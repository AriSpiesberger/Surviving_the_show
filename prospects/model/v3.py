"""v3 scorer: GRU sequence encoder + per-event single-stage GBM bags over a base joint bundle.

v3 won the debut target in era (AP@3 +0.020 over v2.4, calibration 0.88 -> 0.98) and out of
era (+0.054 on the corrected walk-forward). The 2026-09-23 full-size screen found that EQUAL-WEIGHT
blends of calibrated components beat the base joint model on every event, with the best blend
differing by event (DEFAULT_SPEC). Components:

    base   the base joint bundle's calibrated trajectory (v2.4 / v2.5)
    v3     the event's own single-stage GBM over panel features + encoder logits
    cond   v3-debut x P(event | debut), calibrated as a product
    haz    discrete-time hazard GBM, P(by h) = 1 - prod(1 - q_t), then calibrated
    nhaz   neural hazard net (exp_nhaz): transformer trunk + tabular MLP, 4 events x 10 years

Each covered event is published as the mean of its components' calibrated trajectories, then
(v3_sidebyside's recalibrate step) a monotone Platt per horizon fit on the held-out build's val
predictions. Uncovered events keep the base model. The sheet builder needs no change: the v3
bundle goes where a joint bundle went (`--xgb`) and the calibrator file it writes where the
joint calibrators went (`--calibrators`).

Build (mirrors v2.4 / v2.5: held-out for thresholds and metrics, 100% for the sheet):

    python -m prospects.model.v3 build --fit <long.csv> --aug-long <recent_long.csv> \
        --base-xgb models/joint_xgb_v2.4.pkl --base-cal models/calibrators_v2.4.pkl \
        --out models/v3.pkl --out-cal models/calibrators_v3.pkl

Stacking contract (the lesson of 2026-09-22): the encoder output fed to the GBM is the
encoder's 6 multi-horizon debut LOGITS, whose meaning is fixed across encoders. Training rows
get logits from an encoder that never saw that player (3 player folds); scored rows from the
full encoder, which never saw a held-out player. The GBM calibrator is fit on 3-fold cross-fit
predictions (snaps >= 2008), exactly as the v2.4 recipe does.
"""
from __future__ import annotations

import argparse
import io
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from prospects import config
from prospects.features.scouting import FEATURE_NAMES
from prospects.model.joint import EVENTS, H_MAX, PUBLISH_H, add_cond_cols, prep_base
from prospects.model.joint2 import HYip2Calibrator, attach_raw_features

KIND = "v3_seq_gbm"
# Chosen 2026-09-23 on the full-size in-era screen (exp_v3_all + hazard + exp_combo), equal
# weights only, from a small candidate set fixed in advance. AP vs deployed v2.4 (h3 / h6):
#   debut        base+v3(+nhaz)          +0.012 / +0.006; adding the neural hazard net (exp_nhaz,
#                                        2026-09-23) gave a further +0.007 / +0.004 (significant)
#   top-100      base+haz                +0.011 / +0.012   (ns, consistent)
#   established  base+v3+cond+haz        +0.025 / +0.018
#   star+        cond+haz (no base)      +0.054 / +0.061   (base is weakest here)
DEFAULT_SPEC = ("MLB_DEBUT=base,v3,nhaz", "TOP_100_PROSPECT=base,haz",
                "ESTABLISHED_MLB=base,v3,cond,haz", "STAR_PLUS_ELITE=cond,haz")
PARAMS = {"max_depth": 8, "min_child_weight": 100, "colsample_bytree": 0.6, "learning_rate": 0.03}
ROUNDS = 340


def _seq():
    # imported lazily: torch + the encoder live with the experiment that validated them
    from prospects.model.train import exp_seq_d
    return exp_seq_d


def _emb_cols():
    return [f"seq_emb_{i}" for i in range(_seq().EMB)]


def _pack(frame, e):
    return pd.concat([frame[["player_id", "snap_year"]].reset_index(drop=True),
                      pd.DataFrame(e, columns=_emb_cols())], axis=1)


def _fit_gbm(X, y, feats, seed):
    from prospects.model.train.exp_cdf_timing2 import BASE_PARAMS, _mono_string
    p = dict(BASE_PARAMS)
    p.update(PARAMS)
    p["seed"] = seed
    p["monotone_constraints"] = _mono_string(feats)
    return xgb.train(p, xgb.QuantileDMatrix(X, label=y, feature_names=feats),
                     num_boost_round=ROUNDS, verbose_eval=False)


def _state_bytes(net) -> bytes:
    buf = io.BytesIO()
    torch.save(net.state_dict(), buf)
    return buf.getvalue()


ENCODERS = ("gru", "transformer_rank")


def _make_tokens(encoder, db):
    if encoder == "transformer_rank":
        from prospects.model.train.exp_seq_c import TokensRank
        return TokensRank(db, rank_tokens=True)
    return _seq().Tokens(db)


def _train_encoder(encoder, tok, samples, epochs, seed):
    if encoder == "transformer_rank":
        from prospects.model.train.exp_seq_c import train_c
        return train_c(tok, samples, epochs, seed, "transformer", lambda m: None)
    return _seq().train_encoder(tok, samples, epochs, seed, lambda m: None)


def _embed(encoder, net, tok, frame):
    if encoder == "transformer_rank":
        from prospects.model.train.exp_seq_c import embed_c
        return embed_c(net, tok, frame)
    return _seq().embed(net, tok, frame)


def build(fit_csv, aug_csv, base_xgb, base_cal, out, out_cal, db, events, seeds=5, epochs=8,
          max_entry=2020, cal_min_snap_year=2008, threads=16, log=print, modes=None, mix=0.5,
          encoder="gru"):
    """Train a v3 bundle on fit_csv (+ aug_csv) and write it plus the merged calibrators."""
    from prospects.model.train.exp_cdf_timing2 import stamp_extra_cols
    from prospects.model.train.exp_macro_bc import feature_sets
    from prospects.model.train.joint_xgb import _assemble, _prep_train
    sq = _seq()
    torch.set_num_threads(threads)
    t0 = time.time()
    tick = lambda m: log(f"[v3] {m}  [{(time.time() - t0) / 60:.1f}m]")

    with open(base_xgb, "rb") as fh:
        base = pickle.load(fh)
    live = {f"rw_{n}" for n in FEATURE_NAMES}
    keep_raw = [c for c in base["keep_raw"] if c in live]
    fs = feature_sets(keep_raw)
    emb_cols = _emb_cols()
    feats = fs["B"] + emb_cols
    raw_all = sorted(set(keep_raw) | (set(fs["B"]) & live))

    fit_base = _prep_train(pd.read_csv(fit_csv, low_memory=False), db, max_entry)
    if aug_csv and Path(aug_csv).exists():
        aug = prep_base(pd.read_csv(aug_csv, low_memory=False), db)
        for ev in ("TOP_100_PROSPECT", "MLB_DEBUT"):
            if f"eligible_{ev}" in aug.columns:
                aug = aug[aug[f"eligible_{ev}"] == 1]
        fit_base = pd.concat([fit_base, aug], ignore_index=True)
    fit_base = attach_raw_features(fit_base, db, raw_all, verbose=False)
    snap_cap = int(fit_base["snap_year"].max())
    tick(f"fit {fit_base.player_id.nunique():,} players, snaps <= {snap_cap}")

    # ---- encoder: 3 out-of-fold encoders for the training rows, one full encoder --------
    tok = _make_tokens(encoder, db)
    tok.fit_scaler(set(fit_base.player_id), snap_cap)
    pl = np.array(sorted(set(fit_base.player_id)))
    fold_of = {p: i % 3 for i, p in enumerate(np.random.default_rng(11).permutation(pl))}
    fb = fit_base.drop_duplicates(["player_id", "snap_year"]).copy()
    fb["fold"] = fb.player_id.map(fold_of)
    samples = sq.sample_frame(fb, snap_cap)
    samples["fold"] = samples.player_id.map(fold_of)
    parts = []
    for k in range(3):
        net = _train_encoder(encoder, tok, samples[samples.fold != k], epochs, 100 + k)
        b = fb[fb.fold == k]
        parts.append(_pack(b, _embed(encoder, net, tok, b)))
        tick(f"encoder fold {k}: {len(b):,} snapshots embedded out-of-fold")
    net_full = _train_encoder(encoder, tok, samples, epochs, 200)
    tick("full encoder trained")
    emb_df = pd.concat(parts, ignore_index=True)
    fit_base = fit_base.merge(emb_df, on=["player_id", "snap_year"], how="left")

    # ---- neural hazard component (exp_nhaz, "nhaz"), only if the spec asks for it ---------
    want_nhaz = any("nhaz" in (modes or {}).get(ev, []) for ev in events
                    if isinstance((modes or {}).get(ev), (list, tuple)))
    nhaz = (_build_nhaz(fit_base, [c for c in fs["B"] if c != "h_centered"], db, epochs, cal_min_snap_year, tick)
            if want_nhaz else None)

    # ---- per-event GBM bags with cross-fit HYip2 calibrators -----------------------------
    fl, Y = _assemble(fit_base, H_MAX)
    fl = stamp_extra_cols(fl)
    del fit_base
    X = fl[feats].values.astype(np.float32)
    h_arr = fl["h"].astype(int).to_numpy()
    yip_arr = fl["snap_offset"].to_numpy()
    era_ok = fl["snap_year"].to_numpy() >= cal_min_snap_year
    pids = fl["player_id"].to_numpy()
    uniq = np.unique(pids)
    gfold = {p: i % 3 for i, p in enumerate(np.random.default_rng(7).permutation(uniq))}
    fold = np.array([gfold[p] for p in pids])
    modes = dict(modes or {})
    kd = EVENTS.index("MLB_DEBUT")
    yd = Y[:, kd].astype(np.float32)

    def elig(ev):
        return (fl[f"eligible_{ev}"] == 1).to_numpy() if f"eligible_{ev}" in fl.columns else np.ones(len(Y), bool)

    def oof_and_bag(y, m, seed0):
        """3-fold cross-fit raw predictions on ALL rows (training on m) + a bag on all of m."""
        oof = np.full(len(y), np.nan)
        for f in range(3):
            b = _fit_gbm(X[m & (fold != f)], y[m & (fold != f)], feats, seed0 + f)
            oof[fold == f] = b.predict(xgb.DMatrix(X[fold == f], feature_names=feats))
            del b
        return oof, [_fit_gbm(X[m], y[m], feats, seed0 + 900 + s) for s in range(seeds)]

    def fit_cal(score, y, m):
        ok = m & era_ok & np.isfinite(score)
        return HYip2Calibrator().fit(np.clip(score[ok], 1e-7, 1 - 1e-7), h_arr[ok], yip_arr[ok],
                                     y[ok].astype(int))

    # previous-horizon row of the same (player, snap), for the hazard mode's at-risk set
    snap_arr = fl["snap_year"].to_numpy()
    kdf = pd.DataFrame({"pid": pids, "snap": snap_arr, "h": h_arr, "i": np.arange(len(pids))})
    prv = kdf.assign(h=kdf.h + 1).rename(columns={"i": "iprev"})[["pid", "snap", "h", "iprev"]]
    iprev = kdf.merge(prv, on=["pid", "snap", "h"], how="left")["iprev"].to_numpy()
    has_prev = np.isfinite(iprev)
    ip = np.where(has_prev, iprev, 0).astype(int)

    # spec: event -> list of components, equal-weighted after calibration. "base" = the base
    # joint's calibrated trajectory; "v3" = the event's own v3 GBM; "cond" = v3-debut x
    # P(event | debut); "haz" = discrete-time hazard. Legacy (modes + mix=0.5) maps to
    # ["base", <mode>].
    spec = {ev: list(modes[ev]) if isinstance(modes.get(ev), (list, tuple))
            else ["base", modes.get(ev, "v3")] for ev in events}
    heads, debut_oof, debut_bag = {}, None, None
    if any(ev == "MLB_DEBUT" or "cond" in spec[ev] for ev in events):
        debut_oof, debut_bag = oof_and_bag(yd, elig("MLB_DEBUT"), 242)
        tick(f"MLB_DEBUT v3: cross-fit + {seeds}-seed bag")
    for ev in events:
        k = EVENTS.index(ev)
        y = Y[:, k].astype(np.float32)
        m = elig(ev)
        heads[ev] = {}
        for comp in spec[ev]:
            if comp == "base":
                continue
            if comp == "v3" and ev == "MLB_DEBUT":
                heads[ev][comp] = {"bag": None, "cal": fit_cal(debut_oof, y, m)}   # shares debut_bag
            elif comp == "v3":
                oofe, bage = oof_and_bag(y, m, 300 + 10 * k)
                heads[ev][comp] = {"bag": bage, "cal": fit_cal(oofe, y, m)}
            elif comp == "cond":       # v3-debut x P(event | debut), calibrated as a product
                oofc, bagc = oof_and_bag(y, m & (yd == 1), 500 + 10 * k)
                heads[ev][comp] = {"bag": bagc, "cal": fit_cal(debut_oof * oofc, y, m)}
            elif comp == "haz":        # discrete-time hazard: P(in year h | not yet)
                y_prev = np.where(has_prev, y[ip], 0.0)
                at_risk = (h_arr == 1) | (has_prev & (y_prev == 0))
                q, bagh = oof_and_bag((y - y_prev).astype(np.float32), m & at_risk, 800 + 10 * k)
                surv = np.full(len(q), np.nan)
                for hh in range(1, H_MAX + 1):
                    idx = np.where(h_arr == hh)[0]
                    prev_s = np.ones(len(idx)) if hh == 1 else np.where(has_prev[idx], surv[ip[idx]], np.nan)
                    surv[idx] = (1 - q[idx]) * prev_s
                heads[ev][comp] = {"bag": bagh, "cal": fit_cal(1 - surv, y, m)}
            elif comp == "nhaz":       # neural hazard net, trained once for all events above
                heads[ev][comp] = {"cal": nhaz["cals"][ev]}
            else:
                raise ValueError(f"unknown component {comp!r} for {ev}")
            tick(f"{ev}: component {comp} (cross-fit calibrator + {seeds}-seed bag)")

    bundle = {"kind": KIND, "base_xgb": str(base_xgb), "base_cal": str(base_cal), "spec": spec,
              "events": list(events), "heads": heads, "debut_bag": debut_bag,
              "nhaz": {k: v for k, v in nhaz.items() if k != "cals"} if nhaz else None,
              "feature_names": feats, "keep_raw": raw_all,
              "encoder_state": _state_bytes(net_full), "n_static": len(sq.STATIC), "encoder": encoder,
              "tok_width": 2 * getattr(tok, "n_num", len(sq.NUM)) + 1,
              "tok_mu": tok.mu, "tok_sd": tok.sd, "seq_out": sq.SEQ_OUT, "snap_cap": snap_cap,
              "params": PARAMS, "rounds": ROUNDS, "h_max": H_MAX, "publish_h": PUBLISH_H,
              "built": time.strftime("%Y-%m-%d %H:%M")}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump(bundle, fh)
    with open(base_cal, "rb") as fh:
        cal_bundle = pickle.load(fh)
    merged = dict(cal_bundle)
    # covered events leave score_v3 already calibrated (mixed): no calibrator for them here,
    # so make_cal_fn passes them through; uncovered events keep the base calibrators
    merged["calibrators"] = {e: c for e, c in cal_bundle["calibrators"].items() if e not in events}
    merged["v3_events"] = list(events)
    merged["v3_spec"] = spec
    with open(out_cal, "wb") as fh:
        pickle.dump(merged, fh)
    tick(f"wrote {out} and {out_cal}")
    return bundle


def _build_nhaz(fit_base, tab_cols, db, epochs, cal_min_snap_year, tick):
    """Train the neural hazard component (architecture D, exp_nhaz): 3 player-fold nets for
    cross-fit calibration + one full net. Returns what scoring needs, plus per-event HYip2
    calibrators fit on the cross-fit cumulative predictions."""
    from prospects.model.joint import realized_by_h
    from prospects.model.train.exp_nhaz import EV4, predict, train_net
    from prospects.model.train.exp_seq_c import TokensRank
    frame = fit_base.drop_duplicates(["player_id", "snap_year"]).reset_index(drop=True)
    A = frame[tab_cols].to_numpy(np.float64)
    mu, sd = np.nanmean(A, axis=0), np.nanstd(A, axis=0) + 1e-6
    miss = np.where(np.isnan(A).any(axis=0))[0]
    T = _nhaz_tab(frame, tab_cols, mu, sd, miss)
    tok = TokensRank(db, rank_tokens=True)
    tok.fit_scaler(set(frame.player_id), int(frame.snap_year.max()))
    uniq = np.unique(frame.player_id)
    fold_of = {p: i % 3 for i, p in enumerate(np.random.default_rng(7).permutation(uniq))}
    fold = frame.player_id.map(fold_of).to_numpy()
    oof = np.full((len(frame), len(EV4), H_MAX), np.nan)
    for f in range(3):
        tr = fold != f
        net = train_net(tok, frame[tr].reset_index(drop=True), T[tr], epochs, 300 + f, lambda m: None)
        oof[fold == f] = predict(net, tok, frame[fold == f], T[fold == f])
        tick(f"nhaz: fold {f} cross-fit")
    net = train_net(tok, frame, T, epochs, 400, lambda m: None)
    yf, snap, yip = frame.years_fwd.to_numpy(), frame.snap_year.to_numpy(), frame.snap_offset.to_numpy()
    cals = {}
    for k, ev in enumerate(EV4):
        el = (frame[f"eligible_{ev}"] == 1).to_numpy()
        pp, hh, yy, yp = [], [], [], []
        for h in range(1, H_MAX + 1):
            m = el & (yf >= h) & (snap >= cal_min_snap_year)
            pp.append(oof[m, k, h - 1]); hh.append(np.full(int(m.sum()), h)); yp.append(yip[m])
            yy.append(np.asarray(realized_by_h(frame[m], ev, h), dtype=int))
        cals[ev] = HYip2Calibrator().fit(np.clip(np.concatenate(pp), 1e-7, 1 - 1e-7), np.concatenate(hh),
                                         np.concatenate(yp), np.concatenate(yy))
    tick("nhaz: full net + calibrators")
    return {"state": _state_bytes(net), "tab_cols": tab_cols, "mu": mu, "sd": sd, "miss": miss,
            "tok_mu": tok.mu, "tok_sd": tok.sd, "tok_width": 2 * tok.n_num + 1, "n_tab": T.shape[1],
            "events": list(EV4), "cals": cals}


def _nhaz_tab(frame, tab_cols, mu, sd, miss):
    a = (frame[tab_cols].to_numpy(np.float64) - mu) / sd
    flags = np.isnan(a[:, miss]).astype(np.float32)
    return np.concatenate([np.clip(np.nan_to_num(a), -8, 8).astype(np.float32), flags], axis=1)


def _score_nhaz(nh, d, db):
    """(n, 4, H) raw cumulative P(event by h) from the stored full net."""
    from prospects.model.train.exp_nhaz import NHaz, predict
    from prospects.model.train.exp_seq_c import TokensRank
    tok = TokensRank(db, rank_tokens=True)
    tok.mu, tok.sd = nh["tok_mu"], nh["tok_sd"]
    net = NHaz(nh["tok_width"], nh["n_tab"], len(_seq().STATIC))
    net.load_state_dict(torch.load(io.BytesIO(nh["state"])))
    net.eval()
    return predict(net, tok, d, _nhaz_tab(d, nh["tab_cols"], nh["mu"], nh["sd"], nh["miss"]))


def is_v3(bundle: dict) -> bool:
    return bundle.get("kind") == KIND


class PlattByH:
    """Monotone final recalibration of a published trajectory: logit p' = a_h + b_h * logit p.

    The fixed mix ranks best but inherits the base joint's under-prediction of the ceiling
    events on held-out players (EST printed at ~0.62-0.74x, STAR+ at ~0.47-0.77x of realized in
    the 2026-09-23 full screen); a learned stacker fixed calibration but lost AP. Platt per
    horizon fixes the level without touching the order within a horizon (b_h > 0 is enforced),
    so AP is unchanged. Fit on the HELD-OUT v3's val predictions, applied to v3.5 — the same
    pattern as the per-yip thresholds. Same predict(values, h, yip) signature as HYip2."""

    def __init__(self):
        self.ab = {}

    def fit(self, p, h, y):
        from sklearn.linear_model import LogisticRegression
        for hh in np.unique(h):
            m = h == hh
            if y[m].sum() < 5:
                continue
            x = _logit(p[m])[:, None]
            lr = LogisticRegression(C=1e6, max_iter=1000).fit(x, y[m])
            a, b = float(lr.intercept_[0]), float(lr.coef_[0, 0])
            if b > 0:
                self.ab[int(hh)] = (a, b)
        return self

    def predict(self, values, h, yip=None):
        v = np.asarray(values, dtype=float)
        out = v.copy()
        hs = np.asarray(h).astype(int)
        for hh, (a, b) in self.ab.items():
            m = hs == hh
            out[m] = 1 / (1 + np.exp(-(a + b * _logit(v[m]))))
        return out


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def recalibrate(heldout_xgb, heldout_cal, val_csv, db, cal_files, max_entry=2020, log=print):
    """Fit PlattByH per covered event on the held-out v3's (mixed, calibrated) val trajectory and
    write it into every calibrator file in cal_files for those events."""
    from prospects.model.joint import realized_by_h
    from prospects.model.joint2 import apply_calibrators_frame, load_calibrators, score_trajectory
    with open(heldout_xgb, "rb") as fh:
        events = pickle.load(fh)["events"]
    val = prep_base(pd.read_csv(val_csv, low_memory=False), db, max_entry=max_entry)
    scored, _ = score_trajectory(heldout_xgb, val, db)
    cb0 = load_calibrators(heldout_cal)          # idempotent: ignore any recalibrator already written
    cb0 = dict(cb0, calibrators={e: c for e, c in cb0["calibrators"].items() if e not in events})
    scored = apply_calibrators_frame(scored, cb0)
    yf = scored["years_fwd"].to_numpy()
    recal = {}
    for ev in events:
        el = (scored[f"eligible_{ev}"] == 1).to_numpy() if f"eligible_{ev}" in scored.columns \
            else np.ones(len(scored), bool)
        P, H, Yv = [], [], []
        for h in range(1, H_MAX + 1):
            m = (yf >= h) & el
            if not m.any():
                continue
            P.append(scored.loc[m, f"xp_{ev}_h{h}"].to_numpy(float))
            H.append(np.full(int(m.sum()), h))
            Yv.append(np.asarray(realized_by_h(scored[m], ev, h), dtype=int))
        p, hh, yy = np.concatenate(P), np.concatenate(H), np.concatenate(Yv)
        # instantiate through the importable module: run as `python -m prospects.model.v3`,
        # the bare name would pickle as __main__.PlattByH and no other process could load it
        from prospects.model import v3 as _v3mod
        recal[ev] = _v3mod.PlattByH().fit(p, hh, yy)
        before = {h: (p[hh == h].mean() / max(yy[hh == h].mean(), 1e-9)) for h in (3, 6)}
        log(f"[v3-recal] {ev}: calibration before h3 {before[3]:.2f} / h6 {before[6]:.2f}; "
            f"Platt fit for {len(recal[ev].ab)} horizons")
    for f in cal_files:
        with open(f, "rb") as fh:
            cb = pickle.load(fh)
        cb["calibrators"] = dict(cb["calibrators"])
        cb["calibrators"].update(recal)
        cb["v3_recal"] = "PlattByH on held-out v3 val predictions"
        with open(f, "wb") as fh:
            pickle.dump(cb, fh)
        log(f"[v3-recal] wrote recalibrators for {list(recal)} into {f}")
    return recal


def score_v3(bundle: dict, df: pd.DataFrame, db: str):
    """Base joint trajectory for every event, then v3's raw trajectory for bundle['events'].
    Returns (scored_df, base_bundle) — the same shape score_trajectory returns."""
    from prospects.model.joint2 import score_trajectory
    from prospects.model.train.exp_cdf_timing2 import stamp_extra_cols
    sq = _seq()
    scored, base = score_trajectory(bundle["base_xgb"], df, db)

    d = attach_raw_features(df, db, [c for c in bundle["keep_raw"] if c not in df.columns], verbose=False)
    encoder = bundle.get("encoder", "gru")
    tok = _make_tokens(encoder, db)
    tok.mu, tok.sd = bundle["tok_mu"], bundle["tok_sd"]
    if encoder == "transformer_rank":
        from prospects.model.train.exp_seq_c import EncoderC
        net = EncoderC(bundle["tok_width"], bundle["n_static"], "transformer")
    else:
        net = sq.Encoder(bundle["n_static"])
    net.load_state_dict(torch.load(io.BytesIO(bundle["encoder_state"])))
    net.eval()
    snaps = d.drop_duplicates(["player_id", "snap_year"])
    emb = _pack(snaps, _embed(encoder, net, tok, snaps))
    d = d.drop(columns=[c for c in emb.columns if c.startswith("seq_emb_")], errors="ignore")
    d = d.merge(emb, on=["player_id", "snap_year"], how="left")
    feats, h_max = bundle["feature_names"], int(bundle.get("h_max", H_MAX))
    publish_h = int(bundle.get("publish_h", PUBLISH_H))
    subs = [stamp_extra_cols(add_cond_cols(d, h))[feats].values.astype(np.float32) for h in range(1, h_max + 1)]
    yip = scored["snap_offset"].to_numpy()

    def bag_traj(bag):
        return np.column_stack([np.mean([b.predict(xgb.DMatrix(s, feature_names=feats)) for b in bag], axis=0)
                                for s in subs])

    def cal_traj(raw, cal):
        raw = np.maximum.accumulate(np.clip(raw, 1e-7, 1 - 1e-7), axis=1)
        P = np.column_stack([cal.predict(raw[:, h - 1], np.full(len(raw), h), yip) for h in range(1, h_max + 1)])
        return np.maximum.accumulate(P, axis=1)

    from prospects.model.joint2 import load_calibrators, make_cal_fn
    base_cal = make_cal_fn(load_calibrators(bundle["base_cal"]), scored)
    debut_raw = np.maximum.accumulate(bag_traj(bundle["debut_bag"]), axis=1) if bundle.get("debut_bag") else None
    nh = bundle.get("nhaz")
    nh_raw = _score_nhaz(nh, d, db) if nh else None
    for ev in bundle["events"]:
        parts = []
        for comp in bundle["spec"][ev]:
            if comp == "base":
                raw_b = np.maximum.accumulate(np.column_stack([scored[f"xp_{ev}_h{h}"].to_numpy(float)
                                                               for h in range(1, h_max + 1)]), axis=1)
                parts.append(np.maximum.accumulate(np.column_stack(
                    [base_cal(raw_b[:, h - 1], ev, h) for h in range(1, h_max + 1)]), axis=1))
                continue
            head = bundle["heads"][ev][comp]
            if comp == "v3" and ev == "MLB_DEBUT":
                raw_v = debut_raw
            elif comp == "cond":
                raw_v = debut_raw * bag_traj(head["bag"])
            elif comp == "haz":
                raw_v = 1 - np.cumprod(1 - np.clip(bag_traj(head["bag"]), 0, 1), axis=1)
            elif comp == "nhaz":
                raw_v = nh_raw[:, nh["events"].index(ev), :h_max]
            else:
                raw_v = bag_traj(head["bag"])
            parts.append(cal_traj(raw_v, head["cal"]))
        P = np.maximum.accumulate(np.mean(parts, axis=0), axis=1)
        for hi in range(h_max):
            scored[f"xp_{ev}_h{hi + 1}"] = P[:, hi]
        scored[f"xp_{ev}"] = P[:, min(publish_h, h_max) - 1]
    return scored, base


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--fit", required=True)
    b.add_argument("--aug-long", default=None)
    b.add_argument("--base-xgb", required=True)
    b.add_argument("--base-cal", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--out-cal", required=True)
    b.add_argument("--db", default=str(config.model_db()))
    b.add_argument("--spec", nargs="+", default=list(DEFAULT_SPEC),
                   help="EVENT=comp,comp,... with comps from base/v3/cond/haz, equal-weighted after "
                        "calibration; events not listed keep the base model")
    b.add_argument("--seeds", type=int, default=5)
    b.add_argument("--epochs", type=int, default=8)
    b.add_argument("--encoder", choices=ENCODERS, default="gru")
    b.add_argument("--max-entry", type=int, default=2020)
    b.add_argument("--threads", type=int, default=16)
    r = sub.add_parser("recalibrate")
    r.add_argument("--heldout-xgb", required=True)
    r.add_argument("--heldout-cal", required=True)
    r.add_argument("--val", default=str(config.run().oof_val_long))
    r.add_argument("--db", default=str(config.model_db()))
    r.add_argument("--write", nargs="+", required=True, help="calibrator files to receive the recalibrators")
    a = ap.parse_args()
    if a.cmd == "recalibrate":
        recalibrate(a.heldout_xgb, a.heldout_cal, a.val, a.db, a.write, log=lambda m: print(m, flush=True))
    if a.cmd == "build":
        spec = {kv.split("=", 1)[0]: kv.split("=", 1)[1].split(",") for kv in a.spec}
        build(a.fit, a.aug_long, a.base_xgb, a.base_cal, a.out, a.out_cal, a.db, list(spec),
              seeds=a.seeds, epochs=a.epochs, max_entry=a.max_entry, threads=a.threads, encoder=a.encoder,
              log=lambda m: print(m, flush=True), modes=spec)


if __name__ == "__main__":
    main()
