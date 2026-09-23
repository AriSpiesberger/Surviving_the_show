"""In-era gate for candidate v3 (2026-09-22).

v3 = single-stage, debut-only GBM on all 325 raw panel features + a 32-d GRU sequence
embedding, 5-seed bag, 340 rounds (exp_ap_opt c00). Out of era it beat production by
+0.052 AP@3 [+0.041, +0.064]. Before it can go near the sheet it must not give that back
in era — the check that killed the first blend (exp_macro_inera).

Same fit players, same held-out val players, same recent-cohort augmentation and the same
HYip2 calibration protocol as the deployed v2.4 (exp_cdf_timing5), which is the control and
is loaded from disk, not refit. Embeddings are stacked honestly: fit rows get embeddings from
an encoder trained without that player (3 player folds); val rows from the full-fit encoder,
which never saw a val player.

Arms: v3 (recency-weighted, as selected out of era) and v3-flat (no recency weights —
recency weighting cost the C1 blend 1.3 basket points in era).

Reported on held-out val: AP / AUC / calibration / ECE at h = 1, 3, 6; the >=0.70 region;
and the basket test — precision of each model's top-N at the same N per years-in-pro that
the deployed sheet rule selects.

    python -m prospects.model.train.exp_v3_inera
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

from prospects import config
from prospects.config import REPO_ROOT
from prospects.features.scouting import FEATURE_NAMES
from prospects.model.joint import EVENTS, H_MAX, add_cond_cols, prep_base, realized_by_h
from prospects.model.joint2 import HYip2Calibrator, attach_raw_features
from prospects.model.train.exp_cal_topend import reliability
from prospects.model.train.exp_cdf_timing2 import BASE_PARAMS, _mono_string, stamp_extra_cols
from prospects.model.train.exp_cdf_timing4 import sweep_val
from prospects.model.train.exp_macro_bc import feature_sets
from prospects.model.train.exp_seq_d import EMB, SEQ_OUT, Tokens, embed, sample_frame, train_encoder
from prospects.model.train.joint_xgb import _assemble, _prep_train

_RUN = config.run()
DB = str(config.model_db())
EV = "MLB_DEBUT"
K = EVENTS.index(EV)
EMB_COLS = [f"seq_emb_{i}" for i in range(EMB)]
PARAMS = {"max_depth": 8, "min_child_weight": 100, "colsample_bytree": 0.6, "learning_rate": 0.03}
ROUNDS = 340


def fit(X, y, w, feats, seed):
    p = dict(BASE_PARAMS)
    p.update(PARAMS)
    p["seed"] = seed
    p["monotone_constraints"] = _mono_string(feats)
    d = xgb.QuantileDMatrix(X, label=y, weight=w, feature_names=feats)
    return xgb.train(p, d, num_boost_round=ROUNDS, verbose_eval=False)


def pack(frame, e):
    return pd.concat([frame[["player_id", "snap_year"]].reset_index(drop=True),
                      pd.DataFrame(e, columns=EMB_COLS)], axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit", default=str(_RUN.oof_stacked_long))
    ap.add_argument("--val", default=str(_RUN.oof_val_long))
    ap.add_argument("--aug-long", default=str(_RUN.training / "recent_long.csv"))
    ap.add_argument("--bag", default=str(_RUN.scratch / "v24_build" / "joint_xgb_exp5_bag.pkl"))
    ap.add_argument("--cal", default=str(_RUN.scratch / "v24_build" / "calibrators_hyip2.pkl"))
    ap.add_argument("--max-entry", type=int, default=2020)
    ap.add_argument("--cal-min-snap-year", type=int, default=2008)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "runs" / "exp_v3_inera"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tick = lambda m: print(f"[v3] {m}  [{(time.time() - t0) / 60:.1f}m]", flush=True)

    with open(args.bag, "rb") as fh:
        bagA = pickle.load(fh)
    with open(args.cal, "rb") as fh:
        calA = pickle.load(fh)["calibrators"][EV]
    live = {f"rw_{n}" for n in FEATURE_NAMES}
    keep_raw = [c for c in bagA["keep_raw"] if c in live]
    fs = feature_sets(keep_raw)
    featsA = list(bagA["feature_names"])
    feats = fs["B"] + EMB_COLS
    raw_all = sorted(set(keep_raw) | (set(fs["B"]) & live))

    # ---- frames: identical to the deployed recipe -----------------------------------
    fit_base = _prep_train(pd.read_csv(args.fit, low_memory=False), DB, args.max_entry)
    if Path(args.aug_long).exists():
        aug = prep_base(pd.read_csv(args.aug_long, low_memory=False), DB)
        for ev in ("TOP_100_PROSPECT", "MLB_DEBUT"):
            if f"eligible_{ev}" in aug.columns:
                aug = aug[aug[f"eligible_{ev}"] == 1]
        fit_base = pd.concat([fit_base, aug], ignore_index=True)
    val_base = prep_base(pd.read_csv(args.val, low_memory=False), DB, max_entry=args.max_entry)
    fit_base = attach_raw_features(fit_base, DB, raw_all, verbose=False)
    val_base = attach_raw_features(val_base, DB, raw_all, verbose=False)
    val_pl = set(val_base.player_id)
    assert not (set(fit_base.player_id) & val_pl), "fit/val player overlap"
    tick(f"fit {fit_base.player_id.nunique():,} players / val {len(val_pl):,} players")

    # ---- sequence embeddings, stacked out-of-fold ------------------------------------
    emb_f = out / f"embeddings_{SEQ_OUT}.npz"
    if emb_f.exists():
        z = np.load(emb_f, allow_pickle=True)
        emb_df = pd.concat([pd.DataFrame({"player_id": z["pid"], "snap_year": z["snap"]}),
                            pd.DataFrame(z["emb"], columns=EMB_COLS)], axis=1)
        tick("embeddings loaded from cache")
    else:
        tok = Tokens()
        tok.fit_scaler(set(fit_base.player_id), 2025)
        pl = np.array(sorted(set(fit_base.player_id)))
        fold_of = {p: i % 3 for i, p in enumerate(np.random.default_rng(11).permutation(pl))}
        fb = fit_base.drop_duplicates(["player_id", "snap_year"]).copy()
        fb["fold"] = fb.player_id.map(fold_of)
        samples = sample_frame(fb, 2025)
        samples["fold"] = samples.player_id.map(fold_of)
        parts = []
        for k in range(3):
            net = train_encoder(tok, samples[samples.fold != k], args.epochs, 100 + k, lambda m: None)
            b = fb[fb.fold == k]
            parts.append(pack(b, embed(net, tok, b)))
            tick(f"encoder fold {k}: {len(b):,} fit snapshots embedded out-of-fold")
        net = train_encoder(tok, samples, args.epochs, 200, lambda m: None)
        vb = val_base.drop_duplicates(["player_id", "snap_year"])
        parts.append(pack(vb, embed(net, tok, vb)))
        emb_df = pd.concat(parts, ignore_index=True)
        np.savez_compressed(emb_f, pid=emb_df.player_id.to_numpy(), snap=emb_df.snap_year.to_numpy(),
                            emb=emb_df[EMB_COLS].to_numpy(np.float32))
        tick("full encoder done; val embedded; cached")
    fit_base = fit_base.merge(emb_df, on=["player_id", "snap_year"], how="left")
    val_base = val_base.merge(emb_df, on=["player_id", "snap_year"], how="left")

    # ---- debut-only training frame ----------------------------------------------------
    fit_long, Y = _assemble(fit_base, H_MAX)
    fit_long = stamp_extra_cols(fit_long)
    del fit_base
    X = fit_long[feats].values.astype(np.float32)
    y = Y[:, K].astype(np.float32)
    h_arr = fit_long["h"].astype(int).to_numpy()
    yip_arr = fit_long["snap_offset"].to_numpy()
    snap_yr = fit_long["snap_year"].to_numpy()
    pids = fit_long["player_id"].to_numpy()
    elig = (fit_long[f"eligible_{EV}"] == 1).to_numpy() if f"eligible_{EV}" in fit_long.columns \
        else np.ones(len(fit_long), bool)
    era_ok = snap_yr >= args.cal_min_snap_year
    uniq = np.unique(pids)
    rng = np.random.default_rng(7)
    rng.choice(uniq, size=max(1, len(uniq) // 10), replace=False)      # same stream as exp5
    fold_of = {p: i % 3 for i, p in enumerate(rng.permutation(uniq))}
    fold_idx = np.array([fold_of[p] for p in pids])
    del fit_long
    tick(f"debut-only rows {int(elig.sum()):,} of {len(y):,}; {len(feats)} features")

    def val_traj(predict_fn, cal):
        cols = []
        for h in range(1, H_MAX + 1):
            sub = stamp_extra_cols(add_cond_cols(val_base, h))
            cols.append(predict_fn(sub))
        raw = np.maximum.accumulate(np.column_stack(cols), axis=1)
        yip = val_base["snap_offset"].to_numpy()
        P = np.column_stack([cal.predict(raw[:, h - 1], np.full(len(val_base), h), yip)
                             for h in range(1, H_MAX + 1)])
        return np.maximum.accumulate(P, axis=1)

    # ---- control: the deployed v2.4, loaded not refit --------------------------------
    sv = sweep_val(bagA["models"], featsA, val_base)
    rawA = np.maximum.accumulate(sv[[f"xp_{EV}_h{h}" for h in range(1, H_MAX + 1)]]
                                 .to_numpy(dtype=np.float64), axis=1)
    yip_v = sv["snap_offset"].to_numpy()
    P_A = np.maximum.accumulate(np.column_stack([calA.predict(rawA[:, h - 1], np.full(len(sv), h), yip_v)
                                                 for h in range(1, H_MAX + 1)]), axis=1)
    preds = {"A (deployed v2.4)": P_A}
    tick("control scored")

    w_rec = (0.5 ** ((2025 - snap_yr) / 4.0)).astype(np.float32)
    for name, w in (("v3 (recency)", w_rec), ("v3-flat", np.ones_like(w_rec))):
        m = elig
        oof = np.full(len(y), np.nan)
        for f in range(3):
            tr, ho = m & (fold_idx != f), m & (fold_idx == f)
            b = fit(X[tr], y[tr], w[tr], feats, 242 + f)
            oof[ho] = b.predict(xgb.DMatrix(X[ho], feature_names=feats))
            del b
        ok = m & era_ok & np.isfinite(oof)
        cal = HYip2Calibrator().fit(oof[ok], h_arr[ok], yip_arr[ok], y[ok].astype(int))
        bag = [fit(X[m], y[m], w[m], feats, 1142 + s) for s in range(args.seeds)]
        predict = lambda sub: np.mean([bst.predict(xgb.DMatrix(sub[feats].values.astype(np.float32),
                                                               feature_names=feats)) for bst in bag], axis=0)
        preds[name] = val_traj(predict, cal)
        with open(out / f"bag_{name.split()[0]}{'_flat' if 'flat' in name else ''}.pkl", "wb") as fh:
            pickle.dump({"models": bag, "feature_names": feats, "calibrator": cal, "rounds": ROUNDS,
                         "params": PARAMS, "recency": "flat" not in name}, fh)
        del bag
        tick(f"{name}: cross-fit calibrator + {args.seeds}-seed bag + val scored")

    np.savez_compressed(out / "val_preds.npz", pid=val_base["player_id"].to_numpy(),
                        snap_year=val_base["snap_year"].to_numpy(), yip=yip_v,
                        **{k.split()[0].replace("-", "_"): v for k, v in preds.items()})

    # ---- metrics ----------------------------------------------------------------------
    yf = val_base["years_fwd"].to_numpy()
    el = (val_base[f"eligible_{EV}"] == 1).to_numpy() if f"eligible_{EV}" in val_base.columns \
        else np.ones(len(val_base), bool)
    rows = []
    for name, P in preds.items():
        for h in (1, 3, 6):
            mm = (yf >= h) & el
            p, yy = P[mm, h - 1], np.asarray(realized_by_h(val_base[mm], EV, h), dtype=float)
            top = p >= 0.70
            rows.append({"model": name, "h": h, "n": int(mm.sum()), "ap": average_precision_score(yy, p),
                         "auc": roc_auc_score(yy, p), "calib": p.mean() / yy.mean(),
                         "ece": reliability(p, yy), "brier": float(np.mean((p - yy) ** 2)),
                         "top_n": int(top.sum()), "top_pred": float(p[top].mean()) if top.any() else np.nan,
                         "top_actual": float(yy[top].mean()) if top.any() else np.nan})
    res = pd.DataFrame(rows)

    thr = {int(k): float(v) for k, v in json.load(open(_RUN.yip_thresholds(60))).items()}
    mm = (yf >= 3) & el & (yip_v <= 3)
    y3 = np.asarray(realized_by_h(val_base[mm], EV, 3), dtype=float)
    yv, pa = yip_v[mm], preds["A (deployed v2.4)"][mm, 2]
    rng = np.random.default_rng(0)
    basket = []
    for name, P in preds.items():
        p = P[mm, 2]
        sel = np.zeros(len(p), bool)
        for yy_ in range(4):
            idx = np.where(yv == yy_)[0]
            sel[idx[np.argsort(-p[idx])[:int((pa[idx] >= thr.get(yy_, 9)).sum())]]] = True
        basket.append({"model": name, "n_selected": int(sel.sum()), "precision": float(y3[sel].mean()),
                       "debuts": int(y3[sel].sum()), "of_total": int(y3.sum())})
    bk = pd.DataFrame(basket)

    # paired bootstrap on AP@3 vs control (player-level resampling of the h=3 rows)
    m3 = (yf >= 3) & el
    y3a = np.asarray(realized_by_h(val_base[m3], EV, 3), dtype=float)
    pl3 = val_base.loc[m3, "player_id"].to_numpy()
    upl = np.unique(pl3)
    pos = {p: np.where(pl3 == p)[0] for p in upl}
    boot = {}
    for name, P in preds.items():
        if name.startswith("A "):
            continue
        d = []
        for _ in range(300):
            ix = np.concatenate([pos[p] for p in rng.choice(upl, len(upl), replace=True)])
            if y3a[ix].sum():
                d.append(average_precision_score(y3a[ix], P[m3, 2][ix])
                         - average_precision_score(y3a[ix], preds["A (deployed v2.4)"][m3, 2][ix]))
        d = np.array(d)
        boot[name] = (d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5))

    res.to_csv(out / "metrics.csv", index=False)
    bk.to_csv(out / "basket.csv", index=False)
    pd.set_option("display.width", 220)
    for h in (3, 6, 1):
        print(f"\n===== held-out val, P(debut <= {h}y) =====")
        print(res[res.h == h].drop(columns="h").round(4).to_string(index=False))
    print("\n===== basket test, h=3, yip<=3, same N per yip as the deployed rule =====")
    print(bk.round(4).to_string(index=False))
    print("\n===== AP@3 vs deployed, paired player-bootstrap (300) =====")
    for k, (m_, lo, hi) in boot.items():
        print(f"  {k:14s} {m_:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    tick("done")


if __name__ == "__main__":
    main()
