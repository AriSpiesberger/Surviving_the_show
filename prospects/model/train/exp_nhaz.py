"""D: neural discrete-time hazard model over sequence + tabular features (2026-09-23).

Everything else in v3 is a GBM on top of an encoder's 6 debut logits. D is one network trained
end to end on the survival objective for all four sheet events:

  trunk     transformer + ranking tokens over the season sequence (architecture C, adopted)
  tabular   MLP over the 335 per-snapshot panel features of v3's set B (z-scored, NaN -> 0 +
            missing flags)
  heads     4 events x 10 years of hazard logits: q[e, t] = P(event e in year t | not yet)
  loss      the discrete-time survival likelihood — BCE on cell (e, t) only when the player is
            eligible for e at the snapshot, the year has resolved (years_fwd >= t) and e has not
            happened by year t-1. Each positive counts once, in its year.
  output    P(e by h) = 1 - prod_{t<=h} (1 - q[e, t]), calibrated with HYip2 on 3-fold cross-fit
            predictions over the fit players (snaps >= 2008), like every other component.

Held-out val scoring comes from the full-fit net (never saw a val player). Writes
runs/exp_nhaz/val_preds.npz in exp_v3_all's format ("<event>__nhaz") and scores D alone and as
an extra component of each event's DEFAULT_SPEC blend (components from runs/exp_v3_all_trf).

    python -m prospects.model.train.exp_nhaz
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

from prospects import config
from prospects.config import REPO_ROOT
from prospects.features.scouting import FEATURE_NAMES
from prospects.model.joint import H_MAX, prep_base, realized_by_h
from prospects.model.joint2 import HYip2Calibrator, attach_raw_features
from prospects.model.train import exp_seq_d as sq
from prospects.model.train.exp_macro_bc import feature_sets
from prospects.model.train.exp_seq_c import TokensRank
from prospects.model.train.joint_xgb import _prep_train

_RUN = config.run()
DB = str(config.model_db())
EV4 = ["TOP_100_PROSPECT", "MLB_DEBUT", "ESTABLISHED_MLB", "STAR_PLUS_ELITE"]
EV_W = np.array([2.0, 1.0, 1.5, 3.0], np.float32)       # rare events are not drowned by debut
SPEC = {"MLB_DEBUT": ["control", "v3"], "TOP_100_PROSPECT": ["control", "haz"],
        "ESTABLISHED_MLB": ["control", "v3", "v3cond", "haz"], "STAR_PLUS_ELITE": ["v3cond", "haz"]}


class NHaz(nn.Module):
    def __init__(self, n_tok, n_tab, n_static, d=48, layers=2):
        super().__init__()
        self.lvl = nn.Embedding(7, 4)
        self.inp = nn.Linear(n_tok + 4, d)
        self.pos = nn.Embedding(sq.MAXLEN, d)
        layer = nn.TransformerEncoderLayer(d, 4, dim_feedforward=2 * d, dropout=0.1, batch_first=True)
        self.seq = nn.TransformerEncoder(layer, num_layers=layers)
        self.tab = nn.Sequential(nn.Linear(n_tab, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU())
        self.mix = nn.Sequential(nn.Linear(d + 64 + n_static, 128), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(128, 64), nn.ReLU())
        self.head = nn.Linear(64, len(EV4) * H_MAX)

    def forward(self, X, L, lens, T, S):
        h = torch.relu(self.inp(torch.cat([X, self.lvl(L)], dim=-1)))
        n = h.shape[1]
        h = h + self.pos(torch.arange(n))[None]
        pad = torch.arange(n)[None, :] >= lens[:, None]
        causal = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        z = self.seq(h, mask=causal, src_key_padding_mask=pad)[torch.arange(len(lens)), lens - 1]
        return self.head(self.mix(torch.cat([z, self.tab(T), S], dim=-1))).view(-1, len(EV4), H_MAX)


def targets(frame):
    """(n, 4, H) event-in-year labels and (n, 4, H) at-risk & resolved & eligible masks."""
    snap = frame["snap_year"].to_numpy(float)
    yf = frame["years_fwd"].to_numpy(float)
    t = np.arange(1, H_MAX + 1)[None, :]
    Yl = np.zeros((len(frame), len(EV4), H_MAX), np.float32)
    M = np.zeros_like(Yl)
    for k, ev in enumerate(EV4):
        trig = pd.to_numeric(frame.get(f"trigger_{ev}"), errors="coerce").to_numpy(float)
        el = (frame[f"eligible_{ev}"] == 1).to_numpy() if f"eligible_{ev}" in frame.columns else np.ones(len(frame), bool)
        rel = trig[:, None] - snap[:, None]                     # year of the event relative to the snap
        Yl[:, k] = (rel == t).astype(np.float32)
        at_risk = ~(rel < t)                                    # NaN (never) -> at risk throughout
        M[:, k] = (el[:, None] & at_risk & (yf[:, None] >= t)).astype(np.float32) * EV_W[k]
    return Yl, M


def train_net(tok, frame, T, epochs, seed, log, bs=1024):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pl = frame.player_id.unique()
    hold = frame.player_id.isin(set(rng.choice(pl, size=max(1, len(pl) // 10), replace=False))).to_numpy()
    Yl, M = targets(frame)
    S = sq.static_tensor(frame)
    P, Sn = frame.player_id.to_numpy(), frame.snap_year.to_numpy()
    net = NHaz(2 * tok.n_num + 1, T.shape[1], len(sq.STATIC))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Tt, Yt, Mt = torch.from_numpy(T), torch.from_numpy(Yl), torch.from_numpy(M)

    def run(idx, train):
        net.train(train)
        order = rng.permutation(idx) if train else idx
        tot, wsum = 0.0, 0.0
        for i in range(0, len(order), bs):
            b = order[i:i + bs]
            X, L, lens = tok.batch(P[b], Sn[b])
            with torch.set_grad_enabled(train):
                logit = net(X, L, lens, Tt[b], S[b])
                loss = (nn.functional.binary_cross_entropy_with_logits(logit, Yt[b], reduction="none")
                        * Mt[b]).sum() / Mt[b].sum().clamp(min=1)
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            tot += loss.item() * float(Mt[b].sum())
            wsum += float(Mt[b].sum())
        return tot / max(wsum, 1)

    tr_idx, va_idx = np.where(~hold)[0], np.where(hold)[0]
    best, best_state = 1e9, None
    for ep in range(epochs):
        ltr, lva = run(tr_idx, True), run(va_idx, False)
        if lva < best - 1e-6:
            best, best_state = lva, {k: v.clone() for k, v in net.state_dict().items()}
        log(f"      epoch {ep + 1}/{epochs}  train {ltr:.5f}  val {lva:.5f}")
    net.load_state_dict(best_state)
    net.eval()
    return net


@torch.no_grad()
def predict(net, tok, frame, T, bs=4096):
    """(n, 4, H) cumulative P(event by h)."""
    S = sq.static_tensor(frame)
    P, Sn = frame.player_id.to_numpy(), frame.snap_year.to_numpy()
    out = []
    for i in range(0, len(frame), bs):
        X, L, lens = tok.batch(P[i:i + bs], Sn[i:i + bs])
        q = torch.sigmoid(net(X, L, lens, torch.from_numpy(T[i:i + bs]), S[i:i + bs])).numpy()
        out.append(1 - np.cumprod(1 - q, axis=2))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit", default=str(_RUN.oof_stacked_long))
    ap.add_argument("--val", default=str(_RUN.oof_val_long))
    ap.add_argument("--aug-long", default=str(_RUN.training / "recent_long.csv"))
    ap.add_argument("--bag", default=str(_RUN.scratch / "v24_build" / "joint_xgb_exp5_bag.pkl"))
    ap.add_argument("--components", default=str(REPO_ROOT / "runs" / "exp_v3_all_trf" / "val_preds.npz"))
    ap.add_argument("--max-entry", type=int, default=2020)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "runs" / "exp_nhaz"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tick = lambda m: print(f"[D] {m}  [{(time.time() - t0) / 60:.1f}m]", flush=True)

    with open(args.bag, "rb") as fh:
        bag = pickle.load(fh)
    live = {f"rw_{n}" for n in FEATURE_NAMES}
    keep_raw = [c for c in bag["keep_raw"] if c in live]
    tab_cols = [c for c in feature_sets(keep_raw)["B"] if c != "h_centered"]
    raw_all = sorted(set(keep_raw) | (set(tab_cols) & live))

    fit = _prep_train(pd.read_csv(args.fit, low_memory=False), DB, args.max_entry)
    if Path(args.aug_long).exists():
        aug = prep_base(pd.read_csv(args.aug_long, low_memory=False), DB)
        for ev in ("TOP_100_PROSPECT", "MLB_DEBUT"):
            if f"eligible_{ev}" in aug.columns:
                aug = aug[aug[f"eligible_{ev}"] == 1]
        fit = pd.concat([fit, aug], ignore_index=True)
    val = prep_base(pd.read_csv(args.val, low_memory=False), DB, max_entry=args.max_entry)
    fit = attach_raw_features(fit, DB, raw_all, verbose=False).drop_duplicates(["player_id", "snap_year"]).reset_index(drop=True)
    val = attach_raw_features(val, DB, raw_all, verbose=False)

    A = fit[tab_cols].to_numpy(np.float64)
    mu, sd = np.nanmean(A, axis=0), np.nanstd(A, axis=0) + 1e-6
    miss_cols = np.where(np.isnan(A).any(axis=0))[0]

    def tab(frame):
        a = (frame[tab_cols].to_numpy(np.float64) - mu) / sd
        flags = np.isnan(a[:, miss_cols]).astype(np.float32)
        return np.concatenate([np.clip(np.nan_to_num(a), -8, 8).astype(np.float32), flags], axis=1)
    T_fit, T_val = tab(fit), tab(val)
    tok = TokensRank(DB, rank_tokens=True)
    tok.fit_scaler(set(fit.player_id), int(fit.snap_year.max()))
    tick(f"fit {len(fit):,} snapshots / {fit.player_id.nunique():,} players; val {len(val):,}; "
         f"tabular {T_fit.shape[1]} (incl. {len(miss_cols)} missing flags)")

    # 3-fold cross-fit (same player folds as the GBM calibrators) + a full net for val
    uniq = np.unique(fit.player_id)
    fold_of = {p: i % 3 for i, p in enumerate(np.random.default_rng(7).permutation(uniq))}
    fold = fit.player_id.map(fold_of).to_numpy()
    oof = np.full((len(fit), len(EV4), H_MAX), np.nan)
    for f in range(3):
        tr = fold != f
        net = train_net(tok, fit[tr].reset_index(drop=True), T_fit[tr], args.epochs, 300 + f, lambda m: None)
        oof[fold == f] = predict(net, tok, fit[fold == f], T_fit[fold == f])
        tick(f"fold {f} cross-fit")
    net = train_net(tok, fit, T_fit, args.epochs, 400, tick)
    Pv = predict(net, tok, val, T_val)
    torch.save(net.state_dict(), out / "nhaz_full.pt")
    tick("full net trained, val scored")

    # calibrate each event's cumulative trajectory with HYip2 on the cross-fit predictions
    yf_f, snap_f, yip_f = fit.years_fwd.to_numpy(), fit.snap_year.to_numpy(), fit.snap_offset.to_numpy()
    yip_v = val.snap_offset.to_numpy()
    preds = {}
    for k, ev in enumerate(EV4):
        el = (fit[f"eligible_{ev}"] == 1).to_numpy()
        pp, hh, yy, yp = [], [], [], []
        for h in range(1, H_MAX + 1):
            m = el & (yf_f >= h) & (snap_f >= 2008)
            pp.append(oof[m, k, h - 1]); hh.append(np.full(m.sum(), h)); yp.append(yip_f[m])
            yy.append(np.asarray(realized_by_h(fit[m], ev, h), dtype=int))
        cal = HYip2Calibrator().fit(np.clip(np.concatenate(pp), 1e-7, 1 - 1e-7), np.concatenate(hh),
                                    np.concatenate(yp), np.concatenate(yy))
        P = np.column_stack([cal.predict(np.clip(Pv[:, k, h - 1], 1e-7, 1 - 1e-7), np.full(len(val), h), yip_v)
                             for h in range(1, H_MAX + 1)])
        preds[ev] = np.maximum.accumulate(P, axis=1)
    np.savez_compressed(out / "val_preds.npz", pid=val.player_id.to_numpy(), snap_year=val.snap_year.to_numpy(),
                        yip=yip_v, **{f"{ev}__nhaz": P for ev, P in preds.items()})

    # ---- score: D alone, the current blend, the blend + D (paired bootstrap vs control) -------
    z = np.load(args.components, allow_pickle=True)
    assert (z["pid"] == val.player_id.to_numpy()).all(), "component predictions are for other val rows"
    yf, vpid = val.years_fwd.to_numpy(), val.player_id.to_numpy()
    rng = np.random.default_rng(0)
    rows = []
    for ev in EV4:
        el = (val[f"eligible_{ev}"] == 1).to_numpy()
        comps = {c: z[f"{ev}__{c}"] for c in SPEC[ev]}
        for h in (3, 6):
            mm = (yf >= h) & el
            yy = np.asarray(realized_by_h(val[mm], ev, h), dtype=float)
            pl = vpid[mm]
            upl = np.unique(pl)
            pos = {q: np.where(pl == q)[0] for q in upl}
            draws = [np.concatenate([pos[q] for q in rng.choice(upl, len(upl), replace=True)]) for _ in range(300)]
            base = z[f"{ev}__control"][mm, h - 1]
            blend = np.mean([c[mm, h - 1] for c in comps.values()], axis=0)
            nh = preds[ev][mm, h - 1]
            blend_d = np.mean([c[mm, h - 1] for c in comps.values()] + [nh], axis=0)
            for name, p in (("control", base), ("nhaz", nh), ("spec_blend", blend), ("spec_blend+nhaz", blend_d)):
                d = np.array([average_precision_score(yy[ix], p[ix]) - average_precision_score(yy[ix], blend[ix])
                              for ix in draws if yy[ix].sum()])
                rows.append({"event": ev, "h": h, "arm": name, "ap": average_precision_score(yy, p),
                             "calib": p.mean() / yy.mean(), "d_vs_blend": d.mean(),
                             "lo": np.percentile(d, 2.5), "hi": np.percentile(d, 97.5)})
    res = pd.DataFrame(rows)
    res.to_csv(out / "metrics.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(4).to_string(index=False))
    tick("done")


if __name__ == "__main__":
    main()
