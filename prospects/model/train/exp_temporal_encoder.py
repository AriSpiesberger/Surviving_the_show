"""EXPERIMENT: temporal encoder — learned trajectory embeddings as features.

Every top hand-feature the trees rely on (promotion velocity, deltas,
accelerations, windows, years-since-max-level) is a fixed-lens approximation
of trajectory SHAPE, and the model's known misses are sequence-pattern
misses (rehab-shaped seasons, fast risers). This experiment learns the shape
directly: a small GRU over per-stint vectors (level, age, workload, rates,
year-gaps) produces a 24-dim embedding per (player, snap), trained
multi-task on debut/establishment horizons using FIT players only (val
players' labels never touch the encoder), then the embedding is concatenated
into the v2.4 joint feature set and A/B'd on the clean val.

Stages (each cached under runs/exp_temporal_encoder/):
  1. stint table from season_stats (14-dim vectors, sorted, per player)
  2. (pid, snap) sequences (last 10 stints with year <= snap)
  3. GRU encoder trained on fit+aug snaps, masked BCE over
     [debut<=1,2,3,5; established<=6]
  4. embeddings for fit/aug/val snaps -> emb_0..emb_23 columns
  5. joint layer A/B: FEAT2 + raw160 + emb24 vs v2.4 reference

    python -m prospects.model.train.exp_temporal_encoder          # full
    python -m prospects.model.train.exp_temporal_encoder --quick  # smoke
"""
from __future__ import annotations

import argparse
import json
import pickle
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from prospects import config
from prospects.config import REPO_ROOT
from prospects.model.joint import EVENTS, H_MAX, add_cond_cols, prep_base, realized_by_h
from prospects.model.joint2 import HYip2Calibrator, attach_raw_features
from prospects.model.train.joint_xgb import _assemble, _prep_train
from prospects.model.train.exp_cdf_timing import (
    cdf_timing, score_lasso_timing, timing_report, weighted_ap_at,
)
from prospects.model.train.exp_cdf_timing2 import FEAT2, stamp_extra_cols
from prospects.model.train.exp_cdf_timing4 import predict_rows, train_one

_RUN = config.run()
DB = str(config.model_db())
OUT_DIR = REPO_ROOT / "runs" / "exp_temporal_encoder"
G3_SLOW = {"max_depth": 8, "min_child_weight": 100,
           "colsample_bytree": 0.6, "learning_rate": 0.03}
V24_REF = {"deb_ap_h1": 0.4421, "deb_ap_h3": 0.6120, "deb_ap_h6": 0.6683,
           "wap_h6": 0.4759, "tim_mae": 1.034}

SEQ_LEN = 10
STINT_DIM = 14
EMB_DIM = 24
LEVEL_RANK = {"DSL": 1, "FCL": 1, "CPX": 1, "RK": 1, "ROK": 1, "A-": 2,
              "A": 3, "A+": 4, "AA": 5, "AAA": 6, "MLB": 7}
HORIZONS = [1, 2, 3, 5]          # debut heads
EST_H = 6                        # established head


def build_stints(db: str) -> dict[str, np.ndarray]:
    """pid -> (n_stints, 1+STINT_DIM) array: [season_year, features...],
    sorted by (year, level)."""
    con = sqlite3.connect(db)
    df = pd.read_sql(
        "SELECT player_id, season_year, level, pa, ip, woba, k_pct, bb_pct, "
        "iso, fip, whip, era, age_during_season FROM season_stats", con)
    con.close()
    df = df.dropna(subset=["season_year"])
    df["yr"] = df["season_year"].astype(int)
    df["lv"] = df["level"].astype(str).str.upper().map(LEVEL_RANK).fillna(3)
    df = df.sort_values(["player_id", "yr", "lv"])
    df["is_p"] = (df["ip"].fillna(0) > 0).astype(float)

    def _n(col, div, clip=3.0):
        return np.clip(pd.to_numeric(df[col], errors="coerce")
                       .fillna(0).to_numpy() / div, -clip, clip)

    F = np.column_stack([
        df["lv"].to_numpy() / 7.0,
        _n("age_during_season", 30.0),
        np.log1p(pd.to_numeric(df["pa"], errors="coerce").fillna(0)) / 7.0,
        np.log1p(pd.to_numeric(df["ip"], errors="coerce").fillna(0)) / 5.0,
        _n("woba", 0.400), _n("k_pct", 0.35), _n("bb_pct", 0.15),
        _n("iso", 0.250), _n("fip", 6.0), _n("whip", 2.0), _n("era", 8.0),
        np.zeros(len(df)),               # year-gap, filled per player below
        df["is_p"].to_numpy(),
        np.zeros(len(df)),               # within-year stint idx
    ]).astype(np.float32)
    out: dict[str, np.ndarray] = {}
    yrs = df["yr"].to_numpy()
    pids = df["player_id"].to_numpy()
    order = np.arange(len(df))
    for pid, idx in pd.Series(order).groupby(pids).groups.items():
        idx = np.asarray(idx)
        y = yrs[idx]
        f = F[idx].copy()
        f[1:, 11] = np.clip(y[1:] - y[:-1], 0, 3) / 3.0     # gap
        wy = np.zeros(len(idx))
        for j in range(1, len(idx)):
            wy[j] = wy[j - 1] + 1 if y[j] == y[j - 1] else 0
        f[:, 13] = np.clip(wy, 0, 3) / 3.0
        out[pid] = np.column_stack([y.astype(np.float32), f])
    return out


def sequences_for(snaps: pd.DataFrame, stints: dict) -> np.ndarray:
    """(n, SEQ_LEN, STINT_DIM) float32, last SEQ_LEN stints with yr <= snap
    (zero-padded at the front)."""
    S = np.zeros((len(snaps), SEQ_LEN, STINT_DIM), dtype=np.float32)
    pid_arr = snaps["player_id"].to_numpy()
    yr_arr = snaps["snap_year"].to_numpy()
    for i in range(len(snaps)):
        st = stints.get(pid_arr[i])
        if st is None:
            continue
        sel = st[st[:, 0] <= yr_arr[i]][-SEQ_LEN:, 1:]
        if len(sel):
            S[i, -len(sel):, :] = sel
    return S


class TrajEncoder(nn.Module):
    def __init__(self, d_in=STINT_DIM, d_h=EMB_DIM, n_heads=len(HORIZONS) + 1):
        super().__init__()
        self.gru = nn.GRU(d_in, d_h, batch_first=True)
        self.head = nn.Linear(d_h, n_heads)

    def embed(self, x):
        _, h = self.gru(x)
        return h[-1]

    def forward(self, x):
        return self.head(self.embed(x))


def train_encoder(S, targets, masks, epochs, seed, verbose=True):
    torch.manual_seed(seed)
    enc = TrajEncoder()
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    X = torch.from_numpy(S)
    Y = torch.from_numpy(targets.astype(np.float32))
    M = torch.from_numpy(masks.astype(np.float32))
    n = len(X)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        rng.shuffle(idx)
        tot = cnt = 0.0
        for lo in range(0, n, 2048):
            b = idx[lo:lo + 2048]
            opt.zero_grad()
            out = enc(X[b])
            l = (lossf(out, Y[b]) * M[b]).sum() / (M[b].sum() + 1e-9)
            l.backward()
            opt.step()
            tot += float(l) * len(b)
            cnt += len(b)
        if verbose:
            print(f"    epoch {ep+1}/{epochs} masked-bce {tot/cnt:.4f}",
                  flush=True)
    enc.eval()
    return enc


@torch.no_grad()
def extract(enc, S, bs=8192):
    out = np.zeros((len(S), EMB_DIM), dtype=np.float32)
    X = torch.from_numpy(S)
    for lo in range(0, len(S), bs):
        out[lo:lo + bs] = enc.embed(X[lo:lo + bs]).numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default=str(_RUN.oof_stacked_long))
    ap.add_argument("--val", default=str(_RUN.oof_val_long))
    ap.add_argument("--aug-long",
                    default=str(_RUN.training / "recent_long.csv"))
    ap.add_argument("--db", default=DB)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bag-seeds", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    with open(_RUN.models / "joint_xgb_v2.4.pkl", "rb") as fh:
        keep_raw = list(pickle.load(fh)["keep_raw"])

    print(f"[enc] loading longs")
    fit_base = _prep_train(pd.read_csv(args.fit), args.db, 2020)
    if Path(args.aug_long).exists():
        aug = prep_base(pd.read_csv(args.aug_long), args.db)
        for ev in ("TOP_100_PROSPECT", "MLB_DEBUT"):
            c = f"eligible_{ev}"
            if c in aug.columns:
                aug = aug[aug[c] == 1]
        fit_base = pd.concat([fit_base, aug], ignore_index=True)
    val_base = prep_base(pd.read_csv(args.val), args.db, max_entry=2020)

    rounds_cap, estop = 2000, 50
    if args.quick:
        rng = np.random.default_rng(0)
        pids = fit_base["player_id"].unique()
        keep = set(rng.choice(pids, size=max(200, len(pids) // 7),
                              replace=False))
        fit_base = fit_base[fit_base["player_id"].isin(keep)]
        args.epochs, rounds_cap, estop, args.bag_seeds = 2, 100, 20, 1
        print(f"  [quick] {fit_base.player_id.nunique():,} players")

    # ---- 1-2: stints + sequences ---------------------------------------
    print(f"[enc] building stints + sequences [{(time.time()-t0)/60:.1f}m]")
    stints = build_stints(args.db)
    S_fit = sequences_for(fit_base, stints)
    S_val = sequences_for(val_base, stints)
    print(f"  fit seqs {S_fit.shape}, val seqs {S_val.shape} "
          f"[{(time.time()-t0)/60:.1f}m]")

    # ---- 3: encoder targets (fit players only; masked by resolution) ----
    snap = fit_base["snap_year"].astype(float).to_numpy()
    yfwd = fit_base["years_fwd"].astype(float).to_numpy()
    t_deb = pd.to_numeric(fit_base["trigger_MLB_DEBUT"],
                          errors="coerce").to_numpy()
    t_est = pd.to_numeric(fit_base.get("trigger_ESTABLISHED_MLB", np.nan),
                          errors="coerce").to_numpy()
    T, M = [], []
    for hz in HORIZONS:
        T.append((~np.isnan(t_deb)) & (t_deb > snap) & (t_deb <= snap + hz))
        M.append(yfwd >= hz)
    T.append((~np.isnan(t_est)) & (t_est > snap) & (t_est <= snap + EST_H))
    M.append(yfwd >= EST_H)
    targets = np.column_stack(T)
    masks = np.column_stack(M)
    print(f"[enc] training encoder ({args.epochs} epochs, "
          f"{targets.shape[1]} heads)")
    enc = train_encoder(S_fit, targets, masks, args.epochs, args.seed)
    torch.save(enc.state_dict(), out_dir / "traj_encoder.pt")

    # ---- 4: embeddings ---------------------------------------------------
    E_fit = extract(enc, S_fit)
    E_val = extract(enc, S_val)
    emb_cols = [f"emb_{j}" for j in range(EMB_DIM)]
    for j, cname in enumerate(emb_cols):
        fit_base[cname] = E_fit[:, j]
        val_base[cname] = E_val[:, j]
    print(f"  embeddings done [{(time.time()-t0)/60:.1f}m]")

    # ---- 5: joint A/B ----------------------------------------------------
    feats = list(FEAT2) + keep_raw + emb_cols
    fit_base = attach_raw_features(fit_base, args.db, keep_raw, verbose=False)
    val_base = attach_raw_features(val_base, args.db, keep_raw, verbose=False)
    fit_long, Y = _assemble(fit_base, H_MAX)
    fit_long = stamp_extra_cols(fit_long)
    X = fit_long[feats].values.astype(np.float32)
    print(f"[enc] joint fit rows: {len(fit_long):,} "
          f"({len(feats)} feats) [{(time.time()-t0)/60:.1f}m]")

    fpids = fit_long["player_id"].to_numpy()
    h_arr = fit_long["h"].astype(int).to_numpy()
    yip_arr = fit_long["snap_offset"].to_numpy()
    era = fit_long["snap_year"].to_numpy() >= 2008
    uniq = np.unique(fpids)
    rng = np.random.default_rng(7)
    es_players = set(rng.choice(uniq, size=max(1, len(uniq) // 10),
                                replace=False))
    es_m = np.isin(fpids, list(es_players))
    half = set(rng.choice(uniq, size=len(uniq) // 2, replace=False))
    hm = np.isin(fpids, list(half))

    b = train_one(X[~es_m], Y[~es_m], feats, G3_SLOW, rounds_cap, estop,
                  args.seed, X[es_m], Y[es_m])
    nrounds = int(b.best_iteration) + 1
    print(f"  ES rounds = {nrounds} [{(time.time()-t0)/60:.1f}m]")
    del b

    oof = np.full((len(fit_long), len(EVENTS)), np.nan)
    for tr_m, ho_m in ((~hm, hm), (hm, ~hm)):
        bb = train_one(X[tr_m], Y[tr_m], feats, G3_SLOW, nrounds, 0,
                       args.seed)
        oof[ho_m] = predict_rows(bb, X[ho_m], feats)
        del bb
        print(f"  cross-fit fold done [{(time.time()-t0)/60:.1f}m]")
    cals = {}
    for k, ev in enumerate(EVENTS):
        elig = (fit_long[f"eligible_{ev}"] == 1).to_numpy() \
            if f"eligible_{ev}" in fit_long.columns \
            else np.ones(len(fit_long), bool)
        ok = elig & era & np.isfinite(oof[:, k])
        yv = Y[ok, k].astype(int)
        if 25 <= yv.sum() < len(yv):
            cals[ev] = HYip2Calibrator().fit(oof[ok, k], h_arr[ok],
                                             yip_arr[ok], yv)
    bag = [train_one(X, Y, feats, G3_SLOW, nrounds, 0, args.seed + 100 + s)
           for s in range(args.bag_seeds)]
    print(f"  bag done [{(time.time()-t0)/60:.1f}m]")
    del X
    with open(out_dir / "joint_emb_bag.pkl", "wb") as fh:
        pickle.dump({"models": bag, "feature_names": feats,
                     "keep_raw": keep_raw, "emb_cols": emb_cols,
                     "nrounds": nrounds, "kind": "exp_temporal_encoder"}, fh)

    # eval
    print(f"\n[enc] scoring val")
    import xgboost as xgb_
    preds_by_h = {}
    for h in range(1, H_MAX + 1):
        sub = stamp_extra_cols(add_cond_cols(val_base, h))
        Xv = sub[feats].values.astype(np.float32)
        d = xgb_.DMatrix(Xv, feature_names=list(feats))
        preds_by_h[h] = np.mean([m.predict(d) for m in bag], axis=0)
    sv = val_base.copy()
    yipv = sv["snap_offset"].to_numpy()
    new = {}
    for k, ev in enumerate(EVENTS):
        Mm = np.column_stack([preds_by_h[h][:, k]
                              for h in range(1, H_MAX + 1)])
        Mm = np.maximum.accumulate(Mm, axis=1)
        Cc = Mm.copy()
        if ev in cals:
            for hi, h in enumerate(range(1, H_MAX + 1)):
                Cc[:, hi] = cals[ev].predict(Mm[:, hi],
                                             np.full(len(sv), h), yipv)
            Cc = np.maximum.accumulate(Cc, axis=1)
        for hi, h in enumerate(range(1, H_MAX + 1)):
            new[f"xp_{ev}_h{h}"] = Mm[:, hi]
            new[f"cal_xp_{ev}_h{h}"] = Cc[:, hi]
    sv = pd.concat([sv, pd.DataFrame(new, index=sv.index)], axis=1)

    from prospects.model.train.exp_cdf_timing import per_h_metrics
    rows = per_h_metrics(sv, "", "T_raw(emb)")
    rows += per_h_metrics(sv, "cal_", "T_cal(emb,OOF)")
    met = pd.DataFrame(rows)
    met.to_csv(out_dir / "per_event_h_metrics.csv", index=False)
    print(f"\n===== debut vs v2.4 (clean val) =====")
    for scorer in met["scorer"].unique():
        for h in (1, 3, 6):
            r = met[(met.scorer == scorer) & (met.event == "MLB_DEBUT")
                    & (met.h == h)]
            if len(r):
                r = r.iloc[0]
                print(f"{scorer:<18} h={h}  AP={r['ap']:.4f}  "
                      f"AUC={r['auc']:.4f}  calib={r['calib']:.2f}")
    print(f"v2.4 ref           h=1  AP={V24_REF['deb_ap_h1']:.4f} | "
          f"h=3 {V24_REF['deb_ap_h3']:.4f} | h=6 {V24_REF['deb_ap_h6']:.4f}")
    for scorer in met["scorer"].unique():
        sub_ = met[(met.scorer == scorer) & (met.h == 6)]
        print(f"  wAP@6 {scorer:<18} "
              f"{weighted_ap_at(sub_.to_dict(orient='records'), 6):.4f} "
              f"(v2.4 {V24_REF['wap_h6']:.4f})")

    tim = cdf_timing(sv, "MLB_DEBUT", "cal_")
    trep = timing_report(val_base, {
        "T_cdf_median(cal)": tim["t_med"].to_numpy(),
        "lasso_timing.pkl": score_lasso_timing(val_base, _RUN.timing,
                                               args.db)},
        {"T_cdf_median(cal)": (tim["t_q25"].to_numpy(),
                               tim["t_q75"].to_numpy())})
    print(f"\n[enc] timing (v2.4 ref {V24_REF['tim_mae']:.3f}):")
    print(trep.to_string(index=False))

    # how much do the trees actually use the embedding?
    gain = {}
    for m in bag:
        for f, g in m.get_score(importance_type="total_gain").items():
            gain[f] = gain.get(f, 0.0) + g
    tot = sum(gain.values())
    emb_share = sum(g for f, g in gain.items() if f.startswith("emb_")) / tot
    top_emb = sorted(((g, f) for f, g in gain.items()
                      if f.startswith("emb_")), reverse=True)[:5]
    print(f"\n[enc] embedding gain share: {emb_share:.1%} of total; "
          f"top dims: {[(f, round(g/tot, 4)) for g, f in top_emb]}")

    with open(out_dir / "summary.json", "w") as fh:
        json.dump({"when": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "quick": bool(args.quick), "nrounds": nrounds,
                   "emb_gain_share": emb_share,
                   "timing": trep.to_dict(orient="records"),
                   "elapsed_min": round((time.time() - t0) / 60, 1)},
                  fh, indent=2, default=float)
    print(f"\n[enc] wrote {out_dir} ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
