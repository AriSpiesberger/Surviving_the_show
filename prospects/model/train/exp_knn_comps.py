"""Historical-comparables features (feature construction, 2026-09-24).

For every (player, snapshot S) row, find the k most similar EARLIER player-snapshots (same
hitter/pitcher pool, standardized age / level / age-for-level / stats-vs-level / pedigree /
rank features) and use their RESOLVED outcomes as features:

  knn_debut3   share of comps that debuted within 3 years   (comps from snapshots <= S-3)
  knn_top3     share that made a top-100 list within 3 years (<= S-3)
  knn_est6     share that became established within 6 years  (<= S-6)
  knn_star6    share that reached star+ within 6 years       (<= S-6)
  knn_dist     mean distance to the comps (how typical the player is)

Only information available at S: every comp's outcome window closed by S. Comps come only from
FIT players (never val), so the held-out evaluation stays clean. Output: a CSV keyed by
player_id, snap_year for exp_joint_feats / exp_resid_value --extra.

    python -m prospects.model.train.exp_knn_comps
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from prospects import config
from prospects.config import REPO_ROOT
from prospects.model.joint import prep_base, realized_by_h
from prospects.model.joint2 import attach_raw_features
from prospects.model.train.joint_xgb import _prep_train

_RUN = config.run()
DB = str(config.model_db())
FEATS = ["age_at_snap_centered", "years_in_pro", "draft_round_filled", "is_ifa", "rw_level_rank_yT",
         "rw_age_vs_level_yT", "rw_best_org_rank", "rw_pa_yT", "rw_ip_yT"]
HIT = ["rw_woba_vs_level_yT", "rw_iso_vs_level_yT", "rw_k_pct_vs_level_yT", "rw_bb_pct_vs_level_yT"]
PIT = ["rw_fip_vs_level_yT", "rw_k9_vs_level_yT", "rw_bb9_vs_level_yT", "rw_era_vs_level_yT"]
K = 50


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--out", default=str(REPO_ROOT / "reference" / "knn" / "knn_comps.csv"))
    a = ap.parse_args()
    t0 = time.time()
    raw = sorted({c for c in FEATS + HIT + PIT if c.startswith("rw_")} | {"rw_is_pitcher"})
    fit = _prep_train(pd.read_csv(_RUN.oof_stacked_long, low_memory=False), DB, 2020)
    aug = prep_base(pd.read_csv(_RUN.training / "recent_long.csv", low_memory=False), DB)
    val = prep_base(pd.read_csv(_RUN.oof_val_long, low_memory=False), DB, max_entry=2020)
    snap = prep_base(pd.read_csv(_RUN.snap_long(2026), low_memory=False), DB)
    frames = {"fit": pd.concat([fit, aug], ignore_index=True), "val": val, "snap": snap}
    for k_, f in frames.items():
        f = attach_raw_features(f, DB, raw, verbose=False).drop_duplicates(["player_id", "snap_year"])
        f["src"] = k_
        frames[k_] = f
    allr = pd.concat(frames.values(), ignore_index=True)
    allr["rw_best_org_rank"] = np.log1p(allr["rw_best_org_rank"].fillna(60).clip(upper=60))
    pool = frames["fit"].copy()
    pool["rw_best_org_rank"] = np.log1p(pool["rw_best_org_rank"].fillna(60).clip(upper=60))
    labels = {}
    for ev, h in (("MLB_DEBUT", 3), ("TOP_100_PROSPECT", 3), ("ESTABLISHED_MLB", 6), ("STAR_PLUS_ELITE", 6)):
        ok = (pool.years_fwd >= h) & (pool.get(f"eligible_{ev}", 1) == 1)
        lab = pd.Series(np.nan, index=pool.index)
        lab[ok] = np.asarray(realized_by_h(pool[ok], ev, h), dtype=float)
        labels[(ev, h)] = lab.to_numpy()
    print(f"[knn] pool {len(pool):,} fit snapshots; query {len(allr):,} rows  [{(time.time() - t0) / 60:.1f}m]", flush=True)

    out = []
    for is_pit, stat in ((0, HIT), (1, PIT)):
        cols = FEATS + stat
        P = pool[pool.rw_is_pitcher.fillna(0) == is_pit]
        Q = allr[allr.rw_is_pitcher.fillna(0) == is_pit]
        mu, sd = P[cols].mean(), P[cols].std() + 1e-6
        zP = ((P[cols] - mu) / sd).fillna(0).clip(-5, 5).to_numpy()
        zQ = ((Q[cols] - mu) / sd).fillna(0).clip(-5, 5).to_numpy()
        py = P.snap_year.to_numpy()
        res = {c: np.full(len(Q), np.nan) for c in ("knn_debut3", "knn_top3", "knn_est6", "knn_star6", "knn_dist")}
        qy = Q.snap_year.to_numpy()
        for S in np.unique(qy):
            qi = np.where(qy == S)[0]
            for (ev, h), name in ((("MLB_DEBUT", 3), "knn_debut3"), (("TOP_100_PROSPECT", 3), "knn_top3"),
                                  (("ESTABLISHED_MLB", 6), "knn_est6"), (("STAR_PLUS_ELITE", 6), "knn_star6")):
                lab = labels[(ev, h)][P.index.to_numpy() - P.index.min()] if False else labels[(ev, h)][pool.index.get_indexer(P.index)]
                cand = np.where((py <= S - h) & np.isfinite(lab))[0]
                if len(cand) < a.k:
                    continue
                nn = NearestNeighbors(n_neighbors=a.k).fit(zP[cand])
                dist, idx = nn.kneighbors(zQ[qi])
                res[name][qi] = lab[cand][idx].mean(axis=1)
                if name == "knn_debut3":
                    res["knn_dist"][qi] = dist.mean(axis=1)
        o = Q[["player_id", "snap_year"]].copy()
        for c, v in res.items():
            o[c] = v
        out.append(o)
        print(f"[knn] {'pitchers' if is_pit else 'hitters'} done  [{(time.time() - t0) / 60:.1f}m]", flush=True)
    f = pd.concat(out, ignore_index=True).drop_duplicates(["player_id", "snap_year"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    f.to_csv(a.out, index=False)
    print(f"[knn] wrote {a.out}: {len(f):,} rows; coverage {f.knn_debut3.notna().mean():.1%}")


if __name__ == "__main__":
    main()
