"""Model momentum: same model, one year less data — Δ P(debut) per player.

Scores every buy-universe player twice with the FROZEN current stack:
  p_now  = P(debut <= 3y) at snap 2026 (from the current buy list)
  p_then = P(debut <= 3y) at a counterfactual snap 2025 (stats <= 2025 only)

Δ = p_now - p_then is pure player trajectory — no model-version or
threshold noise (unlike a sheet-to-sheet diff). Big positive Δ = "putting
it together" (the Jake Cunningham 4.6%->21.5% pattern); big negative Δ =
eroding stock the flat probability column hides.

Output: runs/current/buy_lists/momentum.csv + top movers printed.

    python -m prospects.buylist.momentum
"""
from __future__ import annotations

import argparse
import pickle
import sqlite3
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from prospects import config
from prospects.model.hazards import landmark as lm
from prospects.model.hazards.survival import EXIT_KEY
from prospects.model.pipelines.oof import _HK_EVENTS, _HK_STEPS
from prospects.model.pipelines.stage_a import _ev_name
from prospects.model.joint import add_cond_cols, prep_base
from prospects.model.joint2 import attach_raw_features, load_calibrators
from prospects.model.train.exp_cdf_timing2 import stamp_extra_cols

_RUN = config.run()
DB = str(config.model_db())
THEN_SNAP = 2025
H = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",
                    default=str(_RUN.root / "buy_lists" / "momentum.csv"))
    args = ap.parse_args()
    t0 = time.time()

    now = pd.read_csv(_RUN.root / "buy_lists" / "all_scored.csv")
    pids = set(now.player_id)
    print(f"[momentum] {len(pids):,} buy-universe players")

    con = sqlite3.connect(DB)
    cur = con.execute(
        "SELECT p.*, o.mlb_debut_year, o.year_established_mlb, "
        "o.year_top_100, o.year_top_25, o.year_all_star_once, "
        "o.year_all_star_three, o.year_major_award, o.year_hof_trajectory, "
        "o.events_json, o.final_mlb_year "
        "FROM prospects p JOIN career_outcomes o "
        "ON o.player_id = p.player_id")
    cols = [d[0] for d in cur.description]
    prospects = [dict(zip(cols, x)) for x in cur.fetchall()
                 if x[cols.index("player_id")] in pids]
    scur = con.execute("SELECT * FROM season_stats")
    scols = [d[0] for d in scur.description]
    stats_by_pid: dict = {}
    for x in scur.fetchall():
        d = dict(zip(scols, x))
        if d["player_id"] in pids and (d.get("season_year") or 0) <= THEN_SNAP:
            stats_by_pid.setdefault(d["player_id"], []).append(d)
    con.close()
    for p in prospects:
        p["_top100_rankings"] = []
        p["_org_rankings"] = []
    # counterfactual universe: players who existed by THEN_SNAP (entry <= it)
    def _entry(p):
        dy = p.get("draft_year")
        if dy is not None and not p.get("is_international"):
            return int(dy)
        yrs = [s["season_year"] for s in stats_by_pid.get(p["player_id"], [])
               if s.get("season_year")]
        return min(yrs) if yrs else (int(dy) if dy else None)
    cohort = [p for p in prospects
              if (_entry(p) or 9999) <= THEN_SNAP]
    print(f"  {len(cohort):,} existed by {THEN_SNAP} "
          f"(rest get momentum=NaN)")

    with open(_RUN.hazards, "rb") as fh:
        hazards = pickle.load(fh)
    print(f"[momentum] hazard-scoring counterfactual snap {THEN_SNAP} "
          f"[{(time.time()-t0)/60:.1f}m]")
    out = lm.predict_cumulative_batch_landmark(
        hazards, cohort,
        {p["player_id"]: stats_by_pid.get(p["player_id"], [])
         for p in cohort},
        current_year=THEN_SNAP, horizon=15)
    del hazards

    rows = []
    ev_keys = [k for k in out if not isinstance(k, tuple)]
    for i, p in enumerate(cohort):
        row = {"player_id": p["player_id"], "name": p.get("name"),
               "draft_year": p.get("draft_year"),
               "draft_round": p.get("draft_round"),
               "is_international": int(p.get("is_international") or 0),
               "snap_year": THEN_SNAP,
               "snap_offset": THEN_SNAP - (_entry(p) or THEN_SNAP),
               "years_fwd": 0, "entry_year": _entry(p)}
        per = {}
        for e in ev_keys:
            en = _ev_name(e)
            row[f"p_{en}"] = float(out[e][i])
            row[f"eligible_{en}"] = 1
            row[f"trigger_{en}"] = np.nan
            mt = out.get(("mean_t", e))
            st_ = out.get(("sd_t", e))
            if mt is not None:
                row[f"mean_t_{en}"] = float(mt[i])
            if st_ is not None:
                row[f"sd_t_{en}"] = float(st_[i])
            hk = out.get(("haz_k", e))
            if hk is not None and en in _HK_EVENTS:
                for j in range(_HK_STEPS):
                    row[f"hk{j+1}_{en}"] = float(hk[i, j])
            per[en] = float(out[e][i])
        row["p_STAR_PLUS_ELITE"] = 1 - (1 - per.get("STAR", 0)) * \
            (1 - per.get("ELITE", 0))
        row["eligible_STAR_PLUS_ELITE"] = 1
        row["trigger_STAR_PLUS_ELITE"] = np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    df = prep_base(df, DB)

    with open(_RUN.models / "joint_xgb_v2.4.pkl", "rb") as fh:
        bundle = pickle.load(fh)
    cal = load_calibrators(
        _RUN.models / "calibrators_v2.4.pkl")["calibrators"].get("MLB_DEBUT")
    feats = bundle["feature_names"]
    print(f"[momentum] raw features + joint sweep "
          f"[{(time.time()-t0)/60:.1f}m]")
    df = attach_raw_features(df, DB, bundle["keep_raw"], verbose=True)
    sub = stamp_extra_cols(add_cond_cols(df, H))
    X = sub[feats].values.astype(np.float32)
    d = xgb.DMatrix(X, feature_names=list(feats))
    p_raw = np.mean([m.predict(d) for m in bundle["models"]], axis=0)[:, 1]
    yip = df["snap_offset"].to_numpy()
    p_then = (cal.predict(p_raw, np.full(len(df), H), yip)
              if cal is not None else p_raw)

    mom = pd.DataFrame({"player_id": df["player_id"],
                        f"p_debut_{H}y_then": p_then})
    m = now.merge(mom, on="player_id", how="left")
    m["momentum"] = m["p_MLB_DEBUT"] - m[f"p_debut_{H}y_then"]
    keep = ["player_id", "name", "bucket", "cur_level_2026",
            f"p_debut_{H}y_then", "p_MLB_DEBUT", "momentum",
            "passes_filter", "ebay_price_median"]
    m[keep].sort_values("momentum", ascending=False).to_csv(
        args.out, index=False)
    print(f"[momentum] wrote {args.out} "
          f"({int(m.momentum.notna().sum()):,} scored)")

    mm = m[m.momentum.notna()].sort_values("momentum", ascending=False)
    print(f"\n=== TOP RISERS (same model, +1y of data) ===")
    print(mm[keep[1:]].head(15).to_string(index=False))
    print(f"\n=== TOP FADERS ===")
    print(mm[keep[1:]].tail(10).to_string(index=False))
    print(f"\n({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
