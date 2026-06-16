"""Per-snap walk-forward for the HELD-OUT 2021-entry cohort.

The 2021 draft/IFA class entered after the training cutoff (entry <= 2020), so
it's a clean forward-test: score these players at each snap year 2021..2026 with
the PRODUCTION model (100% hazards + censoring-corrected XGB) and watch the
predictions track the realized outcomes as the cohort matures.

Writes evaluation/v2.0b_landmark/walkforward_2021entry_by_year/snap{YYYY}.csv

    python -m scripts_v17.validate.gen_wf2021_cohort
"""
import pickle
from pathlib import Path

import pandas as pd

from prospects.storage import ProspectDB
from scripts_v17.train.train_v2_0b_prod import score_snap_with_landmark
from scripts_v17.validate.regen_full_eval_v2_0b import (
    _prep_for_xgb, _score_xgb, _join_current_level,
    _build_walkforward_2021_entry, OUT_DIR, DB, XGB_PKL,
)

HAZ = "models/event_classifiers_v2.0b_prod.pkl"   # 100% production hazards
TMP = Path("scratch/wf2021")


def _entry_year(p, stats_by_pid):
    dy = p.get("draft_year")
    if dy is not None and int(p.get("is_international") or 0) == 0:
        return int(dy)
    yrs = [int(s["season_year"]) for s in stats_by_pid.get(p["player_id"], [])
           if s.get("season_year") is not None
           and (s.get("level") or "").upper() != "MLB"]
    if yrs:
        return int(min(yrs))
    return int(dy) if dy is not None else None


def main():
    hazards = pickle.load(open(HAZ, "rb"))
    db = ProspectDB(str(DB))
    with db._connect() as conn:
        prospects = [dict(r) for r in conn.execute("""
            SELECT p.*, o.mlb_debut_year, o.year_established_mlb, o.year_top_100,
                   o.year_top_25, o.year_all_star_once, o.year_all_star_three,
                   o.year_major_award, o.year_hof_trajectory, o.events_json,
                   o.final_mlb_year
            FROM prospects p LEFT JOIN career_outcomes o ON o.player_id=p.player_id
            WHERE (p.draft_year IS NOT NULL AND p.draft_year BETWEEN 2010 AND 2026)
               OR COALESCE(p.is_international,0)=1
        """).fetchall()]
        stats_rows = conn.execute("SELECT * FROM season_stats").fetchall()
    stats_by_pid: dict = {}
    for s in stats_rows:
        d = dict(s)
        stats_by_pid.setdefault(d["player_id"], []).append(d)

    cohort = [p for p in prospects if _entry_year(p, stats_by_pid) == 2021]
    print(f"2021-entry held-out cohort: {len(cohort):,} players")

    TMP.mkdir(parents=True, exist_ok=True)
    longs = []
    for sy in range(2021, 2027):
        out = TMP / f"snap{sy}.csv"
        score_snap_with_landmark(hazards, cohort, stats_by_pid,
                                 snap_year=sy, out_csv=out, verbose=False)
        longs.append(pd.read_csv(out))
    big = pd.concat(longs, ignore_index=True)
    big.to_csv("results/scored/wf2021_cohort_long.csv", index=False)

    df = _prep_for_xgb(big, str(DB), 9999)      # keep all entry years
    df = _score_xgb(df, XGB_PKL)
    df = _join_current_level(df, str(DB))

    od = OUT_DIR / "walkforward_2021entry_by_year"
    od.mkdir(parents=True, exist_ok=True)
    for sy in range(2021, 2027):
        w = _build_walkforward_2021_entry(df, sy)
        if len(w):
            w.to_csv(od / f"snap{sy}.csv", index=False)
            print(f"  snap{sy}.csv: {len(w)} event-rows")


if __name__ == "__main__":
    main()
