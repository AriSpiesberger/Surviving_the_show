"""Score recent entry cohorts (post-training-cutoff) into an augmentation long.

Players entering after the training cutoff (2021+) have resolved short-h
outcomes that production training discarded. The walk-forward A/B
(exp_walkforward3) showed adding their resolved (row,h) pairs to joint
training lifts out-of-era debut@3 AP by +0.04..+0.07 AND cuts era-drift
calib (they carry the current promotion regime). This module produces
`runs/current/training/recent_long.csv` — same schema as the OOF longs, so
`exp_cdf_timing5 --aug-long` can concat it.

Honesty: cohorts are scored with a VAL-EXCLUDED <=2020-trained hazards model
(default: the clean hz0_default fold-0 hazards), so the held-out val metrics
stay untouched by the augmentation.

    python -m prospects.model.train.score_recent_cohorts
"""
from __future__ import annotations

import argparse
import pickle
import sqlite3
import time
from pathlib import Path

from prospects import config
from prospects.config import REPO_ROOT
from prospects.model.pipelines import oof as oof_mod
from prospects.model.pipelines.oof import _entry_year

_RUN = config.run()
DB = str(REPO_ROOT / "prospects_snapshot.db")
DEFAULT_HAZARDS = (REPO_ROOT / "runs" / "hz0_default" / "scratch" / "oof"
                   / "fold0_hazards.pkl")
OUT_CSV = _RUN.training / "recent_long.csv"


def load_universe(db: str):
    con = sqlite3.connect(db)
    cur = con.execute("""
        SELECT p.*, o.mlb_debut_year, o.year_established_mlb,
               o.year_top_100, o.year_top_25,
               o.year_all_star_once, o.year_all_star_three,
               o.year_major_award, o.year_hof_trajectory,
               o.events_json, o.final_mlb_year
        FROM prospects p
        JOIN career_outcomes o ON o.player_id = p.player_id""")
    cols = [d[0] for d in cur.description]
    prospects = [dict(zip(cols, r)) for r in cur.fetchall()]
    scur = con.execute("SELECT * FROM season_stats")
    scols = [d[0] for d in scur.description]
    stats_by_pid: dict = {}
    for r in scur.fetchall():
        d = dict(zip(scols, r))
        stats_by_pid.setdefault(d["player_id"], []).append(d)
    try:
        ranks = con.execute(
            "SELECT player_id, CAST(substr(as_of,1,4) AS INTEGER), "
            "overall_rank, source FROM rankings_history "
            "WHERE overall_rank IS NOT NULL").fetchall()
    except Exception:
        ranks = []
    try:
        orgs = con.execute(
            "SELECT player_id, as_of, org_rank FROM rankings_history "
            "WHERE org_rank IS NOT NULL").fetchall()
    except Exception:
        orgs = []
    con.close()
    rb: dict = {}
    for r in ranks:
        rb.setdefault(r[0], []).append((r[1], r[2], r[3]))
    ob: dict = {}
    for r in orgs:
        try:
            yr = int(str(r[1])[:4])
        except (ValueError, TypeError):
            continue
        ob.setdefault(r[0], []).append((yr, int(r[2])))
    for p in prospects:
        p["_top100_rankings"] = rb.get(p["player_id"], [])
        p["_org_rankings"] = ob.get(p["player_id"], [])
    return prospects, stats_by_pid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hazards", default=str(DEFAULT_HAZARDS),
                    help="Val-excluded <=2020-trained landmark hazards pkl.")
    ap.add_argument("--min-entry", type=int, default=2021)
    ap.add_argument("--max-entry", type=int, default=2025)
    ap.add_argument("--observe-through", type=int, default=None,
                    help="Default: MAX_OBS_YEAR from the hazard module.")
    ap.add_argument("--out", default=str(OUT_CSV))
    args = ap.parse_args()
    t0 = time.time()

    from prospects.model.hazards.survival import MAX_OBS_YEAR
    obs = args.observe_through or MAX_OBS_YEAR

    print(f"[recent] loading universe + hazards")
    prospects, stats_by_pid = load_universe(DB)
    with open(args.hazards, "rb") as fh:
        hazards = pickle.load(fh)

    entry_by_pid = {p["player_id"]: _entry_year(p, stats_by_pid)
                    for p in prospects}
    pid_set = {p for p, e in entry_by_pid.items()
               if e is not None and args.min_entry <= e <= args.max_entry}
    print(f"[recent] cohort entry {args.min_entry}-{args.max_entry}: "
          f"{len(pid_set):,} players; observe_through={obs}")

    out = Path(args.out)
    n = oof_mod._score_checkpointed(
        hazards, prospects, stats_by_pid, pid_set, out,
        out.parent / "recent_partial",
        max_entry_year=args.max_entry, observe_through=obs,
        max_offset=10, horizon=15)
    print(f"[recent] wrote {out}: {n:,} rows "
          f"({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
