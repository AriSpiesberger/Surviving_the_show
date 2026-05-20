"""Incremental panel build: extend panel_v1.13.npz with 2021-2025 drafted
players (who were excluded by max_draft_year=2020). Avoids the full
~500k-row rebuild that has been segfaulting under memory pressure.

Output: panel_v1.14c.npz + panel_v1.14c.joined.pkl
"""
from __future__ import annotations

import argparse
import gc
import pickle
import sys

import numpy as np

import pandas as pd

from prospects.classifier.architectures.survival import (
    MAX_OBS_YEAR, N_FEATURES, build_windowed_features,
)
from prospects.classifier.build_panel import (
    _apply_corrected_positions, _load_corrected_positions,
)
from prospects.storage import ProspectDB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="prospects_snapshot.db")
    ap.add_argument("--base", default="panel_v1.13.npz",
                    help="Existing panel to extend")
    ap.add_argument("--out", default="panel_v1.14c.npz")
    ap.add_argument("--new-draft-min", type=int, default=2021,
                    help="Add drafted players with draft_year >= this")
    ap.add_argument("--new-draft-max", type=int, default=2025)
    ap.add_argument("--min-year", type=int, default=2005)
    ap.add_argument("--max-year", type=int, default=MAX_OBS_YEAR)
    args = ap.parse_args()

    print(f"Loading base panel from {args.base}")
    with np.load(args.base, allow_pickle=True) as d:
        X_base = d["X"]
        pids_base = d["pids"].tolist()
        years_base = d["years"].tolist()
    with open(args.base.replace(".npz", ".joined.pkl"), "rb") as fh:
        joined_base = pickle.load(fh)
    print(f"  base: {X_base.shape[0]:,} rows, "
          f"{len(set(pids_base)):,} players")
    existing_pids = set(pids_base)

    print(f"Querying drafted players with draft_year in "
          f"[{args.new_draft_min}, {args.new_draft_max}]")
    db = ProspectDB(args.db)
    with db._connect() as conn:
        new_prospects = [dict(r) for r in conn.execute("""
            SELECT p.*, o.mlb_debut_year, o.year_established_mlb,
                   o.year_top_100, o.year_top_25,
                   o.year_all_star_once, o.year_all_star_three,
                   o.year_major_award, o.year_hof_trajectory,
                   o.events_json, o.final_mlb_year
            FROM prospects p
            JOIN career_outcomes o ON o.player_id = p.player_id
            WHERE p.draft_year IS NOT NULL
              AND p.draft_year >= ?
              AND p.draft_year <= ?
        """, (args.new_draft_min, args.new_draft_max)).fetchall()]
        stats_rows = conn.execute("SELECT * FROM season_stats").fetchall()
        try:
            rank_rows = conn.execute(
                "SELECT player_id, year, rank, source FROM prospect_rankings"
            ).fetchall()
        except Exception:
            rank_rows = []

    # Only include players actually NEW (not already in base panel)
    new_prospects = [p for p in new_prospects
                     if p["player_id"] not in existing_pids]
    print(f"  {len(new_prospects):,} new drafted players to add")
    _apply_corrected_positions(new_prospects, _load_corrected_positions())

    stats_by_pid: dict[str, list[dict]] = {}
    for s in stats_rows:
        d = dict(s)
        stats_by_pid.setdefault(d["player_id"], []).append(d)
    rankings_by_pid: dict[str, list] = {}
    for r in rank_rows:
        rankings_by_pid.setdefault(r[0], []).append((r[1], r[2], r[3]))
    for p in new_prospects:
        p["_top100_rankings"] = rankings_by_pid.get(p["player_id"], [])

    # Plan rows for new players (drafted players use draft_year as start)
    plan: list[tuple[dict, list, int]] = []
    new_pids: list[str] = []
    new_years: list[int] = []
    new_joined: list[dict] = []
    for p in new_prospects:
        stats = stats_by_pid.get(p["player_id"], [])
        start_year = int(p["draft_year"])
        for year in range(max(start_year + 1, args.min_year),
                          args.max_year + 1):
            plan.append((p, stats, year))
            new_pids.append(p["player_id"])
            new_years.append(year)
            new_joined.append(p)
    n_new = len(plan)
    print(f"  {n_new:,} new panel rows to build")

    X_new = np.empty((n_new, N_FEATURES), dtype=np.float32)
    CHUNK = 1000
    for chunk_start in range(0, n_new, CHUNK):
        chunk_end = min(chunk_start + CHUNK, n_new)
        for i in range(chunk_start, chunk_end):
            p, stats, year = plan[i]
            vec = build_windowed_features(p, stats, year - 1, milb_only=True)
            X_new[i, :] = vec
        gc.collect()
        pct = 100.0 * chunk_end / n_new
        print(f"  built {chunk_end:,}/{n_new:,} ({pct:.0f}%)", flush=True)
    del plan
    gc.collect()

    print(f"Concatenating: base {X_base.shape[0]:,} + new {X_new.shape[0]:,}")
    X_all = np.concatenate([X_base, X_new], axis=0)
    pids_all = pids_base + new_pids
    years_all = years_base + new_years
    joined_all = joined_base + new_joined

    print(f"Saving {args.out}: X shape = {X_all.shape}")
    np.savez_compressed(
        args.out,
        X=X_all,
        pids=np.array(pids_all, dtype=object),
        years=np.array(years_all, dtype=np.int32),
    )
    joined_path = args.out.replace(".npz", ".joined.pkl")
    with open(joined_path, "wb") as fh:
        pickle.dump(joined_all, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {args.out} and {joined_path}")


if __name__ == "__main__":
    main()
