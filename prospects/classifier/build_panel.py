"""Build the hazard panel once and save to disk.

Splitting the panel build into its own subprocess avoids the transient
memory pressure that compounds when training in the same process. The
saved panel (X, pids, years, joined) is then loaded by cv_runner.py per
fold without re-running the build.

Usage:
    python -m prospects.classifier.build_panel \\
        --db prospects_snapshot.db \\
        --out panel_v1.13.npz
"""
from __future__ import annotations

import argparse
import gc
import os
import pickle

import numpy as np
import pandas as pd

from prospects.classifier.architectures.survival import (
    MAX_OBS_YEAR, build_hazard_panel,
)
from prospects.storage import ProspectDB


def _load_corrected_positions(path="models/player_position_from_stats.csv"):
    """Modal position from season_stats (PA+IP weighted). Overrides
    prospects.primary_position which has known mislabels."""
    for p in (path, "player_position_from_stats.csv"):
        try:
            df = pd.read_csv(p)
            return dict(zip(df.player_id, df.pos_seasonstats))
        except FileNotFoundError:
            continue
    print(f"WARNING: position lookup not found at {path} or fallback — using raw prospect positions")
    return {}


def _apply_corrected_positions(prospects, pos_lookup):
    n_fix = 0
    for p in prospects:
        new_pos = pos_lookup.get(p["player_id"])
        if new_pos and new_pos != p.get("primary_position"):
            p["primary_position"] = new_pos
            n_fix += 1
    print(f"  position corrected for {n_fix:,} of {len(prospects):,} prospects")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="prospects_snapshot.db")
    ap.add_argument("--out", default="panel_v1.13.npz")
    ap.add_argument("--max-draft-year", type=int, default=2020)
    ap.add_argument("--max-year", type=int, default=MAX_OBS_YEAR)
    ap.add_argument("--partition", type=int, default=None,
                    help="If set, build only this partition (0 or 1) of "
                         "prospects. Writes a partial npz; combine with --merge.")
    ap.add_argument("--n-partitions", type=int, default=2)
    ap.add_argument("--merge", action="store_true",
                    help="Concatenate panel_v1.13.part{0..N-1}.npz into out.")
    args = ap.parse_args()

    if args.merge:
        parts_X = []
        parts_pids = []
        parts_years = []
        parts_joined = []
        for k in range(args.n_partitions):
            path = args.out.replace(".npz", f".part{k}.npz")
            joined_path = args.out.replace(".npz", f".part{k}.joined.pkl")
            with np.load(path, allow_pickle=True) as d:
                parts_X.append(d["X"])
                parts_pids.append(d["pids"])
                parts_years.append(d["years"])
            with open(joined_path, "rb") as fh:
                parts_joined.append(pickle.load(fh))
            print(f"loaded {path}: X={parts_X[-1].shape}")
        X = np.concatenate(parts_X, axis=0)
        del parts_X; gc.collect()
        pids = np.concatenate(parts_pids).tolist()
        years = np.concatenate(parts_years).tolist()
        joined = [j for part in parts_joined for j in part]
        print(f"Merged X: {X.shape}")
        np.savez_compressed(args.out, X=X,
                            pids=np.array(pids, dtype=object),
                            years=np.array(years, dtype=np.int32))
        with open(args.out.replace(".npz", ".joined.pkl"), "wb") as fh:
            pickle.dump(joined, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Wrote {args.out}")
        return

    db = ProspectDB(args.db)
    # Skip if this partition already complete — retries don't redo work
    if args.partition is not None:
        skip_path = args.out.replace(".npz", f".part{args.partition}.npz")
        if os.path.exists(skip_path):
            print(f"[part {args.partition}] already exists at {skip_path} — skipping")
            return

    # Cap thread oversubscription on sequential partitions
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    if args.partition is None:
        X, pids, years, joined = build_hazard_panel(
            db, max_draft_year=args.max_draft_year, max_year=args.max_year,
        )
    else:
        # Partition prospects deterministically by sorted player_id hash, then
        # build only this partition's slice.
        from prospects.classifier.architectures.survival import (
            build_windowed_features, N_FEATURES,
        )
        with db._connect() as conn:
            prospects = [dict(r) for r in conn.execute("""
                SELECT p.*, o.mlb_debut_year, o.year_established_mlb,
                       o.year_top_100, o.year_top_25,
                       o.year_all_star_once, o.year_all_star_three,
                       o.year_major_award, o.year_hof_trajectory,
                       o.events_json, o.final_mlb_year
                FROM prospects p
                JOIN career_outcomes o ON o.player_id = p.player_id
                WHERE (p.draft_year IS NOT NULL AND p.draft_year <= ?)
                   OR COALESCE(p.is_international, 0) = 1
            """, (args.max_draft_year,)).fetchall()]
            stats_rows = conn.execute("SELECT * FROM season_stats").fetchall()
        stats_by_pid: dict[str, list[dict]] = {}
        for s in stats_rows:
            d = dict(s)
            stats_by_pid.setdefault(d["player_id"], []).append(d)
        del stats_rows
        _apply_corrected_positions(prospects, _load_corrected_positions())
        prospects.sort(key=lambda p: p["player_id"])
        # This partition's slice of prospects
        prospects_slice = [
            p for i, p in enumerate(prospects)
            if i % args.n_partitions == args.partition
        ]
        del prospects; gc.collect()
        print(f"[part {args.partition}/{args.n_partitions}] "
              f"prospects in slice: {len(prospects_slice):,}")

        # Plan rows for this slice
        plan = []
        pids = []
        years = []
        joined = []
        for p in prospects_slice:
            stats = stats_by_pid.get(p["player_id"], [])
            dy = p.get("draft_year")
            if dy is None:
                yrs = [int(s["season_year"]) for s in stats
                       if s.get("season_year") is not None]
                if not yrs:
                    continue
                start_year = min(yrs)
            else:
                start_year = int(dy)
            for year in range(max(start_year + 1, 2005), args.max_year + 1):
                plan.append((p, stats, year))
                pids.append(p["player_id"])
                years.append(year)
                joined.append(p)

        n_rows = len(plan)
        print(f"[part {args.partition}] panel rows: {n_rows:,}")
        X = np.empty((n_rows, N_FEATURES), dtype=np.float32)
        # Smaller chunks reduce peak memory; aggressive gc avoids accumulation
        CHUNK = 2000
        for cs in range(0, n_rows, CHUNK):
            ce = min(cs + CHUNK, n_rows)
            for i in range(cs, ce):
                p, stats, year = plan[i]
                X[i, :] = build_windowed_features(
                    p, stats, year - 1, milb_only=True
                )
            # Free per-chunk references then gc twice (Python ref + cycle gc)
            gc.collect(); gc.collect()
            print(f"  [part {args.partition}] built {ce:,}/{n_rows:,} "
                  f"({100.0*ce/n_rows:.0f}%)", flush=True)
        del plan

    X = X.astype(np.float32, copy=False)
    print(f"X shape: {X.shape}  bytes: {X.nbytes / 1e6:.0f}MB")

    out_path = (args.out.replace(".npz", f".part{args.partition}.npz")
                if args.partition is not None else args.out)
    joined_path = out_path.replace(".npz", ".joined.pkl")
    np.savez_compressed(out_path, X=X,
                        pids=np.array(pids, dtype=object),
                        years=np.array(years, dtype=np.int32))
    print(f"Wrote {out_path}")
    with open(joined_path, "wb") as fh:
        pickle.dump(joined, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {joined_path}")
    del X, pids, years, joined
    gc.collect()


if __name__ == "__main__":
    main()
