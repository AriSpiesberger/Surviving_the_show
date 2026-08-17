"""
prospects/data/backfills/woba_backfill.py
=========================================

Derives the stored advanced-stat columns on `season_stats` from the raw
counting stats the pull now keeps, and writes them back.

Why this exists: `features/advanced.py` deliberately does not touch the DB —
it derives rates from counts so weights can change without a re-pull. But
`woba` is a *stored* column that `features/windowed.py` reads as the
`woba_proxy` model feature, and nothing ever wrote it. Every row fell through
to the `obp + 0.5*iso` fallback. This is the missing link: run the derivation
over the table once after a pull, so the model sees real wOBA.

Only columns that already exist on `season_stats` are written. The other ~50
metrics `advanced.py` computes (swstr_pct, xfip, lob_pct, ...) have no column
and stay on-the-fly derivations — adding columns for them is a schema change,
not a backfill.

Write policy:
  * `woba` is always (re)computed. It is the point of the job, and the era
    weights it depends on can change between runs.
  * every other shared column (iso, babip, k_pct, era, fip, ...) is filled
    ONLY where currently NULL. Those came from the API and are authoritative;
    silently replacing them would move model features as a side effect of a
    wOBA backfill. `--overwrite` opts into recomputing them.

Hitter and pitcher metrics are applied independently per row: the upsert in
storage.py COALESCEs batting and pitching into a single (player_id, year,
level) row, so a two-way player has both pa > 0 and ip > 0.

Usage:
    python -m prospects.data.backfills.woba_backfill --dry-run   # report only
    python -m prospects.data.backfills.woba_backfill             # write woba
    python -m prospects.data.backfills.woba_backfill --overwrite # recompute all
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from prospects.config import REPO_ROOT
from prospects.features.advanced import hitter_advanced, pitcher_advanced

# Recomputed on every run — see the write policy in the module docstring.
ALWAYS = {"woba"}

CHUNK = 5000


def _table_columns(conn: sqlite3.Connection) -> set[str]:
    return {r[1] for r in conn.execute("PRAGMA table_info(season_stats)")}


def backfill(db_path: str, overwrite: bool = False,
             dry_run: bool = False, verbose: bool = True) -> dict:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cols = _table_columns(conn)

    # Which derived metrics actually have somewhere to land.
    probe = {"season_year": 2024, "pa": 100, "ab": 90, "hits": 25,
             "doubles": 5, "triples": 1, "home_runs": 3, "bb": 8, "ibb": 0,
             "hbp": 1, "sf": 1, "so": 20, "total_bases": 41, "ip": 50}
    h_targets = sorted(set(hitter_advanced(probe)) & cols)
    p_targets = sorted(set(pitcher_advanced(probe)) & cols)
    if verbose:
        print(f"[woba] hitter columns:  {', '.join(h_targets)}")
        print(f"[woba] pitcher columns: {', '.join(p_targets)}")

    total = conn.execute("SELECT COUNT(*) FROM season_stats").fetchone()[0]
    stats = {"rows": 0, "hitter_rows": 0, "pitcher_rows": 0,
             "woba_written": 0, "woba_null_inputs": 0, "cells_written": 0}
    updates: list[tuple] = []

    def _flush() -> None:
        """Apply the pending updates. Column set varies per row, so each
        update carries its own SET list; group by that list to batch."""
        if not updates or dry_run:
            updates.clear()
            return
        by_shape: dict[tuple, list] = {}
        for names, values, pk in updates:
            by_shape.setdefault(names, []).append(tuple(values) + pk)
        with conn:
            for names, payload in by_shape.items():
                sets = ", ".join(f"{n} = ?" for n in names)
                conn.executemany(
                    f"UPDATE season_stats SET {sets} WHERE player_id = ? "
                    f"AND season_year = ? AND level = ?", payload)
        updates.clear()

    cur = conn.execute("SELECT * FROM season_stats")
    while True:
        batch = cur.fetchmany(CHUNK)
        if not batch:
            break
        for r in batch:
            row = dict(r)
            stats["rows"] += 1
            derived: dict = {}

            pa = row.get("pa") or 0
            ip = row.get("ip") or 0
            if pa > 0:
                stats["hitter_rows"] += 1
                derived.update({k: v for k, v in hitter_advanced(row).items()
                                if k in h_targets})
            if ip > 0:
                stats["pitcher_rows"] += 1
                # A two-way row keeps its hitting wOBA: the pitcher dict has no
                # woba key, so update() below cannot clobber it.
                derived.update({k: v for k, v in pitcher_advanced(row).items()
                                if k in p_targets})

            names, values = [], []
            for col, val in derived.items():
                if val is None:
                    continue
                if col in ALWAYS or overwrite or row.get(col) is None:
                    names.append(col)
                    values.append(val)
            if "woba" in names:
                stats["woba_written"] += 1
            elif pa > 0:
                stats["woba_null_inputs"] += 1

            if names:
                stats["cells_written"] += len(names)
                updates.append((tuple(names), values,
                                (row["player_id"], row["season_year"],
                                 row["level"])))
        _flush()
        if verbose:
            print(f"  {stats['rows']:,}/{total:,} rows "
                  f"({stats['woba_written']:,} wOBA)", flush=True)
    _flush()
    conn.close()
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(REPO_ROOT / "prospects.db"))
    ap.add_argument("--overwrite", action="store_true",
                    help="Also recompute shared columns (iso, babip, k_pct, "
                         "era, fip, ...) that already hold API values. Off by "
                         "default: those are authoritative and moving them "
                         "moves model features.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be written; change nothing.")
    args = ap.parse_args()

    if not Path(args.db).exists():
        raise SystemExit(f"[woba] no such db: {args.db}")

    print(f"[woba] {args.db}"
          f"{'  (DRY RUN)' if args.dry_run else ''}"
          f"{'  (OVERWRITE)' if args.overwrite else ''}")
    s = backfill(args.db, overwrite=args.overwrite, dry_run=args.dry_run)

    print(f"\n[woba] rows scanned     {s['rows']:,}")
    print(f"[woba]   hitter rows     {s['hitter_rows']:,}")
    print(f"[woba]   pitcher rows    {s['pitcher_rows']:,}")
    print(f"[woba] wOBA computed     {s['woba_written']:,}")
    print(f"[woba] wOBA missing raw  {s['woba_null_inputs']:,}  "
          f"(hitter rows whose counting stats are absent)")
    print(f"[woba] cells written     {s['cells_written']:,}"
          f"{'  (dry run — nothing written)' if args.dry_run else ''}")


if __name__ == "__main__":
    main()
