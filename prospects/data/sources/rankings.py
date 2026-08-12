"""
prospects/data/sources/rankings.py
==================================

Load the baseballcube rankings history into the ``rankings_history`` table.

The CSV (``reference/baseballcube/rankings_history.csv``) is the matched output
of ``baseballcube_integrate``: one row per (player, year, source) carrying the
MLB-wide ``overall_rank`` (global top-100) and the per-org ``org_rank``
(inter-team top-N). Loading it feeds two things:

  * the model's ``org_rank`` features (panel/oof/prod all read rankings_history), and
  * the ``year_top_100`` / ``year_top_25`` labels, once ``outcomes`` reruns
    (it derives them from ``overall_rank`` in this table).

This step was missing from the pull, so ``rankings_history`` sat empty and the
TOP_100 / TOP_25 events had zero positives. Belongs in ``--phase all``.

Usage:
    python -m prospects.data.sources.rankings --db prospects.db
"""

from __future__ import annotations

import argparse
import csv
import sqlite3

from prospects.config import REPO_ROOT

DEFAULT_CSV = REPO_ROOT / "reference" / "baseballcube" / "rankings_history.csv"


def _toint(v):
    v = (v or "").strip()
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def load_rankings(db_path: str, csv_path=DEFAULT_CSV, verbose: bool = True) -> dict:
    """Upsert the rankings CSV into rankings_history. Returns a stats dict."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    data = [
        (r["player_id"], r["as_of"], (r.get("source") or "BBC"),
         _toint(r.get("overall_rank")), _toint(r.get("org_rank")),
         _toint(r.get("list_size")) or 100)
        for r in rows if (r.get("player_id") or "").strip() and (r.get("as_of") or "").strip()
    ]
    con = sqlite3.connect(db_path)
    con.executemany(
        "INSERT INTO rankings_history "
        "(player_id, as_of, source, overall_rank, org_rank, list_size) "
        "VALUES (?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(player_id, as_of, source) DO UPDATE SET "
        "overall_rank=excluded.overall_rank, org_rank=excluded.org_rank, "
        "list_size=excluded.list_size",
        data,
    )
    con.commit()
    total = con.execute("SELECT COUNT(*) FROM rankings_history").fetchone()[0]
    t100 = con.execute(
        "SELECT COUNT(DISTINCT player_id) FROM rankings_history "
        "WHERE overall_rank IS NOT NULL AND overall_rank <= 100").fetchone()[0]
    orgn = con.execute(
        "SELECT COUNT(*) FROM rankings_history WHERE org_rank IS NOT NULL").fetchone()[0]
    con.close()
    if verbose:
        print(f"[rankings] upserted {len(data):,} rows -> rankings_history {total:,} rows; "
              f"{t100:,} players ever global top-100; {orgn:,} rows with org_rank")
    return {"loaded": len(data), "total": total, "top100_players": t100}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="prospects.db")
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    args = ap.parse_args()
    load_rankings(args.db, args.csv)


if __name__ == "__main__":
    main()
