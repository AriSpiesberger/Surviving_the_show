"""Chain script: run after the MiLB re-pull finishes.

Steps:
  1. Sync season_stats from prospects.db to prospects_snapshot.db
  2. Rebuild career_outcomes via the patched outcomes_loader
     (strict mlbam matching, populates year_top_100 / year_top_25
     from prospect_rankings)
  3. Sync career_outcomes to snapshot DB
  4. Print verification summary

After this script finishes, the data side is clean and v1.11 retrain
can run via:
    python -m prospects.classifier.architectures.survival \\
        --out models/event_classifiers_v1.11.pkl

Usage:
    python -m prospects.ingestion.backfills.post_repull_chain
"""
from __future__ import annotations

import sqlite3

from prospects.ingestion.backfills.post_dedup_verify import verify
from prospects.ingestion.outcomes_loader import pull_outcomes
from prospects.storage import ProspectDB


def _sync_table(src_path: str, dst_path: str, table: str,
                truncate: bool = True) -> int:
    """Copy all rows of `table` from src to dst. Returns rows copied."""
    src = sqlite3.connect(src_path)
    dst = sqlite3.connect(dst_path)
    try:
        # Ensure schema parity
        src_cols = [r[1] for r in src.execute(
            f"PRAGMA table_info({table})").fetchall()]
        dst_cols = [r[1] for r in dst.execute(
            f"PRAGMA table_info({table})").fetchall()]
        if set(src_cols) != set(dst_cols):
            print(f"  WARNING: schema mismatch for {table}: "
                  f"src has {set(src_cols) - set(dst_cols)} extra, "
                  f"dst has {set(dst_cols) - set(src_cols)} extra")
            cols = list(set(src_cols) & set(dst_cols))
        else:
            cols = src_cols
        col_csv = ",".join(cols)
        placeholders = ",".join(["?"] * len(cols))

        if truncate:
            dst.execute(f"DELETE FROM {table}")

        rows = src.execute(f"SELECT {col_csv} FROM {table}").fetchall()
        dst.executemany(
            f"INSERT INTO {table} ({col_csv}) VALUES ({placeholders})", rows
        )
        dst.commit()
        return len(rows)
    finally:
        src.close()
        dst.close()


def main():
    print("=" * 70)
    print("  POST-REPULL CHAIN")
    print("=" * 70)

    # 1. Sync season_stats from live -> snapshot
    print("\n[1/4] Syncing season_stats: prospects.db -> prospects_snapshot.db")
    n = _sync_table("prospects.db", "prospects_snapshot.db",
                    "season_stats", truncate=True)
    print(f"  copied {n:,} season_stats rows")

    # 2. Rebuild career_outcomes (writes to live DB)
    print("\n[2/4] Rebuilding career_outcomes with strict mlbam matching")
    print("       (also populates year_top_100 / year_top_25 from BBC ranks)")
    db = ProspectDB("prospects.db")
    summary = pull_outcomes(db, verbose=True)
    print(f"  outcomes summary: {summary}")

    # 3. Sync career_outcomes to snapshot
    print("\n[3/4] Syncing career_outcomes: prospects.db -> prospects_snapshot.db")
    n = _sync_table("prospects.db", "prospects_snapshot.db",
                    "career_outcomes", truncate=True)
    print(f"  copied {n:,} career_outcomes rows")

    # 4. Verify
    print("\n[4/4] Verification")
    verify("prospects.db")
    verify("prospects_snapshot.db")

    print("\nDONE. Ready for v1.11 retrain:")
    print("  python -u -m prospects.classifier.architectures.survival "
          "--out models/event_classifiers_v1.11.pkl")


if __name__ == "__main__":
    main()
