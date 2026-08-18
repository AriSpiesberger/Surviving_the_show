"""
prospects/data/backfills/prune_orphan_stats.py
==============================================

Deletes season_stats rows left behind by the re-draft dedup.

A player drafted twice — taken once, didn't sign, went to college, taken again
and signed — gets two draft records. `pull --phase dedup` collapses them and
keeps one prospect row, but the season_stats written under the discarded id
stay behind, referencing a player_id that no longer exists in `prospects`.
18,060 such rows, 2,698 player_ids.

They cost nothing in training: the panel builds from prospects JOIN
career_outcomes, so an orphan is never read. They are not harmless, though —
features/percentiles.py ranks every row in a (level, season_year) cohort, and
a duplicate of a player who is ALSO in the cohort under their live id
double-counts that player in the reference distribution.

Canonical identity is the LATEST draft record, so the earlier record's rows
are the ones to drop.

Deletes only what it can prove is a duplicate: the orphan's every
(season_year, level) must already exist under a live prospect parsed to the
same name. Everything else is kept, because a shared name is not a shared
player. There are two Matt Halls — one drafted 2005 with 2008-09 stats, one
drafted 2015 with 2016-17 — and merging them on name is exactly the collision
that made milb.py go strict-mlbam-only in v1.10. Those disjoint cases are real
player-seasons for players who left the prospect universe; they stay, and they
legitimately enrich the cohorts as league context.

Usage:
    python -m prospects.data.backfills.prune_orphan_stats --dry-run
    python -m prospects.data.backfills.prune_orphan_stats

Re-run percentile_backfill afterwards: the cohorts change.
"""

from __future__ import annotations

import argparse
import collections
import re
import sqlite3
from pathlib import Path

from prospects.config import REPO_ROOT

ID_RE = re.compile(r"draft_(\d{4})_(.+?)_r(\d+)p(\d+)$")


def _parse(pid: str) -> tuple[int | None, str]:
    """(draft_year, name) from a synthetic draft id, else (None, id)."""
    m = ID_RE.match(pid)
    return (int(m.group(1)), m.group(2)) if m else (None, pid)


def plan(conn: sqlite3.Connection) -> tuple[list[str], dict]:
    live_by_name: dict[str, list[tuple[int | None, str]]] = collections.defaultdict(list)
    for (pid,) in conn.execute("SELECT player_id FROM prospects"):
        yr, nm = _parse(pid)
        live_by_name[nm].append((yr, pid))

    rowsets: dict[str, set] = collections.defaultdict(set)
    for pid, yr, lv in conn.execute(
            "SELECT player_id, season_year, level FROM season_stats"):
        rowsets[pid].add((yr, lv))

    orphans = [p for (p,) in conn.execute(
        "SELECT DISTINCT s.player_id FROM season_stats s WHERE NOT EXISTS "
        "(SELECT 1 FROM prospects p WHERE p.player_id = s.player_id)")]

    doomed: list[str] = []
    stats = collections.Counter()
    for o in orphans:
        oyr, nm = _parse(o)
        candidates = [c for c in live_by_name.get(nm, []) if c[0] is not None]
        if oyr is None or not candidates:
            stats["kept: no live record of that name"] += 1
            continue
        if any(y == oyr for y, _ in candidates):
            # Same name AND same draft year — cannot tell which record the
            # rows belong to. Leaving them is the reversible choice.
            stats["kept: same name and draft year"] += 1
            continue
        _, canonical = max(candidates)  # latest draft wins
        mine = rowsets.get(o, set())
        if mine and mine <= rowsets.get(canonical, set()):
            doomed.append(o)
            stats["deleted: duplicated by the canonical record"] += 1
        else:
            stats["kept: seasons disjoint from the same-name record"] += 1
    return doomed, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(REPO_ROOT / "prospects.db"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not Path(args.db).exists():
        raise SystemExit(f"[prune] no such db: {args.db}")
    conn = sqlite3.connect(args.db)
    doomed, stats = plan(conn)

    n_rows = conn.execute(
        "SELECT COUNT(*) FROM season_stats WHERE player_id IN "
        f"({','.join('?' * len(doomed))})", doomed).fetchone()[0] if doomed else 0
    print(f"[prune] {args.db}{'  (DRY RUN)' if args.dry_run else ''}")
    for k, v in sorted(stats.items()):
        print(f"  {v:>6,} ids   {k}")
    print(f"\n[prune] player_ids to delete {len(doomed):,}")
    print(f"[prune] rows to delete       {n_rows:,}")

    if not args.dry_run and doomed:
        with conn:
            for i in range(0, len(doomed), 500):
                chunk = doomed[i:i + 500]
                conn.execute(
                    "DELETE FROM season_stats WHERE player_id IN "
                    f"({','.join('?' * len(chunk))})", chunk)
        left = conn.execute(
            "SELECT COUNT(*) FROM season_stats s WHERE NOT EXISTS "
            "(SELECT 1 FROM prospects p WHERE p.player_id = s.player_id)"
        ).fetchone()[0]
        print(f"[prune] deleted. orphan rows remaining: {left:,} "
              f"(kept deliberately — distinct players)")
        print("[prune] now re-run percentile_backfill; the cohorts changed.")
    conn.close()


if __name__ == "__main__":
    main()
