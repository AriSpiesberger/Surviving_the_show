"""Backfill missing current_org in buy_scores by mapping season_stats.org
(minor-league affiliate code) to MLB parent org.

The prospects table only sets current_org for drafted players; 100% of
IFAs come through with NULL. Their MLB affiliation is, however, fully
implied by season_stats.org (the affiliate they played at). We learn the
affiliate -> parent map empirically from drafted-player co-occurrence in
recent seasons, then apply it.

Usage:
    python -m prospects.scripts.backfill_orgs \\
        --in  buy_scores_v13_v3.csv \\
        --out buy_scores_v13_v3_orgfilled.csv
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
from collections import Counter, defaultdict


def build_affiliate_map(db: str, min_year: int = 2024,
                        min_votes: int = 5,
                        min_confidence: float = 0.30
                        ) -> dict[str, tuple[str, float, int]]:
    """Return {affiliate_code: (parent_org, confidence, n_votes)}."""
    conn = sqlite3.connect(db)
    rows = conn.execute(
        """
        SELECT s.org AS affiliate, p.current_org AS parent
        FROM season_stats s
        JOIN prospects p ON p.player_id = s.player_id
        WHERE p.current_org IS NOT NULL
          AND s.org IS NOT NULL
          AND s.season_year >= ?
        """,
        (min_year,),
    ).fetchall()
    votes: dict[str, Counter] = defaultdict(Counter)
    for aff, parent in rows:
        votes[aff][parent] += 1
    result: dict[str, tuple[str, float, int]] = {}
    for aff, c in votes.items():
        top, n = c.most_common(1)[0]
        total = sum(c.values())
        conf = n / total
        if total >= min_votes and conf >= min_confidence:
            result[aff] = (top, conf, total)
    return result


def load_most_recent_orgs(db: str) -> dict[str, str]:
    """For each player, return their most-recent season_stats.org."""
    conn = sqlite3.connect(db)
    rows = conn.execute(
        """
        SELECT player_id, org, season_year
        FROM season_stats
        WHERE org IS NOT NULL
        """
    ).fetchall()
    latest: dict[str, tuple[int, str]] = {}
    for pid, org, yr in rows:
        if pid not in latest or yr > latest[pid][0]:
            latest[pid] = (yr, org)
    return {pid: org for pid, (_, org) in latest.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="buy_scores_v13_v3.csv")
    ap.add_argument("--out", default="buy_scores_v13_v3_orgfilled.csv")
    ap.add_argument("--db", default="prospects_snapshot.db")
    ap.add_argument("--min-year", type=int, default=2024,
                    help="Restrict affiliate->parent learning to this year+")
    ap.add_argument("--min-confidence", type=float, default=0.20,
                    help="Minimum plurality share to accept a mapping")
    ap.add_argument("--min-votes", type=int, default=5,
                    help="Minimum total drafted-player votes per affiliate")
    args = ap.parse_args()

    aff_map = build_affiliate_map(args.db, min_year=args.min_year,
                                  min_votes=args.min_votes,
                                  min_confidence=args.min_confidence)
    print(f"Learned {len(aff_map)} affiliate->parent mappings "
          f"(from {args.min_year}+ data)")

    recent = load_most_recent_orgs(args.db)
    print(f"Most-recent season_stats.org for {len(recent):,} players")

    with open(args.inp, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
        fieldnames = rows[0].keys() if rows else []
    # Add diagnostic columns
    extra = ["org_source", "org_confidence", "affiliate_code"]
    out_fields = list(fieldnames)
    for c in extra:
        if c not in out_fields:
            # Insert right after current_org for readability
            if "current_org" in out_fields:
                idx = out_fields.index("current_org")
                out_fields.insert(idx + 1, c)
            else:
                out_fields.append(c)

    n_total = len(rows)
    n_already = 0
    n_filled = 0
    n_only_affiliate = 0
    n_missing = 0
    for r in rows:
        existing = (r.get("current_org") or "").strip()
        if existing:
            r["org_source"] = "prospects.current_org"
            r["org_confidence"] = ""
            r["affiliate_code"] = ""
            n_already += 1
            continue
        aff = recent.get(r["player_id"])
        if aff is None:
            r["current_org"] = ""
            r["org_source"] = "none"
            r["org_confidence"] = ""
            r["affiliate_code"] = ""
            n_missing += 1
            continue
        mapping = aff_map.get(aff)
        if mapping is None:
            # We know the affiliate but couldn't confidently map it
            r["current_org"] = aff  # surface raw affiliate so it isn't blank
            r["org_source"] = "affiliate_only"
            r["org_confidence"] = ""
            r["affiliate_code"] = aff
            n_only_affiliate += 1
            continue
        parent, conf, _votes = mapping
        r["current_org"] = parent
        r["org_source"] = f"affiliate_vote_{args.min_year}+"
        r["org_confidence"] = f"{conf:.2f}"
        r["affiliate_code"] = aff
        n_filled += 1

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=out_fields)
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {len(rows):,} rows to {args.out}")
    print(f"  already had current_org    : {n_already:>6,}")
    print(f"  filled via affiliate vote  : {n_filled:>6,}")
    print(f"  affiliate known, unmapped  : {n_only_affiliate:>6,}")
    print(f"  no season_stats activity   : {n_missing:>6,}")


if __name__ == "__main__":
    main()
