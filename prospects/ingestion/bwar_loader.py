"""
prospects/ingestion/bwar_loader.py
====================================

Populate career_outcomes.career_war by pulling Baseball-Reference WAR from
pybaseball (bwar_bat + bwar_pitch) and joining to our prospects by
(lowercased name, debut year).

Why this exists: the original outcomes_loader wrote career_war=0 because
Lahman doesn't carry WAR. The continuous WAR regressor needs real labels.

Matching strategy:
  1. For each bbref player, take min(year_ID) as their MLB debut year and
     sum WAR across all seasons.
  2. For each prospect with mlb_debut_year set, look up by normalized name.
     If multiple candidates, pick the one whose bWAR debut year is within
     +/- 1 year of our recorded mlb_debut_year.
  3. Unmatched debuted prospects keep career_war=0 (and are flagged).

Usage:
    python -m prospects.ingestion.bwar_loader [--db prospects.db]
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import time

import pandas as pd


def _norm(name: str) -> str:
    if not name:
        return ""
    n = name.lower().strip()
    n = re.sub(r"[^\w\s]", "", n)  # drop punctuation
    n = re.sub(r"\s+", " ", n)
    # drop common suffixes
    for suf in (" jr", " sr", " ii", " iii", " iv"):
        if n.endswith(suf):
            n = n[: -len(suf)]
    return n.strip()


def build_bwar_career_index() -> dict:
    """Returns {bbref_id: {'name': str, 'debut': int, 'war': float}}."""
    import pybaseball as pb

    print("[bwar] downloading bwar_bat...")
    t0 = time.time()
    bat = pb.bwar_bat(return_all=True)
    print(f"  {len(bat)} batting rows in {time.time()-t0:.1f}s")

    print("[bwar] downloading bwar_pitch...")
    t0 = time.time()
    pit = pb.bwar_pitch(return_all=True)
    print(f"  {len(pit)} pitching rows in {time.time()-t0:.1f}s")

    keep_bat = bat[["player_ID", "name_common", "year_ID", "WAR"]].copy()
    keep_pit = pit[["player_ID", "name_common", "year_ID", "WAR"]].copy()
    keep_bat["WAR"] = pd.to_numeric(keep_bat["WAR"], errors="coerce").fillna(0.0)
    keep_pit["WAR"] = pd.to_numeric(keep_pit["WAR"], errors="coerce").fillna(0.0)

    combined = pd.concat([keep_bat, keep_pit], ignore_index=True)
    combined = combined.dropna(subset=["player_ID"])
    combined["year_ID"] = pd.to_numeric(combined["year_ID"], errors="coerce")
    combined = combined.dropna(subset=["year_ID"])

    grp = combined.groupby("player_ID").agg(
        name=("name_common", "first"),
        debut=("year_ID", "min"),
        war=("WAR", "sum"),
    )
    print(f"[bwar] {len(grp)} unique bbref players")

    idx: dict[str, dict] = {}
    for bbref_id, row in grp.iterrows():
        idx[bbref_id] = {
            "name": str(row["name"]) if pd.notna(row["name"]) else "",
            "debut": int(row["debut"]),
            "war": float(row["war"]),
        }
    return idx


def build_name_lookup(idx: dict) -> dict:
    """Returns {normalized_name: [(bbref_id, debut, war), ...]}."""
    out: dict[str, list] = {}
    for bbref_id, rec in idx.items():
        k = _norm(rec["name"])
        if not k:
            continue
        out.setdefault(k, []).append((bbref_id, rec["debut"], rec["war"]))
    return out


def match_prospect(name: str, debut_year: int | None, lookup: dict) -> tuple[float, str | None]:
    """Returns (career_war, matched_bbref_id_or_None).

    v1.11: name-only matching with a debut-year tolerance now REQUIRES
    debut_year and a tight (<= 1 year) gap. The old behavior of falling
    back to "max-WAR candidate" when debut_year was missing was a major
    source of label pollution (modern prospects getting HOFer WAR
    attributed). With strict matching, players without a debut_year just
    return 0.0/None and the row stays clean."""
    cands = lookup.get(_norm(name))
    if not cands:
        return 0.0, None
    if debut_year is None:
        # No debut year — cannot disambiguate by career window. Return
        # zero rather than guess. Modern un-debuted prospects fall here
        # and correctly get no bWAR.
        return 0.0, None
    if len(cands) == 1:
        bbref_id, debut, war = cands[0]
        if abs(debut - debut_year) <= 1:
            return war, bbref_id
        return 0.0, None
    # Multiple candidates: only accept if exactly one has a debut within
    # 1 year of ours. Ambiguous matches are dropped.
    close = [(b, d, w) for (b, d, w) in cands if abs(d - debut_year) <= 1]
    if len(close) == 1:
        bbref_id, _, war = close[0]
        return war, bbref_id
    return 0.0, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="prospects.db")
    args = parser.parse_args()

    print(f"[bwar] DB: {args.db}")
    idx = build_bwar_career_index()
    lookup = build_name_lookup(idx)
    print(f"[bwar] {len(lookup)} unique normalized names in lookup")

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT p.player_id, p.name, o.mlb_debut_year
        FROM prospects p
        JOIN career_outcomes o ON p.player_id = o.player_id
        WHERE o.mlb_debut_year IS NOT NULL
        """
    ).fetchall()
    print(f"[bwar] {len(rows)} prospects with mlb_debut_year")

    matched = 0
    nonzero = 0
    updates = []
    for r in rows:
        war, bbref = match_prospect(r["name"], r["mlb_debut_year"], lookup)
        if bbref is not None:
            matched += 1
        if war > 0 or war < 0:
            nonzero += 1
        updates.append((round(war, 2), r["player_id"]))

    conn.executemany(
        "UPDATE career_outcomes SET career_war = ? WHERE player_id = ?",
        updates,
    )
    conn.commit()
    conn.close()

    print(f"[bwar] matched: {matched}/{len(rows)} ({matched/max(len(rows),1):.1%})")
    print(f"[bwar] non-zero WAR: {nonzero}")
    print(f"[bwar] updated career_outcomes.career_war for {len(updates)} players")


if __name__ == "__main__":
    main()
