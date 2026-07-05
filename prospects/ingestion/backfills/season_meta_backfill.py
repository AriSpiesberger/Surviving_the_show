"""
prospects/ingestion/backfills/season_meta_backfill.py
=============================================

Backfill three derived columns on season_stats:

  games_played       int   — actual games played (Lahman MLB rows) or
                              estimated PA/4.4 for hitter MiLB rows.
                              NULL where we genuinely don't know.

  season_complete    int   — 1 if the season's calendar year is past
                              (year < current_year), else 0. The 0 rows
                              are still mid-season and shouldn't be used
                              as feature inputs without rescaling.

  injury_suspected   int   — 1 if this season has both PA < 100 AND is
                              sandwiched between two seasons where the
                              same player had PA >= 400. Same logic with
                              IP < 30 between IP >= 100 for pitchers.

Run after fresh ingestion or schema migration.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from datetime import date
from typing import Optional

import numpy as np

from prospects.storage import ProspectDB


def _norm(s) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s).lower()).strip()


def _split(full: str) -> tuple[str, str]:
    parts = _norm(full).split()
    sfx = {"jr", "sr", "ii", "iii", "iv"}
    while parts and parts[-1] in sfx:
        parts.pop()
    if len(parts) <= 1:
        return ("", parts[0] if parts else "")
    return parts[0], parts[-1]


def backfill_games_from_lahman(db: ProspectDB, verbose: bool = True) -> int:
    """For every MLB-level row whose prospect has a bbref match, write G
    from Lahman batting (or pitching, if no batting row that year)."""
    import pybaseball as pyb
    from prospects.ingestion.pybaseball_loader import _ensure_lahman
    if not _ensure_lahman():
        return 0
    if verbose:
        print("[games] loading Lahman + Chadwick register...")
    reg = pyb.chadwick_register()
    name_idx: dict[tuple, list] = {}
    for r in reg.itertuples(index=False):
        last = _norm(getattr(r, "name_last", ""))
        first = _norm(getattr(r, "name_first", ""))
        name_idx.setdefault((last, first), []).append(r)
    bat = pyb.lahman.batting().groupby(["playerID", "yearID"]).agg(G=("G", "sum"))
    pit = pyb.lahman.pitching().groupby(["playerID", "yearID"]).agg(G=("G", "sum"))
    bat_dict = bat["G"].to_dict()
    pit_dict = pit["G"].to_dict()

    with db._connect() as conn:
        rows = conn.execute("""
            SELECT s.rowid AS rid, s.player_id, s.season_year, s.pa, s.ip,
                   p.name, p.draft_year
            FROM season_stats s
            JOIN prospects p ON p.player_id = s.player_id
            WHERE s.level = 'MLB'
        """).fetchall()
    if verbose:
        print(f"[games] {len(rows):,} MLB rows to enrich")

    updates = []
    matched = 0
    for r in rows:
        first, last = _split(r["name"])
        cands = name_idx.get((last, first), [])
        chosen = None
        for c in cands:
            debut = getattr(c, "mlb_played_first", None)
            if (debut is not None
                and not (isinstance(debut, float) and np.isnan(debut))
                and r["draft_year"]
                and 0 <= int(debut) - r["draft_year"] <= 15):
                chosen = c; break
        if chosen is None and cands:
            chosen = cands[0]
        if chosen is None:
            continue
        bbref = getattr(chosen, "key_bbref", None)
        if not bbref:
            continue
        g = bat_dict.get((bbref, r["season_year"]))
        if g is None:
            g = pit_dict.get((bbref, r["season_year"]))
        if g is not None:
            updates.append((int(g), r["rid"]))
            matched += 1

    with db._connect() as conn:
        conn.executemany(
            "UPDATE season_stats SET games_played = ? WHERE rowid = ?",
            updates,
        )
    if verbose:
        print(f"[games] wrote games_played on {matched:,}/{len(rows):,} MLB rows")
    return matched


def estimate_milb_games(db: ProspectDB, verbose: bool = True) -> int:
    """Estimate games_played for MiLB hitters as PA/4.4 (rough lineup-spot proxy).
    Skips rows that already have games_played set."""
    with db._connect() as conn:
        rows = conn.execute("""
            SELECT rowid, pa FROM season_stats
            WHERE level != 'MLB'
              AND games_played IS NULL
              AND pa IS NOT NULL AND pa > 0
        """).fetchall()
    updates = [(max(1, round((r["pa"] or 0) / 4.4)), r["rowid"]) for r in rows]
    with db._connect() as conn:
        conn.executemany(
            "UPDATE season_stats SET games_played = ? WHERE rowid = ?",
            updates,
        )
    if verbose:
        print(f"[games] estimated games_played for {len(updates):,} MiLB hitter rows")
    return len(updates)


def mark_completeness(db: ProspectDB, current_year: int, verbose: bool = True) -> int:
    """season_complete = 1 iff season_year < current_year (calendar year is past)."""
    with db._connect() as conn:
        n_done = conn.execute(
            "UPDATE season_stats SET season_complete = 1 WHERE season_year < ?",
            (current_year,),
        ).rowcount
        n_pending = conn.execute(
            "UPDATE season_stats SET season_complete = 0 WHERE season_year >= ?",
            (current_year,),
        ).rowcount
    if verbose:
        print(f"[complete] marked {n_done:,} rows complete, "
              f"{n_pending:,} rows still in-progress (season_year >= {current_year})")
    return n_done + n_pending


def derive_injury_suspected(db: ProspectDB, verbose: bool = True) -> int:
    """Tag a season as injury_suspected if PA<100 (or IP<30 for pitchers) and
    surrounded by full-volume seasons. Walks each player's career timeline."""
    with db._connect() as conn:
        # Pull everything in one shot
        rows = conn.execute("""
            SELECT rowid, player_id, season_year, pa, ip
            FROM season_stats
            ORDER BY player_id, season_year
        """).fetchall()
    by_player: dict[str, list] = {}
    for r in rows:
        by_player.setdefault(r["player_id"], []).append(dict(r))

    flag_updates = []
    for pid, seasons in by_player.items():
        # Sort by year, then merge multi-level rows to per-year totals
        per_year: dict[int, dict] = {}
        for s in seasons:
            y = s["season_year"]
            agg = per_year.setdefault(y, {"pa": 0, "ip": 0.0, "rids": []})
            agg["pa"] += s["pa"] or 0
            agg["ip"] += s["ip"] or 0.0
            agg["rids"].append(s["rowid"])
        years = sorted(per_year)
        for i, y in enumerate(years):
            if i == 0:
                continue  # need a prior season as a baseline
            cur = per_year[y]
            prev = per_year[years[i - 1]]
            nxt = per_year[years[i + 1]] if i < len(years) - 1 else None
            # (a) sandwiched-low: usage craters between two full seasons.
            sand_h = (nxt is not None and cur["pa"] < 100
                      and prev["pa"] >= 400 and nxt["pa"] >= 400)
            sand_p = (nxt is not None and cur["ip"] < 30
                      and prev["ip"] >= 100 and nxt["ip"] >= 100)
            # (b) one-sided sharp drop after a full season — fires WITHOUT a
            # full next season, so it catches season-ending injuries and the
            # most-recent / current year (the live-scoring blind spot). The
            # prev>= guards gate by role (a pitcher's prev_pa~0, a hitter's
            # prev_ip~0), so the two paths don't cross-fire on two-way lines.
            drop_h = (prev["pa"] >= 300 and cur["pa"] < 250
                      and cur["pa"] < 0.40 * prev["pa"])
            drop_p = (prev["ip"] >= 80 and cur["ip"] < 60
                      and cur["ip"] < 0.40 * prev["ip"])
            if sand_h or sand_p or drop_h or drop_p:
                for rid in cur["rids"]:
                    flag_updates.append((1, rid))

    with db._connect() as conn:
        conn.executemany(
            "UPDATE season_stats SET injury_suspected = ? WHERE rowid = ?",
            flag_updates,
        )
        conn.execute(
            "UPDATE season_stats SET injury_suspected = 0 WHERE injury_suspected IS NULL"
        )
    if verbose:
        print(f"[injury] flagged {len(flag_updates):,} season-rows as suspected-injury")
    return len(flag_updates)


def run(db_path: str = "prospects.db", current_year: Optional[int] = None) -> None:
    db = ProspectDB(db_path)
    cy = current_year or date.today().year
    print(f"Backfilling season metadata (current_year={cy})...")
    backfill_games_from_lahman(db)
    estimate_milb_games(db)
    mark_completeness(db, cy)
    derive_injury_suspected(db)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="prospects.db")
    parser.add_argument("--current-year", type=int, default=None)
    args = parser.parse_args()
    run(args.db, args.current_year)
