"""League-context percentiles for the pre-2005 rows of prospects_ext.db (2026-09-24).

pct_* columns rank each season line within its (level, season_year) cohort. For 1996-2004 the
ext DB holds only the 1996-2004 draft classes, so ranking inside the DB would compare a player
with other young draftees instead of the whole league. This builds a scratch DB with EVERY
player's 1996-2004 MiLB line (reference/pre2005/seasons), runs the production wOBA derivation
and percentile ranking over the full league, and writes pct_* back to the cohort's rows in
prospects_ext.db. Run AFTER the ext DB's own percentile_backfill (which would otherwise rank the
pre-2005 cohorts against the subset).

    python tools/build_pre2005_context.py
    python tools/build_pre2005_context.py --start 1993 --db prospects_ext2.db
"""
from __future__ import annotations

import argparse
import glob
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRE = ROOT / "reference" / "pre2005"
EXT = ROOT / "prospects_ext.db"
CTX = ROOT / "runs" / "current" / "scratch" / "pre2005_ctx.db"
LEVELS = {11: "AAA", 12: "AA", 13: "A+", 14: "A", 15: "A-", 16: "RK"}


def main():
    from prospects.core.storage import ProspectDB
    from prospects.data.backfills import woba_backfill
    from prospects.data.sources import milb
    from prospects.features.percentiles import HIT_PCT_NAMES, PIT_PCT_NAMES, attach_percentiles
    global EXT, CTX
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1996)
    ap.add_argument("--db", default=str(EXT))
    args = ap.parse_args()
    EXT = Path(args.db) if Path(args.db).is_absolute() else ROOT / args.db
    CTX = ROOT / "runs" / "current" / "scratch" / f"pre2005_ctx_{EXT.stem}.db"
    files = [f for f in glob.glob(str(PRE / "seasons" / "*.json")) if int(Path(f).stem.split("_")[0]) >= args.start]

    ext = sqlite3.connect(EXT)
    ddl = ext.execute("SELECT sql FROM sqlite_master WHERE name='season_stats'").fetchone()[0]
    cohort = {}
    for pid, m in ext.execute("SELECT player_id, mlbam_id FROM prospects WHERE mlbam_id IS NOT NULL"):
        try:
            cohort[int(float(m))] = pid
        except (TypeError, ValueError):
            pass
    pre_ids = {r[0] for r in ext.execute(
        "SELECT DISTINCT player_id FROM season_stats WHERE season_year < 2005 AND level != 'MLB'")}

    CTX.parent.mkdir(parents=True, exist_ok=True)
    if CTX.exists():
        CTX.unlink()
    ctx = sqlite3.connect(CTX)
    ctx.execute(ddl)
    ctx.commit()
    ctx.close()

    # every player's line; ids are the ext id for the cohort, a placeholder for everyone else
    all_ids = set()
    for f in files:
        for s in json.loads(Path(f).read_text(encoding="utf-8")):
            if (s.get("player") or {}).get("id"):
                all_ids.add(int(s["player"]["id"]))
    milb._MLBAM_TO_PROSPECT = {str(m): (cohort[m] if cohort.get(m) in pre_ids else f"ctx_{m}") for m in all_ids}
    rows = []
    for f in files:
        y, sid, grp = Path(f).stem.split("_")
        y, sid = int(y), int(sid)
        for s in json.loads(Path(f).read_text(encoding="utf-8")):
            pl = s.get("player") or {}
            if not pl.get("id"):
                continue
            flat = dict(s.get("stat") or {})
            flat.update({"playerId": pl["id"], "playerFullName": pl.get("fullName", ""),
                         "playerFirstName": pl.get("firstName", ""), "playerLastName": pl.get("lastName", ""),
                         "primaryPositionAbbrev": (s.get("position") or {}).get("abbreviation", ""),
                         "teamAbbrev": "", "leagueName": (s.get("league") or {}).get("name"),
                         "leagueId": (s.get("league") or {}).get("id")})
            r = milb._parse_player_stats(flat, y, LEVELS[sid], "batting" if grp == "hitting" else "pitching", 0)
            if r is not None:
                rows.append(r)
    merged = {}
    for r in rows:
        k = (r.player_id, r.season_year, r.level)
        if k in merged:
            for fld, v in vars(r).items():
                if getattr(merged[k], fld) in (None, 0, "") and v not in (None, 0, ""):
                    setattr(merged[k], fld, v)
        else:
            merged[k] = r
    ProspectDB(str(CTX)).upsert_season_stats_many(merged.values())
    print(f"[ctx] league context: {len(merged):,} lines, {len({k[0] for k in merged}):,} players ({args.start}-2004)")
    woba_backfill.backfill(str(CTX), verbose=False)

    ctx = sqlite3.connect(CTX)
    ctx.row_factory = sqlite3.Row
    all_rows = [dict(r) for r in ctx.execute("SELECT * FROM season_stats")]
    ctx.close()
    attach_percentiles(all_rows)
    names = HIT_PCT_NAMES + PIT_PCT_NAMES
    upd = []
    for r in all_rows:
        if r["player_id"].startswith("ctx_"):
            continue
        upd.append(tuple(r.get(n) for n in names) + (r["player_id"], r["season_year"], r["level"]))
    with ext:
        ext.executemany(f"UPDATE season_stats SET {', '.join(n + ' = ?' for n in names)} "
                        "WHERE player_id = ? AND season_year = ? AND level = ?", upd)
    ext.close()
    print(f"[ctx] wrote league-context pct_* for {len(upd):,} pre-2005 cohort lines into {EXT.name}")


if __name__ == "__main__":
    main()
