"""Build prospects_ext.db: the live DB + 1996-2004 draft classes and their MiLB seasons (2026-09-24).

The learning curve (runs/experiments/exp_lc*) is still rising at 100% of the training players, so more
resolved careers are the main remaining AP lever. Scope, by the populate-only-where-complete rule:

  * players drafted 1996-2004 (MLB Stats API draft endpoint, reference/pre2005/drafts) who played
    in the minors 1996-2004 — their careers start in the covered window, so they are complete;
  * NOT players drafted before 1996 (their early seasons are outside the window) and NOT
    international signees (DSL / VSL are absent before 2006).

Existing players (matched strictly by MLBAM id; e.g. Curtis Granderson, whose 2002-04 seasons
were missing) keep their player_id and gain their pre-2005 seasons; new players are inserted
as ifa_<mlbam> with their draft record, following the existing pre-2005 draftee convention.
Season lines come from the cached league-wide splits (one summed row per player-level-season,
the same key as season_stats) and are parsed by the production loader's parser.

Writes ONLY prospects_ext.db. Run the derived backfills and the pipeline against it with
PROSPECT_DB / PROSPECT_MODEL_DB pointing there and a RUN_TAG.

    python tools/build_pre2005_ext.py                                   # 1996-2004 -> prospects_ext.db
    python tools/build_pre2005_ext.py --start 1993 --db prospects_ext2.db   # 1993-2004 (1992 lacks FSL/CAR)
"""
from __future__ import annotations

import glob
import argparse
import json
import shutil
import sqlite3
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
PRE = ROOT / "reference" / "pre2005"
EXT = ROOT / "prospects_ext.db"
LEVELS = {11: "AAA", 12: "AA", 13: "A+", 14: "A", 15: "A-", 16: "RK"}
API = "https://statsapi.mlb.com/api/v1"


def draftees(start=1996):
    """mlbam -> (year, round, pick) of the LAST start-2004 selection."""
    out = {}
    for f in sorted(glob.glob(str(PRE / "drafts" / "*.json"))):
        y = int(Path(f).stem)
        if y < start:
            continue
        for rd in json.loads(Path(f).read_text(encoding="utf-8")).get("drafts", {}).get("rounds", []):
            for k in rd.get("picks", []):
                pid = (k.get("person") or {}).get("id")
                if pid:
                    try:
                        rnd = int(str(k.get("pickRound")).rstrip("CSABcsab"))
                    except ValueError:
                        rnd = None
                    out[int(pid)] = (y, rnd, k.get("pickNumber"))
    return out


def main():
    global EXT
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1996, help="first draft class / MiLB season")
    ap.add_argument("--db", default=str(EXT))
    args = ap.parse_args()
    EXT = Path(args.db) if Path(args.db).is_absolute() else ROOT / args.db
    if not EXT.exists():
        shutil.copy2(ROOT / "prospects.db", EXT)
        print(f"[ext] copied prospects.db -> {EXT.name}")
    picks = draftees(args.start)
    splits = {}
    for f in glob.glob(str(PRE / "seasons" / "*.json")):
        y, sid, grp = Path(f).stem.split("_")
        if int(y) < args.start:
            continue
        splits[(int(y), int(sid), grp)] = json.loads(Path(f).read_text(encoding="utf-8"))
    played = {(s.get("player") or {}).get("id") for v in splits.values() for s in v}
    keep = {m for m in picks if m in played}
    print(f"[ext] {args.start}-2004 draftees who played {args.start}-2004 MiLB: {len(keep):,}")

    con = sqlite3.connect(EXT)
    have = {}
    for pid, m in con.execute("SELECT player_id, mlbam_id FROM prospects WHERE mlbam_id IS NOT NULL"):
        try:
            have[int(float(m))] = pid
        except (TypeError, ValueError):
            pass
    new = sorted(m for m in keep if m not in have)
    print(f"[ext] already in DB: {len(keep) - len(new):,}; new: {len(new):,}")

    # bio for the new players (strict by MLBAM id)
    bio = {}
    for i in range(0, len(new), 100):
        r = requests.get(f"{API}/people?personIds=" + ",".join(map(str, new[i:i + 100])), timeout=90).json()
        for p in r.get("people", []):
            bio[p["id"]] = p
        time.sleep(0.1)
    rows = []
    for m in new:
        p = bio.get(m, {})
        pos = (p.get("primaryPosition") or {}).get("abbreviation") or ""
        h = p.get("height") or ""
        try:
            ft, inch = h.replace('"', "").split("' ")
            hin = int(ft) * 12 + int(inch)
        except ValueError:
            hin = None
        y, rnd, pk = picks[m]
        rows.append((f"ifa_{m}", p.get("fullName") or f"mlbam_{m}", int(pos in ("P", "RHP", "LHP", "SP", "RP")),
                     pos, p.get("birthDate"), y, rnd, pk, 0, p.get("birthCountry"), str(m), hin,
                     p.get("weight"), (p.get("batSide") or {}).get("code"), (p.get("pitchHand") or {}).get("code")))
    existing_ids = {r[0] for r in con.execute("SELECT player_id FROM prospects")}
    clash = [r[0] for r in rows if r[0] in existing_ids]
    if clash:
        raise SystemExit(f"[ext] {len(clash)} new ids already exist without a matching mlbam: {clash[:5]}")
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with con:          # plain INSERT: a constraint failure must raise, not vanish (OR IGNORE hid NOT NULL)
        con.executemany(
            "INSERT INTO prospects (player_id, name, is_pitcher, primary_position, birth_date, draft_year, "
            "draft_round, draft_pick, is_international, origin, mlbam_id, height_inches, weight_lbs, bats, throws, "
            "tj_history, has_current_injury, current_injury_type, notes, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,0,'','pre2005 draft class (ext build)',?)",
            [r + (stamp,) for r in rows])
    con.close()
    print(f"[ext] inserted {len(rows):,} new players")

    # season lines 1996-2004 through the production parser
    from prospects.core.storage import ProspectDB
    from prospects.data.sources import milb
    db = ProspectDB(str(EXT))
    milb._MLBAM_TO_PROSPECT = None      # the loader caches its map; None forces a reload from EXT
    milb._load_mlbam_map(db)
    teams = {}
    for y in range(args.start, 2005):
        for sid in LEVELS:
            for t in requests.get(f"{API}/teams?sportId={sid}&season={y}", timeout=60).json().get("teams", []):
                teams[(y, t["id"])] = t.get("abbreviation")
    out = []
    for (y, sid, grp), sp in splits.items():
        for s in sp:
            pl = s.get("player") or {}
            if pl.get("id") not in keep:
                continue
            team = s.get("team") or {}
            flat = dict(s.get("stat") or {})
            flat.update({"playerId": pl["id"], "playerFullName": pl.get("fullName", ""),
                         "playerFirstName": pl.get("firstName", ""), "playerLastName": pl.get("lastName", ""),
                         "primaryPositionAbbrev": (s.get("position") or {}).get("abbreviation", ""),
                         "teamAbbrev": teams.get((y, team.get("id"))) or "",
                         "leagueName": (s.get("league") or {}).get("name"), "leagueId": (s.get("league") or {}).get("id")})
            row = milb._parse_player_stats(flat, y, LEVELS[sid], "batting" if grp == "hitting" else "pitching",
                                           team.get("id") or 0)
            if row is not None:
                out.append(row)
    # a two-way player can have a hitting and a pitching line at one level: keep both by merging
    merged = {}
    for r in out:
        k = (r.player_id, r.season_year, r.level)
        if k in merged:
            base = merged[k]
            for f, v in vars(r).items():
                if getattr(base, f) in (None, 0, "") and v not in (None, 0, ""):
                    setattr(base, f, v)
        else:
            merged[k] = r
    n = db.upsert_season_stats_many(merged.values())
    print(f"[ext] upserted {n:,} {args.start}-2004 season lines for {len({k[0] for k in merged}):,} players")


if __name__ == "__main__":
    main()
