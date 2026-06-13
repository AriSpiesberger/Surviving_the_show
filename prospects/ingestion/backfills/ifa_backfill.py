"""Repair the `ifa_<mlbam>` bucket against the MLB Stats API.

That bucket is a catch-all for any player without a pre-2020 draft record, so it
mixes three populations the model must treat differently. The people endpoint's
`draftYear` field is NOT a reliable "was drafted" signal — it returned 2025 for
Logan Maxwell, who was an UNDRAFTED free agent (Arkansas -> Yankees 2025). The
only reliable signal is membership in the actual `/draft/{year}` pick roster.

Classification:
  1. DRAFTED        mlbam in /draft/{year} picks
                    -> set draft_year/round/pick, is_international = 0
  2. DOMESTIC UDFA  birthCountry in {USA, Canada, Puerto Rico}, not drafted
                    -> is_international = 0 (they're draft-eligible domestics,
                       just not selected) ; no draft cols
  3. INTERNATIONAL  foreign birthCountry, not drafted
                    -> keep is_international = 1 ; fill origin, age_at_signing

Always fills origin = birthCountry (lets downstream distinguish USA-UDFA from a
genuine IFA), plus age_at_signing and physicals where missing.

    python -m prospects.ingestion.backfills.ifa_backfill              # dry-run, active (2024+)
    python -m prospects.ingestion.backfills.ifa_backfill --all        # dry-run, every IFA-bucket row
    python -m prospects.ingestion.backfills.ifa_backfill --all --apply # write
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB = REPO_ROOT / "prospects_snapshot.db"
PEOPLE = "https://statsapi.mlb.com/api/v1/people?personIds={ids}"
DRAFT = "https://statsapi.mlb.com/api/v1/draft/{year}"
BATCH = 40
DOMESTIC = {"USA", "Canada", "Puerto Rico"}        # draft-eligible -> not IFA
DRAFT_YEARS = range(1995, 2027)                      # catch pre-2016 draftees too
                                                     # (vets dumped in the IFA bucket)
PITCH = {"P", "RHP", "LHP", "SP", "RP"}


def _mlbam(player_id: str, mlbam_id) -> int | None:
    if mlbam_id is not None and str(mlbam_id).strip() not in ("", "None", "nan"):
        try:
            return int(float(mlbam_id))
        except (TypeError, ValueError):
            pass
    m = re.match(r"ifa_(\d+)$", str(player_id))
    return int(m.group(1)) if m else None


def _height_in(h) -> int | None:
    m = re.match(r"(\d+)'\s*(\d+)", str(h or ""))
    return int(m.group(1)) * 12 + int(m.group(2)) if m else None


def _get(url: str):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                print(f"  fetch failed ({e!s}) {url[:60]}")
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


def _draft_roster() -> dict[int, tuple[int, int, int]]:
    """{mlbam: (year, round, pick)} across all DRAFT_YEARS."""
    out: dict[int, tuple[int, int, int]] = {}
    for yr in DRAFT_YEARS:
        j = _get(DRAFT.format(year=yr))
        if not j:
            continue
        for rnd in j.get("drafts", {}).get("rounds", []):
            for p in rnd.get("picks", []):
                pid = (p.get("person") or {}).get("id")
                if pid is None:
                    continue
                try:
                    rd = int(str(p.get("pickRound") or 0)) or None
                except (ValueError, TypeError):
                    rd = None
                out[int(pid)] = (yr, rd, p.get("pickNumber"))
        time.sleep(0.15)
    print(f"draft rosters {DRAFT_YEARS.start}-{DRAFT_YEARS.stop-1}: "
          f"{len(out):,} drafted mlbam ids")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    where = "p.is_international = 1"
    if not args.all:
        where += (" AND p.player_id IN (SELECT DISTINCT player_id FROM "
                  "season_stats WHERE season_year >= 2024)")
    cols = ["player_id", "name", "mlbam_id", "birth_date", "origin",
            "signing_year", "age_at_signing", "height_inches", "weight_lbs",
            "primary_position", "draft_year"]
    rows = conn.execute(
        "SELECT p.player_id, p.name, p.mlbam_id, p.birth_date, p.origin, "
        "p.international_signing_year, p.age_at_signing, p.height_inches, "
        "p.weight_lbs, p.primary_position, p.draft_year "
        f"FROM prospects p WHERE {where}").fetchall()
    recs = [dict(zip(cols, r)) for r in rows]
    for r in recs:
        r["mlbam"] = _mlbam(r["player_id"], r["mlbam_id"])
    targets = [r for r in recs if r["mlbam"] is not None]
    print(f"IFA-bucket rows: {len(recs):,} | resolvable: {len(targets):,} "
          f"({'ALL' if args.all else 'active 2024+'})")

    drafted = _draft_roster()

    people: dict[int, dict] = {}
    ids = [r["mlbam"] for r in targets]
    for i in range(0, len(ids), BATCH):
        j = _get(PEOPLE.format(ids=",".join(map(str, ids[i:i + BATCH]))))
        for p in (j or {}).get("people", []):
            if "id" in p:
                people[int(p["id"])] = p
        if (i // BATCH) % 10 == 0:
            print(f"  people {min(i+BATCH, len(ids)):,}/{len(ids):,}", flush=True)
        time.sleep(0.25)

    n_draft = n_udfa = n_intl = n_origin = n_age = n_phys = 0
    updates: list[tuple] = []
    sample = {"drafted": [], "udfa": [], "intl": []}
    for r in targets:
        p = people.get(r["mlbam"])
        if not p:
            continue
        sets, vals = [], []
        country = p.get("birthCountry")
        if country and not r["origin"]:
            sets.append("origin = ?"); vals.append(country); n_origin += 1

        d = drafted.get(r["mlbam"])
        if d:                                   # (1) DRAFTED
            yr, rd, pick = d
            sets += ["draft_year = ?", "is_international = 0"]; vals.append(yr)
            if rd is not None:
                sets.append("draft_round = ?"); vals.append(rd)
            if pick is not None:
                sets.append("draft_pick = ?"); vals.append(pick)
            n_draft += 1
            if len(sample["drafted"]) < 10:
                sample["drafted"].append((r["name"], yr, rd, country))
        elif country in DOMESTIC:               # (2) DOMESTIC UDFA
            sets.append("is_international = 0"); n_udfa += 1
            if len(sample["udfa"]) < 10:
                sample["udfa"].append((r["name"], country))
        else:                                   # (3) genuine INTERNATIONAL
            n_intl += 1
            if country and len(sample["intl"]) < 6:
                sample["intl"].append((r["name"], country))

        if r["age_at_signing"] is None and r["signing_year"] and r["birth_date"]:
            try:
                sets.append("age_at_signing = ?")
                vals.append(round(int(r["signing_year"]) - int(str(r["birth_date"])[:4]), 1))
                n_age += 1
            except (ValueError, TypeError):
                pass
        if r["height_inches"] is None and _height_in(p.get("height")):
            sets.append("height_inches = ?"); vals.append(_height_in(p.get("height")))
        if r["weight_lbs"] is None and p.get("weight"):
            try:
                sets.append("weight_lbs = ?"); vals.append(int(p["weight"]))
            except (ValueError, TypeError):
                pass
        pos = (p.get("primaryPosition") or {}).get("abbreviation")
        if not r["primary_position"] and pos:
            sets += ["primary_position = ?", "is_pitcher = ?"]
            vals += [pos, 1 if pos in PITCH else 0]; n_phys += 1
        if sets:
            updates.append((sets, vals, r["player_id"]))

    print(f"\nresolved people for {len(people):,}. Classification:")
    print(f"  DRAFTED   (in draft roster) -> set draft cols, is_intl=0: {n_draft:,}")
    print(f"  UDFA      (domestic, undrafted) -> is_intl=0:             {n_udfa:,}")
    print(f"  INTL      (foreign, kept is_intl=1):                      {n_intl:,}")
    print(f"  fills: origin={n_origin:,}  age_at_signing={n_age:,}  physicals={n_phys:,}")
    for tag, items in sample.items():
        if items:
            print(f"  sample {tag}: " + " | ".join(str(x) for x in items[:6]))

    if not args.apply:
        print("\nDRY-RUN — re-run with --apply to write.")
        conn.close()
        return
    cur = conn.cursor()
    for sets, vals, pid in updates:
        cur.execute(f"UPDATE prospects SET {', '.join(sets)} WHERE player_id = ?",
                    (*vals, pid))
    conn.commit()
    print(f"\nWROTE {len(updates):,} prospect updates to {args.db}")
    conn.close()


if __name__ == "__main__":
    main()
