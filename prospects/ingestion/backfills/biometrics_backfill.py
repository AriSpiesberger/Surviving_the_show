"""Backfill height, weight, bats, throws from MLB Stats API.

Strategy mirrors birthdate_backfill: for drafted players we use the
/v1/draft/{year} endpoint (already cached behavior). For prospects with
mlbam_id but no draft match (mostly IFAs), we fall back to
/v1/people/{id}.

Height in the API is a string like "6' 6\\\"" — we parse to inches.
Weight is an integer (lbs).

Usage:
    python -m prospects.ingestion.backfills.biometrics_backfill [--db prospects.db]
"""
from __future__ import annotations

import argparse
import re
import time
from typing import Optional

import requests

from prospects.storage import ProspectDB


UA = {"User-Agent": "Mozilla/5.0 (research)"}


def _parse_height(s) -> Optional[float]:
    """'6\\' 6\"' -> 78.0 inches. Also handles '6-6', '78', 78."""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip()
    # "6' 6\""
    m = re.match(r"^\s*(\d+)\s*'\s*(\d+)\s*\"?\s*$", s)
    if m:
        return float(int(m.group(1)) * 12 + int(m.group(2)))
    # "6-6"
    m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", s)
    if m:
        return float(int(m.group(1)) * 12 + int(m.group(2)))
    # bare number
    try:
        v = float(s)
        if v <= 90:
            return v
    except ValueError:
        pass
    return None


def _parse_int(v) -> Optional[int]:
    try:
        return int(v) if v is not None and v != "" else None
    except (TypeError, ValueError):
        return None


def _hand(d) -> Optional[str]:
    """Extract R/L/S from a {code, description} dict or string."""
    if d is None:
        return None
    if isinstance(d, str):
        return d.upper()[:1] if d else None
    code = d.get("code") if isinstance(d, dict) else None
    return code.upper()[:1] if code else None


def _ensure_columns(conn) -> None:
    cur = conn.execute("PRAGMA table_info(prospects)")
    have = {r[1] for r in cur.fetchall()}
    to_add = []
    if "height_inches" not in have:
        to_add.append("height_inches REAL")
    if "weight_lbs" not in have:
        to_add.append("weight_lbs INTEGER")
    if "bats" not in have:
        to_add.append("bats TEXT")
    if "throws" not in have:
        to_add.append("throws TEXT")
    # signing_bonus_usd already exists in schema; we'll populate it here.
    if "pick_value_usd" not in have:
        to_add.append("pick_value_usd REAL")
    for ddl in to_add:
        conn.execute(f"ALTER TABLE prospects ADD COLUMN {ddl}")
        print(f"  added column: {ddl}")
    conn.commit()


def _parse_money(v) -> Optional[float]:
    """API returns strings like '9200000' or None. Strip currency chars if any."""
    if v is None:
        return None
    s = str(v).replace("$", "").replace(",", "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _extract_bio(person: dict) -> dict:
    """Pull biometric fields from a /people response (or draft pick person)."""
    return {
        "height_inches": _parse_height(person.get("height")),
        "weight_lbs": _parse_int(person.get("weight")),
        "bats": _hand(person.get("batSide")),
        "throws": _hand(person.get("pitchHand")),
    }


def _backfill_from_drafts(db: ProspectDB) -> tuple[int, int]:
    """Walk draft years; update biometrics for any pick we recognize."""
    with db._connect() as conn:
        years = [r[0] for r in conn.execute(
            "SELECT DISTINCT draft_year FROM prospects "
            "WHERE draft_year IS NOT NULL ORDER BY draft_year"
        ).fetchall()]
    n_seen = 0
    n_updated = 0
    for yr in years:
        try:
            r = requests.get(
                f"https://statsapi.mlb.com/api/v1/draft/{yr}",
                headers=UA, timeout=60,
            )
            if r.status_code != 200:
                continue
            picks = []
            for rnd in r.json().get("drafts", {}).get("rounds", []):
                picks.extend(rnd.get("picks", []))
        except Exception as e:
            print(f"  {yr}: fetch error {e}")
            continue
        with db._connect() as conn:
            for p in picks:
                person = p.get("person") or {}
                mlbam = str(person.get("id") or "") if person.get("id") else None
                if not mlbam:
                    continue
                n_seen += 1
                bio = _extract_bio(person)
                bonus = _parse_money(p.get("signingBonus"))
                pick_val = _parse_money(p.get("pickValue"))
                if (not any(v is not None for v in bio.values())
                        and bonus is None and pick_val is None):
                    continue
                cur = conn.execute(
                    "UPDATE prospects SET "
                    "height_inches    = COALESCE(?, height_inches), "
                    "weight_lbs       = COALESCE(?, weight_lbs), "
                    "bats             = COALESCE(?, bats), "
                    "throws           = COALESCE(?, throws), "
                    "signing_bonus_usd = COALESCE(?, signing_bonus_usd), "
                    "pick_value_usd   = COALESCE(?, pick_value_usd), "
                    "updated_at = datetime('now') "
                    "WHERE mlbam_id = ?",
                    (bio["height_inches"], bio["weight_lbs"],
                     bio["bats"], bio["throws"],
                     bonus, pick_val, mlbam),
                )
                n_updated += cur.rowcount
            conn.commit()
        print(f"  {yr}: {len(picks):>4} picks seen   "
              f"(cumulative updated={n_updated})")
        time.sleep(0.4)
    return n_seen, n_updated


def _backfill_remaining_via_people(db: ProspectDB,
                                   batch_size: int = 50) -> int:
    """For prospects with mlbam_id but still no height/weight, call /people."""
    with db._connect() as conn:
        rows = [r for r in conn.execute(
            "SELECT player_id, mlbam_id FROM prospects "
            "WHERE mlbam_id IS NOT NULL AND mlbam_id != '' "
            "AND (height_inches IS NULL OR weight_lbs IS NULL "
            "     OR bats IS NULL OR throws IS NULL)"
        ).fetchall()]
    print(f"  remaining without complete bio: {len(rows):,} prospects")
    n_updated = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        ids = ",".join(str(r[1]) for r in batch)
        try:
            r = requests.get(
                f"https://statsapi.mlb.com/api/v1/people?personIds={ids}",
                headers=UA, timeout=60,
            )
            if r.status_code != 200:
                continue
            people = r.json().get("people", [])
        except Exception as e:
            print(f"  batch {i}: fetch error {e}")
            continue
        with db._connect() as conn:
            for person in people:
                mlbam = str(person.get("id") or "")
                if not mlbam:
                    continue
                bio = _extract_bio(person)
                if not any(v is not None for v in bio.values()):
                    continue
                cur = conn.execute(
                    "UPDATE prospects SET "
                    "height_inches = COALESCE(height_inches, ?), "
                    "weight_lbs    = COALESCE(weight_lbs,    ?), "
                    "bats          = COALESCE(bats,          ?), "
                    "throws        = COALESCE(throws,        ?), "
                    "updated_at = datetime('now') "
                    "WHERE mlbam_id = ?",
                    (bio["height_inches"], bio["weight_lbs"],
                     bio["bats"], bio["throws"], mlbam),
                )
                n_updated += cur.rowcount
            conn.commit()
        if (i // batch_size) % 10 == 0:
            print(f"  ...{i+len(batch):,}/{len(rows):,}  updated={n_updated}")
        time.sleep(0.25)
    return n_updated


def _coverage_report(db: ProspectDB) -> None:
    with db._connect() as conn:
        cur = conn.execute("""
            SELECT
              COUNT(*)                                       AS total,
              SUM(CASE WHEN height_inches    IS NOT NULL THEN 1 ELSE 0 END) AS h,
              SUM(CASE WHEN weight_lbs       IS NOT NULL THEN 1 ELSE 0 END) AS w,
              SUM(CASE WHEN bats             IS NOT NULL THEN 1 ELSE 0 END) AS b,
              SUM(CASE WHEN throws           IS NOT NULL THEN 1 ELSE 0 END) AS t,
              SUM(CASE WHEN signing_bonus_usd IS NOT NULL THEN 1 ELSE 0 END) AS sb,
              SUM(CASE WHEN pick_value_usd   IS NOT NULL THEN 1 ELSE 0 END) AS pv
            FROM prospects
        """)
        total, h, w, b, t, sb, pv = cur.fetchone()
    print("\nCoverage after backfill:")
    print(f"  height:        {h:>6,}/{total:>6,}  ({100*h/total:.1f}%)")
    print(f"  weight:        {w:>6,}/{total:>6,}  ({100*w/total:.1f}%)")
    print(f"  bats:          {b:>6,}/{total:>6,}  ({100*b/total:.1f}%)")
    print(f"  throws:        {t:>6,}/{total:>6,}  ({100*t/total:.1f}%)")
    print(f"  signing_bonus: {sb:>6,}/{total:>6,}  ({100*sb/total:.1f}%)")
    print(f"  pick_value:    {pv:>6,}/{total:>6,}  ({100*pv/total:.1f}%)")


def run(db_path: str = "prospects.db") -> None:
    db = ProspectDB(db_path)
    with db._connect() as conn:
        _ensure_columns(conn)

    print("\n[draft endpoint] backfilling drafted prospects...")
    n_seen, n_updated_draft = _backfill_from_drafts(db)
    print(f"  draft pass: {n_seen:,} picks seen, {n_updated_draft:,} rows updated")

    print("\n[people endpoint] backfilling remaining (mostly IFAs)...")
    n_updated_people = _backfill_remaining_via_people(db)
    print(f"  people pass: {n_updated_people:,} rows updated")

    _coverage_report(db)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="prospects.db")
    args = ap.parse_args()
    run(args.db)


if __name__ == "__main__":
    main()
