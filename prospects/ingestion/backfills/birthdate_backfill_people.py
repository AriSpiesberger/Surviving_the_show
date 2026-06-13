"""Backfill remaining missing prospects.birth_date via the MLB Stats *people*
endpoint, keyed on MLBAM id.

Complements birthdate_backfill.py (which uses the /draft/{year} endpoint and so
only covers DRAFTED players). IFAs — e.g. Logan Maxwell (`ifa_800741`) — have no
draft record and fall through that path, leaving birth_date NULL. Silent age
imputation then defaults them to AGE_CENTER=22 (the prime-prospect age), routing
unknown-birthdate players into a flattering region of feature space (they score
high on debut/establish). This repairs the data at the source.

Resolves each missing player's MLBAM id from the `mlbam_id` column, else parses
it from the `ifa_<id>` player_id (the id IS the MLBAM number) or a trailing
6-7 digit id, then batch-queries /api/v1/people and writes birthDate (+ mlbam_id).

    python -m prospects.ingestion.backfills.birthdate_backfill_people          # dry-run
    python -m prospects.ingestion.backfills.birthdate_backfill_people --apply   # write
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
API = "https://statsapi.mlb.com/api/v1/people?personIds={ids}"
BATCH = 40


def _resolve_mlbam(player_id: str, mlbam_id) -> int | None:
    if mlbam_id is not None and str(mlbam_id).strip() not in ("", "None", "nan"):
        try:
            return int(float(mlbam_id))
        except (TypeError, ValueError):
            pass
    m = re.match(r"ifa_(\d+)$", str(player_id))
    if m:
        return int(m.group(1))
    m = re.search(r"_(\d{6,7})$", str(player_id))   # trailing MLBAM id
    return int(m.group(1)) if m else None


def _fetch(ids: list[int]) -> dict[int, dict]:
    url = API.format(ids=",".join(str(i) for i in ids))
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                people = json.load(r).get("people", [])
            return {int(p["id"]): p for p in people if "id" in p}
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                print(f"  batch failed ({e!s})")
                return {}
            time.sleep(1.5 * (attempt + 1))
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--apply", action="store_true",
                    help="Write to the DB (default: dry-run preview).")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    rows = conn.execute(
        "SELECT player_id, name, mlbam_id FROM prospects "
        "WHERE birth_date IS NULL OR birth_date = ''").fetchall()
    targets = []
    for pid, name, mlbam in rows:
        mid = _resolve_mlbam(pid, mlbam)
        if mid is not None:
            targets.append((pid, name, mid))
    print(f"missing birth_date: {len(rows)} | resolvable MLBAM: {len(targets)}")

    id_to_player = {mid: (pid, name) for pid, name, mid in targets}
    ids = list(id_to_player)
    updates, preview = [], []
    for i in range(0, len(ids), BATCH):
        chunk = ids[i:i + BATCH]
        got = _fetch(chunk)
        for mid in chunk:
            p = got.get(mid)
            pid, name = id_to_player[mid]
            bd = (p or {}).get("birthDate")
            if bd:
                updates.append((bd, mid, pid))
                preview.append((name, bd, (p or {}).get("currentAge")))
        print(f"  {min(i+BATCH, len(ids))}/{len(ids)} fetched, "
              f"{len(updates)} birthdates so far", flush=True)
        time.sleep(0.3)

    print(f"\nresolved {len(updates)} birthdates of {len(targets)} targets")
    for name, bd, age in preview[:15]:
        print(f"  {name:<26} {bd}  (age {age})")

    if not args.apply:
        print("\nDRY-RUN — re-run with --apply to write.")
        conn.close()
        return
    cur = conn.cursor()
    for bd, mid, pid in updates:
        cur.execute(
            "UPDATE prospects SET birth_date = ?, "
            "mlbam_id = COALESCE(mlbam_id, ?) WHERE player_id = ?",
            (bd, mid, pid))
    conn.commit()
    print(f"\nWROTE {len(updates)} birthdates to {args.db}")
    conn.close()


if __name__ == "__main__":
    main()
