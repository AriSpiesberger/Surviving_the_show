"""
prospects/ingestion/baseballcube_xref.py
==========================================

Resolve TheBaseballCube player ids to MLBAM (and Retrosheet) ids.

The per-team ranking scrape (baseballcube_loader) tags every row with TBC's
internal ``tbc_player_id``. Each TBC player page exposes a cross-reference
block ("MLBAM ID", "Retrosheet#") plus proper name, birthdate, and draft
tuple -- so this is an *exact* id map, not a fuzzy name/birthdate match.

This builds a standalone xref table keyed on tbc_player_id, with mlbam_id as
the durable join key into the project's prospect universe (prospects.mlbam_id)
and downstream stats/outcomes. Players who never reached affiliated tracking
simply have a blank mlbam_id -- expected, and they can still be matched on the
draft tuple if needed.

Usage
-----
    python -m prospects.ingestion.baseballcube_xref            # all ids in the rankings CSV
    python -m prospects.ingestion.baseballcube_xref --limit 50 # smoke test

Output: data/baseballcube_player_xref.csv (one row per tbc_player_id).
Resumable: ids already present are skipped.
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from prospects.ingestion.baseballcube_loader import (
    BASE, DEFAULT_CSV as RANKINGS_CSV, _get,
)

PLAYER_URL = f"{BASE}/content/player/{{pid}}/"
DEFAULT_XREF = Path(__file__).resolve().parents[2] / "data" / "baseballcube_player_xref.csv"

FIELDNAMES = [
    "tbc_player_id", "name", "proper_name", "birthdate",
    "mlbam_id", "retrosheet_id",
    "draft_year", "draft_round", "draft_pick", "draft_team",
    "signing_bonus", "high_school", "college", "place",
]

_MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}

_PAIR_RE = re.compile(
    r"pi-subject'>(.*?)</div><div class='pi-value'>(.*?)</div>", re.S)
_TAG_RE = re.compile(r"<[^>]+>")


def _clean(s: str) -> str:
    s = _TAG_RE.sub(" ", s).replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", s).strip()


def _parse_birthdate(val: str) -> str:
    """'March 28,2001 Age: 25.0' -> '2001-03-28' (or '' if unparseable)."""
    m = re.search(r"([A-Z][a-z]+)\s+(\d{1,2}),\s*(\d{4})", val)
    if not m:
        return ""
    mon = _MONTHS.get(m.group(1))
    if not mon:
        return ""
    return f"{int(m.group(3)):04d}-{mon:02d}-{int(m.group(2)):02d}"


def _parse_draft(val: str) -> Dict[str, str]:
    """'2022 - 3 - 96 - Atlanta Braves' or '2015 - UDFA - ATL'."""
    parts = [p.strip() for p in val.split(" - ")]
    out = {"draft_year": "", "draft_round": "", "draft_pick": "", "draft_team": ""}
    if parts and re.match(r"^\d{4}$", parts[0]):
        out["draft_year"] = parts[0]
    if len(parts) >= 2:
        out["draft_round"] = parts[1]          # number, "UDFA", "IFA", ...
    if len(parts) >= 4:
        out["draft_pick"] = parts[2] if parts[2].isdigit() else ""
        out["draft_team"] = parts[3]
    elif len(parts) == 3:
        out["draft_team"] = parts[2]           # UDFA/IFA: no overall pick
    return out


def parse_player(pid: str, name_hint: str = "") -> Optional[Dict[str, str]]:
    """Fetch one TBC player page and return its xref row (None on fetch fail)."""
    html = _get(PLAYER_URL.format(pid=pid))
    if html is None:
        return None
    pairs = {_clean(k): v for k, v in _PAIR_RE.findall(html)}

    row = {f: "" for f in FIELDNAMES}
    row["tbc_player_id"] = pid
    row["name"] = name_hint
    row["proper_name"] = _clean(pairs.get("Proper Name", ""))
    row["birthdate"] = _parse_birthdate(_clean(pairs.get("Birthdate", "")))
    row["place"] = _clean(pairs.get("Place", ""))
    row["high_school"] = _clean(pairs.get("High School", ""))
    row["college"] = _clean(pairs.get("Colleges", ""))
    row["signing_bonus"] = _clean(pairs.get("Signing Bonus", "")).lstrip("$ ").replace(",", "")

    mlbam = _clean(pairs.get("MLBAM ID", ""))
    row["mlbam_id"] = mlbam if mlbam.isdigit() else ""
    row["retrosheet_id"] = _clean(pairs.get("Retrosheet#", ""))

    row.update(_parse_draft(_clean(pairs.get("Drafted/Signed", ""))))
    return row


def _distinct_players(rankings_csv: Path) -> List[tuple]:
    """[(tbc_player_id, name), ...] distinct, in first-seen order."""
    seen: Dict[str, str] = {}
    with rankings_csv.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pid = r.get("tbc_player_id", "").strip()
            if pid and pid not in seen:
                seen[pid] = r.get("player", "")
    return list(seen.items())


def _done_ids(xref_csv: Path) -> set:
    if not xref_csv.exists():
        return set()
    with xref_csv.open(newline="", encoding="utf-8") as f:
        return {r["tbc_player_id"] for r in csv.DictReader(f)}


def build(
    rankings_csv: Path = RANKINGS_CSV,
    xref_csv: Path = DEFAULT_XREF,
    limit: Optional[int] = None,
    delay: float = 0.25,
    verbose: bool = True,
) -> int:
    players = _distinct_players(rankings_csv)
    done = _done_ids(xref_csv)
    todo = [(pid, nm) for pid, nm in players if pid not in done]
    if limit:
        todo = todo[:limit]
    if verbose:
        print(f"[xref] {len(players)} distinct TBC players; "
              f"{len(done)} done; {len(todo)} to fetch")

    xref_csv.parent.mkdir(parents=True, exist_ok=True)
    is_new = not xref_csv.exists()
    n = mapped = 0
    with xref_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if is_new:
            w.writeheader()
        for pid, nm in todo:
            row = parse_player(pid, nm)
            if row is None:
                if verbose:
                    print(f"[xref] {pid} {nm}: FETCH FAILED, skipping")
                continue
            w.writerow(row)
            f.flush()
            n += 1
            if row["mlbam_id"]:
                mapped += 1
            if verbose and n % 100 == 0:
                print(f"[xref] {n}/{len(todo)}  mlbam-resolved {mapped} "
                      f"({100*mapped//max(n,1)}%)  last: {nm} -> {row['mlbam_id'] or '-'}")
            time.sleep(delay)

    if verbose:
        print(f"\n[xref] wrote {n} rows; {mapped} had an MLBAM id. -> {xref_csv}")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Map TBC player ids -> MLBAM ids")
    ap.add_argument("--rankings", type=Path, default=RANKINGS_CSV)
    ap.add_argument("--out", type=Path, default=DEFAULT_XREF)
    ap.add_argument("--limit", type=int, default=None, help="cap players (smoke test)")
    ap.add_argument("--delay", type=float, default=0.25)
    args = ap.parse_args()
    build(args.rankings, args.out, limit=args.limit, delay=args.delay)


if __name__ == "__main__":
    main()
