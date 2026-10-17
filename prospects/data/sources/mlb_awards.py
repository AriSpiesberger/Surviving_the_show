"""
prospects/data/sources/mlb_awards.py
====================================

All-Star selections and major awards (MVP, Cy Young, Rookie of the Year) from
the MLB Stats API, keyed by MLBAM id.

Why: outcomes.py took these from Lahman, which stops at 2021, while the model
treats outcomes as observed through MAX_OBS_YEAR. Every All-Star and award
winner from 2022 on was therefore a training negative.

The pulled table is cached at reference/mlb_awards.csv so an outcomes rebuild
is reproducible offline; pass refresh=True (or run this module) to re-pull.

    python -m prospects.data.sources.mlb_awards            # refresh the cache
"""

from __future__ import annotations

import csv
import datetime as dt
from collections import defaultdict

from prospects.config import REFERENCE_DIR
from prospects.data.sources.milb import _get_json

AWARDS_CSV = REFERENCE_DIR / "mlb_awards.csv"
FIRST_SEASON = 1990

# Stats API award id -> our kind
AWARD_KIND = {
    "ALAS": "all_star", "NLAS": "all_star",
    "ALMVP": "mvp", "NLMVP": "mvp",
    "ALCY": "cy_young", "NLCY": "cy_young",
    "ALROY": "roy", "NLROY": "roy",
}
_URL = "https://statsapi.mlb.com/api/v1/awards/{award}/recipients?season={season}"


def pull(first: int = FIRST_SEASON, last: int | None = None) -> list[dict]:
    """Every recipient of every award in AWARD_KIND for seasons first..last.
    A season with no event (no 2020 All-Star Game, a season not yet awarded)
    returns 404 or an empty list and is skipped."""
    last = last or dt.date.today().year
    rows = []
    for season in range(first, last + 1):
        for award, kind in AWARD_KIND.items():
            # None = 404 (no game in 2020, season not yet awarded); transient
            # faults are retried inside _get_json and raise if they persist
            for a in (_get_json(_URL.format(award=award, season=season)) or {}).get("awards", []):
                rows.append({"season": season, "award": award, "kind": kind,
                             "mlbam_id": a["player"]["id"],
                             "name": a["player"].get("nameFirstLast", "")})
    return rows


def write_cache(rows: list[dict]) -> None:
    AWARDS_CSV.parent.mkdir(parents=True, exist_ok=True)
    tmp = AWARDS_CSV.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["season", "award", "kind", "mlbam_id", "name"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["season"], r["award"], r["mlbam_id"])))
    tmp.replace(AWARDS_CSV)


def load(refresh: bool = False) -> dict[str, dict[str, list[int]]]:
    """{kind: {mlbam_id (str): sorted seasons}}. One entry per (player, season,
    kind): a player named to the All-Star team once in a season counts once."""
    if refresh or not AWARDS_CSV.exists():
        write_cache(pull())
    out: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    with AWARDS_CSV.open() as f:
        for r in csv.DictReader(f):
            out[r["kind"]][str(int(r["mlbam_id"]))].add(int(r["season"]))
    return {k: {pid: sorted(ys) for pid, ys in v.items()} for k, v in out.items()}


if __name__ == "__main__":
    rows = pull()
    write_cache(rows)
    seasons = sorted({r["season"] for r in rows})
    print(f"wrote {len(rows)} rows, seasons {seasons[0]}-{seasons[-1]} -> {AWARDS_CSV}")
