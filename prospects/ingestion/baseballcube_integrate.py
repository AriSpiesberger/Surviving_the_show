"""
prospects/ingestion/baseballcube_integrate.py
===============================================

Join the scraped TBC per-team rankings to the project universe and emit rows
that conform to the rankings_history schema.

Inputs
------
  data/baseballcube_team_rankings.csv   (baseballcube_loader)   -- the lists
  data/baseballcube_player_xref.csv     (baseballcube_xref)     -- tbc -> mlbam
  db_dump/prospects.csv  (or a live ProspectDB)                 -- mlbam -> player_id

Output
------
  data/baseballcube_rankings_history.csv with columns:
      player_id, as_of, source, overall_rank, org_rank, list_size,
      mlbam_id, tbc_player_id, team, matched

  - player_id : project key (resolved via mlbam_id); blank if not in universe
  - as_of     : '{year}-01-01' (RankingSnapshot is date-typed)
  - source    : full source name ("Baseball America", ...)
  - overall_rank : MLB-wide rank when the player also made the top-100 that year
  - org_rank  : team prospect rank (1..N)
  - list_size : size of that team-year-source list
  - matched   : 1 if player_id resolved into the project universe
  - match_method : mlbam | draft | birthdate+name | '' (how it resolved)

Only the `matched` rows are ready to load straight into rankings_history; the
rest remain useful keyed on mlbam_id once those players enter the universe.

Temporal alignment (no lookahead)
---------------------------------
The source org lists are PRESEASON annual rankings (BA Handbook, MLB Pipeline,
etc. publish before the season, built on through-prior-season information). We
anchor each to ``as_of = {year}-01-01`` -- i.e. "known at the start of season
Y". The rank-trajectory deltas only ever look back (years Y-1, Y-2), so every
feature on a row is observable at its own as_of; nothing peeks forward.

CAUTION: the *bio* columns on baseballcube_team_rankings.csv (hilvl, mlb_years,
stat_years) are present-day/career-to-date values, NOT point-in-time, so they
are leaky as features. They are intentionally NOT carried into this output;
use them for identity only. Only rank/org_rank/overall_rank/list_size and the
backward-looking deltas here are safe as-of-season features.

Usage
-----
    python -m prospects.ingestion.baseballcube_integrate
"""

from __future__ import annotations

import argparse
import csv
import re
import sqlite3
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Tuple

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def norm_name(name: str) -> str:
    """'Cody A. Reed' / 'Cody Reed Jr.' -> 'cody reed'. Accent/punct/middle-name
    insensitive; collapses to first + last token. '' if unusable."""
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = re.sub(r"[.,'`]", "", s).lower()
    toks = [t for t in s.split() if t not in _SUFFIXES and len(t) > 1]
    if not toks:
        return ""
    return f"{toks[0]} {toks[-1]}" if len(toks) >= 2 else toks[0]

ROOT = Path(__file__).resolve().parents[2]
RANKINGS_CSV = ROOT / "data" / "baseballcube_team_rankings.csv"
XREF_CSV = ROOT / "data" / "baseballcube_player_xref.csv"
PROSPECTS_CSV = ROOT / "db_dump" / "prospects.csv"
PROSPECTS_DB = ROOT / "prospects.db"
OUT_CSV = ROOT / "data" / "baseballcube_rankings_history.csv"

OUT_FIELDS = [
    "player_id", "as_of", "source", "overall_rank", "org_rank", "list_size",
    # Rank trajectory: within the same (player, org, source), how the org rank
    # moved vs 1 and 2 years prior. Delta = prior_rank - current_rank, so a
    # POSITIVE delta means the player CLIMBED the list (rank number shrank,
    # moving toward the top of the org). Blank = no comparable prior list
    # (new to org, traded in, gap year, or older than the 2-year window).
    "org_rank_1y_ago", "org_rank_2y_ago", "org_rank_delta_1y", "org_rank_delta_2y",
    "mlbam_id", "tbc_player_id", "team", "matched", "match_method",
]


def _load_xref(xref_csv: Path) -> Dict[str, dict]:
    """tbc_player_id -> full xref row (mlbam, birthdate, draft tuple, names)."""
    out: Dict[str, dict] = {}
    with xref_csv.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["tbc_player_id"]] = r
    return out


def _drafttuple(year, rnd, pick) -> Optional[Tuple[str, str, str]]:
    """Normalize a (year, round, pick) tuple to comparable strings, or None
    unless all three are present and round/pick are numeric (real picks only)."""
    y, rd, pk = str(year).strip(), str(rnd).strip(), str(pick).strip()
    # DB stores ints; CSV stores strings. Coerce '54'/'54.0'/54 -> '54'.
    def _i(x):
        try:
            return str(int(float(x)))
        except (ValueError, TypeError):
            return ""
    y, rd, pk = _i(y), _i(rd), _i(pk)
    if y and rd and pk:
        return (y, rd, pk)
    return None


class Universe:
    """Multi-key index of the project universe for a precision match cascade.

    Each index maps a key -> player_id, but only when that key is UNIQUE in the
    universe; ambiguous keys are dropped so a fallback never invents a match.
    """

    def __init__(self) -> None:
        self.by_mlbam: Dict[str, str] = {}
        self._draft: Dict[tuple, set] = {}
        self._bdname: Dict[tuple, set] = {}

    def add(self, pid, mlbam, birth_date, name, draft_year, draft_round, draft_pick):
        mlbam = str(mlbam).strip() if mlbam else ""
        if mlbam and mlbam not in self.by_mlbam:
            self.by_mlbam[mlbam] = pid
        dt = _drafttuple(draft_year, draft_round, draft_pick)
        if dt:
            self._draft.setdefault(dt, set()).add(pid)
        nn = norm_name(name)
        bd = (birth_date or "").strip()
        if nn and bd:
            self._bdname.setdefault((bd, nn), set()).add(pid)

    def finalize(self):
        self.by_draft = {k: next(iter(v)) for k, v in self._draft.items() if len(v) == 1}
        self.by_bdname = {k: next(iter(v)) for k, v in self._bdname.items() if len(v) == 1}
        return self

    def resolve(self, mlbam, birth_date, name, draft_year, draft_round, draft_pick):
        """Return (player_id, method) via the precision cascade, or ('', '')."""
        if mlbam:
            pid = self.by_mlbam.get(str(mlbam).strip())
            if pid:
                return pid, "mlbam"
        dt = _drafttuple(draft_year, draft_round, draft_pick)
        if dt:
            pid = self.by_draft.get(dt)
            if pid:
                return pid, "draft"
        nn = norm_name(name)
        bd = (birth_date or "").strip()
        if nn and bd:
            pid = self.by_bdname.get((bd, nn))
            if pid:
                return pid, "birthdate+name"
        return "", ""


def _load_universe(prospects_csv: Path, db_path: Optional[Path]) -> Universe:
    """Build the multi-key Universe from the live DB (preferred) or CSV snapshot."""
    u = Universe()
    if db_path and Path(db_path).exists():
        con = sqlite3.connect(str(db_path))
        try:
            for row in con.execute(
                "SELECT player_id, mlbam_id, birth_date, name, "
                "draft_year, draft_round, draft_pick FROM prospects"
            ):
                u.add(*row)
        finally:
            con.close()
    elif prospects_csv.exists():
        with prospects_csv.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                u.add(r["player_id"], r.get("mlbam_id"), r.get("birth_date"),
                      r.get("name"), r.get("draft_year"),
                      r.get("draft_round"), r.get("draft_pick"))
    return u.finalize()


def integrate(
    rankings_csv: Path = RANKINGS_CSV,
    xref_csv: Path = XREF_CSV,
    prospects_csv: Path = PROSPECTS_CSV,
    out_csv: Path = OUT_CSV,
    db_path: Optional[Path] = PROSPECTS_DB,
    verbose: bool = True,
) -> Dict[str, int]:
    xref = _load_xref(xref_csv)
    uni = _load_universe(prospects_csv, db_path)
    if verbose:
        src = "prospects.db" if (db_path and Path(db_path).exists()) else "prospects.csv"
        print(f"[integrate] xref: {len(xref)} tbc players; universe ({src}): "
              f"{len(uni.by_mlbam)} mlbam / {len(uni.by_draft)} draft-tuple / "
              f"{len(uni.by_bdname)} birthdate+name keys")

    # First pass: list sizes per (year, team_id, source), and a rank index
    # keyed (tbc_player_id, team_id, source, year) -> org_rank for lookbacks.
    list_size: Counter = Counter()
    rank_idx: Dict[tuple, int] = {}
    with rankings_csv.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            list_size[(r["year"], r["team_id"], r["source"])] += 1
            if r["org_rank"].isdigit() and r["tbc_player_id"]:
                rank_idx[(r["tbc_player_id"], r["team_id"],
                          r["source"], int(r["year"]))] = int(r["org_rank"])

    def lookback(tbc: str, team_id: str, source: str, year: int, back: int):
        """(prior_rank, delta) for `back` years ago, or (None, None)."""
        if not tbc:
            return None, None
        prior = rank_idx.get((tbc, team_id, source, year - back))
        cur = rank_idx.get((tbc, team_id, source, year))
        if prior is None or cur is None:
            return prior, None
        return prior, prior - cur   # positive = climbed toward #1

    stats = Counter()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with rankings_csv.open(newline="", encoding="utf-8") as f, \
            out_csv.open("w", newline="", encoding="utf-8") as g:
        w = csv.DictWriter(g, fieldnames=OUT_FIELDS)
        w.writeheader()
        for r in csv.DictReader(f):
            stats["rows"] += 1
            tbc = r["tbc_player_id"]
            x = xref.get(tbc, {})
            mlbam = (x.get("mlbam_id") or "").strip()
            pid, method = uni.resolve(
                mlbam, x.get("birthdate"), x.get("proper_name") or x.get("name") or r["player"],
                x.get("draft_year"), x.get("draft_round"), x.get("draft_pick"),
            )
            matched = 1 if pid else 0
            stats["with_mlbam"] += 1 if mlbam else 0
            stats["matched"] += matched
            if method:
                stats[f"via_{method}"] += 1

            yr = int(r["year"])
            p1, d1 = lookback(r["tbc_player_id"], r["team_id"], r["source"], yr, 1)
            p2, d2 = lookback(r["tbc_player_id"], r["team_id"], r["source"], yr, 2)
            stats["has_1y"] += 1 if d1 is not None else 0
            stats["has_2y"] += 1 if d2 is not None else 0

            w.writerow({
                "player_id": pid,
                "as_of": f"{r['year']}-01-01",
                "source": r["source_name"],
                "overall_rank": r["mlb_rank"],
                "org_rank": r["org_rank"],
                "list_size": list_size[(r["year"], r["team_id"], r["source"])],
                "org_rank_1y_ago": p1 if p1 is not None else "",
                "org_rank_2y_ago": p2 if p2 is not None else "",
                "org_rank_delta_1y": d1 if d1 is not None else "",
                "org_rank_delta_2y": d2 if d2 is not None else "",
                "mlbam_id": mlbam,
                "tbc_player_id": tbc,
                "team": r["team"],
                "matched": matched,
                "match_method": method,
            })

    if verbose:
        n = stats["rows"] or 1
        print(f"[integrate] {stats['rows']} ranking rows")
        print(f"            {stats['with_mlbam']} have mlbam ({100*stats['with_mlbam']//n}%)")
        print(f"            {stats['matched']} matched into universe "
              f"({100*stats['matched']//n}%)")
        print(f"              by method: mlbam={stats['via_mlbam']} "
              f"draft={stats['via_draft']} birthdate+name={stats['via_birthdate+name']}")
        print(f"            {stats['has_1y']} rows with a 1y-ago org rank "
              f"({100*stats['has_1y']//n}%); {stats['has_2y']} with 2y-ago "
              f"({100*stats['has_2y']//n}%)")
        print(f"            -> {out_csv}")
    return dict(stats)


def load_rankings_history(
    out_csv: Path = OUT_CSV,
    db_path: Path = PROSPECTS_DB,
    verbose: bool = True,
) -> int:
    """Load the MATCHED rows from the integrated CSV into rankings_history.

    Only rows with a resolved player_id are loaded (the table is keyed on it).
    A player can appear on at most one org list per (source, year); the rare
    traded-twice collision is de-duped (first row wins) via INSERT OR IGNORE.
    Returns rows inserted.
    """
    con = sqlite3.connect(str(db_path))
    inserted = skipped = dup = 0
    try:
        with out_csv.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["matched"] != "1" or not r["player_id"]:
                    skipped += 1
                    continue
                cur = con.execute(
                    "INSERT OR IGNORE INTO rankings_history "
                    "(player_id, as_of, source, overall_rank, org_rank, list_size) "
                    "VALUES (?,?,?,?,?,?)",
                    (r["player_id"], r["as_of"], r["source"],
                     int(r["overall_rank"]) if r["overall_rank"].strip() else None,
                     int(r["org_rank"]) if r["org_rank"].strip() else None,
                     int(r["list_size"]) if r["list_size"].strip() else None),
                )
                if cur.rowcount:
                    inserted += 1
                else:
                    dup += 1
        con.commit()
    finally:
        con.close()
    if verbose:
        print(f"[load] inserted {inserted} rows into rankings_history "
              f"(skipped {skipped} unmatched, {dup} dup PK) -> {db_path}")
    return inserted


def main() -> None:
    ap = argparse.ArgumentParser(description="Join TBC rankings -> rankings_history rows")
    ap.add_argument("--rankings", type=Path, default=RANKINGS_CSV)
    ap.add_argument("--xref", type=Path, default=XREF_CSV)
    ap.add_argument("--prospects", type=Path, default=PROSPECTS_CSV,
                    help="CSV fallback for mlbam->player_id if --db is absent")
    ap.add_argument("--db", type=Path, default=PROSPECTS_DB,
                    help="live prospects.db (preferred universe source)")
    ap.add_argument("--no-db", action="store_true",
                    help="ignore the DB and use the CSV snapshot")
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    ap.add_argument("--load-db", action="store_true",
                    help="after integrating, load matched rows into rankings_history")
    args = ap.parse_args()
    integrate(args.rankings, args.xref, args.prospects, args.out,
              db_path=None if args.no_db else args.db)
    if args.load_db:
        load_rankings_history(args.out, args.db)


if __name__ == "__main__":
    main()
