"""
prospects/ingestion/baseballcube_loader.py
============================================

Historical per-team Top-30 prospect lists from TheBaseballCube.

TBC publishes, for every franchise, the annual organizational prospect
rankings from each major source (Baseball America back to 1983, plus MLB
Pipeline / ESPN / Baseball Prospectus / FanGraphs in the modern era). This is
the per-org complement to the MLB-wide top-100 lists already captured in
db_dump/{year}_rankings.csv.

Data model
----------
Each franchise page (``/content/prospects_team/{TEAM_ID}/``) embeds a history
grid ("grid2") that links to one ranked list per (year, source):

    /content/prospects_team_year/{YEAR}~{TEAM_ID}~{SOURCE}/

That grid is therefore a self-documenting index of exactly which lists exist
for a team -- no blind year x source guessing. Each list page exposes the full
ranked table (rank, mlb_rank, player, bio, draft info). Three "current status"
columns are premium-locked and ignored.

robots.txt allows /content/ (only search/tracker/premium/etc. are disallowed).
We still throttle to be polite.

Usage
-----
    # one team, to validate
    python -m prospects.ingestion.baseballcube_loader --team 3

    # everything (resumable; skips lists already in the CSV)
    python -m prospects.ingestion.baseballcube_loader --all

Output: data/baseballcube_team_rankings.csv (one row per player-year-source).
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional


BASE = "https://www.thebaseballcube.com"
PROSPECTS_URL = f"{BASE}/content/prospects/"
DROPDOWNS_URL = f"{BASE}/code_2026/ajax/dropdowns.asp?dd=prospects&view=&ID="
TEAM_URL = f"{BASE}/content/prospects_team/{{tid}}/"
LIST_URL = f"{BASE}/content/prospects_team_year/{{key}}/"

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

SOURCE_NAMES = {
    "BA": "Baseball America",
    "PIPE": "MLB Pipeline",
    "ESPN": "ESPN",
    "BP": "Baseball Prospectus",
    "FG": "FanGraphs",
}

DEFAULT_CSV = Path(__file__).resolve().parents[2] / "data" / "baseballcube_team_rankings.csv"

# CSV schema. tbc_player_id is TBC's internal id (for later MLBAM mapping).
FIELDNAMES = [
    "year", "team_id", "team", "source", "source_name",
    "org_rank", "mlb_rank", "tbc_player_id", "player",
    "pos", "ht", "wt", "bats", "throws", "born", "place",
    "hilvl", "mlb_years", "stat_years", "draft_info",
]


# --------------------------------------------------------------------------- #
# HTTP                                                                         #
# --------------------------------------------------------------------------- #
# The site sits behind Cloudflare, which fingerprints the TLS/HTTP stack and
# challenges python-requests (returns a "Just a moment..." 403). curl's
# fingerprint passes cleanly with a browser User-Agent, so we shell out to it.
def _get(url: str, retries: int = 3, delay: float = 1.0) -> Optional[str]:
    """GET via curl with retry/backoff. Returns body, or None on failure."""
    cmd = [
        "curl", "-s", "-A", UA, "-e", PROSPECTS_URL,
        "-w", "\n%{http_code}", "--max-time", "30", url,
    ]
    for attempt in range(retries):
        try:
            raw = subprocess.run(
                cmd, capture_output=True, timeout=45,
            ).stdout
            out = raw.decode("utf-8", errors="replace")
            nl = out.rfind("\n")
            body, code = out[:nl], out[nl + 1:].strip()
            if code == "200":
                return body
            if code == "404":
                return None
            # 403/429/5xx -> back off and retry
        except Exception:
            pass
        time.sleep(delay * (attempt + 1) * 2)
    return None


# --------------------------------------------------------------------------- #
# Parsing                                                                      #
# --------------------------------------------------------------------------- #
_TD_RE = re.compile(r"<td.*?</td>", re.S)
_TR_RE = re.compile(r"<tr.*?</tr>", re.S)
_TAG_RE = re.compile(r"<[^>]+>")


def _cells(row_html: str) -> List[str]:
    out = []
    for c in _TD_RE.findall(row_html):
        t = _TAG_RE.sub(" ", c)
        t = t.replace("&nbsp;", " ")
        t = re.sub(r"\s+", " ", t).strip()
        out.append(t)
    return out


def _grid(html: str, grid_id: str) -> Optional[str]:
    marker = f"id='{grid_id}'"
    i = html.find(marker)
    if i < 0:
        return None
    j = html.find("</table>", i)
    return html[i:j] if j > 0 else html[i:]


def get_teams() -> Dict[int, str]:
    """team_id -> franchise name, from the prospects dropdown AJAX."""
    html = _get(DROPDOWNS_URL)
    if not html:
        raise RuntimeError("could not load team dropdown")
    teams: Dict[int, str] = {}
    for m in re.finditer(r"prospects_team/(\d+)/'>([^<]+)</option>", html):
        teams[int(m.group(1))] = m.group(2).strip()
    if not teams:  # fall back to ?ID= form
        for m in re.finditer(r"prospects_team\.asp\?ID=(\d+)[^>]*>([^<]+)</option>", html):
            teams[int(m.group(1))] = m.group(2).strip()
    return dict(sorted(teams.items()))


def get_team_list_keys(team_id: int) -> List[str]:
    """All '{year}~{team_id}~{source}' list keys available for a franchise."""
    html = _get(TEAM_URL.format(tid=team_id))
    if not html:
        return []
    g2 = _grid(html, "grid2")
    if not g2:
        return []
    keys = re.findall(r"prospects_team_year/(\d+~\d+~[A-Z]+)/", g2)
    return list(dict.fromkeys(keys))  # de-dup, preserve order


def parse_list(key: str) -> List[Dict[str, str]]:
    """Fetch one team-year list and return ranked player rows."""
    year_s, tid_s, source = key.split("~")
    html = _get(LIST_URL.format(key=key))
    if not html:
        return []
    g1 = _grid(html, "grid1")
    if not g1:
        return []
    rows = _TR_RE.findall(g1)
    if not rows:
        return []

    header = _cells(rows[0])
    # Map header label -> index so we're resilient to column reordering.
    hidx = {h.lower(): i for i, h in enumerate(header)}

    def col(cells: List[str], *names: str) -> str:
        for n in names:
            i = hidx.get(n)
            if i is not None and i < len(cells):
                return cells[i]
        return ""

    out: List[Dict[str, str]] = []
    for r in rows[1:]:
        cells = _cells(r)
        if not cells:
            continue
        rank = col(cells, "rank")
        if not rank.isdigit():
            continue  # skip footer rows like "31 record(s)"
        pid_m = re.search(r"/content/player/(\d+)/", r)
        out.append({
            "year": year_s,
            "team_id": tid_s,
            "team": col(cells, "team"),
            "source": source,
            "source_name": SOURCE_NAMES.get(source, source),
            "org_rank": rank,
            "mlb_rank": col(cells, "mlb rank"),
            "tbc_player_id": pid_m.group(1) if pid_m else "",
            "player": col(cells, "player"),
            "pos": col(cells, "pos"),
            "ht": col(cells, "ht"),
            "wt": col(cells, "wt"),
            "bats": col(cells, "ba"),
            "throws": col(cells, "th"),
            "born": col(cells, "born"),
            "place": col(cells, "place"),
            "hilvl": col(cells, "hilvl"),
            "mlb_years": col(cells, "mlb years"),
            "stat_years": col(cells, "stat years"),
            "draft_info": col(cells, "draft info"),
        })
    return out


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #
def _load_done_keys(csv_path: Path) -> set:
    """Set of '{year}~{team_id}~{source}' already present in the CSV (resume)."""
    done = set()
    if not csv_path.exists():
        return done
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add(f"{row['year']}~{row['team_id']}~{row['source']}")
    return done


def pull(
    team_ids: Optional[Iterable[int]] = None,
    csv_path: Path = DEFAULT_CSV,
    delay: float = 1.0,
    verbose: bool = True,
) -> int:
    """Scrape team-year lists into csv_path. Resumable. Returns rows written."""
    teams = get_teams()
    if verbose:
        print(f"[tbc] {len(teams)} franchises discovered")
    if team_ids is not None:
        team_ids = list(team_ids)
        teams = {t: teams.get(t, f"team-{t}") for t in team_ids}

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done_keys(csv_path)
    if verbose and done:
        print(f"[tbc] resuming -- {len(done)} (year,team,source) lists already saved")

    is_new = not csv_path.exists()
    written = 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if is_new:
            w.writeheader()

        for tid, name in teams.items():
            keys = get_team_list_keys(tid)
            time.sleep(delay)
            todo = [k for k in keys if k not in done]
            if verbose:
                print(f"[tbc] team {tid:>3} {name:<24} "
                      f"{len(keys):>3} lists, {len(todo):>3} new")
            for k in todo:
                rows = parse_list(k)
                for row in rows:
                    row["team"] = row["team"] or name
                    w.writerow(row)
                written += len(rows)
                done.add(k)
                f.flush()
                if verbose and rows:
                    print(f"        {k:<16} {len(rows):>2} players  "
                          f"(#1 {rows[0]['player']})")
                time.sleep(delay)

    if verbose:
        print(f"\n[tbc] done. {written} ranking rows written to {csv_path}")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Scrape TBC per-team Top-30 lists")
    ap.add_argument("--team", type=int, action="append",
                    help="franchise id (repeatable). Omit with --all for everything.")
    ap.add_argument("--all", action="store_true", help="scrape all franchises")
    ap.add_argument("--out", type=Path, default=DEFAULT_CSV, help="output CSV path")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    ap.add_argument("--list-teams", action="store_true",
                    help="just print the franchise id->name map and exit")
    args = ap.parse_args()

    if args.list_teams:
        for tid, name in get_teams().items():
            print(f"{tid:>3}  {name}")
        return

    if not args.all and not args.team:
        ap.error("pass --all or one/more --team IDs")

    pull(
        team_ids=None if args.all else args.team,
        csv_path=args.out,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
