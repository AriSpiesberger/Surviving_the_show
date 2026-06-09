"""Scrape FanGraphs Prospect Board season snapshots (20/80 scouting grades).

The Board is a Next.js app; each season's edition lives at
    https://www.fangraphs.com/prospects/the-board/{year}-prospect-list
and the full row set (FV, hit/raw/power/speed/field, pitch grades, present-vs-
future splits, ~94 cols) is embedded in the page's __NEXT_DATA__ blob under
props.pageProps.dehydratedState (React-Query cache). One page = the whole
board for that season, point-in-time (rows carry Season={year}).

Cloudflare is bypassed via curl_cffi's Chrome TLS impersonation — no cookie
needed in practice. If FG ever hardens it, drop a cf_clearance cookie in via
the FG_CF_CLEARANCE / FG_USER_AGENT env vars (see _build_cookies).

Cache layout:
  scratch/fangraphs_board/
    raw/{year}/board.html            # raw page (so re-parse never re-hits net)
    raw/{year}/board.meta.json       # status, row count, season check
    parsed/{year}.csv                # flattened rows, one per prospect

Usage:
    python -m scripts.scrape_fangraphs_board --year 2023        # one season
    python -m scripts.scrape_fangraphs_board --start 2014 --end 2026
    python -m scripts.scrape_fangraphs_board --parse-only       # re-parse cache

Rate-limited to one request / 3s by default.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
try:
    from curl_cffi import requests as crequests
except ImportError:
    crequests = None

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "scratch" / "fangraphs_board"
RAW_DIR = CACHE_DIR / "raw"
PARSED_DIR = CACHE_DIR / "parsed"

BOARD_URL = "https://www.fangraphs.com/prospects/the-board/{year}-prospect-list"

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/120.0.0.0 Safari/537.36")
HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.fangraphs.com/prospects/the-board/",
}

_NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', re.S)


def _build_cookies() -> dict:
    """Optional cf_clearance / FG session cookies from env (rarely needed)."""
    cookies = {}
    for env_name, cookie_name in (
        ("FG_CF_CLEARANCE", "cf_clearance"),
        ("FG_CF_BM", "__cf_bm"),
        ("FG_WORDPRESS_COOKIE", "wordpress_logged_in"),
    ):
        v = os.environ.get(env_name)
        if v:
            cookies[cookie_name] = v
    return cookies


def _http_get(url: str, cookies: dict, ua: str | None = None):
    headers = dict(HEADERS)
    if ua:
        headers["User-Agent"] = ua
    if crequests is not None:
        return crequests.get(url, headers=headers, cookies=cookies,
                             impersonate="chrome120", timeout=60)
    return requests.get(url, headers=headers, cookies=cookies, timeout=60)


def _extract_board_rows(html: str, expect_year: int | None = None
                        ) -> tuple[list[dict] | None, str | None]:
    """Pull the prospect-list rows out of the page's __NEXT_DATA__ blob.

    Returns (rows, error). rows is the largest list-of-dicts found in the
    React-Query dehydratedState (the board itself; the other query is a tiny
    team-info lookup).

    If expect_year is given, the rows' dominant Season MUST equal it — FG
    serves HTTP 200 + the CURRENT board for nonexistent years (e.g. pre-2017),
    so without this check we'd silently save duplicate current-board copies."""
    m = _NEXT_DATA_RE.search(html)
    if not m:
        return None, "no __NEXT_DATA__ script (Cloudflare page or layout change?)"
    try:
        nd = json.loads(m.group(1))
    except Exception as e:
        return None, f"__NEXT_DATA__ not valid JSON: {e}"
    queries = (nd.get("props", {}).get("pageProps", {})
               .get("dehydratedState", {}).get("queries", []))
    best: list[dict] | None = None
    for q in queries:
        data = q.get("state", {}).get("data")
        if (isinstance(data, list) and data and isinstance(data[0], dict)
                and (best is None or len(data) > len(best))):
            best = data
    if not best:
        return None, "no row list in dehydratedState (board empty for this year?)"
    if expect_year is not None:
        seasons = [str(r.get("Season")) for r in best
                   if r.get("Season") is not None]
        if seasons:
            from collections import Counter
            top = Counter(seasons).most_common(1)[0][0]
            if top != str(expect_year):
                return None, (f"FG served Season={top}, not {expect_year} — "
                              f"no board exists for this year (got current "
                              f"board fallback); discarded")
    return best, None


def _fetch_year(year: int, cookies: dict, ua: str | None,
                sleep_s: float) -> dict[str, Any]:
    url = BOARD_URL.format(year=year)
    time.sleep(sleep_s)
    try:
        r = _http_get(url, cookies, ua)
    except Exception as e:
        return {"year": year, "url": url, "status": "ERR", "error": str(e),
                "html": "", "rows": None}
    html = r.content.decode("utf-8", "replace")
    rows, err = (None, f"HTTP {r.status_code}")
    if r.status_code == 200:
        rows, err = _extract_board_rows(html, expect_year=year)
    return {"year": year, "url": url, "status": r.status_code,
            "html": html, "rows": rows, "error": err}


def _save(year: int, res: dict[str, Any]) -> int:
    ydir = RAW_DIR / str(year)
    ydir.mkdir(parents=True, exist_ok=True)
    if res.get("html"):
        (ydir / "board.html").write_text(res["html"], encoding="utf-8")
    rows = res.get("rows")
    n = len(rows) if rows else 0
    seasons = {}
    if rows:
        df = pd.DataFrame(rows)
        PARSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(PARSED_DIR / f"{year}.csv", index=False, encoding="utf-8")
        if "Season" in df.columns:
            seasons = df["Season"].astype(str).value_counts().to_dict()
    (ydir / "board.meta.json").write_text(json.dumps({
        "year": year, "url": res["url"], "status": res["status"],
        "n_rows": n, "season_values": seasons, "error": res.get("error"),
        "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2))
    return n


def _summary(year: int, res: dict[str, Any], n: int):
    if not n:
        print(f"  {year}: status={res['status']}  rows=0  "
              f"-> {res.get('error')}")
        return
    rows = res["rows"]
    df = pd.DataFrame(rows)
    season = (df["Season"].astype(str).value_counts().to_dict()
              if "Season" in df.columns else "?")
    print(f"  {year}: status=200  rows={n:,}  Season={season}")
    grade_cols = [c for c in ("FV_Current", "cFV", "Hit", "Raw", "Game",
                              "FB", "SL", "CMD") if c in df.columns]
    print(f"     {len(df.columns)} cols; grade cols present: {grade_cols}")
    s = rows[0]
    print(f"     sample: {s.get('playerName')} "
          f"FV={s.get('FV_Current') or s.get('cFV')} "
          f"PlayerId={s.get('PlayerId')} Pos={s.get('Position')}")


def _parse_only():
    htmls = sorted(RAW_DIR.glob("*/board.html"))
    if not htmls:
        print(f"No cached board.html under {RAW_DIR}")
        return
    for h in htmls:
        year = int(h.parent.name)
        html = h.read_text(encoding="utf-8")
        rows, err = _extract_board_rows(html, expect_year=year)
        res = {"year": year, "url": BOARD_URL.format(year=year),
               "status": 200 if rows else "parse", "html": "", "rows": rows,
               "error": err}
        n = _save(year, res)
        _summary(year, {**res, "rows": rows}, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--start", type=int, default=2014)
    ap.add_argument("--end", type=int, default=2026)
    ap.add_argument("--sleep", type=float, default=3.0)
    ap.add_argument("--force", action="store_true",
                    help="Re-fetch even if board.html cache exists")
    ap.add_argument("--parse-only", action="store_true",
                    help="Re-parse cached board.html without hitting network")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PARSED_DIR.mkdir(parents=True, exist_ok=True)

    if args.parse_only:
        _parse_only()
        return

    if crequests is None:
        print("WARNING: curl_cffi not installed — plain requests will likely "
              "be Cloudflare-blocked. Run: pip install curl_cffi")
    cookies = _build_cookies()
    ua_override = os.environ.get("FG_USER_AGENT")
    if cookies:
        print(f"Using cookies: {list(cookies)}  UA={ua_override or '(default)'}")

    years = ([args.year] if args.year is not None
             else list(range(args.start, args.end + 1)))
    print(f"Years: {years}\nCache: {RAW_DIR}\n")

    for y in years:
        cache = RAW_DIR / str(y) / "board.html"
        if cache.exists() and not args.force:
            print(f"  {y}: cache hit — re-parsing (use --force to refetch)")
            html = cache.read_text(encoding="utf-8")
            rows, err = _extract_board_rows(html, expect_year=y)
            res = {"year": y, "url": BOARD_URL.format(year=y),
                   "status": 200, "html": "", "rows": rows, "error": err}
            _summary(y, res, _save(y, res))
            continue
        print(f"  {y}: fetching {BOARD_URL.format(year=y)}")
        res = _fetch_year(y, cookies, ua_override, args.sleep)
        _summary(y, res, _save(y, res))


if __name__ == "__main__":
    main()
