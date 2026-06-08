"""Scrape FanGraphs Prospect Board year-end snapshots.

The Board is FG's living prospect ranking with 20/80 scouting grades.
We pull each year's end-of-season snapshot, cache the raw JSON to disk,
and report what we got so we can validate the schema before any DB
ingestion.

The endpoint is undocumented but matches what the public Board page
calls. Format may change without notice — that's why we cache raw JSON
per year, so re-parsing doesn't re-hit the network.

Cache layout:
  scratch/fangraphs_board/
    raw/{year}/scouting_summary.json    # one shot, all positions
    raw/{year}/scouting_summary.meta.json
    parsed/{year}.csv                   # flattened per year (later step)

Usage:
    # Sanity test on one year (saves to cache, prints summary):
    python -m scripts.scrape_fangraphs_board --year 2023

    # Full backfill 2009-current:
    python -m scripts.scrape_fangraphs_board --start 2009 --end 2026

    # Re-parse cache without re-fetching:
    python -m scripts.scrape_fangraphs_board --parse-only

Rate-limited to one request per 3s by default to be polite.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import os
import requests
try:
    from curl_cffi import requests as crequests
except ImportError:
    crequests = None

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "scratch" / "fangraphs_board"
RAW_DIR = CACHE_DIR / "raw"
PARSED_DIR = CACHE_DIR / "parsed"

# These are the endpoints the Board page calls under the hood.
# We're targeting "scouting summary" (the all-grades view).
BOARD_ENDPOINTS = [
    # Newer endpoint shape (live page calls these as of 2025)
    ("scouting-grades",
     "https://www.fangraphs.com/api/prospects/board/data"
     "?statgroup={statgroup}&stattype=grades&type=0&pos=all&team=0"
     "&pageitems=100000&season={season}&seasonend={season}"
     "&hand=all&rookieyearstart=&rookieyearend="
     "&signed=all&active=&minip=&minpa="),
    # Stats-style endpoint (fallback)
    ("scouting-stats",
     "https://www.fangraphs.com/api/prospects/board/data"
     "?statgroup={statgroup}&stattype=stat&type=0&pos=all&team=0"
     "&pageitems=100000&season={season}&seasonend={season}"),
    # Older endpoint pattern (some legacy years)
    ("scouting-summary-legacy",
     "https://www.fangraphs.com/prospects/the-board/json"
     "?year={season}&group={statgroup}"),
]

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
       "AppleWebKit/537.36 (KHTML, like Gecko) "
       "Chrome/120.0.0.0 Safari/537.36")
HEADERS = {
    "User-Agent": UA,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.fangraphs.com/prospects/the-board/scouting-summary",
}


def _build_cookies() -> dict:
    """Read CF clearance + any FG session cookies from env vars.

    Open FG in your browser → DevTools → Application → Cookies →
    fangraphs.com → copy the values, then:

        $env:FG_CF_CLEARANCE = "long-encoded-string..."
        $env:FG_USER_AGENT   = "<the same UA shown in DevTools>"
    """
    cookies = {}
    for env_name, cookie_name in (
        ("FG_CF_CLEARANCE", "cf_clearance"),
        ("FG_WORDPRESS_COOKIE", "wordpress_logged_in"),
        ("FG_PHP_SESSION", "PHPSESSID"),
    ):
        v = os.environ.get(env_name)
        if v:
            cookies[cookie_name] = v
    return cookies


def _http_get(url: str, cookies: dict, ua: str | None = None):
    """Fetch with curl_cffi (chrome TLS fingerprint) + injected cookies.
    Falls back to plain requests if curl_cffi is unavailable."""
    headers = dict(HEADERS)
    if ua:
        headers["User-Agent"] = ua
    if crequests is not None:
        return crequests.get(url, headers=headers, cookies=cookies,
                              impersonate="chrome120", timeout=60)
    return requests.get(url, headers=headers, cookies=cookies, timeout=60)


def _fetch_year(year: int, sleep_s: float, cookies: dict,
                 ua_override: str | None = None) -> dict[str, Any]:
    """Try each endpoint until one returns a usable JSON payload. Returns
    a dict with both hitters + pitchers + metadata."""
    out = {"year": year, "tried": [], "ok": [], "data": {}}
    for statgroup in ("fielders", "pitchers"):
        success = None
        for name, tmpl in BOARD_ENDPOINTS:
            url = tmpl.format(statgroup=statgroup, season=year)
            try:
                time.sleep(sleep_s)
                r = _http_get(url, cookies, ua=ua_override)
                out["tried"].append({"statgroup": statgroup, "name": name,
                                      "url": url, "status": r.status_code,
                                      "length": len(r.content)})
                if r.status_code != 200:
                    continue
                ct = r.headers.get("content-type", "")
                # FG sometimes returns HTML when the endpoint is wrong
                if "json" not in ct.lower() and not r.text.strip().startswith("{"):
                    continue
                try:
                    payload = r.json()
                except Exception:
                    continue
                # Heuristic: payload should be a list or wrap one
                rows = payload if isinstance(payload, list) else (
                    payload.get("data")
                    or payload.get("rows")
                    or payload.get("results")
                    or []
                )
                if not isinstance(rows, list) or len(rows) == 0:
                    continue
                out["data"][statgroup] = {
                    "endpoint": name, "url": url, "row_count": len(rows),
                    "rows": rows,
                }
                out["ok"].append(
                    {"statgroup": statgroup, "name": name,
                     "row_count": len(rows)})
                success = name
                break
            except Exception as e:
                out["tried"].append({"statgroup": statgroup, "name": name,
                                      "url": url, "error": str(e)})
        if success is None:
            print(f"  [WARN] {year} {statgroup}: no endpoint returned data")
    return out


def _save_raw(year: int, payload: dict[str, Any]) -> Path:
    ydir = RAW_DIR / str(year)
    ydir.mkdir(parents=True, exist_ok=True)
    raw_path = ydir / "scouting_summary.json"
    meta_path = ydir / "scouting_summary.meta.json"
    raw_path.write_text(json.dumps(payload["data"], indent=2,
                                    default=str))
    meta = {
        "year": year,
        "tried": payload["tried"],
        "ok": payload["ok"],
        "row_counts": {sg: payload["data"][sg]["row_count"]
                        for sg in payload["data"]},
        "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return raw_path


def _summary(year: int, payload: dict[str, Any]):
    n_hit = payload["data"].get("fielders", {}).get("row_count", 0)
    n_pit = payload["data"].get("pitchers", {}).get("row_count", 0)
    print(f"  {year}: hitters={n_hit:,}  pitchers={n_pit:,}")
    # Sample the first row from each so we can see field names
    for sg in ("fielders", "pitchers"):
        d = payload["data"].get(sg)
        if not d or not d["rows"]:
            continue
        sample = d["rows"][0]
        print(f"    [{sg}] {len(sample)} cols, sample keys: "
              f"{sorted(sample.keys())[:14]}...")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=None,
                    help="Single year for sanity testing")
    ap.add_argument("--start", type=int, default=2009)
    ap.add_argument("--end", type=int, default=2026)
    ap.add_argument("--sleep", type=float, default=3.0,
                    help="Seconds between requests (be polite)")
    ap.add_argument("--force", action="store_true",
                    help="Re-fetch even if cache exists")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PARSED_DIR.mkdir(parents=True, exist_ok=True)

    cookies = _build_cookies()
    ua_override = os.environ.get("FG_USER_AGENT")
    if not cookies:
        print("WARNING: no FG_CF_CLEARANCE env var set. Cloudflare will "
              "almost certainly block this. Open FG in your browser, "
              "DevTools -> Application -> Cookies -> fangraphs.com, copy "
              "the cf_clearance value, then in PowerShell:")
        print('  $env:FG_CF_CLEARANCE = "long-encoded-string..."')
        print('  $env:FG_USER_AGENT   = "<your browser UA from DevTools>"')
        print("Then re-run.")
        print()
    else:
        masked = {k: f"{v[:8]}...{v[-4:]}" for k, v in cookies.items()}
        print(f"Using cookies: {masked}")
        print(f"UA override: {ua_override or '(default chrome)'}")

    years = ([args.year] if args.year is not None
              else list(range(args.start, args.end + 1)))
    print(f"Years to fetch: {years}")
    print(f"Cache: {RAW_DIR}")
    print()

    for y in years:
        cache = RAW_DIR / str(y) / "scouting_summary.json"
        if cache.exists() and not args.force:
            print(f"  {y}: cache hit ({cache.stat().st_size/1024:.0f} KB) — "
                  f"skipping fetch")
            continue
        print(f"  {y}: fetching...")
        payload = _fetch_year(y, sleep_s=args.sleep, cookies=cookies,
                                ua_override=ua_override)
        if not payload["data"]:
            print(f"    [FAIL] {y}: nothing returned. See tried endpoints "
                  f"in cache meta.")
            # Still save the meta so we can inspect what we tried
            ydir = RAW_DIR / str(y); ydir.mkdir(parents=True, exist_ok=True)
            (ydir / "scouting_summary.meta.json").write_text(
                json.dumps({"year": y, "tried": payload["tried"]},
                            indent=2))
            continue
        _save_raw(y, payload)
        _summary(y, payload)


if __name__ == "__main__":
    main()
