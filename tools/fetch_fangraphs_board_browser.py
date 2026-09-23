"""Fetch the FanGraphs Board through a real Chrome when Cloudflare challenges plain HTTP.

As of 2026-09 fangraphs.com answers every scripted request (requests, and curl_cffi with
chrome120-136 / safari TLS profiles) with a Cloudflare JS challenge ("Just a moment...",
cf-mitigated: challenge). A real browser passes it by running the page's JavaScript. This
drives the locally installed Chrome (headed, throwaway profile) via Playwright, waits for
the board's __NEXT_DATA__ blob, validates it with the scraper's own parser, and only then
replaces the cached raw/<year>/board.html. Then run:

    python tools/scrape_fangraphs_board.py --parse-only
    python tools/build_scouting_grades.py

Usage:
    python tools/fetch_fangraphs_board_browser.py            # current season only
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from playwright.sync_api import sync_playwright   # noqa: E402

import scrape_fangraphs_board as fg                # noqa: E402


# A finished season's board is FROZEN (2026-09-22). FanGraphs keeps pruning graduates from
# the still-live prior-season board: re-scraping 2025 in Sept 2026 dropped 75 players, 66 of
# whom had debuted in 2026. Training rows at snap 2025 would then lose their scouting row
# exactly when the label is positive — outcome-dependent missingness, the same class of leak
# as scout_servicetime. Archived boards (2017-2024) are not pruned (11-16% next-year
# debutants each). Only the current season may be refreshed.
def frozen_years(years, allow_past=False):
    from datetime import date
    cur = date.today().year
    past = [y for y in years if y < cur]
    if past and not allow_past:
        raise SystemExit(f"refusing to re-fetch finished season(s) {past}: their boards are "
                         f"frozen (graduate pruning leaks the debut label). Use --allow-past-seasons "
                         f"only for a season never fetched before.")
    return years


def fetch(page, year: int, timeout_s: int) -> str | None:
    url = fg.BOARD_URL.format(year=year)
    page.goto(url, wait_until="domcontentloaded", timeout=timeout_s * 1000)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        html = page.content()
        if "__NEXT_DATA__" in html and "Just a moment" not in html:
            return html
        time.sleep(2)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    from datetime import date
    ap.add_argument("--years", nargs="*", type=int, default=[date.today().year])
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--allow-past-seasons", action="store_true")
    ap.add_argument("--headless", action="store_true",
                    help="usually re-challenged by Cloudflare; headed is the default")
    args = ap.parse_args()
    frozen_years(args.years, args.allow_past_seasons)

    ok = 0
    with sync_playwright() as p, tempfile.TemporaryDirectory() as prof:
        ctx = p.chromium.launch_persistent_context(
            prof, channel="chrome", headless=args.headless,
            args=["--disable-blink-features=AutomationControlled"],
            viewport={"width": 1280, "height": 900})
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        for year in args.years:
            html = fetch(page, year, args.timeout)
            if html is None:
                print(f"  {year}: challenge not cleared within {args.timeout}s — cache left untouched")
                continue
            rows, err = fg._extract_board_rows(html, expect_year=year)
            if not rows:
                print(f"  {year}: page loaded but parse failed ({err}) — cache left untouched")
                continue
            d = fg.RAW_DIR / str(year)
            d.mkdir(parents=True, exist_ok=True)
            (d / "board.html").write_text(html, encoding="utf-8")
            (d / "board.meta.json").write_text(json.dumps({
                "year": year, "url": fg.BOARD_URL.format(year=year), "status": 200,
                "rows": len(rows), "fetched_utc": datetime.now(timezone.utc).isoformat(),
                "via": "playwright/chrome"}, indent=1))
            print(f"  {year}: {len(rows):,} board rows -> {d / 'board.html'}")
            ok += 1
            time.sleep(4)
        ctx.close()
    print(f"done: {ok}/{len(args.years)} years refreshed")
    sys.exit(0 if ok == len(args.years) else 1)


if __name__ == "__main__":
    main()
