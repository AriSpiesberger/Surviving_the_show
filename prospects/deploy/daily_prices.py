"""Daily eBay price refresh — runs on Railway at 13:00 UTC.

Pulls base 1st Bowman Chrome auto raw-card prices for:
  1) every player in the latest buy list (from --buy-list)
  2) every player_id in holdings (from --holdings, optional)

Writes:
  - prices/prices_buylist_YYYY-MM-DD.csv          (today's snapshot)
  - prices/prices_buylist_latest.csv              (symlink-like; rewritten)
  - prices/prices_holdings_YYYY-MM-DD.csv         (if holdings provided)
  - prices/prices_holdings_latest.csv             (if holdings provided)

The output schema includes lowest_buynow_price + lowest_buynow_url, which is
what the alerts job consumes for 2x triggers and what you use for buys.

Usage (local):
    python -m prospects.deploy.daily_prices \\
        --buy-list buy_list_v1.17_FINAL.csv \\
        --db prospects.db \\
        --out-dir prices \\
        --holdings holdings.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from prospects.market.ebay_client import EbayBrowseClient
from prospects.market.price_aggregator import summarize
from prospects.market.query_builder import build_card_spec


OUT_FIELDS = [
    "player_id", "name", "card_year", "denominator",
    "n_listings", "n_auctions", "n_fixed",
    "price_min", "price_p25", "price_median", "price_mean",
    "price_p75", "price_max",
    "lowest_buynow_price", "lowest_buynow_url",
    "top_listing_url", "top_listing_title",
    "has_market", "snapshot_date",
]


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


def _enrich_with_db(rows: list[dict], db_path: str) -> int:
    """Add draft_year / is_international / start_year via prospects.db join."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    n_matched = 0
    for r in rows:
        pid = r.get("player_id")
        if not pid:
            r["draft_year"] = r.get("draft_year", "")
            r["is_international"] = r.get("is_international", 0)
            r["start_year"] = r.get("start_year", "")
            continue
        p = conn.execute(
            "SELECT draft_year, is_international FROM prospects "
            "WHERE player_id = ?",
            (pid,),
        ).fetchone()
        if p is None:
            r.setdefault("draft_year", "")
            r.setdefault("is_international", 0)
            r.setdefault("start_year", "")
            continue
        n_matched += 1
        r["draft_year"] = p["draft_year"] if p["draft_year"] is not None else ""
        r["is_international"] = p["is_international"] or 0
        if r["draft_year"] == "" and r["is_international"]:
            sy = conn.execute(
                "SELECT MIN(season_year) AS sy FROM season_stats "
                "WHERE player_id = ?",
                (pid,),
            ).fetchone()
            r["start_year"] = sy["sy"] if sy and sy["sy"] is not None else ""
        else:
            r["start_year"] = ""
    conn.close()
    return n_matched


def _fetch_one(client: EbayBrowseClient, row: dict, snapshot_date: str,
               per_query_limit: int, max_pages: int, single_query: bool
               ) -> list[dict]:
    spec = build_card_spec(row)
    if spec is None:
        return []
    queries = spec.queries[:1] if single_query else spec.queries
    all_listings = []
    for q in queries:
        try:
            lst = client.search(q, limit=per_query_limit, max_pages=max_pages)
        except Exception:
            continue
        all_listings.extend(lst)
        if lst:
            break
    rows_out: list[dict] = []
    for s in summarize(spec.player_id, spec.name, spec.card_year, all_listings):
        if s.denominator != 0:
            continue  # base only
        d = s.as_dict()
        d["snapshot_date"] = snapshot_date
        rows_out.append(d)
    return rows_out


def _run_cohort(client: EbayBrowseClient, cohort: list[dict],
                snapshot_date: str, workers: int, per_query_limit: int,
                max_pages: int, single_query: bool) -> list[dict]:
    out: list[dict] = []
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_fetch_one, client, r, snapshot_date,
                            per_query_limit, max_pages, single_query)
                for r in cohort]
        for fut in as_completed(futs):
            rows = fut.result()
            with lock:
                out.extend(rows)
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in OUT_FIELDS})


def _read_holdings(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    seen: set[str] = set()
    deduped: list[dict] = []
    for r in rows:
        pid = r.get("player_id")
        if not pid or pid in seen:
            continue
        seen.add(pid)
        deduped.append({"player_id": pid, "name": r.get("name", "")})
    return deduped


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--buy-list", required=True,
                   help="Path to the buy-list CSV produced by the weekly score")
    p.add_argument("--db", default=os.environ.get("PROSPECT_DB",
                                                  "/data/prospects.db"))
    p.add_argument("--out-dir", default=os.environ.get("PRICES_DIR",
                                                       "/data/prices"))
    p.add_argument("--holdings", default=os.environ.get("HOLDINGS_PATH",
                                                        "/data/holdings.csv"))
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--per-query-limit", type=int, default=50)
    p.add_argument("--max-pages", type=int, default=1)
    p.add_argument("--single-query-only", action="store_true",
                   help="Issue only the tight query; skip wider fallbacks")
    args = p.parse_args()

    _load_env_file(Path(".env"))

    t0 = time.time()
    snapshot_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"[daily_prices] snapshot_date={snapshot_date}")
    print(f"[daily_prices] buy-list={args.buy_list}  db={args.db}  "
          f"out={args.out_dir}")

    client = EbayBrowseClient()
    if not (client._bearer or (client.client_id and client.client_secret)):
        print("ERROR: no eBay credentials (EBAY_BEARER_TOKEN or "
              "EBAY_CLIENT_ID + EBAY_CLIENT_SECRET)", file=sys.stderr)
        return 2

    # ---- Buy list ----
    with open(args.buy_list, encoding="utf-8") as f:
        buy_rows = list(csv.DictReader(f))
    n_buy = _enrich_with_db(buy_rows, args.db)
    print(f"[daily_prices] buy-list: {len(buy_rows)} players "
          f"({n_buy} matched in db)")

    buy_prices = _run_cohort(client, buy_rows, snapshot_date,
                             args.workers, args.per_query_limit,
                             args.max_pages, args.single_query_only)
    n_market_buy = sum(1 for r in buy_prices if r.get("has_market"))
    out_dir = Path(args.out_dir)
    dated = out_dir / f"prices_buylist_{snapshot_date}.csv"
    latest = out_dir / "prices_buylist_latest.csv"
    _write_csv(dated, buy_prices)
    _write_csv(latest, buy_prices)
    print(f"[daily_prices] buy-list prices: {len(buy_prices)} rows, "
          f"{n_market_buy} with market -> {dated}")

    # ---- Holdings ----
    holdings = _read_holdings(Path(args.holdings))
    if holdings:
        n_h = _enrich_with_db(holdings, args.db)
        print(f"[daily_prices] holdings: {len(holdings)} unique players "
              f"({n_h} matched in db)")
        h_prices = _run_cohort(client, holdings, snapshot_date,
                               args.workers, args.per_query_limit,
                               args.max_pages, args.single_query_only)
        n_market_h = sum(1 for r in h_prices if r.get("has_market"))
        h_dated = out_dir / f"prices_holdings_{snapshot_date}.csv"
        h_latest = out_dir / "prices_holdings_latest.csv"
        _write_csv(h_dated, h_prices)
        _write_csv(h_latest, h_prices)
        print(f"[daily_prices] holdings prices: {len(h_prices)} rows, "
              f"{n_market_h} with market -> {h_dated}")
    else:
        print(f"[daily_prices] holdings file empty or missing "
              f"({args.holdings}); skipping holdings revaluation")

    print(f"[daily_prices] DONE in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
