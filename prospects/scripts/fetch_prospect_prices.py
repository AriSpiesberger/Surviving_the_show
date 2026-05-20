"""
For each player in the top-N of a grades CSV, query eBay Browse API for their
1st Bowman Chrome auto (numbered /99 and /499) and write a prices CSV.

Usage:
    python -m prospects.scripts.fetch_prospect_prices \\
        --grades grades_probs_v1.4b.csv \\
        --top-n 100 \\
        --sort-by composite_score_raw \\
        --out prices_top100.csv

Reads credentials from environment / .env (see .env.example).
"""
from __future__ import annotations

import argparse
import csv
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Load .env if present
_env = Path(__file__).resolve().parents[2] / ".env"
if _env.exists():
    for line in _env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

from prospects.market.ebay_client import EbayBrowseClient
from prospects.market.price_aggregator import summarize
from prospects.market.query_builder import build_card_spec

try:
    from tqdm import tqdm
except ImportError:
    # Fallback: pass-through identity
    def tqdm(iterable, **kw):
        return iterable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grades", default="grades_probs_v1.4b.csv")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--sort-by", default="composite_score_raw")
    parser.add_argument("--out", default="prices_top100.csv")
    parser.add_argument("--per-query-limit", type=int, default=50)
    parser.add_argument("--sleep-between-players", type=float, default=0.0,
                        help="Sleep between players (set 0 with --workers>1)")
    parser.add_argument("--workers", type=int, default=6,
                        help="Parallel eBay request workers (eBay app tier "
                             "tolerates ~5/sec; 4-6 is the sweet spot)")
    parser.add_argument("--max-pages", type=int, default=1,
                        help="Pages per query. 1 is usually enough since "
                             "Browse API sorts by price ascending")
    parser.add_argument("--single-query-only", action="store_true",
                        default=False,
                        help="Only issue the tight (1st) query — skip wider "
                             "fallbacks. Default False (use fallback if first "
                             "query is empty).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print queries without hitting the API")
    args = parser.parse_args()

    with open(args.grades, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows):,} prospects from {args.grades}")
    rows.sort(key=lambda r: float(r.get(args.sort_by) or 0), reverse=True)
    cohort = rows[:args.top_n]
    print(f"Selecting top {len(cohort):,} by {args.sort_by}")
    env = os.environ.get("EBAY_ENV", "production")
    print(f"eBay env: {env}")

    client = None
    if not args.dry_run:
        client = EbayBrowseClient()
        if not (client._bearer or (client.client_id and client.client_secret)):
            raise SystemExit(
                "No eBay credentials. Set EBAY_BEARER_TOKEN or "
                "EBAY_CLIENT_ID + EBAY_CLIENT_SECRET (in .env or env vars)."
            )

    out_rows = []
    n_with_market = 0
    n_no_card_year = 0
    n_api_errors = 0
    n_zero_results = 0
    error_samples: list[str] = []
    out_lock = threading.Lock()

    def _one_player(row: dict) -> tuple[bool, bool, list[dict], int, int, str | None]:
        """Returns (skipped_no_year, has_market, output_rows,
                    api_errors, zero_result_queries, first_error_msg)."""
        spec = build_card_spec(row)
        if spec is None:
            return (True, False, [], 0, 0, None)
        queries = spec.queries[:1] if args.single_query_only else spec.queries
        all_listings = []
        api_errors = 0
        zero_results = 0
        first_err: str | None = None
        for q in queries:
            try:
                lst = client.search(q, limit=args.per_query_limit,
                                    max_pages=args.max_pages)
            except Exception as e:
                api_errors += 1
                if first_err is None:
                    first_err = f"{spec.name!r}: {type(e).__name__}: {e}"
                continue
            if not lst:
                zero_results += 1
            all_listings.extend(lst)
            if lst:
                break
        summaries = summarize(spec.player_id, spec.name, spec.card_year,
                              all_listings)
        rows_out: list[dict] = []
        any_market = False
        for s in summaries:
            d = s.as_dict()
            for col in ("p_MLB_DEBUT", "p_ESTABLISHED_MLB", "p_ALL_STAR_ONCE",
                        "p_ELITE", "composite_score", "composite_score_raw",
                        "grade", "percentile", "cur_level", "is_international",
                        "draft_round", "draft_pick"):
                d[col] = row.get(col, "")
            rows_out.append(d)
            if s.has_market:
                any_market = True
        return (False, any_market, rows_out, api_errors, zero_results, first_err)

    if args.dry_run:
        for i, row in enumerate(cohort, 1):
            spec = build_card_spec(row)
            if spec is None:
                continue
            print(f"[{i:3d}] {spec.name:<30} card_year={spec.card_year} "
                  f"queries={spec.queries[0]!r}")
        return

    # Streaming CSV writer — flushes after every player so a kill mid-run
    # preserves all completed work.
    out_fh = None
    out_writer = None
    fieldnames: list[str] | None = None

    pbar = tqdm(total=len(cohort), desc="eBay fetch", unit="player",
                dynamic_ncols=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_one_player, r) for r in cohort]
        for fut in as_completed(futures):
            skipped, any_market, rows, errs, zeros, err_msg = fut.result()
            with out_lock:
                if skipped:
                    n_no_card_year += 1
                if any_market:
                    n_with_market += 1
                n_api_errors += errs
                n_zero_results += zeros
                if err_msg and len(error_samples) < 5:
                    error_samples.append(err_msg)
                out_rows.extend(rows)
                # Stream-write any new rows
                if rows:
                    if out_writer is None:
                        fieldnames = list(rows[0].keys())
                        out_fh = open(args.out, "w", newline="",
                                      encoding="utf-8")
                        out_writer = csv.DictWriter(out_fh,
                                                    fieldnames=fieldnames)
                        out_writer.writeheader()
                    for d in rows:
                        # Force consistent schema across rows
                        out_writer.writerow({k: d.get(k, "")
                                             for k in fieldnames})
                    out_fh.flush()
            pbar.update(1)
            pbar.set_postfix(
                market=n_with_market,
                skipped=n_no_card_year,
                err=n_api_errors,
                zero=n_zero_results,
            )
            if args.sleep_between_players > 0:
                time.sleep(args.sleep_between_players)
    pbar.close()
    if out_fh is not None:
        out_fh.close()
    if error_samples:
        print("\nAPI error samples:")
        for s in error_samples:
            print(f"  {s}")
    print(f"\nDiagnostics:")
    print(f"  API errors      : {n_api_errors}")
    print(f"  zero-result queries: {n_zero_results}")
    print(f"  prospects skipped (no card year): {n_no_card_year}")
    print(f"  prospects with at least one accepted listing: {n_with_market}")

    print(f"\nWrote {len(out_rows):,} rows to {args.out}")
    print(f"  prospects with at least one accepted listing: {n_with_market}")
    print(f"  prospects skipped (no card year): {n_no_card_year}")


if __name__ == "__main__":
    main()
