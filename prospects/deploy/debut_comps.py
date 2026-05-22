"""Weekly eBay price refresh for current-season MLB debut comps.

Pulls base 1st Bowman Chrome auto raw-card prices for every player who has
debuted in MLB in --year, restricted to the R2-R3, R4-R10, R11+, and IFA
draft buckets. (R1 picks are typically already on the buy list.)

These rows are comparison data — they reveal what the realized market does
for non-R1 prospects who actually reach The Show, which is the population
the buy-list filter excludes by design.

Writes:
  - prices/prices_debut_comps_YYYY-MM-DD.csv  (today's snapshot)
  - prices/prices_debut_comps_latest.csv      (rewritten each run)

Usage:
    python -m prospects.deploy.debut_comps --year 2026
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
from datetime import date, datetime, timezone
from pathlib import Path

import requests

from prospects.market.ebay_client import EbayBrowseClient
from prospects.market.price_aggregator import summarize
from prospects.market.query_builder import build_card_spec


OUT_FIELDS = [
    "player_id", "name", "card_year", "denominator",
    "draft_year", "draft_round", "is_international", "bucket",
    "debut_year", "debut_date", "days_since_debut",
    "games_pitched", "games_batted", "innings_pitched", "ip_per_game",
    "role",
    "primary_position", "current_org",
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


def _bucket_for(draft_round) -> str:
    if draft_round is None:
        return "IFA"
    r = int(draft_round)
    if r == 1:
        return "R1"
    if r <= 3:
        return "R2-R3"
    if r <= 10:
        return "R4-R10"
    return "R11+"


def load_debut_cohort(db_path: str, year: int) -> list[dict]:
    """Return one row per true {year} debutant in the R2+/IFA buckets.

    "True debutant" = appears in season_stats at level='MLB' in {year} and
    has NO prior season_stats row at level='MLB'. More reliable than reading
    career_outcomes.mlb_debut_year, which is refreshed out-of-band by the
    Chadwick / Lahman ingestion path and lags real call-ups by days-to-weeks.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    q = """
        WITH mlb_curr AS (
            SELECT DISTINCT player_id FROM season_stats
            WHERE season_year = ? AND level = 'MLB'
        ),
        mlb_prior AS (
            SELECT DISTINCT player_id FROM season_stats
            WHERE season_year < ? AND level = 'MLB'
        )
        SELECT p.player_id, p.name, p.draft_year, p.draft_round, p.draft_pick,
               p.is_international, p.primary_position, p.current_org,
               p.mlbam_id
        FROM mlb_curr m
        JOIN prospects p USING (player_id)
        WHERE m.player_id NOT IN (SELECT player_id FROM mlb_prior)
          AND (p.draft_round IS NULL OR p.draft_round >= 2)
        ORDER BY p.draft_round IS NULL, p.draft_round, p.draft_pick
    """
    rows = []
    for r in conn.execute(q, (year, year)):
        d = dict(r)
        d["debut_year"] = year
        d["bucket"] = _bucket_for(d.get("draft_round"))
        if d["is_international"] and (d["draft_year"] is None or d["draft_year"] == ""):
            sy = conn.execute(
                "SELECT MIN(season_year) AS sy FROM season_stats "
                "WHERE player_id = ?",
                (d["player_id"],),
            ).fetchone()
            d["start_year"] = sy["sy"] if sy and sy["sy"] is not None else ""
        else:
            d["start_year"] = ""
        rows.append(d)
    conn.close()
    return rows


def _parse_ip(ip_str) -> float | None:
    """MLB API returns IP as a string like '21.2' meaning 21 and 2/3 innings."""
    if ip_str is None or ip_str == "":
        return None
    s = str(ip_str)
    whole, _, frac = s.partition(".")
    try:
        return int(whole) + (int(frac) / 3 if frac else 0)
    except (ValueError, TypeError):
        return None


def fetch_mlb_stats(mlbam_id: str, season: int, timeout: float = 10.0) -> dict:
    """Hit MLB Stats API for one player's current-season pitching + hitting
    summary. Returns {debut_date, games_pitched, games_batted, innings_pitched}.

    Network failures and missing fields degrade to None — this is enrichment,
    not a hard dependency."""
    out = {
        "debut_date": None,
        "games_pitched": None,
        "games_batted": None,
        "innings_pitched": None,
    }
    if not mlbam_id:
        return out
    try:
        info = requests.get(
            f"https://statsapi.mlb.com/api/v1/people/{mlbam_id}",
            timeout=timeout,
        ).json()
        out["debut_date"] = (
            info.get("people", [{}])[0].get("mlbDebutDate") or None
        )
    except Exception:
        pass
    try:
        stats = requests.get(
            f"https://statsapi.mlb.com/api/v1/people/{mlbam_id}/stats"
            f"?stats=season&season={season}&group=hitting,pitching",
            timeout=timeout,
        ).json()
        for s in stats.get("stats", []):
            grp = s.get("group", {}).get("displayName")
            splits = s.get("splits", [])
            if not splits:
                continue
            st = splits[0].get("stat", {})
            if grp == "pitching":
                out["games_pitched"] = st.get("gamesPitched")
                out["innings_pitched"] = _parse_ip(st.get("inningsPitched"))
            elif grp == "hitting":
                out["games_batted"] = st.get("gamesPlayed")
    except Exception:
        pass
    return out


def _role_for(games_pitched, innings_pitched, games_batted) -> str:
    """Tag SP / SWG / RP based on IP/G; BAT if no pitching appearances."""
    if games_pitched and innings_pitched is not None and games_pitched > 0:
        ipg = innings_pitched / games_pitched
        if ipg >= 2.5:
            return "SP"
        if ipg >= 1.5:
            return "SWG"
        return "RP"
    if games_batted:
        return "BAT"
    return ""


def enrich_with_mlb_api(cohort: list[dict], season: int, workers: int = 10
                        ) -> int:
    """Mutate cohort in place, adding debut_date / games / IP / IP/G / role.
    Returns count of players for whom we resolved a debut date."""
    today = datetime.now(timezone.utc).date()
    n_resolved = 0

    def _work(row):
        return row, fetch_mlb_stats(row.get("mlbam_id"), season)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row, stats in pool.map(_work, cohort):
            dstr = stats.get("debut_date")
            row["debut_date"] = dstr or ""
            if dstr:
                try:
                    d = datetime.strptime(dstr, "%Y-%m-%d").date()
                    row["days_since_debut"] = (today - d).days
                    n_resolved += 1
                except ValueError:
                    row["days_since_debut"] = ""
            else:
                row["days_since_debut"] = ""
            row["games_pitched"] = stats.get("games_pitched") or ""
            row["games_batted"] = stats.get("games_batted") or ""
            ip = stats.get("innings_pitched")
            row["innings_pitched"] = round(ip, 2) if ip is not None else ""
            if stats.get("games_pitched") and ip is not None and stats["games_pitched"] > 0:
                row["ip_per_game"] = round(ip / stats["games_pitched"], 2)
            else:
                row["ip_per_game"] = ""
            row["role"] = _role_for(stats.get("games_pitched"), ip,
                                    stats.get("games_batted"))
    return n_resolved


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
        # Carry through the cohort metadata for downstream comp analysis.
        d["draft_year"] = row.get("draft_year") or ""
        d["draft_round"] = row.get("draft_round") if row.get("draft_round") is not None else ""
        d["is_international"] = row.get("is_international") or 0
        d["bucket"] = row.get("bucket", "")
        d["debut_year"] = row.get("debut_year") or ""
        d["primary_position"] = row.get("primary_position") or ""
        d["current_org"] = row.get("current_org") or ""
        # MLB-API enrichment (added by enrich_with_mlb_api before this runs).
        for k in ("debut_date", "days_since_debut", "games_pitched",
                  "games_batted", "innings_pitched", "ip_per_game", "role"):
            d[k] = row.get(k, "")
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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int,
                   default=datetime.now(timezone.utc).year,
                   help="MLB debut year to pull comps for (default: current UTC year)")
    p.add_argument("--db", default=os.environ.get("PROSPECT_DB",
                                                  "/data/prospects.db"))
    p.add_argument("--out-dir", default=os.environ.get("PRICES_DIR",
                                                       "/data/prices"))
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--per-query-limit", type=int, default=50)
    p.add_argument("--max-pages", type=int, default=1)
    p.add_argument("--single-query-only", action="store_true")
    args = p.parse_args()

    _load_env_file(Path(".env"))

    t0 = time.time()
    snapshot_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"[debut_comps] snapshot_date={snapshot_date}  year={args.year}")
    print(f"[debut_comps] db={args.db}  out={args.out_dir}")

    client = EbayBrowseClient()
    if not (client._bearer or (client.client_id and client.client_secret)):
        print("ERROR: no eBay credentials (EBAY_BEARER_TOKEN or "
              "EBAY_CLIENT_ID + EBAY_CLIENT_SECRET)", file=sys.stderr)
        return 2

    cohort = load_debut_cohort(args.db, args.year)
    by_bucket: dict[str, int] = {}
    for r in cohort:
        by_bucket[r["bucket"]] = by_bucket.get(r["bucket"], 0) + 1
    print(f"[debut_comps] cohort: {len(cohort)} debutants  "
          + "  ".join(f"{k}={v}" for k, v in sorted(by_bucket.items())))

    if not cohort:
        print("[debut_comps] no debutants in target buckets; nothing to do")
        return 0

    # MLB Stats API: debut date + games + IP for role classification.
    # Fail-soft — comp pricing is the primary output, this is enrichment.
    try:
        n_dates = enrich_with_mlb_api(cohort, args.year, workers=args.workers)
        print(f"[debut_comps] MLB Stats API enriched {n_dates}/{len(cohort)} "
              f"with debut date + games/IP")
    except Exception as e:
        print(f"[debut_comps] WARN: MLB API enrichment failed ({e}); "
              f"continuing without debut date / IP fields", file=sys.stderr)
        for r in cohort:
            for k in ("debut_date", "days_since_debut", "games_pitched",
                      "games_batted", "innings_pitched", "ip_per_game", "role"):
                r.setdefault(k, "")

    prices = _run_cohort(client, cohort, snapshot_date,
                         args.workers, args.per_query_limit,
                         args.max_pages, args.single_query_only)
    n_market = sum(1 for r in prices if r.get("has_market"))

    out_dir = Path(args.out_dir)
    dated = out_dir / f"prices_debut_comps_{snapshot_date}.csv"
    latest = out_dir / "prices_debut_comps_latest.csv"
    _write_csv(dated, prices)
    _write_csv(latest, prices)
    print(f"[debut_comps] {len(prices)} rows, {n_market} with market -> {dated}")
    print(f"[debut_comps] DONE in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
