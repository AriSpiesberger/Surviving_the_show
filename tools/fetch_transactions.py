"""Pull MLB + MiLB transactions from the MLB Stats API into a local month cache (2026-09-24).

New information source for the model: 40-man selections (Rule 5 protection), options /
recalls, IL placements, in-season assignments (promotions), releases, trades, retirements.
One JSON file per (sportId, month) under reference/transactions/raw/<sid>/<YYYY-MM>.json;
existing files are skipped, so reruns only fetch what is missing (the current month is always
refetched). Nothing is written to the databases.

    python tools/fetch_transactions.py --start 2005 --end 2026
"""
from __future__ import annotations

import argparse
import calendar
import datetime as dt
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "reference" / "transactions" / "raw"
SPORTS = {1: "MLB", 11: "AAA", 12: "AA", 13: "A+", 14: "A", 15: "A-", 16: "RK"}
URL = "https://statsapi.mlb.com/api/v1/transactions?startDate={a}&endDate={b}&sportId={sid}"


def fetch(sid, y, m, force=False):
    out = RAW / str(sid) / f"{y}-{m:02d}.json"
    if out.exists() and not force:
        return sid, y, m, "cached", None
    last = calendar.monthrange(y, m)[1]
    url = URL.format(a=f"{y}-{m:02d}-01", b=f"{y}-{m:02d}-{last}", sid=sid)
    for attempt in range(5):
        try:
            r = requests.get(url, timeout=90)
            r.raise_for_status()
            t = r.json().get("transactions", [])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(t), encoding="utf-8")
            return sid, y, m, "ok", len(t)
        except Exception as e:                      # transient: back off and retry
            err = e
            time.sleep(2 * (attempt + 1))
    return sid, y, m, f"FAILED {err}", None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=2005)
    ap.add_argument("--end", type=int, default=dt.date.today().year)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    today = dt.date.today()
    jobs = []
    for y in range(a.start, a.end + 1):
        for m in range(1, 13):
            if (y, m) > (today.year, today.month):
                continue
            for sid in SPORTS:
                jobs.append((sid, y, m, (y, m) == (today.year, today.month)))
    t0 = time.time()
    fails = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for i, (sid, y, m, st, n) in enumerate(ex.map(lambda j: fetch(*j), jobs)):
            if st.startswith("FAILED"):
                fails += 1
                print(f"  {SPORTS[sid]} {y}-{m:02d} {st}", flush=True)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(jobs)} [{(time.time() - t0) / 60:.1f}m]", flush=True)
    print(f"done: {len(jobs)} month-sport files, {fails} failed [{(time.time() - t0) / 60:.1f}m]")


if __name__ == "__main__":
    main()
