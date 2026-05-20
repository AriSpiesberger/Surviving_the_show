"""Refresh per-outcome pop magnitudes from current eBay base 1st auto prices.

Procedure:
  1. Load prices_bowman_chrome_auto_v13.csv, filter to denominator=0 (base).
  2. For each historical debutee 2020-2024, compute outcome label
     via Model B logic from MLB stats.
  3. Group by outcome × position; report n, median, P25, P75 of current price.
  4. Save lookup table for downstream EV.
"""
from __future__ import annotations

import sqlite3
from collections import Counter

import numpy as np
import pandas as pd

DB = "prospects_snapshot.db"
PRICES = "prices_bowman_chrome_auto_v13.csv"
DEBUT_LO, DEBUT_HI = 2020, 2024
OUT = "pop_magnitudes_v1.csv"


def label_player(rows: pd.DataFrame, debut_year: int) -> str:
    post = rows[rows.season_year >= debut_year].copy()
    if post.empty:
        return "cup"
    has_breakout = (
        ((post.woba >= 0.350) & (post.pa >= 300)).any()
        or ((post.era <= 3.50) & (post.ip >= 100)).any()
    )
    if has_breakout:
        return "breakout"
    has_regular = ((post.pa >= 350) | (post.ip >= 80)).any()
    if has_regular:
        return "regular"
    n_seasons = post.season_year.nunique()
    if n_seasons >= 2:
        return "utility"
    career_pa = post.pa.sum()
    came_back = (post.season_year > debut_year + 1).any()
    if career_pa >= 100 or came_back:
        return "utility"
    return "cup"


def pos_group(p):
    p = str(p).upper()
    if p in ("C",): return "C"
    if p in ("2B", "SS"): return "MI"
    if p in ("1B", "3B", "LF", "RF", "OF", "CF"): return "CI_OF"
    if p == "SP": return "SP"
    if p == "RP": return "RP"
    return "OTH"


def main():
    print(f"Loading prices from {PRICES}")
    prices = pd.read_csv(PRICES)
    prices = prices[prices.denominator == 0].copy()
    prices = prices[prices.has_market == 1].copy()
    prices["price"] = prices["price_median"].fillna(prices["price_mean"])
    prices = prices[prices.price.notna()].copy()
    prices = prices.groupby("player_id", as_index=False).agg(
        price=("price", "median"),
        n_listings=("n_listings", "sum"),
    )
    print(f"  {len(prices):,} unique players with market prices")

    print(f"\nLoading career_outcomes for {DEBUT_LO}-{DEBUT_HI} debutees")
    c = sqlite3.connect(DB)
    debs = pd.read_sql(
        f"SELECT player_id, mlb_debut_year FROM career_outcomes "
        f"WHERE mlb_debut_year BETWEEN {DEBUT_LO} AND {DEBUT_HI}", c)
    print(f"  {len(debs):,} debutees")

    print("\nLoading MLB season_stats")
    mlb = pd.read_sql(
        "SELECT player_id, season_year, pa, ip, woba, era, primary_position "
        "FROM season_stats WHERE UPPER(level)='MLB'", c)
    mlb["pa"] = mlb["pa"].fillna(0)
    mlb["ip"] = mlb["ip"].fillna(0)
    c.close()

    print("\nLabeling outcomes...")
    rows = []
    for pid, dy in zip(debs.player_id, debs.mlb_debut_year):
        sub = mlb[mlb.player_id == pid]
        if sub.empty:
            continue
        outcome = label_player(sub, int(dy))
        pos = sub.primary_position.dropna().mode()
        pos = pos.iloc[0] if len(pos) else "UNK"
        rows.append({"player_id": pid, "outcome": outcome, "position": pos,
                     "pos_grp": pos_group(pos), "debut_year": int(dy)})
    lbl = pd.DataFrame(rows)
    print(f"  {len(lbl):,} labeled debutees")
    print("  outcome distribution:")
    for o, n in Counter(lbl.outcome).most_common():
        print(f"    {o:<10s} {n:>4d}")

    merged = lbl.merge(prices, on="player_id", how="inner")
    print(f"\n  with eBay price: {len(merged):,} ({len(merged)/len(lbl):.0%})")
    print("  outcome distribution (priced subset):")
    for o, n in Counter(merged.outcome).most_common():
        print(f"    {o:<10s} {n:>4d}")

    print("\n" + "=" * 78)
    print("POP MAGNITUDES by outcome (all positions)")
    print("=" * 78)
    print(f"{'outcome':<10s} {'n':>4s} {'median':>8s} {'P25':>8s} {'P75':>8s} {'mean':>8s}")
    out_rows = []
    for outcome in ["cup", "utility", "regular", "breakout"]:
        sub = merged[merged.outcome == outcome]
        if sub.empty:
            continue
        med = sub.price.median()
        p25 = sub.price.quantile(0.25)
        p75 = sub.price.quantile(0.75)
        mean = sub.price.mean()
        print(f"{outcome:<10s} {len(sub):>4d} ${med:>7.0f} ${p25:>7.0f} ${p75:>7.0f} ${mean:>7.0f}")
        out_rows.append({"outcome": outcome, "pos_grp": "ALL", "n": len(sub),
                         "median": med, "p25": p25, "p75": p75, "mean": mean})

    print("\n" + "=" * 78)
    print("POP MAGNITUDES by outcome × position")
    print("=" * 78)
    print(f"{'outcome':<10s} {'pos':<6s} {'n':>4s} {'median':>8s} {'P25':>8s} {'P75':>8s}")
    for outcome in ["cup", "utility", "regular", "breakout"]:
        for pg in ["C", "MI", "CI_OF", "SP", "RP", "OTH"]:
            sub = merged[(merged.outcome == outcome) & (merged.pos_grp == pg)]
            if len(sub) < 3:
                continue
            med = sub.price.median()
            p25 = sub.price.quantile(0.25)
            p75 = sub.price.quantile(0.75)
            print(f"{outcome:<10s} {pg:<6s} {len(sub):>4d} ${med:>7.0f} ${p25:>7.0f} ${p75:>7.0f}")
            out_rows.append({"outcome": outcome, "pos_grp": pg, "n": len(sub),
                             "median": med, "p25": p25, "p75": p75,
                             "mean": sub.price.mean()})

    print("\n" + "=" * 78)
    print("TOP PRICED PLAYERS in BREAKOUT class (sanity check)")
    print("=" * 78)
    brk = merged[merged.outcome == "breakout"].sort_values("price", ascending=False)
    brk_names = pd.read_sql_query(
        f"SELECT player_id, name FROM prospects WHERE player_id IN "
        f"({','.join(repr(p) for p in brk.player_id.head(15))})",
        sqlite3.connect(DB))
    brk = brk.merge(brk_names, on="player_id", how="left")
    print(brk[["name", "position", "debut_year", "price"]].head(15).to_string(index=False))

    pd.DataFrame(out_rows).to_csv(OUT, index=False)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
