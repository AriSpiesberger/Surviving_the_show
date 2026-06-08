"""
Resolve each player in card_holdings.csv to their CURRENT level.

Primary source : results/buy_lists/buy_list_v2.0b_FINAL.csv  (repo ground truth:
                 name -> cur_level_2026, current_org, primary_position)
Fallback       : live MLB Stats API (statsapi.mlb.com) for players not on the
                 buy list -- name index from /sports/{id}/players + currentTeam
                 -> /teams/{id}.sport.

Outputs:
  - card_levels.csv               (per unique player: level, org, source ...)
  - card_holdings_by_category.csv (full holdings, Category = current level,
                                    grouped by level, exact 7-col format)
"""
from __future__ import annotations

import csv
import time

import requests

BASE = "https://statsapi.mlb.com/api/v1"
HOLDINGS = "card_holdings.csv"
BUYLIST = "results/buy_lists/buy_list_v2.0b_FINAL.csv"

LEVEL_ORDER = {"MLB": 0, "AAA": 1, "AA": 2, "A+": 3, "A": 4, "A-": 5,
               "RK": 6, "?": 9}
SPORT_TO_TAG = {
    "Major League Baseball": "MLB", "Triple-A": "AAA", "Double-A": "AA",
    "High-A": "A+", "Single-A": "A", "Class A Short Season": "A-",
    "Rookie": "RK", "Rookie Advanced": "RK",
}


def load_buylist() -> dict:
    bl = {}
    with open(BUYLIST, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            bl[r["name"].strip().lower()] = (
                r.get("cur_level_2026", "").strip() or "?",
                r.get("current_org", "").strip(),
            )
    return bl


def build_api_index() -> dict:
    idx = {}
    for sid in (1, 11, 12, 13, 14, 16):
        js = requests.get(f"{BASE}/sports/{sid}/players",
                          params={"season": 2026}, timeout=60).json()
        for p in js.get("people", []):
            idx.setdefault(p["fullName"].lower(), p["id"])
    return idx


_team_cache: dict[int, tuple[str, str]] = {}


def team_level_org(team_id: int) -> tuple[str, str]:
    if team_id not in _team_cache:
        t = requests.get(f"{BASE}/teams/{team_id}", timeout=20).json()["teams"][0]
        tag = SPORT_TO_TAG.get((t.get("sport") or {}).get("name", ""), "?")
        _team_cache[team_id] = (tag, t.get("parentOrgName", ""))
    return _team_cache[team_id]


def api_resolve(name: str, idx: dict) -> tuple[str, str, str]:
    pid = idx.get(name.lower())
    if pid is None:
        r = requests.get(f"{BASE}/people/search", params={"names": name},
                         timeout=20).json().get("people", [])
        if not r:
            return "?", "", "not found"
        pid = r[0]["id"]
    js = requests.get(f"{BASE}/people/{pid}", params={"hydrate": "currentTeam"},
                      timeout=20).json().get("people", [])
    ct = js[0].get("currentTeam") if js else None
    if not ct:
        return "?", "", "no current team"
    lvl, org = team_level_org(ct["id"])
    return lvl, org, f"API: {ct.get('name','')}"


def main() -> None:
    buylist = load_buylist()
    rows = list(csv.DictReader(open(HOLDINGS, newline="", encoding="utf-8")))

    # resolve each unique player once
    resolved: dict[str, tuple[str, str, str]] = {}
    api_idx = None
    for r in rows:
        p = r["Player"].strip()
        if p in resolved:
            continue
        hit = buylist.get(p.lower())
        if hit:
            resolved[p] = (hit[0], hit[1], "buy_list")
        else:
            if api_idx is None:
                print("(building live-API index for off-buy-list players...)")
                api_idx = build_api_index()
            resolved[p] = api_resolve(p, api_idx)
            time.sleep(0.1)

    # detail sheet
    with open("card_levels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Player", "Level", "Org", "Source"])
        for p, (lvl, org, src) in sorted(
                resolved.items(), key=lambda kv: (LEVEL_ORDER.get(kv[1][0], 9), kv[0])):
            w.writerow([p, lvl, org, src])
            print(f"  {lvl:>4} | {p:<22} | {org or '-':<22} | {src}")

    # holdings grouped by level (Category = level), exact 7-col format
    out = sorted(rows, key=lambda r: (
        LEVEL_ORDER.get(resolved[r["Player"].strip()][0], 9),
        r["Player"].strip()))
    cols = ["Date Purchased", "Player", "Year", "Set", "Category",
            "Condition", "Investment"]
    with open("card_holdings_by_category.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in out:
            r = dict(r)
            r["Category"] = resolved[r["Player"].strip()][0]  # level
            w.writerow({c: r[c] for c in cols})

    print("\nWrote card_levels.csv and card_holdings_by_category.csv")


if __name__ == "__main__":
    main()
