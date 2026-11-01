"""All-Star rosters 2015-2026 from the MLB Stats API (awards ALAS / NLAS) -> raw/allstars.csv.
career_outcomes.year_all_star_once stops at 2021 (training-label cutoff), so the card studies need
the real recent selections."""
import json
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

rows = []
for y in range(2015, 2027):
    for aw in ("ALAS", "NLAS"):
        u = f"https://statsapi.mlb.com/api/v1/awards/{aw}/recipients?season={y}"
        try:
            awards = json.load(urllib.request.urlopen(u, timeout=60)).get("awards", [])
        except urllib.error.HTTPError:  # 2020: no All-Star Game
            awards = []
        for a in awards:
            p = a.get("player") or {}
            rows.append((y, aw, p.get("id"), p.get("nameFirstLast")))
d = pd.DataFrame(rows, columns=["season", "award", "mlbam", "name"]).drop_duplicates(["season", "mlbam"])
d.to_csv(Path(__file__).parents[1] / "raw" / "allstars.csv", index=False)
print(d.groupby("season").size().to_dict())
