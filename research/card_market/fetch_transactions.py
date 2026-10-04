"""MLB transactions (Stats API) for the Rule 5 protection window each November,
when clubs add not-yet-debuted prospects to the 40-man roster."""
import json, urllib.request
from pathlib import Path

OUT = Path(__file__).parent / "raw" / "transactions_nov.jsonl"
with OUT.open("w") as f:
    for y in (2021, 2022, 2023, 2024, 2025):
        url = f"https://statsapi.mlb.com/api/v1/transactions?sportId=1&startDate={y}-11-01&endDate={y}-11-30"
        tx = json.load(urllib.request.urlopen(url, timeout=120))["transactions"]
        for t in tx:
            f.write(json.dumps({"year": y, "date": t.get("date"), "type": t.get("typeCode"),
                                "desc": t.get("description"), "pid": (t.get("person") or {}).get("id"),
                                "team": (t.get("toTeam") or {}).get("name")}) + "\n")
        print(y, len(tx))
