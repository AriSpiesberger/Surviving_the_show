"""MLB debut dates (and current team/position) for every prospect with an mlbam_id."""
import json, sqlite3, urllib.request
from pathlib import Path

OUT = Path(__file__).parent / "raw" / "mlb_people.jsonl"
con = sqlite3.connect(Path(__file__).parents[2] / "prospects.db")
ids = [r[0] for r in con.execute(
    "select distinct mlbam_id from prospects where mlbam_id is not null and mlbam_id != ''")]
with OUT.open("w") as f:
    for i in range(0, len(ids), 150):
        chunk = ",".join(ids[i:i + 150])
        url = f"https://statsapi.mlb.com/api/v1/people?personIds={chunk}&hydrate=currentTeam"
        people = json.load(urllib.request.urlopen(url, timeout=60))["people"]
        for p in people:
            f.write(json.dumps({k: p.get(k) for k in (
                "id", "fullName", "mlbDebutDate", "lastPlayedDate", "active",
                "birthDate", "primaryPosition")}) + "\n")
print(len(ids), "ids;", sum(1 for _ in OUT.open()), "people written")
