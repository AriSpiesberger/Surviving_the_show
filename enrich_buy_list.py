"""Join buy_list against prospects.db to add draft_year/is_international/start_year."""
import csv
import sqlite3
import sys

src, db_path, dst = sys.argv[1], sys.argv[2], sys.argv[3]

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row

with open(src, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

extra_cols = ["draft_year", "is_international", "start_year"]
in_fields = list(rows[0].keys())
out_fields = in_fields + [c for c in extra_cols if c not in in_fields]

n_matched = 0
n_missing = 0
for r in rows:
    pid = r["player_id"]
    p = conn.execute(
        "SELECT draft_year, is_international FROM prospects WHERE player_id = ?",
        (pid,),
    ).fetchone()
    if p is None:
        n_missing += 1
        for c in extra_cols:
            r.setdefault(c, "")
        continue
    n_matched += 1
    # Only overwrite if missing in source
    if not r.get("draft_year"):
        r["draft_year"] = p["draft_year"] if p["draft_year"] is not None else ""
    if not r.get("is_international"):
        r["is_international"] = p["is_international"] or 0
    if not r.get("start_year"):
        is_intl = str(r.get("is_international") or "0") in ("1", "True", "true")
        if not r.get("draft_year") and is_intl:
            sy = conn.execute(
                "SELECT MIN(season_year) AS sy FROM season_stats WHERE player_id = ?",
                (pid,),
            ).fetchone()
            r["start_year"] = sy["sy"] if sy and sy["sy"] is not None else ""
        else:
            r["start_year"] = ""

with open(dst, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=out_fields)
    w.writeheader()
    w.writerows(rows)

print(f"matched={n_matched} missing={n_missing} -> {dst}")
