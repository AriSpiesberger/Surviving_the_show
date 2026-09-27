#!/usr/bin/env bash
# Promote the 1996-2004 build (runs/ext + prospects_ext.db) to production (2026-09-25, user-approved).
# Backs up both DBs and runs/current first; overlays runs/ext onto runs/current; rewrites the
# v3 bundles' base-bundle paths; keeps the previous sheet next to the new one.
set -eu
cd "$(dirname "$0")/.."
D=20260925
python -c "open('runs/current/buy_lists/final.csv','a').close()" || { echo "[swap] final.csv is locked (open in Excel?); nothing changed"; exit 1; }
[ -f prospects.db.bak_pre_ext_$D ] || cp -p prospects.db prospects.db.bak_pre_ext_$D
[ -f prospects_snapshot.db.bak_pre_ext_$D ] || cp -p prospects_snapshot.db prospects_snapshot.db.bak_pre_ext_$D
[ -d runs/current_pre_ext_$D ] || cp -rp runs/current runs/current_pre_ext_$D
cp -p runs/current/buy_lists/final.csv runs/current/buy_lists/final_v3_pre_ext_2026-09-25.csv
cp -p prospects_ext.db prospects.db
cp -p prospects_ext.db prospects_snapshot.db
cp -rp runs/ext/. runs/current/
python - <<'PY'
import pickle
for f in ("runs/current/models/v3.pkl", "runs/current/models/v3.5.pkl"):
    b = pickle.load(open(f, "rb"))
    for k in ("base_xgb", "base_cal"):
        b[k] = b[k].replace("runs/ext/", "runs/current/")
    pickle.dump(b, open(f, "wb"))
    print(f, b["base_xgb"], b["base_cal"])
PY
echo "[swap] done"
