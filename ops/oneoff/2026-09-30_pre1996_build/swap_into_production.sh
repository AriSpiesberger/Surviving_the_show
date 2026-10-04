#!/usr/bin/env bash
# Promote the 1988-2004 build (runs/ext3 + prospects_ext3.db) to production (2026-10-01, user-approved).
# Backs up both DBs and runs/current first; overlays runs/ext3 onto runs/current; repoints every v3
# bundle's base-bundle path at runs/current (relative or absolute); keeps the previous sheet.
set -eu
cd "$(dirname "$0")/../../.."
D=20261001
python -c "open('runs/current/buy_lists/final.csv','a').close()" || { echo "[swap] final.csv is locked (open in Excel?); nothing changed"; exit 1; }
mkdir -p archive/db_backups runs/backups
[ -f archive/db_backups/prospects.db.bak_pre_ext3_$D ] || cp -p prospects.db archive/db_backups/prospects.db.bak_pre_ext3_$D
[ -f archive/db_backups/prospects_snapshot.db.bak_pre_ext3_$D ] || cp -p prospects_snapshot.db archive/db_backups/prospects_snapshot.db.bak_pre_ext3_$D
[ -d runs/backups/current_pre_ext3_$D ] || cp -rp runs/current runs/backups/current_pre_ext3_$D
cp -p runs/current/buy_lists/final.csv runs/current/buy_lists/history/final_v3_pre_ext3_2026-10-01.csv
cp -p prospects_ext3.db prospects.db
cp -p prospects_ext3.db prospects_snapshot.db
cp -rp runs/ext3/. runs/current/
python - <<'PY'
import pickle
from pathlib import Path
for f in ("runs/current/models/v3.pkl", "runs/current/models/v3.5.pkl"):
    b = pickle.load(open(f, "rb"))
    for k in ("base_xgb", "base_cal"):
        b[k] = (Path("runs") / "current" / "models" / Path(b[k].replace(chr(92), "/")).name).as_posix()
        assert Path(b[k]).exists(), b[k]
    pickle.dump(b, open(f, "wb"))
    print(f, "->", b["base_xgb"], b["base_cal"])
PY
echo "[swap] done"
