#!/usr/bin/env bash
# Train the v2.4 joint recipe on prospects_ext2.db (the live DB + 1993-2004 draft classes) under
# RUN_TAG=ext2, with the SAME held-out val players as production, so its AP compares directly
# with runs/current (2026-09-24). Production files are never touched: every DB read goes to
# prospects_ext2.db via PROSPECT_DB / PROSPECT_MODEL_DB and every artifact lands in runs/ext2/.
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
export PROSPECT_DB="$PWD/prospects_ext2.db" PROSPECT_MODEL_DB="$PWD/prospects_ext2.db"
export RUN_TAG=ext2 PROSPECT_MIN_LANDMARK=1995
export OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
LOG=logs/ext2_pipeline_$(date +%Y%m%d).log
step() { echo "===== $1 $(date '+%F %T')" >> "$LOG"; shift; "$@" >> "$LOG" 2>&1; rc=$?; echo "[$rc]" >> "$LOG"; return $rc; }

# 1. split (stage A's scoring step reads fit_pids.txt, so it must exist first): production's val list verbatim; every other eligible player (incl. the new cohort) is fit
step split python - <<'EOF' || exit 1
from prospects import config
from prospects.model.train.make_split import eligible_pids
cur, ext = config.run("current"), config.run()
val = [p for p in cur.val_pids.read_text().split() if p]
vs = set(val)
pids = eligible_pids(str(config.model_db()), 2020)
fit = [p for p in pids if p not in vs]
ext.val_pids.parent.mkdir(parents=True, exist_ok=True)
ext.val_pids.write_text("\n".join(val) + "\n")
ext.fit_pids.write_text("\n".join(fit) + "\n")
print(f"[ext split] val {len(val):,} (production list), fit {len(fit):,} of {len(pids):,} eligible")
EOF
# 2. landmark panel + hazards (stage A) from 1998; single-threaded is fastest here
OMP_NUM_THREADS=1 PROSPECT_NUM_THREADS=1 step stage_a python -m prospects.model.pipelines.stage_a --db prospects_ext2.db --min-landmark-year 1995 || exit 1
export OMP_NUM_THREADS=16 PROSPECT_NUM_THREADS=16
step oof python -m prospects.model.pipelines.oof --db prospects_ext2.db || exit 1
step hazards python -m prospects.model.train.hazards --force || exit 1
step v2.0b python -m prospects.model.pipelines.prod --db prospects_ext2.db --skip-buylist || exit 1
step recent python -m prospects.model.train.score_recent_cohorts --hazards runs/ext2/scratch/oof/fold0_hazards.pkl || exit 1   # production default hz0_fvfix2 is the old 329-feature contract
step v2.4-joint python -m prospects.model.train.exp_cdf_timing5 --aug-long runs/ext2/training/recent_long.csv \
    --cal-min-snap-year 2008 --db prospects_ext2.db --out-dir runs/ext2/scratch/v24_build || exit 1
echo "===== ext pipeline done $(date '+%F %T')" >> "$LOG"
