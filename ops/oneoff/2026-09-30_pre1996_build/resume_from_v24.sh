#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/../../.."
E=prospects_ext3.db
export PYTHONIOENCODING=utf-8 PROSPECT_DB="$PWD/$E" PROSPECT_MODEL_DB="$PWD/$E"
export RUN_TAG=ext3 PROSPECT_MIN_LANDMARK=1990 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
D=$(date +%Y%m%d)
LOG=$(ls -t logs/ext3_pipeline_*.log | head -1)
step(){ echo "===== $1 $(date '+%F %T')" >> "$LOG"; shift; "$@" >> "$LOG" 2>&1; rc=$?; echo "[$rc]" >> "$LOG"; return $rc; }
export OMP_NUM_THREADS=16 PROSPECT_NUM_THREADS=16
step v2.4-joint python -m prospects.model.train.exp_cdf_timing5 --aug-long runs/ext3/training/recent_long.csv \
    --cal-min-snap-year 2008 --db $E --out-dir runs/ext3/scratch/v24_build || exit 1
M=runs/ext3/models
step promote python -m prospects.model.train.promote_v22 --source runs/ext3/scratch/v24_build --bag-name joint_xgb_exp5_bag.pkl --version v2.4 || exit 1
step v3-heldout python -m prospects.model.v3 build --fit runs/ext3/training/oof_stacked_long.csv --aug-long runs/ext3/training/recent_long.csv \
    --base-xgb $M/joint_xgb_v2.4.pkl --base-cal $M/calibrators_v2.4.pkl --out $M/v3.pkl --out-cal $M/calibrators_v3.pkl \
    --seeds 5 --threads 16 --encoder transformer_rank || exit 1
step recalibrate python -m prospects.model.v3 recalibrate --heldout-xgb $M/v3.pkl --heldout-cal $M/calibrators_v3.pkl --write $M/calibrators_v3.pkl || exit 1
step eval python -m prospects.evaluation.run --xgb $M/v3.pkl --calibrators $M/calibrators_v3.pkl --threshold 0.6 --out-dir runs/ext3/evaluation_v3 || exit 1
step leak-audit python tools/leak_audit.py
echo "===== ext3 all done $(date '+%F %T')" >> "$LOG"
