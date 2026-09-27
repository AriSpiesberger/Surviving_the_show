#!/usr/bin/env bash
# Production-candidate v3 (with nhaz, DEFAULT_SPEC) on prospects_ext.db under RUN_TAG=ext (2026-09-25).
# Held-out build only (v2.4 base) + recalibration + the evaluation packet -> runs/ext/evaluation_v3,
# for a like-for-like comparison with runs/current/scratch/evaluation_v3. Touches nothing in runs/current.
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PROSPECT_DB="$PWD/prospects_ext.db" PROSPECT_MODEL_DB="$PWD/prospects_ext.db"
export RUN_TAG=ext PROSPECT_MIN_LANDMARK=1998 OMP_NUM_THREADS=16 PROSPECT_NUM_THREADS=16 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
M=runs/ext/models
LOG=logs/ext_v3cand_$(date +%Y%m%d).log; : > $LOG
step() { echo "===== $1 $(date '+%F %T')" >> "$LOG"; shift; "$@" >> "$LOG" 2>&1; rc=$?; echo "[$rc]" >> "$LOG"; return $rc; }
step promote python -m prospects.model.train.promote_v22 --source runs/ext/scratch/v24_build --bag-name joint_xgb_exp5_bag.pkl --version v2.4 || exit 1
step v3-heldout python -m prospects.model.v3 build --fit runs/ext/training/oof_stacked_long.csv --aug-long runs/ext/training/recent_long.csv \
    --base-xgb $M/joint_xgb_v2.4.pkl --base-cal $M/calibrators_v2.4.pkl --out $M/v3.pkl --out-cal $M/calibrators_v3.pkl \
    --seeds 5 --threads 16 --encoder transformer_rank || exit 1
step recalibrate python -m prospects.model.v3 recalibrate --heldout-xgb $M/v3.pkl --heldout-cal $M/calibrators_v3.pkl --write $M/calibrators_v3.pkl || exit 1
step eval python -m prospects.evaluation.run --xgb $M/v3.pkl --calibrators $M/calibrators_v3.pkl --threshold 0.6 --out-dir runs/ext/evaluation_v3 || exit 1
step leak-audit python tools/leak_audit.py
echo "===== done $(date '+%F %T')" >> $LOG
