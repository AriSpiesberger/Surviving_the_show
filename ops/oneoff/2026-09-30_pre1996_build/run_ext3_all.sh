#!/usr/bin/env bash
# 1988-2004 draft classes (2026-09-30): MLB API 1988-1995 lines + StatsCrew for the API's 1989 / 1992
# holes (tools/fetch_statscrew_gaps.py). Builds prospects_ext3.db, runs the v2.4 pipeline under
# RUN_TAG=ext3 (production's val list verbatim), then the held-out v3 with the production spec,
# recalibration, the evaluation packet and the leak audit. Production is never touched.
set -u
cd "$(dirname "$0")/../../.."
E=prospects_ext3.db
export PYTHONIOENCODING=utf-8 PROSPECT_DB="$PWD/$E" PROSPECT_MODEL_DB="$PWD/$E"
export RUN_TAG=ext3 PROSPECT_MIN_LANDMARK=1990 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
D=$(date +%Y%m%d)
LOG=logs/ext3_data_$D.log; : > $LOG
run(){ echo "===== $1 $(date '+%F %T')" >> $LOG; shift; "$@" >> $LOG 2>&1 || { echo "FAILED rc=$?" >> $LOG; exit 1; }; }
run build python tools/build_pre2005_ext.py --start 1988 --db $E
run mlb_seasons python -m prospects.data.backfills.mlb_seasons_stitch --db $E --start 1988 --end 2026
run rank_integrate python -m prospects.data.sources.baseballcube_integrate --db $E --out reference/pre2005/rankings_history_ext3.csv
run rank_load python -m prospects.data.sources.rankings --db $E --csv reference/pre2005/rankings_history_ext3.csv
run outcomes python -c "from prospects.data.sources.outcomes import pull_outcomes; from prospects.core.storage import ProspectDB; print(pull_outcomes(ProspectDB('$E')))"
run birthdates python -m prospects.data.backfills.birthdate_backfill_people --db $E --apply
run agefill python - <<'PY'
import os, sqlite3
from prospects.data.backfills.birthdate_backfill import _baseball_age
c = sqlite3.connect(os.environ["PROSPECT_DB"])
rows = c.execute("SELECT s.rowid, s.season_year, p.birth_date FROM season_stats s JOIN prospects p USING(player_id) "
                 "WHERE s.age_during_season IS NULL AND p.birth_date IS NOT NULL").fetchall()
upd = [(a, rid) for rid, y, b in rows if (a := _baseball_age(b, y)) is not None]
with c:
    c.executemany("UPDATE season_stats SET age_during_season=? WHERE rowid=?", upd)
print("ages filled:", len(upd), "| pre-2005 rows (n, with age):",
      c.execute("select count(*), sum(age_during_season is not null) from season_stats where season_year<2005 and level!='MLB'").fetchone())
PY
run ages python -m prospects.data.backfills.repair_impossible_ages --db $E
run rankings_ages python -m prospects.data.backfills.repair_impossible_rankings --db $E
run woba python -m prospects.data.backfills.woba_backfill --db $E
run percentiles python -m prospects.data.backfills.percentile_backfill --db $E
run context python tools/build_pre2005_context.py --start 1988 --db $E
echo "===== data done $(date '+%F %T')" >> $LOG

LOG=logs/ext3_pipeline_$D.log; : > $LOG
step(){ echo "===== $1 $(date '+%F %T')" >> "$LOG"; shift; "$@" >> "$LOG" 2>&1; rc=$?; echo "[$rc]" >> "$LOG"; return $rc; }
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
print(f"[ext3 split] val {len(val):,} (production list), fit {len(fit):,} of {len(pids):,} eligible")
EOF
OMP_NUM_THREADS=1 PROSPECT_NUM_THREADS=1 step stage_a python -m prospects.model.pipelines.stage_a --db $E --min-landmark-year 1990 || exit 1
export OMP_NUM_THREADS=16 PROSPECT_NUM_THREADS=16
step oof python -m prospects.model.pipelines.oof --db $E || exit 1
step hazards python -m prospects.model.train.hazards --force || exit 1
step v2.0b python -m prospects.model.pipelines.prod --db $E --skip-buylist || exit 1
step recent python -m prospects.model.train.score_recent_cohorts --hazards runs/ext3/scratch/oof/fold0_hazards.pkl || exit 1
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
