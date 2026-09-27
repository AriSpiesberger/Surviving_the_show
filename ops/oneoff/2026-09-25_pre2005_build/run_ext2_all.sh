#!/usr/bin/env bash
# 1993-2004 draft classes (2026-09-25): build prospects_ext2.db, run the v2.4 pipeline, then the
# transformer encoder and the v3 components, all under RUN_TAG=ext2. 1992 is excluded: the MLB API
# lacks that year's FSL and Carolina League, so careers starting before 1993 are incomplete.
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PROSPECT_DB="$PWD/prospects_ext2.db" PROSPECT_MODEL_DB="$PWD/prospects_ext2.db"
E=prospects_ext2.db
LOG=logs/ext2_data_$(date +%Y%m%d).log; : > $LOG
run(){ echo "===== $1 $(date '+%T')" >> $LOG; shift; "$@" >> $LOG 2>&1 || { echo "FAILED rc=$?" >> $LOG; exit 1; }; }
run build python tools/build_pre2005_ext.py --start 1993 --db $E
run mlb_seasons python -m prospects.data.backfills.mlb_seasons_stitch --db $E --start 1993 --end 2026
run rank_integrate python -m prospects.data.sources.baseballcube_integrate --db $E --out reference/pre2005/rankings_history_ext2.csv
run rank_load python -m prospects.data.sources.rankings --db $E --csv reference/pre2005/rankings_history_ext2.csv
run outcomes python -c "from prospects.data.sources.outcomes import pull_outcomes; from prospects.core.storage import ProspectDB; print(pull_outcomes(ProspectDB('$E')))"
run birthdates python -m prospects.data.backfills.birthdate_backfill_people --db $E --apply
run agefill python - <<'PY'
import sqlite3
from prospects.data.backfills.birthdate_backfill import _baseball_age
c = sqlite3.connect("prospects_ext2.db")
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
run context python tools/build_pre2005_context.py --start 1993 --db $E
echo "===== data done $(date '+%T')" >> $LOG

bash ops/run_ext2_pipeline.sh || exit 1
grep -q "ext pipeline done" logs/ext2_pipeline_$(date +%Y%m%d).log logs/ext2_pipeline_*.log 2>/dev/null || exit 1
export RUN_TAG=ext2 OMP_NUM_THREADS=24 PROSPECT_NUM_THREADS=24 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
LOG=logs/ext2_v3_$(date +%Y%m%d).log; : > $LOG
echo "===== encoder $(date '+%F %T')" >> $LOG
python -u -m prospects.model.train.exp_seq_c --arch transformer --rank-tokens --threads 24 --out runs/ext2/scratch/emb_transformer_rank.npz >> $LOG 2>&1 || { echo FAILED >> $LOG; exit 1; }
echo "===== v3-all ext2 $(date '+%F %T')" >> $LOG
python -u -m prospects.model.train.exp_v3_all --haz --n-boot 300 --emb-cache runs/ext2/scratch/emb_transformer_rank.npz --out-dir runs/exp_v3_all_ext2 >> $LOG 2>&1
echo "===== done $(date '+%F %T')" >> $LOG
