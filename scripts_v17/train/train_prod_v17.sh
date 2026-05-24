#!/bin/bash
# v1.17 PRODUCTION training pipeline — 100% of the panel on every stage.
#
# Outputs the artifacts the buy sheet consumes:
#   models/event_classifiers_v1.17_prod.pkl
#   models/event_classifiers_v1.17_prod_calibrated.pkl
#   v1.17_prod_fit_long.csv   (seed=42 lasso-fit pids re-scored with prod hazards)
#   v1.17_prod_val_long.csv   (seed=42 lasso-val pids re-scored with prod hazards)
#   models/debut_lasso_universe_v1.17_prod.pkl
#   models/top100_lasso_v1.17_prod.pkl
#   models/model_b_outcomes_v1.17_prod.pkl
#
# Notes vs the TEST script:
#   - Hazards see ALL players. No held-out slice — DO NOT use this model for
#     validation, only for scoring the production buy sheet.
#   - Calibrators are fit on the SAME seed=42 10% slice the test pipeline uses
#     (consistent presentation layer; the slice is just a calibration set here,
#     not held out from anything).
#   - lasso/model_b refit on prod-hazard outputs. Hazards saw those players in
#     training, so lasso inputs are mildly optimistic on the fit slice. The
#     honest comparison numbers live in the TEST pipeline's validation packet.
set -e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

PANEL=${PANEL:-panel_v1.17.npz}
HAZ=models/event_classifiers_v1.17_prod.pkl
CAL=models/event_classifiers_v1.17_prod_calibrated.pkl

if [ ! -f "$PANEL" ]; then
  echo "Panel $PANEL not found. Run scripts_v17/train/build_panel_v17.sh first."
  exit 1
fi

# 1. Hazards trained on 100% of the panel.
echo "=== [1/6] Training v1.17 PROD hazards on 100% panel ==="
python -m prospects.classifier.train_full_v14d \
  --panel "$PANEL" \
  --lasso-fit-frac 0.0 --lasso-val-frac 0.0 \
  --seed 42 \
  --out "$HAZ"

# 2. Recompute the seed=42 fit/val pids (same as test pipeline) — used as a
#    calibration set here, and as the fit slice for prod lasso/model_b refits.
echo "=== [2/6] Reproducing seed=42 fit/val pids ==="
python <<'EOF'
import numpy as np
with np.load("panel_v1.17.npz", allow_pickle=True) as d:
    pids = sorted(set(d["pids"].tolist()))
rng = np.random.default_rng(42)
perm = rng.permutation(len(pids))
n_fit = int(round(0.10 * len(pids)))
n_val = int(round(0.10 * len(pids)))
with open("v17_prod_fit_pids.txt","w") as f:
    for i in perm[:n_fit]: f.write(pids[i]+"\n")
with open("v17_prod_val_pids.txt","w") as f:
    for i in perm[n_fit:n_fit+n_val]: f.write(pids[i]+"\n")
print(f"fit:{n_fit}  val:{n_val}")
EOF

# 3. Beta calibrators on the seed=42 fit pids (consistent with test pipeline).
echo "=== [3/6] Fitting prod Beta calibrators on 10% slice ==="
python -m prospects.classifier.fit_hazard_calibrators \
  --model "$HAZ" \
  --panel "$PANEL" \
  --players-file v17_prod_fit_pids.txt \
  --out "$CAL"

# 4. Score fit + val slices with prod hazards (chunked + retry).
for slice in fit val; do
  echo "=== [4/6] Scoring prod ${slice} slice ==="
  PLIST="v17_prod_${slice}_pids.txt"
  OUT="v1.17_prod_${slice}_long.csv"
  if [ -f "$OUT" ]; then
    echo "  $OUT already exists, skipping"
    continue
  fi
  split -d -l 100 -a 3 "$PLIST" "v17_prod_${slice}_chunk_"
  mkdir -p "v17_prod_${slice}_out"
  for f in v17_prod_${slice}_chunk_*; do
    out="v17_prod_${slice}_out/${f}.csv"
    [ -f "$out" ] && continue
    for try in 1 2 3; do
      python -m prospects.classifier.score_v14c_cal_slice_raw \
        --model "$CAL" --panel "$PANEL" \
        --players-file "$f" --max-entry-year 2020 \
        --observe-through 2026 --max-offset 10 \
        --out "$out" > /tmp/prod_score_${slice}.log 2>&1 || true
      [ -f "$out" ] && break
      sleep 2
    done
    [ -f "$out" ] || { echo "  FAIL on $f"; exit 1; }
  done
  python -c "
import pandas as pd, glob
files = sorted(glob.glob('v17_prod_${slice}_out/*.csv'))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df.to_csv('$OUT', index=False)
print(f'wrote $OUT: {len(df):,} rows')
"
  rm -f v17_prod_${slice}_chunk_*
done

# 5. Refit debut_lasso + top100_lasso + model_b on prod long files.
#    refit_models_honest.py is parametrized by the literal filenames it reads
#    (v1.17h_fit_long.csv / v1.17h_val_long.csv) — swap aliases.
echo "=== [5/6] Refitting prod lasso + model_b ==="
cp -f v1.17_prod_fit_long.csv v1.17h_fit_long.csv
cp -f v1.17_prod_val_long.csv v1.17h_val_long.csv
python scripts_v17/train/refit_models_honest.py
# Rename outputs to _prod suffix so they sit alongside the test artifacts.
mv -f models/debut_lasso_universe_v1.17h.pkl  models/debut_lasso_universe_v1.17_prod.pkl
mv -f models/top100_lasso_v1.17h.pkl          models/top100_lasso_v1.17_prod.pkl
mv -f models/model_b_outcomes_v1.17h.pkl      models/model_b_outcomes_v1.17_prod.pkl

echo "=== [6/6] v1.17 PROD pipeline done ==="
ls -la models/event_classifiers_v1.17_prod*.pkl \
       models/debut_lasso_universe_v1.17_prod.pkl \
       models/top100_lasso_v1.17_prod.pkl \
       models/model_b_outcomes_v1.17_prod.pkl \
       v1.17_prod_fit_long.csv v1.17_prod_val_long.csv
