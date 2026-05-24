#!/bin/bash
# v1.17 TEST training pipeline — 80% hazard / 10% cal+lasso-fit / 10% lasso-val.
#
# Outputs the artifacts used by the validation packet:
#   models/event_classifiers_v1.17.pkl
#   models/event_classifiers_v1.17_lasso_fit_players.txt
#   models/event_classifiers_v1.17_lasso_val_players.txt
#   models/event_classifiers_v1.17_calibrated.pkl
#   v1.17_fit_long.csv      (10% lasso-fit slice scored with 80%-trained hazards)
#   v1.17_val_long.csv      (10% held-out, untouched until validation)
#   models/debut_lasso_universe_v1.17h.pkl
#   models/top100_lasso_v1.17h.pkl
#   models/model_b_outcomes_v1.17h.pkl
#
# The 10% lasso-fit slice serves TWO independent purposes on the same players:
#   (1) Beta calibrators (presentation layer)
#   (2) Lasso fit on RAW hazards (ranking)
# No leakage because lasso never consumes calibrated outputs.
set -e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

PANEL=${PANEL:-panel_v1.17.npz}
HAZ=models/event_classifiers_v1.17.pkl
CAL=models/event_classifiers_v1.17_calibrated.pkl

if [ ! -f "$PANEL" ]; then
  echo "Panel $PANEL not found. Run scripts_v17/train/build_panel_v17.sh first."
  exit 1
fi

# 1. Hazards on 80% train slice (player-grouped, seed=42).
echo "=== [1/6] Training v1.17 TEST hazards (80% train / 10% fit / 10% val) ==="
python -m prospects.classifier.train_full_v14d \
  --panel "$PANEL" \
  --lasso-fit-frac 0.10 --lasso-val-frac 0.10 \
  --seed 42 \
  --out "$HAZ"

# 2. Beta calibrators on the 10% lasso-fit slice.
echo "=== [2/6] Fitting Beta calibrators on 10% lasso-fit slice ==="
python -m prospects.classifier.fit_hazard_calibrators \
  --model "$HAZ" \
  --panel "$PANEL" \
  --players-file "${HAZ%.pkl}_lasso_fit_players.txt" \
  --out "$CAL"

# 3. Score lasso-fit slice + held-out val slice using the calibrated model.
#    (CSV retains p_<event>_raw columns; lasso consumes those.)
for slice in fit val; do
  echo "=== [3/6] Scoring v1.17 ${slice} slice ==="
  PLIST="${HAZ%.pkl}_lasso_${slice}_players.txt"
  OUT="v1.17_${slice}_long.csv"
  if [ -f "$OUT" ]; then
    echo "  $OUT already exists, skipping"
    continue
  fi
  split -d -l 100 -a 3 "$PLIST" "v17_${slice}_chunk_"
  mkdir -p "v17_${slice}_out"
  for f in v17_${slice}_chunk_*; do
    out="v17_${slice}_out/${f}.csv"
    [ -f "$out" ] && continue
    for try in 1 2 3; do
      python -m prospects.classifier.score_v14c_cal_slice_raw \
        --model "$CAL" --panel "$PANEL" \
        --players-file "$f" --max-entry-year 2020 \
        --observe-through 2026 --max-offset 10 \
        --out "$out" > /tmp/score_${slice}.log 2>&1 || true
      [ -f "$out" ] && break
      sleep 2
    done
    [ -f "$out" ] || { echo "  FAIL on $f"; exit 1; }
  done
  python -c "
import pandas as pd, glob
files = sorted(glob.glob('v17_${slice}_out/*.csv'))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df.to_csv('$OUT', index=False)
print(f'wrote $OUT: {len(df):,} rows')
"
  rm -f v17_${slice}_chunk_*
done

# 4. Refit debut_lasso, top100_lasso, model_b on the honest fit/val long files.
#    (refit_models_honest.py reads v1.17h_fit_long.csv / v1.17h_val_long.csv —
#    symlink/copy to match its expected names.)
echo "=== [4/6] Aliasing long files for refit_models_honest.py ==="
cp -f v1.17_fit_long.csv v1.17h_fit_long.csv
cp -f v1.17_val_long.csv v1.17h_val_long.csv

echo "=== [5/6] Refitting debut_lasso + top100_lasso + model_b ==="
python scripts_v17/train/refit_models_honest.py

echo "=== [6/6] v1.17 TEST pipeline done ==="
ls -la models/event_classifiers_v1.17.pkl \
       models/event_classifiers_v1.17_calibrated.pkl \
       models/debut_lasso_universe_v1.17h.pkl \
       models/top100_lasso_v1.17h.pkl \
       models/model_b_outcomes_v1.17h.pkl \
       v1.17_fit_long.csv v1.17_val_long.csv
