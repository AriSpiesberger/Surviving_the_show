#!/bin/bash
# End-to-end v1.17 pipeline. Runs after panel_v1.17.npz exists.
set -e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

PANEL=panel_v1.17.npz
HAZ=models/event_classifiers_v1.17_prod.pkl

# 1. Train hazards on 100% of v1.17 panel
echo "=== Training v1.17 hazards on full panel ==="
python -m prospects.classifier.train_full_v14d \
  --panel "$PANEL" \
  --lasso-fit-frac 0.0 --lasso-val-frac 0.0 \
  --out "$HAZ"

# 2. Score fit slice + val slice
# We need to determine which players belong to fit/val for v1.17.
# Use the SAME seed=42 perm on v1.17 panel pids.
echo "=== Reproducing seed=42 fit/val players ==="
python <<'EOF'
import numpy as np
with np.load("panel_v1.17.npz", allow_pickle=True) as d:
    pids = sorted(set(d["pids"].tolist()))
rng = np.random.default_rng(42)
perm = rng.permutation(len(pids))
n_cal = int(round(0.10 * len(pids)))
n_val = int(round(0.10 * len(pids)))
with open("v17_fit_pids.txt","w") as f:
    for i in perm[:n_cal]: f.write(pids[i]+"\n")
with open("v17_val_pids.txt","w") as f:
    for i in perm[n_cal:n_cal+n_val]: f.write(pids[i]+"\n")
print(f"fit:{n_cal}  val:{n_val}")
EOF

# 3. Score fit slice + val slice (chunked)
for slice in fit val; do
  echo "=== Scoring v17 $slice slice ==="
  split -d -l 100 -a 3 v17_${slice}_pids.txt v17_${slice}_chunk_
  mkdir -p v17_${slice}_out
  for f in v17_${slice}_chunk_*; do
    out="v17_${slice}_out/${f}.csv"
    [ -f "$out" ] && continue
    for try in 1 2 3; do
      python -m prospects.classifier.score_v14c_cal_slice_raw \
        --model "$HAZ" --panel "$PANEL" \
        --players-file "$f" --max-entry-year 2020 \
        --observe-through 2026 --max-offset 10 \
        --out "$out" > /tmp/score.log 2>&1 || true
      [ -f "$out" ] && break
      sleep 2
    done
  done
  python -c "
import pandas as pd, glob
files = sorted(glob.glob('v17_${slice}_out/*.csv'))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df.to_csv('v1.17_${slice}_long.csv', index=False)
print(f'wrote v1.17_${slice}_long.csv: {len(df):,} rows')
"
  rm -f v17_${slice}_chunk_*
done

# 4. Fit lasso on v1.17 fit slice (uses existing lasso_composite logic)
echo "=== Fitting v1.17 lasso ==="
python -m prospects.classifier.lasso_composite \
  --long v1.17_fit_long.csv \
  --time-decay "TOP_100_PROSPECT=3,MLB_DEBUT=4" \
  --require-eligible "TOP_100_PROSPECT,MLB_DEBUT" \
  --out-prefix models/lasso_v1.17_td

# 5. Fit model B on v1.17 fit+val
echo "=== Fitting v1.17 model B ==="
python <<'EOF'
import subprocess, shutil
# Patch fit_model_b.py to point at v1.17 longs and write v1.17 output
import pathlib
src = pathlib.Path("prospects/classifier/fit_model_b.py").read_text()
src2 = src.replace('FIT = "v1.14n_fit_long.csv"', 'FIT = "v1.17_fit_long.csv"')
src2 = src2.replace('VAL = "v1.14n_val_long.csv"', 'VAL = "v1.17_val_long.csv"')
src2 = src2.replace('OUT_MODEL = "models/model_b_outcomes_v1.14n.pkl"',
                    'OUT_MODEL = "models/model_b_outcomes_v1.17.pkl"')
pathlib.Path("prospects/classifier/fit_model_b.py").write_text(src2)
subprocess.run(["python","-m","prospects.classifier.fit_model_b"], check=True)
EOF

echo "=== v1.17 pipeline done ==="
ls models/*v1.17* v1.17_*_long.csv
