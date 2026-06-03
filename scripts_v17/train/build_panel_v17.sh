#!/bin/bash
set -e
OUT="panels/panel_v1.17.npz"
mkdir -p panels
N=64
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "Building $OUT with $N partitions"
for p in $(seq 0 $((N - 1))); do
    part="${OUT%.npz}.part${p}.npz"
    if [ -f "$part" ]; then continue; fi
    for try in 1 2 3 4 5 6 7; do
        python -m prospects.classifier.build_panel \
            --out "$OUT" --max-draft-year 2025 --max-year 2026 \
            --n-partitions "$N" --partition "$p" > /tmp/build_p$p.log 2>&1 || true
        if [ -f "$part" ]; then echo "[$p] OK (try $try)"; break; fi
        sleep 3
    done
    [ -f "$part" ] || { echo "[$p] FAIL after 7 tries"; exit 1; }
done
echo "All $N partitions present, merging..."
python -m prospects.classifier.build_panel --out "$OUT" --n-partitions "$N" --merge
echo "Done."
