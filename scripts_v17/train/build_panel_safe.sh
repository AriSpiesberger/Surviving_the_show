#!/bin/bash
# Safer panel build: skips existing partitions, low thread count, sleeps between partitions.
# Usage: ./build_panel_safe.sh <out_panel.npz> <max_draft_year>

set -e
OUT="${1:-panel_v1.14n.npz}"
MAXDRAFT="${2:-2025}"
MAXYEAR="${3:-2026}"
N=16

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Building $OUT (skip-existing, single-threaded BLAS)"
for p in $(seq 0 $((N - 1))); do
    part_file="${OUT%.npz}.part${p}.npz"
    if [ -f "$part_file" ]; then
        echo "[part $p] exists, skipping"
        continue
    fi
    success=0
    for try in 1 2 3 4 5; do
        echo "[part $p] try $try"
        python -m prospects.classifier.build_panel \
            --out "$OUT" --max-draft-year "$MAXDRAFT" --max-year "$MAXYEAR" \
            --n-partitions "$N" --partition "$p" > /dev/null 2>&1
        if [ -f "$part_file" ]; then
            echo "[part $p] OK"
            success=1
            break
        fi
        echo "[part $p] try $try failed"
        sleep 3
    done
    if [ $success -eq 0 ]; then
        echo "[part $p] FAILED after 5 tries — aborting"
        exit 1
    fi
    # Brief pause for OS to reclaim memory between partitions
    sleep 2
done

echo "All partitions present, merging..."
python -m prospects.classifier.build_panel --out "$OUT" --n-partitions "$N" --merge
echo "Done."
