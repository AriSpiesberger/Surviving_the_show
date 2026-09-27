#!/usr/bin/env bash
# One-off data cleanup (2026-09-27, requested by the user). Deletes extraneous data; keeps one
# rollback point (the 2026-09-25 pre-switch DB backups + runs/backups/current_pre_ext_20260925
# and backup_v2.1c_20260925) and the small result files (metrics / summaries) of each experiment.
set -eu
cd "$(dirname "$0")/../.."
B=$(du -sm . | cut -f1)

# archive/: keep only the two pre-switch DB backups
find archive/db_backups -maxdepth 1 -type f ! -name "*pre_ext_20260925" -delete
for d in archive/*/; do [ "$d" = "archive/db_backups/" ] || rm -rf "$d"; done
find archive -maxdepth 1 -type f -delete

# runs/backups/: keep the pre-switch copy and the latest promote backup
for d in runs/backups/*/; do
  case $d in */current_pre_ext_20260925/|*/backup_v2.1c_20260925/) ;; *) rm -rf "$d" ;; esac
done

# runs/experiments/: legacy heavy runs and the pre-2005 build copies go whole
( cd runs/experiments && rm -rf hz0_default hz0_fvfix hz0_fvfix2 hz1_capacity hz2_wide hz3_max fs308 fsbase \
    exp_walkforward exp_walkforward2 exp_walkforward2_leaky_20260906 exp_walkforward3 exp_walkforward4 \
    exp_walkforward_h ext ext2 aug_twins )
# the rest keep only small result files (metrics.csv, bootstrap.csv, summary.json, ...)
find runs/experiments -type f \( -name "*.pkl" -o -name "*.npz" -o -name "*.pt" -o -name "*.db" -o -name "*.parquet" \) -delete
find runs/experiments -type f -size +2M -delete
find runs/experiments -type d -empty -delete

A=$(du -sm . | cut -f1)
echo "freed $(( (B - A) / 1024 )) GB; repo now $(( A / 1024 )) GB"
