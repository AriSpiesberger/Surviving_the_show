"""Train v1.18 production artifacts from prod fit + val longs.

Pipeline (invoked by prospects.deploy.weekly_score during retrain):
  1. Concatenate v1.17_prod_fit_long.csv + v1.17_prod_val_long.csv
     (the prod hazard-scored slices written by refit_models_honest /
     score_cal_slice).
  2. Train per-event L1-logistic bundle ->
       models/lasso_logits_v1.18_prod.pkl
  3. Train time-to-debut Lasso regression ->
       models/time_to_debut_v1.18_prod.pkl

Atomic: writes to .tmp then renames so a partial fail leaves the prior
artifacts in place.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit",
                    default=str(REPO_ROOT / "results" / "training" /
                                 "v1.17_prod_fit_long.csv"))
    ap.add_argument("--val",
                    default=str(REPO_ROOT / "results" / "training" /
                                 "v1.17_prod_val_long.csv"))
    ap.add_argument("--db", default=str(REPO_ROOT / "prospects_snapshot.db"))
    ap.add_argument("--combined-out",
                    default=str(REPO_ROOT / "v1.17_prod_all_long.csv"))
    ap.add_argument("--bundle-out",
                    default=str(REPO_ROOT / "models" /
                                 "lasso_logits_v1.18_prod.pkl"))
    ap.add_argument("--timing-out",
                    default=str(REPO_ROOT / "models" /
                                 "time_to_debut_v1.18_prod.pkl"))
    args = ap.parse_args()

    fit = pd.read_csv(args.fit)
    val = pd.read_csv(args.val)
    combined = pd.concat([fit, val], ignore_index=True)
    combined.to_csv(args.combined_out, index=False)
    print(f"[v1.18 train] combined: {len(combined):,} rows, "
          f"{combined.player_id.nunique():,} players "
          f"-> {args.combined_out}")

    # Train bundle
    bundle_tmp = args.bundle_out + ".tmp"
    rc = subprocess.run(
        [sys.executable, "-m", "scripts_v17.train.fit_lasso_logits_v18",
         "--fit", args.combined_out,
         "--val", args.combined_out,
         "--db", args.db,
         "--out", bundle_tmp],
        cwd=REPO_ROOT,
    ).returncode
    if rc != 0:
        sys.exit(rc)
    shutil.move(bundle_tmp, args.bundle_out)
    print(f"[v1.18 train] wrote {args.bundle_out}")

    # Train timing
    timing_tmp = args.timing_out + ".tmp"
    rc = subprocess.run(
        [sys.executable, "-m", "scripts_v17.train.fit_time_to_debut_v18",
         "--fit", args.combined_out,
         "--val", args.combined_out,
         "--db", args.db,
         "--bundle", args.bundle_out,
         "--include-p-debut",
         "--out", timing_tmp],
        cwd=REPO_ROOT,
    ).returncode
    if rc != 0:
        sys.exit(rc)
    shutil.move(timing_tmp, args.timing_out)
    print(f"[v1.18 train] wrote {args.timing_out}")


if __name__ == "__main__":
    main()
