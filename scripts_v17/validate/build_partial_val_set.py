"""Build a PARTIAL validation set: score the held-out val slice on
partial-sampled (mid-season) inputs through a chosen hazards model.

This is the controlled A/B harness for the partial-final-season experiment.
The val players are right-censored away from every fold's training, so
scoring them with any fold's hazards is honest. By holding the val cohort +
the partial down-sampling seed FIXED and varying only the hazards model
(baseline vs partial) + its XGB head, we isolate the effect of TRAINING on
partial seasons from the effect of partial inputs simply carrying less signal.

Both arms must use the SAME --partial-seed and the SAME val cohort so the
emitted (player, snap) rows line up one-for-one:

  # arm A — baseline model on partial inputs
  python -m scripts_v17.validate.build_partial_val_set \
      --hazards scratch/v20b_oof/fold5_hazards.pkl \
      --out results/training/v2.0b_valpartial_baselineHaz_long.csv

  # arm B — partial model on partial inputs
  python -m scripts_v17.validate.build_partial_val_set \
      --hazards scratch/v20b_oof_partial/fold5_hazards.pkl \
      --out results/training/v2.0b_valpartial_partialHaz_long.csv

Then score each long with its matching XGB via regen_eval_v2_0b_honest
(--xgb ... --val-long ...).
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from prospects.classifier.architectures.survival import MAX_OBS_YEAR
from scripts_v17.train.run_v2_0b_oof import _score_checkpointed

VAL_PIDS_PATH = REPO_ROOT / "results" / "training" / "v17_prod_val_pids.txt"
DEFAULT_META = REPO_ROOT / "scratch" / "v20b_oof_partial" / "panel_meta.pkl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hazards", required=True,
                    help="Fold hazards pkl to score val with (baseline or "
                         "partial). Val is held out of all folds, so any is "
                         "honest.")
    ap.add_argument("--out", required=True, help="Output val_long CSV.")
    ap.add_argument("--partial-seed", type=int, default=42,
                    help="Down-sample seed for the val inputs (MUST match "
                         "across arms so rows align). Use the same 42 the "
                         "training run used.")
    ap.add_argument("--meta", default=str(DEFAULT_META),
                    help="panel_meta.pkl providing prospects + raw "
                         "stats_by_pid (raw seasons; partial sampling is "
                         "applied at scoring time).")
    ap.add_argument("--max-entry", type=int, default=2020)
    ap.add_argument("--max-offset", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=15)
    args = ap.parse_args()

    print(f"Loading hazards {Path(args.hazards).name}")
    with open(args.hazards, "rb") as fh:
        hazards = pickle.load(fh)

    print(f"Loading meta {Path(args.meta).name}")
    with open(args.meta, "rb") as fh:
        meta = pickle.load(fh)
    prospects_all = meta["prospects"]
    stats_by_pid = meta["stats_by_pid"]
    print(f"  {len(prospects_all):,} prospects, "
          f"{len(stats_by_pid):,} with stats")

    val_pid_set = {ln.strip() for ln in VAL_PIDS_PATH.read_text().splitlines()
                   if ln.strip()}
    print(f"  {len(val_pid_set):,} val pids")

    out_csv = Path(args.out)
    partial_dir = out_csv.parent / (out_csv.stem + "_snaps")
    print(f"Scoring val on partial inputs (seed={args.partial_seed})...")
    n = _score_checkpointed(
        hazards, prospects_all, stats_by_pid, val_pid_set,
        out_csv, partial_dir,
        max_entry_year=args.max_entry, observe_through=MAX_OBS_YEAR,
        max_offset=args.max_offset, horizon=args.horizon,
        partial_seed=args.partial_seed,
    )
    print(f"Wrote {n:,} rows -> {out_csv}")


if __name__ == "__main__":
    main()
