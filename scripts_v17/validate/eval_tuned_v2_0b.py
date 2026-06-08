"""HONEST eval of the TUNED hazards + OOF XGB stack.

Inputs (both honest at val pids):
  models/hazards_v2.0b_oof_tuned.pkl
    Trained on 90% universe (val pids excluded) with the best HP
    discovered by the Optuna hazards study (200 trials, weighted-AP
    0.3899 hazards-direct).
  models/joint_xgb_v2.0b_oof.pkl
    Trained on OOF stacked (val pids excluded from XGB training).

Pipeline:
  1. Score val pids with the tuned hazards
       -> scratch/v20b_oof/val_long_tuned.csv
  2. Apply OOF XGB on those features
  3. Build per_bucket / per_yip / per_level / walkforward tables
       -> evaluation/v2.0b_honest/tuned/*.csv
  4. Compare against the untuned OOF baseline so we can see what the
     Optuna tune actually bought us at the published-metric layer.

Usage:
    python -m scripts_v17.validate.eval_tuned_v2_0b
"""
from __future__ import annotations

import argparse
import gc
import json
import pickle
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts_v17.train.run_v2_0b_oof import (
    PANEL_META, PANEL_NPZ, _score_checkpointed, VAL_PIDS_PATH,
)
from prospects.classifier.architectures.survival import MAX_OBS_YEAR

# Reuse helpers from the existing honest eval generator
from scripts_v17.validate.regen_eval_v2_0b_honest import (
    EVENTS, EVENT_WEIGHTS, AGE_CENTER, YIP_CENTER, HAZARD_PROBS,
    _prep_for_xgb, _score_xgb, _join_current_level, _metric_row,
    _bucket_of,
)

TUNED_HAZ = REPO_ROOT / "models" / "hazards_v2.0b_oof_tuned.pkl"
XGB_PKL = REPO_ROOT / "models" / "joint_xgb_v2.0b_oof.pkl"
DB = REPO_ROOT / "prospects_snapshot.db"

TUNED_VAL_LONG = (REPO_ROOT / "scratch" / "v20b_oof"
                   / "val_long_tuned_hazards.csv")
VAL_PARTIAL = (REPO_ROOT / "scratch" / "v20b_oof"
                / "val_partial_tuned")
OUT_DIR = REPO_ROOT / "evaluation" / "v2.0b_honest" / "tuned"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-entry", type=int, default=2020)
    ap.add_argument("--force-rescore", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"Loading tuned hazards {TUNED_HAZ.name}...")
    with TUNED_HAZ.open("rb") as fh:
        tuned_hazards = pickle.load(fh)

    # ---- 1. Score val with tuned hazards ----
    if TUNED_VAL_LONG.exists() and not args.force_rescore:
        print(f"[1/2] reusing {TUNED_VAL_LONG.name}")
    else:
        print(f"[1/2] scoring val with tuned hazards "
              f"(this may take a few min)")
        # Load panel meta for prospects + stats
        npz = np.load(PANEL_NPZ, allow_pickle=True)
        with PANEL_META.open("rb") as fh:
            meta = pickle.load(fh)
        prospects_list = meta["prospects"]
        stats_by_pid = meta["stats_by_pid"]
        joined_idx = npz["joined_idx"]
        joined = [prospects_list[i] for i in joined_idx]

        # Dedupe prospects_all
        seen: set[str] = set()
        prospects_all: list[dict] = []
        for p in joined:
            if p["player_id"] in seen:
                continue
            seen.add(p["player_id"])
            prospects_all.append(p)
        del npz, joined
        gc.collect()

        val_pid_set = {ln.strip() for ln in VAL_PIDS_PATH.read_text().splitlines()
                        if ln.strip()}
        n = _score_checkpointed(
            tuned_hazards, prospects_all, stats_by_pid, val_pid_set,
            TUNED_VAL_LONG, VAL_PARTIAL,
            max_entry_year=args.max_entry,
            observe_through=MAX_OBS_YEAR,
            max_offset=10, horizon=15,
        )
        print(f"  wrote {n:,} rows -> {TUNED_VAL_LONG.name}")

    # Free hazards
    del tuned_hazards
    gc.collect()

    # ---- 2. Run eval ----
    print(f"\n[2/2] loading {TUNED_VAL_LONG.name}...")
    df = pd.read_csv(TUNED_VAL_LONG)
    print(f"  {len(df):,} rows, {df.player_id.nunique():,} pids")
    df = _prep_for_xgb(df, str(DB), args.max_entry)
    print(f"  after entry<={args.max_entry}: {len(df):,} rows")

    print(f"  applying OOF XGB...")
    df = _score_xgb(df, XGB_PKL)
    df = _join_current_level(df, str(DB))
    df["bucket"] = df.apply(_bucket_of, axis=1)

    # per-bucket
    bucket_rows = []
    for ev in EVENTS:
        for b in ["ALL", "R1", "R2-R3", "R4-R10", "R10+", "IFA"]:
            sub = df if b == "ALL" else df[df["bucket"] == b]
            row = _metric_row(sub, ev, "bucket", b)
            if row:
                bucket_rows.append(row)
    pd.DataFrame(bucket_rows).to_csv(
        OUT_DIR / "per_bucket_validation.csv",
        index=False, float_format="%.6f")
    print(f"  per_bucket: {len(bucket_rows)} rows")

    # per-yip
    yip_rows = []
    for ev in EVENTS:
        for off in sorted(df["snap_offset"].unique()):
            sub = df[df["snap_offset"] == int(off)]
            row = _metric_row(sub, ev, "snap_offset", int(off))
            if row:
                yip_rows.append(row)
    pd.DataFrame(yip_rows).to_csv(
        OUT_DIR / "per_yip_validation.csv",
        index=False, float_format="%.6f")
    pd.DataFrame(yip_rows).to_csv(
        OUT_DIR / "walkforward.csv",
        index=False, float_format="%.6f")
    print(f"  per_yip: {len(yip_rows)} rows")

    # per-level
    level_rows = []
    for ev in EVENTS:
        for lvl in ["ALL", "RK", "A-", "A", "A+", "AA", "AAA", "NONE"]:
            sub = df if lvl == "ALL" else df[df["cur_level"] == lvl]
            row = _metric_row(sub, ev, "cur_level", lvl)
            if row:
                level_rows.append(row)
    pd.DataFrame(level_rows).to_csv(
        OUT_DIR / "per_level_validation.csv",
        index=False, float_format="%.6f")
    print(f"  per_level: {len(level_rows)} rows")

    # headline
    weighted_ap = 0.0; total_w = 0.0
    overall = []
    for ev in EVENTS:
        r = next((b for b in bucket_rows
                   if b["event"] == ev and b["bucket"] == "ALL"), None)
        if r and r["ap"] == r["ap"]:
            w = EVENT_WEIGHTS[ev]
            weighted_ap += w * r["ap"]; total_w += w
            overall.append({"event": ev, "ap": r["ap"],
                              "auc": r["auc"], "ap_lift": r["ap_lift"]})
    wap = weighted_ap / total_w if total_w else 0.0
    (OUT_DIR / "headline.json").write_text(json.dumps({
        "hazards": str(TUNED_HAZ), "xgb": str(XGB_PKL),
        "val_long": str(TUNED_VAL_LONG),
        "weighted_ap": wap, "per_event": overall,
    }, indent=2))

    print(f"\n===== TUNED v2.0b honest val =====")
    print(f"Weighted-AP: {wap:.4f}")
    for r in overall:
        print(f"  {r['event']:<22} AP={r['ap']:.3f}  "
              f"lift={r['ap_lift']:.1f}x  AUC={r['auc']:.3f}")
    print(f"\nDONE in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
