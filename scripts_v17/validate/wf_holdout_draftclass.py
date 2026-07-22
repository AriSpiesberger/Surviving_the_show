"""Leave-one-draft-class-out walkforward validation.

Drop an entire draft class from hazard TRAINING, then score that class at every
snap (entry+0..max_offset) and evaluate the cumulative predictions vs their
now-resolved outcomes, per snap_offset (walkforward). A fully held-out cohort is
the strictest generalization test — no player-level leakage of any kind.

Reuses an existing landmark panel cache (features are model-independent; only the
train mask changes), so it's a fast refit rather than a full panel rebuild.

    python -m scripts_v17.validate.wf_holdout_draftclass --draft-year 2010 \
        --panel scratch/v20b_oof_v3/panel_cache.npz \
        --meta  scratch/v20b_oof_v3/panel_meta.pkl \
        --out   results/wf_holdout_2010_long.csv
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from prospects.classifier.architectures import landmark_survival as lm
from scripts_v17.train.train_v1_18b_prod import score_pids_with_landmark

EVENTS = ["TOP_100_PROSPECT", "MLB_DEBUT", "ESTABLISHED_MLB",
          "STAR_PLUS_ELITE"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft-year", type=int, default=2010,
                    help="Draft class to hold out of training + evaluate.")
    ap.add_argument("--panel", default="scratch/v20b_oof_v3/panel_cache.npz")
    ap.add_argument("--meta", default="scratch/v20b_oof_v3/panel_meta.pkl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-offset", type=int, default=10)
    args = ap.parse_args()
    dy = args.draft_year
    out_csv = Path(args.out or f"results/wf_holdout_{dy}_long.csv")

    print(f"Loading panel {Path(args.panel).name}")
    npz = np.load(args.panel, allow_pickle=True)
    X_lm = npz["X_lm"]
    pids = npz["pids"].tolist()
    S_yrs = npz["S_yrs"].tolist()
    joined_idx = npz["joined_idx"]
    meta = pickle.load(open(args.meta, "rb"))
    prospects_list = meta["prospects"]
    stats_by_pid = meta["stats_by_pid"]
    joined = [prospects_list[i] for i in joined_idx]

    holdout = {p["player_id"] for p in prospects_list
               if p.get("draft_year") == dy}
    train_mask = np.array([pid not in holdout for pid in pids], dtype=bool)
    print(f"Holdout draft class {dy}: {len(holdout):,} players  "
          f"({int((~train_mask).sum()):,} of {len(pids):,} landmark rows dropped "
          f"from training)")

    print(f"Refitting landmark hazards WITHOUT the {dy} class...")
    hazards = lm.fit_landmark_hazards(
        X_lm, joined, S_yrs, stats_by_pid,
        train_mask=train_mask, seed=args.seed, verbose=True,
    )

    print(f"\nScoring the held-out {dy} class walkforward...")
    n = score_pids_with_landmark(
        hazards, prospects_list, stats_by_pid, holdout, out_csv,
        max_entry_year=dy, max_offset=args.max_offset, verbose=True,
    )
    print(f"  wrote {n:,} (player, snap) rows -> {out_csv}")

    # ---- Walkforward evaluation ----
    df = pd.read_csv(out_csv)
    print(f"\n=== {dy} held-out walkforward — cumulative hazard predictions ===")
    for ev in EVENTS:
        pcol, ecol, rcol = f"p_{ev}", f"eligible_{ev}", f"realized_{ev}"
        if pcol not in df.columns:
            continue
        print(f"\n{ev}")
        print(f"{'yip':>3}{'n':>7}{'pos':>6}{'base%':>8}{'AUC':>8}{'AP':>8}"
              f"{'AP_lift':>9}")
        for off in range(0, args.max_offset + 1):
            sub = df[df["snap_offset"] == off]
            if ecol in sub.columns:
                sub = sub[sub[ecol] == 1]
            n_ = len(sub)
            if n_ < 20:
                continue
            y = sub[rcol].astype(int).values
            p = sub[pcol].astype(float).values
            pos = int(y.sum())
            base = y.mean()
            auc = roc_auc_score(y, p) if 0 < pos < n_ else float("nan")
            aps = average_precision_score(y, p) if pos > 0 else float("nan")
            lift = aps / base if base > 0 and aps == aps else float("nan")
            print(f"{off:>3}{n_:>7}{pos:>6}{base*100:>7.1f}%{auc:>8.3f}"
                  f"{aps:>8.3f}{lift:>9.1f}")


if __name__ == "__main__":
    main()
