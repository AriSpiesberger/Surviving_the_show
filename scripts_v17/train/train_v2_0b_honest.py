"""Honest non-OOF v2.0b validation — end-to-end, no shortcuts, no leaks.

The contract:
  Universe (entry <= 2020) is split into three disjoint slices by pid:
    HAZ ≈ 71% of pids   used to train hazards (and NOTHING else)
    FIT ≈ 14.7% of pids used to train the XGB head
    VAL ≈ 14.7% of pids used to validate everything

  Critically:
    - Hazards train only on HAZ pids  → fit+val features are honest
    - XGB sees fit_long features that the hazards never trained on
    - XGB validates on val_long features that the hazards never trained on
    - The XGB-layer split (fit vs val pids) keeps the XGB itself honest

Outputs (every file is unambiguously tagged "honest" so it can't be
confused with the leaky prod variants):
  scratch/v20b_honest/hazards.pkl                   honest landmark hazards
  results/training/v2.0b_honest_fit_long.csv        FIT scored by honest haz
  results/training/v2.0b_honest_val_long.csv        VAL scored by honest haz
  models/joint_xgb_v2.0b_honest.pkl                 final honest XGB twin
  results/training/v2.0b_honest_val_metrics.json    headline honest numbers

Each scored long CSV also gets a sibling .meta.json that names the
hazards pickle path + sha, the train_mask used (HAZ pid list path),
and a timestamp — so anything downstream can sanity-check provenance.

Resumable per stage (skips on existence). Re-run picks up where it died.

Usage:
    python -m scripts_v17.train.train_v2_0b_honest
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from prospects.classifier.architectures import landmark_survival as lm
from prospects.classifier.architectures.survival import MAX_OBS_YEAR
from scripts_v17.train.run_v2_0b_oof import (
    PANEL_META, PANEL_NPZ, _score_checkpointed,
)

REPO_TRAIN = REPO_ROOT / "results" / "training"
FIT_PIDS = REPO_TRAIN / "v17_prod_fit_pids.txt"
VAL_PIDS = REPO_TRAIN / "v17_prod_val_pids.txt"

HONEST_DIR = REPO_ROOT / "scratch" / "v20b_honest"
HAZARDS_PKL = HONEST_DIR / "hazards.pkl"
HAZ_PIDS_TXT = HONEST_DIR / "haz_pids.txt"

FIT_LONG = REPO_TRAIN / "v2.0b_honest_fit_long.csv"
VAL_LONG = REPO_TRAIN / "v2.0b_honest_val_long.csv"
XGB_OUT = REPO_ROOT / "models" / "joint_xgb_v2.0b_honest.pkl"
METRICS_JSON = REPO_TRAIN / "v2.0b_honest_val_metrics.json"


def _read_pids(p: Path) -> set[str]:
    return {ln.strip() for ln in p.read_text().splitlines() if ln.strip()}


def _file_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _write_csv_meta(csv_path: Path, hazards_pkl: Path,
                     haz_pid_list: Path, slice_name: str):
    meta = {
        "csv": csv_path.name,
        "hazards_pkl": str(hazards_pkl),
        "hazards_sha16": _file_sha(hazards_pkl),
        "train_mask_pid_list": str(haz_pid_list),
        "train_mask_sha16": _file_sha(haz_pid_list),
        "slice": slice_name,
        "produced_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "produced_by": Path(__file__).name,
    }
    sidecar = csv_path.with_suffix(csv_path.suffix + ".meta.json")
    sidecar.write_text(json.dumps(meta, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO_ROOT / "prospects_snapshot.db"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-entry-year", type=int, default=2020)
    args = ap.parse_args()

    HONEST_DIR.mkdir(parents=True, exist_ok=True)
    REPO_TRAIN.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("=" * 78)
    print("v2.0b HONEST non-OOF — 80/10/10  HAZ / FIT / VAL  no leakage")
    print("=" * 78)

    fit_pids = _read_pids(FIT_PIDS)
    val_pids = _read_pids(VAL_PIDS)
    overlap = fit_pids & val_pids
    if overlap:
        sys.exit(f"FATAL: {len(overlap)} pids in BOTH fit and val. "
                 f"Refusing to proceed.")
    print(f"FIT pids: {len(fit_pids):,}   VAL pids: {len(val_pids):,}")

    # ---- Stage 1: load panel cache ----
    print("\n[1/4] Loading panel cache...")
    npz = np.load(PANEL_NPZ, allow_pickle=True)
    X_lm = npz["X_lm"]
    pids_list = npz["pids"].tolist()
    S_yrs = npz["S_yrs"].tolist()
    joined_idx = npz["joined_idx"]
    with PANEL_META.open("rb") as fh:
        meta = pickle.load(fh)
    prospects_list = meta["prospects"]
    stats_by_pid = meta["stats_by_pid"]
    joined = [prospects_list[i] for i in joined_idx]
    print(f"  X_lm={X_lm.shape}")

    # Build the HAZ pid set: universe minus fit minus val
    universe = set(p["player_id"] for p in prospects_list)
    haz_pids = universe - fit_pids - val_pids
    pct = 100.0 * len(haz_pids) / len(universe)
    print(f"  HAZ pids: {len(haz_pids):,} of {len(universe):,} "
          f"({pct:.1f}% — the 80%-ish slice)")
    HAZ_PIDS_TXT.write_text("\n".join(sorted(haz_pids)) + "\n")

    haz_mask = np.array([p in haz_pids for p in pids_list], dtype=bool)
    print(f"  HAZ landmark rows: {int(haz_mask.sum()):,} / "
          f"{len(pids_list):,}")
    # Hard sanity check: NO val or fit landmark rows in the training mask
    val_or_fit = (fit_pids | val_pids)
    leaked = sum(1 for i, p in enumerate(pids_list)
                  if haz_mask[i] and p in val_or_fit)
    if leaked:
        sys.exit(f"FATAL: train_mask contains {leaked} fit/val landmark "
                 f"rows. Refusing to train leaky hazards.")
    print(f"  train_mask leak check: 0 fit/val rows in HAZ mask ✓")

    # ---- Stage 2: train honest hazards ----
    if HAZARDS_PKL.exists():
        print(f"\n[2/4] reusing {HAZARDS_PKL.name}")
        with HAZARDS_PKL.open("rb") as fh:
            hazards = pickle.load(fh)
    else:
        print(f"\n[2/4] training honest hazards on HAZ slice only "
              f"(no fit, no val)")
        t = time.time()
        hazards = lm.fit_landmark_hazards(
            X_lm, joined, S_yrs, stats_by_pid,
            train_mask=haz_mask, seed=args.seed, verbose=True,
        )
        print(f"  hazards trained in {time.time()-t:.0f}s")
        tmp = HAZARDS_PKL.with_suffix(".pkl.tmp")
        with tmp.open("wb") as fh:
            pickle.dump(hazards, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(HAZARDS_PKL)
        print(f"  saved {HAZARDS_PKL}")

    # Dedupe prospects_all for scoring
    seen: set[str] = set()
    prospects_all: list[dict] = []
    for p in joined:
        if p["player_id"] in seen:
            continue
        seen.add(p["player_id"])
        prospects_all.append(p)
    del X_lm, pids_list, S_yrs, joined_idx, joined, npz
    gc.collect()

    # ---- Stage 3: score FIT and VAL with the honest hazards ----
    for slice_name, slice_pids, out_csv in (
        ("FIT", fit_pids, FIT_LONG),
        ("VAL", val_pids, VAL_LONG),
    ):
        if out_csv.exists():
            print(f"\n[3/4] {slice_name}: reusing {out_csv.name}")
        else:
            print(f"\n[3/4] {slice_name}: scoring {len(slice_pids):,} pids "
                  f"with honest hazards")
            partial_dir = HONEST_DIR / f"{slice_name.lower()}_partial"
            n = _score_checkpointed(
                hazards, prospects_all, stats_by_pid, slice_pids, out_csv,
                partial_dir,
                max_entry_year=args.max_entry_year,
                observe_through=MAX_OBS_YEAR, max_offset=10, horizon=15,
            )
            print(f"  wrote {n:,} rows -> {out_csv.name}")
        _write_csv_meta(out_csv, HAZARDS_PKL, HAZ_PIDS_TXT, slice_name)

    # ---- Stage 4: fit honest XGB ----
    print(f"\n[4/4] training honest XGB on {FIT_LONG.name} + {VAL_LONG.name}")
    tmp = str(XGB_OUT) + ".tmp"
    rc = subprocess.run([
        sys.executable, "-m", "scripts_v17.train.fit_joint_xgb_v2",
        "--fit", str(FIT_LONG),
        "--val", str(VAL_LONG),
        "--db", args.db,
        "--out", tmp,
    ], cwd=REPO_ROOT).returncode
    if rc != 0:
        sys.exit(rc)
    Path(tmp).replace(XGB_OUT)

    # ---- Report + metrics file ----
    with XGB_OUT.open("rb") as fh:
        bundle = pickle.load(fh)
    metrics = bundle.get("metrics_val", [])
    weighted_ap = 0.0
    weight_total = 0.0
    EVENT_WEIGHTS = {
        "TOP_100_PROSPECT": 1.0, "MLB_DEBUT": 2.0,
        "ESTABLISHED_MLB": 1.0, "STAR_PLUS_ELITE": 1.0,
    }
    for r in metrics:
        w = EVENT_WEIGHTS.get(r["event"], 1.0)
        weighted_ap += w * float(r["ap"])
        weight_total += w
    weighted_ap = weighted_ap / weight_total if weight_total else 0.0

    print(f"\n===== v2.0b HONEST val ({XGB_OUT.name}) =====")
    print(f"  best_iter: {bundle.get('best_iteration')}")
    for r in metrics:
        print(f"  {r['event']:<22} AP={r['ap']:.3f}  "
              f"lift={r['ap_lift']:.1f}x  AUC={r['auc']:.3f}")
    print(f"  weighted-AP (MLB_DEBUT 2x): {weighted_ap:.4f}")

    METRICS_JSON.write_text(json.dumps({
        "model": str(XGB_OUT),
        "fit_long": str(FIT_LONG),
        "val_long": str(VAL_LONG),
        "hazards_pkl": str(HAZARDS_PKL),
        "hazards_sha16": _file_sha(HAZARDS_PKL),
        "best_iteration": int(bundle.get("best_iteration", -1)),
        "weighted_ap": weighted_ap,
        "metrics_val": metrics,
        "wall_min": (time.time() - t0) / 60,
        "produced_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2))
    print(f"\nwrote {METRICS_JSON}")
    print(f"DONE in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
