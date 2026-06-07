"""Phase 2 of chunked panel build: build feature matrix for ONE chunk.

Loads panel_plan.npz + panel_meta.pkl, processes rows [lo:hi] of the
plan, saves panel_chunk_NN.npz.

Each chunk handles ~25k rows, runs in ~1-1.5 min — short enough that
intermittent Windows process kills don't reliably catch it.
"""
from __future__ import annotations

import argparse
import gc
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from prospects.classifier.architectures.landmark_survival import N_FEATURES
from prospects.features.windowed import build_windowed_features

CACHE_DIR = REPO_ROOT / "scratch" / "v20b_oof"
PLAN_NPZ = CACHE_DIR / "panel_plan.npz"
META_PKL = CACHE_DIR / "panel_meta.pkl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=int, required=True)
    ap.add_argument("--hi", type=int, required=True)
    ap.add_argument("--out", required=True,
                    help="Output npz path (panel_chunk_NN.npz)")
    args = ap.parse_args()

    t0 = time.time()
    if not PLAN_NPZ.exists() or not META_PKL.exists():
        sys.exit(f"FATAL: missing plan/meta. Run build_panel_prep first.")

    plan = np.load(PLAN_NPZ, allow_pickle=True)
    pids = plan["pids"]
    S_yrs = plan["S_yrs"]
    prospect_idx = plan["prospect_idx"]
    n_rows = int(plan["n_rows"])

    lo = max(0, args.lo)
    hi = min(n_rows, args.hi)
    chunk_size = hi - lo
    if chunk_size <= 0:
        sys.exit(f"FATAL: empty chunk lo={lo} hi={hi} n_rows={n_rows}")

    with META_PKL.open("rb") as fh:
        meta = pickle.load(fh)
    prospects = meta["prospects"]
    stats_by_pid = meta["stats_by_pid"]

    print(f"[chunk] rows [{lo}..{hi}) ({chunk_size:,} rows of "
          f"{n_rows:,} total)", flush=True)

    X_chunk = np.empty((chunk_size, N_FEATURES), dtype=np.float32)
    REPORT = 5000
    for j in range(chunk_size):
        i = lo + j
        p = prospects[int(prospect_idx[i])]
        stats = stats_by_pid.get(p["player_id"], [])
        S = int(S_yrs[i])
        vec = build_windowed_features(p, stats, S, milb_only=True)
        X_chunk[j, :] = vec
        if (j + 1) % REPORT == 0:
            pct = 100.0 * (j + 1) / chunk_size
            print(f"  [chunk] {j+1:,}/{chunk_size:,} ({pct:.0f}%)",
                  flush=True)
    gc.collect()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    np.savez(tmp, X_chunk=X_chunk, lo=np.int64(lo), hi=np.int64(hi))
    tmp.replace(out)
    print(f"[chunk] wrote {out.name} ({out.stat().st_size/1e6:.0f} MB) "
          f"in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
