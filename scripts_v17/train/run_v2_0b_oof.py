"""End-to-end v2.0b K-fold OOF training — single process, tqdm progress.

Run in YOUR OWN terminal (not via the Claude Code background task runner),
so the Windows process killer doesn't interfere:

    python -m scripts_v17.train.run_v2_0b_oof

Every stage is resumable. If the script dies mid-way, just re-run it —
finished stages are detected from on-disk artifacts and skipped.

Pipeline:
  Stage 1 — landmark panel  (~25 min on cold cache)
            scratch/v20b_oof/panel_cache.npz   X_lm + pids + S_yrs + idx
            scratch/v20b_oof/panel_meta.pkl    prospects + stats_by_pid

  Stage 2 — fold partitioning  (instant)
            scratch/v20b_oof/fold{k}_pids.txt   K folds of ~5k pids each
            scratch/v20b_oof/train{k}_pids.txt  the K-1 train pids per fold

  Stage 3 — K hazard fits + OOF scoring  (~5-10 min per fold)
            scratch/v20b_oof/fold{k}_long.csv  ~47k scored rows per fold

  Stage 4 — stack OOF folds  (instant)
            results/training/v2.0b_oof_stacked_long.csv  ~250k rows

  Stage 5 — score val with fold-K-1 hazards  (~5 min, val is held out)
            results/training/v2.0b_oof_val_long.csv  ~30k rows

  Stage 6 — joint XGB fit on stacked OOF  (~5 min)
            models/joint_xgb_v2.0b_oof.pkl
"""
from __future__ import annotations

import argparse
import gc
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from prospects.classifier.architectures import landmark_survival as lm
from prospects.classifier.architectures.landmark_survival import (
    N_FEATURES, _start_year,
)
from prospects.classifier.architectures.survival import (
    MAX_OBS_YEAR, build_windowed_features,
)
from prospects.storage import ProspectDB
from scripts_v17.train.train_v1_18b_prod import score_pids_with_landmark

# ---- paths ----
SCRATCH = REPO_ROOT / "scratch" / "v20b_oof"
TRAIN_DIR = REPO_ROOT / "results" / "training"
VAL_PIDS_PATH = TRAIN_DIR / "v17_prod_val_pids.txt"
PANEL_NPZ = SCRATCH / "panel_cache.npz"
PANEL_META = SCRATCH / "panel_meta.pkl"
OOF_STACKED = TRAIN_DIR / "v2.0b_oof_stacked_long.csv"
OOF_VAL = TRAIN_DIR / "v2.0b_oof_val_long.csv"
XGB_OUT = REPO_ROOT / "models" / "joint_xgb_v2.0b_oof.pkl"


def _entry_year(player: dict, stats_by_pid: dict) -> int | None:
    dy = player.get("draft_year")
    if dy is not None and int(player.get("is_international") or 0) == 0:
        return int(dy)
    yrs = [s.get("season_year")
           for s in stats_by_pid.get(player["player_id"], [])
           if s.get("season_year") is not None
           and (s.get("level") or "").upper() != "MLB"]
    if yrs:
        return int(min(yrs))
    return int(dy) if dy is not None else None


# ---- Stage 1: panel build with tqdm ----
def stage_panel(db_path: str, max_draft_year: int) -> tuple:
    if PANEL_NPZ.exists() and PANEL_META.exists():
        print(f"[Stage 1] loading panel cache {PANEL_NPZ.name}")
        npz = np.load(PANEL_NPZ, allow_pickle=True)
        X_lm = npz["X_lm"]
        pids = npz["pids"].tolist()
        S_yrs = npz["S_yrs"].tolist()
        joined_idx = npz["joined_idx"]
        with PANEL_META.open("rb") as fh:
            meta = pickle.load(fh)
        prospects_list = meta["prospects"]
        stats_by_pid = meta["stats_by_pid"]
        joined = [prospects_list[i] for i in joined_idx]
        print(f"           X_lm={X_lm.shape}  prospects="
              f"{len(prospects_list):,}")
        return X_lm, pids, S_yrs, joined, stats_by_pid

    print(f"[Stage 1] building landmark panel (max_draft_year={max_draft_year})")
    db = ProspectDB(db_path)
    with db._connect() as conn:
        prospects = [dict(r) for r in conn.execute("""
            SELECT p.*, o.mlb_debut_year, o.year_established_mlb,
                   o.year_top_100, o.year_top_25,
                   o.year_all_star_once, o.year_all_star_three,
                   o.year_major_award, o.year_hof_trajectory,
                   o.events_json, o.final_mlb_year
            FROM prospects p
            JOIN career_outcomes o ON o.player_id = p.player_id
            WHERE (p.draft_year IS NOT NULL AND p.draft_year <= ?)
               OR COALESCE(p.is_international, 0) = 1
        """, (max_draft_year,)).fetchall()]
        stats_rows = conn.execute("SELECT * FROM season_stats").fetchall()
        try:
            rank_rows = conn.execute(
                "SELECT player_id, year, rank, source "
                "FROM prospect_rankings").fetchall()
        except Exception:
            rank_rows = []

    stats_by_pid: dict[str, list[dict]] = {}
    for s in stats_rows:
        d = dict(s)
        stats_by_pid.setdefault(d["player_id"], []).append(d)

    rankings_by_pid: dict[str, list[tuple[int, int, str]]] = {}
    for r in rank_rows:
        rankings_by_pid.setdefault(r[0], []).append((r[1], r[2], r[3]))
    for p in prospects:
        p["_top100_rankings"] = rankings_by_pid.get(p["player_id"], [])

    n_draft = sum(1 for p in prospects if p.get("draft_year") is not None)
    print(f"           {len(prospects):,} prospects "
          f"(drafted {n_draft:,} + IFA {len(prospects)-n_draft:,})")

    plan = []
    pids: list[str] = []
    S_list: list[int] = []
    joined: list[dict] = []
    n_skipped = 0
    for p in prospects:
        stats = stats_by_pid.get(p["player_id"], [])
        sy = _start_year(p, stats_by_pid)
        if sy is None:
            n_skipped += 1
            continue
        lo = max(sy + 1, 2007)
        hi = MAX_OBS_YEAR - 1
        if lo > hi:
            continue
        for S in range(lo, hi + 1):
            plan.append((p, stats, S))
            pids.append(p["player_id"])
            S_list.append(S)
            joined.append(p)
    n_rows = len(plan)
    print(f"           {n_rows:,} landmark rows  ({n_skipped} skipped)")

    X_lm = np.empty((n_rows, N_FEATURES), dtype=np.float32)
    for i in tqdm(range(n_rows), desc="panel", unit="row",
                  mininterval=1.0, smoothing=0.05):
        p, stats, S = plan[i]
        X_lm[i, :] = build_windowed_features(p, stats, S, milb_only=True)
        if (i + 1) % 25000 == 0:
            gc.collect()

    print(f"[Stage 1] writing cache...")
    pid_to_idx: dict[str, int] = {}
    prospects_list: list[dict] = []
    joined_idx = np.empty(len(joined), dtype=np.int32)
    for r, p in enumerate(joined):
        if p["player_id"] not in pid_to_idx:
            pid_to_idx[p["player_id"]] = len(prospects_list)
            prospects_list.append(p)
        joined_idx[r] = pid_to_idx[p["player_id"]]

    SCRATCH.mkdir(parents=True, exist_ok=True)
    tmp = PANEL_NPZ.with_suffix(".npz.tmp")
    np.savez_compressed(tmp, X_lm=X_lm,
                        pids=np.array(pids, dtype=object),
                        S_yrs=np.array(S_list, dtype=np.int32),
                        joined_idx=joined_idx)
    tmp.replace(PANEL_NPZ)
    tmp = PANEL_META.with_suffix(".pkl.tmp")
    with tmp.open("wb") as fh:
        pickle.dump({"prospects": prospects_list,
                     "stats_by_pid": stats_by_pid},
                    fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(PANEL_META)
    print(f"           {PANEL_NPZ.name}: "
          f"{PANEL_NPZ.stat().st_size/1e6:.0f} MB")
    print(f"           {PANEL_META.name}: "
          f"{PANEL_META.stat().st_size/1e6:.0f} MB")
    del plan; gc.collect()
    return X_lm, pids, S_list, joined, stats_by_pid


# ---- Stage 2: partition into K folds ----
def stage_partition(prospects: list[dict], stats_by_pid: dict,
                     val_pids: set[str], k: int, seed: int,
                     max_entry_year: int):
    fold_pid_files = [SCRATCH / f"fold{i}_pids.txt" for i in range(k)]
    train_pid_files = [SCRATCH / f"train{i}_pids.txt" for i in range(k)]
    if all(f.exists() for f in fold_pid_files + train_pid_files):
        print(f"[Stage 2] reusing existing fold/train pid lists")
        fold_sets = [set(f.read_text().splitlines()) - {""}
                     for f in fold_pid_files]
        return fold_sets

    universe = []
    for p in prospects:
        if p["player_id"] in val_pids:
            continue
        ey = _entry_year(p, stats_by_pid)
        if ey is None or ey > max_entry_year:
            continue
        universe.append(p["player_id"])
    universe = sorted(set(universe))
    print(f"[Stage 2] universe = {len(universe):,} pids, K={k}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(universe))
    fold_arrays = np.array_split(perm, k)
    fold_sets = [set(universe[i] for i in arr) for arr in fold_arrays]
    for j in range(k):
        fold_pid_files[j].write_text("\n".join(sorted(fold_sets[j])) + "\n")
        train_set: set[str] = set()
        for m in range(k):
            if m != j:
                train_set |= fold_sets[m]
        train_pid_files[j].write_text("\n".join(sorted(train_set)) + "\n")
        print(f"           fold {j}: heldout={len(fold_sets[j]):,}  "
              f"train={len(train_set):,}")
    return fold_sets


# ---- Stage 3 helper: one fold ----
def run_one_fold(X_lm, pids, S_yrs, joined, stats_by_pid,
                  prospects_all, train_set: set[str], score_set: set[str],
                  out_csv: Path, max_entry_year: int, seed: int):
    t = time.time()
    train_mask = np.array([p in train_set for p in pids], dtype=bool)
    print(f"           train_mask: {int(train_mask.sum()):,} / "
          f"{len(pids):,} landmark rows")
    hazards = lm.fit_landmark_hazards(
        X_lm, joined, S_yrs, stats_by_pid,
        train_mask=train_mask, seed=seed, verbose=True,
    )
    print(f"           hazards fit in {time.time()-t:.0f}s, scoring...")
    n = score_pids_with_landmark(
        hazards, prospects_all, stats_by_pid, score_set, out_csv,
        max_entry_year=max_entry_year,
        observe_through=MAX_OBS_YEAR, max_offset=10, horizon=15,
        verbose=True,
    )
    df = pd.read_csv(out_csv)
    for c in df.columns:
        if c.startswith("mean_t_"):
            df[c] = df[c].fillna(15.0)
        elif c.startswith("sd_t_"):
            df[c] = df[c].fillna(0.0)
    df.to_csv(out_csv, index=False)
    return n, hazards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO_ROOT / "prospects_snapshot.db"))
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-draft-year", type=int, default=2020)
    ap.add_argument("--max-entry-year", type=int, default=2020)
    args = ap.parse_args()

    SCRATCH.mkdir(parents=True, exist_ok=True)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    print("="*78)
    print(f"v2.0b OOF — K={args.k}, seed={args.seed}, "
          f"max_entry={args.max_entry_year}")
    print("="*78)
    t_start = time.time()

    val_pid_set = {ln.strip() for ln in VAL_PIDS_PATH.read_text().splitlines()
                   if ln.strip()}
    print(f"Val pids held out: {len(val_pid_set):,}")

    # Stage 1
    X_lm, pids, S_yrs, joined, stats_by_pid = stage_panel(
        args.db, args.max_draft_year)

    # Stage 2 — need a deduped prospects list for partitioning
    prospects_seen: set[str] = set()
    prospects_all: list[dict] = []
    for p in joined:
        if p["player_id"] not in prospects_seen:
            prospects_seen.add(p["player_id"])
            prospects_all.append(p)

    fold_sets = stage_partition(
        prospects_all, stats_by_pid, val_pid_set,
        args.k, args.seed, args.max_entry_year)

    # Stage 3 — K folds
    fold_csvs = [SCRATCH / f"fold{j}_long.csv" for j in range(args.k)]
    last_hazards = None
    last_train_set = None
    for j in range(args.k):
        out = fold_csvs[j]
        train_set: set[str] = set()
        for m in range(args.k):
            if m != j:
                train_set |= fold_sets[m]
        if out.exists():
            print(f"\n[Stage 3] fold {j+1}/{args.k}: reusing {out.name}")
            last_train_set = train_set
            last_hazards = None  # need to retrain for val score below
            continue
        print(f"\n[Stage 3] fold {j+1}/{args.k} -> {out.name}")
        n, hazards = run_one_fold(
            X_lm, pids, S_yrs, joined, stats_by_pid, prospects_all,
            train_set=train_set, score_set=fold_sets[j], out_csv=out,
            max_entry_year=args.max_entry_year, seed=args.seed,
        )
        print(f"           wrote {n:,} rows")
        last_hazards = hazards
        last_train_set = train_set

    # Stage 4 — stack
    if OOF_STACKED.exists():
        print(f"\n[Stage 4] reusing stacked {OOF_STACKED.name}")
    else:
        print(f"\n[Stage 4] stacking {args.k} folds -> {OOF_STACKED.name}")
        stacked = pd.concat([pd.read_csv(f) for f in fold_csvs],
                            ignore_index=True)
        stacked.to_csv(OOF_STACKED, index=False)
        print(f"           {len(stacked):,} rows, "
              f"{stacked.player_id.nunique():,} pids")

    # Stage 5 — score val using last fold's train hazards
    if OOF_VAL.exists():
        print(f"\n[Stage 5] reusing val {OOF_VAL.name}")
    else:
        print(f"\n[Stage 5] scoring {len(val_pid_set):,} val pids")
        if last_hazards is None:
            # Train from last_train_set fresh
            print(f"           retraining last-fold hazards for val score")
            train_mask = np.array([p in last_train_set for p in pids],
                                   dtype=bool)
            last_hazards = lm.fit_landmark_hazards(
                X_lm, joined, S_yrs, stats_by_pid,
                train_mask=train_mask, seed=args.seed, verbose=True,
            )
        n = score_pids_with_landmark(
            last_hazards, prospects_all, stats_by_pid, val_pid_set, OOF_VAL,
            max_entry_year=args.max_entry_year,
            observe_through=MAX_OBS_YEAR, max_offset=10, horizon=15,
            verbose=True,
        )
        dfv = pd.read_csv(OOF_VAL)
        for c in dfv.columns:
            if c.startswith("mean_t_"):
                dfv[c] = dfv[c].fillna(15.0)
            elif c.startswith("sd_t_"):
                dfv[c] = dfv[c].fillna(0.0)
        dfv.to_csv(OOF_VAL, index=False)
        print(f"           wrote {n:,} rows")

    # Stage 6 — fit joint XGB
    if XGB_OUT.exists():
        print(f"\n[Stage 6] reusing XGB {XGB_OUT.name}")
    else:
        print(f"\n[Stage 6] fitting joint XGB on stacked OOF -> "
              f"{XGB_OUT.name}")
        tmp = str(XGB_OUT) + ".tmp"
        rc = subprocess.run([
            sys.executable, "-m", "scripts_v17.train.fit_joint_xgb_v2",
            "--fit", str(OOF_STACKED),
            "--val", str(OOF_VAL),
            "--db", args.db,
            "--out", tmp,
        ], cwd=REPO_ROOT).returncode
        if rc != 0:
            sys.exit(rc)
        Path(tmp).replace(XGB_OUT)

    print(f"\n=== DONE in {(time.time()-t_start)/60:.1f} min ===")
    print(f"  fold csvs:   {SCRATCH}/fold[0..{args.k-1}]_long.csv")
    print(f"  stacked OOF: {OOF_STACKED}")
    print(f"  val OOF:     {OOF_VAL}")
    print(f"  XGB:         {XGB_OUT}")


if __name__ == "__main__":
    main()
