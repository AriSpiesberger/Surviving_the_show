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

# Threading: respect any pre-set env vars. Was pinned to 1 during the
# BSOD-instability window; the box is stable now so let HistGB's OpenMP
# use all cores. Set OMP_NUM_THREADS=1 in your shell to force single-
# threaded if you need to.

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
    ELITE_KEY, MAX_OBS_YEAR, STAR_KEY, _trigger_year,
    build_windowed_features,
)
from prospects.features.partial_sample import partial_for_features
from prospects.storage import ProspectDB
from scripts_v17.train.train_v1_18b_prod import (
    _bucket_of, _entry_year as _entry_year_v18, _ev_name,
    score_pids_with_landmark,
)

# ---- paths ----
SCRATCH = REPO_ROOT / "scratch" / "v20b_oof"
TRAIN_DIR = REPO_ROOT / "results" / "training"
LOG_DIR = REPO_ROOT / "logs"


class _TeeUnbuffered:
    """Mirror writes to stdout AND a log file, flushing every line so a
    silent process kill still leaves the last line on disk."""
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            s.write(data)
            try:
                s.flush()
            except Exception:
                pass
    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass


def _install_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "run_v2_0b_oof.log"
    fh = log_path.open("a", buffering=1, encoding="utf-8")  # line-buffered
    fh.write(f"\n===== run started "
             f"{time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
    fh.flush()
    sys.stdout = _TeeUnbuffered(sys.__stdout__, fh)
    sys.stderr = _TeeUnbuffered(sys.__stderr__, fh)
    return log_path
VAL_PIDS_PATH = TRAIN_DIR / "v17_prod_val_pids.txt"
PANEL_NPZ = SCRATCH / "panel_cache.npz"
PANEL_META = SCRATCH / "panel_meta.pkl"
OOF_STACKED = TRAIN_DIR / "v2.0b_oof_stacked_long.csv"
OOF_VAL = TRAIN_DIR / "v2.0b_oof_val_long.csv"
XGB_OUT = REPO_ROOT / "models" / "joint_xgb_v2.0b_oof.pkl"

# Per-year hazard curve emission: hand the raw h_k (k=1..N) to the joint XGB so
# it can integrate the curve itself instead of consuming only the cumulative.
_HK_STEPS = 10
_HK_EVENTS = {"TOP_100_PROSPECT", "MLB_DEBUT", "ESTABLISHED_MLB", "ELITE", "STAR"}


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
def stage_panel(db_path: str, max_draft_year: int,
                partial_seed: int | None = None) -> tuple:
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
        try:
            # v2.0c: TBC org (team-level) rankings. as_of is 'YYYY-01-01'.
            org_rank_rows = conn.execute(
                "SELECT player_id, as_of, org_rank FROM rankings_history "
                "WHERE org_rank IS NOT NULL").fetchall()
        except Exception:
            org_rank_rows = []

    stats_by_pid: dict[str, list[dict]] = {}
    for s in stats_rows:
        d = dict(s)
        stats_by_pid.setdefault(d["player_id"], []).append(d)

    rankings_by_pid: dict[str, list[tuple[int, int, str]]] = {}
    for r in rank_rows:
        rankings_by_pid.setdefault(r[0], []).append((r[1], r[2], r[3]))
    for p in prospects:
        p["_top100_rankings"] = rankings_by_pid.get(p["player_id"], [])

    # v2.0c: attach TBC org rankings as (year, org_rank) tuples, point-in-time.
    org_rankings_by_pid: dict[str, list[tuple[int, int]]] = {}
    for r in org_rank_rows:
        try:
            yr = int(str(r[1])[:4])
        except (ValueError, TypeError):
            continue
        org_rankings_by_pid.setdefault(r[0], []).append((yr, int(r[2])))
    for p in prospects:
        p["_org_rankings"] = org_rankings_by_pid.get(p["player_id"], [])
    n_with_org = sum(1 for p in prospects if p["_org_rankings"])
    print(f"           {n_with_org:,} prospects have >=1 org ranking")

    n_draft = sum(1 for p in prospects if p.get("draft_year") is not None)
    print(f"           {len(prospects):,} prospects "
          f"(drafted {n_draft:,} + IFA {len(prospects)-n_draft:,})")

    plan = []
    pids: list[str] = []
    S_list: list[int] = []
    joined: list[dict] = []
    n_skipped = 0
    n_ifa_capped = 0
    for p in prospects:
        stats = stats_by_pid.get(p["player_id"], [])
        sy = _start_year(p, stats_by_pid)
        if sy is None:
            n_skipped += 1
            continue
        # C3 (v2.1): cap IFA entry year the same way draft_year is SQL-capped.
        # IFAs have no draft_year, so they were never filtered -> post-cutoff
        # IFAs leaked into this panel, which the PROD hazards (train_mask=None)
        # reuse 100%, contaminating the "held-out" walk-forward.
        if p.get("draft_year") is None and sy > max_draft_year:
            n_ifa_capped += 1
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
    print(f"           {n_rows:,} landmark rows  ({n_skipped} skipped, "
          f"{n_ifa_capped:,} IFAs capped at entry<={max_draft_year})")

    X_lm = np.empty((n_rows, N_FEATURES), dtype=np.float32)
    for i in tqdm(range(n_rows), desc="panel", unit="row",
                  mininterval=1.0, smoothing=0.05):
        p, stats, S = plan[i]
        stats_S = partial_for_features(stats, S, p["player_id"], partial_seed)
        X_lm[i, :] = build_windowed_features(p, stats_S, S, milb_only=True)
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
    tmp = PANEL_NPZ.with_suffix(".tmp.npz")  # must end .npz; savez appends it otherwise
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


# ---- Stage 3 helper: one fold with per-snap checkpointing ----
def _score_checkpointed(hazards, prospects_all, stats_by_pid,
                         pid_set, out_csv: Path, partial_dir: Path,
                         max_entry_year: int, observe_through: int,
                         max_offset: int, horizon: int,
                         partial_seed: int | None = None):
    """Per-snap checkpointed scorer. Writes one CSV per snap to
    partial_dir/snap_NNNN.csv, then concats into out_csv. If process
    dies, re-running picks up from the first missing snap."""
    partial_dir.mkdir(parents=True, exist_ok=True)
    cohort = [p for p in prospects_all if p["player_id"] in pid_set]
    enriched: list[dict] = []
    for r in cohort:
        ent = _entry_year_v18(r, stats_by_pid)
        if ent is None or ent > max_entry_year:
            continue
        rc = dict(r)
        rc["_entry_year"] = ent
        rc["_bucket"] = _bucket_of(r)
        enriched.append(rc)
    print(f"           cohort {len(enriched):,} / {len(cohort):,} "
          f"after entry<={max_entry_year}")

    snap_groups: dict[int, list[dict]] = {}
    for r in enriched:
        ent = r["_entry_year"]
        debut = r.get("mlb_debut_year")
        for off in range(0, max_offset + 1):
            snap = ent + off
            if snap > observe_through:
                break
            if debut is not None and debut <= snap:
                continue
            rc = dict(r)
            rc["_snap"] = snap
            rc["_offset"] = off
            snap_groups.setdefault(snap, []).append(rc)

    event_keys = [k for k in hazards if not isinstance(k, str)
                  or k in (ELITE_KEY, STAR_KEY)]
    snap_keys = sorted(snap_groups.keys())
    PLAYER_CHUNK = 100000  # effectively no chunking — process each snap
                            # group in one shot. Bigger batch = better
                            # HistGB OpenMP utilization. Was throttled
                            # during the BSOD-instability window;
                            # SpeedShift fix removed the need.
    for si, snap in enumerate(snap_keys):
        partial_csv = partial_dir / f"snap_{snap}.csv"
        if partial_csv.exists():
            if partial_csv.stat().st_size == 0:
                print(f"  [score] snap={snap} EMPTY partial — "
                      f"deleting and re-scoring", flush=True)
                partial_csv.unlink()
            else:
                print(f"  [score] snap={snap} reusing partial "
                      f"({si+1}/{len(snap_keys)})", flush=True)
                continue
        group_full = snap_groups[snap]
        rows: list[dict] = []
        for chunk_lo in range(0, len(group_full), PLAYER_CHUNK):
            group = group_full[chunk_lo:chunk_lo + PLAYER_CHUNK]
            sub_stats = {
                r["player_id"]: partial_for_features(
                    [s for s in stats_by_pid.get(r["player_id"], [])
                     if (s.get("season_year") or 0) <= snap],
                    snap, r["player_id"], partial_seed)
                for r in group
            }
            out = lm.predict_cumulative_batch_landmark(
                hazards, group, sub_stats,
                current_year=snap, horizon=horizon,
            )
            for i, r in enumerate(group):
                row = {
                    "player_id": r["player_id"],
                    "name": r.get("name"),
                    "draft_year": r.get("draft_year"),
                    "draft_round": r.get("draft_round"),
                    "is_international": int(r.get("is_international") or 0),
                    "bucket": r["_bucket"],
                    "entry_year": r["_entry_year"],
                    "snap_year": snap,
                    "snap_offset": r["_offset"],
                    "years_fwd": observe_through - snap,
                    "mlb_debut_year": r.get("mlb_debut_year"),
                }
                per_ev = {}
                for e in event_keys:
                    ename = _ev_name(e)
                    p_cal = float(out[e][i])
                    trig = _trigger_year(r, e)
                    eligible = int(trig is None or trig > snap)
                    realized = int(trig is not None and trig > snap
                                   and trig <= observe_through)
                    per_ev[ename] = (p_cal, trig, eligible, realized)
                    row[f"p_{ename}"] = p_cal
                    row[f"eligible_{ename}"] = eligible
                    row[f"realized_{ename}"] = realized
                    row[f"trigger_{ename}"] = trig
                    mt = out.get(("mean_t", e))
                    st = out.get(("sd_t", e))
                    if mt is not None:
                        row[f"mean_t_{ename}"] = float(mt[i])
                    if st is not None:
                        row[f"sd_t_{ename}"] = float(st[i])
                    hk = out.get(("haz_k", e))
                    if hk is not None and ename in _HK_EVENTS:
                        for j in range(_HK_STEPS):
                            row[f"hk{j+1}_{ename}"] = float(hk[i, j])
                if "STAR" in per_ev and "ELITE" in per_ev:
                    ps, ts, _, _ = per_ev["STAR"]
                    pe, te, _, _ = per_ev["ELITE"]
                    p_u = 1.0 - (1.0 - ps) * (1.0 - pe)
                    trigs = [t for t in (ts, te) if t is not None]
                    trig_u = min(trigs) if trigs else None
                    elig_u = int(trig_u is None or trig_u > snap)
                    real_u = int(trig_u is not None and trig_u > snap
                                 and trig_u <= observe_through)
                    row["p_STAR_PLUS_ELITE"] = p_u
                    row["eligible_STAR_PLUS_ELITE"] = elig_u
                    row["realized_STAR_PLUS_ELITE"] = real_u
                    row["trigger_STAR_PLUS_ELITE"] = trig_u
                rows.append(row)
            del out
            gc.collect()
        pd.DataFrame(rows).to_csv(partial_csv, index=False)
        del rows
        gc.collect()
        print(f"  [score] snap={snap} group={len(group_full)} wrote "
              f"({si+1}/{len(snap_keys)})", flush=True)

    # Concat all partials
    partials = sorted(partial_dir.glob("snap_*.csv"))
    if not partials:
        out_csv.write_text("")
        return 0
    df = pd.concat([pd.read_csv(f) for f in partials], ignore_index=True)
    for c in df.columns:
        if c.startswith("mean_t_"):
            df[c] = df[c].fillna(15.0)
        elif c.startswith("sd_t_"):
            df[c] = df[c].fillna(0.0)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return len(df)


def run_one_fold(X_lm, pids, S_yrs, joined, stats_by_pid,
                  prospects_all, train_set: set[str], score_set: set[str],
                  out_csv: Path, max_entry_year: int, seed: int,
                  partial_dir: Path, hazards_pkl: Path,
                  hazard_hp: dict | None = None,
                  partial_seed: int | None = None):
    train_mask = np.array([p in train_set for p in pids], dtype=bool)
    print(f"           train_mask: {int(train_mask.sum()):,} / "
          f"{len(pids):,} landmark rows")
    hazards = None
    if hazards_pkl.exists():
        try:
            print(f"           loading cached hazards {hazards_pkl.name}")
            with hazards_pkl.open("rb") as fh:
                hazards = pickle.load(fh)
        except Exception as e:
            print(f"           cached hazards corrupt ({e!s}) — "
                  f"deleting and refitting")
            hazards_pkl.unlink(missing_ok=True)
            hazards = None
    if hazards is None:
        t = time.time()
        hazards = lm.fit_landmark_hazards(
            X_lm, joined, S_yrs, stats_by_pid,
            train_mask=train_mask, seed=seed, verbose=True,
            hazard_hp=hazard_hp,
        )
        print(f"           hazards fit in {time.time()-t:.0f}s, "
              f"saving {hazards_pkl.name}")
        hazards_pkl.parent.mkdir(parents=True, exist_ok=True)
        # Write atomically via tmp + rename so a mid-save crash leaves no
        # corrupt .pkl in place
        tmp = hazards_pkl.with_suffix(".pkl.tmp")
        with tmp.open("wb") as fh:
            pickle.dump(hazards, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(hazards_pkl)
    print(f"           scoring...")
    n = _score_checkpointed(
        hazards, prospects_all, stats_by_pid, score_set, out_csv,
        partial_dir,
        max_entry_year=max_entry_year,
        observe_through=MAX_OBS_YEAR, max_offset=10, horizon=15,
        partial_seed=partial_seed,
    )
    return n, hazards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO_ROOT / "prospects_snapshot.db"))
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-draft-year", type=int, default=2020)
    ap.add_argument("--max-entry-year", type=int, default=2020)
    ap.add_argument("--partial-seed", type=int, default=None,
                    help="Enable partial-final-season training augmentation "
                         "with this seed (down-samples each landmark's current "
                         "season to an in-progress line). Omit for the "
                         "complete-season baseline.")
    ap.add_argument("--tag", default=None,
                    help="Artifact namespace suffix (e.g. 'partial'). When set, "
                         "the scratch dir + stacked/val longs + XGB are written "
                         "side-by-side under this tag, leaving the baseline "
                         "v2.0b artifacts intact.")
    args = ap.parse_args()

    # Namespace all outputs under --tag so a partial run sits beside the
    # complete-season baseline (the panel cache that train_v2_0b_prod_hazards
    # reuses lives in this tagged scratch dir).
    global SCRATCH, PANEL_NPZ, PANEL_META, OOF_STACKED, OOF_VAL, XGB_OUT
    if args.tag:
        SCRATCH = REPO_ROOT / "scratch" / f"v20b_oof_{args.tag}"
        PANEL_NPZ = SCRATCH / "panel_cache.npz"
        PANEL_META = SCRATCH / "panel_meta.pkl"
        OOF_STACKED = TRAIN_DIR / f"v2.0b_{args.tag}_oof_stacked_long.csv"
        OOF_VAL = TRAIN_DIR / f"v2.0b_{args.tag}_oof_val_long.csv"
        XGB_OUT = REPO_ROOT / "models" / f"joint_xgb_v2.0b_{args.tag}_oof.pkl"

    SCRATCH.mkdir(parents=True, exist_ok=True)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _install_logging()
    print("="*78)
    print(f"v2.0b OOF — K={args.k}, seed={args.seed}, "
          f"max_entry={args.max_entry_year}"
          + (f", partial_seed={args.partial_seed}, tag={args.tag}"
             if args.partial_seed is not None or args.tag else ""))
    print(f"log -> {log_path}")
    print("="*78)
    t_start = time.time()

    val_pid_set = {ln.strip() for ln in VAL_PIDS_PATH.read_text().splitlines()
                   if ln.strip()}
    print(f"Val pids held out: {len(val_pid_set):,}")

    # Stage 1
    X_lm, pids, S_yrs, joined, stats_by_pid = stage_panel(
        args.db, args.max_draft_year, partial_seed=args.partial_seed)

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
        partial_dir = SCRATCH / f"fold{j}_partial"
        hazards_pkl = SCRATCH / f"fold{j}_hazards.pkl"
        n, hazards = run_one_fold(
            X_lm, pids, S_yrs, joined, stats_by_pid, prospects_all,
            train_set=train_set, score_set=fold_sets[j], out_csv=out,
            max_entry_year=args.max_entry_year, seed=args.seed,
            partial_dir=partial_dir, hazards_pkl=hazards_pkl,
            partial_seed=args.partial_seed,
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
        val_partial_dir = SCRATCH / "val_partial"
        n = _score_checkpointed(
            last_hazards, prospects_all, stats_by_pid, val_pid_set,
            OOF_VAL, val_partial_dir,
            max_entry_year=args.max_entry_year,
            observe_through=MAX_OBS_YEAR, max_offset=10, horizon=15,
            partial_seed=args.partial_seed,
        )
        print(f"           wrote {n:,} rows")

    # Stage 6 — fit joint XGB
    if XGB_OUT.exists():
        print(f"\n[Stage 6] reusing XGB {XGB_OUT.name}")
    else:
        print(f"\n[Stage 6] fitting joint XGB on stacked OOF -> "
              f"{XGB_OUT.name}")
        tmp = str(XGB_OUT) + ".tmp"
        rc = subprocess.run([
            sys.executable, "-m", "scripts_v17.train.fit_joint_xgb_cond",
            "--fit", str(OOF_STACKED),
            "--val", str(OOF_VAL),
            "--db", args.db,
            # v2.1c: per-horizon censoring is built in (keep (row,h) iff
            # years_fwd>=h) — no --censor-window. h-max 10, publish at h=6.
            "--h-max", "10",
            "--publish-h", "6",
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
