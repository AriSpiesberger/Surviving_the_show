"""Train v2.0b production artifacts: landmark hazards → joint XGBoost.

v2.0b mirrors v2.0 (joint multi-output XGBoost head) but the upstream
hazards are the landmark-trained HistGBTs (v1.18b) instead of the v1.17
contemporaneous ones. We reuse v1.18b's downstream timing model
(time_to_debut_v1.18b_prod.pkl) because it already consumes the
mean_t / sd_t features the landmark inference produces — that pair tested
roughly tied with the v1.18 baseline timing in aggregate MAE and better
at mid-horizon.

Pipeline (three stages):
  1. Train joint XGBoost (fit_joint_xgb_v2.py) on the existing
     v1.18b_landmark_{fit,val}_long.csv files (produced by
     train_v1_18b_prod.py) -> models/joint_xgb_v2.0b_prod.pkl
  2. Score snap=2026 with landmark hazards
     -> results/scored/snap2026_v1.18b_landmark_long.csv
     (cumulative p_<event>, plus mean_t / sd_t / eligible / realized /
     trigger columns the v2.0 builder consumes.)
  3. Build the v2.0b buy list via build_v2.0_buylist.py, pointed at the
     new XGB + landmark snap_long + v1.18b timing.
     -> results/buy_lists/buy_list_v2.0b_FINAL.csv

v2.0 stays untouched as the baseline. Both buy lists exist side by side
for head-to-head.

Usage:
    python -m scripts_v17.train.train_v2_0b_prod

    # Skip XGB retrain (use existing pkl), just rebuild snap + buy list:
    python -m scripts_v17.train.train_v2_0b_prod --skip-xgb

    # Skip snap rescoring (use existing landmark snap_long):
    python -m scripts_v17.train.train_v2_0b_prod --skip-score
"""
from __future__ import annotations

import argparse
import csv
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from prospects.classifier.architectures import landmark_survival as lm
from prospects.classifier.architectures.survival import (
    ELITE_KEY, MAX_OBS_YEAR, STAR_KEY, _trigger_year, _last_active_year,
)
from prospects.storage import ProspectDB
from scripts_v17.train.train_v1_18b_prod import (
    HAZ_OUT as LM_HAZ_OUT, LM_FIT_LONG, LM_VAL_LONG,
)

XGB_OUT = REPO_ROOT / "models" / "joint_xgb_v2.0b_prod.pkl"
TIMING_PKL = REPO_ROOT / "models" / "time_to_debut_v2.0b_prod.pkl"  # M6 (v2.1): was v1.18b (fit+val contaminated)
# v2.1: load the PROD hazards that train_v2_0b_prod_hazards actually writes
# (was importing v1.18b_landmark_prod — a different, older file = pipeline drift).
LM_HAZ_OUT = REPO_ROOT / "models" / "event_classifiers_v2.0b_prod.pkl"
SNAP_LONG = REPO_ROOT / "results" / "scored" / "snap2026_v1.18b_landmark_long.csv"
BUYLIST_ALL = REPO_ROOT / "results" / "buy_lists" / "buy_list_v2.0b_ALL_SCORED.csv"
BUYLIST_FINAL = REPO_ROOT / "results" / "buy_lists" / "buy_list_v2.0b_FINAL.csv"
PRICES_FROM = REPO_ROOT / "data" / "prices_bowman_chrome_auto_v13.csv"

# Per-year hazard curve emission (must match run_v2_0b_oof + fit_joint_xgb_v2).
_HK_STEPS = 10
_HK_EVENTS = {"TOP_100_PROSPECT", "MLB_DEBUT", "ESTABLISHED_MLB", "ELITE", "STAR"}


def _ev_name(e) -> str:
    return e.name if hasattr(e, "name") else str(e).lstrip("_")


def score_snap_with_landmark(
    hazards: dict, prospects: list[dict], stats_by_pid: dict,
    snap_year: int, out_csv: Path, horizon: int = 15,
    verbose: bool = True, exit_grace: int = 2,
) -> int:
    """Score every prospect at one fixed snap year using landmark hazards.

    Filters: drop players who debuted by snap-1 (no longer in the buy
    universe) and players whose entry_year > snap (haven't entered yet).

    Emits the column set the v2.0 buy list builder reads:
      player_id, name, draft_year, draft_round, is_international, bucket,
      entry_year, snap_year, snap_offset, mlb_debut_year,
      p_<event>, eligible_<event>, realized_<event>, trigger_<event>,
      mean_t_<event>, sd_t_<event>.

    The v2.0 builder filters on eligible_MLB_DEBUT == 1 internally, so we
    emit all snap_offsets >= 0 and let the builder pick.
    """
    def entry_year(p):
        dy = p.get("draft_year")
        if dy is not None and int(p.get("is_international") or 0) == 0:
            return int(dy)
        yrs = [int(s["season_year"])
               for s in stats_by_pid.get(p["player_id"], [])
               if s.get("season_year") is not None
               and (s.get("level") or "").upper() != "MLB"]
        if yrs:
            return int(min(yrs))
        if dy is not None:
            return int(dy)
        return None

    def bucket(p):
        if int(p.get("is_international") or 0) == 1:
            return "IFA"
        r = p.get("draft_round")
        if r is None:
            return "IFA"
        r = int(r)
        if r == 1:
            return "R1"
        if r <= 3:
            return "R2-R3"
        if r <= 10:
            return "R4-R10"
        return "R10+"

    cohort = []
    for p in prospects:
        ey = entry_year(p)
        if ey is None or ey > snap_year:
            continue
        debut = p.get("mlb_debut_year")
        if debut is not None and debut <= snap_year - 1:
            # Already debuted before snap year — outside the buy universe.
            continue
        # EXIT event: a never-debuted player whose last MiLB season is more
        # than exit_grace years before the snap has washed out / retired and is
        # no longer at-risk. Mirrors the right-censoring used in hazard training
        # (which drops never-firer rows where label_year > last_active), so the
        # scoring universe matches the training universe. (la is None => brand-
        # new entrant with no stats yet — keep.)
        # M5 (v2.1): compute last_active POINT-IN-TIME (seasons <= snap only).
        # Using full-history last_active leaks future comebacks at historical
        # walk-forward snaps (a guy who returns in 2025 looks "active" at 2022).
        _filt = {p["player_id"]: [s for s in stats_by_pid.get(p["player_id"], [])
                                  if (s.get("season_year") or 0) <= snap_year]}
        la = _last_active_year(p, _filt)
        if la is not None and la < snap_year - exit_grace:
            continue
        rc = dict(p)
        rc["_entry_year"] = ey
        rc["_bucket"] = bucket(p)
        rc["_snap_offset"] = snap_year - ey
        cohort.append(rc)

    if verbose:
        print(f"[score-snap] cohort at snap={snap_year}: {len(cohort):,} "
              f"prospects", flush=True)

    sub_stats = {
        r["player_id"]: [s for s in stats_by_pid.get(r["player_id"], [])
                         if (s.get("season_year") or 0) <= snap_year]
        for r in cohort
    }
    # Chunk the inference. predict_cumulative_batch_landmark builds the
    # 238-feature vector per player in a tight Python loop; doing that for
    # 35k prospects at once fragments heap badly enough to MemoryError on
    # Windows. 2000-player chunks keep peak RSS bounded and let the GC
    # reclaim between batches.
    CHUNK = 2000
    event_keys = [k for k in hazards
                  if (not isinstance(k, str)) or k in (ELITE_KEY, STAR_KEY)]
    # Stitch per-event outputs across chunks.
    n_total = len(cohort)
    out: dict = {}
    import gc
    t0 = time.time()
    for chunk_start in range(0, n_total, CHUNK):
        chunk_end = min(chunk_start + CHUNK, n_total)
        chunk = cohort[chunk_start:chunk_end]
        chunk_stats = {r["player_id"]: sub_stats[r["player_id"]]
                       for r in chunk}
        chunk_out = lm.predict_cumulative_batch_landmark(
            hazards, chunk, chunk_stats,
            current_year=snap_year, horizon=horizon,
        )
        for key, arr in chunk_out.items():
            if key not in out:
                # preserve trailing dims so (n, horizon) haz_k curves survive
                out[key] = np.empty((n_total,) + arr.shape[1:], dtype=arr.dtype)
            out[key][chunk_start:chunk_end] = arr
        gc.collect()
        if verbose:
            print(f"  [score-snap] {chunk_end:,}/{n_total:,} "
                  f"({100*chunk_end/n_total:.0f}%)", flush=True)
    if verbose:
        print(f"  [score-snap] inference in {time.time()-t0:.0f}s",
              flush=True)

    out_rows = []
    for i, r in enumerate(cohort):
        row = {
            "player_id":         r["player_id"],
            "name":              r.get("name"),
            "draft_year":        r.get("draft_year"),
            "draft_round":       r.get("draft_round"),
            "is_international":  int(r.get("is_international") or 0),
            "bucket":            r["_bucket"],
            "entry_year":        r["_entry_year"],
            "snap_year":         snap_year,
            "snap_offset":       r["_snap_offset"],
            "years_fwd":         MAX_OBS_YEAR - snap_year,
            "mlb_debut_year":    r.get("mlb_debut_year"),
        }
        per_ev = {}
        for e in event_keys:
            ename = _ev_name(e)
            p_cal = float(out[e][i])
            trig = _trigger_year(r, e)
            elig = int(trig is None or trig > snap_year)
            real = int(trig is not None and trig > snap_year
                       and trig <= MAX_OBS_YEAR)
            per_ev[ename] = (p_cal, trig, elig, real)
            row[f"p_{ename}"] = p_cal
            row[f"eligible_{ename}"] = elig
            row[f"realized_{ename}"] = real
            row[f"trigger_{ename}"] = trig
            # mean_t / sd_t are NaN when sum_p ≈ 0 (event has essentially
            # zero probability in horizon — timing is undefined). The
            # downstream timing Lasso can't take NaN; substitute the
            # horizon max for mean_t (semantically "if it ever fires
            # it'd be way out") and 0 for sd_t (no mass to spread).
            mt = out.get(("mean_t", e))
            st = out.get(("sd_t", e))
            if mt is not None:
                v = float(mt[i])
                row[f"mean_t_{ename}"] = float(horizon) if np.isnan(v) else v
            if st is not None:
                v = float(st[i])
                row[f"sd_t_{ename}"] = 0.0 if np.isnan(v) else v
            hk = out.get(("haz_k", e))
            if hk is not None and ename in _HK_EVENTS:
                for j in range(_HK_STEPS):
                    row[f"hk{j+1}_{ename}"] = float(hk[i, j])
        if "STAR" in per_ev and "ELITE" in per_ev:
            ps, ts, _, _ = per_ev["STAR"]
            pe, te, _, _ = per_ev["ELITE"]
            # m1 (v2.1): ELITE's trigger components (all_star_three, major_award)
            # are a SUBSET of STAR's (all_star_once + all_star_three + award), so
            # STAR ∪ ELITE == STAR. The old 1-(1-ps)(1-pe) double-counted the
            # shared events. The union probability is just P(STAR).
            p_u = ps
            trigs = [t for t in (ts, te) if t is not None]
            trig_u = min(trigs) if trigs else None
            elig_u = int(trig_u is None or trig_u > snap_year)
            real_u = int(trig_u is not None and trig_u > snap_year
                         and trig_u <= MAX_OBS_YEAR)
            row["p_STAR_PLUS_ELITE"] = p_u
            row["eligible_STAR_PLUS_ELITE"] = elig_u
            row["realized_STAR_PLUS_ELITE"] = real_u
            row["trigger_STAR_PLUS_ELITE"] = trig_u
        out_rows.append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fnames = list(out_rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fnames)
        w.writeheader()
        w.writerows(out_rows)
    return len(out_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO_ROOT / "prospects_snapshot.db"))
    ap.add_argument("--snap-year", type=int, default=2026)
    ap.add_argument("--skip-xgb", action="store_true")
    ap.add_argument("--skip-score", action="store_true")
    ap.add_argument("--skip-buylist", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    print("=" * 78)
    print("v2.0b PROD TRAIN — landmark hazards + joint XGBoost")
    print("=" * 78)

    # --- Stage 1: joint XGBoost on OOF stacked (HONEST) ---
    if args.skip_xgb and XGB_OUT.exists():
        print(f"[1/3] skip-xgb: using existing {XGB_OUT.name}", flush=True)
    else:
        # HONEST: train the XGB on OUT-OF-FOLD hazard features. The old path
        # trained on v1.18b_landmark_all_long, whose hazard probs come from
        # hazards fit on 100% of players then scoring those same players —
        # i.e. the features had already seen each player's label. That leaks
        # into the XGB and inflates the rare-event (est/star) heads. The OOF
        # stacked CSV scores every row with a hazard model that never trained
        # on it, so the XGB learns the real "hazard prob -> outcome" mapping.
        oof_fit = REPO_ROOT / "results" / "training" / "v2.0b_oof_stacked_long.csv"
        oof_val = REPO_ROOT / "results" / "training" / "v2.0b_oof_val_long.csv"
        if not oof_fit.exists() or not oof_val.exists():
            sys.exit("FATAL: missing OOF stacked/val long CSVs. Run "
                     "run_v2_0b_oof first (produces honest out-of-fold longs).")
        tmp = str(XGB_OUT) + ".tmp"
        print(f"\n[1/3] Training joint XGBoost on OOF stacked (HONEST, no "
              f"in-sample hazard leakage) -> {XGB_OUT.name}", flush=True)
        rc = subprocess.run(
            [sys.executable, "-m", "scripts_v17.train.fit_joint_xgb_cond",
             "--fit", str(oof_fit),
             "--val", str(oof_val),
             "--db",  args.db,
             # v2.1c conditional refinement: per-horizon censoring built in;
             # train h in 1..10, publish the buy list at h=6.
             "--h-max", "10",
             "--publish-h", "6",
             "--out", tmp],
            cwd=REPO_ROOT,
        ).returncode
        if rc != 0:
            sys.exit(rc)
        shutil.move(tmp, XGB_OUT)
        print(f"  wrote {XGB_OUT}", flush=True)

    # --- Stage 2: score snap=2026 with landmark hazards ---
    if args.skip_score and SNAP_LONG.exists():
        print(f"\n[2/3] skip-score: using existing {SNAP_LONG.name}",
              flush=True)
    else:
        print(f"\n[2/3] Loading landmark hazards + DB rows for snap="
              f"{args.snap_year}", flush=True)
        with LM_HAZ_OUT.open("rb") as f:
            hazards = pickle.load(f)
        db = ProspectDB(args.db)
        with db._connect() as conn:
            prospects = [dict(r) for r in conn.execute("""
                SELECT p.*, o.mlb_debut_year, o.year_established_mlb,
                       o.year_top_100, o.year_top_25,
                       o.year_all_star_once, o.year_all_star_three,
                       o.year_major_award, o.year_hof_trajectory,
                       o.events_json, o.final_mlb_year
                FROM prospects p
                LEFT JOIN career_outcomes o ON o.player_id = p.player_id
                WHERE (p.draft_year IS NOT NULL
                        AND p.draft_year BETWEEN 2010 AND ?)
                   OR COALESCE(p.is_international, 0) = 1
            """, (args.snap_year - 1,)).fetchall()]
            stats_rows = conn.execute(
                "SELECT * FROM season_stats").fetchall()
        stats_by_pid: dict[str, list] = {}
        for s in stats_rows:
            d = dict(s)
            stats_by_pid.setdefault(d["player_id"], []).append(d)
        print(f"  {len(prospects):,} candidate prospects, "
              f"{len(stats_by_pid):,} with season_stats", flush=True)
        t0 = time.time()
        n = score_snap_with_landmark(
            hazards, prospects, stats_by_pid,
            snap_year=args.snap_year, out_csv=SNAP_LONG,
            verbose=True,
        )
        print(f"  wrote {n:,} rows in {time.time()-t0:.0f}s -> {SNAP_LONG}",
              flush=True)

    # --- Stage 3: build the v2.0b buy list ---
    if args.skip_buylist:
        print(f"\n[3/3] skip-buylist: stopping after snap scoring",
              flush=True)
    else:
        print(f"\n[3/3] Building v2.0b buy list", flush=True)
        rc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts_v17" /
                                 "buylist" / "build_v2.0_buylist.py"),
             "--long", str(SNAP_LONG),
             "--xgb", str(XGB_OUT),
             "--timing", str(TIMING_PKL),
             "--prices", str(PRICES_FROM),
             "--db", args.db,
             "--out-all", str(BUYLIST_ALL),
             "--out-final", str(BUYLIST_FINAL)],
            cwd=REPO_ROOT,
        ).returncode
        if rc != 0:
            sys.exit(rc)
        print(f"  wrote {BUYLIST_FINAL}", flush=True)

    print(f"\n=== v2.0b TRAIN COMPLETE in "
          f"{(time.time()-t_start)/60:.1f} min ===")
    print(f"  xgb:        {XGB_OUT}")
    print(f"  snap_long:  {SNAP_LONG}")
    print(f"  buy list:   {BUYLIST_FINAL}")


if __name__ == "__main__":
    main()
