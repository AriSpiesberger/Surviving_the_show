"""Weekly v2.0b full retrain + scoring + buy list build.

v2.0b = landmark hazards + joint XGBoost downstream. Replaces the prior
v1.18 / v2.0 contemporaneous pipeline. Held-out validation showed AU-PR
gains of +0.30 to +0.83 on rare events vs v2.0 (see
results/v20*_landmark*/report.txt).

Pipeline (default order):
  0a. PANEL/HAZARDS (v1.18b): rebuild panel -> train landmark HistGBT
      hazards (k-as-feature) -> score fit/val slices -> refit
      v1.18b L1-logistic bundle + time-to-debut (with mean_t/sd_t).
  0b. JOINT XGB (v2.0b): retrain the multi-output XGBoost downstream
      on landmark hazard outputs.
  1.  SCORE:   landmark inference at snap=2026
              -> results/scored/snap2026_v1.18b_landmark_long.csv
  2.  BUYLIST: scripts_v17/buylist/build_v2.0_buylist.py with v2.0b
              artifacts and v1.18b timing
              -> results/buy_lists/buy_list_v2.0b_FINAL.csv
  3.  COMPS:   prospects.deploy.debut_comps (eBay refresh, fail-soft)

The retrain block is the slow part (~60-75 min total: panel ~25, hazards
~10, score+buylist the rest). It's pure-Python orchestrated so the Windows
Task Scheduler invocation doesn't need bash.

Idempotent in degraded modes — score_panel_v17 caches per-chunk outputs in
scratch/v17_score_panel so reruns only fill gaps.

Usage:
    # Full retrain + score + buylist (the weekly cron):
    python -m prospects.deploy.weekly_score --season 2026

    # Skip retrain; just rescore with existing models (saves ~35 min):
    python -m prospects.deploy.weekly_score --season 2026 --skip-retrain

    # Force chunk-cache wipe before rescoring (when hazards change):
    python -m prospects.deploy.weekly_score --season 2026 --fresh

    # Run scoring only (skip retrain + buylist):
    python -m prospects.deploy.weekly_score --season 2026 \\
        --skip-retrain --score-only

    # Run buy-list rebuild only:
    python -m prospects.deploy.weekly_score --season 2026 \\
        --skip-retrain --buylist-only

Exit codes:
    0 = success
    1 = scoring failed
    2 = buylist build failed
    3 = required artifacts missing
    4 = retrain failed
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Required artifacts for the v2.0b production pipeline (landmark hazards +
# joint XGBoost). The retrain block produces all of these; check_artifacts
# runs AFTER retrain so a clean run never trips it.
REQUIRED = [
    # Landmark hazards (v1.18b) — the upstream
    "models/event_classifiers_v1.18b_landmark_prod.pkl",
    # v1.18b downstream — used for time-to-debut. The L1 bundle is now
    # superseded by the v2.0b joint XGB but the bundle pkl is still needed
    # by fit_time_to_debut_v18b for the p_debut_lasso feature.
    "models/lasso_logits_v1.18b_prod.pkl",
    "models/time_to_debut_v1.18b_prod.pkl",
    # v2.0b joint XGBoost — the actual scoring head
    "models/joint_xgb_v2.0b_prod.pkl",
    # Shared infra
    "models/player_position_from_stats.csv",
    "panels/panel_v1.17.npz",
    "prospects_snapshot.db",
    "scripts_v17/buylist/build_v2.0_buylist.py",
]

# Retrain pipeline knobs.
RETRAIN_PARTITIONS = 64
RETRAIN_PART_RETRIES = 7
RETRAIN_PANEL_NAME = "panels/panel_v1.17.npz"


def check_artifacts() -> list[str]:
    missing = []
    for rel in REQUIRED:
        if not (REPO_ROOT / rel).exists():
            missing.append(rel)
    return missing


def run_step(label: str, cmd: list[str], cwd: Path,
             quiet: bool = False) -> int:
    """Run a subprocess, stream its output. Return exit code.

    `quiet=True` suppresses child stdout/stderr (used for noisy retried-
    partition workers; we summarize success/failure at the call site).
    """
    if not quiet:
        print(f"\n{'='*70}\n[{label}] {' '.join(cmd)}\n{'='*70}", flush=True)
    proc = subprocess.run(
        cmd, cwd=cwd,
        env={
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONIOENCODING": "utf-8",
        },
        stdout=subprocess.DEVNULL if quiet else None,
        stderr=subprocess.DEVNULL if quiet else None,
    )
    if not quiet:
        print(f"\n[{label}] exit={proc.returncode}", flush=True)
    return proc.returncode


# ---------------- retrain orchestration ----------------

def _rebuild_position_lookup() -> int:
    """Rebuild player_position_from_stats.csv from prospects_snapshot.db's
    season_stats. Quick (~1s). Returns 0 on success."""
    import sqlite3
    import pandas as pd  # local import keeps weekly_score importable when
    # pandas is missing (e.g. running --help in a stripped env).
    print("\n[retrain/positions] rebuilding player_position_from_stats.csv",
          flush=True)
    c = sqlite3.connect(REPO_ROOT / "prospects_snapshot.db")
    try:
        df = pd.read_sql(
            "SELECT player_id, primary_position, pa, ip FROM season_stats "
            "WHERE primary_position IS NOT NULL", c)
    finally:
        c.close()
    df["weight"] = df["pa"].fillna(0) + df["ip"].fillna(0) * 3
    m = (df.groupby(["player_id", "primary_position"])["weight"]
           .sum().reset_index()
           .sort_values("weight", ascending=False)
           .drop_duplicates("player_id", keep="first"))
    out = m[["player_id", "primary_position"]].rename(
        columns={"primary_position": "pos_seasonstats"})
    out_root = REPO_ROOT / "player_position_from_stats.csv"
    out_models = REPO_ROOT / "models" / "player_position_from_stats.csv"
    out.to_csv(out_root, index=False)
    out.to_csv(out_models, index=False)
    print(f"  wrote {len(m):,} player positions (root + models/)", flush=True)
    return 0


def _build_panel(fresh: bool) -> int:
    """Build panel_v1.17.npz with N partitions and per-partition retries.

    Mirrors scripts_v17/train/build_panel_v17.sh in pure Python so we don't
    depend on bash being in PATH. Skips already-completed partitions (the
    partition writer's checkpointing) so reruns just fill gaps.

    Returns 0 on success, non-zero if any partition fails after retries.
    """
    panel_path = REPO_ROOT / RETRAIN_PANEL_NAME
    if fresh:
        # Wipe partial state so we build clean. Keep .npz files matched to a
        # specific data version; a fresh data pull means a fresh panel.
        for ext in (".npz", ".joined.pkl"):
            f = panel_path.with_suffix(ext)
            if f.exists():
                print(f"[retrain/panel] wiping {f.name}", flush=True)
                f.unlink()
        for p in REPO_ROOT.glob(f"{panel_path.stem}.part*.npz"):
            p.unlink()
    print(f"\n[retrain/panel] building {RETRAIN_PANEL_NAME} "
          f"({RETRAIN_PARTITIONS} partitions, up to "
          f"{RETRAIN_PART_RETRIES} retries each)", flush=True)
    failed: list[int] = []
    for part in range(RETRAIN_PARTITIONS):
        part_file = REPO_ROOT / f"{panel_path.stem}.part{part}.npz"
        if part_file.exists():
            continue
        ok = False
        for attempt in range(1, RETRAIN_PART_RETRIES + 1):
            rc = run_step(
                f"panel.part{part}.try{attempt}",
                [sys.executable, "-m", "prospects.classifier.build_panel",
                 "--out", str(panel_path),
                 "--max-draft-year", "2025", "--max-year", "2026",
                 "--n-partitions", str(RETRAIN_PARTITIONS),
                 "--partition", str(part)],
                REPO_ROOT, quiet=True,
            )
            if part_file.exists():
                ok = True
                print(f"  part {part:>2d}/{RETRAIN_PARTITIONS}: OK "
                      f"(try {attempt})", flush=True)
                break
        if not ok:
            print(f"  part {part:>2d}/{RETRAIN_PARTITIONS}: FAILED after "
                  f"{RETRAIN_PART_RETRIES} tries", flush=True)
            failed.append(part)
    if failed:
        print(f"[retrain/panel] {len(failed)} partition(s) failed: "
              f"{failed}", flush=True)
        return 1
    # Merge
    rc = run_step("panel.merge",
                  [sys.executable, "-m", "prospects.classifier.build_panel",
                   "--out", str(panel_path),
                   "--n-partitions", str(RETRAIN_PARTITIONS), "--merge"],
                  REPO_ROOT)
    if rc != 0:
        return rc
    if not panel_path.exists():
        print("[retrain/panel] merge succeeded but panel file missing",
              flush=True)
        return 1
    return 0


def _train_hazards_and_models() -> int:
    """Hazards (100% panel) → Beta calibrators → score fit/val slices →
    refit lasso + model_b. Mirrors scripts_v17/train/train_prod_v17.sh in
    pure Python.

    On any non-zero exit code from a step, returns that code so the caller
    can abort. We do NOT continue past a hazard/calibration failure — the
    later stages would consume corrupt inputs."""
    panel_path = REPO_ROOT / RETRAIN_PANEL_NAME
    haz = REPO_ROOT / "models" / "event_classifiers_v1.17_prod.pkl"
    cal = REPO_ROOT / "models" / "event_classifiers_v1.17_prod_calibrated.pkl"

    rc = run_step("retrain/hazards",
                  [sys.executable, "-m", "prospects.classifier.train_full_v14d",
                   "--panel", str(panel_path),
                   "--lasso-fit-frac", "0.0", "--lasso-val-frac", "0.0",
                   "--seed", "42", "--out", str(haz)],
                  REPO_ROOT)
    if rc != 0:
        return rc

    # Recompute the seed=42 fit/val pid lists. The prod calibrator and the
    # honest refit both consume these. (We regenerate them every retrain
    # because the panel's pid set can shift when new prospects enter.)
    print("\n[retrain/pids] regenerating seed=42 fit/val pid lists",
          flush=True)
    import numpy as np
    with np.load(panel_path, allow_pickle=True) as d:
        pids = sorted(set(d["pids"].tolist()))
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(pids))
    n_fit = int(round(0.10 * len(pids)))
    n_val = int(round(0.10 * len(pids)))
    train_dir = REPO_ROOT / "results" / "training"
    train_dir.mkdir(parents=True, exist_ok=True)
    fit_pids_path = train_dir / "v17_prod_fit_pids.txt"
    val_pids_path = train_dir / "v17_prod_val_pids.txt"
    fit_pids_path.write_text(
        "\n".join(pids[i] for i in perm[:n_fit]) + "\n")
    val_pids_path.write_text(
        "\n".join(pids[i] for i in perm[n_fit:n_fit + n_val]) + "\n")
    print(f"  fit: {n_fit:,}  val: {n_val:,}", flush=True)

    rc = run_step("retrain/calibrators",
                  [sys.executable, "-m",
                   "prospects.classifier.fit_hazard_calibrators",
                   "--model", str(haz), "--panel", str(panel_path),
                   "--players-file", str(fit_pids_path),
                   "--out", str(cal)],
                  REPO_ROOT)
    if rc != 0:
        return rc

    # Score fit + val slices using the calibrated model. Chunked + retried
    # the same way as the buy-sheet scoring (score_v14c_cal_slice_raw is
    # the segfault-prone one; we feed it 100-player chunks).
    # NB: we are inside _train_hazards_and_models, called only after the
    # hazards were freshly trained. Stale fit/val long CSVs and stale chunk
    # outputs from a prior week's hazards MUST be wiped here, otherwise the
    # downstream lasso/model_b refit happily reuses last week's hazard
    # outputs and the buy list ends up with today's hazards * last week's
    # reweighting. (Caching is fine for partial-run recovery but not across
    # a hazard refit, which is the case here.)
    for slice_name, plist in (
        ("fit", fit_pids_path), ("val", val_pids_path),
    ):
        out_csv = train_dir / f"v1.17_prod_{slice_name}_long.csv"
        chunk_dir = REPO_ROOT / "scratch" / f"v17_prod_{slice_name}_out"
        if out_csv.exists():
            print(f"[retrain/score-{slice_name}] wiping stale {out_csv.name} "
                  f"(hazards just retrained)", flush=True)
            out_csv.unlink()
        if chunk_dir.exists():
            shutil.rmtree(chunk_dir)
        chunk_dir.mkdir(parents=True, exist_ok=True)
        rc = _chunk_score(plist, chunk_dir, slice_name, cal, panel_path)
        if rc != 0:
            return rc
        # Merge chunk outputs
        import pandas as pd
        files = sorted(chunk_dir.glob("*.csv"))
        if not files:
            print(f"[retrain/score-{slice_name}] no chunk outputs",
                  flush=True)
            return 1
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        df.to_csv(out_csv, index=False)
        print(f"[retrain/score-{slice_name}] wrote {out_csv.name} "
              f"({len(df):,} rows)", flush=True)

    # v1.18b retrain: fit per-event L1-logistic bundle + time-to-debut
    # regression with mean_t/sd_t features on landmark hazard outputs.
    # The bundle is needed by fit_time_to_debut_v18b (which takes
    # p_debut_lasso as a feature); the joint XGB downstream (step below)
    # supersedes the bundle for scoring.
    rc = run_step("retrain/v1.18b",
                  [sys.executable, "-m",
                   "scripts_v17.train.fit_lasso_logits_v18",
                   "--fit", str(train_dir / "v1.18b_landmark_fit_long.csv"),
                   "--val", str(train_dir / "v1.18b_landmark_val_long.csv"),
                   "--db", str(REPO_ROOT / "prospects_snapshot.db"),
                   "--out", str(REPO_ROOT / "models" /
                                 "lasso_logits_v1.18b_prod.pkl")],
                  REPO_ROOT)
    if rc != 0:
        return rc
    rc = run_step("retrain/v1.18b-timing",
                  [sys.executable, "-m",
                   "scripts_v17.train.fit_time_to_debut_v18b",
                   "--fit", str(train_dir / "v1.18b_landmark_fit_long.csv"),
                   "--val", str(train_dir / "v1.18b_landmark_val_long.csv"),
                   "--db", str(REPO_ROOT / "prospects_snapshot.db"),
                   "--bundle", str(REPO_ROOT / "models" /
                                    "lasso_logits_v1.18b_prod.pkl"),
                   "--include-p-debut",
                   "--out", str(REPO_ROOT / "models" /
                                 "time_to_debut_v1.18b_prod.pkl")],
                  REPO_ROOT)
    if rc != 0:
        return rc

    # v2.0b joint XGB: train the multi-output XGBoost head on the landmark
    # hazard outputs. This is the actual scoring head used by the buy list.
    rc = run_step("retrain/v2.0b-xgb",
                  [sys.executable, "-m",
                   "scripts_v17.train.fit_joint_xgb_v2",
                   "--fit", str(train_dir / "v1.18b_landmark_fit_long.csv"),
                   "--val", str(train_dir / "v1.18b_landmark_val_long.csv"),
                   "--db", str(REPO_ROOT / "prospects_snapshot.db"),
                   "--out", str(REPO_ROOT / "models" /
                                 "joint_xgb_v2.0b_prod.pkl")],
                  REPO_ROOT)
    if rc != 0:
        return rc
    return 0


def _chunk_score(pids_file: Path, chunk_dir: Path, label: str,
                 cal_model: Path, panel: Path) -> int:
    """Split pids_file into 100-pid chunks and score each via
    score_v14c_cal_slice_raw with up to 3 retries (segfault-resilient)."""
    pids = [p for p in pids_file.read_text().splitlines() if p.strip()]
    chunk_size = 100
    for i in range(0, len(pids), chunk_size):
        chunk_idx = i // chunk_size
        chunk_file = chunk_dir / f"chunk_{chunk_idx:04d}.txt"
        out_file = chunk_dir / f"chunk_{chunk_idx:04d}.csv"
        if out_file.exists():
            continue
        chunk_file.write_text("\n".join(pids[i:i + chunk_size]) + "\n")
        ok = False
        for attempt in range(1, 4):
            rc = run_step(
                f"retrain/score-{label}.chunk{chunk_idx}.try{attempt}",
                [sys.executable, "-m",
                 "prospects.classifier.score_v14c_cal_slice_raw",
                 "--model", str(cal_model), "--panel", str(panel),
                 "--players-file", str(chunk_file),
                 "--max-entry-year", "2020",
                 "--observe-through", "2026", "--max-offset", "10",
                 "--out", str(out_file)],
                REPO_ROOT, quiet=True,
            )
            if out_file.exists():
                ok = True
                break
        if not ok:
            print(f"[retrain/score-{label}] chunk {chunk_idx} FAILED after "
                  f"3 tries", flush=True)
            return 1
    return 0


def run_retrain(fresh: bool) -> int:
    """Full retrain orchestration. Returns 0 on success, non-zero on first
    failed step.

    Delegates the heavy lifting to two orchestrator scripts that already
    encode the v2.0b production pipeline:

      - scripts_v17.train.train_v1_18b_prod : panel rebuild + landmark
        hazards + score fit/val + downstream lasso/timing refit.
      - scripts_v17.train.train_v2_0b_prod : joint XGB on landmark longs
        + snap=2026 scoring + buy list build.

    The legacy inline _rebuild_position_lookup / _build_panel /
    _train_hazards_and_models helpers are still defined above for ad-hoc
    use (and so the panel/lookup work shares one impl with the v1.18b
    orchestrator, which calls back into them via direct import)."""
    print(f"\n{'#'*70}\n# WEEKLY RETRAIN (v2.0b)\n{'#'*70}", flush=True)
    # Stage A: v1.18b — panel + landmark hazards + fit/val scoring +
    # downstream bundle/timing. Runs end-to-end as its own subprocess so
    # a failure isolates cleanly.
    v18b_cmd = [sys.executable, "-m", "scripts_v17.train.train_v1_18b_prod"]
    if not fresh:
        # If panel exists from a recent build, skip-hazards lets us reuse it.
        # The orchestrator's --skip-hazards reuses the .pkl. For a true clean
        # weekly retrain we want fresh, so omit the flag.
        pass
    rc = run_step("retrain/v1.18b", v18b_cmd, REPO_ROOT)
    if rc != 0:
        return rc

    # Stage B: v2.0b — joint XGB on the landmark longs + snap=2026 scoring
    # + buy list. Produces results/buy_lists/buy_list_v2.0b_FINAL.csv as
    # the prod artifact.
    rc = run_step("retrain/v2.0b",
                  [sys.executable, "-m", "scripts_v17.train.train_v2_0b_prod"],
                  REPO_ROOT)
    if rc != 0:
        return rc

    print(f"\n[retrain] OK\n", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True,
                    help="snap year (e.g. 2026)")
    ap.add_argument("--fresh", action="store_true",
                    help="wipe scratch/v17_score_panel before scoring (forces "
                         "full re-score; use after hazard model updates)")
    ap.add_argument("--skip-retrain", action="store_true",
                    help="skip the retrain block (panel + hazards + lasso + "
                         "model_b). Use for ad-hoc rescoring against existing "
                         "models.")
    ap.add_argument("--score-only", action="store_true",
                    help="run only the scoring step (implies --skip-retrain)")
    ap.add_argument("--buylist-only", action="store_true",
                    help="run only the buylist build step (implies "
                         "--skip-retrain)")
    ap.add_argument("--keep-chunks", action="store_true",
                    help="don't clean intermediate per-chunk CSVs after merge")
    ap.add_argument("--skip-debut-comps", action="store_true",
                    help="skip the trailing debut_comps eBay refresh")
    args = ap.parse_args()

    if args.score_only and args.buylist_only:
        sys.exit("--score-only and --buylist-only are mutually exclusive")

    print(f"=== weekly_score for season={args.season} ===")

    # Refresh prospects_snapshot.db from the live prospects.db so the buy-list
    # build sees today's MiLB stats. daily_data.py writes to prospects.db;
    # build_v17_buylist.py reads from prospects_snapshot.db. This bridge keeps
    # the weekly buy list current. Idempotent.
    live_db = REPO_ROOT / "prospects.db"
    snap_db = REPO_ROOT / "prospects_snapshot.db"
    if live_db.exists():
        print(f"[snapshot] copying {live_db.name} -> {snap_db.name}")
        shutil.copy2(live_db, snap_db)
    else:
        print(f"[snapshot] WARN: {live_db.name} not found; "
              f"using existing {snap_db.name} if present")

    # Step 0: retrain (skipped for ad-hoc rescoring / partial runs)
    skip_retrain = (args.skip_retrain or args.score_only or args.buylist_only)
    if not skip_retrain:
        rc = run_retrain(fresh=args.fresh)
        if rc != 0:
            print(f"\nFATAL: retrain failed (rc={rc})", flush=True)
            sys.exit(4)
    else:
        print("[retrain] skipped (--skip-retrain or --score-only/"
              "--buylist-only set)", flush=True)

    missing = check_artifacts()
    if missing:
        print(f"\nFATAL: required artifacts missing:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(3)

    scored_dir = REPO_ROOT / "results" / "scored"
    scored_dir.mkdir(parents=True, exist_ok=True)
    snap_long = scored_dir / f"snap{args.season}_v1.18b_landmark_long.csv"

    # Step 1+2: snap=2026 landmark scoring + v2.0b buy list. When the full
    # retrain ran, train_v2_0b_prod already did this — so on a full-retrain
    # run these are no-ops (the orchestrator's --skip-xgb path is the
    # ad-hoc rescoring entry point used by --skip-retrain modes).
    if skip_retrain and not args.buylist_only:
        # Rescore snap with EXISTING landmark hazards + rebuild buy list.
        # --skip-xgb keeps the prod XGB pkl; --skip-buylist is set when the
        # caller really only wants the snap_long.
        cmd = [sys.executable, "-m",
               "scripts_v17.train.train_v2_0b_prod",
               "--skip-xgb"]
        if args.score_only:
            cmd.append("--skip-buylist")
        rc = run_step("score+buylist", cmd, REPO_ROOT)
        if rc != 0:
            sys.exit(1)
    elif skip_retrain and args.buylist_only:
        # Snap_long must already exist; just rebuild the buy list.
        if not snap_long.exists():
            print(f"FATAL: need {snap_long.name} but it doesn't exist; "
                  f"run without --buylist-only first")
            sys.exit(3)
        rc = run_step("buylist", [
            sys.executable,
            str(REPO_ROOT / "scripts_v17" / "buylist" /
                "build_v2.0_buylist.py"),
            "--long", str(snap_long),
            "--xgb", str(REPO_ROOT / "models" /
                          "joint_xgb_v2.0b_prod.pkl"),
            "--timing", str(REPO_ROOT / "models" /
                             "time_to_debut_v1.18b_prod.pkl"),
            "--out-all", str(REPO_ROOT / "results" / "buy_lists" /
                              "buy_list_v2.0b_ALL_SCORED.csv"),
            "--out-final", str(REPO_ROOT / "results" / "buy_lists" /
                                "buy_list_v2.0b_FINAL.csv"),
        ], REPO_ROOT)
        if rc != 0:
            sys.exit(2)

    # Step 3: refresh debut comps (eBay prices for non-R1 current-season
    # debutants). Fail-soft: a comp ingestion error must not block the buy
    # list update.
    if not args.skip_debut_comps and not args.score_only:
        rc = run_step("debut_comps", [
            sys.executable, "-m", "prospects.deploy.debut_comps",
            "--year", str(args.season),
        ], REPO_ROOT)
        if rc != 0:
            print(f"[debut_comps] WARN: exited {rc}; buy list still valid",
                  flush=True)

    bl_dir = REPO_ROOT / "results" / "buy_lists"
    print(f"\n=== weekly_score season={args.season} OK ===")
    print(f"  snap long file: {snap_long}")
    print(f"  buy list:       {bl_dir / 'buy_list_v2.0b_FINAL.csv'}")
    print(f"  all scored:     {bl_dir / 'buy_list_v2.0b_ALL_SCORED.csv'}")


if __name__ == "__main__":
    main()
