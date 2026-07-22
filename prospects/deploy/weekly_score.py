"""Weekly v2.0b full retrain + scoring + buy list build.

v2.0b = landmark hazards + joint XGBoost downstream. Replaces the prior
v1.18 / v2.0 contemporaneous pipeline. Held-out validation showed AU-PR
gains of +0.30 to +0.83 on rare events vs v2.0 (see
results/v20*_landmark*/report.txt).

Pipeline (default order):
  0a. PANEL/HAZARDS (v1.18b): rebuild panel -> train landmark HistGBT
      hazards (k-as-feature) -> score fit/val slices -> refit
      v1.18b L1-logistic bundle + time-to-debut (with mean_t/sd_t).
  0b. JOINT XGB (v2.0b): retrain the conditional multi-output XGBoost
      downstream on landmark hazard outputs, then score snap=2026 and
      build the buy list.
  1.  COMPS:   prospects.deploy.debut_comps (eBay refresh, fail-soft)

Steps 0a/0b are delegated wholesale to the two orchestrator scripts (see
run_retrain); this module only sequences them, bridges the database and
checks the resulting artifacts. The retrain block is the slow part
(~60-75 min total). It's pure-Python orchestrated so the Windows Task
Scheduler invocation doesn't need bash.

Usage:
    # Full retrain + score + buylist (the weekly cron):
    python -m prospects.deploy.weekly_score --season 2026

    # Skip retrain; just rescore with existing models (saves ~35 min):
    python -m prospects.deploy.weekly_score --season 2026 --skip-retrain

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
# conditional joint XGBoost). check_artifacts runs AFTER retrain, so a clean
# run never trips it — anything listed here that retrain does NOT produce is
# a real external dependency and is called out as such.
REQUIRED = [
    # --- Stage A (scripts_v17.train.train_v1_18b_prod) ---
    # Landmark hazards, the upstream of everything downstream.
    "models/event_classifiers_v1.18b_landmark_prod.pkl",
    # The L1 bundle is superseded as a scoring head by the joint XGB, but
    # fit_time_to_debut_v18b still reads it for the p_debut_lasso feature.
    "models/lasso_logits_v1.18b_prod.pkl",
    "models/time_to_debut_v1.18b_prod.pkl",
    # --- Stage B (scripts_v17.train.train_v2_0b_prod) ---
    # The conditional joint XGB (fit_joint_xgb_cond) is the actual scoring
    # head, plus its retrained timing model.
    "models/joint_xgb_v2.0b_prod.pkl",
    "models/time_to_debut_v2.0b_prod.pkl",
    # NOT produced by run_retrain: train_v2_0b_prod READS these 100%-panel
    # prod hazards, but only train_v2_0b_prod_hazards writes them and that
    # script is hand-run. Listed so a missing/stale file fails loudly here
    # rather than silently scoring against whatever is on disk.
    "models/event_classifiers_v2.0b_prod.pkl",
    # --- Shared infra ---
    "models/player_position_from_stats.csv",
    "prospects_snapshot.db",
    "scripts_v17/buylist/build_v2.0_buylist.py",
]


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


def run_retrain() -> int:
    """Full retrain orchestration. Returns 0 on success, non-zero on first
    failed step.

    Delegates the heavy lifting to two orchestrator scripts that already
    encode the v2.0b production pipeline:

      - scripts_v17.train.train_v1_18b_prod : panel rebuild + landmark
        hazards + score fit/val + downstream lasso/timing refit.
      - scripts_v17.train.train_v2_0b_prod : conditional joint XGB
        (fit_joint_xgb_cond) on landmark longs + snap=2026 scoring + buy
        list build."""
    print(f"\n{'#'*70}\n# WEEKLY RETRAIN (v2.0b)\n{'#'*70}", flush=True)
    # Stage A: v1.18b — panel + landmark hazards + fit/val scoring +
    # downstream bundle/timing. Runs end-to-end as its own subprocess so
    # a failure isolates cleanly.
    # Always a clean rebuild: we omit --skip-hazards so the orchestrator
    # retrains rather than reusing the existing .pkl.
    v18b_cmd = [sys.executable, "-m", "scripts_v17.train.train_v1_18b_prod"]
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
    ap.add_argument("--skip-retrain", action="store_true",
                    help="skip the retrain block (panel + hazards + "
                         "downstream refit). Use for ad-hoc rescoring "
                         "against existing models.")
    ap.add_argument("--score-only", action="store_true",
                    help="run only the scoring step (implies --skip-retrain)")
    ap.add_argument("--buylist-only", action="store_true",
                    help="run only the buylist build step (implies "
                         "--skip-retrain)")
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
        rc = run_retrain()
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
            # Match Stage B: the v1.18b timing model was fit on contaminated
            # fit+val data and was replaced by the v2.0b one in train_v2_0b_prod.
            # This path was still passing the old model.
            "--timing", str(REPO_ROOT / "models" /
                             "time_to_debut_v2.0b_prod.pkl"),
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
