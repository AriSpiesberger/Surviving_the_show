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


# Required artifacts for the v2.0b production pipeline (landmark hazards +
# conditional joint XGBoost). check_artifacts runs AFTER retrain, so a clean
# run never trips it — anything listed here that retrain does NOT produce is
# a real external dependency and is called out as such.
from prospects import config
from prospects.config import REPO_ROOT

_RUN = config.run()  # runs/current
# Required artifacts, as absolute paths under the current run. check_artifacts
# runs AFTER retrain, so a clean run never trips it — anything here that
# retrain does NOT produce is a real external dependency, flagged as such.
REQUIRED = [
    # --- Stage A (prospects.model.pipelines.stage_a) ---
    _RUN.hazards_landmark,   # landmark hazards, upstream of everything
    _RUN.lasso_logits,       # L1 bundle; feeds the p_debut_lasso timing feature
    _RUN.timing_stage_a,
    # --- Stage B (prospects.model.pipelines.prod) ---
    _RUN.joint_xgb_prod,     # conditional joint XGB — the scoring head
    _RUN.timing,             # its retrained timing model
    # NOT produced by run_retrain: prod READS these 100%-panel prod hazards,
    # but only model.train.hazards writes them and that step is hand-run.
    # Listed so a missing/stale file fails loudly rather than scoring silently.
    _RUN.hazards,
    # --- Stage C (v2.4, the deployed scorer) ---
    _RUN.models / "joint_xgb_v2.4.pkl",
    _RUN.models / "calibrators_v2.4.pkl",
    # --- the sheet's scorer: the v2.4 recipe refit on 100% of players ---
    _RUN.models / "joint_xgb_v2.5.pkl",
    _RUN.models / "calibrators_v2.5.pkl",
    # --- Shared infra ---
    config.POSITION_LOOKUP,
    config.model_db(),
]


def check_artifacts() -> list[str]:
    missing = []
    for path in REQUIRED:
        p = Path(path)
        if not p.is_absolute():
            p = REPO_ROOT / p
        if not p.exists():
            missing.append(str(path))
    return missing


DEFAULT_THREADS = "16"   # 32-core box; override with PROSPECT_NUM_THREADS


def run_step(label: str, cmd: list[str], cwd: Path,
             quiet: bool = False, threads: str | None = None) -> int:
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
            # The old blanket pin to 1 dated from a BSOD-instability window.
            # Measured on the 32-core box (2026-09): OOF stage 159 -> 27 min and
            # the joint fit 315 -> ~75 min at 16 threads. The exception is Stage
            # A's L1 refit, which THRASHES unpinned (2h+ vs ~30 min), so that
            # step passes threads="1" explicitly.
            "OMP_NUM_THREADS": threads or os.environ.get("PROSPECT_NUM_THREADS", DEFAULT_THREADS),
            "OPENBLAS_NUM_THREADS": threads or os.environ.get("PROSPECT_NUM_THREADS", DEFAULT_THREADS),
            "MKL_NUM_THREADS": threads or os.environ.get("PROSPECT_NUM_THREADS", DEFAULT_THREADS),
            "PYTHONIOENCODING": "utf-8",
        },
        stdout=subprocess.DEVNULL if quiet else None,
        stderr=subprocess.DEVNULL if quiet else None,
    )
    if not quiet:
        print(f"\n[{label}] exit={proc.returncode}", flush=True)
    return proc.returncode


def sheet_cmd(extra: list[str] | None = None) -> list[str]:
    """The production buy list: scored by v2.5 (the v2.4 recipe refit on 100% of
    players, val included), filtered by the per-yip thresholds computed on the
    HELD-OUT v2.4 run (recomputing them with v2.5 would be in-sample), priced
    from the latest daily pull when there is one."""
    cmd = [sys.executable, "-m", "prospects.buylist.build",
           "--xgb", str(_RUN.models / "joint_xgb_v2.5.pkl"),
           "--calibrators", str(_RUN.models / "calibrators_v2.5.pkl"),
           "--yip-thresholds", str(_RUN.yip_thresholds(60))]
    latest = REPO_ROOT / "prices" / "prices_buylist_latest.csv"
    if latest.exists():
        cmd += ["--prices", str(latest)]
    return cmd + (extra or [])


def run_retrain() -> int:
    """Full retrain orchestration. Returns 0 on success, non-zero on first
    failed step.

    Delegates the heavy lifting to two orchestrator scripts that already
    encode the v2.0b production pipeline:

      - prospects.model.pipelines.stage_a : panel rebuild + landmark
        hazards + score fit/val + downstream lasso/timing refit.
      - prospects.model.pipelines.prod : conditional joint XGB
        (fit_joint_xgb_cond) on landmark longs + snap=2026 scoring + buy
        list build."""
    print(f"\n{'#'*70}\n# WEEKLY RETRAIN (held-out v2.4 for metrics, "
          f"100% v2.5 for the sheet)\n{'#'*70}", flush=True)
    py = sys.executable
    scratch = REPO_ROOT / "runs" / "current" / "scratch"
    v24_dir, v25_dir = str(scratch / "v24_build"), str(scratch / "v25_full_build")
    aug_long = str(_RUN.training / "recent_long.csv")
    all_long = str(_RUN.training / "oof_all_long.csv")
    concat = ("import pandas as pd;"
              f"a=pd.read_csv(r'{_RUN.oof_stacked_long}',low_memory=False);"
              f"b=pd.read_csv(r'{_RUN.oof_val_long}',low_memory=False);"
              f"pd.concat([a,b],ignore_index=True).to_csv(r'{all_long}',index=False);"
              "print('[full] oof_all_long rows',len(a)+len(b))")
    exp5 = [py, "-m", "prospects.model.train.exp_cdf_timing5",
            "--aug-long", aug_long, "--cal-min-snap-year", "2008"]
    models = _RUN.models
    steps = [
        # Legacy v1.18b bundle (lasso/timing). Feeds nothing on the sheet but is a
        # required artifact; pinned to one thread, where it is ~4x faster.
        ("retrain/v1.18b", [py, "-m", "prospects.model.pipelines.stage_a"], "1"),
        # Fresh universe, panel, OOF longs and PRODUCTION hazards every week.
        # Until 2026-09-19 none of these ran here: hazards.pkl was "hand-run" and
        # sat frozen at 2026-09-08 through two full retrains.
        ("retrain/split", [py, "-m", "prospects.model.train.make_split"], None),
        ("retrain/oof", [py, "-m", "prospects.model.pipelines.oof"], None),
        ("retrain/hazards", [py, "-m", "prospects.model.train.hazards", "--force"], None),
        ("retrain/v2.0b", [py, "-m", "prospects.model.pipelines.prod", "--skip-buylist"], None),
        ("retrain/recent", [py, "-m", "prospects.model.train.score_recent_cohorts"], None),
        # v2.4 = held-out: the honest metrics, README and per-yip thresholds.
        ("retrain/v2.4-joint", exp5 + ["--out-dir", v24_dir], None),
        ("retrain/v2.4-promote", [py, "-m", "prospects.model.train.promote_v22",
                                  "--source", v24_dir, "--bag-name", "joint_xgb_exp5_bag.pkl",
                                  "--version", "v2.4"], None),
        ("retrain/v2.4-eval", [py, "-m", "prospects.evaluation.run",
                               "--xgb", str(models / "joint_xgb_v2.4.pkl"),
                               "--calibrators", str(models / "calibrators_v2.4.pkl"),
                               "--threshold", "0.6"], None),
        ("retrain/v2.4-report", [py, "-m", "prospects.evaluation.report"], None),
        ("retrain/v2.4-thresholds", [py, "-m", "prospects.buylist.build",
                                     "--xgb", str(models / "joint_xgb_v2.4.pkl"),
                                     "--calibrators", str(models / "calibrators_v2.4.pkl"),
                                     "--out-all", str(scratch / "all_scored_v24.csv"),
                                     "--out-final", str(scratch / "final_v24.csv")], None),
        # v2.5 = the same recipe on 100% of players (fit + val). This scores the sheet.
        ("retrain/v2.5-long", [py, "-c", concat], None),
        ("retrain/v2.5-joint", exp5 + ["--fit", all_long, "--val", str(_RUN.oof_val_long),
                                       "--out-dir", v25_dir], None),
        ("retrain/v2.5-promote", [py, "-m", "prospects.model.train.promote_v22",
                                  "--source", v25_dir, "--bag-name", "joint_xgb_exp5_bag.pkl",
                                  "--version", "v2.5"], None),
        # Gate: no sheet from a model that fails the leak / freshness audit.
        ("retrain/leak-audit", [py, str(REPO_ROOT / "tools" / "leak_audit.py")], None),
        ("retrain/v2.5-buylist", sheet_cmd(), None),
    ]
    for label, cmd, threads in steps:
        rc = run_step(label, cmd, REPO_ROOT, threads=threads)
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

    _RUN.scored.mkdir(parents=True, exist_ok=True)
    snap_long = _RUN.snap_long(args.season)

    # Step 1+2: snap=2026 landmark scoring + v2.0b buy list. When the full
    # retrain ran, train_v2_0b_prod already did this — so on a full-retrain
    # run these are no-ops (the orchestrator's --skip-xgb path is the
    # ad-hoc rescoring entry point used by --skip-retrain modes).
    if skip_retrain and not args.buylist_only:
        # Rescore snap with EXISTING landmark hazards + rebuild buy list.
        # --skip-xgb keeps the prod XGB pkl; --skip-buylist is set when the
        # caller really only wants the snap_long.
        # Rescore only: prod's own buy-list step uses the fallback v2.1c scorer,
        # so it is always skipped; the sheet is built by sheet_cmd (v2.5).
        rc = run_step("score", [sys.executable, "-m", "prospects.model.pipelines.prod",
                                "--skip-xgb", "--skip-buylist"], REPO_ROOT)
        if rc != 0:
            sys.exit(1)
        if not args.score_only:
            rc = run_step("buylist", sheet_cmd([
                "--long", str(snap_long),
                "--out-all", str(_RUN.buy_list_all),
                "--out-final", str(_RUN.buy_list_final),
            ]), REPO_ROOT)
            if rc != 0:
                sys.exit(2)
    elif skip_retrain and args.buylist_only:
        # Snap_long must already exist; just rebuild the buy list.
        if not snap_long.exists():
            print(f"FATAL: need {snap_long.name} but it doesn't exist; "
                  f"run without --buylist-only first")
            sys.exit(3)
        # The weekly buy list is scored with the promoted v2.3 bag +
        # calibrators (timing comes from the calibrated debut CDF inside
        # buylist.build; the Lasso --timing path is legacy-bundle only).
        rc = run_step("buylist", sheet_cmd([
            "--long", str(snap_long),
            "--out-all", str(_RUN.buy_list_all),
            "--out-final", str(_RUN.buy_list_final),
        ]), REPO_ROOT)
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

    print(f"\n=== weekly_score season={args.season} OK ===")
    print(f"  snap long file: {snap_long}")
    print(f"  buy list:       {_RUN.buy_list_final}")
    print(f"  all scored:     {_RUN.buy_list_all}")


if __name__ == "__main__":
    main()
