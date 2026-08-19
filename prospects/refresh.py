"""One-command driver: data update -> validation -> production build -> buy list.

This is the runbook in docs/runbook.md, executed. Everything it does can be
done by hand; the point is that the order is load-bearing and easy to get
wrong. Four ordering traps it removes:

  * prospects.db's `dedup` phase runs LAST inside `pull --phase all`, so
    career_outcomes is stale w.r.t. the collapsed player_ids until
    post_repull_chain rebuilds it. Modeling reads the SNAPSHOT db, so the
    rebuild has to land before the snapshot copy, not after.
  * prod.py hard-fails without oof_stacked_long/oof_val_long, and reads
    models/hazards.pkl — which oof does NOT write. model.train.hazards is a
    separate hand-run step that is easy to skip and silently scores stale.
  * a bare `prospects.buylist.build` defaults to joint_xgb.pkl (the OOF
    model). The production list wants joint_xgb_prod.pkl and must say so.
  * calibrators.pkl is fit against a SPECIFIC joint_xgb.pkl. Retraining the
    XGB without refitting them leaves last run's raw-score -> probability map
    in place, which mis-scales every calibrated number downstream without
    erroring. So calibrators refits after prod, before the buy list.

Usage:
    python -m prospects.refresh --season 2026            # the whole thing
    python -m prospects.refresh --dry-run                # print the plan
    python -m prospects.refresh --from oof               # resume after a failure
    python -m prospects.refresh --only buylist           # one step
    python -m prospects.refresh --skip pull --skip tests
    python -m prospects.refresh --tag cand               # isolated trial run

Steps, in order:
    tests     pytest — fast sanity on config + foundation
    backup    copy prospects.db -> prospects.db.bak_pre_pull_<stamp>
    pull      prospects.data.pull --phase all (FULL, never current-season-only)
    outcomes  post_repull_chain — rebuild career_outcomes after dedup + verify
    woba      derive wOBA on season_stats from the raw counting stats
    percentiles  rank each row within its (level, season_year) cohort
    snapshot  copy prospects.db -> prospects_snapshot.db (what modeling reads)
    split     regenerate the fit/val split over the current universe
    oof       OOF folds + hazards + joint_xgb.pkl
    evaluate  held-out metrics + evaluation/README.md  (non-blocking)
    hazards   full-panel production hazards -> models/hazards.pkl
    prod      prod joint XGB + snap scoring + buy list
    calibrators  refit isotonic/Platt calibrators on THIS run's OOF val
    buylist   explicit rebuild against joint_xgb_prod.pkl

There are two useful runs here, and picking the wrong one wastes hours.
The EVAL run answers "did this modelling change help?" and stops at the
metrics — everything it needs comes out of oof:

    python -m prospects.refresh --from split \
        --skip hazards --skip prod --skip calibrators --skip buylist

The FULL run additionally builds the production artifacts and the buy list.
Only that second half needs prod hazards, the prod XGB and calibrators, and
none of it feeds `evaluate`.

`evaluate` is deliberately non-blocking: a metrics regression is something to
read, not a reason to throw away a completed multi-hour rebuild. Every other
step is fail-fast.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from prospects import config
from prospects.config import REPO_ROOT

# Steps whose failure is reported but does not stop the run.
NON_BLOCKING = {"evaluate"}

# Ordered step names. Also the vocabulary for --from / --only / --skip.
# `evaluate` sits directly behind `oof` on purpose. It reads only
# oof_val_long + joint_xgb, both OOF outputs, and nothing downstream — so
# putting it after the production build meant paying for prod hazards, the
# prod XGB, calibrators and a buy list just to read a metric. Validating a
# modelling change is now:
#     prospects-refresh --from split --skip hazards --skip prod \
#                       --skip calibrators --skip buylist
STEP_ORDER = ["tests", "backup", "pull", "outcomes", "birthdates", "woba",
              "percentiles", "snapshot", "baselines", "split", "oof",
              "evaluate", "hazards", "prod", "calibrators", "buylist"]


def _banner(text: str) -> None:
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}", flush=True)


def run_cmd(label: str, cmd: list[str]) -> int:
    """Run a subprocess from the repo root, streaming output.

    Threads are pinned the same way weekly_score.py pins them: the hazard and
    XGB fits oversubscribe badly on a many-core box and run slower unpinned.
    """
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    proc = subprocess.run(
        cmd, cwd=REPO_ROOT,
        env={
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONIOENCODING": "utf-8",
        },
    )
    return proc.returncode


def _py(*args: str) -> list[str]:
    return [sys.executable, "-u", "-m", *args]


def step_backup(args) -> int:
    """Snapshot prospects.db before the pull rewrites it.

    Matches the existing prospects.db.bak_pre_pull_* convention on disk.
    """
    live = REPO_ROOT / "prospects.db"
    if not live.exists():
        print(f"  {live.name} does not exist yet; nothing to back up")
        return 0
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    dst = REPO_ROOT / f"prospects.db.bak_pre_pull_{stamp}"
    print(f"  {live.name} -> {dst.name} ({live.stat().st_size / 1e6:.1f} MB)")
    shutil.copy2(live, dst)
    return 0


def step_snapshot(args) -> int:
    """Refresh prospects_snapshot.db from the live db.

    All modeling reads the snapshot (config.model_db()), so a pull that never
    reaches the snapshot is a pull that changes nothing downstream. A wholesale
    copy rather than a per-table sync, so parity is guaranteed rather than
    argued about.
    """
    live = REPO_ROOT / "prospects.db"
    snap = REPO_ROOT / "prospects_snapshot.db"
    if not live.exists():
        print(f"  FATAL: {live.name} missing; cannot refresh the snapshot")
        return 1
    print(f"  {live.name} -> {snap.name} ({live.stat().st_size / 1e6:.1f} MB)")
    shutil.copy2(live, snap)
    return 0


def build_plan(args) -> list[tuple[str, str, object]]:
    """(name, description, action) for every step, in order.

    `action` is either an argv list to subprocess or a callable taking args.
    """
    run = config.run(args.tag) if args.tag else config.run()
    tag_args = ["--tag", args.tag] if args.tag else []

    plan = [
        ("tests", "pytest sanity", [sys.executable, "-m", "pytest", "-q"]),
        ("backup", "back up prospects.db", step_backup),
        ("pull", "FULL data pull (--phase all)",
         _py("prospects.data.pull", "--phase", "all", "--db", "prospects.db")),
        ("outcomes", "rebuild career_outcomes post-dedup + verify",
         _py("prospects.data.backfills.post_repull_chain")),
        # Both derive columns on season_stats from the rows the pull just
        # wrote, so they belong between `pull` and `snapshot` — modeling reads
        # the snapshot, so a derivation landing after the copy is invisible to
        # every model. Order matters: percentiles rank pct_woba, so woba first.
        # season_stats.age_during_season was empty for every one of 177,710
        # rows because these two were never wired in. That silently killed
        # eight panel features — age_yT/y1/y2, age_vs_level_*, delta_age,
        # delta_age_vs_level — and age-for-level is the single most load-
        # bearing concept in prospect evaluation. The `people` pass covers
        # anyone with an mlbam id (IFAs included); the draft pass covers
        # drafted players and derives age_during_season from birth_date.
        ("birthdates", "backfill birth_date + derive age_during_season",
         _py("prospects.data.backfills.birthdate_backfill_people",
             "--db", "prospects.db", "--apply")),
        ("woba", "derive wOBA from raw counting stats",
         _py("prospects.data.backfills.woba_backfill")),
        ("percentiles", "rank each row within its (level, year) cohort",
         _py("prospects.data.backfills.percentile_backfill")),
        ("snapshot", "refresh prospects_snapshot.db", step_snapshot),
        # Baselines are league medians per level, read from the SNAPSHOT, so
        # they have to be rebuilt after the copy and before any feature is
        # built. The checked-in file was computed while woba, fip and age
        # were all empty columns, so it carried no entry for any of them and
        # woba_vs_level / fip_vs_level / age_vs_level were dead in the panel.
        ("baselines", "recompute league baselines per level",
         _py("prospects.features.scouting", "--compute-baselines",
             "--out", "reference/milb_baselines.json")),
        ("split", "regenerate the fit/val split over the current universe",
         _py("prospects.model.train.make_split", *tag_args)),
        ("oof", "OOF folds + hazards + joint_xgb.pkl",
         _py("prospects.model.pipelines.oof", *tag_args)),
        ("evaluate", "held-out metrics + report",
         _py("prospects.evaluation.run", *tag_args)),
        ("hazards", "full-panel production hazards",
         _py("prospects.model.train.hazards", *tag_args)),
        ("prod", f"prod XGB + score snap={args.season} + buy list",
         _py("prospects.model.pipelines.prod",
             "--snap-year", str(args.season), *tag_args)),
        ("calibrators", "refit probability calibrators on THIS run's OOF val",
         _py("prospects.model.train.calibrators",
             "--val-long", str(run.oof_val_long),
             "--xgb", str(run.joint_xgb),
             "--out", str(run.calibrators))),
        ("buylist", "buy list against joint_xgb_prod.pkl",
         _py("prospects.buylist.build",
             "--long", str(run.snap_long(args.season)),
             "--xgb", str(run.joint_xgb_prod),
             "--timing", str(run.timing),
             "--out-all", str(run.buy_list_all),
             "--out-final", str(run.buy_list_final))),
    ]
    return plan


def select(plan, args) -> list:
    """Apply --only / --from / --skip to the full plan."""
    if args.only:
        return [s for s in plan if s[0] in args.only]
    out = plan
    if args.start_from:
        idx = STEP_ORDER.index(args.start_from)
        out = [s for s in out if STEP_ORDER.index(s[0]) >= idx]
    if args.skip:
        out = [s for s in out if s[0] not in args.skip]
    return out


def main():
    ap = argparse.ArgumentParser(
        description="One-command data refresh -> production build -> buy list.")
    ap.add_argument("--season", type=int, default=datetime.now().year,
                    help="Snap year to score (default: current year).")
    ap.add_argument("--tag", default=None,
                    help="Run namespace (runs/<tag>/). Default: current.")
    ap.add_argument("--from", dest="start_from", choices=STEP_ORDER,
                    help="Resume at this step (skip everything before it).")
    ap.add_argument("--only", action="append", choices=STEP_ORDER,
                    help="Run only this step. Repeatable.")
    ap.add_argument("--skip", action="append", choices=STEP_ORDER, default=[],
                    help="Skip this step. Repeatable.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan and exit without running anything.")
    args = ap.parse_args()

    plan = select(build_plan(args), args)

    _banner(f"REFRESH  season={args.season}  run={args.tag or 'current'}\n"
            f"{len(plan)} step(s): {', '.join(s[0] for s in plan)}")

    if args.dry_run:
        for name, desc, action in plan:
            shown = " ".join(action) if isinstance(action, list) \
                else f"<python: {action.__name__}>"
            flag = "  (non-blocking)" if name in NON_BLOCKING else ""
            print(f"\n[{name}] {desc}{flag}\n    {shown}")
        print("\n(dry run — nothing executed)")
        return 0

    t0 = time.time()
    results: list[tuple[str, int, float]] = []

    for name, desc, action in plan:
        _banner(f"[{name}] {desc}")
        t = time.time()
        if callable(action):
            rc = action(args)
        else:
            rc = run_cmd(name, action)
        dt = time.time() - t
        results.append((name, rc, dt))
        print(f"\n[{name}] exit={rc}  ({dt / 60:.1f} min)", flush=True)

        if rc != 0:
            if name in NON_BLOCKING:
                print(f"[{name}] WARN: failed but non-blocking; continuing",
                      flush=True)
                continue
            _banner(f"FAILED at [{name}] (exit {rc})\n"
                    f"Fix, then resume with:\n"
                    f"  python -m prospects.refresh --season {args.season} "
                    f"--from {name}")
            _summary(results, t0)
            return rc

    # The evaluation README regenerates from whatever run.py just wrote; it is
    # cheap and pointless to fail the run over, so it rides along with evaluate.
    if any(n == "evaluate" and rc == 0 for n, rc, _ in results):
        run_cmd("report", _py("prospects.evaluation.report",
                              *(["--tag", args.tag] if args.tag else [])))

    _banner("REFRESH COMPLETE")
    _summary(results, t0)

    run = config.run(args.tag) if args.tag else config.run()
    if run.buy_list_final.exists():
        n = sum(1 for _ in open(run.buy_list_final, encoding="utf-8")) - 1
        print(f"\nBuy list: {run.buy_list_final}  ({n:,} picks)")
        print(f"All scored: {run.buy_list_all}")
    return 0


def _summary(results, t0) -> None:
    print(f"\n{'step':<12} {'exit':>5}  {'minutes':>8}")
    print("-" * 30)
    for name, rc, dt in results:
        print(f"{name:<12} {rc:>5}  {dt / 60:>8.1f}")
    print("-" * 30)
    print(f"{'TOTAL':<12} {'':>5}  {(time.time() - t0) / 60:>8.1f}")


if __name__ == "__main__":
    sys.exit(main())
