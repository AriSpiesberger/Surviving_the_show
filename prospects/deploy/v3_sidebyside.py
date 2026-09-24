"""Build v3 next to production and write a parallel buy list (2026-09-23).

Nothing here touches the production sheet, its thresholds or the evaluation README. It needs
the artifacts a weekly retrain leaves behind (v2.4 / v2.5 bundles, the OOF longs, the 100%
long, the snapshot long) and does NOT refresh prospects_snapshot.db, so v3 is trained and
scored on exactly the data the base models saw.

  held-out v3   (fit players only, base v2.4)  -> per-yip P60 thresholds, honest metrics
  v3.5          (fit + val players, base v2.5) -> scores the parallel sheet
  leak audit    (now covers v3*.pkl bundles)
  sheet         runs/current/buy_lists/final_v3.csv (+ all_scored_v3.csv)

    python -m prospects.deploy.v3_sidebyside
    python -m prospects.deploy.v3_sidebyside --events MLB_DEBUT TOP_100_PROSPECT \
        ESTABLISHED_MLB STAR_PLUS_ELITE --cond ESTABLISHED_MLB STAR_PLUS_ELITE
"""
from __future__ import annotations

import argparse
import sys

from prospects import config
from prospects.config import REPO_ROOT
from prospects.deploy.weekly_score import run_step

_RUN = config.run()


THR_V3 = str(_RUN.yip_thresholds(60)).replace("p60", "v3_p60")
DEFAULT_ENCODER = "transformer_rank"   # adopted 2026-09-23 (full-size screen, exp_v3_all_trf)


def v3_build_steps(encoder=DEFAULT_ENCODER, seeds=5, threads="16", spec=None, build=True):
    """(label, cmd) steps: held-out v3 (base v2.4), v3.5 (base v2.5), recalibration, v3's own
    per-yip thresholds. Shared by this runner and prospects.deploy.weekly_score."""
    py, models, scratch = sys.executable, _RUN.models, REPO_ROOT / "runs" / "current" / "scratch"
    aug = str(_RUN.training / "recent_long.csv")
    all_long = str(_RUN.training / "oof_all_long.csv")
    ev = (["--spec", *spec] if spec else []) + ["--seeds", str(seeds), "--threads", str(threads),
                                                "--encoder", encoder]
    steps = []
    if build:
        steps += [
            ("v3/heldout", [py, "-m", "prospects.model.v3", "build", "--fit", str(_RUN.oof_stacked_long),
                            "--aug-long", aug, "--base-xgb", str(models / "joint_xgb_v2.4.pkl"),
                            "--base-cal", str(models / "calibrators_v2.4.pkl"),
                            "--out", str(models / "v3.pkl"), "--out-cal", str(models / "calibrators_v3.pkl"),
                            *ev]),
            ("v3/full", [py, "-m", "prospects.model.v3", "build", "--fit", all_long, "--aug-long", aug,
                         "--base-xgb", str(models / "joint_xgb_v2.5.pkl"),
                         "--base-cal", str(models / "calibrators_v2.5.pkl"),
                         "--out", str(models / "v3.5.pkl"), "--out-cal", str(models / "calibrators_v3.5.pkl"),
                         *ev]),
        ]
    steps += [
        # monotone Platt per horizon, fit on the held-out v3's val predictions, written into both
        # calibrator files: fixes the ceiling under-prediction without changing the order
        ("v3/recalibrate", [py, "-m", "prospects.model.v3", "recalibrate",
                            "--heldout-xgb", str(models / "v3.pkl"),
                            "--heldout-cal", str(models / "calibrators_v3.pkl"),
                            "--write", str(models / "calibrators_v3.pkl"), str(models / "calibrators_v3.5.pkl")]),
        # held-out v3 on the held-out val slice -> its own per-yip P60 thresholds
        ("v3/thresholds", [py, "-m", "prospects.buylist.build", "--xgb", str(models / "v3.pkl"),
                           "--calibrators", str(models / "calibrators_v3.pkl"), "--thresholds-out", THR_V3,
                           "--out-all", str(scratch / "all_scored_v3held.csv"),
                           "--out-final", str(scratch / "final_v3held.csv")]),
    ]
    return steps


# The user's buy rule (2026-09-23): every player on the sheet must INDIVIDUALLY have
# P(debut <= 3y) >= 0.60. (Until then the rule was basket precision 60% per yip, whose
# individual cutoffs were 0.37-0.45; that is still computed by v3/thresholds for reference.)
MIN_P_DEBUT_3Y = 0.60


def v3_sheet_cmd(out_all, out_final, extra=None):
    """The v3.5-scored buy list: individual P(debut <= 3y) >= MIN_P_DEBUT_3Y, latest prices."""
    models = _RUN.models
    cmd = [sys.executable, "-m", "prospects.buylist.build", "--xgb", str(models / "v3.5.pkl"),
           "--calibrators", str(models / "calibrators_v3.5.pkl"),
           "--precision", "0", "--threshold", str(MIN_P_DEBUT_3Y),
           "--out-all", str(out_all), "--out-final", str(out_final)]
    latest = REPO_ROOT / "prices" / "prices_buylist_latest.csv"
    if latest.exists():
        cmd += ["--prices", str(latest)]
    return cmd + (extra or [])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", nargs="*", default=None,
                    help="EVENT=comp,... (default: prospects.model.v3.DEFAULT_SPEC)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--encoder", default=DEFAULT_ENCODER, choices=["gru", "transformer_rank"])
    ap.add_argument("--threads", default="16")
    ap.add_argument("--skip-build", action="store_true", help="reuse existing v3 bundles")
    args = ap.parse_args()
    steps = v3_build_steps(args.encoder, args.seeds, args.threads, args.spec, build=not args.skip_build)
    steps.append(("v3/leak-audit", [sys.executable, str(REPO_ROOT / "tools" / "leak_audit.py")]))
    steps.append(("v3/sheet", v3_sheet_cmd(str(_RUN.buy_list_all).replace("all_scored", "all_scored_v3"),
                                           str(_RUN.buy_list_final).replace("final", "final_v3"))))
    for label, cmd in steps:
        rc = run_step(label, cmd, REPO_ROOT, threads=args.threads)
        if rc != 0:
            print(f"\nFATAL: {label} failed (rc={rc})", flush=True)
            sys.exit(rc)
    print("\n[v3] side-by-side OK", flush=True)


if __name__ == "__main__":
    main()
