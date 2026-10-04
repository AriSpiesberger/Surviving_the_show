"""Finish the production artifacts for a candidate run (RUN_TAG=ext3, 1988-2004 classes, 2026-10-01).

The ext run already holds the panel, OOF longs, hazards, v2.4 (held-out) and the held-out v3.
This runs the rest of the weekly retrain's tail against it -- v2.4 evaluation / thresholds, v2.5,
v3.5, recalibration, v3 thresholds, leak audit, both sheets -- so runs/ext can be swapped into
runs/current whole. Run with RUN_TAG=ext and PROSPECT_(MODEL_)DB=prospects_ext.db.
"""
import os
import sys

assert os.environ.get("RUN_TAG") not in (None, "", "current"), "run under the candidate RUN_TAG"
from prospects import config
from prospects.deploy.v3_sidebyside import v3_build_steps
from prospects.deploy.weekly_score import publish_cmd, run_step, sheet_cmd
from prospects.config import REPO_ROOT

R = config.run()
py, models, scratch = sys.executable, R.models, R.scratch
v25_dir = str(scratch / "v25_full_build")
all_long = str(R.training / "oof_all_long.csv")
aug = str(R.training / "recent_long.csv")
concat = ("import pandas as pd;"
          f"a=pd.read_csv(r'{R.oof_stacked_long}',low_memory=False);"
          f"b=pd.read_csv(r'{R.oof_val_long}',low_memory=False);"
          f"pd.concat([a,b],ignore_index=True).to_csv(r'{all_long}',index=False);"
          "print('[full] oof_all_long rows',len(a)+len(b))")
exp5 = [py, "-m", "prospects.model.train.exp_cdf_timing5", "--aug-long", aug, "--cal-min-snap-year", "2008",
        "--db", str(config.model_db())]
steps = [
    ("v2.4-eval", [py, "-m", "prospects.evaluation.run", "--xgb", str(models / "joint_xgb_v2.4.pkl"),
                   "--calibrators", str(models / "calibrators_v2.4.pkl"), "--threshold", "0.6"]),
    ("v2.4-report", [py, "-m", "prospects.evaluation.report"]),
    ("v2.4-thresholds", [py, "-m", "prospects.buylist.build", "--xgb", str(models / "joint_xgb_v2.4.pkl"),
                         "--calibrators", str(models / "calibrators_v2.4.pkl"),
                         "--out-all", str(scratch / "all_scored_v24.csv"), "--out-final", str(scratch / "final_v24.csv")]),
    ("v2.5-long", [py, "-c", concat]),
    ("v2.5-joint", exp5 + ["--fit", all_long, "--val", str(R.oof_val_long), "--out-dir", v25_dir]),
    ("v2.5-promote", [py, "-m", "prospects.model.train.promote_v22", "--source", v25_dir,
                      "--bag-name", "joint_xgb_exp5_bag.pkl", "--version", "v2.5"]),
    ("v3/full", [py, "-m", "prospects.model.v3", "build", "--fit", all_long, "--aug-long", aug,
                 "--base-xgb", str(models / "joint_xgb_v2.5.pkl"), "--base-cal", str(models / "calibrators_v2.5.pkl"),
                 "--out", str(models / "v3.5.pkl"), "--out-cal", str(models / "calibrators_v3.5.pkl"),
                 "--seeds", "5", "--threads", "16", "--encoder", "transformer_rank"]),
]
steps += v3_build_steps(threads="16", build=False)          # recalibrate (both files) + v3 thresholds
steps += [
    ("leak-audit", [py, str(REPO_ROOT / "tools" / "leak_audit.py")]),
    ("v2.5-buylist", sheet_cmd(["--out-all", str(R.buy_list_all).replace("all_scored", "all_scored_v25"),
                                "--out-final", str(R.buy_list_final).replace("final", "final_v25")])),
]
# the sheet: publish_cmd() is resolved only after v3.5 exists (it silently fell back to v2.5 on 2026-09-25)
steps.append(("v3-buylist", None))
for label, cmd in steps:
    if cmd is None:
        cmd = publish_cmd()
        assert "v3.5.pkl" in " ".join(cmd), "v3.5 missing: refusing to publish the v2.5 fallback as the sheet"
    rc = run_step(f"promote/{label}", cmd, REPO_ROOT, threads=None)
    if rc != 0:
        sys.exit(rc)
print("[ext-promote] OK", flush=True)
