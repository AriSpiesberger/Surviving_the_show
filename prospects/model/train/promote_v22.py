"""Promote the exp4 champion ("F") to v2.2 artifacts under runs/current/models.

- Backs up runs/current/{models,evaluation,buy_lists} to
  runs/backup_v2.1c_<date>/ first (ground state restorable by copying back).
- Writes models/joint_xgb_v2.2.pkl  (5-seed monotone bag + raw-feature list)
  and models/calibrators_v2.2.pkl   (per-event h/yip calibrators, re-homed
  from the experiment module to prospects.model.joint2 so prod never imports
  experiment code).
- Leaves joint_xgb.pkl / calibrators.pkl (v2.1c) untouched: consumers select
  v2.2 via --xgb/--calibrators. NOTE: the weekly retrain still produces
  v2.1c-recipe artifacts until pipelines are ported to the F recipe.

    python -m prospects.model.train.promote_v22
"""
from __future__ import annotations

import pickle
import shutil
import time
from pathlib import Path

from prospects import config
from prospects.config import REPO_ROOT
from prospects.model.joint2 import HYip2Calibrator

_RUN = config.run()
# exp5 = the exp4 recipe retrained with FULL-coverage raw features (the
# deployable variant — exp4's panel-cache attachment left pre-2007 and
# post-panel snaps NaN, which deployment cannot reproduce).
SOURCE = REPO_ROOT / "runs" / "exp_cdf_timing5"
BAG_PKL = "joint_xgb_exp5_bag.pkl"
BACKUP = REPO_ROOT / "runs" / f"backup_v2.1c_{time.strftime('%Y%m%d')}"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(SOURCE),
                    help="Experiment dir holding the bag + calibrators.")
    ap.add_argument("--bag-name", default=BAG_PKL)
    ap.add_argument("--version", default="v2.2",
                    help="Version stamp -> models/joint_xgb_<v>.pkl + "
                         "models/calibrators_<v>.pkl")
    args = ap.parse_args()
    src = Path(args.source)
    ver = args.version

    # ---- 1. backup ground state ----------------------------------------
    if BACKUP.exists():
        print(f"[promote] backup already exists: {BACKUP}")
    else:
        BACKUP.mkdir(parents=True)
        for sub in ("models", "evaluation", "buy_lists"):
            bsrc = _RUN.root / sub
            if bsrc.exists():
                shutil.copytree(bsrc, BACKUP / sub)
                print(f"[promote] backed up {sub}/ "
                      f"({sum(1 for _ in (BACKUP / sub).rglob('*'))} files)")

    # ---- 2. joint XGB bag ----------------------------------------------
    with open(src / args.bag_name, "rb") as fh:
        bag = pickle.load(fh)
    bundle = {
        "models": bag["models"],
        "feature_names": list(bag["feature_names"]),
        "keep_raw": list(bag["keep_raw"]),
        "events": list(bag["events"]),
        "h_max": int(bag["h_max"]),
        "publish_h": 6,
        "version": ver,
        "kind": f"joint_xgb_bag_{ver}",
        "recipe": {"overrides": bag.get("overrides"),
                   "num_rounds": bag.get("num_rounds"),
                   "monotone": "h_centered + horizon margins (+1)",
                   "raw_coverage": bag.get("raw_coverage", "panel-only"),
                   "source": str(src.name)},
    }
    out_xgb = _RUN.models / f"joint_xgb_{ver}.pkl"
    with open(out_xgb, "wb") as fh:
        pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[promote] wrote {out_xgb.name}: {len(bundle['models'])} models, "
          f"{len(bundle['feature_names'])} feats "
          f"({len(bundle['keep_raw'])} raw)")

    # ---- 3. calibrators, re-homed to joint2 ----------------------------
    with open(src / "calibrators_hyip2.pkl", "rb") as fh:
        cal_exp = pickle.load(fh)     # classes resolve via the exp module
    rehomed = {}
    for ev, c in cal_exp["calibrators"].items():
        nc = HYip2Calibrator()
        nc.lr = c.lr                  # fitted sklearn model carries over
        rehomed[ev] = nc
    out_cal = _RUN.models / f"calibrators_{ver}.pkl"
    with open(out_cal, "wb") as fh:
        pickle.dump({"calibrators": rehomed,
                     "events": list(cal_exp.get("events", rehomed)),
                     "h_max": int(cal_exp.get("h_max", 10)),
                     "version": ver,
                     "kind": "hyip2_logistic_oof"},
                    fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[promote] wrote {out_cal.name}: {len(rehomed)} per-event "
          f"h/yip calibrators")
    print(f"[promote] ground state restorable from {BACKUP}")


if __name__ == "__main__":
    main()
