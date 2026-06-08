"""Auto-retry wrapper around run_v2_0b_oof.

Keeps invoking the OOF script in fresh subprocesses until it actually
completes (signaled by models/joint_xgb_v2.0b_oof.pkl existing). Each
subprocess picks up where the last one died thanks to per-snap +
hazards-pkl checkpointing.

Usage:
    python -m scripts_v17.train.run_v2_0b_oof_until_done

Bails after MAX_ATTEMPTS so it can't loop forever on a real bug.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
XGB_OUT = REPO_ROOT / "models" / "joint_xgb_v2.0b_oof.pkl"

MAX_ATTEMPTS = 200
SLEEP_BETWEEN = 5  # seconds


def main():
    t0 = time.time()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if XGB_OUT.exists():
            print(f"\n{'='*70}\n  COMPLETE — {XGB_OUT.name} exists after "
                  f"{attempt-1} attempts, {(time.time()-t0)/60:.1f} min\n"
                  f"{'='*70}")
            return 0
        print(f"\n{'#'*70}")
        print(f"# attempt {attempt}/{MAX_ATTEMPTS}  "
              f"(elapsed {(time.time()-t0)/60:.1f} min)")
        print(f"{'#'*70}\n", flush=True)
        rc = subprocess.run([
            sys.executable, "-m", "scripts_v17.train.run_v2_0b_oof",
        ], cwd=REPO_ROOT).returncode
        if rc == 0 and XGB_OUT.exists():
            print(f"\n  ✓ XGB built. Total {attempt} attempts, "
                  f"{(time.time()-t0)/60:.1f} min")
            return 0
        print(f"\n  ✗ attempt {attempt} exited rc={rc}, "
              f"XGB_OUT exists={XGB_OUT.exists()}. "
              f"Sleeping {SLEEP_BETWEEN}s before retry...", flush=True)
        time.sleep(SLEEP_BETWEEN)
    print(f"\nGiving up after {MAX_ATTEMPTS} attempts.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
