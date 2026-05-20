"""Bundle the v1.17 deploy seed into seed_v1.17.tar.gz.

Run locally. Upload the resulting tarball to a public URL (GitHub Release
recommended — free, fast, and you can keep the repo private), then set the
Railway service env var SEED_URL to that URL. On first boot (or any time
/data is empty), auto_seed.py downloads and extracts it.

Usage:
    python scripts/prepare_seed.py
    # -> seed_v1.17.tar.gz in the repo root (~150 MB)
"""
from __future__ import annotations

import os
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "seed_v1.17.tar.gz"

# Paths are relative to REPO_ROOT. Listed in the order auto_seed.py will
# extract them onto /data — directory structure is preserved verbatim.
REQUIRED: list[str] = [
    "prospects.db",
    "prospects_snapshot.db",
    "panels/panel_v1.17.npz",
    "models/event_classifiers_v1.17_prod.pkl",
    "models/debut_lasso_universe_v1.17h.pkl",
    "models/top100_lasso_v1.17h.pkl",
    "models/model_b_outcomes_v1.17h.pkl",
    "models/player_position_from_stats.csv",
    "buy_list_v1.17_FINAL.csv",
]

# Optional — included if present.
OPTIONAL: list[str] = [
    "holdings.csv",
    "alerts_state.json",
    "scripts_v17/score/score_panel_v17.py",
    "scripts_v17/buylist/build_v17_buylist.py",
]


def main() -> int:
    missing = [p for p in REQUIRED if not (REPO_ROOT / p).exists()]
    if missing:
        print("ERROR: required files missing:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        return 1

    print(f"Building {OUT.name} ...")
    total_bytes = 0
    n_files = 0
    with tarfile.open(OUT, "w:gz") as tar:
        for rel in REQUIRED + OPTIONAL:
            p = REPO_ROOT / rel
            if not p.exists():
                if rel in OPTIONAL:
                    print(f"  (skipping optional, not present: {rel})")
                continue
            tar.add(p, arcname=rel)
            sz = p.stat().st_size
            total_bytes += sz
            n_files += 1
            print(f"  + {rel}  ({sz / 1024 / 1024:.1f} MB)")

    out_size = OUT.stat().st_size
    print()
    print(f"Wrote {OUT}: {out_size / 1024 / 1024:.1f} MB compressed "
          f"({total_bytes / 1024 / 1024:.1f} MB uncompressed, {n_files} files)")
    print()
    print("Next steps:")
    print("  1. Create a GitHub Release on your repo (any tag, e.g. seed-v1.17-2026-05-20).")
    print("  2. Upload seed_v1.17.tar.gz as a release asset.")
    print("  3. Right-click the uploaded asset and copy its download URL.")
    print("     It will look like:")
    print("     https://github.com/<you>/<repo>/releases/download/<tag>/seed_v1.17.tar.gz")
    print("  4. In Railway: set service env var  SEED_URL=<that URL>")
    print("  5. railway up")
    return 0


if __name__ == "__main__":
    sys.exit(main())
