"""Boot-time volume seeder. Runs on every container start.

If $DATA_DIR (default /data) is missing key seed files, downloads
$SEED_URL and extracts it onto $DATA_DIR. Otherwise no-op. Idempotent —
safe to run every boot.

Strategy is "presence of sentinel files" not "directory empty", because
the cron jobs write into /data and we don't want to overwrite live state
on every restart.

Sentinel: presence of prospects.db AND the v1.17 hazard pkl.

Usage:
    python scripts/auto_seed.py
    # (in Railway startCommand, runs before the cron host idles)
"""
from __future__ import annotations

import os
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
SEED_URL = os.environ.get("SEED_URL", "")

# If both of these exist, we consider the volume already seeded.
SENTINELS = [
    DATA_DIR / "prospects.db",
    DATA_DIR / "models" / "event_classifiers_v1.17_prod.pkl",
]


def already_seeded() -> bool:
    return all(p.exists() for p in SENTINELS)


def download(url: str, dst: Path) -> int:
    print(f"[auto_seed] downloading {url}")
    t0 = time.time()
    last_pct = -1
    with urllib.request.urlopen(url) as r, open(dst, "wb") as out:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        chunk = 1024 * 1024
        while True:
            buf = r.read(chunk)
            if not buf:
                break
            out.write(buf)
            got += len(buf)
            if total:
                pct = int(100 * got / total)
                if pct != last_pct and pct % 5 == 0:
                    print(f"  {pct}%  ({got/1024/1024:.0f} MB / "
                          f"{total/1024/1024:.0f} MB)")
                    last_pct = pct
    dt = time.time() - t0
    sz = dst.stat().st_size
    print(f"[auto_seed] downloaded {sz/1024/1024:.1f} MB in {dt:.1f}s")
    return sz


def extract(tar_path: Path, dst_dir: Path) -> int:
    print(f"[auto_seed] extracting {tar_path.name} -> {dst_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with tarfile.open(tar_path, "r:gz") as tar:
        for m in tar.getmembers():
            # Guard against path traversal in arcnames.
            target = (dst_dir / m.name).resolve()
            if not str(target).startswith(str(dst_dir.resolve())):
                print(f"[auto_seed] skipping suspicious member: {m.name}",
                      file=sys.stderr)
                continue
            tar.extract(m, dst_dir)
            n += 1
    return n


def main() -> int:
    print(f"[auto_seed] DATA_DIR={DATA_DIR}")
    if already_seeded():
        print("[auto_seed] volume already seeded (sentinels present); skipping")
        return 0
    if not SEED_URL:
        print("[auto_seed] ERROR: /data is unseeded and SEED_URL is not set",
              file=sys.stderr)
        return 2

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tf:
        tar_path = Path(tf.name)
    try:
        download(SEED_URL, tar_path)
        n = extract(tar_path, DATA_DIR)
        print(f"[auto_seed] extracted {n} entries")
    finally:
        try:
            tar_path.unlink()
        except OSError:
            pass

    if not already_seeded():
        print("[auto_seed] ERROR: post-extract sentinels still missing — "
              "bundle may be incomplete", file=sys.stderr)
        return 3
    print("[auto_seed] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
