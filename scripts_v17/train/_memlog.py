"""Tiny memory logger — call rss('label') anywhere to print/append the
process resident set size in MB to logs/memlog.txt and stdout."""
from __future__ import annotations

import os
import time
from pathlib import Path

import psutil

_PROC = psutil.Process(os.getpid())
_LOG = (Path(__file__).resolve().parents[2] / "logs" / "memlog.txt")
_LOG.parent.mkdir(parents=True, exist_ok=True)
_T0 = time.time()


def rss(label: str = "") -> float:
    mb = _PROC.memory_info().rss / (1024 * 1024)
    line = (f"[mem {time.time()-_T0:6.0f}s] RSS={mb:7.1f} MB  "
            f"VMS={_PROC.memory_info().vms/(1024*1024):7.1f} MB  "
            f"{label}")
    print(line, flush=True)
    with _LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    return mb
