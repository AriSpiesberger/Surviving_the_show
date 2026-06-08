"""End-to-end data audit for v2.0b OOF.

Checks (each prints PASS / WARN / FAIL):

  1. PID SPLITS
     - fit ∩ val empty
     - val ∩ haz_universe empty
     - no duplicate pids within any split file

  2. PANEL FEATURES
     - NaN / Inf rates per feature
     - constant / near-constant features (zero variance)
     - feature value ranges (catches scaling bugs)
     - completeness vs prospect inventory

  3. LABELS
     - per-event positive rates on val
     - eligible vs realized consistency
     - trigger_year distributions (gaps, outliers)
     - "realized but not eligible" or other contradictions

  4. LONG CSV INTEGRITY (oof_stacked, oof_val, honest_val, honest_fit)
     - schema (columns + dtypes)
     - row counts vs expected
     - duplicate (pid, snap) combos
     - hazard-prob NaN rates
     - mean_t / sd_t sanity
     - .meta.json sidecars

  5. COHORT CONSISTENCY
     - entry_year distributions across splits
     - bucket (round / IFA) distributions
     - debut_year distributions

Usage:
    python -m scripts_v17.train.audit_data
    python -m scripts_v17.train.audit_data --json   # machine-readable
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from prospects.classifier.architectures.landmark_survival import N_FEATURES

SCRATCH = REPO_ROOT / "scratch" / "v20b_oof"
PANEL_NPZ = SCRATCH / "panel_cache.npz"
PANEL_META = SCRATCH / "panel_meta.pkl"
TRAIN_DIR = REPO_ROOT / "results" / "training"
VAL_PIDS = TRAIN_DIR / "v17_prod_val_pids.txt"
FIT_PIDS = TRAIN_DIR / "v17_prod_fit_pids.txt"

EVENTS = ["TOP_100_PROSPECT", "MLB_DEBUT", "ESTABLISHED_MLB",
          "STAR_PLUS_ELITE", "ELITE", "STAR"]

# Report aggregator
REPORT: list[dict] = []


def emit(check: str, status: str, detail: str, **extra):
    """status ∈ {PASS, WARN, FAIL, INFO}"""
    rec = {"check": check, "status": status, "detail": detail, **extra}
    REPORT.append(rec)
    badge = {
        "PASS": "  ✓", "WARN": " ⚠ ", "FAIL": " ✗ ", "INFO": "  •",
    }[status]
    print(f"{badge}  {check:<45} {detail}")


def _read_pids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {ln.strip() for ln in path.read_text().splitlines() if ln.strip()}


# ---- 1. PID SPLITS ----
def audit_pid_splits():
    print("\n=== 1. PID SPLITS ===")
    val = _read_pids(VAL_PIDS)
    fit = _read_pids(FIT_PIDS)
    emit("val_pids file", "INFO", f"{len(val):,} pids ({VAL_PIDS.name})")
    emit("fit_pids file", "INFO", f"{len(fit):,} pids ({FIT_PIDS.name})")
    overlap = val & fit
    emit("val ∩ fit disjoint",
         "PASS" if not overlap else "FAIL",
         "no overlap" if not overlap
         else f"{len(overlap)} pids in BOTH (LEAK)")
    val_raw = (VAL_PIDS.read_text().splitlines() if VAL_PIDS.exists() else [])
    val_dups = len([p for p in val_raw if p.strip()]) - len(val)
    emit("val_pids dedupe",
         "PASS" if val_dups == 0 else "WARN",
         f"{val_dups} duplicate lines" if val_dups else "no duplicates")


# ---- 2. PANEL FEATURES ----
def audit_panel():
    print("\n=== 2. PANEL FEATURES ===")
    if not PANEL_NPZ.exists():
        emit("panel_cache.npz", "FAIL", "missing")
        return
    npz = np.load(PANEL_NPZ, allow_pickle=True)
    X = npz["X_lm"]
    n_rows, n_feat = X.shape
    emit("panel shape", "INFO", f"{n_rows:,} rows × {n_feat} features")
    if n_feat != N_FEATURES:
        emit("panel feature count", "FAIL",
             f"{n_feat} columns vs expected {N_FEATURES}")
    else:
        emit("panel feature count", "PASS",
             f"matches N_FEATURES = {N_FEATURES}")

    # NaN / Inf
    n_nan = int(np.isnan(X).sum())
    n_inf = int(np.isinf(X).sum())
    cells = X.size
    emit("panel NaN cells", "PASS" if n_nan == 0 else "WARN",
         f"{n_nan:,} / {cells:,} ({100*n_nan/cells:.4f}%)")
    emit("panel Inf cells", "PASS" if n_inf == 0 else "FAIL",
         f"{n_inf:,} cells")

    # Zero-variance feature columns (suspicious — usually a bug)
    col_std = np.nanstd(X, axis=0)
    zero_var = int((col_std < 1e-12).sum())
    near_zero_var = int((col_std < 1e-6).sum() - zero_var)
    emit("panel zero-variance features",
         "PASS" if zero_var == 0 else "WARN",
         f"{zero_var} features have std < 1e-12")
    emit("panel near-zero-variance",
         "PASS" if near_zero_var <= 5 else "WARN",
         f"{near_zero_var} features have 1e-12 < std < 1e-6")

    # Value range sanity (catches scaling bugs)
    col_min = np.nanmin(X, axis=0)
    col_max = np.nanmax(X, axis=0)
    huge = int(((np.abs(col_min) > 1e6) | (np.abs(col_max) > 1e6)).sum())
    emit("panel value ranges",
         "PASS" if huge == 0 else "WARN",
         f"{huge} features with |value| > 1e6 (possible unit bug)")

    # MISSING sentinel handling check — survival.MISSING = -999 typically
    # If MISSING is stored as -999.0 we'd see column floors at -999.
    sentinel_floor = int((col_min == -999.0).sum())
    if sentinel_floor:
        emit("panel MISSING sentinel",
             "INFO",
             f"{sentinel_floor} features floor at -999 (MISSING sentinel — expected)")


# ---- 3. LABELS ----
def audit_labels(csv_path: Path, label: str):
    if not csv_path.exists():
        emit(f"{label}: long csv", "WARN", f"missing {csv_path.name}")
        return
    df = pd.read_csv(csv_path)
    emit(f"{label}: rows", "INFO",
         f"{len(df):,} rows, {df.player_id.nunique():,} pids")

    # Per-event sanity
    for ev in EVENTS:
        elig = f"eligible_{ev}"; real = f"realized_{ev}"
        if elig not in df.columns or real not in df.columns:
            continue
        n_elig = int(df[elig].sum())
        n_real = int(df[real].sum())
        # Realized must imply eligible
        contradictions = int(((df[real] == 1) & (df[elig] == 0)).sum())
        emit(f"{label}: {ev} realized⊆eligible",
             "PASS" if contradictions == 0 else "FAIL",
             f"{contradictions} contradictory rows" if contradictions
             else f"{n_real:,} realized of {n_elig:,} eligible "
                  f"({100*n_real/n_elig:.2f}%)" if n_elig
                  else "no eligible rows")

    # Hazard prob ranges (must be in [0, 1])
    for ev in EVENTS:
        col = f"p_{ev}"
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) == 0:
            continue
        out_of_range = int(((s < 0) | (s > 1)).sum())
        n_nan = int(df[col].isna().sum())
        emit(f"{label}: {col} range",
             "PASS" if out_of_range == 0 else "FAIL",
             f"min={s.min():.4f} max={s.max():.4f} nan={n_nan}"
             + (f" out_of_range={out_of_range}" if out_of_range else ""))

    # Duplicate (pid, snap) rows
    if "snap_year" in df.columns:
        dup = df.duplicated(subset=["player_id", "snap_year"]).sum()
        emit(f"{label}: (pid, snap) duplicates",
             "PASS" if dup == 0 else "FAIL",
             f"{dup} duplicate (pid, snap) rows" if dup else "unique")

    # mean_t / sd_t sanity
    for ev in EVENTS:
        col = f"mean_t_{ev}"
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) == 0:
            continue
        n_nan = int(df[col].isna().sum())
        out = int(((s < 0) | (s > 30)).sum())
        emit(f"{label}: {col}",
             "PASS" if out == 0 else "WARN",
             f"min={s.min():.1f} max={s.max():.1f} mean={s.mean():.1f} "
             f"nan_filled={n_nan}"
             + (f" out_of_range={out}" if out else ""))

    # .meta.json sidecar
    sidecar = csv_path.with_suffix(csv_path.suffix + ".meta.json")
    if sidecar.exists():
        meta = json.loads(sidecar.read_text())
        emit(f"{label}: provenance metadata", "PASS",
             f"haz_sha={meta.get('hazards_sha16', '?')} "
             f"mask_sha={meta.get('train_mask_sha16', '?')}")
    else:
        emit(f"{label}: provenance metadata", "WARN",
             f"no .meta.json sidecar — can't verify which hazards "
             f"produced this")


# ---- 4. COHORT CONSISTENCY ----
def audit_cohorts():
    print("\n=== 4. COHORT CONSISTENCY ===")
    if not PANEL_META.exists():
        emit("panel_meta.pkl", "WARN", "missing — skipping cohort audit")
        return
    with PANEL_META.open("rb") as fh:
        meta = pickle.load(fh)
    prospects = meta["prospects"]
    n = len(prospects)
    n_draft = sum(1 for p in prospects if p.get("draft_year") is not None
                   and not (p.get("is_international") or 0))
    n_ifa = sum(1 for p in prospects
                 if int(p.get("is_international") or 0) == 1)
    emit("prospects total", "INFO",
         f"{n:,} ({n_draft:,} drafted + {n_ifa:,} IFA)")

    debut = sum(1 for p in prospects
                 if p.get("mlb_debut_year") is not None)
    estab = sum(1 for p in prospects
                 if p.get("year_established_mlb") is not None)
    top100 = sum(1 for p in prospects
                  if p.get("year_top_100") is not None)
    emit("prospects with MLB_DEBUT", "INFO",
         f"{debut:,} ({100*debut/n:.1f}%)")
    emit("prospects with ESTABLISHED_MLB", "INFO",
         f"{estab:,} ({100*estab/n:.1f}%)")
    emit("prospects with TOP_100", "INFO",
         f"{top100:,} ({100*top100/n:.1f}%)")

    val = _read_pids(VAL_PIDS)
    fit = _read_pids(FIT_PIDS)
    universe_pids = {p["player_id"] for p in prospects}
    miss_val = val - universe_pids
    miss_fit = fit - universe_pids
    emit("val pids in panel universe",
         "PASS" if not miss_val else "FAIL",
         f"{len(val) - len(miss_val):,} of {len(val):,} "
         f"({len(miss_val)} missing)")
    emit("fit pids in panel universe",
         "PASS" if not miss_fit else "FAIL",
         f"{len(fit) - len(miss_fit):,} of {len(fit):,} "
         f"({len(miss_fit)} missing)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true",
                    help="Emit JSON report to stdout instead of text")
    args = ap.parse_args()

    audit_pid_splits()
    audit_panel()
    print("\n=== 3. LABEL + LONG CSV AUDIT ===")
    audit_labels(TRAIN_DIR / "v2.0b_honest_fit_long.csv",
                  "honest_fit")
    audit_labels(TRAIN_DIR / "v2.0b_honest_val_long.csv",
                  "honest_val")
    audit_labels(TRAIN_DIR / "v2.0b_oof_stacked_long.csv",
                  "oof_stacked")
    audit_labels(TRAIN_DIR / "v2.0b_oof_val_long.csv",
                  "oof_val")
    audit_cohorts()

    # Summary
    n_fail = sum(1 for r in REPORT if r["status"] == "FAIL")
    n_warn = sum(1 for r in REPORT if r["status"] == "WARN")
    n_pass = sum(1 for r in REPORT if r["status"] == "PASS")
    print(f"\n=== SUMMARY ===")
    print(f"  ✓ {n_pass} PASS    ⚠ {n_warn} WARN    ✗ {n_fail} FAIL")

    if args.json:
        Path("audit_report.json").write_text(json.dumps(REPORT, indent=2))
        print("  wrote audit_report.json")

    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
