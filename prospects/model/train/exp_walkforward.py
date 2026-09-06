"""Temporal walk-forward backtest — hazard layer, deployment-mimicking.

The random player split measures interpolation; deployment is extrapolation:
score entry-cohorts the model never saw, with a model whose LABELS stop six
years earlier (the 2020-cap / 2026-score relationship). This backtest
reproduces that geometry at three historical origins:

  For Y in {2012, 2014, 2016}:
    TRAIN  landmark hazards on players with entry <= Y, labels observed only
           through Y+6 (max_obs_year) — no future information at all.
    SCORE  snap = Y+6, players with entry in (Y, Y+6], not yet debuted —
           the exact analog of scoring the 2026 cohort today.
    EVAL   P(debut <= 3y) from the hazard curve vs realized debut by Y+9
           (fully resolved: Y+9 <= 2025). AP / AUC / calib + buckets.

Uses the panel cache (features are split-independent). Hazard HP = default
(production recipe). ~15 min per origin.

    python -m prospects.model.train.exp_walkforward
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from prospects import config
from prospects.config import REPO_ROOT
from prospects.model.hazards import landmark as lm
from prospects.model.pipelines.oof import _entry_year, stage_panel

OUT_DIR = REPO_ROOT / "runs" / "exp_walkforward"
DB = str(REPO_ROOT / "prospects_snapshot.db")
GAP = 6           # label cutoff -> scoring snap gap (2020 -> 2026)
EVAL_H = 3        # debut horizon evaluated


def bucket_print(p, y):
    edges = [0, .05, .10, .20, .30, .40, .60, .80, 1.001]
    lab = ["0-5", "5-10", "10-20", "20-30", "30-40", "40-60", "60-80",
           "80-100"]
    b = pd.cut(pd.Series(p), edges, labels=lab)
    print(f"    {'bucket':<8}{'n':>6}{'pred':>7}{'actual':>8}{'diff':>8}")
    for L in lab:
        m = (b == L).to_numpy()
        if m.sum() < 10:
            continue
        print(f"    {L:<8}{int(m.sum()):>6,}{p[m].mean():>6.1%}"
              f"{y[m].mean():>8.1%}{y[m].mean()-p[m].mean():>+8.1%}")


def run_origin(Y, X_lm, pids, S_yrs, joined, stats_by_pid, entry_by_pid,
               t0):
    snap = Y + GAP
    print(f"\n===== origin Y={Y}: train entry<={Y}, labels<= {snap}, "
          f"score snap={snap}, eval debut<= {EVAL_H}y "
          f"[{(time.time()-t0)/60:.0f}m] =====")
    train_mask = np.array(
        [entry_by_pid.get(p) is not None and entry_by_pid[p] <= Y
         for p in pids], dtype=bool)
    print(f"  train landmarks: {int(train_mask.sum()):,} rows")
    hazards = lm.fit_landmark_hazards(
        X_lm, joined, S_yrs, stats_by_pid, train_mask=train_mask,
        seed=42, max_obs_year=snap, verbose=False,
    )

    # scoring cohort: entry in (Y, snap], never trained on, not yet debuted
    seen: set = set()
    cohort = []
    for p in joined:
        pid = p["player_id"]
        if pid in seen:
            continue
        seen.add(pid)
        ent = entry_by_pid.get(pid)
        if ent is None or not (Y < ent <= snap):
            continue
        deb = p.get("mlb_debut_year")
        if deb is not None and deb <= snap:
            continue
        cohort.append(p)
    sub_stats = {
        p["player_id"]: [s for s in stats_by_pid.get(p["player_id"], [])
                         if (s.get("season_year") or 0) <= snap]
        for p in cohort
    }
    print(f"  scoring cohort: {len(cohort):,} players (entry {Y+1}-{snap})")
    out = lm.predict_cumulative_batch_landmark(
        hazards, cohort, sub_stats, current_year=snap, horizon=10)

    from prospects.core.schema import CareerEvent
    hk = out[("haz_k", CareerEvent.MLB_DEBUT)]
    cum3 = 1.0 - np.prod(1.0 - np.clip(hk[:, :EVAL_H], 0, 1), axis=1)
    deb_yr = np.array([p.get("mlb_debut_year") or 0 for p in cohort],
                      dtype=float)
    y = ((deb_yr > snap) & (deb_yr <= snap + EVAL_H)).astype(int)

    ap = float(average_precision_score(y, cum3)) if y.sum() else float("nan")
    auc = float(roc_auc_score(y, cum3)) if 0 < y.sum() < len(y) else float("nan")
    base = float(y.mean())
    calib = float(cum3.mean() / base) if base else float("nan")
    print(f"  n={len(y):,} pos={int(y.sum()):,} base={base:.1%}  "
          f"AP={ap:.4f} (lift {ap/base:.1f}x)  AUC={auc:.4f}  "
          f"calib={calib:.2f}")
    bucket_print(cum3, y)
    return {"Y": Y, "snap": snap, "n": len(y), "pos": int(y.sum()),
            "base": base, "ap": ap, "auc": auc, "calib": calib}


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--origins", nargs="*", type=int,
                     default=[2012, 2014, 2016])
    args = ap_.parse_args()
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X_lm, pids, S_yrs, joined, stats_by_pid = stage_panel(DB, 2020)
    entry_by_pid: dict = {}
    for p in joined:
        pid = p["player_id"]
        if pid not in entry_by_pid:
            entry_by_pid[pid] = _entry_year(p, stats_by_pid)

    results = [run_origin(Y, X_lm, pids, S_yrs, joined, stats_by_pid,
                          entry_by_pid, t0)
               for Y in args.origins]

    print(f"\n===== walk-forward summary (hazard layer, debut<= {EVAL_H}y) =====")
    print(f"{'Y':>5}{'snap':>6}{'n':>7}{'base%':>7}{'AP':>8}{'AUC':>8}"
          f"{'calib':>7}")
    for r in results:
        print(f"{r['Y']:>5}{r['snap']:>6}{r['n']:>7,}{r['base']*100:>6.1f}%"
              f"{r['ap']:>8.4f}{r['auc']:>8.4f}{r['calib']:>7.2f}")
    (OUT_DIR / "summary.json").write_text(json.dumps(
        {"results": results,
         "elapsed_min": round((time.time() - t0) / 60, 1)}, indent=2))
    print(f"\nwrote {OUT_DIR / 'summary.json'}  "
          f"({(time.time()-t0)/60:.0f} min)")


if __name__ == "__main__":
    main()
