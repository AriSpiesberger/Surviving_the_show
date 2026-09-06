"""EXPERIMENT 5 (probe): honest equal-weight blends of the D/E/F bags.

D = exp2 (FEAT2, deep HP, no raw features)
E = exp3 (raw-feature pass-through, e2_rawcol)
F = exp4 (raw features, g3_slow)

Equal weights only — no weight fitting on val, so the comparison stays
honest. Averages the RAW trajectories (pre-calibration), cummaxes the blend,
scores per_h_metrics on the val slice. Prediction-first readout: debut AP at
h=1/2/3/6 and the h=6 weighted composite.

Writes runs/exp_blend/blend_metrics.csv. Read-only wrt all other artifacts.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from prospects import config
from prospects.config import REPO_ROOT
from prospects.model.joint import EVENTS, H_MAX, PUBLISH_H, prep_base
from prospects.model.train.exp_cdf_timing import (
    per_h_metrics, weighted_ap_at,
)
from prospects.model.train.exp_cdf_timing2 import sweep_val as sweep_feat2
from prospects.model.train.exp_cdf_timing3 import load_panel_map, raw_rows_for
from prospects.model.train.exp_cdf_timing4 import sweep_val as sweep_raw

_RUN = config.run()
DB = str(config.model_db())
OUT_DIR = REPO_ROOT / "runs" / "exp_blend"
BAGS = {
    "D": REPO_ROOT / "runs" / "exp_cdf_timing2" / "joint_xgb_exp2_bag.pkl",
    "E": REPO_ROOT / "runs" / "exp_cdf_timing3" / "joint_xgb_exp3_bag.pkl",
    "F": REPO_ROOT / "runs" / "exp_cdf_timing4" / "joint_xgb_exp4_bag.pkl",
}


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    val_base = prep_base(pd.read_csv(_RUN.oof_val_long), DB, max_entry=2020)

    bundles = {}
    for k, p in BAGS.items():
        with open(p, "rb") as fh:
            bundles[k] = pickle.load(fh)
        print(f"  {k}: {p.name} ({len(bundles[k]['models'])} models, "
              f"{len(bundles[k]['feature_names'])} feats)")

    # attach the union of raw columns E and F need
    need_raw = sorted(set(bundles["E"].get("keep_raw", []))
                      | set(bundles["F"].get("keep_raw", [])))
    if need_raw:
        from prospects.features.scouting import FEATURE_NAMES
        X_lm, key_to_row = load_panel_map()
        raw_val, cov = raw_rows_for(val_base, X_lm, key_to_row)
        name_to_col = {f"rw_{n}": i for i, n in enumerate(FEATURE_NAMES)}
        val_base = pd.concat([val_base, pd.DataFrame(
            {n: raw_val[:, name_to_col[n]] for n in need_raw},
            index=val_base.index)], axis=1)
        print(f"  attached {len(need_raw)} raw cols (coverage {cov:.1%})")
        del X_lm, raw_val

    # per-model raw trajectory sweeps
    trajs = {}
    for k, b in bundles.items():
        sweep = sweep_feat2 if k == "D" else sweep_raw
        sv = sweep(b["models"], b["feature_names"], val_base)
        cols = {f"xp_{ev}_h{h}": sv[f"xp_{ev}_h{h}"].to_numpy()
                for ev in EVENTS for h in range(1, H_MAX + 1)}
        trajs[k] = cols
        print(f"  swept {k} [{time.time()-t0:,.0f}s]")

    def _frame(mix: list[str]) -> pd.DataFrame:
        out = val_base.copy()
        new = {}
        for ev in EVENTS:
            M = np.mean([np.column_stack(
                [trajs[k][f"xp_{ev}_h{h}"] for h in range(1, H_MAX + 1)])
                for k in mix], axis=0)
            M = np.maximum.accumulate(M, axis=1)
            for hi, h in enumerate(range(1, H_MAX + 1)):
                new[f"xp_{ev}_h{h}"] = M[:, hi]
        return pd.concat([out, pd.DataFrame(new, index=out.index)], axis=1)

    mixes = [["F"], ["E", "F"], ["D", "F"], ["D", "E", "F"]]
    rows = []
    for mix in mixes:
        name = "+".join(mix)
        rows += per_h_metrics(_frame(mix), "", name)
    met = pd.DataFrame(rows)
    met.to_csv(OUT_DIR / "blend_metrics.csv", index=False)

    print(f"\n===== blends (raw, equal weight) =====")
    print(f"{'mix':<10}{'AP_h1':>8}{'AP_h2':>8}{'AP_h3':>8}{'AP_h6':>8}"
          f"{'wAP_h6':>8}")
    for mix in mixes:
        name = "+".join(mix)
        deb = {h: met[(met.scorer == name) & (met.event == "MLB_DEBUT")
                      & (met.h == h)].iloc[0]["ap"] for h in (1, 2, 3, 6)}
        sub = met[(met.scorer == name) & (met.h == PUBLISH_H)]
        wap = weighted_ap_at(sub.to_dict(orient="records"), PUBLISH_H)
        print(f"{name:<10}{deb[1]:>8.4f}{deb[2]:>8.4f}{deb[3]:>8.4f}"
              f"{deb[6]:>8.4f}{wap:>8.4f}")
    print(f"\nwrote {OUT_DIR / 'blend_metrics.csv'}  "
          f"({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
