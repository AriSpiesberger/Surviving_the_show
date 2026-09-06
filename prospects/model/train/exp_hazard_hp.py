"""EXPERIMENT 6: hazard-layer HP — the untouched source of ranking power.

Every gain so far came from reshaping what the hazard layer already knew
(exp1-4 live at the XGB refinement layer; F = exp4 is the champion). The
hazards themselves still train with sklearn-default-ish HistGBT (max_iter
200, depth 6, lr .05) — `fit_landmark_hazards(hazard_hp=...)` exists but
nothing wires it. This driver runs the REAL 6-fold OOF pipeline per HP
config, namespaced under runs/<tag>/, then trains the F-recipe XGB on the
new longs and scores the same val slice, so numbers are directly comparable
to F.

Comparability guarantees:
  - panel_cache.npz / panel_meta.pkl copied from runs/current (same features,
    same signature -> stage_panel cache-hits);
  - fold/train pid lists copied (identical player partition);
  - same val_pids.txt (runs/current), same seed, same max_entry;
  - XGB step reuses exp4's winner recipe (g3_slow HP + its 160 raw features)
    with the same internal ES split (rng seed 7), 2-seed bag, raw AP readout
    (prediction-first: no calibration in the screen).

The val-scoring hazards refit passes hazard_hp too (the stock Stage 5
refit would silently use default HP).

Sequential configs; each ~1-2h. All stages resume from checkpoints if the
process dies — just re-run. Writes runs/hz_*/ + runs/exp_hazard_hp/.
Touches nothing under runs/current/.

Usage:
    python -m prospects.model.train.exp_hazard_hp
    python -m prospects.model.train.exp_hazard_hp --configs hz1_capacity
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from prospects import config
from prospects.config import REPO_ROOT
from prospects.model.pipelines import oof as oof_mod
from prospects.model.joint import EVENTS, H_MAX, PUBLISH_H, prep_base
from prospects.model.train.joint_xgb import _assemble, _prep_train
from prospects.model.train.exp_cdf_timing import per_h_metrics, weighted_ap_at
from prospects.model.train.exp_cdf_timing2 import (
    FEAT2, debut_ap_at, stamp_extra_cols,
)
from prospects.model.train.exp_cdf_timing3 import load_panel_map, raw_rows_for
from prospects.model.train.exp_cdf_timing4 import (
    predict_rows, sweep_val, train_one,
)

CUR = config.run()
OUT_DIR = REPO_ROOT / "runs" / "exp_hazard_hp"
EXP4_BAG = REPO_ROOT / "runs" / "exp_cdf_timing4" / "joint_xgb_exp4_bag.pkl"
CUR_SCRATCH = CUR.scratch / "oof"

# F (exp4 champion) honest-val reference, for the printout.
F_REF = {"ap_h1": 0.5339, "ap_h2": 0.6397, "ap_h3": 0.6855,
         "ap_h6": 0.7256, "wap_h6": 0.5568}

G3_SLOW = {"max_depth": 8, "min_child_weight": 100,
           "colsample_bytree": 0.6, "learning_rate": 0.03}

CONFIGS: dict[str, dict] = {
    # historical default HP — the honest-baseline reference config
    "hz0_default": {},
    # more capacity, slower lr, leaf-count-limited depth
    "hz1_capacity": dict(max_iter=600, learning_rate=0.03, max_depth=None,
                         max_leaf_nodes=63, min_samples_leaf=50,
                         l2_regularization=1.0, early_stopping=True,
                         n_iter_no_change=15, validation_fraction=0.1),
    # wider-but-shallow trees, light regularization
    "hz2_wide": dict(max_iter=400, learning_rate=0.05, max_depth=None,
                     max_leaf_nodes=31, min_samples_leaf=20,
                     l2_regularization=0.1, early_stopping=True,
                     n_iter_no_change=10, validation_fraction=0.1),
    # saturation probe past hz1: hz1 (600 iters, 63 leaves) beat default HP
    # by a wide margin — find where more capacity stops paying.
    "hz3_max": dict(max_iter=1000, learning_rate=0.02, max_depth=None,
                    max_leaf_nodes=127, min_samples_leaf=80,
                    l2_regularization=1.0, early_stopping=True,
                    n_iter_no_change=20, validation_fraction=0.1),
}

K = 6
SEED = 42
MAX_ENTRY = 2020
DB = str(REPO_ROOT / "prospects_snapshot.db")


def _seed_scratch(tag_scratch: Path) -> None:
    """Copy the panel cache from runs/current so the tagged run cache-hits
    Stage 1. Fold pid lists are NOT copied (2026-09-05): the Aug-15 fold
    lists went stale against a regenerated val_pids.txt and put 90% of val
    inside training. stage_partition rebuilds folds deterministically from
    the CURRENT val set (and its new guard purges any stale cached lists)."""
    tag_scratch.mkdir(parents=True, exist_ok=True)
    for n in ("panel_cache.npz", "panel_meta.pkl"):
        src, dst = CUR_SCRATCH / n, tag_scratch / n
        if not dst.exists():
            if not src.exists():
                raise SystemExit(f"missing source artifact {src}")
            shutil.copy2(src, dst)
            print(f"  seeded {n}")


def run_config(tag: str, hp: dict, t0: float) -> dict:
    run = config.run(tag)
    scratch = run.scratch / "oof"
    _seed_scratch(scratch)
    run.training.mkdir(parents=True, exist_ok=True)
    run.models.mkdir(parents=True, exist_ok=True)

    # Point the oof module's globals at the tag namespace. VAL_PIDS_PATH
    # stays on runs/current (same held-out players as every experiment).
    oof_mod.SCRATCH = scratch
    oof_mod.TRAIN_DIR = run.training
    oof_mod.PANEL_NPZ = scratch / "panel_cache.npz"
    oof_mod.PANEL_META = scratch / "panel_meta.pkl"
    oof_mod.OOF_STACKED = run.oof_stacked_long
    oof_mod.OOF_VAL = run.oof_val_long
    oof_mod.XGB_OUT = run.joint_xgb          # unused (we run F-recipe here)
    oof_mod._FEATURE_MASK = None

    print(f"\n[{tag}] Stage 1: panel (cache-hit expected)")
    X_lm, pids, S_yrs, joined, stats_by_pid = oof_mod.stage_panel(DB, 2020)

    seen: set[str] = set()
    prospects_all: list[dict] = []
    for p in joined:
        if p["player_id"] not in seen:
            seen.add(p["player_id"])
            prospects_all.append(p)
    val_pid_set = {ln.strip()
                   for ln in oof_mod.VAL_PIDS_PATH.read_text().splitlines()
                   if ln.strip()}
    fold_sets = oof_mod.stage_partition(
        prospects_all, stats_by_pid, val_pid_set, K, SEED, MAX_ENTRY)

    fold_csvs = [scratch / f"fold{j}_long.csv" for j in range(K)]
    last_hazards, last_train_set = None, None
    for j in range(K):
        train_set: set[str] = set()
        for m in range(K):
            if m != j:
                train_set |= fold_sets[m]
        last_train_set = train_set
        if fold_csvs[j].exists():
            print(f"[{tag}] fold {j+1}/{K}: reusing {fold_csvs[j].name}")
            last_hazards = None
            continue
        print(f"[{tag}] fold {j+1}/{K}  [{(time.time()-t0)/60:,.0f}m]")
        n, last_hazards = oof_mod.run_one_fold(
            X_lm, pids, S_yrs, joined, stats_by_pid, prospects_all,
            train_set=train_set, score_set=fold_sets[j],
            out_csv=fold_csvs[j], max_entry_year=MAX_ENTRY, seed=SEED,
            partial_dir=scratch / f"fold{j}_partial",
            hazards_pkl=scratch / f"fold{j}_hazards.pkl",
            hazard_hp=hp,
        )
        print(f"[{tag}]   wrote {n:,} rows")

    if not oof_mod.OOF_STACKED.exists():
        print(f"[{tag}] stacking")
        stacked = pd.concat([pd.read_csv(f) for f in fold_csvs],
                            ignore_index=True)
        stacked.to_csv(oof_mod.OOF_STACKED, index=False)
        del stacked

    if not oof_mod.OOF_VAL.exists():
        print(f"[{tag}] scoring val  [{(time.time()-t0)/60:,.0f}m]")
        if last_hazards is None:
            hz_pkl = scratch / "val_hazards.pkl"
            if hz_pkl.exists():
                with hz_pkl.open("rb") as fh:
                    last_hazards = pickle.load(fh)
            else:
                train_mask = np.array([p in last_train_set for p in pids],
                                      dtype=bool)
                # NB: pass hazard_hp — the stock Stage 5 refit would not.
                last_hazards = oof_mod.lm.fit_landmark_hazards(
                    X_lm, joined, S_yrs, stats_by_pid,
                    train_mask=train_mask, seed=SEED, verbose=True,
                    hazard_hp=hp, feature_mask=None,
                )
                with hz_pkl.open("wb") as fh:
                    pickle.dump(last_hazards, fh,
                                protocol=pickle.HIGHEST_PROTOCOL)
        n = oof_mod._score_checkpointed(
            last_hazards, prospects_all, stats_by_pid, val_pid_set,
            oof_mod.OOF_VAL, scratch / "val_partial",
            max_entry_year=MAX_ENTRY,
            observe_through=oof_mod.MAX_OBS_YEAR, max_offset=10, horizon=15,
        )
        print(f"[{tag}]   wrote {n:,} val rows")
    del X_lm, last_hazards
    import gc; gc.collect()

    # ---- F-recipe XGB on the new longs ----------------------------------
    print(f"[{tag}] XGB (F recipe) on new longs  [{(time.time()-t0)/60:,.0f}m]")
    with open(EXP4_BAG, "rb") as fh:
        f_bundle = pickle.load(fh)
    keep_raw = f_bundle["keep_raw"]
    feats = list(FEAT2) + list(keep_raw)

    fit_base = _prep_train(pd.read_csv(oof_mod.OOF_STACKED), DB, MAX_ENTRY)
    val_base = prep_base(pd.read_csv(oof_mod.OOF_VAL), DB,
                         max_entry=MAX_ENTRY)
    fit_long, Y_fit = _assemble(fit_base, H_MAX)
    fit_long = stamp_extra_cols(fit_long)

    from prospects.features.scouting import FEATURE_NAMES
    X_lm2, key_to_row = load_panel_map()   # raw features (same panel)
    raw_fit, cov = raw_rows_for(fit_long, X_lm2, key_to_row)
    name_to_col = {f"rw_{n}": i for i, n in enumerate(FEATURE_NAMES)}
    raw_idx = [name_to_col[n] for n in keep_raw]
    X = np.hstack([fit_long[FEAT2].values.astype(np.float32),
                   raw_fit[:, raw_idx]])
    del raw_fit

    fit_pids = fit_long["player_id"].to_numpy()
    h_arr = fit_long["h"].astype(int).to_numpy()
    uniq = np.unique(fit_pids)
    rng = np.random.default_rng(7)
    es_players = set(rng.choice(uniq, size=max(1, len(uniq) // 10),
                                replace=False))
    es_mask = np.isin(fit_pids, list(es_players))

    bst = train_one(X[~es_mask], Y_fit[~es_mask], feats, G3_SLOW, 2000, 50,
                    SEED, X[es_mask], Y_fit[es_mask])
    bi = int(bst.best_iteration)
    P = predict_rows(bst, X[es_mask], feats)
    es_ap3 = debut_ap_at(h_arr[es_mask], Y_fit[es_mask], P, 3)
    print(f"[{tag}]   ES best_iter={bi} debut ap_h3={es_ap3:.4f}")
    del bst, P

    bag = []
    for s in range(2):
        bag.append(train_one(X, Y_fit, feats, G3_SLOW, bi + 1, 0,
                             SEED + 100 + s))
        print(f"[{tag}]   bag seed {s} done [{(time.time()-t0)/60:,.0f}m]")
    del X

    raw_val, _ = raw_rows_for(val_base, X_lm2, key_to_row)
    val_base = pd.concat([val_base, pd.DataFrame(
        {n: raw_val[:, name_to_col[n]] for n in keep_raw},
        index=val_base.index)], axis=1)
    del raw_val, X_lm2

    sv = sweep_val(bag, feats, val_base)
    for ev in EVENTS:
        cols = [f"xp_{ev}_h{h}" for h in range(1, H_MAX + 1)]
        sv[cols] = np.maximum.accumulate(
            sv[cols].to_numpy(dtype=np.float64), axis=1)
    rows = per_h_metrics(sv, "", tag)
    met = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    met.to_csv(OUT_DIR / f"{tag}_val_metrics.csv", index=False)
    with open(run.models / "joint_xgb_hz_bag.pkl", "wb") as fh:
        pickle.dump({"models": bag, "feature_names": feats,
                     "keep_raw": keep_raw, "events": list(EVENTS),
                     "hazard_hp": hp, "kind": f"exp6_{tag}"}, fh)

    deb = {h: met[(met.event == "MLB_DEBUT") & (met.h == h)].iloc[0]["ap"]
           for h in (1, 2, 3, 6)}
    wap = weighted_ap_at(
        met[met.h == PUBLISH_H].to_dict(orient="records"), PUBLISH_H)
    res = {"tag": tag, "hp": hp, "es_best_iter": bi, "es_ap_h3": es_ap3,
           "ap_h1": deb[1], "ap_h2": deb[2], "ap_h3": deb[3],
           "ap_h6": deb[6], "wap_h6": wap}
    print(f"[{tag}] VAL debut ap_h3={deb[3]:.4f} (F {F_REF['ap_h3']:.4f})  "
          f"wap_h6={wap:.4f} (F {F_REF['wap_h6']:.4f})")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    args = ap.parse_args()

    t0 = time.time()
    results = []
    for tag in args.configs:
        results.append(run_config(tag, CONFIGS[tag], t0))
        with open(OUT_DIR / "summary.json", "w") as fh:
            json.dump({"F_ref": F_REF, "results": results,
                       "elapsed_min": round((time.time() - t0) / 60, 1)},
                      fh, indent=2, default=str)

    print(f"\n===== hazard HP vs F (val, raw bag) =====")
    print(f"{'config':<16}{'AP_h1':>8}{'AP_h2':>8}{'AP_h3':>8}"
          f"{'AP_h6':>8}{'wAP_h6':>8}")
    print(f"{'F (exp4 ref)':<16}{F_REF['ap_h1']:>8.4f}{F_REF['ap_h2']:>8.4f}"
          f"{F_REF['ap_h3']:>8.4f}{F_REF['ap_h6']:>8.4f}"
          f"{F_REF['wap_h6']:>8.4f}")
    for r in results:
        print(f"{r['tag']:<16}{r['ap_h1']:>8.4f}{r['ap_h2']:>8.4f}"
              f"{r['ap_h3']:>8.4f}{r['ap_h6']:>8.4f}{r['wap_h6']:>8.4f}")
    print(f"\nwrote {OUT_DIR}  ({(time.time()-t0)/60:.0f} min)")


if __name__ == "__main__":
    main()
