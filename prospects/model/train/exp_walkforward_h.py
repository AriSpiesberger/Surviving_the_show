"""Walk-forward test of H: train the hazard layer on ALL resolved landmark cells (2026-09-19).

The user's point: every label here is a landmark cell ("did the event fire in year S+k"),
known as soon as S+k has happened, so nothing requires an entry-year cutoff — recent
entrants simply contribute short horizons. Production caps the hazard panel at entry <= 2020
and lets recent cohorts in only through the joint layer's augmentation. exp_walkforward4
tried lifting the cap and rejected it, but in the leaky era (scout_servicetime, whose
coverage is strongly era-dependent) — and with the flaw below.

HARNESS FIX. exp_walkforward2/3/4 (and exp_macro_bc's stack arms) score the joint layer's
TRAINING rows with hazards fit on those same players. The joint model therefore trains on
in-sample hazard features and is evaluated on out-of-sample ones, which handicaps every
stacked arm and none of the single-stage ones. Production avoids this with K-fold OOF
hazards; so does this script.

Per origin Y (score the entry-(Y, Y+6] cohort at snap = Y+6, label = debut within 3 more
years), two regimes, identical otherwise:

  capY   hazards see entry <= Y            (fair control; production's policy)
  fresh  hazards see entry <= Y+5          (H: every landmark cell resolved by the snap)

In each: K-fold player OOF hazards -> the training long; a full-fit hazard model scores the
players it did not train on and the eval snapshot (as production does); then the production
joint recipe (FEAT2 + kept raw, 2-fold cross-fit calibrator, refit on 100%). The single-stage
B1 predictions from exp_macro_bc are reused for the blends (they have no hazard inputs).

    python -m prospects.model.train.exp_walkforward_h
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from prospects.config import REPO_ROOT
from prospects.features.scouting import FEATURE_NAMES
from prospects.model.hazards import landmark as lm
from prospects.model.joint import H_MAX, prep_base
from prospects.model.joint2 import attach_raw_features
from prospects.model.pipelines import oof as oof_mod
from prospects.model.pipelines.oof import _entry_year, stage_panel
from prospects.model.train.exp_cdf_timing import _logit
from prospects.model.train.exp_cdf_timing2 import stamp_extra_cols
from prospects.model.train.exp_macro_bc import OUT_DIR as BC_DIR, feature_sets, run_variant
from prospects.model.train.exp_walkforward2 import DB, EVAL_H, GAP, _metrics, bucket_rows
from prospects.model.train.joint_xgb import _assemble

OUT_DIR = REPO_ROOT / "runs" / "exp_walkforward_h"


def build_longs(Y, regime, cap, K, X_lm, pids, S_yrs, joined, stats_by_pid, entry_by_pid,
                prospects_all, log):
    """OOF training long(s), the rest-cohort long, and the eval long for one regime."""
    snap = Y + GAP
    odir = OUT_DIR / f"Y{Y}" / regime
    odir.mkdir(parents=True, exist_ok=True)
    train_pids = sorted(p for p, e in entry_by_pid.items() if e is not None and e <= cap)
    rest_pids = {p for p, e in entry_by_pid.items() if e is not None and cap < e <= snap}
    eval_pids = {p for p, e in entry_by_pid.items() if e is not None and Y < e <= snap}
    rng = np.random.default_rng(11)
    fold_of = {p: i % K for i, p in enumerate(rng.permutation(train_pids))}
    pid_arr = np.asarray(pids)
    in_train = np.array([p in fold_of for p in pid_arr])
    row_fold = np.array([fold_of.get(p, -1) for p in pid_arr])

    fold_csvs = []
    for f in range(K):
        csv = odir / f"oof_fold{f}.csv"
        fold_csvs.append(csv)
        if csv.exists():
            continue
        hz = lm.fit_landmark_hazards(X_lm, joined, S_yrs, stats_by_pid,
                                     train_mask=in_train & (row_fold != f),
                                     seed=42, max_obs_year=snap, verbose=False)
        n = oof_mod._score_checkpointed(
            hz, prospects_all, stats_by_pid, {p for p, g in fold_of.items() if g == f}, csv,
            odir / f"oof_fold{f}_partial", max_entry_year=cap, observe_through=snap,
            max_offset=10, horizon=15)
        del hz
        log(f"    [{regime}] fold {f}: {n:,} OOF rows")

    rest_csv, eval_csv = odir / "rest_long.csv", odir / "eval_long.csv"
    if not (rest_csv.exists() and eval_csv.exists()):
        hz_pkl = odir / "hazards_full.pkl"
        if hz_pkl.exists():
            hz = pickle.load(open(hz_pkl, "rb"))
        else:
            hz = lm.fit_landmark_hazards(X_lm, joined, S_yrs, stats_by_pid, train_mask=in_train,
                                         seed=42, max_obs_year=snap, verbose=False)
            pickle.dump(hz, open(hz_pkl, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        if not rest_csv.exists():
            if rest_pids:
                oof_mod._score_checkpointed(hz, prospects_all, stats_by_pid, rest_pids, rest_csv,
                                            odir / "rest_partial", max_entry_year=snap,
                                            observe_through=snap, max_offset=10, horizon=15)
            else:
                pd.DataFrame().to_csv(rest_csv, index=False)
        if not eval_csv.exists():
            # production scores the live snapshot with the full-fit hazards, including
            # players whose earlier landmarks it trained on
            oof_mod._score_checkpointed(hz, prospects_all, stats_by_pid, eval_pids, eval_csv,
                                        odir / "eval_partial", max_entry_year=snap,
                                        observe_through=snap, max_offset=10, horizon=15)
        del hz
    log(f"    [{regime}] train players {len(train_pids):,} | rest {len(rest_pids):,} | eval {len(eval_pids):,}")
    return fold_csvs, rest_csv, eval_csv


def frames(Y, fold_csvs, rest_csv, eval_csv, keep_raw):
    snap = Y + GAP
    parts = [pd.read_csv(c, low_memory=False) for c in fold_csvs]
    if rest_csv.stat().st_size > 5:
        r = pd.read_csv(rest_csv, low_memory=False)
        parts.append(r[r.snap_year < snap])
    both = prep_base(pd.concat(parts, ignore_index=True), DB)
    both = both[both.snap_year < snap].copy()
    for ev in ("TOP_100_PROSPECT", "MLB_DEBUT"):
        col = f"eligible_{ev}"
        if col in both.columns:
            both = both[both[col] == 1]
    both = attach_raw_features(both, DB, keep_raw, verbose=False)
    fit_long, Y_fit = _assemble(both, H_MAX)
    fit_long = stamp_extra_cols(fit_long)

    ev_base = prep_base(pd.read_csv(eval_csv, low_memory=False), DB)
    ev_base = ev_base[(ev_base.snap_year == snap)
                      & (ev_base.get("eligible_MLB_DEBUT", 1) == 1)].copy()
    ev_base = attach_raw_features(ev_base, DB, keep_raw, verbose=False)
    trig = pd.to_numeric(ev_base["trigger_MLB_DEBUT"], errors="coerce")
    y = ((trig > snap) & (trig <= snap + EVAL_H)).fillna(False).to_numpy().astype(int)
    return fit_long, Y_fit, ev_base, y


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--origins", nargs="*", type=int, default=[2016, 2014, 2012])
    ap.add_argument("--folds", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(f"{m}  [{(time.time() - t0) / 60:.0f}m]", flush=True)

    with open(REPO_ROOT / "runs" / "current" / "models" / "joint_xgb_v2.3.pkl", "rb") as fh:
        keep_raw = list(pickle.load(fh)["keep_raw"])
    live = {f"rw_{n}" for n in FEATURE_NAMES}
    keep_raw = [c for c in keep_raw if c in live]
    feats_A = feature_sets(keep_raw)["A"]

    X_lm, pids, S_yrs, joined, stats_by_pid = stage_panel(DB, 2020)
    entry_by_pid, prospects_all, seen = {}, [], set()
    for p in joined:
        pid = p["player_id"]
        if pid not in seen:
            seen.add(pid)
            prospects_all.append(p)
            entry_by_pid[pid] = _entry_year(p, stats_by_pid)

    results = []
    for Y in args.origins:
        snap = Y + GAP
        log(f"\n===== origin Y={Y} (score snap {snap}) =====")
        preds = {}
        for regime, cap in (("capY", Y), ("fresh", snap - 1)):
            fold_csvs, rest_csv, eval_csv = build_longs(
                Y, regime, cap, args.folds, X_lm, pids, S_yrs, joined, stats_by_pid,
                entry_by_pid, prospects_all, log)
            fit_long, Y_fit, ev_base, y = frames(Y, fold_csvs, rest_csv, eval_csv, keep_raw)
            recent = int((fit_long["entry_year"] > Y).sum())
            log(f"    [{regime}] joint rows {len(fit_long):,} ({recent:,} from entries > {Y}); "
                f"eval n={len(y):,} pos={int(y.sum()):,}")
            name = f"A_oof_{regime}"
            r = run_variant(name, feats_A, fit_long, Y_fit, ev_base, y, Y, None, False, t0,
                            lambda m: print(m, flush=True))
            r.update({"Y": Y, "regime": regime, "n": int(len(y)), "base": float(y.mean())})
            results.append(r)
            z = np.load(BC_DIR / f"preds_Y{Y}_{name}.npz", allow_pickle=True)
            preds[regime] = pd.Series(z["cal3"], index=z["pid"])
            ytrue = pd.Series(y, index=ev_base["player_id"].to_numpy())
            del fit_long, Y_fit, ev_base
            (OUT_DIR / "results.json").write_text(json.dumps(results, indent=1))

        # blends with the single-stage B1 (no hazard inputs), aligned by player
        b1f = BC_DIR / f"preds_Y{Y}_B1.npz"
        if b1f.exists():
            zb = np.load(b1f, allow_pickle=True)
            b1 = pd.Series(zb["cal3"], index=zb["pid"])
            for regime, p in preds.items():
                idx = p.index.intersection(b1.index)
                yy = ytrue.reindex(idx).to_numpy()
                for nm, pr in ((f"B1 (same {len(idx):,} players)", b1.reindex(idx).to_numpy()),
                               (f"blend A_oof_{regime} 50 / B1 50",
                                1 / (1 + np.exp(-(0.5 * _logit(p.reindex(idx).to_numpy())
                                                  + 0.5 * _logit(b1.reindex(idx).to_numpy())))))):
                    if nm.startswith("B1") and regime != "capY":
                        continue
                    r = {"variant": nm, **_metrics(pr, yy, nm), "buckets": bucket_rows(pr, yy),
                         "Y": Y, "regime": regime, "n": int(len(idx)), "base": float(yy.mean())}
                    results.append(r)
            (OUT_DIR / "results.json").write_text(json.dumps(results, indent=1))

    df = pd.DataFrame([{k: r.get(k) for k in ("Y", "variant", "ap", "auc", "calib")} for r in results])
    df["variant"] = df["variant"].str.replace(r" \(same .*\)", "", regex=True)
    pd.set_option("display.width", 200)
    for m, lab in (("ap", "AP"), ("auc", "AUC"), ("calib", "calibration pred/actual")):
        piv = df.pivot_table(index="variant", columns="Y", values=m)
        if m != "calib":
            piv["mean"] = piv.mean(axis=1)
        print(f"\n===== OUT-OF-ERA debut<=3y: {lab} =====")
        print(piv.round(4).to_string())
    print("\nreference (exp_macro_bc, in-sample hazard features): A .5591/.3914/.5931  B1 .6538/.3721/.6508")
    df.to_csv(OUT_DIR / "summary.csv", index=False)
    log("\ndone")


if __name__ == "__main__":
    main()
