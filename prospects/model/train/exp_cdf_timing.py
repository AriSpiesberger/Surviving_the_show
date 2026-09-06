"""EXPERIMENT: monotone-h conditional XGB + OOF-fit h-covariate calibration
+ CDF-derived debut timing.

Hypothesis (three linked changes vs v2.1c):
  1. `monotone_constraints` on h_centered inside the XGB replaces the post-hoc
     cummax. cummax takes a max over noise, biasing long-h up and corrupting
     the implied per-year pmf; the in-model constraint regularizes instead.
  2. Calibrators move OFF the val slice. v2.1c fits per-(event,h) calibrators
     on the same 10% val that early-stops the XGB and tunes thresholds —
     triple-spent, so reported calibration is optimistic and the calibrators
     see only ~2k rows. Here: 5-fold player-grouped cross-fit over the FIT
     longs generates honest predictions for every fit row, and ONE calibrator
     per event with h as a covariate (logistic on [logit(p), h_c,
     logit(p)*h_c]) is fit on those. Smooth in h, pools strength for rare
     events, and val stays purely a reporting set.
  3. Timing is read off the calibrated trajectory instead of the separate
     Lasso: pmf_j = cal_xp_hj - cal_xp_h(j-1); conditional E[T], median and
     the q25-q75 debut window follow. If the CDF is calibrated, timing is
     calibrated by construction.

Baseline A = stored runs/current/models/joint_xgb.pkl (+ its val-fit
calibrators, labeled OPTIMISTIC in the output). Experiment C trains on the
same FEAT_COND rows; early stopping uses an internal 10% player slice of fit
(never val), then refits on 100% of fit at that round count.

Writes to runs/exp_cdf_timing/ (models + metric CSVs + summary.json).
Touches nothing under runs/current/.

Usage:
    python -m prospects.model.train.exp_cdf_timing            # full run
    python -m prospects.model.train.exp_cdf_timing --quick    # smoke test
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, roc_auc_score,
)

from prospects import config
from prospects.config import REPO_ROOT
from prospects.model.joint import (
    EVENTS, FEAT_COND, H_MAX, PUBLISH_H, add_cond_cols, predict_trajectory,
    prep_base, realized_by_h,
)
from prospects.model.train.joint_xgb import _assemble, _prep_train

EVENT_WEIGHTS = {"TOP_100_PROSPECT": 1.0, "MLB_DEBUT": 2.0,
                 "ESTABLISHED_MLB": 1.0, "STAR_PLUS_ELITE": 1.0}
EPS = 1e-6

_RUN = config.run()
DB = str(config.model_db())
OUT_DIR = REPO_ROOT / "runs" / "exp_cdf_timing"

XGB_PARAMS_BASE = {
    "tree_method": "hist",
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 6,
    "learning_rate": 0.05,
    "min_child_weight": 30,
    "reg_lambda": 1.0,
    "verbosity": 0,
}
NUM_ROUNDS = 800
EARLY_STOP = 25


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _monotone_string(feat_names: list[str]) -> str:
    """+1 on h_centered only: every head's cumulative P(event by h) is
    monotone non-decreasing in h. (Deliberately NOT constraining the
    haz_cum anchors — with multi-label training the same constraint would
    apply to every output head, and cross-event monotonicity isn't a truth.)
    """
    cons = [1 if n == "h_centered" else 0 for n in feat_names]
    return "(" + ",".join(str(c) for c in cons) + ")"


def _train_xgb(X_tr, Y_tr, X_es=None, Y_es=None, monotone=True,
               num_rounds=NUM_ROUNDS, seed=42, multi_output_tree=False):
    params = dict(XGB_PARAMS_BASE)
    params["seed"] = seed
    if multi_output_tree:
        params["multi_strategy"] = "multi_output_tree"
    if monotone:
        params["monotone_constraints"] = _monotone_string(FEAT_COND)
    dtr = xgb.DMatrix(X_tr, label=Y_tr, feature_names=FEAT_COND)
    kw = {}
    evals = [(dtr, "train")]
    if X_es is not None:
        des = xgb.DMatrix(X_es, label=Y_es, feature_names=FEAT_COND)
        evals.append((des, "es"))
        kw = dict(early_stopping_rounds=EARLY_STOP)
    bst = xgb.train(params, dtr, num_boost_round=num_rounds,
                    evals=evals, verbose_eval=False, **kw)
    return bst


def _predict_rows(bst, X, best_iter=None):
    d = xgb.DMatrix(X, feature_names=FEAT_COND)
    if best_iter is not None:
        return bst.predict(d, iteration_range=(0, best_iter + 1))
    return bst.predict(d)


def predict_trajectory_c(bst, df: pd.DataFrame, h_max: int = H_MAX,
                         cummax: bool = True) -> pd.DataFrame:
    """Sweep h=1..h_max with the experiment booster (no scaler). Emits
    xp_<ev>_h{h}. cummax kept as a defensive no-op — with the monotone
    constraint it should change nothing (violation stats are reported)."""
    out = df.copy()
    preds_by_h = {}
    for h in range(1, h_max + 1):
        sub = add_cond_cols(df, h)
        X = sub[FEAT_COND].values.astype(np.float32)
        preds_by_h[h] = _predict_rows(bst, X)
    for k, ev in enumerate(EVENTS):
        M = np.column_stack([preds_by_h[h][:, k] for h in range(1, h_max + 1)])
        if cummax:
            M = np.maximum.accumulate(M, axis=1)
        for hi, h in enumerate(range(1, h_max + 1)):
            out[f"xp_{ev}_h{h}"] = M[:, hi]
    return out


# --------------------------------------------------------------------------
# h-covariate calibrator
# --------------------------------------------------------------------------
class HCovCalibrator:
    """One calibrator per event across ALL horizons:
        p_cal = sigmoid(b0 + b1*logit(p) + b2*h_c + b3*logit(p)*h_c)
    Fit on honest (cross-fitted) predictions. Monotone in p at fixed h as
    long as (b1 + b3*h_c) > 0, which the fit produces in practice; cross-h
    monotonicity of a trajectory is re-enforced by cummax after applying."""

    def __init__(self):
        self.lr = LogisticRegression(C=1e4, solver="lbfgs", max_iter=2000)

    @staticmethod
    def _feats(p, h):
        lp = _logit(p)
        hc = np.asarray(h, dtype=np.float64) - 5.0
        return np.column_stack([lp, hc, lp * hc])

    def fit(self, p, h, y):
        self.lr.fit(self._feats(p, h), np.asarray(y, dtype=int))
        return self

    def predict(self, p, h):
        return self.lr.predict_proba(self._feats(p, h))[:, 1]


def apply_hcov_trajectory(df: pd.DataFrame, cals: dict,
                          h_max: int = H_MAX) -> pd.DataFrame:
    """Calibrate xp_<ev>_h{h} -> cal_xp_<ev>_h{h}, then cummax across h so
    the calibrated trajectory is a valid CDF."""
    out = df.copy()
    for ev in EVENTS:
        cal = cals.get(ev)
        cols = [f"xp_{ev}_h{h}" for h in range(1, h_max + 1)]
        M = out[cols].to_numpy(dtype=np.float64).copy()
        if cal is not None:
            for hi, h in enumerate(range(1, h_max + 1)):
                M[:, hi] = cal.predict(M[:, hi], np.full(len(out), h))
        M = np.maximum.accumulate(M, axis=1)
        for hi, h in enumerate(range(1, h_max + 1)):
            out[f"cal_xp_{ev}_h{h}"] = M[:, hi]
    return out


def apply_baseline_calibrators(df: pd.DataFrame, cal_pkl: Path,
                               h_max: int = H_MAX) -> pd.DataFrame:
    """v2.1c per-(event,h) calibrators (fit ON val -> optimistic here).
    Applied per-h independently, exactly as buylist.build does; NO cross-h
    re-monotonization, so we can also measure the inversions it creates."""
    with open(cal_pkl, "rb") as fh:
        cals = pickle.load(fh)["calibrators"]
    out = df.copy()
    for ev in EVENTS:
        for h in range(1, h_max + 1):
            col = f"xp_{ev}_h{h}"
            c = cals.get((ev, h))
            v = out[col].astype(float).to_numpy()
            out[f"calA_xp_{ev}_h{h}"] = c.predict(v) if c is not None else v
    return out


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def per_h_metrics(val: pd.DataFrame, prefix: str, scorer: str,
                  h_max: int = H_MAX) -> list[dict]:
    rows = []
    for ev in EVENTS:
        elig = (val[f"eligible_{ev}"] == 1) if f"eligible_{ev}" in val.columns \
            else pd.Series(True, index=val.index)
        for h in range(1, h_max + 1):
            col = f"{prefix}xp_{ev}_h{h}"
            if col not in val.columns:
                continue
            d = val[elig & (val["years_fwd"] >= h)].copy()
            d = d[d[col].notna()]
            if not len(d):
                continue
            y = realized_by_h(d, ev, h).astype(int)
            p = d[col].astype(float).to_numpy()
            pos = int(y.sum())
            n = len(d)
            rows.append({
                "scorer": scorer, "event": ev, "h": h, "n": n, "pos": pos,
                "base": pos / n,
                "auc": float(roc_auc_score(y, p)) if 0 < pos < n else np.nan,
                "ap": float(average_precision_score(y, p)) if pos else np.nan,
                "brier": float(brier_score_loss(y, p)),
                "calib": float(p.mean() / y.mean()) if pos else np.nan,
            })
    return rows


def weighted_ap_at(rows: list[dict], h: int) -> float:
    num = den = 0.0
    for r in rows:
        if r["h"] != h or not np.isfinite(r.get("ap", np.nan)):
            continue
        w = EVENT_WEIGHTS.get(r["event"], 1.0)
        num += w * r["ap"]
        den += w
    return num / den if den else np.nan


def inversion_stats(df: pd.DataFrame, prefix: str,
                    h_max: int = H_MAX) -> dict:
    """Cross-h monotonicity violations of a trajectory (pre any cummax we
    add): fraction of snaps with any decrease, and mean total decrease."""
    out = {}
    for ev in EVENTS:
        cols = [f"{prefix}xp_{ev}_h{h}" for h in range(1, h_max + 1)]
        if not all(c in df.columns for c in cols):
            continue
        M = df[cols].to_numpy(dtype=np.float64)
        drops = np.clip(-np.diff(M, axis=1), 0.0, None)
        out[ev] = {"frac_snaps_with_inversion": float((drops.max(axis=1) > 1e-9).mean()),
                   "mean_total_drop": float(drops.sum(axis=1).mean()),
                   "max_drop": float(drops.max())}
    return out


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------
def cdf_timing(df: pd.DataFrame, ev: str, prefix: str,
               h_max: int = H_MAX) -> pd.DataFrame:
    """Debut-time distribution from a (calibrated) trajectory.
    pmf_j = F(j) - F(j-1); conditional on event-by-h_max:
      t_mean = sum j*pmf / F(h_max)
      t_med / t_q25 / t_q75 = first j where F(j)/F(h_max) >= q
    """
    cols = [f"{prefix}xp_{ev}_h{h}" for h in range(1, h_max + 1)]
    F = df[cols].to_numpy(dtype=np.float64)
    F = np.maximum.accumulate(F, axis=1)
    Ftot = F[:, -1]
    pmf = np.diff(np.column_stack([np.zeros(len(F)), F]), axis=1)
    j = np.arange(1, h_max + 1, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_mean = (pmf * j).sum(axis=1) / Ftot
        Fc = F / Ftot[:, None]
    def _q(q):
        idx = (Fc >= q).argmax(axis=1) + 1  # first h crossing q
        return np.where(Ftot > 1e-9, idx.astype(float), np.nan)
    out = pd.DataFrame({
        "t_mean": np.where(Ftot > 1e-9, t_mean, np.nan),
        "t_med": _q(0.5), "t_q25": _q(0.25), "t_q75": _q(0.75),
    }, index=df.index)
    return out


def _timing_feats_for_lasso(df: pd.DataFrame, db: str) -> pd.DataFrame:
    """Replicate time_to_debut.add_feats (calendar-year age) so the stored
    Lasso timing model scores on the features it was trained with."""
    import sqlite3
    c = sqlite3.connect(db)
    birth = pd.read_sql("SELECT player_id, birth_date FROM prospects", c)
    c.close()
    birth["birth_year"] = pd.to_datetime(birth["birth_date"],
                                         errors="coerce").dt.year
    df = df.merge(birth[["player_id", "birth_year"]], on="player_id",
                  how="left")
    df["age_at_snap_centered_cal"] = (
        (df["snap_year"] - df["birth_year"]).fillna(22.0) - 22)
    df["yip_centered_cal"] = df["snap_offset"] - 3
    return df


def score_lasso_timing(df: pd.DataFrame, timing_pkl: Path,
                       db: str) -> np.ndarray | None:
    try:
        with open(timing_pkl, "rb") as fh:
            m = pickle.load(fh)
    except FileNotFoundError:
        return None
    sc, lasso, feat = m["scaler"], m["lasso"], m["feature_names"]
    d = _timing_feats_for_lasso(df.copy(), db)
    # The timing model's own naming (add_feats) collides with prep_base's;
    # rebuild its exact columns from the calendar-age versions.
    d["age_at_snap_centered"] = d["age_at_snap_centered_cal"]
    d["yip_centered"] = d["yip_centered_cal"]
    d["years_in_pro"] = d["snap_offset"]
    for col in feat:
        if col in d.columns:
            continue
        if col.endswith("_x_yip_centered"):
            base = col[:-len("_x_yip_centered")]
            if base in d.columns:
                d[col] = d[base] * d["yip_centered"]
    missing = [c for c in feat if c not in d.columns]
    if missing:
        print(f"  [timing] lasso baseline skipped, missing {missing}")
        return None
    X = d[feat].astype(float).values
    ok = np.isfinite(X).all(axis=1)
    pred = np.full(len(d), np.nan)
    if ok.any():
        pred[ok] = lasso.predict(sc.transform(X[ok]))
    return pred


def timing_report(val: pd.DataFrame, estimators: dict[str, np.ndarray],
                  windows: dict[str, tuple[np.ndarray, np.ndarray]],
                  h_max: int = H_MAX) -> pd.DataFrame:
    """Debut-timing metrics among val debutees (eligible pre-debut snaps,
    1 <= dt <= h_max)."""
    trig = pd.to_numeric(val["trigger_MLB_DEBUT"], errors="coerce")
    dt = (trig - val["snap_year"]).astype(float)
    mask = (val["eligible_MLB_DEBUT"] == 1) & trig.notna() \
        & (dt >= 1) & (dt <= h_max)
    dt = dt[mask].to_numpy()
    rows = []
    for name, pred in estimators.items():
        if pred is None:
            continue
        p = np.asarray(pred, dtype=float)[mask.to_numpy()]
        ok = np.isfinite(p)
        if ok.sum() < 30:
            continue
        mae = float(np.abs(p[ok] - dt[ok]).mean())
        rho = float(spearmanr(p[ok], dt[ok]).correlation)
        bias = float((p[ok] - dt[ok]).mean())
        row = {"estimator": name, "n": int(ok.sum()), "mae": mae,
               "spearman": rho, "bias": bias}
        if name in windows:
            lo, hi = windows[name]
            lo = np.asarray(lo, dtype=float)[mask.to_numpy()]
            hi = np.asarray(hi, dtype=float)[mask.to_numpy()]
            okw = ok & np.isfinite(lo) & np.isfinite(hi)
            row["q25_q75_coverage"] = float(
                ((dt >= lo) & (dt <= hi))[okw].mean())
            row["mean_window_width"] = float((hi - lo)[okw].mean())
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default=str(_RUN.oof_stacked_long))
    ap.add_argument("--val", default=str(_RUN.oof_val_long))
    ap.add_argument("--db", default=DB)
    ap.add_argument("--max-entry", type=int, default=2020)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true",
                    help="Smoke test: subsample fit players 15%, 2 folds, "
                         "120 rounds.")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"[exp] loading longs\n  fit: {args.fit}\n  val: {args.val}")
    fit_base = _prep_train(pd.read_csv(args.fit), args.db, args.max_entry)
    val_base = prep_base(pd.read_csv(args.val), args.db,
                         max_entry=args.max_entry)

    num_rounds = NUM_ROUNDS
    folds = args.folds
    if args.quick:
        rng = np.random.default_rng(0)
        pids = fit_base["player_id"].unique()
        keep = set(rng.choice(pids, size=max(200, len(pids) // 7),
                              replace=False))
        fit_base = fit_base[fit_base["player_id"].isin(keep)]
        num_rounds, folds = 120, 2
        print(f"  [quick] fit subsampled to {fit_base.player_id.nunique():,} "
              f"players")

    fit_long, Y_fit = _assemble(fit_base, H_MAX)
    print(f"  fit long: {len(fit_long):,} (row,h) rows / "
          f"{fit_long.player_id.nunique():,} players")
    X_fit = fit_long[FEAT_COND].values.astype(np.float32)
    fit_pids = fit_long["player_id"].to_numpy()

    # ---- C: monotone XGB. Step 1: pick rounds via internal 10% player ES.
    uniq = np.unique(fit_pids)
    rng = np.random.default_rng(7)
    es_players = set(rng.choice(uniq, size=max(1, len(uniq) // 10),
                                replace=False))
    es_mask = np.isin(fit_pids, list(es_players))
    print(f"\n[exp] C step 1: ES split {len(uniq) - len(es_players):,}/"
          f"{len(es_players):,} players; training monotone XGB "
          f"(rounds<={num_rounds})")
    bst_es = _train_xgb(X_fit[~es_mask], Y_fit[~es_mask],
                        X_fit[es_mask], Y_fit[es_mask],
                        monotone=True, num_rounds=num_rounds, seed=args.seed)
    best_iter = int(bst_es.best_iteration)
    print(f"  best_iteration = {best_iter}")

    # Step 2: refit on 100% of fit at best_iter+1 rounds.
    print(f"[exp] C step 2: refit on 100% fit at {best_iter + 1} rounds")
    bst_c = _train_xgb(X_fit, Y_fit, monotone=True,
                       num_rounds=best_iter + 1, seed=args.seed)
    with open(out_dir / "joint_xgb_mono.pkl", "wb") as fh:
        pickle.dump({"model": bst_c, "feature_names": list(FEAT_COND),
                     "events": list(EVENTS), "h_max": H_MAX,
                     "publish_h": PUBLISH_H, "monotone": "h_centered:+1",
                     "num_rounds": best_iter + 1,
                     "kind": "exp_cdf_timing_monotone"}, fh)

    # Step 3: cross-fit for honest calibration data.
    print(f"[exp] C step 3: {folds}-fold player-grouped cross-fit")
    fold_of = {p: i % folds for i, p in enumerate(
        rng.permutation(uniq))}
    fold_idx = np.array([fold_of[p] for p in fit_pids])
    oof_pred = np.full((len(fit_long), len(EVENTS)), np.nan, dtype=np.float64)
    for f in range(folds):
        tr, ho = fold_idx != f, fold_idx == f
        bst_f = _train_xgb(X_fit[tr], Y_fit[tr], monotone=True,
                           num_rounds=best_iter + 1, seed=args.seed + f)
        oof_pred[ho] = _predict_rows(bst_f, X_fit[ho])
        print(f"  fold {f}: trained {int(tr.sum()):,} rows, "
              f"predicted {int(ho.sum()):,}  "
              f"[{time.time() - t0:,.0f}s]")

    # Step 4: h-covariate calibrators on cross-fitted predictions.
    print(f"[exp] C step 4: h-covariate calibrators per event")
    h_arr = fit_long["h"].astype(int).to_numpy()
    cals: dict = {}
    for k, ev in enumerate(EVENTS):
        elig = (fit_long[f"eligible_{ev}"] == 1).to_numpy() \
            if f"eligible_{ev}" in fit_long.columns else np.ones(len(fit_long), bool)
        ok = elig & np.isfinite(oof_pred[:, k])
        y = Y_fit[ok, k].astype(int)
        if y.sum() < 25 or y.sum() == len(y):
            print(f"  {ev:<22} skip (pos={int(y.sum())})")
            continue
        cals[ev] = HCovCalibrator().fit(oof_pred[ok, k], h_arr[ok], y)
        print(f"  {ev:<22} n={int(ok.sum()):,} pos={int(y.sum()):,} "
              f"coef={np.round(cals[ev].lr.coef_.ravel(), 3).tolist()} "
              f"b0={cals[ev].lr.intercept_[0]:+.3f}")
    with open(out_dir / "calibrators_hcov.pkl", "wb") as fh:
        pickle.dump({"calibrators": cals, "events": list(EVENTS),
                     "h_max": H_MAX, "kind": "hcov_logistic_oof"}, fh)

    # ---- score val: baseline A and experiment C -------------------------
    print(f"\n[exp] scoring val ({len(val_base):,} snaps)")
    with open(_RUN.joint_xgb, "rb") as fh:
        bundle_a = pickle.load(fh)
    val_a = predict_trajectory(bundle_a, val_base)          # cummax inside
    # A pre-cummax inversion stats need the un-cummaxed sweep; recompute:
    val_a_raw = val_base.copy()
    for h in range(1, H_MAX + 1):
        sub = add_cond_cols(val_base, h)
        Xh = bundle_a["scaler"].transform(
            sub[bundle_a["feature_names"]].values.astype(np.float32))
        P = bundle_a["model"].predict(
            xgb.DMatrix(Xh, feature_names=list(bundle_a["feature_names"])),
            iteration_range=(0, bundle_a["best_iteration"] + 1))
        for k, ev in enumerate(EVENTS):
            val_a_raw[f"xp_{ev}_h{h}"] = P[:, k]
    inv_a = inversion_stats(val_a_raw, "")
    if _RUN.calibrators.exists():
        val_a = apply_baseline_calibrators(val_a, _RUN.calibrators)
        inv_a_cal = inversion_stats(val_a, "calA_")
    else:
        inv_a_cal = {}

    val_c_raw = predict_trajectory_c(bst_c, val_base, cummax=False)
    inv_c = inversion_stats(val_c_raw, "")
    val_c = predict_trajectory_c(bst_c, val_base, cummax=True)
    val_c = apply_hcov_trajectory(val_c, cals)

    # ---- metrics --------------------------------------------------------
    rows = []
    rows += per_h_metrics(val_a, "", "A_raw(v2.1c)")
    if _RUN.calibrators.exists():
        rows += per_h_metrics(val_a, "calA_", "A_cal(VAL-FIT,optimistic)")
    rows += per_h_metrics(val_c_raw, "", "C_raw(monotone)")
    rows += per_h_metrics(val_c, "cal_", "C_cal(OOF,honest)")
    met = pd.DataFrame(rows)
    met.to_csv(out_dir / "per_event_h_metrics.csv", index=False)

    print(f"\n===== weighted AP @ h={PUBLISH_H} (2x debut) =====")
    for scorer in met["scorer"].unique():
        wap = weighted_ap_at([r for r in rows if r["scorer"] == scorer],
                             PUBLISH_H)
        print(f"  {scorer:<28} {wap:.4f}")

    print(f"\n===== MLB_DEBUT per-h (val) =====")
    print(f"{'scorer':<28}{'h':>3}{'n':>7}{'AP':>8}{'AUC':>8}"
          f"{'Brier':>9}{'calib':>7}")
    for scorer in met["scorer"].unique():
        for h in (1, 2, 3, 6):
            r = met[(met.scorer == scorer) & (met.event == "MLB_DEBUT")
                    & (met.h == h)]
            if len(r):
                r = r.iloc[0]
                print(f"{scorer:<28}{h:>3}{r['n']:>7,.0f}{r['ap']:>8.3f}"
                      f"{r['auc']:>8.3f}{r['brier']:>9.4f}{r['calib']:>7.2f}")

    print(f"\n===== calib ratio @ h={PUBLISH_H} (want 1.00) =====")
    print(f"{'event':<22}" + "".join(f"{s[:14]:>16}"
                                     for s in met["scorer"].unique()))
    for ev in EVENTS:
        cells = []
        for scorer in met["scorer"].unique():
            r = met[(met.scorer == scorer) & (met.event == ev)
                    & (met.h == PUBLISH_H)]
            cells.append(f"{r.iloc[0]['calib']:>16.2f}" if len(r)
                         else f"{'—':>16}")
        print(f"{ev:<22}" + "".join(cells))

    print(f"\n===== cross-h inversions (pre-cummax) =====")
    for name, inv in (("A raw", inv_a), ("A cal(per-h)", inv_a_cal),
                      ("C raw (monotone)", inv_c)):
        for ev, s in inv.items():
            print(f"  {name:<18} {ev:<20} "
                  f"frac_snaps={s['frac_snaps_with_inversion']:.3f} "
                  f"mean_drop={s['mean_total_drop']:.4f} "
                  f"max_drop={s['max_drop']:.4f}")

    # ---- timing ---------------------------------------------------------
    print(f"\n[exp] timing (MLB_DEBUT)")
    tim_c = cdf_timing(val_c, "MLB_DEBUT", "cal_")
    tim_a = cdf_timing(val_a, "MLB_DEBUT",
                       "calA_" if _RUN.calibrators.exists() else "")
    lasso_pred = score_lasso_timing(val_base, _RUN.timing, args.db)
    mean_t_col = (val_base["mean_t_MLB_DEBUT"].to_numpy()
                  if "mean_t_MLB_DEBUT" in val_base.columns else None)
    est = {
        "C_cdf_mean(cal,OOF)": tim_c["t_mean"].to_numpy(),
        "C_cdf_median(cal,OOF)": tim_c["t_med"].to_numpy(),
        "A_cdf_mean(cal_valfit)": tim_a["t_mean"].to_numpy(),
        "lasso_timing.pkl": lasso_pred,
        "hazard_mean_t": mean_t_col,
    }
    windows = {"C_cdf_median(cal,OOF)": (tim_c["t_q25"].to_numpy(),
                                         tim_c["t_q75"].to_numpy())}
    trep = timing_report(val_base.assign(
        trigger_MLB_DEBUT=val_base["trigger_MLB_DEBUT"]), est, windows)
    trep.to_csv(out_dir / "timing_metrics.csv", index=False)
    print(trep.to_string(index=False))

    # ---- summary --------------------------------------------------------
    summary = {
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "quick": bool(args.quick),
        "fit_rows": int(len(fit_long)),
        "best_iteration": best_iter,
        "weighted_ap_h6": {
            s: weighted_ap_at([r for r in rows if r["scorer"] == s],
                              PUBLISH_H)
            for s in met["scorer"].unique()},
        "inversions": {"A_raw": inv_a, "A_cal_perh": inv_a_cal,
                       "C_raw_monotone": inv_c},
        "timing": trep.to_dict(orient="records"),
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"\n[exp] wrote {out_dir}  ({summary['elapsed_s']:.0f}s)")


if __name__ == "__main__":
    main()
