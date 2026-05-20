"""Train v1.13: 100% of data, OOF-derived Beta calibrators.

Step 1: Train all hazards on EVERY row of the panel (no holdout). This
        gives the model maximum training signal.

Step 2: Fit Beta calibrators on the existing 5-fold OOF predictions
        (oof_predictions_v1.13.csv). Those OOF predictions came from
        models that never saw the players being predicted — they are
        leakage-free labels for fitting a calibrator. Attach the fit
        calibrators to the 100%-trained hazards.

Result: v1.13.pkl — most powerful model + honest calibration.

Usage:
    python -m prospects.classifier.train_full_v13 \\
        --panel panel_v1.13.npz \\
        --oof oof_predictions_v1.13.csv \\
        --out models/event_classifiers_v1.13.pkl
"""
from __future__ import annotations

import argparse
import csv
import gc
import pickle
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

from prospects.classifier.architectures.survival import (
    ELITE_KEY,
    EVENT_TRIGGER_COL,
    EXIT_KEY,
    MAX_OBS_YEAR,
    STAR_KEY,
    _BetaCalibrator,
    _PlattCalibrator,
    exit_labels,
    labels_and_eligibility,
)
from prospects.features.scouting import FEATURE_NAMES
from prospects.schema import CareerEvent
from prospects.storage import ProspectDB

sys.modules["__main__"]._PlattCalibrator = _PlattCalibrator
sys.modules["__main__"]._BetaCalibrator = _BetaCalibrator


def _ename(e) -> str:
    if isinstance(e, str):
        return e.lstrip("_")
    return e.name


def _train_event(X_tr: np.ndarray, y_tr: np.ndarray,
                 seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=200, max_depth=6, learning_rate=0.05,
        min_samples_leaf=30, random_state=seed,
        early_stopping=True, n_iter_no_change=10,
        validation_fraction=0.1,
    ).fit(X_tr, y_tr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="panel_v1.13.npz")
    ap.add_argument("--oof", default="oof_predictions_v1.13.csv")
    ap.add_argument("--db", default="prospects_snapshot.db")
    ap.add_argument("--out", default="models/event_classifiers_v1.13.pkl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--right-censor", action="store_true",
                    help="Enable right-censoring of post-exit rows.")
    ap.add_argument("--censor-events", default="TOP_100_PROSPECT,MLB_DEBUT",
                    help="Comma-separated event names to apply censoring "
                         "to (only when --right-censor is set). Default = "
                         "near-horizon events only.")
    args = ap.parse_args()

    print("=" * 70)
    print("  v1.13 TRAIN: 100% data + OOF-derived Beta calibrators")
    print("=" * 70)

    # ---- Load panel ----
    print(f"\nLoading panel from {args.panel}")
    with np.load(args.panel, allow_pickle=True) as d:
        X = d["X"].astype(np.float32, copy=False)
        pids = d["pids"].tolist()
        years = d["years"].tolist()
    with open(args.panel.replace(".npz", ".joined.pkl"), "rb") as fh:
        joined = pickle.load(fh)
    print(f"  panel: {X.shape[0]:,} rows over "
          f"{len(set(pids)):,} players, {X.shape[1]} features")

    # ---- Train each event on 100% of eligible rows ----
    print("\n--- Phase 1: Training hazards on 100% of data ---")
    censor_set = (set(s.strip() for s in args.censor_events.split(","))
                  if args.right_censor else None)
    if censor_set is not None:
        print(f"  RIGHT-CENSORING applied to: {sorted(censor_set)}")
    hazards: dict = {}
    train_events = list(CareerEvent.all_events()) + [ELITE_KEY, STAR_KEY]
    stats_for_censor: dict | None = None
    if args.right_censor:
        db_pre = ProspectDB(args.db)
        with db_pre._connect() as conn:
            stats_rows_pre = conn.execute(
                "SELECT * FROM season_stats").fetchall()
        stats_for_censor = {}
        for s in stats_rows_pre:
            d = dict(s)
            stats_for_censor.setdefault(d["player_id"], []).append(d)
    print(f"\n{'Event':<22} {'n':>9} {'pos':>7} {'AUC':>7} {'Brier':>9}")
    print("-" * 60)
    for event in train_events:
        if (event not in (ELITE_KEY, STAR_KEY)
                and event not in EVENT_TRIGGER_COL):
            continue
        ename = event.name if hasattr(event, "name") else str(event).lstrip("_")
        apply_rc = bool(args.right_censor and (
            censor_set is None or ename in censor_set))
        eligible, y_all = labels_and_eligibility(
            joined, years, event,
            stats_by_pid=stats_for_censor,
            right_censor=apply_rc,
        )
        X_tr = X[eligible]
        y_tr = y_all[eligible]
        n_pos = int(y_tr.sum())
        ev_label = _ename(event)
        if n_pos < 10 or n_pos > X_tr.shape[0] - 10:
            print(f"{ev_label:<22} {X_tr.shape[0]:>9,d} {n_pos:>7d} "
                  f"  skip (pos out of range)")
            continue
        clf = _train_event(X_tr, y_tr, seed=args.seed)
        # Honest in-sample AUC just for visibility (not a held-out metric)
        try:
            p = clf.predict_proba(X_tr)[:, 1]
            auc = roc_auc_score(y_tr, p)
            brier = brier_score_loss(y_tr, p)
        except Exception:
            auc = float("nan"); brier = float("nan")
        print(f"{ev_label:<22} {X_tr.shape[0]:>9,d} {n_pos:>7d} "
              f"{auc:>7.3f} {brier:>9.4f}")
        hazards[event] = {
            "hazard": clf,
            "feature_names": list(FEATURE_NAMES),
        }
        del X_tr, y_tr, eligible, y_all
        gc.collect()

    # ---- Exit hazard ----
    print(f"\nExit hazard...")
    db = ProspectDB(args.db)
    with db._connect() as conn:
        stats_rows = conn.execute("SELECT * FROM season_stats").fetchall()
    stats_by_pid: dict[str, list] = {}
    for s in stats_rows:
        d = dict(s)
        stats_by_pid.setdefault(d["player_id"], []).append(d)
    elig_e, y_e_all = exit_labels(joined, years, stats_by_pid)
    X_tr_e = X[elig_e]
    y_tr_e = y_e_all[elig_e]
    if int(y_tr_e.sum()) >= 10:
        clf_e = HistGradientBoostingClassifier(
            max_iter=300, max_depth=6, learning_rate=0.05,
            min_samples_leaf=30, random_state=args.seed,
        ).fit(X_tr_e, y_tr_e)
        hazards[EXIT_KEY] = {
            "hazard": clf_e,
            "feature_names": list(FEATURE_NAMES),
        }
        print(f"  EXIT trained on {X_tr_e.shape[0]:,} rows, "
              f"pos={int(y_tr_e.sum()):,}")
    del X, X_tr_e, y_tr_e, elig_e, y_e_all, stats_rows, stats_by_pid
    gc.collect()

    # ---- Phase 2: Fit calibrators using OOF predictions ----
    print(f"\n--- Phase 2: Fitting Beta calibrators from {args.oof} ---")
    with open(args.oof, encoding="utf-8") as fh:
        oof_rows = list(csv.DictReader(fh))
    print(f"  loaded {len(oof_rows):,} OOF rows")

    # Build (raw, label) pairs per event from OOF data
    event_to_pairs: dict = {}
    for event in hazards:
        if event == EXIT_KEY:
            continue
        ename = _ename(event)
        raw_col = f"p_{ename}_raw"
        real_col = f"realized_{ename}"
        if raw_col not in oof_rows[0]:
            print(f"  {ename}: missing column {raw_col}, skipping cal")
            continue
        raws = []
        labels = []
        for r in oof_rows:
            try:
                raw = float(r[raw_col])
                lab = int(r[real_col])
            except (TypeError, ValueError):
                continue
            if raw != raw:  # NaN
                continue
            raws.append(raw)
            labels.append(lab)
        event_to_pairs[event] = (
            np.array(raws, dtype=np.float64),
            np.array(labels, dtype=np.int8),
        )

    print(f"\n{'Event':<22} {'n':>7} {'pos':>5} {'a':>7} {'b':>7} {'c':>7} "
          f"{'top10_obs':>10}")
    print("-" * 70)
    for event, (raws, labels) in event_to_pairs.items():
        pos = int(labels.sum())
        ename = _ename(event)
        if pos < 5 or pos > len(labels) - 5:
            print(f"{ename:<22} {len(raws):>7,d} {pos:>5d}  "
                  f"skip (positives out of range)")
            continue
        cal = _BetaCalibrator().fit(raws, labels)
        # OOF predictions came from 80%-trained models; the 100%-trained
        # v1.13 model produces tighter, more confident raw probabilities
        # that exceed the OOF 99th percentile for elite prospects. The
        # default _BetaCalibrator clip would map them all to the same
        # value. Since OOF covers the full prediction range, extrapolation
        # risk is low — disable the clip.
        cal._raw_min = 0.0
        cal._raw_max = 1.0
        hazards[event]["calibrator"] = cal
        # Sanity: top-decile observed
        order = np.argsort(raws)[::-1]
        top10 = order[:max(1, len(raws) // 10)]
        obs10 = float(labels[top10].mean())
        print(f"{ename:<22} {len(raws):>7,d} {pos:>5d} "
              f"{cal.a:>7.3f} {cal.b:>7.3f} {cal.c:>7.3f} "
              f"{obs10:>10.3f}")

    # ---- Save ----
    print(f"\nSaving to {args.out}")
    with open(args.out, "wb") as fh:
        pickle.dump(hazards, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
