"""Train ONLY the STAR (= AS1 OR AS3+ OR MAJOR_AWARD OR HOF) hazard +
Platt calibrator, and add it to an existing trained hazards pickle.

Why only STAR: the v1.4 model is otherwise well-validated. Retraining
everything risks perturbing the load-bearing MLB_DEBUT / ESTABLISHED
predictions. STAR is the new pooled rare-event target we want for v1.5
to stabilize predictions on a larger positive class.

Usage:
    python -m prospects.classifier.train_star_only \\
        --in  models/event_classifiers_v1.4_platt.pkl \\
        --out models/event_classifiers_v1.5_star.pkl
"""
from __future__ import annotations

import argparse
import gc
import pickle
import sys
from collections import defaultdict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

from prospects.classifier.architectures.survival import (
    EXIT_KEY,
    MAX_OBS_YEAR,
    STAR_KEY,
    _PlattCalibrator,
    _trigger_year,
    build_hazard_panel,
    labels_and_eligibility,
    predict_cumulative_batch,
)
from prospects.features.scouting import FEATURE_NAMES
from prospects.storage import ProspectDB

# Pickle compat: existing model files reference _PlattCalibrator under __main__.
sys.modules["__main__"]._PlattCalibrator = _PlattCalibrator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="prospects_snapshot.db")
    ap.add_argument("--in", dest="in_path",
                    default="models/event_classifiers_v1.4_platt.pkl")
    ap.add_argument("--out",
                    default="models/event_classifiers_v1.5_star.pkl")
    ap.add_argument("--max-draft-year", type=int, default=2020)
    ap.add_argument("--max-year", type=int, default=MAX_OBS_YEAR)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--horizon", type=int, default=15)
    args = ap.parse_args()

    with open(args.in_path, "rb") as f:
        hazards = pickle.load(f)
    print(f"Loaded existing hazards: {list(hazards.keys())}")

    db = ProspectDB(args.db)
    X, pids, years, joined = build_hazard_panel(
        db, max_draft_year=args.max_draft_year, max_year=args.max_year,
    )
    X = X.astype(np.float32, copy=False)

    # Reproduce the same train/val/test split as fit_hazards uses (seed=42).
    unique_players = sorted(set(pids))
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(unique_players))
    n_p = len(unique_players)
    n_test = int(round(0.10 * n_p))
    n_val = int(round(0.10 * n_p))
    test_players = set(unique_players[i] for i in perm[:n_test])
    val_players = set(unique_players[i] for i in perm[n_test:n_test + n_val])
    split = np.array([
        "test" if p in test_players else ("val" if p in val_players else "train")
        for p in pids
    ])

    # STAR labels via the existing labels_and_eligibility (uses _trigger_year).
    eligible, y_all = labels_and_eligibility(joined, years, STAR_KEY)
    tr = (split == "train") & eligible
    te = (split == "test") & eligible
    X_tr, y_tr = X[tr], y_all[tr]
    X_te, y_te = X[te], y_all[te]
    n_pos_tr = int(y_tr.sum())
    n_pos_te = int(y_te.sum())
    print(f"\nSTAR positives: train={n_pos_tr} test={n_pos_te} "
          f"(rows: tr={X_tr.shape[0]:,} te={X_te.shape[0]:,})")

    if n_pos_tr < 10:
        raise SystemExit(f"Too few STAR positives in train: {n_pos_tr}")

    clf = HistGradientBoostingClassifier(
        max_iter=200, max_depth=6, learning_rate=0.05,
        min_samples_leaf=30, random_state=args.seed,
        early_stopping=True, n_iter_no_change=10,
        validation_fraction=0.1,
    ).fit(X_tr, y_tr)

    p_te = clf.predict_proba(X_te)[:, 1]
    try:
        auc = roc_auc_score(y_te, p_te)
    except Exception:
        auc = float("nan")
    brier = brier_score_loss(y_te, p_te)
    print(f"STAR test AUC={auc:.3f} Brier={brier:.4f}")

    hazards[STAR_KEY] = {
        "hazard": clf,
        "feature_names": list(FEATURE_NAMES),
    }

    # Free training arrays before the calibrator pass
    del X_tr, y_tr, X_te, y_te, p_te, eligible, y_all
    gc.collect()

    # --- Calibrator for STAR ---
    # Re-score VAL players at multi-snapshots; fit Platt on (raw_cumP, label).
    with db._connect() as conn:
        rows = [dict(r) for r in conn.execute("""
            SELECT p.*, o.events_json, o.mlb_debut_year, o.year_established_mlb, o.year_top_100, o.year_top_25,
                   o.year_all_star_once, o.year_all_star_three,
                   o.year_major_award, o.year_hof_trajectory, o.final_mlb_year
            FROM prospects p
            JOIN career_outcomes o ON o.player_id = p.player_id
            WHERE (p.draft_year IS NOT NULL AND p.draft_year <= 2018)
               OR COALESCE(p.is_international, 0) = 1
        """).fetchall()]
        stats_rows = conn.execute("SELECT * FROM season_stats").fetchall()
    stats_by_pid: dict[str, list] = {}
    for s in stats_rows:
        d = dict(s)
        stats_by_pid.setdefault(d["player_id"], []).append(d)

    val_rows = [r for r in rows if r["player_id"] in val_players]
    print(f"\nFitting STAR Platt calibrator on {len(val_rows)} val players")

    snapshot_offsets = (1, 2, 3, 5)
    groups_by_year: dict[int, list[dict]] = defaultdict(list)
    for r in val_rows:
        dy = r.get("draft_year")
        if dy is None:
            stat_yrs = [int(s["season_year"])
                        for s in stats_by_pid.get(r["player_id"], [])
                        if s.get("season_year") is not None]
            if not stat_yrs:
                continue
            start = min(stat_yrs)
        else:
            start = int(dy)
        for off in snapshot_offsets:
            cur_year = start + off
            if cur_year >= args.max_year:
                continue
            groups_by_year[cur_year].append(r)

    preds: list[float] = []
    labels: list[int] = []
    for cur_year, group in groups_by_year.items():
        sub_stats = {r["player_id"]: stats_by_pid.get(r["player_id"], [])
                     for r in group}
        out = predict_cumulative_batch(
            hazards, group, sub_stats,
            current_year=cur_year, horizon=args.horizon,
        )
        raw = out.get(("raw", STAR_KEY), out.get(STAR_KEY))
        for i, r in enumerate(group):
            preds.append(float(raw[i]))
            trig = _trigger_year(r, STAR_KEY)
            labels.append(int(trig is not None and trig <= args.max_year))

    preds_a = np.array(preds, dtype=np.float64)
    labels_a = np.array(labels, dtype=np.int8)
    pos = int(labels_a.sum())
    print(f"  cal samples: {len(preds_a):,}  positives: {pos}")
    if pos < 3:
        raise SystemExit(f"Too few STAR positives in val: {pos}")

    calibrator = _PlattCalibrator().fit(preds_a, labels_a)
    hazards[STAR_KEY]["calibrator"] = calibrator
    print(f"  Platt fit: a={calibrator.a:.4f} b={calibrator.b:.4f}")

    # Sanity check: top-decile observed vs predicted
    order = np.argsort(preds_a)[::-1]
    top10_n = max(1, len(preds_a) // 10)
    top10 = order[:top10_n]
    cal_a = calibrator.predict(preds_a)
    print(f"  top-10% raw_mean={preds_a[top10].mean():.3f} "
          f"cal_mean={cal_a[top10].mean():.3f} "
          f"observed_rate={labels_a[top10].mean():.3f}")

    with open(args.out, "wb") as f:
        pickle.dump(hazards, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
