"""
prospects/classifier/architectures/stacked.py
==================================

Two-specialist + meta-blender architecture.

For each CareerEvent we train:

    M_early   — HistGradientBoosting trained on EARLY-career rows
                (years_in_pro <= 2). Specializes on pedigree + thin-data
                snapshots; calibration matches early-career base rate.

    M_late    — HistGradientBoosting trained on LATE-career rows
                (years_in_pro >= 3). Specializes on richer stat profiles
                where late-MiLB performance reveals MLB-readiness.

    M_blend   — Final HistGradientBoosting whose inputs are:
                (p_early, p_late, years_in_pro, n_years_observed,
                 max_level_seen, age_at_as_of, n_total_pa_window,
                 n_total_ip_window, has_draft_data).
                Learns how to combine the two specialists conditioned on
                data context. No hand-coded weights.

Player-grouped 80/10/10 split. Train M_early and M_late on the 80%.
Score the 10% blend slice with both → train M_blend on those. Score the
10% test slice to evaluate.

Per event, persisted as a dict:
    {"early": fitted_clf, "late": fitted_clf, "blend": fitted_clf,
     "ctx_feature_names": [...], "n_features": int}

Usage:
    python -m prospects.classifier.architectures.stacked \\
        [--db prospects_snapshot.db] \\
        [--max-draft-year 2020] \\
        [--out models/event_classifiers_v0.9_stacked.pkl] \\
        [--early-cutoff 2]
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from prospects.classifier.train_utils import _future_only_labels
from prospects.features.windowed import (
    FEATURE_NAMES,
    N_FEATURES,
    build_panel_dataset,
)
from prospects.schema import CareerEvent
from prospects.storage import ProspectDB


MODEL_VERSION = "v0.9-stacked"

# Context features the blender sees alongside the two specialist predictions.
CTX_FEATURE_NAMES = [
    "p_early",
    "p_late",
    "years_in_pro",
    "n_years_observed_in_window",
    "max_level_seen_in_window",
    "age_at_as_of",
    "total_pa_window",
    "total_ip_window",
    "has_draft_data",
    "is_pitcher",
]


def _ctx_features(X: np.ndarray, p_early: np.ndarray, p_late: np.ndarray) -> np.ndarray:
    """Build the blender input matrix from windowed features + specialist preds."""
    f = FEATURE_NAMES
    yip = X[:, f.index("years_in_pro")]
    n_yrs = X[:, f.index("n_years_observed_in_window")]
    max_lvl = X[:, f.index("max_level_seen_in_window")]
    age = X[:, f.index("age_at_as_of")]
    is_p = X[:, f.index("is_pitcher")]
    # Sum PA / IP across lag rows in the window
    pa_cols = [f.index(c) for c in f if c.startswith("pa_y")]
    ip_cols = [f.index(c) for c in f if c.startswith("ip_y")]
    pa_total = np.where(X[:, pa_cols] < 0, 0, X[:, pa_cols]).sum(axis=1)
    ip_total = np.where(X[:, ip_cols] < 0, 0, X[:, ip_cols]).sum(axis=1)
    has_draft = (X[:, f.index("draft_round")] != -1.0).astype(np.float64)
    return np.column_stack([
        p_early, p_late, yip, n_yrs, max_lvl, age,
        pa_total, ip_total, has_draft, is_p,
    ])


def _fit_hgb(X, y, seed=0, **kw):
    if y.sum() < 5 or y.sum() > len(y) - 5:
        return None
    return HistGradientBoostingClassifier(
        max_iter=kw.pop("max_iter", 300),
        max_depth=kw.pop("max_depth", 6),
        learning_rate=kw.pop("learning_rate", 0.05),
        min_samples_leaf=kw.pop("min_samples_leaf", 20),
        random_state=seed,
        **kw,
    ).fit(X, y)


def _metrics(y, p):
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan"), float("nan")
    try:
        auc = roc_auc_score(y, p)
    except Exception:
        auc = float("nan")
    return auc, brier_score_loss(y, p)


def train_stack(
    db: ProspectDB,
    max_draft_year: int = 2020,
    max_as_of_year: int = 2024,
    early_cutoff: int = 2,
    seed: int = 42,
) -> dict[CareerEvent, dict]:
    print(f"Building panel dataset (draft <= {max_draft_year}, as_of <= {max_as_of_year})...")
    X, pids, years, joined = build_panel_dataset(db, max_year=max_as_of_year)
    keep = [i for i, j in enumerate(joined)
            if j.get("draft_year") and j["draft_year"] <= max_draft_year]
    X = X[keep]
    pids = [pids[i] for i in keep]
    years = [years[i] for i in keep]
    joined = [joined[i] for i in keep]
    n = X.shape[0]
    print(f"  {n:,} rows from {len(set(pids)):,} players, features={X.shape[1]}")

    # Player-grouped 75/10/5/10 split: train -> M_early & M_late
    #                                     blend -> M_blend training
    #                                     calib -> isotonic on blender output
    #                                     test  -> evaluation
    unique_players = sorted(set(pids))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(unique_players))
    n_p = len(unique_players)
    n_test = int(round(0.10 * n_p))
    n_blend = int(round(0.10 * n_p))
    n_calib = int(round(0.05 * n_p))
    test_players = set(unique_players[i] for i in perm[:n_test])
    blend_players = set(unique_players[i] for i in perm[n_test:n_test + n_blend])
    calib_players = set(
        unique_players[i] for i in perm[n_test + n_blend:n_test + n_blend + n_calib]
    )
    train_players = set(
        unique_players[i] for i in perm[n_test + n_blend + n_calib:]
    )
    print(f"  player split: train {len(train_players)} | "
          f"blend {len(blend_players)} | calib {len(calib_players)} | "
          f"test {len(test_players)}")

    split = np.array([
        "test" if p in test_players else
        ("blend" if p in blend_players else
         ("calib" if p in calib_players else "train"))
        for p in pids
    ])

    yip_idx = FEATURE_NAMES.index("years_in_pro")
    yip = X[:, yip_idx]
    is_early = (yip != -1.0) & (yip <= early_cutoff)

    print(f"  early rows (yip 0..{early_cutoff}): {is_early.sum():,}  "
          f"late rows: {(~is_early).sum():,}")
    print()

    results: dict[CareerEvent, dict] = {}
    for event in CareerEvent.all_events():
        mask, y_all = _future_only_labels(joined, years, event)

        tr = (split == "train") & mask
        bl = (split == "blend") & mask
        ca = (split == "calib") & mask
        te = (split == "test") & mask

        # Train specialists on TRAIN slice
        tr_early = tr & is_early
        tr_late = tr & ~is_early
        M_e = _fit_hgb(X[tr_early], y_all[tr_early], seed=seed)
        M_l = _fit_hgb(X[tr_late], y_all[tr_late], seed=seed)

        if M_e is None or M_l is None:
            print(f"{event.name:<22}  insufficient positives in train (early={int(y_all[tr_early].sum())}, "
                  f"late={int(y_all[tr_late].sum())}); SKIP")
            continue

        # Build blender training data
        Xb = X[bl]
        pe_b = M_e.predict_proba(Xb)[:, 1]
        pl_b = M_l.predict_proba(Xb)[:, 1]
        Cb = _ctx_features(Xb, pe_b, pl_b)
        yb = y_all[bl]
        M_b = _fit_hgb(Cb, yb, seed=seed, max_iter=200, max_depth=4,
                       learning_rate=0.05, min_samples_leaf=30)

        if M_b is None:
            # Fall back to simple mean for events too rare in blend split
            M_b = "mean"

        # Fit isotonic calibrator on the CALIB slice (held out from blender too).
        iso = None
        Xc = X[ca]
        yc = y_all[ca]
        if M_b != "mean" and yc.sum() >= 5 and yc.sum() <= len(yc) - 5:
            pe_c = M_e.predict_proba(Xc)[:, 1]
            pl_c = M_l.predict_proba(Xc)[:, 1]
            Cc = _ctx_features(Xc, pe_c, pl_c)
            p_c = M_b.predict_proba(Cc)[:, 1]
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(p_c, yc)

        # Eval on test
        Xt = X[te]
        pe_t = M_e.predict_proba(Xt)[:, 1]
        pl_t = M_l.predict_proba(Xt)[:, 1]
        if M_b == "mean":
            p_t = (pe_t + pl_t) / 2
        else:
            Ct = _ctx_features(Xt, pe_t, pl_t)
            p_t_raw = M_b.predict_proba(Ct)[:, 1]
            p_t = iso.predict(p_t_raw) if iso is not None else p_t_raw
        yt = y_all[te]

        auc_e, brier_e = _metrics(yt, pe_t)
        auc_l, brier_l = _metrics(yt, pl_t)
        auc_b, brier_b = _metrics(yt, p_t)

        print(f"{event.name:<22}  test n={te.sum():,} pos={int(yt.sum()):,}  "
              f"AUC: early={auc_e:.3f} late={auc_l:.3f} STACK={auc_b:.3f}  "
              f"Brier_stack={brier_b:.4f}")

        results[event] = {
            "early": M_e,
            "late": M_l,
            "blend": M_b,
            "calibrator": iso,
            "early_cutoff": early_cutoff,
            "ctx_feature_names": list(CTX_FEATURE_NAMES),
            "n_features": int(X.shape[1]),
        }
    return results


def predict_stack(models: dict, X: np.ndarray, event: CareerEvent) -> np.ndarray:
    """Score a (n, N_FEATURES) matrix with a per-event stack."""
    m = models.get(event)
    if not m:
        return np.zeros(X.shape[0])
    p_e = m["early"].predict_proba(X)[:, 1]
    p_l = m["late"].predict_proba(X)[:, 1]
    if m["blend"] == "mean":
        return (p_e + p_l) / 2
    C = _ctx_features(X, p_e, p_l)
    p_raw = m["blend"].predict_proba(C)[:, 1]
    iso = m.get("calibrator")
    return iso.predict(p_raw) if iso is not None else p_raw


def save_stack(models: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({int(e): m for e, m in models.items()}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)


def load_stack(path: str) -> dict:
    with open(path, "rb") as f:
        raw = pickle.load(f)
    return {CareerEvent(int(k)): v for k, v in raw.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="prospects_snapshot.db")
    parser.add_argument("--max-draft-year", type=int, default=2020)
    parser.add_argument("--max-as-of-year", type=int, default=2024)
    parser.add_argument("--early-cutoff", type=int, default=2,
                        help="years_in_pro <= this is 'early'")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="models/event_classifiers_v0.9_stacked.pkl")
    args = parser.parse_args()

    db = ProspectDB(args.db)
    models = train_stack(
        db, args.max_draft_year, args.max_as_of_year,
        args.early_cutoff, args.seed,
    )
    save_stack(models, args.out)
    print(f"\nSaved stack -> {args.out}  (version: {MODEL_VERSION})")


if __name__ == "__main__":
    main()
