"""
prospects/classifier/architectures/cascaded.py
==================================

Cascaded event stacks: each downstream event's classifier sees the prior
events' predicted probabilities as additional features. Captures the natural
ordering MLB_DEBUT -> ESTABLISHED_MLB -> ALL_STAR_ONCE -> ALL_STAR_3+ ->
MAJOR_AWARD -> HOF_TRAJECTORY without a full survival rewrite.

Training is K-fold cross-fit so each event's training features (the prior
events' predictions) are out-of-fold and don't leak. At inference, the full
final stacks are chained.

Output artifact:
    {
        event_int: {
            stacks: [stack_fold0, stack_fold1, ..., stack_foldK-1],   # per-fold for OOF use
            final_stack: stack_full,                                  # trained on all train data
            cascade_priors: [event_int, ...],                         # which priors feed this event
        }, ...
    }

Usage:
    python -m prospects.classifier.architectures.cascaded \\
        [--db prospects_snapshot.db] \\
        [--out models/event_classifiers_v0.9.2_cascaded.pkl] \\
        [--kfolds 5]
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

from prospects.classifier.architectures.stacked import (
    CTX_FEATURE_NAMES, _ctx_features, _fit_hgb, train_stack,
    predict_stack,
)
from prospects.classifier.train_utils import _future_only_labels
from prospects.features.windowed import (
    FEATURE_NAMES, build_panel_dataset,
)
from prospects.schema import CareerEvent
from prospects.storage import ProspectDB


MODEL_VERSION = "v0.9.2-cascaded"

# Order matters — earlier events feed later ones.
CASCADE_ORDER = [
    CareerEvent.MLB_DEBUT,
    CareerEvent.ESTABLISHED_MLB,
    CareerEvent.ALL_STAR_ONCE,
    CareerEvent.ALL_STAR_THREE_PLUS,
    CareerEvent.MAJOR_AWARD,
    CareerEvent.HOF_TRAJECTORY,
]


def _kfold_player_splits(players: list[str], k: int, seed: int) -> list[set]:
    """Return k disjoint sets of held-out player_ids (player-grouped K-fold)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(players))
    folds = np.array_split(perm, k)
    return [{players[i] for i in fold} for fold in folds]


def train_cascade(
    db: ProspectDB,
    max_draft_year: int = 2020,
    max_as_of_year: int = 2024,
    early_cutoff: int = 2,
    k: int = 5,
    seed: int = 42,
) -> dict:
    """Train a cascade of stacks. Returns dict keyed by event int."""
    print(f"Loading panel data...")
    X_base, pids, years, joined = build_panel_dataset(db, max_year=max_as_of_year)
    keep = [i for i, j in enumerate(joined)
            if j.get("draft_year") and j["draft_year"] <= max_draft_year]
    X_base = X_base[keep]
    pids = [pids[i] for i in keep]
    years = [years[i] for i in keep]
    joined = [joined[i] for i in keep]
    n = X_base.shape[0]
    print(f"  {n:,} rows from {len(set(pids)):,} players, features={X_base.shape[1]}")

    # Final test split (10% by player) — never touched until end
    unique_players = sorted(set(pids))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(unique_players))
    n_test = int(round(0.10 * len(unique_players)))
    test_players = set(unique_players[i] for i in perm[:n_test])
    train_players = sorted(p for p in unique_players if p not in test_players)
    print(f"  hold-out test: {len(test_players)} players")
    print(f"  cascade-train: {len(train_players)} players (will K-fold internally)")

    # Indices
    is_test = np.array([p in test_players for p in pids])
    train_idx = np.where(~is_test)[0]
    test_idx = np.where(is_test)[0]
    pids_train = [pids[i] for i in train_idx]

    # K-fold on training player set
    fold_test_players = _kfold_player_splits(train_players, k, seed)

    cascade: dict[int, dict] = {}
    # X_aug grows as we add prior-event predictions
    X_aug = X_base.copy()
    extra_feature_names: list[str] = []

    for event in CASCADE_ORDER:
        print(f"\n=== Cascading event: {event.name} ===")
        mask, y_all = _future_only_labels(joined, years, event)
        if (y_all & mask).sum() < 10:
            print(f"  insufficient positives ({(y_all & mask).sum()}); SKIP")
            continue

        # K-fold OOF predictions for THIS event on the training rows.
        # Used as the NEXT event's prior feature in the cascade.
        oof = np.full(n, np.nan, dtype=np.float64)
        all_fold_stacks = []
        for k_idx, fold_holdout_players in enumerate(fold_test_players):
            # rows in this fold's held-out subset (of train rows)
            ho = np.array([p in fold_holdout_players for p in pids])
            in_fold_train = (~ho) & (~is_test)
            in_fold_holdout = ho & (~is_test)

            # train sub-stack only on in_fold_train rows
            stack_sub = _train_one_stack(
                X_aug[in_fold_train], y_all[in_fold_train], mask[in_fold_train],
                pids_for_split=[pids[i] for i in np.where(in_fold_train)[0]],
                years_arr=np.array([years[i] for i in np.where(in_fold_train)[0]]),
                X_indices=None, early_cutoff=early_cutoff, seed=seed + k_idx,
            )
            all_fold_stacks.append(stack_sub)
            if stack_sub is None:
                continue
            # OOF predictions for this fold's holdout
            if in_fold_holdout.sum() > 0:
                p_ho = _score_stack(stack_sub, X_aug[in_fold_holdout])
                oof[in_fold_holdout] = p_ho

        # Final stack: trained on ALL train rows (no holdout) — used at inference.
        final_stack = _train_one_stack(
            X_aug[~is_test], y_all[~is_test], mask[~is_test],
            pids_for_split=pids_train,
            years_arr=np.array([years[i] for i in train_idx]),
            X_indices=None, early_cutoff=early_cutoff, seed=seed,
        )

        # Evaluate on test
        if final_stack is not None:
            test_mask = mask[is_test]
            X_te = X_aug[is_test][test_mask]
            y_te = y_all[is_test][test_mask]
            p_te = _score_stack(final_stack, X_te)
            try:
                auc = roc_auc_score(y_te, p_te)
            except Exception:
                auc = float("nan")
            brier = brier_score_loss(y_te, p_te)
            print(f"  TEST  n={X_te.shape[0]:,}  pos={int(y_te.sum()):,}  "
                  f"AUC={auc:.3f}  Brier={brier:.4f}")

        cascade[int(event)] = {
            "final_stack": final_stack,
            "extra_feature_names": list(extra_feature_names),
        }

        # Score the TEST rows with the final stack for the NEXT event's input
        # (test rows never see OOF priors; we use the final stack's predictions).
        if final_stack is not None and event != CASCADE_ORDER[-1]:
            test_priors = _score_stack(final_stack, X_aug[is_test])
            oof[is_test] = test_priors
            # Append the OOF column as a new feature for downstream events.
            new_col = np.where(np.isnan(oof), 0.0, oof).reshape(-1, 1)
            X_aug = np.hstack([X_aug, new_col])
            extra_feature_names = extra_feature_names + [f"p_prior_{event.name}"]
            print(f"  appended prior feature: p_prior_{event.name} "
                  f"(now {X_aug.shape[1]} feats)")

    return cascade


def _train_one_stack(X, y, mask, pids_for_split, years_arr, X_indices,
                     early_cutoff: int, seed: int):
    """Train a single early/late/blend/iso stack on the given rows.
    Splits internally into blend/calib using player-grouped subsplit.
    """
    if mask.sum() < 50 or y[mask].sum() < 5:
        return None
    # rows considered are those with mask=True
    Xm = X[mask]
    ym = y[mask]
    pidsm = [pids_for_split[i] for i in np.where(mask)[0]]

    # Sub-split: 80% sub_train, 10% blend, 10% calib (of the masked subset)
    unique_p = sorted(set(pidsm))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(unique_p))
    n_blend = int(round(0.10 * len(unique_p)))
    n_calib = int(round(0.10 * len(unique_p)))
    blend_p = set(unique_p[i] for i in perm[:n_blend])
    calib_p = set(unique_p[i] for i in perm[n_blend:n_blend + n_calib])
    role = np.array([
        "blend" if p in blend_p else ("calib" if p in calib_p else "train")
        for p in pidsm
    ])

    is_early_mask = (Xm[:, FEATURE_NAMES.index("years_in_pro")
                       if "years_in_pro" in FEATURE_NAMES else 0] != -1.0) & \
                    (Xm[:, FEATURE_NAMES.index("years_in_pro")
                        if "years_in_pro" in FEATURE_NAMES else 0] <= early_cutoff)

    tr = role == "train"
    bl = role == "blend"
    ca = role == "calib"

    Xe = Xm[tr & is_early_mask]; ye = ym[tr & is_early_mask]
    Xl = Xm[tr & ~is_early_mask]; yl = ym[tr & ~is_early_mask]
    M_e = _fit_hgb(Xe, ye, seed=seed)
    M_l = _fit_hgb(Xl, yl, seed=seed)
    if M_e is None or M_l is None:
        return None

    Xb = Xm[bl]; yb = ym[bl]
    if yb.size < 10 or yb.sum() < 3:
        return {"early": M_e, "late": M_l, "blend": "mean", "calibrator": None,
                "early_cutoff": early_cutoff}
    pe_b = M_e.predict_proba(Xb)[:, 1]
    pl_b = M_l.predict_proba(Xb)[:, 1]
    Cb = _ctx_features(Xb, pe_b, pl_b)
    M_b = _fit_hgb(Cb, yb, seed=seed, max_iter=200, max_depth=4,
                   learning_rate=0.05, min_samples_leaf=30)
    if M_b is None:
        M_b = "mean"

    # Isotonic on calib
    iso = None
    Xc = Xm[ca]; yc = ym[ca]
    if M_b != "mean" and yc.size >= 10 and yc.sum() >= 3:
        from sklearn.isotonic import IsotonicRegression
        pe_c = M_e.predict_proba(Xc)[:, 1]
        pl_c = M_l.predict_proba(Xc)[:, 1]
        Cc = _ctx_features(Xc, pe_c, pl_c)
        p_c = M_b.predict_proba(Cc)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p_c, yc)

    return {"early": M_e, "late": M_l, "blend": M_b, "calibrator": iso,
            "early_cutoff": early_cutoff}


def _score_stack(stack: dict, X: np.ndarray) -> np.ndarray:
    if stack is None:
        return np.zeros(X.shape[0])
    pe = stack["early"].predict_proba(X)[:, 1]
    pl = stack["late"].predict_proba(X)[:, 1]
    if stack["blend"] == "mean":
        return (pe + pl) / 2
    C = _ctx_features(X, pe, pl)
    raw = stack["blend"].predict_proba(C)[:, 1]
    iso = stack.get("calibrator")
    return iso.predict(raw) if iso is not None else raw


# Nesting: P(narrower) <= P(broader). Each tuple is (broader, narrower).
MONOTONIC_PAIRS = [
    (CareerEvent.MLB_DEBUT, CareerEvent.ESTABLISHED_MLB),
    (CareerEvent.MLB_DEBUT, CareerEvent.ALL_STAR_ONCE),
    (CareerEvent.ESTABLISHED_MLB, CareerEvent.ALL_STAR_ONCE),
    (CareerEvent.ALL_STAR_ONCE, CareerEvent.ALL_STAR_THREE_PLUS),
    (CareerEvent.ALL_STAR_ONCE, CareerEvent.MAJOR_AWARD),
    (CareerEvent.ESTABLISHED_MLB, CareerEvent.MAJOR_AWARD),
    (CareerEvent.ALL_STAR_ONCE, CareerEvent.HOF_TRAJECTORY),
    (CareerEvent.ALL_STAR_THREE_PLUS, CareerEvent.HOF_TRAJECTORY),
]


def predict_cascade(cascade: dict, X_base: np.ndarray,
                    enforce_monotonic: bool = True) -> dict[CareerEvent, np.ndarray]:
    """Score the full cascade. Returns {event: predictions array}.

    If `enforce_monotonic`, clamps narrower events' P to be <= their broader
    parent event's P (so you can't have P(AS3+) > P(AS1)).
    """
    X_aug = X_base.copy()
    out: dict[CareerEvent, np.ndarray] = {}
    for event in CASCADE_ORDER:
        entry = cascade.get(int(event))
        if entry is None or entry["final_stack"] is None:
            out[event] = np.zeros(X_base.shape[0])
            continue
        p = _score_stack(entry["final_stack"], X_aug)
        out[event] = p
        if event != CASCADE_ORDER[-1]:
            X_aug = np.hstack([X_aug, p.reshape(-1, 1)])

    if enforce_monotonic:
        # Apply repeatedly until stable (one pass is enough for our DAG)
        for broader, narrower in MONOTONIC_PAIRS:
            if broader in out and narrower in out:
                out[narrower] = np.minimum(out[narrower], out[broader])
    return out


def save_cascade(cascade: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(cascade, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_cascade(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="prospects_snapshot.db")
    parser.add_argument("--out",
                        default="models/event_classifiers_v0.9.2_cascaded.pkl")
    parser.add_argument("--max-draft-year", type=int, default=2020)
    parser.add_argument("--max-as-of-year", type=int, default=2024)
    parser.add_argument("--early-cutoff", type=int, default=2)
    parser.add_argument("--kfolds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    db = ProspectDB(args.db)
    cascade = train_cascade(
        db, args.max_draft_year, args.max_as_of_year,
        args.early_cutoff, args.kfolds, args.seed,
    )
    save_cascade(cascade, args.out)
    print(f"\nSaved cascade -> {args.out}  (version: {MODEL_VERSION})")


if __name__ == "__main__":
    main()
