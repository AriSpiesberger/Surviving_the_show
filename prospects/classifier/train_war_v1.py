"""
prospects/classifier/train_war_v1.py
======================================

Continuous career-WAR regressor — companion to the v0.5 event classifier.

Target framing: E[career_WAR | reached MLB]. We train only on debuted players
because non-debuters have undefined WAR (and counting them as WAR=0 would
collapse the regressor into a "did they make MLB" classifier — work the
v0.5 P(MLB_DEBUT) head already does). At inference time the expected
career WAR is

    E[career_WAR]  =  P(debut) * E[career_WAR | debut]

so the regressor only needs to learn "given they made it, how good are they."

Selection-bias guard (same as v0.5): train only on draftees who've had time
to accumulate WAR. Default cutoff: draft_year <= 2018 (>= 8 years post-draft
as of 2026). A 2022 draftee who debuted in 2025 has at most 1 year of WAR
on the books and would bias the model toward small numbers.

Features: same windowed MiLB-only feature vector as v0.5 (so we can share a
preprocessing path). For debuted prospects this means we predict their
career WAR from the MiLB record we'd have seen the year before they were
called up.

Usage:
    python -m prospects.classifier.train_war_v1 \\
        [--max-draft-year 2018] \\
        [--out models/war_regressor_v1.pkl]
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from prospects.features.windowed import (
    FEATURE_NAMES,
    N_FEATURES,
    build_windowed_features,
)
from prospects.storage import ProspectDB


MODEL_VERSION_WAR_V1 = "war-v1-hgb-conditional-debut"


def _load_debuted(db: ProspectDB, max_draft_year: int):
    with db._connect() as conn:
        rows = conn.execute(
            """
            SELECT p.*, o.mlb_debut_year, o.career_war
            FROM prospects p
            JOIN career_outcomes o ON p.player_id = o.player_id
            WHERE o.mlb_debut_year IS NOT NULL
              AND p.draft_year IS NOT NULL
              AND p.draft_year <= ?
            """,
            (max_draft_year,),
        ).fetchall()
        prospects = [dict(r) for r in rows]
        stats_rows = conn.execute("SELECT * FROM season_stats").fetchall()
    by_pid: dict[str, list] = {}
    for s in stats_rows:
        d = dict(s)
        by_pid.setdefault(d["player_id"], []).append(d)
    return prospects, by_pid


def _as_of_year_for_training(prospect: dict, stats: list[dict]) -> int:
    """The last MiLB year before MLB debut — most realistic "scouting"
    snapshot. Falls back to debut_year - 1 or draft_year + 1."""
    debut = prospect.get("mlb_debut_year")
    milb_years = sorted({
        s["season_year"] for s in stats
        if s.get("season_year") is not None
        and (s.get("level") or "").upper() != "MLB"
    })
    pre = [y for y in milb_years if (debut is None or y < debut)]
    if pre:
        return pre[-1]
    if debut:
        return int(debut) - 1
    dy = prospect.get("draft_year")
    return (dy or 2017) + 1


def _build_xy(prospects, stats_by_pid):
    X_rows, y_rows, meta = [], [], []
    for p in prospects:
        stats = stats_by_pid.get(p["player_id"], [])
        as_of = _as_of_year_for_training(p, stats)
        x = build_windowed_features(p, stats, as_of, milb_only=True)
        X_rows.append(x)
        y_rows.append(float(p["career_war"] or 0.0))
        meta.append({"player_id": p["player_id"], "name": p["name"],
                     "draft_year": p.get("draft_year"),
                     "mlb_debut_year": p.get("mlb_debut_year")})
    X = np.vstack(X_rows) if X_rows else np.zeros((0, N_FEATURES))
    y = np.array(y_rows, dtype=np.float64)
    return X, y, meta


def _fit_bootstraps(X, y, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    members = []
    for k in range(n_boot):
        idx = rng.integers(0, n, size=n)
        reg = HistGradientBoostingRegressor(
            max_iter=400, max_depth=6, learning_rate=0.04,
            min_samples_leaf=20, l2_regularization=1.0,
            random_state=seed + k,
            loss="absolute_error",  # heavy right tail; MAE is more robust than squared
        )
        reg.fit(X[idx], y[idx])
        members.append(reg)
    return members


def _ensemble_predict(members, X):
    P = np.column_stack([m.predict(X) for m in members])
    return P  # (n_samples, n_boot)


def _metrics(y_true, y_pred):
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    # Pearson r
    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        r = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        r = float("nan")
    # Spearman (rank corr): cheap implementation
    if len(y_true) > 1:
        order_t = np.argsort(np.argsort(y_true))
        order_p = np.argsort(np.argsort(y_pred))
        rho = float(np.corrcoef(order_t, order_p)[0, 1])
    else:
        rho = float("nan")
    return mae, rmse, r, rho


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="prospects.db")
    parser.add_argument("--out", default="models/war_regressor_v1.pkl")
    parser.add_argument("--max-draft-year", type=int, default=2018)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstraps", type=int, default=12)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--test-frac", type=float, default=0.10)
    args = parser.parse_args()

    db = ProspectDB(args.db)
    print(f"[war-v1] loading debuted players with draft_year <= {args.max_draft_year}...")
    prospects, by_pid = _load_debuted(db, args.max_draft_year)
    print(f"  {len(prospects)} debuted, matured prospects")

    X, y, meta = _build_xy(prospects, by_pid)
    print(f"  features: {X.shape}; WAR target: mean={y.mean():.2f}, "
          f"median={np.median(y):.2f}, p90={np.percentile(y,90):.2f}, max={y.max():.2f}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(X.shape[0])
    n_val = int(round(args.val_frac * X.shape[0]))
    n_test = int(round(args.test_frac * X.shape[0]))
    val_idx = perm[:n_val]
    test_idx = perm[n_val:n_val + n_test]
    train_idx = perm[n_val + n_test:]
    print(f"  split: train {len(train_idx)} | val {len(val_idx)} | test {len(test_idx)}")

    members = _fit_bootstraps(X[train_idx], y[train_idx], args.bootstraps, args.seed)
    print(f"[war-v1] fit {len(members)} bootstrap regressors")

    for split_name, idx in (("val", val_idx), ("test", test_idx)):
        if len(idx) == 0:
            continue
        P = _ensemble_predict(members, X[idx])
        y_pred = P.mean(axis=1)
        mae, rmse, r, rho = _metrics(y[idx], y_pred)
        print(f"  [{split_name}] n={len(idx)}  MAE={mae:.2f}  RMSE={rmse:.2f}  "
              f"r={r:.3f}  spearman={rho:.3f}")

    artifact = {
        "version": MODEL_VERSION_WAR_V1,
        "feature_names": list(FEATURE_NAMES),
        "members": members,
        "max_draft_year": args.max_draft_year,
        "n_train": len(train_idx),
        "y_train_mean": float(y[train_idx].mean()),
        "y_train_p90": float(np.percentile(y[train_idx], 90)),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[war-v1] saved -> {args.out}")


if __name__ == "__main__":
    main()
