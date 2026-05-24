"""SHAP feature attributions for Justin Lamkin's DEBUT hazard, years 1-3."""
from __future__ import annotations

import sys
import numpy as np
import shap

from prospects.classifier.architectures.survival import (
    _PlattCalibrator, build_windowed_features, load_hazards,
)
from prospects.features.scouting import FEATURE_NAMES
from prospects.schema import CareerEvent
from prospects.storage import ProspectDB

sys.modules["__main__"]._PlattCalibrator = _PlattCalibrator

MODEL = "models/event_classifiers_v1.17_prod.pkl"
DB = "prospects.db"
NAME = "Justin Lamkin"
SEASON = 2026
YEARS = [1, 2, 3]
TOPK = 15


def main():
    hazards = load_hazards(MODEL)
    clf = hazards[CareerEvent.MLB_DEBUT]["hazard"]

    db = ProspectDB(DB)
    with db._connect() as conn:
        row = conn.execute("SELECT * FROM prospects WHERE name = ? LIMIT 1",
                           (NAME,)).fetchone()
        if row is None:
            print(f"Not found: {NAME}")
            return
        prospect = dict(row)
        stats = [dict(r) for r in conn.execute(
            "SELECT * FROM season_stats WHERE player_id=? "
            "ORDER BY season_year, level", (prospect["player_id"],)
        ).fetchall()]

    X0 = build_windowed_features(prospect, stats, SEASON,
                                 milb_only=True).reshape(1, -1)
    yip_i = FEATURE_NAMES.index("years_in_pro")
    age_i = FEATURE_NAMES.index("age_at_as_of")
    yics_i = FEATURE_NAMES.index("years_in_current_system")

    rows = []
    for step in YEARS:
        X = X0.copy()
        if not np.isnan(X0[0, yip_i]):
            X[0, yip_i] = X0[0, yip_i] + step
        if not np.isnan(X0[0, age_i]):
            X[0, age_i] = X0[0, age_i] + step
        if not np.isnan(X0[0, yics_i]):
            X[0, yics_i] = X0[0, yics_i] + step
        rows.append(X[0])
    X_query = np.vstack(rows)

    print(f"Player: {NAME}  snap={SEASON}  event=MLB_DEBUT")
    print(f"Model: {MODEL}")
    p = clf.predict_proba(X_query)[:, 1]
    for i, yr in enumerate(YEARS):
        print(f"  year {yr}: h_DEBUT = {p[i]:.4f}")

    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X_query)
    # sklearn HGB binary: shap_values returns array (n, n_features) for positive class
    if isinstance(sv, list):
        sv = sv[1]
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = float(np.asarray(base).ravel()[-1])
    print(f"\nbase (log-odds, positive class) = {base:.4f}")

    for i, yr in enumerate(YEARS):
        contrib = sv[i]
        feat_vals = X_query[i]
        order = np.argsort(-np.abs(contrib))[:TOPK]
        print(f"\n=== Year {yr}  (logit shift sum = {contrib.sum():+.4f}, "
              f"raw_p = {p[i]:.4f}) ===")
        print(f"{'feature':<40} {'value':>12} {'shap (logit)':>14}")
        for j in order:
            v = feat_vals[j]
            vs = f"{v:.4g}" if not np.isnan(v) else "nan"
            print(f"{FEATURE_NAMES[j]:<40} {vs:>12} {contrib[j]:>+14.4f}")


if __name__ == "__main__":
    main()
