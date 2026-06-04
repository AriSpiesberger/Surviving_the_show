"""SHAP feature attributions for Coy James on MLB_DEBUT year 1, 2, 3
(v1.17 prod hazard)."""
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
NAME = "Coy James"
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

    print(f"Player: {NAME} ({prospect['player_id']})")
    print(f"  pos={prospect.get('primary_position')}  "
          f"org={prospect.get('current_org')}  "
          f"draft_year={prospect.get('draft_year')}")
    print(f"  yip0={X0[0, yip_i]}  age0={X0[0, age_i]}  yics0={X0[0, yics_i]}")

    rows = []
    for step in YEARS:
        X = X0.copy()
        for j, base in [(yip_i, X0[0, yip_i]),
                        (age_i, X0[0, age_i]),
                        (yics_i, X0[0, yics_i])]:
            if not np.isnan(base):
                X[0, j] = base + step
        rows.append(X[0])
    X_query = np.vstack(rows)
    p = clf.predict_proba(X_query)[:, 1]
    for i, yr in enumerate(YEARS):
        print(f"  year {yr}: h_DEBUT = {p[i]:.4f}")

    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X_query)
    if isinstance(sv, list):
        sv = sv[1]
    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(np.asarray(base_val).ravel()[-1])
    print(f"\nbase (log-odds, positive class) = {base_val:.4f}")

    for i, yr in enumerate(YEARS):
        contrib = sv[i]
        order = np.argsort(-np.abs(contrib))[:TOPK]
        print(f"\n=== Year {yr}  (logit shift sum = {contrib.sum():+.4f}, "
              f"raw_p = {p[i]:.4f}) ===")
        print(f"{'feature':<40} {'value':>12} {'shap (logit)':>14}")
        for j in order:
            v = X_query[i, j]
            vs = f"{v:.4g}" if not np.isnan(v) else "nan"
            print(f"{FEATURE_NAMES[j]:<40} {vs:>12} {contrib[j]:>+14.4f}")


if __name__ == "__main__":
    main()
