"""SHAP feature attributions for Spencer Giesting on the next two hazards
(MLB_DEBUT year 1 and ESTABLISHED_MLB year 1)."""
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
NAME = "Spencer Giesting"
SEASON = 2026
TOPK = 15
EVENTS = [CareerEvent.MLB_DEBUT, CareerEvent.ESTABLISHED_MLB]


def main():
    hazards = load_hazards(MODEL)
    db = ProspectDB(DB)
    with db._connect() as conn:
        row = conn.execute("SELECT * FROM prospects WHERE name=? LIMIT 1",
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

    # Year 1 feature row (advance yip / age / yics by 1)
    X1 = X0.copy()
    for j, base in [(yip_i, X0[0, yip_i]), (age_i, X0[0, age_i]),
                     (yics_i, X0[0, yics_i])]:
        if not np.isnan(base):
            X1[0, j] = base + 1

    print(f"Player: {NAME}  ({prospect['player_id']})")
    print(f"  pos={prospect.get('primary_position')}  "
          f"is_pitcher={prospect.get('is_pitcher')}  "
          f"draft_year={prospect.get('draft_year')}")
    print(f"  yip0={X0[0, yip_i]}  age0={X0[0, age_i]}  "
          f"yics0={X0[0, yics_i]}")

    for event in EVENTS:
        clf = hazards[event]["hazard"]
        p = float(clf.predict_proba(X1)[0, 1])
        ev_name = event.name
        print(f"\n{'='*72}")
        print(f"{ev_name}  year-1 hazard = {p:.4f}")
        print(f"{'='*72}")
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X1)
        if isinstance(sv, list):
            sv = sv[1]
        base_val = explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = float(np.asarray(base_val).ravel()[-1])
        contrib = sv[0]
        order = np.argsort(-np.abs(contrib))[:TOPK]
        print(f"base (log-odds) = {base_val:.4f}   "
              f"sum shifts = {contrib.sum():+.4f}")
        print(f"{'feature':<40} {'value':>12} {'shap (logit)':>14}")
        for j in order:
            v = X1[0, j]
            vs = f"{v:.4g}" if not np.isnan(v) else "nan"
            print(f"{FEATURE_NAMES[j]:<40} {vs:>12} {contrib[j]:>+14.4f}")


if __name__ == "__main__":
    main()
