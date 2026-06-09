"""SHAP feature attributions for Ethan Pecko on the v2.0b LANDMARK hazards
(next-step k=1 hazard for MLB_DEBUT, ESTABLISHED_MLB, STAR)."""
from __future__ import annotations

import pickle
import sys

import numpy as np
import shap

from prospects.classifier.architectures.survival import build_windowed_features
from prospects.classifier.architectures.landmark_survival import (
    FEATURE_NAMES_LM, STAR_KEY,
)
from prospects.features.scouting import N_FEATURES
from prospects.schema import CareerEvent

MODEL = "models/event_classifiers_v2.0b_prod.pkl"
DB = "prospects_snapshot.db"
PID = sys.argv[1] if len(sys.argv) > 1 else "draft_2023_ethan_pecko_r6p194"
SEASON = 2026
TOPK = 18
EVENTS = [CareerEvent.MLB_DEBUT, CareerEvent.ESTABLISHED_MLB, STAR_KEY]


def main():
    with open(MODEL, "rb") as fh:
        hazards = pickle.load(fh)
    import sqlite3
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row
    prospect = dict(c.execute("SELECT * FROM prospects WHERE player_id=?",
                              (PID,)).fetchone())
    stats = [dict(r) for r in c.execute(
        "SELECT * FROM season_stats WHERE player_id=? ORDER BY season_year,level",
        (PID,)).fetchall()]
    c.close()

    X = build_windowed_features(prospect, stats, SEASON, milb_only=True)
    assert X.shape[0] == N_FEATURES, (X.shape, N_FEATURES)
    X_lm = np.append(X, 1.0).reshape(1, -1)  # k=1 (next-step hazard)

    print(f"Ethan Pecko ({PID})  pos={prospect.get('primary_position')}  "
          f"snap={SEASON}, k=1")
    print(f"  features: {X_lm.shape[1]} (incl k)\n")

    for ev in EVENTS:
        if ev not in hazards:
            print(f"[{ev}] not in hazards"); continue
        haz = hazards[ev]["hazard"]
        try:
            expl = shap.TreeExplainer(haz)
            sv = expl.shap_values(X_lm)
            base = expl.expected_value
        except Exception:
            expl = shap.Explainer(haz.predict_proba, np.zeros((1, X_lm.shape[1])),
                                  feature_names=FEATURE_NAMES_LM)
            sv = expl(X_lm).values[..., 1]
            base = 0.0
        sv = np.asarray(sv)
        if sv.ndim == 3:        # (n, feat, classes) -> positive class
            sv = sv[0, :, -1]
        elif isinstance(base, (list, np.ndarray)) and np.ndim(sv) == 2:
            sv = sv[0]
        else:
            sv = sv.reshape(-1)
        p = float(haz.predict_proba(X_lm)[0, 1])
        name = ev if isinstance(ev, str) else ev.name
        print(f"=== {name}  p(k=1 hazard)={p:.3f} ===")
        order = np.argsort(-np.abs(sv))[:TOPK]
        for i in order:
            print(f"  {FEATURE_NAMES_LM[i]:<34}{X_lm[0,i]:>10.3f}  shap={sv[i]:+.4f}")
        print()


if __name__ == "__main__":
    main()
