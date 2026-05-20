"""
prospects/classifier/model.py
==============================

One gradient-boosted binary classifier per CareerEvent. Each model maps a
feature vector to P(event triggered).

Design choices:
  - HistGradientBoosting: handles NaN/missing natively, fast, robust to scale.
  - Per-event independent training (no chained constraints). Monotonicity
    across nested events (TOP_25 ⊆ TOP_100) is enforced *post hoc* during
    prediction by floor-ing the broader event by the narrower one.
  - Credible intervals via bootstrap: train K models on bootstrap resamples,
    use percentiles of P across them as p_lo / p_hi.
  - Calibration via sklearn's CalibratedClassifierCV (isotonic) on a held-out
    fold so probabilities are well-calibrated, not just well-ranked.

Persisted model artifact = one pickle per event holding a list of
(calibrated_classifier) bootstrap members.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from prospects.features.build import (
    FEATURE_NAMES,
    N_FEATURES,
    build_feature_vector,
    build_training_matrix,
)
from prospects.schema import CareerEvent, EventProbability, ProspectPrediction
from prospects.storage import ProspectDB


MODEL_VERSION = "v0.1-hgb-boot"


# Pairs (broader_event, narrower_event) where narrower implies broader.
MONOTONIC_PAIRS = [
    (CareerEvent.TOP_100_PROSPECT, CareerEvent.TOP_25_PROSPECT),
    (CareerEvent.MLB_DEBUT, CareerEvent.ESTABLISHED_MLB),
    (CareerEvent.MLB_DEBUT, CareerEvent.ALL_STAR_ONCE),
    (CareerEvent.ALL_STAR_ONCE, CareerEvent.ALL_STAR_THREE_PLUS),
    (CareerEvent.ALL_STAR_ONCE, CareerEvent.MAJOR_AWARD),
    (CareerEvent.ESTABLISHED_MLB, CareerEvent.HOF_TRAJECTORY),
]


@dataclass
class EventClassifier:
    event: CareerEvent
    members: list = field(default_factory=list)  # list of fitted CalibratedClassifierCV
    n_train: int = 0
    n_pos: int = 0
    feature_names: list = field(default_factory=lambda: list(FEATURE_NAMES))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns (n_samples, n_bootstraps) matrix of probabilities."""
        if not self.members:
            raise ValueError(f"EventClassifier({self.event}) is not fitted")
        cols = [m.predict_proba(X)[:, 1] for m in self.members]
        return np.column_stack(cols)

    def predict_intervals(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        P = self.predict_proba(X)
        return (
            P.mean(axis=1),
            np.percentile(P, 10, axis=1),
            np.percentile(P, 90, axis=1),
        )


def _y_from_outcomes(outcomes: list[dict], event: CareerEvent) -> np.ndarray:
    """Pull the y vector for one event from outcome dicts that carry events_json."""
    y = np.zeros(len(outcomes), dtype=np.int8)
    key = str(int(event))
    for i, o in enumerate(outcomes):
        ej = o.get("events_json")
        if not ej:
            continue
        d = json.loads(ej) if isinstance(ej, str) else ej
        if d.get(key):
            y[i] = 1
    return y


def _train_one_event(
    X: np.ndarray,
    y: np.ndarray,
    event: CareerEvent,
    n_bootstraps: int = 10,
    seed: int = 0,
    verbose: bool = True,
) -> EventClassifier:
    """Train K bootstrap members for one event."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    n_pos = int(y.sum())

    if verbose:
        print(f"  {event.name:<22} n={n} pos={n_pos} ({n_pos/max(n,1):.1%})", end=" ")

    members = []
    if n_pos < 5 or n_pos > n - 5:
        # Degenerate: not enough variation to train. Fall back to a single
        # constant-predicting classifier via tiny added jitter.
        if verbose:
            print("[degenerate -> base rate only]")
        base = max(n_pos / max(n, 1), 1e-3)
        members = [_ConstantClassifier(base)]
        return EventClassifier(event=event, members=members, n_train=n, n_pos=n_pos)

    for k in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        Xb, yb = X[idx], y[idx]
        if yb.sum() == 0 or yb.sum() == n:
            # resample didn't capture both classes — extend a few minority rows
            pos_idx = np.where(y == 1)[0]
            if len(pos_idx):
                Xb = np.vstack([Xb, X[pos_idx[:5]]])
                yb = np.concatenate([yb, np.ones(min(5, len(pos_idx)), dtype=np.int8)])
        base = HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=5,
            learning_rate=0.06,
            min_samples_leaf=10,
            random_state=seed + k,
        )
        # CalibratedClassifierCV with isotonic is robust + cheap on small data
        clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
        try:
            clf.fit(Xb, yb)
            members.append(clf)
        except Exception as e:
            if verbose:
                print(f"[boot {k} skipped: {type(e).__name__}]", end=" ")
    if verbose:
        print(f"[{len(members)} members]")
    return EventClassifier(event=event, members=members, n_train=n, n_pos=n_pos)


class _ConstantClassifier:
    """Fallback for degenerate-class events. Mimics predict_proba API."""
    def __init__(self, p: float):
        self.p = float(p)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        return np.column_stack([1 - np.full(n, self.p), np.full(n, self.p)])


def train_all_events(
    db: ProspectDB,
    n_bootstraps: int = 10,
    verbose: bool = True,
) -> dict[CareerEvent, EventClassifier]:
    """
    Train one classifier per CareerEvent against all labeled players in DB.

    Returns dict mapping event -> EventClassifier.
    """
    X, player_ids, outcomes = build_training_matrix(db, require_outcome=True)
    if verbose:
        print(f"Training matrix: {X.shape[0]} players × {X.shape[1]} features")

    if X.shape[0] == 0:
        raise RuntimeError(
            "No labeled players in DB. Run the outcomes phase first."
        )

    models: dict[CareerEvent, EventClassifier] = {}
    for event in CareerEvent.all_events():
        y = _y_from_outcomes(outcomes, event)
        models[event] = _train_one_event(
            X, y, event, n_bootstraps=n_bootstraps, verbose=verbose,
        )
    return models


def predict_prospect(
    db: ProspectDB,
    models: dict[CareerEvent, EventClassifier],
    player_id: str,
    as_of: Optional[date] = None,
) -> ProspectPrediction:
    """Inference: score one current prospect across all events."""
    prospect = db.get_prospect(player_id)
    if not prospect:
        raise KeyError(player_id)
    stats = db.get_season_stats(player_id)
    x = build_feature_vector(prospect, stats, outcome=None).reshape(1, -1)

    events_out: dict[CareerEvent, EventProbability] = {}
    mean_p: dict[CareerEvent, tuple[float, float, float]] = {}
    for event, clf in models.items():
        mean, lo, hi = clf.predict_intervals(x)
        mean_p[event] = (float(mean[0]), float(lo[0]), float(hi[0]))

    # Enforce monotonic constraints: narrower event <= broader event.
    for broader, narrower in MONOTONIC_PAIRS:
        if broader in mean_p and narrower in mean_p:
            bm, bl, bh = mean_p[broader]
            nm, nl, nh = mean_p[narrower]
            mean_p[narrower] = (min(nm, bm), min(nl, bl), min(nh, bh))

    avg_width = 0.0
    for event, (m, lo, hi) in mean_p.items():
        m_c = float(np.clip(m, 0.0, 1.0))
        lo_c = float(np.clip(min(lo, m_c), 0.0, m_c))
        hi_c = float(np.clip(max(hi, m_c), m_c, 1.0))
        events_out[event] = EventProbability(event, m_c, lo_c, hi_c)
        avg_width += hi_c - lo_c
    confidence = float(np.clip(1.0 - (avg_width / max(len(events_out), 1)), 0.0, 1.0))

    return ProspectPrediction(
        player_id=player_id,
        as_of_date=as_of or date.today(),
        events=events_out,
        confidence=confidence,
        model_version=MODEL_VERSION,
        features_used=N_FEATURES,
        features_imputed=int((x == 0).sum()),
    )


# ============================================================================
# PERSISTENCE
# ============================================================================

def save_models(models: dict[CareerEvent, EventClassifier], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(
            {int(e): m for e, m in models.items()},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def load_models(path: str) -> dict[CareerEvent, EventClassifier]:
    with open(path, "rb") as f:
        raw = pickle.load(f)
    return {CareerEvent(int(k)): v for k, v in raw.items()}
