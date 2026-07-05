"""Smooth, monotone probability calibrators for the buy-list pipeline.

Lives in a stable importable module (NOT a __main__ script) so pickled
calibrators unpickle the same way from every consumer (fit_prob_calibrators,
build_v2.0_buylist, gen_yip_thresholds_cond).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def logit(p, eps: float = 1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


class LogitCalibrator:
    """Platt scaling on the logit of the raw score:
    p_cal = sigmoid(a * logit(p_raw) + b).

    Smooth and strictly monotone, so unlike isotonic it is CONTINUOUS — it
    preserves full ranking resolution (every distinct raw score maps to a
    distinct calibrated value; no plateaus that collapse thousands of players
    onto one value and make thresholding all-or-nothing). a≈1, b≈0 reproduces
    an already-calibrated input. Consumers call .predict(raw) exactly like a
    sklearn IsotonicRegression."""

    def __init__(self):
        self.lr = LogisticRegression(C=1e6, solver="lbfgs")

    def fit(self, x_raw, y):
        self.lr.fit(logit(x_raw).reshape(-1, 1), np.asarray(y))
        return self

    def predict(self, x_raw):
        return self.lr.predict_proba(logit(x_raw).reshape(-1, 1))[:, 1]
