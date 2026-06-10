"""Conditional refinement of the landmark hazard model (v2.1c).

The old v2.0 path ran a joint XGB as a *terminal* head: it consumed 6 collapsed
cumulative hazard probs + baseline and emitted ONE scalar per event — the entire
hazard *trajectory* (the per-year curves hk1..hk10) was discarded.

This module reframes the XGB as a *conditional refinement* of the hazard model's
own trajectory. The model reads:

  - the full per-year hazard curves (hk1..hk10 for the 5 curve events),
  - the collapsed cumulative probs + baseline (age, yip, scouting),
  - the hazard model's OWN cumulative-incidence at the target horizon h
    (``haz_cum_h_<event>`` = 1 - prod_{j<=h}(1 - hk_j)),
  - the horizon h itself (as a feature),

and outputs, per event, the refined **cumulative P(event by snap+h)**. Sweeping
h = 1..H_MAX yields a per-year trajectory vector instead of a single scalar.

Horizon-as-a-feature (rather than a wide [events x H] output) is the same trick
``landmark_survival`` uses to kill its train/inference mismatch, and it solves
the per-cell censoring problem for free: ``years_fwd`` is row-level (identical
across events), so a (row, h) cell is fully resolved for every event head iff
``years_fwd >= h`` — no per-output masking needed.

Shared by the trainer (fit_joint_xgb_cond), the OOF/prod orchestrators, the buy
list builder and the honest evaluator so feature ordering can never drift.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import xgboost as xgb

from prospects.features.scouting_grades import (
    SCOUTING_SUMMARY_COLS, attach_scouting_summary,
)

# --- horizon config -------------------------------------------------------
H_MAX = 10          # train/predict horizons h in {1..H_MAX}; matches stored hk1..hk10
PUBLISH_H = 6       # buy-list / headline horizon (P(event by snap+6))
H_CENTER = 5        # centering for the h feature (immaterial to trees, keeps scaler tidy)
AGE_CENTER, YIP_CENTER = 22, 3

# --- events ---------------------------------------------------------------
# The 4 refined output heads (unchanged from v2.0).
EVENTS = ["TOP_100_PROSPECT", "MLB_DEBUT", "ESTABLISHED_MLB", "STAR_PLUS_ELITE"]

# Collapsed cumulative hazard probs the hazard layer emits (full ~15y horizon).
HAZARD_PROBS = [
    "p_TOP_100_PROSPECT", "p_MLB_DEBUT", "p_ESTABLISHED_MLB",
    "p_ELITE", "p_STAR", "p_STAR_PLUS_ELITE",
]

# Per-year step-hazard curves emitted by run_v2_0b_oof / score_snap_with_landmark.
HAZARD_CURVE_EVENTS = ["TOP_100_PROSPECT", "MLB_DEBUT", "ESTABLISHED_MLB",
                       "ELITE", "STAR"]
HK_STEPS = 10
HAZARD_CURVE = [f"hk{j}_{ev}" for ev in HAZARD_CURVE_EVENTS
                for j in range(1, HK_STEPS + 1)]

# Which stored curve feeds each output head's haz_cum_h anchor. STAR u ELITE ==
# STAR (ELITE's trigger components are a subset of STAR's — see train_v2_0b_prod
# m1 note), so the STAR_PLUS_ELITE anchor reads the STAR curve.
CURVE_FOR = {
    "TOP_100_PROSPECT": "TOP_100_PROSPECT",
    "MLB_DEBUT":        "MLB_DEBUT",
    "ESTABLISHED_MLB":  "ESTABLISHED_MLB",
    "STAR_PLUS_ELITE":  "STAR",
}

# --- feature vector -------------------------------------------------------
# Base (the v2.0 19-d set): collapsed probs + age/yip + yip interactions + scouting.
FEAT_BASE = (
    HAZARD_PROBS
    + ["age_at_snap_centered", "years_in_pro"]
    + [f"{p}_x_yip_centered" for p in HAZARD_PROBS]
    + SCOUTING_SUMMARY_COLS
)
# Horizon-specific anchors: the hazard model's own cumulative answer at h.
HAZ_CUM_H = [f"haz_cum_h_{e}" for e in EVENTS]
# Full conditional feature vector.
FEAT_COND = FEAT_BASE + HAZARD_CURVE + HAZ_CUM_H + ["h_centered"]


# --------------------------------------------------------------------------
# Base feature engineering (one row per player-snap; horizon-independent).
# --------------------------------------------------------------------------
def prep_base(df: pd.DataFrame, db: str, max_entry: int | None = None) -> pd.DataFrame:
    """Attach the horizon-independent features used by FEAT_BASE.

    Merges birth year (for age), builds yip-centered hazard interactions and
    point-in-time scouting summary cols. Does NOT filter on eligibility (callers
    that train apply that gate themselves; inference keeps all rows).
    """
    c = sqlite3.connect(db)
    birth = pd.read_sql("SELECT player_id, birth_date FROM prospects", c)
    c.close()
    birth["birth_year"] = pd.to_datetime(
        birth["birth_date"], errors="coerce").dt.year
    df = df.merge(birth[["player_id", "birth_year"]], on="player_id", how="left")
    df["age_at_snap_centered"] = (
        (df["snap_year"] - df["birth_year"]).fillna(22.0) - AGE_CENTER
    )
    df["years_in_pro"] = df["snap_offset"]
    df["yip_centered"] = df["snap_offset"] - YIP_CENTER
    for p in HAZARD_PROBS:
        if p in df.columns:
            df[f"{p}_x_yip_centered"] = df[p] * df["yip_centered"]
    df = attach_scouting_summary(df)
    if max_entry is not None and "entry_year" in df.columns:
        df = df[df.entry_year <= max_entry].copy()
    return df


def _haz_cum(df: pd.DataFrame, curve_event: str, h: int) -> np.ndarray:
    """Hazard model's own cumulative incidence by horizon h from its step curve:
    1 - prod_{j=1..min(h,HK_STEPS)} (1 - hk_j). Missing steps treated as 0 hazard."""
    n = len(df)
    surv = np.ones(n, dtype=np.float64)
    for j in range(1, min(h, HK_STEPS) + 1):
        col = f"hk{j}_{curve_event}"
        if col in df.columns:
            hk = df[col].to_numpy(dtype=np.float64)
            hk = np.nan_to_num(np.clip(hk, 0.0, 1.0), nan=0.0)
        else:
            hk = np.zeros(n, dtype=np.float64)
        surv *= (1.0 - hk)
    return 1.0 - surv


def add_cond_cols(df: pd.DataFrame, h: int) -> pd.DataFrame:
    """Stamp the horizon-h conditional features onto a base-prepped frame:
    fills/clips the raw hazard curves, the per-event haz_cum_h anchors and
    h_centered. Returns the same frame (mutated copy-safe)."""
    df = df.copy()
    for col in HAZARD_CURVE:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0).clip(0.0, 1.0)
    for ev in EVENTS:
        df[f"haz_cum_h_{ev}"] = _haz_cum(df, CURVE_FOR[ev], h)
    df["h"] = h
    df["h_centered"] = h - H_CENTER
    return df


# --------------------------------------------------------------------------
# Training: long-expand to resolved (row, h) pairs with per-horizon labels.
# --------------------------------------------------------------------------
def realized_by_h(df: pd.DataFrame, event: str, h: int) -> np.ndarray:
    """Per-horizon cumulative label: 1 if the event fired in (snap, snap+h].

    Uses trigger_<event> (the calendar fire year) vs snap_year. Already-fired
    (trigger <= snap) -> 0, consistent with the eligibility gate. Only call on
    rows where years_fwd >= h, otherwise negatives are right-censored."""
    trig = pd.to_numeric(df[f"trigger_{event}"], errors="coerce")
    snap = df["snap_year"].astype(float)
    fired = trig.notna() & (trig > snap) & (trig <= snap + h)
    return fired.to_numpy(dtype=np.float32)


def expand_long(df: pd.DataFrame, h_max: int = H_MAX) -> tuple[pd.DataFrame, np.ndarray]:
    """Expand one-row-per-snap into resolved (row, h) pairs for h in 1..h_max.

    A (row, h) pair is kept iff years_fwd >= h (the full h-year window is
    observed, so every event head's label is trustworthy). Returns (long_df with
    FEAT_COND columns populated, Y) where Y is (n_long, len(EVENTS)) float32.
    """
    frames: list[pd.DataFrame] = []
    ys: list[np.ndarray] = []
    for h in range(1, h_max + 1):
        sub = df[df["years_fwd"] >= h]
        if sub.empty:
            continue
        sub = add_cond_cols(sub, h)
        Y = np.empty((len(sub), len(EVENTS)), dtype=np.float32)
        for k, ev in enumerate(EVENTS):
            Y[:, k] = realized_by_h(sub, ev, h)
        frames.append(sub)
        ys.append(Y)
    if not frames:
        return df.iloc[0:0].copy(), np.empty((0, len(EVENTS)), dtype=np.float32)
    long_df = pd.concat(frames, ignore_index=True)
    Y = np.vstack(ys)
    return long_df, Y


# --------------------------------------------------------------------------
# Inference: sweep h to produce the per-year cumulative trajectory per event.
# --------------------------------------------------------------------------
def predict_trajectory(bundle: dict, df: pd.DataFrame,
                       h_max: int | None = None) -> pd.DataFrame:
    """Score a one-row-per-snap frame, emitting the full cumulative trajectory.

    Adds columns ``xp_<event>_h{1..H}`` (monotone non-decreasing in h, enforced
    via cummax) and an alias ``xp_<event>`` = the publish-horizon slice (h=6 by
    default). Horizons beyond H_MAX are NOT extrapolated — that range is the
    hazard layer's opinion, not the XGB's.
    """
    feat = bundle["feature_names"]
    scaler = bundle["scaler"]
    booster = bundle["model"]
    best_iter = bundle.get("best_iteration")
    events = bundle["events"]
    h_max = h_max or int(bundle.get("h_max", H_MAX))
    publish_h = int(bundle.get("publish_h", PUBLISH_H))

    out = df.copy()
    preds_by_h: dict[int, np.ndarray] = {}
    for h in range(1, h_max + 1):
        sub = add_cond_cols(df, h)
        X = scaler.transform(sub[feat].values.astype(np.float32))
        d = xgb.DMatrix(X, feature_names=list(feat))
        if best_iter is not None:
            P = booster.predict(d, iteration_range=(0, best_iter + 1))
        else:
            P = booster.predict(d)
        preds_by_h[h] = P  # (n, len(events))

    for k, ev in enumerate(events):
        M = np.column_stack([preds_by_h[h][:, k] for h in range(1, h_max + 1)])
        M = np.maximum.accumulate(M, axis=1)  # enforce monotone cumulative
        for hi, h in enumerate(range(1, h_max + 1)):
            out[f"xp_{ev}_h{h}"] = M[:, hi]
        pub = min(publish_h, h_max)
        out[f"xp_{ev}"] = out[f"xp_{ev}_h{pub}"]
    return out
