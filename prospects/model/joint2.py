"""v2.2 conditional scoring engine — bag bundles, raw features, h/yip calibration.

v2.2 (promoted from the exp4 experiment line, 2026-09-05) changes the joint
layer three ways relative to v2.1c:

  1. The bundle is a BAG of monotone XGBs (`monotone_constraints` +1 on
     `h_centered` and the horizon-margin features), averaged at inference —
     no scaler, no post-hoc reliance on cummax (kept only as a residual
     cleanup for the tiny leakage through the haz_cum anchors).
  2. The feature vector adds, on top of v2.1c's FEAT_COND: the hazard layer's
     per-event timing moments (mean_t/sd_t), p_ALL_STAR_ONCE/p_MAJOR_AWARD,
     explicit horizon margins (h - mean_t), and the top-160 RAW landmark panel
     features (`rw_*`) — the refinement head sees the evidence, not just the
     hazard layer's verdict on it.
  3. Calibration is ONE per-event logistic map fit on player-grouped
     cross-fitted (OOF) predictions — sigmoid over [logit(p), h_c, yip_c,
     interactions, quadratics] — instead of per-(event,h) maps fit on the val
     slice. Val is a pure reporting set again, and calibration is flat across
     career stage (yip).

Timing is read off the calibrated debut trajectory (CDF): median crossing +
q25-q75 window. Honest val vs v2.1c: debut AP@h3 0.685 vs 0.647, weighted
AP@h6 0.556 vs 0.514, timing MAE 0.97 vs 1.16 (Lasso).

This module is self-contained (no imports from model/train experiment
scripts) so pickled v2.2 artifacts resolve here forever.
"""
from __future__ import annotations

import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

from prospects.model.joint import (
    EVENTS, H_MAX, PUBLISH_H, add_cond_cols, predict_trajectory,
)

EPS = 1e-6

# ---- v2.2 feature stamping ------------------------------------------------
CURVE_EVS = ["TOP_100_PROSPECT", "MLB_DEBUT", "ESTABLISHED_MLB",
             "ELITE", "STAR"]
TIMING_FEATS = ([f"mean_t_{e}" for e in CURVE_EVS]
                + [f"sd_t_{e}" for e in CURVE_EVS])
EXTRA_PROBS = ["p_ALL_STAR_ONCE", "p_MAJOR_AWARD"]
MARGIN_FEATS = ["h_minus_mean_t_MLB_DEBUT", "h_minus_mean_t_ESTABLISHED_MLB",
                "z_h_debut"]
MEAN_T_SENTINEL = 15.0


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def stamp_extra_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Fill/derive the v2.2 extra features on a frame that has an `h` column
    (i.e. after add_cond_cols)."""
    out = df.copy()
    for c in TIMING_FEATS:
        if c not in out.columns:
            out[c] = MEAN_T_SENTINEL if c.startswith("mean_t") else 0.0
        elif c.startswith("mean_t"):
            out[c] = out[c].fillna(MEAN_T_SENTINEL)
        else:
            out[c] = out[c].fillna(0.0)
    for c in EXTRA_PROBS:
        out[c] = out[c].fillna(0.0) if c in out.columns else 0.0
    h = out["h"].astype(float)
    out["h_minus_mean_t_MLB_DEBUT"] = h - out["mean_t_MLB_DEBUT"]
    out["h_minus_mean_t_ESTABLISHED_MLB"] = h - out["mean_t_ESTABLISHED_MLB"]
    sd = out["sd_t_MLB_DEBUT"].astype(float).clip(lower=0.25)
    out["z_h_debut"] = (h - out["mean_t_MLB_DEBUT"]) / sd
    return out


# ---- calibrator -----------------------------------------------------------
class HYip2Calibrator:
    """Per-event calibration across all horizons and career stages:
    p_cal = sigmoid(b . [logit(p), h_c, yip_c, logit*h_c, logit*yip_c,
    h_c^2, yip_c^2]). Fit on cross-fitted (OOF) predictions only."""

    def __init__(self):
        self.lr = LogisticRegression(C=1e4, solver="lbfgs", max_iter=4000)

    @staticmethod
    def _feats(p, h, yip):
        lp = _logit(p)
        hc = np.asarray(h, dtype=np.float64) - 5.0
        yc = np.clip(np.asarray(yip, dtype=np.float64), 0, 10) - 3.0
        return np.column_stack([lp, hc, yc, lp * hc, lp * yc,
                                hc * hc, yc * yc])

    def fit(self, p, h, yip, y):
        self.lr.fit(self._feats(p, h, yip), np.asarray(y, dtype=int))
        return self

    def predict(self, p, h, yip):
        return self.lr.predict_proba(self._feats(p, h, yip))[:, 1]


# ---- raw landmark features ------------------------------------------------
def attach_raw_features(df: pd.DataFrame, db: str,
                        keep_raw: list[str],
                        panel_npz: str | Path | None = "auto",
                        verbose: bool = True) -> pd.DataFrame:
    """Attach the `rw_<name>` raw landmark-panel features as-of each row's
    snap_year, with FULL coverage: rows found in the landmark panel cache take
    its values (fast path; verified bit-identical to a fresh build), every
    other (player, snap) — pre-2007 snaps, post-panel snaps, the current
    scoring cohort — is built from the DB with the same feature builder the
    hazard panel uses. Rows whose rw_ columns already exist are left
    untouched.

    panel_npz: "auto" = runs/current/scratch/oof/panel_cache.npz if present;
    None = always build from the DB; or an explicit path."""
    missing = [c for c in keep_raw if c not in df.columns]
    if not missing:
        return df
    from prospects.features.scouting import FEATURE_NAMES
    from prospects.model.hazards.survival import build_windowed_features

    name_to_idx = {f"rw_{n}": i for i, n in enumerate(FEATURE_NAMES)}
    bad = [c for c in missing if c not in name_to_idx]
    if bad:
        raise ValueError(f"unknown raw feature names: {bad[:5]}")

    # Fast path: panel-cache lookup for covered (player, snap) keys.
    panel_hit: dict[tuple, np.ndarray] = {}
    if panel_npz == "auto":
        from prospects import config as _cfg
        cand = _cfg.run().scratch / "oof" / "panel_cache.npz"
        panel_npz = cand if cand.exists() else None
    if panel_npz is not None:
        d = np.load(panel_npz, allow_pickle=True)
        X_lm, p_pids, p_S = d["X_lm"], d["pids"], d["S_yrs"]
        if X_lm.shape[1] == len(FEATURE_NAMES):
            idxs0 = [name_to_idx[c] for c in missing]
            want = set(zip(df["player_id"], df["snap_year"].astype(int)))
            for i, (pp, ss) in enumerate(zip(p_pids, p_S)):
                k = (pp, int(ss))
                if k in want and k not in panel_hit:
                    panel_hit[k] = X_lm[i][idxs0].astype(np.float32)
        del d

    conn = sqlite3.connect(db)
    prospects = [dict(zip([d[0] for d in cur.description], row))
                 for cur in [conn.execute("""
        SELECT p.*, o.mlb_debut_year, o.year_established_mlb,
               o.year_top_100, o.year_top_25,
               o.year_all_star_once, o.year_all_star_three,
               o.year_major_award, o.year_hof_trajectory,
               o.events_json, o.final_mlb_year
        FROM prospects p
        JOIN career_outcomes o ON o.player_id = p.player_id""")]
                 for row in cur.fetchall()]
    stats_rows = conn.execute("SELECT * FROM season_stats")
    cols = [d[0] for d in stats_rows.description]
    stats_by_pid: dict[str, list[dict]] = {}
    for row in stats_rows.fetchall():
        d = dict(zip(cols, row))
        stats_by_pid.setdefault(d["player_id"], []).append(d)
    try:
        rank_rows = conn.execute(
            "SELECT player_id, CAST(substr(as_of, 1, 4) AS INTEGER), "
            "overall_rank, source FROM rankings_history "
            "WHERE overall_rank IS NOT NULL").fetchall()
    except Exception:
        rank_rows = []
    try:
        org_rows = conn.execute(
            "SELECT player_id, as_of, org_rank FROM rankings_history "
            "WHERE org_rank IS NOT NULL").fetchall()
    except Exception:
        org_rows = []
    conn.close()

    by_pid = {p["player_id"]: p for p in prospects}
    ranks_by_pid: dict[str, list] = {}
    for r in rank_rows:
        ranks_by_pid.setdefault(r[0], []).append((r[1], r[2], r[3]))
    orgs_by_pid: dict[str, list] = {}
    for r in org_rows:
        try:
            yr = int(str(r[1])[:4])
        except (ValueError, TypeError):
            continue
        orgs_by_pid.setdefault(r[0], []).append((yr, int(r[2])))
    for p in prospects:
        p["_top100_rankings"] = ranks_by_pid.get(p["player_id"], [])
        p["_org_rankings"] = orgs_by_pid.get(p["player_id"], [])

    idxs = [name_to_idx[c] for c in missing]
    cache: dict[tuple, np.ndarray] = dict(panel_hit)
    vals = np.full((len(df), len(missing)), np.nan, dtype=np.float32)
    n_miss = n_built = 0
    pid_arr = df["player_id"].to_numpy()
    snap_arr = df["snap_year"].to_numpy()
    for i in range(len(df)):
        key = (pid_arr[i], int(snap_arr[i]))
        if key not in cache:
            p = by_pid.get(key[0])
            if p is None:
                cache[key] = None
                n_miss += 1
            else:
                vec = build_windowed_features(
                    p, stats_by_pid.get(key[0], []), key[1], milb_only=True)
                cache[key] = np.asarray(vec, dtype=np.float32)[idxs]
                n_built += 1
        v = cache[key]
        if v is not None:
            vals[i] = v
    if verbose:
        print(f"  [joint2] raw features for {len(cache):,} (player,snap) "
              f"keys: {len(panel_hit):,} from panel cache, {n_built:,} "
              f"built from DB, {n_miss:,} players not in DB -> NaN")
    return pd.concat([df, pd.DataFrame(
        {c: vals[:, j] for j, c in enumerate(missing)}, index=df.index)],
        axis=1)


# ---- scoring --------------------------------------------------------------
def is_bag_bundle(bundle: dict) -> bool:
    return "models" in bundle and isinstance(bundle["models"], (list, tuple))


def predict_trajectory2(bundle: dict, df: pd.DataFrame,
                        h_max: int | None = None) -> pd.DataFrame:
    """Bag-averaged trajectory sweep. Emits xp_<ev>_h{1..H} (monotone via
    cummax residual cleanup) + xp_<ev> = publish-horizon alias, exactly the
    schema predict_trajectory produces."""
    feats = bundle["feature_names"]
    models = bundle["models"]
    events = bundle["events"]
    h_max = h_max or int(bundle.get("h_max", H_MAX))
    publish_h = int(bundle.get("publish_h", PUBLISH_H))

    out = df.copy()
    preds_by_h = {}
    for h in range(1, h_max + 1):
        sub = stamp_extra_cols(add_cond_cols(df, h))
        X = sub[feats].values.astype(np.float32)
        d = xgb.DMatrix(X, feature_names=list(feats))
        preds_by_h[h] = np.mean([m.predict(d) for m in models], axis=0)
    new = {}
    for k, ev in enumerate(events):
        M = np.column_stack([preds_by_h[h][:, k] for h in range(1, h_max + 1)])
        M = np.maximum.accumulate(M, axis=1)
        for hi, h in enumerate(range(1, h_max + 1)):
            new[f"xp_{ev}_h{h}"] = M[:, hi]
        new[f"xp_{ev}"] = M[:, min(publish_h, h_max) - 1]
    return pd.concat([out, pd.DataFrame(new, index=out.index)], axis=1)


def score_trajectory(xgb_pkl: str | Path, df: pd.DataFrame,
                     db: str) -> tuple[pd.DataFrame, dict]:
    """Load a joint bundle and score df's trajectory, dispatching on bundle
    shape: v2.2 bag bundles (raw features attached from the DB as needed) vs
    legacy v2.1c scaler bundles. Returns (scored_df, bundle)."""
    with open(xgb_pkl, "rb") as fh:
        bundle = pickle.load(fh)
    if is_bag_bundle(bundle):
        keep_raw = list(bundle.get("keep_raw", []))
        if keep_raw:
            df = attach_raw_features(df, db, keep_raw)
        return predict_trajectory2(bundle, df), bundle
    return predict_trajectory(bundle, df), bundle


# ---- calibration application ---------------------------------------------
def load_calibrators(path: str | Path) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def is_hyip2(cal_bundle: dict) -> bool:
    return str(cal_bundle.get("kind", "")).startswith("hyip2")


def make_cal_fn(cal_bundle: dict, df: pd.DataFrame):
    """Return _cal(values, event, h) working over BOTH calibrator formats.
    For hyip2, yip comes from df['snap_offset'] — values must be aligned with
    df row order (they always are: scores are columns of df)."""
    cals = cal_bundle["calibrators"]
    if is_hyip2(cal_bundle):
        yip = df["snap_offset"].to_numpy()

        def _cal(values, event, h):
            c = cals.get(event)
            if c is None:
                return values
            return c.predict(np.asarray(values, dtype=float),
                             np.full(len(values), h), yip)
        return _cal

    def _cal_legacy(values, event, h):
        c = cals.get((event, h))
        if c is None:
            return values
        return c.predict(values)
    return _cal_legacy


def apply_calibrators_frame(df: pd.DataFrame, cal_bundle: dict,
                            h_max: int = H_MAX) -> pd.DataFrame:
    """Calibrate every xp_<ev>_h{h} column in place (raw kept as *_raw),
    re-monotonizing each event's calibrated trajectory across h. Also
    refreshes the xp_<ev> publish alias."""
    _cal = make_cal_fn(cal_bundle, df)
    out = df.copy()
    for ev in EVENTS:
        cols = [f"xp_{ev}_h{h}" for h in range(1, h_max + 1)
                if f"xp_{ev}_h{h}" in out.columns]
        if not cols:
            continue
        M = np.column_stack([
            _cal(out[c].astype(float).to_numpy(), ev, int(c.rsplit("h", 1)[1]))
            for c in cols])
        M = np.maximum.accumulate(M, axis=1)
        for j, c in enumerate(cols):
            out[f"{c}_raw"] = out[c]
            out[c] = M[:, j]
        alias = f"xp_{ev}_h{min(PUBLISH_H, h_max)}"
        if alias in out.columns:
            out[f"xp_{ev}"] = out[alias]
    return out


# ---- timing from the trajectory CDF --------------------------------------
def cdf_timing(df: pd.DataFrame, ev: str = "MLB_DEBUT",
               h_max: int = H_MAX) -> pd.DataFrame:
    """Debut-time distribution from the (calibrated) trajectory:
    pmf_j = F(j) - F(j-1); conditional on event-by-h_max, t_mean / t_med /
    t_q25 / t_q75. NaN where the trajectory carries ~no event mass."""
    cols = [f"xp_{ev}_h{h}" for h in range(1, h_max + 1)]
    F = df[cols].to_numpy(dtype=np.float64)
    F = np.maximum.accumulate(F, axis=1)
    Ftot = F[:, -1]
    pmf = np.diff(np.column_stack([np.zeros(len(F)), F]), axis=1)
    j = np.arange(1, h_max + 1, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_mean = (pmf * j).sum(axis=1) / Ftot
        Fc = F / Ftot[:, None]

    def _q(q):
        idx = (Fc >= q).argmax(axis=1) + 1
        return np.where(Ftot > 1e-9, idx.astype(float), np.nan)

    return pd.DataFrame({
        "t_mean": np.where(Ftot > 1e-9, t_mean, np.nan),
        "t_med": _q(0.5), "t_q25": _q(0.25), "t_q75": _q(0.75),
    }, index=df.index)
