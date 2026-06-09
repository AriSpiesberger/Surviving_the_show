"""Granular per-yip × per-percentile validation.

Reports per-yip per-event the realized outcome rate in fine-grained percentile
slabs:

  TOP HEAD     (precision matters most here)
    0.0-0.5%   0.5-1.0%   1.0-1.5%   1.5-2.0%
  MID
    2-3%   3-4%   4-5%
  WIDER
    5-10%   10-20%   20-50%   bottom 50%

For each (yip, slab) cell, emits:
    score floor, score ceiling, mean score, n, realized rate, expected lift
    (rate/base_rate)

And a LOOK-BACK retrospective:
    Across all val players in slab, list the actual outcomes (cup/utility/
    regular/breakout if model_b available) and the names of the top picks
    to spot-check.

Output per event:
    val_<prefix>_<event>_pct_slabs.csv      machine-readable per-cell metrics
    val_<prefix>_<event>_top_lookback.csv   names + outcomes of top picks per yip

Console: per-yip × slab table for the primary event (default MLB_DEBUT).

Usage:
    python validate_full.py \\
        --long v1.14n_val_long.csv \\
        --lasso lasso_v1.14n_td.pkl \\
        --model-b model_b_outcomes_v1.14n.pkl \\
        --out-prefix v14n
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import io
import os
import pickle
import sqlite3

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, roc_auc_score,
)


def _make_out_dir(prefix: str) -> str:
    """results/<prefix>_<YYYY-MM-DD>/ — created if missing."""
    date = _dt.date.today().isoformat()
    out_dir = os.path.join("results", f"{prefix}_{date}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


class _Tee:
    """Mirror writes to multiple text streams (stdout + report buffer)."""
    def __init__(self, *streams):
        self._streams = streams
    def write(self, s):
        for st in self._streams:
            st.write(s)
    def flush(self):
        for st in self._streams:
            st.flush()

EVENTS = ["MLB_DEBUT", "TOP_100_PROSPECT", "ESTABLISHED_MLB", "STAR_PLUS_ELITE"]
SLABS = [
    (0.0, 0.5, "0.0-0.5%"),
    (0.5, 1.0, "0.5-1.0%"),
    (1.0, 1.5, "1.0-1.5%"),
    (1.5, 2.0, "1.5-2.0%"),
    (2.0, 3.0, "2-3%"),
    (3.0, 4.0, "3-4%"),
    (4.0, 5.0, "4-5%"),
    (5.0, 10.0, "5-10%"),
    (10.0, 20.0, "10-20%"),
    (20.0, 50.0, "20-50%"),
    (50.0, 100.0, "bottom 50%"),
]


_LEVEL_RANK = {  # higher = higher tier
    "RK": 0, "A-": 1, "A": 2, "A+": 3, "AA": 4, "AAA": 5, "MLB": 6,
}
_RANK_TO_LABEL = {0: "RK", 1: "A-", 2: "A", 3: "A+",
                   4: "AA", 5: "AAA", 6: "MLB"}


def _attach_current_level(df: pd.DataFrame, db: str) -> pd.DataFrame:
    """For each (player_id, snap_year), look up the highest level the
    player played at in snap_year (from season_stats). Adds a `cur_level`
    column with values in {RK, A-, A, A+, AA, AAA, MLB, NONE}.
    """
    c = sqlite3.connect(db)
    s = pd.read_sql(
        "SELECT player_id, season_year, level FROM season_stats", c)
    c.close()
    s = s.dropna(subset=["season_year"])
    s["season_year"] = s["season_year"].astype(int)
    s["rank"] = s["level"].astype(str).str.upper().map(_LEVEL_RANK)
    s = s.dropna(subset=["rank"])
    s["rank"] = s["rank"].astype(int)
    hi = (s.groupby(["player_id", "season_year"])["rank"].max()
            .rename("cur_rank").reset_index())
    df = df.merge(hi, left_on=["player_id", "snap_year"],
                   right_on=["player_id", "season_year"], how="left")
    df = df.drop(columns=["season_year"], errors="ignore")
    df["cur_level"] = (df["cur_rank"].map(_RANK_TO_LABEL)
                         .fillna("NONE"))
    df = df.drop(columns=["cur_rank"])
    return df


def _prepare_long(long_csv, age_center=22, yip_center=3,
                   db="prospects_snapshot.db"):
    """Load val long, add the standard centered + interaction columns, and
    filter to entry_year <= 2020 + eligible_{TOP_100,MLB_DEBUT}."""
    df = pd.read_csv(long_csv)
    df = df[df.entry_year <= 2020].copy()
    for ev in ("TOP_100_PROSPECT", "MLB_DEBUT"):
        df = df[df[f"eligible_{ev}"] == 1]
    c = sqlite3.connect(db)
    birth = pd.read_sql("SELECT player_id, birth_date FROM prospects", c)
    c.close()
    birth["birth_year"] = pd.to_datetime(
        birth["birth_date"], errors="coerce").dt.year
    df = df.merge(birth[["player_id", "birth_year"]],
                  on="player_id", how="left")
    df = _attach_current_level(df, db)
    df["age_at_snap_centered"] = (
        (df["snap_year"] - df["birth_year"]).fillna(22.0) - age_center
    )
    df["years_in_pro"] = df["snap_offset"]
    df["yip_centered"] = df["snap_offset"] - yip_center
    # Compute yip-interactions for every p_<event> column actually present
    # in the long. This way the same code handles v1.17/v1.18 (which carry
    # p_TOP_100_PROSPECT/p_MLB_DEBUT/p_ESTABLISHED_MLB/p_STAR_PLUS_ELITE etc)
    # and v1.19 (which carries p_Minors/p_MLB_service/p_All_Star).
    for col in [c for c in df.columns if c.startswith("p_")]:
        df[f"{col}_x_yip_centered"] = df[col] * df["yip_centered"]
    from prospects.features.scouting_grades import attach_scouting_summary
    df = attach_scouting_summary(df)  # point-in-time scouting summary cols
    return df


def score_with_lasso(long_csv, lasso_pkl, age_center=22, yip_center=3,
                      db="prospects_snapshot.db"):
    """Legacy single-lasso path: one `lasso_score` column used for every
    event. Kept for backwards compatibility with the v1.17 debut lasso."""
    df = _prepare_long(long_csv, age_center, yip_center, db)
    with open(lasso_pkl, "rb") as fh:
        m = pickle.load(fh)
    sc, lasso, feat = m["scaler"], m["lasso"], m["feature_names"]
    df["lasso_score"] = lasso.predict(sc.transform(df[feat].values))
    return df


def score_with_xgb(long_csv, xgb_pkl,
                    age_center=22, yip_center=3,
                    db="prospects_snapshot.db"):
    """v2.0 joint XGBoost path. Loads a multi-output XGBoost booster (saved
    by fit_joint_xgb_v2.py), scores each row, and writes one
    `lasso_score_<EVENT>` column per event head — same column shape as the
    per-event bundle so every downstream analysis works unchanged.
    """
    import xgboost as xgb
    with open(xgb_pkl, "rb") as fh:
        bundle = pickle.load(fh)
    df = _prepare_long(long_csv, age_center, yip_center, db)
    booster = bundle["model"]
    scaler = bundle["scaler"]
    feat = bundle["feature_names"]
    events = bundle["events"]
    best_iter = bundle.get("best_iteration")
    X = scaler.transform(df[feat].values.astype(np.float32))
    d = xgb.DMatrix(X, feature_names=list(feat))
    if best_iter is not None:
        P = booster.predict(d, iteration_range=(0, best_iter + 1))
    else:
        P = booster.predict(d)
    for k, ev in enumerate(events):
        df[f"lasso_score_{ev}"] = P[:, k]
    if "MLB_DEBUT" in events:
        df["lasso_score"] = df["lasso_score_MLB_DEBUT"]
    return df, list(events)


def score_with_lasso_bundle(long_csv, bundle_pkl,
                             age_center=22, yip_center=3,
                             db="prospects_snapshot.db"):
    """v1.18 multi-event path: writes one column per event,
    `lasso_score_<EVENT>`, holding the L1-logistic's predicted probability
    of the positive class (i.e., sigmoid of the decision function). Range
    is [0, 1], directly readable as P(event).

    Ranking is identical to using the raw logit, but score floors / slabs
    / thresholds are now interpretable as event probabilities.
    """
    with open(bundle_pkl, "rb") as fh:
        bundle = pickle.load(fh)
    age_center = bundle.get("age_center", age_center)
    yip_center = bundle.get("yip_center", yip_center)
    df = _prepare_long(long_csv, age_center, yip_center, db)

    per_event = bundle["per_event"]
    feat = bundle["feature_names"]
    for ev, art in per_event.items():
        sc = art["scaler"]
        lasso = art["lasso"]
        f_ev = art.get("feature_names", feat)
        # LogisticRegression: predict_proba[:, 1] = P(positive class).
        # This is the logistic/sigmoid transform of decision_function.
        prob = lasso.predict_proba(sc.transform(df[f_ev].values))[:, 1]
        df[f"lasso_score_{ev}"] = prob
    if "MLB_DEBUT" in per_event:
        df["lasso_score"] = df["lasso_score_MLB_DEBUT"]
    return df, list(per_event.keys())


def slab_metrics(scores, y, lo_pct, hi_pct):
    n = len(scores)
    lo = max(0, int(round(n*lo_pct/100)))
    hi = min(n, int(round(n*hi_pct/100)))
    if hi <= lo:
        return None
    order = np.argsort(-scores)
    idx = order[lo:hi]
    seg_scores = scores[idx]; seg_y = y[idx]
    return {
        "n": len(idx),
        "score_lo": float(seg_scores.min()),
        "score_hi": float(seg_scores.max()),
        "score_mean": float(seg_scores.mean()),
        "rate": float(seg_y.mean()),
        "tp": int(seg_y.sum()),
        "idx": idx,
    }


# Percentile cut points used by both cumulative-threshold and score-bucket
# views. Coarse 5% steps from bottom to 95th percentile, then 1% steps inside
# the top 5%. Direction note: percentile p here means "score floor at the p-th
# percentile of this yip's scores", so "above p" = top (100-p)% of players.
PCTILE_CUTS = list(range(5, 96, 5)) + [96, 97, 98, 99]


def cumulative_above_threshold(df, event):
    """Per (yip, percentile cut p): score floor T = score at p-th pctile;
    n_above = players with score >= T; rate_above = mean realized; lift vs base.

    Lets you pick (yip, T) and see what your pool looks like above that score.
    """
    rows = []
    for yip in sorted(df.snap_offset.unique()):
        sub = df[df.snap_offset == yip]
        if len(sub) < 50:
            continue
        elig = sub[sub[f"eligible_{event}"] == 1]
        if len(elig) < 50:
            continue
        y = elig[f"realized_{event}"].values.astype(int)
        if y.sum() == 0:
            continue
        scores = elig["lasso_score"].values
        base = float(y.mean())
        for p in PCTILE_CUTS:
            T = float(np.percentile(scores, p))
            mask = scores >= T
            n_above = int(mask.sum())
            if n_above == 0:
                continue
            rate = float(y[mask].mean())
            rows.append({
                "event": event, "yip": int(yip), "pctile_cut": p,
                "score_floor": T, "n_above": n_above,
                "rate_above": rate, "base_rate": base,
                "lift": rate / base if base > 0 else float("nan"),
                "tp_above": int(y[mask].sum()),
            })
    return pd.DataFrame(rows)


def per_yip_score_buckets(df, event):
    """Per (yip, score bucket): non-cumulative — players whose score falls
    between consecutive percentile cuts. Same percentile grid as the
    cumulative view, so the cuts line up.

    Buckets in ascending percentile order:
        [0,5), [5,10), ..., [90,95), [95,96), [96,97), [97,98), [98,99), [99,100]
    """
    bucket_edges = [0] + PCTILE_CUTS + [100]
    rows = []
    for yip in sorted(df.snap_offset.unique()):
        sub = df[df.snap_offset == yip]
        if len(sub) < 50:
            continue
        elig = sub[sub[f"eligible_{event}"] == 1]
        if len(elig) < 50:
            continue
        y = elig[f"realized_{event}"].values.astype(int)
        if y.sum() == 0:
            continue
        scores = elig["lasso_score"].values
        base = float(y.mean())
        # Score thresholds at every edge (inclusive of 0 = min, 100 = max+eps)
        edge_T = {p: float(np.percentile(scores, p)) for p in bucket_edges}
        for lo, hi in zip(bucket_edges[:-1], bucket_edges[1:]):
            T_lo, T_hi = edge_T[lo], edge_T[hi]
            if hi == 100:
                mask = scores >= T_lo
            else:
                mask = (scores >= T_lo) & (scores < T_hi)
            n = int(mask.sum())
            if n == 0:
                continue
            rate = float(y[mask].mean())
            rows.append({
                "event": event, "yip": int(yip),
                "pctile_lo": lo, "pctile_hi": hi,
                "bucket": f"p{lo}-p{hi}",
                "score_lo": T_lo, "score_hi": T_hi,
                "n": n, "tp": int(y[mask].sum()),
                "rate": rate, "base_rate": base,
                "lift": rate / base if base > 0 else float("nan"),
            })
    return pd.DataFrame(rows)


# Absolute lasso-score bins (stable across yip and draft bucket — unlike the
# percentile slabs above, these let you read off "what score threshold do I
# need" directly).
SCORE_BINS = [
    (-np.inf, 0.0,   "<0.0"),
    (0.0,     0.5,   "0.0-0.5"),
    (0.5,     1.0,   "0.5-1.0"),
    (1.0,     1.5,   "1.0-1.5"),
    (1.5,     2.0,   "1.5-2.0"),
    (2.0,     3.0,   "2.0-3.0"),
    (3.0,     5.0,   "3.0-5.0"),
    (5.0,     np.inf,"5.0+"),
]
DRAFT_BUCKETS = ["R1", "R2-R3", "R4-R10", "R10+", "IFA"]


def _score_bin_rows(elig_df, event, group_col, group_values):
    """Per (group_value, score_bin) → n, rate, base, lift, score range."""
    rows = []
    for gv in group_values:
        sub = elig_df[elig_df[group_col] == gv]
        if len(sub) < 30:
            continue
        y = sub[f"realized_{event}"].values.astype(int)
        if y.sum() == 0:
            continue
        scores = sub["lasso_score"].values
        base = float(y.mean())
        for lo, hi, label in SCORE_BINS:
            mask = (scores >= lo) & (scores < hi)
            n = int(mask.sum())
            if n == 0:
                continue
            rate = float(y[mask].mean())
            rows.append({
                "event": event, group_col: gv,
                "score_bin": label, "score_lo": lo, "score_hi": hi,
                "n": n, "tp": int(y[mask].sum()),
                "rate": rate, "base_rate": base,
                "lift": rate / base if base > 0 else float("nan"),
            })
    return pd.DataFrame(rows)


def score_vs_yip(df, event):
    elig = df[df[f"eligible_{event}"] == 1]
    return _score_bin_rows(elig, event, "snap_offset",
                           sorted(elig.snap_offset.unique()))


def score_vs_draft_bucket(df, event):
    elig = df[df[f"eligible_{event}"] == 1]
    buckets = [b for b in DRAFT_BUCKETS if b in set(elig.bucket.unique())]
    return _score_bin_rows(elig, event, "bucket", buckets)


def walkforward_summary(df, event):
    """Per snap_offset: n, pos, base%, pred%, AU-PR (avg precision),
    AU-PR_lift, lift@5, ECE.

    Dropping AUC: with rare positives + many easy negatives, AUC is dominated
    by the negative ranking. AU-PR (= average precision) integrates precision
    over recall and is the balanced rare-positive metric.
    """
    from sklearn.metrics import average_precision_score
    rows = []
    for off in sorted(df.snap_offset.unique()):
        sub = df[(df.snap_offset == off) & (df[f"eligible_{event}"] == 1)]
        if len(sub) < 50:
            continue
        y = sub[f"realized_{event}"].values.astype(int)
        if y.sum() == 0:
            rows.append({"event": event, "snap_offset": int(off),
                         "n": len(sub), "pos": 0,
                         "base_pct": 0.0, "pred_pct": float("nan"),
                         "auc": float("nan"), "lift@5": float("nan"),
                         "ece": float("nan")})
            continue
        # Predicted prob for this event: use whatever ranker we're
        # validating (lasso first, raw hazard only as last resort).
        # Priority:
        #   1) lasso_score_<event>  — per-event lasso bundle: ALREADY a
        #      probability (logistic of decision_function).
        #   2) lasso_score          — single composite lasso (v1.17-prod):
        #      raw regression score, sigmoid for an ECE-comparable prob.
        #   3) p_<event>            — raw hazard if no lasso scored at all.
        if f"lasso_score_{event}" in sub.columns:
            p = sub[f"lasso_score_{event}"].values
        elif "lasso_score" in sub.columns:
            p = 1.0 / (1.0 + np.exp(-sub["lasso_score"].values))
        else:
            p = sub[f"p_{event}"].values
        base = float(y.mean())
        pred = float(p.mean())
        try:
            ap = float(average_precision_score(y, p))
        except Exception:
            ap = float("nan")
        # Lift @ top 2% by predicted prob (where buy-list cutoffs actually live)
        n = len(sub)
        k = max(1, int(round(0.02 * n)))
        order = np.argsort(-p)[:k]
        rate_top = float(y[order].mean())
        lift2 = rate_top / base if base > 0 else float("nan")
        # ECE: 10 equal-width bins on p
        bins = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
        ece = 0.0
        for b in range(10):
            m = idx == b
            if not m.any():
                continue
            ece += (m.mean()) * abs(p[m].mean() - y[m].mean())
        ap_lift = ap / base if base > 0 else float("nan")
        rows.append({"event": event, "snap_offset": int(off),
                     "n": int(n), "pos": int(y.sum()),
                     "base_pct": base * 100, "pred_pct": pred * 100,
                     "au_pr": ap, "au_pr_lift": ap_lift,
                     "lift@2": lift2, "ece": ece})
    return pd.DataFrame(rows)


def per_yip_threshold(df, event, target_precision=0.60, min_n=10):
    """For each yip, find the lowest lasso_score s.t. cumulative-from-top
    realized rate stays at or above target_precision (with at least min_n
    players above the cut).

    Returns rows: yip, threshold, n_above, n_pos_above, precision, recall.
    """
    rows = []
    elig = df[df[f"eligible_{event}"] == 1].copy()
    for yip in sorted(elig.snap_offset.unique()):
        sub = elig[elig.snap_offset == yip]
        if len(sub) < min_n:
            continue
        y_all = sub[f"realized_{event}"].values.astype(int)
        n_pos_total = int(y_all.sum())
        if n_pos_total == 0:
            continue
        # Sort by score descending; walk from top, find largest k s.t.
        # cumulative realized rate >= target AND k >= min_n
        order = np.argsort(-sub["lasso_score"].values)
        scores_sorted = sub["lasso_score"].values[order]
        y_sorted = y_all[order]
        cum_tp = np.cumsum(y_sorted)
        counts = np.arange(1, len(y_sorted) + 1)
        rate = cum_tp / counts
        ok = (rate >= target_precision) & (counts >= min_n)
        if not ok.any():
            rows.append({"yip": int(yip), "threshold": float("nan"),
                         "n_above": 0, "tp_above": 0,
                         "precision": float("nan"), "recall": float("nan"),
                         "n_total": len(sub), "n_pos_total": n_pos_total})
            continue
        k = int(counts[ok].max())
        thr = float(scores_sorted[k - 1])
        prec = float(rate[k - 1])
        rec = cum_tp[k - 1] / n_pos_total if n_pos_total else float("nan")
        rows.append({"yip": int(yip), "threshold": thr,
                     "n_above": k, "tp_above": int(cum_tp[k - 1]),
                     "precision": prec, "recall": float(rec),
                     "n_total": len(sub), "n_pos_total": n_pos_total})
    return pd.DataFrame(rows)


def _score_time_to_debut(df, model_pkl):
    """Score df with the time-to-debut regression. Adds `t_pred` column
    (predicted years-to-debut)."""
    with open(model_pkl, "rb") as fh:
        m = pickle.load(fh)
    sc, lasso, feat = m["scaler"], m["lasso"], m["feature_names"]
    if "p_debut_lasso" in feat and "p_debut_lasso" not in df.columns:
        # The regression expects p_debut_lasso — synthesize it from
        # lasso_score_MLB_DEBUT if available (it's the same thing).
        if "lasso_score_MLB_DEBUT" in df.columns:
            df["p_debut_lasso"] = df["lasso_score_MLB_DEBUT"]
        else:
            df["p_debut_lasso"] = 0.0  # fallback
    df["t_pred"] = lasso.predict(sc.transform(df[feat].values))
    return df


def per_current_level(df, event):
    """For each cur_level value (RK, A-, A, A+, AA, AAA, NONE — MLB excluded
    because debut already triggered), report:
      n      eligible rows at that current level
      pos    realized positives
      base%  realized rate
      AU-PR  average precision on lasso_score
      lift@5 realized-rate@top-5 / base
      lift@10 realized-rate@top-10 / base
    """
    from sklearn.metrics import average_precision_score
    order = ["NONE", "RK", "A-", "A", "A+", "AA", "AAA"]
    rows = []
    elig = df[df[f"eligible_{event}"] == 1].copy()
    if "cur_level" not in elig.columns:
        return pd.DataFrame()
    for lv in order:
        sub = elig[elig["cur_level"] == lv]
        if len(sub) < 30:
            continue
        y = sub[f"realized_{event}"].values.astype(int)
        n = len(sub)
        n_pos = int(y.sum())
        base = float(y.mean()) if n else float("nan")
        s = sub["lasso_score"].values
        try:
            ap = (float(average_precision_score(y, s))
                  if n_pos and n_pos < n else float("nan"))
        except Exception:
            ap = float("nan")
        # Top-5 / top-10 lift
        def _lift(pct):
            k = max(1, int(round(pct/100 * n)))
            top = np.argsort(-s)[:k]
            rate = float(y[top].mean())
            return rate / base if base > 0 else float("nan"), rate
        l5, r5 = _lift(5)
        l10, r10 = _lift(10)
        rows.append({"event": event, "cur_level": lv,
                     "n": n, "pos": n_pos, "base_pct": base * 100,
                     "au_pr": ap, "lift@5": l5, "rate@5": r5 * 100,
                     "lift@10": l10, "rate@10": r10 * 100})
    return pd.DataFrame(rows)


def per_yip_slabs(df, event, slabs):
    rows = []
    base_by_yip = {}
    for yip in sorted(df.snap_offset.unique()):
        sub = df[df.snap_offset == yip]
        if len(sub) < 100: continue
        y = sub[f"realized_{event}"].values.astype(int)
        if y.sum() == 0: continue
        base = y.mean(); base_by_yip[yip] = base
        scores = sub["lasso_score"].values
        for lo, hi, label in slabs:
            r = slab_metrics(scores, y, lo, hi)
            if r is None: continue
            rows.append({
                "event": event, "yip": yip, "slab": label,
                "slab_lo": lo, "slab_hi": hi,
                "n": r["n"], "tp": r["tp"], "rate": r["rate"],
                "base_rate": base, "lift": r["rate"]/base if base>0 else np.nan,
                "score_lo": r["score_lo"], "score_hi": r["score_hi"],
                "score_mean": r["score_mean"],
            })
    return pd.DataFrame(rows), base_by_yip


def lookback_top_picks(df, event, slabs, top_slab_idx=4, model_b=None, scaler_b=None,
                      classes_b=None, feat_names_b=None):
    """Capture names and full outcome distribution for the union of the top
    `top_slab_idx` slabs (default 4 = 0-2%) per yip."""
    rows = []
    for yip in sorted(df.snap_offset.unique()):
        sub = df[df.snap_offset == yip]
        if len(sub) < 100: continue
        y = sub[f"realized_{event}"].values.astype(int)
        if y.sum() == 0: continue
        scores = sub["lasso_score"].values
        n = len(sub)
        # Top union of slabs
        top_hi = max(slabs[i][1] for i in range(min(top_slab_idx, len(slabs))))
        k = max(1, int(round(n*top_hi/100)))
        order = np.argsort(-scores)[:k]
        seg = sub.iloc[order].copy()
        seg["realized"] = seg[f"realized_{event}"].astype(int).values
        # Slab assignment within this top
        seg["slab"] = ""
        n_total = n
        for j, idx in enumerate(order):
            pct = (j+1) / n_total * 100
            for lo, hi, label in slabs[:top_slab_idx]:
                if lo < pct <= hi:
                    seg.iloc[j, seg.columns.get_loc("slab")] = label
                    break
        # Add model_b outcomes if provided
        if model_b is not None:
            # logit features
            eps = 1e-6
            feat_df = pd.DataFrame()
            haz_map = {
                "logit_p_TOP_100_PROSPECT": "p_TOP_100_PROSPECT",
                "logit_p_MLB_DEBUT": "p_MLB_DEBUT",
                "logit_p_ESTABLISHED_MLB": "p_ESTABLISHED_MLB",
                "logit_p_STAR_PLUS_ELITE": "p_STAR_PLUS_ELITE",
            }
            for lc, sc_col in haz_map.items():
                p = seg[sc_col].clip(eps, 1-eps).values
                feat_df[lc] = np.log(p / (1-p))
            feat_df["yrs_pre_debut"] = seg["snap_offset"].astype(float).values
            buckets = ["b_R1","b_R10+","b_R2-R3","b_R4-R10"]
            for b in buckets:
                feat_df[b] = (seg["bucket"] == b.replace("b_","")).astype(float).values
            # pos_grp default OTH
            feat_df["pos_IF"] = 0.0; feat_df["pos_OF"] = 0.0; feat_df["pos_OTH"] = 0.0
            try:
                feat_df = feat_df[feat_names_b]
                P = model_b.predict_proba(scaler_b.transform(feat_df.values))
                for i, c in enumerate(classes_b):
                    seg[f"p_b_{c}"] = P[:, i]
            except Exception as e:
                print(f"  WARN model_b apply failed: {e}")
        rows.append(seg)
    if not rows: return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    keep = ["snap_offset","slab","player_id","name","bucket","lasso_score",
            f"realized_{event}","realized",
            "p_MLB_DEBUT","p_TOP_100_PROSPECT","p_ESTABLISHED_MLB","p_STAR_PLUS_ELITE"]
    keep += [c for c in out.columns if c.startswith("p_b_")]
    keep = [c for c in keep if c in out.columns]
    return out[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", required=True)
    ap.add_argument("--lasso", default=None,
                    help="Path to v1.17 single-lasso .pkl (one score for "
                         "all events). Mutually exclusive with --lasso-bundle.")
    ap.add_argument("--lasso-bundle", default=None,
                    help="Path to v1.18+ per-event lasso bundle .pkl. When "
                         "set, each event is ranked by its own lasso logit.")
    ap.add_argument("--xgb-model", default=None,
                    help="Path to v2.0 joint XGBoost .pkl. Per-event scores "
                         "come from the multi-output booster's heads.")
    ap.add_argument("--model-b", default=None)
    ap.add_argument("--out-prefix", default="val")
    ap.add_argument("--db", default="prospects_snapshot.db")
    ap.add_argument("--events", nargs="+", default=EVENTS)
    ap.add_argument("--cohort2021-long", default=None,
                    help="Path to long file scored on 2021 draftees (e.g. "
                         "walkforward_v1.14n_2021_long.csv). If set, runs a "
                         "look-back at snap=2022 and snap=2023.")
    ap.add_argument("--time-to-debut-model", default=None,
                    help="Path to a time-to-debut regression .pkl. If "
                         "given, reports MAE / Spearman by snap_offset for "
                         "the MLB_DEBUT event.")
    ap.add_argument("--target-precision", type=float, default=0.60,
                    help="Per-yip MLB_DEBUT threshold target (default 0.60).")
    args = ap.parse_args()

    n_models = sum(x is not None for x in
                    (args.lasso, args.lasso_bundle, args.xgb_model))
    if n_models != 1:
        ap.error("Pass exactly one of --lasso / --lasso-bundle / --xgb-model.")

    out_dir = _make_out_dir(args.out_prefix)
    report_buf = io.StringIO()
    import sys as _sys
    _real_stdout = _sys.stdout
    _sys.stdout = _Tee(_real_stdout, report_buf)

    print(f"validate_full.py  prefix={args.out_prefix}  out_dir={out_dir}")
    print(f"  long={args.long}")
    print(f"  lasso={args.lasso}")
    print(f"  lasso_bundle={args.lasso_bundle}")
    print(f"  model_b={args.model_b}")
    print(f"  db={args.db}")
    print(f"  date={_dt.date.today().isoformat()}\n")

    bundle_events = None
    if args.lasso_bundle:
        df, bundle_events = score_with_lasso_bundle(
            args.long, args.lasso_bundle, db=args.db)
        print(f"scored val (per-event lasso bundle): {len(df):,} rows, "
              f"{df.player_id.nunique():,} players")
        print(f"  bundle events: {bundle_events}")
    elif args.xgb_model:
        df, bundle_events = score_with_xgb(
            args.long, args.xgb_model, db=args.db)
        print(f"scored val (joint XGBoost): {len(df):,} rows, "
              f"{df.player_id.nunique():,} players")
        print(f"  xgb events: {bundle_events}")
    else:
        df = score_with_lasso(args.long, args.lasso, db=args.db)
        print(f"scored val: {len(df):,} rows, "
              f"{df.player_id.nunique():,} players")

    # Optional model B
    model_b = scaler_b = classes_b = feat_names_b = None
    if args.model_b:
        with open(args.model_b, "rb") as fh:
            mb = pickle.load(fh)
        model_b = mb["model"]; scaler_b = mb["scaler"]
        classes_b = mb["classes"]; feat_names_b = mb["feature_names"]
        print(f"loaded model B: classes={classes_b}")

    for ev in args.events:
        print(f"\n{'='*78}\nEVENT: {ev}\n{'='*78}")
        # When a per-event bundle is loaded, point `lasso_score` at that
        # event's column so every downstream function (slabs, cumulative,
        # buckets, score_vs_yip, score_vs_bucket, top-lookback) ranks by
        # the event-specific logit without further refactoring.
        if bundle_events is not None:
            col = f"lasso_score_{ev}"
            if col not in df.columns:
                print(f"  (no {col} in bundle — skipping {ev})")
                continue
            df["lasso_score"] = df[col]
            print(f"  ranking by {col} "
                  f"(range [{df[col].min():+.2f}, {df[col].max():+.2f}])")
        slab_df, base_by_yip = per_yip_slabs(df, ev, SLABS)
        if slab_df.empty:
            print("  (no data)"); continue
        out_slab = os.path.join(out_dir, f"{ev}_pct_slabs.csv")
        slab_df.to_csv(out_slab, index=False)
        print(f"saved {out_slab}")

        # Pretty pivot per event
        rate = slab_df.pivot(index="slab", columns="yip", values="rate").reindex([s[2] for s in SLABS])
        n    = slab_df.pivot(index="slab", columns="yip", values="n").reindex([s[2] for s in SLABS])
        sc_lo = slab_df.pivot(index="slab", columns="yip", values="score_lo").reindex([s[2] for s in SLABS])
        print(f"\n--- realized {ev} rate (%) by yip × slab ---")
        print(rate.map(lambda x: f"{x*100:>5.1f}" if pd.notna(x) else "  -  ").to_string())
        print(f"\n--- n per cell ---")
        print(n.map(lambda x: f"{int(x):>5d}" if pd.notna(x) else "     ").to_string())
        print(f"\n--- score floor per cell ---")
        print(sc_lo.map(lambda x: f"{x:>+6.3f}" if pd.notna(x) else "       ").to_string())
        print(f"\n--- base rate by yip ---")
        for y, b in sorted(base_by_yip.items()):
            print(f"  yip={y}: {b:.2%}")

        # ---- A. Cumulative rate above a score floor (per yip × percentile cut)
        cum_df = cumulative_above_threshold(df, ev)
        if not cum_df.empty:
            out_cum = os.path.join(out_dir, f"{ev}_cum_above_threshold.csv")
            cum_df.to_csv(out_cum, index=False)
            print(f"\nsaved {out_cum}")
            print(f"\n--- {ev}: rate above score floor (%) by yip × pctile cut "
                  f"[cut p = top (100-p)%] ---")
            rate_piv = cum_df.pivot(index="pctile_cut", columns="yip",
                                    values="rate_above")
            print(rate_piv.map(
                lambda x: f"{x*100:>5.1f}" if pd.notna(x) else "  -  "
            ).to_string())
            print(f"\n--- {ev}: n above score floor by yip × pctile cut ---")
            n_piv = cum_df.pivot(index="pctile_cut", columns="yip",
                                 values="n_above")
            print(n_piv.map(
                lambda x: f"{int(x):>5d}" if pd.notna(x) else "     "
            ).to_string())
            print(f"\n--- {ev}: score floor by yip × pctile cut ---")
            sf_piv = cum_df.pivot(index="pctile_cut", columns="yip",
                                  values="score_floor")
            print(sf_piv.map(
                lambda x: f"{x:>+6.3f}" if pd.notna(x) else "       "
            ).to_string())

        # B/C/D removed in v2.1 report cleanup:
        #   B = per-yip percentile-bucket pivot (redundant with the slab
        #       table above, just finer-grained)
        #   C = absolute score bin × yip   (fixed bins don't fit lasso
        #       logit scale)
        #   D = absolute score bin × bucket (same reason)

        # ---- MLB_DEBUT extras: per-yip ≥X% threshold + time-to-debut
        if ev == "MLB_DEBUT":
            thr_df = per_yip_threshold(df, ev,
                                        target_precision=args.target_precision)
            if not thr_df.empty:
                out_thr = os.path.join(
                    out_dir,
                    f"{ev}_thresholds_at_p{int(args.target_precision*100)}.csv")
                thr_df.to_csv(out_thr, index=False)
                print(f"\nsaved {out_thr}")
                tgt_pct = int(args.target_precision * 100)
                print(f"\n--- {ev}: per-yip threshold for "
                      f">={tgt_pct}% precision ---")
                print(f"{'yip':>4} {'threshold':>10} {'n_above':>8} "
                      f"{'tp_above':>9} {'precision':>10} "
                      f"{'recall':>8} {'n_total':>8}")
                for _, r in thr_df.iterrows():
                    fmt = lambda v, w, p: (f"{v:>{w}.{p}f}" if pd.notna(v)
                                            else f"{'nan':>{w}}")
                    print(f"{int(r.yip):>4d} {fmt(r.threshold, 10, 3)} "
                          f"{int(r.n_above):>8d} {int(r.tp_above):>9d} "
                          f"{fmt(r.precision, 10, 1 if False else 3)} "
                          f"{fmt(r.recall, 8, 3)} "
                          f"{int(r.n_total):>8d}")

            if args.time_to_debut_model:
                from scipy.stats import spearmanr
                from sklearn.metrics import mean_absolute_error
                print(f"\nscoring time-to-debut with "
                      f"{args.time_to_debut_model}")
                df_t = _score_time_to_debut(df.copy(),
                                             args.time_to_debut_model)
                # Only debutees with a known mlb_debut_year and pre-debut snap
                deb = df_t.dropna(subset=["mlb_debut_year"]).copy()
                deb["mlb_debut_year"] = deb["mlb_debut_year"].astype(int)
                deb = deb[deb.snap_year < deb.mlb_debut_year]
                deb["actual_h"] = deb["mlb_debut_year"] - deb["snap_year"]
                if len(deb):
                    out_t = os.path.join(out_dir,
                                          f"{ev}_time_to_debut.csv")
                    deb[["player_id", "snap_year", "snap_offset",
                          "mlb_debut_year", "actual_h", "t_pred"]
                        ].to_csv(out_t, index=False)
                    print(f"saved {out_t}")
                    mae = float(mean_absolute_error(
                        deb["actual_h"], deb["t_pred"]))
                    sp = float(spearmanr(deb["t_pred"],
                                          deb["actual_h"]).correlation)
                    print(f"\n--- {ev}: time-to-debut (n={len(deb)} "
                          f"debutee-snaps, {deb.player_id.nunique()} "
                          f"players) ---")
                    print(f"  overall: MAE={mae:.2f}  Spearman={sp:+.3f}")
                    print(f"\n  {'snap_off':>8} {'n':>5} {'spearman':>10} "
                          f"{'MAE':>5} {'mean_act':>9} {'mean_pred':>10}")
                    for so in sorted(deb.snap_offset.unique()):
                        s = deb[deb.snap_offset == so]
                        if len(s) < 20:
                            continue
                        sp_so = float(spearmanr(
                            s["t_pred"], s["actual_h"]).correlation)
                        mae_so = float(mean_absolute_error(
                            s["actual_h"], s["t_pred"]))
                        print(f"  {int(so):>8d} {len(s):>5d} "
                              f"{sp_so:>+9.3f} {mae_so:>5.2f} "
                              f"{s.actual_h.mean():>8.2f} "
                              f"{s.t_pred.mean():>9.2f}")

            pl_df = per_current_level(df, ev)
            if not pl_df.empty:
                out_pl = os.path.join(out_dir, f"{ev}_per_current_level.csv")
                pl_df.to_csv(out_pl, index=False)
                print(f"\nsaved {out_pl}")
                print(f"\n--- {ev}: performance conditional on current "
                      f"MiLB level ---")
                print(f"{'cur_level':>10} {'n':>6} {'pos':>5} "
                      f"{'base%':>6} {'AU-PR':>6} "
                      f"{'lift@5':>7} {'rate@5%':>8} "
                      f"{'lift@10':>8} {'rate@10%':>9}")
                for _, r in pl_df.iterrows():
                    fmt = lambda v, w, p: (f"{v:>{w}.{p}f}" if pd.notna(v)
                                            else f"{'nan':>{w}}")
                    print(f"{r.cur_level:>10} {int(r.n):>6d} {int(r.pos):>5d} "
                          f"{fmt(r.base_pct, 6, 2)} {fmt(r.au_pr, 6, 3)} "
                          f"{fmt(r['lift@5'], 7, 2)} {fmt(r['rate@5'], 8, 2)} "
                          f"{fmt(r['lift@10'], 8, 2)} "
                          f"{fmt(r['rate@10'], 9, 2)}")

        # ---- E. Walk-forward summary (per snap_offset)
        wf_df = walkforward_summary(df, ev)
        if not wf_df.empty:
            out_wf = os.path.join(out_dir, f"{ev}_walkforward.csv")
            wf_df.to_csv(out_wf, index=False)
            print(f"\nsaved {out_wf}")
            print(f"\n--- {ev}: walk-forward by snap_offset ---")
            print(f"{'offset':>6} {'n':>5} {'pos':>4} {'base%':>6} "
                  f"{'pred%':>6} {'AU-PR':>6} {'AU-PR_lift':>10} "
                  f"{'lift@2':>7} {'ECE':>6}")
            for _, r in wf_df.iterrows():
                fmt = lambda v, w, p: (f"{v:>{w}.{p}f}" if pd.notna(v)
                                       else f"{'nan':>{w}}")
                print(f"{r.snap_offset:>6d} {r.n:>5d} {r.pos:>4d} "
                      f"{fmt(r.base_pct, 6, 2)} {fmt(r.pred_pct, 6, 2)} "
                      f"{fmt(r.au_pr, 6, 3)} {fmt(r.au_pr_lift, 10, 2)} "
                      f"{fmt(r['lift@2'], 7, 2)} {fmt(r.ece, 6, 3)}")

        # Look-back: actual top picks
        lb = lookback_top_picks(df, ev, SLABS, top_slab_idx=4,
                                model_b=model_b, scaler_b=scaler_b,
                                classes_b=classes_b, feat_names_b=feat_names_b)
        if not lb.empty:
            out_lb = os.path.join(out_dir, f"{ev}_top_lookback.csv")
            lb.to_csv(out_lb, index=False)
            print(f"\nsaved {out_lb}  ({len(lb)} top-2% rows)")
            # Show summary: per yip × slab → realized rate
            agg = lb.groupby(["snap_offset","slab"]).agg(
                n=("realized","size"), realized=("realized","sum")
            ).reset_index()
            agg["rate"] = agg["realized"]/agg["n"]
            print("\nlook-back per yip × slab (top 0-2%):")
            print(agg.to_string(index=False))

    if args.cohort2021_long:
        cohort2021_lookback(args.cohort2021_long, args.lasso, args.out_prefix,
                            db=args.db, out_dir=out_dir)

    _sys.stdout = _real_stdout
    report_path = os.path.join(out_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report_buf.getvalue())
    print(f"\nwrote {report_path}")


def _unused_cohort2021_lookback_placeholder():
    pass


def cohort2021_lookback(long_csv, lasso_pkl, prefix, db="prospects_snapshot.db",
                        out_dir="."):
    """Score 2021 draftees with the same lasso at snap=2022 and snap=2023,
    bin by per-yip percentile slab, and report realized rates so far."""
    print(f"\n{'='*78}\n2021 DRAFTEE LOOK-BACK — apply same lasso at snap 2022/2023\n{'='*78}")
    df = pd.read_csv(long_csv)
    print(f"  loaded {len(df):,} rows, {df.player_id.nunique():,} players")
    # Walk-forward file already has cumulative event probs at each snap
    c = sqlite3.connect(db)
    birth = pd.read_sql("SELECT player_id, birth_date FROM prospects", c); c.close()
    birth["birth_year"] = pd.to_datetime(birth["birth_date"], errors="coerce").dt.year
    df = df.merge(birth[["player_id","birth_year"]], on="player_id", how="left")
    df["age_at_snap_centered"] = ((df["snap_year"]-df["birth_year"]).fillna(22.0) - 22)
    df["years_in_pro"] = df["snap_offset"]
    df["yip_centered"] = df["snap_offset"] - 3
    # walkforward long has BOTH p_<event> (Beta-calibrated) AND p_<event>_raw.
    # Lasso was trained on raw hazards, so drop the calibrated ones first then
    # rename _raw -> base name to match feature names.
    drop_cal = [c for c in df.columns
                if c.startswith("p_") and not c.endswith("_raw")
                and (c + "_raw") in df.columns]
    df = df.drop(columns=drop_cal)
    rename_map = {}
    for col in df.columns:
        if col.endswith("_raw") and col.startswith("p_"):
            rename_map[col] = col[:-4]
    df = df.rename(columns=rename_map)
    for ev in ("TOP_100_PROSPECT","MLB_DEBUT","ESTABLISHED_MLB","STAR_PLUS_ELITE"):
        if f"p_{ev}" not in df.columns: continue
        df[f"p_{ev}_x_yip_centered"] = df[f"p_{ev}"] * df["yip_centered"]
    with open(lasso_pkl,"rb") as fh:
        m = pickle.load(fh)
    sc, lasso, feat = m["scaler"], m["lasso"], m["feature_names"]
    missing = [f for f in feat if f not in df.columns]
    if missing:
        print(f"  WARN missing features in 2021 long: {missing}")
        return
    df["lasso_score"] = lasso.predict(sc.transform(df[feat].values))

    # walkforward long uses 'realized_after_snap_<E>' (or similar); detect
    real_col = None
    for cand in ("realized_after_snap_MLB_DEBUT","realized_MLB_DEBUT"):
        if cand in df.columns: real_col = cand; break
    if real_col is None:
        # try any "realized_*MLB_DEBUT" column
        for c in df.columns:
            if c.startswith("realized") and "MLB_DEBUT" in c:
                real_col = c; break
    if real_col is None:
        print("  WARN no realized_MLB_DEBUT col in 2021 long")
        return
    df["realized_MLB_DEBUT"] = df[real_col].astype(int)

    print(f"  scored 2021 cohort with same lasso (features match)")
    print(f"  realized MLB_DEBUT by snap (out of {df.player_id.nunique()} players):")
    for snap in sorted(df.snap_year.unique()):
        sub = df[df.snap_year == snap]
        print(f"    snap={int(snap)}  yip={int(sub.snap_offset.iloc[0])}  "
              f"n={len(sub)}  realized={int(sub.realized_MLB_DEBUT.sum())} "
              f"({sub.realized_MLB_DEBUT.mean():.1%})")

    # Per-snap slab analysis (snap=2022, 2023)
    rows = []
    for snap in (2022, 2023):
        sub = df[df.snap_year == snap]
        if len(sub) < 50: continue
        y = sub["realized_MLB_DEBUT"].values
        if y.sum() == 0:
            print(f"\n  snap={snap}: zero realized — no signal to evaluate yet")
            continue
        base = y.mean()
        print(f"\n--- 2021 draftees at snap={snap} (yip={int(sub.snap_offset.iloc[0])}) ---")
        print(f"  n={len(sub)}  base_rate={base:.1%}  realized={int(y.sum())}")
        print(f"  {'slab':>10} {'n':>4} {'TP':>4} {'rate':>7} {'lift':>5} {'score_lo':>9} {'score_hi':>9}")
        scores = sub["lasso_score"].values
        for lo, hi, label in SLABS:
            r = slab_metrics(scores, y, lo, hi)
            if r is None or r["n"] < 1: continue
            lift = r["rate"]/base if base>0 else np.nan
            print(f"  {label:>10} {r['n']:>4d} {r['tp']:>4d} {r['rate']*100:>6.1f}% {lift:>4.1f}x {r['score_lo']:>+9.3f} {r['score_hi']:>+9.3f}")
            rows.append({"snap":snap,"yip":int(sub.snap_offset.iloc[0]),"slab":label,
                         "n":r["n"],"tp":r["tp"],"rate":r["rate"],"lift":lift,
                         "score_lo":r["score_lo"],"score_hi":r["score_hi"]})
        # Top picks: list them
        top_k = max(1, int(len(sub) * 0.02))
        order = np.argsort(-scores)[:top_k]
        top = sub.iloc[order][["player_id","name","snap_offset","lasso_score","realized_MLB_DEBUT","p_MLB_DEBUT"]].copy()
        out_lb = os.path.join(out_dir, f"2021lookback_snap{snap}_top2pct.csv")
        top.to_csv(out_lb, index=False)
        print(f"  saved top-2% picks → {out_lb}")

    if rows:
        pd.DataFrame(rows).to_csv(
            os.path.join(out_dir, "2021lookback_slabs.csv"), index=False)


if __name__ == "__main__":
    main()
