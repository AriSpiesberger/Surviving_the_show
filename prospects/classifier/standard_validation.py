"""Standard validation suite — one script, one report, every model.

Pass any model file (--model path/to/event_classifiers_vX.pkl) and get
the same uniform validation report against the held-out validation set
(first 10% by seed=42, never seen by training OR calibration).

Two reports per event, all forward-looking, leakage-safe:

  1) BUCKET REPORT — one row per (draft_bucket, event) at snap = entry+2.
     Per cell: n_eligible, base_rate, AUC, Brier, Brier-skill,
               lift@{5,10,20}%, recall@{5,10,20}%, ECE, Spiegelhalter_p.

  2) WALK-FORWARD REPORT — one row per (snap_offset, event), 0..max_offset.
     Same metrics, but grouped by years-of-data since entry. Tells you
     how predictions sharpen with each additional MiLB season.

Events reported: TOP_100_PROSPECT, MLB_DEBUT, ESTABLISHED_MLB,
                 STAR_PLUS_ELITE (= STAR or ELITE, union).

Buckets: R1, R2-R3, R4-R10, R10+, IFA.

Eligibility: a player is included in event E's row at snap S only if
event E had not yet fired by snap S. AUC/Brier are computed on the
eligible subset only.

Usage:
    python -m prospects.classifier.standard_validation \\
        --model models/event_classifiers_v1.14.pkl \\
        --out-prefix val_v14
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

from prospects.classifier.architectures.survival import (
    ELITE_KEY, STAR_KEY, _trigger_year, load_hazards,
    predict_cumulative_batch,
)
from prospects.schema import CareerEvent
from prospects.storage import ProspectDB


REPORT_EVENTS = ("TOP_100_PROSPECT", "MLB_DEBUT", "ESTABLISHED_MLB",
                 "STAR_PLUS_ELITE")
BUCKET_ORDER = ("R1", "R2-R3", "R4-R10", "R10+", "IFA")
TOP_K_PCT = (1, 5, 10)
N_BOOT = 200

# Debut-lasso reporting
LASSO_FEATURES = (
    "p_TOP_100_PROSPECT", "p_MLB_DEBUT", "p_ESTABLISHED_MLB",
    "p_STAR_PLUS_ELITE",
    "age_at_snap_centered", "years_in_pro",
    "p_TOP_100_PROSPECT_x_yip_centered", "p_MLB_DEBUT_x_yip_centered",
    "p_ESTABLISHED_MLB_x_yip_centered", "p_STAR_PLUS_ELITE_x_yip_centered",
)
LASSO_DECILE_EDGES = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100)
LASSO_RATE_TARGETS = (0.05, 0.10, 0.25, 0.50, 0.75)
LASSO_MIN_BUCKET_N = 30  # below this, a (bucket, snap_offset) cell is reported but flagged


# ---------- helpers ----------

def _bucket_of(player: dict) -> str:
    if int(player.get("is_international") or 0) == 1:
        return "IFA"
    r = player.get("draft_round")
    if r is None:
        return "IFA"
    r = int(r)
    if r == 1:
        return "R1"
    if r <= 3:
        return "R2-R3"
    if r <= 10:
        return "R4-R10"
    return "R10+"


def _entry_year(player: dict, stats_by_pid: dict) -> int | None:
    dy = player.get("draft_year")
    is_intl = int(player.get("is_international") or 0)
    if dy is not None and not is_intl:
        return int(dy)
    yrs = [s.get("season_year")
           for s in stats_by_pid.get(player["player_id"], [])
           if s.get("season_year") is not None
           and (s.get("level") or "").upper() != "MLB"]
    if yrs:
        return int(min(yrs))
    if dy is not None:
        return int(dy)
    return None


def _ev_name(e) -> str:
    if hasattr(e, "name"):
        return e.name
    return str(e).lstrip("_")


def _heldout_validation_players(rows, seed: int, max_draft_year: int) -> set[str]:
    """First 10% of seed=42 perm over the (drafted<=max_draft_year + IFA)
    universe. Matches the held-out slice from training/calibration."""
    pool = [r for r in rows
            if (r.get("draft_year") is not None
                and r["draft_year"] <= max_draft_year)
            or int(r.get("is_international") or 0) == 1]
    unique = sorted({r["player_id"] for r in pool})
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(unique))
    n = int(round(0.10 * len(unique)))
    return {unique[i] for i in perm[:n]}


# ---------- metrics ----------

def _auc_with_ci(y: np.ndarray, p: np.ndarray,
                 n_boot: int = N_BOOT) -> tuple[float, float, float]:
    if y.size == 0 or y.sum() == 0 or y.sum() == y.size:
        return float("nan"), float("nan"), float("nan")
    auc = roc_auc_score(y, p)
    rng = np.random.default_rng(0)
    idx = np.arange(y.size)
    boots = []
    for _ in range(n_boot):
        s = rng.choice(idx, size=y.size, replace=True)
        ys, ps = y[s], p[s]
        if 0 < ys.sum() < ys.size:
            try:
                boots.append(roc_auc_score(ys, ps))
            except Exception:
                pass
    if not boots:
        return float(auc), float("nan"), float("nan")
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(auc), float(lo), float(hi)


def _brier_skill(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    if y.size == 0:
        return float("nan"), float("nan")
    base = y.mean()
    if base == 0 or base == 1:
        return float(brier_score_loss(y, p)), float("nan")
    br = brier_score_loss(y, p)
    br_base = brier_score_loss(y, np.full_like(p, base, dtype=float))
    return float(br), float(1.0 - br / br_base)


def _lift_recall_at_k(y: np.ndarray, p: np.ndarray,
                      k_pct: int) -> tuple[float, float, int]:
    """Top k_pct% by predicted probability. Returns (lift, recall, k)."""
    n = y.size
    if n == 0 or y.sum() == 0:
        return float("nan"), float("nan"), 0
    k = max(1, int(round(n * k_pct / 100)))
    order = np.argsort(-p)
    top = order[:k]
    tp = int(y[top].sum())
    precision = tp / k
    base = y.mean()
    recall = tp / int(y.sum())
    lift = precision / base if base > 0 else float("nan")
    return float(lift), float(recall), k


def _ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    if y.size == 0:
        return float("nan")
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        ece += n / y.size * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def _spiegelhalter_p(y: np.ndarray, p: np.ndarray) -> float:
    """Two-sided p-value for H0: model is calibrated."""
    if y.size == 0:
        return float("nan")
    # Z = sum((y - p)) / sqrt(sum(p(1-p)))
    var = float((p * (1 - p)).sum())
    if var <= 0:
        return float("nan")
    z = float((y - p).sum() / np.sqrt(var))
    # Normal two-sided p
    from math import erf, sqrt
    p_two = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return float(p_two)


def _cell_metrics(p: np.ndarray, y: np.ndarray) -> dict:
    n = int(y.size)
    pos = int(y.sum())
    base = float(y.mean()) if n else float("nan")
    auc, auc_lo, auc_hi = _auc_with_ci(y, p)
    br, bss = _brier_skill(y, p)
    out = {
        "n": n, "pos": pos, "base_rate": base, "pred_mean": float(p.mean()) if n else float("nan"),
        "auc": auc, "auc_lo": auc_lo, "auc_hi": auc_hi,
        "brier": br, "brier_skill": bss,
        "ece": _ece(y, p), "spiegelhalter_p": _spiegelhalter_p(y, p),
    }
    for kp in TOP_K_PCT:
        lift, rec, k = _lift_recall_at_k(y, p, kp)
        out[f"lift@{kp}%"] = lift
        out[f"recall@{kp}%"] = rec
        out[f"k@{kp}%"] = k
    return out


# ---------- scoring engine ----------

def _score_walkforward(
    cohort_rows: list[dict],
    stats_by_pid: dict,
    hazards,
    event_keys,
    observe_through: int,
    horizon: int,
    max_offset: int,
) -> list[dict]:
    """One row per (player, snap) for snap=entry+0..entry+max_offset, with
    p_<E>, eligible_at_snap_<E>, realized_after_snap_<E> for each event
    plus the synthetic STAR_PLUS_ELITE."""
    # Bucket players by snap so we batch predictions
    snap_groups: dict[int, list[dict]] = {}
    for r in cohort_rows:
        ent = _entry_year(r, stats_by_pid)
        if ent is None:
            continue
        debut = r.get("mlb_debut_year")
        for off in range(0, max_offset + 1):
            snap = ent + off
            if snap > observe_through:
                break
            if debut is not None and debut <= snap:
                continue
            rc = dict(r)
            rc["_entry_year"] = ent
            rc["_snap"] = snap
            rc["_offset"] = off
            rc["_bucket"] = _bucket_of(r)
            snap_groups.setdefault(snap, []).append(rc)

    out_rows: list[dict] = []
    for snap, group in sorted(snap_groups.items()):
        sub_stats = {
            r["player_id"]: [s for s in stats_by_pid.get(r["player_id"], [])
                             if (s.get("season_year") or 0) <= snap]
            for r in group
        }
        cumP = predict_cumulative_batch(
            hazards, group, sub_stats,
            current_year=snap, horizon=horizon,
        )
        for i, r in enumerate(group):
            row = {
                "player_id": r["player_id"],
                "name": r.get("name"),
                "draft_year": r.get("draft_year"),
                "draft_round": r.get("draft_round"),
                "is_international": int(r.get("is_international") or 0),
                "bucket": r["_bucket"],
                "entry_year": r["_entry_year"],
                "snap_year": snap,
                "snap_offset": r["_offset"],
                "years_fwd": observe_through - snap,
                "mlb_debut_year": r.get("mlb_debut_year"),
            }
            per_ev = {}
            for e in event_keys:
                ename = _ev_name(e)
                p_cal = float(cumP[e][i])
                trig = _trigger_year(r, e)
                eligible = int(trig is None or trig > snap)
                realized = int(trig is not None and trig > snap
                               and trig <= observe_through)
                per_ev[ename] = (p_cal, trig, eligible, realized)
                row[f"p_{ename}"] = p_cal
                row[f"eligible_{ename}"] = eligible
                row[f"realized_{ename}"] = realized
                row[f"trigger_{ename}"] = trig
            # STAR_PLUS_ELITE = union
            if "STAR" in per_ev and "ELITE" in per_ev:
                ps, ts, _, _ = per_ev["STAR"]
                pe, te, _, _ = per_ev["ELITE"]
                p_u = 1.0 - (1.0 - ps) * (1.0 - pe)
                trigs = [t for t in (ts, te) if t is not None]
                trig_u = min(trigs) if trigs else None
                elig_u = int(trig_u is None or trig_u > snap)
                real_u = int(trig_u is not None and trig_u > snap
                             and trig_u <= observe_through)
                row["p_STAR_PLUS_ELITE"] = p_u
                row["eligible_STAR_PLUS_ELITE"] = elig_u
                row["realized_STAR_PLUS_ELITE"] = real_u
                row["trigger_STAR_PLUS_ELITE"] = trig_u
            out_rows.append(row)
    return out_rows


# ---------- report assembly ----------

def _arrays_for_event(rows: list[dict], ename: str) -> tuple[np.ndarray, np.ndarray]:
    elig = [r for r in rows if r.get(f"eligible_{ename}") == 1]
    if not elig:
        return np.array([]), np.array([])
    p = np.array([r[f"p_{ename}"] for r in elig], dtype=float)
    y = np.array([r[f"realized_{ename}"] for r in elig], dtype=float)
    return p, y


def _build_bucket_report(rows: list[dict], snap_offset: int) -> list[dict]:
    """One row per (bucket, event) at the chosen snap_offset."""
    sub = [r for r in rows if r["snap_offset"] == snap_offset]
    out = []
    for ename in REPORT_EVENTS:
        # Aggregate row (all buckets combined)
        p, y = _arrays_for_event(sub, ename)
        m = _cell_metrics(p, y)
        out.append({"event": ename, "bucket": "ALL",
                    "snap_offset": snap_offset, **m})
        for b in BUCKET_ORDER:
            brows = [r for r in sub if r["bucket"] == b]
            p, y = _arrays_for_event(brows, ename)
            m = _cell_metrics(p, y)
            out.append({"event": ename, "bucket": b,
                        "snap_offset": snap_offset, **m})
    return out


def _build_walkforward_report(rows: list[dict],
                              max_offset: int) -> list[dict]:
    """One row per (event, snap_offset). All buckets combined."""
    out = []
    for ename in REPORT_EVENTS:
        for off in range(0, max_offset + 1):
            sub = [r for r in rows if r["snap_offset"] == off]
            p, y = _arrays_for_event(sub, ename)
            m = _cell_metrics(p, y)
            mean_fwd = (float(np.mean([r["years_fwd"] for r in sub]))
                        if sub else float("nan"))
            out.append({"event": ename, "snap_offset": off,
                        "mean_fwd_years": mean_fwd, **m})
    return out


def _fmt(v, p=3):
    if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
        return " nan"
    if isinstance(v, float):
        return f"{v:.{p}f}"
    return str(v)


def _write_text_report(bucket_rows, wf_rows, path,
                       cohort_n, observe_through, primary_offset):
    lines = []
    lines.append("=" * 88)
    lines.append("STANDARD VALIDATION REPORT")
    lines.append("=" * 88)
    lines.append(f"held-out validation players: {cohort_n:,}")
    lines.append(f"realization window per player: (snap, {observe_through}]")
    lines.append(f"primary snap_offset for bucket report: {primary_offset}")
    lines.append(f"events: {', '.join(REPORT_EVENTS)}")
    lines.append(f"buckets: {', '.join(BUCKET_ORDER)}")
    lines.append("")

    # Bucket report block
    lines.append("-" * 88)
    lines.append(f"BUCKET REPORT  (at snap_offset = {primary_offset})")
    lines.append("-" * 88)
    for ename in REPORT_EVENTS:
        lines.append(f"\n  Event: {ename}")
        header = (f"  {'bucket':<8} {'n':>5} {'pos':>4} {'base%':>6} "
                  f"{'pred%':>6} {'AUC':>6} {'[CI]':>14} "
                  f"{'BSS':>6}")
        for kp in TOP_K_PCT:
            header += f" {'lift@'+str(kp):>7} {'rec@'+str(kp):>6}"
        header += f" {'ECE':>5} {'spgl_p':>6}"
        lines.append(header)
        ev_rows = [r for r in bucket_rows if r["event"] == ename]
        for r in ev_rows:
            ci = f"[{_fmt(r['auc_lo'],2)},{_fmt(r['auc_hi'],2)}]"
            line = (
                f"  {r['bucket']:<8} {r['n']:>5d} {r['pos']:>4d} "
                f"{100*r['base_rate']:>5.2f}% "
                f"{100*r['pred_mean']:>5.2f}% "
                f"{_fmt(r['auc'],3):>6} {ci:>14} "
                f"{_fmt(r['brier_skill'],3):>6}"
            )
            for kp in TOP_K_PCT:
                line += (f" {_fmt(r[f'lift@{kp}%'],2):>7} "
                         f"{_fmt(r[f'recall@{kp}%'],2):>6}")
            line += f" {_fmt(r['ece'],3):>5} {_fmt(r['spiegelhalter_p'],3):>6}"
            lines.append(line)

    # Walk-forward block
    lines.append("")
    lines.append("-" * 88)
    lines.append("WALK-FORWARD REPORT  (one row per snap_offset)")
    lines.append("-" * 88)
    for ename in REPORT_EVENTS:
        lines.append(f"\n  Event: {ename}")
        header = (f"  {'offset':>6} {'mean_fwd':>8} {'n':>5} {'pos':>4} "
                  f"{'base%':>6} {'pred%':>6} {'AUC':>6} {'BSS':>6}")
        for kp in TOP_K_PCT:
            header += f" {'lift@'+str(kp):>7}"
        header += f" {'ECE':>5}"
        lines.append(header)
        ev_rows = [r for r in wf_rows if r["event"] == ename]
        for r in ev_rows:
            line = (
                f"  {r['snap_offset']:>6d} {_fmt(r['mean_fwd_years'],1):>8} "
                f"{r['n']:>5d} {r['pos']:>4d} "
                f"{100*r['base_rate']:>5.2f}% "
                f"{100*r['pred_mean']:>5.2f}% "
                f"{_fmt(r['auc'],3):>6} {_fmt(r['brier_skill'],3):>6}"
            )
            for kp in TOP_K_PCT:
                line += f" {_fmt(r[f'lift@{kp}%'],2):>7}"
            line += f" {_fmt(r['ece'],3):>5}"
            lines.append(line)

    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


# ---------- debut lasso ----------

def _load_lasso(path: str) -> dict:
    with open(path, "rb") as f:
        m = pickle.load(f)
    feat = list(m["feature_names"])
    if tuple(feat) != LASSO_FEATURES:
        # Not fatal — log so reviewer sees it, but still try.
        print(f"  WARN lasso feature_names differ from expected:"
              f"\n    got:      {feat}\n    expected: {list(LASSO_FEATURES)}")
    return m


def _score_lasso(long_rows: list[dict], lasso_model: dict,
                 birth_year_by_pid: dict) -> np.ndarray:
    """Compute debut_lasso score for every long row. Modifies rows in place
    to attach 'debut_score' and returns the score array aligned with rows."""
    sc = lasso_model["scaler"]
    ls = lasso_model["lasso"]
    feat = list(lasso_model["feature_names"])
    X = np.zeros((len(long_rows), len(feat)), dtype=float)
    for i, r in enumerate(long_rows):
        by = birth_year_by_pid.get(r["player_id"])
        snap = int(r["snap_year"])
        age_c = float(snap - by - 22) if by is not None else 0.0
        yip = float(r["snap_offset"])
        yip_c = yip - 3.0
        vals = {
            "p_TOP_100_PROSPECT": float(r.get("p_TOP_100_PROSPECT", 0.0) or 0.0),
            "p_MLB_DEBUT": float(r.get("p_MLB_DEBUT", 0.0) or 0.0),
            "p_ESTABLISHED_MLB": float(r.get("p_ESTABLISHED_MLB", 0.0) or 0.0),
            "p_STAR_PLUS_ELITE": float(r.get("p_STAR_PLUS_ELITE", 0.0) or 0.0),
            "age_at_snap_centered": age_c,
            "years_in_pro": yip,
        }
        vals["p_TOP_100_PROSPECT_x_yip_centered"] = vals["p_TOP_100_PROSPECT"] * yip_c
        vals["p_MLB_DEBUT_x_yip_centered"]       = vals["p_MLB_DEBUT"] * yip_c
        vals["p_ESTABLISHED_MLB_x_yip_centered"] = vals["p_ESTABLISHED_MLB"] * yip_c
        vals["p_STAR_PLUS_ELITE_x_yip_centered"] = vals["p_STAR_PLUS_ELITE"] * yip_c
        for j, fname in enumerate(feat):
            X[i, j] = vals[fname]
    scores = ls.predict(sc.transform(X))
    for r, s in zip(long_rows, scores):
        r["debut_score"] = float(s)
    return scores


def _build_lasso_curve_report(long_rows: list[dict],
                              snap_offset: int) -> list[dict]:
    """Per (bucket, event, decile-bin) curve at the given snap_offset.

    For each (bucket, event) eligible cohort, sort by debut_lasso score
    descending, slice by percentile edges, report n / realized / rate / lift.
    """
    sub = [r for r in long_rows if r["snap_offset"] == snap_offset
           and r.get("debut_score") is not None]
    if not sub:
        return []
    out = []
    for bucket_label, bucket_pred in (
        ("ALL", lambda r: True),
        ("R1",    lambda r: r["bucket"] == "R1"),
        ("R2-R3", lambda r: r["bucket"] == "R2-R3"),
        ("R4-R10",lambda r: r["bucket"] == "R4-R10"),
        ("R10+",  lambda r: r["bucket"] == "R10+"),
        ("IFA",   lambda r: r["bucket"] == "IFA"),
    ):
        brows = [r for r in sub if bucket_pred(r)]
        for ename in REPORT_EVENTS:
            elig = [r for r in brows if r.get(f"eligible_{ename}") == 1]
            if not elig:
                continue
            scores = np.array([r["debut_score"] for r in elig], dtype=float)
            y = np.array([r[f"realized_{ename}"] for r in elig], dtype=float)
            n_tot = len(elig)
            base = float(y.mean()) if n_tot else float("nan")
            order = np.argsort(-scores)  # high score first
            for lo_pct, hi_pct in zip(LASSO_DECILE_EDGES[:-1],
                                      LASSO_DECILE_EDGES[1:]):
                lo = int(round(n_tot * lo_pct / 100))
                hi = int(round(n_tot * hi_pct / 100))
                if hi <= lo:
                    continue
                idx = order[lo:hi]
                seg_scores = scores[idx]; seg_y = y[idx]
                n_seg = int(seg_y.size)
                pos_seg = int(seg_y.sum())
                rate = float(seg_y.mean())
                out.append({
                    "snap_offset": snap_offset,
                    "bucket": bucket_label,
                    "event": ename,
                    "pct_lo": lo_pct, "pct_hi": hi_pct,
                    "label": f"{lo_pct}-{hi_pct}%",
                    "score_lo": float(seg_scores.min()),
                    "score_hi": float(seg_scores.max()),
                    "score_mean": float(seg_scores.mean()),
                    "n": n_seg,
                    "pos": pos_seg,
                    "rate": rate,
                    "base_rate": base,
                    "lift": rate / base if base > 0 else float("nan"),
                    "small_sample": n_tot < LASSO_MIN_BUCKET_N,
                })
    return out


def _build_lasso_thresholds_report(long_rows: list[dict],
                                   snap_offset: int) -> list[dict]:
    """For each (bucket, event), find the highest score-threshold T such that
    among players with debut_score >= T, realized_<event> rate >= target.

    Sweep descending: walk down the score-sorted list keeping a cumulative
    realized rate; for each target rate, record the smallest k (largest T)
    where cumulative_rate >= target AND k >= LASSO_MIN_K. Lets the reviewer
    read off, e.g., 'score >= 0.71 -> 50% MLB_DEBUT rate (n=42)'."""
    LASSO_MIN_K = 10
    sub = [r for r in long_rows if r["snap_offset"] == snap_offset
           and r.get("debut_score") is not None]
    out = []
    for bucket_label, bucket_pred in (
        ("ALL", lambda r: True),
        ("R1",    lambda r: r["bucket"] == "R1"),
        ("R2-R3", lambda r: r["bucket"] == "R2-R3"),
        ("R4-R10",lambda r: r["bucket"] == "R4-R10"),
        ("R10+",  lambda r: r["bucket"] == "R10+"),
        ("IFA",   lambda r: r["bucket"] == "IFA"),
    ):
        brows = [r for r in sub if bucket_pred(r)]
        for ename in REPORT_EVENTS:
            elig = [r for r in brows if r.get(f"eligible_{ename}") == 1]
            if not elig:
                continue
            scores = np.array([r["debut_score"] for r in elig], dtype=float)
            y = np.array([r[f"realized_{ename}"] for r in elig], dtype=float)
            n_tot = len(elig)
            base = float(y.mean()) if n_tot else float("nan")
            order = np.argsort(-scores)
            cum_y = np.cumsum(y[order])
            ks = np.arange(1, n_tot + 1)
            cum_rate = cum_y / ks
            for target in LASSO_RATE_TARGETS:
                # Largest k (highest threshold inclusive) with cum_rate >= target
                ok = (cum_rate >= target) & (ks >= LASSO_MIN_K)
                if not ok.any():
                    out.append({
                        "snap_offset": snap_offset,
                        "bucket": bucket_label, "event": ename,
                        "target_rate": target,
                        "score_threshold": float("nan"),
                        "k": 0, "realized": 0, "rate": float("nan"),
                        "base_rate": base, "lift": float("nan"),
                        "pct_of_cohort": float("nan"),
                        "achievable": False,
                    })
                    continue
                # Pick the largest k that is still >= target. Walking from the
                # top, the rate is monotone-noisy; we want the maximal k where
                # rate >= target so the user gets the most inclusive threshold
                # that hits the target.
                k_pick = int(np.max(np.where(ok)[0]) + 1)
                thr = float(scores[order[k_pick - 1]])
                pos = int(cum_y[k_pick - 1])
                rate = float(cum_rate[k_pick - 1])
                out.append({
                    "snap_offset": snap_offset,
                    "bucket": bucket_label, "event": ename,
                    "target_rate": target,
                    "score_threshold": thr,
                    "k": k_pick, "realized": pos, "rate": rate,
                    "base_rate": base,
                    "lift": rate / base if base > 0 else float("nan"),
                    "pct_of_cohort": k_pick / n_tot if n_tot else float("nan"),
                    "achievable": True,
                })
    return out


def _write_lasso_text(curve_rows, thr_rows, path, snap_offset, lasso_model_path):
    lines = []
    lines.append("=" * 88)
    lines.append("DEBUT LASSO -> OUTCOME REPORT")
    lines.append("=" * 88)
    lines.append(f"lasso model: {lasso_model_path}")
    lines.append(f"snap_offset evaluated: {snap_offset}")
    lines.append(f"buckets: ALL, {', '.join(BUCKET_ORDER)}")
    lines.append(f"events: {', '.join(REPORT_EVENTS)}")
    lines.append("")
    lines.append("CURVE: per (bucket, event), realized rate by lasso-score percentile band")
    lines.append("       (bands are within-cohort percentiles; top-of-list first)")
    lines.append("")
    by_be: dict[tuple[str, str], list[dict]] = {}
    for r in curve_rows:
        by_be.setdefault((r["bucket"], r["event"]), []).append(r)
    for ename in REPORT_EVENTS:
        lines.append(f"--- event: {ename} ---")
        header = (f"  {'bucket':<8} {'band':>10} {'n':>5} {'pos':>4} "
                  f"{'rate':>6} {'lift':>6} {'score_lo':>9} {'score_hi':>9} "
                  f"{'score_mean':>10}")
        lines.append(header)
        for b in ("ALL",) + BUCKET_ORDER:
            rows = by_be.get((b, ename), [])
            if not rows:
                lines.append(f"  {b:<8}   (no eligible rows)")
                continue
            for r in rows:
                flag = "*" if r["small_sample"] else " "
                lines.append(
                    f" {flag}{r['bucket']:<8} {r['label']:>10} "
                    f"{r['n']:>5d} {r['pos']:>4d} "
                    f"{100*r['rate']:>5.1f}% {_fmt(r['lift'],2):>6} "
                    f"{r['score_lo']:>+9.3f} {r['score_hi']:>+9.3f} "
                    f"{r['score_mean']:>+10.3f}"
                )
        lines.append("")
    lines.append("THRESHOLDS: smallest score s.t. realized rate >= target  (k = players above)")
    lines.append("")
    by_be2: dict[tuple[str, str], list[dict]] = {}
    for r in thr_rows:
        by_be2.setdefault((r["bucket"], r["event"]), []).append(r)
    for ename in REPORT_EVENTS:
        lines.append(f"--- event: {ename} ---")
        header = (f"  {'bucket':<8} {'target':>7} {'score>=':>9} {'k':>5} "
                  f"{'pos':>4} {'rate':>6} {'lift':>6} {'%cohort':>8}")
        lines.append(header)
        for b in ("ALL",) + BUCKET_ORDER:
            rows = by_be2.get((b, ename), [])
            if not rows:
                continue
            for r in rows:
                if not r["achievable"]:
                    lines.append(
                        f"  {r['bucket']:<8} {100*r['target_rate']:>6.0f}% "
                        f"{'-':>9} {'-':>5} {'-':>4} {'-':>6} {'-':>6} {'-':>8}"
                    )
                    continue
                lines.append(
                    f"  {r['bucket']:<8} {100*r['target_rate']:>6.0f}% "
                    f"{r['score_threshold']:>+9.3f} {r['k']:>5d} "
                    f"{r['realized']:>4d} {100*r['rate']:>5.1f}% "
                    f"{_fmt(r['lift'],2):>6} {100*r['pct_of_cohort']:>7.2f}%"
                )
        lines.append("")
    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="prospects_snapshot.db")
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-draft-year", type=int, default=2020,
                    help="Reproduces the training-time split universe")
    ap.add_argument("--max-eval-entry-year", type=int, default=2015,
                    help="Restrict validation cohort to ensure mature "
                         "forward observation for slow events")
    ap.add_argument("--observe-through", type=int, default=2025)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--max-offset", type=int, default=10,
                    help="Walk-forward through this many years post-entry")
    ap.add_argument("--primary-offset", type=int, default=2,
                    help="snap_offset used for the bucket report")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--results-dir", default="results",
                    help="Parent folder. All artifacts go to "
                         "<results-dir>/<out-prefix>/. Pass '' for legacy "
                         "flat output in CWD.")
    ap.add_argument("--debut-lasso", default=None,
                    help="Optional path to debut_lasso_universe_v*.pkl. If "
                         "given, emit per-(bucket,event) lasso-score curves "
                         "and threshold tables.")
    args = ap.parse_args()

    print(f"Loading model: {args.model}")
    hazards = load_hazards(args.model)
    event_keys = [k for k in hazards
                  if k in (CareerEvent.TOP_100_PROSPECT,
                           CareerEvent.MLB_DEBUT,
                           CareerEvent.ESTABLISHED_MLB)
                  or k == STAR_KEY or k == ELITE_KEY]
    print(f"  events available: {[_ev_name(e) for e in event_keys]}")

    db = ProspectDB(args.db)
    with db._connect() as conn:
        rows = [dict(r) for r in conn.execute("""
            SELECT p.*, o.mlb_debut_year, o.year_established_mlb,
                   o.year_top_100, o.year_top_25,
                   o.year_all_star_once, o.year_all_star_three,
                   o.year_major_award, o.year_hof_trajectory, o.final_mlb_year
            FROM prospects p
            LEFT JOIN career_outcomes o ON o.player_id = p.player_id
        """).fetchall()]
        stats_rows = conn.execute("SELECT * FROM season_stats").fetchall()
    # birth_year map for the debut-lasso feature builder
    birth_year_by_pid: dict[str, int] = {}
    for r in rows:
        bd = r.get("birth_date")
        if not bd:
            continue
        try:
            birth_year_by_pid[r["player_id"]] = int(str(bd)[:4])
        except Exception:
            pass
    stats_by_pid: dict[str, list] = {}
    for s in stats_rows:
        d = dict(s)
        stats_by_pid.setdefault(d["player_id"], []).append(d)

    val_ids = _heldout_validation_players(rows, args.seed, args.max_draft_year)
    cohort = []
    for r in rows:
        if r["player_id"] not in val_ids:
            continue
        ent = _entry_year(r, stats_by_pid)
        if ent is None or ent > args.max_eval_entry_year:
            continue
        cohort.append(r)
    print(f"Validation cohort: {len(cohort):,} players "
          f"(held-out 10%, entry<={args.max_eval_entry_year})")

    long_rows = _score_walkforward(
        cohort_rows=cohort,
        stats_by_pid=stats_by_pid,
        hazards=hazards,
        event_keys=event_keys,
        observe_through=args.observe_through,
        horizon=args.horizon,
        max_offset=args.max_offset,
    )
    print(f"  scored {len(long_rows):,} (player, snap) rows")

    # Optional: debut-lasso scoring (attaches r['debut_score'] in place)
    lasso_model = None
    if args.debut_lasso:
        print(f"Loading debut lasso: {args.debut_lasso}")
        lasso_model = _load_lasso(args.debut_lasso)
        _score_lasso(long_rows, lasso_model, birth_year_by_pid)
        print(f"  attached debut_score to {len(long_rows):,} rows "
              f"(min={min(r['debut_score'] for r in long_rows):+.3f}, "
              f"max={max(r['debut_score'] for r in long_rows):+.3f})")

    # Route artifacts into results/<prefix>_<YYYY-MM-DD>/ unless --results-dir empty.
    import datetime as _dt
    prefix_base = os.path.basename(args.out_prefix.rstrip(os.sep).rstrip("/"))
    date = _dt.date.today().isoformat()
    if args.results_dir:
        out_dir = os.path.join(args.results_dir, f"{prefix_base}_{date}")
        artifact = lambda name: os.path.join(out_dir, name)
        long_path   = artifact("long.csv")
        bucket_path = artifact("bucket.csv")
        wf_path     = artifact("walkforward.csv")
        report_path = artifact("report.txt")
        lasso_curve_path = artifact("lasso_curve.csv")
        lasso_thr_path   = artifact("lasso_thresholds.csv")
        lasso_report_path = artifact("lasso_report.txt")
    else:
        out_dir = os.path.dirname(args.out_prefix) or "."
        long_path = f"{args.out_prefix}_long.csv"
        bucket_path = f"{args.out_prefix}_bucket.csv"
        wf_path = f"{args.out_prefix}_walkforward.csv"
        report_path = f"{args.out_prefix}_report.txt"
        lasso_curve_path = f"{args.out_prefix}_lasso_curve.csv"
        lasso_thr_path   = f"{args.out_prefix}_lasso_thresholds.csv"
        lasso_report_path = f"{args.out_prefix}_lasso_report.txt"
    os.makedirs(out_dir, exist_ok=True)

    fnames = list(long_rows[0].keys())
    with open(long_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fnames)
        w.writeheader()
        w.writerows(long_rows)
    print(f"  wrote {long_path}")

    # Free model + stats before heavy bootstrap work to ease memory.
    cohort_n = len(cohort)
    del hazards, stats_by_pid, cohort
    import gc; gc.collect()

    bucket_rows = _build_bucket_report(long_rows, args.primary_offset)
    wf_rows = _build_walkforward_report(long_rows, args.max_offset)
    with open(bucket_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(bucket_rows[0].keys()))
        w.writeheader()
        w.writerows(bucket_rows)
    with open(wf_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(wf_rows[0].keys()))
        w.writeheader()
        w.writerows(wf_rows)

    _write_text_report(
        bucket_rows, wf_rows, report_path,
        cohort_n=cohort_n,
        observe_through=args.observe_through,
        primary_offset=args.primary_offset,
    )

    # Lasso curve + thresholds (only if --debut-lasso was provided)
    if lasso_model is not None:
        lasso_curve = _build_lasso_curve_report(long_rows, args.primary_offset)
        lasso_thr   = _build_lasso_thresholds_report(long_rows, args.primary_offset)
        if lasso_curve:
            with open(lasso_curve_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(lasso_curve[0].keys()))
                w.writeheader(); w.writerows(lasso_curve)
        if lasso_thr:
            with open(lasso_thr_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(lasso_thr[0].keys()))
                w.writeheader(); w.writerows(lasso_thr)
        _write_lasso_text(lasso_curve, lasso_thr, lasso_report_path,
                          snap_offset=args.primary_offset,
                          lasso_model_path=args.debut_lasso)

    print(f"\nWrote:")
    print(f"  {long_path}")
    print(f"  {bucket_path}")
    print(f"  {wf_path}")
    print(f"  {report_path}")
    if lasso_model is not None:
        print(f"  {lasso_curve_path}")
        print(f"  {lasso_thr_path}")
        print(f"  {lasso_report_path}")


if __name__ == "__main__":
    main()
