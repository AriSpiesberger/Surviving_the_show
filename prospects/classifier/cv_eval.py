"""Statistical evaluation suite for the OOF predictions from cv_runner.

Implements Phases 2-5 of CV_VALIDATION_PLAN.md:

  Phase 1.2 — integrity gate (asserts on the OOF CSV)
  Phase 2   — aggregate per-event metrics with bootstrap CIs
  Phase 3   — per-bucket cells with bootstrap CIs + MDE flags
  Phase 4   — cross-fold stability (variance of per-fold metrics)
  Phase 5   — decision-grade summary

Usage:
    python -m prospects.classifier.cv_eval \\
        --oof oof_predictions_v1.13.csv \\
        --out-prefix cv_v1.13
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict

import numpy as np
from scipy.stats import chi2, norm
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


BUCKETS = ("R1", "R2-R3", "R4-R10", "R10+", "IFA")
EVENTS = ("TOP_100_PROSPECT", "MLB_DEBUT", "ESTABLISHED_MLB",
          "ALL_STAR_ONCE", "STAR")
TOP_PCTS = (0.01, 0.05, 0.10, 0.20)
N_BOOTSTRAP = 500
SEED = 42


def _f(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Phase 1.2 — integrity gate
# ---------------------------------------------------------------------------

def integrity_gate(rows: list[dict]) -> list[str]:
    issues: list[str] = []
    n = len(rows)
    if n == 0:
        issues.append("EMPTY: no OOF rows")
        return issues
    pids = [r["player_id"] for r in rows]
    if len(set(pids)) != n:
        issues.append(f"player_id not unique: {n - len(set(pids))} duplicates")
    folds = [_f(r.get("fold")) for r in rows]
    if any(f is None for f in folds):
        issues.append("missing fold assignment")
    fold_sizes = np.bincount([int(f) for f in folds if f is not None])
    if len(fold_sizes) >= 2:
        cv = float(fold_sizes.std() / fold_sizes.mean())
        if cv > 0.10:
            issues.append(f"fold sizes uneven: cv={cv:.3f}, "
                          f"sizes={list(fold_sizes)}")
    for ev in EVENTS:
        col = f"p_{ev}"
        if col not in rows[0]:
            issues.append(f"missing column {col}")
            continue
        ps = [_f(r[col]) for r in rows]
        if any(p is None for p in ps):
            issues.append(f"{col}: {sum(p is None for p in ps)} NaN entries")
        else:
            mn = min(ps); mx = max(ps)
            if mn < -1e-6 or mx > 1 + 1e-6:
                issues.append(f"{col}: out of [0,1]: min={mn} max={mx}")
    return issues


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filter_eligible(rows: list[dict], event: str) -> list[dict]:
    p_col = f"p_{event}"
    real_col = f"realized_{event}"
    elig_col = f"eligible_at_snap_{event}"
    out = []
    for r in rows:
        if r.get(p_col) in (None, ""):
            continue
        if r.get(real_col) in (None, ""):
            continue
        if elig_col in r and r[elig_col] not in (None, ""):
            try:
                if int(r[elig_col]) != 1:
                    continue
            except (TypeError, ValueError):
                pass
        out.append(r)
    return out


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n * n))
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _bootstrap_ci(values: np.ndarray, stat_fn,
                  n_boot: int = N_BOOTSTRAP, seed: int = SEED,
                  alpha: float = 0.05) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    stats = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            stats[i] = stat_fn(values[idx])
        except Exception:
            stats[i] = float("nan")
    point = stat_fn(values)
    valid = stats[~np.isnan(stats)]
    if len(valid) < 10:
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(valid, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def _ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    if len(p) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    edges[-1] += 1e-9
    n = len(p)
    total = 0.0
    for i in range(n_bins):
        mask = (p >= edges[i]) & (p < edges[i + 1])
        if not mask.any():
            continue
        total += (mask.sum() / n) * abs(p[mask].mean() - y[mask].mean())
    return total


def _spiegelhalter_z(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spiegelhalter Z-test for calibration.
    Z = sum(y - p) / sqrt(sum(p (1 - p) (1 - 2p)^2))  (under H0 approx)
    Actually the standard formulation:
        num = sum((y - p) (1 - 2 p))
        den = sqrt(sum(p (1 - p) (1 - 2 p)^2))
    Returns (Z, two-sided p-value)."""
    if len(p) == 0:
        return float("nan"), float("nan")
    num = np.sum((y - p) * (1 - 2 * p))
    den = np.sqrt(np.sum(p * (1 - p) * (1 - 2 * p) ** 2))
    if den == 0:
        return float("nan"), float("nan")
    z = num / den
    pval = 2 * (1 - norm.cdf(abs(z)))
    return float(z), float(pval)


def _hosmer_lemeshow(p: np.ndarray, y: np.ndarray,
                     g: int = 10) -> tuple[float, float, int]:
    """Hosmer-Lemeshow Ĥ statistic using g groups by predicted-quantile."""
    n = len(p)
    if n < g * 5:
        return float("nan"), float("nan"), 0
    order = np.argsort(p)
    p_sorted = p[order]
    y_sorted = y[order]
    H = 0.0
    valid_groups = 0
    for i in range(g):
        lo = (i * n) // g
        hi = ((i + 1) * n) // g
        if hi <= lo:
            continue
        ps = p_sorted[lo:hi]
        ys = y_sorted[lo:hi]
        E1 = ps.sum()
        E0 = (1 - ps).sum()
        O1 = ys.sum()
        O0 = (1 - ys).sum()
        if E1 == 0 or E0 == 0:
            continue
        H += (O1 - E1) ** 2 / E1 + (O0 - E0) ** 2 / E0
        valid_groups += 1
    dof = max(1, valid_groups - 2)
    pval = 1 - chi2.cdf(H, dof)
    return float(H), float(pval), valid_groups


def _mde_auc(n_pos: int) -> float:
    """Minimum detectable AUC delta at 80% power, alpha=0.05 (Hanley-McNeil
    approximation)."""
    if n_pos < 4:
        return float("nan")
    return float(2.802 / math.sqrt(n_pos))  # ~Z(0.80)+Z(0.975) over sqrt(n_pos)


# ---------------------------------------------------------------------------
# Phase 2 — aggregate metrics
# ---------------------------------------------------------------------------

def aggregate_metrics(rows: list[dict], event: str) -> dict:
    items = _filter_eligible(rows, event)
    if not items:
        return {}
    p = np.array([_f(r[f"p_{event}"]) or 0 for r in items])
    y = np.array([int(_f(r[f"realized_{event}"]) or 0) for r in items])
    n = len(p); pos = int(y.sum())
    base = float(y.mean()) if n > 0 else float("nan")
    if pos == 0 or pos == n:
        return {"n": n, "pos": pos, "base": base, "mean_p": float(p.mean()),
                "auc": (float("nan"),)*3, "brier_skill": (float("nan"),)*3,
                "ll_skill": (float("nan"),)*3,
                "ece": float("nan"), "spiegelhalter": (float("nan"),)*2,
                "hosmer_lemeshow": (float("nan"),)*3}

    py = np.column_stack([p, y])
    auc = _bootstrap_ci(py, lambda a: roc_auc_score(a[:, 1], a[:, 0]))

    base_brier = base * (1 - base)
    def _bsk(arr):
        return 1 - brier_score_loss(arr[:, 1], arr[:, 0]) / base_brier
    bsk = _bootstrap_ci(py, _bsk)

    def _llsk(arr):
        clipped = np.clip(arr[:, 0], 1e-7, 1 - 1e-7)
        b = arr[:, 1].mean()
        if b in (0, 1):
            return float("nan")
        base_pred = np.clip(np.full_like(clipped, b), 1e-7, 1 - 1e-7)
        return 1 - log_loss(arr[:, 1], clipped) / log_loss(arr[:, 1], base_pred)
    llsk = _bootstrap_ci(py, _llsk)

    ece = _ece(p, y)
    sp_z, sp_p = _spiegelhalter_z(p, y)
    hl_h, hl_p, hl_g = _hosmer_lemeshow(p, y)

    return {
        "n": n, "pos": pos, "base": base, "mean_p": float(p.mean()),
        "auc": auc,
        "brier_skill": bsk,
        "ll_skill": llsk,
        "ece": ece,
        "spiegelhalter": (sp_z, sp_p),
        "hosmer_lemeshow": (hl_h, hl_p, hl_g),
    }


# ---------------------------------------------------------------------------
# Phase 3 — per-bucket metrics + top-N% with CIs + MDE
# ---------------------------------------------------------------------------

def per_bucket_metrics(rows: list[dict], event: str) -> dict:
    out: dict = {}
    for b in BUCKETS:
        b_rows = [r for r in rows if r.get("bucket") == b]
        out[b] = aggregate_metrics(b_rows, event)
        if out[b]:
            out[b]["mde_auc"] = _mde_auc(out[b]["pos"])
            out[b]["topn"] = topn_metrics(b_rows, event)
    return out


def topn_metrics(rows: list[dict], event: str) -> dict:
    items = _filter_eligible(rows, event)
    out: dict = {}
    for pct in TOP_PCTS:
        if not items:
            out[pct] = None
            continue
        p = np.array([_f(r[f"p_{event}"]) or 0 for r in items])
        y = np.array([int(_f(r[f"realized_{event}"]) or 0) for r in items])
        n = len(p); pos = int(y.sum())
        if pos == 0:
            out[pct] = None
            continue
        k = max(1, int(math.ceil(n * pct)))
        order = np.argsort(p)[::-1]
        top = order[:k]
        tp = int(y[top].sum())
        precision = tp / k
        recall = tp / pos
        base = pos / n
        lift = precision / base if base > 0 else float("nan")
        prec_lo, prec_hi = _wilson(tp, k)
        rec_lo, rec_hi = _wilson(tp, pos)

        # Bootstrap CI on lift
        def _stat(arr, _k=k):
            o = np.argsort(arr[:, 0])[::-1][:_k]
            tp_b = arr[o, 1].sum()
            prec_b = tp_b / _k
            base_b = arr[:, 1].mean()
            return prec_b / base_b if base_b > 0 else float("nan")
        py = np.column_stack([p, y])
        _, lift_lo, lift_hi = _bootstrap_ci(py, _stat)

        out[pct] = {
            "k": k, "tp": tp, "precision": precision,
            "precision_ci": (prec_lo, prec_hi),
            "recall": recall, "recall_ci": (rec_lo, rec_hi),
            "lift": lift, "lift_ci": (lift_lo, lift_hi),
        }
    return out


# ---------------------------------------------------------------------------
# Phase 4 — cross-fold stability
# ---------------------------------------------------------------------------

def cross_fold_stability(rows: list[dict], event: str) -> dict:
    """For each (bucket, fold), compute AUC and Brier-skill. Return per-cell
    mean and std across folds."""
    by_bucket_fold: dict = defaultdict(list)
    for r in rows:
        f = _f(r.get("fold"))
        b = r.get("bucket")
        if f is None or b not in BUCKETS:
            continue
        if r.get(f"p_{event}") in (None, ""):
            continue
        elig = r.get(f"eligible_at_snap_{event}")
        if elig not in (None, ""):
            try:
                if int(elig) != 1:
                    continue
            except (TypeError, ValueError):
                pass
        p = _f(r[f"p_{event}"]) or 0
        y = int(_f(r[f"realized_{event}"]) or 0)
        by_bucket_fold[(b, int(f))].append((p, y))

    out: dict = {}
    for b in BUCKETS:
        aucs = []; bsks = []
        for f in range(5):
            data = by_bucket_fold.get((b, f), [])
            if len(data) < 5:
                continue
            ps = np.array([d[0] for d in data])
            ys = np.array([d[1] for d in data])
            if ys.sum() == 0 or ys.sum() == len(ys):
                continue
            try:
                auc = roc_auc_score(ys, ps)
            except Exception:
                continue
            base = ys.mean()
            base_brier = base * (1 - base)
            bsk = 1 - brier_score_loss(ys, ps) / base_brier if base_brier > 0 else float("nan")
            aucs.append(auc); bsks.append(bsk)
        if not aucs:
            out[b] = None
        else:
            out[b] = {
                "n_folds": len(aucs),
                "auc_mean": float(np.mean(aucs)),
                "auc_std": float(np.std(aucs)),
                "auc_cv": float(np.std(aucs) / np.mean(aucs))
                          if np.mean(aucs) > 0 else float("nan"),
                "brsk_mean": float(np.mean(bsks)),
                "brsk_std": float(np.std(bsks)),
            }
    return out


# ---------------------------------------------------------------------------
# Phase 5 — decision-grade
# ---------------------------------------------------------------------------

def cell_verdict(agg: dict, topn: dict) -> str:
    """One of: GREEN, YELLOW, RED, UNDERPOWERED."""
    if not agg or agg.get("pos", 0) < 4:
        return "UNDERPOWERED"
    auc_ci = agg.get("auc", (float("nan"),)*3)
    bsk_ci = agg.get("brier_skill", (float("nan"),)*3)
    auc_pass = (auc_ci[1] == auc_ci[1] and auc_ci[1] > 0.5)
    bsk_pass = (bsk_ci[1] == bsk_ci[1] and bsk_ci[1] > 0)
    lift_pass = False
    if topn:
        top5 = topn.get(0.05)
        if top5 and top5.get("lift_ci") and top5["lift_ci"][0] == top5["lift_ci"][0]:
            lift_pass = top5["lift_ci"][0] > 1.0
    passes = sum([auc_pass, bsk_pass, lift_pass])
    if passes == 3: return "GREEN"
    if passes >= 1: return "YELLOW"
    return "RED"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_ci(triple, prec=3):
    if not triple or len(triple) < 3:
        return "n/a"
    point, lo, hi = triple
    if point != point:
        return "n/a"
    return f"{point:.{prec}f} [{lo:.{prec}f}, {hi:.{prec}f}]"


def write_report(out_prefix: str, integrity: list[str],
                 aggregate: dict, per_bucket: dict,
                 stability: dict) -> None:
    txt = f"{out_prefix}_assessment.txt"
    lines = []
    lines.append("=" * 100)
    lines.append("  CV ASSESSMENT (5-fold OOF, statistical tests)")
    lines.append("=" * 100)

    lines.append("")
    lines.append("[Phase 1.2] Integrity gate:")
    if integrity:
        for issue in integrity:
            lines.append(f"  FAIL: {issue}")
    else:
        lines.append("  PASS (all checks)")

    lines.append("")
    lines.append("=" * 100)
    lines.append("  [Phase 2] Aggregate metrics per event (with 95% CIs)")
    lines.append("=" * 100)
    lines.append(f"{'Event':<22} {'n':>5} {'pos':>5} {'base':>6} "
                 f"{'AUC':>22} {'Brier-skill':>22} {'ECE':>6} "
                 f"{'Spieg p':>8} {'HL p':>6}")
    for ev in EVENTS:
        a = aggregate.get(ev) or {}
        if not a:
            continue
        sp_z, sp_p = a.get("spiegelhalter", (float("nan"),)*2)
        hl_h, hl_p, _ = a.get("hosmer_lemeshow", (float("nan"),)*3)
        lines.append(
            f"{ev:<22} {a['n']:>5,d} {a['pos']:>5d} {a['base']:>6.3f} "
            f"{_fmt_ci(a['auc']):>22} {_fmt_ci(a['brier_skill']):>22} "
            f"{a['ece']:>6.3f} {sp_p:>8.4f} {hl_p:>6.4f}"
        )

    lines.append("")
    lines.append("=" * 100)
    lines.append("  [Phase 3] Per-bucket × event (AUC and Brier-skill with CIs)")
    lines.append("=" * 100)
    for ev in EVENTS:
        lines.append("")
        lines.append(f"--- {ev} ---")
        lines.append(f"{'Bucket':<8} {'n':>5} {'pos':>4} {'base':>6} "
                     f"{'AUC':>20} {'Brier-skill':>20} "
                     f"{'MDE_AUC':>8} {'Verdict':>14}")
        for b in BUCKETS:
            a = per_bucket.get(ev, {}).get(b) or {}
            if not a:
                lines.append(f"{b:<8} (no eligible rows)")
                continue
            verdict = cell_verdict(a, a.get("topn", {}))
            lines.append(
                f"{b:<8} {a['n']:>5d} {a['pos']:>4d} {a['base']:>6.3f} "
                f"{_fmt_ci(a['auc']):>20} {_fmt_ci(a['brier_skill']):>20} "
                f"{a['mde_auc']:>8.3f} {verdict:>14}"
            )

    lines.append("")
    lines.append("=" * 100)
    lines.append("  [Phase 3] Top-N% precision / lift per (event, bucket)")
    lines.append("  (lift_lo > 1 means model significantly beats random within bucket)")
    lines.append("=" * 100)
    for ev in EVENTS:
        lines.append("")
        lines.append(f"--- {ev} ---")
        header = f"{'Bucket':<8} " + " | ".join(
            f"{'top'+str(int(p*100))+'%':<32}" for p in TOP_PCTS
        )
        lines.append(header)
        lines.append("-" * len(header))
        for b in BUCKETS:
            cell = per_bucket.get(ev, {}).get(b) or {}
            topn = cell.get("topn", {})
            row = f"{b:<8} "
            chunks = []
            for pct in TOP_PCTS:
                t = topn.get(pct) if topn else None
                if not t:
                    chunks.append(f"{'(none)':<32}")
                else:
                    lift_ci = t["lift_ci"]
                    lift_s = (f"{t['lift']:.1f} [{lift_ci[0]:.1f},{lift_ci[1]:.1f}]"
                              if lift_ci[0] == lift_ci[0] else f"{t['lift']:.1f}")
                    chunks.append(
                        f"k={t['k']} tp={t['tp']} "
                        f"prec={t['precision']:.2f} lift={lift_s}"[:32].ljust(32)
                    )
            row += " | ".join(chunks)
            lines.append(row)

    lines.append("")
    lines.append("=" * 100)
    lines.append("  [Phase 4] Cross-fold stability (per bucket × event)")
    lines.append("  Flag if AUC CV (std/mean) across folds > 0.30")
    lines.append("=" * 100)
    for ev in EVENTS:
        lines.append("")
        lines.append(f"--- {ev} ---")
        lines.append(f"{'Bucket':<8} {'folds':>5} {'AUC mean±std':>20} "
                     f"{'AUC cv':>8} {'Brsk mean±std':>20} {'flag':>8}")
        for b in BUCKETS:
            s = stability.get(ev, {}).get(b)
            if not s:
                lines.append(f"{b:<8} (insufficient data)")
                continue
            flag = "UNSTABLE" if (s["auc_cv"] == s["auc_cv"] and s["auc_cv"] > 0.30) else ""
            lines.append(
                f"{b:<8} {s['n_folds']:>5d} "
                f"{s['auc_mean']:>8.3f} +/- {s['auc_std']:>5.3f}     "
                f"{s['auc_cv']:>8.3f} "
                f"{s['brsk_mean']:>8.3f} +/- {s['brsk_std']:>5.3f}     "
                f"{flag:>8}"
            )

    lines.append("")
    lines.append("=" * 100)
    lines.append("  [Phase 5] Decision-grade summary")
    lines.append("=" * 100)
    lines.append(f"{'Event':<22} " + "".join(f"{b:>10}" for b in BUCKETS))
    for ev in EVENTS:
        row = f"{ev:<22} "
        for b in BUCKETS:
            a = per_bucket.get(ev, {}).get(b)
            v = cell_verdict(a or {}, (a or {}).get("topn", {}))
            sym = {"GREEN": "[GREEN] ", "YELLOW": "[YELLOW]",
                   "RED": "[RED]   ", "UNDERPOWERED": "[UNDER] "}.get(v, "?")
            row += f"{sym:>10}"
        lines.append(row)
    lines.append("")
    lines.append("Legend:")
    lines.append("  GREEN  — AUC, Brier-skill, and top-5% lift all CI-significant")
    lines.append("  YELLOW — at least one CI-significant, others not")
    lines.append("  RED    — no metric significantly distinguishable from baseline")
    lines.append("  UNDER  — < 4 positives; insufficient power to evaluate")

    text = "\n".join(lines)
    with open(txt, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    print(f"\nWrote {txt}")


def write_csvs(out_prefix: str, aggregate: dict, per_bucket: dict,
               stability: dict) -> None:
    # Aggregate CSV
    rows = []
    for ev in EVENTS:
        a = aggregate.get(ev) or {}
        if not a:
            continue
        sp_z, sp_p = a.get("spiegelhalter", (float("nan"),)*2)
        hl_h, hl_p, hl_g = a.get("hosmer_lemeshow", (float("nan"),)*3)
        rows.append({
            "event": ev, "n": a["n"], "positives": a["pos"],
            "base_rate": a["base"], "mean_p": a["mean_p"],
            "auc": a["auc"][0], "auc_lo": a["auc"][1], "auc_hi": a["auc"][2],
            "brier_skill": a["brier_skill"][0],
            "brsk_lo": a["brier_skill"][1], "brsk_hi": a["brier_skill"][2],
            "ll_skill": a["ll_skill"][0],
            "llsk_lo": a["ll_skill"][1], "llsk_hi": a["ll_skill"][2],
            "ece": a["ece"],
            "spiegelhalter_z": sp_z, "spiegelhalter_p": sp_p,
            "hosmer_h": hl_h, "hosmer_p": hl_p, "hosmer_groups": hl_g,
        })
    if rows:
        with open(f"{out_prefix}_aggregate.csv", "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # Per-bucket CSV
    bucket_rows = []
    for ev in EVENTS:
        for b in BUCKETS:
            a = per_bucket.get(ev, {}).get(b) or {}
            if not a:
                continue
            topn = a.get("topn", {})
            for pct in TOP_PCTS:
                t = topn.get(pct) if topn else None
                row = {
                    "event": ev, "bucket": b, "top_pct": pct,
                    "n": a["n"], "positives": a["pos"], "base_rate": a["base"],
                    "auc": a["auc"][0], "auc_lo": a["auc"][1], "auc_hi": a["auc"][2],
                    "brier_skill": a["brier_skill"][0],
                    "brsk_lo": a["brier_skill"][1], "brsk_hi": a["brier_skill"][2],
                    "mde_auc": a.get("mde_auc"),
                    "verdict": cell_verdict(a, topn),
                }
                if t:
                    row.update({
                        "k": t["k"], "tp": t["tp"],
                        "precision": t["precision"],
                        "prec_lo": t["precision_ci"][0],
                        "prec_hi": t["precision_ci"][1],
                        "recall": t["recall"],
                        "rec_lo": t["recall_ci"][0],
                        "rec_hi": t["recall_ci"][1],
                        "lift": t["lift"],
                        "lift_lo": t["lift_ci"][0],
                        "lift_hi": t["lift_ci"][1],
                    })
                bucket_rows.append(row)
    if bucket_rows:
        # Collect superset of keys
        all_keys = list({k for r in bucket_rows for k in r.keys()})
        ordered = ["event", "bucket", "top_pct", "n", "positives", "base_rate",
                   "auc", "auc_lo", "auc_hi",
                   "brier_skill", "brsk_lo", "brsk_hi",
                   "mde_auc", "verdict",
                   "k", "tp", "precision", "prec_lo", "prec_hi",
                   "recall", "rec_lo", "rec_hi",
                   "lift", "lift_lo", "lift_hi"]
        fieldnames = [k for k in ordered if k in all_keys]
        with open(f"{out_prefix}_per_bucket.csv", "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            for r in bucket_rows:
                w.writerow({k: r.get(k) for k in fieldnames})

    # Stability CSV
    stab_rows = []
    for ev in EVENTS:
        for b in BUCKETS:
            s = stability.get(ev, {}).get(b)
            if not s:
                continue
            stab_rows.append({
                "event": ev, "bucket": b,
                "n_folds_with_data": s["n_folds"],
                "auc_mean": s["auc_mean"], "auc_std": s["auc_std"],
                "auc_cv": s["auc_cv"],
                "brsk_mean": s["brsk_mean"], "brsk_std": s["brsk_std"],
            })
    if stab_rows:
        with open(f"{out_prefix}_stability.csv", "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(stab_rows[0].keys()))
            w.writeheader()
            w.writerows(stab_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default="oof_predictions_v1.13.csv")
    ap.add_argument("--out-prefix", default="cv_v1.13")
    args = ap.parse_args()

    with open(args.oof, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print(f"Loaded {len(rows):,} OOF rows from {args.oof}")

    # Phase 1.2
    issues = integrity_gate(rows)
    if issues:
        print("\nIntegrity gate FAILED:")
        for i in issues:
            print(f"  - {i}")
    else:
        print("\nIntegrity gate PASSED")

    # Phase 2, 3
    print("\nComputing aggregate metrics (bootstrap CIs, this takes a minute)...")
    aggregate: dict = {}
    per_bucket: dict = {}
    for ev in EVENTS:
        aggregate[ev] = aggregate_metrics(rows, ev)
        per_bucket[ev] = per_bucket_metrics(rows, ev)

    # Phase 4
    print("Computing cross-fold stability...")
    stability: dict = {}
    for ev in EVENTS:
        stability[ev] = cross_fold_stability(rows, ev)

    # Report
    write_report(args.out_prefix, issues, aggregate, per_bucket, stability)
    write_csvs(args.out_prefix, aggregate, per_bucket, stability)
    print(f"Wrote {args.out_prefix}_aggregate.csv, "
          f"{args.out_prefix}_per_bucket.csv, "
          f"{args.out_prefix}_stability.csv")


if __name__ == "__main__":
    main()
