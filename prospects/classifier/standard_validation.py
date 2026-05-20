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

    out_dir = os.path.dirname(args.out_prefix) or "."
    os.makedirs(out_dir, exist_ok=True)
    long_path = f"{args.out_prefix}_long.csv"
    bucket_path = f"{args.out_prefix}_bucket.csv"
    wf_path = f"{args.out_prefix}_walkforward.csv"
    report_path = f"{args.out_prefix}_report.txt"

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

    print(f"\nWrote:")
    print(f"  {long_path}")
    print(f"  {bucket_path}")
    print(f"  {wf_path}")
    print(f"  {report_path}")


if __name__ == "__main__":
    main()
