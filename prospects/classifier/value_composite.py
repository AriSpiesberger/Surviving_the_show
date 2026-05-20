"""Expected-value composite for prospect ranking.

For each player:
    EV = sum_e  P(event_e) * incremental_multiplier(e) * discount^t_e

where:
  - incremental_multiplier(e) is the *additional* card-value gain at
    event e beyond the prior tier (avoids double-counting nested events).
  - t_e is the conditional expected time-to-event from the survival model.
  - discount per year reflects both time value of money and the model's
    prediction uncertainty at longer horizons.

The conditional-mean-time output from the survival simulator does NOT
enforce the structural ordering t_DEBUT <= t_EST <= t_STAR because the
hazard models are trained independently. Roughly 22% of the cohort
shows t_STAR < t_DEBUT, which means a rare-event contribution would get
*less* time-discounting than the prerequisite debut contribution -
incoherent. We clamp timings monotonically before discounting (and emit
both the raw and clamped value composites so the effect is auditable).

Usage:
    python -m prospects.classifier.value_composite \\
        --probs grades_probs_2026_v7.csv \\
        --timing grades_timing_2026_v7.csv \\
        --out value_v7.csv
"""
from __future__ import annotations

import argparse
import csv

import numpy as np


# Incremental dollar multipliers per milestone. STAR is incremental on
# top of ESTABLISHED, which is incremental on top of MLB_DEBUT. Placeholder
# values pending empirical calibration from sold-listing data.
INCR_MULT = {
    "MLB_DEBUT": 2.5,
    "ESTABLISHED_MLB": 2.0,
    "STAR": 4.0,
}

# Per-year discount factor. 0.85 is aggressive: implicitly downweights
# far-horizon predictions where the model has more uncertainty.
DISCOUNT = 0.85

# Fallback horizon when a per-event mean_t is missing (rare-event with
# near-zero cumulative probability => undefined conditional time). We use
# a conservative late-horizon value rather than 0 so missing-timing
# doesn't artificially boost the contribution.
T_FALLBACK = 8.0


def _f(x):
    try:
        v = float(x)
        if v != v:
            return None
        return v
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", default="grades_probs_2026_v7.csv")
    ap.add_argument("--timing", default="grades_timing_2026_v7.csv")
    ap.add_argument("--out", default="value_v7.csv")
    ap.add_argument("--discount", type=float, default=DISCOUNT)
    args = ap.parse_args()

    with open(args.probs, encoding="utf-8") as fh:
        probs = {r["player_id"]: r for r in csv.DictReader(fh)}
    with open(args.timing, encoding="utf-8") as fh:
        timings = {r["player_id"]: r for r in csv.DictReader(fh)}
    print(f"Loaded {len(probs):,} prob rows, {len(timings):,} timing rows")

    n_inv_debut_est = 0
    n_inv_est_star = 0
    n_inv_debut_star = 0

    out_rows = []
    for pid, pr in probs.items():
        ti = timings.get(pid, {})
        p_mlb = _f(pr.get("p_MLB_DEBUT")) or 0.0
        p_est = _f(pr.get("p_ESTABLISHED_MLB")) or 0.0
        p_star = _f(pr.get("p_STAR")) or 0.0

        t_mlb = _f(ti.get("t_MLB_DEBUT_mean"))
        t_est = _f(ti.get("t_ESTABLISHED_MLB_mean"))
        t_star = _f(ti.get("t_STAR_mean"))

        # Fallbacks for missing timings
        if t_mlb is None: t_mlb = T_FALLBACK
        if t_est is None: t_est = T_FALLBACK
        if t_star is None: t_star = T_FALLBACK

        # Track raw inversions
        if t_mlb > t_est: n_inv_debut_est += 1
        if t_est > t_star: n_inv_est_star += 1
        if t_mlb > t_star: n_inv_debut_star += 1

        # Clamp monotonically: t_STAR >= t_EST >= t_DEBUT
        t_mlb_c = t_mlb
        t_est_c = max(t_est, t_mlb_c)
        t_star_c = max(t_star, t_est_c)

        def _contrib(p, mult, t, disc):
            return p * mult * (disc ** t)

        # Raw (no clamp) - for audit
        ev_raw = (
            _contrib(p_mlb, INCR_MULT["MLB_DEBUT"], t_mlb, args.discount) +
            _contrib(p_est, INCR_MULT["ESTABLISHED_MLB"], t_est, args.discount) +
            _contrib(p_star, INCR_MULT["STAR"], t_star, args.discount)
        )
        # Clamped (structurally consistent)
        c_mlb = _contrib(p_mlb, INCR_MULT["MLB_DEBUT"], t_mlb_c, args.discount)
        c_est = _contrib(p_est, INCR_MULT["ESTABLISHED_MLB"], t_est_c, args.discount)
        c_star = _contrib(p_star, INCR_MULT["STAR"], t_star_c, args.discount)
        ev_clamped = c_mlb + c_est + c_star

        row = dict(pr)
        row["t_MLB"] = round(t_mlb, 3)
        row["t_EST"] = round(t_est, 3)
        row["t_STAR"] = round(t_star, 3)
        row["t_MLB_clamped"] = round(t_mlb_c, 3)
        row["t_EST_clamped"] = round(t_est_c, 3)
        row["t_STAR_clamped"] = round(t_star_c, 3)
        row["ev_contrib_MLB"] = round(c_mlb, 4)
        row["ev_contrib_EST"] = round(c_est, 4)
        row["ev_contrib_STAR"] = round(c_star, 4)
        row["ev_raw"] = round(ev_raw, 4)
        row["ev_clamped"] = round(ev_clamped, 4)
        out_rows.append(row)

    print(f"\nTiming inversions in cohort (n={len(out_rows):,}):")
    print(f"  t_DEBUT > t_EST   : {n_inv_debut_est:,}  "
          f"({100*n_inv_debut_est/len(out_rows):.1f}%)")
    print(f"  t_EST > t_STAR    : {n_inv_est_star:,}  "
          f"({100*n_inv_est_star/len(out_rows):.1f}%)")
    print(f"  t_DEBUT > t_STAR  : {n_inv_debut_star:,}  "
          f"({100*n_inv_debut_star/len(out_rows):.1f}%)")

    out_rows.sort(key=lambda r: -r["ev_clamped"])

    fieldnames = list(out_rows[0].keys())
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nWrote {len(out_rows):,} rows to {args.out}")

    print(f"\nTop 25 by ev_clamped (incremental multipliers, discount={args.discount}/yr):")
    print(f"{'Rk':>3} {'Name':<26} {'ev_clamp':>9} {'ev_raw':>8}  "
          f"{'P(MLB)':>7} {'t_MLB':>6}  {'P(EST)':>7} {'t_EST':>6}  "
          f"{'P(STAR)':>8} {'t_STAR':>7}  "
          f"{'c_MLB':>6} {'c_EST':>6} {'c_STAR':>7}")
    for i, r in enumerate(out_rows[:25], 1):
        print(f"{i:>3} {r['name'][:26]:<26} {r['ev_clamped']:>9.3f} "
              f"{r['ev_raw']:>8.3f}  "
              f"{_f(r['p_MLB_DEBUT']):>7.3f} {r['t_MLB_clamped']:>6.2f}  "
              f"{_f(r['p_ESTABLISHED_MLB']):>7.3f} {r['t_EST_clamped']:>6.2f}  "
              f"{_f(r['p_STAR']):>8.3f} {r['t_STAR_clamped']:>7.2f}  "
              f"{r['ev_contrib_MLB']:>6.3f} {r['ev_contrib_EST']:>6.3f} "
              f"{r['ev_contrib_STAR']:>7.3f}")


if __name__ == "__main__":
    main()
