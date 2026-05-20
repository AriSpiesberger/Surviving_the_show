"""Out-of-distribution prospect ranking.

The composite score is a weighted sum across events. It tells you the
expected-value-style ranking, but it doesn't directly answer the buying
question: "Is this player meaningfully unusual compared to the typical
prospect, and in which dimension?"

This script reframes the per-event probabilities as:

  - PER-EVENT PERCENTILE: where in the cohort does this player sit on
    P(MLB), P(Est), P(STAR)? A player at the 99.5th percentile of P(STAR)
    is materially OOD on star upside.

  - LIFT vs cohort median: P(player) / median(P_cohort). 10x lift means
    the model thinks they're 10x more likely than a typical prospect.

  - LOG-ODDS DELTA vs cohort base rate. Symmetric and additive: a
    log-odds delta of +3 on STAR means odds are e^3 ≈ 20x the base rate.

  - OOD SCORE: max percentile across events. The single number tells you
    "this player is at the X-th percentile of the cohort on at least one
    dimension." More directly buy-actionable than a weighted composite.

  - OOD TAG: which event(s) drove the OOD score, so the analyst knows
    whether the player is a debut-tier signal, an established-tier
    signal, or a star-tier signal. A player at 99.9th P(STAR) is a very
    different buy than one at 99.9th P(MLB) but median P(STAR).

Usage:
    python -m prospects.classifier.ood_ranking \\
        --grades grades_probs_2026_v7.csv \\
        --out ood_v7.csv
"""
from __future__ import annotations

import argparse
import csv

import numpy as np


EVENTS = ("p_MLB_DEBUT", "p_ESTABLISHED_MLB", "p_STAR")
EVENT_SHORT = {"p_MLB_DEBUT": "MLB",
               "p_ESTABLISHED_MLB": "EST",
               "p_STAR": "STAR"}

# Event reliability factors derived from v7 holdout validation (AUC and
# top-decile observed rate). MLB_DEBUT has AUC 0.84 and a clean top-decile;
# STAR has AUC 0.77 with P@10 of only 0.20. Trust scales the deviation
# from the cohort baseline that we credit to each event in the composite.
# A trust of 1.0 means take the prediction at face value; 0.5 means pull
# halfway back toward the baseline before scoring.
EVENT_TRUST = {
    "p_MLB_DEBUT": 0.90,
    "p_ESTABLISHED_MLB": 0.75,
    "p_STAR": 0.50,
}

# Composite weights match the grader: MLB=1, EST=3, STAR=10. These are
# market-multiplier-weighted contributions in the underlying EV calc.
EVENT_WEIGHTS = {
    "p_MLB_DEBUT": 1.0,
    "p_ESTABLISHED_MLB": 3.0,
    "p_STAR": 10.0,
}

# Player-side data-thickness shrinkage. The blender produces virtual full
# seasons but tags them with `_n_obs_pa` / `cur_pa`. Players with thin
# partial samples or short careers get more shrinkage toward the cohort
# baseline. K_PLAYER controls the half-trust point.
K_PLAYER_PA = 250.0
K_PLAYER_IP = 60.0


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _percentile_ranks(values: np.ndarray) -> np.ndarray:
    """Return percentile rank in [0, 1]: fraction of cohort below each value.
    Ties get average rank — so identical predictions get identical pct."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    n = len(values)
    # average rank within ties
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks / max(n - 1, 1)


def _log_odds_delta(p: np.ndarray, baseline: float) -> np.ndarray:
    """log odds(p) - log odds(baseline). NaN-safe at 0/1."""
    p_clipped = np.clip(p, 1e-9, 1 - 1e-9)
    b_clipped = float(np.clip(baseline, 1e-9, 1 - 1e-9))
    return np.log(p_clipped / (1 - p_clipped)) - np.log(b_clipped / (1 - b_clipped))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grades", default="grades_probs_2026_v7.csv")
    ap.add_argument("--out", default="ood_v7.csv")
    args = ap.parse_args()

    with open(args.grades, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    n = len(rows)
    print(f"Loaded {n:,} rows from {args.grades}")

    # Build arrays per event
    event_arrays: dict[str, np.ndarray] = {}
    for col in EVENTS:
        arr = np.array([_f(r.get(col)) or 0.0 for r in rows], dtype=np.float64)
        event_arrays[col] = arr

    # Per-event percentile and log-odds-delta vs cohort median
    pct: dict[str, np.ndarray] = {}
    lift: dict[str, np.ndarray] = {}
    lod: dict[str, np.ndarray] = {}
    baselines: dict[str, float] = {}
    for col, arr in event_arrays.items():
        pct[col] = _percentile_ranks(arr)
        med = float(np.median(arr)) or 1e-9
        baselines[col] = med
        lift[col] = arr / med
        lod[col] = _log_odds_delta(arr, med)

    print("\nCohort baselines (median P):")
    for col in EVENTS:
        print(f"  {EVENT_SHORT[col]:<5} median={baselines[col]:.4f} "
              f"mean={event_arrays[col].mean():.4f} "
              f"p99={np.percentile(event_arrays[col], 99):.4f} "
              f"max={event_arrays[col].max():.4f}")

    # OOD score: max percentile across the three events
    pct_matrix = np.stack([pct[c] for c in EVENTS], axis=1)
    ood_score = pct_matrix.max(axis=1)
    # Which event is driving it
    driver_idx = pct_matrix.argmax(axis=1)
    driver_short = [EVENT_SHORT[EVENTS[i]] for i in driver_idx]

    # Secondary signal: count of events where player is above 95th percentile
    above_95 = (pct_matrix >= 0.95).sum(axis=1)

    # ---- Confidence-adjusted composite ----
    # For each event:   credited_p = baseline + (P - baseline) * trust_event * trust_player
    # Then composite_conf = sum_e weight_e * credited_p_e
    # trust_player saturates with sample size; thin samples get pulled
    # back toward the cohort baseline.
    pa = np.array([_f(r.get("cur_pa")) or 0.0 for r in rows], dtype=np.float64)
    ip = np.array([_f(r.get("cur_ip")) or 0.0 for r in rows], dtype=np.float64)
    is_pitcher = np.array([int(_f(r.get("is_pitcher")) or 0) for r in rows])
    # Pitcher trust uses IP; hitter trust uses PA.
    trust_pa = pa / (pa + K_PLAYER_PA)
    trust_ip = ip / (ip + K_PLAYER_IP)
    trust_player = np.where(is_pitcher == 1, trust_ip, trust_pa)
    # Floor at 0.1 so a zero-PA prospect still gets some credit; cap at 1.0.
    trust_player = np.clip(trust_player, 0.1, 1.0)

    credited: dict[str, np.ndarray] = {}
    composite_conf = np.zeros(n, dtype=np.float64)
    for col in EVENTS:
        b = baselines[col]
        t_event = EVENT_TRUST[col]
        # Per-player credited probability
        credited[col] = b + (event_arrays[col] - b) * t_event * trust_player
        composite_conf += EVENT_WEIGHTS[col] * credited[col]

    # Augmented rows
    out_rows = []
    for i, r in enumerate(rows):
        out = dict(r)
        for col in EVENTS:
            short = EVENT_SHORT[col]
            out[f"pct_{short}"] = round(float(pct[col][i]), 5)
            out[f"lift_{short}"] = round(float(lift[col][i]), 3)
            out[f"lod_{short}"] = round(float(lod[col][i]), 3)
            out[f"credited_{short}"] = round(float(credited[col][i]), 5)
        out["trust_player"] = round(float(trust_player[i]), 3)
        out["composite_conf"] = round(float(composite_conf[i]), 4)
        out["ood_score"] = round(float(ood_score[i]), 5)
        out["ood_driver"] = driver_short[i]
        out["n_above_95"] = int(above_95[i])
        out_rows.append(out)

    # Sort by composite_conf desc (the confidence-adjusted ranking is the
    # primary buy signal; ood_score is a secondary lens).
    out_rows.sort(key=lambda r: (-r["composite_conf"], -r["ood_score"]))

    # Drop the legacy composite-style fields from the front; keep the file
    # superset of grades_probs so it can replace the composite output.
    fieldnames = list(out_rows[0].keys()) if out_rows else []
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nWrote {n:,} rows to {args.out}")

    # Top-25 by composite_conf
    print(f"\nTop 25 by composite_conf (confidence-adjusted EV ranking):")
    print(f"{'Rnk':>3} {'Name':<26} {'comp_c':>7} {'OOD':>6} {'drv':>4} "
          f"{'trust':>6} {'P(MLB)':>7} {'P(EST)':>7} {'P(STAR)':>8} "
          f"{'cMLB':>6} {'cEST':>6} {'cSTAR':>6}")
    for i, r in enumerate(out_rows[:25], 1):
        print(f"{i:>3} {r['name'][:26]:<26} {r['composite_conf']:>7.3f} "
              f"{r['ood_score']:>6.3f} {r['ood_driver']:>4} "
              f"{r['trust_player']:>6.2f} "
              f"{_f(r['p_MLB_DEBUT']):>7.3f} {_f(r['p_ESTABLISHED_MLB']):>7.3f} "
              f"{_f(r['p_STAR']):>8.3f} "
              f"{r['credited_MLB']:>6.3f} {r['credited_EST']:>6.3f} "
              f"{r['credited_STAR']:>6.3f}")

    # Also top-25 by raw OOD for comparison
    by_ood = sorted(out_rows, key=lambda r: (-r["ood_score"], -r["n_above_95"]))
    print(f"\nTop 25 by OOD score (most distributionally extreme on any event):")
    print(f"{'Rnk':>3} {'Name':<26} {'OOD':>6} {'drv':>4} "
          f"{'comp_c':>7} {'pct_MLB':>8} {'pct_EST':>8} {'pct_STAR':>9} "
          f"{'#>95':>5}")
    for i, r in enumerate(by_ood[:25], 1):
        print(f"{i:>3} {r['name'][:26]:<26} {r['ood_score']:>6.3f} "
              f"{r['ood_driver']:>4} {r['composite_conf']:>7.3f} "
              f"{r['pct_MLB']:>8.4f} {r['pct_EST']:>8.4f} {r['pct_STAR']:>9.4f} "
              f"{r['n_above_95']:>5d}")

    # Triple-extreme: players in the top 5% on ALL three events
    print(f"\nTriple-extreme prospects (top 5% on ALL three events):")
    triple = [r for r in out_rows if r["n_above_95"] == 3]
    print(f"  count: {len(triple)}")
    for r in triple[:20]:
        print(f"  {r['name'][:28]:<28} "
              f"pct(MLB,EST,STAR)=({r['pct_MLB']:.3f}, {r['pct_EST']:.3f}, "
              f"{r['pct_STAR']:.3f})")


if __name__ == "__main__":
    main()
