"""Buy score with regime-conditioned discounting and per-cell trust.

Goes beyond the composite_score by combining:

  1. Per-event probability (from the model)
  2. Per-event trust          (CV-derived: how reliable is this event's prediction?)
  3. Per-(bucket, event) trust (CV-derived AUC -> [0, 1] skill)
  4. Incremental card-value multiplier (placeholder, calibrated later)
  5. Age × level regime discount (different rate for young vs old, AAA vs not)
  6. Time-to-event from the survival model's E[t_event]

The discount is regime-conditioned because a uniform discount rate conflates
two things: (a) time-value of capital and (b) accumulated path uncertainty.
Young prospects have high path uncertainty (steep discount). Old AAA
prospects have mostly resolved development (low discount). Old non-AAA
prospects show adverse-selection (steep discount for a different reason).

Usage:
    python -m prospects.classifier.buy_score \\
        --probs grades_probs_2026_v13.csv \\
        --timing grades_timing_2026_v13.csv \\
        --bucket-cv cv_v1.13_per_bucket.csv \\
        --out buy_scores_v13.csv
"""
from __future__ import annotations

import argparse
import csv
import math
from datetime import date


EVENTS = ("TOP_100_PROSPECT", "MLB_DEBUT", "ESTABLISHED_MLB", "STAR")

# Per-event trust — from the CV evaluation. AUC + calibration quality.
# MLB_DEBUT is highest (cleanest AUC + best calibration). Rare events get
# discounted because of higher per-prediction noise even when AUC is solid.
EVENT_TRUST = {
    "TOP_100_PROSPECT": 0.85,
    "MLB_DEBUT": 1.00,
    "ESTABLISHED_MLB": 0.90,
    "STAR": 0.75,
}

# Incremental card-value multipliers. PLACEHOLDERS — calibrated empirically
# once the eBay price extraction has enough sold history per milestone.
# Reflect the *additional* value gain at each milestone beyond prior:
#   TOP_100 = early signal a player will be flagged by industry
#   MLB_DEBUT = first major card-price step up
#   ESTABLISHED = sustained mlb value, adds beyond debut bump
#   STAR = the long right tail
INCR_MULT = {
    "TOP_100_PROSPECT": 1.5,
    "MLB_DEBUT": 2.5,
    "ESTABLISHED_MLB": 2.0,
    "STAR": 4.0,
}

# Bucket label normalization
BUCKETS = ("R1", "R2-R3", "R4-R10", "R10+", "IFA")
LEVEL_RANK = {"DSL": 1, "FCL": 1, "CPX": 1, "RK": 1, "ROK": 1,
              "A-": 2, "A": 3, "A+": 4, "AA": 5, "AAA": 6, "MLB": 7}


def _f(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def _bucket(r):
    if int(_f(r.get("is_international")) or 0) == 1:
        return "IFA"
    rd = _f(r.get("draft_round"))
    if rd is None:
        return "UNK"
    rd = int(rd)
    if rd == 1: return "R1"
    if rd <= 3: return "R2-R3"
    if rd <= 10: return "R4-R10"
    return "R10+"


def _age_today(birth_date_iso: str, today: date) -> float | None:
    if not birth_date_iso:
        return None
    try:
        y, m, d = (int(x) for x in str(birth_date_iso)[:10].split("-"))
        bd = date(y, m, d)
        return (today - bd).days / 365.25
    except (TypeError, ValueError):
        return None


def _regime(age: float | None, cur_level: str) -> tuple[str, float]:
    """Return (regime_label, annual_discount_rate)."""
    lvl = (cur_level or "").upper()
    is_aaa = lvl == "AAA"
    is_upper = lvl in ("AA", "AAA")
    if age is None:
        return ("UNKNOWN_AGE", 0.10)
    if age < 21:
        # High path uncertainty regardless of level.
        return ("YOUNG", 0.12 if is_upper else 0.13)
    if age <= 23:
        if is_upper:
            return ("MID_UPPER", 0.05)
        return ("MID_LOWER", 0.10)
    # age >= 24
    if is_aaa:
        return ("OLD_AAA", 0.02)
    # Old + not yet AAA = adverse selection
    return ("OLD_STALLED", 0.22)


def _load_bucket_trust(cv_csv: str) -> dict[tuple[str, str], float]:
    """Derive per-(bucket, event) trust from the CV per-bucket CSV.

    Trust = max(0, 2 * (AUC - 0.5)).
    AUC of 0.5 -> trust 0 (no skill). AUC of 1.0 -> trust 1.
    """
    trust: dict[tuple[str, str], float] = {}
    try:
        with open(cv_csv, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                ev = r.get("event"); b = r.get("bucket")
                auc = _f(r.get("auc"))
                if ev is None or b is None or auc is None:
                    continue
                trust[(b, ev)] = max(0.0, 2.0 * (auc - 0.5))
    except FileNotFoundError:
        pass
    # Fallback defaults baked from the v1.13 CV report — used if the CSV
    # is missing a cell or the file isn't present.
    fallback = {
        ("R1", "TOP_100_PROSPECT"): 0.39,
        ("R2-R3", "TOP_100_PROSPECT"): 0.46,
        ("R4-R10", "TOP_100_PROSPECT"): 0.76,
        ("R10+", "TOP_100_PROSPECT"): 0.81,
        ("IFA", "TOP_100_PROSPECT"): 0.87,
        ("R1", "MLB_DEBUT"): 0.62,
        ("R2-R3", "MLB_DEBUT"): 0.52,
        ("R4-R10", "MLB_DEBUT"): 0.59,
        ("R10+", "MLB_DEBUT"): 0.74,
        ("IFA", "MLB_DEBUT"): 0.73,
        ("R1", "ESTABLISHED_MLB"): 0.46,
        ("R2-R3", "ESTABLISHED_MLB"): 0.44,
        ("R4-R10", "ESTABLISHED_MLB"): 0.59,
        ("R10+", "ESTABLISHED_MLB"): 0.66,
        ("IFA", "ESTABLISHED_MLB"): 0.63,
        ("R1", "STAR"): 0.33,
        ("R2-R3", "STAR"): 0.32,
        ("R4-R10", "STAR"): 0.62,
        ("R10+", "STAR"): 0.79,
        ("IFA", "STAR"): 0.81,
    }
    for k, v in fallback.items():
        trust.setdefault(k, v)
    return trust


def _write_legend(path: str, today: date,
                  bucket_trust: dict[tuple[str, str], float]) -> None:
    """Statistician-facing column glossary + formula spec."""
    lines = []
    W = lines.append
    W("BUY SCORE — COLUMN LEGEND & METHODOLOGY")
    W("=" * 70)
    W(f"Generated: {today.isoformat()}")
    W("Source model: v1.13 (92.5/7.5 split, Beta-calibrated hazards)")
    W("")
    W("EVENTS (E)")
    W("-" * 70)
    W("  TOP_100_PROSPECT  Player appears on any industry Top-100 list")
    W("  MLB_DEBUT         First MLB appearance")
    W("  ESTABLISHED_MLB   Sustained MLB role (per CareerEvent definition)")
    W("  STAR              Star-tier outcome (long right tail)")
    W("")
    W("FORMULA")
    W("-" * 70)
    W("  buy_score = sum over events E of:")
    W("      prob_E")
    W("    * trust_event_E        (per-event reliability, CV-derived)")
    W("    * trust_bucket_E       (per-(bucket,event) AUC-skill, CV-derived)")
    W("    * value_mult_E         (incremental card-value multiplier)")
    W("    * time_discount_E      (= (1 - annual_discount_rate)^years_to_E)")
    W("")
    W("  trust_combined_E = trust_event_E * trust_bucket_E")
    W("  value_contrib_E  = prob_E * trust_combined_E * value_mult_E")
    W("                     * time_discount_E")
    W("")
    W("COLUMN DICTIONARY")
    W("-" * 70)
    cols = [
        ("buy_rank", "Rank by buy_score (1 = highest)"),
        ("player_id", "Internal player ID (joins to prospects table)"),
        ("name", "Player name"),
        ("primary_position", "Primary position"),
        ("current_org", "Current MLB organization"),
        ("age", "Age in years as of as-of date"),
        ("current_level", "Current minor-league level (or MLB)"),
        ("bucket", "Acquisition bucket: R1 / R2-R3 / R4-R10 / R10+ / IFA"),
        ("is_international", "1 if signed as international free agent"),
        ("draft_year", "Year drafted (NULL for IFA)"),
        ("draft_round", "Draft round (NULL for IFA)"),
        ("draft_pick", "Overall pick number"),
        ("signing_bonus_usd", "Signing bonus in USD"),
        ("current_top100_rank", "Most-recent industry Top-100 rank (NULL if never)"),
        ("best_top100_rank", "Best (lowest) Top-100 rank ever achieved"),
        ("times_top100", "Number of years player has appeared on a Top-100"),
        ("birth_date", "Date of birth (YYYY-MM-DD)"),
        ("regime", "Discount regime: YOUNG / MID_UPPER / MID_LOWER / "
                   "OLD_AAA / OLD_STALLED / UNKNOWN_AGE"),
        ("annual_discount_rate", "Per-year discount applied to future events"),
        ("prob_E", "P(event E occurs | covariates), Beta-calibrated"),
        ("years_to_E", "E[time to event E | E occurs], from survival model"),
        ("trust_event_E", "Per-event reliability multiplier (constant per event)"),
        ("trust_bucket_E", "Per-(bucket,event) reliability: max(0, 2*(AUC-0.5))"),
        ("trust_combined_E", "trust_event_E * trust_bucket_E"),
        ("value_mult_E", "Incremental card-value multiplier at this milestone"),
        ("time_discount_E", "(1 - annual_discount_rate)^years_to_E"),
        ("value_contrib_E", "Contribution of event E to buy_score"),
        ("buy_score", "Sum of value_contrib over all events (THE rank metric)"),
        ("composite_score_v13", "Legacy composite score from v1.13 grader, "
                                "for comparison"),
    ]
    for name, desc in cols:
        W(f"  {name:<24} {desc}")
    W("")
    W("CONSTANTS")
    W("-" * 70)
    W("  trust_event values (per-event reliability):")
    for ev, v in EVENT_TRUST.items():
        W(f"    {ev:<20} {v:>5.2f}")
    W("")
    W("  value_mult values (incremental card-value, PLACEHOLDER until")
    W("  empirically calibrated from eBay sold history):")
    for ev, v in INCR_MULT.items():
        W(f"    {ev:<20} {v:>5.2f}")
    W("")
    W("  Discount regimes (annual_discount_rate by (age, current_level)):")
    W("    YOUNG        age < 21                       12-13%/yr")
    W("    MID_UPPER    21 <= age <= 23  &  AA/AAA      5%/yr")
    W("    MID_LOWER    21 <= age <= 23  &  below AA   10%/yr")
    W("    OLD_AAA      age >= 24        &  AAA         2%/yr")
    W("    OLD_STALLED  age >= 24        &  below AAA  22%/yr")
    W("    UNKNOWN_AGE  age unknown                    10%/yr")
    W("")
    W("PER-BUCKET TRUST (from CV: trust = max(0, 2 * (AUC - 0.5)))")
    W("-" * 70)
    W(f"  {'bucket':<8} " + " ".join(f"{e[:12]:>13}" for e in EVENTS))
    for b in BUCKETS:
        cells = []
        for e in EVENTS:
            v = bucket_trust.get((b, e))
            cells.append(f"{v:>13.3f}" if v is not None else f"{'.':>13}")
        W(f"  {b:<8} " + " ".join(cells))
    W("")
    W("CAVEATS")
    W("-" * 70)
    W("  - value_mult constants are PLACEHOLDERS; the buy_score's absolute")
    W("    magnitude is therefore not meaningful — its RANK is.")
    W("  - For already-realized events (e.g. player is already on Top-100),")
    W("    prob_E = 1.0 and time_discount_E uses years_to_E = 0.")
    W("  - bucket_trust uses fallback CV values from v1.13 per-bucket eval")
    W("    when --bucket-cv CSV is missing.")
    W("  - Rare events (STAR, TOP_100 in R1) have wider uncertainty even")
    W("    when calibrated; treat magnitude as conservative.")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", default="grades_probs_2026_v13.csv")
    ap.add_argument("--timing", default="grades_timing_2026_v13.csv")
    ap.add_argument("--bucket-cv", default="cv_v1.13_per_bucket.csv")
    ap.add_argument("--out", default="buy_scores_v13.csv")
    ap.add_argument("--as-of",
                    default=date.today().isoformat(),
                    help="ISO date for age calculation (default: today)")
    args = ap.parse_args()
    today = date.fromisoformat(args.as_of)

    bucket_trust = _load_bucket_trust(args.bucket_cv)
    print(f"Loaded bucket trust for {len(bucket_trust)} cells")

    with open(args.probs, encoding="utf-8") as fh:
        probs = {r["player_id"]: r for r in csv.DictReader(fh)}
    with open(args.timing, encoding="utf-8") as fh:
        timing = {r["player_id"]: r for r in csv.DictReader(fh)}
    print(f"Loaded {len(probs):,} probability rows, "
          f"{len(timing):,} timing rows")

    out_rows = []
    for pid, p in probs.items():
        t = timing.get(pid, {})
        bucket = _bucket(p)
        age = _age_today(p.get("birth_date"), today)
        cur_level = p.get("cur_level", "")
        regime, disc_rate = _regime(age, cur_level)
        keep = (1.0 - disc_rate)  # per-year keep factor

        per_event_detail = {}
        total = 0.0
        for ev in EVENTS:
            P = _f(p.get(f"p_{ev}")) or 0.0
            t_mean = _f(t.get(f"t_{ev}_mean"))
            if t_mean is None or t_mean != t_mean:
                t_eff = 8.0  # Fallback long horizon if timing missing
            else:
                t_eff = max(0.0, float(t_mean))
            ev_trust = EVENT_TRUST[ev]
            b_trust = bucket_trust.get((bucket, ev), 0.5)
            mult = INCR_MULT[ev]
            time_disc = keep ** t_eff
            contrib = P * ev_trust * b_trust * mult * time_disc
            per_event_detail[ev] = {
                "P": P, "t": t_eff, "ev_trust": ev_trust,
                "bucket_trust": b_trust, "trust": ev_trust * b_trust,
                "mult": mult, "time_disc": time_disc, "contrib": contrib,
            }
            total += contrib

        # ---- Identity ----
        row = {
            "player_id": pid,
            "name": p.get("name"),
            "primary_position": p.get("primary_position"),
            "current_org": p.get("current_org"),
            # ---- Context ----
            "age": (round(age, 2) if age is not None else None),
            "current_level": cur_level,
            "bucket": bucket,
            "is_international": p.get("is_international"),
            "draft_year": p.get("draft_year"),
            "draft_round": p.get("draft_round"),
            "draft_pick": p.get("draft_pick"),
            "signing_bonus_usd": p.get("signing_bonus_usd"),
            "current_top100_rank": p.get("recent_top100_rank"),
            "best_top100_rank": p.get("best_top100_rank"),
            "times_top100": p.get("times_top100"),
            "birth_date": p.get("birth_date"),
            # ---- Regime / discounting ----
            "regime": regime,
            "annual_discount_rate": disc_rate,
        }

        # ---- Per-event blocks (probability, timing, trust, discount, value) ----
        for ev in EVENTS:
            d = per_event_detail[ev]
            row[f"prob_{ev}"] = round(d["P"], 5)
            row[f"years_to_{ev}"] = round(d["t"], 2)
            row[f"trust_event_{ev}"] = round(d["ev_trust"], 3)
            row[f"trust_bucket_{ev}"] = round(d["bucket_trust"], 3)
            row[f"trust_combined_{ev}"] = round(d["trust"], 3)
            row[f"value_mult_{ev}"] = d["mult"]
            row[f"time_discount_{ev}"] = round(d["time_disc"], 4)
            row[f"value_contrib_{ev}"] = round(d["contrib"], 4)

        # ---- Final scores ----
        row["buy_score"] = round(total, 4)
        row["composite_score_v13"] = _f(p.get("composite_score"))
        out_rows.append(row)

    out_rows.sort(key=lambda r: -r["buy_score"])
    # Add buy_rank as first column
    for i, r in enumerate(out_rows, 1):
        r["buy_rank"] = i
    fieldnames = ["buy_rank"] + [k for k in out_rows[0].keys() if k != "buy_rank"]

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {len(out_rows):,} rows to {args.out}")

    # ---- Write a sidecar legend file for column documentation ----
    legend_path = args.out.replace(".csv", "_LEGEND.txt")
    if legend_path == args.out:
        legend_path = args.out + ".LEGEND.txt"
    _write_legend(legend_path, today, bucket_trust)
    print(f"Wrote column legend to {legend_path}")

    # Top 25 preview
    print("\nTop 25 by buy_score:")
    hdr = (f"{'Rk':>3} {'Name':<26} {'Lvl':<4} {'Age':>4} "
           f"{'T100':>4} {'Regime':<12} "
           f"{'P(T100)':>7} {'P(MLB)':>7} {'P(EST)':>7} {'P(STAR)':>7} "
           f"{'buy':>6}")
    print(hdr)
    for r in out_rows[:25]:
        rk = r.get("current_top100_rank")
        rk_s = f"{int(rk):>4d}" if rk not in (None, "") else f"{'-':>4}"
        print(
            f"{r['buy_rank']:>3} {(r['name'] or '')[:26]:<26} "
            f"{(r['current_level'] or '')[:4]:<4} "
            f"{(r['age'] if r['age'] is not None else 0):>4.1f} "
            f"{rk_s} {r['regime']:<12} "
            f"{r['prob_TOP_100_PROSPECT']:>7.3f} "
            f"{r['prob_MLB_DEBUT']:>7.3f} {r['prob_ESTABLISHED_MLB']:>7.3f} "
            f"{r['prob_STAR']:>7.3f} {r['buy_score']:>6.3f}"
        )

    # Regime distribution
    from collections import Counter
    reg_counts = Counter(r["regime"] for r in out_rows)
    print("\nRegime distribution:")
    for reg, c in reg_counts.most_common():
        print(f"  {reg:<14} {c:>5,}")


if __name__ == "__main__":
    main()
