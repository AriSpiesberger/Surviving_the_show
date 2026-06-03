"""Time-to-debut diagnostic for the v1.17 hazard survival model on the
honest val cohort. For each val debutee at each pre-debut snap, runs the
hazard chain forward (feature-aging yip/age/yics each step) to compute:

    cum_P(debut by year snap+h) for h in [1..H]
    E[time-to-debut | event]    = sum_h h * step_p[h] / sum step_p

then compares to the actual time-to-debut.
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

from prospects.classifier.architectures.survival import (
    EXIT_KEY, _PlattCalibrator, build_windowed_features, load_hazards,
)
from prospects.features.scouting import FEATURE_NAMES
from prospects.schema import CareerEvent
from prospects.storage import ProspectDB

sys.modules["__main__"]._PlattCalibrator = _PlattCalibrator

MODEL = "models/event_classifiers_v1.17.pkl"
DB = "prospects_snapshot.db"
HORIZON = 12
VAL_PLAYERS = "models/event_classifiers_v1.17_lasso_val_players.txt"


def _forward_chain(hazards, prospect, stats, snap_year, horizon=HORIZON):
    """Returns (cum_P_debut[h], step_p_debut[h]) for h in 1..horizon."""
    yip_i = FEATURE_NAMES.index("years_in_pro")
    age_i = FEATURE_NAMES.index("age_at_as_of")
    yics_i = FEATURE_NAMES.index("years_in_current_system")
    X0 = build_windowed_features(prospect, stats, snap_year,
                                  milb_only=True).reshape(1, -1)
    yip0, age0, yics0 = X0[0, yip_i], X0[0, age_i], X0[0, yics_i]
    has_exit = EXIT_KEY in hazards
    exit_clf = hazards[EXIT_KEY]["hazard"] if has_exit else None
    debut_clf = hazards[CareerEvent.MLB_DEBUT]["hazard"]

    surv = 1.0
    in_bb = 1.0
    step_p = []
    cum = []
    for step in range(horizon):
        X = X0.copy()
        if not np.isnan(yip0): X[0, yip_i] = yip0 + (step + 1)
        if not np.isnan(age0): X[0, age_i] = age0 + (step + 1)
        if not np.isnan(yics0): X[0, yics_i] = yics0 + (step + 1)
        h = float(debut_clf.predict_proba(X)[0, 1])
        sp = surv * in_bb * h
        step_p.append(sp)
        surv = surv * (1.0 - in_bb * h)
        if has_exit:
            he = float(exit_clf.predict_proba(X)[0, 1])
            in_bb = in_bb * (1.0 - he)
        cum.append(1.0 - surv)
    return np.array(cum), np.array(step_p)


def main():
    print(f"Loading {MODEL}")
    hazards = load_hazards(MODEL)
    with open(VAL_PLAYERS) as fh:
        val_pids = {ln.strip() for ln in fh if ln.strip()}
    print(f"  {len(val_pids):,} val players")

    db = ProspectDB(DB)
    with db._connect() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT p.*, o.mlb_debut_year, o.year_established_mlb, "
            "       o.year_top_100, o.year_top_25, "
            "       o.year_all_star_once, o.year_all_star_three, "
            "       o.year_major_award, o.year_hof_trajectory, "
            "       o.final_mlb_year "
            "FROM prospects p LEFT JOIN career_outcomes o "
            "  ON o.player_id = p.player_id "
            "WHERE p.player_id IN ({})"
            .format(",".join("?" * len(val_pids))),
            list(val_pids)).fetchall()]
        stats_by_pid = {}
        for s in conn.execute(
                "SELECT * FROM season_stats WHERE player_id IN ({}) "
                "ORDER BY season_year, level"
                .format(",".join("?" * len(val_pids))),
                list(val_pids)).fetchall():
            d = dict(s)
            stats_by_pid.setdefault(d["player_id"], []).append(d)
    print(f"  loaded {len(rows):,} prospect rows")

    debutees = [r for r in rows if r.get("mlb_debut_year")]
    print(f"  debutees in val: {len(debutees):,}")

    diag = []
    for r in debutees:
        debut = int(r["mlb_debut_year"])
        pid = r["player_id"]
        stats = stats_by_pid.get(pid, [])
        if not stats:
            continue
        # Snap years: from draft/entry year up to debut-1
        entry = r.get("draft_year") or r.get("international_signing_year")
        if not entry:
            continue
        for snap in range(int(entry), debut):
            actual_h = debut - snap
            if actual_h < 1 or actual_h > HORIZON:
                continue
            try:
                cum, step_p = _forward_chain(hazards, r, stats, snap)
            except Exception:
                continue
            total = step_p.sum()
            if total < 1e-9:
                continue
            e_time = float(np.sum(np.arange(1, HORIZON + 1) * step_p) / total)
            # The cumulative P(debut by horizon h) — argmax of step_p[h]
            mode_h = int(np.argmax(step_p)) + 1
            p_debut_overall = float(cum[-1])
            diag.append({
                "player_id": pid, "snap_year": snap,
                "actual_h": actual_h, "e_time": e_time,
                "mode_h": mode_h, "p_debut_overall": p_debut_overall,
                "snap_offset": snap - int(entry),
            })

    df = pd.DataFrame(diag)
    print(f"\n{len(df):,} (debutee, snap) rows scored")
    if df.empty:
        return

    # Summary: per actual time-to-debut, mean and percentiles of E[time]
    print(f"\n--- E[time-to-debut | event] by actual time-to-debut ---")
    print(f"{'actual_h':>9} {'n':>6} {'mean_E[t]':>10} {'med_E[t]':>10} "
          f"{'mean_mode':>10} {'corr_dir':>9}")
    for h in range(1, HORIZON + 1):
        s = df[df.actual_h == h]
        if len(s) < 10: continue
        print(f"{h:>9d} {len(s):>6,d} "
              f"{s.e_time.mean():>9.2f} {s.e_time.median():>9.2f} "
              f"{s.mode_h.mean():>9.2f}")

    print(f"\nSpearman corr (E[t] vs actual): "
          f"{df[['e_time','actual_h']].corr(method='spearman').iloc[0,1]:+.3f}")
    print(f"Pearson  corr (E[t] vs actual): "
          f"{df[['e_time','actual_h']].corr(method='pearson').iloc[0,1]:+.3f}")
    print(f"Spearman corr (mode_h vs actual): "
          f"{df[['mode_h','actual_h']].corr(method='spearman').iloc[0,1]:+.3f}")

    # Also break out by snap_offset
    print(f"\n--- corrs by snap_offset (years-in-pro at snap) ---")
    print(f"{'snap_off':>8} {'n':>6} {'spearman':>10} {'pearson':>10} "
          f"{'mean_actual':>12} {'mean_E[t]':>10}")
    for so in sorted(df.snap_offset.unique()):
        s = df[df.snap_offset == so]
        if len(s) < 30: continue
        sp = s[['e_time','actual_h']].corr(method='spearman').iloc[0,1]
        pr = s[['e_time','actual_h']].corr(method='pearson').iloc[0,1]
        print(f"{so:>8d} {len(s):>6,d} {sp:>+9.3f} {pr:>+9.3f} "
              f"{s.actual_h.mean():>11.2f} {s.e_time.mean():>9.2f}")


if __name__ == "__main__":
    main()
