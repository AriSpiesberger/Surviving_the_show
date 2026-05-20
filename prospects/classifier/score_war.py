"""
prospects/classifier/score_war.py
====================================

Score recent prospects with continuous expected career WAR.

    pred_career_war = P(MLB_DEBUT)  *  E[career_WAR | debut]

The first factor comes from the v0.5 binary classifier (already trained,
selection-bias-corrected MiLB-only). The second comes from war_regressor_v1
(also MiLB-only feature window; trained only on debuted, matured draftees).

This avoids the discontinuity the user noticed in the discrete classifier
when prospects reach MLB: rookies who just debuted get a high P(debut)~1 by
construction and the regressor handles "given they debuted, how good are
they" as a continuous quantity.

Output: extends scored_prospects.csv with three columns:
  - pred_career_war       (mean)
  - pred_career_war_lo    (10th pct across bootstraps & via P(debut)*lo)
  - pred_career_war_hi    (90th pct)
  - pred_war_given_debut  (the conditional, useful for ranking already-debuted players)

Usage:
    python -m prospects.classifier.score_war \\
        [--db prospects.db] \\
        [--war-model models/war_regressor_v1.pkl] \\
        [--event-model models/event_classifiers_v0.5_pre2021_milb.pkl] \\
        [--years 2024 2025] \\
        [--out scored_prospects.csv]
"""

from __future__ import annotations

import argparse
import csv
import pickle

import numpy as np

from prospects.classifier.model import load_models
from prospects.features.windowed import build_windowed_features
from prospects.schema import CareerEvent
from prospects.storage import ProspectDB


def _latest_milb_year(stats, fallback):
    yrs = [s["season_year"] for s in stats
           if s.get("season_year") is not None
           and (s.get("level") or "").upper() != "MLB"]
    return max(yrs) if yrs else fallback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="prospects.db")
    parser.add_argument("--war-model", default="models/war_regressor_v1.pkl")
    parser.add_argument("--event-model",
                        default="models/event_classifiers_v0.5_pre2021_milb.pkl")
    parser.add_argument("--years", type=int, nargs="+", default=[2024, 2025])
    parser.add_argument("--current-year", type=int, default=2026)
    parser.add_argument("--out", default="scored_prospects.csv")
    args = parser.parse_args()

    db = ProspectDB(args.db)
    print(f"Loading event classifier: {args.event_model}")
    event_models = load_models(args.event_model)
    print(f"Loading WAR regressor: {args.war_model}")
    with open(args.war_model, "rb") as f:
        war = pickle.load(f)
    war_members = war["members"]
    print(f"  {len(war_members)} bootstrap regressors, "
          f"trained on {war.get('n_train')} players")

    with db._connect() as conn:
        placeholders = ",".join("?" * len(args.years))
        prospects = [dict(r) for r in conn.execute(
            f"""SELECT * FROM prospects
                WHERE draft_year IN ({placeholders})
                ORDER BY draft_year, draft_round, draft_pick""",
            args.years,
        ).fetchall()]
        stats_rows = conn.execute("SELECT * FROM season_stats").fetchall()
    stats_by_pid: dict[str, list] = {}
    for s in stats_rows:
        d = dict(s)
        stats_by_pid.setdefault(d["player_id"], []).append(d)

    print(f"Scoring {len(prospects)} prospects from years {args.years}...")
    rows = []
    for p in prospects:
        stats = stats_by_pid.get(p["player_id"], [])
        as_of = _latest_milb_year(stats, fallback=p.get("draft_year") or args.current_year)
        x = build_windowed_features(p, stats, as_of, milb_only=True).reshape(1, -1)

        # Event probs (mean across bootstraps)
        event_p: dict[str, float] = {}
        for event in CareerEvent.all_events():
            P = event_models[event].predict_proba(x)
            event_p[event.name] = float(P.mean())
            event_p[event.name + "_lo"] = float(np.percentile(P, 10))
            event_p[event.name + "_hi"] = float(np.percentile(P, 90))

        # WAR | debut from regressor bootstraps
        war_samples = np.array([m.predict(x)[0] for m in war_members])
        war_mean = float(war_samples.mean())
        war_lo = float(np.percentile(war_samples, 10))
        war_hi = float(np.percentile(war_samples, 90))
        # Floor at 0 for display (negative career WAR exists in training, but
        # for a card-EV perspective we treat fringe MLB the same as zero).
        war_mean_clip = max(war_mean, 0.0)

        p_debut = event_p["MLB_DEBUT"]
        # Expected career WAR. Bound CI by combining both uncertainty sources:
        #   lower = lo(P_debut) * lo(WAR|debut), upper = hi * hi
        pred_war = p_debut * war_mean_clip
        pred_war_lo = max(event_p["MLB_DEBUT_lo"] * max(war_lo, 0.0), 0.0)
        pred_war_hi = event_p["MLB_DEBUT_hi"] * max(war_hi, 0.0)

        out = {
            "player_id": p["player_id"],
            "name": p["name"],
            "draft_year": p.get("draft_year"),
            "draft_round": p.get("draft_round"),
            "draft_pick": p.get("draft_pick"),
            "primary_position": p.get("primary_position"),
            "is_pitcher": int(bool(p.get("is_pitcher"))),
            "current_org": p.get("current_org"),
            "origin": p.get("origin"),
            "as_of_year": as_of,
            "has_milb_data": int(any(
                (s.get("level") or "").upper() != "MLB" for s in stats
            )),
            "pred_career_war": round(pred_war, 2),
            "pred_career_war_lo": round(pred_war_lo, 2),
            "pred_career_war_hi": round(pred_war_hi, 2),
            "pred_war_given_debut": round(war_mean_clip, 2),
            "pred_war_given_debut_lo": round(max(war_lo, 0.0), 2),
            "pred_war_given_debut_hi": round(max(war_hi, 0.0), 2),
        }
        for event in CareerEvent.all_events():
            out[f"p_{event.name}"] = round(event_p[event.name], 4)
            out[f"p_{event.name}_lo"] = round(event_p[event.name + "_lo"], 4)
            out[f"p_{event.name}_hi"] = round(event_p[event.name + "_hi"], 4)
        # Composite kept for back-compat with prior CSV
        out["score"] = round(
            out["p_MLB_DEBUT"] * 1
            + out["p_ESTABLISHED_MLB"] * 3
            + out["p_ALL_STAR_ONCE"] * 8
            + out["p_ALL_STAR_THREE_PLUS"] * 12
            + out["p_MAJOR_AWARD"] * 20,
            3,
        )
        rows.append(out)

    rows.sort(key=lambda r: r["pred_career_war"], reverse=True)

    fieldnames = list(rows[0].keys()) if rows else []
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {args.out}")

    print(f"\nTop 25 by expected career WAR:")
    print(f"{'Rank':>4} {'E[WAR]':>7} {'|debut':>7} {'P(MLB)':>7}  "
          f"{'Player':<28} {'Yr':>4} Pick")
    print("-" * 90)
    for i, r in enumerate(rows[:25], 1):
        pick_str = f"R{r['draft_round']}.{r['draft_pick']}" if r['draft_round'] else "—"
        print(f"{i:>4} {r['pred_career_war']:>7.2f} "
              f"{r['pred_war_given_debut']:>7.2f} "
              f"{r['p_MLB_DEBUT']:>7.3f}  "
              f"{r['name'][:28]:<28} {r['draft_year']:>4} {pick_str}")


if __name__ == "__main__":
    main()
