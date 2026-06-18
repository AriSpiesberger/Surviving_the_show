"""Regenerate evaluation/README.md from the CSVs in evaluation/v2.0b_landmark/.

Static prose is templated here; every TABLE is rebuilt from the latest
per_bucket / per_yip / per_level / thresholds CSVs + headline.json, so the
README never drifts from the numbers.

    python -m scripts_v17.validate.gen_eval_readme
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
EV = REPO / "evaluation" / "v2.0b_landmark"
OUT = REPO / "evaluation" / "README.md"

EVENTS = ["TOP_100_PROSPECT", "MLB_DEBUT", "ESTABLISHED_MLB", "STAR_PLUS_ELITE"]
BUCKET_ORDER = ["ALL", "R1", "R2-R3", "R4-R10", "R10+", "IFA"]
LEVEL_ORDER = ["ALL", "RK", "A-", "A", "A+", "AA", "AAA", "NONE"]
COLS = "| {grp} | {n} | {pos} | {base:.2f}% | {auc} | {ap} | {lift} | {sp} | {prec} | {rec} | {f1} | {tp} | {fp} | {fn} |"
HDR = ("| {g} | n | pos | base% | AUC | AP | AP_lift | spearman | precision "
       "| recall | F1 | TP | FP | FN |\n"
       "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")


def _row(r, grp):
    empty = int(r["pos"]) == 0
    f3 = lambda v: "—" if (empty or pd.isna(v)) else f"{v:.3f}"
    return COLS.format(
        grp=grp, n=int(r["n"]), pos=int(r["pos"]), base=r["base_rate"] * 100,
        auc=f3(r["auc"]), ap=f3(r["ap"]),
        lift="—" if empty or pd.isna(r["ap_lift"]) else f"{r['ap_lift']:.1f}×",
        sp=f3(r["spearman_rho"]),
        prec=f"{r['precision']:.3f}" if int(r["predicted_positives"]) > 0 else "—",
        rec="—" if empty else f"{r['recall']:.3f}",
        f1="—" if (empty or pd.isna(r["f1"]) or r["f1"] == 0) else f"{r['f1']:.3f}",
        tp=int(r["tp"]), fp=int(r["fp"]), fn=int(r["fn"]))


def _section(df, group_col, order, label):
    out = []
    for ev in EVENTS:
        sub = df[df.event == ev]
        rows = []
        keys = order if order else sorted(sub[group_col].unique())
        for k in keys:
            cell = sub[sub[group_col] == k]
            if not cell.empty:
                rows.append(_row(cell.iloc[0], str(k)))
        if rows:
            out.append(f"\n#### {ev}\n\n{HDR.format(g=label)}\n" + "\n".join(rows))
    return "\n".join(out)


def _per_horizon(df):
    """Trajectory-quality table: AP/AUC/Brier/calibration by event x horizon h,
    each row evaluated on the slice resolved at that h (years_fwd >= h)."""
    hdr = ("| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |\n"
           "|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    out = []
    for ev in EVENTS:
        sub = df[df.event == ev].sort_values("horizon")
        if sub.empty:
            continue
        rows = [hdr]
        for _, r in sub.iterrows():
            calib = "—" if pd.isna(r["calib_ratio"]) else f"{r['calib_ratio']:.2f}"
            rows.append(
                f"| {int(r['horizon'])} | {int(r['n'])} | {int(r['pos'])} | "
                f"{r['base_rate']*100:.2f}% | {r['auc']:.3f} | {r['ap']:.3f} | "
                f"{r['ap_lift']:.1f}× | {r['brier']:.4f} | {calib} |")
        out.append(f"\n#### {ev}\n\n" + "\n".join(rows))
    return "\n".join(out)


def main():
    global EV, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=str(EV),
                    help="Directory of per_* CSVs + headline.json to render.")
    ap.add_argument("--out", default=str(OUT),
                    help="Output markdown path.")
    ap.add_argument("--tag", default=None,
                    help="Convenience: render the tagged eval "
                         "(evaluation/v2.0b_<tag>_landmark/ -> "
                         "evaluation/README_<tag>.md) unless overridden.")
    args = ap.parse_args()
    if args.tag:
        if args.in_dir == str(EV):
            args.in_dir = str(REPO / "evaluation" / f"v2.0b_{args.tag}_landmark")
        if args.out == str(OUT):
            args.out = str(REPO / "evaluation" / f"README_{args.tag}.md")
    EV = Path(args.in_dir)
    OUT = Path(args.out)

    bucket = pd.read_csv(EV / "per_bucket_validation.csv")
    yip = pd.read_csv(EV / "per_yip_validation.csv")
    level = pd.read_csv(EV / "per_level_validation.csv")
    horizon = pd.read_csv(EV / "per_horizon.csv")
    head = json.loads((EV / "headline.json").read_text())
    H = int(head.get("eval_horizon", 6))

    # headline (ALL bucket per event) + weighted
    hl = ["| Event | n | base% | AP | lift | AUC | spearman | precision | recall | F1 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for ev in EVENTS:
        r = bucket[(bucket.event == ev) & (bucket.bucket == "ALL")].iloc[0]
        prec = f"{r['precision']:.3f}" if int(r["predicted_positives"]) > 0 else "—"
        f1 = "—" if (pd.isna(r["f1"]) or r["f1"] == 0) else f"{r['f1']:.3f}"
        hl.append(f"| {ev} | {int(r['n'])} | {r['base_rate']*100:.2f}% | "
                  f"**{r['ap']:.3f}** | {r['ap_lift']:.1f}× | {r['auc']:.3f} | "
                  f"{r['spearman_rho']:.3f} | {prec} | {r['recall']:.3f} | {f1} |")
    hl.append(f"| **weighted-AP** | | | **{head['weighted_ap']:.3f}** | | | | | | |")

    md = f"""# Held-out validation — v2.1c conditional refinement

Reproducible evaluation of the v2.1c landmark stack against the **10% val
player slice** of the v1.17 seed=42 split — players neither the landmark
hazards nor the joint XGBoost head trained on. Validation universe: drafted
players with `draft_year ≤ 2020` (plus IFAs).

**Conditional refinement.** The joint XGB is no longer a terminal scalar head.
It is a *conditional refinement* of the hazard trajectory: given a player's full
per-year hazard curves (`hk1..hk10`) + baseline + a **target horizon h**, it
outputs the refined cumulative `P(event by snap+h)`. Sweeping h=1..10 yields a
per-year trajectory per event instead of one collapsed scalar. Horizon `h` is an
input feature (the same trick the landmark hazards use to kill train/inference
mismatch), and the hazard model's own cumulative answer at h
(`haz_cum_h_<event>`) is fed in as the quantity to refine — `FEAT_COND` = 74
features (6 cumulative probs + age/yip + 6 yip-interactions + 5 scouting + 50
hazard-curve steps + 4 per-event anchors + h).

**Yardstick: per-horizon, resolved slice.** Labels are right-censored, so each
`(player-snap, h)` cell is used only where it is *resolved* — `years_fwd >= h`,
which (since `years_fwd` is row-level) makes every event head's label
trustworthy with no per-cell masking. Training keeps resolved `(row, h)` pairs;
evaluation scores `xp_<event>_h{{h}}` vs `realized_by_h` on the rows resolved at
that h. The headline below is at **h={H}** (the publish horizon); the per-horizon
section reports the full h=1..10 trajectory. The **hazards** are survival models
— censoring-aware by construction. Anything at h>10 is the hazard layer's
opinion, not the XGB's (no extrapolation).

**Data integrity:** birthdates backfilled for 2024–25 draft classes, FG/TWTC
crosswalk 89%→96%, trade-aware `current_org`, IFA entry-year anchors,
signing-bonus backfill. Point-in-time scouting (FanGraphs Board 2017–26 +
Trouble-With-The-Curve 2013–19): 76 grade/physical/velo/rank/ETA columns in the
hazard panel (no-lookahead, season ≤ snapshot) + a 5-col current-snapshot
summary (`scout_fv, scout_ovr_rank, scout_eta_gap, scout_risk,
scout_is_scouted`) fed to the XGB. HOF_TRAJECTORY dropped from the event set.

## Stack

| Layer | Model | Trained on |
|---|---|---|
| Hazards (per-fold OOF, eval) | `scratch/v20b_oof/fold[0-5]_hazards.pkl` | Each fold trained on the OTHER 5 (val pids excluded). HistGBT, default HP, 314 features (incl. 76 scouting). Survival → censoring-aware. |
| Hazards (production) | `models/event_classifiers_v2.0b_prod.pkl` | 100% of ≤2020 data. Scores the 2026 cohort (entry 2024–26 — not in training, so no leakage). |
| Conditional joint XGB | `models/joint_xgb_v2.0b_{{oof,prod}}.pkl` (`fit_joint_xgb_cond.py`) | OOF stacked, expanded to resolved `(row, h)` pairs for h=1..10. `multi_output_tree` over the 4 heads; per-horizon censoring built in (no `--censor-window`). Outputs `P(event by snap+h)`; monotone in h via cummax at inference. |
| Timing | `models/time_to_debut_v2.0b_prod.pkl` | LassoCV on v2.0b hazard probs + `mean_t`/`sd_t`. MAE 1.14 yr, Spearman 0.66. |

**Buy-list (`build_v2.0_buylist.py`):** thesis = **`P(MLB_DEBUT ≤ 3y)`**
(`xp_MLB_DEBUT_h3`) — filter, sort, and the output `p_MLB_DEBUT` column all use
the 3-year debut slice; ceiling events (top100/established/star) reported at
h={H} for context (`p_MLB_DEBUT_6y` carried alongside). Universe filters: EXIT
washouts, point-in-time top-100 drop, currently-MLB drop, R1 kept.

**Calibration finding.** Ranking (AUC) is 0.95–0.99 across all events and all h.
MLB_DEBUT is near-perfectly calibrated (`calib` ≈ 1.0 from h≥3). **STAR_PLUS_ELITE
is well-ranked but under-calibrated at long horizons** (`calib` ≈ 0.7 by h≥4) —
the magnitude of stardom is under-predicted; a per-horizon isotonic recal on that
head is the fix (ranking needs none).

## Headline (ALL bucket, h={H}, threshold = 0.60)

{chr(10).join(hl)}

(MLB_DEBUT 2× weight, others 1×, per-event eligibility filters. Scores =
`xp_<event>_h{H}` vs realized-within-{H}y, on rows resolved at h={H}.)

## Per-horizon trajectory (h=1..10, resolved at each h)
{_per_horizon(horizon)}

## Per-bucket (h={H}, threshold = 0.60)
{_section(bucket, "bucket", BUCKET_ORDER, "bucket")}

## Per-yip (h={H}, threshold = 0.60)
{_section(yip, "snap_offset", list(range(11)), "yip")}

## Per-level (h={H}, threshold = 0.60)
{_section(level, "cur_level", LEVEL_ORDER, "level")}

## Statistics glossary

| Metric | Meaning |
|---|---|
| `ap` | Average Precision = AU-PR. Headline rare-event metric. |
| `ap_lift` | `ap / base_rate` — how many × random the ranking is. |
| `auc` | Area under ROC. Insensitive to class imbalance. |
| `brier` | Mean squared error of the probability. Lower = better calibrated. |
| `calib` | Mean-predicted ÷ observed rate. 1.0 = calibrated; <1 under-predicts. |
| `spearman_rho` | Rank correlation between score and realized 0/1. |
| `precision/recall/f1` | At threshold 0.60. `—` = undefined (no predicted positives / no positives). |
| `bucket` | Draft pedigree: R1, R2-R3, R4-R10, R10+ (rounds 11+), IFA. |
| `snap_offset` (yip) | Years since entry. |
| `cur_level` | Player's level at snapshot: RK/A-/A/A+/AA/AAA/NONE. |

## Reproducing

```bash
# OOF folds + hazards, then the conditional joint XGB (per-horizon censoring is
# built in; wired into run_v2_0b_oof stage 6 and train_v2_0b_prod stage 1)
python -m scripts_v17.train.run_v2_0b_oof
python -m scripts_v17.train.train_v2_0b_prod    # 100% prod hazards + cond XGB + score 2026

# validation — per-horizon, headline at the publish horizon (h={H})
python -m scripts_v17.validate.regen_eval_v2_0b_honest --eval-horizon {H}
python -m scripts_v17.validate.gen_eval_readme

# buy list — P(debut <= 3y) thesis
python scripts_v17/buylist/build_v2.0_buylist.py \\
    --long results/scored/snap2026_v1.18b_landmark_long.csv \\
    --xgb models/joint_xgb_v2.0b_prod.pkl --debut-horizon 3 --threshold 0.60
```
"""
    OUT.write_text(md, encoding="utf-8")
    print(f"Wrote {OUT} ({len(md):,} chars)")
    print(f"weighted-AP = {head['weighted_ap']:.4f}")


if __name__ == "__main__":
    main()
