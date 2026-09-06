"""Regenerate evaluation/README.md from the CSVs in evaluation/v2.0b_landmark/.

Static prose is templated here; every TABLE is rebuilt from the latest
per_bucket / per_yip / per_level / thresholds CSVs + headline.json, so the
README never drifts from the numbers.

    python -m prospects.evaluation.report
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from prospects import config
from prospects.config import REPO_ROOT as REPO
_RUN = config.run()
EV = _RUN.evaluation
OUT = _RUN.evaluation / "README.md"

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


def _reliability(df):
    """Probability-bucket reliability tables — THE calibration view: a sheet
    probability is trustworthy iff its bucket's realized rate matches it."""
    if df is None or df.empty:
        return "\n(reliability.csv missing — rerun evaluation.run)"
    hdr = ("| predicted | n | avg pred | actual | diff |\n"
           "|---|---:|---:|---:|---:|")
    out = []
    for ev in EVENTS:
        for h in sorted(df[df.event == ev]["h"].unique()):
            sub = df[(df.event == ev) & (df.h == h)]
            if sub.empty:
                continue
            rows = [hdr]
            for _, r in sub.iterrows():
                rows.append(
                    f"| {r['bucket']} | {int(r['n']):,} | "
                    f"{r['avg_pred']*100:.1f}% | {r['actual']*100:.1f}% | "
                    f"{r['diff']*100:+.1f}% |")
            out.append(f"\n#### {ev} — P(within {int(h)}y)\n\n"
                       + "\n".join(rows))
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
                    help="Convenience: render runs/<tag>/evaluation/ into "
                         "runs/<tag>/evaluation/README.md unless overridden.")
    args = ap.parse_args()
    if args.tag:
        tagged = config.run(args.tag)
        if args.in_dir == str(EV):
            args.in_dir = str(tagged.evaluation)
        if args.out == str(OUT):
            args.out = str(tagged.evaluation / "README.md")
    EV = Path(args.in_dir)
    OUT = Path(args.out)

    try:
        reliability = pd.read_csv(EV / "reliability.csv")
    except FileNotFoundError:
        reliability = None
    bucket = pd.read_csv(EV / "per_bucket_validation.csv")
    yip = pd.read_csv(EV / "per_yip_validation.csv")
    level = pd.read_csv(EV / "per_level_validation.csv")
    horizon = pd.read_csv(EV / "per_horizon.csv")
    head = json.loads((EV / "headline.json").read_text())
    H = int(head.get("eval_horizon", 6))
    DH = config.DEFAULT_DEBUT_HORIZON
    THR = config.DEFAULT_THRESHOLD

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

    md = f"""# Held-out validation — v2.4 (raw-feature bag + recent-cohort augmentation)

Reproducible evaluation of the v2.4 stack against the **10% val player
slice** of the v1.17 seed=42 split — players neither the landmark hazards nor
the joint XGBoost head trained on. Validation universe: drafted players with
`draft_year ≤ 2020` (plus IFAs). The numbers below are the **deployable
calibrated probabilities** (calibrators applied before metrics), and the
calibrators were fit on cross-fitted OOF predictions — never on this val
slice.

**SPLIT-LEAK CORRECTION (2026-09-05).** `val_pids.txt` regenerated on Sep 1
(the universe grew, `make_split` reshuffles) while `stage_partition` silently
reused the Aug-15 fold lists — **90% of "held-out" val players were inside
training** for every evaluation Sep 1–5. All READMEs from that window are
inflated (the v2.1c baseline read 0.647 debut@3; its honest value is 0.557).
`stage_partition` now hard-verifies zero val overlap and purges stale
partitions. The tables below are from the rebuilt, verified-clean split.

**What survived the correction:** the joint-layer gains (raw features,
monotone-h, full coverage, era calibration) are real — honest debut@3
**0.614 vs 0.557** baseline (+10%), corroborated throughout by the val-free
internal screens. What did NOT survive: the apparent hazard-capacity gains —
`hz3_max` HP (kept, harmless) measures within noise of default HP on the
clean split; its dramatic "wins" were the leak rewarding memorization.

**Recent-cohort augmentation (v2.4).** The joint layer also trains on
post-cutoff entry cohorts' (2021+) resolved short-horizon (row, h) pairs,
scored with val-excluded hazards (`model/train/score_recent_cohorts`). The
random-split val below CANNOT see this gain (it holds only ≤2020 entries) —
the walk-forward A/B measured it where it matters: **+0.04..+0.07 out-of-era
debut@3 AP and roughly a third of the era-drift over-prediction removed**
(`model/train/exp_walkforward3`) — the recent cohorts carry the current
promotion regime.

**Conditional refinement, un-bottlenecked (v2.2, retained).** The joint
layer is a *conditional refinement* of the hazard trajectory: given a
player's per-year hazard curves (`hk1..hk10`) + baseline + a **target
horizon h**, it outputs the refined cumulative `P(event by snap+h)`;
sweeping h=1..10 yields the per-year trajectory per event. Relative to
v2.1c:

1. **The head sees the evidence, not just the hazards' verdict**: on top of
   v2.1c's `FEAT_COND` (74), it reads the hazard layer's per-event timing
   moments (`mean_t`/`sd_t`), `p_ALL_STAR_ONCE`/`p_MAJOR_AWARD`, explicit
   horizon margins (`h − mean_t`), and the **top-160 raw landmark-panel
   features** (age-vs-level, level-adjusted rates, trajectory deltas,
   scouting grades) built as-of the snap for every row — 252 features total
   (`joint2.attach_raw_features`, full coverage incl. the scoring cohort).
2. **Monotone in h by construction**: a 5-seed bag of XGBs with
   `monotone_constraints` +1 on `h_centered` and the horizon margins —
   cummax survives only as residual cleanup, not as the source of
   monotonicity.
3. **Honest, career-stage-aware calibration**: ONE per-event logistic map
   over `[logit(p), h, yip, interactions, quadratics]`, fit on 3-fold
   player-grouped cross-fitted predictions of the training longs — and (new
   in v2.3) **only on snaps ≥ 2008**: the pre-2008 snaps are a different
   data regime (≤2 years of stat history exist in the 2005+ DB; era calib
   0.79 vs 0.91–1.09 for 2008+) and were dragging the map away from the
   deployment-relevant eras. The val slice is a pure reporting set (v2.1c
   fit per-(event,h) calibrators on the same val rows the XGB
   early-stopped on).

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
| Hazards (per-fold OOF, eval) | `runs/hz0_default/scratch/oof/fold[0-5]_hazards.pkl` | Each fold trained on the OTHER 5 (val pids excluded, partition verified). HistGBT, default HP (capacity retune measured NEUTRAL on the clean split), 327 features. Survival → censoring-aware. |
| Hazards (production) | `runs/current/models/hazards.pkl` | 100% of ≤2020 data, default HP. Scores the 2026 cohort (entry 2024–26 — not in training, so no leakage). |
| Conditional joint XGB | `runs/current/models/joint_xgb_v2.4.pkl` (`model/joint2.py`; trained via `model/train/exp_cdf_timing5.py`, incl. recent-cohort augmentation) | OOF stacked, resolved `(row, h)` pairs h=1..10, 252 features incl. 160 raw panel features (full coverage). 5-seed bag, depth 8 / mcw 100 / colsample 0.6 / lr 0.03, monotone in h. |
| Calibrators | `runs/current/models/calibrators_v2.4.pkl` | Per-event logistic over `[logit(p), h, yip, …]`, fit on 3-fold cross-fitted OOF predictions, snaps ≥ 2008 only (val never used). |
| Timing | derived — calibrated debut CDF (`joint2.cdf_timing`) | No separate model: `pmf_j = F(j) − F(j−1)` off the calibrated trajectory. Clean-val debutees: median-MAE **1.04 yr** (Spearman 0.61); mean-MAE 1.13 (0.63). Lasso baseline: 1.29 / 0.56. |

**Buy-list (`buylist/build.py`):** thesis = **`P(MLB_DEBUT ≤ 3y)`**
(`xp_MLB_DEBUT_h3`, calibrated) — filter, sort, and the output `p_MLB_DEBUT`
column all use the 3-year debut slice; ceiling events reported at h={H}
(`p_MLB_DEBUT_6y` carried alongside). `time_to_debut` = calibrated-CDF median,
with a `debut_eta_lo`/`debut_eta_hi` (q25–q75) window. Universe filters: EXIT
washouts, point-in-time top-100 drop, currently-MLB drop, R1 kept.

**Calibration finding (v2.3, clean split).** The Reliability section below
is the source of truth: probabilities are calibrated on cross-fitted OOF
predictions (2008+ snaps, never val), and the honest reliability evidence is
the fit-OOF bucket table being flat (±1–2% everywhere). Pooled calib ratios
in these tables include the pre-2008 regime the map deliberately ignores and
read below 1.0 for that reason. Judge sheet trustworthiness by the 2008+
bucket tables, and expect high-probability buckets to be thin (small n) on a
10% val sample — bucket wobble of ±5–10pts at n≈100 is sampling noise, not
miscalibration. STAR_PLUS_ELITE below h=4 is a ranking signal, not a rate.

**Era-shift bound (full-stack walk-forward, `model/train/exp_walkforward2`).**
Scoring never-seen entry cohorts with label-frozen models at three historical
origins: ranking holds (AP 0.48–0.73, AUC 0.87–0.96 out-of-era) but absolute
probabilities swing **0.7×–2× by era** (COVID, draft-size and minors-
restructuring shocks) — and neither the calibration layer nor recency
weighting can remove it, because the shocks aren't learnable from history.
Read the sheet accordingly: rank-order and relative comparisons are robust;
absolute probabilities are honest to the historical average with era-level
uncertainty around them.

## Headline (ALL bucket, h={H}, threshold = 0.60)

{chr(10).join(hl)}

(MLB_DEBUT 2× weight, others 1×, per-event eligibility filters. Scores =
`xp_<event>_h{H}` vs realized-within-{H}y, on rows resolved at h={H}.)

## Reliability — probability buckets vs realized rates (2008+ snaps)

This is the table that decides whether a sheet probability can be trusted:
players are bucketed by their PRINTED probability and each bucket's realized
rate is shown beside it. `diff` ≈ 0 everywhere = calibrated; positive diff =
the printed number is a floor (model conservative in that range).
{_reliability(reliability)}

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

All paths resolve through `prospects.config` to `runs/current/`; the commands
below take no explicit artifact paths.

```bash
# OOF folds (default hazard HP; stage_partition verifies the split)
python -m prospects.model.pipelines.oof

# v2.3 joint layer: full-coverage raw-feature bag + era-aware OOF calibrators
python -m prospects.model.train.exp_cdf_timing5 --cal-min-snap-year 2008 \\
    --out-dir runs/current/scratch/v23_build
python -m prospects.model.train.promote_v22 \\
    --source runs/current/scratch/v23_build --version v2.3

# prod hazards + rescore the 2026 cohort
python -m prospects.model.train.hazards --force
python -m prospects.model.pipelines.prod --skip-xgb --skip-buylist

# validation — calibrated, headline at the publish horizon (h={H})
python -m prospects.evaluation.run --xgb runs/current/models/joint_xgb_v2.4.pkl \\
    --calibrators runs/current/models/calibrators_v2.4.pkl --threshold 0.6 --eval-horizon {H}
python -m prospects.evaluation.report

# buy list — P(debut <= {DH}y) thesis, CDF timing + debut window
python -m prospects.buylist.build --xgb runs/current/models/joint_xgb_v2.4.pkl \\
    --calibrators runs/current/models/calibrators_v2.4.pkl --debut-horizon {DH}
```

The weekly retrain (`deploy/weekly_score.py`, ported 2026-09-05) now runs
Stage C = the v2.4 steps above automatically after stage_a + prod; the
Monday job produces the v2.3 buy list end-to-end.
"""
    OUT.write_text(md, encoding="utf-8")
    print(f"Wrote {OUT} ({len(md):,} chars)")
    print(f"weighted-AP = {head['weighted_ap']:.4f}")


if __name__ == "__main__":
    main()
