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

Reproducible evaluation of the v2.4 stack against the **15% val player
slice** (`model/train/make_split`, seed=42) — players neither the landmark
hazards nor the joint XGBoost head trained on. Validation universe: drafted
players with `draft_year ≤ 2020` **and international signees whose first
non-MLB season is ≤ 2020** (IFAs entered the split on 2026-09-10; before that
the joint layer trained on ~14k of them but val held none, so they were never
held-out-evaluated).

**The sheet is scored by v2.5, not by the model evaluated here.** v2.5 is this
same recipe refit on 100% of players (fit + val). It therefore has no held-out
number of its own; v2.4 below is the recipe's report card, and the per-yip buy
thresholds are computed on THIS held-out run and passed to v2.5 as a file. The numbers below are the **deployable
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

**LABEL-LEAK CORRECTION (2026-09-17).** Two inputs were leaking the debut
label and both are cut:

1. `scout_servicetime` — MLB service time *as of the scrape date*, stamped
   onto every historical season of the "point-in-time" scouting file (99.8% of
   players carry one value across all their seasons; 2,573 scouting rows dated
   BEFORE the player's debut show service time > 0). Both layers keyed on it.
   Held-out rows carry the stamp too, so validation looked excellent while the
   live board — where nobody has service time yet — was inverted: 2026 AA/AAA
   players aged 23.5+ scored p3y **0.06 if org-ranked vs 0.30 if unranked**
   (history: 0.63 vs 0.40). After the fix: **0.57 vs 0.29**.
2. Signing-bonus *presence* — on file for ~9% of 2005–2016 draftees who never
   debuted vs ~60% of those who did. Rule now enforced in
   `features/pedigree_rules.py`: a field is usable only where its coverage does
   not depend on outcome. The bonus survives only for drafts 2017+, rounds 1–10
   (100% covered every year, identical for debuted and never-debuted).

Cost of honesty, same split: debut@3 AP **0.593 → 0.525**, debut@6 **0.611 →
0.569**. Every number below is post-fix. `tools/leak_audit.py` re-runs the
model-free checks (future-blindness of all features, outcome-dependent
coverage, source stamping, identity across the split, banned features and
staleness inside the promoted bundles) and gates the weekly sheet.

A third defect surfaced on the way: `model.train.hazards` exits 0 with "already
exist" unless forced, so the production hazards that score the live sheet had
sat frozen at 2026-09-08 through two "full" retrains. `refresh` and the weekly
now force it, and the audit checks its feature contract and freshness.

**What survived the correction:** the joint-layer gains (raw features,
monotone-h, full coverage, era calibration) were measured before the label-leak fix (debut@3 0.614 vs 0.557). On the
clean data the margin is small: at h=6 the v2.1c-recipe OOF model reads **0.561**
against v2.4's **0.569**. Treat the older +10% as inflated. What did NOT survive: the apparent hazard-capacity gains —
`hz3_max` HP (kept, harmless) measures within noise of default HP on the
clean split; its dramatic "wins" were the leak rewarding memorization.

**Recent-cohort augmentation (v2.4).** The joint layer also trains on
post-cutoff entry cohorts' (2021+) resolved short-horizon (row, h) pairs,
scored with val-excluded hazards (`model/train/score_recent_cohorts`). The
random-split val below CANNOT see this gain (it holds only ≤2020 entries) —
the original walk-forward A/B (`model/train/exp_walkforward3`) reported
+0.04..+0.07 out-of-era debut@3 AP, but that figure predates the label-leak fix
AND used a harness that stacked the joint layer on in-sample hazard features
(see the era section). **It has not been re-measured and should be treated as
unverified.** The augmentation is kept — it is the only route by which post-2020
entrants reach the model, and lifting the entry cap inside the hazard layer as
well was tested fairly on 2026-09-19 and adds nothing on top of it.

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
signing bonus gated to the completely-covered block (2017+, rounds 1–10),
IFA birth dates backfilled from the MLB people endpoint (a NULL birth date used
to impute age 22 and inflate scores), `age_during_season` derived for 20k rows,
252 false-negative debut labels repaired from strict-mlbam MLB rows.
Point-in-time scouting (FanGraphs Board 2017–26 + Trouble-With-The-Curve
2013–19): 73 grade/physical/velo/rank/ETA columns in the hazard panel
(no-lookahead, season ≤ snapshot; `servicetime`, `contact_style` and
`versatility_count` banned) + a 5-col current-snapshot
summary (`scout_fv, scout_ovr_rank, scout_eta_gap, scout_risk,
scout_is_scouted`) fed to the XGB. HOF_TRAJECTORY dropped from the event set.

## Stack

| Layer | Model | Trained on |
|---|---|---|
| Hazards (per-fold OOF, eval) | `runs/hz0_default/scratch/oof/fold[0-5]_hazards.pkl` | Each fold trained on the OTHER 5 (val pids excluded, partition verified). HistGBT, default HP (capacity retune measured NEUTRAL on the clean split), 325 features + the landmark offset k. Survival → censoring-aware. |
| Hazards (production) | `runs/current/models/hazards.pkl` | 100% of ≤2020 data, default HP, **force-retrained on every refresh and weekly run** (it was silently frozen at 2026-09-08 until 2026-09-17). Scores the 2026 cohort. |
| Conditional joint XGB | `runs/current/models/joint_xgb_v2.4.pkl` (`model/joint2.py`; trained via `model/train/exp_cdf_timing5.py`, incl. recent-cohort augmentation) | OOF stacked, resolved `(row, h)` pairs h=1..10, 251 features incl. 159 raw panel features (full coverage); ~45% of rows are international signees. 5-seed bag, depth 8 / mcw 100 / colsample 0.6 / lr 0.03, monotone in h. |
| **Sheet scorer** | `runs/current/models/joint_xgb_v2.5.pkl` + `calibrators_v2.5.pkl` | The row above refit on 100% of players (fit + val). No held-out metric by construction; agreement with v2.4 on the 2026 pool: p3y correlation 0.994. |
| Calibrators | `runs/current/models/calibrators_v2.4.pkl` | Per-event logistic over `[logit(p), h, yip, …]`, fit on 3-fold cross-fitted OOF predictions, snaps ≥ 2008 only (val never used). |
| Timing | derived — calibrated debut CDF (`joint2.cdf_timing`) | No separate model: `pmf_j = F(j) − F(j−1)` off the calibrated trajectory. Clean-val debutees: median-MAE **1.04 yr** (Spearman 0.61); mean-MAE 1.13 (0.63). Lasso baseline: 1.29 / 0.56. |

**Buy-list (`buylist/build.py`):** thesis = **`P(MLB_DEBUT ≤ 3y)`**
(`xp_MLB_DEBUT_h3`, calibrated) — filter, sort, and the output `p_MLB_DEBUT`
column all use the 3-year debut slice; ceiling events reported at h={H}
(`p_MLB_DEBUT_6y` carried alongside). `time_to_debut` = calibrated-CDF median,
with a `debut_eta_lo`/`debut_eta_hi` (q25–q75) window. Universe filters: EXIT
washouts, point-in-time top-100 drop, currently-MLB drop, R1 kept, **IFAs
included**, and a years-in-pro ≤ 3 cap that applies **only to players aged 23+**
(an IFA signed at 16 is in his 5th pro year at 21). Selection = per-yip
thresholds at 60% held-out precision. Prices: raw base 1st Bowman Chrome autos
only — in-person / TTM / third-party-authenticated signatures and non-Chrome
cards are rejected (they were 5% of listings but set the quoted lowest buy-now).

**Calibration finding (v2.3, clean split).** The Reliability section below
is the source of truth: probabilities are calibrated on cross-fitted OOF
predictions (2008+ snaps, never val), and the honest reliability evidence is
the fit-OOF bucket table being flat (±1–2% everywhere). Pooled calib ratios
in these tables include the pre-2008 regime the map deliberately ignores and
read below 1.0 for that reason. Judge sheet trustworthiness by the 2008+
bucket tables, and expect high-probability buckets to be thin (small n) on a
10% val sample — bucket wobble of ±5–10pts at n≈100 is sampling noise, not
miscalibration. STAR_PLUS_ELITE below h=4 is a ranking signal, not a rate.

**Measured on the clean split (2026-09-17):** everything above the per-yip bars
— the actual buy rule — printed **0.609** and realized **0.602** on 1,040
held-out rows. Below 0.70 the printed debut probabilities are trustworthy as
written; **above 0.70 they run 5–8 points hot** (printed 0.85 → realized ~0.79),
in every slice. Six alternative calibrators (spline hinges, isotonic, fits on
bag-matched cross-fit predictions; `model/train/exp_cal_topend`) all tie the
deployed one to the fourth decimal, so the residual is a held-out-population
difference, not the calibrator's shape.

**Era-shift bound (full-stack walk-forward, `model/train/exp_walkforward2`).**
Re-measured 2026-09-19 on leak-free features with a corrected harness
(`model/train/exp_walkforward_h`). The earlier harness scored the joint layer's
training rows with hazards fit on those same players — in-sample hazard
features in training, out-of-sample at evaluation — which handicapped every
stacked model; production stacks out-of-fold and so does the corrected harness.
Train on everything known at origin Y, score the entry-(Y, Y+6] cohort at Y+6,
judge on debuts within 3 more years:

| origin → scored cohort | AP | AUC | pred ÷ actual |
|---|---|---|---|
| 2016 → 2022 | 0.634 | 0.950 | 1.21 |
| 2014 → 2020 (no MiLB season) | 0.374 | 0.869 | 0.30 |
| 2012 → 2018 | 0.645 | 0.954 | 1.17 |

In the two normal origins ranking holds, levels over-predict by ~1.2×, and the
top of the range is honest out of era (printed 0.86 → realized 0.90; 0.87 →
0.91). The 2020 snapshot — every current-season feature stale — is
under-predicted 3–4× by **every** model tried; a missing season is the one
consistent failure, and it also applies to any player who loses a year.
Read the sheet accordingly: rank-order and relative comparisons are robust;
absolute probabilities carry era-level uncertainty.

**Is the two-stage stack the right macro?** Tested on the same harness, paired
bootstrap on the mean over origins: a single-stage GBM with no hazard inputs
(all 325 raw features, recency-weighted) **+0.008** [+0.000, +0.016]; a 50/50
logit blend of the stack and that model **+0.011** [+0.007, +0.016], positive
at every origin, and a tie in era. Training the hazard layer on every resolved
landmark cell instead of capping entry at Y: **−0.008** [−0.015, −0.001] (a
tie in normal years, worse at the 2020 origin). An online era intercept:
nothing. The production architecture stands; the blend is a small, optional
gain. An earlier same-day reading (+0.04 for the blend, "the stack is
overconfident out of era") was the harness flaw and is retracted.

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
