# Held-out validation — v2.4 (raw-feature bag + recent-cohort augmentation)

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
evaluation scores `xp_<event>_h{h}` vs `realized_by_h` on the rows resolved at
that h. The headline below is at **h=6** (the publish horizon); the per-horizon
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
column all use the 3-year debut slice; ceiling events reported at h=6
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

## Headline (ALL bucket, h=6, threshold = 0.60)

| Event | n | base% | AP | lift | AUC | spearman | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TOP_100_PROSPECT | 35057 | 1.05% | **0.430** | 41.0× | 0.969 | 0.165 | 0.696 | 0.174 | 0.278 |
| MLB_DEBUT | 35279 | 8.68% | **0.567** | 6.5× | 0.923 | 0.413 | 0.748 | 0.248 | 0.372 |
| ESTABLISHED_MLB | 35279 | 2.71% | **0.325** | 12.0× | 0.936 | 0.245 | 0.650 | 0.027 | 0.052 |
| STAR_PLUS_ELITE | 35279 | 0.43% | **0.135** | 31.1× | 0.952 | 0.103 | — | 0.000 | — |
| **weighted-AP** | | | **0.405** | | | | | | |

(MLB_DEBUT 2× weight, others 1×, per-event eligibility filters. Scores =
`xp_<event>_h6` vs realized-within-6y, on rows resolved at h=6.)

## Reliability — probability buckets vs realized rates (2008+ snaps)

This is the table that decides whether a sheet probability can be trusted:
players are bucketed by their PRINTED probability and each bucket's realized
rate is shown beside it. `diff` ≈ 0 everywhere = calibrated; positive diff =
the printed number is a floor (model conservative in that range).

#### TOP_100_PROSPECT — P(within 3y)

| predicted | n | avg pred | actual | diff |
|---|---:|---:|---:|---:|
| 0-5% | 36,027 | 0.1% | 0.1% | +0.0% |
| 5-10% | 228 | 7.3% | 8.8% | +1.5% |
| 10-20% | 168 | 14.0% | 10.1% | -3.9% |
| 20-30% | 80 | 24.9% | 21.2% | -3.6% |
| 30-40% | 35 | 34.8% | 31.4% | -3.4% |
| 40-50% | 38 | 45.2% | 57.9% | +12.7% |
| 50-60% | 27 | 54.5% | 77.8% | +23.3% |
| 60-70% | 20 | 64.7% | 30.0% | -34.7% |
| 70-80% | 15 | 75.2% | 80.0% | +4.8% |
| 80-90% | 17 | 84.3% | 70.6% | -13.7% |

#### TOP_100_PROSPECT — P(within 6y)

| predicted | n | avg pred | actual | diff |
|---|---:|---:|---:|---:|
| 0-5% | 29,144 | 0.2% | 0.2% | +0.0% |
| 5-10% | 246 | 7.2% | 9.8% | +2.5% |
| 10-20% | 211 | 14.1% | 11.8% | -2.3% |
| 20-30% | 94 | 24.7% | 18.1% | -6.6% |
| 30-40% | 43 | 34.8% | 23.3% | -11.5% |
| 40-50% | 40 | 45.0% | 57.5% | +12.5% |
| 50-60% | 27 | 53.8% | 63.0% | +9.1% |
| 60-70% | 27 | 64.8% | 48.1% | -16.6% |
| 80-90% | 17 | 84.5% | 82.4% | -2.1% |

#### MLB_DEBUT — P(within 3y)

| predicted | n | avg pred | actual | diff |
|---|---:|---:|---:|---:|
| 0-5% | 31,943 | 0.5% | 0.7% | +0.2% |
| 5-10% | 1,442 | 7.2% | 8.9% | +1.8% |
| 10-20% | 1,312 | 14.3% | 18.0% | +3.7% |
| 20-30% | 692 | 24.8% | 24.3% | -0.5% |
| 30-40% | 464 | 34.4% | 37.9% | +3.6% |
| 40-50% | 275 | 44.7% | 43.3% | -1.4% |
| 50-60% | 221 | 54.6% | 62.9% | +8.3% |
| 60-70% | 150 | 64.6% | 70.7% | +6.1% |
| 70-80% | 126 | 75.0% | 71.4% | -3.6% |
| 80-90% | 163 | 85.5% | 81.0% | -4.5% |
| 90-100% | 67 | 92.3% | 83.6% | -8.8% |

#### MLB_DEBUT — P(within 6y)

| predicted | n | avg pred | actual | diff |
|---|---:|---:|---:|---:|
| 0-5% | 23,557 | 0.7% | 1.0% | +0.3% |
| 5-10% | 1,907 | 7.2% | 9.7% | +2.5% |
| 10-20% | 1,676 | 14.3% | 16.8% | +2.5% |
| 20-30% | 890 | 24.8% | 29.0% | +4.2% |
| 30-40% | 599 | 34.5% | 36.4% | +1.9% |
| 40-50% | 416 | 44.8% | 45.0% | +0.2% |
| 50-60% | 296 | 55.0% | 55.7% | +0.8% |
| 60-70% | 212 | 64.9% | 67.0% | +2.1% |
| 70-80% | 202 | 74.6% | 81.2% | +6.6% |
| 80-90% | 184 | 85.5% | 84.8% | -0.8% |
| 90-100% | 76 | 92.4% | 90.8% | -1.6% |

#### ESTABLISHED_MLB — P(within 3y)

| predicted | n | avg pred | actual | diff |
|---|---:|---:|---:|---:|
| 0-5% | 36,366 | 0.1% | 0.2% | +0.1% |
| 5-10% | 211 | 7.0% | 10.4% | +3.4% |
| 10-20% | 139 | 13.8% | 19.4% | +5.7% |
| 20-30% | 60 | 24.4% | 36.7% | +12.3% |
| 30-40% | 42 | 33.9% | 35.7% | +1.8% |
| 40-50% | 25 | 44.0% | 48.0% | +4.0% |

#### ESTABLISHED_MLB — P(within 6y)

| predicted | n | avg pred | actual | diff |
|---|---:|---:|---:|---:|
| 0-5% | 28,246 | 0.3% | 0.6% | +0.2% |
| 5-10% | 727 | 7.1% | 10.3% | +3.2% |
| 10-20% | 567 | 13.9% | 19.2% | +5.3% |
| 20-30% | 217 | 24.2% | 35.5% | +11.3% |
| 30-40% | 116 | 34.4% | 41.4% | +7.0% |
| 40-50% | 73 | 44.5% | 50.7% | +6.2% |
| 50-60% | 43 | 54.7% | 76.7% | +22.0% |
| 60-70% | 23 | 63.2% | 69.6% | +6.3% |

#### STAR_PLUS_ELITE — P(within 3y)

| predicted | n | avg pred | actual | diff |
|---|---:|---:|---:|---:|
| 0-5% | 36,832 | 0.0% | 0.0% | +0.0% |
| 5-10% | 19 | 6.8% | 15.8% | +9.0% |

#### STAR_PLUS_ELITE — P(within 6y)

| predicted | n | avg pred | actual | diff |
|---|---:|---:|---:|---:|
| 0-5% | 29,761 | 0.1% | 0.2% | +0.0% |
| 5-10% | 159 | 7.0% | 5.7% | -1.3% |
| 10-20% | 77 | 13.2% | 27.3% | +14.1% |

## Per-horizon trajectory (h=1..10, resolved at each h)

#### TOP_100_PROSPECT

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 45096 | 136 | 0.30% | 0.993 | 0.488 | 161.8× | 0.0020 | 1.01 |
| 2 | 43586 | 245 | 0.56% | 0.984 | 0.473 | 84.1× | 0.0038 | 1.00 |
| 3 | 41850 | 319 | 0.76% | 0.977 | 0.455 | 59.7× | 0.0053 | 0.97 |
| 4 | 39858 | 359 | 0.90% | 0.972 | 0.449 | 49.8× | 0.0064 | 0.97 |
| 5 | 37603 | 368 | 0.98% | 0.968 | 0.435 | 44.5× | 0.0070 | 0.98 |
| 6 | 35057 | 368 | 1.05% | 0.969 | 0.430 | 41.0× | 0.0076 | 0.99 |
| 7 | 32276 | 354 | 1.10% | 0.968 | 0.433 | 39.4× | 0.0079 | 0.98 |
| 8 | 29532 | 337 | 1.14% | 0.968 | 0.434 | 38.0× | 0.0082 | 0.98 |
| 9 | 26820 | 313 | 1.17% | 0.968 | 0.434 | 37.2× | 0.0084 | 0.99 |
| 10 | 23999 | 289 | 1.20% | 0.968 | 0.422 | 35.0× | 0.0088 | 1.01 |

#### MLB_DEBUT

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 45376 | 795 | 1.75% | 0.965 | 0.394 | 22.5× | 0.0131 | 0.88 |
| 2 | 43862 | 1555 | 3.55% | 0.953 | 0.492 | 13.9× | 0.0238 | 0.87 |
| 3 | 42119 | 2217 | 5.26% | 0.943 | 0.531 | 10.1× | 0.0337 | 0.89 |
| 4 | 40115 | 2703 | 6.74% | 0.933 | 0.548 | 8.1× | 0.0421 | 0.90 |
| 5 | 37844 | 2978 | 7.87% | 0.926 | 0.559 | 7.1× | 0.0485 | 0.91 |
| 6 | 35279 | 3063 | 8.68% | 0.923 | 0.567 | 6.5× | 0.0528 | 0.92 |
| 7 | 32481 | 3008 | 9.26% | 0.920 | 0.568 | 6.1× | 0.0562 | 0.91 |
| 8 | 29722 | 2882 | 9.70% | 0.917 | 0.564 | 5.8× | 0.0591 | 0.91 |
| 9 | 26998 | 2713 | 10.05% | 0.913 | 0.560 | 5.6× | 0.0614 | 0.90 |
| 10 | 24168 | 2529 | 10.46% | 0.909 | 0.555 | 5.3× | 0.0642 | 0.90 |

#### ESTABLISHED_MLB

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 45376 | 7 | 0.02% | 0.983 | 0.116 | 754.6× | 0.0002 | 0.71 |
| 2 | 43862 | 90 | 0.21% | 0.975 | 0.197 | 95.9× | 0.0019 | 0.52 |
| 3 | 42119 | 291 | 0.69% | 0.967 | 0.248 | 36.0× | 0.0059 | 0.62 |
| 4 | 40115 | 540 | 1.35% | 0.957 | 0.292 | 21.7× | 0.0111 | 0.64 |
| 5 | 37844 | 769 | 2.03% | 0.947 | 0.318 | 15.6× | 0.0163 | 0.69 |
| 6 | 35279 | 956 | 2.71% | 0.936 | 0.325 | 12.0× | 0.0215 | 0.74 |
| 7 | 32481 | 1088 | 3.35% | 0.928 | 0.335 | 10.0× | 0.0263 | 0.74 |
| 8 | 29722 | 1166 | 3.92% | 0.921 | 0.340 | 8.7× | 0.0306 | 0.74 |
| 9 | 26998 | 1182 | 4.38% | 0.916 | 0.341 | 7.8× | 0.0341 | 0.74 |
| 10 | 24168 | 1165 | 4.82% | 0.910 | 0.346 | 7.2× | 0.0374 | 0.73 |

#### STAR_PLUS_ELITE

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 45376 | 5 | 0.01% | 0.984 | 0.075 | 679.8× | 0.0001 | 0.57 |
| 2 | 43862 | 18 | 0.04% | 0.966 | 0.062 | 152.2× | 0.0004 | 0.36 |
| 3 | 42119 | 44 | 0.10% | 0.965 | 0.060 | 57.3× | 0.0010 | 0.47 |
| 4 | 40115 | 79 | 0.20% | 0.964 | 0.094 | 47.5× | 0.0019 | 0.58 |
| 5 | 37844 | 118 | 0.31% | 0.959 | 0.112 | 36.0× | 0.0029 | 0.70 |
| 6 | 35279 | 153 | 0.43% | 0.952 | 0.135 | 31.1× | 0.0040 | 0.77 |
| 7 | 32481 | 181 | 0.56% | 0.949 | 0.156 | 28.0× | 0.0051 | 0.81 |
| 8 | 29722 | 201 | 0.68% | 0.945 | 0.160 | 23.6× | 0.0061 | 0.80 |
| 9 | 26998 | 211 | 0.78% | 0.941 | 0.168 | 21.4× | 0.0071 | 0.78 |
| 10 | 24168 | 216 | 0.89% | 0.937 | 0.186 | 20.8× | 0.0080 | 0.73 |

## Per-bucket (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 35057 | 368 | 1.05% | 0.969 | 0.430 | 41.0× | 0.165 | 0.696 | 0.174 | 0.278 | 64 | 28 | 304 |
| R1 | 439 | 109 | 24.83% | 0.922 | 0.792 | 3.2× | 0.632 | 0.806 | 0.459 | 0.585 | 50 | 12 | 59 |
| R2-R3 | 1076 | 46 | 4.28% | 0.879 | 0.325 | 7.6× | 0.266 | 0.556 | 0.109 | 0.182 | 5 | 4 | 41 |
| R4-R10 | 4278 | 49 | 1.15% | 0.920 | 0.174 | 15.2× | 0.155 | 0.250 | 0.020 | 0.038 | 1 | 3 | 48 |
| R10+ | 13414 | 65 | 0.48% | 0.972 | 0.274 | 56.5× | 0.113 | 0.500 | 0.015 | 0.030 | 1 | 1 | 64 |
| IFA | 15850 | 99 | 0.62% | 0.959 | 0.270 | 43.2× | 0.125 | 0.467 | 0.071 | 0.123 | 7 | 8 | 92 |

#### MLB_DEBUT

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 35279 | 3063 | 8.68% | 0.923 | 0.567 | 6.5× | 0.413 | 0.748 | 0.248 | 0.372 | 759 | 256 | 2304 |
| R1 | 583 | 272 | 46.66% | 0.877 | 0.858 | 1.8× | 0.651 | 0.760 | 0.801 | 0.780 | 218 | 69 | 54 |
| R2-R3 | 1093 | 360 | 32.94% | 0.852 | 0.691 | 2.1× | 0.573 | 0.739 | 0.417 | 0.533 | 150 | 53 | 210 |
| R4-R10 | 4292 | 778 | 18.13% | 0.864 | 0.550 | 3.0× | 0.486 | 0.717 | 0.206 | 0.320 | 160 | 63 | 618 |
| R10+ | 13437 | 987 | 7.35% | 0.894 | 0.431 | 5.9× | 0.356 | 0.694 | 0.110 | 0.191 | 109 | 48 | 878 |
| IFA | 15874 | 666 | 4.20% | 0.933 | 0.509 | 12.1× | 0.301 | 0.841 | 0.183 | 0.301 | 122 | 23 | 544 |

#### ESTABLISHED_MLB

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 35279 | 956 | 2.71% | 0.936 | 0.325 | 12.0× | 0.245 | 0.650 | 0.027 | 0.052 | 26 | 14 | 930 |
| R1 | 583 | 127 | 21.78% | 0.849 | 0.548 | 2.5× | 0.499 | 0.696 | 0.126 | 0.213 | 16 | 7 | 111 |
| R2-R3 | 1093 | 120 | 10.98% | 0.851 | 0.339 | 3.1× | 0.380 | 0.429 | 0.025 | 0.047 | 3 | 4 | 117 |
| R4-R10 | 4292 | 241 | 5.62% | 0.863 | 0.285 | 5.1× | 0.289 | 0.667 | 0.008 | 0.016 | 2 | 1 | 239 |
| R10+ | 13437 | 274 | 2.04% | 0.913 | 0.216 | 10.6× | 0.202 | 0.500 | 0.004 | 0.007 | 1 | 1 | 273 |
| IFA | 15874 | 194 | 1.22% | 0.957 | 0.340 | 27.8× | 0.174 | 0.800 | 0.021 | 0.040 | 4 | 1 | 190 |

#### STAR_PLUS_ELITE

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 35279 | 153 | 0.43% | 0.952 | 0.135 | 31.1× | 0.103 | — | 0.000 | — | 0 | 0 | 153 |
| R1 | 583 | 40 | 6.86% | 0.871 | 0.307 | 4.5× | 0.325 | — | 0.000 | — | 0 | 0 | 40 |
| R2-R3 | 1093 | 18 | 1.65% | 0.817 | 0.131 | 8.0× | 0.140 | — | 0.000 | — | 0 | 0 | 18 |
| R4-R10 | 4292 | 40 | 0.93% | 0.928 | 0.101 | 10.9× | 0.142 | — | 0.000 | — | 0 | 0 | 40 |
| R10+ | 13437 | 29 | 0.22% | 0.895 | 0.023 | 10.7× | 0.063 | — | 0.000 | — | 0 | 0 | 29 |
| IFA | 15874 | 26 | 0.16% | 0.974 | 0.112 | 68.2× | 0.066 | — | 0.000 | — | 0 | 0 | 26 |

## Per-yip (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4933 | 151 | 3.06% | 0.926 | 0.524 | 17.1× | 0.254 | 0.800 | 0.265 | 0.398 | 40 | 10 | 111 |
| 1 | 4583 | 105 | 2.29% | 0.933 | 0.406 | 17.7× | 0.224 | 0.636 | 0.133 | 0.220 | 14 | 8 | 91 |
| 2 | 4168 | 69 | 1.66% | 0.944 | 0.345 | 20.8× | 0.196 | 0.500 | 0.087 | 0.148 | 6 | 6 | 63 |
| 3 | 3758 | 31 | 0.82% | 0.967 | 0.349 | 42.3× | 0.146 | 0.500 | 0.129 | 0.205 | 4 | 4 | 27 |
| 4 | 3382 | 9 | 0.27% | 0.984 | 0.394 | 148.0× | 0.086 | — | 0.000 | — | 0 | 0 | 9 |
| 5 | 3049 | 3 | 0.10% | 0.997 | 0.704 | 715.2× | 0.054 | — | 0.000 | — | 0 | 0 | 3 |
| 6 | 2732 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 7 | 2487 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 8 | 2230 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 9 | 1994 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 1741 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

#### MLB_DEBUT

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4936 | 674 | 13.65% | 0.830 | 0.493 | 3.6× | 0.393 | 0.713 | 0.169 | 0.273 | 114 | 46 | 560 |
| 1 | 4615 | 679 | 14.71% | 0.872 | 0.583 | 4.0× | 0.457 | 0.772 | 0.259 | 0.388 | 176 | 52 | 503 |
| 2 | 4220 | 611 | 14.48% | 0.903 | 0.641 | 4.4× | 0.492 | 0.763 | 0.342 | 0.472 | 209 | 65 | 402 |
| 3 | 3806 | 453 | 11.90% | 0.919 | 0.611 | 5.1× | 0.470 | 0.747 | 0.313 | 0.442 | 142 | 48 | 311 |
| 4 | 3412 | 285 | 8.35% | 0.934 | 0.596 | 7.1× | 0.416 | 0.720 | 0.253 | 0.374 | 72 | 28 | 213 |
| 5 | 3066 | 182 | 5.94% | 0.943 | 0.553 | 9.3× | 0.363 | 0.760 | 0.209 | 0.328 | 38 | 12 | 144 |
| 6 | 2742 | 91 | 3.32% | 0.944 | 0.382 | 11.5× | 0.276 | 0.600 | 0.066 | 0.119 | 6 | 4 | 85 |
| 7 | 2496 | 46 | 1.84% | 0.949 | 0.340 | 18.4× | 0.209 | 0.667 | 0.043 | 0.082 | 2 | 1 | 44 |
| 8 | 2238 | 25 | 1.12% | 0.938 | 0.324 | 29.0× | 0.159 | — | 0.000 | — | 0 | 0 | 25 |
| 9 | 2001 | 10 | 0.50% | 0.920 | 0.309 | 61.7× | 0.103 | — | 0.000 | — | 0 | 0 | 10 |
| 10 | 1747 | 7 | 0.40% | 0.895 | 0.502 | 125.3× | 0.086 | — | 0.000 | — | 0 | 0 | 7 |

#### ESTABLISHED_MLB

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4936 | 186 | 3.77% | 0.864 | 0.274 | 7.3× | 0.240 | 0.800 | 0.043 | 0.082 | 8 | 2 | 178 |
| 1 | 4615 | 212 | 4.59% | 0.894 | 0.328 | 7.1× | 0.286 | 0.462 | 0.028 | 0.053 | 6 | 7 | 206 |
| 2 | 4220 | 212 | 5.02% | 0.914 | 0.388 | 7.7× | 0.313 | 0.727 | 0.038 | 0.072 | 8 | 3 | 204 |
| 3 | 3806 | 151 | 3.97% | 0.927 | 0.365 | 9.2× | 0.289 | 0.667 | 0.026 | 0.051 | 4 | 2 | 147 |
| 4 | 3412 | 103 | 3.02% | 0.948 | 0.380 | 12.6× | 0.265 | — | 0.000 | — | 0 | 0 | 103 |
| 5 | 3066 | 57 | 1.86% | 0.961 | 0.362 | 19.5× | 0.216 | — | 0.000 | — | 0 | 0 | 57 |
| 6 | 2742 | 21 | 0.77% | 0.945 | 0.099 | 12.9× | 0.134 | — | 0.000 | — | 0 | 0 | 21 |
| 7 | 2496 | 11 | 0.44% | 0.958 | 0.071 | 16.0× | 0.105 | — | 0.000 | — | 0 | 0 | 11 |
| 8 | 2238 | 3 | 0.13% | 0.967 | 0.044 | 32.6× | 0.059 | — | 0.000 | — | 0 | 0 | 3 |
| 9 | 2001 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 1747 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

#### STAR_PLUS_ELITE

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4936 | 41 | 0.83% | 0.908 | 0.194 | 23.3× | 0.128 | — | 0.000 | — | 0 | 0 | 41 |
| 1 | 4615 | 39 | 0.85% | 0.923 | 0.113 | 13.3× | 0.134 | — | 0.000 | — | 0 | 0 | 39 |
| 2 | 4220 | 37 | 0.88% | 0.932 | 0.219 | 24.9× | 0.140 | — | 0.000 | — | 0 | 0 | 37 |
| 3 | 3806 | 21 | 0.55% | 0.929 | 0.122 | 22.1× | 0.110 | — | 0.000 | — | 0 | 0 | 21 |
| 4 | 3412 | 10 | 0.29% | 0.940 | 0.038 | 13.1× | 0.082 | — | 0.000 | — | 0 | 0 | 10 |
| 5 | 3066 | 5 | 0.16% | 0.970 | 0.031 | 19.0× | 0.066 | — | 0.000 | — | 0 | 0 | 5 |
| 6 | 2742 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 7 | 2496 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 8 | 2238 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 9 | 2001 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 1747 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

## Per-level (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 35057 | 368 | 1.05% | 0.969 | 0.430 | 41.0× | 0.165 | 0.696 | 0.174 | 0.278 | 64 | 28 | 304 |
| RK | 5654 | 76 | 1.34% | 0.932 | 0.359 | 26.7× | 0.172 | 0.867 | 0.171 | 0.286 | 13 | 2 | 63 |
| A- | 1451 | 26 | 1.79% | 0.921 | 0.459 | 25.6× | 0.193 | 0.667 | 0.231 | 0.343 | 6 | 3 | 20 |
| A | 2024 | 68 | 3.36% | 0.954 | 0.558 | 16.6× | 0.283 | 0.684 | 0.191 | 0.299 | 13 | 6 | 55 |
| A+ | 1953 | 39 | 2.00% | 0.956 | 0.435 | 21.8× | 0.221 | 0.556 | 0.128 | 0.208 | 5 | 4 | 34 |
| AA | 1675 | 33 | 1.97% | 0.984 | 0.572 | 29.0× | 0.233 | 0.667 | 0.364 | 0.471 | 12 | 6 | 21 |
| AAA | 2126 | 5 | 0.24% | 0.998 | 0.561 | 238.4× | 0.084 | 0.500 | 0.200 | 0.286 | 1 | 1 | 4 |
| NONE | 20163 | 121 | 0.60% | 0.979 | 0.372 | 62.0× | 0.128 | 0.700 | 0.116 | 0.199 | 14 | 6 | 107 |

#### MLB_DEBUT

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 35279 | 3063 | 8.68% | 0.923 | 0.567 | 6.5× | 0.413 | 0.748 | 0.248 | 0.372 | 759 | 256 | 2304 |
| RK | 5658 | 347 | 6.13% | 0.809 | 0.288 | 4.7× | 0.256 | 0.654 | 0.049 | 0.091 | 17 | 9 | 330 |
| A- | 1452 | 209 | 14.39% | 0.794 | 0.462 | 3.2× | 0.358 | 0.857 | 0.115 | 0.203 | 24 | 4 | 185 |
| A | 2038 | 382 | 18.74% | 0.825 | 0.587 | 3.1× | 0.439 | 0.788 | 0.215 | 0.337 | 82 | 22 | 300 |
| A+ | 1979 | 437 | 22.08% | 0.828 | 0.615 | 2.8× | 0.471 | 0.820 | 0.240 | 0.372 | 105 | 23 | 332 |
| AA | 1735 | 528 | 30.43% | 0.842 | 0.713 | 2.3× | 0.545 | 0.769 | 0.436 | 0.556 | 230 | 69 | 298 |
| AAA | 2177 | 423 | 19.43% | 0.896 | 0.682 | 3.5× | 0.543 | 0.798 | 0.374 | 0.509 | 158 | 40 | 265 |
| NONE | 20229 | 736 | 3.64% | 0.959 | 0.478 | 13.1× | 0.297 | 0.615 | 0.193 | 0.294 | 142 | 89 | 594 |

#### ESTABLISHED_MLB

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 35279 | 956 | 2.71% | 0.936 | 0.325 | 12.0× | 0.245 | 0.650 | 0.027 | 0.052 | 26 | 14 | 930 |
| RK | 5658 | 67 | 1.18% | 0.869 | 0.173 | 14.6× | 0.138 | — | 0.000 | — | 0 | 0 | 67 |
| A- | 1452 | 50 | 3.44% | 0.864 | 0.286 | 8.3× | 0.230 | 1.000 | 0.040 | 0.077 | 2 | 0 | 48 |
| A | 2038 | 110 | 5.40% | 0.861 | 0.349 | 6.5× | 0.283 | 1.000 | 0.027 | 0.053 | 3 | 0 | 107 |
| A+ | 1979 | 133 | 6.72% | 0.843 | 0.337 | 5.0× | 0.297 | 0.667 | 0.015 | 0.029 | 2 | 1 | 131 |
| AA | 1735 | 191 | 11.01% | 0.867 | 0.456 | 4.1× | 0.398 | 0.611 | 0.058 | 0.105 | 11 | 7 | 180 |
| AAA | 2177 | 141 | 6.48% | 0.867 | 0.391 | 6.0× | 0.313 | 0.500 | 0.021 | 0.041 | 3 | 3 | 138 |
| NONE | 20229 | 264 | 1.31% | 0.970 | 0.263 | 20.1× | 0.185 | 0.625 | 0.019 | 0.037 | 5 | 3 | 259 |

#### STAR_PLUS_ELITE

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 35279 | 153 | 0.43% | 0.952 | 0.135 | 31.1× | 0.103 | — | 0.000 | — | 0 | 0 | 153 |
| RK | 5658 | 8 | 0.14% | 0.963 | 0.226 | 159.6× | 0.060 | — | 0.000 | — | 0 | 0 | 8 |
| A- | 1452 | 8 | 0.55% | 0.902 | 0.188 | 34.2× | 0.103 | — | 0.000 | — | 0 | 0 | 8 |
| A | 2038 | 18 | 0.88% | 0.867 | 0.105 | 11.9× | 0.119 | — | 0.000 | — | 0 | 0 | 18 |
| A+ | 1979 | 21 | 1.06% | 0.860 | 0.187 | 17.6× | 0.128 | — | 0.000 | — | 0 | 0 | 21 |
| AA | 1735 | 31 | 1.79% | 0.914 | 0.227 | 12.7× | 0.190 | — | 0.000 | — | 0 | 0 | 31 |
| AAA | 2177 | 14 | 0.64% | 0.956 | 0.328 | 50.9× | 0.126 | — | 0.000 | — | 0 | 0 | 14 |
| NONE | 20229 | 53 | 0.26% | 0.973 | 0.088 | 33.5× | 0.084 | — | 0.000 | — | 0 | 0 | 53 |

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
python -m prospects.model.train.exp_cdf_timing5 --cal-min-snap-year 2008 \
    --out-dir runs/current/scratch/v23_build
python -m prospects.model.train.promote_v22 \
    --source runs/current/scratch/v23_build --version v2.3

# prod hazards + rescore the 2026 cohort
python -m prospects.model.train.hazards --force
python -m prospects.model.pipelines.prod --skip-xgb --skip-buylist

# validation — calibrated, headline at the publish horizon (h=6)
python -m prospects.evaluation.run --xgb runs/current/models/joint_xgb_v2.4.pkl \
    --calibrators runs/current/models/calibrators_v2.4.pkl --threshold 0.6 --eval-horizon 6
python -m prospects.evaluation.report

# buy list — P(debut <= 3y) thesis, CDF timing + debut window
python -m prospects.buylist.build --xgb runs/current/models/joint_xgb_v2.4.pkl \
    --calibrators runs/current/models/calibrators_v2.4.pkl --debut-horizon 3
```

The weekly retrain (`deploy/weekly_score.py`, ported 2026-09-05) now runs
Stage C = the v2.4 steps above automatically after stage_a + prod; the
Monday job produces the v2.3 buy list end-to-end.
