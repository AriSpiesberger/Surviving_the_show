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
| TOP_100_PROSPECT | 35125 | 1.10% | **0.478** | 43.6× | 0.964 | 0.167 | 0.738 | 0.197 | 0.311 |
| MLB_DEBUT | 35371 | 8.49% | **0.592** | 7.0× | 0.930 | 0.415 | 0.779 | 0.238 | 0.364 |
| ESTABLISHED_MLB | 35371 | 2.70% | **0.382** | 14.1× | 0.941 | 0.248 | 0.737 | 0.044 | 0.083 |
| STAR_PLUS_ELITE | 35371 | 0.43% | **0.192** | 44.4× | 0.952 | 0.103 | — | 0.000 | — |
| **weighted-AP** | | | **0.447** | | | | | | |

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
| 0-5% | 36,090 | 0.1% | 0.1% | +0.0% |
| 5-10% | 250 | 7.1% | 6.0% | -1.1% |
| 10-20% | 172 | 14.5% | 14.5% | +0.1% |
| 20-30% | 76 | 24.5% | 14.5% | -10.0% |
| 30-40% | 41 | 34.8% | 39.0% | +4.2% |
| 40-50% | 26 | 44.7% | 61.5% | +16.8% |
| 50-60% | 31 | 54.7% | 58.1% | +3.4% |
| 60-70% | 20 | 65.2% | 55.0% | -10.2% |
| 70-80% | 20 | 74.5% | 70.0% | -4.5% |
| 80-90% | 15 | 84.3% | 86.7% | +2.4% |

#### TOP_100_PROSPECT — P(within 6y)

| predicted | n | avg pred | actual | diff |
|---|---:|---:|---:|---:|
| 0-5% | 29,208 | 0.2% | 0.2% | +0.1% |
| 5-10% | 273 | 7.3% | 8.8% | +1.5% |
| 10-20% | 200 | 14.1% | 13.0% | -1.1% |
| 20-30% | 92 | 24.5% | 18.5% | -6.0% |
| 30-40% | 40 | 34.9% | 25.0% | -9.9% |
| 40-50% | 37 | 44.5% | 54.1% | +9.6% |
| 50-60% | 27 | 55.2% | 59.3% | +4.1% |
| 60-70% | 20 | 64.5% | 50.0% | -14.5% |
| 70-80% | 22 | 74.1% | 81.8% | +7.7% |
| 80-90% | 15 | 84.8% | 86.7% | +1.9% |

#### MLB_DEBUT — P(within 3y)

| predicted | n | avg pred | actual | diff |
|---|---:|---:|---:|---:|
| 0-5% | 31,923 | 0.4% | 0.6% | +0.1% |
| 5-10% | 1,592 | 7.2% | 7.9% | +0.6% |
| 10-20% | 1,304 | 14.2% | 17.9% | +3.7% |
| 20-30% | 749 | 24.6% | 26.2% | +1.6% |
| 30-40% | 466 | 34.3% | 35.8% | +1.5% |
| 40-50% | 279 | 44.6% | 54.8% | +10.3% |
| 50-60% | 184 | 55.1% | 61.4% | +6.3% |
| 60-70% | 159 | 64.8% | 69.2% | +4.4% |
| 70-80% | 136 | 74.8% | 73.5% | -1.2% |
| 80-90% | 145 | 84.6% | 85.5% | +0.9% |
| 90-100% | 21 | 91.9% | 95.2% | +3.4% |

#### MLB_DEBUT — P(within 6y)

| predicted | n | avg pred | actual | diff |
|---|---:|---:|---:|---:|
| 0-5% | 23,530 | 0.6% | 0.8% | +0.2% |
| 5-10% | 2,023 | 7.2% | 8.4% | +1.2% |
| 10-20% | 1,780 | 14.1% | 17.2% | +3.2% |
| 20-30% | 819 | 24.6% | 29.4% | +4.8% |
| 30-40% | 591 | 34.6% | 36.0% | +1.4% |
| 40-50% | 432 | 44.6% | 45.4% | +0.8% |
| 50-60% | 288 | 54.7% | 62.5% | +7.8% |
| 60-70% | 206 | 65.0% | 63.6% | -1.4% |
| 70-80% | 200 | 74.7% | 82.0% | +7.3% |
| 80-90% | 198 | 84.6% | 86.4% | +1.8% |
| 90-100% | 34 | 91.8% | 91.2% | -0.6% |

#### ESTABLISHED_MLB — P(within 3y)

| predicted | n | avg pred | actual | diff |
|---|---:|---:|---:|---:|
| 0-5% | 36,384 | 0.1% | 0.2% | +0.1% |
| 5-10% | 246 | 7.0% | 8.9% | +1.9% |
| 10-20% | 181 | 13.9% | 18.8% | +4.9% |
| 20-30% | 68 | 24.7% | 30.9% | +6.2% |
| 30-40% | 45 | 34.3% | 40.0% | +5.7% |
| 40-50% | 21 | 44.8% | 57.1% | +12.3% |

#### ESTABLISHED_MLB — P(within 6y)

| predicted | n | avg pred | actual | diff |
|---|---:|---:|---:|---:|
| 0-5% | 28,184 | 0.4% | 0.5% | +0.2% |
| 5-10% | 822 | 7.2% | 8.8% | +1.6% |
| 10-20% | 580 | 13.9% | 17.2% | +3.4% |
| 20-30% | 227 | 24.0% | 35.7% | +11.7% |
| 30-40% | 132 | 34.7% | 42.4% | +7.8% |
| 40-50% | 83 | 44.8% | 45.8% | +0.9% |
| 50-60% | 45 | 55.3% | 73.3% | +18.0% |
| 60-70% | 18 | 64.0% | 66.7% | +2.7% |

#### STAR_PLUS_ELITE — P(within 3y)

| predicted | n | avg pred | actual | diff |
|---|---:|---:|---:|---:|
| 0-5% | 36,926 | 0.0% | 0.0% | +0.0% |
| 5-10% | 26 | 7.4% | 7.7% | +0.2% |

#### STAR_PLUS_ELITE — P(within 6y)

| predicted | n | avg pred | actual | diff |
|---|---:|---:|---:|---:|
| 0-5% | 29,802 | 0.1% | 0.1% | -0.0% |
| 5-10% | 187 | 7.0% | 9.6% | +2.6% |
| 10-20% | 93 | 13.9% | 19.4% | +5.4% |
| 20-30% | 19 | 23.3% | 47.4% | +24.0% |

## Per-horizon trajectory (h=1..10, resolved at each h)

#### TOP_100_PROSPECT

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 45183 | 143 | 0.32% | 0.995 | 0.590 | 186.3× | 0.0019 | 0.92 |
| 2 | 43670 | 258 | 0.59% | 0.984 | 0.542 | 91.8× | 0.0037 | 0.89 |
| 3 | 41932 | 335 | 0.80% | 0.975 | 0.514 | 64.3× | 0.0052 | 0.88 |
| 4 | 39936 | 376 | 0.94% | 0.967 | 0.496 | 52.6× | 0.0063 | 0.87 |
| 5 | 37677 | 385 | 1.02% | 0.965 | 0.485 | 47.4× | 0.0070 | 0.88 |
| 6 | 35125 | 385 | 1.10% | 0.964 | 0.478 | 43.6× | 0.0075 | 0.88 |
| 7 | 32338 | 370 | 1.14% | 0.964 | 0.483 | 42.2× | 0.0078 | 0.87 |
| 8 | 29588 | 352 | 1.19% | 0.964 | 0.485 | 40.8× | 0.0081 | 0.87 |
| 9 | 26871 | 326 | 1.21% | 0.963 | 0.478 | 39.4× | 0.0084 | 0.87 |
| 10 | 24043 | 301 | 1.25% | 0.963 | 0.475 | 37.9× | 0.0087 | 0.87 |

#### MLB_DEBUT

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 45492 | 782 | 1.72% | 0.970 | 0.422 | 24.6× | 0.0124 | 0.86 |
| 2 | 43974 | 1531 | 3.48% | 0.959 | 0.520 | 14.9× | 0.0226 | 0.84 |
| 3 | 42228 | 2183 | 5.17% | 0.948 | 0.560 | 10.8× | 0.0320 | 0.86 |
| 4 | 40219 | 2659 | 6.61% | 0.939 | 0.576 | 8.7× | 0.0400 | 0.86 |
| 5 | 37943 | 2925 | 7.71% | 0.932 | 0.584 | 7.6× | 0.0462 | 0.86 |
| 6 | 35371 | 3002 | 8.49% | 0.930 | 0.592 | 7.0× | 0.0502 | 0.86 |
| 7 | 32566 | 2940 | 9.03% | 0.928 | 0.595 | 6.6× | 0.0532 | 0.85 |
| 8 | 29799 | 2809 | 9.43% | 0.925 | 0.593 | 6.3× | 0.0558 | 0.84 |
| 9 | 27069 | 2639 | 9.75% | 0.922 | 0.593 | 6.1× | 0.0578 | 0.83 |
| 10 | 24231 | 2456 | 10.14% | 0.919 | 0.590 | 5.8× | 0.0603 | 0.83 |

#### ESTABLISHED_MLB

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 45492 | 7 | 0.02% | 0.990 | 0.129 | 836.0× | 0.0001 | 1.42 |
| 2 | 43974 | 90 | 0.20% | 0.984 | 0.275 | 134.4× | 0.0017 | 0.68 |
| 3 | 42228 | 291 | 0.69% | 0.975 | 0.299 | 43.4× | 0.0056 | 0.70 |
| 4 | 40219 | 540 | 1.34% | 0.963 | 0.350 | 26.1× | 0.0105 | 0.68 |
| 5 | 37943 | 769 | 2.03% | 0.953 | 0.372 | 18.3× | 0.0156 | 0.70 |
| 6 | 35371 | 956 | 2.70% | 0.941 | 0.382 | 14.1× | 0.0206 | 0.72 |
| 7 | 32566 | 1088 | 3.34% | 0.932 | 0.387 | 11.6× | 0.0253 | 0.71 |
| 8 | 29799 | 1166 | 3.91% | 0.926 | 0.394 | 10.1× | 0.0294 | 0.71 |
| 9 | 27069 | 1182 | 4.37% | 0.921 | 0.396 | 9.1× | 0.0328 | 0.71 |
| 10 | 24231 | 1165 | 4.81% | 0.915 | 0.400 | 8.3× | 0.0360 | 0.70 |

#### STAR_PLUS_ELITE

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 45492 | 5 | 0.01% | 0.989 | 0.328 | 2984.3× | 0.0001 | 0.59 |
| 2 | 43974 | 18 | 0.04% | 0.965 | 0.136 | 331.2× | 0.0004 | 0.53 |
| 3 | 42228 | 44 | 0.10% | 0.963 | 0.120 | 115.5× | 0.0010 | 0.58 |
| 4 | 40219 | 79 | 0.20% | 0.958 | 0.168 | 85.5× | 0.0018 | 0.67 |
| 5 | 37943 | 118 | 0.31% | 0.955 | 0.191 | 61.5× | 0.0028 | 0.74 |
| 6 | 35371 | 153 | 0.43% | 0.952 | 0.192 | 44.4× | 0.0039 | 0.79 |
| 7 | 32566 | 181 | 0.56% | 0.946 | 0.196 | 35.2× | 0.0049 | 0.81 |
| 8 | 29799 | 201 | 0.67% | 0.944 | 0.196 | 29.0× | 0.0060 | 0.78 |
| 9 | 27069 | 211 | 0.78% | 0.940 | 0.204 | 26.1× | 0.0069 | 0.74 |
| 10 | 24231 | 216 | 0.89% | 0.936 | 0.217 | 24.3× | 0.0078 | 0.69 |

## Per-bucket (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 35125 | 385 | 1.10% | 0.964 | 0.478 | 43.6× | 0.167 | 0.738 | 0.197 | 0.311 | 76 | 27 | 309 |
| R1 | 439 | 109 | 24.83% | 0.942 | 0.832 | 3.4× | 0.662 | 0.800 | 0.514 | 0.626 | 56 | 14 | 53 |
| R2-R3 | 1076 | 46 | 4.28% | 0.900 | 0.384 | 9.0× | 0.280 | 0.833 | 0.109 | 0.192 | 5 | 1 | 41 |
| R4-R10 | 4275 | 51 | 1.19% | 0.941 | 0.224 | 18.8× | 0.166 | 0.300 | 0.059 | 0.098 | 3 | 7 | 48 |
| R10+ | 13485 | 73 | 0.54% | 0.948 | 0.319 | 59.0× | 0.114 | 1.000 | 0.027 | 0.053 | 2 | 0 | 71 |
| IFA | 15850 | 106 | 0.67% | 0.954 | 0.331 | 49.5× | 0.128 | 0.667 | 0.094 | 0.165 | 10 | 5 | 96 |

#### MLB_DEBUT

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 35371 | 3002 | 8.49% | 0.930 | 0.592 | 7.0× | 0.415 | 0.779 | 0.238 | 0.364 | 713 | 202 | 2289 |
| R1 | 583 | 272 | 46.66% | 0.899 | 0.877 | 1.9× | 0.690 | 0.845 | 0.724 | 0.780 | 197 | 36 | 75 |
| R2-R3 | 1093 | 360 | 32.94% | 0.864 | 0.730 | 2.2× | 0.592 | 0.778 | 0.389 | 0.519 | 140 | 40 | 220 |
| R4-R10 | 4294 | 772 | 17.98% | 0.879 | 0.593 | 3.3× | 0.504 | 0.736 | 0.220 | 0.339 | 170 | 61 | 602 |
| R10+ | 13509 | 951 | 7.04% | 0.909 | 0.460 | 6.5× | 0.362 | 0.734 | 0.107 | 0.187 | 102 | 37 | 849 |
| IFA | 15892 | 647 | 4.07% | 0.934 | 0.493 | 12.1× | 0.297 | 0.788 | 0.161 | 0.267 | 104 | 28 | 543 |

#### ESTABLISHED_MLB

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 35371 | 956 | 2.70% | 0.941 | 0.382 | 14.1× | 0.248 | 0.737 | 0.044 | 0.083 | 42 | 15 | 914 |
| R1 | 583 | 127 | 21.78% | 0.891 | 0.655 | 3.0× | 0.559 | 0.750 | 0.213 | 0.331 | 27 | 9 | 100 |
| R2-R3 | 1093 | 120 | 10.98% | 0.854 | 0.387 | 3.5× | 0.383 | 0.429 | 0.025 | 0.047 | 3 | 4 | 117 |
| R4-R10 | 4294 | 241 | 5.61% | 0.886 | 0.348 | 6.2× | 0.308 | 1.000 | 0.017 | 0.033 | 4 | 0 | 237 |
| R10+ | 13509 | 274 | 2.03% | 0.921 | 0.297 | 14.6× | 0.206 | 0.833 | 0.018 | 0.036 | 5 | 1 | 269 |
| IFA | 15892 | 194 | 1.22% | 0.958 | 0.334 | 27.4× | 0.174 | 0.750 | 0.015 | 0.030 | 3 | 1 | 191 |

#### STAR_PLUS_ELITE

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 35371 | 153 | 0.43% | 0.952 | 0.192 | 44.4× | 0.103 | — | 0.000 | — | 0 | 0 | 153 |
| R1 | 583 | 40 | 6.86% | 0.905 | 0.355 | 5.2× | 0.354 | — | 0.000 | — | 0 | 0 | 40 |
| R2-R3 | 1093 | 18 | 1.65% | 0.912 | 0.297 | 18.0× | 0.182 | — | 0.000 | — | 0 | 0 | 18 |
| R4-R10 | 4294 | 40 | 0.93% | 0.924 | 0.092 | 9.9× | 0.141 | — | 0.000 | — | 0 | 0 | 40 |
| R10+ | 13509 | 29 | 0.21% | 0.863 | 0.142 | 66.3× | 0.058 | — | 0.000 | — | 0 | 0 | 29 |
| IFA | 15892 | 26 | 0.16% | 0.979 | 0.131 | 79.8× | 0.067 | — | 0.000 | — | 0 | 0 | 26 |

## Per-yip (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4938 | 158 | 3.20% | 0.903 | 0.514 | 16.1× | 0.246 | 0.778 | 0.266 | 0.396 | 42 | 12 | 116 |
| 1 | 4589 | 112 | 2.44% | 0.933 | 0.493 | 20.2× | 0.232 | 0.704 | 0.170 | 0.273 | 19 | 8 | 93 |
| 2 | 4171 | 71 | 1.70% | 0.945 | 0.471 | 27.6× | 0.199 | 0.688 | 0.155 | 0.253 | 11 | 5 | 60 |
| 3 | 3764 | 32 | 0.85% | 0.965 | 0.360 | 42.3× | 0.148 | 0.667 | 0.125 | 0.211 | 4 | 2 | 28 |
| 4 | 3387 | 9 | 0.27% | 0.982 | 0.358 | 134.5× | 0.086 | — | 0.000 | — | 0 | 0 | 9 |
| 5 | 3055 | 3 | 0.10% | 0.996 | 0.689 | 702.0× | 0.054 | — | 0.000 | — | 0 | 0 | 3 |
| 6 | 2738 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 7 | 2496 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 8 | 2238 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 9 | 2002 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 1747 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

#### MLB_DEBUT

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4941 | 668 | 13.52% | 0.826 | 0.495 | 3.7× | 0.386 | 0.744 | 0.144 | 0.241 | 96 | 33 | 572 |
| 1 | 4622 | 674 | 14.58% | 0.877 | 0.602 | 4.1× | 0.461 | 0.804 | 0.255 | 0.387 | 172 | 42 | 502 |
| 2 | 4227 | 604 | 14.29% | 0.911 | 0.663 | 4.6× | 0.499 | 0.777 | 0.329 | 0.463 | 199 | 57 | 405 |
| 3 | 3814 | 446 | 11.69% | 0.929 | 0.638 | 5.5× | 0.477 | 0.770 | 0.307 | 0.439 | 137 | 41 | 309 |
| 4 | 3420 | 279 | 8.16% | 0.941 | 0.639 | 7.8× | 0.419 | 0.787 | 0.251 | 0.380 | 70 | 19 | 209 |
| 5 | 3075 | 176 | 5.72% | 0.950 | 0.594 | 10.4× | 0.363 | 0.780 | 0.182 | 0.295 | 32 | 9 | 144 |
| 6 | 2751 | 85 | 3.09% | 0.961 | 0.502 | 16.3× | 0.276 | 0.875 | 0.082 | 0.151 | 7 | 1 | 78 |
| 7 | 2507 | 41 | 1.64% | 0.975 | 0.445 | 27.2× | 0.209 | — | 0.000 | — | 0 | 0 | 41 |
| 8 | 2248 | 20 | 0.89% | 0.976 | 0.305 | 34.3× | 0.155 | — | 0.000 | — | 0 | 0 | 20 |
| 9 | 2011 | 6 | 0.30% | 0.995 | 0.369 | 123.7× | 0.093 | — | 0.000 | — | 0 | 0 | 6 |
| 10 | 1755 | 3 | 0.17% | 0.995 | 0.700 | 409.5× | 0.071 | — | 0.000 | — | 0 | 0 | 3 |

#### ESTABLISHED_MLB

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4941 | 186 | 3.76% | 0.871 | 0.307 | 8.2× | 0.244 | 0.667 | 0.032 | 0.062 | 6 | 3 | 180 |
| 1 | 4622 | 212 | 4.59% | 0.901 | 0.405 | 8.8× | 0.291 | 0.750 | 0.071 | 0.129 | 15 | 5 | 197 |
| 2 | 4227 | 212 | 5.02% | 0.919 | 0.452 | 9.0× | 0.317 | 0.765 | 0.061 | 0.114 | 13 | 4 | 199 |
| 3 | 3814 | 151 | 3.96% | 0.929 | 0.406 | 10.3× | 0.290 | 0.625 | 0.033 | 0.063 | 5 | 3 | 146 |
| 4 | 3420 | 103 | 3.01% | 0.951 | 0.427 | 14.2× | 0.267 | 1.000 | 0.029 | 0.057 | 3 | 0 | 100 |
| 5 | 3075 | 57 | 1.85% | 0.969 | 0.446 | 24.1× | 0.219 | — | 0.000 | — | 0 | 0 | 57 |
| 6 | 2751 | 21 | 0.76% | 0.958 | 0.244 | 32.0× | 0.138 | — | 0.000 | — | 0 | 0 | 21 |
| 7 | 2507 | 11 | 0.44% | 0.969 | 0.097 | 22.0× | 0.107 | — | 0.000 | — | 0 | 0 | 11 |
| 8 | 2248 | 3 | 0.13% | 0.975 | 0.040 | 30.2× | 0.060 | — | 0.000 | — | 0 | 0 | 3 |
| 9 | 2011 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 1755 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

#### STAR_PLUS_ELITE

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4941 | 41 | 0.83% | 0.892 | 0.180 | 21.7× | 0.123 | — | 0.000 | — | 0 | 0 | 41 |
| 1 | 4622 | 39 | 0.84% | 0.932 | 0.247 | 29.3× | 0.137 | — | 0.000 | — | 0 | 0 | 39 |
| 2 | 4227 | 37 | 0.88% | 0.938 | 0.247 | 28.2× | 0.141 | — | 0.000 | — | 0 | 0 | 37 |
| 3 | 3814 | 21 | 0.55% | 0.936 | 0.203 | 36.9× | 0.112 | — | 0.000 | — | 0 | 0 | 21 |
| 4 | 3420 | 10 | 0.29% | 0.951 | 0.109 | 37.2× | 0.084 | — | 0.000 | — | 0 | 0 | 10 |
| 5 | 3075 | 5 | 0.16% | 0.973 | 0.042 | 26.0× | 0.066 | — | 0.000 | — | 0 | 0 | 5 |
| 6 | 2751 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 7 | 2507 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 8 | 2248 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 9 | 2011 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 1755 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

## Per-level (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 35125 | 385 | 1.10% | 0.964 | 0.478 | 43.6× | 0.167 | 0.738 | 0.197 | 0.311 | 76 | 27 | 309 |
| RK | 5926 | 101 | 1.70% | 0.929 | 0.381 | 22.4× | 0.192 | 0.792 | 0.188 | 0.304 | 19 | 5 | 82 |
| A- | 1679 | 39 | 2.32% | 0.905 | 0.442 | 19.0× | 0.211 | 0.700 | 0.179 | 0.286 | 7 | 3 | 32 |
| A | 2305 | 91 | 3.95% | 0.946 | 0.508 | 12.9× | 0.301 | 0.609 | 0.154 | 0.246 | 14 | 9 | 77 |
| A+ | 2214 | 59 | 2.66% | 0.961 | 0.535 | 20.1× | 0.257 | 0.750 | 0.203 | 0.320 | 12 | 4 | 47 |
| AA | 1912 | 50 | 2.62% | 0.978 | 0.660 | 25.2× | 0.264 | 0.789 | 0.300 | 0.435 | 15 | 4 | 35 |
| AAA | 2254 | 8 | 0.35% | 0.989 | 0.429 | 120.8× | 0.101 | 1.000 | 0.125 | 0.222 | 1 | 0 | 7 |
| NONE | 18785 | 36 | 0.19% | 0.946 | 0.354 | 184.8× | 0.068 | 0.778 | 0.194 | 0.311 | 7 | 2 | 29 |

#### MLB_DEBUT

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 35371 | 3002 | 8.49% | 0.930 | 0.592 | 7.0× | 0.415 | 0.779 | 0.238 | 0.364 | 713 | 202 | 2289 |
| RK | 5931 | 411 | 6.93% | 0.825 | 0.317 | 4.6× | 0.286 | 0.618 | 0.051 | 0.094 | 21 | 13 | 390 |
| A- | 1680 | 272 | 16.19% | 0.793 | 0.500 | 3.1× | 0.374 | 0.821 | 0.085 | 0.153 | 23 | 5 | 249 |
| A | 2322 | 465 | 20.03% | 0.818 | 0.570 | 2.8× | 0.440 | 0.742 | 0.204 | 0.320 | 95 | 33 | 370 |
| A+ | 2246 | 538 | 23.95% | 0.825 | 0.635 | 2.6× | 0.480 | 0.818 | 0.234 | 0.364 | 126 | 28 | 412 |
| AA | 1992 | 631 | 31.68% | 0.831 | 0.709 | 2.2× | 0.533 | 0.775 | 0.410 | 0.537 | 259 | 75 | 372 |
| AAA | 2316 | 472 | 20.38% | 0.899 | 0.699 | 3.4× | 0.557 | 0.800 | 0.347 | 0.484 | 164 | 41 | 308 |
| NONE | 18826 | 212 | 1.13% | 0.970 | 0.487 | 43.3× | 0.172 | 0.828 | 0.113 | 0.199 | 24 | 5 | 188 |

#### ESTABLISHED_MLB

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 35371 | 956 | 2.70% | 0.941 | 0.382 | 14.1× | 0.248 | 0.737 | 0.044 | 0.083 | 42 | 15 | 914 |
| RK | 5931 | 87 | 1.47% | 0.868 | 0.207 | 14.1× | 0.153 | 1.000 | 0.011 | 0.023 | 1 | 0 | 86 |
| A- | 1680 | 69 | 4.11% | 0.854 | 0.252 | 6.1× | 0.243 | 1.000 | 0.014 | 0.029 | 1 | 0 | 68 |
| A | 2322 | 142 | 6.12% | 0.865 | 0.367 | 6.0× | 0.303 | 1.000 | 0.035 | 0.068 | 5 | 0 | 137 |
| A+ | 2246 | 177 | 7.88% | 0.843 | 0.398 | 5.0× | 0.320 | 0.800 | 0.045 | 0.086 | 8 | 2 | 169 |
| AA | 1992 | 242 | 12.15% | 0.857 | 0.476 | 3.9× | 0.404 | 0.714 | 0.083 | 0.148 | 20 | 8 | 222 |
| AAA | 2316 | 164 | 7.08% | 0.883 | 0.439 | 6.2× | 0.340 | 0.714 | 0.030 | 0.058 | 5 | 2 | 159 |
| NONE | 18826 | 75 | 0.40% | 0.980 | 0.292 | 73.2× | 0.105 | 0.400 | 0.027 | 0.050 | 2 | 3 | 73 |

#### STAR_PLUS_ELITE

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 35371 | 153 | 0.43% | 0.952 | 0.192 | 44.4× | 0.103 | — | 0.000 | — | 0 | 0 | 153 |
| RK | 5931 | 11 | 0.19% | 0.923 | 0.100 | 54.1× | 0.063 | — | 0.000 | — | 0 | 0 | 11 |
| A- | 1680 | 14 | 0.83% | 0.900 | 0.094 | 11.3× | 0.126 | — | 0.000 | — | 0 | 0 | 14 |
| A | 2322 | 24 | 1.03% | 0.892 | 0.149 | 14.4× | 0.137 | — | 0.000 | — | 0 | 0 | 24 |
| A+ | 2246 | 29 | 1.29% | 0.889 | 0.192 | 14.8× | 0.152 | — | 0.000 | — | 0 | 0 | 29 |
| AA | 1992 | 44 | 2.21% | 0.903 | 0.268 | 12.1× | 0.205 | — | 0.000 | — | 0 | 0 | 44 |
| AAA | 2316 | 17 | 0.73% | 0.967 | 0.346 | 47.2× | 0.138 | — | 0.000 | — | 0 | 0 | 17 |
| NONE | 18826 | 14 | 0.07% | 0.930 | 0.299 | 402.3× | 0.041 | — | 0.000 | — | 0 | 0 | 14 |

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
