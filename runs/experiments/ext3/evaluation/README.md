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
| Hazards (per-fold OOF, eval) | `runs/experiments/hz0_default/scratch/oof/fold[0-5]_hazards.pkl` | Each fold trained on the OTHER 5 (val pids excluded, partition verified). HistGBT, default HP (capacity retune measured NEUTRAL on the clean split), 325 features + the landmark offset k. Survival → censoring-aware. |
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
| TOP_100_PROSPECT | 35144 | 1.10% | **0.487** | 44.2× | 0.969 | 0.170 | 0.812 | 0.202 | 0.323 |
| MLB_DEBUT | 35392 | 8.53% | **0.591** | 6.9× | 0.930 | 0.416 | 0.774 | 0.253 | 0.382 |
| ESTABLISHED_MLB | 35392 | 2.72% | **0.363** | 13.3× | 0.936 | 0.246 | 0.743 | 0.027 | 0.052 |
| STAR_PLUS_ELITE | 35392 | 0.43% | **0.139** | 32.2× | 0.949 | 0.102 | — | 0.000 | — |
| **weighted-AP** | | | **0.434** | | | | | | |

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
| 0-5% | 36,054 | 0.1% | 0.1% | +0.0% |
| 5-10% | 234 | 7.2% | 6.4% | -0.8% |
| 10-20% | 195 | 14.0% | 13.3% | -0.7% |
| 20-30% | 81 | 24.2% | 21.0% | -3.2% |
| 30-40% | 42 | 35.4% | 40.5% | +5.1% |
| 40-50% | 41 | 44.8% | 36.6% | -8.2% |
| 50-60% | 27 | 54.8% | 40.7% | -14.1% |
| 60-70% | 18 | 65.5% | 83.3% | +17.9% |
| 70-80% | 17 | 74.2% | 70.6% | -3.6% |

#### TOP_100_PROSPECT — P(within 6y)

| predicted | n | avg pred | actual | diff |
|---|---:|---:|---:|---:|
| 0-5% | 29,156 | 0.2% | 0.2% | +0.0% |
| 5-10% | 285 | 7.1% | 7.4% | +0.3% |
| 10-20% | 214 | 14.2% | 14.0% | -0.2% |
| 20-30% | 93 | 24.9% | 19.4% | -5.6% |
| 30-40% | 44 | 34.2% | 36.4% | +2.1% |
| 40-50% | 47 | 44.7% | 29.8% | -14.9% |
| 50-60% | 30 | 54.9% | 53.3% | -1.5% |
| 60-70% | 17 | 66.4% | 82.4% | +16.0% |
| 70-80% | 15 | 73.6% | 80.0% | +6.4% |
| 80-90% | 15 | 84.3% | 86.7% | +2.4% |

#### MLB_DEBUT — P(within 3y)

| predicted | n | avg pred | actual | diff |
|---|---:|---:|---:|---:|
| 0-5% | 31,831 | 0.4% | 0.6% | +0.2% |
| 5-10% | 1,524 | 7.2% | 7.3% | +0.1% |
| 10-20% | 1,438 | 14.3% | 16.8% | +2.5% |
| 20-30% | 679 | 24.9% | 27.2% | +2.3% |
| 30-40% | 476 | 34.6% | 33.6% | -1.0% |
| 40-50% | 320 | 44.5% | 46.6% | +2.1% |
| 50-60% | 201 | 54.7% | 66.2% | +11.5% |
| 60-70% | 172 | 64.7% | 66.9% | +2.2% |
| 70-80% | 121 | 74.9% | 75.2% | +0.3% |
| 80-90% | 149 | 85.1% | 84.6% | -0.5% |
| 90-100% | 31 | 92.0% | 93.5% | +1.6% |

#### MLB_DEBUT — P(within 6y)

| predicted | n | avg pred | actual | diff |
|---|---:|---:|---:|---:|
| 0-5% | 23,211 | 0.6% | 0.7% | +0.1% |
| 5-10% | 2,076 | 7.2% | 7.9% | +0.7% |
| 10-20% | 1,886 | 14.3% | 15.7% | +1.4% |
| 20-30% | 903 | 24.6% | 27.6% | +3.0% |
| 30-40% | 592 | 34.8% | 35.3% | +0.5% |
| 40-50% | 414 | 44.5% | 44.0% | -0.5% |
| 50-60% | 309 | 54.7% | 57.6% | +2.9% |
| 60-70% | 251 | 64.6% | 67.3% | +2.7% |
| 70-80% | 211 | 74.7% | 75.8% | +1.2% |
| 80-90% | 183 | 85.2% | 92.3% | +7.1% |
| 90-100% | 49 | 91.9% | 85.7% | -6.2% |

#### ESTABLISHED_MLB — P(within 3y)

| predicted | n | avg pred | actual | diff |
|---|---:|---:|---:|---:|
| 0-5% | 36,354 | 0.1% | 0.2% | +0.0% |
| 5-10% | 276 | 7.0% | 12.3% | +5.3% |
| 10-20% | 175 | 13.3% | 15.4% | +2.1% |
| 20-30% | 66 | 24.3% | 34.8% | +10.5% |
| 30-40% | 49 | 33.6% | 36.7% | +3.1% |
| 40-50% | 19 | 45.6% | 63.2% | +17.6% |

#### ESTABLISHED_MLB — P(within 6y)

| predicted | n | avg pred | actual | diff |
|---|---:|---:|---:|---:|
| 0-5% | 28,009 | 0.4% | 0.5% | +0.1% |
| 5-10% | 918 | 7.1% | 7.7% | +0.7% |
| 10-20% | 643 | 13.9% | 19.4% | +5.5% |
| 20-30% | 243 | 24.5% | 30.9% | +6.4% |
| 30-40% | 132 | 34.5% | 47.0% | +12.4% |
| 40-50% | 82 | 44.8% | 47.6% | +2.8% |
| 50-60% | 42 | 54.5% | 61.9% | +7.4% |

#### STAR_PLUS_ELITE — P(within 3y)

| predicted | n | avg pred | actual | diff |
|---|---:|---:|---:|---:|
| 0-5% | 36,909 | 0.0% | 0.0% | +0.0% |
| 5-10% | 30 | 6.5% | 10.0% | +3.5% |

#### STAR_PLUS_ELITE — P(within 6y)

| predicted | n | avg pred | actual | diff |
|---|---:|---:|---:|---:|
| 0-5% | 29,777 | 0.2% | 0.1% | -0.0% |
| 5-10% | 195 | 7.0% | 10.3% | +3.2% |
| 10-20% | 98 | 13.4% | 17.3% | +4.0% |

## Per-horizon trajectory (h=1..10, resolved at each h)

#### TOP_100_PROSPECT

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 45202 | 144 | 0.32% | 0.995 | 0.614 | 192.9× | 0.0018 | 0.97 |
| 2 | 43689 | 260 | 0.60% | 0.985 | 0.546 | 91.7× | 0.0038 | 0.93 |
| 3 | 41951 | 337 | 0.80% | 0.977 | 0.521 | 64.9× | 0.0053 | 0.91 |
| 4 | 39955 | 378 | 0.95% | 0.972 | 0.502 | 53.0× | 0.0063 | 0.90 |
| 5 | 37696 | 387 | 1.03% | 0.969 | 0.492 | 47.9× | 0.0070 | 0.91 |
| 6 | 35144 | 387 | 1.10% | 0.969 | 0.487 | 44.2× | 0.0075 | 0.91 |
| 7 | 32357 | 372 | 1.15% | 0.969 | 0.493 | 42.9× | 0.0078 | 0.90 |
| 8 | 29607 | 354 | 1.20% | 0.969 | 0.494 | 41.3× | 0.0081 | 0.90 |
| 9 | 26890 | 328 | 1.22% | 0.969 | 0.488 | 40.0× | 0.0083 | 0.91 |
| 10 | 24062 | 303 | 1.26% | 0.969 | 0.482 | 38.3× | 0.0087 | 0.92 |

#### MLB_DEBUT

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 45513 | 785 | 1.72% | 0.971 | 0.424 | 24.6× | 0.0124 | 0.89 |
| 2 | 43995 | 1537 | 3.49% | 0.960 | 0.525 | 15.0× | 0.0225 | 0.87 |
| 3 | 42249 | 2192 | 5.19% | 0.948 | 0.558 | 10.8× | 0.0320 | 0.87 |
| 4 | 40240 | 2671 | 6.64% | 0.940 | 0.575 | 8.7× | 0.0401 | 0.88 |
| 5 | 37964 | 2939 | 7.74% | 0.932 | 0.583 | 7.5× | 0.0463 | 0.88 |
| 6 | 35392 | 3018 | 8.53% | 0.930 | 0.591 | 6.9× | 0.0504 | 0.89 |
| 7 | 32587 | 2958 | 9.08% | 0.927 | 0.593 | 6.5× | 0.0535 | 0.88 |
| 8 | 29820 | 2828 | 9.48% | 0.924 | 0.591 | 6.2× | 0.0561 | 0.87 |
| 9 | 27090 | 2659 | 9.82% | 0.922 | 0.590 | 6.0× | 0.0581 | 0.87 |
| 10 | 24252 | 2477 | 10.21% | 0.918 | 0.587 | 5.8× | 0.0607 | 0.86 |

#### ESTABLISHED_MLB

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 45513 | 7 | 0.02% | 0.993 | 0.041 | 264.5× | 0.0002 | 1.20 |
| 2 | 43995 | 90 | 0.20% | 0.977 | 0.231 | 112.8× | 0.0018 | 0.63 |
| 3 | 42249 | 292 | 0.69% | 0.971 | 0.288 | 41.7× | 0.0057 | 0.67 |
| 4 | 40240 | 543 | 1.35% | 0.959 | 0.331 | 24.5× | 0.0108 | 0.67 |
| 5 | 37964 | 774 | 2.04% | 0.949 | 0.356 | 17.5× | 0.0159 | 0.69 |
| 6 | 35392 | 964 | 2.72% | 0.936 | 0.363 | 13.3× | 0.0211 | 0.71 |
| 7 | 32587 | 1099 | 3.37% | 0.929 | 0.368 | 10.9× | 0.0259 | 0.72 |
| 8 | 29820 | 1179 | 3.95% | 0.922 | 0.375 | 9.5× | 0.0302 | 0.71 |
| 9 | 27090 | 1197 | 4.42% | 0.917 | 0.375 | 8.5× | 0.0337 | 0.71 |
| 10 | 24252 | 1182 | 4.87% | 0.912 | 0.382 | 7.8× | 0.0369 | 0.71 |

#### STAR_PLUS_ELITE

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 45513 | 5 | 0.01% | 0.990 | 0.029 | 267.2× | 0.0001 | 0.67 |
| 2 | 43995 | 18 | 0.04% | 0.955 | 0.048 | 118.4× | 0.0004 | 0.54 |
| 3 | 42249 | 44 | 0.10% | 0.955 | 0.057 | 55.1× | 0.0010 | 0.60 |
| 4 | 40240 | 79 | 0.20% | 0.955 | 0.099 | 50.4× | 0.0019 | 0.66 |
| 5 | 37964 | 118 | 0.31% | 0.955 | 0.128 | 41.2× | 0.0029 | 0.74 |
| 6 | 35392 | 153 | 0.43% | 0.949 | 0.139 | 32.2× | 0.0040 | 0.81 |
| 7 | 32587 | 181 | 0.56% | 0.944 | 0.146 | 26.2× | 0.0051 | 0.82 |
| 8 | 29820 | 201 | 0.67% | 0.941 | 0.152 | 22.5× | 0.0061 | 0.80 |
| 9 | 27090 | 211 | 0.78% | 0.939 | 0.166 | 21.3× | 0.0070 | 0.78 |
| 10 | 24252 | 216 | 0.89% | 0.934 | 0.182 | 20.5× | 0.0080 | 0.74 |

## Per-bucket (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 35144 | 387 | 1.10% | 0.969 | 0.487 | 44.2× | 0.170 | 0.812 | 0.202 | 0.323 | 78 | 18 | 309 |
| R1 | 439 | 109 | 24.83% | 0.939 | 0.828 | 3.3× | 0.657 | 0.818 | 0.495 | 0.617 | 54 | 12 | 55 |
| R2-R3 | 1076 | 46 | 4.28% | 0.903 | 0.404 | 9.5× | 0.283 | 0.875 | 0.152 | 0.259 | 7 | 1 | 39 |
| R4-R10 | 4275 | 51 | 1.19% | 0.941 | 0.240 | 20.1× | 0.166 | 0.571 | 0.078 | 0.138 | 4 | 3 | 47 |
| R10+ | 13485 | 73 | 0.54% | 0.966 | 0.332 | 61.4× | 0.119 | 0.800 | 0.055 | 0.103 | 4 | 1 | 69 |
| IFA | 15869 | 108 | 0.68% | 0.957 | 0.332 | 48.8× | 0.130 | 0.900 | 0.083 | 0.153 | 9 | 1 | 99 |

#### MLB_DEBUT

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 35392 | 3018 | 8.53% | 0.930 | 0.591 | 6.9× | 0.416 | 0.774 | 0.253 | 0.382 | 764 | 223 | 2254 |
| R1 | 583 | 272 | 46.66% | 0.899 | 0.879 | 1.9× | 0.689 | 0.840 | 0.735 | 0.784 | 200 | 38 | 72 |
| R2-R3 | 1093 | 360 | 32.94% | 0.868 | 0.736 | 2.2× | 0.598 | 0.798 | 0.417 | 0.547 | 150 | 38 | 210 |
| R4-R10 | 4294 | 772 | 17.98% | 0.875 | 0.585 | 3.3× | 0.499 | 0.710 | 0.222 | 0.338 | 171 | 70 | 601 |
| R10+ | 13509 | 951 | 7.04% | 0.911 | 0.466 | 6.6× | 0.364 | 0.755 | 0.126 | 0.216 | 120 | 39 | 831 |
| IFA | 15913 | 663 | 4.17% | 0.934 | 0.486 | 11.7× | 0.300 | 0.764 | 0.186 | 0.299 | 123 | 38 | 540 |

#### ESTABLISHED_MLB

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 35392 | 964 | 2.72% | 0.936 | 0.363 | 13.3× | 0.246 | 0.743 | 0.027 | 0.052 | 26 | 9 | 938 |
| R1 | 583 | 127 | 21.78% | 0.862 | 0.590 | 2.7× | 0.518 | 0.789 | 0.118 | 0.205 | 15 | 4 | 112 |
| R2-R3 | 1093 | 120 | 10.98% | 0.849 | 0.388 | 3.5× | 0.378 | 0.667 | 0.033 | 0.063 | 4 | 2 | 116 |
| R4-R10 | 4294 | 241 | 5.61% | 0.882 | 0.330 | 5.9× | 0.305 | 0.500 | 0.004 | 0.008 | 1 | 1 | 240 |
| R10+ | 13509 | 274 | 2.03% | 0.915 | 0.290 | 14.3× | 0.203 | 0.667 | 0.007 | 0.014 | 2 | 1 | 272 |
| IFA | 15913 | 202 | 1.27% | 0.953 | 0.323 | 25.5× | 0.176 | 0.800 | 0.020 | 0.039 | 4 | 1 | 198 |

#### STAR_PLUS_ELITE

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 35392 | 153 | 0.43% | 0.949 | 0.139 | 32.2× | 0.102 | — | 0.000 | — | 0 | 0 | 153 |
| R1 | 583 | 40 | 6.86% | 0.843 | 0.249 | 3.6× | 0.300 | — | 0.000 | — | 0 | 0 | 40 |
| R2-R3 | 1093 | 18 | 1.65% | 0.903 | 0.236 | 14.3× | 0.178 | — | 0.000 | — | 0 | 0 | 18 |
| R4-R10 | 4294 | 40 | 0.93% | 0.914 | 0.078 | 8.4× | 0.138 | — | 0.000 | — | 0 | 0 | 40 |
| R10+ | 13509 | 29 | 0.21% | 0.869 | 0.084 | 39.3× | 0.059 | — | 0.000 | — | 0 | 0 | 29 |
| IFA | 15913 | 26 | 0.16% | 0.981 | 0.107 | 65.8× | 0.067 | — | 0.000 | — | 0 | 0 | 26 |

## Per-yip (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4941 | 159 | 3.22% | 0.909 | 0.513 | 15.9× | 0.250 | 0.796 | 0.270 | 0.404 | 43 | 11 | 116 |
| 1 | 4592 | 113 | 2.46% | 0.946 | 0.496 | 20.2× | 0.239 | 0.750 | 0.159 | 0.263 | 18 | 6 | 95 |
| 2 | 4173 | 71 | 1.70% | 0.954 | 0.493 | 29.0× | 0.203 | 1.000 | 0.155 | 0.268 | 11 | 0 | 60 |
| 3 | 3766 | 32 | 0.85% | 0.969 | 0.422 | 49.6× | 0.149 | 0.833 | 0.156 | 0.263 | 5 | 1 | 27 |
| 4 | 3389 | 9 | 0.27% | 0.981 | 0.352 | 132.7× | 0.086 | 1.000 | 0.111 | 0.200 | 1 | 0 | 8 |
| 5 | 3057 | 3 | 0.10% | 0.993 | 0.514 | 524.3× | 0.053 | — | 0.000 | — | 0 | 0 | 3 |
| 6 | 2740 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 7 | 2497 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 8 | 2239 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 9 | 2003 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 1747 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

#### MLB_DEBUT

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4944 | 669 | 13.53% | 0.824 | 0.491 | 3.6× | 0.384 | 0.758 | 0.149 | 0.250 | 100 | 32 | 569 |
| 1 | 4625 | 676 | 14.62% | 0.878 | 0.600 | 4.1× | 0.463 | 0.787 | 0.268 | 0.400 | 181 | 49 | 495 |
| 2 | 4230 | 606 | 14.33% | 0.913 | 0.662 | 4.6× | 0.501 | 0.772 | 0.342 | 0.474 | 207 | 61 | 399 |
| 3 | 3817 | 448 | 11.74% | 0.929 | 0.641 | 5.5× | 0.478 | 0.764 | 0.333 | 0.463 | 149 | 46 | 299 |
| 4 | 3422 | 281 | 8.21% | 0.942 | 0.637 | 7.8× | 0.421 | 0.798 | 0.310 | 0.446 | 87 | 22 | 194 |
| 5 | 3077 | 178 | 5.78% | 0.952 | 0.598 | 10.3× | 0.365 | 0.766 | 0.202 | 0.320 | 36 | 11 | 142 |
| 6 | 2753 | 87 | 3.16% | 0.959 | 0.475 | 15.0× | 0.278 | 0.800 | 0.046 | 0.087 | 4 | 1 | 83 |
| 7 | 2508 | 42 | 1.67% | 0.973 | 0.400 | 23.9× | 0.210 | 0.000 | 0.000 | — | 0 | 1 | 42 |
| 8 | 2249 | 21 | 0.93% | 0.980 | 0.343 | 36.7× | 0.160 | — | 0.000 | — | 0 | 0 | 21 |
| 9 | 2012 | 7 | 0.35% | 0.994 | 0.425 | 122.2× | 0.101 | — | 0.000 | — | 0 | 0 | 7 |
| 10 | 1755 | 3 | 0.17% | 0.993 | 0.582 | 340.4× | 0.071 | — | 0.000 | — | 0 | 0 | 3 |

#### ESTABLISHED_MLB

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4944 | 186 | 3.76% | 0.865 | 0.293 | 7.8× | 0.241 | 0.778 | 0.038 | 0.072 | 7 | 2 | 179 |
| 1 | 4625 | 213 | 4.61% | 0.903 | 0.384 | 8.3× | 0.293 | 0.667 | 0.038 | 0.071 | 8 | 4 | 205 |
| 2 | 4230 | 213 | 5.04% | 0.917 | 0.430 | 8.5× | 0.316 | 0.875 | 0.033 | 0.063 | 7 | 1 | 206 |
| 3 | 3817 | 152 | 3.98% | 0.921 | 0.394 | 9.9× | 0.285 | 0.667 | 0.026 | 0.051 | 4 | 2 | 148 |
| 4 | 3422 | 103 | 3.01% | 0.945 | 0.410 | 13.6× | 0.264 | — | 0.000 | — | 0 | 0 | 103 |
| 5 | 3077 | 57 | 1.85% | 0.966 | 0.414 | 22.3× | 0.218 | — | 0.000 | — | 0 | 0 | 57 |
| 6 | 2753 | 23 | 0.84% | 0.952 | 0.180 | 21.5× | 0.142 | — | 0.000 | — | 0 | 0 | 23 |
| 7 | 2508 | 12 | 0.48% | 0.955 | 0.080 | 16.8× | 0.109 | — | 0.000 | — | 0 | 0 | 12 |
| 8 | 2249 | 4 | 0.18% | 0.972 | 0.087 | 48.6× | 0.069 | — | 0.000 | — | 0 | 0 | 4 |
| 9 | 2012 | 1 | 0.05% | 0.993 | 0.062 | 125.8× | 0.038 | — | 0.000 | — | 0 | 0 | 1 |
| 10 | 1755 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

#### STAR_PLUS_ELITE

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4944 | 41 | 0.83% | 0.878 | 0.166 | 20.1× | 0.119 | — | 0.000 | — | 0 | 0 | 41 |
| 1 | 4625 | 39 | 0.84% | 0.925 | 0.162 | 19.2× | 0.134 | — | 0.000 | — | 0 | 0 | 39 |
| 2 | 4230 | 37 | 0.87% | 0.921 | 0.145 | 16.6× | 0.136 | — | 0.000 | — | 0 | 0 | 37 |
| 3 | 3817 | 21 | 0.55% | 0.932 | 0.161 | 29.3× | 0.111 | — | 0.000 | — | 0 | 0 | 21 |
| 4 | 3422 | 10 | 0.29% | 0.954 | 0.059 | 20.1× | 0.085 | — | 0.000 | — | 0 | 0 | 10 |
| 5 | 3077 | 5 | 0.16% | 0.983 | 0.108 | 66.2× | 0.067 | — | 0.000 | — | 0 | 0 | 5 |
| 6 | 2753 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 7 | 2508 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 8 | 2249 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 9 | 2012 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 1755 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

## Per-level (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 35144 | 387 | 1.10% | 0.969 | 0.487 | 44.2× | 0.170 | 0.812 | 0.202 | 0.323 | 78 | 18 | 309 |
| RK | 5943 | 105 | 1.77% | 0.925 | 0.375 | 21.2× | 0.194 | 0.773 | 0.162 | 0.268 | 17 | 5 | 88 |
| A- | 1685 | 40 | 2.37% | 0.917 | 0.461 | 19.4× | 0.220 | 0.778 | 0.175 | 0.286 | 7 | 2 | 33 |
| A | 2330 | 94 | 4.03% | 0.946 | 0.503 | 12.5× | 0.304 | 0.778 | 0.149 | 0.250 | 14 | 4 | 80 |
| A+ | 2240 | 61 | 2.72% | 0.960 | 0.528 | 19.4× | 0.259 | 0.800 | 0.197 | 0.316 | 12 | 3 | 49 |
| AA | 1940 | 50 | 2.58% | 0.979 | 0.700 | 27.2× | 0.263 | 0.895 | 0.340 | 0.493 | 17 | 2 | 33 |
| AAA | 2268 | 8 | 0.35% | 0.990 | 0.485 | 137.6× | 0.101 | 1.000 | 0.125 | 0.222 | 1 | 0 | 7 |
| NONE | 18688 | 28 | 0.15% | 0.960 | 0.414 | 276.5× | 0.062 | 0.818 | 0.321 | 0.462 | 9 | 2 | 19 |

#### MLB_DEBUT

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 35392 | 3018 | 8.53% | 0.930 | 0.591 | 6.9× | 0.416 | 0.774 | 0.253 | 0.382 | 764 | 223 | 2254 |
| RK | 5948 | 422 | 7.09% | 0.822 | 0.315 | 4.4× | 0.286 | 0.639 | 0.055 | 0.100 | 23 | 13 | 399 |
| A- | 1686 | 276 | 16.37% | 0.789 | 0.493 | 3.0× | 0.371 | 0.829 | 0.105 | 0.186 | 29 | 6 | 247 |
| A | 2347 | 487 | 20.75% | 0.817 | 0.572 | 2.8× | 0.445 | 0.720 | 0.211 | 0.327 | 103 | 40 | 384 |
| A+ | 2272 | 552 | 24.30% | 0.823 | 0.636 | 2.6× | 0.480 | 0.820 | 0.255 | 0.390 | 141 | 31 | 411 |
| AA | 2022 | 650 | 32.15% | 0.826 | 0.706 | 2.2× | 0.528 | 0.776 | 0.422 | 0.546 | 274 | 79 | 376 |
| AAA | 2331 | 486 | 20.85% | 0.896 | 0.699 | 3.4× | 0.558 | 0.787 | 0.364 | 0.498 | 177 | 48 | 309 |
| NONE | 18728 | 144 | 0.77% | 0.963 | 0.377 | 49.0× | 0.140 | 0.800 | 0.111 | 0.195 | 16 | 4 | 128 |

#### ESTABLISHED_MLB

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 35392 | 964 | 2.72% | 0.936 | 0.363 | 13.3× | 0.246 | 0.743 | 0.027 | 0.052 | 26 | 9 | 938 |
| RK | 5948 | 89 | 1.50% | 0.869 | 0.208 | 13.9× | 0.155 | 1.000 | 0.011 | 0.022 | 1 | 0 | 88 |
| A- | 1686 | 71 | 4.21% | 0.853 | 0.239 | 5.7× | 0.246 | 1.000 | 0.014 | 0.028 | 1 | 0 | 70 |
| A | 2347 | 150 | 6.39% | 0.859 | 0.338 | 5.3× | 0.304 | 1.000 | 0.013 | 0.026 | 2 | 0 | 148 |
| A+ | 2272 | 185 | 8.14% | 0.834 | 0.379 | 4.7× | 0.316 | 0.625 | 0.027 | 0.052 | 5 | 3 | 180 |
| AA | 2022 | 255 | 12.61% | 0.844 | 0.455 | 3.6× | 0.396 | 0.750 | 0.047 | 0.089 | 12 | 4 | 243 |
| AAA | 2331 | 172 | 7.38% | 0.865 | 0.415 | 5.6× | 0.331 | 0.750 | 0.017 | 0.034 | 3 | 1 | 169 |
| NONE | 18728 | 42 | 0.22% | 0.947 | 0.228 | 101.7× | 0.073 | 0.667 | 0.048 | 0.089 | 2 | 1 | 40 |

#### STAR_PLUS_ELITE

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 35392 | 153 | 0.43% | 0.949 | 0.139 | 32.2× | 0.102 | — | 0.000 | — | 0 | 0 | 153 |
| RK | 5948 | 11 | 0.18% | 0.889 | 0.078 | 42.0× | 0.058 | — | 0.000 | — | 0 | 0 | 11 |
| A- | 1686 | 15 | 0.89% | 0.878 | 0.138 | 15.5× | 0.123 | — | 0.000 | — | 0 | 0 | 15 |
| A | 2347 | 24 | 1.02% | 0.888 | 0.119 | 11.6× | 0.135 | — | 0.000 | — | 0 | 0 | 24 |
| A+ | 2272 | 31 | 1.36% | 0.890 | 0.166 | 12.2× | 0.157 | — | 0.000 | — | 0 | 0 | 31 |
| AA | 2022 | 44 | 2.18% | 0.904 | 0.216 | 9.9× | 0.204 | — | 0.000 | — | 0 | 0 | 44 |
| AAA | 2331 | 18 | 0.77% | 0.946 | 0.121 | 15.7× | 0.135 | — | 0.000 | — | 0 | 0 | 18 |
| NONE | 18728 | 10 | 0.05% | 0.934 | 0.279 | 523.2× | 0.035 | — | 0.000 | — | 0 | 0 | 10 |

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
