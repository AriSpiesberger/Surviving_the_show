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
| TOP_100_PROSPECT | 35351 | 0.99% | **0.449** | 45.3× | 0.972 | 0.162 | 0.692 | 0.206 | 0.317 |
| MLB_DEBUT | 35658 | 7.85% | **0.569** | 7.2× | 0.930 | 0.401 | 0.753 | 0.264 | 0.391 |
| ESTABLISHED_MLB | 35658 | 2.25% | **0.361** | 16.0× | 0.943 | 0.228 | 0.698 | 0.055 | 0.101 |
| STAR_PLUS_ELITE | 35658 | 0.41% | **0.153** | 37.3× | 0.952 | 0.100 | — | 0.000 | — |
| **weighted-AP** | | | **0.420** | | | | | | |

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
| 0-5% | 36,482 | 0.1% | 0.1% | +0.0% |
| 5-10% | 221 | 7.1% | 8.1% | +1.0% |
| 10-20% | 154 | 14.2% | 17.5% | +3.4% |
| 20-30% | 60 | 24.5% | 36.7% | +12.2% |
| 30-40% | 51 | 33.9% | 31.4% | -2.6% |
| 40-50% | 45 | 45.3% | 33.3% | -11.9% |
| 50-60% | 24 | 55.3% | 41.7% | -13.6% |
| 60-70% | 23 | 64.5% | 60.9% | -3.6% |
| 70-80% | 25 | 74.6% | 64.0% | -10.6% |

#### TOP_100_PROSPECT — P(within 6y)

| predicted | n | avg pred | actual | diff |
|---|---:|---:|---:|---:|
| 0-5% | 29,506 | 0.2% | 0.2% | +0.0% |
| 5-10% | 270 | 7.2% | 7.4% | +0.2% |
| 10-20% | 184 | 14.0% | 17.9% | +3.9% |
| 20-30% | 66 | 24.1% | 34.8% | +10.8% |
| 30-40% | 63 | 34.7% | 33.3% | -1.4% |
| 40-50% | 29 | 45.0% | 37.9% | -7.1% |
| 50-60% | 36 | 54.6% | 47.2% | -7.4% |
| 60-70% | 20 | 64.6% | 55.0% | -9.6% |
| 70-80% | 24 | 74.6% | 62.5% | -12.1% |

#### MLB_DEBUT — P(within 3y)

| predicted | n | avg pred | actual | diff |
|---|---:|---:|---:|---:|
| 0-5% | 32,628 | 0.5% | 0.5% | +0.1% |
| 5-10% | 1,416 | 7.3% | 9.3% | +1.9% |
| 10-20% | 1,275 | 14.1% | 14.7% | +0.5% |
| 20-30% | 599 | 24.7% | 28.4% | +3.7% |
| 30-40% | 434 | 34.9% | 31.3% | -3.6% |
| 40-50% | 320 | 44.8% | 42.8% | -2.0% |
| 50-60% | 225 | 54.7% | 56.0% | +1.3% |
| 60-70% | 143 | 64.8% | 67.1% | +2.3% |
| 70-80% | 142 | 74.7% | 70.4% | -4.2% |
| 80-90% | 144 | 85.2% | 75.0% | -10.2% |
| 90-100% | 67 | 92.3% | 88.1% | -4.2% |

#### MLB_DEBUT — P(within 6y)

| predicted | n | avg pred | actual | diff |
|---|---:|---:|---:|---:|
| 0-5% | 24,109 | 0.7% | 0.7% | -0.0% |
| 5-10% | 1,885 | 7.2% | 9.2% | +2.0% |
| 10-20% | 1,720 | 14.3% | 14.8% | +0.4% |
| 20-30% | 859 | 24.3% | 26.2% | +1.8% |
| 30-40% | 492 | 34.7% | 34.3% | -0.3% |
| 40-50% | 410 | 44.9% | 42.9% | -2.0% |
| 50-60% | 291 | 54.9% | 57.4% | +2.5% |
| 60-70% | 210 | 64.6% | 68.6% | +4.0% |
| 70-80% | 201 | 75.5% | 73.1% | -2.4% |
| 80-90% | 199 | 84.9% | 80.9% | -4.0% |
| 90-100% | 84 | 92.1% | 90.5% | -1.7% |

#### ESTABLISHED_MLB — P(within 3y)

| predicted | n | avg pred | actual | diff |
|---|---:|---:|---:|---:|
| 0-5% | 36,885 | 0.1% | 0.1% | +0.0% |
| 5-10% | 231 | 7.0% | 7.8% | +0.8% |
| 10-20% | 125 | 13.8% | 22.4% | +8.6% |
| 20-30% | 64 | 24.3% | 28.1% | +3.8% |
| 30-40% | 33 | 35.1% | 39.4% | +4.3% |
| 40-50% | 35 | 44.4% | 45.7% | +1.3% |

#### ESTABLISHED_MLB — P(within 6y)

| predicted | n | avg pred | actual | diff |
|---|---:|---:|---:|---:|
| 0-5% | 28,697 | 0.3% | 0.3% | -0.0% |
| 5-10% | 700 | 7.0% | 8.0% | +1.0% |
| 10-20% | 526 | 14.1% | 17.7% | +3.6% |
| 20-30% | 217 | 24.4% | 24.9% | +0.5% |
| 30-40% | 117 | 35.0% | 30.8% | -4.2% |
| 40-50% | 99 | 44.3% | 40.4% | -3.9% |
| 50-60% | 63 | 55.0% | 57.1% | +2.1% |
| 60-70% | 28 | 65.2% | 67.9% | +2.7% |

#### STAR_PLUS_ELITE — P(within 3y)

| predicted | n | avg pred | actual | diff |
|---|---:|---:|---:|---:|
| 0-5% | 37,358 | 0.0% | 0.0% | +0.0% |
| 5-10% | 22 | 6.8% | 9.1% | +2.3% |

#### STAR_PLUS_ELITE — P(within 6y)

| predicted | n | avg pred | actual | diff |
|---|---:|---:|---:|---:|
| 0-5% | 30,155 | 0.1% | 0.1% | -0.0% |
| 5-10% | 176 | 7.1% | 14.8% | +7.6% |
| 10-20% | 94 | 14.0% | 19.1% | +5.2% |
| 20-30% | 19 | 23.2% | 5.3% | -18.0% |

## Per-horizon trajectory (h=1..10, resolved at each h)

#### TOP_100_PROSPECT

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 45520 | 138 | 0.30% | 0.993 | 0.537 | 177.1× | 0.0020 | 0.98 |
| 2 | 44003 | 242 | 0.55% | 0.985 | 0.497 | 90.4× | 0.0037 | 0.97 |
| 3 | 42236 | 312 | 0.74% | 0.978 | 0.468 | 63.3× | 0.0052 | 0.93 |
| 4 | 40210 | 348 | 0.87% | 0.974 | 0.451 | 52.2× | 0.0062 | 0.92 |
| 5 | 37930 | 355 | 0.94% | 0.972 | 0.444 | 47.4× | 0.0067 | 0.93 |
| 6 | 35351 | 350 | 0.99% | 0.972 | 0.449 | 45.3× | 0.0071 | 0.94 |
| 7 | 32520 | 339 | 1.04% | 0.971 | 0.446 | 42.8× | 0.0074 | 0.94 |
| 8 | 29716 | 322 | 1.08% | 0.970 | 0.443 | 40.9× | 0.0078 | 0.94 |
| 9 | 26942 | 307 | 1.14% | 0.969 | 0.447 | 39.2× | 0.0081 | 0.95 |
| 10 | 24068 | 285 | 1.18% | 0.967 | 0.437 | 36.9× | 0.0085 | 0.96 |

#### MLB_DEBUT

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 45887 | 733 | 1.60% | 0.968 | 0.400 | 25.0× | 0.0118 | 0.94 |
| 2 | 44365 | 1428 | 3.22% | 0.956 | 0.489 | 15.2× | 0.0217 | 0.93 |
| 3 | 42591 | 2045 | 4.80% | 0.946 | 0.525 | 10.9× | 0.0309 | 0.95 |
| 4 | 40553 | 2507 | 6.18% | 0.938 | 0.545 | 8.8× | 0.0387 | 0.95 |
| 5 | 38257 | 2743 | 7.17% | 0.932 | 0.559 | 7.8× | 0.0441 | 0.96 |
| 6 | 35658 | 2798 | 7.85% | 0.930 | 0.569 | 7.2× | 0.0476 | 0.97 |
| 7 | 32804 | 2730 | 8.32% | 0.927 | 0.571 | 6.9× | 0.0503 | 0.97 |
| 8 | 29977 | 2608 | 8.70% | 0.925 | 0.573 | 6.6× | 0.0524 | 0.97 |
| 9 | 27180 | 2464 | 9.07% | 0.922 | 0.577 | 6.4× | 0.0544 | 0.97 |
| 10 | 24282 | 2297 | 9.46% | 0.919 | 0.575 | 6.1× | 0.0568 | 0.97 |

#### ESTABLISHED_MLB

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 45887 | 4 | 0.01% | 0.994 | 0.291 | 3334.1× | 0.0001 | 1.21 |
| 2 | 44365 | 80 | 0.18% | 0.983 | 0.194 | 107.8× | 0.0016 | 0.58 |
| 3 | 42591 | 254 | 0.60% | 0.974 | 0.298 | 50.0× | 0.0049 | 0.74 |
| 4 | 40553 | 466 | 1.15% | 0.965 | 0.338 | 29.4× | 0.0091 | 0.79 |
| 5 | 38257 | 655 | 1.71% | 0.954 | 0.351 | 20.5× | 0.0133 | 0.84 |
| 6 | 35658 | 804 | 2.25% | 0.943 | 0.361 | 16.0× | 0.0173 | 0.89 |
| 7 | 32804 | 896 | 2.73% | 0.936 | 0.370 | 13.5× | 0.0207 | 0.92 |
| 8 | 29977 | 946 | 3.16% | 0.931 | 0.379 | 12.0× | 0.0238 | 0.94 |
| 9 | 27180 | 967 | 3.56% | 0.925 | 0.381 | 10.7× | 0.0268 | 0.93 |
| 10 | 24282 | 957 | 3.94% | 0.920 | 0.389 | 9.9× | 0.0295 | 0.92 |

#### STAR_PLUS_ELITE

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 45887 | 4 | 0.01% | 1.000 | 0.470 | 5397.2× | 0.0001 | 0.99 |
| 2 | 44365 | 14 | 0.03% | 0.977 | 0.245 | 777.1× | 0.0003 | 0.62 |
| 3 | 42591 | 36 | 0.08% | 0.977 | 0.188 | 222.1× | 0.0008 | 0.72 |
| 4 | 40553 | 68 | 0.17% | 0.962 | 0.161 | 96.0× | 0.0015 | 0.84 |
| 5 | 38257 | 110 | 0.29% | 0.957 | 0.144 | 50.2× | 0.0026 | 0.87 |
| 6 | 35658 | 146 | 0.41% | 0.952 | 0.153 | 37.3× | 0.0037 | 0.95 |
| 7 | 32804 | 174 | 0.53% | 0.947 | 0.163 | 30.6× | 0.0048 | 0.95 |
| 8 | 29977 | 189 | 0.63% | 0.941 | 0.168 | 26.7× | 0.0057 | 0.97 |
| 9 | 27180 | 200 | 0.74% | 0.935 | 0.177 | 24.1× | 0.0066 | 0.93 |
| 10 | 24282 | 199 | 0.82% | 0.931 | 0.178 | 21.7× | 0.0073 | 0.90 |

## Per-bucket (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 35351 | 350 | 0.99% | 0.972 | 0.449 | 45.3× | 0.162 | 0.692 | 0.206 | 0.317 | 72 | 32 | 278 |
| R1 | 446 | 105 | 23.54% | 0.907 | 0.750 | 3.2× | 0.599 | 0.733 | 0.524 | 0.611 | 55 | 20 | 50 |
| R2-R3 | 1194 | 46 | 3.85% | 0.884 | 0.178 | 4.6× | 0.256 | 0.143 | 0.022 | 0.038 | 1 | 6 | 45 |
| R4-R10 | 4078 | 43 | 1.05% | 0.951 | 0.295 | 28.0× | 0.160 | 0.500 | 0.047 | 0.085 | 2 | 2 | 41 |
| R10+ | 13279 | 55 | 0.41% | 0.955 | 0.223 | 53.8× | 0.101 | 0.500 | 0.018 | 0.035 | 1 | 1 | 54 |
| IFA | 16354 | 101 | 0.62% | 0.972 | 0.410 | 66.5× | 0.128 | 0.812 | 0.129 | 0.222 | 13 | 3 | 88 |

#### MLB_DEBUT

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 35658 | 2798 | 7.85% | 0.930 | 0.569 | 7.2× | 0.401 | 0.753 | 0.264 | 0.391 | 738 | 242 | 2060 |
| R1 | 564 | 300 | 53.19% | 0.841 | 0.833 | 1.6× | 0.589 | 0.779 | 0.800 | 0.789 | 240 | 68 | 60 |
| R2-R3 | 1215 | 378 | 31.11% | 0.845 | 0.680 | 2.2× | 0.554 | 0.739 | 0.360 | 0.484 | 136 | 48 | 242 |
| R4-R10 | 4128 | 675 | 16.35% | 0.861 | 0.539 | 3.3× | 0.463 | 0.709 | 0.206 | 0.319 | 139 | 57 | 536 |
| R10+ | 13310 | 916 | 6.88% | 0.896 | 0.407 | 5.9× | 0.348 | 0.704 | 0.096 | 0.169 | 88 | 37 | 828 |
| IFA | 16441 | 529 | 3.22% | 0.945 | 0.530 | 16.5× | 0.272 | 0.808 | 0.255 | 0.388 | 135 | 32 | 394 |

#### ESTABLISHED_MLB

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 35658 | 804 | 2.25% | 0.943 | 0.361 | 16.0× | 0.228 | 0.698 | 0.055 | 0.101 | 44 | 19 | 760 |
| R1 | 564 | 157 | 27.84% | 0.854 | 0.653 | 2.3× | 0.550 | 0.750 | 0.210 | 0.328 | 33 | 11 | 124 |
| R2-R3 | 1215 | 127 | 10.45% | 0.803 | 0.298 | 2.9× | 0.321 | 0.600 | 0.024 | 0.045 | 3 | 2 | 124 |
| R4-R10 | 4128 | 178 | 4.31% | 0.892 | 0.245 | 5.7× | 0.276 | 0.000 | 0.000 | — | 0 | 1 | 178 |
| R10+ | 13310 | 213 | 1.60% | 0.906 | 0.234 | 14.7× | 0.177 | 0.400 | 0.009 | 0.018 | 2 | 3 | 211 |
| IFA | 16441 | 129 | 0.78% | 0.952 | 0.324 | 41.3× | 0.138 | 0.750 | 0.047 | 0.088 | 6 | 2 | 123 |

#### STAR_PLUS_ELITE

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 35658 | 146 | 0.41% | 0.952 | 0.153 | 37.3× | 0.100 | — | 0.000 | — | 0 | 0 | 146 |
| R1 | 564 | 41 | 7.27% | 0.822 | 0.255 | 3.5× | 0.290 | — | 0.000 | — | 0 | 0 | 41 |
| R2-R3 | 1215 | 28 | 2.30% | 0.793 | 0.123 | 5.3× | 0.152 | — | 0.000 | — | 0 | 0 | 28 |
| R4-R10 | 4128 | 20 | 0.48% | 0.940 | 0.179 | 37.0× | 0.106 | — | 0.000 | — | 0 | 0 | 20 |
| R10+ | 13310 | 29 | 0.22% | 0.884 | 0.064 | 29.3× | 0.062 | — | 0.000 | — | 0 | 0 | 29 |
| IFA | 16441 | 28 | 0.17% | 0.990 | 0.187 | 110.0× | 0.070 | — | 0.000 | — | 0 | 0 | 28 |

## Per-yip (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4939 | 150 | 3.04% | 0.922 | 0.526 | 17.3× | 0.251 | 0.754 | 0.287 | 0.415 | 43 | 14 | 107 |
| 1 | 4586 | 101 | 2.20% | 0.937 | 0.365 | 16.6× | 0.222 | 0.593 | 0.158 | 0.250 | 16 | 11 | 85 |
| 2 | 4199 | 61 | 1.45% | 0.957 | 0.351 | 24.2× | 0.189 | 0.583 | 0.115 | 0.192 | 7 | 5 | 54 |
| 3 | 3807 | 31 | 0.81% | 0.981 | 0.560 | 68.7× | 0.150 | 0.667 | 0.129 | 0.216 | 4 | 2 | 27 |
| 4 | 3414 | 7 | 0.21% | 0.997 | 0.498 | 242.9× | 0.078 | 1.000 | 0.286 | 0.444 | 2 | 0 | 5 |
| 5 | 3082 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 6 | 2774 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 7 | 2504 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 8 | 2256 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 9 | 2030 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 1760 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

#### MLB_DEBUT

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4947 | 633 | 12.80% | 0.844 | 0.488 | 3.8× | 0.398 | 0.709 | 0.177 | 0.283 | 112 | 46 | 521 |
| 1 | 4631 | 630 | 13.60% | 0.882 | 0.585 | 4.3× | 0.453 | 0.764 | 0.283 | 0.413 | 178 | 55 | 452 |
| 2 | 4256 | 562 | 13.20% | 0.907 | 0.612 | 4.6× | 0.478 | 0.729 | 0.320 | 0.445 | 180 | 67 | 382 |
| 3 | 3862 | 429 | 11.11% | 0.932 | 0.661 | 6.0× | 0.470 | 0.825 | 0.340 | 0.482 | 146 | 31 | 283 |
| 4 | 3460 | 265 | 7.66% | 0.942 | 0.623 | 8.1× | 0.407 | 0.775 | 0.298 | 0.431 | 79 | 23 | 186 |
| 5 | 3107 | 148 | 4.76% | 0.943 | 0.502 | 10.5× | 0.327 | 0.733 | 0.223 | 0.342 | 33 | 12 | 115 |
| 6 | 2792 | 68 | 2.44% | 0.946 | 0.340 | 14.0× | 0.238 | 0.636 | 0.103 | 0.177 | 7 | 4 | 61 |
| 7 | 2520 | 33 | 1.31% | 0.960 | 0.233 | 17.8× | 0.181 | 0.667 | 0.061 | 0.111 | 2 | 1 | 31 |
| 8 | 2270 | 19 | 0.84% | 0.976 | 0.280 | 33.4× | 0.150 | 0.500 | 0.053 | 0.095 | 1 | 1 | 18 |
| 9 | 2042 | 9 | 0.44% | 0.989 | 0.464 | 105.2× | 0.112 | 0.000 | 0.000 | — | 0 | 1 | 9 |
| 10 | 1771 | 2 | 0.11% | 0.999 | 0.417 | 369.0× | 0.058 | 0.000 | 0.000 | — | 0 | 1 | 2 |

#### ESTABLISHED_MLB

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4947 | 180 | 3.64% | 0.896 | 0.373 | 10.2× | 0.257 | 0.733 | 0.061 | 0.113 | 11 | 4 | 169 |
| 1 | 4631 | 204 | 4.41% | 0.915 | 0.448 | 10.2× | 0.295 | 0.889 | 0.078 | 0.144 | 16 | 2 | 188 |
| 2 | 4256 | 178 | 4.18% | 0.913 | 0.378 | 9.0× | 0.287 | 0.667 | 0.067 | 0.122 | 12 | 6 | 166 |
| 3 | 3862 | 124 | 3.21% | 0.926 | 0.323 | 10.1× | 0.260 | 0.444 | 0.032 | 0.060 | 4 | 5 | 120 |
| 4 | 3460 | 73 | 2.11% | 0.938 | 0.292 | 13.8× | 0.218 | 0.333 | 0.014 | 0.026 | 1 | 2 | 72 |
| 5 | 3107 | 32 | 1.03% | 0.937 | 0.195 | 19.0× | 0.153 | — | 0.000 | — | 0 | 0 | 32 |
| 6 | 2792 | 11 | 0.39% | 0.944 | 0.122 | 31.0× | 0.096 | — | 0.000 | — | 0 | 0 | 11 |
| 7 | 2520 | 1 | 0.04% | 0.977 | 0.017 | 42.7× | 0.033 | — | 0.000 | — | 0 | 0 | 1 |
| 8 | 2270 | 1 | 0.04% | 0.999 | 0.250 | 567.5× | 0.036 | — | 0.000 | — | 0 | 0 | 1 |
| 9 | 2042 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 1771 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

#### STAR_PLUS_ELITE

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4947 | 29 | 0.59% | 0.903 | 0.144 | 24.6× | 0.107 | — | 0.000 | — | 0 | 0 | 29 |
| 1 | 4631 | 40 | 0.86% | 0.936 | 0.199 | 23.0× | 0.140 | — | 0.000 | — | 0 | 0 | 40 |
| 2 | 4256 | 36 | 0.85% | 0.934 | 0.192 | 22.7× | 0.138 | — | 0.000 | — | 0 | 0 | 36 |
| 3 | 3862 | 24 | 0.62% | 0.941 | 0.128 | 20.6× | 0.120 | — | 0.000 | — | 0 | 0 | 24 |
| 4 | 3460 | 12 | 0.35% | 0.943 | 0.077 | 22.1× | 0.090 | — | 0.000 | — | 0 | 0 | 12 |
| 5 | 3107 | 4 | 0.13% | 0.956 | 0.075 | 58.3× | 0.057 | — | 0.000 | — | 0 | 0 | 4 |
| 6 | 2792 | 1 | 0.04% | 0.876 | 0.003 | 8.0× | 0.025 | — | 0.000 | — | 0 | 0 | 1 |
| 7 | 2520 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 8 | 2270 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 9 | 2042 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 1771 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

## Per-level (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 35351 | 350 | 0.99% | 0.972 | 0.449 | 45.3× | 0.162 | 0.692 | 0.206 | 0.317 | 72 | 32 | 278 |
| RK | 5833 | 83 | 1.42% | 0.919 | 0.317 | 22.3× | 0.172 | 0.889 | 0.096 | 0.174 | 8 | 1 | 75 |
| A- | 1465 | 38 | 2.59% | 0.950 | 0.589 | 22.7× | 0.248 | 0.750 | 0.237 | 0.360 | 9 | 3 | 29 |
| A | 1933 | 55 | 2.85% | 0.964 | 0.545 | 19.2× | 0.267 | 0.607 | 0.309 | 0.410 | 17 | 11 | 38 |
| A+ | 1908 | 36 | 1.89% | 0.968 | 0.478 | 25.4× | 0.221 | 0.615 | 0.222 | 0.327 | 8 | 5 | 28 |
| AA | 1576 | 35 | 2.22% | 0.986 | 0.650 | 29.3× | 0.248 | 0.733 | 0.314 | 0.440 | 11 | 4 | 24 |
| AAA | 2098 | 9 | 0.43% | 0.993 | 0.515 | 120.1× | 0.112 | 1.000 | 0.111 | 0.200 | 1 | 0 | 8 |
| NONE | 20504 | 94 | 0.46% | 0.983 | 0.384 | 83.8× | 0.113 | 0.692 | 0.191 | 0.300 | 18 | 8 | 76 |

#### MLB_DEBUT

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 35658 | 2798 | 7.85% | 0.930 | 0.569 | 7.2× | 0.401 | 0.753 | 0.264 | 0.391 | 738 | 242 | 2060 |
| RK | 5839 | 313 | 5.36% | 0.829 | 0.303 | 5.6× | 0.256 | 0.656 | 0.067 | 0.122 | 21 | 11 | 292 |
| A- | 1468 | 220 | 14.99% | 0.831 | 0.499 | 3.3× | 0.410 | 0.698 | 0.136 | 0.228 | 30 | 13 | 190 |
| A | 1951 | 312 | 15.99% | 0.849 | 0.536 | 3.3× | 0.443 | 0.711 | 0.260 | 0.380 | 81 | 33 | 231 |
| A+ | 1941 | 396 | 20.40% | 0.842 | 0.595 | 2.9× | 0.478 | 0.727 | 0.275 | 0.399 | 109 | 41 | 287 |
| AA | 1648 | 491 | 29.79% | 0.858 | 0.729 | 2.4× | 0.566 | 0.796 | 0.460 | 0.583 | 226 | 58 | 265 |
| AAA | 2154 | 360 | 16.71% | 0.899 | 0.680 | 4.1× | 0.515 | 0.792 | 0.369 | 0.504 | 133 | 35 | 227 |
| NONE | 20605 | 705 | 3.42% | 0.966 | 0.497 | 14.5× | 0.294 | 0.730 | 0.196 | 0.309 | 138 | 51 | 567 |

#### ESTABLISHED_MLB

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 35658 | 804 | 2.25% | 0.943 | 0.361 | 16.0× | 0.228 | 0.698 | 0.055 | 0.101 | 44 | 19 | 760 |
| RK | 5839 | 51 | 0.87% | 0.868 | 0.158 | 18.0× | 0.119 | 0.500 | 0.020 | 0.038 | 1 | 1 | 50 |
| A- | 1468 | 45 | 3.07% | 0.878 | 0.179 | 5.8× | 0.226 | 0.000 | 0.000 | — | 0 | 1 | 45 |
| A | 1951 | 86 | 4.41% | 0.877 | 0.373 | 8.5× | 0.268 | 1.000 | 0.047 | 0.089 | 4 | 0 | 82 |
| A+ | 1941 | 107 | 5.51% | 0.846 | 0.313 | 5.7× | 0.274 | 0.571 | 0.037 | 0.070 | 4 | 3 | 103 |
| AA | 1648 | 170 | 10.32% | 0.881 | 0.499 | 4.8× | 0.401 | 0.742 | 0.135 | 0.229 | 23 | 8 | 147 |
| AAA | 2154 | 97 | 4.50% | 0.907 | 0.451 | 10.0× | 0.292 | 0.667 | 0.062 | 0.113 | 6 | 3 | 91 |
| NONE | 20605 | 248 | 1.20% | 0.973 | 0.343 | 28.5× | 0.179 | 0.667 | 0.024 | 0.047 | 6 | 3 | 242 |

#### STAR_PLUS_ELITE

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 35658 | 146 | 0.41% | 0.952 | 0.153 | 37.3× | 0.100 | — | 0.000 | — | 0 | 0 | 146 |
| RK | 5839 | 6 | 0.10% | 0.934 | 0.074 | 71.8× | 0.048 | — | 0.000 | — | 0 | 0 | 6 |
| A- | 1468 | 7 | 0.48% | 0.897 | 0.187 | 39.2× | 0.095 | — | 0.000 | — | 0 | 0 | 7 |
| A | 1951 | 12 | 0.62% | 0.926 | 0.097 | 15.8× | 0.115 | — | 0.000 | — | 0 | 0 | 12 |
| A+ | 1941 | 22 | 1.13% | 0.884 | 0.241 | 21.3× | 0.141 | — | 0.000 | — | 0 | 0 | 22 |
| AA | 1648 | 37 | 2.25% | 0.940 | 0.204 | 9.1× | 0.226 | — | 0.000 | — | 0 | 0 | 37 |
| AAA | 2154 | 17 | 0.79% | 0.879 | 0.252 | 32.0× | 0.116 | — | 0.000 | — | 0 | 0 | 17 |
| NONE | 20605 | 45 | 0.22% | 0.974 | 0.156 | 71.4× | 0.077 | — | 0.000 | — | 0 | 0 | 45 |

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
