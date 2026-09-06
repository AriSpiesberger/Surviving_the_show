# Held-out validation — v2.4 (raw-feature bag + recent-cohort augmentation)

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
evaluation scores `xp_<event>_h{h}` vs `realized_by_h` on the rows resolved at
that h. The headline below is at **h=6** (the publish horizon); the per-horizon
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
column all use the 3-year debut slice; ceiling events reported at h=6
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

## Headline (ALL bucket, h=6, threshold = 0.60)

| Event | n | base% | AP | lift | AUC | spearman | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TOP_100_PROSPECT | 18863 | 1.60% | **0.543** | 33.9× | 0.969 | 0.204 | 0.814 | 0.189 | 0.306 |
| MLB_DEBUT | 19155 | 13.09% | **0.668** | 5.1× | 0.924 | 0.496 | 0.812 | 0.293 | 0.431 |
| ESTABLISHED_MLB | 19155 | 4.14% | **0.382** | 9.2× | 0.921 | 0.291 | 0.648 | 0.072 | 0.129 |
| STAR_PLUS_ELITE | 19155 | 0.58% | **0.117** | 19.9× | 0.924 | 0.112 | — | 0.000 | — |
| **weighted-AP** | | | **0.476** | | | | | | |

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
| 0-5% | 17,976 | 0.1% | 0.1% | +0.0% |
| 5-10% | 162 | 7.1% | 5.6% | -1.5% |
| 10-20% | 107 | 14.6% | 20.6% | +5.9% |
| 20-30% | 44 | 24.9% | 27.3% | +2.4% |
| 30-40% | 27 | 34.0% | 51.9% | +17.9% |
| 40-50% | 32 | 44.2% | 40.6% | -3.6% |
| 60-70% | 21 | 65.2% | 57.1% | -8.1% |
| 90-100% | 17 | 94.2% | 100.0% | +5.8% |

#### TOP_100_PROSPECT — P(within 6y)

| predicted | n | avg pred | actual | diff |
|---|---:|---:|---:|---:|
| 0-5% | 14,686 | 0.2% | 0.2% | +0.0% |
| 5-10% | 196 | 7.0% | 4.1% | -2.9% |
| 10-20% | 133 | 13.7% | 14.3% | +0.6% |
| 20-30% | 58 | 24.6% | 27.6% | +3.0% |
| 30-40% | 32 | 34.3% | 50.0% | +15.7% |
| 40-50% | 26 | 44.7% | 42.3% | -2.4% |
| 50-60% | 19 | 55.5% | 68.4% | +12.9% |
| 60-70% | 17 | 65.6% | 52.9% | -12.6% |
| 90-100% | 16 | 94.5% | 100.0% | +5.5% |

#### MLB_DEBUT — P(within 3y)

| predicted | n | avg pred | actual | diff |
|---|---:|---:|---:|---:|
| 0-5% | 14,997 | 0.5% | 0.7% | +0.1% |
| 5-10% | 1,079 | 7.2% | 9.3% | +2.0% |
| 10-20% | 981 | 14.2% | 17.7% | +3.5% |
| 20-30% | 442 | 24.4% | 26.2% | +1.9% |
| 30-40% | 284 | 34.7% | 34.5% | -0.2% |
| 40-50% | 195 | 44.9% | 46.7% | +1.7% |
| 50-60% | 171 | 54.5% | 62.0% | +7.5% |
| 60-70% | 131 | 64.9% | 60.3% | -4.6% |
| 70-80% | 114 | 74.9% | 71.9% | -3.0% |
| 80-90% | 122 | 85.4% | 84.4% | -0.9% |
| 90-100% | 157 | 94.9% | 92.4% | -2.6% |

#### MLB_DEBUT — P(within 6y)

| predicted | n | avg pred | actual | diff |
|---|---:|---:|---:|---:|
| 0-5% | 11,097 | 0.6% | 0.7% | +0.1% |
| 5-10% | 1,034 | 7.2% | 10.2% | +2.9% |
| 10-20% | 1,033 | 14.5% | 16.7% | +2.3% |
| 20-30% | 604 | 24.6% | 28.5% | +3.8% |
| 30-40% | 440 | 34.9% | 35.9% | +1.0% |
| 40-50% | 303 | 44.8% | 43.6% | -1.2% |
| 50-60% | 227 | 55.0% | 61.7% | +6.7% |
| 60-70% | 214 | 65.0% | 66.4% | +1.4% |
| 70-80% | 164 | 74.7% | 78.7% | +4.0% |
| 80-90% | 155 | 84.9% | 89.7% | +4.8% |
| 90-100% | 148 | 95.2% | 95.3% | +0.1% |

#### ESTABLISHED_MLB — P(within 3y)

| predicted | n | avg pred | actual | diff |
|---|---:|---:|---:|---:|
| 0-5% | 18,145 | 0.2% | 0.2% | +0.0% |
| 5-10% | 225 | 7.0% | 10.2% | +3.2% |
| 10-20% | 136 | 14.6% | 13.2% | -1.4% |
| 20-30% | 71 | 24.4% | 21.1% | -3.2% |
| 30-40% | 36 | 36.1% | 27.8% | -8.3% |
| 40-50% | 34 | 45.4% | 44.1% | -1.3% |
| 50-60% | 17 | 54.1% | 41.2% | -12.9% |

#### ESTABLISHED_MLB — P(within 6y)

| predicted | n | avg pred | actual | diff |
|---|---:|---:|---:|---:|
| 0-5% | 13,780 | 0.5% | 0.6% | +0.2% |
| 5-10% | 641 | 7.2% | 7.6% | +0.4% |
| 10-20% | 484 | 14.1% | 16.9% | +2.8% |
| 20-30% | 186 | 24.7% | 25.8% | +1.1% |
| 30-40% | 117 | 34.3% | 37.6% | +3.3% |
| 40-50% | 85 | 44.5% | 35.3% | -9.2% |
| 50-60% | 61 | 55.7% | 57.4% | +1.7% |
| 60-70% | 48 | 64.8% | 60.4% | -4.4% |

#### STAR_PLUS_ELITE — P(within 3y)

| predicted | n | avg pred | actual | diff |
|---|---:|---:|---:|---:|
| 0-5% | 18,636 | 0.1% | 0.1% | +0.1% |
| 5-10% | 30 | 7.2% | 3.3% | -3.9% |

#### STAR_PLUS_ELITE — P(within 6y)

| predicted | n | avg pred | actual | diff |
|---|---:|---:|---:|---:|
| 0-5% | 15,094 | 0.2% | 0.2% | -0.0% |
| 5-10% | 173 | 7.0% | 6.4% | -0.6% |
| 10-20% | 107 | 13.6% | 14.0% | +0.4% |
| 20-30% | 35 | 25.1% | 20.0% | -5.1% |

## Per-horizon trajectory (h=1..10, resolved at each h)

#### TOP_100_PROSPECT

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 23519 | 123 | 0.52% | 0.992 | 0.619 | 118.4× | 0.0031 | 0.85 |
| 2 | 22852 | 218 | 0.95% | 0.981 | 0.574 | 60.2× | 0.0059 | 0.83 |
| 3 | 22076 | 276 | 1.25% | 0.975 | 0.552 | 44.2× | 0.0079 | 0.82 |
| 4 | 21158 | 297 | 1.40% | 0.973 | 0.551 | 39.3× | 0.0089 | 0.84 |
| 5 | 20088 | 303 | 1.51% | 0.970 | 0.545 | 36.1× | 0.0096 | 0.85 |
| 6 | 18863 | 302 | 1.60% | 0.969 | 0.543 | 33.9× | 0.0103 | 0.85 |
| 7 | 17518 | 296 | 1.69% | 0.967 | 0.538 | 31.8× | 0.0109 | 0.85 |
| 8 | 16155 | 285 | 1.76% | 0.965 | 0.536 | 30.4× | 0.0114 | 0.84 |
| 9 | 14780 | 272 | 1.84% | 0.963 | 0.531 | 28.9× | 0.0120 | 0.84 |
| 10 | 13404 | 256 | 1.91% | 0.961 | 0.527 | 27.6× | 0.0126 | 0.85 |

#### MLB_DEBUT

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 23864 | 650 | 2.72% | 0.958 | 0.442 | 16.2× | 0.0194 | 0.90 |
| 2 | 23192 | 1281 | 5.52% | 0.946 | 0.556 | 10.1× | 0.0345 | 0.86 |
| 3 | 22409 | 1852 | 8.26% | 0.934 | 0.612 | 7.4× | 0.0479 | 0.86 |
| 4 | 21482 | 2252 | 10.48% | 0.928 | 0.638 | 6.1× | 0.0583 | 0.86 |
| 5 | 20398 | 2456 | 12.04% | 0.925 | 0.654 | 5.4× | 0.0651 | 0.86 |
| 6 | 19155 | 2508 | 13.09% | 0.924 | 0.668 | 5.1× | 0.0691 | 0.86 |
| 7 | 17790 | 2463 | 13.84% | 0.922 | 0.675 | 4.9× | 0.0723 | 0.86 |
| 8 | 16407 | 2362 | 14.40% | 0.920 | 0.673 | 4.7× | 0.0752 | 0.85 |
| 9 | 15013 | 2238 | 14.91% | 0.917 | 0.669 | 4.5× | 0.0781 | 0.84 |
| 10 | 13617 | 2100 | 15.42% | 0.913 | 0.668 | 4.3× | 0.0809 | 0.84 |

#### ESTABLISHED_MLB

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 23864 | 6 | 0.03% | 0.985 | 0.075 | 297.9× | 0.0002 | 0.87 |
| 2 | 23192 | 74 | 0.32% | 0.967 | 0.171 | 53.5× | 0.0029 | 0.72 |
| 3 | 22409 | 238 | 1.06% | 0.955 | 0.267 | 25.1× | 0.0089 | 0.81 |
| 4 | 21482 | 446 | 2.08% | 0.941 | 0.335 | 16.1× | 0.0164 | 0.83 |
| 5 | 20398 | 640 | 3.14% | 0.930 | 0.356 | 11.3× | 0.0242 | 0.83 |
| 6 | 19155 | 793 | 4.14% | 0.921 | 0.382 | 9.2× | 0.0310 | 0.84 |
| 7 | 17790 | 895 | 5.03% | 0.918 | 0.408 | 8.1× | 0.0367 | 0.84 |
| 8 | 16407 | 948 | 5.78% | 0.914 | 0.421 | 7.3× | 0.0416 | 0.84 |
| 9 | 15013 | 962 | 6.41% | 0.909 | 0.426 | 6.6× | 0.0458 | 0.83 |
| 10 | 13617 | 950 | 6.98% | 0.904 | 0.432 | 6.2× | 0.0496 | 0.82 |

#### STAR_PLUS_ELITE

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 23864 | 7 | 0.03% | 0.995 | 0.181 | 616.5× | 0.0003 | 0.43 |
| 2 | 23192 | 19 | 0.08% | 0.954 | 0.090 | 109.7× | 0.0008 | 0.37 |
| 3 | 22409 | 36 | 0.16% | 0.926 | 0.061 | 38.1× | 0.0016 | 0.62 |
| 4 | 21482 | 58 | 0.27% | 0.923 | 0.112 | 41.5× | 0.0025 | 0.88 |
| 5 | 20398 | 86 | 0.42% | 0.925 | 0.115 | 27.3× | 0.0040 | 1.09 |
| 6 | 19155 | 112 | 0.58% | 0.924 | 0.117 | 19.9× | 0.0055 | 1.18 |
| 7 | 17790 | 132 | 0.74% | 0.923 | 0.120 | 16.2× | 0.0070 | 1.23 |
| 8 | 16407 | 147 | 0.90% | 0.923 | 0.126 | 14.0× | 0.0084 | 1.22 |
| 9 | 15013 | 162 | 1.08% | 0.923 | 0.138 | 12.8× | 0.0099 | 1.11 |
| 10 | 13617 | 170 | 1.25% | 0.922 | 0.156 | 12.5× | 0.0113 | 1.02 |

## Per-bucket (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 18863 | 302 | 1.60% | 0.969 | 0.543 | 33.9× | 0.204 | 0.814 | 0.189 | 0.306 | 57 | 13 | 245 |
| R1 | 360 | 109 | 30.28% | 0.911 | 0.821 | 2.7× | 0.655 | 0.821 | 0.422 | 0.558 | 46 | 10 | 63 |
| R2-R3 | 984 | 51 | 5.18% | 0.933 | 0.462 | 8.9× | 0.333 | 0.625 | 0.098 | 0.169 | 5 | 3 | 46 |
| R4-R10 | 4429 | 70 | 1.58% | 0.931 | 0.311 | 19.6× | 0.186 | 1.000 | 0.057 | 0.108 | 4 | 0 | 66 |
| R10+ | 13090 | 72 | 0.55% | 0.960 | 0.352 | 64.1× | 0.118 | 1.000 | 0.028 | 0.054 | 2 | 0 | 70 |

#### MLB_DEBUT

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 19155 | 2508 | 13.09% | 0.924 | 0.668 | 5.1× | 0.496 | 0.812 | 0.293 | 0.431 | 735 | 170 | 1773 |
| R1 | 531 | 366 | 68.93% | 0.852 | 0.908 | 1.3× | 0.565 | 0.883 | 0.724 | 0.796 | 265 | 35 | 101 |
| R2-R3 | 1007 | 372 | 36.94% | 0.882 | 0.799 | 2.2× | 0.638 | 0.806 | 0.503 | 0.619 | 187 | 45 | 185 |
| R4-R10 | 4469 | 857 | 19.18% | 0.893 | 0.643 | 3.4× | 0.536 | 0.766 | 0.233 | 0.358 | 200 | 61 | 657 |
| R10+ | 13148 | 913 | 6.94% | 0.905 | 0.431 | 6.2× | 0.356 | 0.741 | 0.091 | 0.162 | 83 | 29 | 830 |

#### ESTABLISHED_MLB

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 19155 | 793 | 4.14% | 0.921 | 0.382 | 9.2× | 0.291 | 0.648 | 0.072 | 0.129 | 57 | 31 | 736 |
| R1 | 531 | 178 | 33.52% | 0.790 | 0.605 | 1.8× | 0.474 | 0.643 | 0.202 | 0.308 | 36 | 20 | 142 |
| R2-R3 | 1007 | 114 | 11.32% | 0.831 | 0.357 | 3.2× | 0.363 | 0.643 | 0.079 | 0.141 | 9 | 5 | 105 |
| R4-R10 | 4469 | 273 | 6.11% | 0.896 | 0.349 | 5.7× | 0.329 | 0.636 | 0.026 | 0.049 | 7 | 4 | 266 |
| R10+ | 13148 | 228 | 1.73% | 0.898 | 0.196 | 11.3× | 0.180 | 0.714 | 0.022 | 0.043 | 5 | 2 | 223 |

#### STAR_PLUS_ELITE

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 19155 | 112 | 0.58% | 0.924 | 0.117 | 19.9× | 0.112 | — | 0.000 | — | 0 | 0 | 112 |
| R1 | 531 | 40 | 7.53% | 0.731 | 0.164 | 2.2× | 0.211 | — | 0.000 | — | 0 | 0 | 40 |
| R2-R3 | 1007 | 10 | 0.99% | 0.844 | 0.243 | 24.4× | 0.118 | — | 0.000 | — | 0 | 0 | 10 |
| R4-R10 | 4469 | 36 | 0.81% | 0.874 | 0.148 | 18.3× | 0.116 | — | 0.000 | — | 0 | 0 | 36 |
| R10+ | 13148 | 26 | 0.20% | 0.897 | 0.048 | 24.5× | 0.061 | — | 0.000 | — | 0 | 0 | 26 |

## Per-yip (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2695 | 134 | 4.97% | 0.924 | 0.581 | 11.7× | 0.319 | 0.816 | 0.231 | 0.360 | 31 | 7 | 103 |
| 1 | 2507 | 91 | 3.63% | 0.927 | 0.493 | 13.6× | 0.277 | 0.750 | 0.165 | 0.270 | 15 | 5 | 76 |
| 2 | 2299 | 51 | 2.22% | 0.947 | 0.566 | 25.5× | 0.228 | 0.900 | 0.176 | 0.295 | 9 | 1 | 42 |
| 3 | 2031 | 18 | 0.89% | 0.982 | 0.626 | 70.6× | 0.156 | 1.000 | 0.111 | 0.200 | 2 | 0 | 16 |
| 4 | 1794 | 7 | 0.39% | 0.972 | 0.417 | 106.9× | 0.102 | — | 0.000 | — | 0 | 0 | 7 |
| 5 | 1605 | 1 | 0.06% | 0.867 | 0.005 | 7.5× | 0.032 | — | 0.000 | — | 0 | 0 | 1 |
| 6 | 1448 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 7 | 1317 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 8 | 1188 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 9 | 1057 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 922 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

#### MLB_DEBUT

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2701 | 565 | 20.92% | 0.821 | 0.571 | 2.7× | 0.453 | 0.739 | 0.181 | 0.290 | 102 | 36 | 463 |
| 1 | 2551 | 578 | 22.66% | 0.867 | 0.693 | 3.1× | 0.532 | 0.829 | 0.311 | 0.453 | 180 | 37 | 398 |
| 2 | 2359 | 523 | 22.17% | 0.900 | 0.740 | 3.3× | 0.575 | 0.842 | 0.377 | 0.520 | 197 | 37 | 326 |
| 3 | 2093 | 384 | 18.35% | 0.917 | 0.727 | 4.0× | 0.560 | 0.828 | 0.375 | 0.516 | 144 | 30 | 240 |
| 4 | 1838 | 230 | 12.51% | 0.926 | 0.660 | 5.3× | 0.488 | 0.777 | 0.317 | 0.451 | 73 | 21 | 157 |
| 5 | 1630 | 125 | 7.67% | 0.939 | 0.587 | 7.7× | 0.405 | 0.893 | 0.200 | 0.327 | 25 | 3 | 100 |
| 6 | 1463 | 63 | 4.31% | 0.955 | 0.483 | 11.2× | 0.320 | 0.750 | 0.143 | 0.240 | 9 | 3 | 54 |
| 7 | 1329 | 27 | 2.03% | 0.973 | 0.456 | 22.5× | 0.231 | 0.571 | 0.148 | 0.235 | 4 | 3 | 23 |
| 8 | 1196 | 9 | 0.75% | 0.980 | 0.396 | 52.6× | 0.144 | 1.000 | 0.111 | 0.200 | 1 | 0 | 8 |
| 9 | 1065 | 3 | 0.28% | 0.983 | 0.102 | 36.1× | 0.089 | — | 0.000 | — | 0 | 0 | 3 |
| 10 | 930 | 1 | 0.11% | 0.998 | 0.333 | 310.0× | 0.057 | — | 0.000 | — | 0 | 0 | 1 |

#### ESTABLISHED_MLB

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2701 | 164 | 6.07% | 0.837 | 0.299 | 4.9× | 0.278 | 0.700 | 0.043 | 0.080 | 7 | 3 | 157 |
| 1 | 2551 | 201 | 7.88% | 0.874 | 0.402 | 5.1× | 0.349 | 0.613 | 0.095 | 0.164 | 19 | 12 | 182 |
| 2 | 2359 | 191 | 8.10% | 0.888 | 0.451 | 5.6× | 0.367 | 0.694 | 0.131 | 0.220 | 25 | 11 | 166 |
| 3 | 2093 | 120 | 5.73% | 0.906 | 0.415 | 7.2× | 0.327 | 0.500 | 0.042 | 0.077 | 5 | 5 | 115 |
| 4 | 1838 | 72 | 3.92% | 0.925 | 0.455 | 11.6× | 0.286 | 1.000 | 0.014 | 0.027 | 1 | 0 | 71 |
| 5 | 1630 | 31 | 1.90% | 0.915 | 0.185 | 9.7× | 0.196 | — | 0.000 | — | 0 | 0 | 31 |
| 6 | 1463 | 11 | 0.75% | 0.946 | 0.093 | 12.4× | 0.133 | — | 0.000 | — | 0 | 0 | 11 |
| 7 | 1329 | 3 | 0.23% | 0.956 | 0.032 | 14.1× | 0.075 | — | 0.000 | — | 0 | 0 | 3 |
| 8 | 1196 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 9 | 1065 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 930 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

#### STAR_PLUS_ELITE

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2701 | 26 | 0.96% | 0.819 | 0.067 | 7.0× | 0.108 | — | 0.000 | — | 0 | 0 | 26 |
| 1 | 2551 | 33 | 1.29% | 0.873 | 0.169 | 13.1× | 0.146 | — | 0.000 | — | 0 | 0 | 33 |
| 2 | 2359 | 25 | 1.06% | 0.900 | 0.160 | 15.1× | 0.142 | — | 0.000 | — | 0 | 0 | 25 |
| 3 | 2093 | 18 | 0.86% | 0.899 | 0.138 | 16.0× | 0.128 | — | 0.000 | — | 0 | 0 | 18 |
| 4 | 1838 | 9 | 0.49% | 0.902 | 0.092 | 18.7× | 0.097 | — | 0.000 | — | 0 | 0 | 9 |
| 5 | 1630 | 1 | 0.06% | 0.996 | 0.125 | 203.8× | 0.043 | — | 0.000 | — | 0 | 0 | 1 |
| 6 | 1463 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 7 | 1329 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 8 | 1196 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 9 | 1065 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 930 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

## Per-level (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 18863 | 302 | 1.60% | 0.969 | 0.543 | 33.9× | 0.204 | 0.814 | 0.189 | 0.306 | 57 | 13 | 245 |
| RK | 1490 | 50 | 3.36% | 0.950 | 0.589 | 17.5× | 0.281 | 0.800 | 0.240 | 0.369 | 12 | 3 | 38 |
| A- | 1065 | 20 | 1.88% | 0.960 | 0.491 | 26.2× | 0.216 | 0.750 | 0.150 | 0.250 | 3 | 1 | 17 |
| A | 1449 | 44 | 3.04% | 0.958 | 0.621 | 20.5× | 0.272 | 0.750 | 0.341 | 0.469 | 15 | 5 | 29 |
| A+ | 1457 | 40 | 2.75% | 0.968 | 0.613 | 22.3× | 0.265 | 0.900 | 0.225 | 0.360 | 9 | 1 | 31 |
| AA | 1312 | 24 | 1.83% | 0.987 | 0.697 | 38.1× | 0.226 | 0.875 | 0.292 | 0.438 | 7 | 1 | 17 |
| AAA | 1051 | 12 | 1.14% | 0.997 | 0.778 | 68.2× | 0.183 | 0.750 | 0.250 | 0.375 | 3 | 1 | 9 |
| NONE | 11005 | 112 | 1.02% | 0.969 | 0.441 | 43.4× | 0.163 | 0.889 | 0.071 | 0.132 | 8 | 1 | 104 |

#### MLB_DEBUT

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 19155 | 2508 | 13.09% | 0.924 | 0.668 | 5.1× | 0.496 | 0.812 | 0.293 | 0.431 | 735 | 170 | 1773 |
| RK | 1496 | 172 | 11.50% | 0.830 | 0.452 | 3.9× | 0.364 | 0.607 | 0.099 | 0.170 | 17 | 11 | 155 |
| A- | 1067 | 168 | 15.75% | 0.843 | 0.543 | 3.5× | 0.432 | 0.731 | 0.113 | 0.196 | 19 | 7 | 149 |
| A | 1474 | 291 | 19.74% | 0.850 | 0.659 | 3.3× | 0.483 | 0.829 | 0.316 | 0.458 | 92 | 19 | 199 |
| A+ | 1480 | 309 | 20.88% | 0.870 | 0.696 | 3.3× | 0.521 | 0.826 | 0.369 | 0.510 | 114 | 24 | 195 |
| AA | 1390 | 461 | 33.17% | 0.865 | 0.774 | 2.3× | 0.596 | 0.826 | 0.443 | 0.576 | 204 | 43 | 257 |
| AAA | 1113 | 353 | 31.72% | 0.858 | 0.762 | 2.4× | 0.578 | 0.819 | 0.462 | 0.591 | 163 | 36 | 190 |
| NONE | 11093 | 753 | 6.79% | 0.961 | 0.617 | 9.1× | 0.402 | 0.833 | 0.166 | 0.277 | 125 | 25 | 628 |

#### ESTABLISHED_MLB

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 19155 | 793 | 4.14% | 0.921 | 0.382 | 9.2× | 0.291 | 0.648 | 0.072 | 0.129 | 57 | 31 | 736 |
| RK | 1496 | 33 | 2.21% | 0.905 | 0.231 | 10.5× | 0.206 | 1.000 | 0.030 | 0.059 | 1 | 0 | 32 |
| A- | 1067 | 41 | 3.84% | 0.872 | 0.297 | 7.7× | 0.247 | 1.000 | 0.024 | 0.048 | 1 | 0 | 40 |
| A | 1474 | 72 | 4.88% | 0.869 | 0.339 | 6.9× | 0.276 | 0.750 | 0.083 | 0.150 | 6 | 2 | 66 |
| A+ | 1480 | 100 | 6.76% | 0.893 | 0.396 | 5.9× | 0.342 | 0.600 | 0.060 | 0.109 | 6 | 4 | 94 |
| AA | 1390 | 155 | 11.15% | 0.860 | 0.464 | 4.2× | 0.392 | 0.594 | 0.123 | 0.203 | 19 | 13 | 136 |
| AAA | 1113 | 105 | 9.43% | 0.871 | 0.475 | 5.0× | 0.376 | 0.680 | 0.162 | 0.262 | 17 | 8 | 88 |
| NONE | 11093 | 286 | 2.58% | 0.952 | 0.354 | 13.7× | 0.248 | 0.667 | 0.021 | 0.041 | 6 | 3 | 280 |

#### STAR_PLUS_ELITE

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 19155 | 112 | 0.58% | 0.924 | 0.117 | 19.9× | 0.112 | — | 0.000 | — | 0 | 0 | 112 |
| RK | 1496 | 8 | 0.53% | 0.915 | 0.043 | 8.0× | 0.105 | — | 0.000 | — | 0 | 0 | 8 |
| A- | 1067 | 7 | 0.66% | 0.836 | 0.266 | 40.6× | 0.094 | — | 0.000 | — | 0 | 0 | 7 |
| A | 1474 | 12 | 0.81% | 0.948 | 0.169 | 20.7× | 0.140 | — | 0.000 | — | 0 | 0 | 12 |
| A+ | 1480 | 12 | 0.81% | 0.882 | 0.081 | 10.0× | 0.119 | — | 0.000 | — | 0 | 0 | 12 |
| AA | 1390 | 24 | 1.73% | 0.885 | 0.152 | 8.8× | 0.174 | — | 0.000 | — | 0 | 0 | 24 |
| AAA | 1113 | 14 | 1.26% | 0.952 | 0.352 | 28.0× | 0.175 | — | 0.000 | — | 0 | 0 | 14 |
| NONE | 11093 | 34 | 0.31% | 0.943 | 0.063 | 20.6× | 0.085 | — | 0.000 | — | 0 | 0 | 34 |

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
