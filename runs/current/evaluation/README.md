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
| TOP_100_PROSPECT | 18863 | 1.60% | **0.540** | 33.7× | 0.971 | 0.205 | 0.812 | 0.185 | 0.302 |
| MLB_DEBUT | 19155 | 13.09% | **0.667** | 5.1× | 0.924 | 0.496 | 0.816 | 0.286 | 0.424 |
| ESTABLISHED_MLB | 19155 | 4.14% | **0.380** | 9.2× | 0.921 | 0.291 | 0.710 | 0.055 | 0.103 |
| STAR_PLUS_ELITE | 19155 | 0.58% | **0.104** | 17.9× | 0.920 | 0.111 | — | 0.000 | — |
| **weighted-AP** | | | **0.472** | | | | | | |

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
| 0-5% | 17,985 | 0.1% | 0.1% | +0.0% |
| 5-10% | 156 | 7.0% | 3.8% | -3.1% |
| 10-20% | 105 | 14.0% | 21.0% | +6.9% |
| 20-30% | 54 | 24.7% | 37.0% | +12.3% |
| 30-40% | 23 | 35.3% | 39.1% | +3.8% |
| 40-50% | 20 | 44.6% | 45.0% | +0.4% |
| 50-60% | 20 | 53.9% | 65.0% | +11.1% |
| 60-70% | 18 | 65.2% | 61.1% | -4.1% |
| 90-100% | 16 | 94.3% | 100.0% | +5.7% |

#### TOP_100_PROSPECT — P(within 6y)

| predicted | n | avg pred | actual | diff |
|---|---:|---:|---:|---:|
| 0-5% | 14,704 | 0.2% | 0.2% | +0.0% |
| 5-10% | 199 | 7.2% | 4.5% | -2.7% |
| 10-20% | 125 | 14.2% | 16.0% | +1.8% |
| 20-30% | 46 | 24.6% | 32.6% | +8.0% |
| 30-40% | 37 | 34.8% | 45.9% | +11.1% |
| 40-50% | 17 | 44.2% | 58.8% | +14.6% |
| 50-60% | 22 | 53.5% | 59.1% | +5.6% |
| 60-70% | 16 | 64.2% | 68.8% | +4.5% |
| 90-100% | 15 | 94.6% | 100.0% | +5.4% |

#### MLB_DEBUT — P(within 3y)

| predicted | n | avg pred | actual | diff |
|---|---:|---:|---:|---:|
| 0-5% | 15,007 | 0.5% | 0.7% | +0.1% |
| 5-10% | 1,095 | 7.3% | 9.3% | +2.0% |
| 10-20% | 977 | 14.3% | 18.5% | +4.2% |
| 20-30% | 441 | 24.4% | 24.9% | +0.5% |
| 30-40% | 266 | 34.7% | 36.8% | +2.1% |
| 40-50% | 195 | 44.6% | 46.7% | +2.1% |
| 50-60% | 185 | 54.9% | 62.7% | +7.8% |
| 60-70% | 123 | 65.0% | 60.2% | -4.9% |
| 70-80% | 98 | 74.8% | 71.4% | -3.4% |
| 80-90% | 143 | 84.9% | 85.3% | +0.4% |
| 90-100% | 143 | 95.3% | 92.3% | -3.0% |

#### MLB_DEBUT — P(within 6y)

| predicted | n | avg pred | actual | diff |
|---|---:|---:|---:|---:|
| 0-5% | 11,122 | 0.6% | 0.7% | +0.1% |
| 5-10% | 998 | 7.2% | 9.5% | +2.3% |
| 10-20% | 1,052 | 14.4% | 17.8% | +3.4% |
| 20-30% | 614 | 24.5% | 30.1% | +5.6% |
| 30-40% | 432 | 34.6% | 33.1% | -1.5% |
| 40-50% | 310 | 44.8% | 47.1% | +2.3% |
| 50-60% | 235 | 55.1% | 61.3% | +6.1% |
| 60-70% | 203 | 64.8% | 66.5% | +1.7% |
| 70-80% | 156 | 75.0% | 82.1% | +7.1% |
| 80-90% | 162 | 84.7% | 87.0% | +2.3% |
| 90-100% | 135 | 95.3% | 95.6% | +0.2% |

#### ESTABLISHED_MLB — P(within 3y)

| predicted | n | avg pred | actual | diff |
|---|---:|---:|---:|---:|
| 0-5% | 18,171 | 0.2% | 0.2% | +0.0% |
| 5-10% | 199 | 6.9% | 10.1% | +3.1% |
| 10-20% | 150 | 14.0% | 12.7% | -1.3% |
| 20-30% | 57 | 25.3% | 21.1% | -4.2% |
| 30-40% | 37 | 34.7% | 27.0% | -7.7% |
| 40-50% | 44 | 44.6% | 47.7% | +3.1% |

#### ESTABLISHED_MLB — P(within 6y)

| predicted | n | avg pred | actual | diff |
|---|---:|---:|---:|---:|
| 0-5% | 13,831 | 0.5% | 0.7% | +0.2% |
| 5-10% | 627 | 7.2% | 8.0% | +0.7% |
| 10-20% | 462 | 14.1% | 15.8% | +1.7% |
| 20-30% | 198 | 24.6% | 28.8% | +4.2% |
| 30-40% | 111 | 34.5% | 36.9% | +2.5% |
| 40-50% | 81 | 44.5% | 45.7% | +1.2% |
| 50-60% | 63 | 55.1% | 52.4% | -2.7% |
| 60-70% | 37 | 63.5% | 70.3% | +6.8% |

#### STAR_PLUS_ELITE — P(within 3y)

| predicted | n | avg pred | actual | diff |
|---|---:|---:|---:|---:|
| 0-5% | 18,636 | 0.1% | 0.1% | +0.1% |
| 5-10% | 29 | 6.8% | 6.9% | +0.1% |

#### STAR_PLUS_ELITE — P(within 6y)

| predicted | n | avg pred | actual | diff |
|---|---:|---:|---:|---:|
| 0-5% | 15,132 | 0.2% | 0.2% | +0.0% |
| 5-10% | 174 | 7.2% | 8.6% | +1.4% |
| 10-20% | 83 | 13.3% | 15.7% | +2.4% |
| 20-30% | 22 | 24.7% | 4.5% | -20.2% |

## Per-horizon trajectory (h=1..10, resolved at each h)

#### TOP_100_PROSPECT

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 23519 | 123 | 0.52% | 0.992 | 0.614 | 117.3× | 0.0031 | 0.85 |
| 2 | 22852 | 218 | 0.95% | 0.983 | 0.570 | 59.8× | 0.0059 | 0.83 |
| 3 | 22076 | 276 | 1.25% | 0.977 | 0.550 | 44.0× | 0.0080 | 0.82 |
| 4 | 21158 | 297 | 1.40% | 0.975 | 0.546 | 38.9× | 0.0090 | 0.83 |
| 5 | 20088 | 303 | 1.51% | 0.973 | 0.541 | 35.9× | 0.0097 | 0.84 |
| 6 | 18863 | 302 | 1.60% | 0.971 | 0.540 | 33.7× | 0.0103 | 0.85 |
| 7 | 17518 | 296 | 1.69% | 0.969 | 0.534 | 31.6× | 0.0110 | 0.84 |
| 8 | 16155 | 285 | 1.76% | 0.968 | 0.533 | 30.2× | 0.0115 | 0.84 |
| 9 | 14780 | 272 | 1.84% | 0.966 | 0.530 | 28.8× | 0.0120 | 0.84 |
| 10 | 13404 | 256 | 1.91% | 0.964 | 0.526 | 27.6× | 0.0126 | 0.85 |

#### MLB_DEBUT

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 23864 | 650 | 2.72% | 0.959 | 0.443 | 16.3× | 0.0194 | 0.91 |
| 2 | 23192 | 1281 | 5.52% | 0.946 | 0.556 | 10.1× | 0.0345 | 0.88 |
| 3 | 22409 | 1852 | 8.26% | 0.935 | 0.614 | 7.4× | 0.0478 | 0.87 |
| 4 | 21482 | 2252 | 10.48% | 0.928 | 0.637 | 6.1× | 0.0583 | 0.86 |
| 5 | 20398 | 2456 | 12.04% | 0.925 | 0.654 | 5.4× | 0.0651 | 0.86 |
| 6 | 19155 | 2508 | 13.09% | 0.924 | 0.667 | 5.1× | 0.0692 | 0.86 |
| 7 | 17790 | 2463 | 13.84% | 0.923 | 0.673 | 4.9× | 0.0724 | 0.86 |
| 8 | 16407 | 2362 | 14.40% | 0.920 | 0.672 | 4.7× | 0.0753 | 0.85 |
| 9 | 15013 | 2238 | 14.91% | 0.917 | 0.668 | 4.5× | 0.0781 | 0.84 |
| 10 | 13617 | 2100 | 15.42% | 0.914 | 0.666 | 4.3× | 0.0810 | 0.84 |

#### ESTABLISHED_MLB

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 23864 | 6 | 0.03% | 0.986 | 0.067 | 264.7× | 0.0002 | 0.81 |
| 2 | 23192 | 74 | 0.32% | 0.969 | 0.178 | 55.8× | 0.0029 | 0.71 |
| 3 | 22409 | 238 | 1.06% | 0.956 | 0.270 | 25.4× | 0.0089 | 0.80 |
| 4 | 21482 | 446 | 2.08% | 0.942 | 0.337 | 16.2× | 0.0164 | 0.81 |
| 5 | 20398 | 640 | 3.14% | 0.931 | 0.359 | 11.4× | 0.0241 | 0.80 |
| 6 | 19155 | 793 | 4.14% | 0.921 | 0.380 | 9.2× | 0.0311 | 0.81 |
| 7 | 17790 | 895 | 5.03% | 0.917 | 0.406 | 8.1× | 0.0369 | 0.81 |
| 8 | 16407 | 948 | 5.78% | 0.914 | 0.420 | 7.3× | 0.0417 | 0.82 |
| 9 | 15013 | 962 | 6.41% | 0.909 | 0.421 | 6.6× | 0.0461 | 0.81 |
| 10 | 13617 | 950 | 6.98% | 0.904 | 0.428 | 6.1× | 0.0499 | 0.80 |

#### STAR_PLUS_ELITE

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 23864 | 7 | 0.03% | 0.995 | 0.103 | 351.3× | 0.0003 | 0.57 |
| 2 | 23192 | 19 | 0.08% | 0.954 | 0.091 | 110.9× | 0.0008 | 0.45 |
| 3 | 22409 | 36 | 0.16% | 0.920 | 0.064 | 40.1× | 0.0016 | 0.69 |
| 4 | 21482 | 58 | 0.27% | 0.918 | 0.109 | 40.4× | 0.0026 | 0.88 |
| 5 | 20398 | 86 | 0.42% | 0.922 | 0.108 | 25.5× | 0.0040 | 1.01 |
| 6 | 19155 | 112 | 0.58% | 0.920 | 0.104 | 17.9× | 0.0055 | 1.07 |
| 7 | 17790 | 132 | 0.74% | 0.921 | 0.107 | 14.4× | 0.0070 | 1.10 |
| 8 | 16407 | 147 | 0.90% | 0.921 | 0.114 | 12.7× | 0.0084 | 1.09 |
| 9 | 15013 | 162 | 1.08% | 0.921 | 0.125 | 11.6× | 0.0100 | 1.02 |
| 10 | 13617 | 170 | 1.25% | 0.920 | 0.139 | 11.1× | 0.0114 | 0.97 |

## Per-bucket (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 18863 | 302 | 1.60% | 0.971 | 0.540 | 33.7× | 0.205 | 0.812 | 0.185 | 0.302 | 56 | 13 | 246 |
| R1 | 360 | 109 | 30.28% | 0.908 | 0.809 | 2.7× | 0.650 | 0.800 | 0.440 | 0.568 | 48 | 12 | 61 |
| R2-R3 | 984 | 51 | 5.18% | 0.939 | 0.495 | 9.5× | 0.337 | 0.800 | 0.078 | 0.143 | 4 | 1 | 47 |
| R4-R10 | 4429 | 70 | 1.58% | 0.932 | 0.284 | 18.0× | 0.187 | 1.000 | 0.043 | 0.082 | 3 | 0 | 67 |
| R10+ | 13090 | 72 | 0.55% | 0.967 | 0.346 | 63.0× | 0.120 | 1.000 | 0.014 | 0.027 | 1 | 0 | 71 |

#### MLB_DEBUT

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 19155 | 2508 | 13.09% | 0.924 | 0.667 | 5.1× | 0.496 | 0.816 | 0.286 | 0.424 | 718 | 162 | 1790 |
| R1 | 531 | 366 | 68.93% | 0.855 | 0.910 | 1.3× | 0.569 | 0.886 | 0.724 | 0.797 | 265 | 34 | 101 |
| R2-R3 | 1007 | 372 | 36.94% | 0.879 | 0.795 | 2.2× | 0.633 | 0.800 | 0.462 | 0.586 | 172 | 43 | 200 |
| R4-R10 | 4469 | 857 | 19.18% | 0.893 | 0.641 | 3.3× | 0.536 | 0.778 | 0.229 | 0.353 | 196 | 56 | 661 |
| R10+ | 13148 | 913 | 6.94% | 0.906 | 0.431 | 6.2× | 0.357 | 0.746 | 0.093 | 0.166 | 85 | 29 | 828 |

#### ESTABLISHED_MLB

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 19155 | 793 | 4.14% | 0.921 | 0.380 | 9.2× | 0.291 | 0.710 | 0.055 | 0.103 | 44 | 18 | 749 |
| R1 | 531 | 178 | 33.52% | 0.790 | 0.599 | 1.8× | 0.475 | 0.676 | 0.140 | 0.233 | 25 | 12 | 153 |
| R2-R3 | 1007 | 114 | 11.32% | 0.830 | 0.361 | 3.2× | 0.363 | 0.636 | 0.061 | 0.112 | 7 | 4 | 107 |
| R4-R10 | 4469 | 273 | 6.11% | 0.895 | 0.357 | 5.8× | 0.327 | 0.889 | 0.029 | 0.057 | 8 | 1 | 265 |
| R10+ | 13148 | 228 | 1.73% | 0.899 | 0.191 | 11.0× | 0.180 | 0.800 | 0.018 | 0.034 | 4 | 1 | 224 |

#### STAR_PLUS_ELITE

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 19155 | 112 | 0.58% | 0.920 | 0.104 | 17.9× | 0.111 | — | 0.000 | — | 0 | 0 | 112 |
| R1 | 531 | 40 | 7.53% | 0.695 | 0.133 | 1.8× | 0.178 | — | 0.000 | — | 0 | 0 | 40 |
| R2-R3 | 1007 | 10 | 0.99% | 0.843 | 0.239 | 24.0× | 0.118 | — | 0.000 | — | 0 | 0 | 10 |
| R4-R10 | 4469 | 36 | 0.81% | 0.879 | 0.129 | 16.0× | 0.117 | — | 0.000 | — | 0 | 0 | 36 |
| R10+ | 13148 | 26 | 0.20% | 0.889 | 0.060 | 30.1× | 0.060 | — | 0.000 | — | 0 | 0 | 26 |

## Per-yip (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2695 | 134 | 4.97% | 0.926 | 0.577 | 11.6× | 0.321 | 0.810 | 0.254 | 0.386 | 34 | 8 | 100 |
| 1 | 2507 | 91 | 3.63% | 0.932 | 0.501 | 13.8× | 0.280 | 0.737 | 0.154 | 0.255 | 14 | 5 | 77 |
| 2 | 2299 | 51 | 2.22% | 0.952 | 0.569 | 25.6× | 0.230 | 1.000 | 0.118 | 0.211 | 6 | 0 | 45 |
| 3 | 2031 | 18 | 0.89% | 0.985 | 0.663 | 74.8× | 0.158 | 1.000 | 0.111 | 0.200 | 2 | 0 | 16 |
| 4 | 1794 | 7 | 0.39% | 0.981 | 0.402 | 103.1× | 0.104 | — | 0.000 | — | 0 | 0 | 7 |
| 5 | 1605 | 1 | 0.06% | 0.923 | 0.008 | 12.8× | 0.037 | — | 0.000 | — | 0 | 0 | 1 |
| 6 | 1448 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 7 | 1317 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 8 | 1188 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 9 | 1057 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 922 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

#### MLB_DEBUT

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2701 | 565 | 20.92% | 0.822 | 0.574 | 2.7× | 0.454 | 0.760 | 0.173 | 0.282 | 98 | 31 | 467 |
| 1 | 2551 | 578 | 22.66% | 0.867 | 0.693 | 3.1× | 0.532 | 0.837 | 0.301 | 0.443 | 174 | 34 | 404 |
| 2 | 2359 | 523 | 22.17% | 0.899 | 0.736 | 3.3× | 0.574 | 0.838 | 0.375 | 0.518 | 196 | 38 | 327 |
| 3 | 2093 | 384 | 18.35% | 0.917 | 0.723 | 3.9× | 0.560 | 0.828 | 0.375 | 0.516 | 144 | 30 | 240 |
| 4 | 1838 | 230 | 12.51% | 0.927 | 0.662 | 5.3× | 0.490 | 0.772 | 0.309 | 0.441 | 71 | 21 | 159 |
| 5 | 1630 | 125 | 7.67% | 0.939 | 0.583 | 7.6× | 0.405 | 0.828 | 0.192 | 0.312 | 24 | 5 | 101 |
| 6 | 1463 | 63 | 4.31% | 0.955 | 0.502 | 11.7× | 0.320 | 0.778 | 0.111 | 0.194 | 7 | 2 | 56 |
| 7 | 1329 | 27 | 2.03% | 0.974 | 0.456 | 22.5× | 0.232 | 0.750 | 0.111 | 0.194 | 3 | 1 | 24 |
| 8 | 1196 | 9 | 0.75% | 0.982 | 0.423 | 56.2× | 0.144 | 1.000 | 0.111 | 0.200 | 1 | 0 | 8 |
| 9 | 1065 | 3 | 0.28% | 0.987 | 0.138 | 49.0× | 0.089 | — | 0.000 | — | 0 | 0 | 3 |
| 10 | 930 | 1 | 0.11% | 0.999 | 0.500 | 465.0× | 0.057 | — | 0.000 | — | 0 | 0 | 1 |

#### ESTABLISHED_MLB

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2701 | 164 | 6.07% | 0.836 | 0.296 | 4.9× | 0.278 | 0.636 | 0.043 | 0.080 | 7 | 4 | 157 |
| 1 | 2551 | 201 | 7.88% | 0.872 | 0.396 | 5.0× | 0.347 | 0.565 | 0.065 | 0.116 | 13 | 10 | 188 |
| 2 | 2359 | 191 | 8.10% | 0.888 | 0.450 | 5.6× | 0.367 | 0.889 | 0.084 | 0.153 | 16 | 2 | 175 |
| 3 | 2093 | 120 | 5.73% | 0.903 | 0.429 | 7.5× | 0.325 | 0.778 | 0.058 | 0.109 | 7 | 2 | 113 |
| 4 | 1838 | 72 | 3.92% | 0.927 | 0.451 | 11.5× | 0.287 | 1.000 | 0.014 | 0.027 | 1 | 0 | 71 |
| 5 | 1630 | 31 | 1.90% | 0.918 | 0.190 | 10.0× | 0.198 | — | 0.000 | — | 0 | 0 | 31 |
| 6 | 1463 | 11 | 0.75% | 0.949 | 0.114 | 15.1× | 0.134 | — | 0.000 | — | 0 | 0 | 11 |
| 7 | 1329 | 3 | 0.23% | 0.957 | 0.033 | 14.8× | 0.075 | — | 0.000 | — | 0 | 0 | 3 |
| 8 | 1196 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 9 | 1065 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 930 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

#### STAR_PLUS_ELITE

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2701 | 26 | 0.96% | 0.822 | 0.062 | 6.4× | 0.109 | — | 0.000 | — | 0 | 0 | 26 |
| 1 | 2551 | 33 | 1.29% | 0.865 | 0.165 | 12.8× | 0.143 | — | 0.000 | — | 0 | 0 | 33 |
| 2 | 2359 | 25 | 1.06% | 0.895 | 0.136 | 12.8× | 0.140 | — | 0.000 | — | 0 | 0 | 25 |
| 3 | 2093 | 18 | 0.86% | 0.895 | 0.119 | 13.8× | 0.126 | — | 0.000 | — | 0 | 0 | 18 |
| 4 | 1838 | 9 | 0.49% | 0.900 | 0.134 | 27.5× | 0.097 | — | 0.000 | — | 0 | 0 | 9 |
| 5 | 1630 | 1 | 0.06% | 0.999 | 0.500 | 815.0× | 0.043 | — | 0.000 | — | 0 | 0 | 1 |
| 6 | 1463 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 7 | 1329 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 8 | 1196 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 9 | 1065 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 930 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

## Per-level (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 18863 | 302 | 1.60% | 0.971 | 0.540 | 33.7× | 0.205 | 0.812 | 0.185 | 0.302 | 56 | 13 | 246 |
| RK | 1490 | 50 | 3.36% | 0.951 | 0.588 | 17.5× | 0.281 | 0.765 | 0.260 | 0.388 | 13 | 4 | 37 |
| A- | 1065 | 20 | 1.88% | 0.961 | 0.503 | 26.8× | 0.217 | 0.800 | 0.200 | 0.320 | 4 | 1 | 16 |
| A | 1449 | 44 | 3.04% | 0.958 | 0.625 | 20.6× | 0.272 | 0.750 | 0.341 | 0.469 | 15 | 5 | 29 |
| A+ | 1457 | 40 | 2.75% | 0.969 | 0.625 | 22.8× | 0.265 | 1.000 | 0.200 | 0.333 | 8 | 0 | 32 |
| AA | 1312 | 24 | 1.83% | 0.988 | 0.671 | 36.7× | 0.227 | 0.833 | 0.208 | 0.333 | 5 | 1 | 19 |
| AAA | 1051 | 12 | 1.14% | 0.997 | 0.812 | 71.1× | 0.183 | 1.000 | 0.083 | 0.154 | 1 | 0 | 11 |
| NONE | 11005 | 112 | 1.02% | 0.971 | 0.439 | 43.1× | 0.164 | 0.833 | 0.089 | 0.161 | 10 | 2 | 102 |

#### MLB_DEBUT

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 19155 | 2508 | 13.09% | 0.924 | 0.667 | 5.1× | 0.496 | 0.816 | 0.286 | 0.424 | 718 | 162 | 1790 |
| RK | 1496 | 172 | 11.50% | 0.827 | 0.444 | 3.9× | 0.362 | 0.615 | 0.093 | 0.162 | 16 | 10 | 156 |
| A- | 1067 | 168 | 15.75% | 0.844 | 0.555 | 3.5× | 0.435 | 0.885 | 0.137 | 0.237 | 23 | 3 | 145 |
| A | 1474 | 291 | 19.74% | 0.851 | 0.657 | 3.3× | 0.484 | 0.838 | 0.285 | 0.426 | 83 | 16 | 208 |
| A+ | 1480 | 309 | 20.88% | 0.869 | 0.692 | 3.3× | 0.520 | 0.831 | 0.366 | 0.508 | 113 | 23 | 196 |
| AA | 1390 | 461 | 33.17% | 0.867 | 0.776 | 2.3× | 0.599 | 0.829 | 0.443 | 0.577 | 204 | 42 | 257 |
| AAA | 1113 | 353 | 31.72% | 0.860 | 0.763 | 2.4× | 0.581 | 0.809 | 0.445 | 0.574 | 157 | 37 | 196 |
| NONE | 11093 | 753 | 6.79% | 0.961 | 0.614 | 9.0× | 0.402 | 0.829 | 0.161 | 0.269 | 121 | 25 | 632 |

#### ESTABLISHED_MLB

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 19155 | 793 | 4.14% | 0.921 | 0.380 | 9.2× | 0.291 | 0.710 | 0.055 | 0.103 | 44 | 18 | 749 |
| RK | 1496 | 33 | 2.21% | 0.900 | 0.207 | 9.4× | 0.203 | 1.000 | 0.030 | 0.059 | 1 | 0 | 32 |
| A- | 1067 | 41 | 3.84% | 0.872 | 0.308 | 8.0× | 0.248 | 1.000 | 0.024 | 0.048 | 1 | 0 | 40 |
| A | 1474 | 72 | 4.88% | 0.870 | 0.335 | 6.8× | 0.276 | 0.667 | 0.056 | 0.103 | 4 | 2 | 68 |
| A+ | 1480 | 100 | 6.76% | 0.895 | 0.393 | 5.8× | 0.343 | 0.571 | 0.040 | 0.075 | 4 | 3 | 96 |
| AA | 1390 | 155 | 11.15% | 0.861 | 0.469 | 4.2× | 0.394 | 0.762 | 0.103 | 0.182 | 16 | 5 | 139 |
| AAA | 1113 | 105 | 9.43% | 0.873 | 0.485 | 5.1× | 0.378 | 0.750 | 0.114 | 0.198 | 12 | 4 | 93 |
| NONE | 11093 | 286 | 2.58% | 0.951 | 0.342 | 13.2× | 0.248 | 0.625 | 0.017 | 0.034 | 5 | 3 | 281 |

#### STAR_PLUS_ELITE

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 19155 | 112 | 0.58% | 0.920 | 0.104 | 17.9× | 0.111 | — | 0.000 | — | 0 | 0 | 112 |
| RK | 1496 | 8 | 0.53% | 0.904 | 0.043 | 8.0× | 0.102 | — | 0.000 | — | 0 | 0 | 8 |
| A- | 1067 | 7 | 0.66% | 0.856 | 0.323 | 49.2× | 0.100 | — | 0.000 | — | 0 | 0 | 7 |
| A | 1474 | 12 | 0.81% | 0.942 | 0.153 | 18.9× | 0.137 | — | 0.000 | — | 0 | 0 | 12 |
| A+ | 1480 | 12 | 0.81% | 0.880 | 0.090 | 11.1× | 0.118 | — | 0.000 | — | 0 | 0 | 12 |
| AA | 1390 | 24 | 1.73% | 0.888 | 0.142 | 8.2× | 0.175 | — | 0.000 | — | 0 | 0 | 24 |
| AAA | 1113 | 14 | 1.26% | 0.953 | 0.240 | 19.1× | 0.175 | — | 0.000 | — | 0 | 0 | 14 |
| NONE | 11093 | 34 | 0.31% | 0.940 | 0.057 | 18.6× | 0.084 | — | 0.000 | — | 0 | 0 | 34 |

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
