# Held-out validation — v2.1c conditional refinement

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
| Hazards (per-fold OOF, eval) | `scratch/v20b_oof/fold[0-5]_hazards.pkl` | Each fold trained on the OTHER 5 (val pids excluded). HistGBT, default HP, 314 features (incl. 76 scouting). Survival → censoring-aware. |
| Hazards (production) | `models/event_classifiers_v2.0b_prod.pkl` | 100% of ≤2020 data. Scores the 2026 cohort (entry 2024–26 — not in training, so no leakage). |
| Conditional joint XGB | `models/joint_xgb_v2.0b_{oof,prod}.pkl` (`fit_joint_xgb_cond.py`) | OOF stacked, expanded to resolved `(row, h)` pairs for h=1..10. `multi_output_tree` over the 4 heads; per-horizon censoring built in (no `--censor-window`). Outputs `P(event by snap+h)`; monotone in h via cummax at inference. |
| Timing | `models/time_to_debut_v2.0b_prod.pkl` | LassoCV on v2.0b hazard probs + `mean_t`/`sd_t`. MAE 1.14 yr, Spearman 0.66. |

**Buy-list (`build_v2.0_buylist.py`):** thesis = **`P(MLB_DEBUT ≤ 3y)`**
(`xp_MLB_DEBUT_h3`) — filter, sort, and the output `p_MLB_DEBUT` column all use
the 3-year debut slice; ceiling events (top100/established/star) reported at
h=6 for context (`p_MLB_DEBUT_6y` carried alongside). Universe filters: EXIT
washouts, point-in-time top-100 drop, currently-MLB drop, R1 kept.

**Calibration finding.** Ranking (AUC) is 0.95–0.99 across all events and all h.
MLB_DEBUT is near-perfectly calibrated (`calib` ≈ 1.0 from h≥3). **STAR_PLUS_ELITE
is well-ranked but under-calibrated at long horizons** (`calib` ≈ 0.7 by h≥4) —
the magnitude of stardom is under-predicted; a per-horizon isotonic recal on that
head is the fix (ranking needs none).

## Headline (ALL bucket, h=6, threshold = 0.60)

| Event | n | base% | AP | lift | AUC | spearman | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TOP_100_PROSPECT | 24092 | 0.99% | **0.523** | 52.7× | 0.983 | 0.166 | 0.767 | 0.234 | 0.359 |
| MLB_DEBUT | 24272 | 7.84% | **0.667** | 8.5× | 0.948 | 0.417 | 0.773 | 0.347 | 0.479 |
| ESTABLISHED_MLB | 24272 | 2.12% | **0.409** | 19.3× | 0.965 | 0.232 | 0.766 | 0.070 | 0.128 |
| STAR_PLUS_ELITE | 24272 | 0.49% | **0.191** | 39.3× | 0.970 | 0.113 | 1.000 | 0.008 | 0.017 |
| **weighted-AP** | | | **0.491** | | | | | | |

(MLB_DEBUT 2× weight, others 1×, per-event eligibility filters. Scores =
`xp_<event>_h6` vs realized-within-6y, on rows resolved at h=6.)

## Per-horizon trajectory (h=1..10, resolved at each h)

#### TOP_100_PROSPECT

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 30817 | 105 | 0.34% | 0.992 | 0.542 | 159.1× | 0.0022 | 0.86 |
| 2 | 29829 | 191 | 0.64% | 0.986 | 0.491 | 76.8× | 0.0044 | 0.84 |
| 3 | 28662 | 242 | 0.84% | 0.984 | 0.518 | 61.4× | 0.0056 | 0.84 |
| 4 | 27323 | 259 | 0.95% | 0.982 | 0.510 | 53.8× | 0.0063 | 0.89 |
| 5 | 25800 | 256 | 0.99% | 0.982 | 0.522 | 52.6× | 0.0065 | 0.93 |
| 6 | 24092 | 239 | 0.99% | 0.983 | 0.523 | 52.7× | 0.0065 | 1.00 |
| 7 | 22205 | 229 | 1.03% | 0.982 | 0.526 | 51.0× | 0.0067 | 1.02 |
| 8 | 20318 | 217 | 1.07% | 0.981 | 0.519 | 48.6× | 0.0070 | 1.04 |
| 9 | 18407 | 203 | 1.10% | 0.979 | 0.513 | 46.5× | 0.0073 | 1.05 |
| 10 | 16424 | 190 | 1.16% | 0.978 | 0.513 | 44.3× | 0.0077 | 1.07 |

#### MLB_DEBUT

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 31046 | 506 | 1.63% | 0.973 | 0.481 | 29.5× | 0.0111 | 0.96 |
| 2 | 30052 | 993 | 3.30% | 0.964 | 0.582 | 17.6× | 0.0199 | 0.95 |
| 3 | 28879 | 1425 | 4.93% | 0.956 | 0.617 | 12.5× | 0.0283 | 0.96 |
| 4 | 27529 | 1721 | 6.25% | 0.951 | 0.641 | 10.2× | 0.0347 | 0.97 |
| 5 | 25993 | 1874 | 7.21% | 0.949 | 0.658 | 9.1× | 0.0388 | 0.97 |
| 6 | 24272 | 1903 | 7.84% | 0.948 | 0.667 | 8.5× | 0.0414 | 0.98 |
| 7 | 22368 | 1871 | 8.36% | 0.947 | 0.672 | 8.0× | 0.0435 | 0.98 |
| 8 | 20463 | 1793 | 8.76% | 0.947 | 0.670 | 7.7× | 0.0454 | 0.98 |
| 9 | 18535 | 1683 | 9.08% | 0.944 | 0.665 | 7.3× | 0.0472 | 0.99 |
| 10 | 16530 | 1564 | 9.46% | 0.941 | 0.662 | 7.0× | 0.0495 | 0.99 |

#### ESTABLISHED_MLB

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 31046 | 3 | 0.01% | 0.993 | 0.019 | 192.4× | 0.0001 | 2.41 |
| 2 | 30052 | 58 | 0.19% | 0.986 | 0.220 | 114.1× | 0.0017 | 0.98 |
| 3 | 28879 | 165 | 0.57% | 0.981 | 0.319 | 55.8× | 0.0047 | 0.98 |
| 4 | 27529 | 295 | 1.07% | 0.974 | 0.358 | 33.4× | 0.0084 | 0.92 |
| 5 | 25993 | 416 | 1.60% | 0.969 | 0.386 | 24.1× | 0.0121 | 0.92 |
| 6 | 24272 | 514 | 2.12% | 0.965 | 0.409 | 19.3× | 0.0156 | 0.93 |
| 7 | 22368 | 581 | 2.60% | 0.961 | 0.417 | 16.0× | 0.0188 | 0.94 |
| 8 | 20463 | 619 | 3.02% | 0.958 | 0.424 | 14.0× | 0.0217 | 0.94 |
| 9 | 18535 | 624 | 3.37% | 0.955 | 0.427 | 12.7× | 0.0241 | 0.95 |
| 10 | 16530 | 613 | 3.71% | 0.953 | 0.435 | 11.7× | 0.0262 | 0.93 |

#### STAR_PLUS_ELITE

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 31046 | 1 | 0.00% | 0.990 | 0.003 | 97.9× | 0.0000 | 4.34 |
| 2 | 30052 | 8 | 0.03% | 0.990 | 0.220 | 825.9× | 0.0003 | 1.46 |
| 3 | 28879 | 24 | 0.08% | 0.981 | 0.185 | 222.5× | 0.0008 | 1.09 |
| 4 | 27529 | 56 | 0.20% | 0.978 | 0.193 | 94.7× | 0.0019 | 0.79 |
| 5 | 25993 | 90 | 0.35% | 0.972 | 0.171 | 49.4× | 0.0031 | 0.75 |
| 6 | 24272 | 118 | 0.49% | 0.970 | 0.191 | 39.3× | 0.0043 | 0.76 |
| 7 | 22368 | 138 | 0.62% | 0.968 | 0.200 | 32.4× | 0.0054 | 0.80 |
| 8 | 20463 | 152 | 0.74% | 0.965 | 0.208 | 28.0× | 0.0065 | 0.80 |
| 9 | 18535 | 164 | 0.88% | 0.963 | 0.222 | 25.1× | 0.0077 | 0.80 |
| 10 | 16530 | 170 | 1.03% | 0.961 | 0.229 | 22.2× | 0.0089 | 0.75 |

## Per-bucket (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 24092 | 239 | 0.99% | 0.983 | 0.523 | 52.7× | 0.166 | 0.767 | 0.234 | 0.359 | 56 | 17 | 183 |
| R1 | 420 | 78 | 18.57% | 0.909 | 0.710 | 3.8× | 0.551 | 0.745 | 0.487 | 0.589 | 38 | 13 | 40 |
| R2-R3 | 751 | 71 | 9.45% | 0.912 | 0.528 | 5.6× | 0.417 | 0.769 | 0.141 | 0.238 | 10 | 3 | 61 |
| R4-R10 | 2499 | 14 | 0.56% | 0.950 | 0.095 | 16.9× | 0.116 | 0.000 | 0.000 | — | 0 | 1 | 14 |
| R10+ | 10468 | 27 | 0.26% | 0.968 | 0.307 | 119.1× | 0.082 | 1.000 | 0.111 | 0.200 | 3 | 0 | 24 |
| IFA | 9954 | 49 | 0.49% | 0.985 | 0.465 | 94.5× | 0.118 | 1.000 | 0.102 | 0.185 | 5 | 0 | 44 |

#### MLB_DEBUT

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 24272 | 1903 | 7.84% | 0.948 | 0.667 | 8.5× | 0.417 | 0.773 | 0.347 | 0.479 | 661 | 194 | 1242 |
| R1 | 503 | 269 | 53.48% | 0.888 | 0.901 | 1.7× | 0.671 | 0.838 | 0.732 | 0.782 | 197 | 38 | 72 |
| R2-R3 | 797 | 312 | 39.15% | 0.849 | 0.782 | 2.0× | 0.591 | 0.763 | 0.506 | 0.609 | 158 | 49 | 154 |
| R4-R10 | 2509 | 410 | 16.34% | 0.873 | 0.572 | 3.5× | 0.478 | 0.673 | 0.251 | 0.366 | 103 | 50 | 307 |
| R10+ | 10477 | 580 | 5.54% | 0.929 | 0.487 | 8.8× | 0.340 | 0.732 | 0.155 | 0.256 | 90 | 33 | 490 |
| IFA | 9986 | 332 | 3.32% | 0.957 | 0.658 | 19.8× | 0.284 | 0.825 | 0.340 | 0.482 | 113 | 24 | 219 |

#### ESTABLISHED_MLB

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 24272 | 514 | 2.12% | 0.965 | 0.409 | 19.3× | 0.232 | 0.766 | 0.070 | 0.128 | 36 | 11 | 478 |
| R1 | 503 | 124 | 24.65% | 0.863 | 0.683 | 2.8× | 0.542 | 0.880 | 0.177 | 0.295 | 22 | 3 | 102 |
| R2-R3 | 797 | 91 | 11.42% | 0.835 | 0.352 | 3.1× | 0.369 | 0.462 | 0.066 | 0.115 | 6 | 7 | 85 |
| R4-R10 | 2509 | 116 | 4.62% | 0.909 | 0.333 | 7.2× | 0.297 | 1.000 | 0.017 | 0.034 | 2 | 0 | 114 |
| R10+ | 10477 | 129 | 1.23% | 0.951 | 0.247 | 20.0× | 0.172 | 1.000 | 0.031 | 0.060 | 4 | 0 | 125 |
| IFA | 9986 | 54 | 0.54% | 0.983 | 0.341 | 63.0× | 0.123 | 0.667 | 0.037 | 0.070 | 2 | 1 | 52 |

#### STAR_PLUS_ELITE

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 24272 | 118 | 0.49% | 0.970 | 0.191 | 39.3× | 0.113 | 1.000 | 0.008 | 0.017 | 1 | 0 | 117 |
| R1 | 503 | 40 | 7.95% | 0.833 | 0.284 | 3.6× | 0.312 | 1.000 | 0.025 | 0.049 | 1 | 0 | 39 |
| R2-R3 | 797 | 24 | 3.01% | 0.879 | 0.169 | 5.6× | 0.224 | — | 0.000 | — | 0 | 0 | 24 |
| R4-R10 | 2509 | 17 | 0.68% | 0.926 | 0.086 | 12.6× | 0.121 | — | 0.000 | — | 0 | 0 | 17 |
| R10+ | 10477 | 23 | 0.22% | 0.958 | 0.166 | 75.6× | 0.074 | — | 0.000 | — | 0 | 0 | 23 |
| IFA | 9986 | 14 | 0.14% | 0.979 | 0.262 | 187.1× | 0.062 | — | 0.000 | — | 0 | 0 | 14 |

## Per-yip (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3329 | 103 | 3.09% | 0.950 | 0.491 | 15.9× | 0.270 | 0.675 | 0.262 | 0.378 | 27 | 13 | 76 |
| 1 | 3114 | 75 | 2.41% | 0.966 | 0.589 | 24.5× | 0.247 | 0.882 | 0.200 | 0.326 | 15 | 2 | 60 |
| 2 | 2863 | 40 | 1.40% | 0.974 | 0.540 | 38.6× | 0.193 | 0.846 | 0.275 | 0.415 | 11 | 2 | 29 |
| 3 | 2593 | 16 | 0.62% | 0.977 | 0.402 | 65.2× | 0.129 | 1.000 | 0.062 | 0.118 | 1 | 0 | 15 |
| 4 | 2336 | 5 | 0.21% | 0.985 | 0.805 | 376.3× | 0.078 | 1.000 | 0.400 | 0.571 | 2 | 0 | 3 |
| 5 | 2099 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 6 | 1905 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 7 | 1719 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 8 | 1541 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 9 | 1380 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 1213 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

#### MLB_DEBUT

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3336 | 426 | 12.77% | 0.895 | 0.584 | 4.6× | 0.457 | 0.752 | 0.249 | 0.374 | 106 | 35 | 320 |
| 1 | 3138 | 434 | 13.83% | 0.920 | 0.677 | 4.9× | 0.502 | 0.754 | 0.366 | 0.493 | 159 | 52 | 275 |
| 2 | 2907 | 398 | 13.69% | 0.941 | 0.748 | 5.5× | 0.525 | 0.790 | 0.455 | 0.577 | 181 | 48 | 217 |
| 3 | 2629 | 275 | 10.46% | 0.947 | 0.735 | 7.0× | 0.474 | 0.793 | 0.418 | 0.548 | 115 | 30 | 160 |
| 4 | 2362 | 175 | 7.41% | 0.949 | 0.695 | 9.4× | 0.407 | 0.763 | 0.406 | 0.530 | 71 | 22 | 104 |
| 5 | 2113 | 92 | 4.35% | 0.943 | 0.567 | 13.0× | 0.313 | 0.778 | 0.228 | 0.353 | 21 | 6 | 71 |
| 6 | 1915 | 52 | 2.72% | 0.956 | 0.493 | 18.1× | 0.257 | 0.889 | 0.154 | 0.262 | 8 | 1 | 44 |
| 7 | 1725 | 30 | 1.74% | 0.967 | 0.342 | 19.7× | 0.212 | — | 0.000 | — | 0 | 0 | 30 |
| 8 | 1546 | 12 | 0.78% | 0.944 | 0.203 | 26.2× | 0.135 | — | 0.000 | — | 0 | 0 | 12 |
| 9 | 1384 | 7 | 0.51% | 0.970 | 0.295 | 58.2× | 0.115 | — | 0.000 | — | 0 | 0 | 7 |
| 10 | 1217 | 2 | 0.16% | 0.901 | 0.067 | 40.6× | 0.056 | — | 0.000 | — | 0 | 0 | 2 |

#### ESTABLISHED_MLB

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3336 | 114 | 3.42% | 0.919 | 0.351 | 10.3× | 0.264 | 0.714 | 0.044 | 0.083 | 5 | 2 | 109 |
| 1 | 3138 | 135 | 4.30% | 0.936 | 0.431 | 10.0× | 0.307 | 0.769 | 0.074 | 0.135 | 10 | 3 | 125 |
| 2 | 2907 | 134 | 4.61% | 0.953 | 0.520 | 11.3× | 0.329 | 0.824 | 0.104 | 0.185 | 14 | 3 | 120 |
| 3 | 2629 | 72 | 2.74% | 0.955 | 0.354 | 12.9× | 0.257 | 0.571 | 0.056 | 0.101 | 4 | 3 | 68 |
| 4 | 2362 | 40 | 1.69% | 0.974 | 0.441 | 26.1× | 0.212 | 1.000 | 0.075 | 0.140 | 3 | 0 | 37 |
| 5 | 2113 | 11 | 0.52% | 0.951 | 0.078 | 15.0× | 0.112 | — | 0.000 | — | 0 | 0 | 11 |
| 6 | 1915 | 5 | 0.26% | 0.965 | 0.100 | 38.1× | 0.082 | — | 0.000 | — | 0 | 0 | 5 |
| 7 | 1725 | 2 | 0.12% | 0.990 | 0.154 | 133.2× | 0.058 | — | 0.000 | — | 0 | 0 | 2 |
| 8 | 1546 | 1 | 0.06% | 0.999 | 0.500 | 773.0× | 0.044 | — | 0.000 | — | 0 | 0 | 1 |
| 9 | 1384 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 1217 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

#### STAR_PLUS_ELITE

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3336 | 28 | 0.84% | 0.924 | 0.127 | 15.1× | 0.134 | 1.000 | 0.036 | 0.069 | 1 | 0 | 27 |
| 1 | 3138 | 33 | 1.05% | 0.943 | 0.200 | 19.0× | 0.157 | — | 0.000 | — | 0 | 0 | 33 |
| 2 | 2907 | 34 | 1.17% | 0.965 | 0.327 | 28.0× | 0.173 | — | 0.000 | — | 0 | 0 | 34 |
| 3 | 2629 | 14 | 0.53% | 0.964 | 0.259 | 48.6× | 0.117 | — | 0.000 | — | 0 | 0 | 14 |
| 4 | 2362 | 7 | 0.30% | 0.983 | 0.254 | 85.7× | 0.091 | — | 0.000 | — | 0 | 0 | 7 |
| 5 | 2113 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 6 | 1915 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 7 | 1725 | 1 | 0.06% | 0.992 | 0.071 | 123.2× | 0.041 | — | 0.000 | — | 0 | 0 | 1 |
| 8 | 1546 | 1 | 0.06% | 0.992 | 0.077 | 118.9× | 0.043 | — | 0.000 | — | 0 | 0 | 1 |
| 9 | 1384 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 1217 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

## Per-level (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 24092 | 239 | 0.99% | 0.983 | 0.523 | 52.7× | 0.166 | 0.767 | 0.234 | 0.359 | 56 | 17 | 183 |
| RK | 3597 | 65 | 1.81% | 0.968 | 0.546 | 30.2× | 0.216 | 0.833 | 0.231 | 0.361 | 15 | 3 | 50 |
| A- | 919 | 19 | 2.07% | 0.966 | 0.501 | 24.2× | 0.230 | 0.556 | 0.263 | 0.357 | 5 | 4 | 14 |
| A | 1265 | 40 | 3.16% | 0.966 | 0.615 | 19.4× | 0.283 | 0.812 | 0.325 | 0.464 | 13 | 3 | 27 |
| A+ | 1237 | 28 | 2.26% | 0.986 | 0.628 | 27.8× | 0.250 | 0.778 | 0.250 | 0.378 | 7 | 2 | 21 |
| AA | 1068 | 26 | 2.43% | 0.986 | 0.708 | 29.1× | 0.260 | 0.833 | 0.385 | 0.526 | 10 | 2 | 16 |
| AAA | 1318 | 5 | 0.38% | 0.999 | 0.900 | 237.2× | 0.106 | 1.000 | 0.400 | 0.571 | 2 | 0 | 3 |
| NONE | 14688 | 56 | 0.38% | 0.986 | 0.264 | 69.2× | 0.104 | 0.571 | 0.071 | 0.127 | 4 | 3 | 52 |

#### MLB_DEBUT

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 24272 | 1903 | 7.84% | 0.948 | 0.667 | 8.5× | 0.417 | 0.773 | 0.347 | 0.479 | 661 | 194 | 1242 |
| RK | 3601 | 204 | 5.67% | 0.923 | 0.534 | 9.4× | 0.339 | 0.750 | 0.147 | 0.246 | 30 | 10 | 174 |
| A- | 921 | 130 | 14.12% | 0.835 | 0.502 | 3.6× | 0.405 | 0.781 | 0.192 | 0.309 | 25 | 7 | 105 |
| A | 1286 | 234 | 18.20% | 0.869 | 0.662 | 3.6× | 0.493 | 0.784 | 0.372 | 0.504 | 87 | 24 | 147 |
| A+ | 1258 | 245 | 19.48% | 0.875 | 0.697 | 3.6× | 0.515 | 0.777 | 0.412 | 0.539 | 101 | 29 | 144 |
| AA | 1142 | 349 | 30.56% | 0.866 | 0.787 | 2.6× | 0.584 | 0.817 | 0.524 | 0.639 | 183 | 41 | 166 |
| AAA | 1351 | 244 | 18.06% | 0.914 | 0.735 | 4.1× | 0.552 | 0.856 | 0.340 | 0.487 | 83 | 14 | 161 |
| NONE | 14713 | 497 | 3.38% | 0.970 | 0.616 | 18.2× | 0.294 | 0.688 | 0.306 | 0.423 | 152 | 69 | 345 |

#### ESTABLISHED_MLB

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 24272 | 514 | 2.12% | 0.965 | 0.409 | 19.3× | 0.232 | 0.766 | 0.070 | 0.128 | 36 | 11 | 478 |
| RK | 3601 | 41 | 1.14% | 0.950 | 0.288 | 25.3× | 0.165 | — | 0.000 | — | 0 | 0 | 41 |
| A- | 921 | 24 | 2.61% | 0.885 | 0.150 | 5.8× | 0.212 | 0.000 | 0.000 | — | 0 | 2 | 24 |
| A | 1286 | 64 | 4.98% | 0.923 | 0.475 | 9.6× | 0.319 | 1.000 | 0.047 | 0.090 | 3 | 0 | 61 |
| A+ | 1258 | 68 | 5.41% | 0.923 | 0.413 | 7.6× | 0.331 | 0.833 | 0.074 | 0.135 | 5 | 1 | 63 |
| AA | 1142 | 102 | 8.93% | 0.897 | 0.477 | 5.3× | 0.392 | 0.632 | 0.118 | 0.198 | 12 | 7 | 90 |
| AAA | 1351 | 55 | 4.07% | 0.941 | 0.518 | 12.7× | 0.302 | 1.000 | 0.127 | 0.226 | 7 | 0 | 48 |
| NONE | 14713 | 160 | 1.09% | 0.982 | 0.396 | 36.4× | 0.173 | 0.900 | 0.056 | 0.106 | 9 | 1 | 151 |

#### STAR_PLUS_ELITE

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 24272 | 118 | 0.49% | 0.970 | 0.191 | 39.3× | 0.113 | 1.000 | 0.008 | 0.017 | 1 | 0 | 117 |
| RK | 3601 | 14 | 0.39% | 0.944 | 0.153 | 39.3× | 0.096 | — | 0.000 | — | 0 | 0 | 14 |
| A- | 921 | 2 | 0.22% | 0.959 | 0.054 | 24.7× | 0.074 | — | 0.000 | — | 0 | 0 | 2 |
| A | 1286 | 13 | 1.01% | 0.949 | 0.298 | 29.5× | 0.156 | — | 0.000 | — | 0 | 0 | 13 |
| A+ | 1258 | 17 | 1.35% | 0.933 | 0.235 | 17.4× | 0.173 | — | 0.000 | — | 0 | 0 | 17 |
| AA | 1142 | 19 | 1.66% | 0.936 | 0.297 | 17.9× | 0.193 | — | 0.000 | — | 0 | 0 | 19 |
| AAA | 1351 | 12 | 0.89% | 0.949 | 0.344 | 38.7× | 0.146 | — | 0.000 | — | 0 | 0 | 12 |
| NONE | 14713 | 41 | 0.28% | 0.986 | 0.173 | 62.0× | 0.089 | 1.000 | 0.024 | 0.048 | 1 | 0 | 40 |

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

# validation — per-horizon, headline at the publish horizon (h=6)
python -m scripts_v17.validate.regen_eval_v2_0b_honest --eval-horizon 6
python -m scripts_v17.validate.gen_eval_readme

# buy list — P(debut <= 3y) thesis
python scripts_v17/buylist/build_v2.0_buylist.py \
    --long results/scored/snap2026_v1.18b_landmark_long.csv \
    --xgb models/joint_xgb_v2.0b_prod.pkl --debut-horizon 3 --threshold 0.60
```
