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
| Hazards (per-fold OOF, eval) | `runs/current/scratch/oof/fold[0-5]_hazards.pkl` | Each fold trained on the OTHER 5 (val pids excluded). HistGBT, default HP, 314 features (incl. 76 scouting). Survival → censoring-aware. |
| Hazards (production) | `runs/current/models/hazards.pkl` | 100% of ≤2020 data. Scores the 2026 cohort (entry 2024–26 — not in training, so no leakage). |
| Conditional joint XGB | `runs/current/models/joint_xgb.pkl` (`model/train/joint_xgb.py`) | OOF stacked, expanded to resolved `(row, h)` pairs for h=1..10. `multi_output_tree` over the 4 heads; per-horizon censoring built in (no `--censor-window`). Outputs `P(event by snap+h)`; monotone in h via cummax at inference. |
| Timing | `runs/current/models/timing.pkl` | LassoCV on v2.0b hazard probs + `mean_t`/`sd_t`. MAE 1.14 yr, Spearman 0.66. |

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
| TOP_100_PROSPECT | 25571 | 0.74% | **0.557** | 75.7× | 0.991 | 0.145 | 0.672 | 0.436 | 0.529 |
| MLB_DEBUT | 25754 | 5.85% | **0.664** | 11.3× | 0.956 | 0.371 | 0.708 | 0.450 | 0.550 |
| ESTABLISHED_MLB | 25754 | 1.70% | **0.427** | 25.2× | 0.973 | 0.212 | 0.651 | 0.162 | 0.260 |
| STAR_PLUS_ELITE | 25754 | 0.30% | **0.172** | 56.7× | 0.977 | 0.091 | — | 0.000 | — |
| **weighted-AP** | | | **0.497** | | | | | | |

(MLB_DEBUT 2× weight, others 1×, per-event eligibility filters. Scores =
`xp_<event>_h6` vs realized-within-6y, on rows resolved at h=6.)

## Per-horizon trajectory (h=1..10, resolved at each h)

#### TOP_100_PROSPECT

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32887 | 89 | 0.27% | 0.996 | 0.459 | 169.6× | 0.0019 | 0.87 |
| 2 | 31815 | 158 | 0.50% | 0.993 | 0.511 | 103.0× | 0.0033 | 0.90 |
| 3 | 30544 | 198 | 0.65% | 0.991 | 0.534 | 82.4× | 0.0041 | 0.92 |
| 4 | 29083 | 210 | 0.72% | 0.990 | 0.539 | 74.7× | 0.0045 | 0.96 |
| 5 | 27427 | 205 | 0.75% | 0.990 | 0.555 | 74.2× | 0.0045 | 1.02 |
| 6 | 25571 | 188 | 0.74% | 0.991 | 0.557 | 75.7× | 0.0044 | 1.09 |
| 7 | 23519 | 178 | 0.76% | 0.991 | 0.567 | 74.9× | 0.0045 | 1.10 |
| 8 | 21466 | 166 | 0.77% | 0.990 | 0.560 | 72.4× | 0.0046 | 1.12 |
| 9 | 19380 | 152 | 0.78% | 0.990 | 0.553 | 70.6× | 0.0047 | 1.14 |
| 10 | 17211 | 139 | 0.81% | 0.990 | 0.555 | 68.7× | 0.0049 | 1.16 |

#### MLB_DEBUT

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 33119 | 433 | 1.31% | 0.979 | 0.511 | 39.1× | 0.0085 | 0.97 |
| 2 | 32041 | 828 | 2.58% | 0.970 | 0.607 | 23.5× | 0.0150 | 0.97 |
| 3 | 30764 | 1175 | 3.82% | 0.962 | 0.624 | 16.3× | 0.0217 | 0.97 |
| 4 | 29292 | 1405 | 4.80% | 0.958 | 0.643 | 13.4× | 0.0266 | 0.98 |
| 5 | 27623 | 1509 | 5.46% | 0.956 | 0.658 | 12.0× | 0.0296 | 0.99 |
| 6 | 25754 | 1507 | 5.85% | 0.956 | 0.664 | 11.3× | 0.0314 | 1.00 |
| 7 | 23685 | 1453 | 6.13% | 0.956 | 0.666 | 10.9× | 0.0326 | 1.00 |
| 8 | 21614 | 1359 | 6.29% | 0.955 | 0.661 | 10.5× | 0.0336 | 1.00 |
| 9 | 19511 | 1242 | 6.37% | 0.954 | 0.656 | 10.3× | 0.0342 | 1.00 |
| 10 | 17320 | 1117 | 6.45% | 0.952 | 0.650 | 10.1× | 0.0350 | 1.01 |

#### ESTABLISHED_MLB

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 33119 | 2 | 0.01% | 0.999 | 0.050 | 828.0× | 0.0001 | 3.28 |
| 2 | 32041 | 42 | 0.13% | 0.990 | 0.193 | 147.3× | 0.0012 | 0.90 |
| 3 | 30764 | 142 | 0.46% | 0.985 | 0.361 | 78.2× | 0.0036 | 0.97 |
| 4 | 29292 | 256 | 0.87% | 0.982 | 0.414 | 47.3× | 0.0064 | 0.94 |
| 5 | 27623 | 360 | 1.30% | 0.977 | 0.409 | 31.4× | 0.0095 | 0.93 |
| 6 | 25754 | 437 | 1.70% | 0.973 | 0.427 | 25.2× | 0.0122 | 0.93 |
| 7 | 23685 | 479 | 2.02% | 0.971 | 0.440 | 21.8× | 0.0143 | 0.94 |
| 8 | 21614 | 495 | 2.29% | 0.969 | 0.443 | 19.4× | 0.0160 | 0.94 |
| 9 | 19511 | 490 | 2.51% | 0.967 | 0.444 | 17.7× | 0.0175 | 0.94 |
| 10 | 17320 | 468 | 2.70% | 0.966 | 0.453 | 16.8× | 0.0187 | 0.90 |

#### STAR_PLUS_ELITE

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 33119 | 1 | 0.00% | 0.996 | 0.007 | 230.0× | 0.0000 | 2.86 |
| 2 | 32041 | 7 | 0.02% | 0.992 | 0.121 | 552.8× | 0.0002 | 0.95 |
| 3 | 30764 | 17 | 0.06% | 0.988 | 0.104 | 188.9× | 0.0005 | 1.07 |
| 4 | 29292 | 38 | 0.13% | 0.982 | 0.198 | 152.6× | 0.0012 | 0.84 |
| 5 | 27623 | 61 | 0.22% | 0.979 | 0.167 | 75.7× | 0.0020 | 0.77 |
| 6 | 25754 | 78 | 0.30% | 0.977 | 0.172 | 56.7× | 0.0027 | 0.76 |
| 7 | 23685 | 89 | 0.38% | 0.977 | 0.180 | 47.9× | 0.0034 | 0.77 |
| 8 | 21614 | 97 | 0.45% | 0.975 | 0.199 | 44.4× | 0.0040 | 0.76 |
| 9 | 19511 | 106 | 0.54% | 0.974 | 0.204 | 37.5× | 0.0048 | 0.71 |
| 10 | 17320 | 109 | 0.63% | 0.974 | 0.214 | 34.0× | 0.0055 | 0.65 |

## Per-bucket (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 25571 | 188 | 0.74% | 0.991 | 0.557 | 75.7× | 0.145 | 0.672 | 0.436 | 0.529 | 82 | 40 | 106 |
| R1 | 320 | 68 | 21.25% | 0.910 | 0.662 | 3.1× | 0.581 | 0.630 | 0.676 | 0.652 | 46 | 27 | 22 |
| R2-R3 | 529 | 50 | 9.45% | 0.927 | 0.557 | 5.9× | 0.433 | 0.696 | 0.320 | 0.438 | 16 | 7 | 34 |
| R4-R10 | 1763 | 8 | 0.45% | 0.964 | 0.128 | 28.2× | 0.108 | 0.000 | 0.000 | — | 0 | 1 | 8 |
| R10+ | 8514 | 5 | 0.06% | 0.987 | 0.608 | 1035.9× | 0.041 | 1.000 | 0.600 | 0.750 | 3 | 0 | 2 |
| IFA | 14445 | 57 | 0.39% | 0.987 | 0.461 | 116.9× | 0.106 | 0.773 | 0.298 | 0.430 | 17 | 5 | 40 |

#### MLB_DEBUT

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 25754 | 1507 | 5.85% | 0.956 | 0.664 | 11.3× | 0.371 | 0.708 | 0.450 | 0.550 | 678 | 279 | 829 |
| R1 | 393 | 203 | 51.65% | 0.890 | 0.900 | 1.7× | 0.674 | 0.794 | 0.798 | 0.796 | 162 | 42 | 41 |
| R2-R3 | 558 | 219 | 39.25% | 0.861 | 0.799 | 2.0× | 0.611 | 0.720 | 0.694 | 0.707 | 152 | 59 | 67 |
| R4-R10 | 1764 | 248 | 14.06% | 0.880 | 0.573 | 4.1× | 0.458 | 0.582 | 0.371 | 0.453 | 92 | 66 | 156 |
| R10+ | 8516 | 339 | 3.98% | 0.943 | 0.481 | 12.1× | 0.300 | 0.673 | 0.206 | 0.316 | 70 | 34 | 269 |
| IFA | 14523 | 498 | 3.43% | 0.956 | 0.619 | 18.1× | 0.288 | 0.721 | 0.406 | 0.519 | 202 | 78 | 296 |

#### ESTABLISHED_MLB

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 25754 | 437 | 1.70% | 0.973 | 0.427 | 25.2× | 0.212 | 0.651 | 0.162 | 0.260 | 71 | 38 | 366 |
| R1 | 393 | 103 | 26.21% | 0.865 | 0.656 | 2.5× | 0.556 | 0.714 | 0.388 | 0.503 | 40 | 16 | 63 |
| R2-R3 | 558 | 67 | 12.01% | 0.853 | 0.419 | 3.5× | 0.397 | 0.571 | 0.179 | 0.273 | 12 | 9 | 55 |
| R4-R10 | 1764 | 78 | 4.42% | 0.917 | 0.317 | 7.2× | 0.297 | 0.500 | 0.038 | 0.071 | 3 | 3 | 75 |
| R10+ | 8516 | 69 | 0.81% | 0.969 | 0.297 | 36.6× | 0.146 | 0.667 | 0.058 | 0.107 | 4 | 2 | 65 |
| IFA | 14523 | 120 | 0.83% | 0.978 | 0.344 | 41.7× | 0.150 | 0.600 | 0.100 | 0.171 | 12 | 8 | 108 |

#### STAR_PLUS_ELITE

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 25754 | 78 | 0.30% | 0.977 | 0.172 | 56.7× | 0.091 | — | 0.000 | — | 0 | 0 | 78 |
| R1 | 393 | 24 | 6.11% | 0.845 | 0.187 | 3.1× | 0.286 | — | 0.000 | — | 0 | 0 | 24 |
| R2-R3 | 558 | 11 | 1.97% | 0.893 | 0.123 | 6.2× | 0.189 | — | 0.000 | — | 0 | 0 | 11 |
| R4-R10 | 1764 | 11 | 0.62% | 0.946 | 0.241 | 38.7× | 0.122 | — | 0.000 | — | 0 | 0 | 11 |
| R10+ | 8516 | 11 | 0.13% | 0.943 | 0.207 | 160.2× | 0.055 | — | 0.000 | — | 0 | 0 | 11 |
| IFA | 14523 | 21 | 0.14% | 0.987 | 0.251 | 173.6× | 0.064 | — | 0.000 | — | 0 | 0 | 21 |

## Per-yip (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3489 | 86 | 2.46% | 0.963 | 0.492 | 20.0× | 0.249 | 0.593 | 0.407 | 0.483 | 35 | 24 | 51 |
| 1 | 3248 | 57 | 1.75% | 0.983 | 0.658 | 37.5× | 0.220 | 0.794 | 0.474 | 0.593 | 27 | 7 | 30 |
| 2 | 3002 | 30 | 1.00% | 0.988 | 0.639 | 63.9× | 0.168 | 0.750 | 0.500 | 0.600 | 15 | 5 | 15 |
| 3 | 2733 | 12 | 0.44% | 0.996 | 0.512 | 116.6× | 0.114 | 0.600 | 0.250 | 0.353 | 3 | 2 | 9 |
| 4 | 2478 | 3 | 0.12% | 1.000 | 0.867 | 715.9× | 0.060 | 0.500 | 0.667 | 0.571 | 2 | 2 | 1 |
| 5 | 2250 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 6 | 2052 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 7 | 1851 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 8 | 1663 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 9 | 1493 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 1312 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

#### MLB_DEBUT

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3508 | 369 | 10.52% | 0.902 | 0.567 | 5.4× | 0.428 | 0.655 | 0.360 | 0.465 | 133 | 70 | 236 |
| 1 | 3276 | 349 | 10.65% | 0.925 | 0.656 | 6.2× | 0.454 | 0.695 | 0.450 | 0.546 | 157 | 69 | 192 |
| 2 | 3042 | 311 | 10.22% | 0.943 | 0.730 | 7.1× | 0.465 | 0.722 | 0.527 | 0.610 | 164 | 63 | 147 |
| 3 | 2766 | 209 | 7.56% | 0.960 | 0.757 | 10.0× | 0.421 | 0.761 | 0.565 | 0.648 | 118 | 37 | 91 |
| 4 | 2503 | 129 | 5.15% | 0.960 | 0.730 | 14.2× | 0.353 | 0.734 | 0.535 | 0.619 | 69 | 25 | 60 |
| 5 | 2262 | 66 | 2.92% | 0.955 | 0.562 | 19.2× | 0.265 | 0.667 | 0.364 | 0.471 | 24 | 12 | 42 |
| 6 | 2060 | 37 | 1.80% | 0.970 | 0.544 | 30.3× | 0.216 | 0.750 | 0.243 | 0.367 | 9 | 3 | 28 |
| 7 | 1856 | 21 | 1.13% | 0.980 | 0.605 | 53.5× | 0.176 | 1.000 | 0.143 | 0.250 | 3 | 0 | 18 |
| 8 | 1668 | 11 | 0.66% | 0.973 | 0.517 | 78.4× | 0.133 | 1.000 | 0.091 | 0.167 | 1 | 0 | 10 |
| 9 | 1497 | 5 | 0.33% | 0.999 | 0.761 | 227.9× | 0.100 | — | 0.000 | — | 0 | 0 | 5 |
| 10 | 1316 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

#### ESTABLISHED_MLB

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3508 | 111 | 3.16% | 0.928 | 0.326 | 10.3× | 0.260 | 0.500 | 0.090 | 0.153 | 10 | 10 | 101 |
| 1 | 3276 | 111 | 3.39% | 0.945 | 0.454 | 13.4× | 0.279 | 0.810 | 0.153 | 0.258 | 17 | 4 | 94 |
| 2 | 3042 | 102 | 3.35% | 0.969 | 0.543 | 16.2× | 0.292 | 0.722 | 0.255 | 0.377 | 26 | 10 | 76 |
| 3 | 2766 | 58 | 2.10% | 0.978 | 0.490 | 23.4× | 0.237 | 0.588 | 0.172 | 0.267 | 10 | 7 | 48 |
| 4 | 2503 | 32 | 1.28% | 0.980 | 0.417 | 32.7× | 0.187 | 0.571 | 0.250 | 0.348 | 8 | 6 | 24 |
| 5 | 2262 | 13 | 0.57% | 0.983 | 0.212 | 36.9× | 0.126 | 0.000 | 0.000 | — | 0 | 1 | 13 |
| 6 | 2060 | 6 | 0.29% | 0.984 | 0.165 | 56.8× | 0.090 | — | 0.000 | — | 0 | 0 | 6 |
| 7 | 1856 | 2 | 0.11% | 0.996 | 0.171 | 159.1× | 0.056 | — | 0.000 | — | 0 | 0 | 2 |
| 8 | 1668 | 2 | 0.12% | 1.000 | 0.833 | 695.0× | 0.060 | — | 0.000 | — | 0 | 0 | 2 |
| 9 | 1497 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 1316 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

#### STAR_PLUS_ELITE

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3508 | 23 | 0.66% | 0.941 | 0.089 | 13.6× | 0.123 | — | 0.000 | — | 0 | 0 | 23 |
| 1 | 3276 | 20 | 0.61% | 0.963 | 0.322 | 52.8× | 0.125 | — | 0.000 | — | 0 | 0 | 20 |
| 2 | 3042 | 19 | 0.62% | 0.975 | 0.247 | 39.5× | 0.130 | — | 0.000 | — | 0 | 0 | 19 |
| 3 | 2766 | 9 | 0.33% | 0.984 | 0.216 | 66.5× | 0.095 | — | 0.000 | — | 0 | 0 | 9 |
| 4 | 2503 | 5 | 0.20% | 0.990 | 0.323 | 161.7× | 0.076 | — | 0.000 | — | 0 | 0 | 5 |
| 5 | 2262 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 6 | 2060 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 7 | 1856 | 1 | 0.05% | 0.992 | 0.067 | 123.7× | 0.040 | — | 0.000 | — | 0 | 0 | 1 |
| 8 | 1668 | 1 | 0.06% | 0.994 | 0.091 | 151.6× | 0.042 | — | 0.000 | — | 0 | 0 | 1 |
| 9 | 1497 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 1316 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

## Per-level (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 25571 | 188 | 0.74% | 0.991 | 0.557 | 75.7× | 0.145 | 0.672 | 0.436 | 0.529 | 82 | 40 | 106 |
| RK | 3771 | 65 | 1.72% | 0.968 | 0.572 | 33.2× | 0.211 | 0.667 | 0.338 | 0.449 | 22 | 11 | 43 |
| A- | 987 | 19 | 1.93% | 0.969 | 0.465 | 24.2× | 0.223 | 0.615 | 0.421 | 0.500 | 8 | 5 | 11 |
| A | 1342 | 40 | 2.98% | 0.969 | 0.574 | 19.2× | 0.276 | 0.688 | 0.550 | 0.611 | 22 | 10 | 18 |
| A+ | 1332 | 28 | 2.10% | 0.987 | 0.606 | 28.8× | 0.242 | 0.667 | 0.429 | 0.522 | 12 | 6 | 16 |
| AA | 1151 | 26 | 2.26% | 0.988 | 0.765 | 33.9× | 0.251 | 0.833 | 0.577 | 0.682 | 15 | 3 | 11 |
| AAA | 1399 | 5 | 0.36% | 0.997 | 0.569 | 159.2× | 0.103 | 0.667 | 0.400 | 0.500 | 2 | 1 | 3 |
| NONE | 15589 | 5 | 0.03% | 0.999 | 0.218 | 680.5× | 0.031 | 0.200 | 0.200 | 0.200 | 1 | 4 | 4 |

#### MLB_DEBUT

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 25754 | 1507 | 5.85% | 0.956 | 0.664 | 11.3× | 0.371 | 0.708 | 0.450 | 0.550 | 678 | 279 | 829 |
| RK | 3775 | 209 | 5.54% | 0.922 | 0.530 | 9.6× | 0.334 | 0.704 | 0.273 | 0.393 | 57 | 24 | 152 |
| A- | 989 | 133 | 13.45% | 0.845 | 0.525 | 3.9× | 0.407 | 0.614 | 0.323 | 0.424 | 43 | 27 | 90 |
| A | 1363 | 240 | 17.61% | 0.875 | 0.655 | 3.7× | 0.494 | 0.675 | 0.458 | 0.546 | 110 | 53 | 130 |
| A+ | 1353 | 255 | 18.85% | 0.861 | 0.684 | 3.6× | 0.490 | 0.696 | 0.502 | 0.583 | 128 | 56 | 127 |
| AA | 1225 | 357 | 29.14% | 0.862 | 0.772 | 2.7× | 0.569 | 0.731 | 0.594 | 0.655 | 212 | 78 | 145 |
| AAA | 1432 | 252 | 17.60% | 0.908 | 0.723 | 4.1× | 0.538 | 0.771 | 0.440 | 0.561 | 111 | 33 | 141 |
| NONE | 15617 | 61 | 0.39% | 0.941 | 0.441 | 112.9× | 0.095 | 0.680 | 0.279 | 0.395 | 17 | 8 | 44 |

#### ESTABLISHED_MLB

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 25754 | 437 | 1.70% | 0.973 | 0.427 | 25.2× | 0.212 | 0.651 | 0.162 | 0.260 | 71 | 38 | 366 |
| RK | 3775 | 49 | 1.30% | 0.950 | 0.256 | 19.8× | 0.176 | 0.250 | 0.020 | 0.038 | 1 | 3 | 48 |
| A- | 989 | 29 | 2.93% | 0.893 | 0.185 | 6.3× | 0.230 | 0.000 | 0.000 | — | 0 | 3 | 29 |
| A | 1363 | 72 | 5.28% | 0.913 | 0.445 | 8.4× | 0.320 | 0.889 | 0.111 | 0.198 | 8 | 1 | 64 |
| A+ | 1353 | 84 | 6.21% | 0.920 | 0.489 | 7.9× | 0.351 | 0.812 | 0.155 | 0.260 | 13 | 3 | 71 |
| AA | 1225 | 128 | 10.45% | 0.902 | 0.537 | 5.1× | 0.426 | 0.680 | 0.266 | 0.382 | 34 | 16 | 94 |
| AAA | 1432 | 65 | 4.54% | 0.938 | 0.491 | 10.8× | 0.316 | 0.609 | 0.215 | 0.318 | 14 | 9 | 51 |
| NONE | 15617 | 10 | 0.06% | 0.997 | 0.247 | 386.4× | 0.044 | 0.250 | 0.100 | 0.143 | 1 | 3 | 9 |

#### STAR_PLUS_ELITE

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 25754 | 78 | 0.30% | 0.977 | 0.172 | 56.7× | 0.091 | — | 0.000 | — | 0 | 0 | 78 |
| RK | 3775 | 14 | 0.37% | 0.955 | 0.111 | 30.0× | 0.096 | — | 0.000 | — | 0 | 0 | 14 |
| A- | 989 | 2 | 0.20% | 0.963 | 0.038 | 19.0× | 0.072 | — | 0.000 | — | 0 | 0 | 2 |
| A | 1363 | 13 | 0.95% | 0.957 | 0.263 | 27.6× | 0.154 | — | 0.000 | — | 0 | 0 | 13 |
| A+ | 1353 | 17 | 1.26% | 0.943 | 0.253 | 20.2× | 0.171 | — | 0.000 | — | 0 | 0 | 17 |
| AA | 1225 | 19 | 1.55% | 0.941 | 0.305 | 19.7× | 0.189 | — | 0.000 | — | 0 | 0 | 19 |
| AAA | 1432 | 12 | 0.84% | 0.915 | 0.278 | 33.2× | 0.131 | — | 0.000 | — | 0 | 0 | 12 |
| NONE | 15617 | 1 | 0.01% | 1.000 | 0.125 | 1952.1× | 0.014 | — | 0.000 | — | 0 | 0 | 1 |

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
# OOF folds + hazards, then the conditional joint XGB (per-horizon censoring
# is built in; wired into pipelines.oof stage 6 and pipelines.prod stage 1)
python -m prospects.model.pipelines.oof
python -m prospects.model.pipelines.prod    # 100% prod hazards + cond XGB + score 2026

# validation — per-horizon, headline at the publish horizon (h=6)
python -m prospects.evaluation.run --eval-horizon 6
python -m prospects.evaluation.report

# buy list — P(debut <= 3y) thesis
python -m prospects.buylist.build --debut-horizon 3 --threshold 0.6
```
