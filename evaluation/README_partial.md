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
| TOP_100_PROSPECT | 25911 | 0.74% | **0.530** | 71.1× | 0.987 | 0.145 | 0.714 | 0.285 | 0.407 |
| MLB_DEBUT | 26122 | 5.37% | **0.605** | 11.3× | 0.947 | 0.349 | 0.768 | 0.290 | 0.421 |
| ESTABLISHED_MLB | 26122 | 1.23% | **0.328** | 26.6× | 0.973 | 0.181 | 0.688 | 0.034 | 0.065 |
| STAR_PLUS_ELITE | 26122 | 0.28% | **0.149** | 52.6× | 0.974 | 0.087 | — | 0.000 | — |
| **weighted-AP** | | | **0.443** | | | | | | |

(MLB_DEBUT 2× weight, others 1×, per-event eligibility filters. Scores =
`xp_<event>_h6` vs realized-within-6y, on rows resolved at h=6.)

## Per-horizon trajectory (h=1..10, resolved at each h)

#### TOP_100_PROSPECT

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 33268 | 90 | 0.27% | 0.996 | 0.525 | 194.1× | 0.0017 | 0.95 |
| 2 | 32192 | 160 | 0.50% | 0.990 | 0.523 | 105.2× | 0.0032 | 0.91 |
| 3 | 30916 | 201 | 0.65% | 0.988 | 0.565 | 86.9× | 0.0040 | 0.92 |
| 4 | 29449 | 214 | 0.73% | 0.987 | 0.544 | 74.9× | 0.0046 | 0.96 |
| 5 | 27782 | 210 | 0.76% | 0.987 | 0.544 | 72.0× | 0.0048 | 1.02 |
| 6 | 25911 | 193 | 0.74% | 0.987 | 0.530 | 71.1× | 0.0048 | 1.09 |
| 7 | 23842 | 183 | 0.77% | 0.987 | 0.533 | 69.4× | 0.0049 | 1.11 |
| 8 | 21766 | 171 | 0.79% | 0.986 | 0.542 | 68.9× | 0.0050 | 1.13 |
| 9 | 19658 | 157 | 0.80% | 0.986 | 0.540 | 67.6× | 0.0051 | 1.15 |
| 10 | 17462 | 144 | 0.82% | 0.986 | 0.538 | 65.2× | 0.0053 | 1.17 |

#### MLB_DEBUT

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 33530 | 402 | 1.20% | 0.974 | 0.451 | 37.6× | 0.0085 | 0.99 |
| 2 | 32448 | 769 | 2.37% | 0.965 | 0.551 | 23.2× | 0.0150 | 1.00 |
| 3 | 31166 | 1090 | 3.50% | 0.956 | 0.578 | 16.5× | 0.0214 | 1.01 |
| 4 | 29688 | 1306 | 4.40% | 0.950 | 0.592 | 13.5× | 0.0264 | 1.02 |
| 5 | 28007 | 1406 | 5.02% | 0.948 | 0.602 | 12.0× | 0.0296 | 1.02 |
| 6 | 26122 | 1402 | 5.37% | 0.947 | 0.605 | 11.3× | 0.0315 | 1.04 |
| 7 | 24034 | 1349 | 5.61% | 0.945 | 0.602 | 10.7× | 0.0329 | 1.04 |
| 8 | 21938 | 1257 | 5.73% | 0.944 | 0.591 | 10.3× | 0.0338 | 1.04 |
| 9 | 19810 | 1142 | 5.76% | 0.942 | 0.576 | 10.0× | 0.0345 | 1.06 |
| 10 | 17588 | 1020 | 5.80% | 0.938 | 0.569 | 9.8× | 0.0352 | 1.07 |

#### ESTABLISHED_MLB

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 33530 | 2 | 0.01% | 0.999 | 0.032 | 528.3× | 0.0001 | 3.45 |
| 2 | 32448 | 35 | 0.11% | 0.989 | 0.184 | 170.7× | 0.0010 | 1.05 |
| 3 | 31166 | 106 | 0.34% | 0.984 | 0.268 | 78.9× | 0.0028 | 0.93 |
| 4 | 29688 | 184 | 0.62% | 0.980 | 0.298 | 48.1× | 0.0050 | 0.95 |
| 5 | 28007 | 259 | 0.92% | 0.977 | 0.319 | 34.5× | 0.0073 | 0.95 |
| 6 | 26122 | 322 | 1.23% | 0.973 | 0.328 | 26.6× | 0.0096 | 0.93 |
| 7 | 24034 | 355 | 1.48% | 0.970 | 0.344 | 23.3× | 0.0113 | 0.95 |
| 8 | 21938 | 365 | 1.66% | 0.968 | 0.339 | 20.4× | 0.0128 | 0.96 |
| 9 | 19810 | 356 | 1.80% | 0.966 | 0.334 | 18.6× | 0.0138 | 0.99 |
| 10 | 17588 | 334 | 1.90% | 0.966 | 0.341 | 18.0× | 0.0144 | 0.99 |

#### STAR_PLUS_ELITE

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 33530 | 1 | 0.00% | 0.999 | 0.031 | 1047.8× | 0.0000 | 3.34 |
| 2 | 32448 | 7 | 0.02% | 0.989 | 0.057 | 264.0× | 0.0002 | 1.32 |
| 3 | 31166 | 16 | 0.05% | 0.980 | 0.059 | 114.1× | 0.0005 | 1.24 |
| 4 | 29688 | 36 | 0.12% | 0.978 | 0.156 | 128.4× | 0.0011 | 1.01 |
| 5 | 28007 | 58 | 0.21% | 0.977 | 0.140 | 67.4× | 0.0019 | 0.90 |
| 6 | 26122 | 74 | 0.28% | 0.974 | 0.149 | 52.6× | 0.0026 | 0.88 |
| 7 | 24034 | 84 | 0.35% | 0.972 | 0.159 | 45.6× | 0.0032 | 0.89 |
| 8 | 21938 | 92 | 0.42% | 0.971 | 0.171 | 40.7× | 0.0038 | 0.86 |
| 9 | 19810 | 101 | 0.51% | 0.969 | 0.173 | 33.8× | 0.0046 | 0.79 |
| 10 | 17588 | 104 | 0.59% | 0.968 | 0.177 | 30.0× | 0.0053 | 0.73 |

## Per-bucket (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 25911 | 193 | 0.74% | 0.987 | 0.530 | 71.1× | 0.145 | 0.714 | 0.285 | 0.407 | 55 | 22 | 138 |
| R1 | 330 | 68 | 20.61% | 0.909 | 0.700 | 3.4× | 0.574 | 0.698 | 0.544 | 0.612 | 37 | 16 | 31 |
| R2-R3 | 544 | 51 | 9.38% | 0.922 | 0.593 | 6.3× | 0.426 | 1.000 | 0.137 | 0.241 | 7 | 0 | 44 |
| R4-R10 | 1814 | 8 | 0.44% | 0.947 | 0.086 | 19.6× | 0.103 | — | 0.000 | — | 0 | 0 | 8 |
| R10+ | 8680 | 9 | 0.10% | 0.961 | 0.354 | 341.0× | 0.051 | 1.000 | 0.222 | 0.364 | 2 | 0 | 7 |
| IFA | 14543 | 57 | 0.39% | 0.985 | 0.360 | 91.9× | 0.105 | 0.600 | 0.158 | 0.250 | 9 | 6 | 48 |

#### MLB_DEBUT

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 26122 | 1402 | 5.37% | 0.947 | 0.605 | 11.3× | 0.349 | 0.768 | 0.290 | 0.421 | 407 | 123 | 995 |
| R1 | 411 | 192 | 46.72% | 0.857 | 0.829 | 1.8× | 0.617 | 0.771 | 0.703 | 0.736 | 135 | 40 | 57 |
| R2-R3 | 582 | 211 | 36.25% | 0.848 | 0.762 | 2.1× | 0.580 | 0.790 | 0.393 | 0.525 | 83 | 22 | 128 |
| R4-R10 | 1815 | 213 | 11.74% | 0.846 | 0.459 | 3.9× | 0.386 | 0.667 | 0.188 | 0.293 | 40 | 20 | 173 |
| R10+ | 8682 | 291 | 3.35% | 0.925 | 0.365 | 10.9× | 0.265 | 0.639 | 0.079 | 0.141 | 23 | 13 | 268 |
| IFA | 14632 | 495 | 3.38% | 0.949 | 0.595 | 17.6× | 0.281 | 0.818 | 0.255 | 0.388 | 126 | 28 | 369 |

#### ESTABLISHED_MLB

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 26122 | 322 | 1.23% | 0.973 | 0.328 | 26.6× | 0.181 | 0.688 | 0.034 | 0.065 | 11 | 5 | 311 |
| R1 | 411 | 76 | 18.49% | 0.837 | 0.461 | 2.5× | 0.453 | 0.714 | 0.066 | 0.120 | 5 | 2 | 71 |
| R2-R3 | 582 | 55 | 9.45% | 0.828 | 0.287 | 3.0× | 0.333 | 0.500 | 0.036 | 0.068 | 2 | 2 | 53 |
| R4-R10 | 1815 | 61 | 3.36% | 0.911 | 0.297 | 8.8× | 0.257 | 0.000 | 0.000 | — | 0 | 1 | 61 |
| R10+ | 8682 | 32 | 0.37% | 0.969 | 0.227 | 61.6× | 0.098 | 1.000 | 0.031 | 0.061 | 1 | 0 | 31 |
| IFA | 14632 | 98 | 0.67% | 0.976 | 0.311 | 46.4× | 0.135 | 1.000 | 0.031 | 0.059 | 3 | 0 | 95 |

#### STAR_PLUS_ELITE

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 26122 | 74 | 0.28% | 0.974 | 0.149 | 52.6× | 0.087 | — | 0.000 | — | 0 | 0 | 74 |
| R1 | 411 | 24 | 5.84% | 0.817 | 0.175 | 3.0× | 0.257 | — | 0.000 | — | 0 | 0 | 24 |
| R2-R3 | 582 | 7 | 1.20% | 0.905 | 0.077 | 6.4× | 0.153 | — | 0.000 | — | 0 | 0 | 7 |
| R4-R10 | 1815 | 11 | 0.61% | 0.955 | 0.248 | 41.0× | 0.122 | — | 0.000 | — | 0 | 0 | 11 |
| R10+ | 8682 | 11 | 0.13% | 0.937 | 0.281 | 222.0× | 0.054 | — | 0.000 | — | 0 | 0 | 11 |
| IFA | 14632 | 21 | 0.14% | 0.985 | 0.256 | 178.6× | 0.064 | — | 0.000 | — | 0 | 0 | 21 |

## Per-yip (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3504 | 86 | 2.45% | 0.957 | 0.461 | 18.8× | 0.245 | 0.571 | 0.279 | 0.375 | 24 | 18 | 62 |
| 1 | 3269 | 59 | 1.80% | 0.978 | 0.621 | 34.4× | 0.221 | 0.850 | 0.288 | 0.430 | 17 | 3 | 42 |
| 2 | 3021 | 31 | 1.03% | 0.985 | 0.654 | 63.8× | 0.169 | 1.000 | 0.387 | 0.558 | 12 | 0 | 19 |
| 3 | 2760 | 13 | 0.47% | 0.984 | 0.499 | 106.0× | 0.115 | 0.667 | 0.154 | 0.250 | 2 | 1 | 11 |
| 4 | 2513 | 4 | 0.16% | 0.992 | 0.553 | 347.7× | 0.068 | — | 0.000 | — | 0 | 0 | 4 |
| 5 | 2288 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 6 | 2092 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 7 | 1891 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 8 | 1702 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 9 | 1528 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 1343 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

#### MLB_DEBUT

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3524 | 337 | 9.56% | 0.893 | 0.510 | 5.3× | 0.400 | 0.719 | 0.190 | 0.300 | 64 | 25 | 273 |
| 1 | 3297 | 321 | 9.74% | 0.913 | 0.603 | 6.2× | 0.424 | 0.740 | 0.293 | 0.420 | 94 | 33 | 227 |
| 2 | 3062 | 284 | 9.28% | 0.938 | 0.678 | 7.3× | 0.440 | 0.745 | 0.380 | 0.503 | 108 | 37 | 176 |
| 3 | 2796 | 194 | 6.94% | 0.952 | 0.701 | 10.1× | 0.398 | 0.844 | 0.392 | 0.535 | 76 | 14 | 118 |
| 4 | 2541 | 124 | 4.88% | 0.950 | 0.674 | 13.8× | 0.336 | 0.845 | 0.395 | 0.538 | 49 | 9 | 75 |
| 5 | 2304 | 63 | 2.73% | 0.935 | 0.478 | 17.5× | 0.246 | 0.714 | 0.159 | 0.260 | 10 | 4 | 53 |
| 6 | 2104 | 37 | 1.76% | 0.947 | 0.417 | 23.7× | 0.204 | 0.800 | 0.108 | 0.190 | 4 | 1 | 33 |
| 7 | 1899 | 21 | 1.11% | 0.972 | 0.455 | 41.1× | 0.171 | 1.000 | 0.048 | 0.091 | 1 | 0 | 20 |
| 8 | 1710 | 12 | 0.70% | 0.935 | 0.313 | 44.6× | 0.126 | 1.000 | 0.083 | 0.154 | 1 | 0 | 11 |
| 9 | 1535 | 7 | 0.46% | 0.968 | 0.291 | 63.8× | 0.109 | — | 0.000 | — | 0 | 0 | 7 |
| 10 | 1350 | 2 | 0.15% | 0.878 | 0.026 | 17.5× | 0.050 | — | 0.000 | — | 0 | 0 | 2 |

#### ESTABLISHED_MLB

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3524 | 84 | 2.38% | 0.930 | 0.251 | 10.5× | 0.227 | 0.500 | 0.012 | 0.023 | 1 | 1 | 83 |
| 1 | 3297 | 85 | 2.58% | 0.943 | 0.346 | 13.4× | 0.243 | 0.667 | 0.047 | 0.088 | 4 | 2 | 81 |
| 2 | 3062 | 77 | 2.51% | 0.963 | 0.388 | 15.4× | 0.251 | 0.600 | 0.039 | 0.073 | 3 | 2 | 74 |
| 3 | 2796 | 43 | 1.54% | 0.976 | 0.400 | 26.0× | 0.203 | 1.000 | 0.047 | 0.089 | 2 | 0 | 41 |
| 4 | 2541 | 23 | 0.91% | 0.982 | 0.494 | 54.6× | 0.158 | 1.000 | 0.043 | 0.083 | 1 | 0 | 22 |
| 5 | 2304 | 5 | 0.22% | 0.988 | 0.130 | 60.1× | 0.079 | — | 0.000 | — | 0 | 0 | 5 |
| 6 | 2104 | 3 | 0.14% | 0.971 | 0.197 | 138.1× | 0.062 | — | 0.000 | — | 0 | 0 | 3 |
| 7 | 1899 | 1 | 0.05% | 0.999 | 0.333 | 633.0× | 0.040 | — | 0.000 | — | 0 | 0 | 1 |
| 8 | 1710 | 1 | 0.06% | 0.995 | 0.111 | 190.0× | 0.041 | — | 0.000 | — | 0 | 0 | 1 |
| 9 | 1535 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 1350 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

#### STAR_PLUS_ELITE

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3524 | 23 | 0.65% | 0.941 | 0.122 | 18.8× | 0.123 | — | 0.000 | — | 0 | 0 | 23 |
| 1 | 3297 | 19 | 0.58% | 0.963 | 0.199 | 34.5× | 0.121 | — | 0.000 | — | 0 | 0 | 19 |
| 2 | 3062 | 18 | 0.59% | 0.969 | 0.251 | 42.7× | 0.124 | — | 0.000 | — | 0 | 0 | 18 |
| 3 | 2796 | 8 | 0.29% | 0.973 | 0.167 | 58.3× | 0.087 | — | 0.000 | — | 0 | 0 | 8 |
| 4 | 2541 | 4 | 0.16% | 0.976 | 0.083 | 52.6× | 0.065 | — | 0.000 | — | 0 | 0 | 4 |
| 5 | 2304 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 6 | 2104 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 7 | 1899 | 1 | 0.05% | 0.981 | 0.026 | 50.0× | 0.038 | — | 0.000 | — | 0 | 0 | 1 |
| 8 | 1710 | 1 | 0.06% | 0.991 | 0.062 | 106.9× | 0.041 | — | 0.000 | — | 0 | 0 | 1 |
| 9 | 1535 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 1350 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

## Per-level (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 25911 | 193 | 0.74% | 0.987 | 0.530 | 71.1× | 0.145 | 0.714 | 0.285 | 0.407 | 55 | 22 | 138 |
| RK | 3771 | 65 | 1.72% | 0.960 | 0.510 | 29.6× | 0.208 | 0.727 | 0.246 | 0.368 | 16 | 6 | 49 |
| A- | 987 | 19 | 1.93% | 0.971 | 0.492 | 25.5× | 0.224 | 0.571 | 0.211 | 0.308 | 4 | 3 | 15 |
| A | 1340 | 40 | 2.99% | 0.968 | 0.589 | 19.7× | 0.276 | 0.684 | 0.325 | 0.441 | 13 | 6 | 27 |
| A+ | 1334 | 28 | 2.10% | 0.983 | 0.560 | 26.7× | 0.240 | 0.800 | 0.286 | 0.421 | 8 | 2 | 20 |
| AA | 1158 | 26 | 2.25% | 0.984 | 0.665 | 29.6× | 0.248 | 0.846 | 0.423 | 0.564 | 11 | 2 | 15 |
| AAA | 1561 | 6 | 0.38% | 0.999 | 0.760 | 197.7× | 0.107 | 1.000 | 0.333 | 0.500 | 2 | 0 | 4 |
| NONE | 15760 | 9 | 0.06% | 0.990 | 0.152 | 266.0× | 0.041 | 0.250 | 0.111 | 0.154 | 1 | 3 | 8 |

#### MLB_DEBUT

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 26122 | 1402 | 5.37% | 0.947 | 0.605 | 11.3× | 0.349 | 0.768 | 0.290 | 0.421 | 407 | 123 | 995 |
| RK | 3775 | 193 | 5.11% | 0.915 | 0.502 | 9.8× | 0.316 | 0.742 | 0.119 | 0.205 | 23 | 8 | 170 |
| A- | 989 | 119 | 12.03% | 0.831 | 0.462 | 3.8× | 0.373 | 0.692 | 0.151 | 0.248 | 18 | 8 | 101 |
| A | 1362 | 221 | 16.23% | 0.870 | 0.633 | 3.9× | 0.473 | 0.764 | 0.308 | 0.439 | 68 | 21 | 153 |
| A+ | 1356 | 235 | 17.33% | 0.859 | 0.663 | 3.8× | 0.471 | 0.832 | 0.336 | 0.479 | 79 | 16 | 156 |
| AA | 1234 | 331 | 26.82% | 0.840 | 0.712 | 2.7× | 0.522 | 0.775 | 0.438 | 0.560 | 145 | 42 | 186 |
| AAA | 1609 | 236 | 14.67% | 0.864 | 0.580 | 4.0× | 0.446 | 0.733 | 0.267 | 0.391 | 63 | 23 | 173 |
| NONE | 15797 | 67 | 0.42% | 0.910 | 0.376 | 88.7× | 0.092 | 0.688 | 0.164 | 0.265 | 11 | 5 | 56 |

#### ESTABLISHED_MLB

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 26122 | 322 | 1.23% | 0.973 | 0.328 | 26.6× | 0.181 | 0.688 | 0.034 | 0.065 | 11 | 5 | 311 |
| RK | 3775 | 37 | 0.98% | 0.945 | 0.168 | 17.2× | 0.152 | — | 0.000 | — | 0 | 0 | 37 |
| A- | 989 | 19 | 1.92% | 0.916 | 0.173 | 9.0× | 0.198 | — | 0.000 | — | 0 | 0 | 19 |
| A | 1362 | 56 | 4.11% | 0.922 | 0.347 | 8.4× | 0.290 | 0.750 | 0.054 | 0.100 | 3 | 1 | 53 |
| A+ | 1356 | 65 | 4.79% | 0.924 | 0.402 | 8.4× | 0.314 | 0.500 | 0.015 | 0.030 | 1 | 1 | 64 |
| AA | 1234 | 91 | 7.37% | 0.898 | 0.407 | 5.5× | 0.361 | 0.714 | 0.055 | 0.102 | 5 | 2 | 86 |
| AAA | 1609 | 49 | 3.05% | 0.916 | 0.374 | 12.3× | 0.248 | 0.667 | 0.041 | 0.077 | 2 | 1 | 47 |
| NONE | 15797 | 5 | 0.03% | 0.999 | 0.166 | 524.9× | 0.031 | — | 0.000 | — | 0 | 0 | 5 |

#### STAR_PLUS_ELITE

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 26122 | 74 | 0.28% | 0.974 | 0.149 | 52.6× | 0.087 | — | 0.000 | — | 0 | 0 | 74 |
| RK | 3775 | 13 | 0.34% | 0.937 | 0.234 | 67.9× | 0.089 | — | 0.000 | — | 0 | 0 | 13 |
| A- | 989 | 2 | 0.20% | 0.981 | 0.073 | 36.1× | 0.075 | — | 0.000 | — | 0 | 0 | 2 |
| A | 1362 | 12 | 0.88% | 0.951 | 0.210 | 23.8× | 0.146 | — | 0.000 | — | 0 | 0 | 12 |
| A+ | 1356 | 16 | 1.18% | 0.934 | 0.172 | 14.6× | 0.162 | — | 0.000 | — | 0 | 0 | 16 |
| AA | 1234 | 18 | 1.46% | 0.917 | 0.218 | 15.0× | 0.173 | — | 0.000 | — | 0 | 0 | 18 |
| AAA | 1609 | 12 | 0.75% | 0.913 | 0.208 | 27.9× | 0.123 | — | 0.000 | — | 0 | 0 | 12 |
| NONE | 15797 | 1 | 0.01% | 1.000 | 0.250 | 3949.2× | 0.014 | — | 0.000 | — | 0 | 0 | 1 |

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
