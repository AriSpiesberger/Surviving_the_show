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
| TOP_100_PROSPECT | 15144 | 1.70% | **0.477** | 28.1× | 0.973 | 0.212 | 0.649 | 0.288 | 0.399 |
| MLB_DEBUT | 15343 | 14.62% | **0.667** | 4.6× | 0.911 | 0.503 | 0.715 | 0.427 | 0.534 |
| ESTABLISHED_MLB | 15343 | 4.48% | **0.379** | 8.5× | 0.918 | 0.299 | 0.578 | 0.140 | 0.225 |
| STAR_PLUS_ELITE | 15343 | 0.84% | **0.122** | 14.5× | 0.904 | 0.128 | 0.750 | 0.023 | 0.045 |
| **weighted-AP** | | | **0.462** | | | | | | |

(MLB_DEBUT 2× weight, others 1×, per-event eligibility filters. Scores =
`xp_<event>_h6` vs realized-within-6y, on rows resolved at h=6.)

## Per-horizon trajectory (h=1..10, resolved at each h)

#### TOP_100_PROSPECT

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 19069 | 109 | 0.57% | 0.985 | 0.468 | 81.9× | 0.0040 | 0.97 |
| 2 | 18488 | 199 | 1.08% | 0.980 | 0.504 | 46.9× | 0.0072 | 0.89 |
| 3 | 17817 | 250 | 1.40% | 0.977 | 0.486 | 34.7× | 0.0095 | 0.90 |
| 4 | 17041 | 267 | 1.57% | 0.975 | 0.481 | 30.7× | 0.0107 | 0.94 |
| 5 | 16150 | 269 | 1.67% | 0.974 | 0.482 | 29.0× | 0.0113 | 0.98 |
| 6 | 15144 | 257 | 1.70% | 0.973 | 0.477 | 28.1× | 0.0115 | 1.02 |
| 7 | 14039 | 247 | 1.76% | 0.972 | 0.482 | 27.4× | 0.0119 | 1.02 |
| 8 | 12964 | 237 | 1.83% | 0.971 | 0.489 | 26.8× | 0.0123 | 1.02 |
| 9 | 11876 | 225 | 1.89% | 0.969 | 0.491 | 25.9× | 0.0128 | 1.01 |
| 10 | 10785 | 210 | 1.95% | 0.968 | 0.492 | 25.3× | 0.0131 | 1.01 |

#### MLB_DEBUT

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 19315 | 594 | 3.08% | 0.947 | 0.452 | 14.7× | 0.0217 | 1.04 |
| 2 | 18729 | 1172 | 6.26% | 0.933 | 0.567 | 9.1× | 0.0387 | 1.00 |
| 3 | 18051 | 1679 | 9.30% | 0.924 | 0.621 | 6.7× | 0.0532 | 1.00 |
| 4 | 17265 | 2034 | 11.78% | 0.918 | 0.651 | 5.5× | 0.0641 | 0.99 |
| 5 | 16361 | 2216 | 13.54% | 0.913 | 0.661 | 4.9× | 0.0722 | 0.99 |
| 6 | 15343 | 2243 | 14.62% | 0.911 | 0.667 | 4.6× | 0.0768 | 0.99 |
| 7 | 14226 | 2185 | 15.36% | 0.907 | 0.661 | 4.3× | 0.0809 | 0.99 |
| 8 | 13141 | 2085 | 15.87% | 0.903 | 0.653 | 4.1× | 0.0843 | 1.00 |
| 9 | 12038 | 1960 | 16.28% | 0.900 | 0.643 | 4.0× | 0.0872 | 1.00 |
| 10 | 10932 | 1825 | 16.69% | 0.895 | 0.638 | 3.8× | 0.0902 | 1.01 |

#### ESTABLISHED_MLB

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 19315 | 4 | 0.02% | 0.959 | 0.008 | 37.3× | 0.0002 | 2.07 |
| 2 | 18729 | 75 | 0.40% | 0.964 | 0.157 | 39.2× | 0.0037 | 0.97 |
| 3 | 18051 | 225 | 1.25% | 0.949 | 0.262 | 21.0× | 0.0105 | 0.96 |
| 4 | 17265 | 402 | 2.33% | 0.937 | 0.319 | 13.7× | 0.0188 | 0.97 |
| 5 | 16361 | 562 | 3.44% | 0.926 | 0.356 | 10.4× | 0.0266 | 0.97 |
| 6 | 15343 | 687 | 4.48% | 0.918 | 0.379 | 8.5× | 0.0338 | 0.97 |
| 7 | 14226 | 766 | 5.38% | 0.911 | 0.388 | 7.2× | 0.0401 | 0.97 |
| 8 | 13141 | 813 | 6.19% | 0.907 | 0.398 | 6.4× | 0.0454 | 0.95 |
| 9 | 12038 | 820 | 6.81% | 0.902 | 0.397 | 5.8× | 0.0498 | 0.95 |
| 10 | 10932 | 793 | 7.25% | 0.897 | 0.399 | 5.5× | 0.0528 | 0.93 |

#### STAR_PLUS_ELITE

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 19315 | 5 | 0.03% | 0.932 | 0.003 | 12.1× | 0.0003 | 0.58 |
| 2 | 18729 | 19 | 0.10% | 0.910 | 0.016 | 15.7× | 0.0011 | 0.64 |
| 3 | 18051 | 41 | 0.23% | 0.904 | 0.031 | 13.9× | 0.0023 | 0.73 |
| 4 | 17265 | 68 | 0.39% | 0.899 | 0.059 | 15.0× | 0.0039 | 0.80 |
| 5 | 16361 | 100 | 0.61% | 0.906 | 0.088 | 14.5× | 0.0059 | 0.81 |
| 6 | 15343 | 129 | 0.84% | 0.904 | 0.122 | 14.5× | 0.0079 | 0.80 |
| 7 | 14226 | 146 | 1.03% | 0.902 | 0.133 | 13.0× | 0.0095 | 0.83 |
| 8 | 13141 | 154 | 1.17% | 0.896 | 0.134 | 11.5× | 0.0109 | 0.87 |
| 9 | 12038 | 162 | 1.35% | 0.892 | 0.143 | 10.6× | 0.0124 | 0.85 |
| 10 | 10932 | 164 | 1.50% | 0.889 | 0.148 | 9.9× | 0.0138 | 0.80 |

## Per-bucket (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 15144 | 257 | 1.70% | 0.973 | 0.477 | 28.1× | 0.212 | 0.649 | 0.288 | 0.399 | 74 | 40 | 183 |
| R1 | 461 | 96 | 20.82% | 0.918 | 0.717 | 3.4× | 0.588 | 0.659 | 0.583 | 0.619 | 56 | 29 | 40 |
| R2-R3 | 966 | 58 | 6.00% | 0.884 | 0.330 | 5.5× | 0.316 | 0.500 | 0.138 | 0.216 | 8 | 8 | 50 |
| R4-R10 | 3535 | 63 | 1.78% | 0.950 | 0.340 | 19.1× | 0.206 | 0.778 | 0.111 | 0.194 | 7 | 2 | 56 |
| R10+ | 10182 | 40 | 0.39% | 0.970 | 0.236 | 60.1× | 0.102 | 0.750 | 0.075 | 0.136 | 3 | 1 | 37 |

#### MLB_DEBUT

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 15343 | 2243 | 14.62% | 0.911 | 0.667 | 4.6× | 0.503 | 0.715 | 0.427 | 0.534 | 957 | 382 | 1286 |
| R1 | 565 | 281 | 49.73% | 0.884 | 0.879 | 1.8× | 0.665 | 0.753 | 0.858 | 0.802 | 241 | 79 | 40 |
| R2-R3 | 983 | 331 | 33.67% | 0.882 | 0.790 | 2.3× | 0.626 | 0.738 | 0.671 | 0.703 | 222 | 79 | 109 |
| R4-R10 | 3561 | 653 | 18.34% | 0.893 | 0.662 | 3.6× | 0.527 | 0.660 | 0.475 | 0.552 | 310 | 160 | 343 |
| R10+ | 10234 | 978 | 9.56% | 0.904 | 0.537 | 5.6× | 0.412 | 0.742 | 0.188 | 0.300 | 184 | 64 | 794 |

#### ESTABLISHED_MLB

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 15343 | 687 | 4.48% | 0.918 | 0.379 | 8.5× | 0.299 | 0.578 | 0.140 | 0.225 | 96 | 70 | 591 |
| R1 | 565 | 147 | 26.02% | 0.808 | 0.594 | 2.3× | 0.468 | 0.587 | 0.415 | 0.486 | 61 | 43 | 86 |
| R2-R3 | 983 | 101 | 10.27% | 0.870 | 0.429 | 4.2× | 0.389 | 0.500 | 0.119 | 0.192 | 12 | 12 | 89 |
| R4-R10 | 3561 | 214 | 6.01% | 0.878 | 0.304 | 5.1× | 0.311 | 0.548 | 0.079 | 0.139 | 17 | 14 | 197 |
| R10+ | 10234 | 225 | 2.20% | 0.919 | 0.246 | 11.2× | 0.213 | 0.857 | 0.027 | 0.052 | 6 | 1 | 219 |

#### STAR_PLUS_ELITE

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 15343 | 129 | 0.84% | 0.904 | 0.122 | 14.5× | 0.128 | 0.750 | 0.023 | 0.045 | 3 | 1 | 126 |
| R1 | 565 | 44 | 7.79% | 0.769 | 0.245 | 3.1× | 0.250 | 0.750 | 0.068 | 0.125 | 3 | 1 | 41 |
| R2-R3 | 983 | 18 | 1.83% | 0.847 | 0.103 | 5.6× | 0.161 | — | 0.000 | — | 0 | 0 | 18 |
| R4-R10 | 3561 | 30 | 0.84% | 0.870 | 0.045 | 5.3× | 0.117 | — | 0.000 | — | 0 | 0 | 30 |
| R10+ | 10234 | 37 | 0.36% | 0.873 | 0.043 | 12.0× | 0.078 | — | 0.000 | — | 0 | 0 | 37 |

## Per-yip (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2220 | 116 | 5.23% | 0.928 | 0.500 | 9.6× | 0.330 | 0.625 | 0.259 | 0.366 | 30 | 18 | 86 |
| 1 | 2055 | 82 | 3.99% | 0.935 | 0.472 | 11.8× | 0.295 | 0.643 | 0.329 | 0.435 | 27 | 15 | 55 |
| 2 | 1870 | 41 | 2.19% | 0.955 | 0.486 | 22.2× | 0.231 | 0.714 | 0.244 | 0.364 | 10 | 4 | 31 |
| 3 | 1666 | 14 | 0.84% | 0.984 | 0.451 | 53.7× | 0.153 | 0.667 | 0.429 | 0.522 | 6 | 3 | 8 |
| 4 | 1436 | 4 | 0.28% | 0.994 | 0.608 | 218.4× | 0.090 | 1.000 | 0.250 | 0.400 | 1 | 0 | 3 |
| 5 | 1263 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 6 | 1117 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 7 | 1017 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 8 | 924 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 9 | 831 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 745 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

#### MLB_DEBUT

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2223 | 516 | 23.21% | 0.835 | 0.608 | 2.6× | 0.489 | 0.626 | 0.386 | 0.477 | 199 | 119 | 317 |
| 1 | 2082 | 521 | 25.02% | 0.859 | 0.698 | 2.8× | 0.539 | 0.732 | 0.461 | 0.565 | 240 | 88 | 281 |
| 2 | 1918 | 465 | 24.24% | 0.876 | 0.727 | 3.0× | 0.558 | 0.758 | 0.518 | 0.616 | 241 | 77 | 224 |
| 3 | 1709 | 340 | 19.89% | 0.891 | 0.704 | 3.5× | 0.540 | 0.749 | 0.456 | 0.567 | 155 | 52 | 185 |
| 4 | 1466 | 202 | 13.78% | 0.896 | 0.647 | 4.7× | 0.473 | 0.747 | 0.366 | 0.492 | 74 | 25 | 128 |
| 5 | 1278 | 106 | 8.29% | 0.925 | 0.604 | 7.3× | 0.406 | 0.780 | 0.302 | 0.435 | 32 | 9 | 74 |
| 6 | 1128 | 52 | 4.61% | 0.941 | 0.475 | 10.3× | 0.320 | 0.632 | 0.231 | 0.338 | 12 | 7 | 40 |
| 7 | 1025 | 24 | 2.34% | 0.944 | 0.285 | 12.2× | 0.232 | 0.500 | 0.125 | 0.200 | 3 | 3 | 21 |
| 8 | 929 | 10 | 1.08% | 0.961 | 0.239 | 22.2× | 0.165 | 0.333 | 0.100 | 0.154 | 1 | 2 | 9 |
| 9 | 836 | 5 | 0.60% | 0.916 | 0.055 | 9.2× | 0.111 | — | 0.000 | — | 0 | 0 | 5 |
| 10 | 749 | 2 | 0.27% | 0.922 | 0.508 | 190.4× | 0.076 | — | 0.000 | — | 0 | 0 | 2 |

#### ESTABLISHED_MLB

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2223 | 151 | 6.79% | 0.864 | 0.320 | 4.7× | 0.317 | 0.583 | 0.093 | 0.160 | 14 | 10 | 137 |
| 1 | 2082 | 168 | 8.07% | 0.879 | 0.401 | 5.0× | 0.357 | 0.532 | 0.149 | 0.233 | 25 | 22 | 143 |
| 2 | 1918 | 153 | 7.98% | 0.888 | 0.450 | 5.6× | 0.364 | 0.542 | 0.170 | 0.259 | 26 | 22 | 127 |
| 3 | 1709 | 105 | 6.14% | 0.889 | 0.374 | 6.1× | 0.323 | 0.606 | 0.190 | 0.290 | 20 | 13 | 85 |
| 4 | 1466 | 63 | 4.30% | 0.899 | 0.401 | 9.3× | 0.280 | 0.786 | 0.175 | 0.286 | 11 | 3 | 52 |
| 5 | 1278 | 30 | 2.35% | 0.923 | 0.198 | 8.4× | 0.222 | — | 0.000 | — | 0 | 0 | 30 |
| 6 | 1128 | 12 | 1.06% | 0.920 | 0.202 | 19.0× | 0.149 | — | 0.000 | — | 0 | 0 | 12 |
| 7 | 1025 | 5 | 0.49% | 0.930 | 0.065 | 13.3× | 0.104 | — | 0.000 | — | 0 | 0 | 5 |
| 8 | 929 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 9 | 836 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 749 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

#### STAR_PLUS_ELITE

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2223 | 33 | 1.48% | 0.814 | 0.103 | 6.9× | 0.132 | — | 0.000 | — | 0 | 0 | 33 |
| 1 | 2082 | 39 | 1.87% | 0.859 | 0.134 | 7.1× | 0.169 | — | 0.000 | — | 0 | 0 | 39 |
| 2 | 1918 | 29 | 1.51% | 0.895 | 0.205 | 13.6× | 0.167 | 0.750 | 0.103 | 0.182 | 3 | 1 | 26 |
| 3 | 1709 | 15 | 0.88% | 0.865 | 0.150 | 17.1× | 0.118 | — | 0.000 | — | 0 | 0 | 15 |
| 4 | 1466 | 11 | 0.75% | 0.808 | 0.088 | 11.8× | 0.092 | — | 0.000 | — | 0 | 0 | 11 |
| 5 | 1278 | 2 | 0.16% | 0.918 | 0.018 | 11.4× | 0.057 | — | 0.000 | — | 0 | 0 | 2 |
| 6 | 1128 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 7 | 1025 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 8 | 929 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 9 | 836 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 749 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

## Per-level (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 15144 | 257 | 1.70% | 0.973 | 0.477 | 28.1× | 0.212 | 0.649 | 0.288 | 0.399 | 74 | 40 | 183 |
| A | 1424 | 41 | 2.88% | 0.977 | 0.631 | 21.9× | 0.276 | 0.719 | 0.561 | 0.630 | 23 | 9 | 18 |
| A+ | 1524 | 31 | 2.03% | 0.982 | 0.482 | 23.7× | 0.236 | 0.529 | 0.290 | 0.375 | 9 | 8 | 22 |
| AA | 1328 | 20 | 1.51% | 0.988 | 0.655 | 43.5× | 0.206 | 0.650 | 0.650 | 0.650 | 13 | 7 | 7 |
| AAA | 946 | 8 | 0.85% | 0.997 | 0.787 | 93.0× | 0.158 | 1.000 | 0.250 | 0.400 | 2 | 0 | 6 |
| NONE | 9922 | 157 | 1.58% | 0.966 | 0.403 | 25.4× | 0.202 | 0.628 | 0.172 | 0.270 | 27 | 16 | 130 |

#### MLB_DEBUT

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 15343 | 2243 | 14.62% | 0.911 | 0.667 | 4.6× | 0.503 | 0.715 | 0.427 | 0.534 | 957 | 382 | 1286 |
| A | 1439 | 262 | 18.21% | 0.870 | 0.668 | 3.7× | 0.495 | 0.688 | 0.454 | 0.547 | 119 | 54 | 143 |
| A+ | 1541 | 294 | 19.08% | 0.866 | 0.674 | 3.5× | 0.499 | 0.730 | 0.507 | 0.598 | 149 | 55 | 145 |
| AA | 1384 | 387 | 27.96% | 0.879 | 0.787 | 2.8× | 0.590 | 0.773 | 0.607 | 0.680 | 235 | 69 | 152 |
| AAA | 991 | 290 | 29.26% | 0.826 | 0.730 | 2.5× | 0.514 | 0.782 | 0.531 | 0.632 | 154 | 43 | 136 |
| NONE | 9988 | 1010 | 10.11% | 0.928 | 0.576 | 5.7× | 0.447 | 0.651 | 0.297 | 0.408 | 300 | 161 | 710 |

#### ESTABLISHED_MLB

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 15343 | 687 | 4.48% | 0.918 | 0.379 | 8.5× | 0.299 | 0.578 | 0.140 | 0.225 | 96 | 70 | 591 |
| A | 1439 | 72 | 5.00% | 0.912 | 0.463 | 9.3× | 0.311 | 0.727 | 0.111 | 0.193 | 8 | 3 | 64 |
| A+ | 1541 | 83 | 5.39% | 0.916 | 0.342 | 6.3× | 0.325 | 0.400 | 0.048 | 0.086 | 4 | 6 | 79 |
| AA | 1384 | 128 | 9.25% | 0.886 | 0.482 | 5.2× | 0.387 | 0.583 | 0.328 | 0.420 | 42 | 30 | 86 |
| AAA | 991 | 81 | 8.17% | 0.851 | 0.487 | 6.0× | 0.333 | 0.595 | 0.272 | 0.373 | 22 | 15 | 59 |
| NONE | 9988 | 323 | 3.23% | 0.927 | 0.305 | 9.4× | 0.262 | 0.556 | 0.062 | 0.111 | 20 | 16 | 303 |

#### STAR_PLUS_ELITE

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 15343 | 129 | 0.84% | 0.904 | 0.122 | 14.5× | 0.128 | 0.750 | 0.023 | 0.045 | 3 | 1 | 126 |
| A | 1439 | 15 | 1.04% | 0.935 | 0.464 | 44.5× | 0.153 | — | 0.000 | — | 0 | 0 | 15 |
| A+ | 1541 | 17 | 1.10% | 0.888 | 0.072 | 6.5× | 0.140 | — | 0.000 | — | 0 | 0 | 17 |
| AA | 1384 | 22 | 1.59% | 0.885 | 0.162 | 10.2× | 0.167 | 0.667 | 0.091 | 0.160 | 2 | 1 | 20 |
| AAA | 991 | 15 | 1.51% | 0.910 | 0.228 | 15.1× | 0.174 | 1.000 | 0.067 | 0.125 | 1 | 0 | 14 |
| NONE | 9988 | 60 | 0.60% | 0.901 | 0.103 | 17.1× | 0.107 | — | 0.000 | — | 0 | 0 | 60 |

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
