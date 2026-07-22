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
| TOP_100_PROSPECT | 25631 | 0.75% | **0.546** | 72.9× | 0.987 | 0.146 | 0.740 | 0.297 | 0.424 |
| MLB_DEBUT | 25814 | 5.85% | **0.662** | 11.3× | 0.953 | 0.369 | 0.789 | 0.349 | 0.484 |
| ESTABLISHED_MLB | 25814 | 1.69% | **0.390** | 23.0× | 0.972 | 0.211 | 0.778 | 0.032 | 0.062 |
| STAR_PLUS_ELITE | 25814 | 0.30% | **0.134** | 44.5× | 0.977 | 0.091 | — | 0.000 | — |
| **weighted-AP** | | | **0.479** | | | | | | |

(MLB_DEBUT 2× weight, others 1×, per-event eligibility filters. Scores =
`xp_<event>_h6` vs realized-within-6y, on rows resolved at h=6.)

## Per-horizon trajectory (h=1..10, resolved at each h)

#### TOP_100_PROSPECT

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32955 | 89 | 0.27% | 0.994 | 0.495 | 183.4× | 0.0018 | 0.87 |
| 2 | 31881 | 159 | 0.50% | 0.990 | 0.536 | 107.4× | 0.0032 | 0.87 |
| 3 | 30609 | 200 | 0.65% | 0.989 | 0.566 | 86.7× | 0.0040 | 0.87 |
| 4 | 29147 | 213 | 0.73% | 0.987 | 0.536 | 73.4× | 0.0046 | 0.91 |
| 5 | 27489 | 209 | 0.76% | 0.987 | 0.546 | 71.8× | 0.0048 | 0.97 |
| 6 | 25631 | 192 | 0.75% | 0.987 | 0.546 | 72.9× | 0.0047 | 1.04 |
| 7 | 23578 | 182 | 0.77% | 0.987 | 0.552 | 71.5× | 0.0048 | 1.06 |
| 8 | 21522 | 170 | 0.79% | 0.987 | 0.552 | 69.9× | 0.0049 | 1.09 |
| 9 | 19436 | 156 | 0.80% | 0.986 | 0.553 | 68.9× | 0.0050 | 1.11 |
| 10 | 17265 | 143 | 0.83% | 0.986 | 0.558 | 67.4× | 0.0051 | 1.13 |

#### MLB_DEBUT

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 33187 | 433 | 1.30% | 0.978 | 0.490 | 37.5× | 0.0087 | 0.96 |
| 2 | 32107 | 827 | 2.58% | 0.970 | 0.602 | 23.4× | 0.0150 | 0.96 |
| 3 | 30829 | 1173 | 3.80% | 0.961 | 0.629 | 16.5× | 0.0215 | 0.99 |
| 4 | 29356 | 1405 | 4.79% | 0.956 | 0.642 | 13.4× | 0.0266 | 1.00 |
| 5 | 27685 | 1511 | 5.46% | 0.954 | 0.655 | 12.0× | 0.0297 | 1.01 |
| 6 | 25814 | 1511 | 5.85% | 0.953 | 0.662 | 11.3× | 0.0315 | 1.03 |
| 7 | 23744 | 1460 | 6.15% | 0.953 | 0.667 | 10.8× | 0.0327 | 1.02 |
| 8 | 21670 | 1367 | 6.31% | 0.952 | 0.663 | 10.5× | 0.0336 | 1.02 |
| 9 | 19567 | 1252 | 6.40% | 0.950 | 0.657 | 10.3× | 0.0343 | 1.03 |
| 10 | 17374 | 1127 | 6.49% | 0.948 | 0.649 | 10.0× | 0.0353 | 1.03 |

#### ESTABLISHED_MLB

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 33187 | 2 | 0.01% | 0.998 | 0.039 | 643.3× | 0.0001 | 3.58 |
| 2 | 32107 | 42 | 0.13% | 0.989 | 0.131 | 99.9× | 0.0012 | 0.90 |
| 3 | 30829 | 142 | 0.46% | 0.985 | 0.305 | 66.1× | 0.0038 | 0.87 |
| 4 | 29356 | 256 | 0.87% | 0.980 | 0.345 | 39.6× | 0.0068 | 0.86 |
| 5 | 27685 | 360 | 1.30% | 0.975 | 0.370 | 28.5× | 0.0099 | 0.88 |
| 6 | 25814 | 437 | 1.69% | 0.972 | 0.390 | 23.0× | 0.0126 | 0.90 |
| 7 | 23744 | 479 | 2.02% | 0.970 | 0.412 | 20.4× | 0.0146 | 0.92 |
| 8 | 21670 | 495 | 2.28% | 0.968 | 0.422 | 18.5× | 0.0164 | 0.93 |
| 9 | 19567 | 490 | 2.50% | 0.967 | 0.421 | 16.8× | 0.0178 | 0.93 |
| 10 | 17374 | 468 | 2.69% | 0.966 | 0.438 | 16.3× | 0.0188 | 0.90 |

#### STAR_PLUS_ELITE

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 33187 | 1 | 0.00% | 0.999 | 0.048 | 1580.3× | 0.0000 | 3.17 |
| 2 | 32107 | 7 | 0.02% | 0.995 | 0.097 | 443.6× | 0.0002 | 1.03 |
| 3 | 30829 | 17 | 0.06% | 0.989 | 0.097 | 176.6× | 0.0005 | 1.02 |
| 4 | 29356 | 38 | 0.13% | 0.984 | 0.165 | 127.7× | 0.0012 | 0.80 |
| 5 | 27685 | 61 | 0.22% | 0.981 | 0.149 | 67.5× | 0.0020 | 0.77 |
| 6 | 25814 | 78 | 0.30% | 0.977 | 0.134 | 44.5× | 0.0028 | 0.79 |
| 7 | 23744 | 89 | 0.37% | 0.977 | 0.153 | 40.7× | 0.0034 | 0.82 |
| 8 | 21670 | 97 | 0.45% | 0.977 | 0.172 | 38.4× | 0.0040 | 0.83 |
| 9 | 19567 | 106 | 0.54% | 0.976 | 0.176 | 32.6× | 0.0049 | 0.77 |
| 10 | 17374 | 109 | 0.63% | 0.975 | 0.187 | 29.9× | 0.0056 | 0.72 |

## Per-bucket (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 25631 | 192 | 0.75% | 0.987 | 0.546 | 72.9× | 0.146 | 0.740 | 0.297 | 0.424 | 57 | 20 | 135 |
| R1 | 320 | 68 | 21.25% | 0.912 | 0.704 | 3.3× | 0.583 | 0.714 | 0.515 | 0.598 | 35 | 14 | 33 |
| R2-R3 | 529 | 50 | 9.45% | 0.915 | 0.535 | 5.7× | 0.421 | 0.714 | 0.200 | 0.312 | 10 | 4 | 40 |
| R4-R10 | 1763 | 8 | 0.45% | 0.940 | 0.081 | 17.9× | 0.102 | — | 0.000 | — | 0 | 0 | 8 |
| R10+ | 8574 | 9 | 0.10% | 0.950 | 0.342 | 326.0× | 0.050 | 1.000 | 0.333 | 0.500 | 3 | 0 | 6 |
| IFA | 14445 | 57 | 0.39% | 0.987 | 0.427 | 108.3× | 0.106 | 0.818 | 0.158 | 0.265 | 9 | 2 | 48 |

#### MLB_DEBUT

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 25814 | 1511 | 5.85% | 0.953 | 0.662 | 11.3× | 0.369 | 0.789 | 0.349 | 0.484 | 528 | 141 | 983 |
| R1 | 393 | 203 | 51.65% | 0.897 | 0.902 | 1.7× | 0.688 | 0.833 | 0.764 | 0.797 | 155 | 31 | 48 |
| R2-R3 | 558 | 219 | 39.25% | 0.851 | 0.796 | 2.0× | 0.594 | 0.771 | 0.507 | 0.612 | 111 | 33 | 108 |
| R4-R10 | 1764 | 248 | 14.06% | 0.884 | 0.588 | 4.2× | 0.462 | 0.707 | 0.262 | 0.382 | 65 | 27 | 183 |
| R10+ | 8576 | 343 | 4.00% | 0.937 | 0.474 | 11.9× | 0.297 | 0.804 | 0.131 | 0.226 | 45 | 11 | 298 |
| IFA | 14523 | 498 | 3.43% | 0.952 | 0.613 | 17.9× | 0.285 | 0.796 | 0.305 | 0.441 | 152 | 39 | 346 |

#### ESTABLISHED_MLB

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 25814 | 437 | 1.69% | 0.972 | 0.390 | 23.0× | 0.211 | 0.778 | 0.032 | 0.062 | 14 | 4 | 423 |
| R1 | 393 | 103 | 26.21% | 0.818 | 0.537 | 2.0× | 0.485 | 0.571 | 0.039 | 0.073 | 4 | 3 | 99 |
| R2-R3 | 558 | 67 | 12.01% | 0.840 | 0.449 | 3.7× | 0.383 | 1.000 | 0.075 | 0.139 | 5 | 0 | 62 |
| R4-R10 | 1764 | 78 | 4.42% | 0.921 | 0.349 | 7.9× | 0.300 | 1.000 | 0.013 | 0.025 | 1 | 0 | 77 |
| R10+ | 8576 | 69 | 0.80% | 0.968 | 0.283 | 35.2× | 0.145 | 1.000 | 0.014 | 0.029 | 1 | 0 | 68 |
| IFA | 14523 | 120 | 0.83% | 0.976 | 0.320 | 38.7× | 0.149 | 0.750 | 0.025 | 0.048 | 3 | 1 | 117 |

#### STAR_PLUS_ELITE

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 25814 | 78 | 0.30% | 0.977 | 0.134 | 44.5× | 0.091 | — | 0.000 | — | 0 | 0 | 78 |
| R1 | 393 | 24 | 6.11% | 0.837 | 0.185 | 3.0× | 0.279 | — | 0.000 | — | 0 | 0 | 24 |
| R2-R3 | 558 | 11 | 1.97% | 0.904 | 0.110 | 5.6× | 0.195 | — | 0.000 | — | 0 | 0 | 11 |
| R4-R10 | 1764 | 11 | 0.62% | 0.927 | 0.110 | 17.7× | 0.116 | — | 0.000 | — | 0 | 0 | 11 |
| R10+ | 8576 | 11 | 0.13% | 0.967 | 0.221 | 172.0× | 0.058 | — | 0.000 | — | 0 | 0 | 11 |
| IFA | 14523 | 21 | 0.14% | 0.984 | 0.144 | 99.6× | 0.064 | — | 0.000 | — | 0 | 0 | 21 |

## Per-yip (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3495 | 86 | 2.46% | 0.962 | 0.532 | 21.6× | 0.248 | 0.737 | 0.326 | 0.452 | 28 | 10 | 58 |
| 1 | 3256 | 58 | 1.78% | 0.978 | 0.603 | 33.8× | 0.219 | 0.680 | 0.293 | 0.410 | 17 | 8 | 41 |
| 2 | 3007 | 31 | 1.03% | 0.983 | 0.597 | 57.9× | 0.169 | 0.818 | 0.290 | 0.429 | 9 | 2 | 22 |
| 3 | 2738 | 13 | 0.47% | 0.984 | 0.492 | 103.6× | 0.115 | 1.000 | 0.154 | 0.267 | 2 | 0 | 11 |
| 4 | 2482 | 4 | 0.16% | 0.986 | 0.615 | 381.3× | 0.068 | 1.000 | 0.250 | 0.400 | 1 | 0 | 3 |
| 5 | 2255 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 6 | 2057 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 7 | 1857 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 8 | 1669 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 9 | 1498 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 1317 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

#### MLB_DEBUT

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3514 | 367 | 10.44% | 0.901 | 0.559 | 5.4× | 0.425 | 0.730 | 0.221 | 0.339 | 81 | 30 | 286 |
| 1 | 3283 | 349 | 10.63% | 0.923 | 0.653 | 6.1× | 0.452 | 0.761 | 0.347 | 0.476 | 121 | 38 | 228 |
| 2 | 3046 | 310 | 10.18% | 0.945 | 0.736 | 7.2× | 0.466 | 0.814 | 0.452 | 0.581 | 140 | 32 | 170 |
| 3 | 2771 | 210 | 7.58% | 0.958 | 0.757 | 10.0× | 0.420 | 0.811 | 0.471 | 0.596 | 99 | 23 | 111 |
| 4 | 2507 | 130 | 5.19% | 0.957 | 0.728 | 14.0× | 0.351 | 0.836 | 0.431 | 0.569 | 56 | 11 | 74 |
| 5 | 2268 | 67 | 2.95% | 0.945 | 0.572 | 19.4× | 0.261 | 0.808 | 0.313 | 0.452 | 21 | 5 | 46 |
| 6 | 2066 | 38 | 1.84% | 0.954 | 0.498 | 27.1× | 0.211 | 0.778 | 0.184 | 0.298 | 7 | 2 | 31 |
| 7 | 1862 | 21 | 1.13% | 0.973 | 0.547 | 48.5× | 0.173 | 1.000 | 0.048 | 0.091 | 1 | 0 | 20 |
| 8 | 1674 | 12 | 0.72% | 0.953 | 0.479 | 66.9× | 0.132 | 1.000 | 0.167 | 0.286 | 2 | 0 | 10 |
| 9 | 1502 | 6 | 0.40% | 0.971 | 0.352 | 88.1× | 0.103 | — | 0.000 | — | 0 | 0 | 6 |
| 10 | 1321 | 1 | 0.08% | 0.652 | 0.002 | 2.9× | 0.014 | — | 0.000 | — | 0 | 0 | 1 |

#### ESTABLISHED_MLB

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3514 | 111 | 3.16% | 0.925 | 0.298 | 9.4× | 0.257 | 0.500 | 0.018 | 0.035 | 2 | 2 | 109 |
| 1 | 3283 | 111 | 3.38% | 0.943 | 0.428 | 12.7× | 0.277 | 1.000 | 0.063 | 0.119 | 7 | 0 | 104 |
| 2 | 3046 | 102 | 3.35% | 0.965 | 0.469 | 14.0× | 0.290 | 0.800 | 0.039 | 0.075 | 4 | 1 | 98 |
| 3 | 2771 | 58 | 2.09% | 0.975 | 0.474 | 22.7× | 0.236 | 1.000 | 0.017 | 0.034 | 1 | 0 | 57 |
| 4 | 2507 | 32 | 1.28% | 0.980 | 0.461 | 36.1× | 0.187 | 0.000 | 0.000 | — | 0 | 1 | 32 |
| 5 | 2268 | 13 | 0.57% | 0.983 | 0.210 | 36.7× | 0.126 | — | 0.000 | — | 0 | 0 | 13 |
| 6 | 2066 | 6 | 0.29% | 0.984 | 0.117 | 40.1× | 0.090 | — | 0.000 | — | 0 | 0 | 6 |
| 7 | 1862 | 2 | 0.11% | 0.998 | 0.250 | 232.8× | 0.056 | — | 0.000 | — | 0 | 0 | 2 |
| 8 | 1674 | 2 | 0.12% | 0.999 | 0.667 | 558.0× | 0.060 | — | 0.000 | — | 0 | 0 | 2 |
| 9 | 1502 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 1321 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

#### STAR_PLUS_ELITE

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3514 | 23 | 0.65% | 0.936 | 0.090 | 13.7× | 0.122 | — | 0.000 | — | 0 | 0 | 23 |
| 1 | 3283 | 20 | 0.61% | 0.964 | 0.229 | 37.5× | 0.125 | — | 0.000 | — | 0 | 0 | 20 |
| 2 | 3046 | 19 | 0.62% | 0.960 | 0.177 | 28.4× | 0.125 | — | 0.000 | — | 0 | 0 | 19 |
| 3 | 2771 | 9 | 0.32% | 0.983 | 0.242 | 74.5× | 0.095 | — | 0.000 | — | 0 | 0 | 9 |
| 4 | 2507 | 5 | 0.20% | 0.989 | 0.262 | 131.3× | 0.076 | — | 0.000 | — | 0 | 0 | 5 |
| 5 | 2268 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 6 | 2066 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 7 | 1862 | 1 | 0.05% | 0.997 | 0.143 | 266.0× | 0.040 | — | 0.000 | — | 0 | 0 | 1 |
| 8 | 1674 | 1 | 0.06% | 0.999 | 0.500 | 837.0× | 0.042 | — | 0.000 | — | 0 | 0 | 1 |
| 9 | 1502 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 1321 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

## Per-level (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 25631 | 192 | 0.75% | 0.987 | 0.546 | 72.9× | 0.146 | 0.740 | 0.297 | 0.424 | 57 | 20 | 135 |
| RK | 3770 | 65 | 1.72% | 0.969 | 0.618 | 35.9× | 0.211 | 0.885 | 0.354 | 0.505 | 23 | 3 | 42 |
| A- | 987 | 19 | 1.93% | 0.972 | 0.448 | 23.3× | 0.225 | 0.429 | 0.158 | 0.231 | 3 | 4 | 16 |
| A | 1340 | 40 | 2.99% | 0.963 | 0.575 | 19.3× | 0.273 | 0.706 | 0.300 | 0.421 | 12 | 5 | 28 |
| A+ | 1332 | 28 | 2.10% | 0.983 | 0.571 | 27.2× | 0.240 | 0.700 | 0.250 | 0.368 | 7 | 3 | 21 |
| AA | 1151 | 26 | 2.26% | 0.979 | 0.613 | 27.1× | 0.246 | 0.800 | 0.308 | 0.444 | 8 | 2 | 18 |
| AAA | 1399 | 5 | 0.36% | 0.995 | 0.735 | 205.7× | 0.102 | 0.750 | 0.600 | 0.667 | 3 | 1 | 2 |
| NONE | 15652 | 9 | 0.06% | 0.991 | 0.131 | 228.6× | 0.041 | 0.333 | 0.111 | 0.167 | 1 | 2 | 8 |

#### MLB_DEBUT

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 25814 | 1511 | 5.85% | 0.953 | 0.662 | 11.3× | 0.369 | 0.789 | 0.349 | 0.484 | 528 | 141 | 983 |
| RK | 3774 | 208 | 5.51% | 0.921 | 0.538 | 9.8× | 0.333 | 0.750 | 0.173 | 0.281 | 36 | 12 | 172 |
| A- | 989 | 133 | 13.45% | 0.844 | 0.501 | 3.7× | 0.406 | 0.647 | 0.165 | 0.263 | 22 | 12 | 111 |
| A | 1361 | 238 | 17.49% | 0.873 | 0.653 | 3.7× | 0.491 | 0.731 | 0.332 | 0.457 | 79 | 29 | 159 |
| A+ | 1353 | 255 | 18.85% | 0.862 | 0.685 | 3.6× | 0.490 | 0.778 | 0.412 | 0.538 | 105 | 30 | 150 |
| AA | 1225 | 357 | 29.14% | 0.866 | 0.775 | 2.7× | 0.576 | 0.811 | 0.515 | 0.630 | 184 | 43 | 173 |
| AAA | 1432 | 252 | 17.60% | 0.905 | 0.724 | 4.1× | 0.534 | 0.873 | 0.353 | 0.503 | 89 | 13 | 163 |
| NONE | 15680 | 68 | 0.43% | 0.910 | 0.418 | 96.3× | 0.093 | 0.867 | 0.191 | 0.313 | 13 | 2 | 55 |

#### ESTABLISHED_MLB

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 25814 | 437 | 1.69% | 0.972 | 0.390 | 23.0× | 0.211 | 0.778 | 0.032 | 0.062 | 14 | 4 | 423 |
| RK | 3774 | 49 | 1.30% | 0.945 | 0.287 | 22.1× | 0.175 | — | 0.000 | — | 0 | 0 | 49 |
| A- | 989 | 29 | 2.93% | 0.893 | 0.218 | 7.4× | 0.230 | 0.500 | 0.034 | 0.065 | 1 | 1 | 28 |
| A | 1361 | 72 | 5.29% | 0.911 | 0.393 | 7.4× | 0.319 | 1.000 | 0.042 | 0.080 | 3 | 0 | 69 |
| A+ | 1353 | 84 | 6.21% | 0.917 | 0.475 | 7.7× | 0.348 | 1.000 | 0.048 | 0.091 | 4 | 0 | 80 |
| AA | 1225 | 128 | 10.45% | 0.895 | 0.469 | 4.5× | 0.419 | 0.714 | 0.039 | 0.074 | 5 | 2 | 123 |
| AAA | 1432 | 65 | 4.54% | 0.931 | 0.411 | 9.1× | 0.311 | 1.000 | 0.015 | 0.030 | 1 | 0 | 64 |
| NONE | 15680 | 10 | 0.06% | 0.996 | 0.092 | 144.2× | 0.043 | 0.000 | 0.000 | — | 0 | 1 | 10 |

#### STAR_PLUS_ELITE

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 25814 | 78 | 0.30% | 0.977 | 0.134 | 44.5× | 0.091 | — | 0.000 | — | 0 | 0 | 78 |
| RK | 3774 | 14 | 0.37% | 0.945 | 0.114 | 30.7× | 0.094 | — | 0.000 | — | 0 | 0 | 14 |
| A- | 989 | 2 | 0.20% | 0.978 | 0.064 | 31.6× | 0.074 | — | 0.000 | — | 0 | 0 | 2 |
| A | 1361 | 13 | 0.96% | 0.942 | 0.233 | 24.3× | 0.149 | — | 0.000 | — | 0 | 0 | 13 |
| A+ | 1353 | 17 | 1.26% | 0.927 | 0.225 | 17.9× | 0.165 | — | 0.000 | — | 0 | 0 | 17 |
| AA | 1225 | 19 | 1.55% | 0.926 | 0.179 | 11.5× | 0.182 | — | 0.000 | — | 0 | 0 | 19 |
| AAA | 1432 | 12 | 0.84% | 0.957 | 0.127 | 15.1× | 0.144 | — | 0.000 | — | 0 | 0 | 12 |
| NONE | 15680 | 1 | 0.01% | 1.000 | 0.200 | 3136.0× | 0.014 | — | 0.000 | — | 0 | 0 | 1 |

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
