# Held-out validation — v2.0b landmark (censoring-corrected)

Reproducible evaluation of the v2.0b landmark stack against the **10% val
player slice** of the v1.17 seed=42 split — players neither the landmark
hazards nor the joint XGBoost head trained on. Validation universe: drafted
players with `draft_year ≤ 2020` (plus IFAs).

**Yardstick: RESOLVED outcomes only.** Raw labels are right-censored (an event
is recorded only if it occurred by the data cutoff), so a model that predicts
*eventual* outcome is unfairly penalized for events that will happen after the
cutoff. The joint XGB therefore trains on — and is evaluated on — **resolved
rows**: the event was observed, OR the player had ≥6 forward years without it.
This fixed a severe debut undercount (AAA arms read ~2% debut before; realistic
now). Knobs: `--censor-window 6` (`fit_joint_xgb_v2.py`), `--resolved-window 6`
(regen scripts). The **hazards** are survival models — censoring-aware by
construction — and need no filter.

**Data integrity (this revision):** birthdates backfilled for 2024–25 draft
classes, FG/TWTC crosswalk 89%→96%, trade-aware `current_org` (latest-season
affiliate → parent org via MLB Stats API), IFA entry-year anchors, signing-bonus
backfill. Point-in-time scouting (FanGraphs Board 2017–26 + Trouble-With-The-
Curve 2013–19): 76 grade/physical/velo/rank/ETA columns in the hazard panel
(no-lookahead, season ≤ snapshot) + a 5-col current-snapshot summary
(`scout_fv, scout_ovr_rank, scout_eta_gap, scout_risk, scout_is_scouted`) fed to
the XGB. HOF_TRAJECTORY dropped from the event set.

## Stack

| Layer | Model | Trained on |
|---|---|---|
| Hazards (per-fold OOF, eval) | `scratch/v20b_oof/fold[0-5]_hazards.pkl` | Each fold trained on the OTHER 5 (val pids excluded). HistGBT, default HP, 314 features (incl. 76 scouting). Survival → censoring-aware. |
| Hazards (production) | `models/event_classifiers_v2.0b_prod.pkl` | 100% of ≤2020 data. Scores the 2026 cohort (entry 2024–26 — not in training, so no leakage). |
| XGB head | `models/joint_xgb_v2.0b_{oof,prod}.pkl` | OOF stacked CSV, **censoring-corrected (`--censor-window 6`)**. Default HP (`max_depth=6, lr=0.05`). FEAT = hazard probs + age/yip + 5-col scouting summary (19). Honest at XGB layer. |

Scouting features are point-in-time (latest grade with `season ≤ snapshot`),
so features never see the future. ~4% of rows are scouted (only ranked
prospects get grades); the rest are NaN/sentinel, which HistGBT handles.

## Headline (ALL bucket, threshold = 0.60)

| Event | n | base% | AP | lift | AUC | spearman | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TOP_100_PROSPECT | 25827 | 0.89% | **0.494** | 55.8× | 0.981 | 0.156 | 0.750 | 0.066 | 0.120 |
| MLB_DEBUT | 26042 | 7.07% | **0.684** | 9.7× | 0.943 | 0.394 | 0.826 | 0.381 | 0.522 |
| ESTABLISHED_MLB | 26042 | 1.74% | **0.427** | 24.6× | 0.969 | 0.212 | 0.765 | 0.086 | 0.155 |
| STAR_PLUS_ELITE | 26042 | 0.49% | **0.227** | 46.2× | 0.971 | 0.114 | — | 0.000 | — |
| **weighted-AP** | | | **0.503** | | | | | | |

(MLB_DEBUT 2× weight, others 1×, per-event eligibility filters.)

## Per-bucket (threshold = 0.60)

#### TOP_100_PROSPECT

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 25827 | 229 | 0.89% | 0.981 | 0.494 | 55.8× | 0.156 | 0.750 | 0.066 | 0.120 | 15 | 5 | 214 |
| R1 | 338 | 80 | 23.67% | 0.902 | 0.713 | 3.0× | 0.592 | 0.778 | 0.087 | 0.157 | 7 | 2 | 73 |
| R2-R3 | 560 | 61 | 10.89% | 0.892 | 0.513 | 4.7× | 0.423 | 0.800 | 0.066 | 0.121 | 4 | 1 | 57 |
| R4-R10 | 1809 | 20 | 1.11% | 0.940 | 0.255 | 23.1× | 0.159 | — | 0.000 | — | 0 | 0 | 20 |
| R10+ | 8629 | 9 | 0.10% | 0.916 | 0.288 | 275.9× | 0.046 | 1.000 | 0.111 | 0.200 | 1 | 0 | 8 |
| IFA | 14491 | 59 | 0.41% | 0.980 | 0.351 | 86.1× | 0.106 | 0.600 | 0.051 | 0.094 | 3 | 2 | 56 |

#### MLB_DEBUT

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 26042 | 1842 | 7.07% | 0.943 | 0.684 | 9.7× | 0.394 | 0.826 | 0.381 | 0.522 | 702 | 148 | 1140 |
| R1 | 428 | 251 | 58.64% | 0.924 | 0.946 | 1.6× | 0.723 | 0.916 | 0.737 | 0.817 | 185 | 17 | 66 |
| R2-R3 | 593 | 264 | 44.52% | 0.870 | 0.860 | 1.9× | 0.638 | 0.839 | 0.591 | 0.693 | 156 | 30 | 108 |
| R4-R10 | 1810 | 306 | 16.91% | 0.878 | 0.624 | 3.7× | 0.490 | 0.741 | 0.327 | 0.454 | 100 | 35 | 206 |
| R10+ | 8631 | 431 | 4.99% | 0.909 | 0.488 | 9.8× | 0.309 | 0.827 | 0.144 | 0.245 | 62 | 13 | 369 |
| IFA | 14580 | 590 | 4.05% | 0.941 | 0.599 | 14.8× | 0.301 | 0.790 | 0.337 | 0.473 | 199 | 53 | 391 |

#### ESTABLISHED_MLB

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 26042 | 452 | 1.74% | 0.969 | 0.427 | 24.6× | 0.212 | 0.765 | 0.086 | 0.155 | 39 | 12 | 413 |
| R1 | 428 | 92 | 21.50% | 0.894 | 0.663 | 3.1× | 0.560 | 0.773 | 0.185 | 0.298 | 17 | 5 | 75 |
| R2-R3 | 593 | 68 | 11.47% | 0.815 | 0.388 | 3.4× | 0.347 | 0.600 | 0.088 | 0.154 | 6 | 4 | 62 |
| R4-R10 | 1810 | 91 | 5.03% | 0.915 | 0.427 | 8.5× | 0.314 | 1.000 | 0.022 | 0.043 | 2 | 0 | 89 |
| R10+ | 8631 | 74 | 0.86% | 0.959 | 0.221 | 25.8× | 0.147 | 1.000 | 0.014 | 0.027 | 1 | 0 | 73 |
| IFA | 14580 | 127 | 0.87% | 0.978 | 0.402 | 46.2× | 0.154 | 0.812 | 0.102 | 0.182 | 13 | 3 | 114 |

#### STAR_PLUS_ELITE

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 26042 | 128 | 0.49% | 0.971 | 0.227 | 46.2× | 0.114 | — | 0.000 | — | 0 | 0 | 128 |
| R1 | 428 | 35 | 8.18% | 0.857 | 0.308 | 3.8× | 0.339 | — | 0.000 | — | 0 | 0 | 35 |
| R2-R3 | 593 | 16 | 2.70% | 0.932 | 0.307 | 11.4× | 0.242 | — | 0.000 | — | 0 | 0 | 16 |
| R4-R10 | 1810 | 21 | 1.16% | 0.943 | 0.298 | 25.7× | 0.164 | — | 0.000 | — | 0 | 0 | 21 |
| R10+ | 8631 | 18 | 0.21% | 0.915 | 0.179 | 85.8× | 0.066 | — | 0.000 | — | 0 | 0 | 18 |
| IFA | 14580 | 38 | 0.26% | 0.987 | 0.221 | 84.9× | 0.086 | — | 0.000 | — | 0 | 0 | 38 |

## Per-yip (threshold = 0.60)

#### TOP_100_PROSPECT

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3509 | 97 | 2.76% | 0.956 | 0.447 | 16.2× | 0.259 | 0.667 | 0.021 | 0.040 | 2 | 1 | 95 |
| 1 | 3278 | 66 | 2.01% | 0.959 | 0.522 | 25.9× | 0.223 | 0.636 | 0.106 | 0.182 | 7 | 4 | 59 |
| 2 | 3044 | 41 | 1.35% | 0.974 | 0.616 | 45.7× | 0.190 | 1.000 | 0.146 | 0.255 | 6 | 0 | 35 |
| 3 | 2782 | 17 | 0.61% | 0.979 | 0.464 | 76.0× | 0.129 | — | 0.000 | — | 0 | 0 | 17 |
| 4 | 2521 | 7 | 0.28% | 0.983 | 0.425 | 153.2× | 0.088 | — | 0.000 | — | 0 | 0 | 7 |
| 5 | 2275 | 1 | 0.04% | 1.000 | 0.500 | 1137.5× | 0.036 | — | 0.000 | — | 0 | 0 | 1 |
| 6 | 2070 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 7 | 1861 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 8 | 1670 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 9 | 1499 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 1318 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

#### MLB_DEBUT

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3528 | 434 | 12.30% | 0.886 | 0.591 | 4.8× | 0.439 | 0.784 | 0.242 | 0.370 | 105 | 29 | 329 |
| 1 | 3311 | 403 | 12.17% | 0.914 | 0.679 | 5.6× | 0.469 | 0.800 | 0.377 | 0.513 | 152 | 38 | 251 |
| 2 | 3089 | 366 | 11.85% | 0.932 | 0.746 | 6.3× | 0.484 | 0.822 | 0.467 | 0.596 | 171 | 37 | 195 |
| 3 | 2823 | 268 | 9.49% | 0.944 | 0.774 | 8.2× | 0.451 | 0.861 | 0.507 | 0.638 | 136 | 22 | 132 |
| 4 | 2549 | 174 | 6.83% | 0.936 | 0.728 | 10.7× | 0.381 | 0.842 | 0.489 | 0.618 | 85 | 16 | 89 |
| 5 | 2292 | 92 | 4.01% | 0.936 | 0.624 | 15.6× | 0.296 | 0.861 | 0.337 | 0.484 | 31 | 5 | 61 |
| 6 | 2082 | 55 | 2.64% | 0.949 | 0.639 | 24.2× | 0.249 | 1.000 | 0.309 | 0.472 | 17 | 0 | 38 |
| 7 | 1867 | 27 | 1.45% | 0.970 | 0.515 | 35.6× | 0.194 | 1.000 | 0.111 | 0.200 | 3 | 0 | 24 |
| 8 | 1676 | 14 | 0.84% | 0.964 | 0.413 | 49.5× | 0.147 | 0.667 | 0.143 | 0.235 | 2 | 1 | 12 |
| 9 | 1503 | 7 | 0.47% | 0.952 | 0.300 | 64.4× | 0.107 | — | 0.000 | — | 0 | 0 | 7 |
| 10 | 1322 | 2 | 0.15% | 0.893 | 0.014 | 9.2× | 0.053 | — | 0.000 | — | 0 | 0 | 2 |

#### ESTABLISHED_MLB

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3528 | 138 | 3.91% | 0.912 | 0.368 | 9.4× | 0.276 | 0.700 | 0.051 | 0.095 | 7 | 3 | 131 |
| 1 | 3311 | 119 | 3.59% | 0.941 | 0.473 | 13.2× | 0.285 | 0.783 | 0.151 | 0.254 | 18 | 5 | 101 |
| 2 | 3089 | 98 | 3.17% | 0.963 | 0.503 | 15.9× | 0.281 | 0.769 | 0.102 | 0.180 | 10 | 3 | 88 |
| 3 | 2823 | 54 | 1.91% | 0.971 | 0.486 | 25.4× | 0.223 | 0.800 | 0.074 | 0.136 | 4 | 1 | 50 |
| 4 | 2549 | 28 | 1.10% | 0.979 | 0.384 | 35.0× | 0.173 | — | 0.000 | — | 0 | 0 | 28 |
| 5 | 2292 | 9 | 0.39% | 0.977 | 0.154 | 39.3× | 0.103 | — | 0.000 | — | 0 | 0 | 9 |
| 6 | 2082 | 4 | 0.19% | 0.980 | 0.218 | 113.4× | 0.073 | — | 0.000 | — | 0 | 0 | 4 |
| 7 | 1867 | 1 | 0.05% | 0.994 | 0.077 | 143.6× | 0.040 | — | 0.000 | — | 0 | 0 | 1 |
| 8 | 1676 | 1 | 0.06% | 1.000 | 1.000 | 1676.0× | 0.042 | — | 0.000 | — | 0 | 0 | 1 |
| 9 | 1503 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 1322 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

#### STAR_PLUS_ELITE

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3528 | 42 | 1.19% | 0.923 | 0.163 | 13.7× | 0.159 | — | 0.000 | — | 0 | 0 | 42 |
| 1 | 3311 | 33 | 1.00% | 0.963 | 0.297 | 29.8× | 0.159 | — | 0.000 | — | 0 | 0 | 33 |
| 2 | 3089 | 28 | 0.91% | 0.969 | 0.409 | 45.1× | 0.154 | — | 0.000 | — | 0 | 0 | 28 |
| 3 | 2823 | 14 | 0.50% | 0.979 | 0.275 | 55.5× | 0.117 | — | 0.000 | — | 0 | 0 | 14 |
| 4 | 2549 | 7 | 0.27% | 0.986 | 0.198 | 72.3× | 0.088 | — | 0.000 | — | 0 | 0 | 7 |
| 5 | 2292 | 1 | 0.04% | 0.817 | 0.002 | 5.5× | 0.023 | — | 0.000 | — | 0 | 0 | 1 |
| 6 | 2082 | 1 | 0.05% | 0.939 | 0.008 | 16.3× | 0.033 | — | 0.000 | — | 0 | 0 | 1 |
| 7 | 1867 | 1 | 0.05% | 0.983 | 0.030 | 56.6× | 0.039 | — | 0.000 | — | 0 | 0 | 1 |
| 8 | 1676 | 1 | 0.06% | 0.996 | 0.125 | 209.5× | 0.042 | — | 0.000 | — | 0 | 0 | 1 |
| 9 | 1503 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 1322 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

## Per-level (threshold = 0.60)

#### TOP_100_PROSPECT

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 25827 | 229 | 0.89% | 0.981 | 0.494 | 55.8× | 0.156 | 0.750 | 0.066 | 0.120 | 15 | 5 | 214 |
| RK | 3771 | 65 | 1.72% | 0.963 | 0.465 | 27.0× | 0.209 | 0.500 | 0.015 | 0.030 | 1 | 1 | 64 |
| A- | 987 | 19 | 1.93% | 0.970 | 0.432 | 22.5× | 0.224 | 1.000 | 0.105 | 0.190 | 2 | 0 | 17 |
| A | 1353 | 45 | 3.33% | 0.944 | 0.446 | 13.4× | 0.276 | 0.400 | 0.044 | 0.080 | 2 | 3 | 43 |
| A+ | 1348 | 29 | 2.15% | 0.975 | 0.586 | 27.2× | 0.239 | 1.000 | 0.172 | 0.294 | 5 | 0 | 24 |
| AA | 1179 | 33 | 2.80% | 0.977 | 0.746 | 26.6× | 0.273 | 0.833 | 0.152 | 0.256 | 5 | 1 | 28 |
| AAA | 1446 | 11 | 0.76% | 0.992 | 0.613 | 80.6× | 0.148 | — | 0.000 | — | 0 | 0 | 11 |
| NONE | 15743 | 27 | 0.17% | 0.981 | 0.248 | 144.6× | 0.069 | — | 0.000 | — | 0 | 0 | 27 |

#### MLB_DEBUT

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 26042 | 1842 | 7.07% | 0.943 | 0.684 | 9.7× | 0.394 | 0.826 | 0.381 | 0.522 | 702 | 148 | 1140 |
| RK | 3775 | 258 | 6.83% | 0.906 | 0.551 | 8.1× | 0.355 | 0.787 | 0.244 | 0.373 | 63 | 17 | 195 |
| A- | 989 | 139 | 14.05% | 0.847 | 0.542 | 3.9× | 0.418 | 0.725 | 0.209 | 0.324 | 29 | 11 | 110 |
| A | 1376 | 271 | 19.69% | 0.863 | 0.678 | 3.4× | 0.501 | 0.778 | 0.387 | 0.517 | 105 | 30 | 166 |
| A+ | 1373 | 280 | 20.39% | 0.847 | 0.701 | 3.4× | 0.484 | 0.830 | 0.436 | 0.571 | 122 | 25 | 158 |
| AA | 1260 | 400 | 31.75% | 0.864 | 0.801 | 2.5× | 0.587 | 0.830 | 0.537 | 0.653 | 215 | 44 | 185 |
| AAA | 1486 | 307 | 20.66% | 0.903 | 0.749 | 3.6× | 0.565 | 0.851 | 0.391 | 0.536 | 120 | 21 | 187 |
| NONE | 15783 | 187 | 1.18% | 0.938 | 0.572 | 48.3× | 0.164 | 1.000 | 0.257 | 0.409 | 48 | 0 | 139 |

#### ESTABLISHED_MLB

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 26042 | 452 | 1.74% | 0.969 | 0.427 | 24.6× | 0.212 | 0.765 | 0.086 | 0.155 | 39 | 12 | 413 |
| RK | 3775 | 74 | 1.96% | 0.939 | 0.325 | 16.6× | 0.211 | — | 0.000 | — | 0 | 0 | 74 |
| A- | 989 | 32 | 3.24% | 0.882 | 0.222 | 6.9× | 0.234 | 0.000 | 0.000 | — | 0 | 1 | 32 |
| A | 1376 | 80 | 5.81% | 0.915 | 0.474 | 8.1× | 0.336 | 0.800 | 0.100 | 0.178 | 8 | 2 | 72 |
| A+ | 1373 | 79 | 5.75% | 0.910 | 0.432 | 7.5× | 0.331 | 0.692 | 0.114 | 0.196 | 9 | 4 | 70 |
| AA | 1260 | 113 | 8.97% | 0.906 | 0.558 | 6.2× | 0.402 | 0.857 | 0.159 | 0.269 | 18 | 3 | 95 |
| AAA | 1486 | 62 | 4.17% | 0.927 | 0.412 | 9.9× | 0.296 | 0.667 | 0.065 | 0.118 | 4 | 2 | 58 |
| NONE | 15783 | 12 | 0.08% | 0.998 | 0.290 | 381.0× | 0.048 | — | 0.000 | — | 0 | 0 | 12 |

#### STAR_PLUS_ELITE

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 26042 | 128 | 0.49% | 0.971 | 0.227 | 46.2× | 0.114 | — | 0.000 | — | 0 | 0 | 128 |
| RK | 3775 | 25 | 0.66% | 0.962 | 0.245 | 36.9× | 0.130 | — | 0.000 | — | 0 | 0 | 25 |
| A- | 989 | 5 | 0.51% | 0.904 | 0.157 | 31.0× | 0.099 | — | 0.000 | — | 0 | 0 | 5 |
| A | 1376 | 23 | 1.67% | 0.950 | 0.331 | 19.8× | 0.200 | — | 0.000 | — | 0 | 0 | 23 |
| A+ | 1373 | 22 | 1.60% | 0.940 | 0.238 | 14.9× | 0.191 | — | 0.000 | — | 0 | 0 | 22 |
| AA | 1260 | 33 | 2.62% | 0.929 | 0.346 | 13.2× | 0.237 | — | 0.000 | — | 0 | 0 | 33 |
| AAA | 1486 | 14 | 0.94% | 0.879 | 0.204 | 21.7× | 0.127 | — | 0.000 | — | 0 | 0 | 14 |
| NONE | 15783 | 6 | 0.04% | 0.974 | 0.024 | 62.2× | 0.032 | — | 0.000 | — | 0 | 0 | 6 |

## Threshold @ precision ≥ 0.60 (MLB_DEBUT per yip)

| yip | threshold | n_above | TP | precision | recall | n_total | n_pos_total |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.382 | 340 | 204 | 0.600 | 0.470 | 3528 | 434 |
| 1 | 0.353 | 398 | 239 | 0.601 | 0.593 | 3311 | 403 |
| 2 | 0.279 | 450 | 270 | 0.600 | 0.738 | 3089 | 366 |
| 3 | 0.279 | 343 | 206 | 0.601 | 0.769 | 2823 | 268 |
| 4 | 0.287 | 200 | 120 | 0.600 | 0.690 | 2549 | 174 |
| 5 | 0.365 | 78 | 47 | 0.603 | 0.511 | 2292 | 92 |
| 6 | 0.296 | 53 | 32 | 0.604 | 0.582 | 2082 | 55 |
| 7 | 0.401 | 13 | 8 | 0.615 | 0.296 | 1867 | 27 |
| 8 | 0.359 | 5 | 3 | 0.600 | 0.214 | 1676 | 14 |
| 9 | — | 0 | 0 | — | — | 1503 | 7 |
| 10 | — | 0 | 0 | — | — | 1322 | 2 |

## Statistics glossary

| Metric | Meaning |
|---|---|
| `ap` | Average Precision = AU-PR. Headline rare-event metric. |
| `ap_lift` | `ap / base_rate` — how many × random the ranking is. |
| `auc` | Area under ROC. Insensitive to class imbalance. |
| `spearman_rho` | Rank correlation between score and realized 0/1. |
| `precision/recall/f1` | At threshold 0.60. `—` = undefined (no predicted positives / no positives). |
| `bucket` | Draft pedigree: R1, R2-R3, R4-R10, R10+ (rounds 11+), IFA. |
| `snap_offset` (yip) | Years since entry. |
| `cur_level` | Player's level at snapshot: RK/A-/A/A+/AA/AAA/NONE. |

## Reproducing

```bash
# data integrity backfills (MLB Stats API) + scouting grades
python -m prospects.ingestion.backfills.birthdate_backfill      # 2024-25 DOB
python -m prospects.ingestion.backfills.org_backfill            # trade-aware current_org
python -m scripts.scrape_fangraphs_board --start 2017 --end 2026   # needs curl_cffi
python -m scripts.build_fg_crosswalk      # 96% match (needs DOB backfill first)
python -m scripts.build_scouting_grades

# model: OOF folds + hazards + censoring-corrected joint XGB (--censor-window 6
# is wired into run_v2_0b_oof stage 6 and train_v2_0b_prod stage 1)
python -m scripts_v17.train.run_v2_0b_oof
python -m scripts_v17.train.train_v2_0b_prod    # 100% prod hazards + W=6 XGB + score 2026

# validation on the RESOLVED yardstick (positives + >=6 fwd-yr negatives)
python -c "import pandas as pd; v=pd.read_csv('results/training/v2.0b_oof_val_long.csv'); \
  r=[c for c in v if c.startswith('realized_')]; \
  v[(v.years_fwd>=6)|(v[r].sum(1)>0)].to_csv('results/training/v2.0b_oof_val_long_resolved.csv',index=False)"
python -m scripts_v17.validate.regen_eval_v2_0b_honest \
    --val-long results/training/v2.0b_oof_val_long_resolved.csv --threshold 0.60
python -m scripts_v17.validate.regen_full_eval_v2_0b \
    --val-long results/training/v2.0b_oof_val_long_resolved.csv
python -m scripts_v17.validate.gen_eval_readme
```
