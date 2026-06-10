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
| TOP_100_PROSPECT | 25631 | 0.75% | **0.478** | 63.8× | 0.985 | 0.145 | 0.637 | 0.266 | 0.375 |
| MLB_DEBUT | 25814 | 5.85% | **0.653** | 11.2× | 0.953 | 0.368 | 0.711 | 0.436 | 0.541 |
| ESTABLISHED_MLB | 25814 | 1.43% | **0.403** | 28.2× | 0.972 | 0.194 | 0.656 | 0.160 | 0.258 |
| STAR_PLUS_ELITE | 25814 | 0.30% | **0.135** | 44.7× | 0.970 | 0.089 | — | 0.000 | — |
| **weighted-AP** | | | **0.464** | | | | | | |

(MLB_DEBUT 2× weight, others 1×, per-event eligibility filters. Scores =
`xp_<event>_h6` vs realized-within-6y, on rows resolved at h=6.)

## Per-horizon trajectory (h=1..10, resolved at each h)

#### TOP_100_PROSPECT

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32955 | 89 | 0.27% | 0.991 | 0.406 | 150.4× | 0.0020 | 0.77 |
| 2 | 31881 | 159 | 0.50% | 0.987 | 0.450 | 90.2× | 0.0035 | 0.77 |
| 3 | 30609 | 200 | 0.65% | 0.986 | 0.499 | 76.4× | 0.0044 | 0.81 |
| 4 | 29147 | 213 | 0.73% | 0.985 | 0.488 | 66.7× | 0.0050 | 0.86 |
| 5 | 27489 | 209 | 0.76% | 0.984 | 0.492 | 64.7× | 0.0052 | 0.91 |
| 6 | 25631 | 192 | 0.75% | 0.985 | 0.478 | 63.8× | 0.0051 | 0.98 |
| 7 | 23578 | 182 | 0.77% | 0.984 | 0.481 | 62.3× | 0.0053 | 1.00 |
| 8 | 21522 | 170 | 0.79% | 0.983 | 0.467 | 59.2× | 0.0055 | 1.04 |
| 9 | 19436 | 156 | 0.80% | 0.982 | 0.457 | 56.9× | 0.0057 | 1.07 |
| 10 | 17265 | 143 | 0.83% | 0.982 | 0.450 | 54.3× | 0.0059 | 1.08 |

#### MLB_DEBUT

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 33187 | 433 | 1.30% | 0.981 | 0.505 | 38.7× | 0.0086 | 0.94 |
| 2 | 32107 | 827 | 2.58% | 0.971 | 0.605 | 23.5× | 0.0150 | 0.94 |
| 3 | 30829 | 1173 | 3.80% | 0.962 | 0.627 | 16.5× | 0.0217 | 0.97 |
| 4 | 29356 | 1405 | 4.79% | 0.956 | 0.640 | 13.4× | 0.0267 | 0.99 |
| 5 | 27685 | 1511 | 5.46% | 0.954 | 0.649 | 11.9× | 0.0300 | 1.00 |
| 6 | 25814 | 1511 | 5.85% | 0.953 | 0.653 | 11.2× | 0.0320 | 1.02 |
| 7 | 23744 | 1460 | 6.15% | 0.952 | 0.654 | 10.6× | 0.0334 | 1.02 |
| 8 | 21670 | 1367 | 6.31% | 0.951 | 0.648 | 10.3× | 0.0344 | 1.02 |
| 9 | 19567 | 1252 | 6.40% | 0.949 | 0.638 | 10.0× | 0.0352 | 1.03 |
| 10 | 17374 | 1127 | 6.49% | 0.947 | 0.632 | 9.7× | 0.0361 | 1.03 |

#### ESTABLISHED_MLB

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 33187 | 2 | 0.01% | 0.999 | 0.049 | 820.2× | 0.0001 | 3.69 |
| 2 | 32107 | 41 | 0.13% | 0.988 | 0.206 | 161.5× | 0.0011 | 0.97 |
| 3 | 30829 | 121 | 0.39% | 0.983 | 0.278 | 70.9× | 0.0032 | 0.91 |
| 4 | 29356 | 210 | 0.72% | 0.979 | 0.366 | 51.2× | 0.0055 | 0.89 |
| 5 | 27685 | 296 | 1.07% | 0.975 | 0.391 | 36.6× | 0.0080 | 0.89 |
| 6 | 25814 | 368 | 1.43% | 0.972 | 0.403 | 28.2× | 0.0105 | 0.88 |
| 7 | 23744 | 407 | 1.71% | 0.968 | 0.398 | 23.2× | 0.0127 | 0.90 |
| 8 | 21670 | 422 | 1.95% | 0.966 | 0.401 | 20.6× | 0.0144 | 0.92 |
| 9 | 19567 | 417 | 2.13% | 0.965 | 0.401 | 18.8× | 0.0157 | 0.93 |
| 10 | 17374 | 397 | 2.28% | 0.964 | 0.407 | 17.8× | 0.0167 | 0.90 |

#### STAR_PLUS_ELITE

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 33187 | 1 | 0.00% | 0.999 | 0.022 | 721.5× | 0.0000 | 3.06 |
| 2 | 32107 | 7 | 0.02% | 0.990 | 0.064 | 294.4× | 0.0002 | 1.02 |
| 3 | 30829 | 17 | 0.06% | 0.985 | 0.063 | 115.0× | 0.0005 | 0.97 |
| 4 | 29356 | 38 | 0.13% | 0.980 | 0.110 | 84.6× | 0.0012 | 0.77 |
| 5 | 27685 | 61 | 0.22% | 0.973 | 0.136 | 61.5× | 0.0021 | 0.71 |
| 6 | 25814 | 78 | 0.30% | 0.970 | 0.135 | 44.7× | 0.0028 | 0.72 |
| 7 | 23744 | 89 | 0.37% | 0.968 | 0.155 | 41.4× | 0.0034 | 0.74 |
| 8 | 21670 | 97 | 0.45% | 0.966 | 0.170 | 37.9× | 0.0040 | 0.75 |
| 9 | 19567 | 106 | 0.54% | 0.965 | 0.181 | 33.5× | 0.0049 | 0.71 |
| 10 | 17374 | 109 | 0.63% | 0.965 | 0.194 | 30.9× | 0.0056 | 0.66 |

## Per-bucket (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 25631 | 192 | 0.75% | 0.985 | 0.478 | 63.8× | 0.145 | 0.637 | 0.266 | 0.375 | 51 | 29 | 141 |
| R1 | 320 | 68 | 21.25% | 0.905 | 0.687 | 3.2× | 0.574 | 0.694 | 0.368 | 0.481 | 25 | 11 | 43 |
| R2-R3 | 529 | 50 | 9.45% | 0.904 | 0.475 | 5.0× | 0.409 | 0.520 | 0.260 | 0.347 | 13 | 12 | 37 |
| R4-R10 | 1763 | 8 | 0.45% | 0.953 | 0.120 | 26.4× | 0.105 | 0.000 | 0.000 | — | 0 | 1 | 8 |
| R10+ | 8574 | 9 | 0.10% | 0.934 | 0.253 | 241.3× | 0.049 | 1.000 | 0.222 | 0.364 | 2 | 0 | 7 |
| IFA | 14445 | 57 | 0.39% | 0.984 | 0.385 | 97.5× | 0.105 | 0.688 | 0.193 | 0.301 | 11 | 5 | 46 |

#### MLB_DEBUT

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 25814 | 1511 | 5.85% | 0.953 | 0.653 | 11.2× | 0.368 | 0.711 | 0.436 | 0.541 | 659 | 268 | 852 |
| R1 | 393 | 203 | 51.65% | 0.897 | 0.902 | 1.7× | 0.687 | 0.835 | 0.773 | 0.803 | 157 | 31 | 46 |
| R2-R3 | 558 | 219 | 39.25% | 0.854 | 0.800 | 2.0× | 0.599 | 0.726 | 0.616 | 0.667 | 135 | 51 | 84 |
| R4-R10 | 1764 | 248 | 14.06% | 0.875 | 0.566 | 4.0× | 0.451 | 0.623 | 0.407 | 0.493 | 101 | 61 | 147 |
| R10+ | 8576 | 343 | 4.00% | 0.934 | 0.475 | 11.9× | 0.295 | 0.673 | 0.204 | 0.313 | 70 | 34 | 273 |
| IFA | 14523 | 498 | 3.43% | 0.955 | 0.601 | 17.5× | 0.287 | 0.683 | 0.394 | 0.499 | 196 | 91 | 302 |

#### ESTABLISHED_MLB

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 25814 | 368 | 1.43% | 0.972 | 0.403 | 28.2× | 0.194 | 0.656 | 0.160 | 0.258 | 59 | 31 | 309 |
| R1 | 393 | 83 | 21.12% | 0.870 | 0.633 | 3.0× | 0.524 | 0.694 | 0.301 | 0.420 | 25 | 11 | 58 |
| R2-R3 | 558 | 61 | 10.93% | 0.822 | 0.364 | 3.3× | 0.348 | 0.526 | 0.164 | 0.250 | 10 | 9 | 51 |
| R4-R10 | 1764 | 69 | 3.91% | 0.914 | 0.389 | 9.9× | 0.278 | 0.833 | 0.072 | 0.133 | 5 | 1 | 64 |
| R10+ | 8576 | 54 | 0.63% | 0.969 | 0.210 | 33.4× | 0.129 | 0.500 | 0.037 | 0.069 | 2 | 2 | 52 |
| IFA | 14523 | 101 | 0.70% | 0.978 | 0.396 | 56.9× | 0.138 | 0.680 | 0.168 | 0.270 | 17 | 8 | 84 |

#### STAR_PLUS_ELITE

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 25814 | 78 | 0.30% | 0.970 | 0.135 | 44.7× | 0.089 | — | 0.000 | — | 0 | 0 | 78 |
| R1 | 393 | 24 | 6.11% | 0.809 | 0.190 | 3.1× | 0.256 | — | 0.000 | — | 0 | 0 | 24 |
| R2-R3 | 558 | 11 | 1.97% | 0.916 | 0.218 | 11.1× | 0.200 | — | 0.000 | — | 0 | 0 | 11 |
| R4-R10 | 1764 | 11 | 0.62% | 0.910 | 0.066 | 10.6× | 0.112 | — | 0.000 | — | 0 | 0 | 11 |
| R10+ | 8576 | 11 | 0.13% | 0.952 | 0.176 | 137.3× | 0.056 | — | 0.000 | — | 0 | 0 | 11 |
| IFA | 14523 | 21 | 0.14% | 0.978 | 0.250 | 172.9× | 0.063 | — | 0.000 | — | 0 | 0 | 21 |

## Per-yip (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3495 | 86 | 2.46% | 0.958 | 0.477 | 19.4× | 0.246 | 0.605 | 0.302 | 0.403 | 26 | 17 | 60 |
| 1 | 3256 | 58 | 1.78% | 0.971 | 0.482 | 27.1× | 0.216 | 0.636 | 0.241 | 0.350 | 14 | 8 | 44 |
| 2 | 3007 | 31 | 1.03% | 0.978 | 0.534 | 51.8× | 0.167 | 0.800 | 0.258 | 0.390 | 8 | 2 | 23 |
| 3 | 2738 | 13 | 0.47% | 0.981 | 0.487 | 102.6× | 0.115 | 0.600 | 0.231 | 0.333 | 3 | 2 | 10 |
| 4 | 2482 | 4 | 0.16% | 0.963 | 0.369 | 229.2× | 0.064 | — | 0.000 | — | 0 | 0 | 4 |
| 5 | 2255 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 6 | 2057 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 7 | 1857 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 8 | 1669 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 9 | 1498 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 1317 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

#### MLB_DEBUT

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3514 | 367 | 10.44% | 0.893 | 0.534 | 5.1× | 0.417 | 0.655 | 0.300 | 0.411 | 110 | 58 | 257 |
| 1 | 3283 | 349 | 10.63% | 0.925 | 0.652 | 6.1× | 0.454 | 0.689 | 0.433 | 0.532 | 151 | 68 | 198 |
| 2 | 3046 | 310 | 10.18% | 0.943 | 0.732 | 7.2× | 0.464 | 0.733 | 0.532 | 0.617 | 165 | 60 | 145 |
| 3 | 2771 | 210 | 7.58% | 0.960 | 0.757 | 10.0× | 0.421 | 0.745 | 0.586 | 0.656 | 123 | 42 | 87 |
| 4 | 2507 | 130 | 5.19% | 0.952 | 0.715 | 13.8× | 0.347 | 0.716 | 0.523 | 0.604 | 68 | 27 | 62 |
| 5 | 2268 | 67 | 2.95% | 0.953 | 0.559 | 18.9× | 0.266 | 0.706 | 0.358 | 0.475 | 24 | 10 | 43 |
| 6 | 2066 | 38 | 1.84% | 0.960 | 0.537 | 29.2× | 0.214 | 0.846 | 0.289 | 0.431 | 11 | 2 | 27 |
| 7 | 1862 | 21 | 1.13% | 0.976 | 0.550 | 48.8× | 0.174 | 0.833 | 0.238 | 0.370 | 5 | 1 | 16 |
| 8 | 1674 | 12 | 0.72% | 0.952 | 0.385 | 53.7× | 0.132 | 1.000 | 0.167 | 0.286 | 2 | 0 | 10 |
| 9 | 1502 | 6 | 0.40% | 0.971 | 0.254 | 63.6× | 0.103 | — | 0.000 | — | 0 | 0 | 6 |
| 10 | 1321 | 1 | 0.08% | 0.808 | 0.004 | 5.2× | 0.029 | — | 0.000 | — | 0 | 0 | 1 |

#### ESTABLISHED_MLB

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3514 | 95 | 2.70% | 0.916 | 0.309 | 11.4× | 0.234 | 0.529 | 0.095 | 0.161 | 9 | 8 | 86 |
| 1 | 3283 | 97 | 2.95% | 0.948 | 0.447 | 15.1× | 0.263 | 0.750 | 0.216 | 0.336 | 21 | 7 | 76 |
| 2 | 3046 | 88 | 2.89% | 0.964 | 0.470 | 16.3× | 0.269 | 0.621 | 0.205 | 0.308 | 18 | 11 | 70 |
| 3 | 2771 | 49 | 1.77% | 0.978 | 0.511 | 28.9× | 0.218 | 0.727 | 0.163 | 0.267 | 8 | 3 | 41 |
| 4 | 2507 | 26 | 1.04% | 0.980 | 0.375 | 36.1× | 0.168 | 0.600 | 0.115 | 0.194 | 3 | 2 | 23 |
| 5 | 2268 | 7 | 0.31% | 0.976 | 0.219 | 70.9× | 0.091 | — | 0.000 | — | 0 | 0 | 7 |
| 6 | 2066 | 4 | 0.19% | 0.963 | 0.262 | 135.2× | 0.070 | — | 0.000 | — | 0 | 0 | 4 |
| 7 | 1862 | 1 | 0.05% | 0.998 | 0.200 | 372.4× | 0.040 | — | 0.000 | — | 0 | 0 | 1 |
| 8 | 1674 | 1 | 0.06% | 1.000 | 1.000 | 1674.0× | 0.042 | — | 0.000 | — | 0 | 0 | 1 |
| 9 | 1502 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 1321 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

#### STAR_PLUS_ELITE

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3514 | 23 | 0.65% | 0.904 | 0.080 | 12.2× | 0.113 | — | 0.000 | — | 0 | 0 | 23 |
| 1 | 3283 | 20 | 0.61% | 0.960 | 0.241 | 39.6× | 0.124 | — | 0.000 | — | 0 | 0 | 20 |
| 2 | 3046 | 19 | 0.62% | 0.956 | 0.161 | 25.7× | 0.124 | — | 0.000 | — | 0 | 0 | 19 |
| 3 | 2771 | 9 | 0.32% | 0.979 | 0.356 | 109.6× | 0.094 | — | 0.000 | — | 0 | 0 | 9 |
| 4 | 2507 | 5 | 0.20% | 0.989 | 0.151 | 76.0× | 0.076 | — | 0.000 | — | 0 | 0 | 5 |
| 5 | 2268 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 6 | 2066 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 7 | 1862 | 1 | 0.05% | 0.990 | 0.053 | 98.0× | 0.039 | — | 0.000 | — | 0 | 0 | 1 |
| 8 | 1674 | 1 | 0.06% | 0.996 | 0.143 | 239.1× | 0.042 | — | 0.000 | — | 0 | 0 | 1 |
| 9 | 1502 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 1321 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

## Per-level (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 25631 | 192 | 0.75% | 0.985 | 0.478 | 63.8× | 0.145 | 0.637 | 0.266 | 0.375 | 51 | 29 | 141 |
| RK | 3770 | 65 | 1.72% | 0.960 | 0.527 | 30.6× | 0.207 | 0.692 | 0.277 | 0.396 | 18 | 8 | 47 |
| A- | 987 | 19 | 1.93% | 0.962 | 0.430 | 22.4× | 0.220 | 0.500 | 0.158 | 0.240 | 3 | 3 | 16 |
| A | 1340 | 40 | 2.99% | 0.944 | 0.458 | 15.3× | 0.262 | 0.706 | 0.300 | 0.421 | 12 | 5 | 28 |
| A+ | 1332 | 28 | 2.10% | 0.975 | 0.489 | 23.3× | 0.236 | 0.538 | 0.250 | 0.341 | 7 | 6 | 21 |
| AA | 1151 | 26 | 2.26% | 0.982 | 0.609 | 27.0× | 0.248 | 0.692 | 0.346 | 0.462 | 9 | 4 | 17 |
| AAA | 1399 | 5 | 0.36% | 0.999 | 0.742 | 207.5× | 0.103 | 0.500 | 0.200 | 0.286 | 1 | 1 | 4 |
| NONE | 15652 | 9 | 0.06% | 0.989 | 0.163 | 284.0× | 0.041 | 0.333 | 0.111 | 0.167 | 1 | 2 | 8 |

#### MLB_DEBUT

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 25814 | 1511 | 5.85% | 0.953 | 0.653 | 11.2× | 0.368 | 0.711 | 0.436 | 0.541 | 659 | 268 | 852 |
| RK | 3774 | 208 | 5.51% | 0.920 | 0.512 | 9.3× | 0.332 | 0.641 | 0.240 | 0.350 | 50 | 28 | 158 |
| A- | 989 | 133 | 13.45% | 0.841 | 0.494 | 3.7× | 0.403 | 0.604 | 0.241 | 0.344 | 32 | 21 | 101 |
| A | 1361 | 238 | 17.49% | 0.870 | 0.652 | 3.7× | 0.486 | 0.682 | 0.441 | 0.536 | 105 | 49 | 133 |
| A+ | 1353 | 255 | 18.85% | 0.858 | 0.686 | 3.6× | 0.484 | 0.718 | 0.529 | 0.609 | 135 | 53 | 120 |
| AA | 1225 | 357 | 29.14% | 0.863 | 0.777 | 2.7× | 0.571 | 0.749 | 0.611 | 0.673 | 218 | 73 | 139 |
| AAA | 1432 | 252 | 17.60% | 0.897 | 0.695 | 3.9× | 0.524 | 0.757 | 0.409 | 0.531 | 103 | 33 | 149 |
| NONE | 15680 | 68 | 0.43% | 0.918 | 0.394 | 90.8× | 0.095 | 0.593 | 0.235 | 0.337 | 16 | 11 | 52 |

#### ESTABLISHED_MLB

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 25814 | 368 | 1.43% | 0.972 | 0.403 | 28.2× | 0.194 | 0.656 | 0.160 | 0.258 | 59 | 31 | 309 |
| RK | 3774 | 41 | 1.09% | 0.938 | 0.268 | 24.7× | 0.157 | 0.333 | 0.024 | 0.045 | 1 | 2 | 40 |
| A- | 989 | 25 | 2.53% | 0.889 | 0.205 | 8.1× | 0.211 | 0.500 | 0.040 | 0.074 | 1 | 1 | 24 |
| A | 1361 | 64 | 4.70% | 0.925 | 0.445 | 9.5× | 0.312 | 0.727 | 0.125 | 0.213 | 8 | 3 | 56 |
| A+ | 1353 | 71 | 5.25% | 0.924 | 0.450 | 8.6× | 0.327 | 0.565 | 0.183 | 0.277 | 13 | 10 | 58 |
| AA | 1225 | 103 | 8.41% | 0.902 | 0.488 | 5.8× | 0.387 | 0.722 | 0.252 | 0.374 | 26 | 10 | 77 |
| AAA | 1432 | 58 | 4.05% | 0.915 | 0.399 | 9.9× | 0.283 | 0.667 | 0.172 | 0.274 | 10 | 5 | 48 |
| NONE | 15680 | 6 | 0.04% | 0.998 | 0.263 | 687.4× | 0.034 | — | 0.000 | — | 0 | 0 | 6 |

#### STAR_PLUS_ELITE

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 25814 | 78 | 0.30% | 0.970 | 0.135 | 44.7× | 0.089 | — | 0.000 | — | 0 | 0 | 78 |
| RK | 3774 | 14 | 0.37% | 0.941 | 0.134 | 36.2× | 0.093 | — | 0.000 | — | 0 | 0 | 14 |
| A- | 989 | 2 | 0.20% | 0.972 | 0.060 | 29.8× | 0.073 | — | 0.000 | — | 0 | 0 | 2 |
| A | 1361 | 13 | 0.96% | 0.924 | 0.234 | 24.5× | 0.143 | — | 0.000 | — | 0 | 0 | 13 |
| A+ | 1353 | 17 | 1.26% | 0.931 | 0.166 | 13.2× | 0.166 | — | 0.000 | — | 0 | 0 | 17 |
| AA | 1225 | 19 | 1.55% | 0.890 | 0.154 | 10.0× | 0.167 | — | 0.000 | — | 0 | 0 | 19 |
| AAA | 1432 | 12 | 0.84% | 0.877 | 0.090 | 10.7× | 0.119 | — | 0.000 | — | 0 | 0 | 12 |
| NONE | 15680 | 1 | 0.01% | 0.999 | 0.111 | 1742.2× | 0.014 | — | 0.000 | — | 0 | 0 | 1 |

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
