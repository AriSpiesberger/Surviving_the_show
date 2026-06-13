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
| TOP_100_PROSPECT | 26093 | 0.92% | **0.444** | 48.5× | 0.981 | 0.159 | 0.634 | 0.247 | 0.355 |
| MLB_DEBUT | 26273 | 7.39% | **0.648** | 8.8× | 0.946 | 0.404 | 0.682 | 0.437 | 0.533 |
| ESTABLISHED_MLB | 26273 | 1.99% | **0.356** | 17.9× | 0.961 | 0.223 | 0.539 | 0.105 | 0.176 |
| STAR_PLUS_ELITE | 26273 | 0.45% | **0.117** | 26.0× | 0.963 | 0.107 | 0.000 | 0.000 | — |
| **weighted-AP** | | | **0.442** | | | | | | |

(MLB_DEBUT 2× weight, others 1×, per-event eligibility filters. Scores =
`xp_<event>_h6` vs realized-within-6y, on rows resolved at h=6.)

## Per-horizon trajectory (h=1..10, resolved at each h)

#### TOP_100_PROSPECT

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 33417 | 105 | 0.31% | 0.987 | 0.364 | 115.9× | 0.0024 | 0.83 |
| 2 | 32343 | 191 | 0.59% | 0.984 | 0.431 | 72.9× | 0.0043 | 0.81 |
| 3 | 31071 | 242 | 0.78% | 0.982 | 0.471 | 60.5× | 0.0053 | 0.84 |
| 4 | 29609 | 259 | 0.87% | 0.981 | 0.456 | 52.2× | 0.0061 | 0.90 |
| 5 | 27951 | 256 | 0.92% | 0.981 | 0.451 | 49.3× | 0.0063 | 0.96 |
| 6 | 26093 | 239 | 0.92% | 0.981 | 0.444 | 48.5× | 0.0064 | 1.03 |
| 7 | 24040 | 229 | 0.95% | 0.980 | 0.446 | 46.8× | 0.0066 | 1.06 |
| 8 | 21986 | 217 | 0.99% | 0.979 | 0.442 | 44.8× | 0.0069 | 1.07 |
| 9 | 19906 | 203 | 1.02% | 0.977 | 0.426 | 41.8× | 0.0072 | 1.09 |
| 10 | 17753 | 190 | 1.07% | 0.976 | 0.419 | 39.1× | 0.0077 | 1.11 |

#### MLB_DEBUT

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 33646 | 520 | 1.55% | 0.972 | 0.453 | 29.3× | 0.0108 | 0.94 |
| 2 | 32566 | 1018 | 3.13% | 0.961 | 0.568 | 18.2× | 0.0193 | 0.96 |
| 3 | 31288 | 1460 | 4.67% | 0.953 | 0.605 | 13.0× | 0.0274 | 0.98 |
| 4 | 29815 | 1762 | 5.91% | 0.948 | 0.623 | 10.5× | 0.0339 | 1.00 |
| 5 | 28144 | 1916 | 6.81% | 0.947 | 0.639 | 9.4× | 0.0379 | 1.01 |
| 6 | 26273 | 1942 | 7.39% | 0.946 | 0.648 | 8.8× | 0.0404 | 1.03 |
| 7 | 24203 | 1906 | 7.88% | 0.945 | 0.654 | 8.3× | 0.0425 | 1.03 |
| 8 | 22131 | 1823 | 8.24% | 0.944 | 0.655 | 8.0× | 0.0441 | 1.03 |
| 9 | 20034 | 1709 | 8.53% | 0.943 | 0.651 | 7.6× | 0.0458 | 1.04 |
| 10 | 17859 | 1585 | 8.88% | 0.940 | 0.646 | 7.3× | 0.0480 | 1.05 |

#### ESTABLISHED_MLB

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 33646 | 3 | 0.01% | 0.993 | 0.023 | 257.0× | 0.0001 | 1.95 |
| 2 | 32566 | 59 | 0.18% | 0.984 | 0.128 | 70.5× | 0.0017 | 0.85 |
| 3 | 31288 | 169 | 0.54% | 0.977 | 0.233 | 43.2× | 0.0046 | 0.84 |
| 4 | 29815 | 301 | 1.01% | 0.971 | 0.314 | 31.1× | 0.0081 | 0.88 |
| 5 | 28144 | 423 | 1.50% | 0.964 | 0.342 | 22.7× | 0.0116 | 0.90 |
| 6 | 26273 | 522 | 1.99% | 0.961 | 0.356 | 17.9× | 0.0151 | 0.93 |
| 7 | 24203 | 590 | 2.44% | 0.957 | 0.366 | 15.0× | 0.0183 | 0.94 |
| 8 | 22131 | 627 | 2.83% | 0.955 | 0.385 | 13.6× | 0.0209 | 0.95 |
| 9 | 20034 | 631 | 3.15% | 0.952 | 0.384 | 12.2× | 0.0231 | 0.97 |
| 10 | 17859 | 619 | 3.47% | 0.951 | 0.395 | 11.4× | 0.0251 | 0.96 |

#### STAR_PLUS_ELITE

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 33646 | 1 | 0.00% | 0.993 | 0.004 | 142.0× | 0.0000 | 3.50 |
| 2 | 32566 | 8 | 0.02% | 0.992 | 0.070 | 283.0× | 0.0002 | 1.32 |
| 3 | 31288 | 24 | 0.08% | 0.984 | 0.069 | 89.6× | 0.0007 | 1.03 |
| 4 | 29815 | 56 | 0.19% | 0.971 | 0.097 | 51.9× | 0.0018 | 0.84 |
| 5 | 28144 | 90 | 0.32% | 0.964 | 0.099 | 31.0× | 0.0030 | 0.82 |
| 6 | 26273 | 118 | 0.45% | 0.963 | 0.117 | 26.0× | 0.0042 | 0.84 |
| 7 | 24203 | 138 | 0.57% | 0.962 | 0.142 | 24.9× | 0.0052 | 0.86 |
| 8 | 22131 | 152 | 0.69% | 0.959 | 0.160 | 23.4× | 0.0062 | 0.88 |
| 9 | 20034 | 164 | 0.82% | 0.957 | 0.170 | 20.8× | 0.0073 | 0.86 |
| 10 | 17859 | 170 | 0.95% | 0.955 | 0.177 | 18.6× | 0.0085 | 0.82 |

## Per-bucket (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 26093 | 239 | 0.92% | 0.981 | 0.444 | 48.5× | 0.159 | 0.634 | 0.247 | 0.355 | 59 | 34 | 180 |
| R1 | 420 | 78 | 18.57% | 0.894 | 0.587 | 3.2× | 0.531 | 0.636 | 0.359 | 0.459 | 28 | 16 | 50 |
| R2-R3 | 751 | 71 | 9.45% | 0.929 | 0.515 | 5.4× | 0.435 | 0.654 | 0.239 | 0.351 | 17 | 9 | 54 |
| R4-R10 | 2499 | 14 | 0.56% | 0.955 | 0.172 | 30.6× | 0.118 | 0.500 | 0.143 | 0.222 | 2 | 2 | 12 |
| R10+ | 12469 | 27 | 0.22% | 0.952 | 0.232 | 107.0× | 0.073 | 0.500 | 0.148 | 0.229 | 4 | 4 | 23 |
| IFA | 9954 | 49 | 0.49% | 0.981 | 0.375 | 76.2× | 0.117 | 0.727 | 0.163 | 0.267 | 8 | 3 | 41 |

#### MLB_DEBUT

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 26273 | 1942 | 7.39% | 0.946 | 0.648 | 8.8× | 0.404 | 0.682 | 0.437 | 0.533 | 849 | 396 | 1093 |
| R1 | 503 | 269 | 53.48% | 0.909 | 0.914 | 1.7× | 0.707 | 0.838 | 0.810 | 0.824 | 218 | 42 | 51 |
| R2-R3 | 797 | 312 | 39.15% | 0.833 | 0.775 | 2.0× | 0.563 | 0.704 | 0.587 | 0.640 | 183 | 77 | 129 |
| R4-R10 | 2509 | 410 | 16.34% | 0.870 | 0.555 | 3.4× | 0.474 | 0.569 | 0.380 | 0.456 | 156 | 118 | 254 |
| R10+ | 12478 | 619 | 4.96% | 0.927 | 0.455 | 9.2× | 0.321 | 0.606 | 0.249 | 0.353 | 154 | 100 | 465 |
| IFA | 9986 | 332 | 3.32% | 0.954 | 0.635 | 19.1× | 0.282 | 0.701 | 0.416 | 0.522 | 138 | 59 | 194 |

#### ESTABLISHED_MLB

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 26273 | 522 | 1.99% | 0.961 | 0.356 | 17.9× | 0.223 | 0.539 | 0.105 | 0.176 | 55 | 47 | 467 |
| R1 | 503 | 124 | 24.65% | 0.822 | 0.515 | 2.1× | 0.481 | 0.567 | 0.137 | 0.221 | 17 | 13 | 107 |
| R2-R3 | 797 | 91 | 11.42% | 0.852 | 0.430 | 3.8× | 0.388 | 0.571 | 0.176 | 0.269 | 16 | 12 | 75 |
| R4-R10 | 2509 | 116 | 4.62% | 0.906 | 0.318 | 6.9× | 0.295 | 0.562 | 0.078 | 0.136 | 9 | 7 | 107 |
| R10+ | 12478 | 137 | 1.10% | 0.944 | 0.215 | 19.6× | 0.160 | 0.500 | 0.051 | 0.093 | 7 | 7 | 130 |
| IFA | 9986 | 54 | 0.54% | 0.978 | 0.325 | 60.2× | 0.121 | 0.429 | 0.111 | 0.176 | 6 | 8 | 48 |

#### STAR_PLUS_ELITE

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 26273 | 118 | 0.45% | 0.963 | 0.117 | 26.0× | 0.107 | 0.000 | 0.000 | — | 0 | 2 | 118 |
| R1 | 503 | 40 | 7.95% | 0.778 | 0.183 | 2.3× | 0.261 | — | 0.000 | — | 0 | 0 | 40 |
| R2-R3 | 797 | 24 | 3.01% | 0.912 | 0.178 | 5.9× | 0.244 | 0.000 | 0.000 | — | 0 | 2 | 24 |
| R4-R10 | 2509 | 17 | 0.68% | 0.898 | 0.046 | 6.8× | 0.113 | — | 0.000 | — | 0 | 0 | 17 |
| R10+ | 12478 | 23 | 0.18% | 0.935 | 0.176 | 95.7× | 0.065 | — | 0.000 | — | 0 | 0 | 23 |
| IFA | 9986 | 14 | 0.14% | 0.976 | 0.150 | 107.2× | 0.062 | — | 0.000 | — | 0 | 0 | 14 |

## Per-yip (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3594 | 103 | 2.87% | 0.942 | 0.392 | 13.7× | 0.256 | 0.615 | 0.233 | 0.338 | 24 | 15 | 79 |
| 1 | 3363 | 75 | 2.23% | 0.959 | 0.466 | 20.9× | 0.235 | 0.606 | 0.267 | 0.370 | 20 | 13 | 55 |
| 2 | 3098 | 40 | 1.29% | 0.963 | 0.549 | 42.5× | 0.181 | 0.722 | 0.325 | 0.448 | 13 | 5 | 27 |
| 3 | 2805 | 16 | 0.57% | 0.990 | 0.449 | 78.7× | 0.128 | 1.000 | 0.125 | 0.222 | 2 | 0 | 14 |
| 4 | 2529 | 5 | 0.20% | 0.991 | 0.456 | 230.9× | 0.076 | 0.000 | 0.000 | — | 0 | 1 | 5 |
| 5 | 2280 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 6 | 2071 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 7 | 1866 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 8 | 1670 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 9 | 1499 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 1318 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

#### MLB_DEBUT

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3601 | 439 | 12.19% | 0.887 | 0.552 | 4.5× | 0.439 | 0.645 | 0.360 | 0.462 | 158 | 87 | 281 |
| 1 | 3387 | 443 | 13.08% | 0.917 | 0.673 | 5.1× | 0.488 | 0.702 | 0.474 | 0.566 | 210 | 89 | 233 |
| 2 | 3142 | 405 | 12.89% | 0.934 | 0.726 | 5.6× | 0.504 | 0.707 | 0.514 | 0.595 | 208 | 86 | 197 |
| 3 | 2841 | 279 | 9.82% | 0.944 | 0.716 | 7.3× | 0.458 | 0.703 | 0.527 | 0.602 | 147 | 62 | 132 |
| 4 | 2555 | 177 | 6.93% | 0.950 | 0.671 | 9.7× | 0.396 | 0.683 | 0.463 | 0.552 | 82 | 38 | 95 |
| 5 | 2294 | 93 | 4.05% | 0.950 | 0.524 | 12.9× | 0.308 | 0.595 | 0.269 | 0.370 | 25 | 17 | 68 |
| 6 | 2081 | 53 | 2.55% | 0.961 | 0.492 | 19.3× | 0.252 | 0.571 | 0.226 | 0.324 | 12 | 9 | 41 |
| 7 | 1872 | 31 | 1.66% | 0.964 | 0.377 | 22.8× | 0.205 | 0.429 | 0.097 | 0.158 | 3 | 4 | 28 |
| 8 | 1675 | 13 | 0.78% | 0.941 | 0.377 | 48.5× | 0.134 | 0.500 | 0.231 | 0.316 | 3 | 3 | 10 |
| 9 | 1503 | 7 | 0.47% | 0.958 | 0.360 | 77.3× | 0.108 | 0.500 | 0.143 | 0.222 | 1 | 1 | 6 |
| 10 | 1322 | 2 | 0.15% | 0.864 | 0.253 | 167.1× | 0.049 | — | 0.000 | — | 0 | 0 | 2 |

#### ESTABLISHED_MLB

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3601 | 116 | 3.22% | 0.910 | 0.292 | 9.1× | 0.251 | 0.429 | 0.078 | 0.131 | 9 | 12 | 107 |
| 1 | 3387 | 137 | 4.04% | 0.936 | 0.390 | 9.6× | 0.298 | 0.586 | 0.124 | 0.205 | 17 | 12 | 120 |
| 2 | 3142 | 136 | 4.33% | 0.952 | 0.450 | 10.4× | 0.318 | 0.552 | 0.118 | 0.194 | 16 | 13 | 120 |
| 3 | 2841 | 73 | 2.57% | 0.946 | 0.367 | 14.3× | 0.244 | 0.700 | 0.096 | 0.169 | 7 | 3 | 66 |
| 4 | 2555 | 41 | 1.60% | 0.963 | 0.373 | 23.2× | 0.201 | 0.600 | 0.146 | 0.235 | 6 | 4 | 35 |
| 5 | 2294 | 11 | 0.48% | 0.965 | 0.094 | 19.7× | 0.111 | 0.000 | 0.000 | — | 0 | 2 | 11 |
| 6 | 2081 | 5 | 0.24% | 0.966 | 0.074 | 30.7× | 0.079 | 0.000 | 0.000 | — | 0 | 1 | 5 |
| 7 | 1872 | 2 | 0.11% | 0.978 | 0.034 | 31.8× | 0.054 | — | 0.000 | — | 0 | 0 | 2 |
| 8 | 1675 | 1 | 0.06% | 0.996 | 0.125 | 209.4× | 0.042 | — | 0.000 | — | 0 | 0 | 1 |
| 9 | 1503 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 1322 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

#### STAR_PLUS_ELITE

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3601 | 28 | 0.78% | 0.896 | 0.064 | 8.2× | 0.120 | 0.000 | 0.000 | — | 0 | 1 | 28 |
| 1 | 3387 | 33 | 0.97% | 0.943 | 0.128 | 13.1× | 0.151 | 0.000 | 0.000 | — | 0 | 1 | 33 |
| 2 | 3142 | 34 | 1.08% | 0.954 | 0.237 | 21.9× | 0.163 | — | 0.000 | — | 0 | 0 | 34 |
| 3 | 2841 | 14 | 0.49% | 0.952 | 0.210 | 42.6× | 0.110 | — | 0.000 | — | 0 | 0 | 14 |
| 4 | 2555 | 7 | 0.27% | 0.948 | 0.210 | 76.5× | 0.081 | — | 0.000 | — | 0 | 0 | 7 |
| 5 | 2294 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 6 | 2081 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 7 | 1872 | 1 | 0.05% | 0.990 | 0.050 | 93.6× | 0.039 | — | 0.000 | — | 0 | 0 | 1 |
| 8 | 1675 | 1 | 0.06% | 0.997 | 0.167 | 279.2× | 0.042 | — | 0.000 | — | 0 | 0 | 1 |
| 9 | 1503 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 1322 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

## Per-level (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 26093 | 239 | 0.92% | 0.981 | 0.444 | 48.5× | 0.159 | 0.634 | 0.247 | 0.355 | 59 | 34 | 180 |
| RK | 3770 | 65 | 1.72% | 0.963 | 0.424 | 24.6× | 0.209 | 0.737 | 0.215 | 0.333 | 14 | 5 | 51 |
| A- | 987 | 19 | 1.93% | 0.955 | 0.374 | 19.5× | 0.217 | 0.375 | 0.158 | 0.222 | 3 | 5 | 16 |
| A | 1340 | 40 | 2.99% | 0.959 | 0.514 | 17.2× | 0.271 | 0.609 | 0.350 | 0.444 | 14 | 9 | 26 |
| A+ | 1332 | 28 | 2.10% | 0.978 | 0.640 | 30.4× | 0.238 | 0.632 | 0.429 | 0.511 | 12 | 7 | 16 |
| AA | 1150 | 26 | 2.26% | 0.987 | 0.690 | 30.5× | 0.251 | 0.800 | 0.462 | 0.585 | 12 | 3 | 14 |
| AAA | 1391 | 5 | 0.36% | 0.997 | 0.401 | 111.7× | 0.103 | 0.000 | 0.000 | — | 0 | 3 | 5 |
| NONE | 16123 | 56 | 0.35% | 0.984 | 0.307 | 88.5× | 0.099 | 0.667 | 0.071 | 0.129 | 4 | 2 | 52 |

#### MLB_DEBUT

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 26273 | 1942 | 7.39% | 0.946 | 0.648 | 8.8× | 0.404 | 0.682 | 0.437 | 0.533 | 849 | 396 | 1093 |
| RK | 3774 | 208 | 5.51% | 0.920 | 0.484 | 8.8× | 0.332 | 0.629 | 0.212 | 0.317 | 44 | 26 | 164 |
| A- | 989 | 133 | 13.45% | 0.842 | 0.488 | 3.6× | 0.404 | 0.642 | 0.323 | 0.430 | 43 | 24 | 90 |
| A | 1361 | 238 | 17.49% | 0.871 | 0.640 | 3.7× | 0.488 | 0.675 | 0.454 | 0.543 | 108 | 52 | 130 |
| A+ | 1353 | 255 | 18.85% | 0.866 | 0.675 | 3.6× | 0.496 | 0.683 | 0.498 | 0.576 | 127 | 59 | 128 |
| AA | 1224 | 357 | 29.17% | 0.870 | 0.777 | 2.7× | 0.583 | 0.733 | 0.591 | 0.654 | 211 | 77 | 146 |
| AAA | 1424 | 252 | 17.70% | 0.901 | 0.704 | 4.0× | 0.530 | 0.724 | 0.437 | 0.545 | 110 | 42 | 142 |
| NONE | 16148 | 499 | 3.09% | 0.966 | 0.603 | 19.5× | 0.279 | 0.640 | 0.413 | 0.502 | 206 | 116 | 293 |

#### ESTABLISHED_MLB

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 26273 | 522 | 1.99% | 0.961 | 0.356 | 17.9× | 0.223 | 0.539 | 0.105 | 0.176 | 55 | 47 | 467 |
| RK | 3774 | 41 | 1.09% | 0.935 | 0.188 | 17.3× | 0.156 | 0.000 | 0.000 | — | 0 | 1 | 41 |
| A- | 989 | 25 | 2.53% | 0.886 | 0.202 | 8.0× | 0.210 | 0.333 | 0.040 | 0.071 | 1 | 2 | 24 |
| A | 1361 | 64 | 4.70% | 0.916 | 0.385 | 8.2× | 0.305 | 0.545 | 0.094 | 0.160 | 6 | 5 | 58 |
| A+ | 1353 | 71 | 5.25% | 0.921 | 0.432 | 8.2× | 0.325 | 0.769 | 0.141 | 0.238 | 10 | 3 | 61 |
| AA | 1224 | 103 | 8.42% | 0.895 | 0.402 | 4.8× | 0.380 | 0.500 | 0.126 | 0.202 | 13 | 13 | 90 |
| AAA | 1424 | 58 | 4.07% | 0.914 | 0.439 | 10.8× | 0.284 | 0.636 | 0.121 | 0.203 | 7 | 4 | 51 |
| NONE | 16148 | 160 | 0.99% | 0.980 | 0.354 | 35.7× | 0.165 | 0.486 | 0.113 | 0.183 | 18 | 19 | 142 |

#### STAR_PLUS_ELITE

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 26273 | 118 | 0.45% | 0.963 | 0.117 | 26.0× | 0.107 | 0.000 | 0.000 | — | 0 | 2 | 118 |
| RK | 3774 | 14 | 0.37% | 0.931 | 0.058 | 15.7× | 0.091 | — | 0.000 | — | 0 | 0 | 14 |
| A- | 989 | 2 | 0.20% | 0.946 | 0.042 | 20.7× | 0.069 | — | 0.000 | — | 0 | 0 | 2 |
| A | 1361 | 13 | 0.96% | 0.929 | 0.239 | 25.0× | 0.144 | — | 0.000 | — | 0 | 0 | 13 |
| A+ | 1353 | 17 | 1.26% | 0.928 | 0.140 | 11.1× | 0.165 | — | 0.000 | — | 0 | 0 | 17 |
| AA | 1224 | 19 | 1.55% | 0.937 | 0.207 | 13.3× | 0.187 | — | 0.000 | — | 0 | 0 | 19 |
| AAA | 1424 | 12 | 0.84% | 0.949 | 0.223 | 26.4× | 0.142 | — | 0.000 | — | 0 | 0 | 12 |
| NONE | 16148 | 41 | 0.25% | 0.978 | 0.111 | 43.7× | 0.083 | 0.000 | 0.000 | — | 0 | 2 | 41 |

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
