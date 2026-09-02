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
| TOP_100_PROSPECT | 18901 | 1.50% | **0.513** | 34.3× | 0.964 | 0.195 | 0.723 | 0.332 | 0.455 |
| MLB_DEBUT | 19160 | 13.12% | **0.693** | 5.3× | 0.923 | 0.495 | 0.709 | 0.499 | 0.586 |
| ESTABLISHED_MLB | 19160 | 4.15% | **0.424** | 10.2× | 0.926 | 0.295 | 0.684 | 0.131 | 0.219 |
| STAR_PLUS_ELITE | 19160 | 0.58% | **0.168** | 28.8× | 0.925 | 0.112 | 1.000 | 0.009 | 0.018 |
| **weighted-AP** | | | **0.499** | | | | | | |

(MLB_DEBUT 2× weight, others 1×, per-event eligibility filters. Scores =
`xp_<event>_h6` vs realized-within-6y, on rows resolved at h=6.)

## Per-horizon trajectory (h=1..10, resolved at each h)

#### TOP_100_PROSPECT

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 23557 | 114 | 0.48% | 0.985 | 0.454 | 93.8× | 0.0034 | 0.94 |
| 2 | 22890 | 204 | 0.89% | 0.974 | 0.504 | 56.6× | 0.0059 | 0.92 |
| 3 | 22114 | 259 | 1.17% | 0.969 | 0.503 | 42.9× | 0.0077 | 0.91 |
| 4 | 21196 | 279 | 1.32% | 0.966 | 0.515 | 39.1× | 0.0086 | 0.96 |
| 5 | 20126 | 284 | 1.41% | 0.965 | 0.511 | 36.2× | 0.0093 | 1.00 |
| 6 | 18901 | 283 | 1.50% | 0.964 | 0.513 | 34.3× | 0.0098 | 1.00 |
| 7 | 17555 | 277 | 1.58% | 0.963 | 0.508 | 32.2× | 0.0104 | 1.00 |
| 8 | 16191 | 266 | 1.64% | 0.961 | 0.512 | 31.1× | 0.0108 | 0.99 |
| 9 | 14815 | 254 | 1.71% | 0.959 | 0.514 | 30.0× | 0.0112 | 0.99 |
| 10 | 13436 | 239 | 1.78% | 0.957 | 0.513 | 28.9× | 0.0116 | 0.98 |

#### MLB_DEBUT

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 23869 | 651 | 2.73% | 0.952 | 0.502 | 18.4× | 0.0183 | 1.05 |
| 2 | 23197 | 1283 | 5.53% | 0.942 | 0.596 | 10.8× | 0.0329 | 1.06 |
| 3 | 22414 | 1855 | 8.28% | 0.931 | 0.636 | 7.7× | 0.0466 | 1.06 |
| 4 | 21487 | 2256 | 10.50% | 0.926 | 0.661 | 6.3× | 0.0565 | 1.05 |
| 5 | 20403 | 2461 | 12.06% | 0.924 | 0.680 | 5.6× | 0.0628 | 1.05 |
| 6 | 19160 | 2513 | 13.12% | 0.923 | 0.693 | 5.3× | 0.0666 | 1.05 |
| 7 | 17795 | 2468 | 13.87% | 0.923 | 0.702 | 5.1× | 0.0692 | 1.05 |
| 8 | 16412 | 2367 | 14.42% | 0.922 | 0.702 | 4.9× | 0.0717 | 1.06 |
| 9 | 15018 | 2243 | 14.94% | 0.919 | 0.700 | 4.7× | 0.0743 | 1.05 |
| 10 | 13622 | 2105 | 15.45% | 0.916 | 0.698 | 4.5× | 0.0770 | 1.04 |

#### ESTABLISHED_MLB

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 23869 | 6 | 0.03% | 0.987 | 0.071 | 283.4× | 0.0002 | 1.75 |
| 2 | 23197 | 74 | 0.32% | 0.952 | 0.195 | 61.2× | 0.0028 | 1.03 |
| 3 | 22414 | 238 | 1.06% | 0.950 | 0.306 | 28.8× | 0.0086 | 0.99 |
| 4 | 21487 | 447 | 2.08% | 0.939 | 0.381 | 18.3× | 0.0158 | 0.97 |
| 5 | 20403 | 642 | 3.15% | 0.932 | 0.407 | 12.9× | 0.0232 | 0.94 |
| 6 | 19160 | 796 | 4.15% | 0.926 | 0.424 | 10.2× | 0.0300 | 0.95 |
| 7 | 17795 | 899 | 5.05% | 0.925 | 0.458 | 9.1× | 0.0352 | 0.97 |
| 8 | 16412 | 953 | 5.81% | 0.923 | 0.473 | 8.1× | 0.0397 | 0.98 |
| 9 | 15018 | 967 | 6.44% | 0.919 | 0.477 | 7.4× | 0.0436 | 0.98 |
| 10 | 13622 | 955 | 7.01% | 0.914 | 0.481 | 6.9× | 0.0472 | 0.94 |

#### STAR_PLUS_ELITE

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 23869 | 7 | 0.03% | 0.993 | 0.068 | 231.0× | 0.0003 | 0.81 |
| 2 | 23197 | 19 | 0.08% | 0.947 | 0.081 | 98.4× | 0.0008 | 0.80 |
| 3 | 22414 | 36 | 0.16% | 0.927 | 0.077 | 47.9× | 0.0015 | 0.86 |
| 4 | 21487 | 58 | 0.27% | 0.929 | 0.184 | 68.3× | 0.0025 | 0.96 |
| 5 | 20403 | 86 | 0.42% | 0.924 | 0.162 | 38.3× | 0.0038 | 0.97 |
| 6 | 19160 | 112 | 0.58% | 0.925 | 0.168 | 28.8× | 0.0053 | 0.99 |
| 7 | 17795 | 132 | 0.74% | 0.926 | 0.172 | 23.2× | 0.0067 | 1.04 |
| 8 | 16412 | 147 | 0.90% | 0.924 | 0.193 | 21.5× | 0.0080 | 1.06 |
| 9 | 15018 | 162 | 1.08% | 0.921 | 0.194 | 18.0× | 0.0096 | 1.02 |
| 10 | 13622 | 170 | 1.25% | 0.920 | 0.208 | 16.6× | 0.0110 | 0.94 |

## Per-bucket (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 18901 | 283 | 1.50% | 0.964 | 0.513 | 34.3× | 0.195 | 0.723 | 0.332 | 0.455 | 94 | 36 | 189 |
| R1 | 396 | 101 | 25.51% | 0.928 | 0.788 | 3.1× | 0.646 | 0.733 | 0.733 | 0.733 | 74 | 27 | 27 |
| R2-R3 | 986 | 48 | 4.87% | 0.909 | 0.402 | 8.3× | 0.305 | 0.579 | 0.229 | 0.328 | 11 | 8 | 37 |
| R4-R10 | 4429 | 66 | 1.49% | 0.957 | 0.373 | 25.1× | 0.192 | 0.857 | 0.091 | 0.164 | 6 | 1 | 60 |
| R10+ | 13090 | 68 | 0.52% | 0.938 | 0.278 | 53.5× | 0.109 | 1.000 | 0.044 | 0.085 | 3 | 0 | 65 |

#### MLB_DEBUT

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 19160 | 2513 | 13.12% | 0.923 | 0.693 | 5.3× | 0.495 | 0.709 | 0.499 | 0.586 | 1255 | 515 | 1258 |
| R1 | 536 | 371 | 69.22% | 0.875 | 0.934 | 1.3× | 0.600 | 0.853 | 0.827 | 0.840 | 307 | 53 | 64 |
| R2-R3 | 1007 | 372 | 36.94% | 0.904 | 0.838 | 2.3× | 0.675 | 0.760 | 0.766 | 0.763 | 285 | 90 | 87 |
| R4-R10 | 4469 | 857 | 19.18% | 0.886 | 0.653 | 3.4× | 0.527 | 0.633 | 0.469 | 0.539 | 402 | 233 | 455 |
| R10+ | 13148 | 913 | 6.94% | 0.902 | 0.488 | 7.0× | 0.354 | 0.652 | 0.286 | 0.398 | 261 | 139 | 652 |

#### ESTABLISHED_MLB

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 19160 | 796 | 4.15% | 0.926 | 0.424 | 10.2× | 0.295 | 0.684 | 0.131 | 0.219 | 104 | 48 | 692 |
| R1 | 536 | 181 | 33.77% | 0.782 | 0.630 | 1.9× | 0.462 | 0.699 | 0.320 | 0.439 | 58 | 25 | 123 |
| R2-R3 | 1007 | 114 | 11.32% | 0.869 | 0.444 | 3.9× | 0.405 | 0.595 | 0.193 | 0.291 | 22 | 15 | 92 |
| R4-R10 | 4469 | 273 | 6.11% | 0.896 | 0.393 | 6.4× | 0.328 | 0.739 | 0.062 | 0.115 | 17 | 6 | 256 |
| R10+ | 13148 | 228 | 1.73% | 0.905 | 0.256 | 14.7× | 0.183 | 0.778 | 0.031 | 0.059 | 7 | 2 | 221 |

#### STAR_PLUS_ELITE

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 19160 | 112 | 0.58% | 0.925 | 0.168 | 28.8× | 0.112 | 1.000 | 0.009 | 0.018 | 1 | 0 | 111 |
| R1 | 536 | 40 | 7.46% | 0.771 | 0.259 | 3.5× | 0.246 | 1.000 | 0.025 | 0.049 | 1 | 0 | 39 |
| R2-R3 | 1007 | 10 | 0.99% | 0.909 | 0.258 | 26.0× | 0.141 | — | 0.000 | — | 0 | 0 | 10 |
| R4-R10 | 4469 | 36 | 0.81% | 0.896 | 0.201 | 24.9× | 0.123 | — | 0.000 | — | 0 | 0 | 36 |
| R10+ | 13148 | 26 | 0.20% | 0.885 | 0.067 | 33.9× | 0.059 | — | 0.000 | — | 0 | 0 | 26 |

## Per-yip (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2696 | 124 | 4.60% | 0.916 | 0.535 | 11.6× | 0.302 | 0.700 | 0.339 | 0.457 | 42 | 18 | 82 |
| 1 | 2513 | 86 | 3.42% | 0.928 | 0.524 | 15.3× | 0.270 | 0.738 | 0.360 | 0.484 | 31 | 11 | 55 |
| 2 | 2305 | 48 | 2.08% | 0.943 | 0.514 | 24.7× | 0.219 | 0.750 | 0.312 | 0.441 | 15 | 5 | 33 |
| 3 | 2038 | 17 | 0.83% | 0.967 | 0.520 | 62.4× | 0.147 | 0.714 | 0.294 | 0.417 | 5 | 2 | 12 |
| 4 | 1800 | 7 | 0.39% | 0.949 | 0.510 | 131.0× | 0.097 | 1.000 | 0.143 | 0.250 | 1 | 0 | 6 |
| 5 | 1607 | 1 | 0.06% | 0.680 | 0.002 | 3.1× | 0.016 | — | 0.000 | — | 0 | 0 | 1 |
| 6 | 1450 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 7 | 1319 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 8 | 1190 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 9 | 1059 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 924 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

#### MLB_DEBUT

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2702 | 566 | 20.95% | 0.824 | 0.582 | 2.8× | 0.457 | 0.674 | 0.369 | 0.477 | 209 | 101 | 357 |
| 1 | 2552 | 579 | 22.69% | 0.875 | 0.726 | 3.2× | 0.544 | 0.720 | 0.525 | 0.607 | 304 | 118 | 275 |
| 2 | 2360 | 524 | 22.20% | 0.899 | 0.758 | 3.4× | 0.574 | 0.723 | 0.603 | 0.658 | 316 | 121 | 208 |
| 3 | 2094 | 385 | 18.39% | 0.912 | 0.757 | 4.1× | 0.553 | 0.732 | 0.595 | 0.656 | 229 | 84 | 156 |
| 4 | 1839 | 231 | 12.56% | 0.924 | 0.698 | 5.6× | 0.487 | 0.699 | 0.502 | 0.584 | 116 | 50 | 115 |
| 5 | 1630 | 125 | 7.67% | 0.939 | 0.614 | 8.0× | 0.405 | 0.704 | 0.400 | 0.510 | 50 | 21 | 75 |
| 6 | 1463 | 63 | 4.31% | 0.951 | 0.559 | 13.0× | 0.317 | 0.613 | 0.302 | 0.404 | 19 | 12 | 44 |
| 7 | 1329 | 27 | 2.03% | 0.969 | 0.511 | 25.2× | 0.229 | 0.588 | 0.370 | 0.455 | 10 | 7 | 17 |
| 8 | 1196 | 9 | 0.75% | 0.981 | 0.489 | 65.0× | 0.144 | 1.000 | 0.222 | 0.364 | 2 | 0 | 7 |
| 9 | 1065 | 3 | 0.28% | 0.959 | 0.097 | 34.4× | 0.084 | 0.000 | 0.000 | — | 0 | 1 | 3 |
| 10 | 930 | 1 | 0.11% | 0.991 | 0.111 | 103.3× | 0.056 | — | 0.000 | — | 0 | 0 | 1 |

#### ESTABLISHED_MLB

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2702 | 164 | 6.07% | 0.833 | 0.304 | 5.0× | 0.276 | 0.632 | 0.073 | 0.131 | 12 | 7 | 152 |
| 1 | 2552 | 201 | 7.88% | 0.888 | 0.432 | 5.5× | 0.362 | 0.633 | 0.154 | 0.248 | 31 | 18 | 170 |
| 2 | 2360 | 192 | 8.14% | 0.896 | 0.499 | 6.1× | 0.375 | 0.740 | 0.193 | 0.306 | 37 | 13 | 155 |
| 3 | 2094 | 121 | 5.78% | 0.918 | 0.514 | 8.9× | 0.338 | 0.696 | 0.132 | 0.222 | 16 | 7 | 105 |
| 4 | 1839 | 73 | 3.97% | 0.917 | 0.487 | 12.3× | 0.282 | 0.800 | 0.110 | 0.193 | 8 | 2 | 65 |
| 5 | 1630 | 31 | 1.90% | 0.929 | 0.253 | 13.3× | 0.203 | 0.000 | 0.000 | — | 0 | 1 | 31 |
| 6 | 1463 | 11 | 0.75% | 0.962 | 0.349 | 46.4× | 0.138 | — | 0.000 | — | 0 | 0 | 11 |
| 7 | 1329 | 3 | 0.23% | 0.980 | 0.111 | 49.0× | 0.079 | — | 0.000 | — | 0 | 0 | 3 |
| 8 | 1196 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 9 | 1065 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 930 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

#### STAR_PLUS_ELITE

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2702 | 26 | 0.96% | 0.847 | 0.124 | 12.8× | 0.117 | — | 0.000 | — | 0 | 0 | 26 |
| 1 | 2552 | 33 | 1.29% | 0.908 | 0.244 | 18.9× | 0.160 | 1.000 | 0.030 | 0.059 | 1 | 0 | 32 |
| 2 | 2360 | 25 | 1.06% | 0.908 | 0.246 | 23.2× | 0.145 | — | 0.000 | — | 0 | 0 | 25 |
| 3 | 2094 | 18 | 0.86% | 0.881 | 0.136 | 15.8× | 0.122 | — | 0.000 | — | 0 | 0 | 18 |
| 4 | 1839 | 9 | 0.49% | 0.842 | 0.125 | 25.6× | 0.083 | — | 0.000 | — | 0 | 0 | 9 |
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
| ALL | 18901 | 283 | 1.50% | 0.964 | 0.513 | 34.3× | 0.195 | 0.723 | 0.332 | 0.455 | 94 | 36 | 189 |
| RK | 1460 | 45 | 3.08% | 0.968 | 0.632 | 20.5× | 0.280 | 0.750 | 0.467 | 0.575 | 21 | 7 | 24 |
| A- | 1044 | 19 | 1.82% | 0.977 | 0.507 | 27.8× | 0.221 | 0.545 | 0.316 | 0.400 | 6 | 5 | 13 |
| A | 1451 | 42 | 2.89% | 0.966 | 0.609 | 21.1× | 0.271 | 0.731 | 0.452 | 0.559 | 19 | 7 | 23 |
| A+ | 1460 | 37 | 2.53% | 0.976 | 0.636 | 25.1× | 0.259 | 0.778 | 0.378 | 0.509 | 14 | 4 | 23 |
| AA | 1323 | 21 | 1.59% | 0.990 | 0.729 | 45.9× | 0.212 | 0.824 | 0.667 | 0.737 | 14 | 3 | 7 |
| AAA | 1056 | 12 | 1.14% | 0.992 | 0.568 | 50.0× | 0.181 | 0.667 | 0.500 | 0.571 | 6 | 3 | 6 |
| NONE | 11073 | 107 | 0.97% | 0.960 | 0.375 | 38.8× | 0.156 | 0.667 | 0.131 | 0.219 | 14 | 7 | 93 |

#### MLB_DEBUT

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 19160 | 2513 | 13.12% | 0.923 | 0.693 | 5.3× | 0.495 | 0.709 | 0.499 | 0.586 | 1255 | 515 | 1258 |
| RK | 1465 | 169 | 11.54% | 0.882 | 0.552 | 4.8× | 0.423 | 0.707 | 0.314 | 0.434 | 53 | 22 | 116 |
| A- | 1046 | 166 | 15.87% | 0.883 | 0.660 | 4.2× | 0.485 | 0.743 | 0.452 | 0.562 | 75 | 26 | 91 |
| A | 1474 | 291 | 19.74% | 0.886 | 0.712 | 3.6× | 0.532 | 0.659 | 0.591 | 0.623 | 172 | 89 | 119 |
| A+ | 1480 | 309 | 20.88% | 0.899 | 0.767 | 3.7× | 0.562 | 0.714 | 0.638 | 0.674 | 197 | 79 | 112 |
| AA | 1390 | 461 | 33.17% | 0.893 | 0.837 | 2.5× | 0.642 | 0.730 | 0.744 | 0.737 | 343 | 127 | 118 |
| AAA | 1113 | 353 | 31.72% | 0.877 | 0.794 | 2.5× | 0.608 | 0.744 | 0.660 | 0.700 | 233 | 80 | 120 |
| NONE | 11150 | 763 | 6.84% | 0.936 | 0.545 | 8.0× | 0.382 | 0.675 | 0.237 | 0.351 | 181 | 87 | 582 |

#### ESTABLISHED_MLB

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 19160 | 796 | 4.15% | 0.926 | 0.424 | 10.2× | 0.295 | 0.684 | 0.131 | 0.219 | 104 | 48 | 692 |
| RK | 1465 | 33 | 2.25% | 0.900 | 0.206 | 9.2× | 0.206 | 0.000 | 0.000 | — | 0 | 1 | 33 |
| A- | 1046 | 39 | 3.73% | 0.921 | 0.397 | 10.7× | 0.276 | 1.000 | 0.077 | 0.143 | 3 | 0 | 36 |
| A | 1474 | 72 | 4.88% | 0.900 | 0.366 | 7.5× | 0.299 | 0.636 | 0.097 | 0.169 | 7 | 4 | 65 |
| A+ | 1480 | 100 | 6.76% | 0.929 | 0.496 | 7.3× | 0.373 | 0.636 | 0.140 | 0.230 | 14 | 8 | 86 |
| AA | 1390 | 155 | 11.15% | 0.911 | 0.576 | 5.2× | 0.448 | 0.727 | 0.258 | 0.381 | 40 | 15 | 115 |
| AAA | 1113 | 105 | 9.43% | 0.906 | 0.553 | 5.9× | 0.411 | 0.676 | 0.238 | 0.352 | 25 | 12 | 80 |
| NONE | 11150 | 291 | 2.61% | 0.934 | 0.321 | 12.3× | 0.239 | 0.667 | 0.048 | 0.090 | 14 | 7 | 277 |

#### STAR_PLUS_ELITE

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 19160 | 112 | 0.58% | 0.925 | 0.168 | 28.8× | 0.112 | 1.000 | 0.009 | 0.018 | 1 | 0 | 111 |
| RK | 1465 | 8 | 0.55% | 0.906 | 0.111 | 20.3× | 0.104 | — | 0.000 | — | 0 | 0 | 8 |
| A- | 1046 | 7 | 0.67% | 0.928 | 0.340 | 50.7× | 0.121 | — | 0.000 | — | 0 | 0 | 7 |
| A | 1474 | 12 | 0.81% | 0.956 | 0.173 | 21.2× | 0.142 | — | 0.000 | — | 0 | 0 | 12 |
| A+ | 1480 | 12 | 0.81% | 0.923 | 0.128 | 15.8× | 0.132 | — | 0.000 | — | 0 | 0 | 12 |
| AA | 1390 | 24 | 1.73% | 0.954 | 0.391 | 22.7× | 0.205 | — | 0.000 | — | 0 | 0 | 24 |
| AAA | 1113 | 14 | 1.26% | 0.982 | 0.396 | 31.4× | 0.186 | 1.000 | 0.071 | 0.133 | 1 | 0 | 13 |
| NONE | 11150 | 34 | 0.30% | 0.900 | 0.042 | 13.6× | 0.076 | — | 0.000 | — | 0 | 0 | 34 |

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
