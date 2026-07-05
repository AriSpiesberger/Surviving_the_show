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
| TOP_100_PROSPECT | 25631 | 0.75% | **0.560** | 74.8× | 0.988 | 0.146 | 0.759 | 0.328 | 0.458 |
| MLB_DEBUT | 25814 | 5.85% | **0.659** | 11.3× | 0.954 | 0.369 | 0.794 | 0.343 | 0.479 |
| ESTABLISHED_MLB | 25814 | 1.69% | **0.428** | 25.3× | 0.973 | 0.211 | 0.743 | 0.059 | 0.110 |
| STAR_PLUS_ELITE | 25814 | 0.30% | **0.164** | 54.3× | 0.979 | 0.091 | — | 0.000 | — |
| **weighted-AP** | | | **0.494** | | | | | | |

(MLB_DEBUT 2× weight, others 1×, per-event eligibility filters. Scores =
`xp_<event>_h6` vs realized-within-6y, on rows resolved at h=6.)

## Per-horizon trajectory (h=1..10, resolved at each h)

#### TOP_100_PROSPECT

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32955 | 89 | 0.27% | 0.995 | 0.511 | 189.3× | 0.0018 | 0.99 |
| 2 | 31881 | 159 | 0.50% | 0.991 | 0.529 | 106.0× | 0.0032 | 0.95 |
| 3 | 30609 | 200 | 0.65% | 0.989 | 0.568 | 86.9× | 0.0040 | 0.95 |
| 4 | 29147 | 213 | 0.73% | 0.988 | 0.550 | 75.3× | 0.0046 | 0.99 |
| 5 | 27489 | 209 | 0.76% | 0.988 | 0.558 | 73.4× | 0.0047 | 1.05 |
| 6 | 25631 | 192 | 0.75% | 0.988 | 0.560 | 74.8× | 0.0047 | 1.13 |
| 7 | 23578 | 182 | 0.77% | 0.988 | 0.564 | 73.0× | 0.0048 | 1.15 |
| 8 | 21522 | 170 | 0.79% | 0.987 | 0.559 | 70.8× | 0.0049 | 1.17 |
| 9 | 19436 | 156 | 0.80% | 0.986 | 0.553 | 68.9× | 0.0051 | 1.20 |
| 10 | 17265 | 143 | 0.83% | 0.986 | 0.557 | 67.2× | 0.0052 | 1.22 |

#### MLB_DEBUT

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 33187 | 433 | 1.30% | 0.980 | 0.508 | 39.0× | 0.0086 | 0.94 |
| 2 | 32107 | 827 | 2.58% | 0.970 | 0.615 | 23.9× | 0.0148 | 0.95 |
| 3 | 30829 | 1173 | 3.80% | 0.960 | 0.632 | 16.6× | 0.0214 | 0.97 |
| 4 | 29356 | 1405 | 4.79% | 0.956 | 0.641 | 13.4× | 0.0266 | 0.98 |
| 5 | 27685 | 1511 | 5.46% | 0.954 | 0.650 | 11.9× | 0.0298 | 0.99 |
| 6 | 25814 | 1511 | 5.85% | 0.954 | 0.659 | 11.3× | 0.0315 | 1.01 |
| 7 | 23744 | 1460 | 6.15% | 0.953 | 0.664 | 10.8× | 0.0328 | 1.01 |
| 8 | 21670 | 1367 | 6.31% | 0.953 | 0.661 | 10.5× | 0.0336 | 1.01 |
| 9 | 19567 | 1252 | 6.40% | 0.951 | 0.653 | 10.2× | 0.0343 | 1.01 |
| 10 | 17374 | 1127 | 6.49% | 0.949 | 0.649 | 10.0× | 0.0351 | 1.02 |

#### ESTABLISHED_MLB

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 33187 | 2 | 0.01% | 0.999 | 0.035 | 578.3× | 0.0001 | 3.71 |
| 2 | 32107 | 42 | 0.13% | 0.990 | 0.229 | 174.7× | 0.0011 | 1.02 |
| 3 | 30829 | 142 | 0.46% | 0.985 | 0.352 | 76.5× | 0.0036 | 0.97 |
| 4 | 29356 | 256 | 0.87% | 0.982 | 0.401 | 46.0× | 0.0065 | 0.94 |
| 5 | 27685 | 360 | 1.30% | 0.976 | 0.411 | 31.6× | 0.0095 | 0.93 |
| 6 | 25814 | 437 | 1.69% | 0.973 | 0.428 | 25.3× | 0.0122 | 0.94 |
| 7 | 23744 | 479 | 2.02% | 0.971 | 0.448 | 22.2× | 0.0142 | 0.96 |
| 8 | 21670 | 495 | 2.28% | 0.969 | 0.453 | 19.8× | 0.0160 | 0.96 |
| 9 | 19567 | 490 | 2.50% | 0.967 | 0.446 | 17.8× | 0.0175 | 0.96 |
| 10 | 17374 | 468 | 2.69% | 0.966 | 0.455 | 16.9× | 0.0186 | 0.93 |

#### STAR_PLUS_ELITE

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 33187 | 1 | 0.00% | 0.997 | 0.010 | 342.1× | 0.0000 | 3.33 |
| 2 | 32107 | 7 | 0.02% | 0.993 | 0.152 | 696.2× | 0.0002 | 1.23 |
| 3 | 30829 | 17 | 0.06% | 0.987 | 0.180 | 327.0× | 0.0005 | 1.15 |
| 4 | 29356 | 38 | 0.13% | 0.984 | 0.170 | 131.4× | 0.0012 | 0.95 |
| 5 | 27685 | 61 | 0.22% | 0.981 | 0.161 | 73.3× | 0.0020 | 0.86 |
| 6 | 25814 | 78 | 0.30% | 0.979 | 0.164 | 54.3× | 0.0027 | 0.87 |
| 7 | 23744 | 89 | 0.37% | 0.979 | 0.176 | 47.0× | 0.0033 | 0.91 |
| 8 | 21670 | 97 | 0.45% | 0.978 | 0.183 | 40.8× | 0.0039 | 0.94 |
| 9 | 19567 | 106 | 0.54% | 0.977 | 0.191 | 35.2× | 0.0048 | 0.88 |
| 10 | 17374 | 109 | 0.63% | 0.977 | 0.206 | 32.8× | 0.0055 | 0.82 |

## Per-bucket (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 25631 | 192 | 0.75% | 0.988 | 0.560 | 74.8× | 0.146 | 0.759 | 0.328 | 0.458 | 63 | 20 | 129 |
| R1 | 320 | 68 | 21.25% | 0.920 | 0.733 | 3.4× | 0.596 | 0.737 | 0.618 | 0.672 | 42 | 15 | 26 |
| R2-R3 | 529 | 50 | 9.45% | 0.919 | 0.544 | 5.8× | 0.424 | 0.714 | 0.200 | 0.312 | 10 | 4 | 40 |
| R4-R10 | 1763 | 8 | 0.45% | 0.964 | 0.116 | 25.5× | 0.108 | — | 0.000 | — | 0 | 0 | 8 |
| R10+ | 8574 | 9 | 0.10% | 0.944 | 0.343 | 327.0× | 0.050 | 1.000 | 0.333 | 0.500 | 3 | 0 | 6 |
| IFA | 14445 | 57 | 0.39% | 0.988 | 0.430 | 108.9× | 0.106 | 0.889 | 0.140 | 0.242 | 8 | 1 | 49 |

#### MLB_DEBUT

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 25814 | 1511 | 5.85% | 0.954 | 0.659 | 11.3× | 0.369 | 0.794 | 0.343 | 0.479 | 518 | 134 | 993 |
| R1 | 393 | 203 | 51.65% | 0.890 | 0.887 | 1.7× | 0.675 | 0.840 | 0.749 | 0.792 | 152 | 29 | 51 |
| R2-R3 | 558 | 219 | 39.25% | 0.855 | 0.797 | 2.0× | 0.600 | 0.775 | 0.489 | 0.599 | 107 | 31 | 112 |
| R4-R10 | 1764 | 248 | 14.06% | 0.880 | 0.574 | 4.1× | 0.458 | 0.708 | 0.254 | 0.374 | 63 | 26 | 185 |
| R10+ | 8576 | 343 | 4.00% | 0.936 | 0.480 | 12.0× | 0.296 | 0.800 | 0.128 | 0.221 | 44 | 11 | 299 |
| IFA | 14523 | 498 | 3.43% | 0.954 | 0.613 | 17.9× | 0.286 | 0.804 | 0.305 | 0.443 | 152 | 37 | 346 |

#### ESTABLISHED_MLB

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 25814 | 437 | 1.69% | 0.973 | 0.428 | 25.3× | 0.211 | 0.743 | 0.059 | 0.110 | 26 | 9 | 411 |
| R1 | 393 | 103 | 26.21% | 0.838 | 0.616 | 2.3× | 0.516 | 0.733 | 0.107 | 0.186 | 11 | 4 | 92 |
| R2-R3 | 558 | 67 | 12.01% | 0.849 | 0.438 | 3.6× | 0.393 | 0.857 | 0.090 | 0.162 | 6 | 1 | 61 |
| R4-R10 | 1764 | 78 | 4.42% | 0.924 | 0.370 | 8.4× | 0.302 | — | 0.000 | — | 0 | 0 | 78 |
| R10+ | 8576 | 69 | 0.80% | 0.969 | 0.292 | 36.3× | 0.145 | 1.000 | 0.029 | 0.056 | 2 | 0 | 67 |
| IFA | 14523 | 120 | 0.83% | 0.977 | 0.355 | 43.0× | 0.149 | 0.636 | 0.058 | 0.107 | 7 | 4 | 113 |

#### STAR_PLUS_ELITE

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 25814 | 78 | 0.30% | 0.979 | 0.164 | 54.3× | 0.091 | — | 0.000 | — | 0 | 0 | 78 |
| R1 | 393 | 24 | 6.11% | 0.862 | 0.241 | 3.9× | 0.300 | — | 0.000 | — | 0 | 0 | 24 |
| R2-R3 | 558 | 11 | 1.97% | 0.908 | 0.110 | 5.6× | 0.197 | — | 0.000 | — | 0 | 0 | 11 |
| R4-R10 | 1764 | 11 | 0.62% | 0.933 | 0.200 | 32.1× | 0.118 | — | 0.000 | — | 0 | 0 | 11 |
| R10+ | 8576 | 11 | 0.13% | 0.969 | 0.266 | 207.8× | 0.058 | — | 0.000 | — | 0 | 0 | 11 |
| IFA | 14523 | 21 | 0.14% | 0.985 | 0.159 | 109.7× | 0.064 | — | 0.000 | — | 0 | 0 | 21 |

## Per-yip (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3495 | 86 | 2.46% | 0.962 | 0.508 | 20.6× | 0.248 | 0.638 | 0.349 | 0.451 | 30 | 17 | 56 |
| 1 | 3256 | 58 | 1.78% | 0.981 | 0.638 | 35.8× | 0.220 | 0.895 | 0.293 | 0.442 | 17 | 2 | 41 |
| 2 | 3007 | 31 | 1.03% | 0.985 | 0.669 | 64.9× | 0.170 | 0.923 | 0.387 | 0.545 | 12 | 1 | 19 |
| 3 | 2738 | 13 | 0.47% | 0.984 | 0.570 | 120.1× | 0.115 | 1.000 | 0.077 | 0.143 | 1 | 0 | 12 |
| 4 | 2482 | 4 | 0.16% | 0.985 | 0.757 | 469.5× | 0.067 | 1.000 | 0.750 | 0.857 | 3 | 0 | 1 |
| 5 | 2255 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 6 | 2057 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 7 | 1857 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 8 | 1669 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 9 | 1498 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 1317 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

#### MLB_DEBUT

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3514 | 367 | 10.44% | 0.900 | 0.543 | 5.2× | 0.424 | 0.716 | 0.226 | 0.344 | 83 | 33 | 284 |
| 1 | 3283 | 349 | 10.63% | 0.923 | 0.653 | 6.1× | 0.452 | 0.748 | 0.332 | 0.460 | 116 | 39 | 233 |
| 2 | 3046 | 310 | 10.18% | 0.945 | 0.731 | 7.2× | 0.466 | 0.813 | 0.435 | 0.567 | 135 | 31 | 175 |
| 3 | 2771 | 210 | 7.58% | 0.958 | 0.752 | 9.9× | 0.420 | 0.845 | 0.467 | 0.601 | 98 | 18 | 112 |
| 4 | 2507 | 130 | 5.19% | 0.956 | 0.739 | 14.3× | 0.350 | 0.857 | 0.462 | 0.600 | 60 | 10 | 70 |
| 5 | 2268 | 67 | 2.95% | 0.951 | 0.582 | 19.7× | 0.265 | 0.895 | 0.254 | 0.395 | 17 | 2 | 50 |
| 6 | 2066 | 38 | 1.84% | 0.956 | 0.496 | 27.0× | 0.212 | 0.857 | 0.158 | 0.267 | 6 | 1 | 32 |
| 7 | 1862 | 21 | 1.13% | 0.969 | 0.473 | 41.9× | 0.172 | 1.000 | 0.048 | 0.091 | 1 | 0 | 20 |
| 8 | 1674 | 12 | 0.72% | 0.946 | 0.507 | 70.8× | 0.130 | 1.000 | 0.167 | 0.286 | 2 | 0 | 10 |
| 9 | 1502 | 6 | 0.40% | 0.960 | 0.702 | 175.8× | 0.101 | — | 0.000 | — | 0 | 0 | 6 |
| 10 | 1321 | 1 | 0.08% | 0.783 | 0.003 | 4.6× | 0.027 | — | 0.000 | — | 0 | 0 | 1 |

#### ESTABLISHED_MLB

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3514 | 111 | 3.16% | 0.925 | 0.325 | 10.3× | 0.258 | 0.667 | 0.018 | 0.035 | 2 | 1 | 109 |
| 1 | 3283 | 111 | 3.38% | 0.945 | 0.449 | 13.3× | 0.278 | 0.846 | 0.099 | 0.177 | 11 | 2 | 100 |
| 2 | 3046 | 102 | 3.35% | 0.968 | 0.530 | 15.8× | 0.291 | 0.643 | 0.088 | 0.155 | 9 | 5 | 93 |
| 3 | 2771 | 58 | 2.09% | 0.977 | 0.468 | 22.4× | 0.236 | 0.667 | 0.034 | 0.066 | 2 | 1 | 56 |
| 4 | 2507 | 32 | 1.28% | 0.983 | 0.536 | 42.0× | 0.188 | 1.000 | 0.062 | 0.118 | 2 | 0 | 30 |
| 5 | 2268 | 13 | 0.57% | 0.984 | 0.221 | 38.6× | 0.127 | — | 0.000 | — | 0 | 0 | 13 |
| 6 | 2066 | 6 | 0.29% | 0.988 | 0.132 | 45.3× | 0.091 | — | 0.000 | — | 0 | 0 | 6 |
| 7 | 1862 | 2 | 0.11% | 0.995 | 0.163 | 151.3× | 0.056 | — | 0.000 | — | 0 | 0 | 2 |
| 8 | 1674 | 2 | 0.12% | 0.999 | 0.583 | 488.2× | 0.060 | — | 0.000 | — | 0 | 0 | 2 |
| 9 | 1502 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 1321 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

#### STAR_PLUS_ELITE

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3514 | 23 | 0.65% | 0.931 | 0.096 | 14.7× | 0.120 | — | 0.000 | — | 0 | 0 | 23 |
| 1 | 3283 | 20 | 0.61% | 0.969 | 0.264 | 43.4× | 0.127 | — | 0.000 | — | 0 | 0 | 20 |
| 2 | 3046 | 19 | 0.62% | 0.966 | 0.248 | 39.7× | 0.127 | — | 0.000 | — | 0 | 0 | 19 |
| 3 | 2771 | 9 | 0.32% | 0.985 | 0.340 | 104.8× | 0.096 | — | 0.000 | — | 0 | 0 | 9 |
| 4 | 2507 | 5 | 0.20% | 0.992 | 0.236 | 118.5× | 0.076 | — | 0.000 | — | 0 | 0 | 5 |
| 5 | 2268 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 6 | 2066 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 7 | 1862 | 1 | 0.05% | 0.998 | 0.250 | 465.5× | 0.040 | — | 0.000 | — | 0 | 0 | 1 |
| 8 | 1674 | 1 | 0.06% | 0.999 | 0.500 | 837.0× | 0.042 | — | 0.000 | — | 0 | 0 | 1 |
| 9 | 1502 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 1321 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

## Per-level (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 25631 | 192 | 0.75% | 0.988 | 0.560 | 74.8× | 0.146 | 0.759 | 0.328 | 0.458 | 63 | 20 | 129 |
| RK | 3770 | 65 | 1.72% | 0.969 | 0.568 | 32.9× | 0.212 | 0.778 | 0.323 | 0.457 | 21 | 6 | 44 |
| A- | 987 | 19 | 1.93% | 0.959 | 0.449 | 23.3× | 0.218 | 0.545 | 0.316 | 0.400 | 6 | 5 | 13 |
| A | 1340 | 40 | 2.99% | 0.969 | 0.593 | 19.9× | 0.277 | 0.824 | 0.350 | 0.491 | 14 | 3 | 26 |
| A+ | 1332 | 28 | 2.10% | 0.983 | 0.574 | 27.3× | 0.240 | 0.750 | 0.214 | 0.333 | 6 | 2 | 22 |
| AA | 1151 | 26 | 2.26% | 0.988 | 0.754 | 33.4× | 0.251 | 0.929 | 0.500 | 0.650 | 13 | 1 | 13 |
| AAA | 1399 | 5 | 0.36% | 0.999 | 0.808 | 226.2× | 0.103 | 0.667 | 0.400 | 0.500 | 2 | 1 | 3 |
| NONE | 15652 | 9 | 0.06% | 0.990 | 0.205 | 357.4× | 0.041 | 0.333 | 0.111 | 0.167 | 1 | 2 | 8 |

#### MLB_DEBUT

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 25814 | 1511 | 5.85% | 0.954 | 0.659 | 11.3× | 0.369 | 0.794 | 0.343 | 0.479 | 518 | 134 | 993 |
| RK | 3774 | 208 | 5.51% | 0.922 | 0.522 | 9.5× | 0.334 | 0.717 | 0.159 | 0.260 | 33 | 13 | 175 |
| A- | 989 | 133 | 13.45% | 0.846 | 0.502 | 3.7× | 0.409 | 0.618 | 0.158 | 0.251 | 21 | 13 | 112 |
| A | 1361 | 238 | 17.49% | 0.875 | 0.650 | 3.7× | 0.493 | 0.771 | 0.311 | 0.443 | 74 | 22 | 164 |
| A+ | 1353 | 255 | 18.85% | 0.860 | 0.682 | 3.6× | 0.488 | 0.814 | 0.412 | 0.547 | 105 | 24 | 150 |
| AA | 1225 | 357 | 29.14% | 0.868 | 0.772 | 2.6× | 0.579 | 0.807 | 0.515 | 0.629 | 184 | 44 | 173 |
| AAA | 1432 | 252 | 17.60% | 0.909 | 0.729 | 4.1× | 0.539 | 0.846 | 0.349 | 0.494 | 88 | 16 | 164 |
| NONE | 15680 | 68 | 0.43% | 0.902 | 0.396 | 91.3× | 0.091 | 0.867 | 0.191 | 0.313 | 13 | 2 | 55 |

#### ESTABLISHED_MLB

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 25814 | 437 | 1.69% | 0.973 | 0.428 | 25.3× | 0.211 | 0.743 | 0.059 | 0.110 | 26 | 9 | 411 |
| RK | 3774 | 49 | 1.30% | 0.946 | 0.268 | 20.6× | 0.175 | 0.000 | 0.000 | — | 0 | 1 | 49 |
| A- | 989 | 29 | 2.93% | 0.887 | 0.161 | 5.5× | 0.226 | 0.000 | 0.000 | — | 0 | 1 | 29 |
| A | 1361 | 72 | 5.29% | 0.913 | 0.407 | 7.7× | 0.320 | 0.667 | 0.056 | 0.103 | 4 | 2 | 68 |
| A+ | 1353 | 84 | 6.21% | 0.917 | 0.481 | 7.7× | 0.349 | 1.000 | 0.083 | 0.154 | 7 | 0 | 77 |
| AA | 1225 | 128 | 10.45% | 0.904 | 0.535 | 5.1× | 0.428 | 0.688 | 0.086 | 0.153 | 11 | 5 | 117 |
| AAA | 1432 | 65 | 4.54% | 0.939 | 0.492 | 10.8× | 0.317 | 1.000 | 0.062 | 0.116 | 4 | 0 | 61 |
| NONE | 15680 | 10 | 0.06% | 0.996 | 0.232 | 363.5× | 0.043 | — | 0.000 | — | 0 | 0 | 10 |

#### STAR_PLUS_ELITE

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 25814 | 78 | 0.30% | 0.979 | 0.164 | 54.3× | 0.091 | — | 0.000 | — | 0 | 0 | 78 |
| RK | 3774 | 14 | 0.37% | 0.948 | 0.136 | 36.6× | 0.094 | — | 0.000 | — | 0 | 0 | 14 |
| A- | 989 | 2 | 0.20% | 0.930 | 0.024 | 11.7× | 0.067 | — | 0.000 | — | 0 | 0 | 2 |
| A | 1361 | 13 | 0.96% | 0.954 | 0.215 | 22.5× | 0.153 | — | 0.000 | — | 0 | 0 | 13 |
| A+ | 1353 | 17 | 1.26% | 0.931 | 0.316 | 25.2× | 0.166 | — | 0.000 | — | 0 | 0 | 17 |
| AA | 1225 | 19 | 1.55% | 0.942 | 0.254 | 16.4× | 0.189 | — | 0.000 | — | 0 | 0 | 19 |
| AAA | 1432 | 12 | 0.84% | 0.960 | 0.207 | 24.7× | 0.145 | — | 0.000 | — | 0 | 0 | 12 |
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
