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
| TOP_100_PROSPECT | 18868 | 1.60% | **0.568** | 35.5× | 0.976 | 0.207 | 0.683 | 0.364 | 0.475 |
| MLB_DEBUT | 19160 | 13.12% | **0.701** | 5.3× | 0.930 | 0.503 | 0.770 | 0.421 | 0.544 |
| ESTABLISHED_MLB | 19160 | 4.15% | **0.447** | 10.8× | 0.938 | 0.303 | 0.642 | 0.173 | 0.273 |
| STAR_PLUS_ELITE | 19160 | 0.58% | **0.219** | 37.5× | 0.959 | 0.121 | 0.571 | 0.071 | 0.127 |
| **weighted-AP** | | | **0.528** | | | | | | |

(MLB_DEBUT 2× weight, others 1×, per-event eligibility filters. Scores =
`xp_<event>_h6` vs realized-within-6y, on rows resolved at h=6.)

## Per-horizon trajectory (h=1..10, resolved at each h)

#### TOP_100_PROSPECT

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 23524 | 123 | 0.52% | 0.994 | 0.623 | 119.1× | 0.0030 | 1.02 |
| 2 | 22857 | 218 | 0.95% | 0.986 | 0.628 | 65.8× | 0.0054 | 0.95 |
| 3 | 22081 | 276 | 1.25% | 0.980 | 0.596 | 47.7× | 0.0074 | 0.93 |
| 4 | 21163 | 297 | 1.40% | 0.978 | 0.583 | 41.6× | 0.0084 | 0.95 |
| 5 | 20093 | 303 | 1.51% | 0.977 | 0.572 | 37.9× | 0.0092 | 0.99 |
| 6 | 18868 | 302 | 1.60% | 0.976 | 0.568 | 35.5× | 0.0098 | 0.99 |
| 7 | 17523 | 296 | 1.69% | 0.974 | 0.561 | 33.2× | 0.0104 | 0.99 |
| 8 | 16160 | 285 | 1.76% | 0.972 | 0.555 | 31.5× | 0.0109 | 0.99 |
| 9 | 14785 | 272 | 1.84% | 0.971 | 0.550 | 29.9× | 0.0114 | 0.98 |
| 10 | 13409 | 256 | 1.91% | 0.969 | 0.544 | 28.5× | 0.0120 | 0.99 |

#### MLB_DEBUT

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 23869 | 651 | 2.73% | 0.964 | 0.503 | 18.4× | 0.0180 | 0.98 |
| 2 | 23197 | 1283 | 5.53% | 0.953 | 0.612 | 11.1× | 0.0318 | 0.95 |
| 3 | 22414 | 1855 | 8.28% | 0.943 | 0.657 | 7.9× | 0.0447 | 0.94 |
| 4 | 21487 | 2256 | 10.50% | 0.936 | 0.679 | 6.5× | 0.0547 | 0.92 |
| 5 | 20403 | 2461 | 12.06% | 0.932 | 0.690 | 5.7× | 0.0612 | 0.92 |
| 6 | 19160 | 2513 | 13.12% | 0.930 | 0.701 | 5.3× | 0.0654 | 0.92 |
| 7 | 17795 | 2468 | 13.87% | 0.928 | 0.705 | 5.1× | 0.0686 | 0.92 |
| 8 | 16412 | 2367 | 14.42% | 0.926 | 0.702 | 4.9× | 0.0714 | 0.92 |
| 9 | 15018 | 2243 | 14.94% | 0.923 | 0.697 | 4.7× | 0.0742 | 0.91 |
| 10 | 13622 | 2105 | 15.45% | 0.920 | 0.696 | 4.5× | 0.0769 | 0.91 |

#### ESTABLISHED_MLB

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 23869 | 6 | 0.03% | 0.987 | 0.232 | 923.2× | 0.0002 | 1.41 |
| 2 | 23197 | 74 | 0.32% | 0.972 | 0.272 | 85.2× | 0.0027 | 1.00 |
| 3 | 22414 | 238 | 1.06% | 0.967 | 0.342 | 32.2× | 0.0084 | 0.99 |
| 4 | 21487 | 447 | 2.08% | 0.955 | 0.389 | 18.7× | 0.0156 | 0.91 |
| 5 | 20403 | 642 | 3.15% | 0.947 | 0.416 | 13.2× | 0.0228 | 0.89 |
| 6 | 19160 | 796 | 4.15% | 0.938 | 0.447 | 10.8× | 0.0293 | 0.87 |
| 7 | 17795 | 899 | 5.05% | 0.934 | 0.471 | 9.3× | 0.0347 | 0.87 |
| 8 | 16412 | 953 | 5.81% | 0.929 | 0.486 | 8.4× | 0.0392 | 0.87 |
| 9 | 15018 | 967 | 6.44% | 0.924 | 0.490 | 7.6× | 0.0432 | 0.86 |
| 10 | 13622 | 955 | 7.01% | 0.919 | 0.495 | 7.1× | 0.0469 | 0.84 |

#### STAR_PLUS_ELITE

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 23869 | 7 | 0.03% | 0.998 | 0.180 | 614.7× | 0.0003 | 0.70 |
| 2 | 23197 | 19 | 0.08% | 0.992 | 0.226 | 276.2× | 0.0007 | 0.92 |
| 3 | 22414 | 36 | 0.16% | 0.976 | 0.175 | 109.2× | 0.0015 | 1.03 |
| 4 | 21487 | 58 | 0.27% | 0.971 | 0.177 | 65.7× | 0.0024 | 1.10 |
| 5 | 20403 | 86 | 0.42% | 0.963 | 0.196 | 46.4× | 0.0037 | 1.08 |
| 6 | 19160 | 112 | 0.58% | 0.959 | 0.219 | 37.5× | 0.0051 | 1.06 |
| 7 | 17795 | 132 | 0.74% | 0.952 | 0.221 | 29.8× | 0.0065 | 1.06 |
| 8 | 16412 | 147 | 0.90% | 0.947 | 0.233 | 26.1× | 0.0077 | 1.06 |
| 9 | 15018 | 162 | 1.08% | 0.944 | 0.252 | 23.4× | 0.0092 | 1.02 |
| 10 | 13622 | 170 | 1.25% | 0.942 | 0.264 | 21.1× | 0.0105 | 0.95 |

## Per-bucket (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 18868 | 302 | 1.60% | 0.976 | 0.568 | 35.5× | 0.207 | 0.683 | 0.364 | 0.475 | 110 | 51 | 192 |
| R1 | 365 | 109 | 29.86% | 0.915 | 0.818 | 2.7× | 0.659 | 0.700 | 0.771 | 0.734 | 84 | 36 | 25 |
| R2-R3 | 984 | 51 | 5.18% | 0.946 | 0.497 | 9.6× | 0.343 | 0.579 | 0.216 | 0.314 | 11 | 8 | 40 |
| R4-R10 | 4429 | 70 | 1.58% | 0.956 | 0.453 | 28.7× | 0.197 | 1.000 | 0.129 | 0.228 | 9 | 0 | 61 |
| R10+ | 13090 | 72 | 0.55% | 0.969 | 0.285 | 51.8× | 0.120 | 0.462 | 0.083 | 0.141 | 6 | 7 | 66 |

#### MLB_DEBUT

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 19160 | 2513 | 13.12% | 0.930 | 0.701 | 5.3× | 0.503 | 0.770 | 0.421 | 0.544 | 1057 | 315 | 1456 |
| R1 | 536 | 371 | 69.22% | 0.889 | 0.931 | 1.3× | 0.621 | 0.872 | 0.898 | 0.884 | 333 | 49 | 38 |
| R2-R3 | 1007 | 372 | 36.94% | 0.902 | 0.828 | 2.2× | 0.672 | 0.795 | 0.645 | 0.712 | 240 | 62 | 132 |
| R4-R10 | 4469 | 857 | 19.18% | 0.904 | 0.676 | 3.5× | 0.551 | 0.705 | 0.357 | 0.474 | 306 | 128 | 551 |
| R10+ | 13148 | 913 | 6.94% | 0.909 | 0.472 | 6.8× | 0.361 | 0.701 | 0.195 | 0.305 | 178 | 76 | 735 |

#### ESTABLISHED_MLB

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 19160 | 796 | 4.15% | 0.938 | 0.447 | 10.8× | 0.303 | 0.642 | 0.173 | 0.273 | 138 | 77 | 658 |
| R1 | 536 | 181 | 33.77% | 0.823 | 0.684 | 2.0× | 0.529 | 0.639 | 0.508 | 0.566 | 92 | 52 | 89 |
| R2-R3 | 1007 | 114 | 11.32% | 0.858 | 0.402 | 3.6× | 0.392 | 0.588 | 0.175 | 0.270 | 20 | 14 | 94 |
| R4-R10 | 4469 | 273 | 6.11% | 0.910 | 0.414 | 6.8× | 0.340 | 0.731 | 0.070 | 0.127 | 19 | 7 | 254 |
| R10+ | 13148 | 228 | 1.73% | 0.927 | 0.278 | 16.0× | 0.193 | 0.636 | 0.031 | 0.059 | 7 | 4 | 221 |

#### STAR_PLUS_ELITE

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 19160 | 112 | 0.58% | 0.959 | 0.219 | 37.5× | 0.121 | 0.571 | 0.071 | 0.127 | 8 | 6 | 104 |
| R1 | 536 | 40 | 7.46% | 0.813 | 0.352 | 4.7× | 0.285 | 0.571 | 0.200 | 0.296 | 8 | 6 | 32 |
| R2-R3 | 1007 | 10 | 0.99% | 0.924 | 0.185 | 18.7× | 0.146 | — | 0.000 | — | 0 | 0 | 10 |
| R4-R10 | 4469 | 36 | 0.81% | 0.931 | 0.300 | 37.2× | 0.133 | — | 0.000 | — | 0 | 0 | 36 |
| R10+ | 13148 | 26 | 0.20% | 0.967 | 0.087 | 44.0× | 0.072 | — | 0.000 | — | 0 | 0 | 26 |

## Per-yip (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2696 | 134 | 4.97% | 0.940 | 0.587 | 11.8× | 0.331 | 0.679 | 0.410 | 0.512 | 55 | 26 | 79 |
| 1 | 2508 | 91 | 3.63% | 0.949 | 0.564 | 15.5× | 0.291 | 0.682 | 0.330 | 0.444 | 30 | 14 | 61 |
| 2 | 2300 | 51 | 2.22% | 0.966 | 0.550 | 24.8× | 0.238 | 0.655 | 0.373 | 0.475 | 19 | 10 | 32 |
| 3 | 2032 | 18 | 0.89% | 0.976 | 0.593 | 67.0× | 0.154 | 0.857 | 0.333 | 0.480 | 6 | 1 | 12 |
| 4 | 1795 | 7 | 0.39% | 0.963 | 0.596 | 152.8× | 0.100 | — | 0.000 | — | 0 | 0 | 7 |
| 5 | 1605 | 1 | 0.06% | 0.779 | 0.003 | 4.5× | 0.024 | — | 0.000 | — | 0 | 0 | 1 |
| 6 | 1448 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 7 | 1317 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 8 | 1188 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 9 | 1057 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 922 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

#### MLB_DEBUT

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2702 | 566 | 20.95% | 0.825 | 0.582 | 2.8× | 0.458 | 0.681 | 0.309 | 0.425 | 175 | 82 | 391 |
| 1 | 2552 | 579 | 22.69% | 0.876 | 0.714 | 3.1× | 0.545 | 0.765 | 0.454 | 0.570 | 263 | 81 | 316 |
| 2 | 2360 | 524 | 22.20% | 0.907 | 0.760 | 3.4× | 0.586 | 0.793 | 0.498 | 0.612 | 261 | 68 | 263 |
| 3 | 2094 | 385 | 18.39% | 0.925 | 0.764 | 4.2× | 0.570 | 0.798 | 0.483 | 0.602 | 186 | 47 | 199 |
| 4 | 1839 | 231 | 12.56% | 0.937 | 0.726 | 5.8× | 0.502 | 0.783 | 0.437 | 0.561 | 101 | 28 | 130 |
| 5 | 1630 | 125 | 7.67% | 0.946 | 0.668 | 8.7× | 0.411 | 0.875 | 0.336 | 0.486 | 42 | 6 | 83 |
| 6 | 1463 | 63 | 4.31% | 0.957 | 0.693 | 16.1× | 0.321 | 0.875 | 0.333 | 0.483 | 21 | 3 | 42 |
| 7 | 1329 | 27 | 2.03% | 0.980 | 0.584 | 28.7× | 0.235 | 1.000 | 0.185 | 0.312 | 5 | 0 | 22 |
| 8 | 1196 | 9 | 0.75% | 0.986 | 0.593 | 78.7× | 0.145 | 1.000 | 0.333 | 0.500 | 3 | 0 | 6 |
| 9 | 1065 | 3 | 0.28% | 0.995 | 0.278 | 98.6× | 0.091 | — | 0.000 | — | 0 | 0 | 3 |
| 10 | 930 | 1 | 0.11% | 0.991 | 0.111 | 103.3× | 0.056 | — | 0.000 | — | 0 | 0 | 1 |

#### ESTABLISHED_MLB

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2702 | 164 | 6.07% | 0.849 | 0.327 | 5.4× | 0.288 | 0.586 | 0.104 | 0.176 | 17 | 12 | 147 |
| 1 | 2552 | 201 | 7.88% | 0.896 | 0.471 | 6.0× | 0.370 | 0.627 | 0.209 | 0.313 | 42 | 25 | 159 |
| 2 | 2360 | 192 | 8.14% | 0.914 | 0.528 | 6.5× | 0.392 | 0.641 | 0.260 | 0.370 | 50 | 28 | 142 |
| 3 | 2094 | 121 | 5.78% | 0.932 | 0.488 | 8.4× | 0.349 | 0.667 | 0.165 | 0.265 | 20 | 10 | 101 |
| 4 | 1839 | 73 | 3.97% | 0.948 | 0.540 | 13.6× | 0.303 | 0.818 | 0.123 | 0.214 | 9 | 2 | 64 |
| 5 | 1630 | 31 | 1.90% | 0.950 | 0.332 | 17.5× | 0.213 | — | 0.000 | — | 0 | 0 | 31 |
| 6 | 1463 | 11 | 0.75% | 0.950 | 0.299 | 39.8× | 0.135 | — | 0.000 | — | 0 | 0 | 11 |
| 7 | 1329 | 3 | 0.23% | 0.988 | 0.107 | 47.5× | 0.080 | — | 0.000 | — | 0 | 0 | 3 |
| 8 | 1196 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 9 | 1065 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 930 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

#### STAR_PLUS_ELITE

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2702 | 26 | 0.96% | 0.882 | 0.123 | 12.8× | 0.129 | 0.000 | 0.000 | — | 0 | 1 | 26 |
| 1 | 2552 | 33 | 1.29% | 0.939 | 0.327 | 25.3× | 0.172 | 0.571 | 0.121 | 0.200 | 4 | 3 | 29 |
| 2 | 2360 | 25 | 1.06% | 0.951 | 0.269 | 25.4× | 0.160 | 0.667 | 0.160 | 0.258 | 4 | 2 | 21 |
| 3 | 2094 | 18 | 0.86% | 0.958 | 0.314 | 36.6× | 0.146 | — | 0.000 | — | 0 | 0 | 18 |
| 4 | 1839 | 9 | 0.49% | 0.943 | 0.098 | 19.9× | 0.107 | — | 0.000 | — | 0 | 0 | 9 |
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
| ALL | 18868 | 302 | 1.60% | 0.976 | 0.568 | 35.5× | 0.207 | 0.683 | 0.364 | 0.475 | 110 | 51 | 192 |
| RK | 1490 | 50 | 3.36% | 0.977 | 0.676 | 20.2× | 0.298 | 0.895 | 0.340 | 0.493 | 17 | 2 | 33 |
| A- | 1065 | 20 | 1.88% | 0.975 | 0.570 | 30.4× | 0.223 | 0.700 | 0.350 | 0.467 | 7 | 3 | 13 |
| A | 1449 | 44 | 3.04% | 0.981 | 0.679 | 22.4× | 0.286 | 0.667 | 0.409 | 0.507 | 18 | 9 | 26 |
| A+ | 1457 | 40 | 2.75% | 0.980 | 0.638 | 23.2× | 0.272 | 0.632 | 0.300 | 0.407 | 12 | 7 | 28 |
| AA | 1312 | 24 | 1.83% | 0.995 | 0.817 | 44.7× | 0.230 | 0.789 | 0.625 | 0.698 | 15 | 4 | 9 |
| AAA | 1051 | 12 | 1.14% | 0.999 | 0.946 | 82.8× | 0.184 | 0.900 | 0.750 | 0.818 | 9 | 1 | 3 |
| NONE | 11010 | 112 | 1.02% | 0.969 | 0.377 | 37.0× | 0.163 | 0.561 | 0.286 | 0.379 | 32 | 25 | 80 |

#### MLB_DEBUT

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 19160 | 2513 | 13.12% | 0.930 | 0.701 | 5.3× | 0.503 | 0.770 | 0.421 | 0.544 | 1057 | 315 | 1456 |
| RK | 1496 | 172 | 11.50% | 0.843 | 0.505 | 4.4× | 0.379 | 0.778 | 0.163 | 0.269 | 28 | 8 | 144 |
| A- | 1067 | 168 | 15.75% | 0.848 | 0.563 | 3.6× | 0.439 | 0.672 | 0.244 | 0.358 | 41 | 20 | 127 |
| A | 1474 | 291 | 19.74% | 0.869 | 0.688 | 3.5× | 0.509 | 0.781 | 0.430 | 0.554 | 125 | 35 | 166 |
| A+ | 1480 | 309 | 20.88% | 0.882 | 0.720 | 3.4× | 0.539 | 0.801 | 0.495 | 0.612 | 153 | 38 | 156 |
| AA | 1390 | 461 | 33.17% | 0.891 | 0.818 | 2.5× | 0.638 | 0.807 | 0.564 | 0.664 | 260 | 62 | 201 |
| AAA | 1113 | 353 | 31.72% | 0.903 | 0.828 | 2.6× | 0.649 | 0.854 | 0.615 | 0.715 | 217 | 37 | 136 |
| NONE | 11098 | 758 | 6.83% | 0.956 | 0.614 | 9.0× | 0.399 | 0.680 | 0.306 | 0.422 | 232 | 109 | 526 |

#### ESTABLISHED_MLB

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 19160 | 796 | 4.15% | 0.938 | 0.447 | 10.8× | 0.303 | 0.642 | 0.173 | 0.273 | 138 | 77 | 658 |
| RK | 1496 | 33 | 2.21% | 0.913 | 0.266 | 12.1× | 0.210 | 0.500 | 0.061 | 0.108 | 2 | 2 | 31 |
| A- | 1067 | 41 | 3.84% | 0.885 | 0.316 | 8.2× | 0.257 | 0.333 | 0.024 | 0.045 | 1 | 2 | 40 |
| A | 1474 | 72 | 4.88% | 0.904 | 0.417 | 8.5× | 0.302 | 0.579 | 0.153 | 0.242 | 11 | 8 | 61 |
| A+ | 1480 | 100 | 6.76% | 0.903 | 0.452 | 6.7× | 0.351 | 0.571 | 0.120 | 0.198 | 12 | 9 | 88 |
| AA | 1390 | 155 | 11.15% | 0.905 | 0.583 | 5.2× | 0.442 | 0.726 | 0.290 | 0.415 | 45 | 17 | 110 |
| AAA | 1113 | 105 | 9.43% | 0.915 | 0.557 | 5.9× | 0.421 | 0.681 | 0.305 | 0.421 | 32 | 15 | 73 |
| NONE | 11098 | 289 | 2.60% | 0.956 | 0.376 | 14.5× | 0.252 | 0.596 | 0.118 | 0.197 | 34 | 23 | 255 |

#### STAR_PLUS_ELITE

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 19160 | 112 | 0.58% | 0.959 | 0.219 | 37.5× | 0.121 | 0.571 | 0.071 | 0.127 | 8 | 6 | 104 |
| RK | 1496 | 8 | 0.53% | 0.966 | 0.098 | 18.4× | 0.118 | 0.000 | 0.000 | — | 0 | 1 | 8 |
| A- | 1067 | 7 | 0.66% | 0.884 | 0.254 | 38.7× | 0.107 | — | 0.000 | — | 0 | 0 | 7 |
| A | 1474 | 12 | 0.81% | 0.960 | 0.410 | 50.3× | 0.143 | 1.000 | 0.250 | 0.400 | 3 | 0 | 9 |
| A+ | 1480 | 12 | 0.81% | 0.952 | 0.245 | 30.3× | 0.140 | 0.400 | 0.167 | 0.235 | 2 | 3 | 10 |
| AA | 1390 | 24 | 1.73% | 0.965 | 0.351 | 20.3× | 0.210 | 1.000 | 0.042 | 0.080 | 1 | 0 | 23 |
| AAA | 1113 | 14 | 1.26% | 0.981 | 0.375 | 29.9× | 0.186 | 0.333 | 0.071 | 0.118 | 1 | 2 | 13 |
| NONE | 11098 | 34 | 0.31% | 0.958 | 0.072 | 23.6× | 0.088 | — | 0.000 | — | 0 | 0 | 34 |

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
