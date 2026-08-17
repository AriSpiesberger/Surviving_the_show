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
| TOP_100_PROSPECT | 15139 | 1.63% | **0.428** | 26.2× | 0.971 | 0.207 | 0.551 | 0.348 | 0.427 |
| MLB_DEBUT | 15326 | 14.73% | **0.654** | 4.4× | 0.909 | 0.503 | 0.680 | 0.440 | 0.534 |
| ESTABLISHED_MLB | 15326 | 4.59% | **0.379** | 8.3× | 0.924 | 0.307 | 0.563 | 0.170 | 0.262 |
| STAR_PLUS_ELITE | 15326 | 0.74% | **0.086** | 11.6× | 0.925 | 0.127 | 0.250 | 0.009 | 0.017 |
| **weighted-AP** | | | **0.440** | | | | | | |

(MLB_DEBUT 2× weight, others 1×, per-event eligibility filters. Scores =
`xp_<event>_h6` vs realized-within-6y, on rows resolved at h=6.)

## Per-horizon trajectory (h=1..10, resolved at each h)

#### TOP_100_PROSPECT

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 18990 | 104 | 0.55% | 0.987 | 0.398 | 72.6× | 0.0041 | 1.21 |
| 2 | 18413 | 185 | 1.00% | 0.980 | 0.464 | 46.2× | 0.0070 | 1.19 |
| 3 | 17753 | 232 | 1.31% | 0.975 | 0.465 | 35.6× | 0.0092 | 1.19 |
| 4 | 16992 | 253 | 1.49% | 0.974 | 0.463 | 31.1× | 0.0105 | 1.21 |
| 5 | 16129 | 256 | 1.59% | 0.973 | 0.445 | 28.0× | 0.0115 | 1.24 |
| 6 | 15139 | 247 | 1.63% | 0.971 | 0.428 | 26.2× | 0.0120 | 1.27 |
| 7 | 14055 | 236 | 1.68% | 0.970 | 0.424 | 25.3× | 0.0124 | 1.26 |
| 8 | 12990 | 221 | 1.70% | 0.969 | 0.420 | 24.7× | 0.0127 | 1.26 |
| 9 | 11920 | 207 | 1.74% | 0.966 | 0.404 | 23.3× | 0.0132 | 1.27 |
| 10 | 10859 | 193 | 1.78% | 0.964 | 0.397 | 22.3× | 0.0137 | 1.28 |

#### MLB_DEBUT

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 19223 | 611 | 3.18% | 0.943 | 0.444 | 14.0× | 0.0228 | 1.10 |
| 2 | 18643 | 1198 | 6.43% | 0.931 | 0.561 | 8.7× | 0.0400 | 1.06 |
| 3 | 17977 | 1717 | 9.55% | 0.923 | 0.620 | 6.5× | 0.0546 | 1.03 |
| 4 | 17207 | 2069 | 12.02% | 0.917 | 0.650 | 5.4× | 0.0654 | 1.02 |
| 5 | 16331 | 2238 | 13.70% | 0.912 | 0.654 | 4.8× | 0.0734 | 1.00 |
| 6 | 15326 | 2257 | 14.73% | 0.909 | 0.654 | 4.4× | 0.0783 | 1.01 |
| 7 | 14232 | 2202 | 15.47% | 0.906 | 0.650 | 4.2× | 0.0822 | 1.00 |
| 8 | 13160 | 2098 | 15.94% | 0.902 | 0.639 | 4.0× | 0.0858 | 1.00 |
| 9 | 12081 | 1973 | 16.33% | 0.898 | 0.630 | 3.9× | 0.0885 | 1.02 |
| 10 | 11006 | 1832 | 16.65% | 0.892 | 0.622 | 3.7× | 0.0915 | 1.02 |

#### ESTABLISHED_MLB

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 19223 | 5 | 0.03% | 0.956 | 0.021 | 82.6× | 0.0003 | 1.88 |
| 2 | 18643 | 77 | 0.41% | 0.957 | 0.196 | 47.5× | 0.0036 | 1.00 |
| 3 | 17977 | 232 | 1.29% | 0.944 | 0.260 | 20.2× | 0.0108 | 0.98 |
| 4 | 17207 | 418 | 2.43% | 0.934 | 0.325 | 13.4× | 0.0192 | 0.98 |
| 5 | 16331 | 581 | 3.56% | 0.930 | 0.359 | 10.1× | 0.0270 | 0.99 |
| 6 | 15326 | 704 | 4.59% | 0.924 | 0.379 | 8.3× | 0.0341 | 1.00 |
| 7 | 14232 | 780 | 5.48% | 0.919 | 0.391 | 7.1× | 0.0401 | 0.99 |
| 8 | 13160 | 814 | 6.19% | 0.915 | 0.400 | 6.5× | 0.0447 | 1.00 |
| 9 | 12081 | 812 | 6.72% | 0.911 | 0.404 | 6.0× | 0.0483 | 1.01 |
| 10 | 11006 | 776 | 7.05% | 0.905 | 0.398 | 5.6× | 0.0509 | 1.00 |

#### STAR_PLUS_ELITE

| h | n | pos | base% | AUC | AP | AP_lift | Brier | calib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 19223 | 4 | 0.02% | 0.938 | 0.003 | 13.9× | 0.0002 | 0.80 |
| 2 | 18643 | 16 | 0.09% | 0.962 | 0.021 | 24.2× | 0.0009 | 0.85 |
| 3 | 17977 | 35 | 0.19% | 0.939 | 0.036 | 18.3× | 0.0020 | 1.02 |
| 4 | 17207 | 60 | 0.35% | 0.930 | 0.055 | 15.9× | 0.0035 | 1.10 |
| 5 | 16331 | 90 | 0.55% | 0.928 | 0.074 | 13.4× | 0.0055 | 1.14 |
| 6 | 15326 | 114 | 0.74% | 0.925 | 0.086 | 11.6× | 0.0073 | 1.16 |
| 7 | 14232 | 134 | 0.94% | 0.926 | 0.101 | 10.8× | 0.0091 | 1.15 |
| 8 | 13160 | 143 | 1.09% | 0.921 | 0.108 | 9.9× | 0.0104 | 1.19 |
| 9 | 12081 | 147 | 1.22% | 0.916 | 0.111 | 9.1× | 0.0117 | 1.24 |
| 10 | 11006 | 145 | 1.32% | 0.912 | 0.115 | 8.7× | 0.0126 | 1.20 |

## Per-bucket (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 15139 | 247 | 1.63% | 0.971 | 0.428 | 26.2× | 0.207 | 0.551 | 0.348 | 0.427 | 86 | 70 | 161 |
| R1 | 477 | 98 | 20.55% | 0.911 | 0.667 | 3.2× | 0.575 | 0.641 | 0.673 | 0.657 | 66 | 37 | 32 |
| R2-R3 | 967 | 58 | 6.00% | 0.873 | 0.249 | 4.1× | 0.307 | 0.281 | 0.155 | 0.200 | 9 | 23 | 49 |
| R4-R10 | 3539 | 58 | 1.64% | 0.939 | 0.283 | 17.3× | 0.193 | 0.727 | 0.138 | 0.232 | 8 | 3 | 50 |
| R10+ | 10156 | 33 | 0.32% | 0.960 | 0.162 | 49.7× | 0.091 | 0.300 | 0.091 | 0.140 | 3 | 7 | 30 |

#### MLB_DEBUT

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 15326 | 2257 | 14.73% | 0.909 | 0.654 | 4.4× | 0.503 | 0.680 | 0.440 | 0.534 | 993 | 467 | 1264 |
| R1 | 596 | 322 | 54.03% | 0.860 | 0.865 | 1.6× | 0.622 | 0.757 | 0.863 | 0.807 | 278 | 89 | 44 |
| R2-R3 | 987 | 354 | 35.87% | 0.859 | 0.774 | 2.2× | 0.596 | 0.707 | 0.655 | 0.680 | 232 | 96 | 122 |
| R4-R10 | 3556 | 635 | 17.86% | 0.878 | 0.621 | 3.5× | 0.502 | 0.643 | 0.417 | 0.506 | 265 | 147 | 370 |
| R10+ | 10187 | 946 | 9.29% | 0.904 | 0.505 | 5.4× | 0.406 | 0.618 | 0.230 | 0.336 | 218 | 135 | 728 |

#### ESTABLISHED_MLB

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 15326 | 704 | 4.59% | 0.924 | 0.379 | 8.3× | 0.307 | 0.563 | 0.170 | 0.262 | 120 | 93 | 584 |
| R1 | 596 | 177 | 29.70% | 0.828 | 0.617 | 2.1× | 0.519 | 0.640 | 0.401 | 0.493 | 71 | 40 | 106 |
| R2-R3 | 987 | 113 | 11.45% | 0.834 | 0.340 | 3.0× | 0.368 | 0.472 | 0.221 | 0.301 | 25 | 28 | 88 |
| R4-R10 | 3556 | 196 | 5.51% | 0.873 | 0.281 | 5.1× | 0.295 | 0.485 | 0.082 | 0.140 | 16 | 17 | 180 |
| R10+ | 10187 | 218 | 2.14% | 0.929 | 0.247 | 11.6× | 0.215 | 0.500 | 0.037 | 0.068 | 8 | 8 | 210 |

#### STAR_PLUS_ELITE

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 15326 | 114 | 0.74% | 0.925 | 0.086 | 11.6× | 0.127 | 0.250 | 0.009 | 0.017 | 1 | 3 | 113 |
| R1 | 596 | 36 | 6.04% | 0.737 | 0.151 | 2.5× | 0.196 | 0.250 | 0.028 | 0.050 | 1 | 3 | 35 |
| R2-R3 | 987 | 21 | 2.13% | 0.843 | 0.078 | 3.7× | 0.172 | — | 0.000 | — | 0 | 0 | 21 |
| R4-R10 | 3556 | 30 | 0.84% | 0.875 | 0.068 | 8.0× | 0.119 | — | 0.000 | — | 0 | 0 | 30 |
| R10+ | 10187 | 27 | 0.27% | 0.948 | 0.044 | 16.5× | 0.080 | — | 0.000 | — | 0 | 0 | 27 |

## Per-yip (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2220 | 113 | 5.09% | 0.930 | 0.491 | 9.6× | 0.328 | 0.667 | 0.354 | 0.462 | 40 | 20 | 73 |
| 1 | 2051 | 71 | 3.46% | 0.918 | 0.389 | 11.2× | 0.265 | 0.467 | 0.394 | 0.427 | 28 | 32 | 43 |
| 2 | 1864 | 40 | 2.15% | 0.955 | 0.405 | 18.9× | 0.228 | 0.407 | 0.275 | 0.328 | 11 | 16 | 29 |
| 3 | 1638 | 19 | 1.16% | 0.985 | 0.558 | 48.1× | 0.180 | 0.857 | 0.316 | 0.462 | 6 | 1 | 13 |
| 4 | 1429 | 4 | 0.28% | 0.994 | 0.323 | 115.5× | 0.090 | 0.500 | 0.250 | 0.333 | 1 | 1 | 3 |
| 5 | 1265 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 6 | 1114 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 7 | 1017 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 8 | 932 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 9 | 854 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 755 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

#### MLB_DEBUT

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2222 | 522 | 23.49% | 0.838 | 0.597 | 2.5× | 0.496 | 0.636 | 0.398 | 0.490 | 208 | 119 | 314 |
| 1 | 2082 | 527 | 25.31% | 0.859 | 0.690 | 2.7× | 0.541 | 0.689 | 0.516 | 0.590 | 272 | 123 | 255 |
| 2 | 1907 | 467 | 24.49% | 0.882 | 0.728 | 3.0× | 0.570 | 0.718 | 0.525 | 0.606 | 245 | 96 | 222 |
| 3 | 1672 | 331 | 19.80% | 0.885 | 0.690 | 3.5× | 0.532 | 0.725 | 0.453 | 0.558 | 150 | 57 | 181 |
| 4 | 1458 | 204 | 13.99% | 0.897 | 0.630 | 4.5× | 0.477 | 0.678 | 0.402 | 0.505 | 82 | 39 | 122 |
| 5 | 1277 | 107 | 8.38% | 0.907 | 0.509 | 6.1× | 0.391 | 0.600 | 0.252 | 0.355 | 27 | 18 | 80 |
| 6 | 1125 | 55 | 4.89% | 0.913 | 0.335 | 6.9× | 0.309 | 0.391 | 0.164 | 0.231 | 9 | 14 | 46 |
| 7 | 1025 | 25 | 2.44% | 0.916 | 0.341 | 14.0× | 0.222 | 0.000 | 0.000 | — | 0 | 1 | 25 |
| 8 | 938 | 11 | 1.17% | 0.947 | 0.252 | 21.5× | 0.167 | — | 0.000 | — | 0 | 0 | 11 |
| 9 | 860 | 5 | 0.58% | 0.943 | 0.164 | 28.1× | 0.117 | — | 0.000 | — | 0 | 0 | 5 |
| 10 | 760 | 3 | 0.39% | 0.911 | 0.059 | 15.1× | 0.089 | — | 0.000 | — | 0 | 0 | 3 |

#### ESTABLISHED_MLB

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2222 | 160 | 7.20% | 0.869 | 0.359 | 5.0× | 0.330 | 0.727 | 0.050 | 0.094 | 8 | 3 | 152 |
| 1 | 2082 | 180 | 8.65% | 0.886 | 0.421 | 4.9× | 0.376 | 0.562 | 0.250 | 0.346 | 45 | 35 | 135 |
| 2 | 1907 | 163 | 8.55% | 0.904 | 0.451 | 5.3× | 0.391 | 0.541 | 0.245 | 0.338 | 40 | 34 | 123 |
| 3 | 1672 | 102 | 6.10% | 0.895 | 0.380 | 6.2× | 0.327 | 0.571 | 0.157 | 0.246 | 16 | 12 | 86 |
| 4 | 1458 | 62 | 4.25% | 0.914 | 0.360 | 8.5× | 0.289 | 0.611 | 0.177 | 0.275 | 11 | 7 | 51 |
| 5 | 1277 | 24 | 1.88% | 0.889 | 0.104 | 5.5× | 0.183 | 0.000 | 0.000 | — | 0 | 2 | 24 |
| 6 | 1125 | 10 | 0.89% | 0.878 | 0.108 | 12.1× | 0.123 | — | 0.000 | — | 0 | 0 | 10 |
| 7 | 1025 | 3 | 0.29% | 0.842 | 0.048 | 16.5× | 0.064 | — | 0.000 | — | 0 | 0 | 3 |
| 8 | 938 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 9 | 860 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 760 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

#### STAR_PLUS_ELITE

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2222 | 29 | 1.31% | 0.840 | 0.085 | 6.5× | 0.134 | — | 0.000 | — | 0 | 0 | 29 |
| 1 | 2082 | 35 | 1.68% | 0.882 | 0.114 | 6.8× | 0.170 | 0.250 | 0.029 | 0.051 | 1 | 3 | 34 |
| 2 | 1907 | 28 | 1.47% | 0.911 | 0.097 | 6.6× | 0.171 | — | 0.000 | — | 0 | 0 | 28 |
| 3 | 1672 | 14 | 0.84% | 0.874 | 0.077 | 9.2× | 0.118 | — | 0.000 | — | 0 | 0 | 14 |
| 4 | 1458 | 8 | 0.55% | 0.853 | 0.109 | 19.8× | 0.090 | — | 0.000 | — | 0 | 0 | 8 |
| 5 | 1277 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 6 | 1125 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 7 | 1025 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 8 | 938 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 9 | 860 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |
| 10 | 760 | 0 | 0.00% | — | — | — | — | — | — | — | 0 | 0 | 0 |

## Per-level (h=6, threshold = 0.60)

#### TOP_100_PROSPECT

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 15139 | 247 | 1.63% | 0.971 | 0.428 | 26.2× | 0.207 | 0.551 | 0.348 | 0.427 | 86 | 70 | 161 |
| A | 1428 | 42 | 2.94% | 0.974 | 0.587 | 20.0× | 0.278 | 0.581 | 0.429 | 0.493 | 18 | 13 | 24 |
| A+ | 1464 | 32 | 2.19% | 0.978 | 0.595 | 27.2× | 0.242 | 0.700 | 0.438 | 0.538 | 14 | 6 | 18 |
| AA | 1299 | 23 | 1.77% | 0.985 | 0.545 | 30.8× | 0.222 | 0.583 | 0.609 | 0.596 | 14 | 10 | 9 |
| AAA | 975 | 6 | 0.62% | 0.993 | 0.546 | 88.8× | 0.134 | 0.571 | 0.667 | 0.615 | 4 | 3 | 2 |
| NONE | 9973 | 144 | 1.44% | 0.964 | 0.335 | 23.2× | 0.192 | 0.486 | 0.250 | 0.330 | 36 | 38 | 108 |

#### MLB_DEBUT

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 15326 | 2257 | 14.73% | 0.909 | 0.654 | 4.4× | 0.503 | 0.680 | 0.440 | 0.534 | 993 | 467 | 1264 |
| A | 1441 | 257 | 17.83% | 0.889 | 0.679 | 3.8× | 0.515 | 0.686 | 0.510 | 0.585 | 131 | 60 | 126 |
| A+ | 1479 | 292 | 19.74% | 0.874 | 0.707 | 3.6× | 0.515 | 0.746 | 0.514 | 0.609 | 150 | 51 | 142 |
| AA | 1354 | 407 | 30.06% | 0.879 | 0.795 | 2.6× | 0.602 | 0.761 | 0.634 | 0.692 | 258 | 81 | 149 |
| AAA | 1014 | 290 | 28.60% | 0.834 | 0.709 | 2.5× | 0.523 | 0.724 | 0.497 | 0.589 | 144 | 55 | 146 |
| NONE | 10038 | 1011 | 10.07% | 0.921 | 0.525 | 5.2× | 0.439 | 0.585 | 0.307 | 0.402 | 310 | 220 | 701 |

#### ESTABLISHED_MLB

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 15326 | 704 | 4.59% | 0.924 | 0.379 | 8.3× | 0.307 | 0.563 | 0.170 | 0.262 | 120 | 93 | 584 |
| A | 1441 | 73 | 5.07% | 0.930 | 0.456 | 9.0× | 0.327 | 0.667 | 0.110 | 0.188 | 8 | 4 | 65 |
| A+ | 1479 | 83 | 5.61% | 0.928 | 0.437 | 7.8× | 0.342 | 0.600 | 0.108 | 0.184 | 9 | 6 | 74 |
| AA | 1354 | 136 | 10.04% | 0.914 | 0.561 | 5.6× | 0.431 | 0.667 | 0.382 | 0.486 | 52 | 26 | 84 |
| AAA | 1014 | 80 | 7.89% | 0.897 | 0.485 | 6.2× | 0.370 | 0.711 | 0.338 | 0.458 | 27 | 11 | 53 |
| NONE | 10038 | 332 | 3.31% | 0.921 | 0.258 | 7.8× | 0.261 | 0.343 | 0.072 | 0.119 | 24 | 46 | 308 |

#### STAR_PLUS_ELITE

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 15326 | 114 | 0.74% | 0.925 | 0.086 | 11.6× | 0.127 | 0.250 | 0.009 | 0.017 | 1 | 3 | 113 |
| A | 1441 | 9 | 0.62% | 0.950 | 0.243 | 38.9× | 0.123 | — | 0.000 | — | 0 | 0 | 9 |
| A+ | 1479 | 20 | 1.35% | 0.929 | 0.159 | 11.8× | 0.172 | — | 0.000 | — | 0 | 0 | 20 |
| AA | 1354 | 21 | 1.55% | 0.909 | 0.098 | 6.3× | 0.175 | 0.000 | 0.000 | — | 0 | 1 | 21 |
| AAA | 1014 | 10 | 0.99% | 0.920 | 0.182 | 18.4× | 0.144 | — | 0.000 | — | 0 | 0 | 10 |
| NONE | 10038 | 54 | 0.54% | 0.922 | 0.069 | 12.9× | 0.107 | 0.333 | 0.019 | 0.035 | 1 | 2 | 53 |

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
