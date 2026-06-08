# Held-out validation — v2.0b landmark

Reproducible evaluation of the v2.0b landmark stack against the **10%
val player slice** of the v1.17 seed=42 split — 3,543 players neither
the landmark hazards nor the joint XGBoost head trained on. Validation
universe: drafted players with `draft_year ≤ 2020` (plus IFAs),
realized window through 2026. Val rows after filters: 34,430.

## Stack

| Layer | Model | Trained on |
|---|---|---|
| Hazards (test, for val scoring) | `scratch/v20b_oof/hazards_full.pkl` | 90% of universe (val pids excluded). **Default HP** — no Optuna, no calibration. |
| Hazards (per-fold OOF) | `scratch/v20b_oof/fold[0-5]_hazards.pkl` | Each fold trained on the OTHER 5 folds (~75% of universe excluding val). Used to score its held-out fold's pids. |
| XGB head | `models/joint_xgb_v2.0b_oof.pkl` | OOF stacked CSV: 287,045 rows / 29,762 pids. Default HP via `fit_joint_xgb_v2.py` — `max_depth=6, lr=0.05, early_stop=25, best_iter=188`. Honest at XGB layer (val pids never in stacked CSV). |

Each landmark row: features computed `as_of S`, with `horizon_offset_k`
as an explicit feature column. Inference sets `k = step+1` per horizon
step instead of advancing yip — train and inference draw from the same
distribution.

## How this is clean at both layers

| Layer | Cleanliness |
|---|---|
| Hazards used to score val | Trained on 90% universe **excluding val pids** → val features are honest |
| XGB training rows | The OOF stacked CSV (287k rows) is the 90% universe scored leave-one-fold-out — no row was scored by a hazards model that trained on it. **Val pids are not in the stacked CSV at all.** |
| XGB val evaluation | Val features come from `hazards_full` (90% honest); val labels are realized 0/1. XGB never saw val pids. |

## Headline (ALL bucket, threshold = 0.60)

| Event | n | base% | AP | lift | AUC | spearman | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TOP_100_PROSPECT | 34,164 | 0.67% | **0.492** | 73.1× | 0.986 | 0.138 | 0.848 | 0.122 | 0.213 |
| MLB_DEBUT | 34,430 | 5.07% | **0.565** | 11.1× | 0.937 | 0.332 | 0.704 | 0.287 | 0.408 |
| ESTABLISHED_MLB | 34,430 | 1.12% | **0.353** | 31.6× | 0.976 | 0.173 | 0.800 | 0.021 | 0.041 |
| STAR_PLUS_ELITE | 34,430 | 0.36% | **0.187** | 52.3× | 0.973 | 0.098 | — | 0.000 | — |
| **weighted-AP** | | | **0.432** | | | | | | |

(MLB_DEBUT 2× weight, others 1×, per-event eligibility filters.)

## File map — [v2.0b_landmark/](v2.0b_landmark/)

| File | What it is |
|---|---|
| `bucket.csv` | Per `(event, bucket)` cell at `snap_offset=2`. Buckets: `ALL, R1, R2-R3, R4-R10, R10+, IFA`. |
| `walkforward.csv` | Per `(event, snap_offset)` cell, offsets 0..10. |
| `per_bucket_validation.csv` | 0.60-threshold confusion matrix + Spearman ρ per bucket. |
| `per_yip_validation.csv` | 0.60-threshold view per `(event, snap_offset)`. |
| `per_level_validation.csv` | 0.60-threshold view per `(event, current_level)` for `ALL, RK, A-, A, A+, AA, AAA, NONE`. |
| `thresholds_at_p60_per_bucket.csv`, `_per_yip.csv`, `_per_level.csv` | Per-cell minimum threshold achieving precision ≥ 0.60. |
| `<EVENT>_pct_slabs.csv`, `_cum_above_threshold.csv`, `_walkforward.csv` | Per-event slab and walkforward analysis. |
| `MLB_DEBUT_per_current_level.csv`, `_thresholds_at_p60.csv`, `_time_to_debut.csv` | MLB_DEBUT-specific tables. |
| `headline.json` | Machine-readable summary. |
| `report.txt` | Human-readable summary from `validate_full.py`. |

## Per-bucket (threshold = 0.60)


#### TOP_100_PROSPECT

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 34164 | 230 | 0.67% | 0.986 | 0.492 | 73.1× | 0.138 | 0.848 | 0.122 | 0.213 | 28 | 5 | 202 |
| R1 | 388 | 80 | 20.62% | 0.927 | 0.764 | 3.7× | 0.599 | 0.909 | 0.250 | 0.392 | 20 | 2 | 60 |
| R2-R3 | 707 | 62 | 8.77% | 0.891 | 0.451 | 5.1× | 0.384 | 0.714 | 0.081 | 0.145 | 5 | 2 | 57 |
| R4-R10 | 2484 | 20 | 0.81% | 0.942 | 0.223 | 27.8× | 0.137 | 1.000 | 0.050 | 0.095 | 1 | 0 | 19 |
| R10+ | 11788 | 9 | 0.08% | 0.949 | 0.291 | 380.7× | 0.043 | — | 0.000 | — | 0 | 0 | 9 |
| IFA | 18797 | 59 | 0.31% | 0.986 | 0.323 | 103.0× | 0.094 | 0.667 | 0.034 | 0.065 | 2 | 1 | 57 |

#### MLB_DEBUT

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 34430 | 1747 | 5.07% | 0.937 | 0.565 | 11.1× | 0.332 | 0.704 | 0.287 | 0.408 | 502 | 211 | 1245 |
| R1 | 491 | 240 | 48.88% | 0.893 | 0.897 | 1.8× | 0.680 | 0.837 | 0.683 | 0.752 | 164 | 32 | 76 |
| R2-R3 | 755 | 256 | 33.91% | 0.825 | 0.713 | 2.1× | 0.532 | 0.701 | 0.430 | 0.533 | 110 | 47 | 146 |
| R4-R10 | 2487 | 279 | 11.22% | 0.857 | 0.436 | 3.9× | 0.390 | 0.519 | 0.201 | 0.289 | 56 | 52 | 223 |
| R10+ | 11790 | 383 | 3.25% | 0.898 | 0.279 | 8.6× | 0.244 | 0.457 | 0.055 | 0.098 | 21 | 25 | 362 |
| IFA | 18907 | 589 | 3.12% | 0.942 | 0.528 | 16.9× | 0.266 | 0.733 | 0.256 | 0.380 | 151 | 55 | 438 |

#### ESTABLISHED_MLB

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 34430 | 385 | 1.12% | 0.976 | 0.353 | 31.6× | 0.173 | 0.800 | 0.021 | 0.041 | 8 | 2 | 377 |
| R1 | 491 | 85 | 17.31% | 0.870 | 0.497 | 2.9× | 0.485 | 0.500 | 0.012 | 0.023 | 1 | 1 | 84 |
| R2-R3 | 755 | 61 | 8.08% | 0.843 | 0.292 | 3.6× | 0.323 | 0.750 | 0.049 | 0.092 | 3 | 1 | 58 |
| R4-R10 | 2487 | 76 | 3.06% | 0.931 | 0.310 | 10.1× | 0.257 | 1.000 | 0.013 | 0.026 | 1 | 0 | 75 |
| R10+ | 11790 | 41 | 0.35% | 0.972 | 0.197 | 56.6× | 0.096 | 1.000 | 0.024 | 0.048 | 1 | 0 | 40 |
| IFA | 18907 | 122 | 0.65% | 0.981 | 0.365 | 56.5× | 0.133 | 1.000 | 0.016 | 0.032 | 2 | 0 | 120 |

#### STAR_PLUS_ELITE

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 34430 | 123 | 0.36% | 0.973 | 0.187 | 52.3× | 0.098 | — | 0.000 | — | 0 | 0 | 123 |
| R1 | 491 | 35 | 7.13% | 0.837 | 0.215 | 3.0× | 0.300 | — | 0.000 | — | 0 | 0 | 35 |
| R2-R3 | 755 | 11 | 1.46% | 0.915 | 0.298 | 20.4× | 0.172 | — | 0.000 | — | 0 | 0 | 11 |
| R4-R10 | 2487 | 21 | 0.84% | 0.954 | 0.250 | 29.6× | 0.144 | — | 0.000 | — | 0 | 0 | 21 |
| R10+ | 11790 | 18 | 0.15% | 0.929 | 0.177 | 116.0× | 0.058 | — | 0.000 | — | 0 | 0 | 18 |
| IFA | 18907 | 38 | 0.20% | 0.987 | 0.202 | 100.7× | 0.076 | — | 0.000 | — | 0 | 0 | 38 |

## Per-yip (threshold = 0.60)


#### TOP_100_PROSPECT

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3523 | 97 | 2.75% | 0.959 | 0.528 | 19.2× | 0.260 | 0.923 | 0.124 | 0.218 | 12 | 1 | 85 |
| 1 | 3481 | 67 | 1.92% | 0.962 | 0.492 | 25.5× | 0.220 | 0.750 | 0.134 | 0.228 | 9 | 3 | 58 |
| 2 | 3433 | 41 | 1.19% | 0.970 | 0.498 | 41.7× | 0.177 | 0.875 | 0.171 | 0.286 | 7 | 1 | 34 |
| 3 | 3346 | 17 | 0.51% | 0.972 | 0.358 | 70.4× | 0.116 | — | 0.000 | — | 0 | 0 | 17 |
| 4 | 3273 | 7 | 0.21% | 0.992 | 0.325 | 152.0× | 0.079 | — | 0.000 | — | 0 | 0 | 7 |
| 5 | 3204 | 1 | 0.03% | 1.000 | 1.000 | 3204.0× | 0.031 | — | 0.000 | — | 0 | 0 | 1 |
| 6 | 3170 | 0 | 0.00% | — | — | nan× | — | — | — | — | 0 | 0 | 0 |
| 7 | 2965 | 0 | 0.00% | — | — | nan× | — | — | — | — | 0 | 0 | 0 |
| 8 | 2767 | 0 | 0.00% | — | — | nan× | — | — | — | — | 0 | 0 | 0 |
| 9 | 2593 | 0 | 0.00% | — | — | nan× | — | — | — | — | 0 | 0 | 0 |
| 10 | 2409 | 0 | 0.00% | — | — | nan× | — | — | — | — | 0 | 0 | 0 |

#### MLB_DEBUT

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3543 | 404 | 11.40% | 0.886 | 0.572 | 5.0× | 0.425 | 0.746 | 0.210 | 0.328 | 85 | 29 | 319 |
| 1 | 3516 | 377 | 10.72% | 0.898 | 0.595 | 5.6× | 0.426 | 0.741 | 0.318 | 0.445 | 120 | 42 | 257 |
| 2 | 3481 | 342 | 9.82% | 0.916 | 0.637 | 6.5× | 0.429 | 0.667 | 0.351 | 0.460 | 120 | 60 | 222 |
| 3 | 3392 | 253 | 7.46% | 0.918 | 0.606 | 8.1× | 0.380 | 0.678 | 0.399 | 0.502 | 101 | 48 | 152 |
| 4 | 3309 | 170 | 5.14% | 0.912 | 0.548 | 10.7× | 0.315 | 0.710 | 0.288 | 0.410 | 49 | 20 | 121 |
| 5 | 3229 | 90 | 2.79% | 0.913 | 0.393 | 14.1× | 0.236 | 0.655 | 0.211 | 0.319 | 19 | 10 | 71 |
| 6 | 3187 | 55 | 1.73% | 0.935 | 0.418 | 24.2× | 0.196 | 0.800 | 0.145 | 0.246 | 8 | 2 | 47 |
| 7 | 2976 | 29 | 0.97% | 0.945 | 0.256 | 26.2× | 0.152 | — | 0.000 | — | 0 | 0 | 29 |
| 8 | 2778 | 16 | 0.58% | 0.943 | 0.175 | 30.4× | 0.116 | — | 0.000 | — | 0 | 0 | 16 |
| 9 | 2602 | 8 | 0.31% | 0.920 | 0.306 | 99.7× | 0.080 | — | 0.000 | — | 0 | 0 | 8 |
| 10 | 2417 | 3 | 0.12% | 0.882 | 0.345 | 278.0× | 0.047 | — | 0.000 | — | 0 | 0 | 3 |

#### ESTABLISHED_MLB

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3543 | 120 | 3.39% | 0.928 | 0.355 | 10.5× | 0.268 | 1.000 | 0.008 | 0.017 | 1 | 0 | 119 |
| 1 | 3516 | 102 | 2.90% | 0.940 | 0.384 | 13.2× | 0.256 | 0.750 | 0.029 | 0.057 | 3 | 1 | 99 |
| 2 | 3481 | 83 | 2.38% | 0.964 | 0.426 | 17.9× | 0.245 | 0.800 | 0.048 | 0.091 | 4 | 1 | 79 |
| 3 | 3392 | 45 | 1.33% | 0.972 | 0.300 | 22.6× | 0.187 | — | 0.000 | — | 0 | 0 | 45 |
| 4 | 3309 | 24 | 0.73% | 0.983 | 0.372 | 51.3× | 0.142 | — | 0.000 | — | 0 | 0 | 24 |
| 5 | 3229 | 6 | 0.19% | 0.983 | 0.098 | 52.7× | 0.072 | — | 0.000 | — | 0 | 0 | 6 |
| 6 | 3187 | 3 | 0.09% | 0.981 | 0.156 | 166.2× | 0.051 | — | 0.000 | — | 0 | 0 | 3 |
| 7 | 2976 | 1 | 0.03% | 0.999 | 0.333 | 992.0× | 0.032 | — | 0.000 | — | 0 | 0 | 1 |
| 8 | 2778 | 1 | 0.04% | 0.976 | 0.015 | 40.9× | 0.031 | — | 0.000 | — | 0 | 0 | 1 |
| 9 | 2602 | 0 | 0.00% | — | — | nan× | — | — | — | — | 0 | 0 | 0 |
| 10 | 2417 | 0 | 0.00% | — | — | nan× | — | — | — | — | 0 | 0 | 0 |

#### STAR_PLUS_ELITE

| yip | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3543 | 41 | 1.16% | 0.925 | 0.120 | 10.3× | 0.157 | — | 0.000 | — | 0 | 0 | 41 |
| 1 | 3516 | 32 | 0.91% | 0.955 | 0.246 | 27.1× | 0.150 | — | 0.000 | — | 0 | 0 | 32 |
| 2 | 3481 | 27 | 0.78% | 0.965 | 0.394 | 50.8× | 0.141 | — | 0.000 | — | 0 | 0 | 27 |
| 3 | 3392 | 13 | 0.38% | 0.976 | 0.197 | 51.5× | 0.102 | — | 0.000 | — | 0 | 0 | 13 |
| 4 | 3309 | 6 | 0.18% | 0.968 | 0.107 | 58.9× | 0.069 | — | 0.000 | — | 0 | 0 | 6 |
| 5 | 3229 | 1 | 0.03% | 0.924 | 0.004 | 13.1× | 0.026 | — | 0.000 | — | 0 | 0 | 1 |
| 6 | 3187 | 1 | 0.03% | 0.979 | 0.014 | 46.2× | 0.029 | — | 0.000 | — | 0 | 0 | 1 |
| 7 | 2976 | 1 | 0.03% | 0.962 | 0.009 | 26.3× | 0.029 | — | 0.000 | — | 0 | 0 | 1 |
| 8 | 2778 | 1 | 0.04% | 0.989 | 0.031 | 86.8× | 0.032 | — | 0.000 | — | 0 | 0 | 1 |
| 9 | 2602 | 0 | 0.00% | — | — | nan× | — | — | — | — | 0 | 0 | 0 |
| 10 | 2417 | 0 | 0.00% | — | — | nan× | — | — | — | — | 0 | 0 | 0 |

## Per-level (threshold = 0.60)


#### TOP_100_PROSPECT

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 34164 | 230 | 0.67% | 0.986 | 0.492 | 73.1× | 0.138 | 0.848 | 0.122 | 0.213 | 28 | 5 | 202 |
| RK | 3928 | 65 | 1.65% | 0.963 | 0.503 | 30.4× | 0.205 | 1.000 | 0.123 | 0.219 | 8 | 0 | 57 |
| A- | 987 | 19 | 1.93% | 0.964 | 0.477 | 24.8× | 0.221 | 0.500 | 0.053 | 0.095 | 1 | 1 | 18 |
| A | 1503 | 45 | 2.99% | 0.954 | 0.506 | 16.9× | 0.268 | 0.800 | 0.178 | 0.291 | 8 | 2 | 37 |
| A+ | 1510 | 29 | 1.92% | 0.974 | 0.453 | 23.6× | 0.225 | 0.750 | 0.103 | 0.182 | 3 | 1 | 26 |
| AA | 1390 | 33 | 2.37% | 0.983 | 0.639 | 26.9× | 0.255 | 0.857 | 0.182 | 0.300 | 6 | 1 | 27 |
| AAA | 1809 | 12 | 0.66% | 0.989 | 0.652 | 98.3× | 0.137 | 1.000 | 0.167 | 0.286 | 2 | 0 | 10 |
| NONE | 23037 | 27 | 0.12% | 0.989 | 0.340 | 290.1× | 0.058 | — | 0.000 | — | 0 | 0 | 27 |

#### MLB_DEBUT

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 34430 | 1747 | 5.07% | 0.937 | 0.565 | 11.1× | 0.332 | 0.704 | 0.287 | 0.408 | 502 | 211 | 1245 |
| RK | 3932 | 242 | 6.15% | 0.901 | 0.539 | 8.8× | 0.334 | 0.816 | 0.165 | 0.275 | 40 | 9 | 202 |
| A- | 989 | 126 | 12.74% | 0.835 | 0.487 | 3.8× | 0.387 | 0.714 | 0.159 | 0.260 | 20 | 8 | 106 |
| A | 1528 | 256 | 16.75% | 0.858 | 0.615 | 3.7× | 0.464 | 0.700 | 0.328 | 0.447 | 84 | 36 | 172 |
| A+ | 1541 | 260 | 16.87% | 0.821 | 0.586 | 3.5× | 0.416 | 0.703 | 0.346 | 0.464 | 90 | 38 | 170 |
| AA | 1478 | 376 | 25.44% | 0.839 | 0.710 | 2.8× | 0.512 | 0.720 | 0.444 | 0.549 | 167 | 65 | 209 |
| AAA | 1865 | 296 | 15.87% | 0.834 | 0.538 | 3.4× | 0.423 | 0.627 | 0.267 | 0.374 | 79 | 47 | 217 |
| NONE | 23097 | 191 | 0.83% | 0.928 | 0.337 | 40.8× | 0.134 | 0.733 | 0.115 | 0.199 | 22 | 8 | 169 |

#### ESTABLISHED_MLB

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 34430 | 385 | 1.12% | 0.976 | 0.353 | 31.6× | 0.173 | 0.800 | 0.021 | 0.041 | 8 | 2 | 377 |
| RK | 3932 | 63 | 1.60% | 0.947 | 0.318 | 19.8× | 0.194 | — | 0.000 | — | 0 | 0 | 63 |
| A- | 989 | 24 | 2.43% | 0.930 | 0.265 | 10.9× | 0.229 | — | 0.000 | — | 0 | 0 | 24 |
| A | 1528 | 69 | 4.52% | 0.928 | 0.369 | 8.2× | 0.308 | 1.000 | 0.014 | 0.029 | 1 | 0 | 68 |
| A+ | 1541 | 71 | 4.61% | 0.919 | 0.354 | 7.7× | 0.304 | 1.000 | 0.028 | 0.055 | 2 | 0 | 69 |
| AA | 1478 | 99 | 6.70% | 0.901 | 0.440 | 6.6× | 0.347 | 0.667 | 0.040 | 0.076 | 4 | 2 | 95 |
| AAA | 1865 | 49 | 2.63% | 0.927 | 0.367 | 14.0× | 0.237 | 1.000 | 0.020 | 0.040 | 1 | 0 | 48 |
| NONE | 23097 | 10 | 0.04% | 0.999 | 0.236 | 545.4× | 0.036 | — | 0.000 | — | 0 | 0 | 10 |

#### STAR_PLUS_ELITE

| level | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 34430 | 123 | 0.36% | 0.973 | 0.187 | 52.3× | 0.098 | — | 0.000 | — | 0 | 0 | 123 |
| RK | 3932 | 23 | 0.58% | 0.953 | 0.156 | 26.6× | 0.120 | — | 0.000 | — | 0 | 0 | 23 |
| A- | 989 | 5 | 0.51% | 0.933 | 0.069 | 13.7× | 0.106 | — | 0.000 | — | 0 | 0 | 5 |
| A | 1528 | 22 | 1.44% | 0.929 | 0.186 | 12.9× | 0.177 | — | 0.000 | — | 0 | 0 | 22 |
| A+ | 1541 | 21 | 1.36% | 0.929 | 0.188 | 13.8× | 0.172 | — | 0.000 | — | 0 | 0 | 21 |
| AA | 1478 | 32 | 2.17% | 0.911 | 0.308 | 14.2× | 0.207 | — | 0.000 | — | 0 | 0 | 32 |
| AAA | 1865 | 14 | 0.75% | 0.874 | 0.271 | 36.1× | 0.112 | — | 0.000 | — | 0 | 0 | 14 |
| NONE | 23097 | 6 | 0.03% | 0.990 | 0.033 | 128.4× | 0.027 | — | 0.000 | — | 0 | 0 | 6 |

## Threshold @ precision ≥ 0.60 (MLB_DEBUT per yip)

| yip | threshold | n_above | TP | precision | recall | n_total | n_pos_total |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.415 | 288 | 173 | 0.601 | 0.428 | 3543 | 404 |
| 1 | 0.458 | 288 | 173 | 0.601 | 0.459 | 3516 | 377 |
| 2 | 0.453 | 305 | 183 | 0.600 | 0.535 | 3481 | 342 |
| 3 | 0.446 | 218 | 131 | 0.601 | 0.518 | 3392 | 253 |
| 4 | 0.460 | 120 | 72 | 0.600 | 0.424 | 3309 | 170 |
| 5 | 0.571 | 35 | 21 | 0.600 | 0.233 | 3229 | 90 |
| 6 | 0.369 | 28 | 17 | 0.607 | 0.309 | 3187 | 55 |
| 7 | 0.447 | 6 | 4 | 0.667 | 0.138 | 2976 | 29 |
| 8 | — | 0 | 0 | — | — | 2778 | 16 |
| 9 | 0.180 | 5 | 3 | 0.600 | 0.375 | 2602 | 8 |
| 10 | 0.322 | 1 | 1 | 1.000 | 0.333 | 2417 | 3 |

## Statistics glossary

### Structural

| Column | Meaning |
|---|---|
| `event` | One of `TOP_100_PROSPECT, MLB_DEBUT, ESTABLISHED_MLB, STAR_PLUS_ELITE, ELITE, STAR`. `STAR_PLUS_ELITE` is the union: `1 − (1 − p_STAR)(1 − p_ELITE)`. |
| `bucket` | Draft-pedigree. `R1` = first round, `R2-R3`, `R4-R10`, `R10+` = rounds 11+, `IFA` = international free agent. |
| `snap_offset` | Years since entry. Also called "yip". |
| `n` | Number of eligible players. **Eligibility**: event hadn't fired by snap year AND player has enough forward observation window. |
| `pos` | Eligible players who realized the event in `(snap_year, 2026]`. |
| `base_rate` | `pos / n`. Random-guess hit rate. |

### Discrimination

| Metric | Meaning |
|---|---|
| `auc` | Area under ROC curve. Insensitive to class imbalance. |
| **`ap`** | **Average Precision = AU-PR.** The headline rare-event metric. |
| **`ap_lift`** | `ap / base_rate`. How many × random the ranking is. |
| **`spearman_rho`** | Rank correlation between predicted score and realized 0/1. Threshold-free. `spearman_p` is its p-value. |

### Calibration

| Metric | Meaning |
|---|---|
| `brier` | MSE of predictions vs realized. Lower is better. |
| **`brier_skill`** | `1 − brier / brier_baseline` (predict base rate for everyone). Positive = better than baseline. |
| **`ece`** | Expected Calibration Error. Bins predictions into 10 buckets, weighted abs difference vs realized rate. Low (< 0.05) = well calibrated. |
| `spiegelhalter_p` | Calibration H0 p-value. |

### Threshold view

| Metric | Meaning |
|---|---|
| `threshold` | XGB probability cutoff. Headline tables use **0.60** (production buy filter). |
| `tp` / `fp` / `tn` / `fn` | At threshold. |
| `precision` | `tp / (tp + fp)` — of the model's picks, fraction that hit. |
| `recall` | `tp / (tp + fn)` — of all hitters, fraction the model picked. |
| `f1` | Harmonic mean of precision and recall. |

## Reproducing

```bash
# 1. Build panel + OOF folds (one-time, ~6 hours, fold longs cached on disk)
python -m scripts_v17.train.run_v2_0b_oof

# 2. Train hazards_full on 90% universe (default HP) + score val
python -m scripts_v17.train.finalize_v2_0b_oof

# 3. Train joint XGB on OOF stacked + val (default HP, no Optuna)
python -m scripts_v17.train.fit_joint_xgb_v2 \
    --fit results/training/v2.0b_oof_stacked_long.csv \
    --val results/training/v2.0b_oof_val_long.csv \
    --db prospects_snapshot.db \
    --out models/joint_xgb_v2.0b_oof.pkl

# 4. Run validate_full.py to produce the per-event detailed tables
python scripts_v17/validate/validate_full.py \
    --long results/training/v2.0b_oof_val_long.csv \
    --xgb-model models/joint_xgb_v2.0b_oof.pkl \
    --time-to-debut-model models/time_to_debut_v1.18_prod.pkl \
    --target-precision 0.60 \
    --out-prefix v20b_clean

# 5. Regenerate compact (per_bucket/per_yip/per_level) + threshold-at-p60 tables
python -m scripts_v17.validate.regen_eval_v2_0b_honest --threshold 0.60
python -m scripts_v17.validate.regen_full_eval_v2_0b
```
