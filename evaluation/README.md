# Held-out validation packets

Reproducible evaluation of the prospect-card models against the **10% val
player slice** of the v1.17 seed=42 split — 3,543 players neither the
landmark hazards nor the joint XGBoost head ever trained on. Validation
universe: drafted players with `draft_year ≤ 2020` (plus IFAs), realized
window through 2026. Val rows after filters: 34,430.

## Production stack

The v2.0b landmark architecture is the sole production model. Hazards
trained on per-`(player, landmark_year_S)` rows with `horizon_offset_k`
as an explicit feature column; inference sets `k = step+1` instead of
advancing yip — train and inference draw from the same distribution.

| Layer | Model | Trained on |
|---|---|---|
| Hazards | `models/event_classifiers_v2.0b_tuned_prod.pkl` | 100% of panel (487k landmark rows). Optuna-tuned: `max_depth=4, max_leaf_nodes=15, lr=0.063, min_samples_leaf=70, l2=4.2, max_bins=211, max_iter=298`. |
| XGB head | `models/joint_xgb_v2.0b_oof_tuned.pkl` | OOF stacked CSV (248k OOF-honest rows). Optuna-tuned: `max_depth=6, lr=0.0129, min_child_weight=46, reg_lambda=6.86, subsample=0.90, colsample_bytree=0.96`. |
| Inference snap | `results/scored/snap2026_v2.0b_tuned_prod_long.csv` | 37,389 prospects at snap=2026. |
| Buy list | `results/buy_lists/buy_list_v2.0b_TUNED_FINAL.csv` | 18 prospects at `P(MLB_DEBUT) ≥ 0.60`. |

Honest at both layers — val pids excluded from hazards training (`hazards_full`
trained on 90% universe used to score val), and val pids never appear in
the XGB's OOF stacked training data.

## File map — [v2.0b_landmark/](v2.0b_landmark/)

| File | What it is |
|---|---|
| `bucket.csv` | Per `(event, bucket)` cell at `snap_offset=2`. Buckets: `ALL, R1, R2-R3, R4-R10, R10+, IFA`. |
| `walkforward.csv` | Per `(event, snap_offset)` cell, offsets 0..10. |
| `per_bucket_validation.csv` | Same buckets as `bucket.csv` but with the 0.50-threshold confusion-matrix view (`precision`, `recall`, `f1`, `accuracy`, `tp`, `fp`, `tn`, `fn`, `predicted_positives`). |
| `per_yip_validation.csv` | 0.50-threshold view per `(event, snap_offset)`. |
| `per_level_validation.csv` | 0.50-threshold view per `(event, current_level)` for `ALL, RK, A-, A, A+, AA, AAA, NONE`. |
| `headline.json` | Machine-readable summary. |

Columns inside `bucket.csv` and `walkforward.csv`: `n`, `pos`,
`base_rate`, `pred_mean`, `mean_fwd_years`, `auc` + bootstrap
`[auc_lo, auc_hi]`, `ap`, `ap_lift`, `brier`, `brier_skill`, `ece`,
`spiegelhalter_p`, plus `lift@{1,5,10}%` and `recall@{1,5,10}%` with the
cutoff index `k@K%`. Columns inside `per_*_validation.csv`: `n`, `pos`,
`base_rate`, `auc`, `ap`, `ap_lift`, `threshold`, `tp`, `fp`, `tn`, `fn`,
`precision`, `recall`, `f1`, `accuracy`, `predicted_positives`.

## Per-bucket validation with XGBoost outputs

**[v2.0b_landmark/per_bucket_validation.csv](v2.0b_landmark/per_bucket_validation.csv)** is
the table to start with if you want a single-glance answer to "how good is
the production model per draft bucket per event?".

For each `(bucket, event)` cell, scored at the per-event eligibility
filter, it reports:

| Column | Meaning |
|---|---|
| `n` | Eligible val players in the cell. |
| `pos` | Players who actually realized the event by 2026. |
| `base_rate` | `pos / n` — the random-guess hit rate. |
| `auc` | Area under the ROC curve. |
| `ap` | Average Precision = AU-PR. |
| `ap_lift` | `ap / base_rate` — how many × better than random the precision-weighted ranking is. |
| `spearman_rho` | Spearman rank correlation between the model's score and the realized 0/1 outcome on the cell. Threshold-free — captures "are higher-scored players actually more likely to fire?". Range −1..+1. Significance in `spearman_p`. |
| `threshold` | The XGBoost probability cutoff used for the confusion-matrix metrics. Fixed at **0.60** (production buy cutoff). |
| `tp`, `fp`, `tn`, `fn` | True/false positives/negatives at `xgb_p_event ≥ 0.50`. |
| `predicted_positives` | `tp + fp` — how many players the model said "yes" to. |
| `precision` | `tp / (tp + fp)`. Of players the model picked, fraction that hit. |
| `recall` | `tp / (tp + fn)`. Of players who hit, fraction the model picked. |
| `f1` | Harmonic mean of precision and recall — the standard balanced summary at this threshold. |
| `accuracy` | `(tp + tn) / n`. Total correct rate. **Warning**: for rare events (STAR/ELITE base rate ~0.4%), accuracy is near 1.00 even for a model that predicts "no" for everyone; use F1, precision, recall instead. |

Buckets: `ALL, R1, R2-R3, R4-R10, R10+, IFA`. The `ALL` row aggregates the
full val cohort.

### Full per-bucket numbers (per-event eligibility, threshold = 0.60 — the production buy cutoff)

The CSV linked above has the exact values below in machine-readable form.
`spearman` is Spearman's rank correlation ρ between the model's score
and the realized outcome on the cell (higher = better rank ordering;
significance test in the CSV's `spearman_p` column).

#### TOP_100_PROSPECT

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | 34164 | 230 | 0.67% | 0.985 | 0.483 | 71.8× | 0.137 | 0.882 | 0.130 | 0.227 | 30 | 4 | 200 |
| R1 | 388 | 80 | 20.62% | 0.931 | 0.757 | 3.7× | 0.604 | 0.957 | 0.275 | 0.427 | 22 | 1 | 58 |
| R2-R3 | 707 | 62 | 8.77% | 0.892 | 0.430 | 4.9× | 0.384 | 0.600 | 0.048 | 0.090 | 3 | 2 | 59 |
| R4-R10 | 2484 | 20 | 0.81% | 0.937 | 0.231 | 28.7× | 0.135 | 1.000 | 0.050 | 0.095 | 1 | 0 | 19 |
| R10+ | 11788 | 9 | 0.08% | 0.949 | 0.354 | 463.1× | 0.043 | — | 0.000 | — | 0 | 0 | 9 |
| IFA | 18797 | 59 | 0.31% | 0.986 | 0.349 | 111.3× | 0.094 | 0.800 | 0.068 | 0.125 | 4 | 1 | 55 |

#### MLB_DEBUT

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | 34430 | 1747 | 5.07% | 0.937 | 0.564 | 11.1× | 0.332 | 0.710 | 0.284 | 0.406 | 496 | 203 | 1251 |
| R1 | 491 | 240 | 48.88% | 0.895 | 0.898 | 1.8× | 0.684 | 0.843 | 0.671 | 0.747 | 161 | 30 | 79 |
| R2-R3 | 755 | 256 | 33.91% | 0.826 | 0.711 | 2.1× | 0.534 | 0.722 | 0.445 | 0.551 | 114 | 44 | 142 |
| R4-R10 | 2487 | 279 | 11.22% | 0.855 | 0.434 | 3.9× | 0.388 | 0.500 | 0.186 | 0.272 | 52 | 52 | 227 |
| R10+ | 11790 | 383 | 3.25% | 0.899 | 0.282 | 8.7× | 0.245 | 0.511 | 0.063 | 0.112 | 24 | 23 | 359 |
| IFA | 18907 | 589 | 3.12% | 0.943 | 0.524 | 16.8× | 0.266 | 0.729 | 0.246 | 0.368 | 145 | 54 | 444 |

#### ESTABLISHED_MLB

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | 34430 | 385 | 1.12% | 0.976 | 0.362 | 32.4× | 0.173 | 0.783 | 0.047 | 0.088 | 18 | 5 | 367 |
| R1 | 491 | 85 | 17.31% | 0.878 | 0.530 | 3.1× | 0.496 | 0.636 | 0.082 | 0.146 | 7 | 4 | 78 |
| R2-R3 | 755 | 61 | 8.08% | 0.842 | 0.295 | 3.7× | 0.322 | 0.750 | 0.049 | 0.092 | 3 | 1 | 58 |
| R4-R10 | 2487 | 76 | 3.06% | 0.928 | 0.306 | 10.0× | 0.255 | 1.000 | 0.013 | 0.026 | 1 | 0 | 75 |
| R10+ | 11790 | 41 | 0.35% | 0.971 | 0.192 | 55.1× | 0.096 | 1.000 | 0.024 | 0.048 | 1 | 0 | 40 |
| IFA | 18907 | 122 | 0.65% | 0.981 | 0.367 | 56.8× | 0.133 | 1.000 | 0.049 | 0.094 | 6 | 0 | 116 |

#### STAR_PLUS_ELITE

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | 34430 | 123 | 0.36% | 0.974 | 0.184 | 51.5× | 0.098 | — | 0.000 | — | 0 | 0 | 123 |
| R1 | 491 | 35 | 7.13% | 0.844 | 0.225 | 3.2× | 0.307 | — | 0.000 | — | 0 | 0 | 35 |
| R2-R3 | 755 | 11 | 1.46% | 0.914 | 0.267 | 18.3× | 0.172 | — | 0.000 | — | 0 | 0 | 11 |
| R4-R10 | 2487 | 21 | 0.84% | 0.952 | 0.248 | 29.4× | 0.143 | — | 0.000 | — | 0 | 0 | 21 |
| R10+ | 11790 | 18 | 0.15% | 0.936 | 0.178 | 116.5× | 0.059 | — | 0.000 | — | 0 | 0 | 18 |
| IFA | 18907 | 38 | 0.20% | 0.989 | 0.198 | 98.6× | 0.076 | — | 0.000 | — | 0 | 0 | 38 |

STAR_PLUS_ELITE never crosses the 0.60 cutoff — the AP=0.184 ranking is
strong but the rare positives sit at p ≈ 0.15-0.45. See the
**threshold-at-precision-≥-0.60** tables below for the tuned-per-cell
operating points that recover meaningful precision/recall numbers on
these rare events.

## Per-yip validation (threshold = 0.50)

Same threshold-0.50 confusion-matrix view as the bucket tables above,
but rows are `snap_offset` (years-in-pro since entry).

#### TOP_100_PROSPECT

| yip | n | pos | base% | AUC | AP | AP_lift | precision | recall | F1 | accuracy | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 3523 | 97 | 2.75% | 0.957 | 0.530 | 19.3× | 0.762 | 0.165 | 0.271 | 0.976 | 16 | 5 | 81 |
| 1 | 3481 | 67 | 1.92% | 0.962 | 0.482 | 25.0× | 0.810 | 0.254 | 0.386 | 0.984 | 17 | 4 | 50 |
| 2 | 3433 | 41 | 1.19% | 0.966 | 0.466 | 39.0× | 0.688 | 0.268 | 0.386 | 0.990 | 11 | 5 | 30 |
| 3 | 3346 | 17 | 0.51% | 0.971 | 0.376 | 74.0× | 0.333 | 0.059 | 0.100 | 0.995 | 1 | 2 | 16 |
| 4 | 3273 | 7 | 0.21% | 0.994 | 0.325 | 152.2× | — | 0.000 | — | 0.998 | 0 | 0 | 7 |
| 5 | 3204 | 1 | 0.03% | 1.000 | 1.000 | 3204.0× | — | 0.000 | — | 1.000 | 0 | 0 | 1 |

(yip 6-10: 0 eligible positives — censoring removes them.)

#### MLB_DEBUT

| yip | n | pos | base% | AUC | AP | AP_lift | precision | recall | F1 | accuracy | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 3543 | 404 | 11.40% | 0.885 | 0.579 | 5.1× | 0.702 | 0.361 | 0.477 | 0.910 | 146 | 62 | 258 |
| 1 | 3516 | 377 | 10.72% | 0.896 | 0.591 | 5.5× | 0.661 | 0.398 | 0.497 | 0.914 | 150 | 77 | 227 |
| 2 | 3481 | 342 | 9.82% | 0.916 | 0.637 | 6.5× | 0.637 | 0.468 | 0.540 | 0.922 | 160 | 91 | 182 |
| 3 | 3392 | 253 | 7.46% | 0.920 | 0.604 | 8.1× | 0.634 | 0.466 | 0.538 | 0.940 | 118 | 68 | 135 |
| 4 | 3309 | 170 | 5.14% | 0.914 | 0.543 | 10.6× | 0.617 | 0.388 | 0.477 | 0.956 | 66 | 41 | 104 |
| 5 | 3229 | 90 | 2.79% | 0.914 | 0.394 | 14.1× | 0.442 | 0.256 | 0.324 | 0.970 | 23 | 29 | 67 |
| 6 | 3187 | 55 | 1.73% | 0.939 | 0.392 | 22.7× | 0.700 | 0.255 | 0.373 | 0.985 | 14 | 6 | 41 |
| 7 | 2976 | 29 | 0.97% | 0.948 | 0.242 | 24.8× | 0.500 | 0.034 | 0.065 | 0.990 | 1 | 1 | 28 |
| 8 | 2778 | 16 | 0.58% | 0.935 | 0.182 | 31.6× | 0.000 | 0.000 | — | 0.994 | 0 | 1 | 16 |
| 9 | 2602 | 8 | 0.31% | 0.922 | 0.305 | 99.2× | — | 0.000 | — | 0.997 | 0 | 0 | 8 |
| 10 | 2417 | 3 | 0.12% | 0.881 | 0.341 | 274.9× | — | 0.000 | — | 0.999 | 0 | 0 | 3 |

#### ESTABLISHED_MLB

| yip | n | pos | base% | AUC | AP | AP_lift | precision | recall | F1 | accuracy | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 3543 | 120 | 3.39% | 0.929 | 0.360 | 10.6× | 0.583 | 0.058 | 0.106 | 0.967 | 7 | 5 | 113 |
| 1 | 3516 | 102 | 2.90% | 0.938 | 0.395 | 13.6× | 0.594 | 0.186 | 0.284 | 0.973 | 19 | 13 | 83 |
| 2 | 3481 | 83 | 2.38% | 0.965 | 0.430 | 18.1× | 0.654 | 0.205 | 0.312 | 0.978 | 17 | 9 | 66 |
| 3 | 3392 | 45 | 1.33% | 0.972 | 0.299 | 22.5× | 0.333 | 0.044 | 0.078 | 0.986 | 2 | 4 | 43 |
| 4 | 3309 | 24 | 0.73% | 0.983 | 0.387 | 53.4× | — | 0.000 | — | 0.993 | 0 | 0 | 24 |
| 5 | 3229 | 6 | 0.19% | 0.982 | 0.084 | 45.3× | — | 0.000 | — | 0.998 | 0 | 0 | 6 |
| 6 | 3187 | 3 | 0.09% | 0.977 | 0.164 | 173.7× | — | 0.000 | — | 0.999 | 0 | 0 | 3 |

#### STAR_PLUS_ELITE

| yip | n | pos | base% | AUC | AP | AP_lift | precision | recall | F1 | accuracy | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 3543 | 41 | 1.16% | 0.930 | 0.143 | 12.4× | — | 0.000 | — | 0.988 | 0 | 0 | 41 |
| 1 | 3516 | 32 | 0.91% | 0.955 | 0.272 | 29.9× | — | 0.000 | — | 0.991 | 0 | 0 | 32 |
| 2 | 3481 | 27 | 0.78% | 0.963 | 0.352 | 45.4× | — | 0.000 | — | 0.992 | 0 | 0 | 27 |
| 3 | 3392 | 13 | 0.38% | 0.976 | 0.255 | 66.6× | — | 0.000 | — | 0.996 | 0 | 0 | 13 |

## Per-level validation (threshold = 0.50)

Rows are `current_level` (the MiLB level the player was at when scored).

#### TOP_100_PROSPECT

| level | n | pos | base% | AUC | AP | AP_lift | precision | recall | F1 | accuracy | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | 34164 | 230 | 0.67% | 0.985 | 0.483 | 71.8× | 0.738 | 0.196 | 0.309 | 0.994 | 45 | 16 | 185 |
| RK | 3928 | 65 | 1.65% | 0.960 | 0.502 | 30.3× | 0.900 | 0.138 | 0.240 | 0.985 | 9 | 1 | 56 |
| A- | 987 | 19 | 1.93% | 0.967 | 0.518 | 26.9× | 0.800 | 0.211 | 0.333 | 0.984 | 4 | 1 | 15 |
| A | 1503 | 45 | 2.99% | 0.954 | 0.502 | 16.8× | 0.722 | 0.289 | 0.413 | 0.975 | 13 | 5 | 32 |
| A+ | 1510 | 29 | 1.92% | 0.972 | 0.449 | 23.4× | 0.545 | 0.207 | 0.300 | 0.981 | 6 | 5 | 23 |
| AA | 1390 | 33 | 2.37% | 0.982 | 0.615 | 25.9× | 0.769 | 0.303 | 0.435 | 0.981 | 10 | 3 | 23 |
| AAA | 1809 | 12 | 0.66% | 0.985 | 0.634 | 95.6× | 1.000 | 0.167 | 0.286 | 0.994 | 2 | 0 | 10 |
| NONE | 23037 | 27 | 0.12% | 0.988 | 0.304 | 259.4× | 0.500 | 0.037 | 0.069 | 0.999 | 1 | 1 | 26 |

#### MLB_DEBUT

| level | n | pos | base% | AUC | AP | AP_lift | precision | recall | F1 | accuracy | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | 34430 | 1747 | 5.07% | 0.937 | 0.564 | 11.1× | 0.643 | 0.388 | 0.484 | 0.958 | 678 | 376 | 1069 |
| RK | 3932 | 242 | 6.15% | 0.900 | 0.538 | 8.7× | 0.743 | 0.310 | 0.437 | 0.951 | 75 | 26 | 167 |
| A- | 989 | 126 | 12.74% | 0.834 | 0.483 | 3.8× | 0.569 | 0.294 | 0.387 | 0.882 | 37 | 28 | 89 |
| A | 1528 | 256 | 16.75% | 0.857 | 0.616 | 3.7× | 0.639 | 0.422 | 0.508 | 0.863 | 108 | 61 | 148 |
| A+ | 1541 | 260 | 16.87% | 0.820 | 0.586 | 3.5× | 0.636 | 0.450 | 0.527 | 0.864 | 117 | 67 | 143 |
| AA | 1478 | 376 | 25.44% | 0.838 | 0.705 | 2.8× | 0.667 | 0.527 | 0.588 | 0.813 | 198 | 99 | 178 |
| AAA | 1865 | 296 | 15.87% | 0.837 | 0.542 | 3.4× | 0.588 | 0.361 | 0.448 | 0.858 | 107 | 75 | 189 |
| NONE | 23097 | 191 | 0.83% | 0.929 | 0.333 | 40.2× | 0.643 | 0.188 | 0.291 | 0.992 | 36 | 20 | 155 |

#### ESTABLISHED_MLB

| level | n | pos | base% | AUC | AP | AP_lift | precision | recall | F1 | accuracy | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | 34430 | 385 | 1.12% | 0.976 | 0.362 | 32.4× | 0.592 | 0.117 | 0.195 | 0.989 | 45 | 31 | 340 |
| RK | 3932 | 63 | 1.60% | 0.945 | 0.330 | 20.6× | 0.000 | 0.000 | — | 0.984 | 0 | 1 | 63 |
| A- | 989 | 24 | 2.43% | 0.931 | 0.282 | 11.6× | 0.333 | 0.042 | 0.074 | 0.975 | 1 | 2 | 23 |
| A | 1528 | 69 | 4.52% | 0.929 | 0.376 | 8.3× | 0.533 | 0.116 | 0.190 | 0.955 | 8 | 7 | 61 |
| A+ | 1541 | 71 | 4.61% | 0.918 | 0.360 | 7.8× | 0.533 | 0.113 | 0.186 | 0.955 | 8 | 7 | 63 |
| AA | 1478 | 99 | 6.70% | 0.900 | 0.447 | 6.7× | 0.639 | 0.232 | 0.341 | 0.940 | 23 | 13 | 76 |
| AAA | 1865 | 49 | 2.63% | 0.928 | 0.374 | 14.2× | 0.833 | 0.102 | 0.182 | 0.976 | 5 | 1 | 44 |
| NONE | 23097 | 10 | 0.04% | 0.999 | 0.255 | 588.9× | — | 0.000 | — | 1.000 | 0 | 0 | 10 |

#### STAR_PLUS_ELITE

| level | n | pos | base% | AUC | AP | AP_lift | precision | recall | F1 | accuracy | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | 34430 | 123 | 0.36% | 0.974 | 0.184 | 51.5× | — | 0.000 | — | 0.996 | 0 | 0 | 123 |
| RK | 3932 | 23 | 0.58% | 0.957 | 0.164 | 28.1× | — | 0.000 | — | 0.994 | 0 | 0 | 23 |
| A- | 989 | 5 | 0.51% | 0.944 | 0.079 | 15.6× | — | 0.000 | — | 0.995 | 0 | 0 | 5 |
| A | 1528 | 22 | 1.44% | 0.939 | 0.194 | 13.4× | — | 0.000 | — | 0.986 | 0 | 0 | 22 |
| A+ | 1541 | 21 | 1.36% | 0.932 | 0.198 | 14.5× | — | 0.000 | — | 0.986 | 0 | 0 | 21 |
| AA | 1478 | 32 | 2.17% | 0.918 | 0.297 | 13.7× | — | 0.000 | — | 0.978 | 0 | 0 | 32 |
| AAA | 1865 | 14 | 0.75% | 0.891 | 0.197 | 26.2× | — | 0.000 | — | 0.992 | 0 | 0 | 14 |
| NONE | 23097 | 6 | 0.03% | 0.982 | 0.021 | 81.6× | — | 0.000 | — | 1.000 | 0 | 0 | 6 |

## Threshold-at-precision-≥-0.60

For each slice, find the **lowest** XGB probability threshold whose
precision among players-at-or-above is ≥ 0.60. This is the buy-list
operating point — "what cutoff should I trust to be 60% right?" — and
the recall reported is "of all eventual hitters in this slice, what
fraction did the model's confident-buys catch?".

CSVs: [`thresholds_at_p60_per_bucket.csv`](v2.0b_landmark/thresholds_at_p60_per_bucket.csv), [`thresholds_at_p60_per_yip.csv`](v2.0b_landmark/thresholds_at_p60_per_yip.csv), [`thresholds_at_p60_per_level.csv`](v2.0b_landmark/thresholds_at_p60_per_level.csv).

`lift` column is `precision / base_rate`. Dash entries mean the
slice has no threshold whose precision reaches 0.60 (too few positives).

### Per bucket

#### TOP_100_PROSPECT

| bucket | n | pos | base% | threshold | n≥thr | TP | precision | recall | lift |
|---|---|---|---|---|---|---|---|---|---|
| ALL | 34164 | 230 | 0.67% | 0.347 | 125 | 75 | 0.600 | 0.326 | 89.1× |
| R1 | 388 | 80 | 20.62% | 0.120 | 120 | 72 | 0.600 | 0.900 | 2.9× |
| R2-R3 | 707 | 62 | 8.77% | 0.632 | 5 | 3 | 0.600 | 0.048 | 6.8× |
| R4-R10 | 2484 | 20 | 0.81% | 0.440 | 3 | 2 | 0.667 | 0.100 | 82.8× |
| R10+ | 11788 | 9 | 0.08% | 0.282 | 5 | 3 | 0.600 | 0.333 | 785.9× |
| IFA | 18797 | 59 | 0.31% | 0.467 | 16 | 10 | 0.625 | 0.169 | 199.1× |

#### MLB_DEBUT

| bucket | n | pos | base% | threshold | n≥thr | TP | precision | recall | lift |
|---|---|---|---|---|---|---|---|---|---|
| ALL | 34430 | 1747 | 5.07% | 0.447 | 1271 | 763 | 0.600 | 0.437 | 11.8× |
| R1 | 491 | 240 | 48.88% | 0.061 | 395 | 237 | 0.600 | 0.988 | 1.2× |
| R2-R3 | 755 | 256 | 33.91% | 0.400 | 296 | 178 | 0.601 | 0.695 | 1.8× |
| R4-R10 | 2487 | 279 | 11.22% | 0.661 | 71 | 43 | 0.606 | 0.154 | 5.4× |
| R10+ | 11790 | 383 | 3.25% | 0.725 | 10 | 6 | 0.600 | 0.016 | 18.5× |
| IFA | 18907 | 589 | 3.12% | 0.444 | 390 | 234 | 0.600 | 0.397 | 19.3× |

#### ESTABLISHED_MLB

| bucket | n | pos | base% | threshold | n≥thr | TP | precision | recall | lift |
|---|---|---|---|---|---|---|---|---|---|
| ALL | 34430 | 385 | 1.12% | 0.506 | 73 | 44 | 0.603 | 0.114 | 53.9× |
| R1 | 491 | 85 | 17.31% | 0.497 | 35 | 21 | 0.600 | 0.247 | 3.5× |
| R2-R3 | 755 | 61 | 8.08% | 0.527 | 10 | 6 | 0.600 | 0.098 | 7.4× |
| R4-R10 | 2487 | 76 | 3.06% | 0.338 | 11 | 7 | 0.636 | 0.092 | 20.8× |
| R10+ | 11790 | 41 | 0.35% | 0.354 | 5 | 3 | 0.600 | 0.073 | 172.5× |
| IFA | 18907 | 122 | 0.65% | 0.477 | 25 | 15 | 0.600 | 0.123 | 93.0× |

#### STAR_PLUS_ELITE

| bucket | n | pos | base% | threshold | n≥thr | TP | precision | recall | lift |
|---|---|---|---|---|---|---|---|---|---|
| ALL | 34430 | 123 | 0.36% | 0.237 | 5 | 3 | 0.600 | 0.024 | 168.0× |
| R1 | 491 | 35 | 7.13% | — | 0 | 0 | — | — | — |
| R2-R3 | 755 | 11 | 1.46% | 0.228 | 3 | 2 | 0.667 | 0.182 | 45.8× |
| R4-R10 | 2487 | 21 | 0.84% | 0.121 | 6 | 4 | 0.667 | 0.190 | 79.0× |
| R10+ | 11790 | 18 | 0.15% | 0.079 | 5 | 3 | 0.600 | 0.167 | 393.0× |
| IFA | 18907 | 38 | 0.20% | 0.241 | 3 | 2 | 0.667 | 0.053 | 331.7× |

### Per yip (snap_offset)

#### TOP_100_PROSPECT

| yip | n | pos | base% | threshold | n≥thr | TP | precision | recall | lift |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 3523 | 97 | 2.75% | 0.293 | 60 | 36 | 0.600 | 0.371 | 21.8× |
| 1 | 3481 | 67 | 1.92% | 0.347 | 38 | 23 | 0.605 | 0.343 | 31.4× |
| 2 | 3433 | 41 | 1.19% | 0.251 | 36 | 22 | 0.611 | 0.537 | 51.2× |
| 3 | 3346 | 17 | 0.51% | 0.589 | 1 | 1 | 1.000 | 0.059 | 196.8× |
| 5 | 3204 | 1 | 0.03% | 0.178 | 1 | 1 | 1.000 | 1.000 | 3204× |

#### MLB_DEBUT

| yip | n | pos | base% | threshold | n≥thr | TP | precision | recall | lift |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 3543 | 404 | 11.40% | 0.408 | 308 | 185 | 0.601 | 0.458 | 5.3× |
| 1 | 3516 | 377 | 10.72% | 0.444 | 290 | 174 | 0.600 | 0.462 | 5.6× |
| 2 | 3481 | 342 | 9.82% | 0.451 | 296 | 178 | 0.601 | 0.520 | 6.1× |
| 3 | 3392 | 253 | 7.46% | 0.466 | 208 | 125 | 0.601 | 0.494 | 8.1× |
| 4 | 3309 | 170 | 5.14% | 0.419 | 133 | 80 | 0.602 | 0.471 | 11.7× |
| 5 | 3229 | 90 | 2.79% | 0.629 | 28 | 17 | 0.607 | 0.189 | 21.8× |
| 6 | 3187 | 55 | 1.73% | 0.409 | 26 | 16 | 0.615 | 0.291 | 35.7× |
| 7 | 2976 | 29 | 0.97% | 0.528 | 1 | 1 | 1.000 | 0.034 | 102.6× |
| 9 | 2602 | 8 | 0.31% | 0.149 | 5 | 3 | 0.600 | 0.375 | 195.2× |
| 10 | 2417 | 3 | 0.12% | 0.296 | 1 | 1 | 1.000 | 0.333 | 805.7× |

#### ESTABLISHED_MLB

| yip | n | pos | base% | threshold | n≥thr | TP | precision | recall | lift |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 3543 | 120 | 3.39% | 0.498 | 13 | 8 | 0.615 | 0.067 | 18.2× |
| 1 | 3516 | 102 | 2.90% | 0.468 | 40 | 24 | 0.600 | 0.235 | 20.7× |
| 2 | 3481 | 83 | 2.38% | 0.428 | 38 | 23 | 0.605 | 0.277 | 25.4× |
| 3 | 3392 | 45 | 1.33% | 0.522 | 3 | 2 | 0.667 | 0.044 | 50.3× |
| 4 | 3309 | 24 | 0.73% | 0.358 | 15 | 9 | 0.600 | 0.375 | 82.7× |

#### STAR_PLUS_ELITE

| yip | n | pos | base% | threshold | n≥thr | TP | precision | recall | lift |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3516 | 32 | 0.91% | 0.222 | 5 | 3 | 0.600 | 0.094 | 65.9× |
| 2 | 3481 | 27 | 0.78% | 0.208 | 8 | 5 | 0.625 | 0.185 | 80.6× |
| 3 | 3392 | 13 | 0.38% | 0.122 | 1 | 1 | 1.000 | 0.077 | 260.9× |

### Per level

#### TOP_100_PROSPECT

| level | n | pos | base% | threshold | n≥thr | TP | precision | recall | lift |
|---|---|---|---|---|---|---|---|---|---|
| ALL | 34164 | 230 | 0.67% | 0.347 | 125 | 75 | 0.600 | 0.326 | 89.1× |
| RK | 3928 | 65 | 1.65% | 0.241 | 46 | 28 | 0.609 | 0.431 | 36.8× |
| A- | 987 | 19 | 1.93% | 0.225 | 15 | 9 | 0.600 | 0.474 | 31.2× |
| A | 1503 | 45 | 2.99% | 0.293 | 40 | 24 | 0.600 | 0.533 | 20.0× |
| A+ | 1510 | 29 | 1.92% | 0.421 | 16 | 10 | 0.625 | 0.345 | 32.5× |
| AA | 1390 | 33 | 2.37% | 0.204 | 35 | 21 | 0.600 | 0.636 | 25.3× |
| AAA | 1809 | 12 | 0.66% | 0.265 | 10 | 6 | 0.600 | 0.500 | 90.5× |
| NONE | 23037 | 27 | 0.12% | 0.366 | 8 | 5 | 0.625 | 0.185 | 533.3× |

#### MLB_DEBUT

| level | n | pos | base% | threshold | n≥thr | TP | precision | recall | lift |
|---|---|---|---|---|---|---|---|---|---|
| ALL | 34430 | 1747 | 5.07% | 0.447 | 1271 | 763 | 0.600 | 0.437 | 11.8× |
| RK | 3932 | 242 | 6.15% | 0.421 | 163 | 98 | 0.601 | 0.405 | 9.8× |
| A- | 989 | 126 | 12.74% | 0.481 | 73 | 44 | 0.603 | 0.349 | 4.7× |
| A | 1528 | 256 | 16.75% | 0.473 | 188 | 113 | 0.601 | 0.441 | 3.6× |
| A+ | 1541 | 260 | 16.87% | 0.474 | 208 | 125 | 0.601 | 0.481 | 3.6× |
| AA | 1478 | 376 | 25.44% | 0.375 | 393 | 236 | 0.601 | 0.628 | 2.4× |
| AAA | 1865 | 296 | 15.87% | 0.519 | 175 | 105 | 0.600 | 0.355 | 3.8× |
| NONE | 23097 | 191 | 0.83% | 0.409 | 83 | 50 | 0.602 | 0.262 | 72.8× |

#### ESTABLISHED_MLB

| level | n | pos | base% | threshold | n≥thr | TP | precision | recall | lift |
|---|---|---|---|---|---|---|---|---|---|
| ALL | 34430 | 385 | 1.12% | 0.506 | 73 | 44 | 0.603 | 0.114 | 53.9× |
| A- | 989 | 24 | 2.43% | 0.601 | 1 | 1 | 1.000 | 0.042 | 41.2× |
| A | 1528 | 69 | 4.52% | 0.443 | 26 | 16 | 0.615 | 0.232 | 13.6× |
| A+ | 1541 | 71 | 4.61% | 0.606 | 5 | 3 | 0.600 | 0.042 | 13.0× |
| AA | 1478 | 99 | 6.70% | 0.482 | 41 | 25 | 0.610 | 0.253 | 9.1× |
| AAA | 1865 | 49 | 2.63% | 0.430 | 15 | 9 | 0.600 | 0.184 | 22.8× |
| NONE | 23097 | 10 | 0.04% | 0.373 | 1 | 1 | 1.000 | 0.100 | 2309.7× |

#### STAR_PLUS_ELITE

| level | n | pos | base% | threshold | n≥thr | TP | precision | recall | lift |
|---|---|---|---|---|---|---|---|---|---|
| ALL | 34430 | 123 | 0.36% | 0.237 | 5 | 3 | 0.600 | 0.024 | 168.0× |
| A+ | 1541 | 21 | 1.36% | 0.216 | 5 | 3 | 0.600 | 0.143 | 44.0× |
| AA | 1478 | 32 | 2.17% | 0.221 | 8 | 5 | 0.625 | 0.156 | 28.9× |

### How to read these tables

- **`base%` is the random-guess hit rate** for that bucket × event. Compare
  `precision` and `AP_lift` against it.
- **AP_lift is the headline discrimination number.** For rare events
  (TOP_100 / STAR_PLUS_ELITE / ESTABLISHED_MLB at IFA / R10+), the
  multiplier is huge (50× to 463×) because the base rate is tiny — the
  model's positives-weighted ranking is dramatically sharper than random.
- **R1 cells are tight** (n = 388-491) because R1 picks are dropped from
  the buy universe in production. R1 numbers are included for
  completeness; the buy list output is dominated by R2-R3 / R4-R10 / R10+
  / IFA.
- **Precision and recall depend on the threshold.** At the default 0.50
  cutoff, the model is selective — it favors precision. For rare events
  it produces zero positives at p ≥ 0.50. To trade precision for recall,
  consult `MLB_DEBUT_thresholds_at_p60.csv` (per-yip 0.60-precision
  cutoffs) or the `<event>_cum_above_threshold.csv` files.
- **Accuracy can be deceptive** for rare events. `STAR_PLUS_ELITE / IFA`
  accuracy is 0.998 — a model predicting "no" for everyone would also score
  0.998. Always read accuracy together with AP / AP_lift on imbalanced
  cells.

## Validate-full per-event tables

The packet additionally carries the deeper per-yip × per-percentile slab
analysis. Per event (MLB_DEBUT, TOP_100_PROSPECT, ESTABLISHED_MLB,
STAR_PLUS_ELITE):

| File suffix | What it is |
|---|---|
| `<event>_walkforward.csv` | Per-snap_offset AU-PR / lift / ECE for this event. |
| `<event>_pct_slabs.csv` | Per (yip, percentile slab) realized rate. Slabs: 0-0.5, 0.5-1.0, 1.0-1.5, 1.5-2.0, 2-3, 3-4, 4-5, 5-10, 10-20, 20-50, bottom 50%. |
| `<event>_cum_above_threshold.csv` | Cumulative-from-top realized rate as you lower the score threshold. |

Plus three MLB_DEBUT-specific tables:

| File | What it is |
|---|---|
| `MLB_DEBUT_per_current_level.csv` | Detailed metrics by the level the player was at when scored (RK, A-, A, A+, AA, AAA, NONE). |
| `MLB_DEBUT_thresholds_at_p60.csv` | Per-yip XGB score threshold that achieves ≥60% precision among players scoring above. |
| `MLB_DEBUT_time_to_debut.csv` | Predicted years-to-debut vs actual, for val-cohort debutees. |

## Statistics glossary — how to read every column

### Structural columns

| Column | Meaning |
|---|---|
| `event` | Career event being predicted. One of `TOP_100_PROSPECT, MLB_DEBUT, ESTABLISHED_MLB, STAR_PLUS_ELITE, ELITE, STAR`. `STAR_PLUS_ELITE` is the union: `1 − (1 − p_STAR)(1 − p_ELITE)`. |
| `bucket` | Draft-pedigree bucket. `R1` = first round, `R2-R3`, `R4-R10`, `R10+` = rounds 11+, `IFA` = international free agent (no draft). |
| `snap_offset` | Years since entry. For a 2019-drafted player, snap_year=2021 → snap_offset=2. Also called "yip" (years-in-pro). |
| `n` | Number of eligible players in this cell. **Eligibility**: the event hadn't fired by the start of the snap year AND the player has enough forward observation window for the event to plausibly fire. Players whose event already fired before the snap (e.g. already debuted) are excluded — the model isn't being asked about them. |
| `pos` | Number of eligible players who realized the event in the window `(snap_year, observe_through=2026]`. |
| `base_rate` | `pos / n`. The naive guess rate — if you predicted "yes" for everyone you'd be right this fraction of the time. |
| `pred_mean` | Mean of the model's predicted probability across the cell. Compare to `base_rate`: well-calibrated models have `pred_mean ≈ base_rate`. |

### Discrimination metrics — "does the model rank the right players higher?"

| Metric | What it measures |
|---|---|
| `auc` | Area Under the ROC Curve. 0.5 = random, 1.0 = perfect. Insensitive to class imbalance — useful but gives the same number whether positives are 50% or 0.5% of the cell. |
| `auc_lo`, `auc_hi` | 95% bootstrap confidence interval on AUC (200 resamples). If the interval includes 0.5, the model isn't doing better than random on this cell. |
| **`ap`** | **Average Precision = Area Under the Precision-Recall Curve (AU-PR).** Far more informative than AUC for rare events. Tells you how good the ranking is *at the positives*. For a 1% base-rate event, AP=0.5 means top-X% selections are about 50× the random precision. |
| **`ap_lift`** | `ap / base_rate`. The "you got X× better than random precision-weighted ranking" multiplier. AP_lift = 1.0 means random; AP_lift = 100 means the model's positives-weighted ranking is 100× sharper than random. |
| **`spearman_rho`** | Spearman rank correlation between the model's score and the realized 0/1 outcome on the cell. Captures monotonic agreement between predicted rank and outcome. 0 = no relationship, +1 = perfectly increasing, −1 = perfectly inverted. Threshold-free, so a useful sanity sibling to AP. `spearman_p` is its two-sided p-value (small = reject "no correlation"). |

### Calibration metrics — "when the model says 30%, do 30% actually happen?"

| Metric | What it measures |
|---|---|
| `brier` | Brier score = mean squared error of predictions vs realized 0/1. Lower is better. Sensitive to both calibration and discrimination. |
| **`brier_skill`** | Brier Skill Score = `1 − brier / brier_baseline`, where the baseline predicts `base_rate` for everyone. **Positive = better than baseline; negative = worse**. A BSS of 0.3 means 30% reduction in Brier vs always-guess-base-rate. |
| **`ece`** | Expected Calibration Error. Bins predictions into 10 buckets, computes `|mean(pred) − mean(realized)|` per bin, averages weighted by bin size. Low (< 0.05) = well-calibrated. |
| `spiegelhalter_p` | Two-sided p-value for "the model is calibrated" (H0). Small (< 0.05) = reject calibration. Use as a sanity flag, not the headline read. |

### Top-K precision/recall metrics — "if I sniped the top K%, what's the hit rate?"

| Metric | What it means |
|---|---|
| `lift@K%` | Precision in top K% by predicted probability, divided by `base_rate`. `lift@5% = 7.2` means: take the top 5% the model is most confident about, and 7.2× the base rate of those will actually fire. |
| `recall@K%` | Fraction of all real positives captured in the top K%. `recall@5% = 0.36` means the top-5% slice contains 36% of all positives. |
| `k@K%` | Number of players in the top-K% slice (`= round(n * K/100)`). |

### What "snap_offset" means in the walk-forward table

Each row in `walkforward.csv` is a snapshot of the model at a specific
years-into-career value. **Walk-forward at offset = 2 means**: "score this
player using only stats available 2 years after entry, and see if any of the
events fire by 2026". As `snap_offset` grows, the cohort shrinks (events
that already fired are removed) and the forward observation window
shortens.

This is why later offsets often show NaN AUC — the cohort got so small
or the remaining `pos` count dropped to zero (right-censoring or because all
the positives already realized by then).

### Slab / threshold tables

`MLB_DEBUT_thresholds_at_p60.csv` answers the question "what score do I
need to be confident this player will hit?" Per yip:

| Column | Meaning |
|---|---|
| `threshold` | The cutoff value. Players scoring `≥ threshold` are the "buy" cohort. |
| `n_above` | How many players in the val cohort scored above that threshold. |
| `tp_above` | How many of those actually fired the event. |
| `precision` | `tp_above / n_above`. The "if I buy at this threshold, hit rate" number. Pinned to ≥ 0.60 by construction. |
| `recall` | `tp_above / pos_total`. Fraction of all eventual hitters captured by this threshold. |

`<event>_pct_slabs.csv` slices each (yip, percentile-band) cell so you can
see how the realized rate falls off as you go lower in the ranking. The
0-0.5% slab is "top half-percent picks at this yip"; the bottom-50% is the
control.

### `<event>_cum_above_threshold.csv`

Walks the score down from the top and reports the cumulative number of
above-threshold picks and their realized rate at each of 27 percentile
cuts (1, 2, …, 20, 25, 30, 40, 50, 60, 75, 100). Useful for "if I'm willing
to accept N false positives, how many true positives do I catch?"

### `MLB_DEBUT_time_to_debut.csv`

Per realized debutee in the val cohort: predicted years until debut vs
actual. Tells you whether the time-to-debut head is right not just on the
"who" but on the "when".

### What's NOT in here

- The 8 MB per-player prediction tables (`long.csv`). Available locally in
  `results/scored/` if you need per-row predictions; excluded here to keep
  the repo small.
- Buy-list outputs (those live in `results/buy_lists/`; current production
  is `buy_list_v2.0b_TUNED_FINAL.csv`).

## Headline result

Per-event eligibility filter, weighted-AP with MLB_DEBUT 2× weight:

| Event | n | base_rate | AP | AP lift × base | AUC |
|---|---:|---:|---:|---:|---:|
| TOP_100_PROSPECT | 34,164 | 0.67% | **0.483** | 71.8× | 0.985 |
| MLB_DEBUT | 34,430 | 5.07% | **0.564** | 11.1× | 0.937 |
| ESTABLISHED_MLB | 34,430 | 1.12% | **0.362** | 32.4× | 0.976 |
| STAR_PLUS_ELITE | 34,430 | 0.36% | **0.184** | 51.5× | 0.974 |
| **weighted-AP** | | | **0.431** | | |

## How to regenerate

```bash
# 1. Build panel + OOF folds (one-time, ~25 min)
python -m scripts_v17.train.run_v2_0b_oof

# 2. Train full-universe hazards + score val
python -m scripts_v17.train.finalize_v2_0b_oof

# 3. Tune hazards HP (overnight, ~5 hours)
python -m scripts_v17.train.tune_hazards_oof --trials 200 --n-jobs 4 \
    --storage sqlite:///scratch/v20b_oof/hazards_study.db

# 4. Tune XGB HP (~2 hours)
python -m scripts_v17.train.tune_joint_xgb_v2_oof --trials 200 \
    --storage sqlite:///oof_xgb_hz149.db

# 5. Train tuned prod hazards on 100% + score snap=2026 + build buy list
python -m scripts_v17.train.train_v2_0b_tuned_prod

# 6. Regenerate all eval tables (compact + detailed)
python -m scripts_v17.validate.regen_eval_v2_0b_honest
python -m scripts_v17.validate.regen_full_eval_v2_0b
```
