# Held-out validation packets

Reproducible evaluation of the prospect-card models against the **10% val
player slice** of the v1.17 seed=42 split — 3,523 players the production
**joint XGBoost head was never trained on**. Validation universe: drafted
players with `draft_year ≤ 2020` (plus IFAs), realized window through 2026.

**Important caveat on honesty**: the XGB used for the per-bucket / per-yip
/ per-level / walkforward tables here (`models/joint_xgb_v2.0b.pkl`) was
trained with the val slice held out — honest at the XGB layer. But the
upstream landmark hazards (`event_classifiers_v1.18b_landmark_prod.pkl`)
were trained on 100% of players, so the hazard outputs the XGB consumes
*did* include val pids during hazard training. That makes the published
numbers an honest read of the XGB head specifically; the hazard layer is
still leaky for these pids. v1.17 has a true honest-twin hazard pkl
(`event_classifiers_v1.17.pkl`); v1.18b does not.

Two model versions are scored against the same val cohort for direct
comparison:

| Directory | Architecture |
|---|---|
| [v2.0_contemporaneous/](v2.0_contemporaneous/) | v2.0 baseline. 8 HistGBT hazards trained on `(features as_of year-1, label = fires in year)` — strict one-year-ahead. Walk-forward at inference advances yip/age/yics on a frozen feature vector (LOCF) and integrates per-year hazards. |
| [v2.0b_landmark/](v2.0b_landmark/) | v2.0b landmark. Same HistGBT shape but `horizon_offset_k` is now an explicit feature column; each (player, landmark, k) row is its own training example. Inference sets `k = step+1` instead of advancing yip. Train and inference now draw from the same distribution. |

Both consume the joint XGBoost downstream + v1.18b time-to-debut model. The
hazards are the only thing that differs.

## File map per directory

| File | What it is |
|---|---|
| `report.txt` | Human-readable summary. Bucket report at `snap_offset=2` + walk-forward per-offset. |
| `bucket.csv` | Per `(event, bucket)` cell at `snap_offset=2`. Buckets: `ALL, R1, R2-R3, R4-R10, R10+, IFA`. |
| `walkforward.csv` | Per `(event, snap_offset)` cell, offsets 0..10. |
| `lasso_curve.csv` | Per `(bucket, event, percentile-band)`: realized rate, lift, score range. Bands `0-10%, 10-20%, ..., 90-95%, 95-99%, 99-100%`. |
| `lasso_thresholds.csv` | Per `(bucket, event, target_rate ∈ {5,10,25,50,75%})`: the maximum-inclusive lasso score above which realized rate hits the target. |
| `lasso_report.txt` | Human-readable lasso analysis. |

Columns inside the metric CSVs include `n`, `pos`, `base_rate`, `auc`,
`ap` (= AU-PR), `ap_lift` (= AP / base_rate), `brier_skill`,
`lift@{1,5,10}%`, `recall@{1,5,10}%`, `ece`, `spiegelhalter_p`. The
contemporaneous packet predates the AU-PR patch so its `bucket.csv` and
`walkforward.csv` lack the `ap` and `ap_lift` columns — use the landmark
packet for those.

## Per-bucket validation with XGBoost outputs (landmark only)

**[v2.0b_landmark/per_bucket_validation.csv](v2.0b_landmark/per_bucket_validation.csv)** is
the table to start with if you want a single-glance answer to "how good is
the production model per draft bucket per event?".

For each `(bucket, event)` cell, scored at `snap_offset = 2` (the
canonical two-years-post-entry view), it reports:

| Column | Meaning |
|---|---|
| `n` | Eligible val players in the cell. |
| `pos` | Players who actually realized the event by 2026. |
| `base_rate` | `pos / n` — the random-guess hit rate. |
| `auc` | Area under the ROC curve. |
| `ap` | Average Precision = AU-PR. |
| `ap_lift` | `ap / base_rate` — how many × better than random the precision-weighted ranking is. |
| `threshold` | The XGBoost probability cutoff used to compute the confusion-matrix metrics below. Fixed at **0.50**. |
| `tp`, `fp`, `tn`, `fn` | True/false positives/negatives at `xgb_p_event ≥ 0.50`. |
| `predicted_positives` | `tp + fp` — how many players the model said "yes" to. |
| `precision` | `tp / (tp + fp)`. Of players the model picked, fraction that hit. |
| `recall` | `tp / (tp + fn)`. Of players who hit, fraction the model picked. |
| `f1` | Harmonic mean of precision and recall — the standard balanced summary at this threshold. |
| `accuracy` | `(tp + tn) / n`. Total correct rate. **Warning**: for rare events (STAR/ELITE base rate ~0.5%), accuracy is near 1.00 even for a model that predicts "no" for everyone; use F1, precision, recall instead. |

Buckets: `ALL, R1, R2-R3, R4-R10, R10+, IFA`. The `ALL` row aggregates the
full val cohort.

The XGBoost output is the production scoring head's calibrated probability
per event (joint multi-output booster trained on the combined fit+val
landmark slice). All metrics in this file are computed on the v1.17
seed=42 val pids — the same held-out cohort the other CSVs in this
directory use.

### Full per-bucket numbers (snap_offset=2, threshold=0.50)

The CSV linked above has the exact values below in machine-readable form.
Buckets: `ALL = aggregate`, `R1 = first round`, `R2-R3`, `R4-R10`,
`R10+ = rounds 11+`, `IFA = international free agents`.

#### TOP_100_PROSPECT

| bucket | n | pos | base% | AUC | AP | AP_lift | precision | recall | F1 | accuracy | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | 3433 | 41 | 1.19% | 0.990 | 0.645 | 54.0× | 0.684 | 0.317 | 0.433 | 0.990 | 13 | 6 | 28 |
| R1 | 49 | 13 | 26.53% | 0.964 | 0.901 | 3.4× | 0.875 | 0.538 | 0.667 | 0.857 | 7 | 1 | 6 |
| R2-R3 | 93 | 10 | 10.75% | 0.898 | 0.486 | 4.5× | 0.333 | 0.100 | 0.154 | 0.882 | 1 | 2 | 9 |
| R4-R10 | 270 | 4 | 1.48% | 0.981 | 0.323 | 21.8× | 0.000 | 0.000 | — | 0.981 | 0 | 1 | 4 |
| R10+ | 1165 | 2 | 0.17% | 0.940 | 0.507 | 295× | 1.000 | 0.500 | 0.667 | 0.999 | 1 | 0 | 1 |
| IFA | 1856 | 12 | 0.65% | 0.997 | 0.609 | 94.2× | 0.667 | 0.333 | 0.444 | 0.995 | 4 | 2 | 8 |

#### MLB_DEBUT

| bucket | n | pos | base% | AUC | AP | AP_lift | precision | recall | F1 | accuracy | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | 3481 | 342 | 9.82% | 0.941 | 0.710 | 7.2× | 0.685 | 0.497 | 0.576 | 0.928 | 170 | 78 | 172 |
| R1 | 73 | 49 | 67.12% | 0.885 | 0.938 | 1.4× | 0.891 | 0.837 | 0.863 | 0.822 | 41 | 5 | 8 |
| R2-R3 | 103 | 54 | 52.43% | 0.835 | 0.873 | 1.7× | 0.759 | 0.759 | 0.759 | 0.748 | 41 | 13 | 13 |
| R4-R10 | 271 | 57 | 21.03% | 0.843 | 0.609 | 2.9× | 0.508 | 0.544 | 0.525 | 0.793 | 31 | 30 | 26 |
| R10+ | 1165 | 71 | 6.09% | 0.914 | 0.497 | 8.2× | 0.581 | 0.254 | 0.353 | 0.943 | 18 | 13 | 53 |
| IFA | 1869 | 111 | 5.94% | 0.945 | 0.651 | 11.0× | 0.696 | 0.351 | 0.467 | 0.952 | 39 | 17 | 72 |

#### ESTABLISHED_MLB

| bucket | n | pos | base% | AUC | AP | AP_lift | precision | recall | F1 | accuracy | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | 3481 | 83 | 2.38% | 0.983 | 0.653 | 27.4× | 0.694 | 0.410 | 0.515 | 0.982 | 34 | 15 | 49 |
| R1 | 73 | 19 | 26.03% | 0.947 | 0.870 | 3.3× | 0.722 | 0.684 | 0.703 | 0.849 | 13 | 5 | 6 |
| R2-R3 | 103 | 15 | 14.56% | 0.893 | 0.583 | 4.0× | 0.556 | 0.333 | 0.417 | 0.864 | 5 | 4 | 10 |
| R4-R10 | 271 | 18 | 6.64% | 0.939 | 0.642 | 9.7× | 1.000 | 0.222 | 0.364 | 0.948 | 4 | 0 | 14 |
| R10+ | 1165 | 9 | 0.77% | 0.984 | 0.653 | 84.5× | 1.000 | 0.333 | 0.500 | 0.995 | 3 | 0 | 6 |
| IFA | 1869 | 22 | 1.18% | 0.989 | 0.623 | 52.9× | 0.600 | 0.409 | 0.486 | 0.990 | 9 | 6 | 13 |

#### STAR_PLUS_ELITE

| bucket | n | pos | base% | AUC | AP | AP_lift | precision | recall | F1 | accuracy | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | 3481 | 27 | 0.78% | 0.984 | 0.678 | 87.4× | — | 0.000 | — | 0.992 | 0 | 0 | 27 |
| R1 | 73 | 8 | 10.96% | 0.971 | 0.850 | 7.8× | — | 0.000 | — | 0.890 | 0 | 0 | 8 |
| R2-R3 | 103 | 3 | 2.91% | 0.967 | 0.591 | 20.3× | — | 0.000 | — | 0.971 | 0 | 0 | 3 |
| R4-R10 | 271 | 6 | 2.21% | 0.997 | 0.873 | 39.4× | — | 0.000 | — | 0.978 | 0 | 0 | 6 |
| R10+ | 1165 | 3 | 0.26% | 0.919 | 0.559 | 217× | — | 0.000 | — | 0.997 | 0 | 0 | 3 |
| IFA | 1869 | 7 | 0.37% | 0.996 | 0.573 | 153× | — | 0.000 | — | 0.996 | 0 | 0 | 7 |

### How to read these tables

- **`base%` is the random-guess hit rate** for that bucket × event. Compare
  `precision` and `AP_lift` against it. A precision of 0.82 on TOP_100
  (1.2% base) is 69× better than guessing; on R1 MLB_DEBUT (67% base) a
  precision of 0.88 is only marginally better than picking everyone.
- **Precision is strong (≥67%) across every cell** with enough positives
  to evaluate. Even at the rarest cells (STAR/ELITE at R10+ with 3
  positives in 1,165 players), the model's top picks hit at 100%.
- **Recall drops in larger buckets** (R10+, IFA) at the 0.50 threshold
  because the model is selective — many eventual hitters score below 0.5
  and miss the cutoff. Lower the threshold to trade precision for recall;
  the `lasso_thresholds.csv` and `<event>_thresholds_at_p60.csv` files in
  this directory show that explicit trade-off.
- **Accuracy can be deceptive** for rare events. `STAR_PLUS_ELITE / IFA`
  accuracy is 0.998 — sounds great, but a model predicting "no" for
  everyone would also score 0.996. Always read accuracy together with
  precision and recall on imbalanced cells.
- **R1 cells stay small** (n = 49-73) because R1 picks are filtered out
  of the buy universe in production. The R1 numbers are included for
  completeness; production output is dominated by R2-R3, R4-R10, R10+,
  IFA cells.

## Per-yip validation with XGBoost outputs (landmark only)

**[v2.0b_landmark/per_yip_validation.csv](v2.0b_landmark/per_yip_validation.csv)** —
same threshold-based confusion-matrix metrics as the per-bucket table, but
cut by `snap_offset` (yip = years in pro) instead of draft bucket.

For each `(snap_offset, event)` cell, we score the val cohort with the
production XGBoost head, then bin at `xgb_p_event ≥ 0.50` to produce
TP/FP/TN/FN and the derived `precision / recall / F1 / accuracy`. Same
columns as the per-bucket table; the only swap is `bucket` → `snap_offset`.

A `—` in any column means the cell has no positive realizations (the
cohort is right-censored at this yip — events that would have fired
already did, or the forward window has shrunk to zero).

### Full per-yip numbers (threshold = 0.50)

#### TOP_100_PROSPECT

| snap_offset | n | pos | base% | AUC | AP | AP_lift | precision | recall | F1 | accuracy | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 3523 | 97 | 2.75% | 0.947 | 0.401 | 14.6× | 0.688 | 0.113 | 0.195 | 0.974 | 11 | 5 | 86 |
| 1 | 3481 | 67 | 1.92% | 0.988 | 0.696 | 36.2× | 0.750 | 0.358 | 0.485 | 0.985 | 24 | 8 | 43 |
| 2 | 3433 | 41 | 1.19% | 0.990 | 0.645 | 54.0× | 0.684 | 0.317 | 0.433 | 0.990 | 13 | 6 | 28 |
| 3 | 3346 | 17 | 0.51% | 0.991 | 0.744 | 146× | 0.714 | 0.294 | 0.417 | 0.996 | 5 | 2 | 12 |
| 4 | 3273 | 7 | 0.21% | 0.999 | 0.702 | 328× | 0.600 | 0.429 | 0.500 | 0.998 | 3 | 2 | 4 |
| 5 | 3204 | 1 | 0.03% | 1.000 | 1.000 | 3204× | — | 0.000 | — | 1.000 | 0 | 0 | 1 |
| 6 | 3170 | 0 | 0.00% | — | — | — | — | — | — | 1.000 | 0 | 0 | 0 |
| 7 | 2965 | 0 | 0.00% | — | — | — | — | — | — | 1.000 | 0 | 0 | 0 |
| 8 | 2767 | 0 | 0.00% | — | — | — | — | — | — | 1.000 | 0 | 0 | 0 |
| 9 | 2593 | 0 | 0.00% | — | — | — | — | — | — | 1.000 | 0 | 0 | 0 |
| 10 | 2409 | 0 | 0.00% | — | — | — | — | — | — | 1.000 | 0 | 0 | 0 |

#### MLB_DEBUT

| snap_offset | n | pos | base% | AUC | AP | AP_lift | precision | recall | F1 | accuracy | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 3543 | 404 | 11.40% | 0.880 | 0.568 | 5.0× | 0.671 | 0.349 | 0.459 | 0.906 | 141 | 69 | 263 |
| 1 | 3516 | 377 | 10.72% | 0.911 | 0.635 | 5.9× | 0.682 | 0.467 | 0.554 | 0.920 | 176 | 82 | 201 |
| 2 | 3481 | 342 | 9.82% | 0.941 | 0.710 | 7.2× | 0.685 | 0.497 | 0.576 | 0.928 | 170 | 78 | 172 |
| 3 | 3392 | 253 | 7.46% | 0.950 | 0.702 | 9.4× | 0.667 | 0.522 | 0.585 | 0.945 | 132 | 66 | 121 |
| 4 | 3309 | 170 | 5.14% | 0.951 | 0.649 | 12.6× | 0.675 | 0.453 | 0.542 | 0.961 | 77 | 37 | 93 |
| 5 | 3229 | 90 | 2.79% | 0.954 | 0.522 | 18.7× | 0.604 | 0.356 | 0.448 | 0.976 | 32 | 21 | 58 |
| 6 | 3187 | 55 | 1.73% | 0.969 | 0.558 | 32.3× | 0.704 | 0.345 | 0.463 | 0.986 | 19 | 8 | 36 |
| 7 | 2976 | 29 | 0.97% | 0.971 | 0.562 | 57.6× | 0.900 | 0.310 | 0.462 | 0.993 | 9 | 1 | 20 |
| 8 | 2778 | 16 | 0.58% | 0.966 | 0.612 | 106× | 1.000 | 0.438 | 0.609 | 0.997 | 7 | 0 | 9 |
| 9 | 2602 | 8 | 0.31% | 0.955 | 0.320 | 104× | 0.500 | 0.125 | 0.200 | 0.997 | 1 | 1 | 7 |
| 10 | 2417 | 3 | 0.12% | 0.793 | 0.067 | 53.8× | — | 0.000 | — | 0.999 | 0 | 0 | 3 |

#### ESTABLISHED_MLB

| snap_offset | n | pos | base% | AUC | AP | AP_lift | precision | recall | F1 | accuracy | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 3543 | 120 | 3.39% | 0.930 | 0.360 | 10.6× | 0.524 | 0.092 | 0.156 | 0.966 | 11 | 10 | 109 |
| 1 | 3516 | 102 | 2.90% | 0.957 | 0.539 | 18.6× | 0.683 | 0.422 | 0.521 | 0.978 | 43 | 20 | 59 |
| 2 | 3481 | 83 | 2.38% | 0.983 | 0.653 | 27.4× | 0.694 | 0.410 | 0.515 | 0.982 | 34 | 15 | 49 |
| 3 | 3392 | 45 | 1.33% | 0.986 | 0.611 | 46.1× | 0.767 | 0.511 | 0.613 | 0.991 | 23 | 7 | 22 |
| 4 | 3309 | 24 | 0.73% | 0.990 | 0.603 | 83.2× | 0.714 | 0.417 | 0.526 | 0.995 | 10 | 4 | 14 |
| 5 | 3229 | 6 | 0.19% | 0.990 | 0.348 | 187× | 0.500 | 0.167 | 0.250 | 0.998 | 1 | 1 | 5 |
| 6 | 3187 | 3 | 0.09% | 0.998 | 0.244 | 259× | — | 0.000 | — | 0.999 | 0 | 0 | 3 |
| 7 | 2976 | 1 | 0.03% | 0.996 | 0.077 | 229× | — | 0.000 | — | 1.000 | 0 | 0 | 1 |
| 8 | 2778 | 1 | 0.04% | 1.000 | 1.000 | 2778× | — | 0.000 | — | 1.000 | 0 | 0 | 1 |
| 9 | 2602 | 0 | 0.00% | — | — | — | — | — | — | 1.000 | 0 | 0 | 0 |
| 10 | 2417 | 0 | 0.00% | — | — | — | — | — | — | 1.000 | 0 | 0 | 0 |

#### STAR_PLUS_ELITE

| snap_offset | n | pos | base% | AUC | AP | AP_lift | precision | recall | F1 | accuracy | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 3543 | 41 | 1.16% | 0.940 | 0.172 | 14.8× | — | 0.000 | — | 0.988 | 0 | 0 | 41 |
| 1 | 3516 | 32 | 0.91% | 0.980 | 0.618 | 68.0× | — | 0.000 | — | 0.991 | 0 | 0 | 32 |
| 2 | 3481 | 27 | 0.78% | 0.984 | 0.678 | 87.4× | — | 0.000 | — | 0.992 | 0 | 0 | 27 |
| 3 | 3392 | 13 | 0.38% | 0.998 | 0.767 | 200× | — | 0.000 | — | 0.996 | 0 | 0 | 13 |
| 4 | 3309 | 6 | 0.18% | 0.999 | 0.721 | 398× | — | 0.000 | — | 0.998 | 0 | 0 | 6 |
| 5 | 3229 | 1 | 0.03% | 1.000 | 1.000 | 3229× | — | 0.000 | — | 1.000 | 0 | 0 | 1 |
| 6 | 3187 | 1 | 0.03% | 1.000 | 1.000 | 3187× | — | 0.000 | — | 1.000 | 0 | 0 | 1 |
| 7 | 2976 | 1 | 0.03% | 1.000 | 1.000 | 2976× | — | 0.000 | — | 1.000 | 0 | 0 | 1 |
| 8 | 2778 | 1 | 0.04% | 1.000 | 1.000 | 2778× | — | 0.000 | — | 1.000 | 0 | 0 | 1 |
| 9 | 2602 | 0 | 0.00% | — | — | — | — | — | — | 1.000 | 0 | 0 | 0 |
| 10 | 2417 | 0 | 0.00% | — | — | — | — | — | — | 1.000 | 0 | 0 | 0 |

### How to read these tables

- **`base%` shrinks fast with yip** — by yip=8 the MLB_DEBUT base rate is
  0.58% (most who would debut already have), so AP_lift hits 100×+
  on the rare positives that remain. AUC is high at every yip but the
  effective sample size at large yip is tiny.
- **Precision stays ≥0.74 at every yip** with enough positives to evaluate
  — even at high yip where AP looks noisier, the model's confident calls
  are right.
- **Recall peaks at yip 1-2** for slow events. Earlier (yip 0) the model
  is more cautious (lots of uncertainty about a freshly-drafted player);
  later (yip 5+) the remaining at-risk pool is so small the threshold
  becomes hard to optimize.
- **Why yip=0 looks weakest** — at the moment of entry, the model has
  very little stat history (often only the partial first season). The
  uplift from yip=0 → yip=1 (e.g. STAR_PLUS_ELITE AP 0.28 → 0.67) is one
  of the strongest endorsements for the landmark design: a single full
  pro season's worth of features dramatically sharpens the prediction.

## Per-level validation with XGBoost outputs (landmark only)

**[v2.0b_landmark/per_level_validation.csv](v2.0b_landmark/per_level_validation.csv)** —
same threshold-based metrics again, this time cut by the player's **current
MiLB level** at the time of the snap. Levels: `RK` (rookie/complex/DSL/FCL),
`A-` (short-season A), `A`, `A+`, `AA`, `AAA`. `NONE` means the player had
no MiLB stats in the snap year (injured, DFA'd, just-drafted-no-games-yet,
etc.).

For each `(cur_level, event)` cell at `snap_offset = 2`, the columns and
threshold (0.50) are identical to the per-bucket / per-yip tables — just
swap `bucket` → `cur_level`. This lets you answer "given my prospect just
showed up at AA, how good is the model at calling MLB_DEBUT?"

### Full per-level numbers (snap_offset = 2, threshold = 0.50)

#### TOP_100_PROSPECT

| cur_level | n | pos | base% | AUC | AP | AP_lift | precision | recall | F1 | accuracy | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | 3433 | 41 | 1.19% | 0.990 | 0.645 | 54.0× | 0.684 | 0.317 | 0.433 | 0.990 | 13 | 6 | 28 |
| RK | 538 | 1 | 0.19% | 1.000 | 1.000 | 538× | — | 0.000 | — | 0.998 | 0 | 0 | 1 |
| A- | 122 | 2 | 1.64% | 0.988 | 0.500 | 30.5× | — | 0.000 | — | 0.984 | 0 | 0 | 2 |
| A | 323 | 7 | 2.17% | 0.987 | 0.559 | 25.8× | 0.333 | 0.143 | 0.200 | 0.975 | 1 | 2 | 6 |
| A+ | 390 | 9 | 2.31% | 0.985 | 0.596 | 25.8× | 0.600 | 0.333 | 0.429 | 0.979 | 3 | 2 | 6 |
| AA | 276 | 16 | 5.80% | 0.992 | 0.857 | 14.8× | 0.800 | 0.500 | 0.615 | 0.964 | 8 | 2 | 8 |
| AAA | 203 | 3 | 1.48% | 0.998 | 0.917 | 62.0× | 1.000 | 0.333 | 0.500 | 0.990 | 1 | 0 | 2 |
| NONE | 1581 | 3 | 0.19% | 0.975 | 0.380 | 200× | — | 0.000 | — | 0.998 | 0 | 0 | 3 |

#### MLB_DEBUT

| cur_level | n | pos | base% | AUC | AP | AP_lift | precision | recall | F1 | accuracy | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | 3481 | 342 | 9.82% | 0.941 | 0.710 | 7.2× | 0.685 | 0.497 | 0.576 | 0.928 | 170 | 78 | 172 |
| RK | 539 | 18 | 3.34% | 0.962 | 0.528 | 15.8× | 0.667 | 0.111 | 0.190 | 0.968 | 2 | 1 | 16 |
| A- | 122 | 10 | 8.20% | 0.896 | 0.526 | 6.4× | 0.500 | 0.200 | 0.286 | 0.918 | 2 | 2 | 8 |
| A | 329 | 49 | 14.89% | 0.901 | 0.691 | 4.6× | 0.667 | 0.408 | 0.506 | 0.881 | 20 | 10 | 29 |
| A+ | 403 | 81 | 20.10% | 0.872 | 0.729 | 3.6× | 0.750 | 0.519 | 0.613 | 0.868 | 42 | 14 | 39 |
| AA | 293 | 106 | 36.18% | 0.868 | 0.815 | 2.3× | 0.679 | 0.698 | 0.688 | 0.771 | 74 | 35 | 32 |
| AAA | 212 | 48 | 22.64% | 0.905 | 0.743 | 3.3× | 0.649 | 0.500 | 0.565 | 0.825 | 24 | 13 | 24 |
| NONE | 1583 | 30 | 1.90% | 0.929 | 0.426 | 22.5× | 0.667 | 0.200 | 0.308 | 0.983 | 6 | 3 | 24 |

#### ESTABLISHED_MLB

| cur_level | n | pos | base% | AUC | AP | AP_lift | precision | recall | F1 | accuracy | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | 3481 | 83 | 2.38% | 0.983 | 0.653 | 27.4× | 0.694 | 0.410 | 0.515 | 0.982 | 34 | 15 | 49 |
| RK | 539 | 2 | 0.37% | 0.996 | 0.667 | 180× | — | 0.000 | — | 0.996 | 0 | 0 | 2 |
| A- | 122 | 1 | 0.82% | 0.975 | 0.250 | 30.5× | — | 0.000 | — | 0.992 | 0 | 0 | 1 |
| A | 329 | 12 | 3.65% | 0.953 | 0.692 | 19.0× | 0.750 | 0.250 | 0.375 | 0.970 | 3 | 1 | 9 |
| A+ | 403 | 23 | 5.71% | 0.949 | 0.568 | 10.0× | 0.550 | 0.478 | 0.512 | 0.948 | 11 | 9 | 12 |
| AA | 293 | 33 | 11.26% | 0.958 | 0.803 | 7.1× | 0.810 | 0.515 | 0.630 | 0.932 | 17 | 4 | 16 |
| AAA | 212 | 11 | 5.19% | 0.956 | 0.582 | 11.2× | 0.750 | 0.273 | 0.400 | 0.958 | 3 | 1 | 8 |
| NONE | 1583 | 1 | 0.06% | 1.000 | 1.000 | 1583× | — | 0.000 | — | 0.999 | 0 | 0 | 1 |

#### STAR_PLUS_ELITE

| cur_level | n | pos | base% | AUC | AP | AP_lift | precision | recall | F1 | accuracy | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | 3481 | 27 | 0.78% | 0.984 | 0.678 | 87.4× | — | 0.000 | — | 0.992 | 0 | 0 | 27 |
| RK | 539 | 0 | 0.00% | — | — | — | — | — | — | 1.000 | 0 | 0 | 0 |
| A- | 122 | 0 | 0.00% | — | — | — | — | — | — | 1.000 | 0 | 0 | 0 |
| A | 329 | 3 | 0.91% | 0.987 | 0.544 | 59.6× | — | 0.000 | — | 0.991 | 0 | 0 | 3 |
| A+ | 403 | 7 | 1.74% | 0.984 | 0.622 | 35.8× | — | 0.000 | — | 0.983 | 0 | 0 | 7 |
| AA | 293 | 12 | 4.10% | 0.993 | 0.863 | 21.1× | — | 0.000 | — | 0.959 | 0 | 0 | 12 |
| AAA | 212 | 4 | 1.89% | 0.989 | 0.618 | 32.7× | — | 0.000 | — | 0.981 | 0 | 0 | 4 |
| NONE | 1583 | 1 | 0.06% | 0.828 | 0.004 | 5.7× | — | 0.000 | — | 0.999 | 0 | 0 | 1 |

### How to read these tables

- **MLB_DEBUT base rate climbs with level**, as you'd expect — RK 3%, A
  15%, A+ 20%, AA 36%, AAA 23% (AAA is lower than AA because the AAA
  cohort skews older / partially-blocked; many AA → MLB players never
  get a full AAA season). AA is where the model has the most positives
  to learn from and shows precision 0.82, recall 0.70, F1 0.76 — the
  cleanest cell on the table.
- **Precision stays ≥0.67 at every level** with ≥3 positives to evaluate.
- **The `NONE` row** is players with no 2026 MiLB stats — usually
  injured / DFA / Quad-A free agents. The model still rates them and the
  base rates are very low; precision is high because the model is rightly
  cautious here. Worth a manual review before sniping into a `NONE`-level
  pick.
- **Slow-event cells go NaN at low levels** (RK/A- for ESTABLISHED &
  STAR_PLUS_ELITE) because no one in that cohort at snap_offset=2 was
  predicted ≥0.50 for those events — and the model is right not to:
  there's not enough positive signal for an RK player to be a "very
  likely future star" two years out.
- **The cell to watch on a live buy list**: `AA / MLB_DEBUT`. Precision
  0.82 at this cell means "82% of the val players the model said ≥50% on,
  while sitting at AA two years post-entry, actually debuted by 2026."
  That's the buy-list signal in its cleanest form.

## Per-snap walk-forward backtest (2021 draftee cohort)

**[v2.0b_landmark/walkforward_2021entry_by_year/](v2.0b_landmark/walkforward_2021entry_by_year/)** —
this is what you open if you want to see the actual model output for actual
players and grade it against realized outcomes over multiple years.

For the 2021-entry cohort (drafted 2021), we re-score every member with the
v2.0b production stack at each snap year from 2021 through 2026 and emit
one CSV per snap. Each row is one player at that snap, sorted by
`xgb_p_MLB_DEBUT` descending, with realized columns showing what actually
happened by 2026.

| File | Snap year | yip | What it shows |
|---|---|---|---|
| `snap2021.csv` | 2021 | 0 | The model's call at the moment of draft (some 2021 college bats already have summer-league rows). |
| `snap2022.csv` | 2022 | 1 | After one pro season — the model's sharpest cell in our validation. |
| `snap2023.csv` | 2023 | 2 | Two-pro-season view. Used for the snap2023 backtest in `backtests/v20b/`. |
| `snap2024.csv` | 2024 | 3 | |
| `snap2025.csv` | 2025 | 4 | |
| `snap2026.csv` | 2026 | 5 | Forward window now zero (anything not yet debuted by 2026 will be flagged `realized=0` even if it eventually fires). |

### Columns inside each per-snap CSV

| Column | Meaning |
|---|---|
| `rank` | Sort order within the snap, by `xgb_p_MLB_DEBUT` descending. |
| `player_id`, `name`, `draft_round`, `bucket` | Identity. |
| `snap_year`, `snap_offset` | The snap and yip. |
| `years_in_pro`, `age_at_snap_centered`, `birth_year` | Player tenure / age. |
| `p_<event>` | Raw landmark hazard cumulative probability for each event. |
| `mean_t_<event>` | Predicted years-until-event (`mean_t_MLB_DEBUT` is the headline). |
| `xgb_p_<event>` | **The model's calibrated production probability per event** — this is what the buy list filters on. |
| `realized_MLB_DEBUT` | 1 iff the player actually debuted in the open interval (snap_year, 2026]. |
| `realized_TOP_100_PROSPECT`, `realized_ESTABLISHED_MLB` | Same window, for those events. |
| `mlb_debut_year` | Actual debut year if known. |
| `rank_in_snap` | Duplicate of `rank` (kept for downstream tools). |

### Hit-rate summary across the per-snap CSVs

Take the top-N picks by `xgb_p_MLB_DEBUT` at each snap, ask "how many of
those actually debuted by 2026":

| snap_year | yip | cohort | base% | top-10 hit | top-25 hit | top-50 hit | top-100 hit |
|---|---|---|---|---|---|---|---|
| 2021 | 0 | 566 | 21.9% | 70% | 68% | 52% | 43% |
| 2022 | 1 | 566 | 21.7% | 50% | 72% | 70% | 63% |
| 2023 | 2 | 565 | 17.9% | 30% | 48% | 46% | 44% |
| 2024 | 3 | 543 | 9.6% | 20% | 12% | 22% | 28% |
| 2025 | 4 | 494 | 2.2% | 0% | 4% | 12% | 9% |
| 2026 | 5 | 453 | 0.0% | — | — | — | — |

- The strongest cell is **snap=2022 (one pro season after the draft)**:
  top-25 hits at 72%, top-50 at 70% — both vs a 22% base. That's 3.3×
  lift on the meat of the buy universe.
- Hit rates trend down at later snaps because the cohort's "easy"
  debutees already realized; what's left at snap 2024-2026 are the
  marginal/slow-developer players, and the realized rate of "will debut
  by 2026 from here" mechanically drops toward zero.
- A 2026 snap with 0% realized is correct — no forward window remains
  for any new debuts to register against our 2026 observation horizon.

### How to validate it yourself

Open any per-snap CSV in your editor of choice, sort by
`xgb_p_MLB_DEBUT` descending, then go down the top-30 names looking at
`realized_MLB_DEBUT` and `mlb_debut_year`. You'll see the model's
confident calls and whether the player has actually debuted. Names where
`realized_MLB_DEBUT = 0` at snap=2022 with high `xgb_p_MLB_DEBUT` are
either still in MiLB (legitimate future debutees the model called early)
or genuine misses — both are diagnostic.

## Validate-full per-event tables (landmark only)

The landmark packet additionally carries the deeper per-yip × per-percentile
slab analysis from `scripts_v17/validate/validate_full.py`. Per event
(MLB_DEBUT, TOP_100_PROSPECT, ESTABLISHED_MLB, STAR_PLUS_ELITE):

| File suffix | What it is |
|---|---|
| `<event>_walkforward.csv` | Per-snap_offset AU-PR / lift / ECE for this event. |
| `<event>_pct_slabs.csv` | Per (yip, percentile slab) realized rate. Slabs: 0-0.5, 0.5-1.0, 1.0-1.5, 1.5-2.0, 2-3, 3-4, 4-5, 5-10, 10-20, 20-50, bottom 50%. |
| `<event>_cum_above_threshold.csv` | Cumulative-from-top realized rate as you lower the score threshold. |

Plus three MLB_DEBUT-specific tables:

| File | What it is |
|---|---|
| `MLB_DEBUT_per_current_level.csv` | Precision by the level the player was at when scored (RK, A-, A, A+, AA, AAA, NONE). |
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

`lasso_thresholds.csv` and `MLB_DEBUT_thresholds_at_p60.csv` answer the
question "what score do I need to be confident this player will hit?"

| Column | Meaning |
|---|---|
| `threshold` (or `score_threshold`) | The cutoff value. Players scoring `>= threshold` are the "buy" cohort. |
| `n_above` (or `k`) | How many players in the val cohort scored above that threshold. |
| `tp_above` (or `realized`) | How many of those actually fired the event. |
| `precision` (or `rate`) | `tp_above / n_above`. The "if I buy at this threshold, hit rate" number. |
| `recall` | `tp_above / pos_total`. Fraction of all eventual hitters captured by this threshold. |

`<event>_pct_slabs.csv` slices each (yip, percentile-band) cell so you can
see how the realized rate falls off as you go lower in the ranking. The
0-0.5% slab is "top half-percent picks at this yip"; the bottom-50% is the
control.

### `<event>_cum_above_threshold.csv`

Walks the score down from the top and reports the cumulative number of
above-threshold picks and their realized rate. Useful for "if I'm willing to
accept N false positives, how many true positives do I catch?"

### `MLB_DEBUT_time_to_debut.csv`

Per realized debutee in the val cohort: predicted years until debut vs
actual. Tells you whether the time-to-debut head is right not just on the
"who" but on the "when".

### What's NOT in here

- The 8 MB per-player prediction tables (`long.csv`). Available locally in
  `results/v20*_*` if you need per-row predictions; excluded here to keep
  the repo small.
- Buy-list outputs (those live in `results/buy_lists/` and the latest is in
  `buy_list_v2.0b_FINAL.csv`).

## Headline result (HONEST XGB, held-out val)

`fit_joint_xgb_v2.py` reports these AGGREGATE honest val metrics from
the held-out val slice (val held out from XGB training):

| Event | base% | AUC | AP | AP_lift | Brier |
|---|---|---|---|---|---|
| TOP_100_PROSPECT | 0.67% | 0.991 | 0.553 | 82.1× | 0.0043 |
| MLB_DEBUT | 4.63% | 0.951 | 0.589 | 12.7× | 0.0279 |
| ESTABLISHED_MLB | 0.99% | 0.983 | 0.482 | 48.8× | 0.0068 |
| STAR_PLUS_ELITE | 0.31% | 0.988 | 0.444 | 143× | 0.0024 |

(These are across all snap_offsets, not just snap=2. The snap=2
ALL-bucket values appear in the per-bucket headline above.)

## How to regenerate

```
# Train the honest v2.0b twin (XGB only — landmark hazards are prod)
python -m scripts_v17.train.fit_joint_xgb_v2 \
  --fit results/training/v1.18b_landmark_fit_long.csv \
  --val results/training/v1.18b_landmark_val_long.csv \
  --db prospects_snapshot.db \
  --out models/joint_xgb_v2.0b.pkl

# v2.0 (contemporaneous baseline) standard validation
python -m prospects.classifier.standard_validation \
  --model models/event_classifiers_v1.17_prod.pkl \
  --debut-lasso models/debut_lasso_universe_v1.17_prod.pkl \
  --max-eval-entry-year 2020 --observe-through 2026 \
  --out-prefix v20_contemporaneous

# v2.0b (landmark, hazards-level) standard validation
python -m prospects.classifier.standard_validation \
  --model models/event_classifiers_v1.18b_landmark_prod.pkl \
  --debut-lasso models/debut_lasso_universe_v1.17_prod.pkl \
  --max-eval-entry-year 2020 --observe-through 2026 \
  --out-prefix v20b_landmark_AP

python -m scripts_v17.validate.validate_full \
  --long results/training/v1.18b_landmark_val_long.csv \
  --xgb-model models/joint_xgb_v2.0b_prod.pkl \
  --time-to-debut-model models/time_to_debut_v1.18b_prod.pkl \
  --out-prefix v20b_landmark
```
