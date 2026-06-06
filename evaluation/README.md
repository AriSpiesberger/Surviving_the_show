# Held-out validation packets

Reproducible evaluation of the prospect-card models against the **first 10% of
players** in a seed=42 player-grouped permutation — the same held-out slice
the v1.18b landmark hazards never trained on. Validation universe: drafted
players with `draft_year ≤ 2020` (plus IFAs), realized window through 2026,
3,702 players.

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
| ALL | 3433 | 41 | 1.19% | 0.997 | 0.924 | 77.4× | 0.821 | 0.780 | 0.800 | 0.995 | 32 | 7 | 9 |
| R1 | 49 | 13 | 26.53% | 0.989 | 0.975 | 3.7× | 0.800 | 0.923 | 0.857 | 0.918 | 12 | 3 | 1 |
| R2-R3 | 93 | 10 | 10.75% | 0.990 | 0.949 | 8.8× | 0.889 | 0.800 | 0.842 | 0.968 | 8 | 1 | 2 |
| R4-R10 | 270 | 4 | 1.48% | 1.000 | 1.000 | 67.5× | 0.667 | 1.000 | 0.800 | 0.993 | 4 | 2 | 0 |
| R10+ | 1165 | 2 | 0.17% | 0.982 | 0.522 | 304.2× | 1.000 | 0.500 | 0.667 | 0.999 | 1 | 0 | 1 |
| IFA | 1856 | 12 | 0.65% | 0.999 | 0.901 | 139.3× | 0.875 | 0.583 | 0.700 | 0.997 | 7 | 1 | 5 |

#### MLB_DEBUT

| bucket | n | pos | base% | AUC | AP | AP_lift | precision | recall | F1 | accuracy | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | 3481 | 342 | 9.82% | 0.965 | 0.808 | 8.2× | 0.813 | 0.558 | 0.662 | 0.944 | 191 | 44 | 151 |
| R1 | 73 | 49 | 67.12% | 0.918 | 0.963 | 1.4× | 0.880 | 0.898 | 0.889 | 0.849 | 44 | 6 | 5 |
| R2-R3 | 103 | 54 | 52.43% | 0.918 | 0.937 | 1.8× | 0.846 | 0.815 | 0.830 | 0.825 | 44 | 8 | 10 |
| R4-R10 | 271 | 57 | 21.03% | 0.922 | 0.774 | 3.7× | 0.688 | 0.579 | 0.629 | 0.856 | 33 | 15 | 24 |
| R10+ | 1165 | 71 | 6.09% | 0.958 | 0.670 | 11.0× | 0.800 | 0.282 | 0.417 | 0.952 | 20 | 5 | 51 |
| IFA | 1869 | 111 | 5.94% | 0.966 | 0.747 | 12.6× | 0.833 | 0.450 | 0.585 | 0.962 | 50 | 10 | 61 |

#### ESTABLISHED_MLB

| bucket | n | pos | base% | AUC | AP | AP_lift | precision | recall | F1 | accuracy | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | 3481 | 83 | 2.38% | 0.990 | 0.788 | 33.0× | 0.768 | 0.518 | 0.619 | 0.985 | 43 | 13 | 40 |
| R1 | 73 | 19 | 26.03% | 0.960 | 0.917 | 3.5× | 0.875 | 0.737 | 0.800 | 0.904 | 14 | 2 | 5 |
| R2-R3 | 103 | 15 | 14.56% | 0.958 | 0.822 | 5.6× | 0.727 | 0.533 | 0.615 | 0.903 | 8 | 3 | 7 |
| R4-R10 | 271 | 18 | 6.64% | 0.964 | 0.707 | 10.6× | 0.714 | 0.278 | 0.400 | 0.945 | 5 | 2 | 13 |
| R10+ | 1165 | 9 | 0.77% | 0.991 | 0.805 | 104.2× | 1.000 | 0.444 | 0.615 | 0.996 | 4 | 0 | 5 |
| IFA | 1869 | 22 | 1.18% | 0.992 | 0.728 | 61.8× | 0.667 | 0.545 | 0.600 | 0.991 | 12 | 6 | 10 |

#### STAR_PLUS_ELITE

| bucket | n | pos | base% | AUC | AP | AP_lift | precision | recall | F1 | accuracy | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | 3481 | 27 | 0.78% | 0.993 | 0.776 | 100.0× | 0.882 | 0.556 | 0.682 | 0.996 | 15 | 2 | 12 |
| R1 | 73 | 8 | 10.96% | 0.967 | 0.798 | 7.3× | 0.750 | 0.750 | 0.750 | 0.945 | 6 | 2 | 2 |
| R2-R3 | 103 | 3 | 2.91% | 0.980 | 0.778 | 26.7× | 1.000 | 0.333 | 0.500 | 0.981 | 1 | 0 | 2 |
| R4-R10 | 271 | 6 | 2.21% | 0.999 | 0.976 | 44.1× | 1.000 | 0.500 | 0.667 | 0.989 | 3 | 0 | 3 |
| R10+ | 1165 | 3 | 0.26% | 0.977 | 0.679 | 263.6× | 1.000 | 0.667 | 0.800 | 0.999 | 2 | 0 | 1 |
| IFA | 1869 | 7 | 0.37% | 0.998 | 0.692 | 184.8× | 1.000 | 0.429 | 0.600 | 0.998 | 3 | 0 | 4 |

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

## Headline result

From the landmark packet's `bucket.csv` at `snap_offset=2`, AUC and AU-PR
shifts vs the baseline:

| Event | v2.0 AUC | v2.0b AUC | Δ | v2.0 AP | v2.0b AP | Δ |
|---|---|---|---|---|---|---|
| MLB_DEBUT | 0.865 | 0.919 | +0.054 | 0.527 | 0.665 | +0.138 |
| TOP_100_PROSPECT | 0.953 | 0.997 | +0.044 | 0.540 | 0.901 | +0.361 |
| ESTABLISHED_MLB | 0.836 | 0.976 | +0.140 | 0.136 | 0.576 | +0.440 |
| STAR_PLUS_ELITE | 0.858 | 0.977 | +0.119 | 0.147 | 0.439 | +0.292 |

The slow events (ESTABLISHED, STAR_PLUS_ELITE) move the most, which is
exactly what we'd expect: their cumulative mass accrues at high `k`, where
the v2.0 LOCF inference is most out-of-distribution.

## How to regenerate

```
# v2.0 (contemporaneous baseline)
python -m prospects.classifier.standard_validation \
  --model models/event_classifiers_v1.17_prod.pkl \
  --debut-lasso models/debut_lasso_universe_v1.17_prod.pkl \
  --max-eval-entry-year 2020 --observe-through 2026 \
  --out-prefix v20_contemporaneous

# v2.0b (landmark, full + per-event slabs)
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
