# Held-out validation packets

Reproducible evaluation of the prospect-card models against the **10% val
player slice** of the v1.17 seed=42 split — 3,543 players neither the
landmark hazards nor the joint XGBoost head trained on. Validation
universe: drafted players with `draft_year ≤ 2020` (plus IFAs), realized
window through 2026. Val rows after filters: 34,430.

The numbers below describe the same production stack the buy list runs
on — calibrated tuned-hazards + tuned-XGB + isotonic per-event
calibration fit on val.

## Production stack

| Layer | Model | Trained on |
|---|---|---|
| Hazards | `models/event_classifiers_v2.0b_tuned_prod.pkl` | 100% of panel (487k landmark rows). Optuna-tuned: `max_depth=4, max_leaf_nodes=15, lr=0.063, min_samples_leaf=70, l2=4.2, max_bins=211, max_iter=298`. |
| XGB head + calibrators | `models/joint_xgb_v2.0b_prod.pkl` | Joint XGB fit on OOF stacked CSV (248k OOF-honest rows). Optuna-tuned: `max_depth=6, lr=0.0129, min_child_weight=46, reg_lambda=6.86, subsample=0.90, colsample_bytree=0.96`. Isotonic per-event calibrators fit on val. |
| Inference snap | `results/scored/snap2026_v2.0b_prod_long.csv` | 37,389 prospects at snap=2026, scored by hazards → XGB → calibrated. |
| Buy list | `results/buy_lists/buy_list_v2.0b_CALIBRATED_FINAL.csv` | 15 prospects at `P_calibrated(MLB_DEBUT) ≥ 0.60`. |

Each landmark row: features computed `as_of S`, with `horizon_offset_k`
as an explicit feature column. Inference sets `k = step+1` per horizon
step instead of advancing yip — train and inference draw from the same
distribution.

## File map — [v2.0b_landmark/](v2.0b_landmark/)

| File | What it is |
|---|---|
| `bucket.csv` | Per `(event, bucket)` cell at `snap_offset=2`. Buckets: `ALL, R1, R2-R3, R4-R10, R10+, IFA`. |
| `walkforward.csv` | Per `(event, snap_offset)` cell, offsets 0..10. |
| `per_bucket_validation.csv` | Same buckets as `bucket.csv` but the 0.60-threshold confusion-matrix view + Spearman ρ. |
| `per_yip_validation.csv` | 0.60-threshold view per `(event, snap_offset)`. |
| `per_level_validation.csv` | 0.60-threshold view per `(event, current_level)` for `ALL, RK, A-, A, A+, AA, AAA, NONE`. |
| `thresholds_at_p60_per_bucket.csv`, `_per_yip.csv`, `_per_level.csv` | Per-cell minimum threshold achieving precision ≥ 0.60. |
| `<EVENT>_pct_slabs.csv`, `_cum_above_threshold.csv`, `_walkforward.csv` | Per-event slab and walkforward analysis. |
| `MLB_DEBUT_per_current_level.csv`, `_thresholds_at_p60.csv`, `_time_to_debut.csv` | MLB_DEBUT-specific tables. |
| `headline.json` | Machine-readable summary. |

Columns in `bucket.csv` / `walkforward.csv`: `n`, `pos`, `base_rate`,
`pred_mean`, `mean_fwd_years`, `auc` + bootstrap `[auc_lo, auc_hi]`, `ap`,
`ap_lift`, `brier`, `brier_skill`, `ece`, `spiegelhalter_p`, plus
`lift@{1,5,10}%` and `recall@{1,5,10}%` with the cutoff index `k@K%`.
Columns in `per_*_validation.csv`: `n`, `pos`, `base_rate`, `auc`, `ap`,
`ap_lift`, `spearman_rho`, `spearman_p`, `threshold`, `tp`, `fp`, `tn`,
`fn`, `precision`, `recall`, `f1`, `accuracy`, `predicted_positives`.

## Per-bucket validation (calibrated, threshold = 0.60)

`spearman` is Spearman's rank correlation ρ between the model's
calibrated score and the realized outcome on the cell (higher = better
rank ordering; significance test in the CSV's `spearman_p` column).

### Full per-bucket numbers

#### TOP_100_PROSPECT

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | 34164 | 230 | 0.67% | 0.976 | 0.420 | 62.4× | 0.152 | 0.738 | 0.196 | 0.309 | 45 | 16 | 185 |
| R1 | 388 | 80 | 20.62% | 0.891 | 0.691 | 3.4× | 0.549 | 0.821 | 0.287 | 0.426 | 23 | 5 | 57 |
| R2-R3 | 707 | 62 | 8.77% | 0.864 | 0.438 | 5.0× | 0.358 | 0.667 | 0.194 | 0.300 | 12 | 6 | 50 |
| R4-R10 | 2484 | 20 | 0.81% | 0.901 | 0.073 | 9.0× | 0.128 | — | 0.000 | — | 0 | 0 | 20 |
| R10+ | 11788 | 9 | 0.08% | 0.916 | 0.338 | 443.3× | 0.045 | 1.000 | 0.333 | 0.500 | 3 | 0 | 6 |
| IFA | 18797 | 59 | 0.31% | 0.986 | 0.330 | 105.1× | 0.111 | 0.583 | 0.119 | 0.197 | 7 | 5 | 52 |

#### MLB_DEBUT

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | 34430 | 1747 | 5.07% | 0.936 | 0.534 | 10.5× | 0.334 | 0.818 | 0.197 | 0.318 | 345 | 77 | 1402 |
| R1 | 491 | 240 | 48.88% | 0.867 | 0.868 | 1.8× | 0.635 | 0.898 | 0.512 | 0.653 | 123 | 14 | 117 |
| R2-R3 | 755 | 256 | 33.91% | 0.808 | 0.685 | 2.0× | 0.505 | 0.835 | 0.297 | 0.438 | 76 | 15 | 180 |
| R4-R10 | 2487 | 279 | 11.22% | 0.838 | 0.407 | 3.6× | 0.370 | 0.653 | 0.115 | 0.195 | 32 | 17 | 247 |
| R10+ | 11790 | 383 | 3.25% | 0.900 | 0.267 | 8.2× | 0.246 | 0.526 | 0.026 | 0.050 | 10 | 9 | 373 |
| IFA | 18907 | 589 | 3.12% | 0.947 | 0.511 | 16.4× | 0.276 | 0.825 | 0.177 | 0.291 | 104 | 22 | 485 |

#### ESTABLISHED_MLB

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | 34430 | 385 | 1.12% | 0.975 | 0.359 | 32.1× | 0.215 | 0.712 | 0.096 | 0.169 | 37 | 15 | 348 |
| R1 | 491 | 85 | 17.31% | 0.880 | 0.553 | 3.2× | 0.500 | 0.714 | 0.176 | 0.283 | 15 | 6 | 70 |
| R2-R3 | 755 | 61 | 8.08% | 0.837 | 0.292 | 3.6× | 0.322 | 0.636 | 0.115 | 0.194 | 7 | 4 | 54 |
| R4-R10 | 2487 | 76 | 3.06% | 0.921 | 0.301 | 9.9× | 0.261 | 1.000 | 0.026 | 0.051 | 2 | 0 | 74 |
| R10+ | 11790 | 41 | 0.35% | 0.968 | 0.175 | 50.3× | 0.124 | 1.000 | 0.024 | 0.048 | 1 | 0 | 40 |
| IFA | 18907 | 122 | 0.65% | 0.983 | 0.368 | 57.0× | 0.180 | 0.706 | 0.098 | 0.173 | 12 | 5 | 110 |

#### STAR_PLUS_ELITE

| bucket | n | pos | base% | AUC | AP | AP_lift | spearman | precision | recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | 34430 | 123 | 0.36% | 0.969 | 0.172 | 48.2× | 0.112 | 1.000 | 0.008 | 0.016 | 1 | 0 | 122 |
| R1 | 491 | 35 | 7.13% | 0.847 | 0.235 | 3.3× | 0.312 | — | 0.000 | — | 0 | 0 | 35 |
| R2-R3 | 755 | 11 | 1.46% | 0.924 | 0.147 | 10.1× | 0.179 | — | 0.000 | — | 0 | 0 | 11 |
| R4-R10 | 2487 | 21 | 0.84% | 0.936 | 0.219 | 25.9× | 0.145 | — | 0.000 | — | 0 | 0 | 21 |
| R10+ | 11790 | 18 | 0.15% | 0.919 | 0.144 | 94.6× | 0.067 | — | 0.000 | — | 0 | 0 | 18 |
| IFA | 18907 | 38 | 0.20% | 0.991 | 0.189 | 93.8× | 0.093 | 1.000 | 0.026 | 0.051 | 1 | 0 | 37 |

STAR_PLUS_ELITE rarely crosses 0.60 — the AP=0.172 ranking is decent but
positives concentrate at p ≈ 0.15-0.30. Use the threshold-at-p60 tables
below for the tuned-per-cell operating points.

## Threshold-at-precision-≥-0.60

For each slice, find the **lowest** XGB probability threshold whose
precision among players-at-or-above is ≥ 0.60. This is the buy-list
operating point — "what cutoff should I trust to be 60% right?" — and
the recall reported is "of all eventual hitters in this slice, what
fraction did the model's confident-buys catch?".

CSVs: [`thresholds_at_p60_per_bucket.csv`](v2.0b_landmark/thresholds_at_p60_per_bucket.csv), [`thresholds_at_p60_per_yip.csv`](v2.0b_landmark/thresholds_at_p60_per_yip.csv), [`thresholds_at_p60_per_level.csv`](v2.0b_landmark/thresholds_at_p60_per_level.csv).

`lift` column is `precision / base_rate`. Dash entries mean the slice
has no threshold whose precision reaches 0.60 (too few positives).

## Statistics glossary

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

### Discrimination metrics

| Metric | What it measures |
|---|---|
| `auc` | Area Under the ROC Curve. 0.5 = random, 1.0 = perfect. Insensitive to class imbalance. |
| `auc_lo`, `auc_hi` | 95% bootstrap confidence interval on AUC (200 resamples). |
| **`ap`** | **Average Precision = Area Under the Precision-Recall Curve (AU-PR).** Far more informative than AUC for rare events. |
| **`ap_lift`** | `ap / base_rate`. The "you got X× better than random precision-weighted ranking" multiplier. |
| **`spearman_rho`** | Spearman rank correlation between the model's calibrated score and the realized 0/1 outcome on the cell. Captures monotonic agreement between predicted rank and outcome. 0 = no relationship, +1 = perfectly increasing, −1 = perfectly inverted. Threshold-free, so a useful sanity sibling to AP. `spearman_p` is its two-sided p-value. |

### Calibration metrics

| Metric | What it measures |
|---|---|
| `brier` | Brier score = MSE of predictions vs realized 0/1. Lower is better. |
| **`brier_skill`** | Brier Skill Score = `1 − brier / brier_baseline`, where the baseline predicts `base_rate` for everyone. Positive = better than baseline. |
| **`ece`** | Expected Calibration Error. Bins predictions into 10 buckets, computes `|mean(pred) − mean(realized)|` per bin, averages weighted by bin size. Low (< 0.05) = well-calibrated. |
| `spiegelhalter_p` | Two-sided p-value for "the model is calibrated" (H0). Small (< 0.05) = reject calibration. |

### Top-K precision/recall metrics

| Metric | What it means |
|---|---|
| `lift@K%` | Precision in top K% by predicted probability, divided by `base_rate`. |
| `recall@K%` | Fraction of all real positives captured in the top K%. |
| `k@K%` | Number of players in the top-K% slice. |

## Headline result

Per-event eligibility filter, weighted-AP with MLB_DEBUT 2× weight,
calibrated production stack:

| Event | n | base_rate | AP | AP lift × base | AUC |
|---|---:|---:|---:|---:|---:|
| TOP_100_PROSPECT | 34,164 | 0.67% | **0.420** | 62.4× | 0.976 |
| MLB_DEBUT | 34,430 | 5.07% | **0.534** | 10.5× | 0.936 |
| ESTABLISHED_MLB | 34,430 | 1.12% | **0.359** | 32.1× | 0.975 |
| STAR_PLUS_ELITE | 34,430 | 0.36% | **0.172** | 48.2× | 0.969 |
| **weighted-AP** | | | **0.404** | | |

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

# 5. Train tuned prod hazards on 100% + score snap=2026
python -m scripts_v17.train.train_v2_0b_tuned_prod

# 6. Calibrate (isotonic per-event on val, applied to snap)
python -m scripts_v17.train.calibrate_v2_0b_tuned

# 7. Regenerate all eval tables (compact + detailed)
python -m scripts_v17.validate.regen_eval_v2_0b_honest --threshold 0.60
python -m scripts_v17.validate.regen_full_eval_v2_0b
```
