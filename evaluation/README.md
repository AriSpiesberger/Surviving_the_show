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
| `threshold` | The XGBoost probability cutoff used for the confusion-matrix metrics. Fixed at **0.50**. |
| `tp`, `fp`, `tn`, `fn` | True/false positives/negatives at `xgb_p_event ≥ 0.50`. |
| `predicted_positives` | `tp + fp` — how many players the model said "yes" to. |
| `precision` | `tp / (tp + fp)`. Of players the model picked, fraction that hit. |
| `recall` | `tp / (tp + fn)`. Of players who hit, fraction the model picked. |
| `f1` | Harmonic mean of precision and recall — the standard balanced summary at this threshold. |
| `accuracy` | `(tp + tn) / n`. Total correct rate. **Warning**: for rare events (STAR/ELITE base rate ~0.4%), accuracy is near 1.00 even for a model that predicts "no" for everyone; use F1, precision, recall instead. |

Buckets: `ALL, R1, R2-R3, R4-R10, R10+, IFA`. The `ALL` row aggregates the
full val cohort.

### Full per-bucket numbers (per-event eligibility, threshold=0.50)

The CSV linked above has the exact values below in machine-readable form.

#### TOP_100_PROSPECT

| bucket | n | pos | base% | AUC | AP | AP_lift | precision | recall | F1 | accuracy | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | 34164 | 230 | 0.67% | 0.985 | 0.483 | 71.8× | 0.738 | 0.196 | 0.309 | 0.994 | 45 | 16 | 185 |
| R1 | 388 | 80 | 20.62% | 0.931 | 0.757 | 3.7× | 0.833 | 0.375 | 0.517 | 0.856 | 30 | 6 | 50 |
| R2-R3 | 707 | 62 | 8.77% | 0.892 | 0.430 | 4.9× | 0.417 | 0.081 | 0.135 | 0.909 | 5 | 7 | 57 |
| R4-R10 | 2484 | 20 | 0.81% | 0.937 | 0.231 | 28.7× | 1.000 | 0.050 | 0.095 | 0.992 | 1 | 0 | 19 |
| R10+ | 11788 | 9 | 0.08% | 0.949 | 0.354 | 463.1× | 1.000 | 0.111 | 0.200 | 0.999 | 1 | 0 | 8 |
| IFA | 18797 | 59 | 0.31% | 0.986 | 0.349 | 111.3× | 0.727 | 0.136 | 0.229 | 0.997 | 8 | 3 | 51 |

#### MLB_DEBUT

| bucket | n | pos | base% | AUC | AP | AP_lift | precision | recall | F1 | accuracy | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | 34430 | 1747 | 5.07% | 0.937 | 0.564 | 11.1× | 0.643 | 0.388 | 0.484 | 0.958 | 678 | 376 | 1069 |
| R1 | 491 | 240 | 48.88% | 0.895 | 0.898 | 1.8× | 0.814 | 0.800 | 0.807 | 0.813 | 192 | 44 | 48 |
| R2-R3 | 755 | 256 | 33.91% | 0.826 | 0.711 | 2.1× | 0.651 | 0.582 | 0.614 | 0.752 | 149 | 80 | 107 |
| R4-R10 | 2487 | 279 | 11.22% | 0.855 | 0.434 | 3.9× | 0.474 | 0.330 | 0.389 | 0.884 | 92 | 102 | 187 |
| R10+ | 11790 | 383 | 3.25% | 0.899 | 0.282 | 8.7× | 0.451 | 0.107 | 0.173 | 0.967 | 41 | 50 | 342 |
| IFA | 18907 | 589 | 3.12% | 0.943 | 0.524 | 16.8× | 0.671 | 0.346 | 0.457 | 0.974 | 204 | 100 | 385 |

#### ESTABLISHED_MLB

| bucket | n | pos | base% | AUC | AP | AP_lift | precision | recall | F1 | accuracy | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | 34430 | 385 | 1.12% | 0.976 | 0.362 | 32.4× | 0.592 | 0.117 | 0.195 | 0.989 | 45 | 31 | 340 |
| R1 | 491 | 85 | 17.31% | 0.878 | 0.530 | 3.1× | 0.606 | 0.235 | 0.339 | 0.841 | 20 | 13 | 65 |
| R2-R3 | 755 | 61 | 8.08% | 0.842 | 0.295 | 3.7× | 0.429 | 0.098 | 0.160 | 0.917 | 6 | 8 | 55 |
| R4-R10 | 2487 | 76 | 3.06% | 0.928 | 0.306 | 10.0× | 1.000 | 0.053 | 0.100 | 0.971 | 4 | 0 | 72 |
| R10+ | 11790 | 41 | 0.35% | 0.971 | 0.192 | 55.1× | 1.000 | 0.024 | 0.048 | 0.997 | 1 | 0 | 40 |
| IFA | 18907 | 122 | 0.65% | 0.981 | 0.367 | 56.8× | 0.583 | 0.115 | 0.192 | 0.994 | 14 | 10 | 108 |

#### STAR_PLUS_ELITE

| bucket | n | pos | base% | AUC | AP | AP_lift | precision | recall | F1 | accuracy | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | 34430 | 123 | 0.36% | 0.974 | 0.184 | 51.5× | — | 0.000 | — | 0.996 | 0 | 0 | 123 |
| R1 | 491 | 35 | 7.13% | 0.844 | 0.225 | 3.2× | — | 0.000 | — | 0.929 | 0 | 0 | 35 |
| R2-R3 | 755 | 11 | 1.46% | 0.914 | 0.267 | 18.3× | — | 0.000 | — | 0.985 | 0 | 0 | 11 |
| R4-R10 | 2487 | 21 | 0.84% | 0.952 | 0.248 | 29.4× | — | 0.000 | — | 0.992 | 0 | 0 | 21 |
| R10+ | 11790 | 18 | 0.15% | 0.936 | 0.178 | 116.5× | — | 0.000 | — | 0.998 | 0 | 0 | 18 |
| IFA | 18907 | 38 | 0.20% | 0.989 | 0.198 | 98.6× | — | 0.000 | — | 0.998 | 0 | 0 | 38 |

STAR_PLUS_ELITE never crosses the 0.50 cutoff in production — the AP=0.184
ranking is strong but the rare positives sit at p ≈ 0.15-0.45.  Use the
`STAR_PLUS_ELITE_cum_above_threshold.csv` / `_pct_slabs.csv` tables to
pick a lower threshold appropriate to the use case.

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
