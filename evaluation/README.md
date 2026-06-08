# Held-out validation — v2.0b landmark (production)

Reproducible evaluation of the production v2.0b landmark stack against
the **10% val player slice** of the v1.17 seed=42 split — 3,543 players
neither the hazards nor the XGB ever saw during training.

Validation universe: drafted players with `draft_year ≤ 2020` (plus
IFAs), realized window through 2026.

## Production architecture

| Layer | Model | Trained on |
|---|---|---|
| Hazards | `models/event_classifiers_v2.0b_tuned_prod.pkl` | 100% of panel (487k landmark rows). Optuna-tuned HP: `max_depth=4, max_leaf_nodes=15, lr=0.063, min_samples_leaf=70, l2=4.2, max_bins=211, max_iter=298`. |
| XGB head | `models/joint_xgb_v2.0b_oof_tuned.pkl` | OOF stacked CSV (248k OOF-honest rows where every row's hazard features came from a model that didn't see that pid). Optuna-tuned HP: `max_depth=6, lr=0.0129, min_child_weight=46, reg_lambda=6.86, subsample=0.90, colsample_bytree=0.96, best_iter=999`. |
| Inference snap | `results/scored/snap2026_v2.0b_tuned_prod_long.csv` | 37,389 prospects scored at snap=2026. |
| Buy list | `results/buy_lists/buy_list_v2.0b_TUNED_FINAL.csv` | 18 prospects at `P(MLB_DEBUT) ≥ 0.60`. |

Each landmark row: features computed `as_of S`, with `horizon_offset_k`
as an explicit feature column. Inference sets `k = step+1` per horizon
step instead of advancing yip — train and inference draw from the same
distribution.

## Headline numbers

Weighted-AP (MLB_DEBUT 2×, others 1×), per-event eligibility filters:

| Event | n | base_rate | AP | AP lift × base | AUC |
|---|---:|---:|---:|---:|---:|
| TOP_100_PROSPECT | 3,433 | 0.67% | **0.483** | 71.8× | 0.985 |
| MLB_DEBUT | 34,164 | 5.07% | **0.564** | 11.1× | 0.937 |
| ESTABLISHED_MLB | 34,164 | 1.12% | **0.362** | 32.4× | 0.976 |
| STAR_PLUS_ELITE | 34,164 | 0.36% | **0.184** | 51.5× | 0.974 |
| **weighted-AP** | | | **0.431** | | |

## File map

| File | What it is |
|---|---|
| [v2.0b_landmark/per_bucket_validation.csv](v2.0b_landmark/per_bucket_validation.csv) | One row per `(event × bucket)`. Buckets: `ALL`, `R1`, `R2-R3`, `R4-R10`, `R10+`, `IFA`. |
| [v2.0b_landmark/per_yip_validation.csv](v2.0b_landmark/per_yip_validation.csv) | One row per `(event × snap_offset)` for offsets 0..10. |
| [v2.0b_landmark/per_level_validation.csv](v2.0b_landmark/per_level_validation.csv) | One row per `(event × current_level)` for `ALL, RK, A-, A, A+, AA, AAA, NONE`. |
| [v2.0b_landmark/walkforward.csv](v2.0b_landmark/walkforward.csv) | Same as `per_yip_validation.csv`, alternate name. |
| [v2.0b_landmark/headline.json](v2.0b_landmark/headline.json) | Machine-readable summary. |

Each cell carries: `n`, `pos`, `base_rate`, `auc`, `ap`, `ap_lift`,
`threshold` (= 0.5 for buy-list cutoff view), `tp`/`fp`/`tn`/`fn`,
`precision`, `recall`, `f1`, `accuracy`, `predicted_positives`.

## Reproducing

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

# 6. Regenerate all eval tables
python -m scripts_v17.validate.regen_eval_v2_0b_honest
```
