# v2.0b honest validation (no hazard-layer leakage)

These are the **real** held-out val numbers for v2.0b. The validation
tables published earlier under [v2.0b_landmark/](../v2.0b_landmark/)
trained their landmark hazards on 100% of the panel — including the val
pids — so the hazard features fed to the XGB at val time came from a
model that had already seen those pids during training. That XGB-layer
honesty without hazard-layer honesty is not enough; the numbers there
were optimistic.

These tables are honest at **both** layers:

- **Hazards** trained with `train_mask` excluding val pids
  (`scratch/v20b_honest/hazards.pkl`, derived from a 71% / 14.5% / 14.5%
  HAZ / FIT / VAL split — see `scripts_v17/train/train_v2_0b_honest.py`).
- **XGB head** trained on FIT, validated on VAL (same as before).

## Setups compared

| Setup | Hazards trained on | XGB trained on | Val scored with |
|---|---|---|---|
| **Honest non-OOF** ([`honest_nonoof_val_metrics.csv`](honest_nonoof_val_metrics.csv)) | 71% (excludes fit + val) | FIT (10%, ~34k rows) | hazards trained on 71% |
| **OOF v2** ([`oof_val_metrics.csv`](oof_val_metrics.csv)) | leave-one-out per fold (75% per fold) | OOF stacked (248k rows) | `hazards_full` trained on 90% (excludes val) |

## Headline numbers (weighted-AP, MLB_DEBUT 2× weight)

| Setup | TOP_100 AP | MLB_DEBUT AP | ESTABLISHED AP | STAR_PLUS_ELITE AP | weighted-AP |
|---|---:|---:|---:|---:|---:|
| Honest non-OOF | 0.393 | 0.472 | 0.343 | 0.148 | **0.366** |
| OOF v2 | **0.489** | **0.500** | 0.328 | **0.191** | **0.401** |
| Δ vs non-OOF | +0.096 | +0.027 | −0.015 | +0.043 | **+0.035** |

OOF wins on 3 of 4 heads and the headline weighted-AP. The 8× more
honest training data (248k OOF-honest rows vs 34k non-OOF rows) beats the
small train/val distribution mismatch that OOF introduces.

## Retraction of prior published numbers

The tables under [v2.0b_landmark/](../v2.0b_landmark/) were generated
with `event_classifiers_v1.18b_landmark_prod.pkl` (hazards trained on
100% of panel, val pids included). They were honest at the XGB layer
only. Use these honest packets instead.

- ~~`TOP_100_PROSPECT AP=0.553`~~ (leaky) → **0.489** (honest OOF)
- ~~`MLB_DEBUT AP=0.589`~~ (leaky) → **0.500** (honest OOF)
- ~~`ESTABLISHED_MLB AP=0.482`~~ (leaky) → **0.328** (honest OOF)
- ~~`STAR_PLUS_ELITE AP=0.444`~~ (leaky) → **0.191** (honest OOF)
- ~~weighted-AP 0.531~~ (leaky) → **0.401** (honest OOF)

The leak inflated weighted-AP by ~32%. The shape of which events do
well (TOP_100 > MLB_DEBUT > ESTABLISHED > STAR_PLUS_ELITE on rarity, but
not all in lift) holds up — the absolute numbers were the problem.

## Reproducibility

Both packets reproduce from a single panel cache + the splits already
on disk:

```
# Honest non-OOF
python -m scripts_v17.train.train_v2_0b_honest
  -> models/joint_xgb_v2.0b_honest.pkl
  -> results/training/v2.0b_honest_val_metrics.json

# OOF v2
python -m scripts_v17.train.run_v2_0b_oof
python -m scripts_v17.train.finalize_v2_0b_oof
  -> models/joint_xgb_v2.0b_oof.pkl
```

Both use `v17_prod_val_pids.txt` as the val split, so the val cohort is
the same 4,630 pids → 34,164 rows after entry≤2020 and eligibility
filters.
