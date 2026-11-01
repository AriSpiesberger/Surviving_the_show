# Model experiments

Retired experiment scripts that production does not import. Most results are negative or a wash,
which is the point of keeping the record.

Run a script from the repo root with the package installed:

```bash
python research/model_experiments/exp_v3_all.py --help
```

Output goes to `runs/experiments/<name>/`, which is local only. Experiments that production depends
on stay in `prospects/model/train/`.

## What shipped

| Experiment | Result | Script |
|---|---|---|
| Per-event blends of calibrated components | Beat v2.4 on every event; star+ AP at 6 years rose 0.135 to 0.191 | `exp_v3_all`, `exp_combo` |
| Sequence encoder logits fed to the GBM | Debut AP at 3 years +0.020 in era, +0.054 out of era | `exp_v3_inera`, `exp_ap_opt` |
| Recent-cohort augmentation | Kept in v2.4; its measured gain predates the leak fixes and is unverified | `exp_walkforward3` |
| Pre-2005 draft classes | Weighted AP at 6 years rose 0.427 to 0.463, then 0.4628 to 0.4665 with 1988-1995 | `tools/history/` |

## What did not ship

| Idea | Result | Script |
|---|---|---|
| Temporal-difference labels, rank:map, playing-time teacher | +0.010 on a 25% sample, gone at full size (0.4587 against 0.4628) | `exp_v3_all`, `spec_compare`, `spec_subsets` |
| Scouting report text embeddings | Top-100 AP -0.034 (significant) | `exp_resid_value` |
| Transactions, MiLB honours, org development, showcases | No AP added over production v3 | `exp_joint_feats`, `exp_resid_value` |
| Historical comparables (kNN) | Debut +0.013, but top-100 and star+ fell | `exp_knn_comps` |
| Synthetic twins and row-level augmentation | Established AP fell; copies carry no new information | `exp_v3_all`, `tools/features/build_twins.py` |
| Multi-task encoder | Worse on debut, top-100 and star+ | `exp_seq_mt` |
| Encoder augmentation and deeper transformers | Neutral or negative | `exp_seq_e` |
| Self-supervised pretraining | A wash at full size | `exp_seq_p` |
| Learned stacker | Lost AP on rare events, though calibrated well | `exp_stacker` |
| Dedicated ceiling-event heads and parameter grids | Replaced by the component blends | `exp_ceiling_heads`, `exp_ceiling_grid` |
| Single-stage model and A/B1 blend | Small gain out of era, tie in era; earlier stack-overconfidence claim retracted | `exp_macro_inera`, `exp_walkforward_h` |
| Calibrator variants | All tie the deployed calibrator | `exp_cal_topend`, `exp_cal_bagfit`, `exp_cal_null` |

## Lessons

- A 25% screen overstates ideas that only reduce variance. Confirm at 50% before a full build.
- Compare arms within one run. Cross-run noise is about 0.005-0.01 AP, which a validation bootstrap
  does not see.
- A walk-forward test that feeds hazard outputs to a second stage must use out-of-fold hazards for
  training rows. `exp_walkforward_h.py` is the correct template; results from `exp_walkforward2/3/4`
  stack arms are not comparable.
- New real players helped. Most other levers did not.

## All scripts

Every script's docstring states its question and how to run it. `exp_blend`, `exp_hazard_form`,
`exp_rank_debut`, `exp_temporal_encoder`, `exp_walkforward` and `exp_walkforward4`
are earlier screens not listed above.
