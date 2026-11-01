# runs

Only the evaluation reports are in git. Everything else here is local.

| Folder | Contents |
|---|---|
| `current/` | Production: models, training data, scored snapshot and the buy sheet (`buy_lists/final.csv`) |
| `current/evaluation_v3/` | [Metrics for the published v3.5 model](current/evaluation_v3/README.md) (in git) |
| `current/evaluation/` | [Metrics for the v2.4 base layer](current/evaluation/README.md) (in git) |
| `backups/` | Restore points written before a promotion |
| `experiments/` | Experiment output. Some folders are live: the retrain writes `exp_cdf_timing5/` and reads it back |

Pass `--tag <name>` to a pipeline command to write to `runs/<name>/`.
