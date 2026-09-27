# runs/

| Folder | What it holds |
|---|---|
| `current/` | **Production.** Models, training longs, scored snapshot, evaluation packet, and the live buy sheet (`buy_lists/final.csv`). Everything the weekly jobs read and write. |
| `current/buy_lists/history/` | Earlier sheets and side-by-side variants, kept for comparison. |
| `backups/` | Point-in-time copies of `current/`: automatic `backup_v2.1c_<date>` snapshots from `promote_v22`, and manual backups before big switches (e.g. `current_pre_ext_20260925`). |
| `experiments/` | One folder per experiment (the default `--out-dir` of each `prospects/model/train/exp_*.py`), plus the `ext` / `ext2` pre-2005 builds and the older `hz*` / `fs*` hazard and feature-selection runs. Not read by production. |

Loose run logs live in `logs/runs/`.
