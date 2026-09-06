# Surviving the Show

A model that scores minor-league baseball prospects on their probability of
reaching MLB milestones, and turns those scores into a card **buy list** —
players whose Bowman 1st Chrome autos look cheap relative to the model's
outlook.

Current model: **v2.4** — landmark hazards feeding a horizon-conditional
joint XGBoost **bag** (monotone-in-h, 160 raw panel features passed through,
per-event h/yip calibration fit on OOF over 2008+ snaps, debut timing read
off the calibrated CDF). On the verified-clean holdout: debut AP@3y **0.61
vs 0.56** for the v2.1c recipe (+10%), weighted AP @ h=6 **0.47 vs 0.42**,
timing MAE 1.04 vs 1.29 — see
[`runs/current/evaluation/README.md`](runs/current/evaluation/README.md),
including its 2026-09-05 split-leak correction note (metrics reported Sep
1–5, e.g. "0.50–0.59 weighted AP", were inflated by a stale fold partition;
absolute numbers are not comparable across that boundary).
v2.4 adds recent-cohort augmentation: 2021+ entries' resolved short-horizon
outcomes join training (walk-forward proven: +0.04–0.07 out-of-era debut@3
AP, and it trims era-drift). Artifacts are `models/joint_xgb_v2.4.pkl` +
`models/calibrators_v2.4.pkl` (+ `models/hazards.pkl`, default HP — the
capacity retune measured neutral), selected via `--xgb`/`--calibrators`; the
weekly retrain (`deploy/weekly_score.py`) produces the full v2.4 stack
automatically (Stage C, ported 2026-09-05/06).

## Layout

```
prospects/            the importable package — one folder per concept
├── config.py         paths, database, environment, run namespaces
├── core/             schema, SQLite storage, outcome labels
├── data/             download & organize: sources/, backfills/, pull.py
├── features/         the panel: scouting, grades, prorate, panel/
├── model/            hazards/, joint.py, train/, pipelines/, calibration, thresholds
├── evaluation/       held-out metrics (run.py) + README generator (report.py)
├── buylist/          build.py — universe filter, scoring, price join
├── market/           eBay client + price aggregation
└── deploy/           the scheduled jobs (daily_data, daily_prices, weekly_score, alerts)

tools/                hand-run maintenance CLIs (price pulls, scouting scrape, backfills)
ops/                  Windows Task Scheduler wrappers (run_job.ps1, register_tasks.ps1)
tests/                pytest suite
reference/            hand-curated inputs (prices, baseballcube, scouting grades, baselines)

runs/current/         the live model run — everything one run produces:
├── models/           hazards.pkl, joint_xgb.pkl, calibrators.pkl, timing.pkl, ...
├── training/         fit/val longs, OOF longs, panel caches, pid lists
├── scored/           snap scoring output
├── buy_lists/        all_scored.csv, final.csv
└── evaluation/       metric CSVs + headline.json + README.md   (committed)

archive/              superseded runs, DB backups, retired artifacts (git-ignored)
```

Every path resolves through `prospects.config`. There are no scattered path
literals: a run's artifacts all live under `runs/<tag>/` (default `current`),
and the model's feature contract (event list, feature order, horizons) lives in
`prospects.model.joint`.

## Databases

- `prospects.db` — the **live** database; `deploy/daily_data.py` appends
  current-season stats nightly.
- `prospects_snapshot.db` — the **modeling** database. `deploy/weekly_score.py`
  copies the live DB over it at the start of each weekly run, so a long training
  job reads a stable file. (The name is historical; it is not frozen.)

## Run it

Install once (editable, so imports and `python -m` just work):

```bash
pip install -e .
```

Rebuild the buy list from the current run's artifacts — **no arguments**, every
default points at `runs/current/`:

```bash
python -m prospects.buylist.build
```

Retrain end to end (the weekly cycle) and re-evaluate:

```bash
python -m prospects.model.pipelines.oof     # OOF folds + hazards + conditional joint XGB
python -m prospects.model.pipelines.prod    # full-panel prod hazards + score 2026 + buy list
python -m prospects.evaluation.run          # held-out metrics -> runs/current/evaluation/
python -m prospects.evaluation.report       # regenerate that run's README
```

A different run tag (backtest, A/B) writes to `runs/<tag>/` instead — pass
`--tag <name>` to the pipeline commands, or set `RUN_TAG=<name>`.

## Scheduled deploy (Windows Task Scheduler)

Four tasks drive the live loop, all through `ops/run_job.ps1`:

| Task | When | Module |
|---|---|---|
| daily_data | 00:30 daily | `prospects.deploy.daily_data` — pull current-season stats |
| weekly_score | Mon 05:00 | `prospects.deploy.weekly_score` — full retrain + buy list |
| daily_prices | 09:00 daily | `prospects.deploy.daily_prices` — eBay pull for buy-list names |
| daily_digest | 10:00 daily | `prospects.deploy.alerts` — email digest |

Register (idempotent) with:

```powershell
powershell -ExecutionPolicy Bypass -File ops\register_tasks.ps1
```

## Docs

- [`docs/runbook.md`](docs/runbook.md) — operating the pipeline, retraining, promoting a run.
- [`docs/architecture.md`](docs/architecture.md) — the model, stage by stage, and how a run is namespaced.
- [`docs/data.md`](docs/data.md) — data sources, the two databases, and the backfills.
