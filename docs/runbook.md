# Runbook

## Scheduled jobs

`ops/run_job.ps1` runs each job. It loads `.env`, sets local paths and logs to
`logs/<job>_<date>.log`.

| Task | When | Module |
|---|---|---|
| `SurvivingShow_daily_data` | Daily 00:30 | `prospects.deploy.daily_data`: current-season stats |
| `SurvivingShow_weekly_score` | Monday 05:00 | `prospects.deploy.weekly_score`: full retrain and buy list |
| `SurvivingShow_daily_prices` | Daily 09:00 | `prospects.deploy.daily_prices`: eBay prices |
| `SurvivingShow_daily_digest` | Daily 10:00 | `prospects.deploy.alerts`: email digest |

```powershell
# Register or update all four tasks (safe to repeat)
powershell -ExecutionPolicy Bypass -File ops\register_tasks.ps1

# Inspect, pause, resume, or run one now
Get-ScheduledTask -TaskName 'SurvivingShow_*'
Disable-ScheduledTask -TaskName 'SurvivingShow_weekly_score'
Enable-ScheduledTask -TaskName 'SurvivingShow_weekly_score'
Start-ScheduledTask -TaskName 'SurvivingShow_daily_digest'
```

## The weekly retrain

`weekly_score` copies `prospects.db` over `prospects_snapshot.db`, then runs:

1. Stage A: panel rebuild, landmark hazards and timing
2. Split, OOF folds, production hazards and the production joint model
3. v2.4 (held-out) and v2.5 (all players) joint layers
4. v3 held-out and v3.5 full builds, recalibration and thresholds
5. `tools/leak_audit.py`, which stops the run if any check fails
6. The v2.5 fallback sheet, then the published v3.5 sheet

The joint-layer step writes to `runs/experiments/exp_cdf_timing5/` and `promote_v22` reads from
there, so that folder is part of the live pipeline. Never delete it.

Run the leak audit by hand after any data refresh or feature change:

```bash
python tools/leak_audit.py
```

## Manual runs

Rebuild the buy list from existing artifacts with the published v3.5 model:

```bash
python -m prospects.deploy.weekly_score --season 2026 --skip-retrain --buylist-only
```

A bare `prospects.buylist.build` defaults to the older OOF bundle, not v3.5.

Run the whole pipeline, or part of it:

```bash
python -m prospects.refresh --dry-run        # print the plan
python -m prospects.refresh --from oof       # resume after a failure
python -m prospects.refresh --only buylist   # one step
```

## Testing a candidate

```bash
python -m prospects.refresh --tag cand --from split
```

Results land in `runs/cand/`. Compare `runs/cand/evaluation/headline.json` with
`runs/current/evaluation/headline.json`. To compare two bundles on identical rows, use
`prospects/evaluation/paired.py`.

## Promoting a candidate

1. Move the old run aside: `mv runs/current runs/backups/current_pre_<name>_<date>`.
2. Move the new run in: `mv runs/cand runs/current`.
3. Run `python tools/leak_audit.py`.
4. Rebuild the sheet and review `runs/current/buy_lists/final.csv`.

## Data refreshes

```bash
python -m prospects.deploy.daily_data       # current-season stats into prospects.db
python -m prospects.data.pull --phase all   # full historical pull
```

Routine refreshes only need recent years. Do a full backfill only when columns or the schema
change.

## Moving files safely

Before moving or deleting anything, search for readers of that exact file, including paths built in
code. Experiment folders, `runs/current/scratch/` and pickled module paths are all read by
production. The [architecture page](model/architecture.md) lists the module constraints.
