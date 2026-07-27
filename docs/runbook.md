# Runbook

Operating the pipeline. Every command resolves its paths through
`prospects.config`; none takes explicit artifact paths unless you are targeting
a non-default run with `--tag`.

## Weekly cycle (what the scheduler does)

`ops/run_job.ps1 -Job weekly_score` runs `prospects.deploy.weekly_score`, which:

1. Copies `prospects.db` → `prospects_snapshot.db` (stable read for training).
2. **Stage A** — `model.pipelines.stage_a`: rebuild panel, train landmark
   hazards, score fit/val, refit the L1 bundle + stage-A timing.
3. **Stage B** — `model.pipelines.prod`: train the prod joint XGB, retrain
   timing, score the 2026 cohort, build the buy list into
   `runs/current/buy_lists/final.csv`.
4. Refresh eBay debut comps (fail-soft).

It then checks that all required artifacts exist (`weekly_score.REQUIRED`) and
exits non-zero if any are missing — including `runs/current/models/hazards.pkl`,
which is written by the hand-run `model.train.hazards`, not by the weekly job.
If that check fails, run:

```bash
python -m prospects.model.train.hazards        # writes runs/current/models/hazards.pkl
```

## Full retrain by hand

```bash
python -m prospects.model.pipelines.oof        # OOF folds + hazards + joint_xgb.pkl
python -m prospects.model.train.hazards        # full-panel prod hazards
python -m prospects.model.pipelines.prod       # prod joint XGB + score 2026 + buy list
python -m prospects.evaluation.run             # held-out metrics
python -m prospects.evaluation.report          # regenerate runs/current/evaluation/README.md
```

Rebuild only the buy list from existing artifacts (fast, deterministic):

```bash
python -m prospects.buylist.build
```

## Running a candidate without touching the live run

```bash
python -m prospects.model.pipelines.oof  --tag cand
python -m prospects.model.train.hazards  --tag cand
python -m prospects.model.pipelines.prod --tag cand
python -m prospects.evaluation.run       --tag cand
```

Everything lands in `runs/cand/`. Compare `runs/cand/evaluation/headline.json`
against `runs/current/evaluation/headline.json`.

## Promoting a candidate to current

`runs/current/` is a plain directory, not a symlink. To promote:

1. Archive the outgoing run: `mv runs/current archive/runs/<old-tag>-<date>`.
2. `mv runs/cand runs/current`.
3. Re-run `python -m prospects.buylist.build` and eyeball `final.csv`.

The weekly retrain will overwrite `runs/current/` in place next Monday, so a
promoted candidate becomes the baseline the next retrain builds on.

## Scheduled tasks

```powershell
# register / update the four tasks (idempotent)
powershell -ExecutionPolicy Bypass -File ops\register_tasks.ps1

# inspect / pause / run one now
Get-ScheduledTask -TaskName 'SurvivingShow_*'
Disable-ScheduledTask -TaskName 'SurvivingShow_weekly_score'
Enable-ScheduledTask  -TaskName 'SurvivingShow_weekly_score'
Start-ScheduledTask   -TaskName 'SurvivingShow_daily_digest'
```

Logs land in `logs/<job>_<date>.log`. `ops/run_job.ps1` loads `.env`, pins BLAS
to one thread, and overrides the DB/prices/holdings paths to the repo root.

## Data refresh

```bash
python -m prospects.deploy.daily_data          # current-season pull into prospects.db
python -m prospects.data.pull --phase all      # full historical re-pull (see docs/data.md)
```

Do a **full** pull (`--phase all`), not current-season-only — a partial refresh
can leave cross-table filters stale.
