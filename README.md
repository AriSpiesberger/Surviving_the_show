# Surviving the Show

Predicts which minor-league baseball prospects will reach MLB milestones, then turns those
probabilities into a buy list for Bowman 1st Chrome autographs.

## The model

**v3.5** estimates the probability of four career events at every horizon from 1 to 10 years. Each
event is a blend of gradient-boosted, hazard and sequence-encoder components, trained on draft
classes from 1988 onward plus international signees.

Held-out results for players the model never trained on, from
[`runs/current/evaluation_v3`](runs/current/evaluation_v3):

| Event | AP at 3 years | AP at 6 years | AUC at 6 years |
|---|---:|---:|---:|
| MLB debut | 0.574 | 0.603 | 0.932 |
| Top-100 prospect | 0.527 | 0.496 | 0.972 |
| Established MLB player | 0.326 | 0.393 | 0.940 |
| Star or elite | 0.141 | 0.236 | 0.955 |

The buy list keeps players with at least a 60% chance of debuting within three years.

## Repository layout

| Folder | Contents |
|---|---|
| [`prospects/`](prospects) | The Python package: data, features, model, evaluation, buy list, deployment |
| [`tools/`](tools) | Hand-run scripts for scraping, feature building and auditing |
| [`research/`](research) | Card-market backtests and retired model experiments |
| [`reference/`](reference) | Checked-in inputs: scouting boards, rankings, price snapshots |
| [`runs/`](runs) | Model runs; only the evaluation reports are in git |
| [`portfolio/`](portfolio) | Card holdings and alert state |
| [`ops/`](ops) | Windows Task Scheduler jobs |
| [`docs/`](docs) | Architecture, runbook and data notes |
| [`tests/`](tests) | pytest suite |

## Getting started

Requires Python 3.11 or newer.

```bash
pip install -e ".[dev]"
pytest
cp .env.example .env    # add eBay and SendGrid credentials
```

The databases and trained models are too large for git, so a fresh clone starts without them. Build
the database first, then train:

```bash
prospects-pull                      # download the full history (slow)
prospects-refresh --dry-run         # preview the full retrain
prospects-refresh                   # train, evaluate and build the buy list
```

If you already have the artifacts and a scored snapshot, rebuild only the buy list:

```bash
python -m prospects.deploy.weekly_score --season 2026 --skip-retrain --buylist-only
```

Use this rather than a bare `prospects.buylist.build`, which defaults to an older bundle. The sheet
is written to `runs/current/buy_lists/final.csv`.

## Documentation

- [Architecture](docs/model/architecture.md): how the model works, stage by stage
- [Runbook](docs/runbook.md): retraining, promoting a run, scheduled jobs
- [Data](docs/data/README.md): sources, databases and known defects
- [Card-market findings](research/card_market/README.md): which trades clear costs
- [Model experiments](research/model_experiments/README.md): what was tried and what shipped
