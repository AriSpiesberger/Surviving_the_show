# Architecture

The pipeline is a three-layer model plus a buy-list stage. Data flows left to
right; each layer is one folder under `prospects/`.

```
data/  ->  features/  ->  model/  ->  evaluation/  +  buylist/
```

## The model — v2.1c

**1. Landmark hazards** (`model/hazards/landmark.py`, over `model/hazards/survival.py`).
Discrete-time survival model: for each player-season it estimates the per-year
hazard of hitting each career event (TOP_100, MLB_DEBUT, ESTABLISHED_MLB, STAR,
ELITE). "Landmark" = years-in-pro is a feature rather than a separate model per
year, which removes a train/inference mismatch. HistGBT, ~314 features
including 76 scouting-grade features. Trained by `model/pipelines/stage_a.py`.

**2. Conditional joint XGB** (`model/joint.py` + `model/train/joint_xgb.py`).
Reframes the hazards' full trajectory as a single conditional refinement: it
reads the per-year hazard curves plus the horizon `h` itself and outputs, per
event, the cumulative `P(event by snap+h)`. Sweeping h=1..10 gives a trajectory
rather than one scalar. Horizon-as-a-feature solves per-cell censoring for free.
`model/joint.py` owns the event list, feature order, and horizon constants
(`H_MAX`, `PUBLISH_H`, `AGE_CENTER`, `YIP_CENTER`) so ordering can never drift
between trainer, scorer, evaluator, and buy list.

**3. Calibration + thresholds** (`model/calibration.py`, `model/train/calibrators.py`,
`model/thresholds.py`). Per-(event, horizon) isotonic calibrators map raw scores
to true probabilities; per-yip thresholds give precision-calibrated debut
cutoffs.

**Timing** (`model/train/time_to_debut.py`) is a LassoCV on the hazard probs
predicting years-to-debut, surfaced on the buy list.

## OOF vs prod

Two joint-XGB artifacts, by design:

- **`joint_xgb.pkl`** — out-of-fold. Every fold is scored by hazards trained on
  the *other* folds, so it is an honest held-out model. This is what
  `evaluation/` and the default `buylist/build.py` read.
- **`joint_xgb_prod.pkl`** — trained on the full panel's in-sample hazards. The
  fresh (2024-26) cohort was never in training either way, so scoring it has no
  leakage. This is what the weekly `pipelines/prod.py` scores the live buy list
  with.

## Pipelines

| Command | Writes (under `runs/current/`) |
|---|---|
| `model.pipelines.oof` | fold hazards, `training/oof_*_long.csv`, `models/joint_xgb.pkl` |
| `model.pipelines.stage_a` | `models/hazards_landmark.pkl`, `lasso_logits.pkl`, `timing_stage_a.pkl`, `training/{fit,val,all}_long.csv` |
| `model.train.hazards` | `models/hazards.pkl` (full-panel prod hazards) |
| `model.pipelines.prod` | `models/joint_xgb_prod.pkl`, `scored/snap2026_long.csv`, `buy_lists/{all_scored,final}.csv` |
| `evaluation.run` | `evaluation/*.csv`, `evaluation/headline.json` |
| `buylist.build` | `buy_lists/{all_scored,final}.csv` |

The weekly retrain runs `stage_a` then `prod`. `oof` and `hazards` are run when
the held-out model or full-panel hazards need regenerating.

## Runs

A **run** is one pass of the pipeline under a tag. `prospects.config.RunPaths`
gives every artifact directory for a tag; `runs/current/` is the live run.
`--tag <name>` (or `RUN_TAG=<name>`) points a command at `runs/<name>/` instead,
so a backtest or A/B experiment sits beside the live run without colliding. The
weekly retrain overwrites `runs/current/` in place; to keep a run, copy it aside
or archive it before retraining.

The model **generation** a file belongs to is git history, not its name — the
tree carries no `v1.18b` / `v2.0b` suffixes. See `docs/runbook.md` for promoting
a candidate run to `current`.
