# Architecture

Data moves through `prospects/` in this order:

```
data/ -> features/ -> model/ -> evaluation/ + buylist/ -> deploy/
```

## Events

| Event | Definition |
|---|---|
| `TOP_100_PROSPECT` | Ever ranked in a Pipeline or Baseball America top 100 |
| `MLB_DEBUT` | Any MLB game |
| `ESTABLISHED_MLB` | 500+ career PA or 200+ career IP |
| `STAR_PLUS_ELITE` | An All-Star selection or major award |

Labels are right-censored, so each player snapshot and horizon is scored only once the outcome is
known.

## The stack

**1. Landmark hazards** (`model/hazards/`). A discrete-time survival model per event. It produces a
per-year hazard curve for each player snapshot.

**2. Joint XGBoost** (`model/joint.py`, `model/joint2.py`). Takes the hazard curves, raw panel
features and the horizon `h` as inputs, and returns the cumulative P(event by snapshot + h). It is
monotone in `h`. This layer is v2.4 when trained on the fit split and v2.5 when trained on all
players.

**3. v3 components** (`model/v3.py`). Each event is published as an equal-weight blend of calibrated
components:

| Event | Components |
|---|---|
| `MLB_DEBUT` | `base`, `v3`, `nhaz` |
| `TOP_100_PROSPECT` | `base`, `haz` |
| `ESTABLISHED_MLB` | `base`, `v3`, `cond`, `haz` |
| `STAR_PLUS_ELITE` | `cond`, `haz` |

- `base`: the joint XGBoost trajectory
- `v3`: a per-event GBM over panel features and sequence-encoder logits
- `cond`: v3 debut probability times P(event given debut)
- `haz`: a discrete-time hazard GBM
- `nhaz`: a neural hazard network with a transformer trunk

The sequence encoder is a transformer over a player's season rows and rank tokens. Its output is
stacked honestly: training rows get logits from an encoder that never saw that player.

**4. Calibration** (`model/calibration.py`, `model/v3.py`). A monotone Platt fit per horizon on
held-out predictions, so printed probabilities can be read as rates.

**5. Buy list** (`buylist/build.py`). Keeps players whose own P(debut within 3 years) is at least
0.60, then joins market prices.

## Held-out and full builds

| Bundle | Trained on | Used for |
|---|---|---|
| `v3.pkl` | Fit split (v2.4 base) | Metrics and thresholds |
| `v3.5.pkl` | All players (v2.5 base) | Scoring the published sheet |

v3.5 has no held-out score of its own, because it trains on every player. `v3.pkl` is its report
card.

## Production `exp_*` modules

Eleven modules in `model/train/` keep their experiment names because the retrain and `v3.py` import
them. The main ones are `exp_cdf_timing5` (the joint-layer build), `exp_seq_c` and `exp_seq_d` (the
encoders) and `exp_nhaz` (the neural hazard). Experiments that production does not use live in
[`research/model_experiments`](../../research/model_experiments).

## Runs

A run is one pass of the pipeline under a tag. `prospects.config.RunPaths` defines every artifact
path for a tag, and `runs/current/` is the live run. Set `--tag <name>` or `RUN_TAG=<name>` to write
to `runs/<name>/` instead.

Pickled bundles reference `prospects.model.{v3,joint2,calibration}` and `prospects.core.schema`, so
moving those modules breaks loading existing models.
