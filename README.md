# Prospect Card Buy List — v1.17 Production

End-to-end pipeline that scores MiLB prospects, ranks the buy universe, and produces a card buy list with model EV vs eBay prices.

## Two-command weekly deploy

```bash
# 1. Score current prospects with v1.17 production hazards
python -m scripts_v17.score.score_panel_v17 --snap-year 2026

# 2. Apply lassos + model B + universe filter + eBay prices
python scripts_v17/buylist/build_v17_buylist.py
```

Outputs:
- `buy_list_v1.17_FINAL.csv` — universe-filtered, per-yip-threshold passing players
- `buy_list_v1.17_ALL_SCORED.csv` — all snap=2026 prospects with full scoring

## Production artifacts

All in `models/`:

| file | what it is |
|---|---|
| `event_classifiers_v1.17_prod.pkl` | hazards trained on 100% of panel (production) |
| `event_classifiers_v1.17.pkl` | 80%-trained twin (kept for honest validation) |
| `debut_lasso_universe_v1.17h.pkl` | debut composite scorer (universe-aware, held-out validated) |
| `top100_lasso_v1.17h.pkl` | top-100 composite scorer (held-out validated) |
| `model_b_outcomes_v1.17h.pkl` | P(cup/utility/regular/breakout \| debut) |
| `player_position_from_stats.csv` | corrected positions from season_stats |

Per-yip filter thresholds in `scripts_v17/v17h_thresholds.json`:
```
yip 0 → lasso ≥ 4.241
yip 1 → lasso ≥ 1.713
yip 2 → lasso ≥ 2.549
yip 3 → lasso ≥ 3.913
yip 4 → lasso ≥ 3.755
```

## Directory layout

```
.
├── README.md
├── CV_VALIDATION_PLAN.md, HANDOFF.md, RUNBOOK.md
├── prospects_snapshot.db       # source DB (46,692 prospects, 237,904 stat rows)
├── panel_v1.14n.npz, panel_v1.17.npz  # active panels (kept at root for speed)
│
├── scripts_v17/                # PRODUCTION pipeline
│   ├── score/score_panel_v17.py       # score current prospects
│   ├── buylist/build_v17_buylist.py   # build final ranked list
│   ├── buylist/apply_model_b_to_buylist.py
│   ├── buylist/merge_prices_to_buylist.py
│   ├── validate/validate_universe.py  # canonical val script (non-R1, non-top100)
│   ├── validate/validate_full.py      # per-yip percentile slabs + 2021 lookback
│   ├── validate/compare_models_full.py# AUC/PR/McNemar/DeLong/Brier/ECE
│   ├── train/refit_models_honest.py   # train lassos + model B on honest data
│   ├── train/train_top100_lasso.py
│   └── v17h_thresholds.json
│
├── models/                     # all model artifacts
├── data/                       # eBay prices, raw inputs
├── backtests/v17/              # snap=2022/2023/2024 backtest sheets
├── buy_lists/                  # historical buy lists
├── panels/                     # old panels (v1.14i, v1.15, etc)
│
├── prospects/                  # core Python package
│   ├── classifier/             # hazard training, feature builders
│   ├── ingestion/              # data sources (MLB Stats, Lahman, NCAA)
│   ├── features/               # scouting.py (238-feature builder)
│   ├── market/                 # eBay client + price aggregation
│   ├── schema.py, storage.py
│
├── archive/                    # everything historical/abandoned
│   ├── v14n_v16_iterations/    # scripts, csvs from v14n/v16 era
│   ├── v17_iterations/         # comparison files, val outputs, intermediate csvs
│   ├── v15_betacal_abandoned/  # the v1.15 beta-calibrator attempt
│   ├── v16_proxy_backfill_abandoned/  # v1.16 (kept for reference)
│   ├── old_scratch_scripts/    # historical _*.py one-off scripts
│   ├── grades/, edges/, etc.   # legacy output dirs
│   └── panel_v1.14i.*, panel_v1.15.*
│
├── scratch/                    # intermediate working files
│   ├── v17_intermediate_chunks/  # chunked scoring outputs (regeneratable)
│   ├── v17_long_files/         # fit/val long files (regeneratable)
│   ├── cohort_pids/            # cohort pid lists (regeneratable)
│   ├── shap_iterations/        # one-off SHAP analyses
│   └── old_logs/
│
├── experiments/                # historical experiment subdirs
└── mnt/                        # external data drop zone
```

## Validation summary (held-out 80%-hazard, universe-filtered)

Per-yip cumulative-from-top score threshold for **≥50% MLB_DEBUT realized**:

| yip | n_total | base | threshold | n above | realized |
|---|---|---|---|---|---|
| 0 | 3388 | 8.4% | ≥4.241 | 30 | 50% |
| 1 | 3378 | 8.1% | ≥1.713 | 144 | 50% |
| 2 | 3356 | 7.5% | ≥2.549 | 188 | 50% |
| 3 | 3297 | 5.9% | ≥3.913 | 124 | 50% |
| 4 | 3237 | 4.1% | ≥3.755 | 74 | 50% |

These are the thresholds baked into `build_v17_buylist.py`.

## Universe definition

The "buy universe" filter applied in build_v17_buylist:
- `bucket != R1` (round-1 draftees are too expensive on Bowman 1st Chrome auto)
- `year_top_100 IS NULL` (never been on a public top-100 prospect list)
- per-yip lasso score >= threshold above

Result: ~300 players per cycle. R1 + ever-top-100 are excluded because their cards are priced at the level the model has already converged on.

## Maintenance

- **DB refresh**: re-ingest `season_stats` via `prospects/ingestion/milb_stats.py` (~30 min for full season)
- **Top-100 refresh**: update `career_outcomes.year_top_100` when new BA / MLB Pipeline lists come out
- **eBay refresh**: `python -m prospects.scripts.fetch_prospect_prices --grades buy_list_v1.17_FINAL.csv --top-n 300`
- **Panel rebuild**: only needed when feature-builder code changes — `python -m prospects.classifier.build_panel ...`
- **Hazard retraining**: only needed when panel rebuilds or new training data arrives
- **Lasso / model B retraining**: only needed if validation drifts; use `scripts_v17/train/refit_models_honest.py`
