# Data

## Databases

Both files are SQLite and live at the repo root, outside git.

| File | Role | Written by |
|---|---|---|
| `prospects.db` | Live database, updated nightly | `deploy/daily_data.py` |
| `prospects_snapshot.db` | Stable copy that training reads | `deploy/weekly_score.py`, refreshed every run |

`config.live_db()` and `config.model_db()` resolve them. Override with `PROSPECT_DB` or
`PROSPECT_MODEL_DB`.

Current contents of `prospects.db`:

| | |
|---|---:|
| Players | 63,749 (20,514 drafted, 43,235 international) |
| Season rows | 367,913 |
| Career outcome rows | 67,476 |
| MLB debuts | 8,486 |
| Ranking history rows | 35,076 |
| Draft classes | 1988-2026 |

Player IDs resolve strictly through MLBAM ID. Name-fallback matching was removed after it
mis-assigned about 8% of MiLB stat rows.

## Sources

Each external source has one module in `prospects/data/sources/`.

| Module | Provides |
|---|---|
| `milb.py` | MLB Stats API season lines at every level; the backbone |
| `milb_advanced.py` | Platoon splits, fielding by position, park factors |
| `pybaseball.py` | Career outcomes, draft data, Lahman awards, Chadwick ID register |
| `bwar.py`, `season_war.py` | Baseball-Reference WAR |
| `baseballcube*.py` | Historical per-team Top-30 lists |
| `rankings.py` | Point-in-time ranking history |
| `spray.py` | Batted-ball direction |
| `draft_align.py`, `mlb_draft.py` | Draft records |
| `ncaa.py`, `college.py` | NCAA D-I lines (thin) |
| `outcomes.py` | Training labels |

`prospects/data/pull.py` runs a bulk pull. `prospects/data/backfills/` holds one-shot repair scripts;
several take `--db` to target either database.

## Scouting grades

`features/grades.py` loads `reference/fangraphs_board/scouting_grades_pointintime.csv` and adds
no-lookahead grade features. Missing values stay NaN. Set `SCOUTING_GRADES_OFF=1` to disable them for
an A/B run. The [`tools/scouting`](../../tools/scouting) scripts rebuild the file.

## Rule for adding a field

A field is usable only where its coverage does not depend on the outcome. Signing bonus and service
time both leaked the debut label this way. `tools/leak_audit.py` checks for it.

## Detail pages

- [Advanced-stat layer](advanced-stats.md)
- [Batted-ball direction](batted-ball.md)
- [Non-affiliate contamination and biometrics](non-affiliate-and-biometrics.md)
- [Plate-discipline defect](plate-discipline-defect.md)
- [Rookie-ball defect](rookie-ball-defect.md)
- [Known gaps](known-gaps.md)
