# Batted-ball direction — gap #6 closed

> Written 2026-08-15 against model v2.1c. Row counts predate the 1988-2004 history build.

Spray direction is **not** Statcast-only. Every game feed, at every level back
to **2007**, carries a `hitData` block on batted-ball play events:

```json
{"trajectory": "line_drive", "hardness": "medium", "location": "8",
 "coordinates": {"coordX": 116.02, "coordY": 103.93}}
```

Below MLB the pitch side is empty — no `startSpeed`, `breaks` is `{}` — but the
*hit* coordinates are populated regardless, because they come from the human
stringer rather than from Hawkeye. Verified at 100% coverage across MLB / AA /
A / RK for 2007, 2010, 2015, 2019 and 2025.

`data/sources/spray.py` scrapes this into `batted_ball_profile`. Geometry is
home plate at (125.42, 203.5); the derived angle was validated against the
feed's own fielder-position tag — location 7 (LF) → −39.5°, 8 (CF) → −5.4°,
9 (RF) → +35.9°. Pull side is resolved from `matchup.batSide.code`, the stance
actually used in that plate appearance, so switch hitters resolve per-PA rather
than per-player.

Validation on a 40-game Double-A 2025 slice (1,938 batted balls, 291 players):

| | Pull | Center | Oppo |
|---|---:|---:|---:|
| Observed | 40.7% | 32.3% | 26.9% |

98.9% of batted balls classified. The independently-derived trajectory split
(GB 42.3% / FB 27.1% / LD 23.3% / PU 7.2%) matches the season-stat grid.

Uses the `playByPlay` endpoint (~280 KB/game) rather than `feed/live`
(~500 KB) — identical content, 43% smaller. Runs are resumable via a
`spray_games_parsed` ledger. A single season across MLB and all full-season
levels is ~6,800 games, so this is an incremental job, not a one-shot pull.

**`hardness` is close to useless.** It is a human grade, and it is
overwhelmingly "medium" at every level *including MLB*:

| | medium | hard | soft |
|---|---:|---:|---:|
| MLB 2025 | 83.5% | 14.3% | 2.2% |
| AAA 2025 | 83.4% | 9.7% | 6.9% |
| AA 2025 | 91.8% | 6.3% | 1.8% |
| A 2025 | 84.1% | 8.6% | 7.4% |
| MLB 2015 | 85.8% | 6.7% | 7.5% |
| AA 2015 | 97.6% | 1.3% | 1.1% |

Only 2–16% of balls get a non-medium tag, and the rate swings with level and
season far more than hitter populations plausibly do — AA went from 97.6%
medium in 2015 to 91.8% in 2025. That is scorer drift. It is collected because
it is free, but it is not a stand-in for exit velocity.
