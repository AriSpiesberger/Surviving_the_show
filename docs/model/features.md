# The feature panel

> Written 2026-08-15 against model v2.1c. The hazard panel has since grown to 326 features.

Features are built per player-snapshot — a point-in-time view with no lookahead:
scouting grades, rankings and stats are all filtered to `season ≤ snapshot`.

| | Count |
|---|---:|
| `scouting.py` | 330 |
| `windowed.py` | 181 |
| Hazard panel | 314 (incl. 76 scouting) |
| `FEAT_COND` (joint XGB contract) | 77 |

The scouting block is the substantive one. For every rate stat it derives a
level-adjusted twin (`woba_vs_level`), then layers per-year values, career bests,
current-vs-best ratios, year-over-year deltas, second differences, and — for
players who repeat a level — the slope across those repeats. That last group is
where genuinely predictive signal tends to live: a hitter who repeats High-A and
improves is a different prospect from one who repeats and stalls.

> **The new stats are not wired in yet.** `features/advanced.py` computes the
> full metric set and is validated, but neither `scouting.py` nor the hazard
> panel consumes it. Every result below was produced *without* plate discipline,
> batted-ball profile, true wOBA, splits, park factors or defense. They are the
> headroom, not part of the current number.
