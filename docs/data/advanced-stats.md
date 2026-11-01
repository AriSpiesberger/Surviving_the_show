# The advanced-stat layer

> Written 2026-08-15 against model v2.1c. Row counts predate the 1988-2004 history build.

Until recently the stat table held eight hitter rates and seven pitcher rates,
with `woba`, `fip` and `velo_avg` declared but **never populated** — the feature
layer reconstructed wOBA and FIP from proxies at read time. The MLB Stats API
endpoint the puller already called was returning ~90 fields per player and the
ingest kept 13 of them.

`season_stats` now carries **136 columns**. The new ones are raw counts rather
than rates, so any rate — including era-specific wOBA weights — can be
recomputed without a re-pull.

What that unlocks:

- **True wOBA** from linear weights, not the OBP+ISO proxy, with era-appropriate
  coefficients.
- **Plate discipline** — Swing%, Whiff%, SwStr%, Contact%, pitches per PA.
  **Only where `pitch_data_valid = 1`.** An earlier draft of this document
  claimed these were "available at every level back to 2005." That was wrong,
  and wrong in the most dangerous way: the fields are present and non-null
  across the whole span, but ~40% of rows are silent undercounts. See § 2.3.
- **Batted-ball profile** — GB/FB/LD/PU%, GB/FB, HR/FB, from the trajectory
  breakdown of both outs and hits.
- **True FIP** including HBP, computed from counts rather than reconstructed
  from rounded per-9 rates, plus **xFIP** and a transparent SIERA-shaped estimator.
- **League-relative indices** — wRC+, ERA−, FIP−, K%+, on per-season/per-level
  baselines aggregated as true league totals.

**Baselines validated against published figures.** Aggregating the full 2024 MLB
population reproduces league ERA 4.079, wOBA .309, K% 22.6%, BB% 8.2%,
BABIP .289. The same machinery on AA returns ERA 3.943, wOBA .304, K% 24.1%,
BB% 9.7% — the expected shape for the level.

## Backfill coverage

Share of rows of that type carrying the new counting stats:

| Level | Hitter rows | Cov. | Pitcher rows | Cov. | State |
|---|---:|---:|---:|---:|---|
| RK | 33,181 | 99.7% | 37,853 | 99.7% | complete |
| AAA | 29,297 | 98.7% | 29,110 | 98.8% | complete |
| AA | 21,814 | 99.3% | 20,931 | 99.2% | complete |
| A+ | 17,839 | 99.4% | 21,680 | 99.5% | complete |
| A | 17,178 | 99.6% | 20,642 | 99.7% | complete |
| A− | 7,171 | 99.4% | 8,146 | 99.6% | complete |
| MLB | 28,942 | 84.5% | 23,024 | 70.3% | partial |
| NCAA-D1 | 48 | 0.0% | 60 | 0.0% | separate source |

**Every MiLB level is now complete.** RK (2005–2026) and A− (2005–2019) were
backfilled on 2026-09-02; they had been empty only because `phase_milb`
defaults to `levels = [AAA, AA, A+, A]` and had never been run with anything
else. Pass `--levels` explicitly:

```bash
python -m prospects.data.pull --phase milb --levels RK --start 2005 --end 2026
python -m prospects.data.pull --phase milb --levels A- --start 2005 --end 2019
```

MLB's shortfall is structural, not a missed pull: those rows run back to 1874
and pitch-level counts do not exist before 2005.

The backfill also populated `league`, which is what makes the DSL/VSL gate in
§ 2.2 operable. Rookie ball is where it bites — **28,603 rows (DSL 26,299 +
VSL 2,304) now have their swing metrics correctly suppressed**, about 40% of
the level. Without the gate those rows would have carried SwStr% and Whiff%
inflated by roughly 2×, concentrated precisely in the youngest cohorts.

## Tables built but not yet populated

Pullers written and tested against the live API; not yet run into `prospects.db`.

| Table | Contents | Availability |
|---|---|---|
| `platoon_splits` | vs-LHP / vs-RHP lines | All levels |
| `fielding_stats` | Innings and rates by position | All levels |
| `park_factors` | Home/road run environment per team | All levels |
| `statcast_batting` | Exit velo, barrel%, chase%, xwOBA | MLB, 2015+ |
| `statcast_pitching` | Velocity, spin, induced whiffs | MLB, 2015+ |
| `pitch_arsenal` | Per-pitch usage, velo, whiff | MLB, 2015+ |
| `catcher_defense` | Framing runs, pop time, arm | MLB only |

Test runs resolved real players correctly at MLB, AA and A: 4,323 split rows and
9,394 fielding rows across three levels for 2024 alone.
