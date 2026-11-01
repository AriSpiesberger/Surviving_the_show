# Rookie-ball advanced stats — and the DSL pitch-count defect

> Written 2026-08-15 against model v2.1c. Row counts predate the 1988-2004 history build.

On **statsapi**, `stats=season` omits the swing and batted-ball grid entirely;
`stats=seasonAdvanced` returns `iso`, `babip`, `swingAndMisses`, `totalSwings`
and the full trajectory breakdown. (Our puller uses the **stitch** endpoint,
whose `stats=season` already returns the union of both — so for this codebase
RK needs no parameter change, just an actual run. Confirmed: sportId 16 returns
all 47 hitter fields, 1,607 rows for 2024 across DSL 832 / ACL 397 / FCL 378.)

**The DSL pitch counts are corrupt — by almost exactly a factor of two.**
Measured on 2024, players with 50+ PA / 20+ IP:

| League | P/PA | P/IP | P/BF | Swing% | Whiff% | SwStr% |
|---|---:|---:|---:|---:|---:|---:|
| DSL | 1.97 | 8.54 | 1.93 | 0.600 | 0.480 | 0.288 |
| ACL | 3.80 | 17.45 | 3.79 | 0.443 | 0.314 | 0.139 |
| FCL | 3.80 | 16.87 | 3.79 | 0.432 | 0.282 | 0.122 |

One refinement worth noting: `totalSwings` and `swingAndMisses` are corrupted
*independently* of `numberOfPitches`. Whiff% is misses-over-swings and never
touches the pitch count, yet it still reads ~1.6× the complex-league rate. So
the whole pitch-tracking block has to be discarded, not only the ratios with
pitches in the denominator.

Everything built on plate appearances and balls in play survives intact:

| League | K% | BB% | GB% | LD% | BABIP |
|---|---:|---:|---:|---:|---:|
| DSL | 0.219 | 0.139 | 0.463 | 0.179 | 0.308 |
| ACL | 0.249 | 0.130 | 0.458 | 0.181 | 0.348 |
| FCL | 0.235 | 0.131 | 0.464 | 0.168 | 0.314 |

Those are league differences, not artifacts. `season_stats` now carries
`league` / `league_id`, and `features/advanced.py` gates on
`PITCH_DATA_UNRELIABLE_LEAGUES = {DSL, VSL}` — the VSL ran through 2015 and
shares the defect. A DSL line still yields wOBA, K%, BB%, FIP and the full
batted-ball profile; only the swing rates are nulled.
