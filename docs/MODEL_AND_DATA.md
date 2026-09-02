# Surviving the Show — model & data summary

Model **v2.1c**. Held-out weighted AP **0.499** @ h=6.
Generated 2026-08-15 from `prospects.db` and `runs/current/evaluation/`.

---

## 1. The data

Everything lives in one SQLite file, `prospects.db`, with a frozen copy
(`prospects_snapshot.db`) used for training so a long fit reads a stable file.
The universe is drafted players plus international free agents, keyed on a
synthetic player ID that resolves **strictly through MLBAM ID** — name-fallback
matching was removed in v1.10 after it mis-assigned ~8% of MiLB stat rows.

| | |
|---|---|
| Players | 48,651 (26,045 pitchers) |
| Season rows | 292,003 (player · season · level) |
| Resolved careers | 50,692 of 52,761 outcome rows |
| MLB debuts | 8,054 |
| Ranking snapshots | 32,538 (point-in-time, no lookahead) |

### season_stats by level

| Level | Rows | Seasons | Hitter rows | Pitcher rows |
|---|---:|---|---:|---:|
| RK | 67,314 | 2005–2026 | 30,621 | 36,592 |
| AAA | 51,710 | 2005–2026 | 29,297 | 29,110 |
| MLB | 42,584 | 1874–2026 | 28,942 | 23,024 |
| A+ | 38,683 | 2005–2026 | 17,839 | 21,680 |
| A | 36,991 | 2005–2026 | 17,178 | 20,642 |
| AA | 36,886 | 2005–2026 | 21,814 | 20,931 |
| A− | 14,478 | 2005–2019 | 6,653 | 7,811 |
| NCAA-D1 | 104 | 2021–2025 | 48 | 60 |
| **Total** | **288,750** | 1874–2026 | 152,392 | 159,850 |

A− stops after 2019 because short-season ball was eliminated in the 2021
reorganization — a true structural gap, not missing data. MLB reaches back to
1874 via the Lahman backfill.

### Sources

| Module | Provides |
|---|---|
| `milb.py` | MLB Stats API, per team-season-level. The backbone: every MiLB and MLB line, 2005+ |
| `milb_advanced.py` | Platoon splits, fielding by position, computed park factors (new) |
| `pybaseball.py` | FanGraphs career outcomes, draft data, Lahman awards, Chadwick ID register |
| `bwar.py`, `season_war.py` | Baseball-Reference WAR, career and per-season |
| `baseballcube.py` | Historical per-team Top-30 lists → point-in-time ranking history |
| `rankings.py` | FanGraphs Board 2017–26, Trouble-With-The-Curve 2013–19 scouting grades |
| `ncaa.py`, `college.py` | NCAA D-I lines (thin — 104 rows, not load-bearing) |
| `outcomes.py` | Assembles training labels |

---

## 2. The advanced-stat layer

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

### Backfill coverage

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

### Tables built but not yet populated

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

### 2.1 Batted-ball direction — gap #6 closed

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

### 2.4 Non-affiliate contamination, biometrics, and the 2005 floor

**Mexican League inside pre-2021 AAA.** `sportId=11` carried leagueId 125
alongside the International and Pacific Coast leagues through 2020 — 16 clubs
a season, **9,399 rows, 18.2% of all AAA**. Every pre-2021 AAA baseline was
therefore computed over a foreign professional league mixed into the affiliate
panel, and those baselines feed `woba_vs_level` and every derived twin.

Rows are now tagged `league_id = 125` (team lists resolved per season from the
API and cached in `reference/mex_league_teams.json` — the filter must be
year-scoped, since `DUR` is Mexican League pre-2021 and Durham after).
`scouting.EXCLUDE_LEAGUE_IDS` drops them from baseline construction. The AAA
baselines moved materially:

| stat | old | new | delta |
|---|---:|---:|---:|
| hit K% | 0.1895 | 0.2000 | +0.0105 |
| hit ISO | 0.1410 | 0.1450 | +0.0040 |
| pit K/9 | 7.66 | 8.11 | +0.45 |
| pit FIP | 4.031 | 3.966 | −0.065 |
| median hitter age | 26.99 | 26.67 | −0.32 |

The age drop is the signature: Mexican League rosters are veteran-heavy, and
they were dragging AAA toward an older, lower-strikeout, contact-oriented
denominator.

**Biometrics purged.** `height_inches`, `weight_lbs` and `bmi` are removed
from the panel (330 → 327 features). They came from `/api/v1/people`, which
returns current state with no measurement date and no dated variant, and were
stored per-player with no season dimension — so a 2010 A-ball snapshot carried
the player's 2026 listed weight. Measured against the FanGraphs boards, which
*do* carry dated physicals: weight genuinely drifts for **40.1%** of players
(median 17 lb, p90 40 lb, max 85 lb). "Filled out and succeeded" was being fed
backwards into the snapshot meant to predict it.

Checked and clean, no action needed: `active` and `currentAge` are never read
from the API anywhere in this codebase, `strikeZoneTop/Bottom` is never
pulled, and both age features already derive from `birthDate` per season
rather than from `currentAge`.

A proposed fix — taking as-drafted measurements from `/api/v1/draft/{year}` —
was tested and rejected: that endpoint embeds the live `/people` record
verbatim. 40/40 picks from the 2011 draft are byte-identical to today's
height and weight, and the payload carries `currentAge: 35` and
`active: true` inside a 2011 draft record.

**The 2005 floor is real, but Rookie ball is not truncated to 2006.** The DSL
genuinely starts in 2006 (2005 returns zero splits). The rest of Rookie ball
does not: RK 2005 holds **1,830 real rows** across GCL, AZL, APP, PIO and VSL
— 23,451 PA in the GCL alone, 558 rows with 100+ PA — and **1,055 players
whose only rookie season is 2005**. Truncating the level to 2006 would orphan
all of them. The gap is DSL-specific, not level-wide.

### 2.3 The plate-discipline block is corrupt across ~40% of the panel

**This is the most important caveat in this document.**

The API returns the same 31 advanced hitting keys for 2005 as for 2026,
non-null, at every level. The schema is byte-identical across the span. But
for roughly half the panel the *values* are undercounts, and nothing in the
response signals it. It is visible only by range-checking.

A full census of this database — 5,313 level-season-team cells with 3+
qualifying hitters — shows a sharply bimodal distribution against a true
pitches-per-PA of ~3.7–4.0:

| P/PA bin | cells | |
|---:|---:|---|
| 1.5 | 1,364 | corrupt |
| 2.0 | 1,049 | corrupt |
| 2.5 | 40 | |
| 3.0 | 95 | |
| 3.5 | 1,070 | credible |
| 4.0 | 1,680 | credible |

47.1% of cells fall below 3.0; only 191 (3.6%) sit in the ambiguous 2.6–3.4
band. `totalSwings` and `swingAndMisses` fail *with* the pitch count, so
Swing%, Whiff%, SwStr% and Contact% are all affected, not just P/PA.

Share of rows with credible pitch counts, by level × season:

| year | MLB | AAA | AA | A+ | A | A− | RK |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2005 | 100% | 0% | 0% | 0% | 0% | 0% | 0% |
| 2006 | 100% | 71% | 0% | 0% | 0% | 0% | 0% |
| 2007 | 100% | 100% | 0% | 0% | 0% | 0% | 0% |
| 2008 | 100% | 95% | 27% | 0% | 0% | 0% | 0% |
| 2011 | 100% | 97% | 61% | 0% | 0% | 0% | 0% |
| 2012 | 100% | 100% | 100% | 0% | 0% | 0% | 0% |
| 2015 | 100% | 98% | 100% | 16% | 17% | 59% | 0% |
| 2016 | 100% | 98% | 100% | 100% | 100% | 88% | 9% |
| 2019 | 100% | 98% | 100% | 100% | 100% | 100% | 0% |
| 2024 | 100% | 100% | 100% | 100% | 100% | — | 49% |
| 2025 | 100% | 100% | 100% | 100% | 100% | — | 97% |

Within Rookie ball the boundary differs by league: GCL and AZL are corrupt for
their entire existence (2005–2019), APP/PIO clear in 2017/2016, ACL and FCL in
2024, the DSL in 2025.

**Why this is worse than an ordinary data gap.** Validity is ordered by level
and era — AAA clears first, the DSL last. So "has credible pitch counts"
encodes "was at a high level in a recent season," which is a short walk from
the outcome. Feeding raw values in lets the model learn the availability
pattern rather than the skill.

**Why the earlier validation missed it.** The league baselines in § 2 were
checked against 2024 MLB, which is a clean cell. Every MLB row in the database
is clean. Validating there and generalising to 2008 A-ball is precisely the
trap the schema stability sets.

**The fix.** `season_stats.pitch_data_valid` is computed per
(level, season_year, **team**) by `data/backfills/pitch_validity.py` — team
granularity because the rollout was team-by-team at the boundaries (AA 2011
was 10 corrupt / 20 clean; RK 2024 was 45/30). `features/advanced.py` gates
every pitch-derived metric on it, and treats undetermined as unreliable.
Current state: 164,879 rows valid (56.5%), 117,617 corrupt (40.3%), 9,507
undetermined (3.2%). Everything built on plate appearances and balls in play
— wOBA, K%, BB%, the batted-ball grid — is unaffected and remains full-span.

```bash
python -m prospects.data.backfills.pitch_validity      # after every pull
```

### 2.2 Rookie-ball advanced stats — and the DSL pitch-count defect

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

---

## 3. The feature panel

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

---

## 4. The model stack

Four events are modeled. `HOF_TRAJECTORY` was deliberately dropped.

| Event | Definition | Base rate @ h=6 |
|---|---|---:|
| TOP_100_PROSPECT | Ever ranked in a Pipeline or BA top 100 | 1.50% |
| MLB_DEBUT | Any MLB game | 13.12% |
| ESTABLISHED_MLB | 500+ career PA or 200+ career IP | 4.15% |
| STAR_PLUS_ELITE | Any major-league recognition — All-Star or major award | 0.58% |

The last is a composite. Two pooled tiers are modeled separately — `_ELITE`
(3+ All-Star selections or a major award) and `_STAR` (those plus a single
All-Star selection) — then combined as a probabilistic union,
`1 − (1 − p_STAR)(1 − p_ELITE)`, trigger year = the earlier of the two. The
pooling exists because the unpooled split produced unstable predictions on a
very small positive count, and even inverted — the rarer event scoring higher
than the commoner one for some players.

**01 · Landmark discrete-time hazards.** A HistGBT survival model per event,
producing per-year hazard curves `hk1…hk10`. Censoring-aware by construction,
which is what makes right-censored career labels usable at all. Trained six ways
for out-of-fold stacking; a production copy fits on 100% of the ≤2020 data.

**02 · Conditional joint XGBoost.** Not a terminal scalar head — a *refinement*
of the hazard trajectory. It takes the full hazard curves, a baseline, and a
target **horizon h as an input feature**, and returns a refined cumulative
P(event by snap+h). Sweeping h=1…10 yields a per-year trajectory per event
rather than one collapsed number. Monotone in h via cummax at inference.
`multi_output_tree` over the 4 heads.

**03 · Timing head.** LassoCV over the hazard probabilities plus distribution
moments, predicting *when* rather than *whether*. MAE 1.14 years, Spearman 0.66.

**04 · Buy list.** The investable thesis is `P(MLB_DEBUT ≤ 3y)` — a three-year
window, not the six-year evaluation horizon, because card prices move on
imminent debuts. Ceiling events carried at h=6 for context. Universe filters
drop washouts, players already on a top-100, and players already in the majors.

Validation uses a 10% held-out player slice — players neither the hazards nor
the XGB head ever saw. Because labels are right-censored, each
(player-snapshot, horizon) cell is scored only where it is actually resolved
(`years_fwd ≥ h`). Nothing beyond h=10 is the XGB's opinion; that range is the
hazard layer extrapolating.

---

## 5. Results

Headline — all buckets, h=6, threshold 0.60:

| Event | n | base% | AP | lift | AUC | precision | recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| TOP_100_PROSPECT | 18,901 | 1.50% | 0.513 | 34.3× | 0.964 | 0.723 | 0.332 |
| MLB_DEBUT | 19,160 | 13.12% | 0.693 | 5.3× | 0.923 | 0.709 | 0.499 |
| ESTABLISHED_MLB | 19,160 | 4.15% | 0.424 | 10.2× | 0.926 | 0.684 | 0.131 |
| STAR_PLUS_ELITE | 19,160 | 0.58% | 0.168 | 28.8× | 0.925 | 1.000 | 0.009 |
| **weighted AP** | | | **0.499** | | | | |

**Ranking is excellent everywhere.** AUC 0.92–0.99 across all four events and
all ten horizons. The model orders prospects well even at a 0.58% base rate.

**Recall is where it costs you.** At threshold 0.60 the two ceiling events barely
fire — ESTABLISHED_MLB recovers 13% of true positives, STAR_PLUS_ELITE catches
1 of 112. Precision is high because the model only commits when certain. For a
buy list that asymmetry is defensible; as a measure of the model it means the
threshold, not the ranking, is doing the discarding.

**MLB_DEBUT is the workhorse and is well calibrated** — calib ≈ 1.05 from h=1
through h=10, AP peaking at 0.702. It is also the buy-list thesis, which is the
right pairing.

**Open calibration defect.** STAR_PLUS_ELITE is well ranked but under-calibrated
at long horizons (calib ≈ 0.7 by h≥4) — the magnitude of stardom is
under-predicted. Fix is a per-horizon isotonic recalibration on that head alone;
the ranking needs no correction.

**Signal concentration.** Performance tracks pedigree and proximity. For
MLB_DEBUT, first-round picks reach AP 0.934 against a 69% base rate, while
rounds 11+ sit at 0.488 against 6.9% — a 7.0× lift, meaning the model adds most
*relative* value exactly where scouting consensus is thinnest. By level, AA is
the sweet spot: AP 0.837 for debut, 0.729 for top-100. Players with no level
recorded at snapshot are the weakest slice. Across years-in-pro, top-100
probability is only meaningful through yip 4 — after that the event essentially
stops occurring, a property of the sport rather than the model.

---

## 6. Known gaps

1. ~~RK and A− carry no advanced stats.~~ **Closed 2026-09-02** — both
   backfilled to 99.7% / 99.4%. Every MiLB level is now complete.
2. **The advanced metrics are not in the panel.** Computed and validated,
   consumed by nothing. Largest single piece of unrealized headroom.
3. **Splits, fielding, park factors and Statcast tables are empty.** Pullers
   written and tested; not yet run.

3a. **RESOLVED — TOP_100_PROSPECT era bias in the label.** The raw scraped
   lists carry 100 names for every year 2004–2026, but integration dropped
   ~20/year in the early panel, so only 78.0 of 100 slots were present for
   2004–2011 against 91.0 for 2012–2015. Since this sits inside the
   *target*, it contaminated every metric computed against it. 220 rankings
   recovered by unique-name match with ±2-year era corroboration
   (`data/backfills/top100_recovery.py`); ambiguous names are skipped, never
   guessed. Slots filled now run 94–97 early and 98–100 late — a 13-point
   gap reduced to 4. Players carrying `year_top_100` went 1,549 → 1,648.
   **This changes the training target: the v2.1c headline numbers were
   computed against the biased label and are stale until a retrain.**

   A competing diagnosis — that the bias came from MLB Pipeline publishing a
   Top 50 before 2012 — was tested and rejected. True of Pipeline, but this
   database holds no Pipeline rows before 2016; its 2004–2015 source is
   Baseball America, which published a full Top 100 throughout (36–49 names
   per year rank 51–100). Capping the label at top-50 would have discarded
   real signal without touching the actual defect.
4. **No pitcher velocity at any minor-league level.** `velo_avg` remains 0%
   populated. Verified directly: a 2025 Double-A game feed returns no
   `startSpeed` and an empty `breaks` object. Velocity is arguably the
   strongest single MiLB→MLB pitcher signal and there is no public source
   below MLB.
5. **No zone-aware discipline below MLB.** Chase rate and zone contact need
   pitch-location data that only Statcast provides. Swing, whiff and contact
   rates are available at every level; the split between them is not.
6. ~~Batted-ball direction unavailable below MLB.~~ **Closed** — see
   § 2.1. Spray angle and pull% are computable at every level back to 2007.
7. **STAR_PLUS_ELITE needs isotonic recalibration** at h≥4.
8. **The 2026 draft class pull is broken** (`draft_align` / `ncaa_bbStats`) and
   has been accepted as low priority.
