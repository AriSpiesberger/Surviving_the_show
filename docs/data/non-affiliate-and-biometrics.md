# Non-affiliate contamination, biometrics, and the 2005 floor

> Written 2026-08-15 against model v2.1c. Row counts predate the 1988-2004 history build.

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
