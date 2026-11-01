# The plate-discipline block is corrupt across ~40% of the panel

> Written 2026-08-15 against model v2.1c. Row counts predate the 1988-2004 history build.

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
