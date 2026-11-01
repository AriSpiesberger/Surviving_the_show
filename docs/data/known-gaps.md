# Known gaps

> Written 2026-08-15 against model v2.1c. Row counts predate the 1988-2004 history build.

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
9. **Established, All-Star and award labels stop at 2021.** `career_outcomes` is built from a Lahman
   copy that ends there, so later stars count as negatives in training.
