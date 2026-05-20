# Prospect Classifier — Handoff Guide for Claude Code

This document is for whoever picks up this project next (Claude Code on a machine with internet access).

## Current state

We built the **foundation** of a probabilistic prospect classifier:

- ✓ `schema.py` — data types with event-based outputs (8 `CareerEvent`s)
- ✓ `outcome_labels.py` — turns career stats into binary event labels
- ✓ `storage.py` — SQLite layer (7 tables)
- ✓ `ingestion/pybaseball_loader.py` — draft data + MLB outcomes
- ✓ `ingestion/milb_stats.py` — MiLB stats from MLB Stats API (armstjc method)
- ✓ `ingestion/ncaa_loader.py` — college stats from ncaa_bbStats package
- ✓ `ingestion/run_bulk_pull.py` — orchestrator
- ✓ `tests/test_foundation.py` — 53 unit tests (all passing)

What's **not yet built**:

- ✗ Feature engineering (transform raw data → numeric feature vector)
- ✗ Classifier training (logistic regression / gradient boosting per event)
- ✗ Prediction pipeline (current prospects → ProspectPrediction)
- ✗ Card EV calculation (P(events) + multipliers → expected card value)
- ✗ Integration with scanner.py (Stage 3 of evaluation)

## Why we stopped here

The original development environment (Claude.ai chat sandbox) cannot reach the data hosts (baseball-reference.com, statsapi.mlb.com, raw.githubusercontent.com — all blocked with `host_not_allowed`). The ingestion code was written based on **documented APIs** for pybaseball/armstjc/ncaa_bbStats but has NOT been run against real data.

**Your first job is to verify the ingestion works.** API column names and behaviors may have changed since this was written. Errors will surface only when run against real services.

## The handoff checklist

### Step 1: Install dependencies

```bash
cd bowman-scanner
pip install pybaseball ncaa_bbStats pandas requests --break-system-packages
```

### Step 2: Run the foundation tests (should already pass)

```bash
python -m prospects.tests.test_foundation
```

Expected: "ALL TESTS PASSED ✓" (53 checks).

### Step 3: Run diagnostics

```bash
python -m prospects.ingestion.run_bulk_pull --phase diagnostics
```

This probes each data source. Output tells you what works:
- `[diag] playerid_lookup OK: ...` — pybaseball reaches Chadwick register
- `[diag] lahman.batting OK: N rows` — Lahman accessible
- `[diag] amateur_draft(2021,1) OK: N rows` — draft data accessible
- `[diag] team list (AA 2024): status=200` — MLB Stats API reachable
- `[diag] ncaa_bbStats available` — NCAA package installed

**If any diagnostic fails:**
- pybaseball errors usually mean the underlying URL changed. Check pybaseball GitHub issues.
- MLB Stats API 403 means user-agent might need updating (look at armstjc's current scripts).
- ncaa_bbStats missing → `pip install ncaa_bbStats`

### Step 4: Pull draft data (smallest test first)

```bash
python -m prospects.ingestion.run_bulk_pull --phase draft --start 2020 --end 2021
```

This pulls 2 years of draft data (smaller test). Expected to populate the `prospects` table with ~2000 rows.

**Likely issues to watch for:**

1. **Column name mismatches.** pybaseball's draft DataFrame may have different column names than expected ("Name" vs "Player" vs "name"). Look at the `get_col` calls in `pybaseball_loader.py:pull_draft_data` and add candidate names if needed.

2. **Year format issues.** Some years may have different DataFrame shapes. Wrap individual year pulls in try/except (already done).

3. **Rate limiting.** pybaseball hits Baseball Reference HTML pages. Don't pull all years at once. Use `time.sleep(0.5)` between calls (already in place).

### Step 5: Once draft works, pull MLB outcomes

```bash
python -m prospects.ingestion.run_bulk_pull --phase outcomes
```

This is the trickiest module. For each prospect in the database, it tries to:
1. Look up MLBAM ID via Chadwick register
2. Pull career stats from Lahman
3. Pull All-Star/award/HOF data from Lahman
4. Build a `CareerOutcome` and label it

**Known limitations of the current code:**

- The Lahman batting stats lookup uses `playerID` column which is bbref-format, but our `player_id` field is MLBAM-format for some players. **You'll need to add ID translation** (Chadwick register has the mapping).

- The `_build_outcome_for_player` function is partial. It correctly pulls All-Star/award/HOF counts but the `career_pa`, `career_ip`, `career_war` calculations need real verification. WAR isn't in Lahman directly — you might need to use `pybaseball.batting_stats_bref()` or join with FanGraphs.

- `db_get_best_rank` is a placeholder. It should call `db.best_rank(player_id)` once rankings are populated.

### Step 6: Pull MiLB stats (small first)

```bash
python -m prospects.ingestion.run_bulk_pull --phase milb --start 2023 --end 2024
```

Start small to verify the MLB Stats API endpoint still works. Expected:
- Populates the `season_stats` table
- ~5,000-15,000 rows per level per year

**This is the most likely to work as written** — the URL is exactly what armstjc uses and it's the official MLB endpoint.

### Step 7: NCAA stats

```bash
python -m prospects.ingestion.run_bulk_pull --phase ncaa
```

Only useful after Step 4 (draft data) populates college-origin players. Pulls college stats for college draftees.

### Step 8: Full historical pull (once individual phases work)

```bash
python -m prospects.ingestion.run_bulk_pull --phase all --start 2005 --end 2024
```

This will run for hours. ~20 years × 4 levels × 2 stat types × ~30 teams per level = ~5000 API calls. With 1s sleep between teams, that's ~1.5 hours just for MiLB.

## What to build next (after ingestion works)

### Feature engineering (`prospects/features/`)

Transform raw `Prospect` + `SeasonStats` records into a numeric feature vector for the classifier. Key features:

```python
def build_features(prospect_id, db, as_of_year) -> dict:
    # Pedigree
    "draft_round", "draft_pick", "signing_bonus_log", "age_at_signing",
    "is_international", "is_high_school_draftee", "is_college_draftee",
    
    # Performance (most recent season at each level reached)
    "best_woba_so_far", "best_iso_so_far", "best_k_pct_so_far",
    "best_bb_pct_so_far", "best_fip_so_far", "best_k9_so_far",
    
    # Age-relative-to-level
    "age_vs_level_avg",  # negative = young for level
    
    # Trajectory
    "promotion_velocity",  # levels per year
    "highest_level_int",   # 1=DSL ... 6=AAA
    
    # Risk
    "tj_history", "has_injury",
    
    # Position premium
    "is_premium_defensive_position",  # SS/C/CF
```

### Classifier (`prospects/classifier/`)

For each of 8 `CareerEvent`s, train a separate binary classifier:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

# Per event:
#   X = features for prospects at "drafted+0 years" (their state when first eligible to be ranked)
#   y = whether that prospect eventually triggered the event
# Train. Calibrate. Output P(event) with confidence interval.
```

Monotonicity isn't strictly needed across our events (TOP_25 ⊆ TOP_100 is natural; ALL_STAR_THREE ⊆ ALL_STAR_ONCE is logical; but the model can violate these slightly without breaking anything).

### EV calculation (`prospects/ev/`)

Given a `ProspectPrediction` (P at each event) and an `EventMultiplier` table (size model), compute expected card value:

```python
def card_ev(prediction, current_price, multipliers) -> CardEV:
    ev = current_price  # baseline
    for event, prob in prediction.events.items():
        mult = multipliers[event]
        # Expected value contribution from this event
        ev += prob.p_mean * current_price * (mult.multiplier_mean - 1.0)
    return CardEV(...)
```

### Ranking (`prospects/ranking/`)

For all cards we know about, compute EV, sort descending. Top of list = best buys.

### Scanner integration

Modify `scanner.py` to add Stage 3 after AI evaluation:
- For each listing the scanner finds, look up the prospect's prediction
- Compute card EV
- Append to scanner output

## Things to watch out for

1. **pybaseball is flaky.** Its scrapers break every 6-12 months. Pin to a known-working version once you find one.

2. **MLB Stats API rate limits.** Don't hammer it. armstjc's scripts use 0.1s sleep; mine use 0.2-1.0s. Stay polite.

3. **ID systems are a mess.** MLBAM, FanGraphs, BBref, Retrosheet IDs all coexist. Chadwick register maps between them. Use it.

4. **NCAA data is shallow.** Only 2021-2025 player stats. Won't help for older training data, only for current college draftees.

5. **Tool grades and prospect rankings are still missing.** Neither is fully covered by the free stack. Manual seeding from MLB Pipeline / Baseball America may be needed for current prospects.

6. **The classifier won't work until you have training data.** Don't try to train before Step 5 produces real outcomes.

## File map

```
prospects/
├── __init__.py             — public API
├── schema.py               — Prospect, CareerOutcome, ProspectPrediction, etc
├── outcome_labels.py       — label_career(), base_rates(), describe_cohort()
├── storage.py              — ProspectDB (SQLite)
├── ingestion/
│   ├── __init__.py
│   ├── pybaseball_loader.py    — draft + MLB outcomes
│   ├── milb_stats.py           — MiLB stats from MLB Stats API
│   ├── ncaa_loader.py          — college stats
│   └── run_bulk_pull.py        — orchestrator (run this)
└── tests/
    ├── __init__.py
    └── test_foundation.py      — 53 tests, all passing
```

## Contact for context

The full project history is in transcripts. The user is building this to evaluate prospect cards (Bowman /99 autos) for both buying and EV ranking. Phillies focus but model is general across all prospects.

Side model = classifier (P at each event).
Size model = card price multipliers per event.
EV = combine the two.
