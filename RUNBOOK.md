# Prospect-Card Model — Experiment Runbook

Standardized end-to-end procedure for training, scoring, and validating a new
model version. Pick a version tag (e.g. `v1.14i`, `v1.15a`) and substitute it
everywhere `vXXX` appears below.

---

## 0. Naming conventions

Every experiment produces a fixed set of artifacts named after its version
tag. Do not deviate.

| Artifact | Pattern |
|---|---|
| Panel | `panel_vXXX.npz` + `panel_vXXX.joined.pkl` |
| Hazards | `models/event_classifiers_vXXX.pkl` |
| Player split lists | `models/event_classifiers_vXXX_hazard_cal_players.txt`<br>`models/event_classifiers_vXXX_lasso_fit_players.txt`<br>`models/event_classifiers_vXXX_lasso_val_players.txt` |
| Calibrated hazards | `models/event_classifiers_vXXX_calibrated.pkl` |
| Scored slices (calibrated, with raw retained as `p_*_raw`) | `vXXX_fit_long.csv`<br>`vXXX_val_long.csv` |
| Lasso | `lasso_vXXX_td.pkl` |
| Model B | `model_b_outcomes_vXXX.pkl` |
| Buy list (raw) | `buy_list_vXXX_STRICT_2026.csv`<br>`buy_list_vXXX_HIGHCONVICTION_2026.csv` |
| Buy list + prices | `buy_list_vXXX_*_with_prices.csv` |
| Buy list final | `buy_list_vXXX_*_FINAL.csv` |
| Buy list tiered | `buy_list_vXXX_*_TIERED.csv` |
| Buy list filtered | `buy_list_vXXX_*_FILTERED.csv` |
| Validation reports | `val_vXXX_report.txt`, `val_vXXX_bucket.csv`, `val_vXXX_walkforward.csv`, `val_vXXX_long.csv`, `val_vXXX_lasso_pctile.csv`, `val_vXXX_lasso_score.csv`, `val_vXXX_threshold_curve.csv` |
| Backtest | `backtest_vXXX_snap2022.csv` |

---

## 1. Pre-flight (data state)

Before any model work:

```bash
# Refresh MiLB stats (current year only is usually enough)
python -m prospects.ingestion.run_bulk_pull --phase milb \
    --start 2026 --end 2026 --db prospects_snapshot.db

# Rebuild the corrected-position lookup (PA+IP-weighted modal from season_stats)
python -c "
import sqlite3, pandas as pd
c = sqlite3.connect('prospects_snapshot.db')
df = pd.read_sql('SELECT player_id, primary_position, pa, ip FROM season_stats WHERE primary_position IS NOT NULL', c)
df['weight'] = df['pa'].fillna(0) + df['ip'].fillna(0) * 3
m = df.groupby(['player_id','primary_position'])['weight'].sum().reset_index()
m = m.sort_values('weight', ascending=False).drop_duplicates('player_id', keep='first')
m[['player_id','primary_position']].rename(columns={'primary_position':'pos_seasonstats'}).to_csv('player_position_from_stats.csv', index=False)
print(f'wrote {len(m):,} player positions')
"
```

The position lookup is consumed by both panel build (via `_apply_corrected_positions`) and Model B fit/apply. **Do not skip** — this fix prevents pitcher/hitter mislabels from polluting `is_pitcher`, `is_premium_position`, etc.

---

## 2. Panel build (`panel_vXXX.npz`)

Use 16 partitions with up to 5 retries each — segfaults are common and
random. Run in background.

```bash
# Each partition runs, retries on failure
for p in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  for try in 1 2 3 4 5; do
    python -m prospects.classifier.build_panel \
        --out panel_vXXX.npz --max-draft-year 2025 --max-year 2026 \
        --n-partitions 16 --partition $p > /dev/null 2>&1
    [ -f panel_vXXX.part$p.npz ] && break
  done
done

# Merge once all 16 partitions exist
python -m prospects.classifier.build_panel --out panel_vXXX.npz \
    --n-partitions 16 --merge
```

Expected size: ~580k rows × 238 features for 46k players.

**If you change `prospects/features/scouting.py` you MUST rebuild the panel.**
Adding a feature changes `N_FEATURES` and breaks all downstream models.

---

## 3. Hazard training (`event_classifiers_vXXX.pkl`)

```bash
python -m prospects.classifier.train_full_v14d \
    --panel panel_vXXX.npz \
    --out models/event_classifiers_vXXX.pkl \
    --seed 42 --hazard-cal-frac 0.075 \
    --lasso-fit-frac 0.075 --lasso-val-frac 0.075
```

**Split** (player-grouped, seed=42):
- **77.5% hazard train** — fits the 8 HistGBT hazards
- **7.5% hazard cal** — fits per-event Beta calibrators (Stage 3.5)
- **7.5% lasso fit** — fits the Lasso composite on CALIBRATED probabilities
- **7.5% lasso val** — held out from everything; only touched by validation

- 8 events trained: TOP_100_PROSPECT, MLB_DEBUT, ESTABLISHED_MLB, ALL_STAR_ONCE, ALL_STAR_THREE_PLUS, MAJOR_AWARD, STAR, ELITE
- HistGBT, 200 trees, max_depth=6, lr=0.05, min_samples_leaf=30
- Right-censoring per `EVENT_POLICY` dict (ESTABLISHED needs min 4 yrs obs, etc.)
- Writes `_hazard_cal_players.txt`, `_lasso_fit_players.txt`, `_lasso_val_players.txt`

## 3.5. Beta calibration on hazard-cal slice

```bash
python -m prospects.classifier.fit_hazard_calibrators \
    --model models/event_classifiers_vXXX.pkl \
    --panel panel_vXXX.npz \
    --players-file models/event_classifiers_vXXX_hazard_cal_players.txt \
    --out models/event_classifiers_vXXX_calibrated.pkl
```

- For each event: score the hazard-cal slice with raw hazards, fit a per-event Beta calibrator (3 params: a, b, c) on the (raw_prob, realized) pairs
- Calibrated probabilities are now honest: predicted X% means observed ≈ X%
- Calibrators saved into the same model pickle under key `"calibrators"`
- The hazard pickle now emits BOTH `("raw", event)` and `event` keys at score time — calibrated by default, raw available for backwards compat

---

## 4. Score fit + val slices (CALIBRATED probs)

```bash
# Use the calibrated model from 3.5
python -m prospects.classifier.score_cal_slice \
    --model models/event_classifiers_vXXX_calibrated.pkl \
    --panel panel_vXXX.npz \
    --players-file models/event_classifiers_vXXX_lasso_fit_players.txt \
    --out vXXX_fit_long.csv

python -m prospects.classifier.score_cal_slice \
    --model models/event_classifiers_vXXX_calibrated.pkl \
    --panel panel_vXXX.npz \
    --players-file models/event_classifiers_vXXX_lasso_val_players.txt \
    --out vXXX_val_long.csv
```

These score each player at every snap_offset 0..10 with **Beta-calibrated**
cumulative hazards. Both files also retain the raw probs as `p_<event>_raw`
columns for diagnostic use. Lasso fits on the calibrated columns.

Segfault-prone — wrap in a 5-try retry loop.

---

## 5. Lasso composite (`lasso_vXXX_td.pkl`)

```bash
python -m prospects.classifier.lasso_composite \
    --long vXXX_fit_long.csv \
    --time-decay "TOP_100_PROSPECT=3,MLB_DEBUT=4" \
    --require-eligible "TOP_100_PROSPECT,MLB_DEBUT" \
    --out-prefix lasso_vXXX_td
```

Lasso consumes the **calibrated** `p_<event>` columns from Stage 4 (not the
`p_<event>_raw` columns). This means Lasso is doing pure ranking refinement
on probability inputs that are already individually honest — no double-cal.

- Time-decay target: weights players who hit events fast
- Filter: only train on rows where player was eligible for both TOP_100 and MLB_DEBUT at the snap (no leak)
- 10 features: 4 raw hazards + age_centered + yip + 4 hazard×yip interactions
- GroupKFold(5) by player_id, alpha selected via LassoCV
- Output is the `buy_score` (single value per player×snap)

---

## 6. Score snap2026 buy list

```bash
python -m prospects.classifier.score_buy_list_v14d \
    --model models/event_classifiers_vXXX.pkl \
    --lasso lasso_vXXX_td.pkl \
    --drop-r1 --drop-already-top100 \
    --out buy_list_vXXX_STRICT_2026.csv
```

`--drop-r1` and `--drop-already-top100` are the **strict arbitrage filter** — drops R1 picks (too expensive) and already-Top-100 players (already priced in). Output ~8,500 rows.

---

## 7. Model B fit (`model_b_outcomes_vXXX.pkl`)

Edit `prospects/classifier/fit_model_b.py`:
```python
FIT = "vXXX_fit_pre2021_raw_long.csv"
VAL = "vXXX_val_pre2021_raw_long.csv"
OUT_MODEL = "model_b_outcomes_vXXX.pkl"
```

Run:
```bash
python -m prospects.classifier.fit_model_b
```

- Trains on fit+val combined (held out of hazard training)
- Filters to debutees 2010-2024 (~650 players)
- Multinomial logistic, GroupKFold(5)
- Outputs 4-class P(cup/utility/regular/breakout | debut)
- Reports OOF log-loss + per-class decile calibration

---

## 8. Apply Model B + eBay merge + EV/edge

```bash
# Merge eBay prices (use existing or fresh pull — see step 9)
python -c "
import pandas as pd
prices = pd.read_csv('prices_bowman_chrome_auto_v13.csv')  # or fresh
prices = prices[prices.denominator == 0]
agg = prices.groupby('player_id', as_index=False).agg(
    ebay_card_year=('card_year','first'), ebay_denom=('denominator','first'),
    ebay_n_listings=('n_listings','sum'),
    ebay_price_min=('price_min','min'), ebay_price_p25=('price_p25','median'),
    ebay_price_median=('price_median','median'), ebay_price_mean=('price_mean','mean'),
    ebay_price_p75=('price_p75','median'), ebay_price_max=('price_max','max'))
df = pd.read_csv('buy_list_vXXX_STRICT_2026.csv').merge(agg, on='player_id', how='left')
df.to_csv('buy_list_vXXX_STRICT_2026_with_prices.csv', index=False)
df[df.buy_score >= 1.0].to_csv('buy_list_vXXX_HIGHCONVICTION_2026_with_prices.csv', index=False)
"

# Update MODEL constant in apply_model_b.py to point at vXXX
python -m prospects.classifier.apply_model_b \
    buy_list_vXXX_STRICT_2026_with_prices.csv buy_list_vXXX_STRICT_FINAL.csv
python -m prospects.classifier.apply_model_b \
    buy_list_vXXX_HIGHCONVICTION_2026_with_prices.csv buy_list_vXXX_HIGHCONVICTION_FINAL.csv

# Rewrite to bake corrected position into primary_position column
python -c "
import pandas as pd
for f in ['buy_list_vXXX_STRICT_FINAL.csv','buy_list_vXXX_HIGHCONVICTION_FINAL.csv']:
    df = pd.read_csv(f)
    df['primary_position'] = df['position_corrected']
    df = df.drop(columns=['pos_seasonstats','position_corrected'])
    df.to_csv(f.replace('.csv','_v2.csv'), index=False)
"
```

EV pop magnitudes (in `apply_model_b.py:54`): `{no_debut:3, cup:15, utility:25, regular:50, breakout:150}` — these are hand-tuned and the largest source of dollar uncertainty.

---

## 9. eBay fresh pull (optional but recommended weekly)

```bash
# Build input CSV from buy list
python -c "
import sqlite3, pandas as pd
buy = pd.read_csv('buy_list_vXXX_HIGHCONVICTION_FINAL_v2.csv')
c = sqlite3.connect('prospects_snapshot.db')
prosp = pd.read_sql('SELECT player_id, draft_year, is_international FROM prospects', c)
stats = pd.read_sql('SELECT player_id, MIN(season_year) AS start_year FROM season_stats GROUP BY player_id', c)
m = buy[['player_id','name']].merge(prosp, on='player_id', how='left').merge(stats, on='player_id', how='left')
m['draft_year'] = m['draft_year'].astype('Int64')
m['is_international'] = m['is_international'].fillna(0).astype('Int64')
m['start_year'] = m['start_year'].astype('Int64')
m['composite_score_raw'] = buy['buy_score']
m.to_csv('ebay_input_vXXX_HC.csv', index=False)
"

# Pull (~1-2 min for 220 names with 6 workers)
python -m prospects.scripts.fetch_prospect_prices \
    --grades ebay_input_vXXX_HC.csv --top-n 250 \
    --sort-by composite_score_raw --workers 6 \
    --out prices_vXXX_HC.csv

# Re-merge fresh prices and recompute EV/edge (script in apply_model_b.py)
```

---

## 10. Validation suite (REQUIRED for every experiment)

### 10a. Bucket × event × walk-forward report

```bash
python -m prospects.classifier.standard_validation \
    --model models/event_classifiers_vXXX.pkl \
    --max-eval-entry-year 2020 --observe-through 2026 \
    --out-prefix val_vXXX
```

Outputs:
- `val_vXXX_report.txt` — human-readable
- `val_vXXX_bucket.csv` — per (bucket, event) at snap_offset=2
- `val_vXXX_walkforward.csv` — per (snap_offset, event) 0..10
- `val_vXXX_long.csv` — raw per-row predictions

**Reads to check:**
- AUC ≥ 0.80 on MLB_DEBUT and ESTABLISHED at aggregate
- ECE ≤ 0.05 on aggregate (R1 bucket can be higher pre-Lasso)
- Lift@5% ≥ 5× on TOP_100 and MLB_DEBUT at snap_offset 2-4
- Walk-forward AUC sharpens as offset increases until ~offset 5

### 10b. Per-yip 50% threshold curve

```bash
python -m prospects.classifier.validate_lasso v14i_val_pre2021_raw_long.csv lasso_vXXX_td.pkl
# (edit script to point at vXXX val file)
```

Outputs `val_vXXX_lasso_pctile.csv`, `val_vXXX_lasso_score.csv`.

Plus the **threshold curve** (compute inline):
```python
# For each yip, find smallest buy_score where rolling observed debut rate >= 0.50
# Save to val_vXXX_threshold_curve.csv
```

Expected shape: T50 climbs from ~0.30 (yip 0-1) to ~2.0+ (yip 5-6). Use these for the per-yip filter in step 12.

### 10c. Model B calibration

Captured during `fit_model_b` run. Verify in stdout:
- OOF log-loss < baseline (priors) log-loss
- Top-decile breakout pred within ±5pp of obs
- Top-decile cup pred within ±10pp of obs
- Top-decile utility pred can drift up to +15pp (known weakness)

Save the printed summary into `val_vXXX_modelB_calibration.txt`.

### 10d. Buy-list backtest

Score the model on a historical snap (snap_year=2022 is the right choice — 4 years of forward data through 2026) and check how the top-N picks actually realized.

```bash
python -m prospects.classifier.score_buy_list_v14d \
    --model models/event_classifiers_vXXX.pkl \
    --lasso lasso_vXXX_td.pkl \
    --snap-year 2022 --min-entry-year 2017 --max-entry-year 2022 \
    --drop-already-top100 \
    --out backtest_vXXX_snap2022.csv

# Then compute realized rates against career_outcomes
python -c "
import sqlite3, pandas as pd
bt = pd.read_csv('backtest_vXXX_snap2022.csv')
c = sqlite3.connect('prospects_snapshot.db')
co = pd.read_sql('SELECT player_id, mlb_debut_year, year_established_mlb, year_top_100 FROM career_outcomes', c)
m = bt.merge(co, on='player_id', how='left')
m['debuted_by_2026'] = (m.mlb_debut_year.notna() & (m.mlb_debut_year <= 2026) & (m.mlb_debut_year > 2022)).astype(int)
m['established_by_2026'] = (m.year_established_mlb.notna() & (m.year_established_mlb <= 2026)).astype(int)
m['top100_by_2026'] = (m.year_top_100.notna() & (m.year_top_100 <= 2026) & (m.year_top_100 > 2022)).astype(int)
for n in [50, 100, 250, 500, 1000]:
    top = m.sort_values('buy_score', ascending=False).head(n)
    print(f'Top {n:>4d}: debut={top.debuted_by_2026.mean():.1%}  '
          f'estab={top.established_by_2026.mean():.1%}  '
          f'top100={top.top100_by_2026.mean():.1%}')
"
```

**Expected reads** (v1.14i baseline):
- Top 50: ~60-70% debut by 2026, ~25% established
- Top 100: ~50-60% debut, ~18% established
- Top 250: ~35% debut, ~10% established
- Top 500: ~25% debut, ~5% established

If your new version moves the top-100 debut rate up by ≥5pp without sacrificing the tail, it's a real improvement.

---

## 11. Per-yip filter application

Compute the empirical T50 from val 10b, then apply:

```python
THRESH = {  # min buy_score for ~50% MLB_DEBUT chance per yip
    0: 1.20, 1: 1.00, 2: 1.35, 3: 1.50,
    4: 1.65, 5: 2.00, 6: 2.00, 7: 2.00,
}
df['min_score_50pct'] = df.years_in_pro.clip(0,7).astype(int).map(THRESH)
df['passes_filter'] = df.buy_score >= df['min_score_50pct']
df[df.passes_filter].to_csv('buy_list_vXXX_HC_FILTERED.csv', index=False)
```

**Update THRESH from each version's actual 10b output** — these are not constants, they shift with re-trains.

---

## 12. Comparison vs baseline

To declare a new version better:

1. Bucket report aggregates: AUC, BSS, ECE, lift@5% — within 0.02 AUC of baseline AND improved in at least one bucket without regression elsewhere
2. Walk-forward sharpening: AUC at offsets 2-4 ≥ baseline
3. Model B OOF log-loss: lower than baseline
4. Buy-list backtest top-100 realized rates: ≥ baseline on debut+established combined

Document the side-by-side in `val_vXXX_vs_baseline.md`.

---

## Anti-patterns to avoid

- **Don't fit Lasso on raw hazards** — that defeats the purpose of the dedicated calibration slice. Lasso consumes calibrated probabilities only.
- **Don't fit calibrators on the lasso-fit slice** — calibrators MUST come from the hazard-cal slice. Cross-contamination breaks the held-out guarantee.
- **Don't use the lasso-val slice during training of anything** — it's the only truly held-out slice; touching it once invalidates all reported metrics.
- **Don't change `EVENT_POLICY` without backfilling old comparisons** — right-censoring policy shifts the entire training set.
- **Don't rebuild the panel without bumping the version tag** — silent data drift will confuse later comparisons.
- **Don't use `prospects.primary_position` directly** — always overlay `player_position_from_stats.csv`. The prospects-table positions have ~7,000 mislabels.
- **Don't pull eBay on the full STRICT (8,500 names) without checking quota** — HC (220) is usually enough.

---

## Quick reference: full pipeline command sequence

For a fresh vXXX experiment, assuming code unchanged from v1.14i:

```bash
# 1. Pre-flight (run before each experiment)
python -m prospects.ingestion.run_bulk_pull --phase milb --start 2026 --end 2026 --db prospects_snapshot.db
# rebuild player_position_from_stats.csv

# 2-3. Panel + hazards (77.5/7.5/7.5/7.5 split)
# loop 16 partitions, merge
python -m prospects.classifier.train_full_v14d --panel panel_vXXX.npz --out models/event_classifiers_vXXX.pkl --hazard-cal-frac 0.075 --lasso-fit-frac 0.075 --lasso-val-frac 0.075

# 3.5. Beta calibration on hazard-cal slice
python -m prospects.classifier.fit_hazard_calibrators --model models/event_classifiers_vXXX.pkl --panel panel_vXXX.npz --players-file models/event_classifiers_vXXX_hazard_cal_players.txt --out models/event_classifiers_vXXX_calibrated.pkl

# 4-5. Score fit/val with CALIBRATED probs, fit Lasso
python -m prospects.classifier.score_cal_slice --model models/event_classifiers_vXXX_calibrated.pkl --panel panel_vXXX.npz --players-file models/event_classifiers_vXXX_lasso_fit_players.txt --out vXXX_fit_long.csv
python -m prospects.classifier.score_cal_slice --model models/event_classifiers_vXXX_calibrated.pkl --panel panel_vXXX.npz --players-file models/event_classifiers_vXXX_lasso_val_players.txt --out vXXX_val_long.csv
python -m prospects.classifier.lasso_composite --long vXXX_fit_long.csv --time-decay "TOP_100_PROSPECT=3,MLB_DEBUT=4" --require-eligible "TOP_100_PROSPECT,MLB_DEBUT" --out-prefix lasso_vXXX_td

# 6-8. Buy list + Model B
python -m prospects.classifier.score_buy_list_v14d --model models/event_classifiers_vXXX.pkl --lasso lasso_vXXX_td.pkl --drop-r1 --drop-already-top100 --out buy_list_vXXX_STRICT_2026.csv
# edit FIT/VAL/OUT_MODEL in fit_model_b.py to vXXX
python -m prospects.classifier.fit_model_b
# edit MODEL in apply_model_b.py to vXXX
# merge eBay prices, run apply_model_b.py

# 10. Validation (all 4)
python -m prospects.classifier.standard_validation --model models/event_classifiers_vXXX.pkl --max-eval-entry-year 2020 --observe-through 2026 --out-prefix val_vXXX
python -m prospects.classifier.validate_lasso vXXX_val_pre2021_raw_long.csv lasso_vXXX_td.pkl
# compute threshold curve
# run backtest_vXXX_snap2022 script

# 11. Filter and ship
# apply per-yip T50 from val_vXXX_threshold_curve.csv
```
