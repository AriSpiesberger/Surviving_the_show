# Railway deploy: prospect pipeline

Four scheduled jobs:

| Job | When (UTC) | What |
|---|---|---|
| `daily_data` | 04:30 daily | Pull MiLB stats: all levels, current season → `prospects.db` |
| `weekly_score` | 09:00 Monday | Re-score with v1.17 bundle → `buy_list_v1.17_FINAL.csv` |
| `daily_prices` | 13:00 daily | eBay buy-now prices for buy list + holdings → `prices/` |
| `daily_digest` | 14:00 daily | Email digest + 2x alerts via SendGrid |

## One-time setup

1. **Volume**: in Railway, attach a persistent volume at `/data`.
2. **Seed data**: copy `prospects.db`, the v1.17 model bundle from `models/`, and an initial `buy_list_v1.17_FINAL.csv` into `/data/`.
3. **Env vars** (Railway service settings):
   - `EBAY_BEARER_TOKEN` (or `EBAY_CLIENT_ID` + `EBAY_CLIENT_SECRET`)
   - `EBAY_ENV=production`
   - `SENDGRID_API_KEY`
   - `ALERT_FROM` (verified SendGrid sender)
   - `ALERT_TO` (your inbox)
   - `PROSPECT_DB=/data/prospects.db`
   - `PRICES_DIR=/data/prices`
   - `HOLDINGS_PATH=/data/holdings.csv`
   - `ALERTS_STATE_PATH=/data/alerts_state.json`
4. **Holdings file** at `/data/holdings.csv`:
   ```
   card_id,player_id,name,denominator,grade,buy_date,buy_price_usd,ebay_item_id,notes
   ```
   `denominator=0` (base) and `grade` empty or `raw` for cards to be tracked / alerted on.

## Manual run (local dry runs)

```
python -m prospects.deploy.daily_data --season 2026 --db prospects.db
python -m prospects.deploy.daily_prices --buy-list buy_list_v1.17_FINAL.csv --db prospects.db --out-dir prices --holdings holdings.csv
python -m prospects.deploy.alerts --holdings holdings.csv --prices-dir prices --state alerts_state.json --dry-run
```

## What scope the pricing covers

Per standing rules:
- **Base 1st Bowman Chrome autos only** — `/99`, `/499`, colored parallels excluded by [listing_parser.py](../market/listing_parser.py).
- **Raw cards only** — PSA / BGS / SGC / CGC / CSG slabs and "gem mint N" titles excluded by the same parser.
- The `lowest_buynow_price` column is the actionable floor for buy decisions and the trigger value for 2x alerts.

## Alert semantics

A held card fires a 2x alert when **today's `lowest_buynow_price` for that player's base raw 1st Bowman Chrome auto ≥ 2 × `buy_price_usd`**. Fired card_ids are persisted in `alerts_state.json` so the alert only fires once per card. Delete the entry from that JSON to re-arm.

## Required artifacts on the volume

The `weekly_score` pre-flight check expects these to be present on `/data`:

```
models/event_classifiers_v1.17_prod.pkl
models/debut_lasso_universe_v1.17h.pkl
models/top100_lasso_v1.17h.pkl
models/model_b_outcomes_v1.17h.pkl
models/player_position_from_stats.csv
panels/panel_v1.17.npz
prospects_snapshot.db
scripts_v17/score/score_panel_v17.py
scripts_v17/buylist/build_v17_buylist.py
```

If any are missing the job exits 3 without running. Add new ones via Railway's volume browser or a one-off shell.

## Exit codes (weekly_score)

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | scoring failed |
| 2 | buylist build failed |
| 3 | required artifacts missing |
