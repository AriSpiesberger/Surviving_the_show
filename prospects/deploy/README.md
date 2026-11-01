# deploy

The scheduled jobs, run through [`ops/run_job.ps1`](../../ops/run_job.ps1). The
[runbook](../../docs/runbook.md) covers the schedule and the retrain.

| Module | Purpose |
|---|---|
| `daily_data.py` | Pull current-season stats into `prospects.db` |
| `weekly_score.py` | Full retrain, leak-audit gate and buy list |
| `v3_sidebyside.py` | Build v3 and v3.5 beside the live sheet without replacing it |
| `daily_prices.py` | eBay buy-now prices for the buy list and holdings |
| `debut_comps.py` | eBay comps for recent debuts |
| `alerts.py` | Daily email digest and 2x alerts through SendGrid |

## Pricing scope

- Base 1st Bowman Chrome autos only. `market/listing_parser.py` excludes numbered parallels.
- Raw cards only. The same parser excludes graded slabs.
- `lowest_buynow_price` is the floor used for buy decisions and alerts.

## Alerts

A held card fires an alert when its `lowest_buynow_price` reaches twice its `buy_price_usd`. Fired
card IDs are saved in `portfolio/alerts_state.json`, so each fires once. Delete an entry to re-arm
it.
