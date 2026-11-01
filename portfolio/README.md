# portfolio

| File | Purpose |
|---|---|
| `holdings.csv` | Cards held, read by the price and alert jobs |
| `alerts_state.json` | Alerts already fired, written by `deploy/alerts.py` |
| `card_holdings.csv`, `card_holdings_by_level.csv`, `card_levels.csv` | Holdings views by player and level |

`holdings.csv` columns: `card_id, player_id, name, denominator, grade, buy_date, buy_price_usd,
ebay_item_id, notes`. Use `denominator=0` for base cards, and leave `grade` empty or set it to `raw`
for cards that should trigger alerts.
