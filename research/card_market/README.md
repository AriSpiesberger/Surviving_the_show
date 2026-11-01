# Card-market backtests

Tests whether Bowman Chrome first autographs can be traded profitably around debuts, rankings and
model signals. Costs bind: a $15 card pays about 30% round trip.

Status: working notes through 2026-10-25, on four years of mostly falling prices (late 2022 onward).

## Findings

| Page | Sections | Result |
|---|---|---|
| [The debut trade](findings/debut-trade.md) | §0, §1, §1b-d | A real pop lasting about four days, but the plain trade loses without hindsight |
| [Top-100 lists and momentum](findings/rankings-and-momentum.md) | §2, §3 | The move happens before a list is published |
| [Model vs the market](findings/model-vs-market.md) | §3b, §3c, §7 | P(debut next season) ≥ 0.7 with a days 0-3 exit clears costs, +24% net; call-up model loses |
| [Strategy search](findings/strategy-search.md) | §8-10 | Best rule below; most variants are dead ends |
| [Per-player EV](findings/player-ev.md) | §11 | EV above 0.10 is profitable; ranking by EV is about the same as ranking by P |
| [Pop ceiling and All-Stars](findings/ceiling-and-execution.md) | §12 | Exit execution matters more than predicting the pop; first All-Star years are priced in |
| [Dead ends and open questions](findings/open-questions.md) | §4-6 | What remains untested |

**Best rule so far (§8):** in November, buy the PSA 10 first auto of a player who has reached AAA and
has P(debut next season) ≥ 0.7. Sell in days 0-3 after the debut, otherwise hold to October. This
returned +33% (CI +21 to +46) and was positive in all four years, on a small sample.

## Layout

| Path | Contents |
|---|---|
| `findings/` | Results by topic |
| `load.py`, `panel.py` | Build the card tables and per-player price series |
| `light_model.py`, `prod_probs.py` | Walk-forward stand-in model; production model probabilities |
| `study_*.py` | One script per backtest |
| `ev.py`, `debut_pop.py`, `callup.py` | Per-player EV, debut pop and call-up models |
| `scrapers/` | Stats API fetchers and the browser price scraper |
| `ev_board/` | EV board template and day-level pop scraper |
| `raw/` | Scraped data; anything derived from `prospects.db` stays out of git |

## Reproduce

Run from this folder:

```bash
python scrapers/fetch_debuts.py        # refresh MLB debut dates
python scrapers/fetch_transactions.py  # refresh November 40-man additions
python load.py                         # sanity check: card counts, match rate
python study_debut.py                  # §1 debut event study
python study_basket.py                 # §1b no-hindsight basket
python study_debut_daily.py            # §1c day-level timing
python study_rankings.py               # §2 Top-100 lists
python study_momentum.py               # §3 momentum and fundamentals
OMP_NUM_THREADS=2 python light_model.py   # §3b walk-forward stand-in (about 3 minutes)
python study_model.py                  # §3b model vs market
```

Run `ev.py` and `prod_probs.py` from the repo root with `PYTHONPATH=research/card_market`, because
the model bundle uses relative paths.

To test the production model instead of the stand-in, write its leak-free historical predictions to
`raw/light_preds.csv` with the columns `player_id, S, p_deb1, p_deb2, p_top100, elig_top100, start`:
one row per not-yet-debuted player per season-end 2022-2025, each predicted only from data available
at that season-end. Then run `study_model.py`.

Refreshing prices means re-scraping through a browser:

1. Run `scrapers/receiver.py`.
2. Open a sportscardspro.com tab, then open `http://127.0.0.1:8765/bridge.html` from it. Chrome blocks
   the page from reaching localhost directly.
3. Run `scrapers/scrape_worker.js` there as a Web Worker.

That takes about two hours for 3,553 cards at the site's rate limit.

## Data

- **Prices:** SportsCardsPro, 3,553 base autographs from 66 Bowman Chrome auto sets, 2011-2026. This
  includes 3,020 first Bowman Chrome prospect and draft autos, of which 97% match a player in
  `prospects.db`. Each card has monthly raw, PSA 9 and PSA 10 prices back to about 2021-22, and up to
  30 recent sales per grade.
- **Debut dates:** MLB Stats API.
- **Stats, rankings, levels:** `prospects.db`.

## Limits

- Monthly prices are SportsCardsPro estimates from sold listings. 47% of card-months show no change,
  so trading is thin.
- Assumed costs: 13.25% + $0.40 on the sale; 7% tax + $1.50 shipping on the buy.
- Past real buys paid 1.48× the monthly estimate, which alone erases every edge. Execution near the
  sold average is a prerequisite.
