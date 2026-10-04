# Card-market backtests — findings so far

Status: 2026-10-02. Working notes; numbers regenerate from the scripts named in each section.

## Reproduce

Committed: the scripts and the raw scrape (`raw/product.jsonl`, `raw/setlist.jsonl`,
`raw/mlb_people.jsonl`, `raw/transactions_nov.jsonl`). Not committed: anything derived from
`prospects.db` (panel cache, predictions, position CSVs), so it is rebuilt against the
current database. Run from `research/card_market/`:

```bash
python fetch_debuts.py                 # refresh MLB debut dates (seconds, Stats API)
python fetch_transactions.py           # refresh November 40-man additions
python load.py                         # sanity check: card counts, match rate
python study_debut.py                  # §1    debut event study
python study_basket.py                 # §1b   no-hindsight basket (writes raw/basket_positions.csv)
python study_debut_daily.py            # §1c   day-level timing
python study_rankings.py               # §2    Top-100 lists
python study_momentum.py               # §3    momentum and fundamentals
OMP_NUM_THREADS=2 python light_model.py   # §3b  walk-forward stand-in model (~3 min on 2 threads)
python study_model.py                  # §3b   model vs market
```

All numbers below were produced on a Mac with the 2026-08-24 copy of `prospects.db`;
expect small differences on a newer database.

**To test the production model instead of the stand-in:** write its leak-free historical
predictions to `raw/light_preds.csv` with the same columns (`player_id, S, p_deb1, p_deb2,
p_top100, elig_top100, start`; one row per not-yet-debuted player per season-end
2022–2025, each predicted only from data available at that season-end), then run
`study_model.py`.

Refreshing prices means re-scraping in a browser: run `receiver.py`, open a
sportscardspro.com tab, open `http://127.0.0.1:8765/bridge.html` from it (Chrome blocks the
page from reaching localhost directly), and run `scrape_worker.js` there as a Web Worker.
At the site's rate limit that is about two hours for the 3,553 cards.

## Data

- **Prices**: SportsCardsPro, scraped through Chrome (`scrape_worker.js` → `receiver.py` → `raw/`).
  3,553 base autographs from 66 Bowman Chrome auto sets, 2011–2026: 3,020 first Bowman
  Chrome prospect/draft autos (92% matched to a player in `prospects.db`), 418 Chrome rookie
  autos. Per card: monthly raw / PSA 9 / PSA 10 price back to ~2021–22, and the most recent
  ~30 completed sales per grade (98k sales).
- **Debut dates**: MLB Stats API (`fetch_debuts.py`).
- **Stats, rankings, levels**: `prospects.db` (the Aug 24 copy on this Mac).
- `load.py` builds the tables; `panel.py` builds one raw-price series per player and an
  equal-weight market index.

Limits that apply to everything below:

- Monthly prices are SportsCardsPro's estimate from sold listings. 47% of card-months show no
  change (thin trading). Where it matters, tests are repeated on cards with recent price moves.
- Four years of history (late 2022 →), most of it a falling market.
- Costs assumed: 13.25% + $0.40 on the sale; 7% tax + $1.50 shipping on the buy. On a $15 card
  that is ~30% round trip.
- The production model has not been tested. The OOF predictions on this Mac predate the
  2026-09-17 label-leak fix. §3b uses a lightweight walk-forward stand-in instead.

## 0. The market drifts down

Equal-weight index of raw first autos: about −57% from 2021 to 2026 (log −0.85), −3%/month
through 2023, roughly flat since late 2024. Holding prospect autos passively lost money, so
every result is also reported net of this index. (`panel.py`)

## 1. Debut: a pop, then an 18-month fade  — `study_debut.py`

454 players who debuted 2022+ with a priced first auto. Raw log returns by month relative to
debut:

| Window | Mean | 95% CI (log) | Cards up |
|---|---|---|---|
| −6 → −1 | +11% | +0.05 to +0.15 | 54% |
| −1 → +1 | +27% | +0.19 to +0.29 | 61% |
| 0 → +6 | −13% | −0.20 to −0.07 | 34% |
| +1 → +12 | −32% | −0.45 to −0.30 | 23% |

- Peak is the month after debut; the gain is gone by about month 8.
- The pop is concentrated in cheap, unheralded players: under $10 pre-debut +49% mean,
  $60+ about 0%. Never-top-100 +40% vs market, top-100 +20%. Pitchers +43%, hitters +24%.
- **Rule: sell in the debut month or the month after.**

## 1b. The same trade without hindsight loses money — `study_basket.py`

Each November 2022–2025, buy every not-yet-debuted minor leaguer's first auto; sell one
month after debut, else at the end of the next season. 2,176 positions.

| Top level that season | n | Debuted | Gross | Net | Net win rate |
|---|---|---|---|---|---|
| AAA | 376 | 43% | +37% | −26% | 19% |
| AA | 733 | 15% | +9% | −41% | 11% |
| A+ | 631 | 3% | +5% | −44% | 9% |
| A | 436 | 1% | +8% | −40% | 13% |

AA/AAA only: a debuter nets +4% to +31% depending on price band, a non-debuter −32% to
−58%. Break-even debut hit rate on a one-year horizon is 61–89%; base rates are 12–60%.
The sell-at-debut rule beats holding to season end (AAA: −26% vs −31%) but does not make
the basket profitable. 2024 and 2025 baskets are much better than 2022–23 (−15% and −10%
dollar-weighted) because the market stopped falling.

## 1c. Day level: the pop is 2-3x the monthly figure and lasts about four days — `study_debut_daily.py`

Individual raw sales for 70 recent debuts (1,206 sales), each relative to the card's own
median sale 15–90 days before debut:

| Days from debut | Mean vs baseline | Sales/day |
|---|---|---|
| −90 → −4 | 0% | 6–8 |
| −3 → −1 | +59% | 15 |
| 0 → 1 | +70% | 64 |
| 2 → 3 | +85% | 24 |
| 4 → 7 | +44% | 17 |
| 15 → 30 | +20% | 6 |
| 61 → 120 | −25% | 1 |

- Under-$15 cards peak near +145% on days 2–3; $15+ cards near +43% on days 0–1.
- Prices do not anticipate the call-up: flat until about three days before. Buying late
  costs nothing and shortens the hold.
- 18% of sales in days −3..+3 still cleared within 10% of the old price.
- Listing side (your daily eBay snapshots): only 3 players had coverage across debut day;
  in all three the lowest Buy-It-Now was unchanged through day 0 and rose 25–63% on days
  1–2. Suggestive that asks lag sales, far too few cases to rely on.
- Sample caveat: only cards whose last ~30 sales still reach back before the debut, so it
  skews to recent debuts and thinner cards.
- **Rule: list held cards the day the call-up is reported; the window is days 0–3.**

PSA 10 copies of the same players move less on monthly data: pop +16% vs +27% raw, fade
−18% vs −35% (335 / 257 players). Moving up in price does not rescue the trade.

## 1d. 40-man additions predict debuts but the monthly-exit trade still loses

November "selected the contract" transactions (Rule 5 protection), MLB Stats API
(`fetch_transactions.py`). Not-yet-debuted AA/AAA players with a priced first auto, bought
in December 2022–2025:

| | n | Debuted next season | News move Oct→Dec vs market | Net (monthly exit) |
|---|---|---|---|---|
| Added to 40-man | 88 | 70% | +11% | −26% (CI −36% to −15%) |
| Not added | 1,021 | 20% | 0% | −36% |

A 70% hit rate is not enough when the exit is a monthly-average price. Rough estimate
combining this hit rate with the day-level pop (two different samples, so treat as an
estimate): +26% net on a debuter sold in days 0–3, −40% on a miss, about +6% per position.

Entering later on the calendar does not help. The strongest candidates debut early (17% of
adds in April, 13% in May), so the remaining pool weakens: rest-of-season debut rate 66%
buying in December, 58% in April, 42% in June, 28% in July, with net between −24% and −34%
at every entry month. Late entry only pays if it is triggered by an event, not a date.

## 2. Top-100 lists: the move happens before publication — `study_rankings.py`

238 new entrants, 220 holdovers, 67 drops, list years 2022–2026.

- New entrants: +23% raw July→December *before* the list (83% beat the market), +7% over
  publication (Dec→Feb), then −7% Feb→Dec.
- Holdovers: +7% Dec→Feb, then −21% Feb→Dec; only 21% of cards rise.
- Dropped: −25% in the run-up, −29% after.
- **Rule: February is the month to sell a ranked prospect.** The buy side needs a forecast
  of who will be newly ranked, made mid-season — that is the model's Top-100 probability,
  untested here.

## 3. Last season's stats predict next season's card return — `study_momentum.py`

Not-yet-debuted minor leaguers, bought in November, price ≥ $5, n=944.

- Rank correlation of season performance percentile with next-season return vs market:
  +0.35 (2022), +0.26, +0.17, +0.09 (2025). Holds on fresh prices only (+0.21 to +0.24).
- Regression with past 6-month price change, age-for-level, price and year effects:
  performance t=3.4, younger-for-level t=2.5, past price change not significant.
- It is a relative effect: top quartile about 0% raw, bottom quartile −30 to −40%.
  Long-only it does not clear costs. **Use: don't hold poor performers through the winter.**
- Best cell — top-quartile performance, young for level, AA/AAA: 57 positions, 53% debuted,
  25% doubled, pooled net about +19%, but −33%, −17%, +70%, +15% by year on 6–18 positions.
  Promising, not established.
- Pure price momentum (6-month formation, 6-month hold): spread +4.6%, t=1.4. Nothing.

## 3b. Model probabilities vs the market — `light_model.py`, `study_model.py`

Probabilities here come from a **lightweight walk-forward stand-in**, not the production
stack: one gradient-boosting model per target on the same leak-fixed panel features, each
season-end Y trained only on landmarks whose outcome was known by then. Out of time it
scores AP 0.45–0.50 for "debuts next season" (base rate 2.4%) and 0.40–0.57 for "new
Top-100 entrant". Treat what follows as a floor for the production probabilities.

2,015 positions: not-yet-debuted players with a priced first auto, bought in November
2022–2025, sold one month after debut else at the end of the next season, net of costs.

**The probability is calibrated and it orders returns.**

| P(debut next season) | n | Debuted | Median price | Gross | Net (95% CI) | Win rate |
|---|---|---|---|---|---|---|
| 0–0.05 | 980 | 0.5% | $4 | +5% | −48% (−51 to −44) | 8% |
| 0.05–0.15 | 400 | 10% | $4 | +9% | −46% (−51 to −41) | 8% |
| 0.15–0.3 | 236 | 19% | $5 | +12% | −38% (−45 to −30) | 12% |
| 0.3–0.5 | 176 | 33% | $9 | +38% | −18% (−33 to +1) | 19% |
| 0.5–0.7 | 123 | 50% | $16 | +26% | −18% (−29 to −7) | 20% |
| 0.7–1.0 | 100 | 83% | $26 | +62% | +15% (−4 to +40) | 43% |

By year the P ≥ 0.7 band nets −22% (2022, n=6), −16% (2023), +29% (2024), +31% (2025).

**The market prices ceiling, not proximity.** Rank correlation of the November price with
P(new Top-100) is +0.60 to +0.71; with P(debut next season) it is +0.13 to +0.35. The three
probabilities explain about half the variance of log price (R² 0.48–0.58).

**"Cheap for its probabilities" predicts returns.** Regress log price on the probabilities
within each year and sort on the residual. To keep price noise out, the residual uses the
September price and the buy is in November. Price ≥ $5, n=899:

| Quartile | n | Return vs market | Net |
|---|---|---|---|
| Cheapest | 226 | +14% | −8% |
| 2 | 224 | −5% | −25% |
| 3 | 224 | −9% | −33% |
| Richest | 225 | −33% | −38% |

Rank correlation of the residual with the forward return is negative in all four years
(−0.31, −0.13, −0.07, −0.16) and survives on cards with fresh prices (+10% vs −18%).

**Combined rule.**

| Rule | n | Debuted | Median price | Gross | Net (95% CI) | Dollar-weighted |
|---|---|---|---|---|---|---|
| P ≥ 0.5 | 221 | 65% | $20 | +43% | −3% (−14 to +10) | −8% |
| P ≥ 0.5, cheap half | 89 | 56% | $10 | +71% | +9% (−15 to +37) | +14% |
| P ≥ 0.5, rich half | 132 | 71% | $30 | +23% | −10% (−19 to −1) | −12% |
| P ≥ 0.7 | 100 | 83% | $26 | +62% | +15% (−4 to +40) | 0% |
| P ≥ 0.7, cheap half | 42 | 74% | $14 | +99% | +34% (−8 to +90) | +42% |
| P ≥ 0.7, rich half | 58 | 90% | $51 | +35% | +1% (−11 to +14) | −7% |

What to hold against this: every confidence interval on a positive rule includes zero; the
median position loses in every rule; five positions (Misiorowski 2024 alone returned 9×)
account for more than all of the P ≥ 0.5 cheap-half profit; 2022–23 lost and 2024–25 won.
In its favour: the exit here is a monthly average, which §1c shows understates the
achievable pop by a factor of two to three.

**Top-100 probability and list season.** Among players not already on a list (price ≥ $5),
P(new Top-100) ≥ 0.02 moved +9% to +12% vs market November→February and +11% to +18%
over the year (n=179); below 0.02, −10% and −18% (n=540).

**40-man status is information the model lacks.** In every probability band, November
40-man additions debuted more often: 30% vs 3%, 64% vs 16%, 65% vs 28%, 61% vs 48%,
100% vs 78%. In a logistic regression on the model's logit, 40-man status has a coefficient
of 1.47 (odds ×4.3). It does not add return — the market reacts to the transaction (+11%)
— but it belongs in the model as a feature.

## 4. Dead ends

- **Calendar seasonality**: ±1.5%/month after removing each year's trend.
- **MLB rookies after their first season**: every performance quartile loses 13–37% raw the
  next season (n=221; the MLB percentile column in the DB looks unreliable).

## 5. Grading — open

Paired actual sales (≥3 raw and ≥3 PSA 10 sales in the last 12 months):

| Raw price | n | PSA 10 / raw |
|---|---|---|
| $10–25 | 146 | 3.8× |
| $25–60 | 108 | 3.5× |
| $60–150 | 71 | 2.7× |
| $150+ | 49 | 2.7× |

With $28 all-in grading, the break-even gem rate falls from ~60% on a $15 card to ~25% at
$25–60 and ~10% at $150+. Not a result yet: raw copies on eBay are likely the ones their
owners judged would not gem, and there is no grading-outcome data here. Needs PSA population
counts at minimum.

## 6. Entry prices

18 base holdings matched to SportsCardsPro: median paid 1.48× the same-month price
($235 vs $167). Shipping and tax explain part of it. Buying at auction/offer nearer the
sold average is worth more than most of the signals above.

## Open

- §3b with the production model's leak-free historical predictions, and with a day 0–3 exit.
- A short-horizon call-up signal (who is called up in the next 30–60 days), since buying
  late is free.
- Numbered parallels, where fixed costs are a smaller share (needs another scrape).
- PSA population data for the grading question.
