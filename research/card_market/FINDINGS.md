# Card-market backtests — findings so far

Status: 2026-10-05 (§3c, §7-10 added). Working notes; numbers regenerate from the scripts named in each section.

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

## 3c. The model rule with a days 0-3 exit clears costs — `study_model_daypop.py`

Same November positions and walk-forward probabilities as §3b, with two changes. First, the
stand-in (`light_model.py`) now trains from 1990 on the 1988-2004 database. Its debut AP is the
same as the 2012-start version (0.47-0.49 for 2022-25). Second, a debuter exits at the card's
price the month before debut times the **measured days 0-3 pop** for its price band, instead of
the monthly price a month after debut. The pop multipliers come from individual sales, each card
against its own baseline (`study_callup.pop_by_band`):

| Baseline price | Cards | Days 0-3 vs baseline |
|---|---|---|
| < $10 | 23 | x2.13 |
| $10-15 | 7 | x1.90 |
| $15-30 | 6 | x1.31 |
| $30+ | 16 | x1.40 |

2,044 positions, Nov 2022-2025:

| Rule | n | Debuted | Median price | Net, monthly exit | Net, days 0-3 exit (95% CI) | Median | Dollar-weighted |
|---|---|---|---|---|---|---|---|
| P >= 0.5 | 192 | 64% | $21 | +4% | +13% (+3 to +24) | +2% | +19% |
| P >= 0.7 | 82 | 82% | $27 | +8% | **+24% (+11 to +37)** | +11% | +27% |
| P >= 0.7, rich half | 50 | 88% | $49 | +20% | +25% (+11 to +39) | +13% | +28% |
| P >= 0.7, cheap half | 32 | 72% | $11 | -11% | +22% (-3 to +49) | 0% | +13% |

- With the right exit, "cheap for its probabilities" stops mattering. The edge is high P(debut)
  plus selling into the pop.
- P >= 0.7 is positive in every year: +28%, +6%, +28%, +33% (2022-25). P >= 0.5 relies on 2025:
  -4%, +4%, +5%, +34%.
- Sensitivity: P >= 0.7 nets +14% at 75% of the measured pop and +4% at 50%; it breaks even near
  40%. The pop rests on 52 recent debuts (2025-26, a market that had stopped falling), and the
  average is applied to every hit, so realised sales will scatter around it.
- **The binding risk is entry price.** Everything assumes buying at the SportsCardsPro monthly
  estimate. §6 found past buys paid 1.48x that. Buying at 1.48x erases the edge, so execution has
  to be auction or best-offer near the sold average.
- **Rule (candidate): in November, buy the raw first auto of players with P(debut next season)
  >= 0.7. List the day the call-up is reported and sell in days 0-3.** Use the production model's
  P(debut <= 1 year), which is stronger than the stand-in.

## 7. A short-horizon call-up model does not pay — `callup.py`, `study_callup.py`

Weekly in-season panel, April-September 2010-2025, built from MLB transactions. Rows are
not-yet-debuted AA/AAA players: 575k rows, 12.6k players. Features: level and days there, 40-man
status, the parent club's injured-list placements by position group in the last 14/30 days, its
recent call-ups, September, age, and the walk-forward season-end P(debut). Target: debut in the
next 3-30 days. Walk-forward by season.

- Out of time, AP 0.10-0.16 (base rate 1.2-2.1%); top-1% precision 16-26%. Adding P(debut)
  helps in 2023-25 (+0.02 AP). Timing within a season is hard to predict from transactions.
- Traded with the measured pop as the exit, it loses at every threshold. P >= 0.30 with a 60-day
  hold: 43% hit, -14% net (-22 to -5). $30+ cards at P >= 0.2: 41% hit, -12%. A miss costs ~25-30%
  (fees, tax, shipping, drift); a $30+ hit nets ~+20%; cheap cards pop more but fixed costs eat
  them.
- In-season entry gives up the season-long selection: the November P >= 0.7 basket (§3c) already
  holds most eventual call-ups, at an 82% hit rate.

## 8. Refining the §3c rule: AAA, PSA 10, buy in November, hold to October — `study_strategies.py`

Same engine as §3c (walk-forward P from `light_model.py`, November entry at the monthly price,
debuters exit at the pre-debut price x the days 0-3 pop, others at October Y+1, same costs). The
pop for each grade and band comes from individual sales, with at least 5 cards per band (otherwise
the pooled median). PSA 10: x1.57 for $30+ (32 cards); thinner bands use the pooled x1.48.

**The best rule: P(debut next season) >= 0.7, reached AAA that season, PSA 10 copy.**

| Nov buy, reached AAA | n | Debuted | Median price | Net (95% CI) | Median | Years + |
|---|---|---|---|---|---|---|
| PSA 10, P >= 0.7 | 62 | 87% | $85 | **+33% (+21 to +46)** | +24% | 4/4 |
| PSA 10, P >= 0.7, fresh entry price | 41 | 90% | $90 | +43% (+26 to +62) | +28% | 4/4 |
| PSA 10, P >= 0.6 | 92 | 82% | $71 | +33% (+20 to +50) | +25% | 4/4 |
| Raw, P >= 0.6 | 103 | 83% | $20 | +28% (+13 to +44) | +14% | 4/4 |
| Raw, P >= 0.7, fresh entry price | 51 | 90% | $24 | +24% (+10 to +41) | +12% | 3/4 |

- **AAA is where the edge lives.** At P >= 0.6, raw nets +28% for AAA vs +2% for AA (debut 82% vs
  52%); PSA 10 +33% vs +12%.
- **PSA 10 beats raw.** The median is about twice the raw one and the result depends less on a
  few big wins; fixed costs are small at $70+. The raw edge is partly a stale-price artifact:
  fresh-price raw entries net +13% at P >= 0.6, against +50% for stale ones. PSA 10 holds up on
  fresh prices (+37%), though most of that comes from 2024-25.
- **Not concentrated:** at AAA, P >= 0.5, the top 5 positions are ~45% of dollar profit; the rest
  still profits (raw $848, PSA 10 $2,334 across the basket).
- **Buy in November.** Returns decay by entry month: P >= 0.7 raw nets +24% (Nov), +21% (Dec),
  +15% (Jan), +13% (Feb), +3% (Mar), -2% (Apr).
- **Hold non-debuters to October.** Giving up in June cuts AAA P >= 0.6 from +28% to +18% raw and
  from +33% to +16% PSA 10; September call-ups matter.
- **Weaker within the basket:** international signees (raw -4%, PSA 10 +13%). The summer price
  trend shows no consistent pattern. Pitchers pop more in raw, hitters do better in PSA 10.
- **Which copy:** among AAA P >= 0.6 players with both copies priced (n=89), when the PSA 10 costs
  <= ~2.5x the raw (bottom tertile, median 2.1x), PSA 10 nets +56% (median +49%) against +26% raw,
  and beats the raw copy for 90% of players. At ~6x, raw nets +3% and PSA 10 +22%. Thirty players
  per tertile, and a low ratio can be a stale PSA 10 price, so treat it as a tie-breaker.
- **40-man adds do not help a December entry.** P >= 0.5 and added nets -12% vs +14% not added,
  because the market prices the add in (§1d). The model underrates adds at P 0.2-0.5 (76% debut,
  +18% net), but those are cheap cards and the dollar-weighted net is -8%.

## 9. The Top-100 list-season trade loses — `study_strategies.py`

Buy in November when walk-forward P(new Top-100) is in a given band and the player is not on a
list; sell in February at publication. Every band loses 10-20% after costs, at every price level,
almost every year (n 13-46 per band). The pre-list run-up happens July->December (§2), before a
season-end model can see it; what remains over publication is too small for ~30% costs.

## 10. More dead ends from the strategy search — `study_strategies.py`

- **Two-season holds for "debuts within two years, probably not next year"** (P(debut <= 2y)
  >= 0.5-0.7, P(next season) < 0.5): -4% to +11% raw, -4% to +8% PSA 10. Only the 2024 cohort is
  positive. Another year of drift and fade outweighs the higher debut rate.
- **Buying established rookies after the debut season** (November after debut, hold 12 months):
  every playing-time and performance split loses: raw -29% to -50%, PSA 10 -26% to -36%, every
  year. The post-debut fade (§1) is market-wide, and early MLB success does not stop it.
- **Buying on a first AAA promotion** (new to AAA, transactions panel): -7% to -37%, almost every
  year negative, raw and PSA 10. Only 25-54% debut by October, and the November basket already
  holds the strong cases.

## 11. Per-player EV — `ev.py`, `debut_pop.py`, `prod_probs.py` (2026-10-22)

One number per player for "what is buying his first auto now worth": the §3c/§8 November trade
written as a two-path expectation per dollar of the SportsCardsPro (SCP) price,

    E[exit / p] = P(debut next season) x M_debut(price band) + (1 - P) x M_miss(price band x new-Top-100 tier)

with M = shrunk mean exit multiples from the 2022-25 November positions, then the usual costs. It
also gives a **max-pay price**, the entry at which EV = 0. Current players use the production v3.5
P(debut within 1y) and P(Top-100 within 1y) (`prod_probs.py`), SCP prices and the eBay lowest
Buy-It-Now pulled 2026-10-22 for 396 candidates. Output: `raw/ev_2026.csv`.

**Calibration, leave one year out.** EV > 0.10 realises +13% raw (CI +2 to +24, 3 of 4 years
positive) and +18% PSA 10 (CI +5 to +32). EV between 0 and 0.10 loses on raw (-15%). The top raw bin
over-promises (predicted +29%, realised +11%), so the sheet also carries an isotonic-calibrated EV
(raw tops out near +26%, PSA 10 near +30%). Ranking by EV is about as good as ranking by P(debut);
the EV adds the dollar scale and the price discipline.

**Market regime is the biggest unknown.** The miss-path multiple was 0.74 (2022), 0.89, 1.07,
1.11 (2025); the debut path 1.63 to 2.39. The sheet reports the all-years fit and a 2024-25 fit.

**Top-100 on the miss path.** Non-debuters with P(new Top-100) >= 0.05 returned about 1.08x vs 0.99x.

**At the fair price only the near-certain call-ups clear.** Paying the SCP price, 18 raw / 20
PSA 10 cards have EV > 0.10 in the all-years market (49 / 52 in the 2024-25 market): the P(debut
in 2027) 0.65-0.8 group (Tibbs, King, Anderson, Bodine, Ayers, Roccaforte, Burke, Cijntje ...) at
+15 to +22% (all years) or +30 to +40% (2024-25). Every raw name with EV > 0.10 *at today's ask*
gets there because the ask is under the SCP price, so verify the listing before trusting it (some
are signed base cards, slabs or variants; asks outside 0.5-2.5x the reference are flagged).

**Which debuts pop more (`debut_pop.py`).** Target: each card's own monthly jump at debut,
log(max(debut month, month after) / month before), 362 debuts 2021-26. Lasso, leave one debut year
out: rho 0.24 against the observed pop, R2 ~0.04; top predicted quintile popped about 1.6x vs 1.2x
for the bottom. Per standard deviation (log): cheaper card -0.13, older at debut -0.09, older card
-0.07, September call-up -0.04, skipped AAA -0.03, IFA -0.02, Top-25 -0.02; outfielder +0.03,
higher PSA 10 premium +0.03, bigger team market (log metro population) positive. Team one-hots
(8-17 debuts per team) are noise; a hand-picked "big market" flag looked stronger (rho 0.31) but
was partly chosen after seeing the teams, so the model uses metro population (rho 0.21 -> 0.24).
Big-metro debuts popped +0.11 log more on average. Draft round, position beyond OF, rank and the
performance percentiles add nothing at this n. PSA 10 pops are unpredictable (rho 0.04).

**But a player-level debut multiple does not improve the EV.** Scored against exits that carry each
debuter's own observed pop, the band EV at > 0.10 realised +24% raw and the player-level EV +10%;
within the band EV > 0.10 basket, the *low* pop-index half did better (+35% vs +14%, debut rate
75% vs 65%). What predicts a bigger pop (young, cheap, unheralded) predicts a lower chance of
debuting, and the hit rate dominates. The pop index stays on the sheet as information only.

**Robustness round (2026-10-24): most of the "predictable pop" was measurement noise.** The monthly
target divides by the month-before price, so a stale or noisy low price both inflates the pop and
shows up as "cheap card / high PSA-10 ratio" in the features. Averaging raw, PSA 9 and PSA 10 pops
per player (they agree at rho 0.44-0.57) and measuring against the median of the three pre-debut
months, the lasso's rho falls from 0.24 to ~0.15 once price features are taken from six months out.

**More data: every priced parallel.** An in-browser scrape (SportsCardsPro, 719 player-sets, 6,057
parallel pages, no errors) pulled refractors and numbered colours of the first autos of every
2021-26 debut. Parallels trade thinly, so their last-30-sales window reaches back over the debut:
day-level (days 0-3 vs 15-90 days before) pops for 209 players, up from 64 base cards
(`raw/variant_daypop.csv`, `debut_pop.day_pops`). Base and parallel reads of the same player agree
only at rho 0.28, so single cards are noisy and averaging matters.

On that day-level target: hype already in the price shrinks the pop (ever-Top-100 1.38x vs 1.78x,
higher FV and price lower), September call-ups pop less (1.47x vs 1.60x). Team market and
ahead-of-ETA effects do **not** replicate (small-market debuts 1.93x vs big 1.81x), so they were
noise in the monthly data. Within a price band the hype effect mostly disappears; the robust
expectation is the price-band multiplier, now on ~200 players:

| Base-auto price | Players | Mean days 0-3 pop (95% CI) | Shrunk, used in EV | Old (56 base cards) |
|---|---|---|---|---|
| < $10 | 73 | 2.57x (2.14-3.42) | 2.48x | 2.14x |
| $10-15 | 20 | 1.62x (1.33-2.06) | 1.70x | 1.83x |
| $15-30 | 28 | 1.41x (1.28-1.55) | 1.53x | 1.30x |
| $30+ | 83 | 1.48x (1.37-1.61) | 1.52x | 1.40x |

`ev.py` now uses this table for raw (PSA 10 unchanged). Leave one year out: EV > 0.10 realises +17%
(CI +7 to +28), EV > 0.20 predicted +33% realised +26% (CI +13 to +40). The backtest's debut exits
are built from the same table, so part of that is mechanical; the gain is that the pop now rests on
~200 players with a tight interval in the $30+ band where the buy-list PSA 10 / AAA cards sit.

**Long run is a star lottery.** Mean current price of a base first auto by settled outcome, 2013-16
cards: raw never $3.9, debuted $4.5, established $8.9, star $44; PSA 10 $30 / $32 / $56 / $175.
Weighting by the production 6-year probabilities, the median buy-list card is worth ~0.35x its raw
price (0.44x PSA 10) a decade out; 6% (13%) are worth more than today's price. Holding past the
debut pop is negative EV; the edge is the pop.

**Data fixes made here.** `load._match_players` breaks name ties by who was playing in the minors
around the card year and falls back to last name + first initial (Chris / Christopher): matched
first autos 92% -> 97% (2,939 / 3,020). SCP's first-auto sets include 2025 Sapphire variations
(CPS) and Mega Box autos (BMA); `ev.latest_scp` prefers the base card and wants >= 2 sales in 180
days, else uses the eBay ask / 1.14 (median ask-to-SCP ratio, IQR 0.9-1.4). The 2025 cohort now
exits at October 2026 (the Oct 5 scrape prices every card that month). 130point sold history keeps
only ~200 sales per search (about a month for a hot player), so it cannot extend the pop sample.

## 12. Pop prediction ceiling, exit execution, and the established / All-Star trade (2026-10-25)

**How much better can pop prediction get?** Two reads of the same player's debut pop (base auto vs
its parallels) agree at rho 0.28. Even a model that knew each player's true pop exactly could reach
only about rho 0.5 against a single card; the November model is at 0.15-0.17. The rest is call-up
timing and context that is not knowable at the buy.

**Execution is the bigger lever.** Individual sales vs each card's median 15-90 days before
debut (median across cards):

| Window | Raw | PSA 10 |
|---|---|---|
| days -3 to -1 (call-up news) | 1.58x | 1.31x |
| days 0-1 | 1.60x | 1.41x |
| days 2-3 | 1.55x | 1.56x |
| days 4-7 | 1.39x | 1.40x |
| days 15-30 | 1.43x | 1.08x |

Within days 0-3 the best sale is 1.25x the median sale and the 75th percentile 1.10x (43 cards
with >= 3 sales there). Raw peaks as the call-up is reported; PSA 10 peaks on days 2-3.

**Ceiling events: event study** (`study_ceiling.py`; labels refreshed, because career_outcomes stops
at 2021: All-Star rosters from the Stats API via `fetch_allstars.py`, established recomputed from
MLB season_stats through 2026 with outcomes.py's rule).

| Event, raw first auto vs market | n | Return (95% CI) | Cards up |
|---|---|---|---|
| First All-Star season, Jan -> Jun | 65 | +39% (+24 to +56) | 71% |
| First All-Star, Jun -> Aug (selection) | 77 | +17% (+8 to +28) | 55% |
| First All-Star, Aug -> next Jun | 59 | +17% (+2 to +34) | 41% |
| First All-Star, calendar year | 45 | +88% (+57 to +127) | 69% |
| Established, the year it happens | 219 | +13% (+3 to +23) | 43% |
| Established, the year after | 193 | +9% (0 to +17) | 38% |

PSA 10 is similar (+67% over the All-Star year). Establishment barely moves cards.

**Buying ahead of it does not pay.**
- Prospect-stage P(star) / P(established), from the held-out v3 validation players, held one to two
  years: -27% to -54% net, every band negative. The probabilities are 3-6 years out; the
  post-debut fade wins.
- A first-All-Star model for young MLB players (`study_allstar.py`; debuted within 5 years, never
  an All-Star, each January): AUC 0.87, AP 0.09-0.39 by season against a 3% base rate; the top 5
  per season hit 32%. Bought in January and sold in August, the top 5 nets -21% (CI -33 to -10),
  the top 10 -19%. Even perfect hindsight nets only +32% on the hits, while misses lose -45%.
  The market already prices it: the card's own price ranks the All-Stars at AUC 0.82, and the
  model's picks cost a median $97. Break-even would need ~58% hits.

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

- §3c with the production model's leak-free historical predictions (the day 0-3 exit is done, §3c).
- A larger day-level pop sample (eBay sold history for 2022-24 debuts) to firm up the §3c multipliers.
- ~~A short-horizon call-up signal~~ (done, §7: does not pay).
- Numbered parallels, where fixed costs are a smaller share (needs another scrape).
- PSA population data for the grading question.
