# The debut trade

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
