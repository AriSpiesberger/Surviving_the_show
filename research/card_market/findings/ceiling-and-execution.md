# Pop ceiling, exit execution and All-Stars

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
