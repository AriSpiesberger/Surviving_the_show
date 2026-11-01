# Per-player EV

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
