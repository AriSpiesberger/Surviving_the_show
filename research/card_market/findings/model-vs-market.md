# Model probabilities vs the market

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
