# Top-100 lists and momentum

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
