# Strategy search

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
