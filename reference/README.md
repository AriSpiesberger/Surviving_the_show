# reference

Checked-in inputs. Paths are defined in `prospects/config.py`.

| Path | Contents |
|---|---|
| `fangraphs_board/` | FanGraphs Board and Trouble-With-The-Curve grades, with the raw scrapes they are built from |
| `baseballcube/` | Historical org rankings and the player crosswalk |
| `pre2005/` | Ranking history and debut dates for the 1988-2004 draft classes |
| `transactions/` | Transaction and organization features, team maps |
| `awards/` | MiLB honours features |
| `showcase/` | Futures Game and winter-league features |
| `prices/` | Buy-list price reference |
| `prices/daily/` | Daily eBay snapshots written by `deploy/daily_prices.py` |
| `milb_baselines.json` | League medians by level |
| `mex_league_teams.json` | Mexican League team list |

Past FanGraphs boards can no longer be retrieved, so the raw HTML is source data rather than a cache.

Raw API caches stay out of git because they can be fetched again: `transactions/raw/`, `awards/raw/`,
`showcase/futures/`, `showcase/winter/`, `pre2005/seasons/`, `pre2005/drafts/` and
`pre2005/statscrew/raw/`.
