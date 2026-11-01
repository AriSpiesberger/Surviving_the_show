# tools

Hand-run scripts. Run them from the repo root, for example `python tools/leak_audit.py`.

`leak_audit.py` checks features and promoted bundles for label leaks and stale models. The weekly
retrain stops if it fails.

## scouting/

| Script | Purpose |
|---|---|
| `scrape_fangraphs_board.py` | Scrape FanGraphs Board season snapshots |
| `fetch_fangraphs_board_browser.py` | Fetch the board through Chrome when Cloudflare blocks plain HTTP |
| `build_fg_crosswalk.py` | Map FanGraphs IDs to player IDs |
| `build_scouting_grades.py` | Build the point-in-time grades file |
| `build_scout_text_emb.py` | Embed scouting report text (tested, not adopted) |

## features/

| Script | Purpose |
|---|---|
| `fetch_transactions.py` | Cache MLB and MiLB transactions |
| `build_tx_features.py` | Transaction features |
| `build_award_features.py` | MiLB honours features |
| `build_org_features.py` | Parent-organization development features |
| `build_showcase_features.py` | Futures Game and winter-league features |
| `build_twins.py` | Synthetic twin players (tested, not adopted) |

## history/

| Script | Purpose |
|---|---|
| `build_pre2005_ext.py` | Add pre-2005 draft classes and their MiLB seasons |
| `build_pre2005_context.py` | League-context percentiles for those seasons |
| `fetch_statscrew_gaps.py` | Fill pre-1993 league holes from StatsCrew |

## market/

| Script | Purpose |
|---|---|
| `check_ebay_auth.py` | Test eBay credentials |
| `check_ebay_quota.py` | Show API rate-limit usage |
| `fetch_prices.py` | Query eBay prices for a list of top players |
