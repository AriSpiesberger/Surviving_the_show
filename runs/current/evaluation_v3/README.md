# Held-out validation: v3

Metrics for `v3.pkl`, scored on players excluded from training. The published sheet uses v3.5, the
same recipe refit on every player, so v3.5 has no held-out score of its own and this is its report
card.

| Event | AP at 3 years | AP at 6 years | AUC at 6 years | Lift at 6 years |
|---|---:|---:|---:|---:|
| `MLB_DEBUT` | 0.574 | 0.603 | 0.932 | 7.1× |
| `TOP_100_PROSPECT` | 0.527 | 0.496 | 0.972 | 45.1× |
| `ESTABLISHED_MLB` | 0.326 | 0.393 | 0.940 | 14.4× |
| `STAR_PLUS_ELITE` | 0.141 | 0.236 | 0.955 | 54.6× |

Weighted AP at 6 years is 0.466, with `MLB_DEBUT` counted twice.

## Files

| File | Contents |
|---|---|
| `headline.json` | The numbers above at the 6-year horizon |
| `per_horizon.csv` | Metrics at each horizon from 1 to 10 years |
| `reliability.csv` | Predicted probability buckets against realized rates |
| `per_bucket_validation.csv` | Metrics by player bucket |
| `per_yip_validation.csv` | Metrics by years in pro |
| `per_level_validation.csv` | Metrics by level |
| `walkforward.csv` | Walk-forward results |
