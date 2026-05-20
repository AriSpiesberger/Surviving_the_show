# Railway deploy runbook

Fully automated after one-time prep. Total time: ~30 minutes.

## What you're shipping

| Job | Cadence (UTC) | Script |
|---|---|---|
| `daily_data` | 04:30 daily | `prospects/deploy/daily_data.py` |
| `weekly_score` | 09:00 Mondays | `prospects/deploy/weekly_score.py` |
| `daily_prices` | 13:00 daily | `prospects/deploy/daily_prices.py` |
| `daily_digest` | 14:00 daily | `prospects/deploy/alerts.py` |

Every cron also runs `scripts/auto_seed.py` first — cheap no-op once the `/data` volume is seeded.

## One-time prep (do once, locally)

### 1. Build the seed bundle

```
python scripts/prepare_seed.py
```

Produces `seed_v1.17.tar.gz` (~150 MB). Bundles:

- `prospects.db`, `prospects_snapshot.db`
- `panels/panel_v1.17.npz`
- `models/event_classifiers_v1.17_prod.pkl`
- `models/debut_lasso_universe_v1.17h.pkl`
- `models/top100_lasso_v1.17h.pkl`
- `models/model_b_outcomes_v1.17h.pkl`
- `models/player_position_from_stats.csv`
- `buy_list_v1.17_FINAL.csv`
- (optional) `holdings.csv`, `alerts_state.json`
- The `scripts_v17/` scoring + buylist code

### 2. Host the seed bundle on GitHub Releases

1. On GitHub, go to your repo → Releases → "Draft a new release".
2. Tag: any string, e.g. `seed-v1.17-2026-05-20`.
3. Upload `seed_v1.17.tar.gz` as a release asset.
4. Publish (can be marked as a pre-release if you don't want it on the front page).
5. Right-click the uploaded asset → copy link. Looks like:
   ```
   https://github.com/<you>/<repo>/releases/download/<tag>/seed_v1.17.tar.gz
   ```
6. Keep this URL — it's the `SEED_URL` env var.

> If your repo is **private**, the release asset URL still requires auth. Either make the release public, or use Backblaze B2 / Cloudflare R2 with a public bucket. Auto-seed just reads `SEED_URL` with no auth headers.

### 3. Push the repo to GitHub if you haven't

Make sure these are present at the repo root:
- `railway.toml`
- `requirements.txt`
- `scripts/prepare_seed.py`, `scripts/auto_seed.py`
- The whole `prospects/` and `scripts_v17/` trees

Make sure these are in `.gitignore` (they're seeded via the bundle, not git):
- `prospects.db`, `prospects_snapshot.db`
- `panels/`, `scratch/`, `prices/`
- `*.npz`, `models/*.pkl`
- `seed_v1.17.tar.gz`

## Railway setup (one-time, ~15 min)

### 4. Create the project

```
npm i -g @railway/cli
railway login
railway init
```

In the Railway dashboard, add a new service from your GitHub repo. Railway uses Nixpacks to build, sees `requirements.txt`, and installs everything automatically.

### 5. Attach the volume

In the Railway dashboard:
- Service → Settings → Volumes → Add Volume
- Mount path: `/data`
- Size: **1 GB** (plenty — current bundle uncompressed is ~200 MB; cron output adds maybe 50 MB/year)

### 6. Set env vars

In the Railway dashboard → Service → Variables:

| Variable | Value |
|---|---|
| `SEED_URL` | the URL from step 2 |
| `DATA_DIR` | `/data` |
| `PROSPECT_DB` | `/data/prospects.db` |
| `PRICES_DIR` | `/data/prices` |
| `HOLDINGS_PATH` | `/data/holdings.csv` |
| `ALERTS_STATE_PATH` | `/data/alerts_state.json` |
| `EBAY_CLIENT_ID` | from your eBay developer account |
| `EBAY_CLIENT_SECRET` | from your eBay developer account |
| `EBAY_ENV` | `production` |
| `SENDGRID_API_KEY` | from SendGrid |
| `ALERT_FROM` | your verified sender email |
| `ALERT_TO` | your inbox |

### 7. Deploy

```
railway up
```

The container builds, the boot command runs `scripts/auto_seed.py`, which downloads + extracts `SEED_URL` into `/data`. After that it idles, waiting for cron entries to fire.

## First-run verification (~30 min)

In the Railway dashboard, each cron job has a "Run now" button. Trigger them manually in order to catch failures early:

1. **`daily_data`** — should finish in ~10 min, populate `/data/prospects.db`. Watch logs for "DONE in ... rows=...".
2. **`weekly_score`** — ~30 min. Watch for "weekly_score season=... OK" and verify `/data/buy_list_v1.17_FINAL.csv` is fresh (touch time = now).
3. **`daily_prices`** — ~3 min. Watch for "DONE in ...s" and check `/data/prices/` for today's snapshots.
4. **`daily_digest`** — ~5s. The email should arrive in your `ALERT_TO` inbox.

If any step fails, the logs are in the Railway dashboard. Fix in the repo, `railway up` again.

## Once it's working

Nothing to do. The cron entries fire on schedule. The first 2x alert email is the real test that the whole pipeline is healthy.

To track new holdings: edit `/data/holdings.csv` via `railway shell`. Schema:

```
card_id,player_id,name,denominator,grade,buy_date,buy_price_usd,ebay_item_id,notes
```

- `denominator=0`  (base auto, not /99 or /499)
- `grade=""` or `raw`  (the 2x trigger ignores graded slabs)

To rotate the model bundle (e.g. v1.18): rerun `prepare_seed.py`, upload the new tarball to a new release, update `SEED_URL` in Railway, and restart the service. The volume's `/data` will get re-seeded only if `auto_seed.py`'s sentinels are missing — to force a re-seed, delete `/data/prospects.db` and `/data/models/event_classifiers_v1.17_prod.pkl` first.

## Cost

Railway hobby plan: $5/mo base. With a 1 GB volume and ~4 small cron jobs/day, total well under $10/mo. The cron runtimes are short (longest is `weekly_score` at ~30 min, once a week).
