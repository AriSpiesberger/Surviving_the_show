# prospects

The importable package. Install it with `pip install -e .` from the repo root.

| Module | Purpose |
|---|---|
| `config.py` | Every path, database and run namespace |
| `refresh.py` | The whole pipeline as one resumable command |
| [`core/`](core) | Schema, SQLite storage, outcome labels |
| [`data/`](data) | Downloads (`sources/`), repairs (`backfills/`), bulk pull (`pull.py`) |
| [`features/`](features) | The point-in-time feature panel |
| [`model/`](model) | Hazards, joint layer, v3 blends, calibration, training pipelines |
| [`evaluation/`](evaluation) | Held-out metrics and the report generator |
| [`buylist/`](buylist) | Universe filter, scoring and price join |
| [`market/`](market) | eBay client and price aggregation |
| [`deploy/`](deploy) | The scheduled jobs |

See the [architecture page](../docs/model/architecture.md) for how these fit together.
