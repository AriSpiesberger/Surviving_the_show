# ops

| Script | Purpose |
|---|---|
| `run_job.ps1` | Wrapper for each scheduled job: loads `.env`, sets paths, logs output |
| `register_tasks.ps1` | Registers the four Windows Task Scheduler tasks (safe to repeat) |
| `keep_awake.ps1` | Keeps the machine awake during a long job |

The schedule is in the [runbook](../docs/runbook.md).
