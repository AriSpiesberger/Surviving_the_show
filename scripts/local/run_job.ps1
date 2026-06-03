#requires -version 5.1
<#
Wrapper for the four scheduled jobs. Invoked by Windows Task Scheduler.

Usage:
    powershell -ExecutionPolicy Bypass -File scripts\local\run_job.ps1 -Job daily_data

Behavior:
  - cd's to repo root
  - loads .env (KEY=VALUE lines, ignores blanks and #-comments)
  - overrides Railway-style /data paths with local repo paths
  - runs the right `python -m prospects.deploy.<module>` command
  - tees stdout+stderr to logs\<job>_<yyyy-MM-dd>.log
  - exits with the python process's exit code
#>

param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('daily_data', 'weekly_score', 'daily_prices', 'daily_digest')]
    [string]$Job
)

$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot

$LogDir = Join-Path $RepoRoot 'logs'
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }
$LogPath = Join-Path $LogDir ("{0}_{1}.log" -f $Job, (Get-Date -Format 'yyyy-MM-dd'))

# Load .env if present
$EnvPath = Join-Path $RepoRoot '.env'
if (Test-Path $EnvPath) {
    foreach ($line in Get-Content $EnvPath) {
        $t = $line.Trim()
        if (-not $t -or $t.StartsWith('#') -or ($t -notmatch '=')) { continue }
        $k, $v = $t -split '=', 2
        [Environment]::SetEnvironmentVariable($k.Trim(), $v.Trim(), 'Process')
    }
}

# Local path overrides (modules default to Railway's /data/* otherwise)
$env:DATA_DIR          = $RepoRoot
$env:PROSPECT_DB       = Join-Path $RepoRoot 'prospects.db'
$env:PRICES_DIR        = Join-Path $RepoRoot 'prices'
$env:HOLDINGS_PATH     = Join-Path $RepoRoot 'holdings.csv'
$env:ALERTS_STATE_PATH = Join-Path $RepoRoot 'alerts_state.json'
# Force UTF-8 stdout/stderr so debut-alert subject lines containing
# emojis / accented names don't crash on cp1252 consoles.
$env:PYTHONIOENCODING  = 'utf-8'
# Single-thread BLAS for the heavy panel+train steps in weekly_score
# (one thread per worker process scores better).
$env:OMP_NUM_THREADS      = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS      = '1'

if (-not (Test-Path $env:PRICES_DIR)) {
    New-Item -ItemType Directory -Path $env:PRICES_DIR | Out-Null
}

$Season = (Get-Date).ToUniversalTime().Year
$BuyList = Join-Path $RepoRoot 'results\buy_lists\buy_list_v1.18_FINAL.csv'

switch ($Job) {
    'daily_data'    { $Args = @('-m', 'prospects.deploy.daily_data') }
    'weekly_score'  { $Args = @('-m', 'prospects.deploy.weekly_score', '--season', "$Season") }
    'daily_prices'  { $Args = @('-m', 'prospects.deploy.daily_prices', '--buy-list', $BuyList) }
    'daily_digest'  { $Args = @('-m', 'prospects.deploy.alerts') }
}

$Header = "=== $Job @ $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz') (cwd=$RepoRoot) ==="
Add-Content -Path $LogPath -Value $Header -Encoding utf8
Write-Output $Header

# Run python and tee output to the log. python.exe is resolved from PATH.
# Use cmd.exe redirection so both stdout and stderr land in the log together,
# preserving real-time order.
$ArgString = ($Args | ForEach-Object { if ($_ -match '\s') { "`"$_`"" } else { $_ } }) -join ' '
$Cmd = "python $ArgString 2>&1"
& cmd.exe /c "$Cmd" | Tee-Object -FilePath $LogPath -Append
$ExitCode = $LASTEXITCODE

$Footer = "=== $Job exit=$ExitCode @ $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz') ==="
Add-Content -Path $LogPath -Value $Footer -Encoding utf8
Write-Output $Footer

exit $ExitCode
