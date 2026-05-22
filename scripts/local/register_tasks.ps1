#requires -version 5.1
<#
Register the four Surviving_the_show scheduled tasks in Windows Task Scheduler.

Run once, from this repo root:
    powershell -ExecutionPolicy Bypass -File scripts\local\register_tasks.ps1

Re-run to overwrite existing tasks with the same names (safe; idempotent).

Tasks run as the current user, in the current user's context (no admin needed).
Schedule (local time):
    SurvivingShow_daily_data    00:30 daily
    SurvivingShow_weekly_score  05:00 Monday
    SurvivingShow_daily_prices  09:00 daily
    SurvivingShow_daily_digest  10:00 daily
#>

$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Wrapper  = Join-Path $RepoRoot 'scripts\local\run_job.ps1'

if (-not (Test-Path $Wrapper)) {
    throw "Wrapper not found: $Wrapper"
}

$Jobs = @(
    @{ Name = 'SurvivingShow_daily_data';    Job = 'daily_data';    At = '00:30'; Day = 'Daily' }
    @{ Name = 'SurvivingShow_weekly_score';  Job = 'weekly_score';  At = '05:00'; Day = 'Monday' }
    @{ Name = 'SurvivingShow_daily_prices';  Job = 'daily_prices';  At = '09:00'; Day = 'Daily' }
    @{ Name = 'SurvivingShow_daily_digest';  Job = 'daily_digest';  At = '10:00'; Day = 'Daily' }
)

foreach ($j in $Jobs) {
    $Name = $j.Name
    $Action = New-ScheduledTaskAction `
        -Execute 'powershell.exe' `
        -Argument "-ExecutionPolicy Bypass -NoProfile -File `"$Wrapper`" -Job $($j.Job)" `
        -WorkingDirectory $RepoRoot

    if ($j.Day -eq 'Daily') {
        $Trigger = New-ScheduledTaskTrigger -Daily -At $j.At
    } else {
        $Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $j.Day -At $j.At
    }

    # Catch up after missed runs (laptop off / restart) and don't kill long jobs.
    $Settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
        -MultipleInstances IgnoreNew

    $Principal = New-ScheduledTaskPrincipal `
        -UserId $env:USERNAME `
        -LogonType Interactive `
        -RunLevel Limited

    if (Get-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $Name -Confirm:$false
        Write-Output "  - removed existing task $Name"
    }

    Register-ScheduledTask `
        -TaskName $Name `
        -Action $Action `
        -Trigger $Trigger `
        -Settings $Settings `
        -Principal $Principal `
        -Description "Surviving_the_show pipeline: $($j.Job)" | Out-Null

    Write-Output "  + registered $Name  ($($j.Day) $($j.At))"
}

Write-Output ""
Write-Output "Done. List with:  Get-ScheduledTask -TaskName 'SurvivingShow_*'"
Write-Output "Run one now with: Start-ScheduledTask -TaskName 'SurvivingShow_daily_digest'"
