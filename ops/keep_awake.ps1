# Hold the machine awake (blocks idle-triggered sleep / Modern Standby) until killed,
# or until the given WINDOWS process id exits.  Usage: powershell -File ops/keep_awake.ps1 [-WatchPid <pid>]
# Note: a bash `$$` is an MSYS pid, not a Windows pid — pass the python process id or nothing.
param([int]$WatchPid = 0)
Add-Type -Namespace KA -Name P -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
$ES_CONTINUOUS = [uint32]"0x80000000"; $ES_SYSTEM_REQUIRED = [uint32]"0x00000001"; $ES_AWAYMODE = [uint32]"0x00000040"
while ($true) {
  [KA.P]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED -bor $ES_AWAYMODE) | Out-Null
  if ($WatchPid -gt 0 -and -not (Get-Process -Id $WatchPid -ErrorAction SilentlyContinue)) { break }
  Start-Sleep -Seconds 30
}
[KA.P]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
