# Audit location queries and errors in jarvis.log

$lines = Get-Content 'jarvis.log' -ErrorAction SilentlyContinue
Write-Output "Total log lines: $($lines.Count)"

Write-Output "`n=== RECENT TRANSCRIPTS IN JARVIS.LOG ==="
$lines | Where-Object { $_ -match "transcript" -or $_ -match "dispatcher" -or $_ -match "intent" -or $_ -match "router" -or $_ -match "location" } | Select-Object -Last 50

Write-Output "`n=== RECENT ERRORS IN JARVIS.LOG ==="
$lines | Where-Object { $_ -match "ERROR" -or $_ -match "Traceback" -or $_ -match "Exception" -or $_ -match "failed" } | Select-Object -Last 30

