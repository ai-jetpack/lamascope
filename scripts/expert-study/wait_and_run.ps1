# Waits until no live llama process holds the model (the user's own session), then runs the study and the analysis.
$here = $PSScriptRoot
$log = Join-Path $here "run.log"
"$(Get-Date -Format s) waiting for running llama processes to exit" | Add-Content $log
while ($true) {
    $live = Get-Process llama-cli, llama-server, llama-bench -ErrorAction SilentlyContinue | Where-Object { $_.Threads.Count -gt 0 }
    if (-not $live) { break }
    Start-Sleep 60
}
Start-Sleep 20   # let the driver release VRAM and the OS release locked pages
"$(Get-Date -Format s) starting study" | Add-Content $log
& (Join-Path $here "run_study.ps1")
if ($LASTEXITCODE -eq 1) { exit 1 }
python -I (Join-Path $here "analyze.py") | Out-File (Join-Path $here "analyze.out") -Encoding utf8
"$(Get-Date -Format s) analysis written" | Add-Content $log
