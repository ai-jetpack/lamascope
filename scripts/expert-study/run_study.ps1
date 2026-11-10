# Runs prompts.jsonl through llamascope-batch in batches (<= 250 prompts, grouped by source, alternating halves A/B),
# one process per batch, each writing a profile with expert counts and the per-token expert trace.
# Restartable: batches with an existing .prof are skipped.
$here = $PSScriptRoot
$bin   = $env:LLAMACPP_PROF_BIN      # folder with the profiler-patched llamascope-batch.exe
$model = $env:LLAMASCOPE_MODEL       # path to the .gguf
if (-not $bin -or -not $model) { throw "set LLAMACPP_PROF_BIN and LLAMASCOPE_MODEL (optional: CUDA_BIN)" }
$out  = Join-Path $here "profiles"
New-Item -ItemType Directory -Force $out | Out-Null
if ($env:CUDA_BIN) { $env:PATH = "$env:CUDA_BIN;" + $env:PATH }

$lines = Get-Content (Join-Path $here "prompts.jsonl") -Encoding UTF8
$sources = $lines | % { ($_ | ConvertFrom-Json).source }
# contiguous runs of the same source, cut into chunks of <= 250
$batches = @(); $start = 0
for ($i = 1; $i -le $sources.Count; $i++) {
    if ($i -eq $sources.Count -or $sources[$i] -ne $sources[$start]) {
        $n = $i - $start; $k = [math]::Ceiling($n / 250); $size = [math]::Ceiling($n / $k)
        for ($s = $start; $s -lt $i; $s += $size) { $batches += ,@($sources[$start], $s, [math]::Min($s + $size, $i)) }
        $start = $i
    }
}
$llama = @('-m', $model, '-ngl', '999', '--n-cpu-moe', '36',
           '-fa', 'on', '-c', '8192', '-b', '4096', '-ub', '1024', '-t', '16', '-tb', '14', '-C', '0xFFFF', '--cpu-strict', '1',
           '--poll', '100', '--load-mode', 'mlock', '-ctk', 'q8_0', '-ctv', 'q8_0', '--parallel', '1', '-n', '128')
$log = Join-Path $here "run.log"
"start $(Get-Date -Format s): $($lines.Count) prompts in $($batches.Count) batches" | Add-Content $log
for ($b = 0; $b -lt $batches.Count; $b++) {
    $src, $s, $e = $batches[$b]
    $half = if ($b % 2 -eq 0) { "A" } else { "B" }
    $name = "{0:D2}-{1}-{2}-{3}" -f $b, $half, $src, $s
    $prof = Join-Path $out "$name.prof"
    if (Test-Path $prof) { continue }
    $env:LLAMASCOPE_PROMPTS = Join-Path $here "prompts.jsonl"
    $env:LLAMASCOPE_RANGE = "${s}:${e}"
    $env:GGML_PROFILE = "block"; $env:GGML_PROFILE_MB = "4"; $env:GGML_PROFILE_FILE = $prof
    $env:GGML_PROFILE_TRACE = "1"; $env:GGML_PROFILE_TRACE_MB = "64"
    "$(Get-Date -Format s) batch $name ($($e - $s) prompts)" | Add-Content $log
    $p = Start-Process (Join-Path $bin "llamascope-batch.exe") -ArgumentList $llama -NoNewWindow -Wait -PassThru `
         -RedirectStandardError (Join-Path $out "$name.err") -RedirectStandardOutput (Join-Path $out "$name.out")
    $last = (Get-Content (Join-Path $out "$name.err") | Select-String "llamascope-batch:" | Select -Last 1).Line
    "$(Get-Date -Format s) batch $name exit $($p.ExitCode): $last" | Add-Content $log
    if ($p.ExitCode -ne 0 -or -not (Test-Path $prof)) {
        "STOPPED: batch $name failed (see $name.err)" | Add-Content $log
        exit 1
    }
}
Remove-Item Env:GGML_PROFILE, Env:GGML_PROFILE_FILE, Env:GGML_PROFILE_TRACE, Env:GGML_PROFILE_TRACE_MB, Env:GGML_PROFILE_MB, Env:LLAMASCOPE_PROMPTS, Env:LLAMASCOPE_RANGE -EA SilentlyContinue
"done $(Get-Date -Format s)" | Add-Content $log
