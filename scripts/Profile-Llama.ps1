# Runs the profiler-patched llama-cli (or llama-bench) with GGML_PROFILE on, writes the profile where LlamaScope's
# "Load latest" finds it (%LOCALAPPDATA%\LlamaScope\profiles\<tool>-<time>.prof), then opens it in LlamaScope.
#
#   $env:LLAMACPP_PROF_BIN = "<llama.cpp-prof>\build-prof\bin"   (required)
#   $env:CUDA_BIN          = "<CUDA>\v13.3\bin"                  (if the CUDA runtime DLLs are not on PATH)
#   .\scripts\Profile-Llama.ps1 -m C:\models\x.gguf -ngl 999 --n-cpu-moe 36 ...     (your usual llama-cli flags)
#
# Options (anywhere among the flags): --profile-tool=llama-cli|llama-bench  --profile-mode=block|op  --profile-bin=<dir>
#                                     --profile-view=0 (do not open LlamaScope afterwards)
#                                     --profile-pause-on-error=1 (wait for Enter before closing when no profile was written)
# LlamaScope.exe is taken from $env:LLAMASCOPE_EXE, else ..\build\Release\LlamaScope.exe relative to this script.
#
# No param() block on purpose: llama flags such as -m or -t would otherwise bind to script parameters.
# End llama-cli with /exit (or one Ctrl+C, then /exit). A double Ctrl+C exits without freeing the GPU backend,
# so GPU block times and expert counts are lost (CPU data is still written).
$tool = "llama-cli"; $mode = "block"; $view = $true; $pause = $false
$bin = $env:LLAMACPP_PROF_BIN
$cudaBin = $env:CUDA_BIN
$llamaArgs = @()
foreach ($a in $args) {
    if ($a -like "--profile-tool=*") { $tool = $a.Substring(15) }
    elseif ($a -like "--profile-mode=*") { $mode = $a.Substring(15) }
    elseif ($a -like "--profile-bin=*") { $bin = $a.Substring(14) }
    elseif ($a -like "--profile-view=*") { $view = $a.Substring(15) -ne "0" }
    elseif ($a -like "--profile-pause-on-error=*") { $pause = $a.Substring(25) -ne "0" }
    else { $llamaArgs += $a }
}
function Fail($msg) { Write-Host $msg -ForegroundColor Red; if ($pause) { Read-Host "Press Enter to close" | Out-Null }; exit 1 }
if (-not $bin) { Fail "set LLAMACPP_PROF_BIN (or pass --profile-bin=<dir>)" }
if (-not (Test-Path (Join-Path $bin "$tool.exe"))) { Fail "$tool.exe not found in $bin" }
if ($cudaBin) { $env:PATH = "$cudaBin;$env:PATH" }
$dir = Join-Path $env:LOCALAPPDATA "LlamaScope\profiles"
New-Item -ItemType Directory -Force $dir | Out-Null
$file = Join-Path $dir ("{0}-{1}.prof" -f $tool, (Get-Date -Format "yyyyMMdd-HHmmss"))
$env:GGML_PROFILE = $mode
$env:GGML_PROFILE_FILE = $file
try {
    & (Join-Path $bin "$tool.exe") @llamaArgs
} finally {
    Remove-Item Env:GGML_PROFILE, Env:GGML_PROFILE_FILE -ErrorAction SilentlyContinue
}
if (-not (Test-Path $file)) { Fail "No profile written (did the process exit normally?)" }
Write-Host "Profile: $file"
if ($view) {
    $exe = if ($env:LLAMASCOPE_EXE) { $env:LLAMASCOPE_EXE } else { Join-Path $PSScriptRoot "..\build\Release\LlamaScope.exe" }
    if (Test-Path $exe) { Start-Process $exe -ArgumentList "--profile `"$file`"" }
    else { Write-Host "LlamaScope not found at $exe (set LLAMASCOPE_EXE); open the profile with Load latest" }
}
