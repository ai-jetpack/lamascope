# LlamaScope

One-screen diagnostics for local LLM inference (llama.cpp) on Windows with an NVIDIA GPU and an Intel CPU.
See [docs/PRD.md](docs/PRD.md) and [docs/DESIGN.md](docs/DESIGN.md).

## Build

Requires Visual Studio 2022/2026 with the C++ x64 tools, CMake and Ninja.

```
build.cmd            # Release -> build\Release\LlamaScope.exe
build.cmd Debug
```

Output is a single statically linked exe (~1.5 MB) that depends only on DLLs shipped with Windows.
NVIDIA data comes from `nvml.dll` / `nvcuda.dll`, loaded at runtime from the driver.

## Run

- `LlamaScope.exe`: UI. Run as administrator for Inference mode and live memory traffic (and later ETW features).
- `LlamaScope.exe --selftest`: headless probes + memory and PCIe bandwidth tests; exit code 0 on success.
- `--run-tests`: press "Run tests" 5 s after startup. `--im-roundtrip [--memtest]`, `--imc-probe`: headless checks.
- `--no-nvml --no-imc --no-pdh-all --no-pdh-gpu --no-procmon`: turn off individual collectors (to measure their cost).

Settings, layout and the log live in `%LOCALAPPDATA%\LlamaScope\` (`llamascope.log` is rewritten each start).

Live memory traffic needs Intel VTune installed (its `emon.exe` and sampling driver read the memory controller).

## Status

| Milestone | State |
|---|---|
| M0 Feasibility spikes (DxgKrnl, ETW overhead, HWiNFO) | done |
| M1 Skeleton, Hardware panel ("Run tests", checks vs spec, Platform details), CPU/GPU runtime panels | done |
| M5 Model topology: GGUF reader, placement, pipeline diagram, VRAM budget, `--n-cpu-moe` what-if | done |
| M2b Inference mode: session-scoped de-bloat, background programs on E-cores, journaled restore | done |
| Live DRAM traffic (memory-controller counters), VRAM holders, diagnosis rules (HAGS, VRAM room, bandwidth) | done |
| M2 Tools & Setup, llama.cpp build manager, installers | next |
| M3 Launcher, run history + diff | planned |
| M4 ETW thread placement, CPU time by module, spin % | planned |

## Third-party (vendored in `third_party/`)

Dear ImGui 1.93.0-docking, ImPlot 1.1, nlohmann/json 3.12.0, miniz 3.1.2 (all MIT).
