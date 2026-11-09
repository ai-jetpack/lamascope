# LlamaScope — Product Requirements Document

| | |
|---|---|
| Status | Draft for review |
| Date | 2026-11-08 |
| Owner | Meir Abergel |
| Companion | [DESIGN.md](DESIGN.md) |

---

## 1. Problem

Tuning local LLM inference on a consumer Windows PC (GPU + CPU split, MoE models) means
juggling many tools that each show one slice: VTune, Nsight Systems, nvidia-smi, GPU-Z,
HWiNFO, Intel MLC, Task Manager, llama-bench CSVs and dozens of hand-edited command lines.

Typical problems hit while tuning a large MoE model split across a consumer GPU and CPU, each of
which took hours to days to find and that one integrated view would expose in minutes:

| Problem | Time to find | What would have caught it |
|---|---|---|
| GPU link trained at PCIe x1 (another card shared the lanes) | hours | link width vs card max, in red |
| "Bus Interface Load 100%" read as saturation; real traffic ~1% of link | long detour | real MB/s and % of capacity |
| Most CPU time spinning in OpenMP barriers, a minority in expert math | VTune setup + quoting fights | live CPU time per module |
| ~10% slower overnight (BIOS PCIe split change, lost env vars) | a morning | snapshot diff vs best run |
| llama-server ~20% slower than llama-cli with the same model | guesswork | thread placement + command diff |
| `--n-cpu-moe` floor found by crash-and-retry | many runs | VRAM planner |
| Unknown whether CPU phase was at the hardware limit | MLC + manual math | bandwidth ceiling gauge |

## 2. Product vision

**One screen that shows the hardware, where the model runs, and what every part of the
machine is doing while tokens are generated, and that says plainly what is wrong.**

## 3. Target user

- Primary: a technically strong individual running local models on their own Windows PC,
  tuning for single-user throughput (one request at a time).
- Comfortable with command lines and hardware concepts. Not necessarily a GPU/CPU
  profiling specialist, and does not want to become one.
- v1 platform: Windows 11, NVIDIA GPU, Intel CPU (hybrid P/E cores), llama.cpp.

## 4. Goals and non-goals

### Goals
- G1. Verify within 1 minute of launch that the hardware performs at spec.
- G2. Visualize how the model is split across GPU and CPU, and the dataflow between them.
- G3. Show live CPU and GPU behavior during inference, including **work vs waiting**.
- G4. Name the current bottleneck in plain language, with evidence.
- G5. Make every run reproducible and comparable: what changed, and what it cost.
- G6. Ship as one exe with no prerequisites besides the NVIDIA driver; install the optional
  external tools from inside the app.

### Non-goals (v1)
- AMD GPUs, AMD CPUs, Intel GPUs, Linux, macOS.
- Inference engines other than llama.cpp (vLLM, ik_llama.cpp, Ollama: later).
- Replacing VTune / Nsight for instruction-level or kernel-level analysis. We launch them.
- Multi-user serving metrics (batching, concurrency).
- Model quality / accuracy evaluation.
- A kernel driver of our own.

## 5. User stories

| ID | As a user I want to… | So that… |
|---|---|---|
| US1 | see all hardware fundamentals with spec-vs-actual coloring | I catch degraded links, lost XMP, wrong power plan before tuning |
| US2 | see which blocks of each layer run on GPU vs CPU and the PCIe crossings | I understand the dataflow and the cost of `--n-cpu-moe` |
| US3 | watch per-core load and where llama's threads run | I can pick `-t`, affinity and poll settings with evidence |
| US4 | see CPU time split into expert math vs OpenMP spin vs CUDA | I know whether I'm compute-, memory- or sync-bound |
| US5 | see GPU utilization, clocks, throttle reasons and real PCIe MB/s | I stop misreading "bus load %" |
| US6 | see a per-token timeline and time budget | I see where the 22 ms per token goes |
| US7 | get a diagnosis list | I don't need to interpret 40 numbers myself |
| US8 | launch llama-cli/server/bench from the app with a validated command | I stop fighting typos, duplicates and PowerShell quoting |
| US9 | compare any two runs, with changes highlighted | I find what changed after a reboot or BIOS edit |
| US10 | download and A/B several llama.cpp builds | I detect regressions between builds |
| US11 | install HWiNFO, MLC, Nsight, VTune, the build toolchain with one button each | I don't hunt for download pages |
| US12 | run without admin, with admin-only views greyed out | I can use it anywhere, and elevate only when needed |
| US13 | predict VRAM use and tok/s for a different `--n-cpu-moe` | I skip crash-and-retry |

## 6. Functional requirements

Priority: **P0** = MVP, **P1** = v1.0, **P2** = later.

### 6.1 Hardware panel
| ID | Requirement | Pri |
|---|---|---|
| HW1 | CPU model, P/E core counts, HT, cache sizes, ISA flags | P0 |
| HW2 | Per-core effective clock vs max turbo | P0 |
| HW3 | RAM total, DIMMs, channels, configured vs rated speed | P0 |
| HW4 | Built-in memory bandwidth test; show measured vs theoretical | P0 |
| HW5 | GPU name, VRAM, driver, CUDA version, clocks, power, power limit, temp | P0 |
| HW6 | GPU throttle reasons decoded by name | P0 |
| HW7 | PCIe gen/width current vs max | P0 |
| HW8 | Built-in host↔GPU bandwidth test | P0 |
| HW9 | Resizable BAR state | P1 |
| HW10 | Power plan, PCIe ASPM, Lock Pages privilege, HW GPU scheduling | P0 |
| HW11 | CPU temperature, package power, VID via HWiNFO (if running) | P1 |
| HW12 | Green/amber/red/grey status per item with tooltip explaining the rule | P0 |

### 6.2 Model topology panel
| ID | Requirement | Pri |
|---|---|---|
| MT1 | Read GGUF metadata and tensor list (no full model load) | P0 |
| MT2 | Layer column: mixer type, router, shared expert, routed experts, sizes | P0 |
| MT3 | Color each block by device from launch parameters (`-ngl`, `--n-cpu-moe`, `-ot`) | P0 |
| MT4 | PCIe crossing arrows with bytes/token | P1 |
| MT5 | VRAM split: weights / KV / compute buffer / free | P1 |
| MT6 | What-if slider for `--n-cpu-moe` → predicted VRAM and tok/s | P1 |
| MT7 | Live highlight of the executing layer (instrumented mode) | P2 |

### 6.3 CPU runtime panel
| ID | Requirement | Pri |
|---|---|---|
| CP1 | Per-logical-core utilization heatmap laid out by topology | P0 |
| CP2 | Thread-to-core placement for the llama process (admin) | P1 |
| CP3 | CPU time by module: ggml-cpu / libomp / CUDA runtime / kernel / other (admin) | P1 |
| CP4 | Per-thread work / spin / off-CPU split (admin) | P1 |
| CP5 | IPC, LLC misses, stalls via hardware counters (admin) | P2 |
| CP6 | Context switches, preemptions by other processes, P↔E migrations (admin) | P1 |

### 6.4 GPU runtime panel
| ID | Requirement | Pri |
|---|---|---|
| GP1 | Live graphs: util, memory util, clocks, power, temperature | P0 |
| GP2 | PCIe RX/TX MB/s and % of link capacity | P0 |
| GP3 | VRAM used; dedicated vs shared GPU memory; sysmem-fallback alert | P0 |
| GP4 | GPU busy/idle intervals (admin, ETW) — pending feasibility spike | P1 |

### 6.5 Timeline, budget, diagnosis
| ID | Requirement | Pri |
|---|---|---|
| TL1 | Swimlanes for last 1–3 tokens: GPU, PCIe, each CPU thread, main thread | P1 |
| TL2 | Stacked per-token budget bar: GPU / copies / CPU / wait / other | P1 |
| TL3 | Speed ceilings: CPU phase vs DRAM BW, GPU phase vs VRAM BW | P1 |
| DG1 | Rule-based diagnosis list with severity, evidence, suggested action | P0 (core rules), P1 (all) |

### 6.6 Launcher, runs, config
| ID | Requirement | Pri |
|---|---|---|
| LR1 | Choose build, model, tool (cli/server/bench), edit parameters, launch, stop | P0 |
| LR2 | Parse prompt and generation tok/s from output; llama-server `/metrics` | P0 |
| LR3 | Save every run: command, build, env, model hash, hardware snapshot, results | P0 |
| LR4 | Diff any two runs; "compare to best" | P0 |
| LR5 | Repeat N times, report mean ± sd | P1 |
| LR6 | Validate flags against the selected build's `--help`; flag duplicates/typos | P1 |
| LR7 | Visual affinity picker producing `--cpu-range` / `--cpu-mask` | P1 |
| LR8 | Attach to an already running llama process (no launch) | P1 |
| LR9 | Parameter sweeps (grid over chosen flags), result heatmap | P2 |

### 6.7 Other views
| ID | Requirement | Pri |
|---|---|---|
| OV1 | Correlation: tok/s, clocks, power, temps, link, spin on one time axis | P1 |
| OV2 | Memory: mapped vs resident model, locked pages, hard faults/s, standby | P1 |
| OV3 | Context/KV: fill level, KV size, tok/s vs context position | P1 |
| OV4 | Interference: DPC/ISR per core, other processes on llama's cores (admin) | P2 |

### 6.8 Tools & Setup
| ID | Requirement | Pri |
|---|---|---|
| TS1 | Row per external component: status, version, path, what it unlocks | P0 |
| TS2 | **[Install]** downloads and spawns the installer as a child process, shows progress and result | P0 |
| TS3 | Integrity check: SHA-256 from GitHub digest / winget manifest / vendor page | P0 |
| TS4 | Gated vendor downloads (license or login): open page, detect file in Downloads, verify, install | P1 |
| TS5 | llama.cpp build manager: list releases, download build + matching cudart, keep several side by side | P0 |
| TS6 | Post-install configuration help (e.g. enable HWiNFO Shared Memory Support) | P1 |
| TS7 | Required vs optional clearly marked; only NVIDIA driver + one llama.cpp build are required | P0 |

### 6.8b Inference mode (runtime de-bloat + P-core isolation)
Temporary, reversible changes that remove background contention while a model runs. Inspired by
ChrisTitusTech/winutil and Raphire/Win11Debloat, but session-scoped: nothing is permanently disabled.

| ID | Requirement | Pri |
|---|---|---|
| IM1 | One toggle "Enter inference mode" / "Restore"; per-item checkboxes with current state, effect, risk | P1 |
| IM2 | Restore journal written **before** each change; restore on toggle-off, app exit, and at next start after a crash | P1 |
| IM3 | P-core isolation: move every accessible process except llama to E-cores, re-apply to new processes every 2 s, restore original masks | P1 |
| IM4 | llama process: P-core affinity (or picker mask), High priority, EcoQoS/power throttling off, timer-resolution throttling off | P1 |
| IM5 | Pause services for the session (stop, restore prior state): SysMain, Windows Update (UsoSvc, wuauserv, BITS, DoSvc), WSearch, DiagTrack, dmwappush, WerSvc | P1 |
| IM6 | Disable scheduled tasks for the session: defrag, Defender scheduled scan, UpdateOrchestrator scans, CEIP, DiskDiagnostic, WinSAT, WindowsAI/Recall | P1 |
| IM7 | Defender: exclusions for the llama build folder (process) and model folder (path), session-only or persistent; separate explicit consent; never disables real-time protection | P1 |
| IM8 | Power: High/Ultimate performance plan, core parking off, PCIe ASPM off for the session | P1 |
| IM9 | Background Store apps off for the session (GlobalUserDisabled) | P2 |
| IM10 | Report what could not be changed (protected processes such as MsMpEng, kernel threads, interrupts) | P1 |
| IM11 | Optional A/B: llama-bench before/after inference mode, gain shown per run | P2 |
| IM12 | Advanced, permanent, reboot required: GPU/NVMe interrupt affinity to E-cores (registry), with undo | P2 |

### 6.9 Privileges
| ID | Requirement | Pri |
|---|---|---|
| PV1 | Runs unelevated; admin-only features greyed out with reason | P0 |
| PV2 | "Restart as administrator" button | P0 |
| PV3 | Grant Lock Pages in Memory privilege (admin, requires sign-out) | P1 |

## 7. Non-functional requirements

| ID | Requirement |
|---|---|
| NF1 | Single `.exe`, x64, no installer, no runtime prerequisites beyond the NVIDIA driver; target < 5 MB |
| NF2 | Monitoring overhead: < 1% of one core unelevated; < 3% with ETW sampling on |
| NF3 | Must not change inference speed by more than 1% (measured A/B with app closed) |
| NF4 | UI at 60 fps; collectors never block the UI |
| NF5 | Start to hardware panel populated (excluding bandwidth tests) < 2 s |
| NF6 | Works with no network (installs/downloads disabled) |
| NF7 | No telemetry. Network only for user-initiated downloads |
| NF8 | Every downloaded file verified before execution |
| NF9 | Data stored under `%LOCALAPPDATA%\LlamaScope`; portable mode if a `portable` file sits next to the exe |
| NF10 | Readable at 1920×1080; usable at 1366×768 with scrolling panels |

## 8. Success metrics

- Reproduce each failure in §1 on a test machine; the tool flags each one within
  60 s of starting a run (7/7).
- Time from launch to first diagnosis < 2 minutes on a fresh machine (incl. installs).
- Monitoring overhead within NF2/NF3.
- Target users stop opening VTune/GPU-Z/nvidia-smi for routine tuning.

## 9. Releases

| Release | Content |
|---|---|
| **MVP (P0)** | Hardware panel + tests, Tools & Setup (winget, llama.cpp builds), launcher, CPU heatmap, GPU/PCIe graphs, GGUF topology (static placement), core diagnosis rules, runs + diff |
| **v1.0 (P1)** | ETW: thread placement, module time, spin %, timeline, budget, ceilings; what-if planner; config check; assisted installs; correlation, memory, context views |
| **v2 (P2)** | PMC counters, interference, sweeps, instrumented mode (live layer highlight), PDB symbols |

## 10. Assumptions and dependencies

- NVIDIA driver installed (provides `nvml.dll`, `nvcuda.dll`).
- winget present (Windows 11 default); otherwise winget-sourced installs fall back to the vendor page.
- llama.cpp keeps publishing `bNNNNN` Windows CUDA zips on GitHub (now marked prerelease).
- HWiNFO free edition shared memory is acceptable for CPU temperature and package power.

## 11. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| CUDA activity not visible in ETW DxgKrnl | No GPU lane in timeline | Spike in M1; fallback: estimate from NVML util at 10 ms + CPU-side waits |
| ETW PMC limited on hybrid CPUs | No IPC/stall data | P2 only; degrade gracefully |
| Monitoring perturbs the spin-barrier workload | Misleading results | Overhead budget NF2/NF3; collector rates configurable; "low-overhead mode" |
| Vendor download URLs change / gated | Install buttons break | URLs resolved at runtime (GitHub API, winget); assisted-install fallback |
| llama.cpp log format changes | tok/s parsing breaks | Parse defensively, prefer llama-server `/metrics` and llama-bench JSON |
| GGUF metadata differs per architecture | Wrong topology | Per-architecture mapping table; generic fallback view |

## 12. Open questions for review

1. Name: "LlamaScope" OK?
2. MVP scope: is ETW (thread placement, spin %) needed in the MVP, or is v1.0 fine?
3. Should the app keep llama.cpp builds in its own folder, or also reuse existing `llama-b*` folders (scan both)?
4. Attach-to-running-process (LR8) priority: many users launch from a terminal. Raise to P0?
5. Sweeps (LR9): P2, or overnight auto-tuning earlier?
