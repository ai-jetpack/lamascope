# LlamaScope — Design Document

| | |
|---|---|
| Status | Draft for review |
| Date | 2026-11-08 |
| Implements | [PRD.md](PRD.md) (requirement IDs referenced as HW1, CP3, …) |

---

## 1. Overview

A single native x64 executable. Collector threads sample the OS, NVIDIA driver, ETW and the
llama.cpp process into an in-memory time-series store. The UI thread renders panels from
that store every frame with Dear ImGui. A rule engine evaluates the store once per second
and produces diagnoses. One-shot probes (hardware inventory, bandwidth tests, GGUF parsing)
run on worker threads on demand.

```
                    ┌──────────────────────────── LlamaScope.exe ───────────────────────────┐
 NVIDIA driver ───► │ NvmlCollector ──┐                                                      │
 (nvml, nvcuda)     │                 │        ┌────────────┐      ┌─────────────────────┐   │
 PDH / CIM ───────► │ PdhCollector ───┼──────► │ TimeSeries │ ───► │ UI thread (ImGui)   │   │
 ETW kernel ──────► │ EtwSession ─────┤        │   Store    │      │ panels, 60 fps      │   │
 ETW DxgKrnl ─────► │                 │        └─────┬──────┘      └─────────────────────┘   │
 HWiNFO SHM ──────► │ HwinfoReader ───┤              │                       ▲             │
 llama process ───► │ LlamaMonitor ───┘              ▼                       │             │
 (stdout, HTTP)     │                         DiagnosisEngine (1 Hz) ────────┘             │
                    │ one-shot: HardwareProbe, MemBwTest, PcieTest, GgufReader,            │
                    │           HelpParser, Installer, BuildManager                        │
                    │ persistence: RunStore (JSON files)                                    │
                    └──────────────────────────────────────────────────────────────────────┘
```

## 2. Technology choices

| Concern | Choice | Reason |
|---|---|---|
| Language | C++20, MSVC (VS 2026), `/MT` static CRT | single exe, direct Win32/ETW access |
| UI | Dear ImGui (docking branch) + ImPlot, D3D11 backend | dense live dashboards, compiled in |
| Build | CMake + Ninja, vendored deps in `third_party/` | reproducible, no package manager at runtime |
| JSON | nlohmann/json (header-only) | runs, settings, GitHub API |
| Zip | miniz | extract llama.cpp builds, MLC |
| HTTP | WinHTTP | built into Windows; TLS via SChannel |
| Hash | BCrypt (CNG) SHA-256 | built into Windows |
| NVIDIA | `LoadLibrary("nvml.dll")`, `LoadLibrary("nvcuda.dll")` | driver-provided; no CUDA toolkit needed at runtime |
| ETW | raw `StartTrace` / `OpenTrace` / `ProcessTrace` + TDH | no external library |
| Tests | doctest (header-only) | unit tests for parsers, rules, math |

Expected exe size: ~3 MB (ImGui ~1 MB, fonts embedded ~0.5 MB).

## 3. Source layout

```
llamascope/
  CMakeLists.txt
  docs/            PRD.md, DESIGN.md
  third_party/     imgui/, implot/, miniz/, json/, doctest/
  src/
    main.cpp                  window, D3D11, frame loop
    app/                      App state, docking layout, settings, theme
    core/                     TimeSeries, RingBuffer, Clock, Log, Status (green/amber/red/grey)
    platform/                 Elevation, Privileges, Process, Registry, WinHttp, Hash, Zip, Shell
    probes/                   HardwareProbe, CpuTopology, MemInventory, PowerConfig,
                              MemBwTest, PcieTest
    collectors/               NvmlCollector, PdhCollector, EtwKernelSession, EtwDxgSession,
                              HwinfoReader, LlamaMonitor
    model/                    GgufReader, ArchTable, PlacementModel, Ceilings
    llama/                    BuildManager, Launcher, CommandLine, HelpParser, OutputParser,
                              ServerMetrics
    diagnosis/                Rule, Rules*.cpp, DiagnosisEngine
    runs/                     RunRecord, RunStore, Snapshot, Diff
    setup/                    Component, Detectors, Installer, ComponentCatalog
    ui/                       panels: Hardware, Topology, CpuRuntime, GpuRuntime, Timeline,
                              Diagnosis, Correlation, Memory, Context, Interference, Runs,
                              ConfigCheck, ToolsSetup, Launcher, Log; widgets: Heatmap,
                              StatusCell, Swimlane, BudgetBar, CoreMap
  tests/
```

## 4. Threading and data model

### 4.1 Threads
| Thread | Rate | Work |
|---|---|---|
| UI | vsync (60 Hz) | render; reads store snapshots; never waits on locks held > µs |
| NvmlCollector | 100 ms (PCIe throughput: NVML's ~20 ms window, read each tick) | GPU metrics |
| PdhCollector | 250 ms | per-core util, clock %, GPU adapter memory, process counters |
| EtwKernelSession | streaming (consumer thread) | CSwitch, SampledProfile, ImageLoad, DPC/ISR, PMC |
| EtwDxgSession | streaming | DxgKrnl GPU queue packets |
| HwinfoReader | 500 ms | shared-memory sensors |
| LlamaMonitor | stdout pipe + 500 ms HTTP | tok/s, logs, `/metrics`, `/slots` |
| Aggregator | 100 ms | ETW events → per-thread/per-module/per-token aggregates |
| Diagnosis | 1 s | rules over store |
| Worker pool (2) | on demand | probes, tests, downloads, GGUF parse |

### 4.2 TimeSeries store
- Each metric is a `Series { id, unit, RingBuffer<(t, value)> }`, capacity 10 minutes at its
  native rate; older data downsampled 10:1 to 1 hour.
- Writers: single producer per series, lock-free SPSC ring. Readers copy the visible window.
- Event streams (timeline intervals) in a separate ring of `Interval { lane, t0, t1, kind }`,
  last 5 s only.
- `Snapshot` = struct of current scalar values + static inventory, used by diagnosis and runs.

### 4.3 Status model
`enum Status { Ok, Warn, Bad, Unavailable }` + `reason` string. Every displayed metric has an
optional `StatusRule` that maps (value, spec) → Status. Colors: Ok green, Warn amber,
Bad red, Unavailable grey. Hover shows the rule (e.g. "configured 4800 < rated 5600 MT/s").

## 5. Probes (one-shot)

### 5.1 CPU topology (HW1)
- `GetLogicalProcessorInformationEx(RelationAll)`: cores, SMT siblings, `EfficiencyClass`
  (higher = P-core on Alder Lake), cache levels/sizes, groups.
- `__cpuid`: brand string, AVX2, AVX-VNNI, AVX-512F (absent on hybrid Alder/Raptor Lake client parts).
- Max turbo: registry `HARDWARE\DESCRIPTION\System\CentralProcessor\N\~MHz` gives base;
  max from PDH `% Processor Performance` peak observed during the bandwidth test, plus a
  small table of known SKUs for spec reference.

### 5.2 Memory inventory (HW3)
- CIM `Win32_PhysicalMemory`: Capacity, Speed (rated), ConfiguredClockSpeed, DeviceLocator,
  BankLabel → DIMM count, channel guess (one DIMM per channel unless locators show 2DPC).
- Theoretical BW = channels × MT/s × 8 B.

### 5.3 Memory bandwidth test (HW4)
- Allocate 2 GB (large pages if the privilege is held, else 4 KB), touch to commit.
- One thread per P-core physical core, pinned with `SetThreadAffinityMask`; AVX2
  non-temporal reads (`_mm256_stream_load_si256` / plain loads with unrolled accumulators).
- 5 passes, report the best and median GB/s. Read-only test is comparable to MLC "all reads".
- Duration ~3 s. Warns if a llama process is running (the result would be wrong).

### 5.4 PCIe test (HW8)
- `nvcuda.dll` driver API: `cuInit`, `cuCtxCreate`, `cuMemAllocHost` (pinned 256 MB),
  `cuMemAlloc`, timed `cuMemcpyHtoD` / `cuMemcpyDtoH` × 10 with `cuEventRecord`.
- Theoretical: per-lane GB/s by gen (Gen3 0.985, Gen4 1.969, Gen5 3.938) × width.
- Bad if measured < 50% of theoretical for the *negotiated* link, or link < card max.

### 5.5 Power and system config (HW10)
- Power plan: `PowerGetActiveScheme` + `PowerReadFriendlyName`.
- ASPM: `PowerReadACValueIndex(SUB_PCIEXPRESS, ASPM)`.
- Lock Pages privilege: `LsaEnumerateAccountRights` for the user SID → `SeLockMemoryPrivilege`
  (and token check for the current session).
- HAGS: `HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers\HwSchMode`.
- Resizable BAR: NVML `nvmlDeviceGetBAR1MemoryInfo` total > 256 MB.

## 6. Collectors

### 6.1 NVML (HW5–7, GP1–3)
Functions resolved at runtime; missing ones mark their metric Unavailable.
`nvmlDeviceGetUtilizationRates`, `GetClockInfo` (graphics, mem), `GetMaxClockInfo`,
`GetPowerUsage`, `GetEnforcedPowerLimit`, `GetTemperature`, `GetCurrentClocksEventReasons`
(decoded with the NVML bitmask names), `GetCurrPcieLinkGeneration/Width`,
`GetMaxPcieLinkGeneration/Width`, `GetPcieThroughput(TX/RX)`, `GetMemoryInfo_v2`,
`GetComputeRunningProcesses` (per-process VRAM), `SystemGetDriverVersion`,
`SystemGetCudaDriverVersion`.

### 6.2 PDH (CP1, HW2, GP3)
- `\Processor Information(*)\% Processor Utility` and `% Processor Performance` per logical CPU.
- `\GPU Adapter Memory(*)\Dedicated Usage`, `Shared Usage`.
- `\GPU Process Memory(pid_*)\Shared Usage` for the llama process.
- `\Process(llama-*)\Page Faults/sec`, `Working Set`, `Thread Count`.
- `\Memory\Standby Cache *`, `\Memory\Pages Input/sec` (hard faults).

### 6.3 ETW kernel session (CP2–CP6, TL1, OV4) — admin
- Session: `SystemTraceProvider` via `EVENT_TRACE_SYSTEM_LOGGER_MODE` (private system logger,
  Windows 8+), so it does not conflict with the global NT Kernel Logger.
- Enabled groups: `PROC_THREAD`, `IMAGE_LOAD`, `CSWITCH`, `PROFILE`, `DPC`, `INTERRUPT`.
- Profile interval: `TraceSetInformation(TraceSampledProfileIntervalInfo)` = 0.25 ms
  (default 1 ms; configurable). Overhead measured in M4 against NF2.
- **Module attribution**: maintain per-process module map from ImageLoad (base, size, name).
  Each SampledProfile event carries the instruction pointer and thread id → module bucket:
  `ggml-cpu*` = work, `libomp*` / `vcomp*` = spin/barrier, `ggml-cuda`/`nvcuda`/`cudart` = CUDA,
  kernel addresses = kernel, else other. No symbols needed.
- **Spin vs work** inside libomp is an approximation (spin and barrier code both live there).
  Refined in v2 with export-table symbol lookup of `__kmpc_barrier` etc.
- **Placement**: CSwitch gives (new thread, CPU, timestamp) → per-thread last CPU, on-CPU
  intervals, preemptions (old thread in llama, new thread elsewhere = preempted by other).
- **PMC (P2)**: `TraceSetInformation(TraceProfileSourceConfigInfo)` with sources from
  `TraceProfileSourceListInfo` (TotalCycles, InstructionRetired, LLCMisses…), sampled on
  CSwitch → per-thread IPC.
- Filtering: events from processes other than llama are reduced to per-core "other busy"
  counts in the consumer, to keep memory flat.

### 6.4 ETW DxgKrnl session (GP4, TL1) — admin, **feasibility spike**
- Provider `Microsoft-Windows-DxgKrnl`, keywords for DmaPacket / QueuePacket start/stop.
- Filter by llama PID's GPU contexts → GPU busy intervals on the compute/copy engines.
- If CUDA on WDDM does not emit usable packets: fallback lane = NVML util sampled at 10 ms
  rendered as a density band, plus CPU-side "waiting in nvcuda" intervals from 6.3.

### 6.5 HWiNFO reader (HW11)
- Open `Global\HWiNFO_SENS_SM2` shared memory, parse the sensor/reading tables by label:
  "CPU Package Power", "CPU Package" temperature, "Core VIDs", per-core clocks, DRAM temps.
- Missing → grey + hint "Enable Shared Memory Support in HWiNFO settings" (link to Tools & Setup).

### 6.6 LlamaMonitor (LR2, OV3)
- Launched processes: stdout/stderr through pipes; attach mode (LR8): process found by name
  or picked from a list, no stdout → rely on `/metrics` (server) or ETW only.
- `OutputParser`: regexes for the `[ Prompt: X t/s | Generation: Y t/s ]` line, `llama_perf_*`
  eval lines, `system_info`, load-time buffer sizes (`CUDA0 model buffer size = … MiB`,
  `CPU_Mapped`, `KV self size`, `compute buffer size`), `n_ctx`.
- llama-bench: launched with `-o json`, parsed directly.
- llama-server: poll `/metrics` (Prometheus text: `llamacpp:tokens_predicted_total`,
  `llamacpp:prompt_tokens_seconds`, `kv_cache_usage_ratio`, …) and `/slots`.

## 7. Model topology (MT1–MT6)

### 7.1 GGUF reader
- Memory-map only the header: magic, version, KV metadata, tensor infos (name, dims, type,
  offset). Never reads weights.
- Byte size per tensor from ggml type block sizes (table for Q4_K, Q5_K, Q6_K, Q8_0, IQ*,
  F16, BF16, F32, MXFP4…).

### 7.2 Architecture table
- Maps `general.architecture` → how to group tensors into blocks:
  - `blk.N.attn_*` → attention (or `ssm_*` / DeltaNet tensors → linear-attention mixer)
  - `blk.N.ffn_gate_inp` → router; `ffn_*_shexp` → shared expert; `ffn_*_exps` → routed experts
  - expert count / used count keys: `<arch>.expert_count`, `<arch>.expert_used_count`
  - layer pattern (e.g. full-attention interval) from `<arch>.full_attention_interval` or by
    tensor presence per layer.
- Unknown arch → generic grouping by tensor-name prefix.

### 7.3 Placement model
- Inputs: tensors, `-ngl`, `--n-cpu-moe N` (experts of layers 0..N-1 on CPU, as llama.cpp
  implements it), `-ot` regex overrides (applied in order), `--no-kv-offload`, `-c`, KV types.
- Output: device per tensor → per block; VRAM estimate = GPU weights + KV
  (layers_with_kv × 2 × n_ctx × n_kv_heads × head_dim × bytes) + compute buffer
  (from the load log if available, else a calibrated estimate).
- When a load log exists, actual buffer sizes override estimates and the error is shown.

### 7.4 Ceilings (TL3, MT6)
- CPU bytes/token = Σ CPU layers × (k_active × bytes per expert) + other CPU tensors read per token.
- CPU floor = CPU bytes/token ÷ measured DRAM BW; GPU floor = GPU active bytes ÷ VRAM BW
  (VRAM BW from a small per-GPU spec table; unknown → hidden).
- Predicted tok/s for the what-if slider = 1 ÷ (CPU floor·α + GPU floor·β + overhead), with
  α, β, overhead fitted from the user's own runs (least squares over RunStore); before any
  runs, α = β = 1 and overhead = 0 (labelled "theoretical").

## 8. Timeline and budget (TL1, TL2)

- Lanes: GPU (6.4), PCIe (NVML throughput as a band; per-copy intervals only in instrumented
  mode), CPU threads (on-CPU intervals from CSwitch, colored by module of samples falling in
  them: work / spin / CUDA), main thread.
- Token boundaries: from the stdout token stream timing (cli/server streaming) or from the
  periodicity of GPU-busy intervals; marked as estimated when inferred.
- Budget = average over last N tokens of: GPU-busy-only time, CPU-work time (any worker in
  ggml-cpu), copy/sync time (main thread in nvcuda with GPU idle), straggler wait (some
  workers spinning while others work), other.

## 9. Diagnosis engine (DG1)

```cpp
struct Finding { Severity sev; std::string id, title, evidence, action; };
struct Rule { std::string id; std::function<std::optional<Finding>(const Snapshot&, const Store&)> eval; };
```
- Evaluated at 1 Hz; findings debounced (must hold 3 s to appear, 10 s to clear).
- MVP rules: PCIe link below max · RAM configured < rated · measured DRAM BW < 85% theoretical ·
  power plan not performance · ASPM on · Lock Pages missing with mlock flag · shared GPU memory
  > 256 MB for llama · GPU throttling · hard faults during generation · llama on E-cores
  (needs ETW) · context > 90% full.
- v1 rules: spin share > 50% · CPU phase ≥ 90% of DRAM ceiling · preemptions > N/s on llama
  cores · tok/s dropped > 5% vs best comparable run (with diff summary) · ignored env vars ·
  duplicate/unknown flags.

## 10. Launcher and command handling (LR1, LR6, LR7)

- Command model: ordered list of `(flag, value)`; canonicalized using aliases learned from
  `--help` (e.g. `-fa` ≡ `--flash-attn`, `-ncmoe` ≡ `--n-cpu-moe`).
- `HelpParser` runs `<tool> --help` once per build, caches flags, aliases, value types.
- Validation: unknown flag, duplicate canonical flag, CPU indices ≥ logical count, tool-specific
  invalid flags, value type errors.
- LlamaScope pins itself to the E-cores (process affinity) at startup. Affinity is inherited, so every
  launched llama process gets its own explicit mask (default: all cores, or what the affinity picker
  produced) via `SetProcessAffinityMask` on the suspended child before `ResumeThread`.
- Process launch: `CreateProcessW` with explicit env block (inherited env + run overrides),
  working dir = build folder, pipes for output, Job Object so children die with the app.
- Env vars recorded per run. Known-ineffective env table (e.g. `KMP_AFFINITY` with LLVM libomp).
- Affinity picker: clicking cores on the CoreMap builds `--cpu-range` (contiguous) or
  `--cpu-mask` (hex) for `-t`/`-tb` separately.

## 11. Runs and diff (LR3–LR5)

- `RunRecord` JSON per run in `runs/YYYYMMDD-HHMMSS-<tool>.json`:
  `{ command, canonical_flags, build {tag, cuda, path}, env, model {path, size, sha256_head},
     snapshot {...all §5 values...}, results {pp_tps[], tg_tps[], mean, sd}, budget, findings }`.
- Model hash: SHA-256 of the first 64 MB + size (fast, stable).
- Diff: flatten both records to key paths; show changed keys grouped (Command / Build / Env /
  Hardware / Results). "Best comparable" = same model + same tool, max tg_tps.

## 12. Tools & Setup (TS1–TS7)

### 12.1 Component catalog

Resolved at runtime where possible; URLs below are current as of 2026-11-08.

| Component | Need | Detect | Source | Install action |
|---|---|---|---|---|
| NVIDIA driver | Required | NVML `SystemGetDriverVersion` | https://www.nvidia.com/en-us/drivers/ | Open page (no stable direct URL) |
| llama.cpp CUDA build | Required (≥1) | scan `builds\`, user folders, winget path | GitHub API `GET https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=30`, keep tags `^b\d+$` (**prereleases**; `/releases/latest` returns `v0.9.0` without Windows assets) | Download `llama-bNNNNN-bin-win-cuda-<cu>-x64.zip` + `cudart-llama-bin-win-cuda-<cu>-x64.zip`, verify `digest` (sha256), extract both to `builds\bNNNNN-cuda<cu>\` |
| HWiNFO | Recommended | uninstall registry key / `HWiNFO64.exe` | winget `REALiX.HWiNFO` (8.54 → `https://www.hwinfo.com/files/hwi64_854.exe`, Inno) | `winget install --id REALiX.HWiNFO -e --silent --accept-package-agreements --accept-source-agreements`; then guide: enable Shared Memory Support |
| Intel MLC | Optional | `mlc.exe` in `tools\mlc\` | Intel Download Center 736633 (v3.13, license click-through, SHA256 `A8537E8FF3FAD626D75A383FABC224CCC4CC98A0111C9989F7FB26B639F12019`) | Assisted: open page → watch Downloads for `mlc_v3.13.tgz` → verify hash → extract `Windows\mlc.exe` |
| Intel VTune | Optional | `%ProgramFiles(x86)%\Intel\oneAPI\vtune\latest\bin64\vtune.exe` | `https://registrationcenter-download.intel.com/akdlm/IRC_NAS/0aa9b155-4329-4996-b242-c17acf8a7dd3/intel-vtune-2026.6.0.141_offline.exe` | Direct download → run elevated: `-s -a --silent --eula accept` (oneAPI installer flags, verify) |
| Nsight Systems | Optional | `%ProgramFiles%\NVIDIA Corporation\Nsight Systems *\target-windows-x64\nsys.exe` | `https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2026_6/nsightsystems-2026.6.1.218-39266636.msi` (path under `/secure/` — likely needs NVIDIA developer login) | Try direct; on 401/403/HTML → assisted (open page, watch Downloads) → `msiexec /i <file> /passive /norestart` |
| Git | Optional (v2 build) | `git.exe` on PATH | winget `Git.Git` | winget install |
| CMake | Optional (v2 build) | `cmake.exe` on PATH | winget `Kitware.CMake` | winget install |
| VS Build Tools 2026 | Optional (v2 build) | `vswhere -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64` | winget `Microsoft.VisualStudio.BuildTools` (18.11.3) | `winget install … --override "--passive --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"` |
| CUDA Toolkit | Optional (v2 build) | `nvcc.exe`, `CUDA_PATH` | winget `Nvidia.CUDA` (13.5 → `https://developer.download.nvidia.com/compute/cuda/13.5.0/local_installers/cuda_13.5.0_windows_x86_64.exe`) | winget install |

CUDA flavor for llama.cpp builds: pick the highest `<cu>` (currently 13.5 or 12.4) whose
major.minor ≤ `nvmlSystemGetCudaDriverVersion`.

### 12.2 Install pipeline

```
[Install] ─► resolve (GitHub API / winget / catalog URL)
          ─► download (WinHTTP, resumable, progress %, to %LOCALAPPDATA%\LlamaScope\downloads)
          ─► verify SHA-256 (GitHub digest | winget manifest hash | catalog hash) — mismatch = abort, delete
          ─► spawn installer as child process:
               winget    : CreateProcess("winget install …")  (winget handles UAC per package)
               exe / msi : ShellExecuteEx(verb="runas") with silent flags
               zip / tgz : extract in-process (no elevation)
          ─► wait on process handle (UI shows "Installing…", Cancel = TerminateProcess for zip only)
          ─► exit code → result (winget/msi codes mapped to messages; 3010 = reboot required)
          ─► re-run detector → row turns green, unlocked features enabled
```
- **Assisted mode** (gated vendors): open the vendor page with `ShellExecute`, then watch
  `%USERPROFILE%\Downloads` with `ReadDirectoryChangesW` for the expected file pattern; when it
  appears and is no longer growing, verify and continue the pipeline.
- No downloaded file is executed without a hash check; for direct vendor URLs without a
  published hash (VTune, Nsight), check the Authenticode signature (`WinVerifyTrust` + signer
  = Intel Corporation / NVIDIA Corporation) instead.
- If winget is missing, winget-sourced rows offer "Open vendor page".

## 12b. Inference mode (IM1–IM12)

Session-scoped tuning. Every action implements `Probe() -> state`, `Apply()`, `Restore(saved)`, and is
journaled to `%LOCALAPPDATA%\LlamaScope\inference-mode.json` (state before change) before `Apply()`.
On startup, a leftover journal means a crash: offer "Restore previous system state" first.

| Action | Mechanism | Restore | Notes |
|---|---|---|---|
| Others → E-cores | enumerate processes (Toolhelp), `OpenProcess(PROCESS_SET_INFORMATION \| PROCESS_QUERY_LIMITED_INFORMATION)`, `SetProcessAffinityMask(E-mask)`; rescan every 2 s for new pids | original mask per (pid, create time) | Skips: llama, System/Idle, protected (PPL: MsMpEng, csrss…), failures listed. Kernel threads and DPC/ISR still use P-cores. |
| llama.cpp priority | `SetPriorityClass(HIGH)`, `SetProcessInformation(ProcessPowerThrottling, ControlMask=EXECUTION_SPEED|IGNORE_TIMER_RESOLUTION, StateMask=0)`. **Never changes the CPU mask of a running llama process**: that overrides llama.cpp's per-thread placement (`--cpu-range`/`--cpu-strict`) and OpenMP has already sized its pool. Warns if it started without the P-cores. | priority back to normal | Children inherit their parent's CPU mask, so terminals and Explorer (launchers) keep or get all cores; processes started during the mode that inherited the E-core mask get all cores back on restore. |
| Services | SCM: `QueryServiceStatusEx`, `ControlService(STOP)`; restore = start only if it was running | prior running state (start type untouched) | SysMain, UsoSvc, wuauserv, BITS, DoSvc, WSearch, DiagTrack, dmwappush, WerSvc. WaaSMedicSvc is protected (skip). |
| Scheduled tasks | Task Scheduler COM `ITaskService` → `IRegisteredTask::put_Enabled(FALSE)` | re-enable those we disabled | Defrag\ScheduledDefrag, Windows Defender\Scheduled Scan, UpdateOrchestrator\Schedule Scan(+Static), CEIP\Consolidator+UsbCeip, DiskDiagnostic\DataCollector, Maintenance\WinSAT, WindowsAI\* |
| Defender exclusions | WMI `root\Microsoft\Windows\Defender` `MSFT_MpPreference.Add(ExclusionPath/ExclusionProcess)` | `Remove(...)` of exactly what we added | Requires Tamper Protection off. Warns about the security trade-off; detects third-party AV. |
| Power | `PowerSetActiveScheme` (High/Ultimate), `PowerWriteACValueIndex` CPMINCORES=100 (core parking off), ASPM=0 | prior scheme + values | |
| Background apps | HKCU `...\BackgroundAccessApplications\GlobalUserDisabled=1` | prior value | P2 |
| Interrupt affinity (advanced) | `HKLM\SYSTEM\CurrentControlSet\Enum\<dev>\Device Parameters\Interrupt Management\Affinity Policy` (DevicePolicy=4, AssignmentSetOverride=E-mask) | delete keys | permanent, reboot; P2 |

UI: panel "Inference mode" with a checklist (state now → state in mode, risk badge), "Enter" / "Restore"
buttons, a log of applied/failed actions, and the list of processes that could not be moved.
Diagnosis integration: "N processes still running on P-cores during inference" (from ETW CSwitch).

## 13. Privileges (PV1–PV3)

- At start: `CheckTokenMembership(Administrators)` + elevation type.
- Admin-only features declare `requiresAdmin`; their panels render greyed with
  "Requires administrator — [Restart as admin]". Restart = `ShellExecuteEx(runas, self, --restore-layout)`.
- Lock Pages: `LsaAddAccountRights(user SID, SeLockMemoryPrivilege)` (admin), then tell the
  user to sign out/in; detector re-checks.

## 14. UI

### 14.1 Default layout (docking)

```
┌ Toolbar: [Launch ▶][Stop][Attach…][Snapshot][Compare…]  build ▾  model ▾  tool ▾  admin ✔/✖ ┐
├──────────────────┬─────────────────────────────────┬─────────────────────────────────────────┤
│ Hardware         │ Model topology & dataflow       │ CPU runtime                             │
│                  │                                 ├─────────────────────────────────────────┤
│                  │                                 │ GPU runtime                             │
├──────────────────┴─────────────────────────────────┴───────────────┬─────────────────────────┤
│ Per-token timeline + budget bar                                    │ Diagnosis               │
├────────────────────────────────────────────────────────────────────┴─────────────────────────┤
│ tabs: Launcher · Correlation · Memory · Context/KV · Interference · Runs & Diff ·            │
│       Config check · Tools & Setup · Log                                                     │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 14.2 Panel sketches

**Hardware** — grouped tables (CPU / Memory / GPU / PCIe / System), each row:
`label | measured | spec | status dot`. Buttons: [Run memory test] [Run PCIe test].

**Model topology** — vertical stack of 48 rows; each row = small boxes
`[mixer][norm][router][shared][experts ×512, k=10]` colored GPU (teal) / CPU (orange);
row height ∝ bytes; arrows on the right edge where consecutive blocks change device, labelled
bytes/token. Side bar: VRAM stacked bar, totals, what-if slider.

**CPU runtime** — CoreMap: 8 P-core tiles each split in 2 HT halves, 8 E-core tiles; fill =
utilization; dots = llama threads (tooltip: tid, work/spin %). Below: module stacked bar,
per-thread bars, counters row.

**GPU runtime** — 4 sparklines (util, clocks, power, temp), PCIe RX/TX graph with link-capacity
line, VRAM dedicated/shared bars, throttle reasons chips.

**Timeline** — ImPlot with custom swimlane rendering, x = ms, last 1–3 tokens, pause/zoom.
Budget bar beneath.

**Diagnosis** — list sorted by severity; click a finding → highlights the panel/metric that
produced it.

**Tools & Setup** — table: component | status | version | unlocks | [Install]/[Open]/[Configure]
with progress bar inline.

### 14.3 Visual language
Dark theme; status colors fixed (green #3fb950, amber #d29922, red #f85149, grey #6e7681);
device colors GPU = teal, CPU = orange, PCIe = violet; monospace digits for live numbers.

## 15. Persistence

`%LOCALAPPDATA%\LlamaScope\` (or exe folder in portable mode):
`settings.json`, `imgui.ini` (layout), `runs\*.json`, `builds\`, `tools\`, `downloads\`, `logs\`.

## 16. Error handling

- Every collector is optional: init failure → its metrics Unavailable with reason; app continues.
- NVML/nvcuda absent → GPU panels grey, Tools & Setup shows driver row red.
- ETW session name collision (stale session from crash) → stop and recreate `LlamaScope-Kernel`.
- Crash safety: ETW sessions are stopped in an atexit + console ctrl handler + on next start.

## 17. Testing

- Unit: GGUF parser (fixture headers, incl. a hybrid attention/linear-layer MoE header),
  placement model, help parser (captured `--help` outputs per build), output parser
  (captured logs), diff, rules (synthetic snapshots), install pipeline with a local file server.
- Integration (manual, test machine): reproduce each PRD §1 failure:
  force the GPU link to x1 (lane-sharing device), lower the RAM speed, Balanced plan, remove Lock Pages, launch
  without `--cpu-strict`, etc. → expect the matching finding within 60 s.
- Overhead: A/B tok/s with app closed vs open (unelevated / ETW on), 5 runs each (NF2, NF3).

## 18. Milestones

| M | Content | Exit criterion |
|---|---|---|
| M0 | Spikes: DxgKrnl visibility of CUDA work; ETW sampling overhead at 0.25 ms; HWiNFO SHM read | go/no-go per feature recorded here |
| M1 | Skeleton, D3D11/ImGui docking, Hardware panel + probes + tests | Hardware panel correct on a test machine |
| M5 (moved up) | GGUF + topology + placement + ceilings + what-if | VRAM estimate within 5% of load log |
| M2b | Inference mode (IM1–IM10) | enter/restore round-trips cleanly; others on E-cores verified |
| M2 | Tools & Setup, BuildManager, installer pipeline | install HWiNFO + 2 llama.cpp builds from the app |
| M3 | Launcher, output parsing, RunStore + diff, NVML/PDH runtime panels | run, see tok/s + GPU/PCIe live, diff two runs |
| M4 | ETW kernel: CoreMap dots, module time, spin %, admin gating | spin % within ±10 pts of VTune on same run |
| M5 | GGUF + topology + placement + ceilings + what-if | VRAM estimate within 5% of load log |
| M6 | Timeline, budget, diagnosis rules | all 7 PRD §1 failures flagged |
| M7 | Correlation, Memory, Context, Interference, Config check, assisted installs | PRD P1 complete |

## 18a. M0 results (hybrid Intel desktop CPU + NVIDIA RTX GPU, llama.cpp CUDA build)

| Spike | Result | Decision |
|---|---|---|
| CUDA work visible in DxgKrnl ETW | **GO.** With keyword `0x1` only: `QueuePacket` Start/Stop (logged in the submitting process, carries `hContext`, `SubmitSequence`) and `DmaPacket` Start/Stop/Info (logged from the VidSch worker, carries the same `hContext`, `uliSubmissionId`). llama-bench used 3 main contexts; ~150 DMA packets per generated token; ~3.5 MB/s of events. | GPU lane = DmaPacket Start→Stop for contexts learned from QueuePacket events whose `EVENT_HEADER.ProcessId` is llama. Do **not** filter DmaPacket by PID. |
| ETW overhead (llama-bench tg128 ×3, sd ≈ 13%) | vs baseline: CSwitch+Profile 1 ms −0.9%; 0.25 ms −1.6%; DxgKrnl only +0.4% (noise); CSwitch+Profile 1 ms + DxgKrnl −2.8%. All within one sd. | Default: CSwitch + 1 ms profile + DxgKrnl 0x1, with a "low-overhead" toggle (CSwitch + DxgKrnl only). Re-measure with ≥10 reps in M4 against NF3. |
| HWiNFO shared memory | **GO.** `Global\HWiNFO_SENS_SM2` v2.1, element size 460 (label @12, unit @268, value @284 work). Provides per-thread effective clocks, package/IA power, PL1/PL2, per-core thermal-throttle and power-limit flags, package temps. No DRAM bandwidth. Requires `SensorsSM=1`; first launch may prompt (motherboard EC warning). | Prefer HWiNFO effective clocks over PDH when present; feed throttle/power-limit flags to diagnosis. Check the free-version 12 h SM limit. |

Ideas adopted from gputaskman (github.com/shanselman/gputaskman):
- Per-process VRAM: use `\GPU Process Memory(*)\Local Usage` / `Non Local Usage` (charged, sums to adapter total); `Dedicated Usage` double-counts shared surfaces (fallback only).
- `\GPU Engine(*)\Utilization Percentage` per pid and engine type (3D / Compute / Copy): no-admin GPU activity per process, incl. CUDA on WDDM.
- llama.cpp load log regex `(\S+)\s+(model|KV|compute|output|RS)\s+buffer size\s*=\s*([\d.]+)\s*MiB` for the exact per-device VRAM split (RS = recurrent state).
- GGUF tensor byte sizes from offset deltas (no per-type table needed).
- Attach mode: mapped GGUF via process memory maps (`VirtualQueryEx` + `GetMappedFileNameW`), command line from the PEB, listening ports → llama-server `/props`, `/slots`, `/metrics`.

## 18b. Live DRAM traffic

- **Source**: memory-controller free-running counters `UNC_MC0/1_RDCAS_COUNT_FREERUN` and `UNC_MC0/1_WRCAS_COUNT_FREERUN` (one count per 64 B transfer; Intel event DB `vtune/config/sampling/alderlake_imc_db.txt`). Windows exposes no such counter and HWiNFO's shared memory has no DRAM-bandwidth reading on Alder Lake.
- **Access**: VTune's `emon.exe` (bin64) via its `sepdrv5` driver, run as a hidden child: `emon -C "<4 events>" -t0.5 -l0`. Each sample's length comes from its TSC column (TSC calibrated against QPC), so pipe bursts do not distort timing. The child is in a kill-on-close job object and inherits LlamaScope's E-core affinity. Needs elevation.
- **Validation**: during the 2 GiB memory test, emon's total matched the test's result plus the idle background traffic (a few GB/s from the desktop and integrated GPU).
- **UI/rules**: CPU runtime panel "Memory traffic" chart (read/write + measured/theoretical reference lines, 10 s average readout, follows Pause); Diagnosis `mem.bw` fires when read+write ≥ 90% of the measured maximum (80% of theoretical if not measured), suppressed during LlamaScope's own memory test.
- **Fallback**: without VTune or admin the section is greyed with the reason. Future: Intel PCM driver as an alternative source.

## 18c. LlamaScope's own cost on inference

llama-bench (-n 128 -r 4, mean of reps 2-4) with LlamaScope in different configurations, relative to runs
without LlamaScope before and after each batch:

| Config | Generation speed |
|---|---|
| full | −4.9% |
| all collectors off (UI drawing only) | −5.0% |
| full, window minimized | −0.2% (noise) |
| full, throttled drawing (fix) | −0.3% (noise) |

Collectors (NVML 100 ms, PDH 250 ms, process scan 2 s, emon 0.5 s) cost nothing measurable; redrawing a high-resolution window at the monitor rate cost ~5% (compositor work on the P-cores plus integrated-GPU traffic on the shared ring/memory bus). Rule: draw at the monitor rate only for 1.5 s after user input; otherwise 4 fps, 2 fps while llama.cpp is computing. All smoothing is time-based (DeltaTime), never per frame. Diagnostic switches `--no-nvml --no-imc --no-pdh-all --no-pdh-gpu --no-procmon` remain for re-measuring.

## 18d. Hardware-accelerated GPU scheduling

On a test machine (MoE model split between CPU and GPU), turning HAGS on (HwSchMode 1 -> 2, reboot) made generation ~15% faster with identical llama.cpp flags. About half of each token's time was not explained by CPU or GPU weight reads, and DRAM traffic was only about a third of the measured maximum. With HAGS off, Windows queues each GPU command on the CPU, adding latency to each of the ~70 CPU<->GPU hand-offs per token. The Hardware panel now checks it (off = Warn, with the setting path and the reboot requirement).

## 19. Open design questions

1. Profile interval default: 1 ms (low overhead) or 0.25 ms (better timeline)? Proposed: 1 ms,
   switchable.
2. Keep the app's own `builds\` folder only, or also index external folders (existing
   `llama-b*` download folders)? Proposed: both, external read-only.
3. Instrumented mode (v2): injected DLL hooking ggml graph compute vs requiring a custom build
   with an eval callback. Proposed: custom build first (safer, no injection).
