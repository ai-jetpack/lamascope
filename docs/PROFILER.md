# In-process profiler for llama.cpp (ggml) — design

| | |
|---|---|
| Status | Implemented as a local spike (llama.cpp branch `llamascope-profiler`, not upstream); cost measured, see §9 |
| Goal | Per-block timing, waits, CPU↔GPU transfer latency and expert usage for every generated token, at a cost low enough to leave on during real runs |
| Target | A patch against llama.cpp/ggml, written to be upstreamable; LlamaScope reads its output and draws it on the model diagram |

## 1. Why not existing mechanisms

| Mechanism | Problem |
|---|---|
| ETW (kernel CSwitch/sampling, DxgKrnl) | Kernel transitions and buffer copies on every event; no tensor/block identity; no expert IDs |
| ggml eval callback (`cb_eval`) | Every observed node forces a device synchronize and breaks CUDA graphs; perturbs exactly what is measured |
| Nsight Systems / VTune | External tool, heavy, no model-level identity, not continuous |

## 2. Principles

- Hot path = plain stores into memory the writing thread owns. No locks, no atomics on the common path, no kernel calls, no allocation.
- User-mode clock: `__rdtsc` (invariant TSC, consistent across cores; converted to seconds against QPC at the end). Fallback `ggml_time_ns`.
- GPU times come from the GPU, never from the CPU side of an asynchronous launch (that would require a sync).
- Off by default; when off, the cost is one predictable branch per node.

## 3. CPU side

Probe in `ggml_graph_compute_thread` (ggml-cpu.c), per node, per thread:

```
t0 = tick();  compute(node);  t1 = tick();  barrier();  t2 = tick();
```

`t1 - t0` = this thread's compute time for the node, `t2 - t1` = time it waited for the other threads (OpenMP spin included).

Entry (32 B, two per cache line):

| Field | Size | Meaning |
|---|---|---|
| `t0`, `t1`, `t2` | 3 × 8 B | ticks |
| `name_hash` | 4 B | hash of the tensor name (`ffn_moe_out-12`); dictionary below |
| `thread` | 1 B | ggml thread index (`ith`), not the OS thread id |
| `kind` | 1 B | op / block summary / sched event |
| `count` | 2 B | nodes folded into this entry (block mode) |

Per-thread buffers are preallocated at init (`GGML_PROFILE_MB` per thread), each in its own cache-line-aligned slot with a thread-local write index. Full buffer → entries are dropped and counted (never allocate on the hot path).

Granularity (`GGML_PROFILE=op|block`):
- `op`: one entry per node per thread.
- `block`: compute and wait times accumulate in registers/thread-local storage and one entry is written when a block-end node is reached (`attn_residual`, `ffn_moe_topk`, `ffn_moe_out`, `ffn_shexp`, `l_out`, end of graph). ~50× fewer entries.

Name dictionary: `hash → name`, filled on first sight (rare insert, only by thread 0); other threads only store the hash.

## 4. Scheduler side (ggml-backend.cpp)

Probe in `ggml_backend_sched_compute_splits`, main thread only: start/end of each split's input copies and of each split's compute submission, plus every `ggml_backend_synchronize` (= CPU waiting for the GPU). Same entry format, `kind = sched`.

## 5. GPU side

CUDA work is asynchronous, so kernel and copy times must come from the GPU:

- **CUPTI activity API** (CUDA toolkit; what Nsight Systems uses): kernel and memcpy records with GPU start/end, delivered in buffers we provide (`cuptiActivityRegisterCallbacks`), drained at the end. Each record carries a correlation ID; the CPU side logs `(correlation id → node name hash)` when ggml-cuda launches a node, so every kernel maps to a tensor and therefore a block.
- GPU timestamps are converted to TSC with an offset measured at start and end (`cuptiGetTimestamp` vs `__rdtsc`), checking drift.
- Fallback if CUPTI is unavailable: a tiny kernel writing `%globaltimer` into a device buffer at block boundaries only (~5 per layer; too costly per op: ~1,500 kernels per token).

## 6. Expert usage

- Layers with experts on the CPU: the chosen IDs (`ffn_moe_topk-N`) are already copied to host memory for the CPU expert matmuls — count them there, free.
- Layers with experts on the GPU: a tiny kernel increments a per-layer, per-expert counter in VRAM (`atomicAdd`), copied back once at the end. No per-token transfer.
- Output: `counts[layer][expert]` for prompt and generation separately.

## 7. Consumer and output

- Default: drained at the end of generation (or at exit) into one binary file: header (tick rate, GPU offset, thread count, drops), entries, CUPTI records, name dictionary, expert counts.
- Optional live drain (bounded memory): single-producer/single-consumer ring per thread; the producer publishes its write index with a release store; a consumer on an E-core reads completed 4 KB chunks well behind the write position (avoids cache-line ping-pong) and polls every 50–100 ms.
- LlamaScope reads the file (later: the live stream) and overlays time per block, PCIe arrow latency, waits, and an expert-usage heatmap on the topology diagram.

## 8. Cost budget

Data rate at ~46 t/s: block mode ≈ 250 KB/token ≈ 12 MB/s; op mode ≈ 1.5 MB/token ≈ 70 MB/s. Even counted twice through DRAM that is 0.03% / 0.2% of measured memory bandwidth (≈83 GB/s). Targets: compiled-in-but-off ≤ 0.2%, block mode ≤ 1%, op mode ≤ 3% generation speed.

## 9. Cost spike (measured with llama-bench `-lm mlock`, A/B)

Generation only (`-p 0 -n 128 -r 5`, mean of reps 2-5), `-ngl 999 -ncmoe 36 -fa on --cpu-strict 1 --poll 100 -ub 1024 -ctk/-ctv q8_0`, i9-12900K class hybrid CPU + 16 GB RTX GPU, Qwen3-Next 80B-A3B Q4.

| Stage | `-t 16 -C 0x5555` | vs off | `-t 16 -C 0xFFFF` | vs off |
|---|---|---|---|---|
| 0 prebuilt b12144 (reference) | 35.66 | | 46.85 | |
| 1 patched, profiler off (two runs) | 38.68 / 38.81 | - | 46.10 / 46.57 | - |
| 3 block mode, GPU part off | 38.75 | 0.0% | 46.41 | +0.2% (noise) |
| 2 op mode + GPU markers | 38.33 | -1.1% | | |
| 4 block mode + GPU markers | 38.18 | -1.5% | 45.80 | -1.1% |

CPU logging costs nothing measurable at either granularity; the GPU marker kernels cost about 1-1.5%. The patched build is a native build (`GGML_NATIVE`, CUDA 13.3) and equals the prebuilt at `0xFFFF`; it is faster at `0x5555` (38.8 vs 35.7), likely from scheduling differences under oversubscription (not investigated).

## 9b. As built (spike)

- llama.cpp worktree `llama.cpp-prof`, branch `llamascope-profiler` (uncommitted). New: `ggml/src/ggml-profiler.{h,cpp}` (ggml-base), `ggml/src/ggml-cuda/profiler.{cu,cuh}`. Hooks: ggml-cpu.c compute loop, ggml-backend.cpp split loop, ggml-cuda.cu (prepare, split begin, after each node incl. fused ones, flush on backend free), one context member in common.cuh.
- Environment: `GGML_PROFILE=block|op`, `GGML_PROFILE_FILE=<path>` (default `ggml-profile.bin`), `GGML_PROFILE_MB` (per thread, default 32), `GGML_PROFILE_THREADS` (slots, default logical CPUs + 1), `GGML_PROFILE_GPU=0` (CPU side only).
- GPU: marker kernels (`%globaltimer`) instead of CUPTI — they are captured into CUDA graphs and replay with them; CUPTI kernel records inside replayed graphs are hard to map back to tensors. Clock pair from the shortest of 16 launch/sync round trips.
- Data is written at process exit; GPU markers and expert counts are handed over when the CUDA backend is freed (normal exit). A forced exit (double Ctrl+C in llama-cli) loses the GPU part.

File format `GGMLPRF2` (little endian):

| Section | Content |
|---|---|
| Header | magic, u32 version=2, u32 level, u64 tsc_hz, u64 tsc0, u64 dropped, u32 n_slots |
| Per slot | u32 n, n x entry (u64 t0, u64 t1, u64 t2, u32 name, u8 thread, u8 kind, u16 count) |
| Names | u32 n, n x (u32 hash, u16 len, bytes) |
| GPU markers | u32 n, n x (u64 tsc, u32 name, u32 seq) — name 1 = split begin |
| Experts | u32 layers, u32 experts, gen[layers][experts] u32, prompt[layers][experts] u32 |

Entry kinds: 1 op (t0 start, t1 compute end, t2 barrier end), 2 block (t0 first start, t1 = sum compute ticks, t2 = sum wait ticks), 3 scheduler event (name = 1 copy-in, 2 submit, 3 sync, 4 graph start; t2 = backend id, or token count for graph start; count = split index).

Running: `scripts/Profile-Llama.ps1 <llama-cli flags>` writes to `%LOCALAPPDATA%\LlamaScope\profiles`; LlamaScope > Model topology > Runtime profile > Load latest.

## 10. Upstreaming path

Small, self-contained: `ggml-profiler.{h,c}` in ggml-base (buffers, clock, dictionary, file writer), probes behind `if (ggml_prof_level)` in ggml-cpu, ggml-backend (sched) and ggml-cuda, CUPTI optional at build time (`GGML_CUDA_CUPTI`), enabled with `GGML_PROFILE=op|block` and `GGML_PROFILE_FILE=path`. No change to llama.cpp's public API.
