# Decisions made overnight (2026-11-08/09) — for review

Context: you asked to implement the in-process profiler (your per-thread buffer design), measure its cost, and integrate it with LlamaScope while you were away. Everything below is local: **nothing is committed or pushed**, in either repository. The llama.cpp changes live on branch `llamascope-profiler` in a separate worktree (`llama.cpp-prof`, next to the `llama.cpp` checkout); your `llama.cpp` checkout and its `build` folder are untouched.

Benchmark settings used everywhere (your command line, translated to llama-bench): `-ngl 999 -ncmoe 36 -fa on -t 16 -C 0x5555 --cpu-strict 1 --poll 100 -ub 1024 -ctk q8_0 -ctv q8_0 -lm mlock`, generation only (`-p 0 -n 128 -r 5`), mean of repetitions 2-5.

## D1. Upstream rules shape the work

llama.cpp's AGENTS.md / code-review skill: features start as a discussion issue (not a PR), first PRs CPU-only with other backends as follow-ups, the contributor must own the design, and agents must not commit, push, or write PR text. So: local branch only; the design doc is written so you can turn it into the discussion issue yourself. I did not search upstream issues/PRs yet (no `gh` installed) — do that before opening the issue.

## D2. Where the hooks are (CPU)

Per node, per thread in `ggml_graph_compute_thread` (ggml-cpu.c): tick before compute, after compute, after the barrier. Gives compute time and barrier wait (OpenMP spin) separately. `rdtsc` clock, plain stores into a cache-line-aligned per-thread slot, no locks/atomics/kernel calls, preallocated and pre-touched buffers (first version used `calloc`, which on Windows does not touch pages -> page faults in the hot path; fixed with `memset`). Entries dropped and counted if a buffer fills (never allocate on the hot path).

## D3. Scheduler events

Main thread only, in `ggml_backend_sched_compute_splits`: graph start (with the token count, found from the `inp_tokens` leaf, to separate prompt from generation), input copies per split, compute submit per split, explicit syncs. Note: a CPU split's input copy from the GPU is a blocking device-to-host copy, so its duration includes waiting for the GPU's queued work; the GPU markers (D4) are used to subtract that.

## D4. GPU timing: marker kernels, not CUPTI

Your design said "one buffer for the GPU"; I proposed CUPTI earlier. Implemented instead: a 1-thread kernel that writes the GPU's `%globaltimer` (ns) plus the block name into a device ring (atomic index) after each block-end node, and one at the start of each GPU split. Reasons: llama.cpp replays generation steps as CUDA graphs, and mapping CUPTI kernel records back to tensors inside replayed graphs is complex; CUPTI is also an extra dependency, harder to upstream. Marker kernels are captured into the graphs and replay with them. Cost: ~5 tiny kernels per layer (measured below). GPU clock -> TSC conversion: 16 round trips at flush, the shortest one's midpoint as the reference pair.

## D5. Granularity

`GGML_PROFILE=op` (one entry per node per thread) and `GGML_PROFILE=block` (compute/wait folded per block, one entry at block-end nodes: `attn_residual`, `ffn_moe_topk`, `ffn_moe_out`, `ffn_shexp`, `l_out`). GPU side is block-level in both modes. `GGML_PROFILE_GPU=0` turns the GPU part off (to measure it separately).

## D6. Expert usage

Counted at `ffn_moe_topk-N` nodes: on the GPU with an `atomicAdd` counting kernel (no per-token transfer, copied once at the end), on the CPU from host memory when the router runs there. In your configuration the router runs on the GPU for all 48 layers, so the GPU path covers everything. Separate counts for prompt (tokens > 1) and generation.

## D7. Output and consumer

Drain at the end (process exit or CUDA backend free) into one binary file, format `GGMLPRF2` (header, per-thread entries, name dictionary, GPU markers converted to TSC, expert counts). No live consumer yet (your E-core consumer design is documented in PROFILER.md §7 as the next step); reason: end-of-run drain has zero runtime cost and was enough for the cost measurement and the diagram.

## D8. LlamaScope integration

- Reader `src/model/Profile.{h,cpp}`: splits the stream into generation steps (graphs with 1 token, skipping the first two steps = warm-up and CUDA graph capture), aggregates per layer and block: CPU wall time (thread 0 compute + wait), barrier-wait share across threads, GPU block time (gap between consecutive markers), transfers (CPU split copy-in minus the GPU wait, GPU split copy-in after CPU work).
- Topology panel: time badges on the context stage, dispatcher, expert row (with % waiting), always-on expert, merge; measured latency on the PCIe labels; per-layer header shows GPU/CPU time; an expert-usage strip per layer (one cell per expert); summary pane "Runtime profile": Open / Load latest (from `%LOCALAPPDATA%\LlamaScope\profiles`), per-token totals, and a layers x experts usage heatmap (generation or prompt).
- `--profile-dump <file>`: headless log of the aggregation, used to check the reader.

## D9. Results

**Profiler cost** (llama-bench, details in PROFILER.md §9):

| Mode | `-C 0x5555` | `-C 0xFFFF` |
|---|---|---|
| block, CPU only | 0.0% | +0.2% (noise) |
| op + GPU markers | -1.1% | not measured |
| block + GPU markers | -1.5% | -1.1% |

CPU logging is free in practice; the GPU marker kernels cost ~1-1.5%. Within the targets (≤1% for CPU logging; the GPU part is slightly above 1%).

**It works on real runs:** `scripts\Profile-Llama.ps1` with your exact llama-cli command line (plus `-st -p ... -n 200` to make it non-interactive) produced a complete profile (146,887 CPU entries, 56,226 GPU markers, expert counts for 48 x 512); LlamaScope loads it and draws it (screenshots were taken with the new `--snapshot` mode because the monitors were asleep).

## D10. Findings worth your attention (morning review)

1. **Your "best" command line is probably not the best.** In llama-bench, `-t 16 --cpu-mask 0x5555 --cpu-strict 1` is ~20% slower than `-t 16 --cpu-mask 0xFFFF` (prebuilt: 35.7 vs 46.9 t/s; patched: 38.8 vs 46.5). With `0x5555` and `--cpu-strict`, 16 threads are placed on 8 logical CPUs (2 per CPU). The profile shows why: CPU threads wait at barriers 21% of their time (vs 4% with `0xFFFF`), CPU expert blocks are slower (284 vs 259 µs per layer), and the main thread's copies slow down (192 vs 135 µs). Please re-check in llama-cli, where you measured it; the profiler makes the difference visible on the diagram.
2. **CPU<->GPU hand-offs: ~10% of each token (corrected 2026-11-09).** The overnight figure (~164 µs per crossing, 5.9 ms, 27%) came from subtracting a GPU timestamp converted to CPU time and included GPU compute. Direct per-copy timing (profiler format v3): per crossing ~64 µs including the non-waiting syncs: hidden state 8 KB GPU->CPU 27.6 µs, expert ids 40 B 6.1 µs, experts' result 80 KB CPU->GPU 27.4 µs, each preceded by a full `ggml_backend_synchronize` (the CPU backend has no async copy, so every input takes the scheduler's fallback path). Step at `0xFFFF`: CPU 9.7 + GPU 8.2 + hand-offs 2.4 + other 2.6 = 22.9 ms; the main thread additionally waits 5.5 ms for the GPU inside copies (part of GPU busy). Lead: pinned buffers and one combined copy per crossing; not tested.
3. **Expert usage is very uneven within a run:** in a 200-token code-generation run, 13,015 of 24,576 (layer, expert) slots were never picked. With longer/varied runs this becomes a basis for smarter placement (keep hot experts on the GPU) — note llama.cpp has no per-expert placement today.
4. The patched native build equals the prebuilt at `0xFFFF` and is ~9% faster at `0x5555`; not investigated.

## D11. Other changes made tonight (LlamaScope)

- `--snapshot <png> [--snapshot-after s]`, `--topology-scroll <layer.frac>`, `--summary-scroll <frac>`: render and save a screenshot without needing the window visible (DirectX stops presenting while monitors sleep; LlamaScope skips rendering when occluded).
- `--profile <file>` / `--profile-latest`: load a profile at startup. `--profile-dump <file>`: headless aggregation log.
- Exited processes kept alive by open handles (0 threads, e.g. old llama-cli sessions) are now ignored by the process monitor and Inference mode (they showed up as "llama-cli running").
- `scripts\Profile-Llama.ps1`: runs the patched llama-cli/llama-bench with profiling into `%LOCALAPPDATA%\LlamaScope\profiles`. It has no `param()` block on purpose (PowerShell bound llama's `-m` to a script parameter); options are `--profile-tool= / --profile-mode= / --profile-bin=`.

## D12. Open items / not done

- No live view: the profile is read after the run. Next step is your E-core consumer (SPSC rings, drained every 50-100 ms) feeding LlamaScope during generation.
- Forced exit (double Ctrl+C) loses the GPU part; a flush on the first Ctrl+C or at `llama_backend_free` would fix it.
- GPU-side "op" granularity is not implemented (block-level only on the GPU).
- Concurrent CUDA streams (when llama.cpp uses them) skip markers.
- Upstream: search existing issues/PRs, then open a discussion issue (you write it). The first PR should be CPU-only (scheduler + CPU hooks), GPU markers as a follow-up.
- The llama.cpp worktree build includes the server (`LLAMA_BUILD_SERVER=ON`) only because `llama-cli` depends on it.

## How to try it

```
# 1. profile a session (your flags; end with /exit, or add -st -p "..." for one turn)
$env:LLAMACPP_PROF_BIN = "<llama.cpp-prof>\build-prof\bin"; $env:CUDA_BIN = "<CUDA v13.3>\bin"
.\scripts\Profile-Llama.ps1 -m <model.gguf> -ngl 999 --n-cpu-moe 36 ...
# 2. view it
.\build\Release\LlamaScope.exe --profile-latest        (or Model topology > Runtime profile > Load latest)
```

## D13. Expert usage study (added 2026-11-09, details in EXPERTS.md §4)

- 2000 prompts (SWE-bench Verified 450, Multilingual 300, Pro 450, Aider polyglot 225, HumanEval 164, MBPP 411), greedy, 128 generated tokens each; 255,631 generated tokens; runner `llamascope-batch` (new tool on the profiler branch) + `GGML_PROFILE_TRACE=1` (per-token expert trace). HF rows API returned 503: SWE-bench read from the parquet exports with a stdlib Parquet reader (`scripts/expert-study/miniparquet.py`).
- Only 137 of 24,576 (layer, expert) slots are never picked over the full run (0.6%); the 53% from the earlier 200-token run was a small-sample artifact.
- Freeing never-picked experts from the GPU layers frees 152 MB = 0.17 of a layer: not enough for another layer. Experts with <= 0.01% of their layer's picks free 1.6 layers. Both need per-expert placement, which stock llama.cpp cannot do.
- Static hot set in today's GPU budget: 68.7% / 68.1% of picks on held-out halves vs 25% today. But only 12.7% of (token, layer) steps have all picks hot, so CPU<->GPU crossings rise from 36 to 42-48 per token. Estimated step time (corrected 2026-11-09 with the per-copy hand-off cost): ~+7% sequentially; ~+19% with GPU/CPU overlap; cutting hand-offs from ~64 to ~20 µs per crossing ~+8% with today's placement. Estimates, not measurements (EXPERTS.md §4.5).
- Runtime movement (simulated): 76-84% hit rate; per-token policies need 3-5 ms of PCIe copies per token; re-planning every 128 tokens needs 0.5 ms/token.
- Scripts in `scripts/expert-study/` use environment variables (`LLAMACPP_PROF_BIN`, `LLAMASCOPE_MODEL`, `CUDA_BIN`); personal paths removed from all repo files, including `scripts/Profile-Llama.ps1`. `prompts.jsonl` and the downloaded datasets are not in the repo (regenerable; dataset licenses not reviewed).
