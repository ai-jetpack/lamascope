# Expert placement: static hot set vs. moving experts at runtime — study and design notes

| | |
|---|---|
| Status | Study complete (§4); design (§5) is a proposal, nothing implemented in llama.cpp |
| Data | 2000 prompts from SWE-bench Verified (450), SWE-bench Multilingual (300), SWE-bench Pro (450), Aider polyglot (225), HumanEval (164), MBPP (411); greedy, 128 generated tokens each, prompts clipped to ~6000 characters |
| Tools | `llamascope-batch` (profiler branch), `GGML_PROFILE_TRACE=1` (experts picked per generated token and layer), `expertstudy/analyze.py` |

## 1. Why

Qwen3-Coder-Next has 48 layers x 512 experts, 10 picked per token per layer. With `--n-cpu-moe 36` the GPU holds all experts of layers 36-47 (6144 expert slots) and serves exactly 25% of all expert picks; the CPU computes the other 75% from system RAM. A first 200-token run suggested usage is very uneven (53% of (layer, expert) slots never picked), so the same VRAM filled with the most-used experts could serve far more picks.

## 2. What llama.cpp allows today

Nothing per expert. Each layer's experts are one tensor per projection (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`, shape `[n_embd, n_ff, n_expert]`); `--n-cpu-moe` and `-ot/--override-tensor` place whole tensors. Per-expert placement needs a llama.cpp change.

## 3. Constraints that decide whether it pays off

- **Hand-offs:** today 36 layers cross PCIe, ~64 µs per crossing including the non-waiting syncs (blocking copies: hidden state 8 KB GPU->CPU 27.6 µs, expert ids 40 B 6.1 µs, experts' result 80 KB CPU->GPU 27.4 µs), ~2.4 ms = ~10% of each token (corrected 2026-11-09: an earlier marker-based estimate said ~164 µs / 27%; see §4.5). With hot experts in every layer, a token whose picks are not all hot still needs the CPU for that layer. Unless the hot path covers most tokens completely, crossings go up, not down.
- **Overlap is the real prize:** today the GPU idles while the CPU computes experts and vice versa (measured: CPU experts ~9.7 ms, GPU ~8.2 ms, hand-offs ~2.4 ms, launches/scheduling ~2.6 ms per token, strictly sequential). If hot experts run on the GPU *while* the CPU computes the cold ones for the same layer, the layer costs max(GPU, CPU) instead of the sum.
- **Static graphs:** ggml builds one graph per token shape; it cannot skip the CPU branch for tokens whose picks are all hot without a new op or graph-level condition.
- **Moving experts:** one expert = ~1.8 MB (gate+up+down, Q4_K). Host-to-device at ~25 GB/s = ~70 µs per expert. Moving a handful per token is affordable only on a separate copy stream, overlapped with compute.

## 4. Results (2026-11-09)

Run: 2000 prompts in 10 batches, 255,631 generated tokens, 662,477 prompt tokens, 2 h 14 min; all batches completed; trace complete (12,270,320 (token, layer) entries = 255,631 x 48). Halves: A = SWE-Verified 1st half, SWE-Multilingual 1st half, SWE-Pro 1st half, Aider, MBPP 1st half; B = the other halves, HumanEval. Files: `expert-study/report.md`, heatmaps `expert-study/*.png`, merged counts `expert-study-merged.prof` (opens in LlamaScope).

**4.1 Usage is concentrated but not sparse.**

| source | generated tokens | never-picked (layer, expert) slots | picks served by each layer's top 10% / 25% experts |
|---|---|---|---|
| aider-polyglot | 28,804 | 562 (2%) | 45% / 70% |
| humaneval | 20,592 | 1,719 (7%) | 47% / 73% |
| mbpp | 52,609 | 1,152 (5%) | 47% / 73% |
| swe-multilingual | 38,406 | 820 (3%) | 52% / 80% |
| swe-pro | 57,609 | 1,273 (5%) | 54% / 83% |
| swe-verified | 57,609 | 687 (3%) | 53% / 81% |
| all | 255,631 | 137 (0.6%) | 41% / 69% |

The 53% never-picked figure from the single 200-token run does not hold: over 255k tokens only 137 of 24,576 slots are never picked. Concentration is real (top 25% of each layer's experts = 69% of picks), and it differs by workload (SWE prompts more concentrated than Aider).

**4.2 Never-used experts in the GPU layers (your VRAM idea).** Layers 36-47 (on the GPU today):

| experts in layers 36-47 with ... | count | VRAM | layers of experts that VRAM could hold instead | share of all picks they served |
|---|---|---|---|---|
| zero picks | 86 | 152 MB | 0.17 | 0% |
| <= 0.01% of their layer's picks | 813 | 1.44 GB | 1.59 | 0.06% |
| <= 0.05% | 2,066 | 3.66 GB | 4.04 | 0.80% |
| <= 0.1% | 3,129 | 5.54 GB | 6.11 | 2.44% |

Zero-pick experts free 0.17 layers: not enough for one more layer. Rarely-used experts (<= 0.01%) free 1.6 layers at a cost of 0.06% of picks moving to the CPU. Either way requires per-expert placement (§2); with stock llama.cpp it is not possible. Generalized, this is the static hot set of 4.3.

**4.3 Static hot set** (same 6,144 GPU slots, chosen by usage instead of whole layers):

| placement | picks served by the GPU |
|---|---|
| today, layers 36-47 | 25.0% |
| hot set learned on A, measured on B / learned on B, measured on A | 68.7% / 68.1% |
| learned and measured on everything | 69.3% |
| half the slots (3,072), A->B | 46.9% |

The held-out numbers are within 1.2 points of the fitted one: a hot set learned from one mix of benchmarks transfers to another (including HumanEval, which is only in B, and Aider, only in A).

But: only 12.7% of (token, layer) steps have all 10 picks hot. In the other 87.3% the CPU still has work in that layer, so crossings per token rise from 36 to 41.9 (if llama.cpp could skip the CPU branch per step) or 48 (static graph, as today).

**4.4 Moving experts at runtime** (per-layer cache of 128 experts, simulated on the trace, causal):

| policy | GPU hit rate | experts moved per token | PCIe time per token (1.77 MB/expert at 25 GB/s) |
|---|---|---|---|
| LRU, move on miss | 83.6% | 78.6 | 5.44 ms |
| decayed LFU, re-plan every token | 84.4% | 60.5 | 4.18 ms |
| decayed LFU, re-plan every 16 tokens | 79.5% | 47.3 | 3.27 ms |
| decayed LFU, re-plan every 128 tokens | 75.6% | 7.4 | 0.51 ms |

Per-token movement (LRU, re-plan every token) costs 3-5 ms of PCIe per token — as much as today's hand-offs — and competes with them on the same link. Re-planning every 128 tokens gains 7 points over the static hot set for 0.5 ms/token of copies (bursty: ~950 experts = 1.7 GB per re-plan, must be spread out). Simulation limits: moves are assumed complete by the next re-plan; per-layer capacity is uniform (the static set let hot layers take more slots).

**4.5 What it would buy (estimate, not measured).** Inputs: per-copy profile (format v3, `-C 0xFFFF`, 30 generation steps): step 22.9 ms = CPU experts 9.7 + GPU 8.2 + hand-offs 2.4 (36 x ~64 µs + per-step inputs) + other 2.6 (launches, scheduling); GPU expert cost ~6 µs per pick; "other" assumed to grow with the number of splits.

| scenario | CPU | GPU | hand-offs | other | step |
|---|---|---|---|---|---|
| today | 9.7 | 8.2 | 2.4 | 2.6 | 22.9 measured |
| static hot set, sequential, 48 crossings | 4.1 + ~1.2 per-layer overhead | 9.5 | 3.1 | ~3.4 | ~21.3: ~+7% |
| static hot set, GPU hot experts overlapped with CPU cold experts | same | same | 3.1 | ~3.4 | ~19.3: ~+19% |
| today's placement, hand-off cut from ~64 to ~20 µs per crossing | 9.7 | 8.2 | 0.8 | 2.6 | ~21.3: ~+8% |

Correction (2026-11-09): the first version of this table used a marker-based hand-off estimate (164 µs per crossing, 5.9 ms per token) and concluded that placement alone gives no gain and that hand-off latency is the larger lever (+26%). Direct per-copy measurement gives ~64 µs per crossing; the marker method mixed GPU compute time into the hand-off because of the GPU-to-CPU clock conversion. With the corrected figure, placement with overlap is the larger estimated gain (~+19%); hand-off reduction ~+8%; they are independent and add up only partly.

Why a hand-off costs ~64 µs (measured): every input of a CPU split takes the scheduler's fallback path (the CPU backend has no async copy): `ggml_backend_synchronize` of the CUDA backend, then a blocking `cudaMemcpy`. 8 KB takes 27.6 µs (~0.3 GB/s), 40 B takes 6.1 µs: latency-bound, the 8 KB copy likely through a pageable-memory staging copy (not verified). Order of work this suggests: (1) pinned buffers / one combined copy per crossing (cheap, independent), (2) GPU/CPU overlap, (3) hot-set placement, (4) runtime movement with long re-plan periods.

Data caveats: greedy decoding, 128 generated tokens per prompt, prompts clipped to 6000 characters, one model and quantization; prompt-processing picks (311M) were recorded but the placement analysis uses generation picks only.

## 5. Design sketch for a llama.cpp change (for discussion, not implemented)

1. **Split storage per layer:** keep the full expert tensors in host memory (as today with `--n-cpu-moe`) plus a GPU *cache tensor* with C expert slots per layer (`[n_embd, n_ff, C]` per projection) and a GPU remap table `slot_of[n_expert]` (int32, -1 = not cached).
2. **Routing:** after top-k, a small kernel splits the picked ids into hits (remapped to slots) and misses. GPU `mul_mat_id` runs on hits against the cache tensor; CPU `mul_mat_id` runs on misses against the host tensors; results are summed with the router weights. Hits and misses for the same layer run concurrently (GPU and CPU), which is where the time is saved.
3. **Static first:** fill the cache from usage counts collected by the profiler (a "hot set" file per model and workload), loaded at startup. No runtime movement, no new synchronization.
4. **Dynamic later:** a policy thread keeps decayed per-expert scores from the topk ids (already on the GPU for counting), re-plans every N tokens, uploads newly hot experts on a copy stream into free slots, then flips the remap entry (double-buffered so a token never sees a half-copied slot).
5. **Upstream path:** discuss as an issue first (new subsystem, touches the scheduler and CUDA backend). Static hot sets alone could be expressed as a GGUF repack (split each expert tensor into hot/cold tensors + remap metadata) to keep core changes small.
