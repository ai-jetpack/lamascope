# Why the CPU expert phase reaches ~84% of the memory-test bandwidth (2026-11-09)

Question: the CPU expert phase reads 684 MB per token in 9.9 ms = 69.1 GB/s, 84% of the 82.7 GB/s memory-test figure. Where do the other 16% go?

## Measurements

**Op-mode profile** (llama-bench, `-t 16 -C 0xFFFF --cpu-strict 1`, 30 generation steps, 36 CPU layers):

| op (per layer) | wall | thread compute min / mean / max | weights | GB/s |
|---|---|---|---|---|
| ffn_moe_gate | 85.3 µs | 74.0 / 78.9 / 83.9 µs | 5.90 MB | 69.1 |
| ffn_moe_up | 83.0 µs | 72.1 / 77.0 / 81.9 µs | 5.90 MB | 71.0 |
| ffn_moe_down | 104.4 µs | 91.7 / 98.1 / 104.2 µs | 7.21 MB | 69.0 |

**Standalone read benchmark** (`scripts/expertbw`): 8 x 512 experts x 576 KB (2.4 GB), 10 random experts per op, AVX2 loads only, spin barrier per op, 3 rounds per setting.

| what was varied | result (GB/s) |
|---|---|
| read schedule: today's 1/16 of every expert per thread (A) / one thread per expert (B) / contiguous 1/16 of the 10 experts (C) / one stream per thread (L) | A 69-75, B 66-70, C 74-76, L 73-76 |
| 2 MB large pages instead of 4 KB | no clear change (A 71, C 74, L 76) |
| barrier after every 1 / 2 / 3 / 6 ops | 71-75 / 76-77 / 76-77 / 76-78 |
| threads: 16 HT on 8 P-cores / 8 P / 8 P + 8 E / all 24 | 72-76 / 73-75 / 71-74 / 64-75 |
| one ~150 MB stream per thread (memory-test pattern), P-core HT or P+E layout | 74-77 (outliers 65-69) |
| LlamaScope memory test at the same time (best / median of 10 windows) | 80.0 / 79.2 and 79.2 / 78.0 |

## Breakdown

| from | to | gap | cause |
|---|---|---|---|
| 82.7 GB/s (earlier best-of-10 memory test) | ~78-79 (median memory test today) | ~5% | the reference: best window on a quiet machine vs median; background load (desktop, iGPU scan-out) varies |
| ~78-79 | ~76-77 (same reads, few barriers) | ~2-3% | harness averages every pass including barriers and slow outliers |
| ~76-77 | ~74-75 (barrier after every op, real op size) | ~3% | per-op barrier and pipeline ramp/drain at each ~80 µs op |
| ~74-75 | 69 (real inference) | ~7-8% | slowest thread ~6% above the mean (load imbalance, measured in the profile), plus unpacking/compute |

## Hypotheses tested and refuted

- Reading each expert in 1/16 slices (short streams) is slower than contiguous ranges: no difference beyond noise.
- One thread per expert: slower (10 work items on 8 cores, idle threads).
- TLB misses with 4 KB pages: large pages give no clear gain.
- More cores (E-cores) raise bandwidth: no.

## What could recover bandwidth (llama.cpp changes, not tested)

1. Balance work between threads in the expert matmul (~6-8% of the CPU phase).
2. Fewer barriers per layer, e.g. gate and up computed as one op (~3%).

Combined: CPU expert phase ~9.9 -> ~8.9 ms, ~4% tokens/s at today's placement (estimate).
