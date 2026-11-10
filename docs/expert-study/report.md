# Expert usage study

10 batches; generation picks 122,703,200 (~255,631 generated tokens); prompt picks 311,364,190.

## 1. Concentration

| source | gen tokens | never-picked (layer, expert) slots | picks served by the top 10% / 25% experts of each layer |
|---|---|---|---|
| aider-polyglot | 28,804 | 562 of 24,576 (2%) | 45% / 70% |
| humaneval | 20,592 | 1,719 of 24,576 (7%) | 47% / 73% |
| mbpp | 52,609 | 1,152 of 24,576 (5%) | 47% / 73% |
| swe-multilingual | 38,406 | 820 of 24,576 (3%) | 52% / 80% |
| swe-pro | 57,609 | 1,273 of 24,576 (5%) | 54% / 83% |
| swe-verified | 57,609 | 687 of 24,576 (3%) | 53% / 81% |
| all | 255,631 | 137 of 24,576 (1%) | 41% / 69% |

## 2. Static placement (same GPU memory: 6144 expert slots)

| placement | share of picks served by the GPU |
|---|---|
| today: experts of layers 36-47 (`--n-cpu-moe 36`) | 25.0% |
| most-used experts, learned on half A, measured on half B | 68.7% |
| most-used experts, learned on half B, measured on half A | 68.1% |
| most-used experts, learned and measured on everything (optimistic) | 69.3% |

- With only 25% of that memory (1536 slots), learned on A, measured on B: 30.5%
- With only 50% of that memory (3072 slots), learned on A, measured on B: 46.9%

## 3. Dynamic placement (per-layer cache of 128 experts, updated at runtime)

| policy | GPU hit rate | experts moved per token | PCIe time per token to move them |
|---|---|---|---|
| LRU (move on miss) | 83.6% | 78.6 | 5.44 ms |
| decayed LFU, re-plan every token (decay 0.98) | 84.4% | 60.5 | 4.18 ms |
| decayed LFU, re-plan every 16 tokens | 79.5% | 47.3 | 3.27 ms |
| decayed LFU, re-plan every 128 tokens (decay 0.995) | 75.6% | 7.4 | 0.51 ms |

Trace: 12,270,320 (token, layer) entries. One expert = 1.77 MB; PCIe 25.0 GB/s host to device.

Heatmaps: heatmap_generation.png, heatmap_prompt.png, heatmap_<source>.png (rows = layers 0-47, columns = experts 0-511, brightness = picks normalized per layer). Merged counts for LlamaScope: expert-study-merged.prof.
