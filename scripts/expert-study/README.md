# Expert usage study scripts

Reproduces docs/EXPERTS.md §4. Requires the profiler-patched llama.cpp (`llamascope-batch`, `GGML_PROFILE_TRACE`, see docs/PROFILER.md). Paths to the model, the llama.cpp build and the CUDA toolkit are hard-coded at the top of `run_study.ps1`.

1. `python -I fetch_parquet.py` - SWE-bench parquet exports from Hugging Face (the HF rows API returned 503).
2. `python -I fetch_prompts.py` - builds `prompts.jsonl` (2000 prompts; Aider via git, HumanEval/MBPP from GitHub). Downloads go to `data/` and are read as text only.
3. `run_study.ps1` (or `wait_and_run.ps1`, which waits until no llama process holds the model) - 10 batches, one profile each.
4. `python -I analyze.py`, `python -I analyze2.py` - report.md, heatmaps, merged counts, follow-up numbers.

`miniparquet.py` is a minimal stdlib Parquet reader (flat string columns, Snappy/GZIP, dictionary and plain encodings).
