"""Downloads the SWE-bench parquet exports (the HF rows API returned 503) into data/parquet."""
import os, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "parquet")
os.makedirs(OUT, exist_ok=True)
FILES = {
    "swe-verified": "https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified/resolve/main/data/test-00000-of-00001.parquet",
    "swe-multilingual": "https://huggingface.co/datasets/SWE-bench/SWE-bench_Multilingual/resolve/main/data/test-00000-of-00001.parquet",
    "swe-pro": "https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro/resolve/main/data/default/test-00000-of-00001.parquet",
}
for name, url in FILES.items():
    path = os.path.join(OUT, name + ".parquet")
    if os.path.exists(path):
        continue
    data = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "llamascope-expert-study"}), timeout=120).read()
    open(path, "wb").write(data)
    print(name, len(data))
