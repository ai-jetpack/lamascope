"""Builds up to 2000 coding prompts from public eval datasets (stdlib only).

Output: prompts.jsonl with {"id", "source", "prompt"}, grouped by source. Downloads are cached under ./data and only
read as text. The Hugging Face rows API returned 503, so SWE-bench comes from the parquet exports (fetch_parquet.py,
read with miniparquet.py) and HumanEval / MBPP from their GitHub copies.

Mix (2000): SWE-bench Verified 450, SWE-bench Multilingual 300, SWE-bench Pro 450, Aider polyglot 225,
HumanEval 164, MBPP 411.
"""
import gzip, json, os, subprocess, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)   # run with -I; only this folder (not data/) is importable
from miniparquet import read_columns
DATA = os.path.join(HERE, "data")
MAX_CHARS = 6000          # ~1.5k tokens: keeps prompt processing bounded
UA = {"User-Agent": "llamascope-expert-study"}

def get(url, name):
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        open(path, "wb").write(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120).read())
    return open(path, "rb").read()

def clip(s):
    s = (s or "").strip()
    return s if len(s) <= MAX_CHARS else s[:MAX_CHARS] + "\n[...truncated]"

def swe(name, n):
    c = read_columns(os.path.join(DATA, "parquet", name + ".parquet"), ["instance_id", "repo", "problem_statement"])
    out = []
    for iid, repo, ps in list(zip(c["instance_id"], c["repo"], c["problem_statement"]))[:n]:
        out.append({"id": f"{name}-{iid}", "source": name,
                    "prompt": clip(f"You are working in the {repo} repository. Here is an issue report:\n\n{ps}\n\n"
                                   "Explain the likely cause and write the code change that fixes it.")})
    return out

def aider(n):
    repo = os.path.join(DATA, "polyglot-benchmark")
    if not os.path.exists(repo):
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/Aider-AI/polyglot-benchmark.git", repo], check=True)
    out = []
    for lang in sorted(os.listdir(repo)):
        ex = os.path.join(repo, lang, "exercises", "practice")
        if not os.path.isdir(ex):
            continue
        for name in sorted(os.listdir(ex)):
            md = os.path.join(ex, name, ".docs", "instructions.md")
            if os.path.exists(md):
                text = open(md, encoding="utf-8", errors="replace").read()
                out.append({"id": f"aider-{lang}-{name}", "source": "aider-polyglot",
                            "prompt": clip(f"Solve this exercise in {lang}. Write the complete solution.\n\n{text}")})
    return out[:n]

def humaneval(n):
    raw = gzip.decompress(get("https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz", "HumanEval.jsonl.gz"))
    rows = [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]
    return [{"id": f"humaneval-{r['task_id']}", "source": "humaneval", "prompt": clip("Complete this Python function:\n\n" + r["prompt"])}
            for r in rows[:n]]

def mbpp(n):
    raw = get("https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl", "mbpp.jsonl")
    rows = [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]
    return [{"id": f"mbpp-{r['task_id']}", "source": "mbpp",
             "prompt": clip(f"{r['text']}\nYour code should pass these tests:\n" + "\n".join(r.get("test_list", [])))}
            for r in rows[:n]]

def main():
    os.makedirs(DATA, exist_ok=True)
    parts = [swe("swe-verified", 450), swe("swe-multilingual", 300), swe("swe-pro", 450), aider(225), humaneval(164)]
    prompts = [p for part in parts for p in part]
    prompts += mbpp(2000 - len(prompts))
    prompts = prompts[:2000]
    with open(os.path.join(HERE, "prompts.jsonl"), "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    counts = {}
    for p in prompts:
        counts[p["source"]] = counts.get(p["source"], 0) + 1
    print(counts, "total", len(prompts))

if __name__ == "__main__":
    main()
