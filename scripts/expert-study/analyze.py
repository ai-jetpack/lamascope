"""Expert-usage study: reads the batch profiles (GGMLPRF2 + expert trace), writes report.md, heatmaps and a merged profile.

Questions answered:
  1. How concentrated is expert usage (per layer, per source)?
  2. Static placement: if the GPU held the most-used experts instead of whole layers, what share of picks would it serve?
     Learned on half A of the batches, measured on half B (and the reverse) to avoid fitting to the test data.
  3. Dynamic placement: per-layer caches updated at runtime from recent picks (LRU, decayed LFU). Hit rate and the PCIe
     traffic needed to move experts, from the per-token expert trace.
"""
import glob, math, os, struct, sys, zlib
from collections import defaultdict, OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
PROF = os.path.join(HERE, "profiles")
GPU_LAYERS = 12                  # --n-cpu-moe 36 on 48 layers
EXPERT_MB = 1.77                 # one expert's gate+up+down weights (Q4_K_XL), from the GGUF tensor sizes
PCIE_GBS = 25.0                  # measured host->device copy speed

def read(path):
    b = open(path, "rb").read()
    assert b[:8] in (b"GGMLPRF2", b"GGMLPRF3"), path
    v3 = b[:8] == b"GGMLPRF3"           # v3: each name entry carries the tensor size (u64)
    o = 8 + 4 + 4 + 8 + 8 + 8
    (n_slots,) = struct.unpack_from("<I", b, o); o += 4
    for _ in range(n_slots):
        (n,) = struct.unpack_from("<I", b, o); o += 4 + n * 32
    (n_names,) = struct.unpack_from("<I", b, o); o += 4
    for _ in range(n_names):
        _, ln = struct.unpack_from("<IH", b, o); o += 6 + ln + (8 if v3 else 0)
    (n_gpu,) = struct.unpack_from("<I", b, o); o += 4 + n_gpu * 16
    nl, ne = struct.unpack_from("<II", b, o); o += 8
    gen = [list(struct.unpack_from(f"<{ne}I", b, o + l * ne * 4)) for l in range(nl)]; o += nl * ne * 4
    prm = [list(struct.unpack_from(f"<{ne}I", b, o + l * ne * 4)) for l in range(nl)]; o += nl * ne * 4
    trace = []
    if o + 4 <= len(b):
        (nt,) = struct.unpack_from("<I", b, o); o += 4
        for i in range(nt):
            layer, n = struct.unpack_from("<HH", b, o + i * 36)
            trace.append((layer, struct.unpack_from(f"<{n}H", b, o + i * 36 + 4)))
    return nl, ne, gen, prm, trace

def add(a, b):
    return [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]

def png(path, rows, cols, values, sx, sy):
    """Heatmap PNG (black-red-yellow-white), values normalized per layer row; stdlib only."""
    def col(t):
        t = max(0.0, min(1.0, t))
        return (int(255 * min(1, 3 * t)), int(255 * min(1, max(0, 3 * t - 1))), int(255 * max(0, 3 * t - 2)))
    raw = bytearray()
    for r in range(rows):
        mx = max(values[r]) or 1
        line = bytearray()
        for c in range(cols):
            line += bytes(col(math.sqrt(values[r][c] / mx))) * sx
        for _ in range(sy):
            raw += b"\x00" + line
    w, h = cols * sx, rows * sy
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + \
           chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b"")
    open(path, "wb").write(data)

def coverage(counts_learn, counts_test, budget):
    flat = sorted(((c, l, e) for l, row in enumerate(counts_learn) for e, c in enumerate(row)), reverse=True)
    hot = {(l, e) for _, l, e in flat[:budget]}
    tot = sum(map(sum, counts_test))
    return sum(counts_test[l][e] for l, e in hot) / tot if tot else 0

def simulate(trace, nl, cap, policy, decay=0.98, period=1):
    """Per-layer cache of `cap` experts. Returns (hit rate, experts moved per token)."""
    caches = [OrderedDict() for _ in range(nl)]
    score = [defaultdict(float) for _ in range(nl)]
    hits = picks = moves = tokens = 0
    step = 0
    for layer, ids in trace:
        if layer == 0:
            tokens += 1
        c = caches[layer]
        for e in ids:
            picks += 1
            if e in c:
                hits += 1
                if policy == "lru":
                    c.move_to_end(e)
            if policy == "lru" and e not in c:
                c[e] = 1
                moves += 1
                if len(c) > cap:
                    c.popitem(last=False)
            if policy == "lfu":
                score[layer][e] += 1
        if policy == "lfu":
            step += 1
            if step % (nl * period) == 0:      # re-plan every `period` tokens
                for L in range(nl):
                    s = score[L]
                    for k in list(s):
                        s[k] *= decay ** period
                    want = set(sorted(s, key=s.get, reverse=True)[:cap])
                    have = set(caches[L])
                    moves += len(want - have)
                    caches[L] = OrderedDict((k, 1) for k in want)
    return (hits / picks if picks else 0), (moves / tokens if tokens else 0)

def main():
    files = sorted(glob.glob(os.path.join(PROF, "*.prof")))
    if not files:
        sys.exit("no profiles")
    data = {}
    for f in files:
        data[os.path.basename(f)[:-5]] = read(f)
    nl, ne = next(iter(data.values()))[:2]
    zero = [[0] * ne for _ in range(nl)]
    gen = {"A": zero, "B": zero}
    prm = zero
    by_src = defaultdict(lambda: zero)
    trace_all = []
    for name, (_, _, g, p, t) in data.items():
        half, src = name.split("-")[1], "-".join(name.split("-")[2:-1])
        gen[half] = add(gen[half], g)
        prm = add(prm, p)
        by_src[src] = add(by_src[src], g)
        trace_all += t
    allg = add(gen["A"], gen["B"])
    tot = sum(map(sum, allg))
    budget = GPU_LAYERS * ne
    lines = ["# Expert usage study", "",
             f"{len(files)} batches; generation picks {tot:,} (~{tot // (10 * nl):,} generated tokens); prompt picks {sum(map(sum, prm)):,}.", ""]

    # 1. concentration
    lines += ["## 1. Concentration", "", "| source | gen tokens | never-picked (layer, expert) slots | picks served by the top 10% / 25% experts of each layer |", "|---|---|---|---|"]
    def conc(c):
        t10 = t25 = 0
        for row in c:
            s = sorted(row, reverse=True)
            t10 += sum(s[: ne // 10]); t25 += sum(s[: ne // 4])
        t = sum(map(sum, c)) or 1
        unused = sum(1 for row in c for x in row if x == 0)
        return t10 / t, t25 / t, unused
    for src, c in sorted(by_src.items()) + [("all", allg)]:
        a, b, u = conc(c)
        lines.append(f"| {src} | {sum(map(sum, c)) // (10 * nl):,} | {u:,} of {nl * ne:,} ({u / (nl * ne):.0%}) | {a:.0%} / {b:.0%} |")
    lines.append("")

    # 2. static
    now = sum(sum(allg[l]) for l in range(nl - GPU_LAYERS, nl)) / tot
    ab, ba = coverage(gen["A"], gen["B"], budget), coverage(gen["B"], gen["A"], budget)
    fit = coverage(allg, allg, budget)
    lines += ["## 2. Static placement (same GPU memory: %d expert slots)" % budget, "",
              "| placement | share of picks served by the GPU |", "|---|---|",
              f"| today: experts of layers {nl - GPU_LAYERS}-{nl - 1} (`--n-cpu-moe {nl - GPU_LAYERS}`) | {now:.1%} |",
              f"| most-used experts, learned on half A, measured on half B | {ab:.1%} |",
              f"| most-used experts, learned on half B, measured on half A | {ba:.1%} |",
              f"| most-used experts, learned and measured on everything (optimistic) | {fit:.1%} |", ""]
    for share in (0.25, 0.5):
        k = int(budget * share)
        lines.append(f"- With only {share:.0%} of that memory ({k} slots), learned on A, measured on B: {coverage(gen['A'], gen['B'], k):.1%}")
    lines.append("")

    # 3. dynamic
    if trace_all:
        cap = budget // nl
        lines += ["## 3. Dynamic placement (per-layer cache of %d experts, updated at runtime)" % cap, "",
                  "| policy | GPU hit rate | experts moved per token | PCIe time per token to move them |", "|---|---|---|---|"]
        for label, pol, kw in [("LRU (move on miss)", "lru", {}),
                               ("decayed LFU, re-plan every token (decay 0.98)", "lfu", {"decay": 0.98, "period": 1}),
                               ("decayed LFU, re-plan every 16 tokens", "lfu", {"decay": 0.98, "period": 16}),
                               ("decayed LFU, re-plan every 128 tokens (decay 0.995)", "lfu", {"decay": 0.995, "period": 128})]:
            h, m = simulate(trace_all, nl, cap, pol, **kw)
            ms = m * EXPERT_MB / 1024 / PCIE_GBS * 1000
            lines.append(f"| {label} | {h:.1%} | {m:.1f} | {ms:.2f} ms |")
        lines += ["", f"Trace: {len(trace_all):,} (token, layer) entries. One expert = {EXPERT_MB} MB; PCIe {PCIE_GBS} GB/s host to device.", ""]

    # outputs
    png(os.path.join(HERE, "heatmap_generation.png"), nl, ne, allg, 2, 8)
    png(os.path.join(HERE, "heatmap_prompt.png"), nl, ne, prm, 2, 8)
    for src, c in by_src.items():
        png(os.path.join(HERE, f"heatmap_{src}.png"), nl, ne, c, 2, 8)
    write_merged(os.path.join(HERE, "expert-study-merged.prof"), nl, ne, allg, prm)
    lines += ["Heatmaps: heatmap_generation.png, heatmap_prompt.png, heatmap_<source>.png (rows = layers 0-47, columns = experts 0-511, "
              "brightness = picks normalized per layer). Merged counts for LlamaScope: expert-study-merged.prof."]
    open(os.path.join(HERE, "report.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))

def write_merged(path, nl, ne, gen, prm):
    """GGMLPRF2 with no timing entries, only expert counts (LlamaScope shows the heatmap)."""
    with open(path, "wb") as f:
        f.write(b"GGMLPRF2" + struct.pack("<IIQQQI", 2, 1, 3_000_000_000, 0, 0, 0))
        f.write(struct.pack("<I", 0))          # names
        f.write(struct.pack("<I", 0))          # gpu markers
        f.write(struct.pack("<II", nl, ne))
        for c in (gen, prm):
            for row in c:
                f.write(struct.pack(f"<{ne}I", *row))
        f.write(struct.pack("<I", 0))          # trace

if __name__ == "__main__":
    main()
