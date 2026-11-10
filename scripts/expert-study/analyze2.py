"""Follow-up questions on the study data:
  1. Never/rarely used experts in the GPU layers (36-47): VRAM they hold, extra layers that would free.
  2. With a static hot set (learned on half A, applied to half B's trace): share of (token, layer) steps whose picks are
     all hot (no CPU work, no crossing) and the number of crossings per token.
"""
import glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze

HERE = os.path.dirname(os.path.abspath(__file__))
files = sorted(glob.glob(os.path.join(HERE, "profiles", "*.prof")))
data = {os.path.basename(f)[:-5]: analyze.read(f) for f in files}
nl, ne = next(iter(data.values()))[:2]
gen = {"A": [[0] * ne for _ in range(nl)], "B": [[0] * ne for _ in range(nl)]}
trace = {"A": [], "B": []}
for name, (_, _, g, p, t) in data.items():
    h = name.split("-")[1]
    gen[h] = analyze.add(gen[h], g)
    trace[h] += t
allg = analyze.add(gen["A"], gen["B"])
tot = sum(map(sum, allg))
LAYER_MB = 0.0
EXPERT_MB = analyze.EXPERT_MB
print(f"one expert {EXPERT_MB} MB, one layer of experts {EXPERT_MB * ne:.0f} MB")

# 1. GPU layers 36..47
print("\n1. GPU layers 36-47 (today on the GPU):")
for thr_label, thr in [("never picked", 0), ("<= 0.01% of the layer's picks", 1e-4), ("<= 0.05%", 5e-4), ("<= 0.1%", 1e-3)]:
    n = 0
    picks_lost = 0
    for l in range(nl - analyze.GPU_LAYERS, nl):
        lt = sum(allg[l])
        for c in allg[l]:
            if (c == 0 if thr == 0 else c / lt <= thr):
                n += 1
                picks_lost += c
    mb = n * EXPERT_MB
    print(f"   {thr_label:32s}: {n:5d} experts, {mb:7.0f} MB = {mb / (EXPERT_MB * ne):.2f} layers of experts; "
          f"those experts served {picks_lost / tot:.4%} of all picks")

# 2. static hot set learned on A, applied to B's trace (and reverse)
budget = analyze.GPU_LAYERS * ne
for learn, test in (("A", "B"), ("B", "A")):
    flat = sorted(((c, l, e) for l, row in enumerate(gen[learn]) for e, c in enumerate(row)), reverse=True)
    hot = set((l, e) for _, l, e in flat[:budget])
    steps = allhot = hits = picks = 0
    per_layer_slots = [0] * nl
    for l, e in hot:
        per_layer_slots[l] += 1
    for layer, ids in trace[test]:
        steps += 1
        h = sum(1 for e in ids if (layer, e) in hot)
        hits += h
        picks += len(ids)
        allhot += h == len(ids)
    tokens = steps / nl
    print(f"\n2. hot set learned on {learn}, measured on {test}: hit rate {hits / picks:.1%}; (token, layer) steps with all picks hot: "
          f"{allhot / steps:.2%}; crossings per token if each step with a cold pick needs the CPU: {(steps - allhot) / tokens:.1f} (today 36)")
    print(f"   hot slots per layer: min {min(per_layer_slots)}, max {max(per_layer_slots)}, layers 0-11: {sum(per_layer_slots[:12])}, "
          f"36-47: {sum(per_layer_slots[36:])}")

# sorted heatmap: each layer's experts in decreasing usage (left = most used), linear scale per layer
srt = [sorted(row, reverse=True) for row in allg]
analyze.png(os.path.join(HERE, "heatmap_generation_sorted.png"), nl, ne, [[(v / (r[0] or 1)) ** 2 * (r[0] or 1) for v in r] for r in srt], 2, 8)
print("\nwrote heatmap_generation_sorted.png")
