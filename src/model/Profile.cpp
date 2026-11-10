#include <windows.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <map>
#include "core/Util.h"
#include "model/Profile.h"

namespace {

struct Entry { uint64_t t0, t1, t2; uint32_t name; uint8_t thread, kind; uint16_t count; };
struct Marker { uint64_t tsc; uint32_t name, seq; };

enum { kOp = 1, kBlock = 2, kSched = 3, kCopy = 4 };
enum { evCopyIn = 1, evSubmit = 2, evSync = 3, evGraph = 4 };
constexpr uint32_t kGpuBegin = 1;

struct Reader {
    const std::vector<char>& b;
    size_t o = 0;
    bool bad = false;
    template <class T> T get() {
        T v{};
        if (o + sizeof(T) > b.size()) { bad = true; return v; }
        memcpy(&v, b.data() + o, sizeof(T));
        o += sizeof(T);
        return v;
    }
    bool bytes(void* dst, size_t n) {
        if (o + n > b.size()) { bad = true; return false; }
        memcpy(dst, b.data() + o, n);
        o += n;
        return true;
    }
};

std::string Prefix(const std::string& n) { return n.substr(0, n.find('-')); }
int LayerOfName(const std::string& n) {
    size_t d = n.rfind('-');
    if (d == std::string::npos) return -1;
    return atoi(n.c_str() + d + 1);
}

} // namespace

const char* ProfBlockName(ProfBlock b) {
    static const char* n[] = {"attention", "router", "experts", "always-on expert", "merge", "other"};
    return n[(int)b];
}

ProfBlock ProfBlockOf(const std::string& p) {
    auto starts = [&](const char* s) { return p.rfind(s, 0) == 0; };
    if (starts("ffn_shexp")) return ProfBlock::Shared;
    if (p == "ffn_moe_probs" || p == "ffn_moe_argsort" || p == "ffn_moe_topk" || p == "ffn_moe_weights" ||
        starts("ffn_moe_logits") || starts("ffn_moe_weights"))
        return ProfBlock::Router;
    if (starts("ffn_moe")) return ProfBlock::Experts;
    if (starts("attn") || starts("linear_attn") || starts("q_") || starts("k_") || starts("v_") || starts("kq") ||
        starts("Qcur") || starts("Kcur") || starts("Vcur"))
        return ProfBlock::Attention;
    if (p == "ffn_out" || p == "post_moe" || p == "l_out" || starts("ffn_inp")) return ProfBlock::Merge;
    return ProfBlock::Other;
}

ProfileData ProfileData::Load(const std::wstring& path) {
    ProfileData d;
    d.path = ToUtf8(path);
    std::ifstream f(path, std::ios::binary);
    if (!f) { d.error = "cannot open file"; return d; }
    std::vector<char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    Reader r{buf};
    char magic[8];
    if (!r.bytes(magic, 8) || (memcmp(magic, "GGMLPRF2", 8) != 0 && memcmp(magic, "GGMLPRF3", 8) != 0)) {
        d.error = "not a ggml profile (GGMLPRF2/3)";
        return d;
    }
    const bool v3 = magic[7] == '3';
    r.get<uint32_t>();                       // version
    d.level = (int)r.get<uint32_t>();
    const uint64_t hz = r.get<uint64_t>();
    r.get<uint64_t>();                       // tsc0
    d.dropped = r.get<uint64_t>();
    const uint32_t nSlots = r.get<uint32_t>();
    std::vector<Entry> entries;
    for (uint32_t s = 0; s < nSlots && !r.bad; ++s) {
        const uint32_t n = r.get<uint32_t>();
        if ((uint64_t)n * 32 > buf.size()) { d.error = "corrupt slot"; return d; }
        size_t at = entries.size();
        entries.resize(at + n);
        for (uint32_t i = 0; i < n; ++i) {
            Entry& e = entries[at + i];
            e.t0 = r.get<uint64_t>(); e.t1 = r.get<uint64_t>(); e.t2 = r.get<uint64_t>();
            e.name = r.get<uint32_t>(); e.thread = r.get<uint8_t>(); e.kind = r.get<uint8_t>(); e.count = r.get<uint16_t>();
        }
    }
    std::map<uint32_t, std::string> names;
    std::map<uint32_t, uint64_t> tensorBytes;
    const uint32_t nNames = r.get<uint32_t>();
    for (uint32_t i = 0; i < nNames && !r.bad; ++i) {
        const uint32_t h = r.get<uint32_t>();
        const uint16_t len = r.get<uint16_t>();
        std::string s(len, '\0');
        r.bytes(s.data(), len);
        names[h] = s;
        if (v3) tensorBytes[h] = r.get<uint64_t>();
    }
    std::vector<Marker> markers(r.get<uint32_t>());
    if (markers.size() * 16 > buf.size()) { d.error = "corrupt GPU section"; return d; }
    for (auto& m : markers) { m.tsc = r.get<uint64_t>(); m.name = r.get<uint32_t>(); m.seq = r.get<uint32_t>(); }
    const uint32_t nl = r.get<uint32_t>(), ne = r.get<uint32_t>();
    if ((uint64_t)nl * ne * 8 > buf.size() || nl > 4096 || ne > 65536) { d.error = "corrupt expert section"; return d; }
    d.nExperts = (int)ne;
    for (auto* counts : {&d.genCounts, &d.promptCounts}) {
        counts->assign(nl, std::vector<uint32_t>(ne));
        for (uint32_t l = 0; l < nl; ++l) r.bytes(counts->at(l).data(), ne * 4);
    }
    if (r.bad || hz == 0) { d.error = "truncated file"; return d; }
    d.tscHz = (double)hz;
    const double us = 1e6 / d.tscHz;

    // Graph windows from the scheduler's graph events (slot 0, chronological).
    struct Win { uint64_t t0, t1; int64_t tokens; };
    std::vector<Win> wins;
    int cpuBackend = 0;
    for (const auto& e : entries)
        if (e.kind == kSched) cpuBackend = std::max(cpuBackend, (int)(e.name == evGraph ? 0 : e.t2));
    for (const auto& e : entries)
        if (e.kind == kSched && e.name == evGraph) wins.push_back({e.t0, UINT64_MAX, (int64_t)e.t2});
    std::sort(wins.begin(), wins.end(), [](const Win& a, const Win& b) { return a.t0 < b.t0; });
    for (size_t i = 0; i + 1 < wins.size(); ++i) wins[i].t1 = wins[i + 1].t0;
    auto winOf = [&](uint64_t t) -> int {
        auto it = std::upper_bound(wins.begin(), wins.end(), t, [](uint64_t v, const Win& w) { return v < w.t0; });
        return it == wins.begin() ? -1 : (int)(it - wins.begin()) - 1;
    };

    // The last generation step is incomplete (no following graph); the first ones include warm-up and graph capture.
    std::vector<char> gen(wins.size(), 0);
    int seen = 0;
    for (size_t i = 0; i + 1 < wins.size(); ++i) {
        if (wins[i].tokens == 1) {
            if (seen++ >= 2) gen[i] = 1;
        } else {
            d.promptGraphs++;
        }
    }
    for (char g : gen) d.genSteps += g;
    if (!d.genSteps) {
        // counts-only profiles (e.g. merged expert-usage studies) still give the expert heatmap and strips
        if (nl == 0) { d.error = "no generation steps in the profile"; return d; }
        d.ok = true;
        return d;
    }

    int maxLayer = (int)nl - 1;
    for (const auto& [h, n] : names) maxLayer = std::max(maxLayer, LayerOfName(n));
    d.layers.assign(maxLayer + 1, {});
    std::vector<std::array<double, (size_t)ProfBlock::Count>> waitSum(d.layers.size()), totSum(d.layers.size());
    int threads = 0;

    for (const auto& e : entries) {
        if (e.kind != kOp && e.kind != kBlock) continue;
        threads = std::max(threads, (int)e.thread + 1);
        int w = winOf(e.t0);
        if (w < 0 || !gen[w]) continue;
        auto it = names.find(e.name);
        if (it == names.end()) continue;
        int L = LayerOfName(it->second);
        if (L < 0 || L >= (int)d.layers.size()) continue;
        size_t b = (size_t)ProfBlockOf(Prefix(it->second));
        const double comp = (e.kind == kOp ? e.t1 - e.t0 : e.t1) * us, wait = (e.kind == kOp ? e.t2 - e.t1 : e.t2) * us;
        if (e.thread == 0) {
            d.layers[L].cpuUs[b] += comp + wait;
            d.cpuBusyUs += comp + wait;
        }
        waitSum[L][b] += wait;
        totSum[L][b] += comp + wait;
        d.layers[L].cpu = true;
    }
    d.threads = threads;

    // GPU block times: gap from the previous marker of the same split (a BEGIN marker starts each split).
    std::sort(markers.begin(), markers.end(), [](const Marker& a, const Marker& b) { return a.tsc < b.tsc; });
    d.hasGpu = !markers.empty();
    std::vector<uint64_t> gpuEnd(d.layers.size(), 0);   // last GPU marker per layer (current step), for transfers
    for (size_t i = 1; i < markers.size(); ++i) {
        const Marker& m = markers[i];
        if (m.name == kGpuBegin) continue;
        int w = winOf(m.tsc);
        if (w < 0 || !gen[w]) continue;
        auto it = names.find(m.name);
        if (it == names.end()) continue;
        int L = LayerOfName(it->second);
        if (L < 0 || L >= (int)d.layers.size()) continue;
        const double dur = (m.tsc - markers[i - 1].tsc) * us;
        d.layers[L].gpuUs[(size_t)ProfBlockOf(Prefix(it->second))] += dur;
        d.layers[L].gpu = true;
        d.gpuBusyUs += dur;
    }

    // Transfers: copy-in of a CPU split -> data arriving from the GPU (layer of the next CPU entry);
    // copy-in of a GPU split right after CPU work -> data going back (layer of the previous CPU entry).
    std::vector<const Entry*> seq;
    for (const auto& e : entries)
        if (e.thread == 0 && (e.kind == kSched || e.kind == kOp || e.kind == kBlock)) seq.push_back(&e);
    std::stable_sort(seq.begin(), seq.end(), [](const Entry* a, const Entry* b) { return a->t0 < b->t0; });
    auto layerOf = [&](const Entry* e) {
        auto it = names.find(e->name);
        return it == names.end() ? -1 : LayerOfName(it->second);
    };
    int lastCpuLayer = -1;
    for (size_t i = 0; i < seq.size(); ++i) {
        const Entry* e = seq[i];
        if (e->kind != kSched) { lastCpuLayer = layerOf(e); continue; }
        int w = winOf(e->t0);
        if (w < 0 || !gen[w]) continue;
        const double dur = (e->t1 - e->t0) * us;
        if (e->name == evSync) d.syncUs += dur;
        if (e->name != evCopyIn) continue;
        d.copyUs += dur;
        if ((int)e->t2 == cpuBackend) {
            int L = -1;
            for (size_t j = i + 1; j < seq.size() && L < 0; ++j)
                if (seq[j]->kind != kSched) L = layerOf(seq[j]);
            if (L < 0 || L >= (int)d.layers.size()) continue;
            // Remove the part spent waiting for the GPU to finish: start counting at the last GPU marker before the end.
            uint64_t start = e->t0;
            if (d.hasGpu) {
                auto it = std::upper_bound(markers.begin(), markers.end(), e->t1,
                                           [](uint64_t v, const Marker& m) { return v < m.tsc; });
                if (it != markers.begin()) start = std::max(start, std::prev(it)->tsc);
            }
            d.layers[L].toCpuUs += (e->t1 - start) * us;
        } else if (lastCpuLayer >= 0 && lastCpuLayer < (int)d.layers.size()) {
            d.layers[lastCpuLayer].toGpuUs += dur;
        }
    }

    // Per-input copies (v3): which backend each split runs on comes from the submit events.
    std::map<int, int> splitBackend;
    for (const auto& e : entries)
        if (e.kind == kSched && e.name == evSubmit) splitBackend[e.count] = (int)e.t2;
    struct Acc { double n = 0, sync = 0, copy = 0; int path = 0; uint64_t bytes = 0; bool toCpu = false; };
    std::map<std::string, Acc> acc;
    for (const auto& e : entries) {
        if (e.kind != kCopy) continue;
        int w = winOf(e.t0);
        if (w < 0 || !gen[w]) continue;
        auto it = names.find(e.name);
        std::string t = it == names.end() ? "?" : Prefix(it->second);
        auto sb = splitBackend.find(e.count);
        bool toCpu = sb != splitBackend.end() && sb->second == cpuBackend;
        Acc& a = acc[t + (toCpu ? "|c" : "|g")];
        a.n += 1;
        a.sync += (e.t1 - e.t0) * us;
        a.copy += (e.t2 - e.t1) * us;
        a.path = e.thread;
        a.toCpu = toCpu;
        auto tb = tensorBytes.find(e.name);
        if (tb != tensorBytes.end()) a.bytes = tb->second;
    }
    // Direct per-layer hand-offs from the copy entries (v3): no GPU clock conversion involved.
    // In each split, the first input's synchronize waits for the GPU to finish its work: that part is GPU wait.
    bool directCopies = false;
    {
        struct SplitKey { int w, split; bool operator<(const SplitKey& o) const { return w != o.w ? w < o.w : split < o.split; } };
        std::map<SplitKey, std::vector<const Entry*>> bySplit;
        for (const auto& e : entries) {
            if (e.kind != kCopy) continue;
            int w = winOf(e.t0);
            if (w < 0 || !gen[w]) continue;
            bySplit[{w, (int)e.count}].push_back(&e);
        }
        if (!bySplit.empty()) {
            directCopies = true;
            for (auto& L : d.layers) L.toCpuUs = L.toGpuUs = 0;
            d.waitGpuUs = 0;
            for (auto& [key, list] : bySplit) {
                std::sort(list.begin(), list.end(), [](const Entry* x, const Entry* y) { return x->t0 < y->t0; });
                auto sb = splitBackend.find(key.split);
                const bool toCpu = sb != splitBackend.end() && sb->second == cpuBackend;
                for (size_t i = 0; i < list.size(); ++i) {
                    const Entry* e = list[i];
                    auto it = names.find(e->name);
                    int L = it == names.end() ? -1 : LayerOfName(it->second);
                    double wait = (i == 0 && toCpu) ? (e->t1 - e->t0) * us : 0;
                    double handoff = (e->t2 - e->t0) * us - wait;
                    d.waitGpuUs += wait;
                    if (L < 0 || L >= (int)d.layers.size()) { d.otherCopyUs += handoff; continue; }
                    (toCpu ? d.layers[L].toCpuUs : d.layers[L].toGpuUs) += handoff;
                }
            }
        }
    }

    for (const auto& [k, a] : acc) {
        Copy c;
        c.tensor = k.substr(0, k.size() - 2);
        c.toCpu = a.toCpu;
        c.path = a.path;
        c.perStep = a.n / d.genSteps;
        c.bytes = a.bytes;
        c.syncUs = a.sync / a.n;
        c.copyUs = a.copy / a.n;
        d.copies.push_back(c);
    }
    std::sort(d.copies.begin(), d.copies.end(),
              [](const Copy& x, const Copy& y) { return x.perStep * (x.syncUs + x.copyUs) > y.perStep * (y.syncUs + y.copyUs); });

    // Averages per generation step.
    const double n = d.genSteps;
    for (size_t L = 0; L < d.layers.size(); ++L) {
        auto& ly = d.layers[L];
        for (size_t b = 0; b < (size_t)ProfBlock::Count; ++b) {
            ly.cpuUs[b] /= n;
            ly.gpuUs[b] /= n;
            ly.spinPct[b] = totSum[L][b] > 0 ? waitSum[L][b] / totSum[L][b] * 100 : 0;
        }
        ly.toCpuUs /= n;
        ly.toGpuUs /= n;
    }
    d.cpuBusyUs /= n; d.gpuBusyUs /= n; d.copyUs /= n; d.syncUs /= n;
    for (const auto& ly : d.layers) d.handoffUs += ly.toCpuUs + ly.toGpuUs;
    if (directCopies) {
        d.waitGpuUs /= n;
        d.otherCopyUs /= n;
        d.handoffUs += d.otherCopyUs;   // per-step inputs without a layer (token, positions, masks)
    } else {
        d.waitGpuUs = std::max(0.0, d.copyUs - d.handoffUs);
    }
    double span = 0;
    for (size_t i = 0; i < wins.size(); ++i)
        if (gen[i]) span += (wins[i].t1 - wins[i].t0) * us;
    d.stepUs = span / n;
    d.ok = true;
    return d;
}
