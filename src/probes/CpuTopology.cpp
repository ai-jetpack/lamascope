#include <windows.h>
#include <powrprof.h>
#include <intrin.h>
#include <algorithm>
#include <cstring>
#include "core/Util.h"
#include "probes/CpuTopology.h"

namespace {

// Documented layout, not declared in the SDK headers.
struct ProcessorPowerInformation {
    ULONG Number, MaxMhz, CurrentMhz, MhzLimit, MaxIdleState, CurrentIdleState;
};

std::string CpuBrand() {
    int r[4];
    char b[49] = {};
    __cpuid(r, 0x80000000);
    if ((unsigned)r[0] < 0x80000004) return "Unknown CPU";
    for (int i = 0; i < 3; ++i) {
        __cpuid(r, 0x80000002 + i);
        memcpy(b + 16 * i, r, 16);
    }
    return Trim(b);
}

// Max turbo (P-core, E-core) in MHz for common hybrid desktop SKUs. F variants share the K/non-K values.
void SpecTurbo(const std::string& brand, int& p, int& e) {
    struct Sku { const char* model; int p, e; };
    static const Sku skus[] = {
        {"12900KS", 5500, 4000}, {"12900K", 5200, 3900}, {"12700K", 5000, 3800}, {"12600K", 4900, 3600},
        {"13900KS", 6000, 4300}, {"13900K", 5800, 4300}, {"13700K", 5400, 4200}, {"13600K", 5100, 3900},
        {"14900KS", 6200, 4500}, {"14900K", 6000, 4400}, {"14700K", 5600, 4300}, {"14600K", 5300, 4000},
    };
    for (const auto& s : skus)
        if (brand.find(s.model) != std::string::npos) { p = s.p; e = s.e; return; }
}

std::vector<int> MaskToLogical(const GROUP_AFFINITY& g) {
    std::vector<int> out;
    for (int b = 0; b < 64; ++b)
        if (g.Mask & (KAFFINITY(1) << b)) out.push_back(g.Group * 64 + b);
    return out;
}

} // namespace

CpuTopology CpuTopology::Probe() {
    CpuTopology t;
    t.brand = CpuBrand();

    int r[4];
    __cpuid(r, 0);
    int maxLeaf = r[0];
    __cpuid(r, 1);
    t.fma = (r[2] >> 12) & 1;
    t.avx = (r[2] >> 28) & 1;
    if (maxLeaf >= 7) {
        __cpuidex(r, 7, 0);
        t.avx2 = (r[1] >> 5) & 1;
        t.avx512f = (r[1] >> 16) & 1;
        __cpuidex(r, 7, 1);
        t.avxVnni = (r[0] >> 4) & 1;
    }
    if (maxLeaf >= 0x16) {
        __cpuid(r, 0x16);
        t.cpuidBaseMhz = r[0] & 0xFFFF;
        t.cpuidMaxMhz = r[1] & 0xFFFF;
    }
    SpecTurbo(t.brand, t.specPMaxMhz, t.specEMaxMhz);
    if (!t.specPMaxMhz) t.specPMaxMhz = t.cpuidMaxMhz;

    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationAll, nullptr, &len);
    std::vector<uint8_t> buf(len);
    if (!GetLogicalProcessorInformationEx(RelationAll, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buf.data(), &len))
        return t;

    struct RawCore { std::vector<int> lps; int eff; };
    struct RawCache { int level; PROCESSOR_CACHE_TYPE type; uint64_t size; std::vector<int> lps; };
    std::vector<RawCore> raw;
    std::vector<RawCache> caches;

    for (uint8_t* p = buf.data(); p < buf.data() + len;) {
        auto* e = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)p;
        if (e->Relationship == RelationProcessorCore) {
            RawCore c{{}, e->Processor.EfficiencyClass};
            for (WORD g = 0; g < e->Processor.GroupCount; ++g) {
                auto l = MaskToLogical(e->Processor.GroupMask[g]);
                c.lps.insert(c.lps.end(), l.begin(), l.end());
            }
            raw.push_back(std::move(c));
        } else if (e->Relationship == RelationCache) {
            caches.push_back({e->Cache.Level, e->Cache.Type, e->Cache.CacheSize, MaskToLogical(e->Cache.GroupMask)});
        }
        p += e->Size;
    }

    int maxEff = 0, minEff = 255;
    for (auto& c : raw) { maxEff = std::max(maxEff, c.eff); minEff = std::min(minEff, c.eff); }
    t.hybrid = maxEff != minEff;

    std::sort(raw.begin(), raw.end(), [&](const RawCore& a, const RawCore& b) {
        bool pa = a.eff == maxEff, pb = b.eff == maxEff;
        if (pa != pb) return pa;
        return a.lps.front() < b.lps.front();
    });

    int maxLp = 0;
    for (auto& c : raw)
        for (int lp : c.lps) maxLp = std::max(maxLp, lp);
    t.logicalCount = maxLp + 1;
    t.logicalToCore.assign(t.logicalCount, -1);

    for (auto& c : raw) {
        CoreInfo ci;
        ci.pcore = c.eff == maxEff;
        ci.ordinal = ci.pcore ? t.pCores++ : t.eCores++;
        ci.logical = c.lps;
        if (c.lps.size() > 1) t.smt = true;
        for (int lp : c.lps) t.logicalToCore[lp] = (int)t.cores.size();
        t.cores.push_back(std::move(ci));
    }

    for (auto& c : caches) {
        if (c.type != CacheData && c.type != CacheUnified) continue;
        bool coversP = !c.lps.empty() && t.IsPCoreLogical(c.lps.front());
        if (c.level == 1 && coversP) t.l1dPerPCore = c.size;
        if (c.level == 2) (coversP ? t.l2PerPCore : t.l2PerECluster) = c.size;
        if (c.level == 3) t.l3Total = c.size;
    }

    std::vector<ProcessorPowerInformation> ppi(t.logicalCount);
    t.baseMhz.assign(t.logicalCount, 0);
    if (CallNtPowerInformation(ProcessorInformation, nullptr, 0, ppi.data(),
                               (ULONG)(ppi.size() * sizeof(ProcessorPowerInformation))) == 0) {
        for (auto& p : ppi)
            if ((int)p.Number < t.logicalCount) t.baseMhz[p.Number] = (int)p.MaxMhz;
    }
    return t;
}
