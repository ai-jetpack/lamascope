#pragma once
#include <cstdint>
#include <string>
#include <vector>

struct CoreInfo {
    bool pcore = true;              // false = efficiency core
    int ordinal = 0;                // P0..Pn / E0..En
    std::vector<int> logical;       // logical processor indices (SMT siblings)
    std::string Label() const { return (pcore ? "P" : "E") + std::to_string(ordinal); }
};

struct CpuTopology {
    std::string brand;
    int logicalCount = 0;
    int pCores = 0, eCores = 0;
    bool hybrid = false, smt = false;
    std::vector<CoreInfo> cores;         // P-cores first, then E-cores, each ordered by logical index
    std::vector<int> logicalToCore;      // logical index -> index in cores
    std::vector<int> baseMhz;            // per logical processor (CallNtPowerInformation MaxMhz)
    int cpuidBaseMhz = 0, cpuidMaxMhz = 0;   // CPUID leaf 0x16 (Intel): nominal and max turbo (often masked)
    int specPMaxMhz = 0, specEMaxMhz = 0;    // max turbo per core type: CPUID or known-SKU table, 0 = unknown
    // Upper bound for a plausible effective-clock sample on this core type.
    float ClockCap(bool pcore) const {
        int m = pcore ? specPMaxMhz : (specEMaxMhz ? specEMaxMhz : specPMaxMhz);
        return (m > 0 ? m : 6500) * 1.05f;
    }

    uint64_t l1dPerPCore = 0, l2PerPCore = 0, l2PerECluster = 0, l3Total = 0;
    bool avx = false, avx2 = false, fma = false, avxVnni = false, avx512f = false;

    bool IsPCoreLogical(int lp) const {
        return lp >= 0 && lp < (int)logicalToCore.size() && logicalToCore[lp] >= 0 && cores[logicalToCore[lp]].pcore;
    }
    static CpuTopology Probe();
};
