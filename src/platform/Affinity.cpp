#include <windows.h>
#include <bit>
#include <format>
#include "core/Log.h"
#include "platform/Affinity.h"

uint64_t Affinity::ECoreMask(const CpuTopology& cpu) {
    if (!cpu.hybrid) return 0;
    uint64_t m = 0;
    for (const auto& c : cpu.cores)
        if (!c.pcore)
            for (int lp : c.logical)
                if (lp < 64) m |= 1ull << lp;
    return m;
}

uint64_t Affinity::PCoreMask(const CpuTopology& cpu) {
    uint64_t m = 0;
    for (const auto& c : cpu.cores)
        if (c.pcore || !cpu.hybrid)
            for (int lp : c.logical)
                if (lp < 64) m |= 1ull << lp;
    return m;
}

uint64_t Affinity::AllCoreMask() {
    DWORD_PTR proc = 0, sys = 0;
    GetProcessAffinityMask(GetCurrentProcess(), &proc, &sys);
    return sys;
}

uint64_t Affinity::CurrentMask() {
    DWORD_PTR proc = 0, sys = 0;
    GetProcessAffinityMask(GetCurrentProcess(), &proc, &sys);
    return proc;
}

bool Affinity::SetMask(uint64_t mask) {
    return mask && SetProcessAffinityMask(GetCurrentProcess(), (DWORD_PTR)mask) != FALSE;
}

bool Affinity::RestrictToECores(const CpuTopology& cpu) {
    uint64_t m = ECoreMask(cpu) & AllCoreMask();
    if (!m) return false;
    bool ok = SetMask(m);
    Log::Write(ok ? Log::Info : Log::Warn,
               ok ? "LlamaScope restricted to " + Describe(cpu) : "Could not restrict LlamaScope to E-cores");
    return ok;
}

std::string Affinity::Describe(const CpuTopology& cpu) {
    uint64_t cur = CurrentMask();
    uint64_t e = ECoreMask(cpu);
    if (e && cur == (e & AllCoreMask())) {
        int first = std::countr_zero(e), last = 63 - std::countl_zero(e);
        return std::format("E-cores only ({} cores, CPUs {}-{})", cpu.eCores, first, last);
    }
    if (cur == AllCoreMask()) return "all cores";
    return std::format("custom mask 0x{:X}", cur);
}

Affinity::ScopedAllCores::ScopedAllCores() : saved_(CurrentMask()) {
    SetMask(AllCoreMask());
}

Affinity::ScopedAllCores::~ScopedAllCores() {
    SetMask(saved_);
}
