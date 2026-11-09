#pragma once
#include <cstdint>
#include <string>
#include "probes/CpuTopology.h"

// Keeps LlamaScope off the P-cores used by inference.
// Affinity is inherited by child processes: anything we launch must set its own mask.
namespace Affinity {
uint64_t ECoreMask(const CpuTopology& cpu);   // group 0 only; 0 if not hybrid
uint64_t PCoreMask(const CpuTopology& cpu);   // group 0 only; all cores if not hybrid
uint64_t AllCoreMask();
// Restricts this process to E-cores. Returns false on non-hybrid CPUs or failure.
bool RestrictToECores(const CpuTopology& cpu);
uint64_t CurrentMask();
bool SetMask(uint64_t mask);
std::string Describe(const CpuTopology& cpu);   // e.g. "E-cores only (8 cores, CPUs 16-23)"

// RAII: temporarily allow all cores (bandwidth tests), restore on scope exit.
class ScopedAllCores {
public:
    ScopedAllCores();
    ~ScopedAllCores();
private:
    uint64_t saved_ = 0;
};
}
