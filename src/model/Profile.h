#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>

// Reader for the ggml in-process profiler output (GGMLPRF2, see docs/PROFILER.md) and per-layer aggregation for the
// topology diagram. Times are microseconds averaged over generation steps (graphs with one token).
enum class ProfBlock { Attention, Router, Experts, Shared, Merge, Other, Count };
const char* ProfBlockName(ProfBlock b);
ProfBlock ProfBlockOf(const std::string& tensorPrefix);

struct ProfileData {
    bool ok = false;
    std::string error, path;
    int level = 0;                    // 1 block, 2 op
    double tscHz = 0;
    uint64_t dropped = 0;
    int threads = 0;
    int genSteps = 0, promptGraphs = 0;
    double stepUs = 0;                // wall time per generation step (graph start to graph start)

    struct Layer {
        std::array<double, (size_t)ProfBlock::Count> cpuUs{};    // wall time on the CPU (thread 0: compute + wait)
        std::array<double, (size_t)ProfBlock::Count> spinPct{};  // share of thread time spent waiting at barriers
        std::array<double, (size_t)ProfBlock::Count> gpuUs{};    // GPU time from marker kernels
        double toCpuUs = 0;           // GPU result -> CPU: syncs + copies, without the wait for the GPU to finish
        double toGpuUs = 0;           // CPU result -> GPU
        bool cpu = false, gpu = false;
    };
    std::vector<Layer> layers;

    double cpuBusyUs = 0, gpuBusyUs = 0, copyUs = 0, syncUs = 0;   // per step
    double handoffUs = 0;             // PCIe hand-offs: copies after the GPU finished (sum of the layers' toCpu + toGpu)
    double waitGpuUs = 0;             // main thread blocked in copies until the GPU finished (contained in gpuBusyUs)
    double otherCopyUs = 0;           // copies of per-step inputs that belong to no layer (token, positions, masks)

    // Per input tensor copied between devices (format v3): averaged over generation steps.
    struct Copy {
        std::string tensor;           // name without the layer suffix
        bool toCpu = false;
        int path = 0;                 // 0 async, 1 sync fallback, 2 user input, 3 callback
        double perStep = 0;           // copies per generated token
        uint64_t bytes = 0;           // per copy
        double syncUs = 0, copyUs = 0;   // per copy: synchronize part, data copy part
    };
    std::vector<Copy> copies;
    bool hasGpu = false;

    // Expert usage: [layer][expert] selection counts.
    int nExperts = 0;
    std::vector<std::vector<uint32_t>> genCounts, promptCounts;

    static ProfileData Load(const std::wstring& path);
};
