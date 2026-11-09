#pragma once
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "core/TimeSeries.h"
#include "probes/CpuTopology.h"

// Per-logical-CPU utilization and effective-clock ratio, plus GPU adapter memory, via PDH.
class PdhCollector {
public:
    static constexpr int kHistory = 240;   // 60 s at 250 ms
    ~PdhCollector() { Stop(); }
    bool noGpuCounters = false;   // diagnostic switch (--no-pdh-gpu): skip the GPU memory counters
    bool Start(const CpuTopology& cpu);

    // Effective clock history in MHz. NaN = core parked for the whole sample (drawn as a gap).
    std::vector<std::unique_ptr<Series>> coreMhz;   // per physical core (max of SMT siblings), index = cpu.cores
    Series pAvgMhz, pMaxMhz, eAvgMhz;               // over active cores of each type
    void Stop();

    std::vector<float> CoreUtil() const { std::lock_guard lk(mu_); return util_; }   // 0..100
    std::vector<float> CorePerf() const { std::lock_guard lk(mu_); return perf_; }   // % of base clock
    // Row-major [logical][kHistory], oldest sample first.
    std::vector<float> UtilHistory() const;

    Series gpuDedicatedGB, gpuSharedGB;

    // Per-process dedicated VRAM on one adapter (the inference GPU).
    struct ProcVram { unsigned pid; std::string name; uint64_t bytes; };
    void SetGpuLuid(const std::wstring& luidTag) { std::lock_guard lk(mu_); luid_ = luidTag; }
    std::vector<ProcVram> GpuProcesses() const { std::lock_guard lk(mu_); return procs_; }

private:
    void UpdateGpuProcesses();
    void* cProcDed_ = nullptr;
    std::wstring luid_;
    std::vector<ProcVram> procs_;
    int tick_ = 0;
    void Loop();
    int n_ = 0;
    CpuTopology cpu_;
    void* query_ = nullptr;
    void* cTime_ = nullptr;
    void* cPerf_ = nullptr;
    void* cDed_ = nullptr;
    void* cShared_ = nullptr;
    std::thread thread_;
    std::atomic<bool> stop_{false};
    mutable std::mutex mu_;
    std::vector<float> util_, perf_;
    std::vector<std::vector<float>> hist_;  // [sample][logical] ring
    size_t histHead_ = 0;
};
