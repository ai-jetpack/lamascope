#pragma once
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include "core/TimeSeries.h"

struct GpuStatic {
    bool ok = false;
    std::string error;
    std::string name, driver;
    int cudaDriverVersion = 0;           // e.g. 13050
    uint64_t vramTotal = 0;
    uint64_t bar1Total = 0;
    unsigned maxGraphicsMHz = 0, maxMemMHz = 0;
    unsigned powerLimitmW = 0;
    unsigned memBusWidth = 0;                // bits
    double VramGBs() const { return memBusWidth / 8.0 * maxMemMHz * 2.0 / 1000.0; }   // DDR: 2 transfers/clock
    int linkMaxGen = 0, linkMaxWidth = 0;    // as reported by NVML (device+system)
    int gpuMaxGen = 0;                       // GPU capability
    int cardWidth = 0;                       // electrical width of the card (override table), 0 = unknown
    std::string cardWidthNote;
};

struct GpuLive {
    bool valid = false;
    int linkGen = 0, linkWidth = 0;
    unsigned util = 0, memUtil = 0;
    unsigned gfxMHz = 0, memMHz = 0;
    unsigned powermW = 0, tempC = 0;
    uint64_t throttle = 0;
    uint64_t vramUsed = 0;
    unsigned rxKBs = 0, txKBs = 0;
};

class NvmlCollector {
public:
    ~NvmlCollector() { Stop(); }
    bool Start();
    void Stop();
    GpuStatic Static() const { std::lock_guard lk(mu_); return static_; }
    GpuLive Live() const { std::lock_guard lk(mu_); return live_; }

    // Highest link observed (links drop to Gen1 at idle and retrain under load).
    int MaxGenSeen() const { return maxGenSeen_; }
    int MaxWidthSeen() const { return maxWidthSeen_; }
    unsigned MaxUtilSeen() const { return maxUtilSeen_; }
    void ResetPeaks() { maxGenSeen_ = 0; maxWidthSeen_ = 0; maxUtilSeen_ = 0; }

    Series util, memUtil, gfxMHz, memMHz, powerW, tempC, rxMBs, txMBs, vramGB;

private:
    void Loop();
    std::thread thread_;
    std::atomic<bool> stop_{false};
    mutable std::mutex mu_;
    GpuStatic static_;
    GpuLive live_;
    std::atomic<int> maxGenSeen_{0}, maxWidthSeen_{0};
    std::atomic<unsigned> maxUtilSeen_{0};
};

// Decodes NVML clock-event (throttle) reason bits into names.
std::string ThrottleReasonsText(uint64_t bits, bool* concerning);
