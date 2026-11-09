#pragma once
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// Every 2 s: all processes with CPU usage (% of one core) and affinity mask.
struct ProcInfo {
    unsigned pid = 0;
    std::string name;
    uint64_t createTime = 0;     // FILETIME, identifies the process across pid reuse
    double cpuPct = 0;           // % of one logical CPU over the last interval
    uint64_t affinity = 0;       // 0 = unknown (access denied)
    bool IsLlama() const { return name.rfind("llama", 0) == 0; }
};

class ProcessMonitor {
public:
    ~ProcessMonitor() { Stop(); }
    void Start();
    void Stop();
    std::vector<ProcInfo> Snapshot() const { std::lock_guard lk(mu_); return procs_; }

private:
    void Loop();
    std::thread thread_;
    std::atomic<bool> stop_{false};
    mutable std::mutex mu_;
    std::vector<ProcInfo> procs_;
};
