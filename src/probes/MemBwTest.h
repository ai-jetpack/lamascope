#pragma once
#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "probes/CpuTopology.h"

// MLC-style "all reads" bandwidth test: one pinned thread per physical core,
// streaming AVX2 loads over a 2 GiB buffer.
struct MemBwResult {
    bool valid = false;
    double bestGBs = 0, medianGBs = 0;
    int threads = 0;
    std::string error;
};

class MemBwTest {
public:
    ~MemBwTest();
    void Start(const CpuTopology& cpu);
    bool Running() const { return running_; }
    float Progress() const { return progress_; }
    MemBwResult Result() const { std::lock_guard lk(mu_); return result_; }

private:
    void Run(std::vector<CoreInfo> cores, bool avx2);
    std::thread worker_;
    std::atomic<bool> running_{false};
    std::atomic<float> progress_{0};
    mutable std::mutex mu_;
    MemBwResult result_;
};
