#pragma once
#include <atomic>
#include <mutex>
#include <string>
#include <thread>

// Host<->device copy bandwidth through the CUDA driver API (nvcuda.dll ships with the driver).
struct PcieResult {
    bool valid = false;
    double h2dGBs = 0, d2hGBs = 0;
    std::string error;
};

class PcieTest {
public:
    ~PcieTest();
    void Start();
    bool Running() const { return running_; }
    float Progress() const { return progress_; }
    PcieResult Result() const { std::lock_guard lk(mu_); return result_; }

private:
    void Run();
    std::thread worker_;
    std::atomic<bool> running_{false};
    std::atomic<float> progress_{0};
    mutable std::mutex mu_;
    PcieResult result_;
};
