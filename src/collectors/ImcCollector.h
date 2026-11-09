#pragma once
#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include "core/TimeSeries.h"

// Live DRAM traffic from the memory controllers' free-running counters (UNC_MC0/1_RDCAS/WRCAS_COUNT_FREERUN,
// one count per 64-byte transfer). Windows has no counter for this and reading the controller needs a kernel
// driver, so this runs Intel's emon.exe (ships with VTune, uses its sepdrv driver) as a hidden child process
// and parses its output. The child is tied to a job object, so it dies with LlamaScope.
class ImcCollector {
public:
    enum class State { Off, NoEmon, NotElevated, Starting, Running, Failed };

    ~ImcCollector() { Stop(); }
    void Start(bool elevated);
    void Stop();

    State GetState() const { return state_; }
    std::string Detail() const { std::lock_guard lk(mu_); return detail_; }   // why it is not running
    std::wstring EmonPath() const { std::lock_guard lk(mu_); return emon_; }

    Series readGBs, writeGBs;   // GB/s per 0.5 s interval, both controllers summed

    static std::wstring FindEmon();   // newest VTune's bin64\emon.exe, "" if none

private:
    void Loop();
    void Fail(const std::string& why);
    std::thread thread_;
    std::atomic<bool> stop_{false};
    std::atomic<State> state_{State::Off};
    mutable std::mutex mu_;
    std::string detail_;
    std::wstring emon_;
    void* job_ = nullptr;
    void* process_ = nullptr;
    void* pipe_ = nullptr;
};
