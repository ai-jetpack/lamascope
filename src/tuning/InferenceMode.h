#pragma once
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "probes/CpuTopology.h"

// Session-scoped system tuning for inference. Every change is journaled (prior state) before it is
// made, and undone on Restore, on app exit, or at the next start after a crash.
struct ImAction {
    std::string id, label, effect, risk;
    bool enabled = true;
    bool needsConsent = false;
    std::string state;      // current state, from the last probe
};

class InferenceMode {
public:
    InferenceMode();
    ~InferenceMode();

    void SetContext(const CpuTopology& cpu, const std::string& modelPath);
    void ProbeAsync();                   // refresh the "current state" column
    void EnterAsync();
    void RestoreAsync();
    void RestoreBlocking();              // app exit
    bool Active() const { return active_; }
    bool Busy() const { return busy_; }
    bool LeftoverJournal() const { return leftover_; }   // a previous session was not restored
    void DismissLeftover() { leftover_ = false; }

    std::vector<ImAction> Actions() const { std::lock_guard lk(mu_); return actions_; }
    void SetEnabled(const std::string& id, bool on);
    std::vector<std::string> Log() const { std::lock_guard lk(mu_); return log_; }
    std::vector<std::string> NotMoved() const { std::lock_guard lk(mu_); return notMoved_; }
    int MovedCount() const { return moved_; }

private:
    void Probe();
    void Enter();
    void Restore();
    void Enforce();                      // background: keeps other programs on E-cores, llama on P-cores
    void Note(const std::string& s);
    bool On(const std::string& id) const;
    void SaveJournal();
    void LoadJournal();

    mutable std::mutex mu_;
    std::vector<ImAction> actions_;
    std::vector<std::string> log_, notMoved_;
    CpuTopology cpu_;
    std::string modelPath_;
    std::atomic<bool> active_{false}, busy_{false}, leftover_{false}, stopEnforce_{false};
    std::atomic<int> moved_{0};
    std::thread worker_, enforcer_;

    // Journal (prior state of everything we changed)
    struct AffinityRec { unsigned pid; uint64_t create; uint64_t mask; std::string name; };
    std::vector<AffinityRec> affinity_;
    std::vector<unsigned> llamaTuned_;
    std::vector<std::string> stoppedServices_, disabledTasks_, addedExclPaths_, addedExclProcs_;
    std::string origScheme_, tunedScheme_;
    int origCpMinCores_ = -1, origAspm_ = -1;
    uint64_t enterTime_ = 0;             // FILETIME when inference mode was entered
    int origBgApps_ = -2;                // -2 = untouched, -1 = value was absent
};
