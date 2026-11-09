#pragma once
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include "core/Status.h"

class App;

// One line per issue, aggregated over the session. Conditions are debounced:
// an episode starts after the condition holds kStartSec and ends after kEndSec without it.
struct Finding {
    std::string id;
    Status severity = Status::Warn;
    std::string text;        // stable one-line summary
    std::string detail;      // latest detailed reason (tooltip)
    bool active = false;
    double lastSeen = 0;     // seconds since app start
};

class DiagnosisEngine {
public:
    static constexpr double kStartSec = 3.0;
    static constexpr double kEndSec = 5.0;

    void Update(App& app);                  // call every frame; evaluates at 1 Hz
    std::vector<Finding> Findings() const { std::lock_guard lk(mu_); return findings_; }

private:
    struct Track {
        Status severity = Status::Warn;
        bool lowerIsWorse = true;
        std::string detail;        // latest reason (tooltip)
        std::string startDetail;   // reason when the episode started (stable line text)
        double condSince = -1;   // condition true since (pending start)
        double falseSince = -1;  // condition false since (pending end)
        bool inEpisode = false;
        int episodes = 0;
        double totalSec = 0;
        double lastSeen = 0;
        float worst = 0;
        bool hasWorst = false;
        bool persistent = false; // configuration issue: no episode stats
    };
    // cond: condition this tick; metric: value for worst-case tracking (optional)
    void Observe(const std::string& id, bool cond, Status sev, const std::string& detail, double now,
                 bool hasMetric = false, float metric = 0, bool lowerIsWorse = true, bool persistent = false);

    double lastEval_ = -1;
    std::map<std::string, Track> tracks_;
    std::map<std::string, std::string> titles_;   // id -> title template
    mutable std::mutex mu_;
    std::vector<Finding> findings_;
    // Peak VRAM while the current llama.cpp process lives (room-for-more-layers rule).
    unsigned peakPid_ = 0;
    uint64_t peakUsed_ = 0;
    bool worked_ = false;
};
