#pragma once
#include "collectors/ImcCollector.h"
#include "collectors/Nvml.h"
#include "collectors/PdhCollector.h"
#include "collectors/ProcessMonitor.h"
#include "tuning/InferenceMode.h"
#include "core/Settings.h"
#include "diagnosis/DiagnosisEngine.h"
#include "model/Profile.h"
#include "model/Topology.h"
#include "probes/CpuTopology.h"
#include "probes/Displays.h"
#include "probes/MemBwTest.h"
#include "probes/MemInventory.h"
#include "probes/PcieTest.h"
#include "probes/SystemConfig.h"
#include "ui/Panels.h"

// Window names double as docking identifiers.
namespace Win {
inline constexpr const char* Hardware = "Hardware";
inline constexpr const char* Topology = "Model topology";
inline constexpr const char* Cpu = "CPU runtime";
inline constexpr const char* Gpu = "GPU runtime";
inline constexpr const char* Timeline = "Timeline";
inline constexpr const char* Diagnosis = "Diagnosis";
inline constexpr const char* Launcher = "Launcher";
inline constexpr const char* Runs = "Runs & diff";
inline constexpr const char* Tools = "Tools & setup";
inline constexpr const char* Log = "Log";
inline constexpr const char* Inference = "Inference mode";
}

class App {
public:
    void Init(bool firstRun);
    void Frame();
    void Shutdown();

    CpuTopology cpu;
    MemInventory mem;
    SystemConfig sys;
    DisplayTopology displays;
    NvmlCollector nvml;
    PdhCollector pdh;
    ImcCollector imc;           // live DRAM read/write traffic (needs VTune's emon + admin)
    MemBwTest memTest;
    PcieTest pcieTest;
    DiagnosisEngine diag;
    ProcessMonitor procMon;
    InferenceMode inference;
    bool showPlatformDetails = false;

    // "Run tests": re-probe configuration (memory, displays, power, privileges), reset the PCIe link peak,
    // run the memory test (also measures all-core clocks) then the PCIe test (also retrains the link),
    // and freeze the Hardware panel's verdicts until the next run.
    struct HwCheck {
        enum class Stage { Idle, Memory, Pcie } stage = Stage::Idle;
        double lastRun = 0;          // NowSeconds() when the last run completed; 0 = only the startup snapshot
        std::string skipped;         // why the bandwidth tests were skipped in the last run ("" = they ran)
        double memT0 = 0;
        double startAt = 0;          // deferred start (command line --run-tests), 0 = none
        float pLoadMhz = 0, eLoadMhz = 0;   // median clock of busy cores during the memory test
        std::vector<HwRow> rows;     // frozen verdicts shown by the Hardware panel
        std::vector<ProcInfo> procs; // process CPU use sampled before the tests (the tests load every core)
    };
    HwCheck hwCheck;
    // Name of a running llama.cpp process ("" if none); *busy = it is computing (>= 10% of a core),
    // as opposed to loaded and idle (e.g. llama-cli waiting at its prompt).
    std::string RunningLlama(bool* busy = nullptr) const;
    void StartHardwareCheck();
    void UpdateHardwareCheck();          // per frame: advances the stages

    // Model under study (topology panel).
    Settings settings;
    ModelInfo model;
    std::vector<std::wstring> modelFiles;   // discovered .gguf files
    void ScanModels();
    void LoadModel(const std::wstring& path);

    // Runtime profile from the ggml profiler (GGML_PROFILE), drawn on the topology diagram.
    ProfileData profile;
    static std::wstring ProfilesDir();           // %LOCALAPPDATA%\LlamaScope\profiles
    void LoadProfile(const std::wstring& path);
    bool LoadLatestProfile();                    // newest *.prof in ProfilesDir()

    // Smoothed clock of loaded cores (>50% busy) and its session maximum, per core type.
    // Raw PDH samples spike above turbo, so the clock rule only uses smoothed values.
    struct ClockTrack { float ema = 0, ref = 0; int loaded = 0; };
    ClockTrack pClock, eClock;
    std::vector<float> EffectiveMhz() const;   // per logical processor

private:
    void DrawMenuBar();
    void BuildDefaultLayout(unsigned dockId);
    bool needLayout_ = false;
};
