#include <windows.h>
#include <algorithm>
#include <cmath>
#include <format>
#include "app/App.h"
#include "core/Log.h"
#include "core/Util.h"
#include "imgui.h"
#include "imgui_internal.h"
#include "platform/Affinity.h"
#include "platform/Elevation.h"
#include "ui/Panels.h"

void App::Init(bool firstRun) {
    needLayout_ = firstRun;
    cpu = CpuTopology::Probe();
    Affinity::RestrictToECores(cpu);   // stay off the P-cores used by inference
    mem = MemInventory::Probe();
    sys = SystemConfig::Probe();
    Log::I(cpu.brand + ", " + std::to_string(cpu.pCores) + "P+" + std::to_string(cpu.eCores) + "E, " +
           std::to_string(cpu.logicalCount) + " threads");
    Log::I(std::format("CPUID 0x16: base {} MHz, max {} MHz; spec max turbo P {} / E {} MHz", cpu.cpuidBaseMhz,
                       cpu.cpuidMaxMhz, cpu.specPMaxMhz, cpu.specEMaxMhz));
    displays = DisplayTopology::Probe(cpu.brand);
    for (const auto& a : displays.adapters)
        for (const auto& m : a.monitors)
            Log::I(std::format("Display {} {}x{}@{} -> {}", m.device, m.width, m.height, m.hz, a.name));
    // Diagnostic switches to measure each collector's cost on inference speed (see docs/DESIGN.md §18c).
    const wchar_t* cl = GetCommandLineW();
    auto off = [&](const wchar_t* sw) {
        bool o = wcsstr(cl, sw) != nullptr;
        if (o) Log::I("collector disabled: " + ToUtf8(sw));
        return o;
    };
    if (!off(L"--no-nvml")) nvml.Start();
    pdh.noGpuCounters = off(L"--no-pdh-gpu");
    if (!off(L"--no-pdh-all")) pdh.Start(cpu);
    if (!off(L"--no-procmon")) procMon.Start();
    if (!off(L"--no-imc")) imc.Start(sys.elevated);

    settings.Load();
    ScanModels();
    if (!settings.modelPath.empty()) LoadModel(ToWide(settings.modelPath));
    else if (!modelFiles.empty()) LoadModel(modelFiles.front());
    inference.SetContext(cpu, settings.modelPath);
    if (displays.nvidiaIndex >= 0) pdh.SetGpuLuid(displays.adapters[displays.nvidiaIndex].LuidTag());
}

std::string App::RunningLlama(bool* busy) const {
    std::string name;
    bool b = false;
    for (const auto& p : procMon.Snapshot())
        if (p.IsLlama()) {
            if (name.empty() || p.cpuPct >= 10) name = p.name;
            b = b || p.cpuPct >= 10;
        }
    if (busy) *busy = b;
    return name;
}

void App::StartHardwareCheck() {
    if (hwCheck.stage != HwCheck::Stage::Idle) return;
    // Configuration can change while the app runs (monitor cable moved, power plan switched, privilege granted).
    mem = MemInventory::Probe();
    sys = SystemConfig::Probe();
    displays = DisplayTopology::Probe(cpu.brand);
    if (displays.nvidiaIndex >= 0) pdh.SetGpuLuid(displays.adapters[displays.nvidiaIndex].LuidTag());
    // The tests saturate the memory bus and PCIe link: next to llama.cpp both would read wrong.
    hwCheck.procs = procMon.Snapshot();
    bool busy = false;
    std::string llama = RunningLlama(&busy);
    hwCheck.skipped = busy ? llama : "";
    if (busy) {
        Log::I("Hardware check: configuration refreshed, tests skipped (" + hwCheck.skipped + " computing)");
        hwCheck.rows = BuildHardwareRows(*this);
        hwCheck.lastRun = NowSeconds();
        return;
    }
    Log::I("Hardware check: started");
    nvml.ResetPeaks();
    hwCheck.memT0 = NowSeconds();
    memTest.Start(cpu);
    hwCheck.stage = HwCheck::Stage::Memory;
}

void App::UpdateHardwareCheck() {
    using Stage = HwCheck::Stage;
    if (hwCheck.startAt > 0 && NowSeconds() >= hwCheck.startAt) { hwCheck.startAt = 0; StartHardwareCheck(); }
    if (hwCheck.stage == Stage::Memory && !memTest.Running()) {
        // The memory test keeps every core busy: a clean all-core clock reading for free.
        for (int kind = 0; kind < 2; ++kind) {
            std::vector<float> vals, xs, ys;
            const float cap = cpu.ClockCap(kind == 0);
            for (size_t ci = 0; ci < cpu.cores.size() && ci < pdh.coreMhz.size(); ++ci) {
                if (cpu.cores[ci].pcore != (kind == 0)) continue;
                pdh.coreMhz[ci]->Copy(hwCheck.memT0 + 1.0, xs, ys);   // skip the first second (allocation)
                for (float y : ys)
                    if (std::isfinite(y) && y > 0 && y <= cap) vals.push_back(y);
            }
            float med = 0;
            if (!vals.empty()) {
                std::nth_element(vals.begin(), vals.begin() + vals.size() / 2, vals.end());
                med = vals[vals.size() / 2];
            }
            (kind == 0 ? hwCheck.pLoadMhz : hwCheck.eLoadMhz) = med;
        }
        pcieTest.Start();
        hwCheck.stage = Stage::Pcie;
    } else if (hwCheck.stage == Stage::Pcie && !pcieTest.Running()) {
        hwCheck.stage = Stage::Idle;
        hwCheck.rows = BuildHardwareRows(*this);
        hwCheck.lastRun = NowSeconds();
        Log::I(std::format("Hardware check: done (all-core clock P {:.0f} / E {:.0f} MHz)", hwCheck.pLoadMhz,
                           hwCheck.eLoadMhz));
    }
    // Startup snapshot (no tests) once the process monitor has its first sample.
    if (hwCheck.rows.empty() && hwCheck.stage == Stage::Idle && !procMon.Snapshot().empty())
        hwCheck.rows = BuildHardwareRows(*this);
}

// Looks for .gguf files (excluding multimodal projectors) in the usual places.
void App::ScanModels() {
    std::vector<std::wstring> dirs;
    wchar_t profile[MAX_PATH] = {};
    if (GetEnvironmentVariableW(L"USERPROFILE", profile, MAX_PATH)) {
        dirs.push_back(std::wstring(profile) + L"\\Downloads\\models");
        dirs.push_back(std::wstring(profile) + L"\\models");
        dirs.push_back(std::wstring(profile) + L"\\.cache\\llama.cpp");
    }
    dirs.push_back(L"C:\\models");
    if (!settings.modelPath.empty()) {
        std::wstring p = ToWide(settings.modelPath);
        dirs.push_back(p.substr(0, p.find_last_of(L"\\/")));
    }
    modelFiles.clear();
    for (const auto& d : dirs) {
        WIN32_FIND_DATAW fd;
        HANDLE h = FindFirstFileW((d + L"\\*.gguf").c_str(), &fd);
        if (h == INVALID_HANDLE_VALUE) continue;
        do {
            std::wstring name = fd.cFileName, lower = name;
            for (auto& c : lower) c = towlower(c);
            if (lower.find(L"mmproj") != std::wstring::npos) continue;
            std::wstring full = d + L"\\" + name;
            if (std::find(modelFiles.begin(), modelFiles.end(), full) == modelFiles.end()) modelFiles.push_back(full);
        } while (FindNextFileW(h, &fd));
        FindClose(h);
    }
}

void App::LoadModel(const std::wstring& path) {
    model = ModelInfo::FromGguf(GgufFile::Read(path));
    if (model.ok) {
        settings.modelPath = ToUtf8(path);
        settings.Save();
        inference.SetContext(cpu, settings.modelPath);
        Log::I(std::format("Model: {} ({}, {} layers, {} experts top-{}, {})", model.name, model.arch, model.nLayers,
                           model.nExpert, model.nExpertUsed, FormatBytes(model.fileSize)));
    } else {
        Log::E("Cannot read model " + ToUtf8(path) + ": " + model.error);
    }
}

void App::Shutdown() {
    settings.Save();
    if (inference.Active() && !inference.LeftoverJournal()) inference.RestoreBlocking();
    procMon.Stop();
    imc.Stop();
    pdh.Stop();
    nvml.Stop();
}

std::vector<float> App::EffectiveMhz() const {
    auto perf = pdh.CorePerf();
    std::vector<float> mhz(perf.size(), 0);
    for (size_t i = 0; i < perf.size() && i < cpu.baseMhz.size(); ++i) mhz[i] = cpu.baseMhz[i] * perf[i] / 100.0f;
    return mhz;
}

void App::DrawMenuBar() {
    if (!ImGui::BeginMainMenuBar()) return;
    ImGui::TextUnformatted("LlamaScope");
    ImGui::Separator();
    if (ImGui::BeginMenu("View")) {
        if (ImGui::MenuItem("Reset layout")) needLayout_ = true;
        ImGui::EndMenu();
    }
    float right = ImGui::GetWindowWidth();
    const char* label = sys.elevated ? "Administrator" : "Not elevated";
    float w = ImGui::CalcTextSize(label).x + (sys.elevated ? 0 : ImGui::CalcTextSize("Restart as administrator").x + 40);
    ImGui::SetCursorPosX(right - w - 20);
    ImGui::TextColored(sys.elevated ? ImVec4(0.25f, 0.73f, 0.31f, 1) : ImVec4(0.82f, 0.6f, 0.13f, 1), "%s", label);
    if (!sys.elevated) {
        if (ImGui::SmallButton("Restart as administrator") && RestartAsAdmin()) PostQuitMessage(0);
    }
    ImGui::EndMainMenuBar();
}

void App::BuildDefaultLayout(unsigned dock) {
    {
        auto* vp = ImGui::GetMainViewport();
        auto& io = ImGui::GetIO();
        Log::I(std::format("Layout: display {:.0f}x{:.0f}, viewport {:.0f}x{:.0f} work {:.0f}x{:.0f}, fb scale {:.2f}, dpi {:.2f}",
                           io.DisplaySize.x, io.DisplaySize.y, vp->Size.x, vp->Size.y, vp->WorkSize.x, vp->WorkSize.y,
                           io.DisplayFramebufferScale.x, vp->DpiScale));
    }
    ImGui::DockBuilderRemoveNode(dock);
    ImGui::DockBuilderAddNode(dock, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodePos(dock, ImGui::GetMainViewport()->WorkPos);
    ImGui::DockBuilderSetNodeSize(dock, ImGui::GetMainViewport()->WorkSize);
    ImGuiID top, bottom, left, rest, mid, right, rtop, rbot, bleft, bright;
    ImGui::DockBuilderSplitNode(dock, ImGuiDir_Down, 0.30f, &bottom, &top);
    ImGui::DockBuilderSplitNode(top, ImGuiDir_Left, 0.24f, &left, &rest);
    ImGui::DockBuilderSplitNode(rest, ImGuiDir_Left, 0.50f, &mid, &right);
    ImGui::DockBuilderSplitNode(right, ImGuiDir_Up, 0.5f, &rtop, &rbot);
    ImGui::DockBuilderSplitNode(bottom, ImGuiDir_Right, 0.30f, &bright, &bleft);
    ImGui::DockBuilderDockWindow(Win::Hardware, left);
    ImGui::DockBuilderDockWindow(Win::Topology, mid);
    ImGui::DockBuilderDockWindow(Win::Cpu, rtop);
    ImGui::DockBuilderDockWindow(Win::Gpu, rbot);
    ImGui::DockBuilderDockWindow(Win::Timeline, bleft);
    ImGui::DockBuilderDockWindow(Win::Launcher, bleft);
    ImGui::DockBuilderDockWindow(Win::Runs, bleft);
    ImGui::DockBuilderDockWindow(Win::Tools, bleft);
    ImGui::DockBuilderDockWindow(Win::Log, bleft);
    ImGui::DockBuilderDockWindow(Win::Inference, bleft);
    ImGui::DockBuilderDockWindow(Win::Diagnosis, bright);
    ImGui::DockBuilderFinish(dock);
}

void App::Frame() {
    // Smoothed loaded-core clocks (reference for "below normal clock" verdicts).
    auto mhz = EffectiveMhz();
    auto util = pdh.CoreUtil();
    for (int kind = 0; kind < 2; ++kind) {
        ClockTrack& ct = kind == 0 ? pClock : eClock;
        const float cap = cpu.ClockCap(kind == 0);
        double sum = 0;
        int n = 0;
        for (int lp = 0; lp < (int)mhz.size() && lp < (int)util.size(); ++lp)
            if (cpu.IsPCoreLogical(lp) == (kind == 0) && util[lp] > 50 && mhz[lp] > 0 && mhz[lp] <= cap) {
                sum += mhz[lp];
                ++n;
            }
        // Only judge clocks under broad load (inference loads most cores); a single busy desktop
        // thread produces noisy, meaningless readings.
        int typeCount = kind == 0 ? cpu.pCores : cpu.eCores;
        ct.loaded = n;
        if (n == 0 || n < std::max(2, typeCount / 2)) { ct.ema = 0; ct.loaded = 0; continue; }
        float avg = (float)(sum / n);
        // ~1 s time constant, independent of the frame rate (2-4 fps when idle, monitor rate when interacting).
        float a = 1.0f - std::exp(-std::min(ImGui::GetIO().DeltaTime, 1.0f) / 1.0f);
        ct.ema = ct.ema == 0 ? avg : ct.ema + a * (avg - ct.ema);
        ct.ref = std::max(ct.ref, ct.ema);
    }

    UpdateHardwareCheck();
    diag.Update(*this);

    DrawMenuBar();
    ImGuiID dock = ImGui::GetID("MainDockSpace");
    // Wait until the viewport has its real (maximized) size before splitting.
    if (needLayout_ && ImGui::GetFrameCount() > 2) {
        BuildDefaultLayout(dock);
        needLayout_ = false;
    }
    ImGui::DockSpaceOverViewport(dock, ImGui::GetMainViewport());

    DrawHardwarePanel(*this);
    DrawTopologyPanel(*this);
    DrawCpuRuntimePanel(*this);
    DrawGpuRuntimePanel(*this);
    DrawTimelinePanel(*this);
    DrawDiagnosisPanel(*this);
    DrawLauncherPanel(*this);
    DrawRunsPanel(*this);
    DrawToolsPanel(*this);
    DrawLogPanel(*this);
    DrawInferenceModePanel(*this);
    if (showPlatformDetails) DrawPlatformDetails(*this, &showPlatformDetails);

    // A previous session left the system tuned (crash / kill): offer to restore it once.
    static bool asked = false;
    if (inference.LeftoverJournal() && sys.elevated && !asked) {
        ImGui::OpenPopup("Restore system settings?");
        asked = true;
    }
    if (ImGui::BeginPopupModal("Restore system settings?", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextUnformatted("LlamaScope did not exit normally while inference mode was on.");
        ImGui::TextUnformatted("Some services, scheduled tasks, power settings or CPU assignments may still be changed.");
        if (ImGui::Button("Restore now")) { inference.RestoreAsync(); ImGui::CloseCurrentPopup(); }
        ImGui::SameLine();
        if (ImGui::Button("Later")) ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }
}

std::wstring App::ProfilesDir() {
    std::wstring d = LocalDataDir() + L"\\profiles";
    CreateDirectoryW(d.c_str(), nullptr);
    return d;
}

void App::LoadProfile(const std::wstring& path) {
    profile = ProfileData::Load(path);
    if (profile.ok)
        Log::I(std::format("Profile: {} generation steps, {:.0f} us per step, {} threads, GPU markers {}", profile.genSteps,
                           profile.stepUs, profile.threads, profile.hasGpu ? "yes" : "no"));
    else
        Log::W("Profile: " + profile.error + " (" + ToUtf8(path) + ")");
}

bool App::LoadLatestProfile() {
    WIN32_FIND_DATAW fd, best{};
    HANDLE h = FindFirstFileW((ProfilesDir() + L"\\*.prof").c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return false;
    do {
        if (CompareFileTime(&fd.ftLastWriteTime, &best.ftLastWriteTime) > 0) best = fd;
    } while (FindNextFileW(h, &fd));
    FindClose(h);
    if (!best.cFileName[0]) return false;
    LoadProfile(ProfilesDir() + L"\\" + best.cFileName);
    return true;
}
