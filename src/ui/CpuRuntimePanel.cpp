#include <algorithm>
#include <cmath>
#include <format>
#include "app/App.h"
#include "imgui.h"
#include "imgui_internal.h"
#include "implot.h"
#include "ui/Panels.h"

namespace {

// Calm single-hue load scale: dark slate (idle) -> steel blue (busy).
const ImVec4 kIdle(0.16f, 0.18f, 0.22f, 1), kMid(0.20f, 0.33f, 0.48f, 1), kBusy(0.33f, 0.56f, 0.80f, 1);

ImU32 LoadColor(float util) {
    float t = std::clamp(util / 100.0f, 0.0f, 1.0f);
    ImVec4 o = t < 0.5f ? ImLerp(kIdle, kMid, t * 2) : ImLerp(kMid, kBusy, (t - 0.5f) * 2);
    return ImGui::GetColorU32(o);
}

// Display-only smoothing so tiles drift instead of flashing at the 250 ms sample rate.
void Smooth(std::vector<float>& state, const std::vector<float>& target, float tau) {
    float a = 1.0f - std::exp(-ImGui::GetIO().DeltaTime / tau);
    if (state.size() != target.size()) state = target;
    for (size_t i = 0; i < state.size(); ++i) state[i] += (target[i] - state[i]) * a;
}

// Core map: one tile per physical core; SMT siblings split the tile vertically.
void DrawCoreMap(App& app) {
    static std::vector<float> util, mhz;
    Smooth(util, app.pdh.CoreUtil(), 1.5f);
    Smooth(mhz, app.EffectiveMhz(), 1.5f);
    const auto& cpu = app.cpu;
    float font = ImGui::GetFontSize();
    float tileW = std::max(font * 3.2f, (ImGui::GetContentRegionAvail().x - font) / std::max(8, cpu.pCores) - 6);
    float tileH = font * 3.4f;
    ImDrawList* dl = ImGui::GetWindowDrawList();

    for (int kind = 0; kind < (cpu.hybrid ? 2 : 1); ++kind) {
        bool p = kind == 0;
        ImGui::TextDisabled(p ? (cpu.hybrid ? "Performance cores" : "Cores") : "Efficiency cores");
        int col = 0;
        for (const auto& core : cpu.cores) {
            if (core.pcore != p) continue;
            if (col > 0 && ImGui::GetContentRegionAvail().x > tileW) ImGui::SameLine(0, 6);
            ImVec2 p0 = ImGui::GetCursorScreenPos();
            ImGui::InvisibleButton(core.Label().c_str(), ImVec2(tileW, tileH));
            int n = (int)core.logical.size();
            float sliceW = tileW / n;
            float maxMhz = 0, avgUtil = 0;
            for (int i = 0; i < n; ++i) {
                int lp = core.logical[i];
                float u = lp < (int)util.size() ? util[lp] : 0;
                avgUtil += u / n;
                maxMhz = std::max(maxMhz, lp < (int)mhz.size() ? mhz[lp] : 0.0f);
                ImVec2 a(p0.x + sliceW * i, p0.y), b(p0.x + sliceW * (i + 1), p0.y + tileH);
                dl->AddRectFilled(a, b, LoadColor(u), 3.0f);
                if (i > 0) dl->AddLine(a, ImVec2(a.x, b.y), IM_COL32(0, 0, 0, 120));
            }
            dl->AddRect(p0, ImVec2(p0.x + tileW, p0.y + tileH), IM_COL32(255, 255, 255, 40), 3.0f);
            // Clock lives in the timeline below; tiles only show load (and parked state).
            std::string l1 = core.Label(), l2 = std::format("{:.0f}%", avgUtil), l3 = maxMhz > 300 ? "" : "parked";
            dl->AddText(ImVec2(p0.x + 4, p0.y + 2), IM_COL32(255, 255, 255, 230), l1.c_str());
            dl->AddText(ImVec2(p0.x + 4, p0.y + 2 + font), IM_COL32(255, 255, 255, 200), l2.c_str());
            dl->AddText(ImVec2(p0.x + 4, p0.y + 2 + font * 2), IM_COL32(255, 255, 255, 150), l3.c_str());
            if (ImGui::IsItemHovered()) {
                std::string tip = core.Label() + "\n";
                for (int lp : core.logical) {
                    float m = lp < (int)mhz.size() ? mhz[lp] : 0;
                    tip += std::format("CPU {}: {:.0f}%  {}\n", lp, lp < (int)util.size() ? util[lp] : 0,
                                       m > 300 ? std::format("{:.0f} MHz", m) : std::string("parked"));
                }
                ImGui::SetTooltip("%s", tip.c_str());
            }
            ++col;
        }
    }
}

// Pause: the time series stop scrolling (data keeps being collected). While paused, the clock charts can be
// dragged / zoomed back through the stored history (Series keeps 6000 samples = 25 min at 250 ms).
double g_pausedAt = 0;      // NowSeconds() when paused, 0 = live
bool g_pauseChanged = false;
constexpr double kWindow = 120;
double ViewEnd() { return g_pausedAt > 0 ? g_pausedAt : NowSeconds(); }
// Live: always pinned to the last 2 min. Paused: pinned once, then free for the user to pan and zoom.
void SetupTimeAxis() {
    double end = ViewEnd();
    if (g_pausedAt == 0 || g_pauseChanged) ImPlot::SetupAxisLimits(ImAxis_X1, end - kWindow, end, ImPlotCond_Always);
}
// Copies the samples to draw: the visible window when live, all history up to the pause point when paused.
void CopyView(const Series& s, std::vector<float>& xs, std::vector<float>& ys) {
    if (g_pausedAt == 0) { s.Copy(NowSeconds() - kWindow - 1, xs, ys); return; }
    s.Copy(0, xs, ys);
    size_t n = std::upper_bound(xs.begin(), xs.end(), (float)g_pausedAt) - xs.begin();
    xs.resize(n);
    ys.resize(n);
}

void ClockPlot(const char* id, const std::vector<std::pair<const Series*, std::pair<std::string, ImVec4>>>& lines,
               float height, bool legendOutside) {
    static std::vector<float> xs, ys;
    if (!ImPlot::BeginPlot(id, ImVec2(-1, height), ImPlotFlags_NoMouseText)) return;
    ImPlot::SetupAxes(nullptr, "GHz", ImPlotAxisFlags_NoTickLabels, ImPlotAxisFlags_AutoFit);
    SetupTimeAxis();
    ImPlot::SetupLegend(legendOutside ? ImPlotLocation_East : ImPlotLocation_NorthWest,
                        legendOutside ? ImPlotLegendFlags_Outside : ImPlotLegendFlags_Horizontal);
    for (const auto& [s, meta] : lines) {
        CopyView(*s, xs, ys);
        for (auto& y : ys) y /= 1000.0f;
        ImPlotSpec spec;
        spec.LineColor = meta.second;
        spec.LineWeight = 1.5f;
        ImPlot::PlotLine(meta.first.c_str(), xs.data(), ys.data(), (int)xs.size(), spec);
    }
    ImPlot::EndPlot();
}

// Overall clock timeline (P avg / P max / E avg) with an expandable per-core view.
void DrawClockTimeline(App& app) {
    const auto& cpu = app.cpu;
    const ImVec4 pCol(0.33f, 0.56f, 0.80f, 1), pMaxCol(0.65f, 0.80f, 0.95f, 1), eCol(0.18f, 0.75f, 0.65f, 1);
    float h = ImGui::GetFontSize() * 9;
    std::vector<std::pair<const Series*, std::pair<std::string, ImVec4>>> overall = {
        {&app.pdh.pAvgMhz, {cpu.hybrid ? "P avg" : "avg", pCol}},
        {&app.pdh.pMaxMhz, {cpu.hybrid ? "P max" : "max", pMaxCol}}};
    if (cpu.hybrid) overall.push_back({&app.pdh.eAvgMhz, {"E avg", eCol}});
    ClockPlot("##clock", overall, h, false);

    // Small multiples: one compact chart per physical core, shared axes (P-cores first).
    if (ImGui::TreeNode("Per core")) {
        const int n = (int)std::min(cpu.cores.size(), app.pdh.coreMhz.size());
        const int cols = 4, rows = (n + cols - 1) / cols;
        const double yMax = std::max(cpu.ClockCap(true) / 1000.0, 1.0);
        static std::vector<float> xs, ys;
        if (n > 0 && ImPlot::BeginSubplots("##percore", rows, cols, ImVec2(-1, rows * ImGui::GetFontSize() * 5.0f),
                                           ImPlotSubplotFlags_NoTitle | ImPlotSubplotFlags_LinkAllX |
                                               ImPlotSubplotFlags_LinkAllY)) {
            for (int c = 0; c < rows * cols; ++c) {
                if (c >= n) {   // keep the grid square
                    if (ImPlot::BeginPlot("##empty", ImVec2(), ImPlotFlags_CanvasOnly)) {
                        ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_NoDecorations);
                        ImPlot::EndPlot();
                    }
                    continue;
                }
                const auto& core = cpu.cores[c];
                std::string title = core.Label();
                if (ImPlot::BeginPlot(title.c_str(), ImVec2(),
                                      ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText | ImPlotFlags_NoMenus)) {
                    ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_NoTickLabels,
                                      c % cols == 0 ? ImPlotAxisFlags_None : ImPlotAxisFlags_NoTickLabels);
                    SetupTimeAxis();
                    ImPlot::SetupAxisLimits(ImAxis_Y1, 0, yMax, ImPlotCond_Always);
                    CopyView(*app.pdh.coreMhz[c], xs, ys);
                    for (auto& y : ys) y /= 1000.0f;
                    ImPlotSpec spec;
                    spec.LineColor = core.pcore ? pCol : eCol;
                    spec.LineWeight = 1.5f;
                    ImPlot::PlotLine(title.c_str(), xs.data(), ys.data(), (int)xs.size(), spec);
                    ImPlot::EndPlot();
                }
            }
            ImPlot::EndSubplots();
        }
        ImGui::TextDisabled("GHz. Gaps = core idle or parked (clock is only measurable while a core is "
                            "at least 10%% busy).");
        ImGui::TreePop();
    }
}

// DRAM traffic (memory controller counters via emon) against the measured and theoretical maximum.
void DrawMemoryTraffic(App& app) {
    using St = ImcCollector::State;
    auto st = app.imc.GetState();
    if (st != St::Running) {
        ImGui::PushTextWrapPos(0);
        if (st == St::Starting) ImGui::TextDisabled("Starting the memory controller counters...");
        else ImGui::TextDisabled("Unavailable: %s", app.imc.Detail().c_str());
        ImGui::PopTextWrapPos();
        return;
    }
    const double measured = app.memTest.Result().valid ? app.memTest.Result().bestGBs : 0;
    const double theoretical = app.mem.theoreticalGBs;

    // Readout: average of the last 10 s (or the 10 s before the pause), so the numbers stay readable.
    static std::vector<float> xs, ys;
    auto avg10 = [&](const Series& s) {
        double end = ViewEnd();
        s.Copy(end - 10, xs, ys);
        double sum = 0;
        int n = 0;
        for (size_t i = 0; i < xs.size(); ++i)
            if (xs[i] <= end) { sum += ys[i]; ++n; }
        return n ? sum / n : 0.0;
    };
    double r = avg10(app.imc.readGBs), w = avg10(app.imc.writeGBs);
    std::string ref = measured > 0 ? std::format(" = {:.0f}% of the measured maximum ({:.0f} GB/s)", (r + w) / measured * 100, measured)
                                   : std::format(" = {:.0f}% of theoretical; press Run tests for the measured maximum", (r + w) / theoretical * 100);
    ImGui::Text("Read %.1f GB/s, write %.1f GB/s%s", r, w, ref.c_str());
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Average of the last 10 s. Counts every transfer between the memory controllers and RAM: "
                          "all programs, the integrated GPU's display scan-out, and PCIe copies to the GPU.");

    if (!ImPlot::BeginPlot("##dram", ImVec2(-1, ImGui::GetFontSize() * 8), ImPlotFlags_NoMouseText)) return;
    ImPlot::SetupAxes(nullptr, "GB/s", ImPlotAxisFlags_NoTickLabels, ImPlotAxisFlags_None);
    SetupTimeAxis();
    ImPlot::SetupAxisLimits(ImAxis_Y1, 0, std::max(theoretical, 10.0) * 1.08, ImPlotCond_Always);
    ImPlot::SetupLegend(ImPlotLocation_NorthWest, ImPlotLegendFlags_Horizontal);
    auto line = [&](const char* name, const Series& s, ImVec4 col) {
        CopyView(s, xs, ys);
        ImPlotSpec spec;
        spec.LineColor = col;
        spec.LineWeight = 1.5f;
        ImPlot::PlotLine(name, xs.data(), ys.data(), (int)xs.size(), spec);
    };
    line("read", app.imc.readGBs, ImVec4(0.33f, 0.56f, 0.80f, 1));
    line("write", app.imc.writeGBs, ImVec4(0.85f, 0.55f, 0.25f, 1));
    ImPlotSpec refSpec;
    refSpec.Flags = ImPlotInfLinesFlags_Horizontal;
    refSpec.LineWeight = 1.0f;
    refSpec.LineColor = ImVec4(0.6f, 0.6f, 0.6f, 0.8f);
    if (measured > 0) ImPlot::PlotInfLines("measured max", &measured, 1, refSpec);
    refSpec.LineColor = ImVec4(0.6f, 0.6f, 0.6f, 0.35f);
    ImPlot::PlotInfLines("theoretical", &theoretical, 1, refSpec);
    ImPlot::EndPlot();
}

void DrawUtilHistory(App& app) {
    const int rows = app.cpu.logicalCount;
    const int cols = PdhCollector::kHistory;
    if (rows == 0) return;
    static std::vector<float> data;
    if (g_pausedAt == 0 || data.empty()) data = app.pdh.UtilHistory();   // frozen while paused
    static std::vector<std::string> labelStore;
    static std::vector<const char*> labels;
    static std::vector<double> ticks;
    if ((int)labels.size() != rows) {
        labelStore.clear();
        labels.clear();
        ticks.clear();
        // Label only the first logical CPU of each core to keep rows readable.
        for (int lp = 0; lp < rows; ++lp) {
            int c = app.cpu.logicalToCore[lp];
            bool first = c >= 0 && app.cpu.cores[c].logical.front() == lp;
            labelStore.push_back(first ? app.cpu.cores[c].Label() : "");
        }
        for (int lp = 0; lp < rows; ++lp) {
            labels.push_back(labelStore[lp].c_str());
            ticks.push_back(rows - lp - 0.5);
        }
    }
    // Same palette as the core tiles.
    static ImPlotColormap cmap = [] {
        const ImVec4 c[] = {kIdle, kMid, kBusy};
        return ImPlot::AddColormap("CoreLoad", c, 3);
    }();
    float h = std::max(ImGui::GetContentRegionAvail().y, ImGui::GetFontSize() * 0.75f * rows + 40);
    if (ImPlot::BeginPlot("##utilhist", ImVec2(-1, h), ImPlotFlags_NoMouseText)) {
        ImPlot::SetupAxes("seconds", nullptr, ImPlotAxisFlags_None, ImPlotAxisFlags_Lock);
        ImPlot::SetupAxisLimits(ImAxis_X1, -60, 0, ImPlotCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0, rows, ImPlotCond_Always);
        ImPlot::SetupAxisTicks(ImAxis_Y1, ticks.data(), rows, labels.data());
        ImPlot::PushColormap(cmap);
        ImPlot::PlotHeatmap("##u", data.data(), rows, cols, 0, 100, nullptr, ImPlotPoint(-60, 0), ImPlotPoint(0, rows));
        ImPlot::PopColormap();
        ImPlot::EndPlot();
    }
}

} // namespace

void DrawCpuRuntimePanel(App& app) {
    if (!ImGui::Begin(Win::Cpu)) { ImGui::End(); return; }
    DrawCoreMap(app);

    // Pause / resume the time series below (the core tiles above stay live).
    g_pauseChanged = false;
    if (ImGui::Button(g_pausedAt > 0 ? "Resume" : "Pause")) {
        g_pausedAt = g_pausedAt > 0 ? 0 : NowSeconds();
        g_pauseChanged = true;
    }
    ImGui::SameLine();
    ImGui::AlignTextToFramePadding();
    if (g_pausedAt > 0) {
        int ago = (int)(NowSeconds() - g_pausedAt);
        ImGui::TextColored(StatusColor(Status::Info), "Paused %d:%02d ago. Drag the clock charts to scroll back "
                           "(up to 25 min), scroll to zoom, double-click to fit.", ago / 60, ago % 60);
    } else {
        ImGui::TextDisabled("Freeze the charts below to read them.");
    }

    ImGui::SeparatorText(g_pausedAt > 0 ? "Core clock (paused)" : "Core clock, last 2 min");
    DrawClockTimeline(app);

    ImGui::SeparatorText(g_pausedAt > 0 ? "Memory traffic (paused)" : "Memory traffic, last 2 min");
    DrawMemoryTraffic(app);

    ImGui::Spacing();
    ImGui::SeparatorText("Thread placement and CPU time by module");
    if (!app.sys.elevated) {
        ImGui::TextDisabled("Requires administrator: which core each llama thread runs on, time in");
        ImGui::TextDisabled("ggml-cpu (work) vs libomp (spin/barrier) vs CUDA, preemptions.");
    } else {
        PlaceholderText("M4", "ETW kernel session: thread dots, module time, spin %, preemptions.");
    }

    ImGui::SeparatorText(g_pausedAt > 0 ? "Utilization, 60 s before the pause (per logical CPU)"
                                        : "Utilization, last 60 s (per logical CPU)");
    DrawUtilHistory(app);
    ImGui::End();
}
