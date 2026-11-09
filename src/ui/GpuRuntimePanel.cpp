#include <algorithm>
#include <format>
#include "app/App.h"
#include "core/Util.h"
#include "imgui.h"
#include "implot.h"
#include "ui/Panels.h"

namespace {

struct Line { const Series* s; const char* label; ImVec4 color; };

void TimePlot(const char* title, std::initializer_list<Line> lines, double ymin, double ymax, float height) {
    static std::vector<float> xs, ys;
    double now = NowSeconds();
    if (!ImPlot::BeginPlot(title, ImVec2(-1, height), ImPlotFlags_NoMouseText)) return;
    ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_NoTickLabels,
                      ymax > ymin ? ImPlotAxisFlags_None : ImPlotAxisFlags_AutoFit);
    ImPlot::SetupAxisLimits(ImAxis_X1, now - 60, now, ImPlotCond_Always);
    if (ymax > ymin) ImPlot::SetupAxisLimits(ImAxis_Y1, ymin, ymax, ImPlotCond_Always);
    ImPlot::SetupLegend(ImPlotLocation_NorthWest, ImPlotLegendFlags_Horizontal);
    for (const auto& l : lines) {
        l.s->Copy(now - 61, xs, ys);
        ImPlotSpec spec;
        spec.LineColor = l.color;
        spec.LineWeight = 1.5f;
        ImPlot::PlotLine(l.label, xs.data(), ys.data(), (int)xs.size(), spec);
    }
    ImPlot::EndPlot();
}

void Chip(const char* text, Status s) {
    ImVec4 c = StatusColor(s);
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(c.x, c.y, c.z, 0.25f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(c.x, c.y, c.z, 0.35f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(c.x, c.y, c.z, 0.35f));
    ImGui::SmallButton(text);
    ImGui::PopStyleColor(3);
}

} // namespace

void DrawGpuRuntimePanel(App& app) {
    if (!ImGui::Begin(Win::Gpu)) { ImGui::End(); return; }
    auto st = app.nvml.Static();
    if (!st.ok) {
        ImGui::TextColored(StatusColor(Status::Bad), "No NVIDIA GPU data: %s", st.error.c_str());
        ImGui::End();
        return;
    }
    auto l = app.nvml.Live();
    bool concerning = false;
    std::string thr = ThrottleReasonsText(l.throttle, &concerning);

    Chip(std::format("{}%  util", l.util).c_str(), Status::Info);
    ImGui::SameLine();
    Chip(std::format("{} MHz", l.gfxMHz).c_str(), Status::Info);
    ImGui::SameLine();
    Chip(std::format("{:.0f} W", l.powermW / 1000.0).c_str(), Status::Info);
    ImGui::SameLine();
    Chip(std::format("{} C", l.tempC).c_str(), l.tempC >= 83 ? Status::Warn : Status::Info);
    ImGui::SameLine();
    Chip(std::format("PCIe Gen{} x{}", l.linkGen, l.linkWidth).c_str(), Status::Info);
    ImGui::SameLine();
    Chip(("limiters: " + thr).c_str(), concerning ? Status::Warn : Status::Ok);

    // VRAM: dedicated vs shared (shared = system RAM used by the GPU, i.e. sysmem fallback).
    float ded = app.pdh.gpuDedicatedGB.Last(), shared = app.pdh.gpuSharedGB.Last();
    float total = (float)(st.vramTotal / double(1ull << 30));
    ImGui::Text("VRAM");
    ImGui::SameLine(ImGui::GetFontSize() * 4);
    ImGui::ProgressBar(total > 0 ? (float)(l.vramUsed / double(st.vramTotal)) : 0, ImVec2(-1, 0),
                       std::format("{} / {}", FormatBytes(l.vramUsed), FormatBytes(st.vramTotal)).c_str());
    ImGui::Text("Shared");
    ImGui::SameLine(ImGui::GetFontSize() * 4);
    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, StatusColor(shared > 1.0f ? Status::Warn : Status::Info));
    ImGui::ProgressBar(std::min(1.0f, shared / std::max(1.0f, total)), ImVec2(-1, 0),
                       std::format("{:.2f} GB system RAM used by GPU{}", shared,
                                   shared > 1.0f ? " - possible sysmem fallback" : "").c_str());
    ImGui::PopStyleColor();
    (void)ded;

    // Who holds VRAM on the inference GPU (Windows per-process counters filtered to this adapter; Task Manager's
    // "Dedicated GPU memory" column sums all GPUs). Whatever the counters don't attribute is driver overhead.
    if (ImGui::TreeNodeEx("Who holds VRAM on this GPU", ImGuiTreeNodeFlags_DefaultOpen)) {
        auto procs = app.pdh.GpuProcesses();
        uint64_t sum = 0;
        for (const auto& p : procs) sum += p.bytes;
        auto note = [](const std::string& n) -> const char* {
            auto is = [&](const char* x) { return _stricmp(n.c_str(), x) == 0; };
            if (n.rfind("llama", 0) == 0) return "inference";
            if (is("dwm.exe")) return "Windows compositor: shared surfaces, mostly freed when no monitor uses this GPU";
            if (is("System") || is("csrss.exe")) return "Windows, not reclaimable";
            if (is("LlamaScope.exe")) return "this app (NVML/CUDA context)";
            if (is("brave.exe") || is("chrome.exe") || is("msedge.exe") || is("firefox.exe") || is("Code.exe") ||
                is("Discord.exe") || is("Teams.exe") || is("ms-teams.exe") || is("slack.exe"))
                return "app with GPU acceleration: Settings > System > Display > Graphics > Power saving moves it to "
                       "the integrated GPU";
            return "close it, or assign it to the integrated GPU in Settings > Graphics";
        };
        if (ImGui::BeginTable("##vramprocs", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("Program", ImGuiTableColumnFlags_WidthStretch, 0.9f);
            ImGui::TableSetupColumn("VRAM", ImGuiTableColumnFlags_WidthFixed, ImGui::GetFontSize() * 5);
            ImGui::TableSetupColumn("Note", ImGuiTableColumnFlags_WidthStretch, 2.2f);
            ImGui::TableHeadersRow();
            auto row = [](const std::string& name, uint64_t bytes, const char* n, bool dim) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                dim ? ImGui::TextDisabled("%s", name.c_str()) : ImGui::TextUnformatted(name.c_str());
                ImGui::TableNextColumn();
                ImGui::Text("%s", FormatBytes(bytes).c_str());
                ImGui::TableNextColumn();
                ImGui::TextDisabled("%s", n);
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", n);
            };
            for (const auto& p : procs) row(std::format("{} ({})", p.name, p.pid), p.bytes, note(p.name), false);
            if (l.vramUsed > sum)
                row("Driver + unattributed", l.vramUsed - sum,
                    "NVIDIA driver reservation (fixed, present on every card) plus memory Windows does not attribute "
                    "to any process (driver contexts, recently freed allocations)", true);
            ImGui::EndTable();
        }
        if (procs.empty()) ImGui::TextDisabled("Per-process counters unavailable.");
        ImGui::TreePop();
    }

    float h = std::max(ImGui::GetFontSize() * 6, (ImGui::GetContentRegionAvail().y - 12) / 4);
    const ImVec4 teal(0.18f, 0.80f, 0.75f, 1), violet(0.66f, 0.50f, 0.95f, 1), amber(0.90f, 0.70f, 0.16f, 1),
        red(0.95f, 0.35f, 0.30f, 1), blue(0.35f, 0.65f, 1.0f, 1);
    TimePlot("Utilization %", {{&app.nvml.util, "SM", teal}, {&app.nvml.memUtil, "memory", blue}}, 0, 100, h);
    TimePlot("PCIe MB/s", {{&app.nvml.rxMBs, "RX (to GPU)", violet}, {&app.nvml.txMBs, "TX (from GPU)", amber}}, 0, 0, h);
    TimePlot("Clock MHz", {{&app.nvml.gfxMHz, "graphics", teal}}, 0, std::max(1.0, st.maxGraphicsMHz * 1.05), h);
    TimePlot("Power W / Temp C", {{&app.nvml.powerW, "power", amber}, {&app.nvml.tempC, "temp", red}}, 0, 0, h);
    ImGui::End();
}
