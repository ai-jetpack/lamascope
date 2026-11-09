#include <format>
#include "app/App.h"
#include "core/Log.h"
#include "imgui.h"
#include "ui/Panels.h"

void StatusDot(Status s, const std::string& tooltip) {
    if (s == Status::None) {
        ImGui::Dummy(ImVec2(ImGui::GetFontSize() * 0.6f, ImGui::GetFontSize()));
        return;
    }
    float r = ImGui::GetFontSize() * 0.3f;
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImGui::Dummy(ImVec2(r * 2 + 2, ImGui::GetFontSize()));
    ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(p.x + r + 1, p.y + ImGui::GetFontSize() * 0.5f), r,
                                                ImGui::GetColorU32(StatusColor(s)));
    if (!tooltip.empty() && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tooltip.c_str());
}

void PlaceholderText(const char* milestone, const char* text) {
    ImGui::TextColored(StatusColor(Status::Info), "[%s]", milestone);
    ImGui::SameLine();
    ImGui::TextWrapped("%s", text);
}

void DrawTimelinePanel(App&) {
    if (ImGui::Begin(Win::Timeline))
        PlaceholderText("M6", "Per-token swimlanes (GPU busy, PCIe, each CPU thread work/spin) and the per-token "
                              "time budget bar.");
    ImGui::End();
}

void DrawDiagnosisPanel(App& app) {
    if (!ImGui::Begin(Win::Diagnosis)) { ImGui::End(); return; }
    // Session-aggregated findings, refreshed once per second by the DiagnosisEngine.
    auto findings = app.diag.Findings();
    for (const auto& f : findings) {
        StatusDot(f.active ? f.severity : Status::Unavailable, f.detail);
        ImGui::SameLine();
        if (!f.active) ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
        ImGui::TextWrapped("%s", f.text.c_str());
        if (!f.active) ImGui::PopStyleColor();
        if (!f.detail.empty() && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", f.detail.c_str());
    }
    if (findings.empty()) {
        StatusDot(Status::Ok, "");
        ImGui::SameLine();
        ImGui::TextWrapped("No issues so far this session. Run the memory and PCIe tests for a full check.");
    }
    ImGui::Spacing();
    ImGui::TextDisabled("Issues must persist %.0f s to be reported; grey = resolved. Hover for the latest reading.",
                        DiagnosisEngine::kStartSec);
    ImGui::End();
}

void DrawLauncherPanel(App&) {
    if (ImGui::Begin(Win::Launcher))
        PlaceholderText("M3", "Pick build, model and tool (cli / server / bench), validated flags, launch and attach, "
                              "live tok/s.");
    ImGui::End();
}

void DrawRunsPanel(App&) {
    if (ImGui::Begin(Win::Runs))
        PlaceholderText("M3", "Every run saved with command, build, env vars and hardware snapshot; diff any two runs.");
    ImGui::End();
}

void DrawToolsPanel(App&) {
    if (ImGui::Begin(Win::Tools))
        PlaceholderText("M2", "Install and detect: llama.cpp CUDA builds, HWiNFO, Intel MLC, VTune, Nsight Systems, "
                              "build toolchain.");
    ImGui::End();
}

void DrawLogPanel(App&) {
    if (!ImGui::Begin(Win::Log)) { ImGui::End(); return; }
    for (const auto& e : Log::Snapshot()) {
        ImVec4 c = e.level == Log::Error ? StatusColor(Status::Bad)
                   : e.level == Log::Warn ? StatusColor(Status::Warn)
                                          : ImGui::GetStyleColorVec4(ImGuiCol_Text);
        ImGui::TextColored(c, "%7.1fs  %s", e.t, e.msg.c_str());
    }
    if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY()) ImGui::SetScrollHereY(1.0f);
    ImGui::End();
}
