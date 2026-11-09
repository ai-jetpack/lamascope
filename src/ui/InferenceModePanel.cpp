#include <windows.h>
#include <format>
#include "app/App.h"
#include "imgui.h"
#include "platform/Elevation.h"
#include "ui/Panels.h"

void DrawInferenceModePanel(App& app) {
    if (!ImGui::Begin(Win::Inference)) { ImGui::End(); return; }
    auto& im = app.inference;
    static bool probed = false;
    if (!probed && app.sys.elevated) { im.ProbeAsync(); probed = true; }

    ImGui::PushTextWrapPos(0);
    ImGui::TextDisabled("Temporary changes that keep background work away from inference. Everything is recorded "
                        "before it is changed and undone on Restore, when LlamaScope exits, or at the next start after a crash.");
    ImGui::PopTextWrapPos();

    if (!app.sys.elevated) {
        StatusDot(Status::Info, "");
        ImGui::SameLine();
        ImGui::TextUnformatted("Requires administrator rights.");
        ImGui::SameLine();
        if (ImGui::SmallButton("Restart as administrator") && RestartAsAdmin()) PostQuitMessage(0);
        ImGui::End();
        return;
    }

    bool busy = im.Busy(), active = im.Active();
    if (im.LeftoverJournal()) {
        StatusDot(Status::Warn, "");
        ImGui::SameLine();
        ImGui::TextWrapped("Settings from a previous session were not restored (LlamaScope did not exit normally).");
    }
    ImGui::BeginDisabled(busy);
    if (!active) {
        if (ImGui::Button("Enter inference mode")) im.EnterAsync();
    } else {
        if (ImGui::Button(im.LeftoverJournal() ? "Restore previous system state" : "Restore system state")) im.RestoreAsync();
    }
    ImGui::SameLine();
    if (ImGui::Button("Refresh")) im.ProbeAsync();
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (busy) ImGui::TextDisabled("working...");
    else if (active) {
        StatusDot(Status::Ok, "");
        ImGui::SameLine();
        ImGui::Text("Active: %d programs moved to E-cores", im.MovedCount());
    }

    if (ImGui::BeginTable("imactions", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerH)) {
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, ImGui::GetFontSize() * 1.6f);
        ImGui::TableSetupColumn("Change", ImGuiTableColumnFlags_WidthStretch, 1.2f);
        ImGui::TableSetupColumn("Now", ImGuiTableColumnFlags_WidthStretch, 1.6f);
        ImGui::TableSetupColumn("Risk", ImGuiTableColumnFlags_WidthFixed, ImGui::GetFontSize() * 4);
        ImGui::TableHeadersRow();
        for (const auto& a : im.Actions()) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            bool on = a.enabled;
            ImGui::BeginDisabled(active || busy);
            if (ImGui::Checkbox(("##" + a.id).c_str(), &on)) im.SetEnabled(a.id, on);
            ImGui::EndDisabled();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(a.label.c_str());
            if (ImGui::IsItemHovered()) {
                ImGui::BeginTooltip();
                ImGui::PushTextWrapPos(ImGui::GetFontSize() * 30);
                ImGui::TextUnformatted(a.effect.c_str());
                ImGui::PopTextWrapPos();
                ImGui::EndTooltip();
            }
            ImGui::TableNextColumn();
            ImGui::PushTextWrapPos(0);
            ImGui::TextDisabled("%s", a.state.empty() ? "..." : a.state.c_str());
            ImGui::PopTextWrapPos();
            ImGui::TableNextColumn();
            ImGui::TextColored(StatusColor(a.risk == "low" ? Status::Ok : Status::Warn), "%s", a.risk.c_str());
        }
        ImGui::EndTable();
    }
    ImGui::TextDisabled("Hover a change for details. Defender exclusions are off by default (security trade-off).");

    auto notMoved = im.NotMoved();
    if (active && !notMoved.empty()) {
        std::string s;
        for (auto& n : notMoved) s += (s.empty() ? "" : ", ") + n;
        ImGui::PushTextWrapPos(0);
        ImGui::TextDisabled("Could not be moved (protected by Windows): %s", s.c_str());
        ImGui::PopTextWrapPos();
    }
    // Warnings stay visible (e.g. llama.cpp started without the P-cores and must be restarted).
    if (active)
        for (const auto& l : im.Log())
            if (l.find("WARNING ") != std::string::npos) {
                ImGui::PushTextWrapPos(0);
                ImGui::TextColored(StatusColor(Status::Warn), "%s", l.c_str());
                ImGui::PopTextWrapPos();
            }
    if (ImGui::CollapsingHeader("Activity")) {
        for (const auto& l : im.Log()) ImGui::BulletText("%s", l.c_str());
    }
    ImGui::End();
}
