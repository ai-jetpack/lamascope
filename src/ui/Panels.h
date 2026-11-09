#pragma once
#include <string>
#include <vector>
#include "core/Status.h"

class App;

void DrawHardwarePanel(App& app);
void DrawTopologyPanel(App& app);
void DrawCpuRuntimePanel(App& app);
void DrawGpuRuntimePanel(App& app);
void DrawTimelinePanel(App& app);
void DrawDiagnosisPanel(App& app);
void DrawLauncherPanel(App& app);
void DrawRunsPanel(App& app);
void DrawToolsPanel(App& app);
void DrawLogPanel(App& app);
void DrawInferenceModePanel(App& app);
void DrawPlatformDetails(App& app, bool* open);

// Hardware verdict rows, shared by the Hardware and Diagnosis panels.
struct HwRow {
    std::string section, label, value, spec;
    Verdict verdict;
};
std::vector<HwRow> BuildHardwareRows(App& app);

// Widgets
void StatusDot(Status s, const std::string& tooltip);
void PlaceholderText(const char* milestone, const char* text);
