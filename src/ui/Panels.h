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

// One-shot scroll requests (command line, for snapshots): topology diagram at layer.fraction, summary at fraction.
inline float g_topologyScroll = -1, g_summaryScroll = -1;
inline std::string g_applyCmdline;   // --cmdline "<llama.cpp command>": applied as if pasted

// Hardware verdict rows, shared by the Hardware and Diagnosis panels.
// Machine: fixed facts (Platform details). Setting: configuration that can be changed. Test: measured by Run tests.
// Runtime: live readings. Settings, tests and runtime rows are shown in the Hardware panel.
enum class HwKind { Machine, Setting, Test, Runtime };

struct HwRow {
    std::string section, label, value, spec;
    Verdict verdict;
    HwKind kind = HwKind::Machine;
};
std::vector<HwRow> BuildHardwareRows(App& app);

// Widgets
void StatusDot(Status s, const std::string& tooltip);
void PlaceholderText(const char* milestone, const char* text);
