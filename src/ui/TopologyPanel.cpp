// Model topology: whole-model pipeline strip + per-layer architecture diagram + memory/speed summary.
#include <windows.h>
#include <commdlg.h>
#include <algorithm>
#include <cmath>
#include <format>
#include "app/App.h"
#include "core/Log.h"
#include "core/Util.h"
#include "imgui.h"
#include "implot.h"
#include "ui/Panels.h"

namespace {

const ImVec4 kGpu(0.13f, 0.50f, 0.52f, 1), kCpu(0.78f, 0.45f, 0.16f, 1), kPcie(0.66f, 0.50f, 0.95f, 1);
const ImU32 kEdge = IM_COL32(170, 178, 190, 255);
const ImU32 kText = IM_COL32(245, 245, 245, 255), kSub = IM_COL32(225, 230, 235, 190);

std::string MB(double bytes) {
    if (bytes >= 10.0 * (1 << 30)) return std::format("{:.1f} GB", bytes / (1 << 30));
    if (bytes >= (1 << 30)) return std::format("{:.2f} GB", bytes / (1 << 30));
    if (bytes >= 10.0 * (1 << 20)) return std::format("{:.0f} MB", bytes / (1 << 20));
    if (bytes >= (1 << 20)) return std::format("{:.1f} MB", bytes / (1 << 20));
    return std::format("{:.0f} KB", bytes / 1024);
}

ImU32 DevCol(Device d, float a = 1.0f) {
    ImVec4 c = d == Device::Gpu ? kGpu : kCpu;
    c.w = a;
    return ImGui::GetColorU32(c);
}
const char* DevName(Device d) { return d == Device::Gpu ? "GPU" : "CPU"; }

std::string Us(double v);

std::wstring BrowseForProfile() {
    wchar_t file[MAX_PATH] = {};
    std::wstring dir = App::ProfilesDir();
    OPENFILENAMEW ofn{sizeof(ofn)};
    ofn.lpstrFilter = L"ggml profiles\0*.prof\0All files\0*.*\0";
    ofn.lpstrFile = file;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrInitialDir = dir.c_str();
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
    return GetOpenFileNameW(&ofn) ? std::wstring(file) : std::wstring();
}

// Measured runtime (ggml profiler output): totals per generation step and expert usage across all layers.
void DrawProfileSummary(App& app, const ModelInfo& m) {
    ImGui::SeparatorText("Runtime profile (measured)");
    if (ImGui::Button("Open profile...")) {
        std::wstring p = BrowseForProfile();
        if (!p.empty()) app.LoadProfile(p);
    }
    ImGui::SameLine();
    if (ImGui::Button("Load latest") && !app.LoadLatestProfile())
        app.profile = ProfileData{false, "no .prof files in " + ToUtf8(App::ProfilesDir())};
    const ProfileData& pr = app.profile;
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Run a patched llama.cpp with GGML_PROFILE=block (or op) and\nGGML_PROFILE_FILE=%s\\<name>.prof", ToUtf8(App::ProfilesDir()).c_str());
    if (!pr.ok) {
        if (!pr.error.empty()) ImGui::TextColored(StatusColor(Status::Warn), "%s", pr.error.c_str());
        else ImGui::TextDisabled("No profile loaded. Times per block, PCIe latencies and expert usage appear on the diagram once one is loaded.");
        return;
    }
    ImGui::SameLine();
    if (ImGui::Button("Clear")) { app.profile = ProfileData{}; return; }
    std::string file = pr.path.substr(pr.path.find_last_of("\\/") + 1);
    ImGui::TextDisabled("%s: %d generation steps averaged, %d CPU threads, %s mode%s", file.c_str(), pr.genSteps, pr.threads,
                        pr.level == 2 ? "op" : "block", pr.dropped ? std::format(", {} entries dropped", pr.dropped).c_str() : "");
    if (pr.nExperts && m.nExpert && pr.nExperts > m.nExpert)
        ImGui::TextColored(StatusColor(Status::Warn), "The profile has %d experts per layer, the selected model %u: wrong model?", pr.nExperts, m.nExpert);

    const double step = pr.stepUs;
    if (pr.genSteps == 0) ImGui::TextDisabled("Expert counts only (no timing in this profile).");
    if (pr.genSteps > 0) {
    ImGui::Text("Per generated token: %s (%.1f tokens/s)", Us(step).c_str(), step > 0 ? 1e6 / step : 0.0);
    auto row = [&](const char* label, double v, const char* tip) {
        ImGui::BulletText("%s: %s (%.0f%%)", label, Us(v).c_str(), step > 0 ? v / step * 100 : 0.0);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tip);
    };
    // Non-overlapping parts of a step (they add up to the step, the remainder is launch/scheduling overhead).
    row("CPU busy (experts on the CPU)", pr.cpuBusyUs, "Thread 0 wall time in CPU blocks: computing plus waiting for the other threads at each step.");
    if (pr.hasGpu) row("GPU busy", pr.gpuBusyUs, "Sum of GPU block times from marker kernels.");
    if (pr.hasGpu) {
        row("PCIe hand-offs", pr.handoffUs, "Copies between CPU and GPU, counted from the moment the GPU finished (the PCIe labels on the diagram).");
        double other = pr.stepUs - pr.cpuBusyUs - pr.gpuBusyUs - pr.handoffUs;
        row("Other (launches, scheduling, unmeasured)", std::max(0.0, other), "Step time not covered by the rows above: kernel launches, graph replay, scheduler bookkeeping.");
        ImGui::TextDisabled("Main thread blocked until the GPU finished: %s per token (already part of GPU busy).", Us(pr.waitGpuUs).c_str());
    } else {
        row("Copies and waits for the GPU", pr.copyUs, "Input copies of each split, including waiting for the GPU (no GPU markers in this profile to separate them).");
    }
    if (pr.syncUs > 0) row("Explicit syncs", pr.syncUs, "ggml_backend_synchronize calls between splits.");

    // Hand-off details: every input tensor copied between devices.
    if (!pr.copies.empty() && ImGui::TreeNode("Hand-off details (each copied tensor)")) {
        static const char* paths[] = {"async", "sync + blocking copy", "user input", "callback"};
        if (ImGui::BeginTable("##copies", 7, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_Borders)) {
            for (const char* h : {"tensor", "direction", "per token", "bytes", "how", "sync / copy (each)", "per token total"})
                ImGui::TableSetupColumn(h);
            ImGui::TableHeadersRow();
            for (const auto& c : pr.copies) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn(); ImGui::TextUnformatted(c.tensor.c_str());
                ImGui::TableNextColumn(); ImGui::TextUnformatted(c.toCpu ? "GPU -> CPU" : "CPU -> GPU");
                ImGui::TableNextColumn(); ImGui::Text("%.0f", c.perStep);
                ImGui::TableNextColumn(); ImGui::TextUnformatted(c.bytes ? MB((double)c.bytes).c_str() : "?");
                ImGui::TableNextColumn(); ImGui::TextUnformatted(paths[std::clamp(c.path, 0, 3)]);
                ImGui::TableNextColumn(); ImGui::Text("%s / %s", Us(c.syncUs).c_str(), Us(c.copyUs).c_str());
                ImGui::TableNextColumn(); ImGui::TextUnformatted(Us(c.perStep * (c.syncUs + c.copyUs)).c_str());
            }
            ImGui::EndTable();
        }
        ImGui::TextDisabled("\"sync\" = waiting in ggml_backend_synchronize before the copy (for the first input of a CPU split this\nincludes the GPU finishing its work); \"copy\" = the blocking data copy itself.");
        ImGui::TreePop();
    }

    // Spin share over all CPU blocks
    double tot = 0, wsum = 0;
    for (const auto& L : pr.layers)
        for (size_t b = 0; b < (size_t)ProfBlock::Count; ++b) { tot += L.cpuUs[b]; wsum += L.cpuUs[b] * L.spinPct[b] / 100; }
    if (tot > 0) ImGui::BulletText("CPU threads waiting at barriers: %.0f%% of their time in CPU blocks", wsum / tot * 100);
    }

    // Expert usage heatmap: rows = layers, columns = experts.
    static bool prompt = false;
    const auto& counts = prompt ? pr.promptCounts : pr.genCounts;
    if (!counts.empty() && pr.nExperts > 0) {
        ImGui::Checkbox("Prompt processing instead of generation", &prompt);
        static std::vector<float> heat;
        const int rows = (int)counts.size(), cols = pr.nExperts;
        heat.assign((size_t)rows * cols, 0);
        float mx = 0;
        for (int l = 0; l < rows; ++l) {
            uint64_t t = 0;
            for (uint32_t c : counts[l]) t += c;
            for (int e = 0; e < cols; ++e) {
                float v = t ? (float)counts[l][e] / t * 100.0f : 0;   // % of this layer's picks
                heat[(size_t)l * cols + e] = v;
                mx = std::max(mx, v);
            }
        }
        if (ImPlot::BeginPlot("##experts", ImVec2(-1, std::max(ImGui::GetFontSize() * 14, rows * 4.0f)), ImPlotFlags_NoMouseText | ImPlotFlags_NoLegend)) {
            ImPlot::SetupAxes("expert", "layer", ImPlotAxisFlags_None, ImPlotAxisFlags_Invert);
            ImPlot::SetupAxesLimits(0, cols, 0, rows, ImPlotCond_Always);
            ImPlot::PushColormap(ImPlotColormap_Hot);
            ImPlot::PlotHeatmap("##h", heat.data(), rows, cols, 0, mx, nullptr, ImPlotPoint(0, rows), ImPlotPoint(cols, 0));
            ImPlot::PopColormap();
            if (ImPlot::IsPlotHovered()) {
                ImPlotPoint mp = ImPlot::GetPlotMousePos();
                int e = std::clamp((int)mp.x, 0, cols - 1), l = std::clamp((int)mp.y, 0, rows - 1);
                ImGui::SetTooltip("layer %d, expert #%d: %u picks (%.2f%% of the layer's picks)", l, e, counts[l][e], heat[(size_t)l * cols + e]);
            }
            ImPlot::EndPlot();
        }
        int never = 0;
        for (const auto& r : counts) for (uint32_t c : r) never += c == 0;
        ImGui::TextDisabled("Brighter = picked more often. %d of %d expert slots were never picked in this run.", never, rows * cols);
    }
}

std::wstring BrowseForModel() {
    wchar_t file[MAX_PATH] = {};
    OPENFILENAMEW ofn{sizeof(ofn)};
    ofn.lpstrFilter = L"GGUF models\0*.gguf\0All files\0*.*\0";
    ofn.lpstrFile = file;
    ofn.nMaxFile = MAX_PATH;
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
    return GetOpenFileNameW(&ofn) ? std::wstring(file) : std::wstring();
}

uint64_t OtherProcessesVram(App& app) {
    uint64_t s = 0;
    for (const auto& p : app.pdh.GpuProcesses())
        if (p.name.rfind("llama", 0) != 0) s += p.bytes;
    return s;
}

double CpuBandwidth(App& app, bool* measured) {
    auto r = app.memTest.Result();
    *measured = r.valid;
    return r.valid ? r.bestGBs * 1e9 : app.mem.theoreticalGBs * 0.9e9;
}

// ------------------------------------------------------------------------------------------------
// Drawing primitives (screen coordinates)

struct Rect {
    ImVec2 a, b;
    ImVec2 Top() const { return ImVec2((a.x + b.x) * 0.5f, a.y); }
    ImVec2 Bottom() const { return ImVec2((a.x + b.x) * 0.5f, b.y); }
    ImVec2 Left() const { return ImVec2(a.x, (a.y + b.y) * 0.5f); }
    ImVec2 Right() const { return ImVec2(b.x, (a.y + b.y) * 0.5f); }
    float W() const { return b.x - a.x; }
};

Rect MakeRect(float cx, float y, float w, float h) { return {ImVec2(cx - w * 0.5f, y), ImVec2(cx + w * 0.5f, y + h)}; }

void CenterText(ImDrawList* dl, ImVec2 c, ImU32 col, const std::string& s, float maxW) {
    if (s.empty()) return;
    ImVec2 ts = ImGui::CalcTextSize(s.c_str());
    if (ts.x > maxW) return;   // never draw text outside its box
    dl->AddText(ImVec2(c.x - ts.x * 0.5f, c.y - ts.y * 0.5f), col, s.c_str());
}

// A component box: device-colored fill, title line, optional subtitle line. Returns hovered.
bool Box(ImDrawList* dl, const Rect& r, ImU32 fill, const std::string& title, const std::string& sub = "",
         bool dashed = false) {
    float f = ImGui::GetFontSize();
    dl->AddRectFilled(r.a, r.b, fill, 5.0f);
    dl->AddRect(r.a, r.b, dashed ? IM_COL32(255, 255, 255, 90) : IM_COL32(255, 255, 255, 50), 5.0f, 0, 1.0f);
    float cy = (r.a.y + r.b.y) * 0.5f;
    std::string t = title;
    if (ImGui::CalcTextSize(t.c_str()).x > r.W() - 6 && t.size() > 3) {   // shorten to fit
        while (t.size() > 2 && ImGui::CalcTextSize((t + ".").c_str()).x > r.W() - 6) t.pop_back();
        t += ".";
    }
    bool showSub = !sub.empty() && (r.b.y - r.a.y) > f * 2.1f;
    CenterText(dl, ImVec2(r.Top().x, showSub ? cy - f * 0.5f : cy), kText, t, r.W() - 4);
    if (showSub) CenterText(dl, ImVec2(r.Top().x, cy + f * 0.55f), kSub, sub, r.W() - 4);
    return ImGui::IsMouseHoveringRect(r.a, r.b);
}

void ArrowHead(ImDrawList* dl, ImVec2 tip, ImVec2 dir, ImU32 col, float s) {
    ImVec2 n(-dir.y, dir.x);
    dl->AddTriangleFilled(tip, ImVec2(tip.x - dir.x * s + n.x * s * 0.55f, tip.y - dir.y * s + n.y * s * 0.55f),
                          ImVec2(tip.x - dir.x * s - n.x * s * 0.55f, tip.y - dir.y * s - n.y * s * 0.55f), col);
}

// Orthogonal connector: vertical, horizontal at mid height, vertical. PCIe edges are violet and thicker.
void Edge(ImDrawList* dl, ImVec2 from, ImVec2 to, bool pcie, float midFrac = 0.5f) {
    ImU32 col = pcie ? ImGui::GetColorU32(kPcie) : kEdge;
    float th = pcie ? 2.4f : 1.4f, s = ImGui::GetFontSize() * 0.38f;
    to.y -= s * 0.2f;
    if (std::fabs(from.x - to.x) < 1.0f) {
        dl->AddLine(from, ImVec2(to.x, to.y - s), col, th);
    } else {
        float my = from.y + (to.y - from.y) * midFrac;
        dl->AddLine(from, ImVec2(from.x, my), col, th);
        dl->AddLine(ImVec2(from.x, my), ImVec2(to.x, my), col, th);
        dl->AddLine(ImVec2(to.x, my), ImVec2(to.x, to.y - s), col, th);
    }
    ArrowHead(dl, to, ImVec2(0, 1), col, s);
}

void DashedLine(ImDrawList* dl, ImVec2 a, ImVec2 b, ImU32 col) {
    float len = std::hypot(b.x - a.x, b.y - a.y), step = 6;
    for (float t = 0; t < len; t += step * 2) {
        float t1 = std::min(len, t + step);
        dl->AddLine(ImVec2(a.x + (b.x - a.x) * t / len, a.y + (b.y - a.y) * t / len),
                    ImVec2(a.x + (b.x - a.x) * t1 / len, a.y + (b.y - a.y) * t1 / len), col, 1.2f);
    }
}

void PcieLabel(ImDrawList* dl, ImVec2 at, const std::string& s) {
    ImVec2 ts = ImGui::CalcTextSize(s.c_str());
    ImVec2 p(at.x + 6, at.y - ts.y * 0.5f);
    dl->AddRectFilled(ImVec2(p.x - 3, p.y - 1), ImVec2(p.x + ts.x + 3, p.y + ts.y + 1), IM_COL32(30, 24, 50, 230), 3);
    dl->AddText(p, ImGui::GetColorU32(kPcie), s.c_str());
}

void SumNode(ImDrawList* dl, ImVec2 c, float r, Device d, const char* sym) {
    dl->AddCircleFilled(c, r, DevCol(d));
    dl->AddCircle(c, r, IM_COL32(255, 255, 255, 80), 0, 1.0f);
    CenterText(dl, c, kText, sym, r * 2);
}

void Tip(const std::vector<std::string>& lines) {
    ImGui::BeginTooltip();
    for (const auto& l : lines) ImGui::TextUnformatted(l.c_str());
    ImGui::EndTooltip();
}

// Runtime profile overlay (GGML_PROFILE): time badges, PCIe latencies, expert usage strip.
const ImU32 kBadge = IM_COL32(255, 214, 120, 255);
std::string Us(double v) {
    return v >= 1000 ? std::format("{:.2f} ms", v / 1000) : std::format("{:.0f} \u00b5s", v);
}
void Badge(ImDrawList* dl, ImVec2 at, const std::string& s) {
    ImVec2 sz = ImGui::CalcTextSize(s.c_str());
    dl->AddRectFilled(ImVec2(at.x - 3, at.y - 1), ImVec2(at.x + sz.x + 3, at.y + sz.y + 1), IM_COL32(60, 48, 20, 230), 3.0f);
    dl->AddText(at, kBadge, s.c_str());
}
const ProfileData::Layer* ProfLayer(const ProfileData* pr, int li) {
    return pr && pr->ok && li < (int)pr->layers.size() ? &pr->layers[li] : nullptr;
}
std::string BlockTime(const ProfileData::Layer& L, ProfBlock b) {
    double g = L.gpuUs[(size_t)b], c = L.cpuUs[(size_t)b];
    std::string s;
    if (g > 0.5) s += "GPU " + Us(g);
    if (c > 0.5) s += (s.empty() ? "" : " + ") + std::string("CPU ") + Us(c) + std::format(" ({:.0f}% waiting)", L.spinPct[(size_t)b]);
    return s;
}
// One thin cell per expert, brighter = picked more often during generation.
void ExpertStrip(ImDrawList* dl, ImVec2 a, float w, float h, const std::vector<uint32_t>& counts, int layer) {
    uint32_t mx = 0;
    uint64_t total = 0;
    for (uint32_t c : counts) { mx = std::max(mx, c); total += c; }
    if (!mx) return;
    const int n = (int)counts.size();
    const float cw = w / n;
    for (int e = 0; e < n; ++e) {
        float t = std::sqrt((float)counts[e] / mx);
        ImU32 col = counts[e] ? ImGui::ColorConvertFloat4ToU32(ImVec4(0.25f + 0.75f * t, 0.18f + 0.6f * t, 0.10f, 1)) : IM_COL32(35, 38, 45, 255);
        dl->AddRectFilled(ImVec2(a.x + e * cw, a.y), ImVec2(a.x + (e + 1) * cw, a.y + h), col);
    }
    dl->AddRect(a, ImVec2(a.x + w, a.y + h), IM_COL32(255, 255, 255, 60));
    if (ImGui::IsMouseHoveringRect(a, ImVec2(a.x + w, a.y + h))) {
        int e = std::clamp((int)((ImGui::GetMousePos().x - a.x) / cw), 0, n - 1);
        std::vector<int> idx(n);
        for (int i = 0; i < n; ++i) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + std::min(5, n), idx.end(), [&](int x, int y) { return counts[x] > counts[y]; });
        std::vector<std::string> t = {std::format("Layer {}: expert #{} picked {} times ({:.2f}% of picks)", layer, e, counts[e], 100.0 * counts[e] / total)};
        int unused = 0;
        for (uint32_t c : counts) unused += c == 0;
        t.push_back(std::format("{} of {} experts never picked in this run", unused, n));
        t.push_back("Most used:");
        for (int k = 0; k < std::min(5, n); ++k)
            t.push_back(std::format("  #{}: {:.2f}%", idx[k], 100.0 * counts[idx[k]] / total));
        Tip(t);
    }
}

std::vector<std::string> TensorLines(const LayerInfo& L, Block b) {
    std::vector<std::string> out;
    for (const auto& t : L.tensors[(int)b])
        out.push_back(std::format("  {:<28} {:<6} {}", t.name, GgmlTypeName(t.type), MB((double)t.bytes)));
    return out;
}

// ------------------------------------------------------------------------------------------------
// Whole-model pipeline: embedding -> 48 layer tiles -> output head. Click a tile to select the layer.

void DrawPipeline(const ModelInfo& m, const Placement& pl, int& sel, int& scrollTo) {
    const float f = ImGui::GetFontSize();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 o = ImGui::GetCursorScreenPos();
    float W = ImGui::GetContentRegionAvail().x;
    float endW = f * 5.2f, gap = f * 0.6f, tileH = f * 2.6f;
    int L = (int)m.layers.size();
    float tileW = std::max(4.0f, (W - endW * 2 - gap * 4) / std::max(1, L));
    float y = o.y + f * 0.2f;

    Rect emb{ImVec2(o.x, y), ImVec2(o.x + endW, y + tileH)};
    if (Box(dl, emb, DevCol(Device::Cpu), "Lookup", "token -> vector"))
        Tip({"Turns the token id into a vector (table lookup, always on the CPU in llama.cpp)",
             MB((double)m.embdBytes) + " table, one row read per token"});
    float x = emb.b.x + gap;
    dl->AddLine(ImVec2(emb.b.x, y + tileH * 0.5f), ImVec2(x - 2, y + tileH * 0.5f), kEdge, 1.4f);

    for (int i = 0; i < L; ++i) {
        const auto& d = pl.dev[i];
        bool hasExp = m.layers[i].bytes[(int)Block::Experts] > 0;
        float x0 = x + i * tileW, x1 = x0 + tileW - 1;
        dl->AddRectFilled(ImVec2(x0, y), ImVec2(x1, y + tileH * 0.5f), DevCol(d[(int)Block::Mixer]));
        dl->AddRectFilled(ImVec2(x0, y + tileH * 0.5f), ImVec2(x1, y + tileH),
                          DevCol(hasExp ? d[(int)Block::Experts] : d[(int)Block::DenseFfn]));
        if (m.layers[i].mixer == MixerKind::Attention)   // full-attention layers get a white cap
            dl->AddRectFilled(ImVec2(x0, y - 3), ImVec2(x1, y), IM_COL32(255, 255, 255, 200));
        if (i == sel) dl->AddRect(ImVec2(x0 - 1, y - 4), ImVec2(x1 + 1, y + tileH + 2), IM_COL32(255, 255, 255, 255), 0, 0, 2.0f);
        if (ImGui::IsMouseHoveringRect(ImVec2(x0, y - 4), ImVec2(x1, y + tileH))) {
            Tip({std::format("Layer {}  ({})", i, m.layers[i].mixer == MixerKind::Attention ? "has a KV cache" : "fixed-size state"),
                 std::format("top: context stage on {}   bottom: experts on {}", DevName(d[(int)Block::Mixer]),
                             DevName(d[hasExp ? (int)Block::Experts : (int)Block::DenseFfn])),
                 "click to scroll the pipeline to this layer"});
            if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) { sel = i; scrollTo = i; }
        }
    }
    float xe = x + L * tileW + gap;
    dl->AddLine(ImVec2(x + L * tileW, y + tileH * 0.5f), ImVec2(xe - 2, y + tileH * 0.5f), kEdge, 1.4f);
    Rect out{ImVec2(xe, y), ImVec2(xe + endW, y + tileH)};
    Device od = pl.outputOnGpu ? Device::Gpu : Device::Cpu;
    if (Box(dl, out, DevCol(od), "Pick token", "vector -> token"))
        Tip({std::format("Scores all {} possible next tokens and picks one", m.nVocab),
             std::format("reads {} per token, on the {}", MB((double)m.outputBytes), DevName(od))});

    // Brackets under runs of identical placement, e.g. "layers 0-35: experts on CPU".
    float by = y + tileH + f * 0.35f;
    int start = 0;
    for (int i = 1; i <= L; ++i) {
        auto key = [&](int k) { return std::make_pair(pl.dev[k][(int)Block::Mixer], pl.dev[k][(int)Block::Experts]); };
        if (i < L && key(i) == key(start)) continue;
        float x0 = x + start * tileW, x1 = x + i * tileW - 2;
        dl->AddLine(ImVec2(x0, by), ImVec2(x1, by), kEdge, 1.0f);
        dl->AddLine(ImVec2(x0, by - 3), ImVec2(x0, by + 1), kEdge);
        dl->AddLine(ImVec2(x1, by - 3), ImVec2(x1, by + 1), kEdge);
        auto [md, ed] = key(start);
        std::string s = md == ed ? std::format("layers {}-{}: all on {}", start, i - 1, DevName(md))
                                 : std::format("layers {}-{}: experts on {}", start, i - 1, DevName(ed));
        if (ImGui::CalcTextSize(s.c_str()).x > x1 - x0) s = std::format("{}-{}", start, i - 1);
        CenterText(dl, ImVec2((x0 + x1) * 0.5f, by + f * 0.7f), kSub, s, x1 - x0 + f * 2);
        start = i;
    }
    ImGui::Dummy(ImVec2(W, tileH + f * 1.9f));
}

// ------------------------------------------------------------------------------------------------
// One layer as an architecture diagram.

Device moeDev(const ModelInfo& m, const Placement& pl, int li) {
    return pl.dev[li][m.layers[li].bytes[(int)Block::Experts] ? (int)Block::Experts : (int)Block::DenseFfn];
}

void DrawLayerDiagram(const ModelInfo& m, const Placement& pl, const LaunchParams& lp, int li, const ProfileData* pr) {
    const LayerInfo& L = m.layers[li];
    const auto& d = pl.dev[li];
    const ProfileData::Layer* P = ProfLayer(pr, li);
    const float f = ImGui::GetFontSize();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 o = ImGui::GetCursorScreenPos();
    float W = ImGui::GetContentRegionAvail().x;
    float cx = o.x + W * 0.5f;
    float vgap = f * 1.5f;
    float y = o.y + f * 0.3f;

    Device ld = pl.layerDev[li], md = d[(int)Block::Mixer], ed = d[(int)Block::Experts], sd = d[(int)Block::Shared],
           rd = d[(int)Block::Router];
    bool moe = L.bytes[(int)Block::Experts] > 0;
    bool attn = L.mixer == MixerKind::Attention;
    int attnLayers = std::max(1, m.CountMixer(MixerKind::Attention)), linLayers = std::max(1, m.CountMixer(MixerKind::Linear));
    std::string vec = std::format("{} values ({})", m.nEmbd, MB((double)m.nEmbd * 4));

    // Section header band for this layer.
    {
        Device exd = moeDev(m, pl, li);
        std::string h = std::format("Layer {}   |   context stage on {}   |   experts on {}   |   {}", li, DevName(md),
                                    DevName(exd), attn ? "has a KV cache" : "fixed-size state");
        if (P) {
            double g = 0, c = 0;
            for (size_t b = 0; b < (size_t)ProfBlock::Count; ++b) { g += P->gpuUs[b]; c += P->cpuUs[b]; }
            h += std::format("   |   measured per token: GPU {}, CPU {}", Us(g), Us(c));
        }
        ImVec2 a(o.x, y), b(o.x + W, y + f * 1.5f);
        dl->AddRectFilled(a, b, IM_COL32(40, 46, 58, 255), 3.0f);
        dl->AddText(ImVec2(a.x + f * 0.5f, a.y + f * 0.25f), IM_COL32(220, 225, 235, 255), h.c_str());
        y = b.y + f * 0.5f;
    }
    // Entry point (the vector arriving from the previous layer); the bypass path starts here.
    Rect in{ImVec2(cx - 3, y), ImVec2(cx + 3, y + 6)};
    dl->AddCircleFilled(ImVec2(cx, y + 3), 3.5f, kEdge);
    y = in.b.y + vgap * 0.7f;

    // Normalize + context stage (+ its store)
    Rect n1 = MakeRect(cx, y, f * 7, f * 1.5f);
    if (Box(dl, n1, DevCol(md, 0.8f), "Normalize")) Tip({"Small scaling step (negligible cost)"});
    Edge(dl, in.Bottom(), n1.Top(), false);
    y = n1.b.y + vgap;

    // Context stage, drawn as three steps around its store:
    //   cache layers: make query/key/value -> APPEND key+value to the KV cache -> LOOK UP all stored keys -> output
    //   state layers: make inputs -> WRITE the fixed-size state store -> READ result -> output
    float mixW = std::min(W * 0.46f, f * 20);
    auto isOutStep = [](const std::string& n) {
        return n.rfind("attn_output", 0) == 0 || n.rfind("ssm_out", 0) == 0 || n.rfind("ssm_norm", 0) == 0;
    };
    double inBytes = 0, outBytes = 0;
    std::vector<std::string> inT, outT;
    for (const auto& t : L.tensors[(int)Block::Mixer]) {
        std::string line = std::format("  {:<24} {:<6} {}", t.name, GgmlTypeName(t.type), MB((double)t.bytes));
        if (isOutStep(t.name)) { outBytes += t.bytes; outT.push_back(line); }
        else { inBytes += t.bytes; inT.push_back(line); }
    }
    const auto& kvt = KvTypes();
    double kvPerTok = m.nHeadKv * (m.keyLen * kvt[lp.ctk].bytesPerElement + m.valLen * kvt[lp.ctv].bytesPerElement);
    double kvLayer = (double)pl.kvBytes / attnLayers, stLayer = (double)pl.rsBytes / linLayers;

    float stepTop = y;
    Rect A = MakeRect(cx, y, mixW, f * 2.4f);
    if (Box(dl, A, DevCol(md), attn ? "Make query, key, value" : "Make update for the state",
            std::format("reads {} of weights", MB(inBytes)))) {
        std::vector<std::string> t = {attn ? "Turns this token's vector into a query (what to look for), a key and a value"
                                           : "Turns this token's vector into an update for the layer's state",
                                      std::string("Runs on the ") + DevName(md), "Tensors:"};
        t.insert(t.end(), inT.begin(), inT.end());
        Tip(t);
    }
    Edge(dl, n1.Bottom(), A.Top(), false);
    y = A.b.y + vgap;
    Rect B = MakeRect(cx, y, mixW, f * 2.4f);
    if (Box(dl, B, DevCol(md, 0.85f), attn ? "Look up: match query to all keys" : "Read the updated state",
            attn ? "blend the matching values" : "fixed cost per token"))
        Tip({attn ? "Compares the query with the key of every earlier token in the KV cache, then blends their values."
                  : "Reads the layer's state after the update; no history is scanned.",
             attn ? std::format("Reads the whole cache for this layer: {} at -c {}. Grows as the conversation grows.", MB(kvLayer), lp.nCtx)
                  : std::format("Reads {} of state, at any context length.", MB(stLayer))});
    Edge(dl, A.Bottom(), B.Top(), false);
    y = B.b.y + vgap;
    Rect C = MakeRect(cx, y, mixW, f * 2.4f);
    if (Box(dl, C, DevCol(md), "Output step", std::format("reads {} of weights", MB(outBytes)))) {
        std::vector<std::string> t = {"Projects the result back onto the main path", "Tensors:"};
        t.insert(t.end(), outT.begin(), outT.end());
        Tip(t);
    }
    Edge(dl, B.Bottom(), C.Top(), false);
    Rect mix = C;

    // Frame grouping the three steps.
    ImVec2 fa(A.a.x - f * 0.5f, stepTop - f * 0.35f), fb(C.b.x + f * 0.5f, C.b.y + f * 0.35f);
    dl->AddRect(fa, fb, IM_COL32(255, 255, 255, 45), 6.0f, 0, 1.0f);
    dl->AddText(ImVec2(fa.x + 4, fa.y - f * 1.05f), kSub, "Context stage");
    if (P) {
        std::string t = BlockTime(*P, ProfBlock::Attention);
        if (!t.empty()) Badge(dl, ImVec2(fa.x + 4 + ImGui::CalcTextSize("Context stage").x + f * 0.6f, fa.y - f * 1.05f), t);
    }

    // The store, to the right, with its write and read arrows.
    float sx = fb.x + f * 3.2f, sw = std::min(f * 11, o.x + W - sx - f * 0.5f);
    if (sw > f * 6) {
        Rect st{ImVec2(sx, A.a.y), ImVec2(sx + sw, B.b.y)};
        dl->AddRectFilled(st.a, st.b, DevCol(ld, 0.30f), 5.0f);
        dl->AddRect(st.a, st.b, IM_COL32(255, 255, 255, 110), 5.0f, 0, 1.2f);
        float ty = st.a.y + f * 0.4f, tx = st.a.x + f * 0.5f;
        auto line = [&](ImU32 c, const std::string& s) { dl->AddText(ImVec2(tx, ty), c, s.c_str()); ty += f * 1.15f; };
        if (attn) {
            line(kText, "KV cache (this layer)");
            line(kSub, std::format("{} at -c {}", MB(kvLayer), lp.nCtx));
            line(kSub, std::format("+{} per new token", MB(kvPerTok)));
            line(kSub, std::format("{}/{}, {} memory", kvt[lp.ctk].name, kvt[lp.ctv].name, DevName(ld)));
        } else {
            line(kText, "State store (this layer)");
            line(kSub, std::format("{}, fixed size", MB(stLayer)));
            line(kSub, "same at any context length");
            line(kSub, std::format("one per request (-np), {}", DevName(ld)));
        }
        if (ImGui::IsMouseHoveringRect(st.a, st.b))
            Tip({attn ? "Key-value store: every token appends its key and value; every new token scans all of them."
                      : "Fixed-size state: overwritten by every token, never grows.",
                 attn ? "Size = tokens so far x per-token entry. Set by -c (capacity) and -ctk/-ctv (entry size)."
                      : "Size is set by the model; multiplied by -np (parallel requests)."});
        ImU32 wc = IM_COL32(120, 200, 255, 255);
        // write: A -> store
        float wy = A.Right().y;
        dl->AddLine(ImVec2(fb.x, wy), ImVec2(st.a.x - 6, wy), wc, 1.8f);
        dl->AddLine(A.Right(), ImVec2(fb.x, wy), wc, 1.8f);
        ArrowHead(dl, ImVec2(st.a.x, wy), ImVec2(1, 0), wc, f * 0.4f);
        dl->AddText(ImVec2(fb.x + f * 0.2f, wy - f * 1.15f), wc, attn ? "append" : "write");
        // read: store -> B
        float ry = B.Right().y;
        dl->AddLine(ImVec2(st.a.x, ry), ImVec2(B.b.x + 6, ry), wc, 1.8f);
        ArrowHead(dl, ImVec2(B.b.x, ry), ImVec2(-1, 0), wc, f * 0.4f);
        dl->AddText(ImVec2(fb.x + f * 0.2f, ry + f * 0.1f), wc, attn ? "read all" : "read");
    }
    y = mix.b.y + vgap;

    // Residual add #1, with the skip connection from the input routed on the left.
    float r = f * 0.62f;
    ImVec2 add1(cx, y + r);
    float skipX = cx - mixW * 0.5f - f * 1.8f;
    dl->AddLine(ImVec2(cx, in.Left().y), ImVec2(skipX, in.Left().y), kEdge, 1.2f);
    dl->AddLine(ImVec2(skipX, in.Left().y), ImVec2(skipX, add1.y), kEdge, 1.2f);
    dl->AddLine(ImVec2(skipX, add1.y), ImVec2(add1.x - r - 4, add1.y), kEdge, 1.2f);
    ArrowHead(dl, ImVec2(add1.x - r, add1.y), ImVec2(1, 0), kEdge, f * 0.35f);
    Edge(dl, mix.Bottom(), ImVec2(add1.x, add1.y - r), false);
    SumNode(dl, add1, r, ld, "+");
    if (ImGui::IsMouseHoveringRect(ImVec2(add1.x - r, add1.y - r), ImVec2(add1.x + r, add1.y + r)))
        Tip({"Merge: adds the stage's input (bypass path on the left) to its output"});
    y = add1.y + r + vgap;

    // Normalize 2
    Rect n2 = MakeRect(cx, y, f * 7, f * 1.5f);
    if (Box(dl, n2, DevCol(rd, 0.8f), "Normalize")) Tip({"Small scaling step (negligible cost)"});
    Edge(dl, ImVec2(add1.x, add1.y + r), n2.Top(), false);
    y = n2.b.y + vgap;

    Rect sum{}, comb{};
    float lane2 = skipX;
    if (moe) {
        // Router
        Rect rt = MakeRect(cx, y, std::min(W * 0.5f, f * 22), f * 2.6f);
        if (Box(dl, rt, DevCol(rd), std::format("Dispatcher: sends the token to {} of {} experts", m.nExpertUsed, m.nExpert),
                std::format("reads {} per token", MB((double)L.bytes[(int)Block::Router]))))
            Tip({std::format("Dispatcher (\"router\") on the {}", DevName(rd)),
                 std::format("Picks which {} of the {} experts process this token, and how much each one counts.", m.nExpertUsed, m.nExpert),
                 "A different set is picked for every token, so all experts must stay loaded."});
        Edge(dl, n2.Bottom(), rt.Top(), false);
        if (P) {
            std::string t = BlockTime(*P, ProfBlock::Router);
            if (!t.empty()) Badge(dl, ImVec2(rt.b.x + f * 0.5f, rt.a.y + f * 0.6f), t);
        }
        y = rt.b.y + vgap * 1.6f;

        // Experts: N routed + 1 shared, side by side. They run in parallel.
        int n = m.nExpertUsed + (L.bytes[(int)Block::Shared] ? 1 : 0);
        float sharedGap = f * 1.4f, eg = f * 0.4f;
        // Leave a lane on the left for the residual skip connection.
        float ew = std::clamp((W - f * 4.5f - sharedGap - eg * (n - 1)) / std::max(1, n), f * 2.2f, f * 6.5f);
        float rowW = ew * n + eg * (n - 1) + sharedGap;
        float ex0 = cx - rowW * 0.5f + f * 1.2f, eh = f * 3.4f;
        lane2 = ex0 - f * 1.1f;
        double perExpert = m.nExpert ? (double)L.bytes[(int)Block::Experts] / m.nExpert : 0;
        bool fanPcie = ed != rd;
        std::vector<Rect> experts;
        for (int k = 0; k < m.nExpertUsed; ++k) {
            Rect e{ImVec2(ex0 + k * (ew + eg), y), ImVec2(ex0 + k * (ew + eg) + ew, y + eh)};
            experts.push_back(e);
            Edge(dl, rt.Bottom(), e.Top(), fanPcie, 0.45f);
            if (Box(dl, e, DevCol(ed), ew > f * 4.5f ? std::format("Expert {}", k + 1) : std::format("E{}", k + 1),
                    MB(perExpert)))
                Tip({std::format("Expert slot {} of {}, runs on the {}", k + 1, m.nExpertUsed, DevName(ed)),
                     "An expert is an independent worker: it reads its own weights and transforms the vector.",
                     std::format("The {} picked experts run in parallel; each reads {} of weights.", m.nExpertUsed, MB(perExpert)),
                     std::format("All {} experts of this layer ({}) live in {} memory.", m.nExpert,
                                 MB((double)L.bytes[(int)Block::Experts]), ed == Device::Gpu ? "GPU" : "system"),
                     ed == Device::Cpu ? "On the CPU, their speed is limited by system memory bandwidth." : ""});
        }
        if (fanPcie)
            PcieLabel(dl, ImVec2(cx + f * 0.3f, rt.b.y + (y - rt.b.y) * 0.45f),
                      "PCIe: vector + expert choice to " + std::string(DevName(ed)) +
                          (P && P->toCpuUs > 0 ? "   [" + Us(P->toCpuUs) + "]" : ""));
        Rect sh{};
        if (L.bytes[(int)Block::Shared]) {
            float sx = ex0 + m.nExpertUsed * (ew + eg) - eg + sharedGap;
            sh = {ImVec2(sx, y), ImVec2(sx + ew, y + eh)};
            // The shared expert gets the normalized vector directly (not through the router).
            float rx = sh.Top().x;
            dl->AddLine(n2.Right(), ImVec2(rx, n2.Right().y), kEdge, 1.4f);
            Edge(dl, ImVec2(rx, n2.Right().y), sh.Top(), false, 1.0f);
            const char* shTitle = ImGui::CalcTextSize("Always-on").x < ew - 8 ? "Always-on" : "Always";
            if (Box(dl, sh, DevCol(sd), shTitle, MB((double)L.bytes[(int)Block::Shared])))
                Tip({"Always-on expert (\"shared expert\"): processes every token, bypassing the dispatcher",
                     "Runs on the " + std::string(DevName(sd)) + ", in parallel with the picked experts"});
        }
        if (P && sh.W() > 0) {
            std::string t = BlockTime(*P, ProfBlock::Shared);
            if (!t.empty()) Badge(dl, ImVec2(sh.a.x, sh.b.y + f * 0.2f), t);
        }
        y += eh + vgap * 1.6f;
        // Expert usage strip: space reserved here, drawn after the edges to the join so it sits on top of them.
        const bool strip = pr && pr->ok && li < (int)pr->genCounts.size() && !pr->genCounts[li].empty();
        const float stripY = y;
        if (strip) y += f * 2.2f;

        // Join: weighted sum of the routed experts (computed where the experts are).
        sum = MakeRect(cx, y, std::min(W * 0.5f, f * 20), f * 2.2f);
        for (const auto& e : experts) Edge(dl, e.Bottom(), sum.Top(), false, 0.55f);
        if (strip) {
            std::string lbl = std::format("Expert usage in this layer ({} experts, brighter = picked more often)", pr->nExperts);
            ImVec2 lp0(ex0, stripY - f * 0.9f), lsz = ImGui::CalcTextSize(lbl.c_str());
            dl->AddRectFilled(ImVec2(lp0.x - 3, lp0.y - 1), ImVec2(lp0.x + lsz.x + 3, lp0.y + lsz.y + 1), ImGui::GetColorU32(ImGuiCol_ChildBg));
            dl->AddText(lp0, kSub, lbl.c_str());
            ExpertStrip(dl, ImVec2(ex0, stripY + f * 0.3f), rowW - f * 1.2f, f * 0.9f, pr->genCounts[li], li);
        }
        if (Box(dl, sum, DevCol(ed), std::format("Join: merge {} expert results", m.nExpertUsed), "weighted by the dispatcher"))
            Tip({"Waits for all picked experts, then merges their outputs into one vector",
                 "Runs where the experts run (" + std::string(DevName(ed)) + ")"});
        if (P) {
            std::string t = BlockTime(*P, ProfBlock::Experts);
            if (!t.empty()) Badge(dl, ImVec2(sum.b.x + f * 0.5f, sum.a.y + f * 0.5f), "experts: " + t);
        }
        y = sum.b.y + vgap;

        // Combine routed + shared + residual on the layer device.
        ImVec2 add2(cx, y + r);
        bool backPcie = ed != ld;
        Edge(dl, sum.Bottom(), ImVec2(add2.x, add2.y - r), backPcie);
        if (backPcie)
            PcieLabel(dl, ImVec2(cx, sum.b.y + (add2.y - r - sum.b.y) * 0.5f),
                      "PCIe: result back to " + std::string(DevName(ld)) + (P && P->toGpuUs > 0 ? "   [" + Us(P->toGpuUs) + "]" : ""));
        if (sh.W() > 0) {
            dl->AddLine(sh.Bottom(), ImVec2(sh.Bottom().x, add2.y), kEdge, 1.4f);
            dl->AddLine(ImVec2(sh.Bottom().x, add2.y), ImVec2(add2.x + r + 4, add2.y), kEdge, 1.4f);
            ArrowHead(dl, ImVec2(add2.x + r, add2.y), ImVec2(-1, 0), kEdge, f * 0.35f);
        }
        // Second residual: from after the first add, down a lane left of the expert row.
        float lx = std::min(skipX, lane2);
        dl->AddLine(ImVec2(add1.x - r, add1.y + r * 0.5f), ImVec2(lx, add1.y + r * 0.5f), kEdge, 1.2f);
        dl->AddLine(ImVec2(lx, add1.y + r * 0.5f), ImVec2(lx, add2.y), kEdge, 1.2f);
        dl->AddLine(ImVec2(lx, add2.y), ImVec2(add2.x - r - 4, add2.y), kEdge, 1.2f);
        ArrowHead(dl, ImVec2(add2.x - r, add2.y), ImVec2(1, 0), kEdge, f * 0.35f);
        SumNode(dl, add2, r, ld, "+");
        if (ImGui::IsMouseHoveringRect(ImVec2(add2.x - r, add2.y - r), ImVec2(add2.x + r, add2.y + r)))
            Tip({"Merge: picked experts + always-on expert + bypass path"});
        if (P) {
            std::string t = BlockTime(*P, ProfBlock::Merge);
            if (!t.empty()) Badge(dl, ImVec2(add2.x + r + f * 0.6f, add2.y - f * 0.5f), "merge: " + t);
        }
        y = add2.y + r + vgap;
        comb = {ImVec2(add2.x - r, add2.y - r), ImVec2(add2.x + r, add2.y + r)};
    } else {
        Rect ffn = MakeRect(cx, y, std::min(W * 0.5f, f * 20), f * 2.6f);
        Box(dl, ffn, DevCol(d[(int)Block::DenseFfn]), "Processing stage",
            std::format("reads {} per token", MB((double)L.bytes[(int)Block::DenseFfn])));
        Edge(dl, n2.Bottom(), ffn.Top(), false);
        y = ffn.b.y + vgap;
        comb = ffn;
    }

    // Exit: continues into the next layer's section (or the token picker).
    Device next = li + 1 < (int)m.layers.size() ? pl.layerDev[li + 1] : (pl.outputOnGpu ? Device::Gpu : Device::Cpu);
    float yEnd = comb.b.y + vgap * 0.9f;
    Edge(dl, comb.Bottom(), ImVec2(cx, yEnd), pl.layerDev[li] != next);
    ImGui::Dummy(ImVec2(W, yEnd - o.y + f * 0.15f));
}

// Terminal box of the pipeline (token lookup at the top, token picker at the bottom).
void TerminalBox(const char* title, const std::string& sub, Device d, bool arrowOut, bool arrowIn) {
    const float f = ImGui::GetFontSize();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 o = ImGui::GetCursorScreenPos();
    float W = ImGui::GetContentRegionAvail().x, cx = o.x + W * 0.5f;
    Rect r = MakeRect(cx, o.y + (arrowIn ? f * 0.2f : f * 0.6f), std::min(W * 0.55f, f * 20), f * 2.6f);
    Box(dl, r, DevCol(d), title, sub);
    float h = r.b.y - o.y;
    if (arrowOut) {
        Edge(dl, r.Bottom(), ImVec2(cx, r.b.y + f * 1.2f), false);
        h += f * 1.2f;
    }
    ImGui::Dummy(ImVec2(W, h + f * 0.6f));
}

void Legend() {
    auto sw = [](ImU32 c, const char* t) {
        ImVec2 p = ImGui::GetCursorScreenPos();
        float s = ImGui::GetFontSize() * 0.8f;
        ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(p.x, p.y + 2), ImVec2(p.x + s, p.y + 2 + s), c, 2.0f);
        ImGui::Dummy(ImVec2(s, s));
        ImGui::SameLine();
        ImGui::TextUnformatted(t);
        ImGui::SameLine(0, ImGui::GetFontSize());
    };
    sw(DevCol(Device::Gpu), "runs on GPU");
    sw(DevCol(Device::Cpu), "runs on CPU");
    sw(ImGui::GetColorU32(kPcie), "PCIe hand-off (one side waits for the other)");
    sw(IM_COL32(255, 255, 255, 200), "white cap = layer with a KV cache");
    ImGui::NewLine();
}

// ------------------------------------------------------------------------------------------------
// Memory and speed summary

void VramBar(const std::vector<std::pair<std::string, std::pair<double, ImVec4>>>& segs, double total) {
    float w = ImGui::GetContentRegionAvail().x, h = ImGui::GetFontSize() * 1.4f;
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    double used = 0;
    for (auto& s : segs) used += s.second.first;
    double scale = std::max(total, used);
    dl->AddRectFilled(p, ImVec2(p.x + w, p.y + h), ImGui::GetColorU32(ImGuiCol_FrameBg), 3.0f);
    float x = p.x;
    for (auto& s : segs) {
        float sw = (float)(s.second.first / scale * w);
        dl->AddRectFilled(ImVec2(x, p.y), ImVec2(x + sw, p.y + h), ImGui::GetColorU32(s.second.second));
        x += sw;
    }
    float tx = p.x + (float)(total / scale * w);   // VRAM capacity marker
    dl->AddLine(ImVec2(tx, p.y - 3), ImVec2(tx, p.y + h + 3), IM_COL32(255, 255, 255, 220), 2.0f);
    ImGui::Dummy(ImVec2(w, h + 4));
    for (auto& s : segs) {
        ImGui::ColorButton(("##" + s.first).c_str(), s.second.second, ImGuiColorEditFlags_NoTooltip,
                           ImVec2(ImGui::GetFontSize() * 0.8f, ImGui::GetFontSize() * 0.8f));
        ImGui::SameLine();
        ImGui::Text("%8s  %s", MB(s.second.first).c_str(), s.first.c_str());
    }
}

int SmallestFittingCpuMoe(const ModelInfo& m, LaunchParams p, double budget) {
    return Placement::SmallestFittingCpuMoe(m, p, budget);
}

void DrawSummary(App& app, const ModelInfo& m, LaunchParams& lp, const Placement& pl) {
    auto gs = app.nvml.Static();
    int attn = m.CountMixer(MixerKind::Attention), lin = m.CountMixer(MixerKind::Linear);
    ImGui::PushTextWrapPos(0);

    ImGui::SeparatorText("Model");
    ImGui::TextUnformatted(m.name.c_str());
    ImGui::TextDisabled("%s, %.1fB parameters, %s file", m.sizeLabel.c_str(), m.nParams / 1e9, MB((double)m.fileSize).c_str());
    ImGui::Text("%d layers in sequence; each token passes through all of them", m.nLayers);
    if (lin) ImGui::Text("%d layers keep a KV cache, %d keep a small fixed-size state", attn, lin);
    if (m.nExpert) ImGui::Text("Each layer has %d experts; %d are picked per token, plus 1 always-on", m.nExpert, m.nExpertUsed);

    ImGui::SeparatorText("Per generated token");
    ImGui::TextColored(kCpu, "CPU reads %s of weights", MB(pl.cpuBytesPerToken).c_str());
    ImGui::TextColored(kGpu, "GPU reads %s of weights", MB(pl.gpuBytesPerToken).c_str());
    if (pl.kvBytes)
        ImGui::TextColored(ImVec4(0.47f, 0.78f, 1.0f, 1), "+ KV cache reads: up to %s (when the context is full)",
                           MB((double)pl.kvBytes).c_str());
    ImGui::TextColored(kPcie, "%d GPU<->CPU hand-offs", pl.crossingsPerToken);
    ImGui::TextDisabled("Stored: GPU %s, CPU %s", MB((double)pl.gpuWeights).c_str(), MB((double)pl.cpuWeights).c_str());

    ImGui::SeparatorText("VRAM budget");
    double total = (double)gs.vramTotal, others = (double)OtherProcessesVram(app);
    const auto& kv = KvTypes();
    std::vector<std::pair<std::string, std::pair<double, ImVec4>>> segs = {
        {"Weights", {(double)pl.gpuWeights, kGpu}},
        {std::format("KV cache ({} layers, {})", attn, kv[lp.ctk].name), {(double)pl.kvBytes, ImVec4(0.30f, 0.70f, 0.95f, 1)}},
        {std::format("State stores ({} layers)", lin), {(double)pl.rsBytes, ImVec4(0.45f, 0.80f, 0.85f, 1)}},
        {"Compute buffers (est.)", {(double)pl.computeEst, ImVec4(0.55f, 0.60f, 0.70f, 1)}},
        {"CUDA context (est.)", {(double)pl.cudaCtxEst, ImVec4(0.40f, 0.45f, 0.55f, 1)}},
        {"Other processes", {others, ImVec4(0.75f, 0.40f, 0.45f, 1)}},
    };
    VramBar(segs, total);
    double margin = total - ((double)pl.GpuTotal() + others);
    int fit = total > 0 ? SmallestFittingCpuMoe(m, lp, total - others - 0.25 * (1 << 30)) : -1;
    if (total > 0) {
        Status s = margin < 0 ? Status::Bad : margin < 0.4 * (1 << 30) ? Status::Warn : Status::Ok;
        StatusDot(s, "");
        ImGui::SameLine();
        ImGui::Text(margin >= 0 ? "Fits, %s free" : "Does not fit: %s over", MB(std::abs(margin)).c_str());
        if (m.nExpert && fit >= 0) ImGui::TextDisabled("Smallest --n-cpu-moe that fits (0.25 GB margin): %d", fit);
    }

    ImGui::SeparatorText("Bandwidth ceiling (upper bound, not a prediction)");
    bool measured = false;
    double cpuBw = CpuBandwidth(app, &measured), gpuBw = gs.VramGBs() * 1e9;
    double cpuMs = cpuBw > 0 ? pl.cpuBytesPerToken / cpuBw * 1000 : 0;
    double gpuMs = gpuBw > 0 ? pl.gpuBytesPerToken / gpuBw * 1000 : 0;
    ImGui::Text("CPU part >= %.1f ms  (%.0f GB/s%s)", cpuMs, cpuBw / 1e9, measured ? ", measured" : ", estimated");
    ImGui::Text("GPU part >= %.1f ms  (%.0f GB/s)", gpuMs, gpuBw / 1e9);
    if (cpuMs + gpuMs > 0) ImGui::Text("Ceiling: %.0f tok/s (%.1f ms per token)", 1000.0 / (cpuMs + gpuMs), cpuMs + gpuMs);
    ImGui::TextDisabled("Only the time to read each token's weights once at full memory speed. Real speed is lower: "
                        "compute below bandwidth peak, CPU<->GPU hand-offs, kernel launches, scheduling and thread sync "
                        "come on top. Load a runtime profile to see the measured parts.");
    if (app.profile.ok && app.profile.genSteps > 0 && cpuMs + gpuMs > 0) {
        const ProfileData& pr = app.profile;
        ImGui::Text("Measured (profile): %.0f tok/s = %.0f%% of the ceiling", 1e6 / pr.stepUs, (cpuMs + gpuMs) * 1000 / pr.stepUs * 100);
        ImGui::TextDisabled("CPU %s vs %.1f ms ceiling, GPU %s vs %.1f ms ceiling, hand-offs %s, other %s", Us(pr.cpuBusyUs).c_str(), cpuMs,
                            Us(pr.gpuBusyUs).c_str(), gpuMs, Us(pr.handoffUs).c_str(),
                            Us(std::max(0.0, pr.stepUs - pr.cpuBusyUs - pr.gpuBusyUs - pr.handoffUs)).c_str());
    }

    if (m.nExpert) {
        ImGui::SeparatorText("What if --n-cpu-moe ...");
        int lo = std::max(0, std::min(lp.nCpuMoe, fit < 0 ? lp.nCpuMoe : fit) - 2);
        int hi = std::min(m.nLayers, lo + 7);
        if (ImGui::BeginTable("whatif", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("N");
            ImGui::TableSetupColumn("GPU mem");
            ImGui::TableSetupColumn("fits");
            ImGui::TableSetupColumn("ceiling tok/s");
            ImGui::TableHeadersRow();
            for (int n = lo; n <= hi; ++n) {
                LaunchParams q = lp;
                q.nCpuMoe = n;
                Placement r = Placement::Compute(m, q);
                double needN = (double)r.GpuTotal() + others;
                double ms = (cpuBw > 0 ? r.cpuBytesPerToken / cpuBw * 1000 : 0) + (gpuBw > 0 ? r.gpuBytesPerToken / gpuBw * 1000 : 0);
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                if (ImGui::Selectable(std::to_string(n).c_str(), n == lp.nCpuMoe, ImGuiSelectableFlags_SpanAllColumns)) {
                    lp.nCpuMoe = n;
                    app.settings.Save();
                }
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(MB((double)r.GpuTotal()).c_str());
                ImGui::TableNextColumn();
                StatusDot(total <= 0 ? Status::Unavailable : needN <= total - 0.25 * (1 << 30) ? Status::Ok
                                                          : needN <= total ? Status::Warn : Status::Bad, "");
                ImGui::TableNextColumn();
                ImGui::Text("%.0f", ms > 0 ? 1000.0 / ms : 0.0);
            }
            ImGui::EndTable();
        }
        ImGui::TextDisabled("N = number of layers whose experts stay on the CPU. Click a row to apply.");
    }
    ImGui::TextDisabled("Estimates; real buffer sizes come from the llama.cpp load log (M3).");
    DrawProfileSummary(app, m);
    ImGui::PopTextWrapPos();
}

// ------------------------------------------------------------------------------------------------
// Launch parameters

void Flow(float itemWidth) {
    float lineEnd = ImGui::GetItemRectMax().x + ImGui::GetStyle().ItemSpacing.x * 2;
    if (lineEnd + itemWidth < ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x)
        ImGui::SameLine(0, ImGui::GetFontSize());
}

void DrawParams(App& app, const ModelInfo& m) {
    auto& p = app.settings.launch;
    float f = ImGui::GetFontSize();
    bool changed = false;
    auto width = [&](float field, const char* label) { return field + ImGui::CalcTextSize(label).x + f * 0.5f; };
    ImGui::SetNextItemWidth(f * 5.5f);
    changed |= ImGui::InputInt("-ngl", &p.ngl, 0, 0);
    if (m.nExpert) {
        Flow(width(f * 9, "--n-cpu-moe"));
        ImGui::SetNextItemWidth(f * 9);
        changed |= ImGui::SliderInt("--n-cpu-moe", &p.nCpuMoe, 0, m.nLayers);
    }
    Flow(width(f * 5.5f, "-c"));
    ImGui::SetNextItemWidth(f * 5.5f);
    changed |= ImGui::InputInt("-c", &p.nCtx, 0, 0);
    const auto& kv = KvTypes();
    auto kvCombo = [&](const char* label, int& idx) {
        Flow(width(f * 4.5f, label));
        ImGui::SetNextItemWidth(f * 4.5f);
        if (ImGui::BeginCombo(label, kv[idx].name)) {
            for (int i = 0; i < (int)kv.size(); ++i)
                if (ImGui::Selectable(kv[i].name, i == idx)) { idx = i; changed = true; }
            ImGui::EndCombo();
        }
    };
    kvCombo("-ctk", p.ctk);
    kvCombo("-ctv", p.ctv);
    Flow(width(f * 4.5f, "-ub"));
    ImGui::SetNextItemWidth(f * 4.5f);
    changed |= ImGui::InputInt("-ub", &p.nUbatch, 0, 0);
    Flow(width(f * 3.5f, "-np"));
    ImGui::SetNextItemWidth(f * 3.5f);
    changed |= ImGui::InputInt("-np", &p.nParallel, 0, 0);
    if (changed) {
        p.ngl = std::clamp(p.ngl, 0, 999);
        p.nCpuMoe = std::clamp(p.nCpuMoe, 0, m.nLayers);
        p.nCtx = std::clamp(p.nCtx, 256, std::max(256, m.ctxTrain ? m.ctxTrain : 1 << 20));
        p.nUbatch = std::clamp(p.nUbatch, 1, 8192);
        p.nParallel = std::clamp(p.nParallel, 1, 64);
        app.settings.Save();
    }
}

} // namespace

void DrawTopologyPanel(App& app) {
    if (!ImGui::Begin(Win::Topology)) { ImGui::End(); return; }
    static int selected = 0;

    float f = ImGui::GetFontSize();
    ImGui::SetNextItemWidth(std::max(f * 16, ImGui::GetContentRegionAvail().x - f * 12));
    if (ImGui::BeginCombo("##model", app.model.fileName.empty() ? "Select a GGUF model" : app.model.fileName.c_str())) {
        for (const auto& path : app.modelFiles) {
            std::string label = ToUtf8(path);
            if (ImGui::Selectable(label.c_str(), label == app.model.path)) app.LoadModel(path);
        }
        if (app.modelFiles.empty()) ImGui::TextDisabled("No .gguf files found in Downloads\\models, models, C:\\models");
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    if (ImGui::Button("Browse...")) {
        std::wstring p = BrowseForModel();
        if (!p.empty()) { app.LoadModel(p); app.ScanModels(); }
    }
    ImGui::SameLine();
    if (ImGui::Button("Rescan")) app.ScanModels();

    const ModelInfo& m = app.model;
    if (!m.ok) {
        if (!m.error.empty()) ImGui::TextColored(StatusColor(Status::Bad), "%s", m.error.c_str());
        ImGui::End();
        return;
    }
    DrawParams(app, m);
    Placement pl = Placement::Compute(m, app.settings.launch);
    selected = std::clamp(selected, 0, (int)m.layers.size() - 1);

    static int scrollTo = -1;
    Legend();
    ImGui::SeparatorText("Model pipeline: the cursor follows the diagram below (click a layer to jump)");
    DrawPipeline(m, pl, selected, scrollTo);

    // Full pipeline in a scroll box; summary beside it (wide) or below it (narrow).
    bool wide = ImGui::GetContentRegionAvail().x > f * 70;
    auto flow = [&](ImVec2 size) {
        static std::vector<float> top, height;
        int L = (int)m.layers.size();
        if ((int)top.size() != L) { top.assign(L, 0); height.assign(L, 0); }
        if (ImGui::BeginChild("##flow", size, ImGuiChildFlags_Borders)) {
            Device firstDev = pl.layerDev.empty() ? Device::Cpu : pl.layerDev[0];
            TerminalBox("Lookup: token -> vector", std::format("table on the CPU, then sent to the {}", DevName(firstDev)),
                        Device::Cpu, true, false);
            for (int i = 0; i < L; ++i) {
                top[i] = ImGui::GetCursorPosY();
                // Draw only what is on screen. Off-screen layers advance the cursor by their measured height
                // (not via Dummy, which would add item spacing and make the cached height grow every frame).
                if (height[i] > 0 && !ImGui::IsRectVisible(ImVec2(ImGui::GetContentRegionAvail().x, height[i]))) {
                    ImGui::SetCursorPosY(top[i] + height[i]);
                } else {
                    DrawLayerDiagram(m, pl, app.settings.launch, i, &app.profile);
                    height[i] = ImGui::GetCursorPosY() - top[i];
                }
            }
            TerminalBox("Pick token: vector -> next token",
                        std::format("scores {} candidates, on the {}", m.nVocab, DevName(pl.outputOnGpu ? Device::Gpu : Device::Cpu)),
                        pl.outputOnGpu ? Device::Gpu : Device::Cpu, false, true);
            // Cursor = layer under the upper third of the view.
            float probe = ImGui::GetScrollY() + ImGui::GetWindowHeight() * 0.33f;
            for (int i = 0; i < L; ++i)
                if (probe >= top[i] && probe < top[i] + height[i]) selected = i;
            if (scrollTo >= 0 && scrollTo < L) {
                ImGui::SetScrollY(top[scrollTo] - f * 0.5f);
                scrollTo = -1;
            }
            if (g_topologyScroll >= 0 && height[0] > 0) {
                int li = std::clamp((int)g_topologyScroll, 0, L - 1);
                if (height[li] > 0) {
                    ImGui::SetScrollY(top[li] + (g_topologyScroll - li) * height[li]);
                    g_topologyScroll = -1;
                }
            }
        }
        ImGui::EndChild();
    };

    if (wide && ImGui::BeginTable("##split", 2, ImGuiTableFlags_Resizable | ImGuiTableFlags_BordersInnerV)) {
        ImGui::TableSetupColumn("diagram", ImGuiTableColumnFlags_WidthStretch, 2.2f);
        ImGui::TableSetupColumn("summary", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        flow(ImVec2(0, ImGui::GetContentRegionAvail().y));
        ImGui::TableNextColumn();
        if (ImGui::BeginChild("##summary", ImVec2(0, ImGui::GetContentRegionAvail().y))) {
            DrawSummary(app, m, app.settings.launch, pl);
            if (g_summaryScroll >= 0 && ImGui::GetScrollMaxY() > 0) { ImGui::SetScrollY(g_summaryScroll * ImGui::GetScrollMaxY()); g_summaryScroll = -1; }
        }
        ImGui::EndChild();
        ImGui::EndTable();
    } else {
        float h = ImGui::GetContentRegionAvail().y;
        flow(ImVec2(0, std::max(f * 20, h * 0.62f)));
        if (ImGui::BeginChild("##summary", ImVec2(0, 0))) {
            DrawSummary(app, m, app.settings.launch, pl);
            if (g_summaryScroll >= 0 && ImGui::GetScrollMaxY() > 0) { ImGui::SetScrollY(g_summaryScroll * ImGui::GetScrollMaxY()); g_summaryScroll = -1; }
        }
        ImGui::EndChild();
    }
    ImGui::End();
}
