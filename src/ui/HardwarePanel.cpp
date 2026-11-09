#include <windows.h>
#include <algorithm>
#include <cmath>
#include <format>
#include "app/App.h"
#include "core/Util.h"
#include "imgui.h"
#include "platform/Affinity.h"
#include "ui/Panels.h"

namespace {

double LaneGBs(int gen) {
    static const double perLane[] = {0, 0.25, 0.5, 0.985, 1.969, 3.938, 7.563};
    return (gen >= 1 && gen <= 6) ? perLane[gen] : 0;
}

std::string Pct(double v) { return std::format("{:.0f}%", v * 100); }

void AddCpuRows(App& app, std::vector<HwRow>& rows) {
    const auto& c = app.cpu;
    const char* S = "CPU";
    rows.push_back({S, "Model", c.brand, "", {Status::None, ""}});
    rows.push_back({S, "Cores",
                    c.hybrid ? std::format("{} P + {} E, {} threads", c.pCores, c.eCores, c.logicalCount)
                             : std::format("{} cores, {} threads", c.pCores, c.logicalCount),
                    c.smt ? "SMT on" : "SMT off", {Status::None, ""}});
    std::string isa = std::string(c.avx2 ? "AVX2 " : "") + (c.avxVnni ? "AVX-VNNI " : "") + (c.fma ? "FMA " : "") +
                      (c.avx512f ? "AVX-512" : "");
    rows.push_back({S, "Vector ISA", Trim(isa), c.avx512f ? "" : "no AVX-512", {Status::None, ""}});
    std::string caches = std::format("L2 {}/P-core", FormatBytes(c.l2PerPCore));
    if (c.hybrid && c.l2PerECluster) caches += std::format(", {}/E-cluster", FormatBytes(c.l2PerECluster));
    caches += std::format(", L3 {}", FormatBytes(c.l3Total));
    rows.push_back({S, "Caches", caches, "", {Status::None, ""}});

    // Measured by "Run tests": the memory test keeps every core busy, so it doubles as an all-core clock check.
    for (int kind = 0; kind < (c.hybrid ? 2 : 1); ++kind) {
        bool p = kind == 0;
        float mhz = p ? app.hwCheck.pLoadMhz : app.hwCheck.eLoadMhz;
        int specMax = p ? c.specPMaxMhz : c.specEMaxMhz;
        std::string label = p ? (c.hybrid ? "P-core clock, all cores busy" : "Clock, all cores busy")
                              : "E-core clock, all cores busy";
        std::string spec = specMax ? std::format("max turbo {:.2f} GHz (one or two cores); all-core turbo is "
                                                 "normally 5-10% lower", specMax / 1000.0)
                                   : "";
        Verdict v{Status::Unavailable, "press Run tests"};
        if (mhz > 0 && specMax) {
            double ratio = mhz / specMax;
            v = ratio >= 0.85 ? Verdict{Status::Ok, std::format("{:.0f}% of max turbo", ratio * 100)}
                              : Verdict{Status::Warn, std::format("only {:.0f}% of max turbo with all cores busy: power "
                                                                  "limit (BIOS PL1/PL2), current limit or cooling",
                                                                  ratio * 100)};
        } else if (mhz > 0) {
            v = {Status::Info, "no reference max turbo for this CPU"};
        }
        rows.push_back({S, label, mhz > 0 ? std::format("{:.2f} GHz", mhz / 1000) : "not measured", spec, v});
    }
}

void AddMemoryRows(App& app, std::vector<HwRow>& rows) {
    const auto& m = app.mem;
    const char* S = "Memory";
    std::string installed = FormatBytes(m.totalBytes);
    if (!m.dimms.empty())
        installed += std::format(" ({} x {} {})", m.dimms.size(), FormatBytes(m.dimms[0].sizeMB << 20), m.typeName);
    rows.push_back({S, "Installed", installed, "", {Status::None, ""}});
    if (!m.dimms.empty())
        rows.push_back({S, "Modules", Trim(m.dimms[0].manufacturer + " " + m.dimms[0].part), "", {Status::None, ""}});

    Verdict sv{Status::Unavailable, "SMBIOS memory information not available"};
    if (m.configuredMTs && m.ratedMTs) {
        if (m.configuredMTs < m.ratedMTs * 0.98)
            sv = {Status::Bad, std::format("running at {} MT/s, modules are rated {} MT/s: XMP/EXPO profile disabled "
                                           "or memory retrained lower",
                                           m.configuredMTs, m.ratedMTs)};
        else
            sv = {Status::Ok, "configured speed matches the module rating"};
    }
    rows.push_back({S, "Speed", m.configuredMTs ? std::format("{} MT/s", m.configuredMTs) : "-",
                    m.ratedMTs ? std::format("rated {} MT/s", m.ratedMTs) : "", sv});
    rows.push_back({S, "Channels", std::to_string(m.channels), "assumed dual-channel platform",
                    {Status::None, ""}});
    rows.push_back({S, "Theoretical bandwidth", std::format("{:.1f} GB/s", m.theoreticalGBs),
                    std::format("{} ch x {} MT/s x 8 B", m.channels, m.configuredMTs), {Status::None, ""}});

    auto r = app.memTest.Result();
    if (app.memTest.Running()) {
        rows.push_back({S, "Measured bandwidth", std::format("testing... {:.0f}%", app.memTest.Progress() * 100), "",
                        {Status::Info, "test running"}});
    } else if (r.valid) {
        double ratio = m.theoreticalGBs > 0 ? r.bestGBs / m.theoreticalGBs : 0;
        Verdict v{Status::Ok, "at least 85% of theoretical"};
        if (ratio < 0.70) v = {Status::Bad, "below 70% of theoretical: check memory configuration and timings"};
        else if (ratio < 0.85) v = {Status::Warn, "70-85% of theoretical: secondary timings or background load?"};
        rows.push_back({S, "Measured bandwidth", std::format("{:.1f} GB/s ({})", r.bestGBs, Pct(ratio)),
                        std::format("median {:.1f}, {} threads", r.medianGBs, r.threads), v});
    } else {
        rows.push_back({S, "Measured bandwidth", "not measured", "", {Status::Unavailable, "press Run tests"}});
    }
}

void AddGpuRows(App& app, std::vector<HwRow>& rows) {
    auto st = app.nvml.Static();
    const char* S = "GPU";
    if (!st.ok) {
        rows.push_back({S, "NVIDIA GPU", "unavailable", "", {Status::Bad, st.error}});
        return;
    }
    auto l = app.nvml.Live();
    rows.push_back({S, "Model", st.name, "", {Status::None, ""}});
    rows.push_back({S, "Driver / CUDA", st.driver,
                    std::format("CUDA {}.{}", st.cudaDriverVersion / 1000, (st.cudaDriverVersion % 1000) / 10),
                    {Status::None, ""}});
    rows.push_back({S, "VRAM", std::format("{} used", FormatBytes(l.vramUsed)),
                    std::format("of {}", FormatBytes(st.vramTotal)), {Status::None, ""}});
    rows.push_back({S, "Graphics clock", std::format("{} MHz", l.gfxMHz), std::format("max {} MHz", st.maxGraphicsMHz),
                    {Status::None, ""}});
    rows.push_back({S, "Memory clock", std::format("{} MHz", l.memMHz), std::format("max {} MHz", st.maxMemMHz),
                    {Status::None, ""}});
    rows.push_back({S, "Power", std::format("{:.0f} W", l.powermW / 1000.0),
                    std::format("limit {:.0f} W", st.powerLimitmW / 1000.0), {Status::None, ""}});
    // Live readings: tracked over time by the GPU runtime and Diagnosis panels, shown here as facts only.
    rows.push_back({S, "Temperature", std::format("{} C", l.tempC), "", {Status::None, ""}});
    bool concerning = false;
    rows.push_back({S, "Clock limiters", ThrottleReasonsText(l.throttle, &concerning), "", {Status::None, ""}});
    bool rebar = st.bar1Total > (256ull << 20);
    rows.push_back({S, "Resizable BAR", rebar ? "enabled" : "disabled", "BAR1 " + FormatBytes(st.bar1Total),
                    rebar ? Verdict{Status::Ok, "CPU can map all of VRAM"}
                          : Verdict{Status::Warn, "Resizable BAR off (enable Above 4G Decoding + Re-Size BAR in BIOS)"}});
}

void AddPcieRows(App& app, std::vector<HwRow>& rows) {
    auto st = app.nvml.Static();
    if (!st.ok) return;
    const char* S = "PCIe";
    auto l = app.nvml.Live();
    int cardGen = st.gpuMaxGen ? std::min(st.gpuMaxGen, st.linkMaxGen ? st.linkMaxGen : st.gpuMaxGen) : st.linkMaxGen;
    int cardWidth = st.cardWidth ? std::min(st.cardWidth, st.linkMaxWidth ? st.linkMaxWidth : st.cardWidth)
                                 : st.linkMaxWidth;
    double linkGBs = LaneGBs(cardGen) * cardWidth;

    rows.push_back({S, "Link now", std::format("Gen{} x{}", l.linkGen, l.linkWidth), "drops at idle to save power",
                    {Status::None, ""}});

    int gSeen = app.nvml.MaxGenSeen(), wSeen = app.nvml.MaxWidthSeen();
    bool evaluated = app.nvml.MaxUtilSeen() >= 20 || app.pcieTest.Result().valid;
    Verdict lv{Status::Unavailable, "press Run tests (the link slows down at idle; the test brings it to full speed)"};
    if (evaluated) {
        if (wSeen < cardWidth)
            lv = {Status::Bad, std::format("link trained at x{} but the card supports x{}: check the slot, lane sharing "
                                           "with M.2/Wi-Fi/other cards, BIOS lane split",
                                           wSeen, cardWidth)};
        else if (gSeen < cardGen)
            lv = {Status::Warn, std::format("link runs at Gen{} but card and slot support Gen{} (BIOS PCIe speed "
                                            "setting, riser cable?)",
                                            gSeen, cardGen)};
        else
            lv = {Status::Ok, "full link width and generation reached under load"};
    }
    std::string spec = std::format("card Gen{} x{}", cardGen, cardWidth);
    if (!st.cardWidthNote.empty()) spec += " (" + st.cardWidthNote + ")";
    rows.push_back({S, "Link under load", evaluated && gSeen ? std::format("Gen{} x{}", gSeen, wSeen) : "not measured",
                    spec, lv});
    rows.push_back({S, "Theoretical bandwidth", std::format("{:.1f} GB/s per direction", linkGBs), "",
                    {Status::None, ""}});

    auto r = app.pcieTest.Result();
    if (app.pcieTest.Running()) {
        rows.push_back({S, "Measured copy speed", std::format("testing... {:.0f}%", app.pcieTest.Progress() * 100), "",
                        {Status::Info, "test running"}});
    } else if (r.valid) {
        double ratio = linkGBs > 0 ? std::min(r.h2dGBs, r.d2hGBs) / linkGBs : 0;
        Verdict v{Status::Ok, "at least 70% of the card's link capacity"};
        if (ratio < 0.4) v = {Status::Bad, "below 40% of link capacity: link degraded or contended"};
        else if (ratio < 0.7) v = {Status::Warn, "40-70% of link capacity"};
        rows.push_back({S, "Measured copy speed", std::format("H2D {:.1f} / D2H {:.1f} GB/s", r.h2dGBs, r.d2hGBs),
                        Pct(ratio) + " of link", v});
    } else if (!r.error.empty()) {
        rows.push_back({S, "Measured copy speed", "failed", "", {Status::Bad, r.error}});
    } else {
        rows.push_back({S, "Measured copy speed", "not measured", "", {Status::Unavailable, "press Run tests"}});
    }
    double rx = l.rxKBs / 1024.0, tx = l.txKBs / 1024.0;
    rows.push_back({S, "Traffic now", std::format("RX {:.0f} / TX {:.0f} MB/s", rx, tx),
                    linkGBs > 0 ? std::format("{:.1f}% of link", std::max(rx, tx) / 1024.0 / linkGBs * 100) : "",
                    {Status::None, ""}});
}

void AddDisplayRows(App& app, std::vector<HwRow>& rows) {
    const auto& d = app.displays;
    const char* S = "Display";
    const AdapterInfo* igpu = d.igpuIndex >= 0 ? &d.adapters[d.igpuIndex] : nullptr;
    const AdapterInfo* nv = d.nvidiaIndex >= 0 ? &d.adapters[d.nvidiaIndex] : nullptr;

    if (igpu)
        rows.push_back({S, "Integrated GPU", igpu->name, "enabled", {Status::None, ""}});
    else if (d.cpuIgpu == 1)
        rows.push_back({S, "Integrated GPU", "not detected", "CPU has one",
                        {Status::Info, "the CPU has an integrated GPU but Windows does not see it: disabled in BIOS "
                                       "(iGPU Multi-Monitor / Primary Display) or no Intel graphics driver"}});
    else
        rows.push_back({S, "Integrated GPU", "none", d.cpuIgpu == 0 ? "F-series CPU" : "", {Status::None, ""}});

    int onNvidia = 0;
    for (const auto& a : d.adapters)
        for (const auto& m : a.monitors) {
            if (a.IsNvidia()) ++onNvidia;
            std::string label = m.primary ? "Monitor (primary)" : "Monitor";
            Verdict mv{Status::None, ""};
            if (m.hz > 0 && m.hz < 50)
                mv = {Status::None, std::format("{} Hz: the port or cable limits the refresh rate (e.g. HDMI 1.4 caps "
                                                "4K at 30 Hz). Use DisplayPort or an HDMI 2.0+ port and cable.",
                                                m.hz)};
            rows.push_back({S, label, std::format("{}x{} @ {} Hz", m.width, m.height, m.hz), "on " + a.name, mv});
        }

    if (!nv) return;
    // VRAM held by other processes on the inference GPU (desktop compositor, browsers, this app...).
    auto procs = app.pdh.GpuProcesses();
    uint64_t others = 0;
    std::string top;
    int shown = 0;
    for (const auto& p : procs) {
        if (p.name.rfind("llama", 0) == 0) continue;
        others += p.bytes;
        if (p.bytes >= (1ull << 20)) {   // every program holding at least 1 MB
            top += std::format("{}{} {}", shown ? ", " : "", p.name, FormatBytes(p.bytes));
            ++shown;
        }
    }

    Verdict v;
    if (onNvidia == 0) {
        v = {Status::Ok, "the inference GPU drives no monitor: all of its VRAM and time are free for the model"};
    } else if (igpu) {
        v = {Status::Warn, std::format("{} monitor(s) on the inference GPU. Moving the cable to the motherboard "
                                       "output (integrated GPU) frees the compositor's VRAM ({} used by other "
                                       "processes now), which can fit another expert layer on the GPU",
                                       onNvidia, FormatBytes(others))};
    } else if (d.cpuIgpu == 1) {
        v = {Status::Warn, "monitor on the inference GPU and the integrated GPU is disabled: enable it in BIOS, "
                           "install the Intel driver and connect the monitor to the motherboard to free VRAM"};
    } else {
        v = {Status::Info, "no integrated GPU, so the display must run on the inference GPU"};
    }
    rows.push_back({S, "Inference GPU drives display", onNvidia ? std::format("yes, {} monitor(s)", onNvidia) : "no",
                    "", v});
    rows.push_back({S, "VRAM used by others", FormatBytes(others), top,
                    others > (1ull << 30) ? Verdict{Status::Warn, "over 1 GB of the inference GPU's VRAM is held by "
                                                                  "other processes: " + top}
                                          : Verdict{Status::None, top}});
}

void AddSystemRows(App& app, std::vector<HwRow>& rows) {
    const auto& s = app.sys;
    const char* S = "System";
    rows.push_back({S, "Power plan", s.powerPlan.empty() ? "unknown" : s.powerPlan, "",
                    s.powerPlanPerformance
                        ? Verdict{Status::Ok, "performance plan"}
                        : Verdict{Status::Warn, "non-performance plan lets cores and PCIe links downclock; select "
                                                "High or Ultimate performance"}});
    const char* aspm[] = {"off", "moderate savings", "maximum savings"};
    rows.push_back({S, "PCIe ASPM", s.aspmAC >= 0 && s.aspmAC <= 2 ? aspm[s.aspmAC] : "unknown", "",
                    s.aspmAC == 0   ? Verdict{Status::Ok, "link power management off"}
                    : s.aspmAC > 0  ? Verdict{Status::Warn, "link power management can add latency to GPU hand-offs"}
                                    : Verdict{Status::Unavailable, "could not read"}});
    // Core parking: forcing every core unparked keeps idle E-cores from sleeping; measured -13% generation speed on a
    // hybrid CPU (repeated A/B with powercfg only).
    if (app.cpu.hybrid && s.cpMinCoresAC >= 0)
        rows.push_back({S, "Core parking", s.cpMinCoresAC >= 100 ? "off (all cores kept awake)" : std::format("allowed (min {}% unparked)", s.cpMinCoresAC),
                        "power plan setting CPMINCORES",
                        s.cpMinCoresAC >= 100
                            ? Verdict{Status::Warn, "every core is kept unparked, so idle E-cores never sleep: measured 13% slower "
                                                    "generation. Allow parking again, e.g.: powercfg /setacvalueindex SCHEME_CURRENT "
                                                    "SUB_PROCESSOR CPMINCORES 10 && powercfg /setactive SCHEME_CURRENT"}
                            : Verdict{Status::Ok, "idle cores can park"}});
    Verdict lp{Status::Unavailable, "could not determine"};
    std::string lpv = "unknown";
    if (s.lockPagesKnown) {
        if (s.lockPagesAssigned && s.lockPagesInToken) { lp = {Status::Ok, "mlock can pin model pages"}; lpv = "granted"; }
        else if (s.lockPagesAssigned) {
            lp = {Status::Warn, "granted to the account but missing from this session's token: sign out and in, or "
                                "processes started from here cannot lock pages"};
            lpv = "granted, not active";
        } else {
            lp = {Status::Warn, "not granted: --mlock / --load-mode mlock silently has no effect"};
            lpv = "not granted";
        }
    }
    rows.push_back({S, "Lock pages in memory", lpv, "", lp});
    // Hardware-accelerated GPU scheduling: with it off, Windows queues every GPU command on the CPU, adding delay to
    // each CPU<->GPU hand-off. Hybrid CPU+GPU inference makes ~70 hand-offs per token: generation was ~15% faster
    // on a test machine when it was turned on.
    rows.push_back({S, "Hardware-accelerated GPU scheduling", s.hags == 2 ? "on" : s.hags == 1 ? "off" : "unknown",
                    "Settings > System > Display > Graphics > Change default graphics settings (reboot required)",
                    s.hags == 2   ? Verdict{Status::Ok, "the GPU schedules its own work"}
                    : s.hags == 1 ? Verdict{Status::Warn, "off: Windows queues every GPU command on the CPU, delaying each "
                                                          "CPU<->GPU hand-off (measured +15% t/s when turned on). Turn on in "
                                                          "Settings > System > Display > Graphics > Change default graphics "
                                                          "settings, then reboot"}
                                  : Verdict{Status::Unavailable, "could not read the setting"}});
    rows.push_back({S, "Windows", s.windowsVersion, "", {Status::None, ""}});
    rows.push_back({S, "LlamaScope runs on", Affinity::Describe(app.cpu), "", {Status::None, ""}});

    // Everything else that may run on the P-cores (affinity allows it) and is using CPU right now.
    if (app.cpu.hybrid) {
        uint64_t pMask = Affinity::PCoreMask(app.cpu);
        // Sampled when "Run tests" was pressed: during the tests every core is busy (and Windows zeroes the
        // test buffer in the System process), which would show up here as competition.
        auto procs = app.hwCheck.procs.empty() ? app.procMon.Snapshot() : app.hwCheck.procs;
        std::sort(procs.begin(), procs.end(), [](auto& a, auto& b) { return a.cpuPct > b.cpuPct; });
        double sum = 0;
        int n = 0;
        std::string top;
        for (const auto& p : procs) {
            if (p.IsLlama() || p.pid == GetCurrentProcessId() || p.cpuPct < 0.3) continue;
            if (p.affinity && !(p.affinity & pMask)) continue;   // already kept off the P-cores
            sum += p.cpuPct;
            if (n < 6) top += std::format("{}{} {:.1f}%", n ? ", " : "", p.name, p.cpuPct);
            ++n;
        }
        bool im = app.inference.Active();
        Verdict v = n == 0 ? Verdict{Status::Ok, "no other program is using the P-cores"}
                  : sum < 5 ? Verdict{Status::Ok, "light background use: " + top}
                            : Verdict{Status::Warn, std::format("{} programs allowed on the P-cores use {:.0f}% of a core: {}. {}", n, sum, top,
                                                                im ? "Some are protected by Windows and cannot be moved."
                                                                   : "Inference mode moves them to the E-cores.")};
        rows.push_back({S, "Competing for P-cores", n ? std::format("{} programs, {:.0f}% of a core", n, sum) : "none",
                        n ? top : "", v});
    }
    rows.push_back({S, "Administrator", s.elevated ? "yes" : "no", "",
                    s.elevated ? Verdict{Status::Ok, "ETW profiling available"}
                               : Verdict{Status::Info, "thread placement and CPU profiling need administrator rights"}});
}

} // namespace

std::vector<HwRow> BuildHardwareRows(App& app) {
    std::vector<HwRow> rows;
    AddCpuRows(app, rows);
    AddMemoryRows(app, rows);
    AddGpuRows(app, rows);
    AddPcieRows(app, rows);
    AddDisplayRows(app, rows);
    AddSystemRows(app, rows);
    return rows;
}

// Hardware panel: checks only (measured vs spec), frozen at the last "Run tests". Facts live in Platform details.
void DrawHardwarePanel(App& app) {
    if (!ImGui::Begin(Win::Hardware)) { ImGui::End(); return; }
    const float em = ImGui::GetFontSize();
    auto& hc = app.hwCheck;
    using Stage = App::HwCheck::Stage;

    // --- Toolbar -----------------------------------------------------------------------------------
    bool llamaBusy = false;
    std::string llama = app.RunningLlama(&llamaBusy);
    bool running = hc.stage != Stage::Idle;
    ImGui::BeginDisabled(running);
    if (ImGui::Button(llamaBusy ? "Refresh" : "Run tests")) app.StartHardwareCheck();
    ImGui::EndDisabled();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip("Re-reads memory, displays, power plan and privileges, then measures memory bandwidth and\n"
                          "all-core clocks (~4 s, every core busy) and PCIe copy speed and link width (~2 s).\n"
                          "The tests are skipped while llama.cpp is computing: they would slow it down and read wrong.");
    ImGui::SameLine();
    if (ImGui::Button("Platform details...")) app.showPlatformDetails = true;
    ImGui::SameLine();

    if (running) {
        bool mem = hc.stage == Stage::Memory;
        float prog = mem ? app.memTest.Progress() * 0.7f : 0.7f + app.pcieTest.Progress() * 0.3f;
        ImGui::ProgressBar(prog, ImVec2(-FLT_MIN, 0),
                           mem ? "Measuring memory bandwidth and clocks..." : "Measuring PCIe copy speed...");
    } else if (hc.lastRun > 0) {
        int ago = (int)(NowSeconds() - hc.lastRun);
        std::string when = ago < 60 ? "just now" : ago < 3600 ? std::format("{} min ago", ago / 60)
                                                              : std::format("{} h ago", ago / 3600);
        ImGui::AlignTextToFramePadding();
        ImGui::TextDisabled("Checked %s", when.c_str());
    } else {
        ImGui::AlignTextToFramePadding();
        ImGui::TextDisabled("Configuration only: press Run tests to measure");
    }
    if (llamaBusy)
        ImGui::TextColored(StatusColor(Status::Warn), "%s is computing: Refresh re-reads the configuration but skips "
                           "the bandwidth tests.", llama.c_str());
    else if (!llama.empty() && running)
        ImGui::TextColored(StatusColor(Status::Warn), "%s is loaded: don't start generating until the test ends.",
                           llama.c_str());
    else if (!hc.skipped.empty() && !running)
        ImGui::TextDisabled("Last run skipped the tests (%s was computing).", hc.skipped.c_str());

    // --- Verdicts (frozen) -------------------------------------------------------------------------
    std::vector<const HwRow*> bad, todo, notes, ok;
    for (const auto& r : hc.rows) {
        switch (r.verdict.status) {
        case Status::Bad: bad.insert(bad.begin(), &r); break;
        case Status::Warn: bad.push_back(&r); break;
        case Status::Unavailable: todo.push_back(&r); break;
        case Status::Info: notes.push_back(&r); break;
        case Status::Ok: ok.push_back(&r); break;
        default: break;
        }
    }

    // Rows: dot | what | measured, aligned across all groups; the reason wraps underneath at full width.
    float labelW = 0;
    for (const auto& r : hc.rows) labelW = std::max(labelW, ImGui::CalcTextSize(r.label.c_str()).x);
    const float textX = em * 1.4f;
    const float valueX = textX + std::min(labelW + em * 1.5f, ImGui::GetContentRegionAvail().x * 0.55f);
    auto table = [&](const char*, const std::vector<const HwRow*>& list, bool reasons) {
        for (const HwRow* r : list) {
            ImGui::PushID(r);
            float x0 = ImGui::GetCursorPosX();
            StatusDot(r->verdict.status, r->verdict.reason);
            ImGui::SameLine(x0 + textX);
            ImGui::TextUnformatted(r->label.c_str());
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", r->section.c_str());
            ImGui::SameLine(x0 + valueX);
            ImGui::PushTextWrapPos(0);
            ImGui::TextUnformatted(r->value.c_str());
            ImGui::PopTextWrapPos();
            if (ImGui::IsItemHovered() && !r->spec.empty()) ImGui::SetTooltip("%s", r->spec.c_str());
            if (reasons && !r->verdict.reason.empty()) {
                ImGui::Indent(textX);
                ImGui::PushTextWrapPos(0);
                ImGui::TextDisabled("%s", r->verdict.reason.c_str());
                ImGui::PopTextWrapPos();
                ImGui::Unindent(textX);
                ImGui::Spacing();
            }
            ImGui::PopID();
        }
    };

    ImGui::BeginChild("##verdicts", ImVec2(0, 0), ImGuiChildFlags_None);
    ImGui::SeparatorText(std::format("Not at spec ({})", bad.size()).c_str());
    if (bad.empty()) ImGui::TextDisabled(hc.lastRun > 0 ? "Nothing: everything checked runs at spec."
                                                        : "Nothing found in the configuration.");
    table("##bad", bad, true);

    if (!todo.empty()) {
        ImGui::SeparatorText(std::format("Not measured ({})", todo.size()).c_str());
        table("##todo", todo, false);
    }
    if (!notes.empty()) {
        ImGui::SeparatorText(std::format("Notes ({})", notes.size()).c_str());
        table("##notes", notes, true);
    }
    ImGui::Spacing();
    if (ImGui::CollapsingHeader(std::format("At spec ({})###atspec", ok.size()).c_str())) table("##ok", ok, false);
    ImGui::EndChild();
    ImGui::End();
}

// Every probed fact, grouped by component.
void DrawPlatformDetails(App& app, bool* visible) {
    ImGui::SetNextWindowSize(ImVec2(ImGui::GetFontSize() * 48, ImGui::GetFontSize() * 40), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Platform details", visible)) { ImGui::End(); return; }
    auto rows = BuildHardwareRows(app);
    // Static extras not used by any check.
    auto memEnd = std::find_if(rows.rbegin(), rows.rend(), [](const HwRow& r) { return r.section == "Memory"; }).base();
    std::vector<HwRow> dimms;
    for (const auto& d : app.mem.dimms)
        dimms.push_back({"Memory", d.locator.empty() ? "DIMM" : d.locator,
                         std::format("{} {} {} MT/s", FormatBytes(d.sizeMB << 20), Trim(d.manufacturer + " " + d.part), d.configuredMTs),
                         d.partNumberSpeed ? std::format("part number rating {} MT/s", d.partNumberSpeed) : "", {Status::None, ""}});
    rows.insert(memEnd, dimms.begin(), dimms.end());
    std::string section;
    bool open = false;
    for (size_t i = 0; i <= rows.size(); ++i) {
        bool newSection = i == rows.size() || rows[i].section != section;
        if (newSection && open) ImGui::EndTable();
        if (i == rows.size()) break;
        if (newSection) {
            section = rows[i].section;
            ImGui::SeparatorText(section.c_str());
            open = ImGui::BeginTable(section.c_str(), 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp);
            if (open) {
                ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, ImGui::GetFontSize());
                ImGui::TableSetupColumn("label", ImGuiTableColumnFlags_WidthStretch, 0.9f);
                ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch, 1.4f);
                ImGui::TableSetupColumn("spec", ImGuiTableColumnFlags_WidthStretch, 1.1f);
            }
        }
        if (!open) continue;
        const auto& r = rows[i];
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        StatusDot(r.verdict.status, r.verdict.reason);
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(r.label.c_str());
        ImGui::TableNextColumn();
        ImGui::TextWrapped("%s", r.value.c_str());
        if (!r.verdict.reason.empty() && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", r.verdict.reason.c_str());
        ImGui::TableNextColumn();
        ImGui::TextDisabled("%s", r.spec.c_str());
    }
    ImGui::End();
}
