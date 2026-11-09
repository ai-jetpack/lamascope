#include <algorithm>
#include <format>
#include "app/App.h"
#include "core/Log.h"
#include "core/Util.h"
#include "platform/Process.h"
#include "diagnosis/DiagnosisEngine.h"
#include "ui/Panels.h"

namespace {

std::string Dur(double s) {
    if (s < 60) return std::format("{:.0f} s", s);
    if (s < 3600) return std::format("{} m {:02} s", (int)s / 60, (int)s % 60);
    return std::format("{} h {:02} m", (int)s / 3600, ((int)s % 3600) / 60);
}

// Rows evaluated by dedicated episode rules instead of the generic passthrough.
bool IsDynamicRow(const HwRow& r) {
    return r.label == "P-core clock" || r.label == "E-core clock" || r.label == "Core clock" ||
           r.label == "Temperature" || r.label == "Clock limiters";
}

} // namespace

void DiagnosisEngine::Observe(const std::string& id, bool cond, Status sev, const std::string& detail, double now,
                              bool hasMetric, float metric, bool lowerIsWorse, bool persistent) {
    Track& t = tracks_[id];
    t.persistent = persistent;
    t.lowerIsWorse = lowerIsWorse;
    double dt = lastEval_ < 0 ? 0 : now - lastEval_;
    if (cond) {
        t.falseSince = -1;
        if (t.condSince < 0) t.condSince = now;
        if (!t.inEpisode && now - t.condSince >= kStartSec) {
            t.inEpisode = true;
            t.startDetail = detail;
            ++t.episodes;
            Log::I("Issue: " + (titles_.count(id) ? titles_[id] : id) + ": " + detail);
            t.totalSec += now - t.condSince;   // count the debounce time too
        } else if (t.inEpisode) {
            t.totalSec += dt;
        }
        if (t.inEpisode) {
            t.lastSeen = now;
            t.detail = detail;
            if (sev > t.severity || t.episodes == 1) t.severity = sev;
            if (hasMetric && (!t.hasWorst || (lowerIsWorse ? metric < t.worst : metric > t.worst))) {
                t.worst = metric;
                t.hasWorst = true;
            }
        }
    } else {
        t.condSince = -1;
        if (t.inEpisode) {
            if (t.falseSince < 0) t.falseSince = now;
            if (now - t.falseSince >= kEndSec) t.inEpisode = false;
        }
    }
}

void DiagnosisEngine::Update(App& app) {
    double now = NowSeconds();
    if (lastEval_ >= 0 && now - lastEval_ < 1.0) return;

    // --- Dynamic rules -------------------------------------------------------------------------
    for (int kind = 0; kind < (app.cpu.hybrid ? 2 : 1); ++kind) {
        const auto& ct = kind == 0 ? app.pClock : app.eClock;
        std::string name = kind == 0 ? (app.cpu.hybrid ? "P-core" : "Core") : "E-core";
        bool cond = ct.loaded && ct.ema > 0 && ct.ref > 0 && ct.ema < 0.9f * ct.ref;
        std::string id = "cpu.clock." + name;
        titles_[id] = std::format("{} clock dropped below 90% of its best sustained {:.2f} GHz", name, ct.ref / 1000);
        Observe(id, cond, Status::Warn,
                std::format("{} loaded cores at {:.2f} GHz (power, thermal or current limit?)", ct.loaded,
                            ct.ema / 1000),
                now, true, ct.ema / 1000, true);
    }

    auto st = app.nvml.Static();
    if (st.ok) {
        auto l = app.nvml.Live();
        struct R { uint64_t bit; const char* name; Status sev; };
        static const R reasons[] = {{0x4, "SW power cap", Status::Warn}, {0x8, "HW slowdown", Status::Bad},
                                    {0x20, "SW thermal slowdown", Status::Warn},
                                    {0x40, "HW thermal slowdown", Status::Bad},
                                    {0x80, "HW power brake", Status::Bad}};
        for (const auto& r : reasons) {
            std::string id = std::string("gpu.thr.") + r.name;
            titles_[id] = std::format("GPU clocks limited by {}", r.name);
            Observe(id, (l.throttle & r.bit) != 0, r.sev,
                    std::format("{} MHz, {:.0f} W, {} C", l.gfxMHz, l.powermW / 1000.0, l.tempC), now, true,
                    (float)l.gfxMHz, true);
        }
        titles_["gpu.temp"] = "GPU temperature at or above 83 C";
        Observe("gpu.temp", l.tempC >= 83, l.tempC >= 90 ? Status::Bad : Status::Warn,
                std::format("{} C", l.tempC), now, true, (float)l.tempC, false);

        // Shared usage alone is normal for CPU+GPU splits (llama.cpp maps the CPU-side weights for the GPU: tens of
        // GB). A spill to system RAM only happens when VRAM is full, so require both.
        float shared = app.pdh.gpuSharedGB.Last();
        bool vramFull = st.vramTotal && l.vramUsed + (256ull << 20) >= st.vramTotal;
        titles_["gpu.shared"] = "VRAM full and the GPU using system RAM (possible spill: very slow)";
        Observe("gpu.shared", vramFull && shared > 1.0f, Status::Warn,
                std::format("{:.2f} GB shared with {} of {} VRAM used", shared, FormatBytes(l.vramUsed),
                            FormatBytes(st.vramTotal)),
                now, true, shared, false);
    }

    // VRAM held by other programs on the inference GPU: every MB is unavailable to the model. Warn when it costs a
    // layer (another --n-cpu-moe step would fit without it); otherwise an Info note while it exceeds 100 MB.
    if (st.ok && st.vramTotal) {
        auto procs = app.pdh.GpuProcesses();
        uint64_t others = 0;
        std::string list;
        for (const auto& p : procs) {
            if (p.name.rfind("llama", 0) == 0) continue;
            others += p.bytes;
            if (p.bytes >= (1ull << 20)) list += std::format("{}{} {:.0f} MB", list.empty() ? "" : ", ", p.name, p.bytes / 1048576.0);
        }
        const double margin = 0.25 * (1 << 30), total = (double)st.vramTotal;
        int fitNow = -1, fitClean = -1;
        const ModelInfo& m = app.model;
        if (m.nLayers > 0 && m.nExpert > 0) {
            fitNow = Placement::SmallestFittingCpuMoe(m, app.settings.launch, total - (double)others - margin);
            fitClean = Placement::SmallestFittingCpuMoe(m, app.settings.launch, total - margin);
        }
        bool costsLayer = fitNow >= 0 && fitClean >= 0 && fitClean < fitNow;
        double mb = others / 1048576.0;
        titles_["gpu.others"] = "Other programs hold VRAM on the inference GPU";
        std::string detail = std::format("{:.0f} MB: {}", mb, list.empty() ? "-" : list);
        if (costsLayer)
            detail += std::format(". Without them --n-cpu-moe {} would fit instead of {} (one more layer of experts on "
                                  "the GPU)", fitClean, fitNow);
        else if (fitNow >= 0)
            detail += ". Not enough to fit another layer on the GPU";
        Observe("gpu.others", costsLayer || mb >= 100, costsLayer ? Status::Warn : Status::Info, detail, now, true,
                (float)mb, false);
    }

    // Room for more expert layers on the GPU. With llama.cpp running: its real --n-cpu-moe (from its command line)
    // and the driver's real free VRAM; each step down moves one layer of experts from the CPU to the GPU, costing
    // that layer's expert bytes (from the GGUF). Without it: the model estimate for the configured parameters.
    if (st.ok && st.vramTotal && app.model.nLayers > 0 && app.model.nExpert > 0) {
        const ModelInfo& m = app.model;
        LaunchParams lp = app.settings.launch;
        std::string who;
        double freeB = -1;
        bool assumed = false;   // llama.cpp runs but its command line could not be read
        bool llamaSeen = false;
        for (const auto& p : app.procMon.Snapshot()) {
            if (!p.IsLlama()) continue;
            llamaSeen = true;
            // Free VRAM at idle overstates the room: processing a prompt needs extra scratch memory (observed: a split
            // that loaded with ~0.5 GB free aborted on the first prompt). So use
            // the peak VRAM seen while this llama process lives, and only after it has done real work.
            auto l0 = app.nvml.Live();
            if (peakPid_ != p.pid) { peakPid_ = p.pid; peakUsed_ = 0; worked_ = false; }
            peakUsed_ = std::max<uint64_t>(peakUsed_, l0.vramUsed);
            if (p.cpuPct >= 200) worked_ = true;     // computing on several cores: a prompt or generation ran
            if (!worked_) break;                     // nothing processed yet: no measured claim
            std::wstring cl = ProcessCommandLine(p.pid);
            int x = CommandLineInt(cl, {L"--n-cpu-moe", L"-ncmoe"});
            if (cl.empty()) assumed = true;          // started as administrator, LlamaScope is not
            else if (x < 0) continue;                // not a MoE split run
            else lp.nCpuMoe = x;
            if (int c = CommandLineInt(cl, {L"-c", L"--ctx-size"}); c > 0) lp.nCtx = c;
            who = p.name;
            freeB = (double)st.vramTotal - (double)peakUsed_;
            break;
        }
        if (!llamaSeen) peakPid_ = 0;
        const double margin = 0.25 * (1 << 30);
        const double base = (double)Placement::Compute(m, lp).GpuTotal();
        int more = 0;
        double need1 = 0;
        if (lp.nCpuMoe > 0) {
            LaunchParams q = lp;
            q.nCpuMoe = lp.nCpuMoe - 1;
            need1 = (double)Placement::Compute(m, q).GpuTotal() - base;
            if (freeB >= 0) {   // measured
                for (int k = 1; k <= lp.nCpuMoe; ++k) {
                    q.nCpuMoe = lp.nCpuMoe - k;
                    if ((double)Placement::Compute(m, q).GpuTotal() - base > freeB - margin) break;
                    more = k;
                }
            } else if (!llamaSeen) {   // nothing running: model estimate + other programs' VRAM
                uint64_t others = 0;
                for (const auto& p : app.pdh.GpuProcesses())
                    if (p.name.rfind("llama", 0) != 0) others += p.bytes;
                int fit = Placement::SmallestFittingCpuMoe(m, lp, (double)st.vramTotal - (double)others - margin);
                if (fit >= 0 && fit < lp.nCpuMoe) more = lp.nCpuMoe - fit;
            }
        }
        titles_["gpu.room"] = "VRAM has room for more expert layers on the GPU";
        std::string detail =
            freeB >= 0 && assumed
                ? std::format("{} runs (its settings are unreadable: run LlamaScope as administrator), assuming the "
                              "configured --n-cpu-moe {}; at its peak the GPU had {:.0f} MB free and one more layer needs {:.0f} MB "
                              "(+256 MB margin): --n-cpu-moe {} should fit",
                              who, lp.nCpuMoe, freeB / 1048576.0, need1 / 1048576.0, lp.nCpuMoe - more)
            : freeB >= 0
                ? std::format("{} runs with --n-cpu-moe {}; at its peak (prompt processing included) the GPU had {:.0f} MB "
                              "free, and one more layer needs {:.0f} MB (+256 MB margin). Restart it with --n-cpu-moe "
                              "{}: less CPU work per token",
                              who, lp.nCpuMoe, freeB / 1048576.0, need1 / 1048576.0, lp.nCpuMoe - more)
                : std::format("estimate for the configured --n-cpu-moe {}: --n-cpu-moe {} fits (each layer needs about "
                              "{:.0f} MB)",
                              lp.nCpuMoe, lp.nCpuMoe - more, need1 / 1048576.0);
        Observe("gpu.room", more > 0, Status::Warn, detail, now, true, (float)more, false);
    }

    // Memory bandwidth saturation (not during LlamaScope's own memory test, which saturates it on purpose).
    if (app.imc.GetState() == ImcCollector::State::Running && app.hwCheck.stage == App::HwCheck::Stage::Idle &&
        !app.memTest.Running()) {
        static std::vector<float> xs, ys;
        auto avg = [&](const Series& s) {
            s.Copy(now - 2, xs, ys);
            double sum = 0;
            for (float y : ys) sum += y;
            return ys.empty() ? 0.0 : sum / ys.size();
        };
        double total = avg(app.imc.readGBs) + avg(app.imc.writeGBs);
        double measured = app.memTest.Result().valid ? app.memTest.Result().bestGBs : 0;
        double limit = measured > 0 ? 0.9 * measured : 0.8 * app.mem.theoreticalGBs;
        titles_["mem.bw"] = measured > 0 ? std::format("Memory bandwidth saturated (over 90% of the measured {:.0f} GB/s)", measured)
                                         : std::format("Memory bandwidth saturated (over 80% of the theoretical {:.0f} GB/s)",
                                                       app.mem.theoreticalGBs);
        Observe("mem.bw", limit > 0 && total >= limit, Status::Warn,
                std::format("{:.1f} GB/s: CPU-side work is waiting on RAM; faster memory or fewer layers on the CPU "
                            "(lower --n-cpu-moe) would help",
                            total),
                now, true, (float)total, false);
    }

    // --- Configuration / hardware rows (stable conditions, no episode stats) ---------------------
    for (const auto& r : BuildHardwareRows(app)) {
        if (IsDynamicRow(r)) continue;
        std::string id = "hw." + r.section + "." + r.label;
        bool cond = r.verdict.status == Status::Warn || r.verdict.status == Status::Bad;
        titles_[id] = r.section + " / " + r.label;
        Observe(id, cond, r.verdict.status, r.verdict.reason, now, false, 0, true, true);
    }

    lastEval_ = now;

    // --- Render findings -------------------------------------------------------------------------
    std::vector<Finding> out;
    for (auto& [id, t] : tracks_) {
        if (t.episodes == 0) continue;
        Finding f;
        f.id = id;
        f.severity = t.severity;
        f.active = t.inEpisode;
        f.lastSeen = t.lastSeen;
        f.detail = t.detail;
        const std::string& title = titles_[id];
        if (t.persistent) {
            f.text = title + ": " + t.startDetail;
            if (!t.inEpisode) f.text += std::format(" (resolved {} ago)", Dur(now - t.lastSeen));
        } else {
            f.text = std::format("{}: {} time{}, {} total", title, t.episodes, t.episodes > 1 ? "s" : "",
                                 Dur(t.totalSec));
            if (t.hasWorst) {
                if (id.rfind("cpu.clock", 0) == 0) f.text += std::format(", lowest {:.2f} GHz", t.worst);
                else if (id == "gpu.temp") f.text += std::format(", peak {:.0f} C", t.worst);
                else if (id == "gpu.shared") f.text += std::format(", peak {:.2f} GB", t.worst);
                else if (id.rfind("gpu.thr", 0) == 0) f.text += std::format(", lowest {:.0f} MHz", t.worst);
            }
            f.text += t.inEpisode ? ", now" : std::format(", last {} ago", Dur(now - t.lastSeen));
        }
        out.push_back(std::move(f));
    }
    std::sort(out.begin(), out.end(), [](const Finding& a, const Finding& b) {
        if (a.active != b.active) return a.active;
        if (a.severity != b.severity) return a.severity > b.severity;
        return a.lastSeen > b.lastSeen;
    });
    std::lock_guard lk(mu_);
    findings_ = std::move(out);
}
