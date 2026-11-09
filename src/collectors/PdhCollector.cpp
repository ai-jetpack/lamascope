#include <windows.h>
#include <pdh.h>
#include <pdhmsg.h>
#include <tlhelp32.h>
#include <algorithm>
#include <chrono>
#include <limits>
#include <map>
#include <string>
#include "collectors/PdhCollector.h"
#include "core/Log.h"
#include "core/Util.h"

namespace {

// Returns instance name -> value for a wildcard counter.
std::map<std::wstring, double> ReadArray(PDH_HCOUNTER c) {
    std::map<std::wstring, double> out;
    DWORD size = 0, count = 0;
    if (PdhGetFormattedCounterArrayW(c, PDH_FMT_DOUBLE | PDH_FMT_NOCAP100, &size, &count, nullptr) != PDH_MORE_DATA)
        return out;
    std::vector<uint8_t> buf(size);
    auto* items = (PDH_FMT_COUNTERVALUE_ITEM_W*)buf.data();
    if (PdhGetFormattedCounterArrayW(c, PDH_FMT_DOUBLE | PDH_FMT_NOCAP100, &size, &count, items) != ERROR_SUCCESS)
        return out;
    for (DWORD i = 0; i < count; ++i)
        if (items[i].FmtValue.CStatus == PDH_CSTATUS_VALID_DATA || items[i].FmtValue.CStatus == PDH_CSTATUS_NEW_DATA)
            out[items[i].szName] = items[i].FmtValue.doubleValue;
    return out;
}

// "0,5" -> 5 (group*64 + index); totals -> -1
int LogicalFromInstance(const std::wstring& name) {
    if (name.find(L'_') != std::wstring::npos) return -1;
    size_t comma = name.find(L',');
    if (comma == std::wstring::npos) return -1;
    return std::stoi(name.substr(0, comma)) * 64 + std::stoi(name.substr(comma + 1));
}

} // namespace

bool PdhCollector::Start(const CpuTopology& cpu) {
    cpu_ = cpu;
    n_ = cpu.logicalCount;
    coreMhz.clear();
    for (size_t i = 0; i < cpu.cores.size(); ++i) coreMhz.push_back(std::make_unique<Series>());
    util_.assign(n_, 0);
    perf_.assign(n_, 0);
    hist_.assign(kHistory, std::vector<float>(n_, 0));
    PDH_HQUERY q = nullptr;
    if (PdhOpenQueryW(nullptr, 0, &q) != ERROR_SUCCESS) {
        Log::E("PDH: cannot open query");
        return false;
    }
    query_ = q;
    PDH_HCOUNTER c;
    if (PdhAddEnglishCounterW(q, L"\\Processor Information(*)\\% Processor Time", 0, &c) == ERROR_SUCCESS) cTime_ = c;
    if (PdhAddEnglishCounterW(q, L"\\Processor Information(*)\\% Processor Performance", 0, &c) == ERROR_SUCCESS) cPerf_ = c;
    if (!noGpuCounters && PdhAddEnglishCounterW(q, L"\\GPU Adapter Memory(*)\\Dedicated Usage", 0, &c) == ERROR_SUCCESS) cDed_ = c;
    if (!noGpuCounters && PdhAddEnglishCounterW(q, L"\\GPU Adapter Memory(*)\\Shared Usage", 0, &c) == ERROR_SUCCESS) cShared_ = c;
    // "Local Usage" is VRAM charged to each process (sums to the adapter total). "Dedicated Usage" also counts
    // surfaces shared from other processes (dwm can read ~15 GB), so it is only a fallback.
    if (!noGpuCounters && PdhAddEnglishCounterW(q, L"\\GPU Process Memory(*)\\Local Usage", 0, &c) == ERROR_SUCCESS ||
        PdhAddEnglishCounterW(q, L"\\GPU Process Memory(*)\\Dedicated Usage", 0, &c) == ERROR_SUCCESS)
        cProcDed_ = c;
    PdhCollectQueryData(q);
    stop_ = false;
    thread_ = std::thread([this] { Loop(); });
    return true;
}

void PdhCollector::Stop() {
    stop_ = true;
    if (thread_.joinable()) thread_.join();
    if (query_) {
        PdhCloseQuery((PDH_HQUERY)query_);
        query_ = nullptr;
    }
}

std::vector<float> PdhCollector::UtilHistory() const {
    std::lock_guard lk(mu_);
    std::vector<float> out((size_t)n_ * kHistory);
    for (int s = 0; s < kHistory; ++s) {
        const auto& col = hist_[(histHead_ + s) % kHistory];
        for (int lp = 0; lp < n_; ++lp) out[(size_t)lp * kHistory + s] = col[lp];
    }
    return out;
}

// Instances look like "pid_1234_luid_0x00000000_0x0000D1A5_phys_0".
void PdhCollector::UpdateGpuProcesses() {
    std::wstring luid;
    {
        std::lock_guard lk(mu_);
        luid = luid_;
    }
    if (!cProcDed_ || luid.empty()) return;
    std::map<unsigned, uint64_t> byPid;
    for (auto& [name, v] : ReadArray((PDH_HCOUNTER)cProcDed_)) {
        if (name.find(luid) == std::wstring::npos || name.rfind(L"pid_", 0) != 0) continue;
        unsigned pid = (unsigned)std::stoul(name.substr(4));
        byPid[pid] += (uint64_t)v;
    }
    std::map<unsigned, std::string> names;
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snap != INVALID_HANDLE_VALUE) {
        PROCESSENTRY32W pe{sizeof(pe)};
        for (BOOL ok = Process32FirstW(snap, &pe); ok; ok = Process32NextW(snap, &pe))
            if (byPid.count(pe.th32ProcessID)) names[pe.th32ProcessID] = ToUtf8(pe.szExeFile);
        CloseHandle(snap);
    }
    std::vector<ProcVram> out;
    for (auto& [pid, bytes] : byPid)
        if (bytes > 0) out.push_back({pid, names.count(pid) ? names[pid] : "pid " + std::to_string(pid), bytes});
    std::sort(out.begin(), out.end(), [](const ProcVram& a, const ProcVram& b) { return a.bytes > b.bytes; });
    std::lock_guard lk(mu_);
    procs_ = std::move(out);
}

void PdhCollector::Loop() {
    auto next = std::chrono::steady_clock::now();
    while (!stop_) {
        next += std::chrono::milliseconds(250);
        std::this_thread::sleep_until(next);
        if (PdhCollectQueryData((PDH_HQUERY)query_) != ERROR_SUCCESS) continue;

        std::vector<float> u(n_, 0), p(n_, 0);
        if (cTime_)
            for (auto& [name, v] : ReadArray((PDH_HCOUNTER)cTime_)) {
                int lp = LogicalFromInstance(name);
                if (lp >= 0 && lp < n_) u[lp] = (float)std::min(100.0, v);
            }
        if (cPerf_)
            for (auto& [name, v] : ReadArray((PDH_HCOUNTER)cPerf_)) {
                int lp = LogicalFromInstance(name);
                if (lp >= 0 && lp < n_) p[lp] = (float)v;
            }

        // The discrete GPU is the adapter with the most dedicated memory in use.
        double ded = 0, shared = 0;
        if (cDed_ && cShared_) {
            auto d = ReadArray((PDH_HCOUNTER)cDed_);
            auto s = ReadArray((PDH_HCOUNTER)cShared_);
            std::wstring best;
            for (auto& [name, v] : d)
                if (v >= ded) { ded = v; best = name; }
            if (s.count(best)) shared = s[best];
            double t = NowSeconds();
            gpuDedicatedGB.Push(t, (float)(ded / double(1ull << 30)));
            gpuSharedGB.Push(t, (float)(shared / double(1ull << 30)));
        }

        // Effective clock per physical core. "% Processor Performance" is measured over the busy part of
        // the sample, so it is noise when a core is almost idle: only trust samples with >= 10% busy
        // time and at most max turbo (+5%). Untrusted / parked -> NaN (gap). Averages are load-weighted.
        {
            const float nan = std::numeric_limits<float>::quiet_NaN();
            double t = NowSeconds();
            double pSum = 0, pW = 0, eSum = 0, eW = 0;
            float pMax = 0;
            for (size_t c = 0; c < cpu_.cores.size(); ++c) {
                float m = 0, w = 0;
                const float cap = cpu_.ClockCap(cpu_.cores[c].pcore);
                for (int lp : cpu_.cores[c].logical) {
                    if (lp >= n_ || lp >= (int)cpu_.baseMhz.size() || u[lp] < 10) continue;
                    float v = cpu_.baseMhz[lp] * p[lp] / 100.0f;
                    if (v <= 0 || v > cap) continue;
                    m = std::max(m, v);
                    w = std::max(w, u[lp]);
                }
                coreMhz[c]->Push(t, m > 0 ? m : nan);
                if (m <= 0) continue;
                if (cpu_.cores[c].pcore) { pSum += m * w; pW += w; pMax = std::max(pMax, m); }
                else { eSum += m * w; eW += w; }
            }
            pAvgMhz.Push(t, pW > 0 ? (float)(pSum / pW) : nan);
            pMaxMhz.Push(t, pW > 0 ? pMax : nan);
            eAvgMhz.Push(t, eW > 0 ? (float)(eSum / eW) : nan);
        }

        if (++tick_ % 4 == 0) UpdateGpuProcesses();   // once per second

        std::lock_guard lk(mu_);
        util_ = u;
        perf_ = p;
        hist_[histHead_] = u;
        histHead_ = (histHead_ + 1) % kHistory;
    }
}
