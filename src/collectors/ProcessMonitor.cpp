#include <windows.h>
#include <tlhelp32.h>
#include <chrono>
#include <map>
#include "collectors/ProcessMonitor.h"
#include "core/Util.h"

namespace {
uint64_t FtToU64(const FILETIME& f) { return ((uint64_t)f.dwHighDateTime << 32) | f.dwLowDateTime; }
}

void ProcessMonitor::Start() {
    stop_ = false;
    thread_ = std::thread([this] { Loop(); });
}

void ProcessMonitor::Stop() {
    stop_ = true;
    if (thread_.joinable()) thread_.join();
}

void ProcessMonitor::Loop() {
    struct Prev { uint64_t create, cpu; };
    std::map<unsigned, Prev> prev;
    auto lastT = std::chrono::steady_clock::now();
    while (!stop_) {
        auto now = std::chrono::steady_clock::now();
        double elapsed100ns = std::chrono::duration<double>(now - lastT).count() * 1e7;
        lastT = now;

        std::vector<ProcInfo> out;
        std::map<unsigned, Prev> next;
        HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
        if (snap != INVALID_HANDLE_VALUE) {
            PROCESSENTRY32W pe{sizeof(pe)};
            for (BOOL ok = Process32FirstW(snap, &pe); ok; ok = Process32NextW(snap, &pe)) {
                if (pe.th32ProcessID == 0) continue;
                ProcInfo p;
                p.pid = pe.th32ProcessID;
                p.name = ToUtf8(pe.szExeFile);
                HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, p.pid);
                DWORD exitCode = STILL_ACTIVE;
                if (h && GetExitCodeProcess(h, &exitCode) && exitCode != STILL_ACTIVE) {   // exited, kept by a handle
                    CloseHandle(h);
                    continue;
                }
                if (h) {
                    FILETIME c, e, k, u;
                    if (GetProcessTimes(h, &c, &e, &k, &u)) {
                        p.createTime = FtToU64(c);
                        uint64_t cpu = FtToU64(k) + FtToU64(u);
                        auto it = prev.find(p.pid);
                        if (it != prev.end() && it->second.create == p.createTime && elapsed100ns > 0)
                            p.cpuPct = (double)(cpu - it->second.cpu) / elapsed100ns * 100.0;
                        next[p.pid] = {p.createTime, cpu};
                    }
                    DWORD_PTR pm = 0, sm = 0;
                    if (GetProcessAffinityMask(h, &pm, &sm)) p.affinity = pm;
                    CloseHandle(h);
                }
                out.push_back(std::move(p));
            }
            CloseHandle(snap);
        }
        prev = std::move(next);
        {
            std::lock_guard lk(mu_);
            procs_ = std::move(out);
        }
        for (int i = 0; i < 20 && !stop_; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}
