#include <cstdio>
#include <share.h>
#include <mutex>
#include "core/Log.h"
#include "core/TimeSeries.h"
#include "core/Util.h"

namespace {
std::mutex g_mu;
std::vector<Log::Entry> g_entries;
FILE* g_file = nullptr;
}

void Log::Write(Level level, const std::string& msg) {
    std::lock_guard lk(g_mu);
    if (g_entries.size() > 5000) g_entries.erase(g_entries.begin(), g_entries.begin() + 1000);
    double t = NowSeconds();
    g_entries.push_back({t, level, msg});
    // Mirror to %LOCALAPPDATA%\LlamaScope\llamascope.log (truncated each start).
    if (!g_file) g_file = _wfsopen((LocalDataDir() + L"\\llamascope.log").c_str(), L"w", _SH_DENYNO);
    if (g_file) {
        static const char* lv[] = {"INFO", "WARN", "ERROR"};
        fprintf(g_file, "%8.2f %-5s %s\n", t, lv[level], msg.c_str());
        fflush(g_file);
    }
}

std::vector<Log::Entry> Log::Snapshot() {
    std::lock_guard lk(g_mu);
    return g_entries;
}
