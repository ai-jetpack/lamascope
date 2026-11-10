#include <windows.h>
#include <combaseapi.h>
#include <powrprof.h>
#include <powersetting.h>
#include <tlhelp32.h>
#include <algorithm>
#include <chrono>
#include <format>
#include <fstream>
#include <map>
#include "core/Log.h"
#include "core/Util.h"
#include "json.hpp"
#include "platform/Affinity.h"
#include "platform/Process.h"
#include "tuning/InferenceMode.h"

using nlohmann::json;

namespace {

const wchar_t* kServices[] = {L"SysMain", L"UsoSvc", L"wuauserv", L"BITS", L"DoSvc", L"WSearch", L"DiagTrack", L"dmwappush", L"WerSvc"};
const wchar_t* kTasks[] = {
    L"\\Microsoft\\Windows\\Defrag\\ScheduledDefrag",
    L"\\Microsoft\\Windows\\Windows Defender\\Windows Defender Scheduled Scan",
    L"\\Microsoft\\Windows\\Windows Defender\\Windows Defender Cache Maintenance",
    L"\\Microsoft\\Windows\\UpdateOrchestrator\\Schedule Scan",
    L"\\Microsoft\\Windows\\UpdateOrchestrator\\Schedule Scan Static Task",
    L"\\Microsoft\\Windows\\Customer Experience Improvement Program\\Consolidator",
    L"\\Microsoft\\Windows\\Customer Experience Improvement Program\\UsbCeip",
    L"\\Microsoft\\Windows\\DiskDiagnostic\\Microsoft-Windows-DiskDiagnosticDataCollector",
    L"\\Microsoft\\Windows\\Maintenance\\WinSAT",
    L"\\Microsoft\\Windows\\Application Experience\\PcaPatchDbTask",
    L"\\Microsoft\\Windows\\WindowsAI\\ClickToDo\\ModelCachingUpdate",
};
const wchar_t* kLlamaExes[] = {L"llama-server.exe", L"llama-cli.exe", L"llama-bench.exe"};
// Never touched by P-core isolation (critical or protected; most refuse anyway).
const char* kSkip[] = {"System", "Registry", "smss.exe", "csrss.exe", "wininit.exe", "services.exe", "lsass.exe",
                       "Memory Compression", "Secure System", "LsaIso.exe"};

const GUID kSubProcessor = {0x54533251, 0x82be, 0x4824, {0x96, 0xc1, 0x47, 0xb6, 0x0b, 0x74, 0x0d, 0x00}};
const GUID kCpMinCores = {0x0cc5b647, 0xc1df, 0x4637, {0x89, 0x1a, 0xde, 0xc3, 0x5c, 0x31, 0x85, 0x83}};
const GUID kSubPcie = {0x501a4d13, 0x42af, 0x4429, {0x9f, 0xd1, 0xa8, 0x21, 0x8c, 0x26, 0x8e, 0x20}};
const GUID kAspm = {0xee12f906, 0xd277, 0x404b, {0xb6, 0xda, 0xe5, 0xfa, 0x1a, 0x57, 0x6d, 0xf5}};
const GUID kHighPerf = {0x8c5e7fda, 0xe8bf, 0x4a96, {0x9a, 0x85, 0xa6, 0xe2, 0x3a, 0x8c, 0x63, 0x5c}};
const wchar_t* kBgKey = L"Software\\Microsoft\\Windows\\CurrentVersion\\BackgroundAccessApplications";

std::wstring JournalPath() { return LocalDataDir() + L"\\inference-mode.json"; }

std::string GuidStr(const GUID& g) {
    wchar_t buf[64];
    StringFromGUID2(g, buf, 64);
    return ToUtf8(buf);
}
GUID StrGuid(const std::string& s) {
    GUID g{};
    CLSIDFromString(ToWide(s).c_str(), &g);
    return g;
}
uint64_t FtU64(const FILETIME& f) { return ((uint64_t)f.dwHighDateTime << 32) | f.dwLowDateTime; }

uint64_t CreateTime(HANDLE h) {
    FILETIME c, e, k, u;
    return GetProcessTimes(h, &c, &e, &k, &u) ? FtU64(c) : 0;
}

bool IsLlamaName(const std::string& n) { return n.rfind("llama", 0) == 0; }
// Programs that start other programs. Windows children inherit their parent's CPU mask, so these keep (or get)
// every core: llama.cpp started from a terminal or Explorer must start with the P-cores available, or its own
// --cpu-range / --cpu-strict placement fails and OpenMP sizes its pool for the E-cores only.
bool IsLauncher(const std::string& n) {
    static const char* k[] = {"explorer.exe", "cmd.exe", "powershell.exe", "pwsh.exe", "WindowsTerminal.exe",
                              "OpenConsole.exe", "conhost.exe", "bash.exe", "sh.exe", "wsl.exe", "wslhost.exe"};
    return std::any_of(std::begin(k), std::end(k), [&](const char* x) { return _stricmp(x, n.c_str()) == 0; });
}

// --- services ---
bool ServiceRunning(SC_HANDLE scm, const wchar_t* name) {
    SC_HANDLE s = OpenServiceW(scm, name, SERVICE_QUERY_STATUS);
    if (!s) return false;
    SERVICE_STATUS_PROCESS ss{};
    DWORD n = 0;
    bool running = QueryServiceStatusEx(s, SC_STATUS_PROCESS_INFO, (BYTE*)&ss, sizeof(ss), &n) &&
                   ss.dwCurrentState != SERVICE_STOPPED;
    CloseServiceHandle(s);
    return running;
}

bool StopService(SC_HANDLE scm, const wchar_t* name) {
    SC_HANDLE s = OpenServiceW(scm, name, SERVICE_STOP | SERVICE_QUERY_STATUS);
    if (!s) return false;
    SERVICE_STATUS st{};
    bool ok = ControlService(s, SERVICE_CONTROL_STOP, &st) || GetLastError() == ERROR_SERVICE_NOT_ACTIVE;
    CloseServiceHandle(s);
    return ok;
}

bool StartSvc(SC_HANDLE scm, const wchar_t* name) {
    SC_HANDLE s = OpenServiceW(scm, name, SERVICE_START);
    if (!s) return false;
    bool ok = StartServiceW(s, 0, nullptr) || GetLastError() == ERROR_SERVICE_ALREADY_RUNNING;
    CloseServiceHandle(s);
    return ok;
}

// --- scheduled tasks (schtasks.exe) ---
int TaskEnabled(const std::wstring& path) {   // 1 enabled, 0 disabled, -1 missing / no access
    std::string out;
    if (RunHidden(L"schtasks /Query /TN \"" + path + L"\" /FO CSV /NH", &out, 10000) != 0) return -1;
    return out.find("Disabled") != std::string::npos ? 0 : 1;
}
bool SetTask(const std::wstring& path, bool enable) {
    return RunHidden(L"schtasks /Change /TN \"" + path + (enable ? L"\" /Enable" : L"\" /Disable"), nullptr, 10000) == 0;
}

// --- Defender (PowerShell cmdlets) ---
std::wstring PsQuote(const std::string& s) {
    std::wstring w = ToWide(s), out = L"'";
    for (wchar_t c : w) out += (c == L'\'') ? std::wstring(L"''") : std::wstring(1, c);
    return out + L"'";
}
std::vector<std::string> DefenderList(const char* prop) {
    std::string out;
    RunHidden(ToWide(std::format("powershell -NoProfile -NonInteractive -Command \"(Get-MpPreference).{} -join '|'\"", prop)), &out, 30000);
    std::vector<std::string> v;
    size_t a = 0;
    out = Trim(out);
    while (a < out.size()) {
        size_t b = out.find('|', a);
        if (b == std::string::npos) b = out.size();
        std::string item = Trim(out.substr(a, b - a));
        if (!item.empty()) v.push_back(item);
        a = b + 1;
    }
    return v;
}
bool DefenderCall(const wchar_t* verb, const wchar_t* param, const std::string& value) {
    std::wstring cmd = std::wstring(L"powershell -NoProfile -NonInteractive -Command \"") + verb + L"-MpPreference -" + param +
                       L" " + PsQuote(value) + L"\"";
    return RunHidden(cmd, nullptr, 30000) == 0;
}
bool ContainsI(const std::vector<std::string>& v, const std::string& s) {
    auto lower = [](std::string x) { for (auto& c : x) c = (char)tolower((unsigned char)c); return x; };
    for (auto& e : v)
        if (lower(e) == lower(s)) return true;
    return false;
}

// --- registry ---
int ReadDword(HKEY root, const wchar_t* key, const wchar_t* name) {
    DWORD v = 0, sz = sizeof(v);
    return RegGetValueW(root, key, name, RRF_RT_REG_DWORD, nullptr, &v, &sz) == ERROR_SUCCESS ? (int)v : -1;
}

} // namespace

// ------------------------------------------------------------------------------------------------

InferenceMode::InferenceMode() {
    actions_ = {
        {"isolate", "Move other programs to E-cores",
         "Every other program is limited to the E-cores, so the P-cores only run llama.cpp. Re-applied every 2 s to new programs. Protected system processes cannot be moved. Terminals and Explorer keep all cores (they are idle), because programs started from them inherit their cores: llama.cpp must start with the P-cores available for its --cpu-range / --cpu-strict / -t settings to work.",
         "low", true, false, ""},
        {"llama", "llama.cpp: high priority, no throttling",
         "Raises the priority of llama-cli / llama-server / llama-bench and turns off Windows power throttling (EcoQoS) for them. Their core placement is left to their own settings (-t, --cpu-range, --cpu-strict): changing the CPU mask of a running process overrides llama.cpp's per-thread placement. Warns if llama.cpp started without access to the P-cores.",
         "low", true, false, ""},
        {"services", "Pause background services",
         "Stops SysMain (prefetch), Windows Update (UsoSvc, wuauserv, BITS, DoSvc), Search indexing, telemetry and error reporting while inference mode is on. Started again on restore.",
         "low", true, false, ""},
        {"tasks", "Pause scheduled maintenance",
         "Disables defrag, Defender scheduled scan, update scans, CEIP, disk diagnostics, WinSAT and Windows AI tasks so they cannot start mid-inference. Re-enabled on restore.",
         "low", true, false, ""},
        {"power", "Performance power settings",
         "High performance plan and PCIe link power saving off. Core parking is deliberately left alone: forcing all cores unparked measured 13% slower generation on a hybrid CPU (idle E-cores can no longer sleep).",
         "low", true, false, ""},
        {"bgapps", "Pause Store apps in the background", "Stops Microsoft Store apps from running in the background.",
         "low", true, false, ""},
        {"defender", "Defender: exclude the model folder and llama.cpp",
         "Stops real-time scanning of the model files and llama.cpp processes (faster model load, no scan stalls). Security trade-off: files in that folder are not scanned. Real-time protection stays on for everything else. Removed on restore.",
         "medium", false, true, ""},
    };
    LoadJournal();
}

InferenceMode::~InferenceMode() {
    if (worker_.joinable()) worker_.join();
    stopEnforce_ = true;
    if (enforcer_.joinable()) enforcer_.join();
}

void InferenceMode::SetContext(const CpuTopology& cpu, const std::string& modelPath) {
    std::lock_guard lk(mu_);
    cpu_ = cpu;
    modelPath_ = modelPath;
}

void InferenceMode::SetEnabled(const std::string& id, bool on) {
    std::lock_guard lk(mu_);
    for (auto& a : actions_)
        if (a.id == id) a.enabled = on;
}

bool InferenceMode::On(const std::string& id) const {
    std::lock_guard lk(mu_);
    for (auto& a : actions_)
        if (a.id == id) return a.enabled;
    return false;
}

void InferenceMode::Note(const std::string& s) {
    Log::I("Inference mode: " + s);
    std::lock_guard lk(mu_);
    log_.push_back(s);
    if (log_.size() > 200) log_.erase(log_.begin());
}

void InferenceMode::ProbeAsync() {
    if (busy_) return;
    if (worker_.joinable()) worker_.join();
    busy_ = true;
    worker_ = std::thread([this] { Probe(); busy_ = false; });
}
void InferenceMode::EnterAsync() {
    if (busy_ || active_) return;
    if (worker_.joinable()) worker_.join();
    busy_ = true;
    worker_ = std::thread([this] { Enter(); Probe(); busy_ = false; });
}
void InferenceMode::RestoreAsync() {
    if (busy_) return;
    if (worker_.joinable()) worker_.join();
    busy_ = true;
    worker_ = std::thread([this] { Restore(); Probe(); busy_ = false; });
}
void InferenceMode::RestoreBlocking() {
    if (worker_.joinable()) worker_.join();
    if (active_) Restore();
}

// ------------------------------------------------------------------------------------------------

void InferenceMode::Probe() {
    CpuTopology cpu;
    std::string model;
    {
        std::lock_guard lk(mu_);
        cpu = cpu_;
        model = modelPath_;
    }
    uint64_t pMask = Affinity::PCoreMask(cpu);
    std::map<std::string, std::string> st;

    // Programs allowed on P-cores; llama processes.
    int onP = 0;
    std::string llama;
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snap != INVALID_HANDLE_VALUE) {
        PROCESSENTRY32W pe{sizeof(pe)};
        for (BOOL ok = Process32FirstW(snap, &pe); ok; ok = Process32NextW(snap, &pe)) {
            if (pe.cntThreads == 0) continue;   // exited process kept alive by an open handle
            std::string name = ToUtf8(pe.szExeFile);
            HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pe.th32ProcessID);
            DWORD_PTR pm = 0, sm = 0;
            bool known = h && GetProcessAffinityMask(h, &pm, &sm);
            DWORD exitCode = STILL_ACTIVE;
            if (h) GetExitCodeProcess(h, &exitCode);
            if (exitCode != STILL_ACTIVE) { CloseHandle(h); continue; }   // exited, kept alive only by an open handle
            if (IsLlamaName(name) && h) {
                DWORD pri = GetPriorityClass(h);
                llama += std::format("{}{} ({}, {} priority)", llama.empty() ? "" : "; ", name,
                                     !(pm & pMask) ? "E-cores only!" : (pm & ~pMask) ? "P+E cores" : "P-cores only", pri == HIGH_PRIORITY_CLASS ? "high" : "normal");
            } else if (pe.th32ProcessID != GetCurrentProcessId() && (!known || (pm & pMask))) {
                ++onP;
            }
            if (h) CloseHandle(h);
        }
        CloseHandle(snap);
    }
    st["isolate"] = std::format("{} programs may run on P-cores", onP);
    st["llama"] = llama.empty() ? "not running (applied when it starts)" : llama;

    std::string running;
    if (SC_HANDLE scm = OpenSCManagerW(nullptr, nullptr, SC_MANAGER_CONNECT)) {
        for (auto* s : kServices)
            if (ServiceRunning(scm, s)) running += (running.empty() ? "" : ", ") + ToUtf8(s);
        CloseServiceHandle(scm);
    }
    st["services"] = running.empty() ? "none running" : "running: " + running;

    int enabled = 0, total = 0;
    for (auto* t : kTasks) {
        int e = TaskEnabled(t);
        if (e >= 0) { ++total; enabled += e; }
    }
    st["tasks"] = std::format("{} of {} enabled", enabled, total);

    GUID* active = nullptr;
    std::string power = "unknown";
    if (PowerGetActiveScheme(nullptr, &active) == ERROR_SUCCESS) {
        wchar_t name[128] = {};
        DWORD sz = sizeof(name);
        PowerReadFriendlyName(nullptr, active, nullptr, nullptr, (UCHAR*)name, &sz);
        DWORD park = 0, aspm = 0;
        PowerReadACValueIndex(nullptr, active, &kSubProcessor, &kCpMinCores, &park);
        PowerReadACValueIndex(nullptr, active, &kSubPcie, &kAspm, &aspm);
        power = std::format("{}, min cores unparked {}%, PCIe power saving {}", ToUtf8(name), park, aspm ? "on" : "off");
        LocalFree(active);
    }
    st["power"] = power;
    int bg = ReadDword(HKEY_CURRENT_USER, kBgKey, L"GlobalUserDisabled");
    st["bgapps"] = bg == 1 ? "already paused" : "allowed";

    std::string folder = model.substr(0, model.find_last_of("\\/"));
    auto paths = DefenderList("ExclusionPath");
    int stale = 0;
    for (auto& p : paths)
        if (GetFileAttributesW(ToWide(p).c_str()) == INVALID_FILE_ATTRIBUTES) ++stale;
    st["defender"] = folder.empty() ? "no model selected"
                                    : std::format("model folder {}excluded{}", ContainsI(paths, folder) ? "" : "not ",
                                                  stale ? std::format("; {} stale exclusions (paths no longer exist)", stale) : "");

    std::lock_guard lk(mu_);
    for (auto& a : actions_) a.state = st[a.id];
}

void InferenceMode::Enter() {
    Note("entering");
    {
        std::lock_guard lk(mu_);
        notMoved_.clear();
    }
    affinity_.clear();
    llamaTuned_.clear();
    stoppedServices_.clear();
    disabledTasks_.clear();
    addedExclPaths_.clear();
    addedExclProcs_.clear();
    origScheme_.clear();
    tunedScheme_.clear();
    origCpMinCores_ = origAspm_ = -1;
    origBgApps_ = -2;
    moved_ = 0;
    FILETIME now;
    GetSystemTimeAsFileTime(&now);
    enterTime_ = (uint64_t(now.dwHighDateTime) << 32) | now.dwLowDateTime;
    active_ = true;
    SaveJournal();   // journal exists before the first change

    if (On("services")) {
        if (SC_HANDLE scm = OpenSCManagerW(nullptr, nullptr, SC_MANAGER_CONNECT)) {
            for (auto* s : kServices) {
                if (!ServiceRunning(scm, s)) continue;
                stoppedServices_.push_back(ToUtf8(s));
                SaveJournal();
                Note(StopService(scm, s) ? "stopped service " + ToUtf8(s) : "could not stop service " + ToUtf8(s));
            }
            CloseServiceHandle(scm);
        }
    }
    if (On("tasks")) {
        for (auto* t : kTasks) {
            if (TaskEnabled(t) != 1) continue;
            disabledTasks_.push_back(ToUtf8(t));
            SaveJournal();
            std::string shortName = ToUtf8(t).substr(ToUtf8(t).find_last_of('\\') + 1);
            if (SetTask(t, false)) {
                Note("paused task " + shortName);
            } else {   // protected: still enabled, nothing to restore
                disabledTasks_.pop_back();
                SaveJournal();
                Note("could not pause task " + shortName + " (protected)");
            }
        }
    }
    if (On("power")) {
        GUID* active = nullptr;
        if (PowerGetActiveScheme(nullptr, &active) == ERROR_SUCCESS) {
            origScheme_ = GuidStr(*active);
            GUID target = *active;
            LocalFree(active);
            if (!IsEqualGUID(target, kHighPerf) && PowerSetActiveScheme(nullptr, &kHighPerf) == ERROR_SUCCESS) {
                target = kHighPerf;
                Note("switched to the High performance plan");
            }
            tunedScheme_ = GuidStr(target);
            DWORD v = 0;
            // Core parking (CPMINCORES) is not changed on purpose: unparking all cores measured -13% t/s (idle
            // E-cores can no longer sleep). But the stock High performance plan often has parking off (100), so when
            // we switched plans on a hybrid CPU, carry over the previous plan's value (journaled, restored).
            if (cpu_.hybrid && !IsEqualGUID(target, StrGuid(origScheme_)) &&
                PowerReadACValueIndex(nullptr, &target, &kSubProcessor, &kCpMinCores, &v) == ERROR_SUCCESS && v >= 100) {
                GUID prev = StrGuid(origScheme_);
                DWORD pv = 10;
                PowerReadACValueIndex(nullptr, &prev, &kSubProcessor, &kCpMinCores, &pv);
                if (pv < 100) {
                    origCpMinCores_ = (int)v;
                    SaveJournal();
                    PowerWriteACValueIndex(nullptr, &target, &kSubProcessor, &kCpMinCores, pv);
                    Note(std::format("kept core parking allowed (min {}% unparked, as in the previous plan)", pv));
                }
            }
            if (PowerReadACValueIndex(nullptr, &target, &kSubPcie, &kAspm, &v) == ERROR_SUCCESS) origAspm_ = (int)v;
            SaveJournal();
            PowerWriteACValueIndex(nullptr, &target, &kSubPcie, &kAspm, 0);
            PowerSetActiveScheme(nullptr, &target);
            Note("PCIe power saving off");
        }
    }
    if (On("bgapps")) {
        origBgApps_ = ReadDword(HKEY_CURRENT_USER, kBgKey, L"GlobalUserDisabled");
        SaveJournal();
        DWORD one = 1;
        if (RegSetKeyValueW(HKEY_CURRENT_USER, kBgKey, L"GlobalUserDisabled", REG_DWORD, &one, sizeof(one)) == ERROR_SUCCESS)
            Note("paused background Store apps");
    }
    if (On("defender")) {
        std::string folder;
        {
            std::lock_guard lk(mu_);
            folder = modelPath_.substr(0, modelPath_.find_last_of("\\/"));
        }
        auto paths = DefenderList("ExclusionPath");
        auto procs = DefenderList("ExclusionProcess");
        if (!folder.empty() && !ContainsI(paths, folder)) {
            addedExclPaths_.push_back(folder);
            SaveJournal();
            Note(DefenderCall(L"Add", L"ExclusionPath", folder) ? "Defender: excluded " + folder
                                                                  : "Defender: could not add exclusion (Tamper Protection?)");
        }
        for (auto* e : kLlamaExes) {
            std::string n = ToUtf8(e);
            if (ContainsI(procs, n)) continue;
            addedExclProcs_.push_back(n);
            SaveJournal();
            DefenderCall(L"Add", L"ExclusionProcess", n);
        }
        if (!addedExclProcs_.empty()) Note("Defender: excluded llama.cpp processes");
    }
    if (On("isolate") || On("llama")) {
        stopEnforce_ = false;
        if (enforcer_.joinable()) enforcer_.join();
        enforcer_ = std::thread([this] { Enforce(); });
    }
    Note("active");
}

void InferenceMode::Enforce() {
    CpuTopology cpu;
    {
        std::lock_guard lk(mu_);
        cpu = cpu_;
    }
    const uint64_t pMask = Affinity::PCoreMask(cpu), eMask = Affinity::ECoreMask(cpu);
    const DWORD self = GetCurrentProcessId();
    while (!stopEnforce_) {
        bool changed = false;
        std::vector<std::string> failed;
        HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
        if (snap != INVALID_HANDLE_VALUE) {
            PROCESSENTRY32W pe{sizeof(pe)};
            for (BOOL ok = Process32FirstW(snap, &pe); ok; ok = Process32NextW(snap, &pe)) {
                if (pe.cntThreads == 0) continue;   // exited process kept alive by an open handle
                DWORD pid = pe.th32ProcessID;
                std::string name = ToUtf8(pe.szExeFile);
                // Any LlamaScope instance manages its own affinity (E-cores; all cores during hardware tests).
                if (pid <= 4 || pid == self || _wcsicmp(pe.szExeFile, L"LlamaScope.exe") == 0 || std::any_of(std::begin(kSkip), std::end(kSkip), [&](const char* s) { return name == s; }))
                    continue;
                bool llama = IsLlamaName(name);
                if (llama && !On("llama")) continue;
                if (!llama && (!On("isolate") || !eMask)) continue;
                HANDLE h = OpenProcess(PROCESS_SET_INFORMATION | PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
                if (!h) {
                    if (!llama) failed.push_back(name);
                    continue;
                }
                uint64_t ct = CreateTime(h);
                DWORD_PTR cur = 0, sys = 0;
                GetProcessAffinityMask(h, &cur, &sys);
                auto known = std::find_if(affinity_.begin(), affinity_.end(), [&](auto& r) { return r.pid == pid && r.create == ct; });
                if (llama) {
                    if (known == affinity_.end()) {
                        affinity_.push_back({pid, ct, 0, name});   // mask 0: affinity never touched
                        SaveJournal();
                        SetPriorityClass(h, HIGH_PRIORITY_CLASS);
                        PROCESS_POWER_THROTTLING_STATE pt{PROCESS_POWER_THROTTLING_CURRENT_VERSION,
                                                          PROCESS_POWER_THROTTLING_EXECUTION_SPEED | PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION, 0};
                        SetProcessInformation(h, ProcessPowerThrottling, &pt, sizeof(pt));
                        if ((cur & pMask) != (pMask & sys))
                            Note(std::format("WARNING {} (pid {}) started without all P-cores (CPU mask 0x{:X}, inherited "
                                             "from the program that launched it): restart it so -t / --cpu-range apply",
                                             name, pid, (uint64_t)cur));
                        else
                            Note(std::format("{} (pid {}): high priority, no power throttling", name, pid));
                    }
                } else if (IsLauncher(name)) {
                    if (cur != sys) {   // inherited the E-core mask from a moved parent: give every core back
                        if (known == affinity_.end()) {
                            affinity_.push_back({pid, ct, cur, name});
                            changed = true;
                        }
                        SetProcessAffinityMask(h, sys);
                    }
                } else if (cur & pMask) {
                    if (known == affinity_.end()) {
                        affinity_.push_back({pid, ct, cur, name});
                        changed = true;
                    }
                    uint64_t target = (cur & eMask) ? (cur & eMask) : eMask;
                    if (SetProcessAffinityMask(h, (DWORD_PTR)target)) ++moved_;
                    else failed.push_back(name);
                }
                CloseHandle(h);
            }
            CloseHandle(snap);
        }
        if (changed) SaveJournal();
        std::sort(failed.begin(), failed.end());
        failed.erase(std::unique(failed.begin(), failed.end()), failed.end());
        {
            std::lock_guard lk(mu_);
            notMoved_ = failed;
        }
        for (int i = 0; i < 20 && !stopEnforce_; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void InferenceMode::Restore() {
    Note("restoring");
    stopEnforce_ = true;
    if (enforcer_.joinable()) enforcer_.join();

    int restored = 0;
    for (const auto& r : affinity_) {
        HANDLE h = OpenProcess(PROCESS_SET_INFORMATION | PROCESS_QUERY_LIMITED_INFORMATION, FALSE, r.pid);
        if (!h) continue;
        if (CreateTime(h) == r.create) {
            if (r.mask && SetProcessAffinityMask(h, (DWORD_PTR)r.mask)) ++restored;   // 0 = llama.cpp, never re-masked
            if (IsLlamaName(r.name)) SetPriorityClass(h, NORMAL_PRIORITY_CLASS);
        }
        CloseHandle(h);
    }
    if (!affinity_.empty()) Note(std::format("restored CPU affinity of {} programs", restored));

    // Programs started during inference mode inherited the E-core mask from a moved parent and were never
    // recorded: without this they would stay on the E-cores after restore.
    if (enterTime_ && cpu_.hybrid) {
        const uint64_t eMask = Affinity::ECoreMask(cpu_);
        int freed = 0;
        HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
        if (snap != INVALID_HANDLE_VALUE) {
            PROCESSENTRY32W pe{sizeof(pe)};
            for (BOOL ok = Process32FirstW(snap, &pe); ok; ok = Process32NextW(snap, &pe)) {
                if (pe.cntThreads == 0) continue;   // exited process kept alive by an open handle
                if (pe.th32ProcessID == GetCurrentProcessId() || _wcsicmp(pe.szExeFile, L"LlamaScope.exe") == 0) continue;
                HANDLE h = OpenProcess(PROCESS_SET_INFORMATION | PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pe.th32ProcessID);
                if (!h) continue;
                DWORD_PTR pm = 0, sm = 0;
                if (CreateTime(h) >= enterTime_ && GetProcessAffinityMask(h, &pm, &sm) && pm == (eMask & sm) && pm != sm &&
                    SetProcessAffinityMask(h, sm))
                    ++freed;
                CloseHandle(h);
            }
            CloseHandle(snap);
        }
        if (freed) Note(std::format("gave all cores back to {} programs started during inference mode", freed));
    }

    if (!stoppedServices_.empty()) {
        if (SC_HANDLE scm = OpenSCManagerW(nullptr, nullptr, SC_MANAGER_CONNECT)) {
            for (auto& s : stoppedServices_) StartSvc(scm, ToWide(s).c_str());
            CloseServiceHandle(scm);
            Note("restarted services: " + std::to_string(stoppedServices_.size()));
        }
    }
    for (auto& t : disabledTasks_) SetTask(ToWide(t), true);
    if (!disabledTasks_.empty()) Note("re-enabled scheduled tasks: " + std::to_string(disabledTasks_.size()));

    if (!tunedScheme_.empty()) {
        GUID tuned = StrGuid(tunedScheme_);
        if (origCpMinCores_ >= 0) PowerWriteACValueIndex(nullptr, &tuned, &kSubProcessor, &kCpMinCores, origCpMinCores_);
        if (origAspm_ >= 0) PowerWriteACValueIndex(nullptr, &tuned, &kSubPcie, &kAspm, origAspm_);
        GUID orig = origScheme_.empty() ? tuned : StrGuid(origScheme_);
        PowerSetActiveScheme(nullptr, &orig);
        Note("power settings restored");
    }
    if (origBgApps_ != -2) {
        if (origBgApps_ == -1) RegDeleteKeyValueW(HKEY_CURRENT_USER, kBgKey, L"GlobalUserDisabled");
        else {
            DWORD v = (DWORD)origBgApps_;
            RegSetKeyValueW(HKEY_CURRENT_USER, kBgKey, L"GlobalUserDisabled", REG_DWORD, &v, sizeof(v));
        }
    }
    for (auto& p : addedExclPaths_) DefenderCall(L"Remove", L"ExclusionPath", p);
    for (auto& p : addedExclProcs_) DefenderCall(L"Remove", L"ExclusionProcess", p);
    if (!addedExclPaths_.empty() || !addedExclProcs_.empty()) Note("Defender exclusions removed");

    affinity_.clear();
    stoppedServices_.clear();
    disabledTasks_.clear();
    addedExclPaths_.clear();
    addedExclProcs_.clear();
    tunedScheme_.clear();
    origBgApps_ = -2;
    active_ = false;
    leftover_ = false;
    DeleteFileW(JournalPath().c_str());
    Note("restored");
}

// ------------------------------------------------------------------------------------------------

void InferenceMode::SaveJournal() {
    json j;
    j["active"] = (bool)active_;
    json aff = json::array();
    for (auto& r : affinity_) aff.push_back({{"pid", r.pid}, {"create", r.create}, {"mask", r.mask}, {"name", r.name}});
    j["affinity"] = aff;
    j["services"] = stoppedServices_;
    j["tasks"] = disabledTasks_;
    j["exclPaths"] = addedExclPaths_;
    j["exclProcs"] = addedExclProcs_;
    j["origScheme"] = origScheme_;
    j["tunedScheme"] = tunedScheme_;
    j["origCpMinCores"] = origCpMinCores_;
    j["origAspm"] = origAspm_;
    j["origBgApps"] = origBgApps_;
    j["enterTime"] = enterTime_;
    std::wstring tmp = JournalPath() + L".tmp";
    {
        std::ofstream f(tmp);
        f << j.dump(1);
    }
    MoveFileExW(tmp.c_str(), JournalPath().c_str(), MOVEFILE_REPLACE_EXISTING);
}

void InferenceMode::LoadJournal() {
    std::ifstream f(JournalPath());
    if (!f) return;
    try {
        json j = json::parse(f);
        for (auto& a : j.value("affinity", json::array()))
            affinity_.push_back({a["pid"], a["create"], a["mask"], a["name"]});
        stoppedServices_ = j.value("services", std::vector<std::string>{});
        disabledTasks_ = j.value("tasks", std::vector<std::string>{});
        addedExclPaths_ = j.value("exclPaths", std::vector<std::string>{});
        addedExclProcs_ = j.value("exclProcs", std::vector<std::string>{});
        origScheme_ = j.value("origScheme", "");
        tunedScheme_ = j.value("tunedScheme", "");
        origCpMinCores_ = j.value("origCpMinCores", -1);
        origAspm_ = j.value("origAspm", -1);
        origBgApps_ = j.value("origBgApps", -2);
        enterTime_ = j.value("enterTime", uint64_t(0));
        leftover_ = true;
        active_ = true;   // so Restore() runs everything
        Log::W("Inference mode: found an unrestored session from a previous run");
    } catch (...) {
        DeleteFileW(JournalPath().c_str());
    }
}
