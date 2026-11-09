#include <windows.h>
#include <intrin.h>
#include <algorithm>
#include <format>
#include <vector>
#include "collectors/ImcCollector.h"
#include "core/Log.h"
#include "core/Util.h"

namespace {

constexpr double kInterval = 0.5;   // seconds per emon sample
const wchar_t* kEvents =
    L"UNC_MC0_RDCAS_COUNT_FREERUN,UNC_MC1_RDCAS_COUNT_FREERUN,UNC_MC0_WRCAS_COUNT_FREERUN,UNC_MC1_WRCAS_COUNT_FREERUN";

// TSC ticks per second (emon reports each interval's length in TSC ticks).
double CalibrateTsc() {
    LARGE_INTEGER f, q0, q1;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&q0);
    uint64_t t0 = __rdtsc();
    Sleep(50);
    QueryPerformanceCounter(&q1);
    uint64_t t1 = __rdtsc();
    return double(t1 - t0) / (double(q1.QuadPart - q0.QuadPart) / double(f.QuadPart));
}

uint64_t ParseCount(const std::string& s) {
    uint64_t v = 0;
    for (char c : s)
        if (c >= '0' && c <= '9') v = v * 10 + (c - '0');
    return v;
}

} // namespace

std::wstring ImcCollector::FindEmon() {
    // VTune installs as ...\oneAPI\vtune\<version>\bin64; "latest" is a junction to the newest.
    const wchar_t* roots[] = {L"C:\\Program Files (x86)\\Intel\\oneAPI\\vtune", L"C:\\Program Files\\Intel\\oneAPI\\vtune"};
    for (const wchar_t* root : roots) {
        std::wstring latest = std::wstring(root) + L"\\latest\\bin64\\emon.exe";
        if (GetFileAttributesW(latest.c_str()) != INVALID_FILE_ATTRIBUTES) return latest;
        std::vector<std::wstring> found;
        WIN32_FIND_DATAW fd;
        HANDLE h = FindFirstFileW((std::wstring(root) + L"\\*").c_str(), &fd);
        if (h == INVALID_HANDLE_VALUE) continue;
        do {
            if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) || fd.cFileName[0] == L'.') continue;
            std::wstring p = std::wstring(root) + L"\\" + fd.cFileName + L"\\bin64\\emon.exe";
            if (GetFileAttributesW(p.c_str()) != INVALID_FILE_ATTRIBUTES) found.push_back(p);
        } while (FindNextFileW(h, &fd));
        FindClose(h);
        if (!found.empty()) return *std::max_element(found.begin(), found.end());
    }
    return {};
}

void ImcCollector::Fail(const std::string& why) {
    {
        std::lock_guard lk(mu_);
        detail_ = why;
    }
    state_ = State::Failed;
    Log::W("Memory traffic: " + why);
}

void ImcCollector::Start(bool elevated) {
    if (thread_.joinable()) return;
    std::wstring emon = FindEmon();
    {
        std::lock_guard lk(mu_);
        emon_ = emon;
    }
    if (emon.empty()) {
        std::lock_guard lk(mu_);
        detail_ = "needs Intel VTune (its emon tool and sampling driver read the memory controller)";
        state_ = State::NoEmon;
        return;
    }
    if (!elevated) {
        std::lock_guard lk(mu_);
        detail_ = "needs administrator rights (the sampling driver only accepts elevated clients)";
        state_ = State::NotElevated;
        return;
    }
    state_ = State::Starting;
    stop_ = false;
    thread_ = std::thread([this] { Loop(); });
}

void ImcCollector::Stop() {
    stop_ = true;
    if (process_) TerminateProcess((HANDLE)process_, 0);   // unblocks the pipe read
    if (thread_.joinable()) thread_.join();
    if (job_) { CloseHandle((HANDLE)job_); job_ = nullptr; }
}

void ImcCollector::Loop() {
    const double tscHz = CalibrateTsc();

    SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
    HANDLE rd = nullptr, wr = nullptr;
    if (!CreatePipe(&rd, &wr, &sa, 1 << 16)) return Fail("could not create pipe");
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);

    // Kill emon if LlamaScope exits or crashes.
    HANDLE job = CreateJobObjectW(nullptr, nullptr);
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION li{};
    li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    SetInformationJobObject(job, JobObjectExtendedLimitInformation, &li, sizeof(li));
    job_ = job;

    std::wstring emon = EmonPath();
    std::wstring cmd = std::format(L"\"{}\" -C \"{}\" -t{} -l0", emon, kEvents, kInterval);
    STARTUPINFOW si{sizeof(si)};
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = wr;
    si.hStdError = wr;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    PROCESS_INFORMATION pi{};
    // Inherits LlamaScope's E-core affinity, so it never competes with inference on the P-cores.
    BOOL ok = CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW | CREATE_SUSPENDED,
                             nullptr, nullptr, &si, &pi);
    CloseHandle(wr);
    if (!ok) {
        CloseHandle(rd);
        return Fail(std::format("could not start emon (error {})", GetLastError()));
    }
    AssignProcessToJobObject(job, pi.hProcess);
    ResumeThread(pi.hThread);
    CloseHandle(pi.hThread);
    process_ = pi.hProcess;
    Log::I("Memory traffic: started " + ToUtf8(emon));

    // Output: one line per event, "NAME\t<interval TSC ticks>\t<count>\t...", each sample ends with "==========".
    std::string buf, lastLines;
    uint64_t rdCas = 0, wrCas = 0, tsc = 0;
    int seen = 0;
    double tPrev = -1;
    char chunk[4096];
    DWORD n = 0;
    while (!stop_ && ReadFile(rd, chunk, sizeof(chunk), &n, nullptr) && n > 0) {
        buf.append(chunk, n);
        size_t pos;
        while ((pos = buf.find('\n')) != std::string::npos) {
            std::string line = buf.substr(0, pos);
            buf.erase(0, pos + 1);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.rfind("UNC_MC", 0) == 0) {
                size_t a = line.find('\t'), b = line.find('\t', a + 1), c = line.find('\t', b + 1);
                if (a == std::string::npos || b == std::string::npos) continue;
                tsc = ParseCount(line.substr(a + 1, b - a - 1));
                uint64_t v = ParseCount(line.substr(b + 1, c == std::string::npos ? std::string::npos : c - b - 1));
                (line.find("RDCAS") != std::string::npos ? rdCas : wrCas) += v;
                ++seen;
            } else if (line.rfind("==========", 0) == 0) {
                if (seen == 4 && tsc > 0) {
                    double dt = tsc / tscHz;
                    // Samples can arrive in bursts (pipe buffering): space them by their real length, never in
                    // the future, and resync if they fall far behind.
                    double now = NowSeconds();
                    double t = tPrev < 0 ? now : std::min(now, tPrev + dt);
                    if (now - t > 3) t = now;
                    tPrev = t;
                    readGBs.Push(t, (float)(rdCas * 64.0 / dt / 1e9));
                    writeGBs.Push(t, (float)(wrCas * 64.0 / dt / 1e9));
                    if (state_ != State::Running) {
                        state_ = State::Running;
                        Log::I(std::format("Memory traffic: first sample {:.1f} GB/s read, {:.1f} GB/s write",
                                           rdCas * 64.0 / dt / 1e9, wrCas * 64.0 / dt / 1e9));
                    }
                }
                rdCas = wrCas = tsc = 0;
                seen = 0;
            } else if (state_ != State::Running && !line.empty()) {
                lastLines = line;   // error text if emon refuses to start
            }
        }
    }
    CloseHandle(rd);
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    process_ = nullptr;
    CloseHandle(pi.hProcess);
    if (!stop_) Fail(std::format("emon exited (code {}): {}", code, lastLines.empty() ? "no output" : lastLines));
}
