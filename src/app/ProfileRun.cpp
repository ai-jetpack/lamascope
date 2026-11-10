// "Run profiling": Inference mode on -> scripts\Profile-Llama.ps1 in its own console -> wait for llama.cpp to exit
// -> load the new profile -> restore Inference mode (if this run turned it on).
#include <windows.h>
#include <format>
#include "app/App.h"
#include "core/Log.h"
#include "core/Util.h"

namespace {

std::wstring ExeDir() {
    wchar_t p[MAX_PATH];
    DWORD n = GetModuleFileNameW(nullptr, p, MAX_PATH);
    std::wstring s(p, n);
    return s.substr(0, s.find_last_of(L'\\'));
}

bool Exists(const std::wstring& p) { return GetFileAttributesW(p.c_str()) != INVALID_FILE_ATTRIBUTES; }

std::wstring FullPath(const std::wstring& p) {
    wchar_t out[MAX_PATH];
    DWORD n = GetFullPathNameW(p.c_str(), MAX_PATH, out, nullptr);
    return n && n < MAX_PATH ? std::wstring(out, n) : p;
}

// build\Release\LlamaScope.exe -> <repo>\scripts; a copied exe -> <exe dir>\scripts.
std::wstring FindScript() {
    for (const wchar_t* rel : {L"\\..\\..\\scripts\\Profile-Llama.ps1", L"\\scripts\\Profile-Llama.ps1", L"\\Profile-Llama.ps1"}) {
        std::wstring p = FullPath(ExeDir() + rel);
        if (Exists(p)) return p;
    }
    return {};
}

// One argument for CreateProcess / CommandLineToArgvW parsing (what powershell -File uses).
std::wstring Quote(const std::wstring& a) {
    if (!a.empty() && a.find_first_of(L" \t\"") == std::wstring::npos) return a;
    std::wstring r = L"\"";
    size_t bs = 0;
    for (wchar_t c : a) {
        if (c == L'\\') { ++bs; continue; }
        if (c == L'"') { r.append(bs * 2 + 1, L'\\'); r += L'"'; }
        else { r.append(bs, L'\\'); r += c; }
        bs = 0;
    }
    r.append(bs * 2, L'\\');
    return r + L"\"";
}

} // namespace

std::wstring App::ProfileScript() { return FindScript(); }

void App::ProfileRunFail(const std::string& why) {
    profileRun.status = why;
    profileRun.bad = true;
    Log::W("Profiling run: " + why);
    if (profileRun.enteredIm && profileRun.restoreIm) {
        inference.RestoreAsync();
        profileRun.stage = ProfileRun::Stage::Restoring;
    } else {
        profileRun.stage = ProfileRun::Stage::Idle;
    }
}

void App::StartProfileRun() {
    auto& r = profileRun;
    if (r.stage != ProfileRun::Stage::Idle) return;
    r = ProfileRun{.restoreIm = r.restoreIm};   // also clears startAt
    if (!model.ok) { r.status = "Select a model first."; r.bad = true; return; }
    if (settings.profBin.empty() || !Exists(ToWide(settings.profBin))) {
        r.status = "Set the folder of the profiler-enabled llama.cpp build (Profiling setup).";
        r.bad = true;
        return;
    }
    if (FindScript().empty()) {
        r.status = "scripts\\Profile-Llama.ps1 not found next to LlamaScope.exe or in ..\\..\\scripts.";
        r.bad = true;
        return;
    }
    if (sys.elevated && !inference.Active() && !inference.Busy()) {
        inference.EnterAsync();
        r.enteredIm = true;
        r.stage = ProfileRun::Stage::EnteringIm;
        r.status = "Entering Inference mode...";
        return;
    }
    if (!sys.elevated) r.note = "Not administrator: ran without Inference mode.";
    else if (inference.Active()) r.note = "Inference mode was already on: left on afterwards.";
    LaunchProfileRun();
}

void App::LaunchProfileRun() {
    auto& r = profileRun;
    // Arguments: the pasted command line without the executable; -m from the selected model if it has none.
    std::vector<std::string> tok = TokenizeArgs(settings.launch.cmdline);
    // Always llama-cli from the profiler build; the executable named in the pasted line is ignored.
    r.tool = "llama-cli";
    if (!tok.empty() && !tok[0].empty() && tok[0][0] != '-') {
        if (tok[0].find("llama-cli") == std::string::npos)
            r.note += (r.note.empty() ? "" : " ") + std::string("Ran llama-cli instead of ") + tok[0] + ".";
        tok.erase(tok.begin());
    }
    if (tok.empty()) {   // no pasted line: the flags of the panel
        const auto& p = settings.launch;
        const auto& kv = KvTypes();
        tok = {"-ngl", std::to_string(p.ngl), "-c", std::to_string(p.nCtx), "-ctk", kv[p.ctk].name, "-ctv", kv[p.ctv].name,
               "-ub", std::to_string(p.nUbatch)};
        if (p.cpuMoe) tok.push_back("--cpu-moe");
        else if (p.nCpuMoe) { tok.push_back("--n-cpu-moe"); tok.push_back(std::to_string(p.nCpuMoe)); }
        for (const auto& [pat, buf] : p.overrides) { tok.push_back("-ot"); tok.push_back(pat + "=" + buf); }
    }
    bool hasModel = false;
    for (const auto& t : tok) hasModel |= t == "-m" || t == "--model" || t.rfind("--model=", 0) == 0;
    if (!hasModel) tok.insert(tok.begin(), {"-m", model.path});

    std::wstring cmd = L"powershell.exe -NoProfile -ExecutionPolicy Bypass -File " + Quote(FindScript());
    for (const auto& t : tok) cmd += L" " + Quote(ToWide(t));
    cmd += L" --profile-tool=" + ToWide(r.tool) + L" --profile-mode=" + ToWide(settings.profMode) +
           L" --profile-view=0 --profile-pause-on-error=1";

    // The child gets its own environment variables and all cores: LlamaScope itself runs on the E-cores, and a child
    // inherits its parent's CPU mask, which would keep llama.cpp off the P-cores.
    SetEnvironmentVariableW(L"LLAMACPP_PROF_BIN", ToWide(settings.profBin).c_str());
    if (!settings.cudaBin.empty()) SetEnvironmentVariableW(L"CUDA_BIN", ToWide(settings.cudaBin).c_str());
    STARTUPINFOW si{sizeof(si)};
    PROCESS_INFORMATION pi{};
    std::wstring title = L"LlamaScope profiling: " + ToWide(r.tool);
    si.lpTitle = title.data();
    std::vector<wchar_t> line(cmd.begin(), cmd.end());
    line.push_back(0);
    FILETIME now;
    GetSystemTimeAsFileTime(&now);
    r.startFt = ((unsigned long long)now.dwHighDateTime << 32) | now.dwLowDateTime;
    if (!CreateProcessW(nullptr, line.data(), nullptr, nullptr, FALSE, CREATE_NEW_CONSOLE | CREATE_SUSPENDED | NORMAL_PRIORITY_CLASS,
                        nullptr, nullptr, &si, &pi)) {
        ProfileRunFail(std::format("Could not start PowerShell (error {}).", GetLastError()));
        return;
    }
    DWORD_PTR procMask = 0, sysMask = 0;
    if (GetProcessAffinityMask(GetCurrentProcess(), &procMask, &sysMask)) SetProcessAffinityMask(pi.hProcess, sysMask);
    ResumeThread(pi.hThread);
    CloseHandle(pi.hThread);
    r.proc = pi.hProcess;
    r.stage = ProfileRun::Stage::Running;
    r.status = r.tool == "llama-cli"
                   ? "llama-cli runs in its own window: type your prompts there, then /exit (a double Ctrl+C loses the GPU part)."
                   : "llama-bench runs in its own window; the profile loads when it ends.";
    Log::I("Profiling run: " + ToUtf8(cmd));
}

void App::UpdateProfileRun() {
    auto& r = profileRun;
    using S = ProfileRun::Stage;
    switch (r.stage) {
    case S::Idle:
        if (r.startAt > 0 && NowSeconds() >= r.startAt && model.ok) { r.startAt = 0; StartProfileRun(); }
        return;
    case S::EnteringIm:
        if (inference.Busy()) return;
        if (!inference.Active()) { r.note = "Inference mode did not turn on (see its panel); ran without it."; r.enteredIm = false; }
        LaunchProfileRun();
        return;
    case S::Running: {
        if (WaitForSingleObject(r.proc, 0) != WAIT_OBJECT_0) return;
        CloseHandle(r.proc);
        r.proc = nullptr;
        // the newest .prof written after the start
        WIN32_FIND_DATAW fd, best{};
        FILETIME start{(DWORD)r.startFt, (DWORD)(r.startFt >> 32)};
        HANDLE h = FindFirstFileW((ProfilesDir() + L"\\*.prof").c_str(), &fd);
        if (h != INVALID_HANDLE_VALUE) {
            do {
                if (CompareFileTime(&fd.ftLastWriteTime, &start) > 0 && CompareFileTime(&fd.ftLastWriteTime, &best.ftLastWriteTime) > 0)
                    best = fd;
            } while (FindNextFileW(h, &fd));
            FindClose(h);
        }
        if (!best.cFileName[0]) {
            r.status = r.tool + " exited without writing a profile (see its window / llamascope.log).";
            r.bad = true;
        } else {
            LoadProfile(ProfilesDir() + L"\\" + best.cFileName);
            r.status = profile.ok ? "Profile loaded: " + ToUtf8(best.cFileName) : "Profile unreadable: " + profile.error;
            r.bad = !profile.ok;
        }
        if (r.enteredIm && r.restoreIm) {
            inference.RestoreAsync();
            r.stage = S::Restoring;
            r.status += "  Restoring Inference mode...";
        } else {
            r.stage = S::Idle;
        }
        return;
    }
    case S::Restoring:
        if (inference.Busy()) return;
        if (auto p = r.status.find("  Restoring"); p != std::string::npos) r.status.erase(p);
        r.stage = S::Idle;
        return;
    }
}

void App::StopWaitingProfileRun() {
    auto& r = profileRun;
    if (r.stage != ProfileRun::Stage::Running) return;
    CloseHandle(r.proc);
    r.proc = nullptr;
    ProfileRunFail("Stopped waiting (the llama.cpp window keeps running; Load latest picks its profile later).");
}
