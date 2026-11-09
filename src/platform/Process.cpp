#include <windows.h>
#include <vector>
#include "platform/Process.h"

int RunHidden(const std::wstring& cmdline, std::string* output, unsigned timeoutMs) {
    SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
    HANDLE rd = nullptr, wr = nullptr;
    if (!CreatePipe(&rd, &wr, &sa, 0)) return -1;
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
    STARTUPINFOW si{sizeof(si)};
    si.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;
    si.hStdOutput = wr;
    si.hStdError = wr;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    PROCESS_INFORMATION pi{};
    std::vector<wchar_t> cmd(cmdline.begin(), cmdline.end());
    cmd.push_back(0);
    BOOL ok = CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
    CloseHandle(wr);
    if (!ok) {
        CloseHandle(rd);
        return -1;
    }
    std::string out;
    char buf[4096];
    DWORD n = 0;
    ULONGLONG deadline = GetTickCount64() + timeoutMs;
    for (;;) {
        DWORD avail = 0;
        if (PeekNamedPipe(rd, nullptr, 0, nullptr, &avail, nullptr) && avail) {
            if (ReadFile(rd, buf, sizeof(buf), &n, nullptr) && n) out.append(buf, n);
            continue;
        }
        if (WaitForSingleObject(pi.hProcess, 50) == WAIT_OBJECT_0) {
            while (PeekNamedPipe(rd, nullptr, 0, nullptr, &avail, nullptr) && avail && ReadFile(rd, buf, sizeof(buf), &n, nullptr) && n)
                out.append(buf, n);
            break;
        }
        if (GetTickCount64() > deadline) {
            TerminateProcess(pi.hProcess, 1);
            break;
        }
    }
    DWORD code = (DWORD)-1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    CloseHandle(rd);
    if (output) *output = out;
    return (int)code;
}

std::wstring ProcessCommandLine(unsigned pid) {
    // NtQueryInformationProcess(ProcessCommandLineInformation = 60), Windows 8.1+: returns a UNICODE_STRING.
    using NtQip = LONG(WINAPI*)(HANDLE, int, PVOID, ULONG, PULONG);
    static auto fn = (NtQip)GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtQueryInformationProcess");
    if (!fn) return {};
    HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
    if (!h) return {};
    std::vector<unsigned char> buf(8192);
    ULONG len = 0;
    LONG st = fn(h, 60, buf.data(), (ULONG)buf.size(), &len);
    if (st != 0 && len > buf.size()) {
        buf.resize(len);
        st = fn(h, 60, buf.data(), (ULONG)buf.size(), &len);
    }
    CloseHandle(h);
    if (st != 0) return {};
    struct US { USHORT Length, MaximumLength; PWSTR Buffer; };
    auto* us = reinterpret_cast<US*>(buf.data());
    return std::wstring(us->Buffer, us->Length / sizeof(wchar_t));
}

int CommandLineInt(const std::wstring& cmd, std::initializer_list<const wchar_t*> flags) {
    for (const wchar_t* f : flags) {
        std::wstring flag(f);
        for (size_t pos = cmd.find(flag); pos != std::wstring::npos; pos = cmd.find(flag, pos + 1)) {
            size_t end = pos + flag.size();
            bool startOk = pos == 0 || cmd[pos - 1] == L' ' || cmd[pos - 1] == L'"';
            if (!startOk || end >= cmd.size() || (cmd[end] != L' ' && cmd[end] != L'=')) continue;
            size_t v = cmd.find_first_not_of(L" =", end);
            if (v == std::wstring::npos || !iswdigit(cmd[v])) continue;
            return _wtoi(cmd.c_str() + v);
        }
    }
    return -1;
}
