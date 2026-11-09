#include <windows.h>
#include <ntsecapi.h>
#include <powrprof.h>
#include <powersetting.h>
#include <format>
#include <vector>
#include "core/Util.h"
#include "platform/Elevation.h"
#include "probes/SystemConfig.h"

namespace {

const GUID kSubPciExpress = {0x501a4d13, 0x42af, 0x4429, {0x9f, 0xd1, 0xa8, 0x21, 0x8c, 0x26, 0x8e, 0x20}};
const GUID kAspmPolicy = {0xee12f906, 0xd277, 0x404b, {0xb6, 0xda, 0xe5, 0xfa, 0x1a, 0x57, 0x6d, 0xf5}};
const GUID kHighPerf = {0x8c5e7fda, 0xe8bf, 0x4a96, {0x9a, 0x85, 0xa6, 0xe2, 0x3a, 0x8c, 0x63, 0x5c}};
const GUID kUltimatePerf = {0xe9a42b02, 0xd5df, 0x448d, {0xaa, 0x00, 0x03, 0xf1, 0x47, 0x49, 0xeb, 0x61}};

void ProbePower(SystemConfig& s) {
    GUID* active = nullptr;
    if (PowerGetActiveScheme(nullptr, &active) != ERROR_SUCCESS) return;
    wchar_t name[256] = {};
    DWORD size = sizeof(name);
    if (PowerReadFriendlyName(nullptr, active, nullptr, nullptr, (UCHAR*)name, &size) == ERROR_SUCCESS)
        s.powerPlan = ToUtf8(name);
    std::wstring lower = name;
    for (auto& c : lower) c = towlower(c);
    s.powerPlanPerformance = IsEqualGUID(*active, kHighPerf) || IsEqualGUID(*active, kUltimatePerf) ||
                             lower.find(L"performance") != std::wstring::npos;
    DWORD v = 0;
    if (PowerReadACValueIndex(nullptr, active, &kSubPciExpress, &kAspmPolicy, &v) == ERROR_SUCCESS)
        s.aspmAC = (int)v;
    LocalFree(active);
}

void ProbeLockPages(SystemConfig& s) {
    HANDLE token = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token)) return;

    LUID lockLuid{};
    LookupPrivilegeValueW(nullptr, SE_LOCK_MEMORY_NAME, &lockLuid);
    DWORD size = 0;
    GetTokenInformation(token, TokenPrivileges, nullptr, 0, &size);
    std::vector<uint8_t> privBuf(size);
    if (GetTokenInformation(token, TokenPrivileges, privBuf.data(), size, &size)) {
        auto* tp = (TOKEN_PRIVILEGES*)privBuf.data();
        for (DWORD i = 0; i < tp->PrivilegeCount; ++i)
            if (tp->Privileges[i].Luid.LowPart == lockLuid.LowPart && tp->Privileges[i].Luid.HighPart == lockLuid.HighPart)
                s.lockPagesInToken = true;
    }

    GetTokenInformation(token, TokenUser, nullptr, 0, &size);
    std::vector<uint8_t> userBuf(size);
    if (GetTokenInformation(token, TokenUser, userBuf.data(), size, &size)) {
        PSID sid = ((TOKEN_USER*)userBuf.data())->User.Sid;
        LSA_OBJECT_ATTRIBUTES oa{};
        LSA_HANDLE policy = nullptr;
        if (LsaOpenPolicy(nullptr, &oa, POLICY_LOOKUP_NAMES, &policy) == 0) {
            PLSA_UNICODE_STRING rights = nullptr;
            ULONG count = 0;
            NTSTATUS st = LsaEnumerateAccountRights(policy, sid, &rights, &count);
            if (st == 0) {
                s.lockPagesKnown = true;
                for (ULONG i = 0; i < count; ++i) {
                    std::wstring r(rights[i].Buffer, rights[i].Length / sizeof(wchar_t));
                    if (r == SE_LOCK_MEMORY_NAME) s.lockPagesAssigned = true;
                }
                LsaFreeMemory(rights);
            } else if ((ULONG)st == 0xC0000034) { // STATUS_OBJECT_NAME_NOT_FOUND: no rights assigned
                s.lockPagesKnown = true;
            }
            LsaClose(policy);
        }
    }
    CloseHandle(token);
    if (s.lockPagesInToken) { s.lockPagesKnown = true; s.lockPagesAssigned = true; }
}

void ProbeHags(SystemConfig& s) {
    DWORD v = 0, size = sizeof(v);
    if (RegGetValueW(HKEY_LOCAL_MACHINE, L"SYSTEM\\CurrentControlSet\\Control\\GraphicsDrivers", L"HwSchMode",
                     RRF_RT_REG_DWORD, nullptr, &v, &size) == ERROR_SUCCESS)
        s.hags = (int)v;
}

void ProbeWindows(SystemConfig& s) {
    using RtlGetVersion_t = LONG(WINAPI*)(OSVERSIONINFOW*);
    auto fn = (RtlGetVersion_t)GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "RtlGetVersion");
    OSVERSIONINFOW v{sizeof(v)};
    if (fn && fn(&v) == 0) {
        DWORD ubr = 0, size = sizeof(ubr);
        RegGetValueW(HKEY_LOCAL_MACHINE, L"SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion", L"UBR", RRF_RT_REG_DWORD,
                     nullptr, &ubr, &size);
        s.windowsVersion = std::format("{} build {}.{}", v.dwBuildNumber >= 22000 ? "Windows 11" : "Windows 10",
                                       v.dwBuildNumber, ubr);
    }
}

} // namespace

SystemConfig SystemConfig::Probe() {
    SystemConfig s;
    ProbePower(s);
    ProbeLockPages(s);
    ProbeHags(s);
    ProbeWindows(s);
    s.elevated = IsProcessElevated();
    return s;
}
