#include <windows.h>
#include <shellapi.h>
#include "platform/Elevation.h"

bool IsProcessElevated() {
    HANDLE token = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token)) return false;
    TOKEN_ELEVATION e{};
    DWORD size = 0;
    bool elevated = GetTokenInformation(token, TokenElevation, &e, sizeof(e), &size) && e.TokenIsElevated;
    CloseHandle(token);
    return elevated;
}

bool RestartAsAdmin() {
    wchar_t path[MAX_PATH];
    GetModuleFileNameW(nullptr, path, MAX_PATH);
    SHELLEXECUTEINFOW si{sizeof(si)};
    si.lpVerb = L"runas";
    si.lpFile = path;
    si.nShow = SW_SHOWNORMAL;
    return ShellExecuteExW(&si) != FALSE;
}
