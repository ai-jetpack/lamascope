#pragma once
#include <initializer_list>
#include <string>

// Runs a command line without a console window, captures stdout+stderr. Returns the exit code, or -1.
int RunHidden(const std::wstring& cmdline, std::string* output = nullptr, unsigned timeoutMs = 30000);

// Command line of another process ("" if it cannot be read). Same-user processes need no elevation.
std::wstring ProcessCommandLine(unsigned pid);
// Value after a flag in a command line ("--n-cpu-moe 36" or "--n-cpu-moe=36"), -1 if absent.
int CommandLineInt(const std::wstring& cmd, std::initializer_list<const wchar_t*> flags);
