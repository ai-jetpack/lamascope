#pragma once
#include <cstdint>
#include <string>

std::string ToUtf8(const std::wstring& w);
std::wstring ToWide(const std::string& s);
std::string Trim(const std::string& s);
std::string FormatBytes(uint64_t bytes);       // "96.0 GB"
std::wstring LocalDataDir();                    // %LOCALAPPDATA%\LlamaScope (created)
