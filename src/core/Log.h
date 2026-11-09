#pragma once
#include <string>
#include <vector>

namespace Log {
enum Level { Info, Warn, Error };
struct Entry { double t; Level level; std::string msg; };

void Write(Level level, const std::string& msg);
inline void I(const std::string& m) { Write(Info, m); }
inline void W(const std::string& m) { Write(Warn, m); }
inline void E(const std::string& m) { Write(Error, m); }
std::vector<Entry> Snapshot();
}
