#pragma once
#include <string>
#include "imgui.h"

// Spec-vs-actual verdict shown as a colored dot next to every metric.
// None = plain information row, drawn without a dot.
enum class Status { None, Ok, Info, Warn, Bad, Unavailable };

inline ImVec4 StatusColor(Status s) {
    switch (s) {
    case Status::Ok:   return ImVec4(0.247f, 0.725f, 0.314f, 1.0f); // #3fb950
    case Status::Info: return ImVec4(0.345f, 0.651f, 1.000f, 1.0f); // #58a6ff
    case Status::Warn: return ImVec4(0.824f, 0.600f, 0.133f, 1.0f); // #d29922
    case Status::Bad:  return ImVec4(0.973f, 0.318f, 0.286f, 1.0f); // #f85149
    default:           return ImVec4(0.431f, 0.463f, 0.506f, 1.0f); // #6e7681
    }
}

struct Verdict {
    Status status = Status::Unavailable;
    std::string reason;
};
