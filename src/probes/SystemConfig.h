#pragma once
#include <string>

struct SystemConfig {
    std::string powerPlan;
    bool powerPlanPerformance = false;   // High performance / Ultimate performance
    int aspmAC = -1;                     // 0 off, 1 moderate, 2 maximum, -1 unknown
    int cpMinCoresAC = -1;               // core parking: minimum unparked cores %, 100 = parking off, -1 unknown
    bool lockPagesKnown = false;
    bool lockPagesAssigned = false;      // account right granted (LSA)
    bool lockPagesInToken = false;       // present in this process token
    int hags = -1;                       // HwSchMode: 2 on, 1 off, -1 unknown
    std::string windowsVersion;
    bool elevated = false;

    static SystemConfig Probe();
};
