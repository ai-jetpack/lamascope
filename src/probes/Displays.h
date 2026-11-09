#pragma once
#include <cstdint>
#include <string>
#include <vector>

struct MonitorInfo {
    std::string device;     // \\.\DISPLAY1
    std::string name;       // monitor model, if the driver reports it
    int width = 0, height = 0, hz = 0;
    bool primary = false;
};

struct AdapterInfo {
    std::string name;
    uint32_t vendor = 0, deviceId = 0;
    uint64_t dedicatedVram = 0;
    uint32_t luidHigh = 0, luidLow = 0;
    std::vector<MonitorInfo> monitors;
    bool IsNvidia() const { return vendor == 0x10DE; }
    bool IsIntel() const { return vendor == 0x8086; }
    // PDH instance tag for this adapter, e.g. "luid_0x00000000_0x0000D1A5"
    std::wstring LuidTag() const;
};

struct DisplayTopology {
    std::vector<AdapterInfo> adapters;   // hardware adapters only (no Basic Render Driver)
    int nvidiaIndex = -1;                // inference GPU
    int igpuIndex = -1;                  // integrated GPU, if enabled and with a driver
    int cpuIgpu = -1;                    // CPU SKU has an iGPU: 1 yes, 0 no (F/KF), -1 unknown

    static DisplayTopology Probe(const std::string& cpuBrand);
};
