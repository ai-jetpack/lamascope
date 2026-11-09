#pragma once
#include <cstdint>
#include <string>
#include <vector>

struct Dimm {
    std::string locator, bank, manufacturer, part;
    uint64_t sizeMB = 0;
    int smbiosSpeed = 0;       // SMBIOS "Speed" (module max as reported by firmware)
    int partNumberSpeed = 0;   // speed decoded from the part number (XMP/EXPO rating), 0 if unknown
    int configuredMTs = 0;     // SMBIOS "Configured Memory Speed"
    int type = 0;              // SMBIOS memory type
};

struct MemInventory {
    uint64_t totalBytes = 0;
    std::vector<Dimm> dimms;
    std::string typeName;
    int channels = 0;          // assumed: min(populated DIMMs, 2) on client platforms
    int ratedMTs = 0;          // max of SMBIOS speed and part-number speed
    int configuredMTs = 0;     // min over DIMMs
    double theoreticalGBs = 0; // channels * configured MT/s * 8 bytes

    static MemInventory Probe();
};
