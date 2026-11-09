#include <windows.h>
#include <algorithm>
#include <cstring>
#include <regex>
#include "core/Util.h"
#include "probes/MemInventory.h"

namespace {

template <typename T> T Rd(const uint8_t* p) { T v; memcpy(&v, p, sizeof(T)); return v; }

const char* MemTypeName(int t) {
    switch (t) {
    case 0x1A: return "DDR4";
    case 0x1E: return "LPDDR4";
    case 0x22: return "DDR5";
    case 0x23: return "LPDDR5";
    default:   return "RAM";
    }
}

// Many kits encode their rated (XMP/EXPO) speed in the part number,
// e.g. CMK96GX5M2B5600C40 or F5-6000J3040G32G. Firmware often reports only JEDEC speed.
int SpeedFromPartNumber(const std::string& part) {
    static const std::regex re("(\\d{4})");
    int best = 0;
    for (auto it = std::sregex_iterator(part.begin(), part.end(), re); it != std::sregex_iterator(); ++it) {
        int v = std::stoi((*it)[1].str());
        if (v >= 3200 && v <= 9600 && v % 100 == 0) best = std::max(best, v);
    }
    return best;
}

} // namespace

MemInventory MemInventory::Probe() {
    MemInventory m;
    MEMORYSTATUSEX ms{sizeof(ms)};
    GlobalMemoryStatusEx(&ms);
    m.totalBytes = ms.ullTotalPhys;

    UINT size = GetSystemFirmwareTable('RSMB', 0, nullptr, 0);
    if (size < 8) return m;
    std::vector<uint8_t> b(size);
    GetSystemFirmwareTable('RSMB', 0, b.data(), size);
    DWORD tableLen = Rd<DWORD>(b.data() + 4);
    const uint8_t* d = b.data() + 8;
    const uint8_t* end = std::min<const uint8_t*>(d + tableLen, b.data() + size);

    while (d + 4 <= end) {
        uint8_t type = d[0], hl = d[1];
        if (hl < 4 || d + hl > end) break;
        // String set: null-terminated strings followed by an extra null.
        std::vector<std::string> strs;
        const char* sp = (const char*)(d + hl);
        while (sp < (const char*)end && *sp) {
            strs.emplace_back(sp);
            sp += strlen(sp) + 1;
        }
        const uint8_t* next = strs.empty() ? d + hl + 2 : (const uint8_t*)sp + 1;
        auto str = [&](uint8_t idx) { return (idx && idx <= strs.size()) ? Trim(strs[idx - 1]) : std::string(); };

        if (type == 17 && hl >= 0x15) {
            WORD sz = Rd<WORD>(d + 0x0C);
            uint64_t mb = 0;
            if (sz == 0x7FFF && hl >= 0x20) mb = Rd<DWORD>(d + 0x1C) & 0x7FFFFFFF;
            else if (sz != 0 && sz != 0xFFFF) mb = (sz & 0x8000) ? (sz & 0x7FFF) / 1024 : sz;
            if (mb > 0) {
                Dimm dm;
                dm.sizeMB = mb;
                dm.locator = str(d[0x10]);
                dm.bank = str(d[0x11]);
                dm.type = d[0x12];
                if (hl >= 0x17) {
                    WORD s = Rd<WORD>(d + 0x15);
                    dm.smbiosSpeed = (s == 0xFFFF && hl >= 0x58) ? (int)Rd<DWORD>(d + 0x54) : s;
                }
                if (hl >= 0x1B) {
                    dm.manufacturer = str(d[0x17]);
                    dm.part = str(d[0x1A]);
                    dm.partNumberSpeed = SpeedFromPartNumber(dm.part);
                }
                if (hl >= 0x22) {
                    WORD cs = Rd<WORD>(d + 0x20);
                    dm.configuredMTs = (cs == 0xFFFF && hl >= 0x5C) ? (int)Rd<DWORD>(d + 0x58) : cs;
                }
                m.dimms.push_back(dm);
            }
        }
        if (type == 127) break;
        d = next;
    }

    if (!m.dimms.empty()) {
        m.typeName = MemTypeName(m.dimms[0].type);
        m.configuredMTs = m.dimms[0].configuredMTs;
        for (auto& dm : m.dimms) {
            m.configuredMTs = std::min(m.configuredMTs, dm.configuredMTs);
            m.ratedMTs = std::max({m.ratedMTs, dm.smbiosSpeed, dm.partNumberSpeed});
        }
        m.channels = std::min<int>((int)m.dimms.size(), 2);
        m.theoreticalGBs = m.channels * (double)m.configuredMTs * 8.0 / 1000.0;
    }
    return m;
}
