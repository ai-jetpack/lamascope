#include <windows.h>
#include <dxgi1_2.h>
#include <format>
#include <regex>
#include "core/Util.h"
#include "probes/Displays.h"

std::wstring AdapterInfo::LuidTag() const {
    return std::format(L"luid_0x{:08X}_0x{:08X}", luidHigh, luidLow);
}

namespace {

// Intel desktop SKUs ending in F (e.g. i9-12900KF, i5-12400F) ship without an iGPU.
int CpuHasIgpu(const std::string& brand) {
    if (brand.find("Intel") == std::string::npos) return -1;
    std::smatch m;
    static const std::regex re("i[3579]-\\d{4,5}([A-Z]*)");
    if (!std::regex_search(brand, m, re)) return -1;
    return m[1].str().find('F') != std::string::npos ? 0 : 1;
}

MonitorInfo DescribeOutput(const DXGI_OUTPUT_DESC& od) {
    MonitorInfo mi;
    mi.device = ToUtf8(od.DeviceName);
    mi.width = od.DesktopCoordinates.right - od.DesktopCoordinates.left;
    mi.height = od.DesktopCoordinates.bottom - od.DesktopCoordinates.top;
    DISPLAY_DEVICEW dd{sizeof(dd)};
    if (EnumDisplayDevicesW(od.DeviceName, 0, &dd, 0)) mi.name = ToUtf8(dd.DeviceString);
    DEVMODEW dm{};
    dm.dmSize = sizeof(dm);
    if (EnumDisplaySettingsW(od.DeviceName, ENUM_CURRENT_SETTINGS, &dm)) {
        mi.hz = (int)dm.dmDisplayFrequency;
        mi.width = (int)dm.dmPelsWidth;
        mi.height = (int)dm.dmPelsHeight;
    }
    MONITORINFO info{sizeof(info)};
    if (GetMonitorInfoW(od.Monitor, &info)) mi.primary = (info.dwFlags & MONITORINFOF_PRIMARY) != 0;
    return mi;
}

} // namespace

DisplayTopology DisplayTopology::Probe(const std::string& cpuBrand) {
    DisplayTopology t;
    t.cpuIgpu = CpuHasIgpu(cpuBrand);

    IDXGIFactory1* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return t;
    IDXGIAdapter1* adapter = nullptr;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 d{};
        adapter->GetDesc1(&d);
        if (!(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
            AdapterInfo a;
            a.name = ToUtf8(d.Description);
            a.vendor = d.VendorId;
            a.deviceId = d.DeviceId;
            a.dedicatedVram = d.DedicatedVideoMemory;
            a.luidHigh = (uint32_t)d.AdapterLuid.HighPart;
            a.luidLow = d.AdapterLuid.LowPart;
            IDXGIOutput* out = nullptr;
            for (UINT o = 0; adapter->EnumOutputs(o, &out) != DXGI_ERROR_NOT_FOUND; ++o) {
                DXGI_OUTPUT_DESC od{};
                if (SUCCEEDED(out->GetDesc(&od)) && od.AttachedToDesktop) a.monitors.push_back(DescribeOutput(od));
                out->Release();
            }
            if (a.IsNvidia() && t.nvidiaIndex < 0) t.nvidiaIndex = (int)t.adapters.size();
            if (a.IsIntel() && t.igpuIndex < 0) t.igpuIndex = (int)t.adapters.size();
            t.adapters.push_back(std::move(a));
        }
        adapter->Release();
    }
    factory->Release();
    return t;
}
