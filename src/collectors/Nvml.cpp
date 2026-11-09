#include <windows.h>
#include <algorithm>
#include <chrono>
#include <format>
#include "collectors/Nvml.h"
#include "core/Log.h"

namespace {

using nvmlReturn_t = int;
using nvmlDevice_t = struct nvmlDevice_st*;
struct nvmlMemory_t { unsigned long long total, free, used; };
struct nvmlUtilization_t { unsigned gpu, memory; };
struct nvmlBAR1Memory_t { unsigned long long total, free, used; };
enum { NVML_CLOCK_GRAPHICS = 0, NVML_CLOCK_MEM = 2 };
enum { NVML_TEMPERATURE_GPU = 0 };
enum { NVML_PCIE_UTIL_TX_BYTES = 0, NVML_PCIE_UTIL_RX_BYTES = 1 };

struct Nvml {
    HMODULE dll = nullptr;
    nvmlReturn_t (*Init)() = nullptr;
    nvmlReturn_t (*Shutdown)() = nullptr;
    nvmlReturn_t (*GetHandleByIndex)(unsigned, nvmlDevice_t*) = nullptr;
    nvmlReturn_t (*GetName)(nvmlDevice_t, char*, unsigned) = nullptr;
    nvmlReturn_t (*SysDriverVersion)(char*, unsigned) = nullptr;
    nvmlReturn_t (*SysCudaVersion)(int*) = nullptr;
    nvmlReturn_t (*GetMemoryInfo)(nvmlDevice_t, nvmlMemory_t*) = nullptr;
    nvmlReturn_t (*GetBar1)(nvmlDevice_t, nvmlBAR1Memory_t*) = nullptr;
    nvmlReturn_t (*GetUtil)(nvmlDevice_t, nvmlUtilization_t*) = nullptr;
    nvmlReturn_t (*GetClock)(nvmlDevice_t, int, unsigned*) = nullptr;
    nvmlReturn_t (*GetMaxClock)(nvmlDevice_t, int, unsigned*) = nullptr;
    nvmlReturn_t (*GetPower)(nvmlDevice_t, unsigned*) = nullptr;
    nvmlReturn_t (*GetPowerLimit)(nvmlDevice_t, unsigned*) = nullptr;
    nvmlReturn_t (*GetTemp)(nvmlDevice_t, int, unsigned*) = nullptr;
    nvmlReturn_t (*GetThrottle)(nvmlDevice_t, unsigned long long*) = nullptr;
    nvmlReturn_t (*CurGen)(nvmlDevice_t, unsigned*) = nullptr;
    nvmlReturn_t (*CurWidth)(nvmlDevice_t, unsigned*) = nullptr;
    nvmlReturn_t (*MaxGen)(nvmlDevice_t, unsigned*) = nullptr;
    nvmlReturn_t (*MaxWidth)(nvmlDevice_t, unsigned*) = nullptr;
    nvmlReturn_t (*GpuMaxGen)(nvmlDevice_t, unsigned*) = nullptr;
    nvmlReturn_t (*Throughput)(nvmlDevice_t, int, unsigned*) = nullptr;
    nvmlReturn_t (*BusWidth)(nvmlDevice_t, unsigned*) = nullptr;

    template <typename F> void Get(F& f, const char* a, const char* b = nullptr) {
        f = (F)GetProcAddress(dll, a);
        if (!f && b) f = (F)GetProcAddress(dll, b);
    }

    bool Load() {
        dll = LoadLibraryW(L"nvml.dll");
        if (!dll) dll = LoadLibraryW(L"C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll");
        if (!dll) return false;
        Get(Init, "nvmlInit_v2", "nvmlInit");
        Get(Shutdown, "nvmlShutdown");
        Get(GetHandleByIndex, "nvmlDeviceGetHandleByIndex_v2", "nvmlDeviceGetHandleByIndex");
        Get(GetName, "nvmlDeviceGetName");
        Get(SysDriverVersion, "nvmlSystemGetDriverVersion");
        Get(SysCudaVersion, "nvmlSystemGetCudaDriverVersion_v2", "nvmlSystemGetCudaDriverVersion");
        Get(GetMemoryInfo, "nvmlDeviceGetMemoryInfo");
        Get(GetBar1, "nvmlDeviceGetBAR1MemoryInfo");
        Get(GetUtil, "nvmlDeviceGetUtilizationRates");
        Get(GetClock, "nvmlDeviceGetClockInfo");
        Get(GetMaxClock, "nvmlDeviceGetMaxClockInfo");
        Get(GetPower, "nvmlDeviceGetPowerUsage");
        Get(GetPowerLimit, "nvmlDeviceGetEnforcedPowerLimit");
        Get(GetTemp, "nvmlDeviceGetTemperature");
        Get(GetThrottle, "nvmlDeviceGetCurrentClocksEventReasons", "nvmlDeviceGetCurrentClocksThrottleReasons");
        Get(CurGen, "nvmlDeviceGetCurrPcieLinkGeneration");
        Get(CurWidth, "nvmlDeviceGetCurrPcieLinkWidth");
        Get(MaxGen, "nvmlDeviceGetMaxPcieLinkGeneration");
        Get(MaxWidth, "nvmlDeviceGetMaxPcieLinkWidth");
        Get(GpuMaxGen, "nvmlDeviceGetGpuMaxPcieLinkGeneration");
        Get(Throughput, "nvmlDeviceGetPcieThroughput");
        Get(BusWidth, "nvmlDeviceGetMemoryBusWidth");
        return Init && GetHandleByIndex;
    }
};

Nvml g_nvml;
nvmlDevice_t g_dev = nullptr;

// Cards whose connector is x16 but which only wire 8 lanes. NVML reports the slot's 16.
int CardWidthOverride(const std::string& name, std::string& note) {
    static const char* x8[] = {"RTX 5060", "RTX 5050", "RTX 4060", "RTX 3050", "RTX A2000"};
    for (auto* n : x8)
        if (name.find(n) != std::string::npos) {
            note = "card is electrically x8 (x16 connector)";
            return 8;
        }
    return 0;
}

} // namespace

std::string ThrottleReasonsText(uint64_t bits, bool* concerning) {
    struct R { uint64_t bit; const char* name; bool bad; };
    static const R reasons[] = {
        {0x1, "GPU idle", false},
        {0x2, "application clocks", false},
        {0x4, "SW power cap", true},
        {0x8, "HW slowdown", true},
        {0x10, "sync boost", false},
        {0x20, "SW thermal", true},
        {0x40, "HW thermal", true},
        {0x80, "HW power brake", true},
        {0x100, "display clock", false},
    };
    std::string s;
    bool bad = false;
    uint64_t known = 0;
    for (auto& r : reasons) {
        known |= r.bit;
        if (bits & r.bit) {
            if (!s.empty()) s += ", ";
            s += r.name;
            bad |= r.bad;
        }
    }
    if (bits & ~known) {
        if (!s.empty()) s += ", ";
        s += std::format("unknown 0x{:X}", bits & ~known);
    }
    if (concerning) *concerning = bad;
    return s.empty() ? "none" : s;
}

bool NvmlCollector::Start() {
    GpuStatic st;
    if (!g_nvml.Load()) {
        st.error = "nvml.dll not found (NVIDIA driver missing?)";
    } else if (g_nvml.Init() != 0 || g_nvml.GetHandleByIndex(0, &g_dev) != 0) {
        st.error = "NVML initialization failed or no NVIDIA GPU";
    } else {
        st.ok = true;
        char buf[128] = {};
        if (g_nvml.GetName && g_nvml.GetName(g_dev, buf, sizeof(buf)) == 0) st.name = buf;
        if (g_nvml.SysDriverVersion && g_nvml.SysDriverVersion(buf, sizeof(buf)) == 0) st.driver = buf;
        if (g_nvml.SysCudaVersion) g_nvml.SysCudaVersion(&st.cudaDriverVersion);
        nvmlMemory_t mem{};
        if (g_nvml.GetMemoryInfo && g_nvml.GetMemoryInfo(g_dev, &mem) == 0) st.vramTotal = mem.total;
        nvmlBAR1Memory_t bar{};
        if (g_nvml.GetBar1 && g_nvml.GetBar1(g_dev, &bar) == 0) st.bar1Total = bar.total;
        if (g_nvml.GetMaxClock) {
            g_nvml.GetMaxClock(g_dev, NVML_CLOCK_GRAPHICS, &st.maxGraphicsMHz);
            g_nvml.GetMaxClock(g_dev, NVML_CLOCK_MEM, &st.maxMemMHz);
        }
        if (g_nvml.GetPowerLimit) g_nvml.GetPowerLimit(g_dev, &st.powerLimitmW);
        if (g_nvml.BusWidth) g_nvml.BusWidth(g_dev, &st.memBusWidth);
        unsigned v = 0;
        if (g_nvml.MaxGen && g_nvml.MaxGen(g_dev, &v) == 0) st.linkMaxGen = (int)v;
        if (g_nvml.MaxWidth && g_nvml.MaxWidth(g_dev, &v) == 0) st.linkMaxWidth = (int)v;
        if (g_nvml.GpuMaxGen && g_nvml.GpuMaxGen(g_dev, &v) == 0) st.gpuMaxGen = (int)v;
        st.cardWidth = CardWidthOverride(st.name, st.cardWidthNote);
        if (!st.cardWidth) st.cardWidth = st.linkMaxWidth;
        Log::I(std::format("NVML: {} driver {} CUDA {}.{}", st.name, st.driver, st.cudaDriverVersion / 1000,
                           (st.cudaDriverVersion % 1000) / 10));
    }
    if (!st.error.empty()) Log::W(st.error);
    {
        std::lock_guard lk(mu_);
        static_ = st;
    }
    if (!st.ok) return false;
    stop_ = false;
    thread_ = std::thread([this] { Loop(); });
    return true;
}

void NvmlCollector::Stop() {
    stop_ = true;
    if (thread_.joinable()) thread_.join();
    if (g_nvml.Shutdown && g_dev) {
        g_nvml.Shutdown();
        g_dev = nullptr;
    }
}

void NvmlCollector::Loop() {
    auto next = std::chrono::steady_clock::now();
    while (!stop_) {
        next += std::chrono::milliseconds(100);
        GpuLive l;
        l.valid = true;
        unsigned v = 0;
        if (g_nvml.CurGen && g_nvml.CurGen(g_dev, &v) == 0) l.linkGen = (int)v;
        if (g_nvml.CurWidth && g_nvml.CurWidth(g_dev, &v) == 0) l.linkWidth = (int)v;
        nvmlUtilization_t u{};
        if (g_nvml.GetUtil && g_nvml.GetUtil(g_dev, &u) == 0) { l.util = u.gpu; l.memUtil = u.memory; }
        if (g_nvml.GetClock) {
            g_nvml.GetClock(g_dev, NVML_CLOCK_GRAPHICS, &l.gfxMHz);
            g_nvml.GetClock(g_dev, NVML_CLOCK_MEM, &l.memMHz);
        }
        if (g_nvml.GetPower) g_nvml.GetPower(g_dev, &l.powermW);
        if (g_nvml.GetTemp) g_nvml.GetTemp(g_dev, NVML_TEMPERATURE_GPU, &l.tempC);
        unsigned long long thr = 0;
        if (g_nvml.GetThrottle && g_nvml.GetThrottle(g_dev, &thr) == 0) l.throttle = thr;
        nvmlMemory_t mem{};
        if (g_nvml.GetMemoryInfo && g_nvml.GetMemoryInfo(g_dev, &mem) == 0) l.vramUsed = mem.used;
        // Each throughput call samples over ~20 ms inside the driver.
        if (g_nvml.Throughput) {
            g_nvml.Throughput(g_dev, NVML_PCIE_UTIL_RX_BYTES, &l.rxKBs);
            g_nvml.Throughput(g_dev, NVML_PCIE_UTIL_TX_BYTES, &l.txKBs);
        }

        double t = NowSeconds();
        util.Push(t, (float)l.util);
        memUtil.Push(t, (float)l.memUtil);
        gfxMHz.Push(t, (float)l.gfxMHz);
        memMHz.Push(t, (float)l.memMHz);
        powerW.Push(t, l.powermW / 1000.0f);
        tempC.Push(t, (float)l.tempC);
        rxMBs.Push(t, l.rxKBs / 1024.0f);
        txMBs.Push(t, l.txKBs / 1024.0f);
        vramGB.Push(t, (float)(l.vramUsed / double(1ull << 30)));

        maxGenSeen_ = std::max<int>(maxGenSeen_, l.linkGen);
        maxWidthSeen_ = std::max<int>(maxWidthSeen_, l.linkWidth);
        maxUtilSeen_ = std::max<unsigned>(maxUtilSeen_, l.util);
        {
            std::lock_guard lk(mu_);
            live_ = l;
        }
        std::this_thread::sleep_until(next);
    }
}
