#include <windows.h>
#include <algorithm>
#include <format>
#include "core/Log.h"
#include "probes/PcieTest.h"

namespace {

using CUresult = int;
using CUdevice = int;
using CUcontext = struct CUctx_st*;
using CUdeviceptr = unsigned long long;

struct Cuda {
    HMODULE dll = nullptr;
    CUresult(__stdcall* Init)(unsigned) = nullptr;
    CUresult(__stdcall* DeviceGet)(CUdevice*, int) = nullptr;
    CUresult(__stdcall* PrimaryCtxRetain)(CUcontext*, CUdevice) = nullptr;
    CUresult(__stdcall* PrimaryCtxRelease)(CUdevice) = nullptr;
    CUresult(__stdcall* CtxSetCurrent)(CUcontext) = nullptr;
    CUresult(__stdcall* MemAllocHost)(void**, size_t) = nullptr;
    CUresult(__stdcall* MemFreeHost)(void*) = nullptr;
    CUresult(__stdcall* MemAlloc)(CUdeviceptr*, size_t) = nullptr;
    CUresult(__stdcall* MemFree)(CUdeviceptr) = nullptr;
    CUresult(__stdcall* MemcpyHtoD)(CUdeviceptr, const void*, size_t) = nullptr;
    CUresult(__stdcall* MemcpyDtoH)(void*, CUdeviceptr, size_t) = nullptr;
    CUresult(__stdcall* CtxSynchronize)() = nullptr;

    template <typename F> bool Get(F& f, const char* name, const char* alt = nullptr) {
        f = (F)GetProcAddress(dll, name);
        if (!f && alt) f = (F)GetProcAddress(dll, alt);
        return f != nullptr;
    }

    bool Load(std::string& err) {
        dll = LoadLibraryW(L"nvcuda.dll");
        if (!dll) { err = "nvcuda.dll not found (NVIDIA driver missing?)"; return false; }
        bool ok = Get(Init, "cuInit") && Get(DeviceGet, "cuDeviceGet") &&
                  Get(PrimaryCtxRetain, "cuDevicePrimaryCtxRetain") &&
                  Get(PrimaryCtxRelease, "cuDevicePrimaryCtxRelease_v2", "cuDevicePrimaryCtxRelease") &&
                  Get(CtxSetCurrent, "cuCtxSetCurrent") && Get(MemAllocHost, "cuMemAllocHost_v2", "cuMemAllocHost") &&
                  Get(MemFreeHost, "cuMemFreeHost") && Get(MemAlloc, "cuMemAlloc_v2", "cuMemAlloc") &&
                  Get(MemFree, "cuMemFree_v2", "cuMemFree") && Get(MemcpyHtoD, "cuMemcpyHtoD_v2", "cuMemcpyHtoD") &&
                  Get(MemcpyDtoH, "cuMemcpyDtoH_v2", "cuMemcpyDtoH") && Get(CtxSynchronize, "cuCtxSynchronize");
        if (!ok) err = "nvcuda.dll is missing required entry points";
        return ok;
    }
    ~Cuda() { if (dll) FreeLibrary(dll); }
};

double QpcSeconds() {
    static LARGE_INTEGER f = [] { LARGE_INTEGER x; QueryPerformanceFrequency(&x); return x; }();
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}

} // namespace

PcieTest::~PcieTest() {
    if (worker_.joinable()) worker_.join();
}

void PcieTest::Start() {
    if (running_) return;
    if (worker_.joinable()) worker_.join();
    running_ = true;
    progress_ = 0;
    worker_ = std::thread([this] { Run(); });
}

void PcieTest::Run() {
    PcieResult res;
    Cuda cu;
    auto finish = [&] {
        if (!res.error.empty()) Log::E("PCIe test: " + res.error);
        std::lock_guard lk(mu_);
        result_ = res;
        running_ = false;
    };
    if (!cu.Load(res.error)) return finish();

    CUdevice dev = 0;
    CUcontext ctx = nullptr;
    if (cu.Init(0) || cu.DeviceGet(&dev, 0) || cu.PrimaryCtxRetain(&ctx, dev) || cu.CtxSetCurrent(ctx)) {
        res.error = "CUDA initialization failed";
        return finish();
    }

    const size_t bytes = size_t(256) << 20;
    void* host = nullptr;
    CUdeviceptr devBuf = 0;
    if (cu.MemAllocHost(&host, bytes) || cu.MemAlloc(&devBuf, bytes)) {
        res.error = "Could not allocate 256 MB pinned host + device buffers";
    } else {
        memset(host, 0x5A, bytes);
        const int iters = 10;
        double bestH2D = 0, bestD2H = 0;
        // Warm-up lets the link leave its idle power state.
        cu.MemcpyHtoD(devBuf, host, bytes);
        cu.MemcpyDtoH(host, devBuf, bytes);
        for (int i = 0; i < iters; ++i) {
            double t0 = QpcSeconds();
            cu.MemcpyHtoD(devBuf, host, bytes);
            cu.CtxSynchronize();
            double t1 = QpcSeconds();
            cu.MemcpyDtoH(host, devBuf, bytes);
            cu.CtxSynchronize();
            double t2 = QpcSeconds();
            bestH2D = std::max(bestH2D, bytes / (t1 - t0) / 1e9);
            bestD2H = std::max(bestD2H, bytes / (t2 - t1) / 1e9);
            progress_ = float(i + 1) / iters;
        }
        res.valid = true;
        res.h2dGBs = bestH2D;
        res.d2hGBs = bestD2H;
        Log::I(std::format("PCIe test: host->device {:.1f} GB/s, device->host {:.1f} GB/s", bestH2D, bestD2H));
    }
    if (devBuf) cu.MemFree(devBuf);
    if (host) cu.MemFreeHost(host);
    cu.PrimaryCtxRelease(dev);
    finish();
}
