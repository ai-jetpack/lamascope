// LlamaScope entry point: Win32 window + Direct3D 11 + Dear ImGui frame loop.
#include <windows.h>
#include <d3d11.h>
#include <wincodec.h>
#include <format>
#include <string>
#include "app/App.h"
#include "core/Log.h"
#include "core/Settings.h"
#include "platform/Affinity.h"
#include "tuning/InferenceMode.h"
#include "core/Util.h"
#include "imgui.h"
#include "imgui_impl_dx11.h"
#include "imgui_impl_win32.h"
#include "implot.h"

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

namespace {

ID3D11Device* g_device = nullptr;
ID3D11DeviceContext* g_context = nullptr;
IDXGISwapChain* g_swapChain = nullptr;
ID3D11RenderTargetView* g_rtv = nullptr;
bool g_occluded = false;
DWORD g_lastInput = 0;   // GetTickCount() of the last mouse/keyboard input to the window
UINT g_resizeW = 0, g_resizeH = 0;

void CreateRenderTarget() {
    ID3D11Texture2D* back = nullptr;
    g_swapChain->GetBuffer(0, IID_PPV_ARGS(&back));
    g_device->CreateRenderTargetView(back, nullptr, &g_rtv);
    back->Release();
}

void CleanupRenderTarget() {
    if (g_rtv) { g_rtv->Release(); g_rtv = nullptr; }
}

bool CreateDeviceD3D(HWND hwnd) {
    DXGI_SWAP_CHAIN_DESC sd{};
    sd.BufferCount = 2;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate = {60, 1};
    sd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = hwnd;
    sd.SampleDesc.Count = 1;
    sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
    const D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_0};
    D3D_FEATURE_LEVEL got;
    HRESULT hr = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, levels, 2,
                                               D3D11_SDK_VERSION, &sd, &g_swapChain, &g_device, &got, &g_context);
    if (hr == DXGI_ERROR_UNSUPPORTED)
        hr = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0, levels, 2, D3D11_SDK_VERSION,
                                           &sd, &g_swapChain, &g_device, &got, &g_context);
    if (FAILED(hr)) return false;
    CreateRenderTarget();
    return true;
}

void CleanupDeviceD3D() {
    CleanupRenderTarget();
    if (g_swapChain) g_swapChain->Release();
    if (g_context) g_context->Release();
    if (g_device) g_device->Release();
}

LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if ((msg >= WM_MOUSEFIRST && msg <= WM_MOUSELAST) || (msg >= WM_KEYFIRST && msg <= WM_KEYLAST) ||
        msg == WM_SIZE || msg == WM_SETFOCUS || msg == WM_ACTIVATE)
        g_lastInput = GetTickCount();
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam)) return true;
    switch (msg) {
    case WM_SIZE:
        if (wParam != SIZE_MINIMIZED) {
            g_resizeW = LOWORD(lParam);
            g_resizeH = HIWORD(lParam);
        }
        return 0;
    case WM_SYSCOMMAND:
        if ((wParam & 0xfff0) == SC_KEYMENU) return 0;
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(hWnd, msg, wParam, lParam);
}

// Saves the current back buffer (RGBA8) as a PNG through WIC. Used by --snapshot.
bool SaveBackBuffer(const std::wstring& path) {
    ID3D11Texture2D* back = nullptr;
    if (FAILED(g_swapChain->GetBuffer(0, IID_PPV_ARGS(&back)))) return false;
    D3D11_TEXTURE2D_DESC d;
    back->GetDesc(&d);
    d.Usage = D3D11_USAGE_STAGING;
    d.BindFlags = 0;
    d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    d.MiscFlags = 0;
    ID3D11Texture2D* staging = nullptr;
    bool ok = SUCCEEDED(g_device->CreateTexture2D(&d, nullptr, &staging));
    if (ok) g_context->CopyResource(staging, back);
    back->Release();
    D3D11_MAPPED_SUBRESOURCE m{};
    ok = ok && SUCCEEDED(g_context->Map(staging, 0, D3D11_MAP_READ, 0, &m));
    IWICImagingFactory* wic = nullptr;
    IWICStream* stream = nullptr;
    IWICBitmapEncoder* enc = nullptr;
    IWICBitmapFrameEncode* frame = nullptr;
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    ok = ok && SUCCEEDED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&wic)));
    ok = ok && SUCCEEDED(wic->CreateStream(&stream)) && SUCCEEDED(stream->InitializeFromFilename(path.c_str(), GENERIC_WRITE));
    ok = ok && SUCCEEDED(wic->CreateEncoder(GUID_ContainerFormatPng, nullptr, &enc)) &&
         SUCCEEDED(enc->Initialize(stream, WICBitmapEncoderNoCache)) && SUCCEEDED(enc->CreateNewFrame(&frame, nullptr)) &&
         SUCCEEDED(frame->Initialize(nullptr)) && SUCCEEDED(frame->SetSize(d.Width, d.Height));
    // swap to BGRA, which the PNG encoder takes natively
    std::vector<BYTE> px;
    if (ok) {
        px.resize((size_t)m.RowPitch * d.Height);
        memcpy(px.data(), m.pData, px.size());
        for (UINT y = 0; y < d.Height; ++y)
            for (UINT x = 0; x < d.Width; ++x) std::swap(px[(size_t)y * m.RowPitch + x * 4], px[(size_t)y * m.RowPitch + x * 4 + 2]);
    }
    WICPixelFormatGUID fmt = GUID_WICPixelFormat32bppBGRA;
    ok = ok && SUCCEEDED(frame->SetPixelFormat(&fmt)) &&
         SUCCEEDED(frame->WritePixels(d.Height, m.RowPitch, m.RowPitch * d.Height, px.data())) &&
         SUCCEEDED(frame->Commit()) && SUCCEEDED(enc->Commit());
    if (frame) frame->Release();
    if (enc) enc->Release();
    if (stream) stream->Release();
    if (wic) wic->Release();
    if (staging) { g_context->Unmap(staging, 0); staging->Release(); }
    return ok;
}

void ApplyTheme(float scale) {
    ImGuiStyle& s = ImGui::GetStyle();
    ImGui::StyleColorsDark();
    s.WindowRounding = 4;
    s.FrameRounding = 3;
    s.GrabRounding = 3;
    s.TabRounding = 3;
    s.WindowPadding = ImVec2(10, 8);
    s.ItemSpacing = ImVec2(8, 5);
    ImVec4* c = s.Colors;
    c[ImGuiCol_WindowBg] = ImVec4(0.08f, 0.09f, 0.11f, 1);
    c[ImGuiCol_ChildBg] = ImVec4(0.08f, 0.09f, 0.11f, 1);
    c[ImGuiCol_TitleBg] = ImVec4(0.06f, 0.07f, 0.09f, 1);
    c[ImGuiCol_TitleBgActive] = ImVec4(0.10f, 0.12f, 0.16f, 1);
    c[ImGuiCol_TableRowBgAlt] = ImVec4(1, 1, 1, 0.03f);
    // input boxes, sliders, combos: a lighter fill and a border so they stand out from the window
    s.FrameBorderSize = 1;
    c[ImGuiCol_FrameBg] = ImVec4(0.16f, 0.18f, 0.23f, 1);
    c[ImGuiCol_FrameBgHovered] = ImVec4(0.21f, 0.24f, 0.31f, 1);
    c[ImGuiCol_FrameBgActive] = ImVec4(0.25f, 0.29f, 0.38f, 1);
    c[ImGuiCol_Border] = ImVec4(0.36f, 0.40f, 0.48f, 0.85f);
    s.ScaleAllSizes(scale);
    s.FontScaleDpi = scale;
}

} // namespace

// Headless check: probes + memory and PCIe tests, results in llamascope.log, no window.
int RunSelfTest() {
    App app;
    app.cpu = CpuTopology::Probe();
    Affinity::RestrictToECores(app.cpu);
    app.mem = MemInventory::Probe();
    app.sys = SystemConfig::Probe();
    Log::I(std::format("selftest: {} | {}P+{}E | {} x {} MT/s (rated {}) | theoretical {:.1f} GB/s", app.cpu.brand,
                       app.cpu.pCores, app.cpu.eCores, app.mem.dimms.size(), app.mem.configuredMTs, app.mem.ratedMTs,
                       app.mem.theoreticalGBs));
    Log::I(std::format("selftest: power plan '{}' aspm {} lockpages assigned={} token={} elevated={}",
                       app.sys.powerPlan, app.sys.aspmAC, app.sys.lockPagesAssigned, app.sys.lockPagesInToken,
                       app.sys.elevated));
    app.memTest.Start(app.cpu);
    while (app.memTest.Running()) Sleep(100);
    app.pcieTest.Start();
    while (app.pcieTest.Running()) Sleep(100);
    return (app.memTest.Result().valid && app.pcieTest.Result().valid) ? 0 : 2;
}

// Headless check: enter inference mode, hold it for a few seconds, restore. Results in llamascope.log.
int RunInferenceModeRoundTrip(bool memTest) {
    CpuTopology cpu = CpuTopology::Probe();
    Affinity::RestrictToECores(cpu);
    Settings s;
    s.Load();
    InferenceMode im;
    im.SetContext(cpu, s.modelPath);
    auto wait = [&] { while (im.Busy()) Sleep(100); };
    if (im.LeftoverJournal()) { im.RestoreAsync(); wait(); }
    im.EnterAsync();
    wait();
    Sleep(5000);   // two enforcement passes
    Log::I(std::format("roundtrip: active={} moved={}", im.Active(), im.MovedCount()));
    if (memTest) {   // memory bandwidth while every other program is packed onto the E-cores
        MemBwTest t;
        t.Start(cpu);
        while (t.Running()) Sleep(100);
    }
    std::string nm;
    for (auto& n : im.NotMoved()) nm += n + " ";
    Log::I("roundtrip: not moved: " + nm);
    im.RestoreAsync();
    wait();
    Log::I(std::format("roundtrip: restored, active={}", im.Active()));
    return im.Active() ? 3 : 0;
}

// Headless check: memory controller counters via emon for ~4 s, results in llamascope.log.
int RunImcProbe() {
    CpuTopology cpu = CpuTopology::Probe();
    Affinity::RestrictToECores(cpu);
    ImcCollector imc;
    imc.Start(SystemConfig::Probe().elevated);
    for (int i = 0; i < 50 && imc.GetState() != ImcCollector::State::Running &&
                    imc.GetState() != ImcCollector::State::Failed && imc.GetState() != ImcCollector::State::NoEmon &&
                    imc.GetState() != ImcCollector::State::NotElevated; ++i)
        Sleep(100);
    Sleep(4000);
    std::vector<float> xs, ys, xw, yw;
    imc.readGBs.Copy(0, xs, ys);
    imc.writeGBs.Copy(0, xw, yw);
    std::string r;
    for (size_t i = 0; i < ys.size(); ++i) r += std::format("{:.1f}@{:.2f} ", ys[i], xs[i]);
    Log::I(std::format("imc-probe: state {} samples {} read: {}| {}", (int)imc.GetState(), ys.size(), r, imc.Detail()));
    imc.Stop();
    return ys.empty() ? 4 : 0;
}

// Headless: hold inference mode with only the listed actions (comma-separated ids, e.g. "isolate,power") until
// %LOCALAPPDATA%\LlamaScope\im-hold.stop appears, then restore. Used to measure each action's effect on t/s.
int RunInferenceModeHold(const std::wstring& actions) {
    CpuTopology cpu = CpuTopology::Probe();
    Affinity::RestrictToECores(cpu);
    Settings s;
    s.Load();
    InferenceMode im;
    im.SetContext(cpu, s.modelPath);
    auto wait = [&] { while (im.Busy()) Sleep(100); };
    if (im.LeftoverJournal()) { im.RestoreAsync(); wait(); }
    std::string list = "," + ToUtf8(actions) + ",";
    for (const auto& a : im.Actions()) im.SetEnabled(a.id, list.find("," + a.id + ",") != std::string::npos);
    std::wstring stop = LocalDataDir() + L"\\im-hold.stop";
    // The caller clears the stop file before launching; deleting it here could swallow an early stop request.
    im.EnterAsync();
    wait();
    Log::I("im-hold: active with " + ToUtf8(actions));
    while (GetFileAttributesW(stop.c_str()) == INVALID_FILE_ATTRIBUTES) Sleep(200);
    DeleteFileW(stop.c_str());
    im.RestoreAsync();
    wait();
    Log::I(std::format("im-hold: restored, active={}", im.Active()));
    return im.Active() ? 3 : 0;
}

// Headless: load a ggml profile and log the per-layer aggregation (checks the reader without the UI).
int RunProfileDump(const std::wstring& path) {
    ProfileData p = ProfileData::Load(path);
    if (!p.ok) { Log::E("profile-dump: " + p.error); return 5; }
    Log::I(std::format("profile-dump: steps {} prompt-graphs {} threads {} step {:.0f} us | cpu {:.0f} gpu {:.0f} copy {:.0f} sync {:.0f} us | gpu markers {} | experts {}x{}",
                       p.genSteps, p.promptGraphs, p.threads, p.stepUs, p.cpuBusyUs, p.gpuBusyUs, p.copyUs, p.syncUs,
                       p.hasGpu, p.genCounts.size(), p.nExperts));
    Log::I(std::format("profile-dump: hand-offs {:.0f} us, main thread waiting for GPU {:.0f} us, other {:.0f} us per step",
                       p.handoffUs, p.waitGpuUs, p.stepUs - p.cpuBusyUs - p.gpuBusyUs - p.handoffUs));
    for (const auto& c : p.copies)
        Log::I(std::format("profile-dump: copy {} {} x{:.0f}/token {} B path {} sync {:.1f} us copy {:.1f} us", c.tensor,
                           c.toCpu ? "GPU->CPU" : "CPU->GPU", c.perStep, c.bytes, c.path, c.syncUs, c.copyUs));
    for (size_t L = 0; L < p.layers.size(); ++L) {
        const auto& ly = p.layers[L];
        std::string s;
        for (size_t b = 0; b < (size_t)ProfBlock::Count; ++b)
            if (ly.cpuUs[b] > 0.5 || ly.gpuUs[b] > 0.5)
                s += std::format(" {}: gpu {:.0f} cpu {:.0f} ({:.0f}% wait);", ProfBlockName((ProfBlock)b), ly.gpuUs[b], ly.cpuUs[b], ly.spinPct[b]);
        uint64_t picks = 0;
        if (L < p.genCounts.size()) for (uint32_t c : p.genCounts[L]) picks += c;
        Log::I(std::format("profile-dump: layer {:2}{} toCpu {:.0f} toGpu {:.0f} us, picks {}", L, s, ly.toCpuUs, ly.toGpuUs, picks));
    }
    return 0;
}

int WINAPI wWinMain(HINSTANCE hInst, HINSTANCE, PWSTR cmdLine, int) {
    if (cmdLine) {
        std::wstring cl0 = cmdLine;
        size_t pd = cl0.find(L"--profile-dump ");
        if (pd != std::wstring::npos) {
            std::wstring path = cl0.substr(pd + 15);
            if (!path.empty() && path.front() == L'"') path = path.substr(1, path.find(L'"', 1) - 1);
            return RunProfileDump(path);
        }
        std::wstring cl = cmdLine;
        size_t k = cl.find(L"--im-hold=");
        if (k != std::wstring::npos) {
            std::wstring v = cl.substr(k + 10);
            v = v.substr(0, v.find(L' '));
            return RunInferenceModeHold(v);
        }
    }
    if (cmdLine && wcsstr(cmdLine, L"--imc-probe")) return RunImcProbe();
    if (cmdLine && wcsstr(cmdLine, L"--selftest")) return RunSelfTest();
    if (cmdLine && wcsstr(cmdLine, L"--im-roundtrip")) return RunInferenceModeRoundTrip(wcsstr(cmdLine, L"--memtest") != nullptr);
    ImGui_ImplWin32_EnableDpiAwareness();
    float scale = ImGui_ImplWin32_GetDpiScaleForMonitor(MonitorFromPoint(POINT{0, 0}, MONITOR_DEFAULTTOPRIMARY));

    WNDCLASSEXW wc{sizeof(wc), CS_CLASSDC, WndProc, 0, 0, hInst, nullptr, nullptr, nullptr, nullptr, L"LlamaScope", nullptr};
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    RegisterClassExW(&wc);
    HWND hwnd = CreateWindowW(wc.lpszClassName, L"LlamaScope", WS_OVERLAPPEDWINDOW, 80, 60, (int)(1600 * scale),
                              (int)(950 * scale), nullptr, nullptr, hInst, nullptr);
    if (!CreateDeviceD3D(hwnd)) {
        CleanupDeviceD3D();
        UnregisterClassW(wc.lpszClassName, hInst);
        MessageBoxW(nullptr, L"Direct3D 11 initialization failed.", L"LlamaScope", MB_ICONERROR);
        return 1;
    }
    ShowWindow(hwnd, SW_SHOWMAXIMIZED);
    UpdateWindow(hwnd);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_DockingEnable;

    static std::string iniPath = ToUtf8(LocalDataDir() + L"\\imgui.ini");
    bool firstRun = GetFileAttributesW(ToWide(iniPath).c_str()) == INVALID_FILE_ATTRIBUTES;
    io.IniFilename = iniPath.c_str();

    ApplyTheme(scale);
    if (GetFileAttributesW(L"C:\\Windows\\Fonts\\segoeui.ttf") != INVALID_FILE_ATTRIBUTES)
        io.Fonts->AddFontFromFileTTF("C:\\Windows\\Fonts\\segoeui.ttf", 16.0f);

    ImGui_ImplWin32_Init(hwnd);
    ImGui_ImplDX11_Init(g_device, g_context);

    App app;
    app.Init(firstRun);
    if (cmdLine && wcsstr(cmdLine, L"--run-tests")) app.hwCheck.startAt = NowSeconds() + 5;   // after startup settles
    if (cmdLine && wcsstr(cmdLine, L"--cmdline \"")) {
        const wchar_t* a = wcsstr(cmdLine, L"--cmdline \"") + 11;
        const wchar_t* e = wcschr(a, L'"');
        g_applyCmdline = ToUtf8(std::wstring(a, e ? e : a + wcslen(a)));
    }
    if (cmdLine && wcsstr(cmdLine, L"--topology-scroll ")) g_topologyScroll = (float)_wtof(wcsstr(cmdLine, L"--topology-scroll ") + 18);
    if (cmdLine && wcsstr(cmdLine, L"--summary-scroll ")) g_summaryScroll = (float)_wtof(wcsstr(cmdLine, L"--summary-scroll ") + 17);
    if (cmdLine && wcsstr(cmdLine, L"--profile-latest")) app.LoadLatestProfile();
    if (cmdLine && wcsstr(cmdLine, L"--run-profiling")) app.profileRun.startAt = NowSeconds() + 3;
    else if (cmdLine && wcsstr(cmdLine, L"--profile ")) {
        std::wstring p = wcsstr(cmdLine, L"--profile ") + 10;
        if (!p.empty() && p.front() == L'"') p = p.substr(1, p.find(L'"', 1) - 1);
        else p = p.substr(0, p.find(L' '));
        app.LoadProfile(p);
    }

    // --snapshot <file.png> [--snapshot-after <seconds>]: render (even when occluded), save the back buffer, exit.
    std::wstring snapshot;
    double snapshotAt = 0;
    if (cmdLine && wcsstr(cmdLine, L"--snapshot ")) {
        snapshot = wcsstr(cmdLine, L"--snapshot ") + 11;
        if (!snapshot.empty() && snapshot.front() == L'"') snapshot = snapshot.substr(1, snapshot.find(L'"', 1) - 1);
        else snapshot = snapshot.substr(0, snapshot.find(L' '));
        const wchar_t* after = wcsstr(cmdLine, L"--snapshot-after ");
        snapshotAt = NowSeconds() + (after ? _wtof(after + 17) : 8.0);
    }

    bool done = false;
    while (!done) {
        // Redrawing costs inference speed: every frame makes the compositor (dwm, often on the P-cores) and the
        // integrated GPU (shares the CPU ring and memory bus) do work; measured -5% t/s at full frame rate.
        // The data changes 4-10 times a second, so only draw at the monitor's rate while the user interacts.
        DWORD sinceInput = GetTickCount() - g_lastInput;
        if (sinceInput > 1500) {
            bool llamaBusy = false;
            app.RunningLlama(&llamaBusy);
            DWORD wait = llamaBusy ? 500 : 250;
            MsgWaitForMultipleObjects(0, nullptr, FALSE, wait, QS_ALLINPUT);
        }
        MSG msg;
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
            if (msg.message == WM_QUIT) done = true;
        }
        if (done) break;

        if (snapshot.empty() && g_occluded && g_swapChain->Present(0, DXGI_PRESENT_TEST) == DXGI_STATUS_OCCLUDED) {
            Sleep(50);
            continue;
        }
        g_occluded = false;
        if (g_resizeW && g_resizeH) {
            CleanupRenderTarget();
            g_swapChain->ResizeBuffers(0, g_resizeW, g_resizeH, DXGI_FORMAT_UNKNOWN, 0);
            g_resizeW = g_resizeH = 0;
            CreateRenderTarget();
        }

        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();
        app.Frame();
        ImGui::Render();

        const float clear[4] = {0.05f, 0.06f, 0.07f, 1.0f};
        g_context->OMSetRenderTargets(1, &g_rtv, nullptr);
        g_context->ClearRenderTargetView(g_rtv, clear);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());

        // vsync; the UI shares the GPU with inference, so keep it cheap.
        if (!snapshot.empty() && NowSeconds() >= snapshotAt) {
            bool ok = SaveBackBuffer(snapshot);
            Log::I(std::string("snapshot ") + (ok ? "saved: " : "FAILED: ") + ToUtf8(snapshot));
            done = true;
        }
        HRESULT hr = g_swapChain->Present(1, 0);
        g_occluded = hr == DXGI_STATUS_OCCLUDED;
    }

    app.Shutdown();
    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    CleanupDeviceD3D();
    DestroyWindow(hwnd);
    UnregisterClassW(wc.lpszClassName, hInst);
    return 0;
}
