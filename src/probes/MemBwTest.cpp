#include <windows.h>
#include <immintrin.h>
#include <algorithm>
#include <barrier>
#include <cstring>
#include <format>
#include "core/Log.h"
#include "platform/Affinity.h"
#include "probes/MemBwTest.h"

namespace {

std::atomic<uint64_t> g_sink{0};

uint64_t ReadAvx2(const uint8_t* p, size_t n) {
    __m256i a = _mm256_setzero_si256(), b = a, c = a, d = a;
    for (size_t i = 0; i + 128 <= n; i += 128) {
        a = _mm256_xor_si256(a, _mm256_load_si256((const __m256i*)(p + i)));
        b = _mm256_xor_si256(b, _mm256_load_si256((const __m256i*)(p + i + 32)));
        c = _mm256_xor_si256(c, _mm256_load_si256((const __m256i*)(p + i + 64)));
        d = _mm256_xor_si256(d, _mm256_load_si256((const __m256i*)(p + i + 96)));
    }
    a = _mm256_xor_si256(_mm256_xor_si256(a, b), _mm256_xor_si256(c, d));
    uint64_t r = (uint64_t)_mm256_extract_epi64(a, 0) ^ (uint64_t)_mm256_extract_epi64(a, 2);
    _mm256_zeroupper();
    return r;
}

uint64_t ReadSse(const uint8_t* p, size_t n) {
    __m128i a = _mm_setzero_si128(), b = a, c = a, d = a;
    for (size_t i = 0; i + 64 <= n; i += 64) {
        a = _mm_xor_si128(a, _mm_load_si128((const __m128i*)(p + i)));
        b = _mm_xor_si128(b, _mm_load_si128((const __m128i*)(p + i + 16)));
        c = _mm_xor_si128(c, _mm_load_si128((const __m128i*)(p + i + 32)));
        d = _mm_xor_si128(d, _mm_load_si128((const __m128i*)(p + i + 48)));
    }
    a = _mm_xor_si128(_mm_xor_si128(a, b), _mm_xor_si128(c, d));
    return (uint64_t)_mm_cvtsi128_si64(a);
}

double QpcSeconds() {
    static LARGE_INTEGER f = [] { LARGE_INTEGER x; QueryPerformanceFrequency(&x); return x; }();
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}

} // namespace

MemBwTest::~MemBwTest() {
    if (worker_.joinable()) worker_.join();
}

void MemBwTest::Start(const CpuTopology& cpu) {
    if (running_) return;
    if (worker_.joinable()) worker_.join();
    running_ = true;
    progress_ = 0;
    worker_ = std::thread([this, cores = cpu.cores, avx2 = cpu.avx2] { Run(cores, avx2); });
}

void MemBwTest::Run(std::vector<CoreInfo> cores, bool avx2) {
    MemBwResult res;
    const size_t total = size_t(2) << 30;
    const int n = (int)cores.size();
    const int passes = 10;
    auto* buf = (uint8_t*)VirtualAlloc(nullptr, total, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!buf || n == 0) {
        res.error = "Could not allocate 2 GiB test buffer";
        std::lock_guard lk(mu_);
        result_ = res;
        running_ = false;
        return;
    }
    const size_t slice = (total / n) & ~size_t(4095);

    // The app normally runs on E-cores only; the test needs one thread on every core.
    Affinity::ScopedAllCores allCores;

    // Time-window measurement: every thread streams continuously and counts its bytes; each sample is
    // (bytes moved by all threads in the window) / window. A thread preempted by other programs only
    // loses its own share, instead of stalling a lock-step pass for everyone (the old barrier method
    // read ~15% low when Inference mode had packed other programs onto the E-cores).
    struct alignas(64) Counter { std::atomic<uint64_t> bytes{0}; };
    std::vector<Counter> done(n);
    std::atomic<int> ready{0};
    std::atomic<bool> stop{false};
    const size_t chunk = size_t(4) << 20;   // ~1 ms per chunk per core: small vs. the window
    std::vector<std::thread> threads;
    for (int i = 0; i < n; ++i) {
        threads.emplace_back([&, i] {
            int lp = cores[i].logical.front();
            GROUP_AFFINITY ga{};
            ga.Group = (WORD)(lp / 64);
            ga.Mask = KAFFINITY(1) << (lp % 64);
            SetThreadGroupAffinity(GetCurrentThread(), &ga, nullptr);
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
            uint8_t* p = buf + slice * i;
            memset(p, i + 1, slice); // first touch: commit pages from the owning core
            ready.fetch_add(1);
            while (ready.load() < n) _mm_pause();
            uint64_t x = 0;
            size_t off = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                x ^= avx2 ? ReadAvx2(p + off, chunk) : ReadSse(p + off, chunk);
                done[i].bytes.fetch_add(chunk, std::memory_order_relaxed);
                off += chunk;
                if (off + chunk > slice) off = 0;
            }
            g_sink.fetch_xor(x, std::memory_order_relaxed);
        });
    }

    while (ready.load() < n) Sleep(5);
    auto sum = [&] { uint64_t s = 0; for (auto& c : done) s += c.bytes.load(); return s; };
    Sleep(300);   // warm-up: let clocks ramp and every thread reach steady state
    std::vector<double> gbs;
    for (int k = 0; k < passes; ++k) {
        uint64_t b0 = sum();
        double t0 = QpcSeconds();
        Sleep(250);
        uint64_t b1 = sum();
        double t1 = QpcSeconds();
        gbs.push_back((double)(b1 - b0) / (t1 - t0) / 1e9);
        progress_ = float(k + 1) / passes;
    }
    stop = true;
    for (auto& t : threads) t.join();
    VirtualFree(buf, 0, MEM_RELEASE);   // E-core restriction is restored when allCores goes out of scope

    std::sort(gbs.begin(), gbs.end());
    res.valid = true;
    res.bestGBs = gbs.back();
    res.medianGBs = gbs[gbs.size() / 2];
    res.threads = n;
    Log::I(std::format("Memory bandwidth test: best {:.1f} GB/s, median {:.1f} GB/s, {} threads", res.bestGBs,
                       res.medianGBs, n));
    std::lock_guard lk(mu_);
    result_ = res;
    running_ = false;
}
