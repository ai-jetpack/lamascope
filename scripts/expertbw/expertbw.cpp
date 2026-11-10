// Standalone memory-bandwidth test of expert-read schedules (no llama.cpp, no LlamaScope).
// Simulates the CPU expert phase of one MoE op: 10 experts picked at random from a 512-expert tensor,
// read by 16 threads pinned to CPUs 0-15, spin barrier after each op.
//
// Schedules:
//   L  one long contiguous stream per thread (memory-test pattern, reference)
//   A  per expert, its bytes split in 16 slices: each thread reads 1/16 of every expert (today's pattern)
//   B  one thread per expert (10 threads busy, placed on distinct physical cores first), 6 idle
//   C  the 10 experts lined up end to end, each thread reads one contiguous 1/16 of the total
// Each of A and C also with 2 MB large pages if SeLockMemoryPrivilege can be enabled.
#define NOMINMAX
#include <windows.h>
#include <immintrin.h>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

static int g_threads = 16;
static std::vector<int> g_cpus;   // CPU per thread; empty = 0..15
constexpr int    kExperts  = 512;
constexpr int    kPicked   = 10;
constexpr size_t kExpert   = 576 * 1024;        // ~gate/up matrix of one expert (0.59 MB measured)
constexpr int    kLayers   = 8;                 // 8 x 288 MB = 2.3 GB, far beyond the 30 MB L3
constexpr int    kOps      = 4000;

static std::atomic<uint64_t> g_sink{0};

static uint64_t read_range(const uint8_t* p, size_t n) {
    __m256i a = _mm256_setzero_si256(), b = a, c = a, d = a;
    for (size_t i = 0; i + 128 <= n; i += 128) {
        a = _mm256_xor_si256(a, _mm256_load_si256((const __m256i*)(p + i)));
        b = _mm256_xor_si256(b, _mm256_load_si256((const __m256i*)(p + i + 32)));
        c = _mm256_xor_si256(c, _mm256_load_si256((const __m256i*)(p + i + 64)));
        d = _mm256_xor_si256(d, _mm256_load_si256((const __m256i*)(p + i + 96)));
    }
    a = _mm256_xor_si256(_mm256_xor_si256(a, b), _mm256_xor_si256(c, d));
    return (uint64_t)_mm256_extract_epi64(a, 0);
}

struct SpinBarrier {
    std::atomic<int> count{0}, gen{0};
    int n;
    explicit SpinBarrier(int n) : n(n) {}
    void wait() {
        int g = gen.load(std::memory_order_acquire);
        if (count.fetch_add(1, std::memory_order_acq_rel) == n - 1) {
            count.store(0, std::memory_order_relaxed);
            gen.fetch_add(1, std::memory_order_release);
        } else {
            while (gen.load(std::memory_order_acquire) == g) _mm_pause();
        }
    }
};

static bool enable_lock_memory() {
    HANDLE tok;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &tok)) return false;
    TOKEN_PRIVILEGES tp{1};
    LookupPrivilegeValueW(nullptr, L"SeLockMemoryPrivilege", &tp.Privileges[0].Luid);
    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
    AdjustTokenPrivileges(tok, FALSE, &tp, 0, nullptr, nullptr);
    bool ok = GetLastError() == ERROR_SUCCESS;
    CloseHandle(tok);
    return ok;
}

static uint8_t* alloc(size_t bytes, bool large) {
    if (large) {
        size_t lp = GetLargePageMinimum();
        bytes = (bytes + lp - 1) / lp * lp;
        return (uint8_t*)VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES, PAGE_READWRITE);
    }
    return (uint8_t*)VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
}

static double now_s() {
    static LARGE_INTEGER f = [] { LARGE_INTEGER x; QueryPerformanceFrequency(&x); return x; }();
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / f.QuadPart;
}

// Logical CPU for thread t. Plain order (0..15) like --cpu-strict with mask 0xFFFF; for B the 10 busy threads go to
// distinct physical cores first (CPU 2k and 2k+1 are the two hyperthreads of P-core k).
static int cpu_for(int t, bool cores_first) {
    if (!g_cpus.empty()) return g_cpus[t];
    if (!cores_first) return t;
    return t < 8 ? 2 * t : 2 * (t - 8) + 1;
}

static int g_every = 1;   // barrier after every N ops (N = 2 ~ gate and up fused into one op)

#include <string>
static void run(const char* label, char sched, uint8_t* base, bool large) {
    std::vector<std::vector<int>> picks(kOps, std::vector<int>(kPicked));
    std::mt19937 rng(42);
    for (auto& p : picks) {
        std::vector<int> all(kExperts);
        for (int i = 0; i < kExperts; ++i) all[i] = i;
        std::shuffle(all.begin(), all.end(), rng);
        std::copy(all.begin(), all.begin() + kPicked, p.begin());
    }
    const size_t layerBytes = (size_t)kExperts * kExpert;
    const size_t opBytes = (size_t)kPicked * kExpert;
    SpinBarrier bar(g_threads);
    std::atomic<int> ready{0};
    double t0 = 0, t1 = 0;
    std::vector<std::thread> th;
    for (int t = 0; t < g_threads; ++t) {
        th.emplace_back([&, t] {
            if (!SetThreadAffinityMask(GetCurrentThread(), 1ull << cpu_for(t, sched == 'B'))) printf("pin failed\n");
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
            uint64_t x = 0;
            bar.wait();
            if (t == 0) t0 = now_s();
            for (int op = 0; op < kOps; ++op) {
                const uint8_t* layer = base + (size_t)(op % kLayers) * layerBytes;
                const auto& p = picks[op];
                if (sched == 'L') {
                    // same bytes per op as the others, as one contiguous stream per thread (rotating through the buffer)
                    size_t per = opBytes / g_threads;
                    size_t off = ((size_t)op * opBytes + t * per) % (kLayers * layerBytes - per);
                    off &= ~(size_t)127;
                    x ^= read_range(base + off, per);
                } else if (sched == 'A') {
                    size_t slice = kExpert / g_threads;
                    for (int e = 0; e < kPicked; ++e) x ^= read_range(layer + (size_t)p[e] * kExpert + t * slice, slice);
                } else if (sched == 'B') {
                    if (t < kPicked) x ^= read_range(layer + (size_t)p[t] * kExpert, kExpert);
                } else {   // 'C'
                    size_t per = opBytes / g_threads, beg = t * per, end = beg + per;
                    while (beg < end) {
                        int e = (int)(beg / kExpert);
                        size_t in = beg % kExpert, n = std::min(end - beg, kExpert - in);
                        x ^= read_range(layer + (size_t)p[e] * kExpert + in, n & ~(size_t)127);
                        beg += n;
                    }
                }
                if ((op + 1) % g_every == 0) bar.wait();
            }
            bar.wait();
            if (t == 0) t1 = now_s();
            g_sink ^= x;
        });
    }
    for (auto& x : th) x.join();
    const double secs = t1 - t0, gb = (double)opBytes * kOps / 1e9;
    printf("%-60s %7.1f GB/s   %6.1f us per op (%.2f MB)\n", label, gb / secs, secs / kOps * 1e6, opBytes / 1e6);
}

int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "--long") == 0) {
        const bool pe = argc > 2 && strcmp(argv[2], "pe") == 0;   // 1 per P-core + 1 per E-core (memory-test layout)
        const int pe_cpus[16] = {0,2,4,6,8,10,12,14,16,17,18,19,20,21,22,23};
        // memory-test pattern: each thread streams its own 1/16 of the buffer (~150 MB) per pass, barrier per pass
        const size_t total = (size_t)kLayers * kExperts * kExpert;
        uint8_t* buf = alloc(total, false);
        memset(buf, 1, total);
        for (int r = 0; r < 4; ++r) {
            SpinBarrier bar(g_threads);
            double t0 = 0, t1 = 0;
            const int passes = 10;
            const size_t per = (total / g_threads) & ~(size_t)127;
            std::vector<std::thread> th;
            for (int t = 0; t < g_threads; ++t)
                th.emplace_back([&, t] {
                    SetThreadAffinityMask(GetCurrentThread(), 1ull << (pe ? pe_cpus[t] : t));
                    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
                    uint64_t x = 0;
                    bar.wait();
                    if (t == 0) t0 = now_s();
                    for (int k = 0; k < passes; ++k) { x ^= read_range(buf + t * per, per); bar.wait(); }
                    if (t == 0) t1 = now_s();
                    g_sink ^= x;
                });
            for (auto& x : th) x.join();
            printf("M  one %.0f MB stream per thread per pass (%s), round %d   %7.1f GB/s\n", per / 1e6, pe ? "pe16" : "p16", r + 1,
                   (double)per * g_threads * passes / 1e9 / (t1 - t0));
        }
        VirtualFree(buf, 0, MEM_RELEASE);
        return 0;
    }
    if (argc > 1 && strcmp(argv[1], "--core-sweep") == 0) {
        const size_t total = (size_t)kLayers * kExperts * kExpert;
        uint8_t* buf = alloc(total, false);
        memset(buf, 1, total);
        struct Set { const char* name; std::vector<int> cpus; };
        std::vector<Set> sets = {
            {"p16  16 hyperthreads on 8 P-cores (-C 0xFFFF)", {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}},
            {"p8   1 thread per P-core", {0,2,4,6,8,10,12,14}},
            {"pe16 1 per P-core + 1 per E-core (memory test)", {0,2,4,6,8,10,12,14,16,17,18,19,20,21,22,23}},
            {"all24 every logical CPU", {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23}},
        };
        printf("core sweep: 4 KB pages, barrier after every op, 3 rounds each (round 1 of the first set is warm-up)\n");
        for (auto& st : sets) {
            g_cpus = st.cpus;
            g_threads = (int)st.cpus.size();
            for (int r = 0; r < 3; ++r) {
                char l1[96], l2[96];
                snprintf(l1, sizeof l1, "%s  L r%d", st.name, r + 1);
                snprintf(l2, sizeof l2, "%s  A r%d", st.name, r + 1);
                run(l1, 'L', buf, false);
                run(l2, 'A', buf, false);
            }
        }
        VirtualFree(buf, 0, MEM_RELEASE);
        return 0;
    }
    if (argc > 1 && strcmp(argv[1], "--barrier-sweep") == 0) {
        const size_t total = (size_t)kLayers * kExperts * kExpert;
        uint8_t* buf = alloc(total, false);
        memset(buf, 1, total);
        printf("barrier sweep: 4 KB pages, 3 rounds each\n");
        for (int every : {1, 2, 3, 6}) {
            g_every = every;
            for (int r = 0; r < 3; ++r) {
                char l1[64], l2[64];
                snprintf(l1, sizeof l1, "L  barrier every %d op(s), round %d", every, r + 1);
                snprintf(l2, sizeof l2, "A  barrier every %d op(s), round %d", every, r + 1);
                run(l1, 'L', buf, false);
                run(l2, 'A', buf, false);
            }
        }
        VirtualFree(buf, 0, MEM_RELEASE);
        return 0;
    }
    const size_t total = (size_t)kLayers * kExperts * kExpert;
    printf("buffer %.2f GB, %d experts x %zu KB per layer, %d picked per op, %d threads, %d ops\n\n", total / 1e9, kExperts,
           kExpert / 1024, kPicked, g_threads, kOps);
    uint8_t* buf4k = alloc(total, false);
    if (!buf4k) { printf("allocation failed\n"); return 1; }
    memset(buf4k, 1, total);
    for (int rep = 0; rep < 2; ++rep) {
        printf("-- 4 KB pages, round %d\n", rep + 1);
        run("L  one long stream per thread (reference)", 'L', buf4k, false);
        run("A  1/16 of every expert per thread (today)", 'A', buf4k, false);
        run("B  one thread per expert (10 busy)", 'B', buf4k, false);
        run("C  contiguous 1/16 of the 10 experts", 'C', buf4k, false);
    }
    VirtualFree(buf4k, 0, MEM_RELEASE);

    if (!enable_lock_memory()) { printf("\nlarge pages: SeLockMemoryPrivilege not enabled in this token, skipped\n"); return 0; }
    uint8_t* buf2m = alloc(total, true);
    if (!buf2m) { printf("\nlarge pages: allocation failed (error %lu; memory fragmented?), skipped\n", GetLastError()); return 0; }
    memset(buf2m, 1, total);
    printf("-- 2 MB large pages\n");
    run("L  one long stream per thread (reference)", 'L', buf2m, true);
    run("A  1/16 of every expert per thread (today)", 'A', buf2m, true);
    run("C  contiguous 1/16 of the 10 experts", 'C', buf2m, true);
    VirtualFree(buf2m, 0, MEM_RELEASE);
    return 0;
}
