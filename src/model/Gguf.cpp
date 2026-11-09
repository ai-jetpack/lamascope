#include <windows.h>
#include <algorithm>
#include <cstring>
#include <format>
#include <stdexcept>
#include "core/Util.h"
#include "model/Gguf.h"

namespace {

// Bounds-checked cursor over an in-memory copy of the header.
class Cursor {
public:
    Cursor(const uint8_t* p, size_t n) : p_(p), n_(n) {}
    template <typename T> T Get() {
        Need(sizeof(T));
        T v;
        memcpy(&v, p_ + pos_, sizeof(T));
        pos_ += sizeof(T);
        return v;
    }
    std::string Str() {
        uint64_t len = Get<uint64_t>();
        Need(len);
        std::string s((const char*)p_ + pos_, (size_t)len);
        pos_ += (size_t)len;
        return s;
    }
    void Skip(uint64_t n) { Need(n); pos_ += (size_t)n; }
    size_t Pos() const { return pos_; }

    struct Truncated {};
private:
    void Need(uint64_t k) {
        if (pos_ + k > n_) throw Truncated{};
    }
    const uint8_t* p_;
    size_t n_, pos_ = 0;
};

size_t ScalarSize(uint32_t t) {
    switch (t) {
    case 0: case 1: case 7: return 1;
    case 2: case 3: return 2;
    case 4: case 5: case 6: return 4;
    case 10: case 11: case 12: return 8;
    default: return 0;
    }
}

GgufValue ReadScalar(Cursor& c, uint32_t t) {
    switch (t) {
    case 0: return (int64_t)c.Get<uint8_t>();
    case 1: return (int64_t)c.Get<int8_t>();
    case 2: return (int64_t)c.Get<uint16_t>();
    case 3: return (int64_t)c.Get<int16_t>();
    case 4: return (int64_t)c.Get<uint32_t>();
    case 5: return (int64_t)c.Get<int32_t>();
    case 6: return (double)c.Get<float>();
    case 7: return c.Get<uint8_t>() != 0;
    case 10: return (int64_t)c.Get<uint64_t>();
    case 11: return c.Get<int64_t>();
    case 12: return c.Get<double>();
    default: throw std::runtime_error(std::format("unknown GGUF value type {}", t));
    }
}

GgufValue ReadValue(Cursor& c, uint32_t t) {
    if (t == 8) return c.Str();
    if (t != 9) return ReadScalar(c, t);
    uint32_t et = c.Get<uint32_t>();
    uint64_t n = c.Get<uint64_t>();
    if (et == 8) {                       // string array (tokenizer): skip, keep the count
        for (uint64_t i = 0; i < n; ++i) c.Skip(c.Get<uint64_t>());
        return (int64_t)n;
    }
    size_t sz = ScalarSize(et);
    if (!sz) throw std::runtime_error("nested GGUF arrays are not supported");
    if (n > 4096) {                      // long numeric array: skip, keep the count
        c.Skip(n * sz);
        return (int64_t)n;
    }
    if (et == 6 || et == 12) {
        std::vector<double> v;
        for (uint64_t i = 0; i < n; ++i) v.push_back(std::get<double>(ReadScalar(c, et)));
        return v;
    }
    std::vector<int64_t> v;
    for (uint64_t i = 0; i < n; ++i) {
        auto s = ReadScalar(c, et);
        v.push_back(std::holds_alternative<bool>(s) ? (int64_t)std::get<bool>(s) : std::get<int64_t>(s));
    }
    return v;
}

} // namespace

std::string GgufFile::Str(const std::string& key, const std::string& def) const {
    auto it = meta.find(key);
    if (it != meta.end() && std::holds_alternative<std::string>(it->second)) return std::get<std::string>(it->second);
    return def;
}

int64_t GgufFile::Int(const std::string& key, int64_t def) const {
    auto it = meta.find(key);
    if (it == meta.end()) return def;
    if (auto p = std::get_if<int64_t>(&it->second)) return *p;
    if (auto p = std::get_if<double>(&it->second)) return (int64_t)*p;
    if (auto p = std::get_if<bool>(&it->second)) return *p;
    if (auto p = std::get_if<std::vector<int64_t>>(&it->second))   // per-layer arrays: use the max
        return p->empty() ? def : *std::max_element(p->begin(), p->end());
    return def;
}

double GgufFile::Float(const std::string& key, double def) const {
    auto it = meta.find(key);
    if (it == meta.end()) return def;
    if (auto p = std::get_if<double>(&it->second)) return *p;
    if (auto p = std::get_if<int64_t>(&it->second)) return (double)*p;
    return def;
}

GgufFile GgufFile::Read(const std::wstring& path) {
    GgufFile g;
    g.path = ToUtf8(path);
    HANDLE h = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                           OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        g.error = "cannot open file";
        return g;
    }
    LARGE_INTEGER sz{};
    GetFileSizeEx(h, &sz);
    g.fileSize = (uint64_t)sz.QuadPart;

    // The header (tokenizer included) is usually < 16 MB; grow the buffer if parsing runs past it.
    std::vector<uint8_t> buf;
    for (size_t want = size_t(16) << 20;; want *= 2) {
        want = (size_t)std::min<uint64_t>(want, g.fileSize);
        buf.resize(want);
        LARGE_INTEGER zero{};
        SetFilePointerEx(h, zero, nullptr, FILE_BEGIN);
        DWORD got = 0;
        size_t done = 0;
        while (done < want && ReadFile(h, buf.data() + done, (DWORD)std::min<size_t>(want - done, 1 << 30), &got, nullptr) && got)
            done += got;
        buf.resize(done);
        try {
            Cursor c(buf.data(), buf.size());
            if (c.Get<uint32_t>() != 0x46554747) { g.error = "not a GGUF file"; break; }   // "GGUF"
            g.version = c.Get<uint32_t>();
            if (g.version < 2) { g.error = std::format("GGUF v{} is not supported", g.version); break; }
            uint64_t nTensors = c.Get<uint64_t>(), nKv = c.Get<uint64_t>();
            g.meta.clear();
            g.tensors.clear();
            for (uint64_t i = 0; i < nKv; ++i) {
                std::string key = c.Str();
                g.meta[key] = ReadValue(c, c.Get<uint32_t>());
            }
            for (uint64_t i = 0; i < nTensors; ++i) {
                GgufTensor t;
                t.name = c.Str();
                uint32_t nd = c.Get<uint32_t>();
                for (uint32_t d = 0; d < nd; ++d) t.dims.push_back(c.Get<uint64_t>());
                t.type = c.Get<uint32_t>();
                t.offset = c.Get<uint64_t>();
                g.tensors.push_back(std::move(t));
            }
            uint64_t align = (uint64_t)std::max<int64_t>(1, g.Int("general.alignment", 32));
            uint64_t dataStart = (c.Pos() + align - 1) / align * align;
            // Sizes from offset deltas (tensors sorted by offset).
            std::vector<size_t> order(g.tensors.size());
            for (size_t i = 0; i < order.size(); ++i) order[i] = i;
            std::sort(order.begin(), order.end(),
                      [&](size_t a, size_t b) { return g.tensors[a].offset < g.tensors[b].offset; });
            for (size_t k = 0; k < order.size(); ++k) {
                uint64_t end = k + 1 < order.size() ? g.tensors[order[k + 1]].offset : g.fileSize - dataStart;
                auto& t = g.tensors[order[k]];
                t.bytes = end > t.offset ? end - t.offset : 0;
            }
            g.ok = true;
            break;
        } catch (Cursor::Truncated&) {
            if (buf.size() >= g.fileSize) { g.error = "truncated GGUF header"; break; }
        } catch (std::exception& e) {
            g.error = e.what();
            break;
        }
    }
    CloseHandle(h);
    return g;
}

const char* GgmlTypeName(uint32_t t) {
    static const char* names[] = {"F32", "F16", "Q4_0", "Q4_1", "?", "?", "Q5_0", "Q5_1", "Q8_0", "Q8_1",
                                  "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K", "IQ2_XXS", "IQ2_XS", "IQ3_XXS",
                                  "IQ1_S", "IQ4_NL", "IQ3_S", "IQ2_S", "IQ4_XS", "I8", "I16", "I32", "I64", "F64",
                                  "IQ1_M", "BF16", "?", "?", "?", "TQ1_0", "TQ2_0", "?", "?", "?", "MXFP4"};
    return t < sizeof(names) / sizeof(names[0]) ? names[t] : "?";
}

const char* GgufFileTypeName(int64_t ft) {
    switch (ft) {
    case 0: return "F32"; case 1: return "F16"; case 2: return "Q4_0"; case 3: return "Q4_1";
    case 7: return "Q8_0"; case 8: return "Q5_0"; case 9: return "Q5_1"; case 10: return "Q2_K";
    case 11: return "Q3_K_S"; case 12: return "Q3_K_M"; case 13: return "Q3_K_L"; case 14: return "Q4_K_S";
    case 15: return "Q4_K_M"; case 16: return "Q5_K_S"; case 17: return "Q5_K_M"; case 18: return "Q6_K";
    case 19: return "IQ2_XXS"; case 20: return "IQ2_XS"; case 21: return "Q2_K_S"; case 22: return "IQ3_XS";
    case 23: return "IQ3_XXS"; case 24: return "IQ1_S"; case 25: return "IQ4_NL"; case 26: return "IQ3_S";
    case 27: return "IQ3_M"; case 28: return "IQ2_S"; case 29: return "IQ2_M"; case 30: return "IQ4_XS";
    case 31: return "IQ1_M"; case 32: return "BF16"; case 38: return "MXFP4_MOE";
    default: return "?";
    }
}
