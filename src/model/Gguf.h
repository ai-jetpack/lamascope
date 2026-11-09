#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

// GGUF header reader. Reads metadata and the tensor table only, never the weights.
// Tensor byte sizes come from offset deltas, so new quant types need no lookup table.
struct GgufTensor {
    std::string name;
    std::vector<uint64_t> dims;
    uint32_t type = 0;
    uint64_t offset = 0;   // relative to the data section
    uint64_t bytes = 0;
    uint64_t Elements() const {
        uint64_t n = 1;
        for (auto d : dims) n *= d;
        return n;
    }
};

// Scalar, string, or (short) numeric array. Long arrays / string arrays are kept as their length only.
using GgufValue = std::variant<int64_t, double, bool, std::string, std::vector<int64_t>, std::vector<double>>;

struct GgufFile {
    bool ok = false;
    std::string error;
    std::string path;
    uint64_t fileSize = 0;
    uint32_t version = 0;
    std::map<std::string, GgufValue> meta;
    std::vector<GgufTensor> tensors;

    std::string Arch() const { return Str("general.architecture"); }
    std::string Str(const std::string& key, const std::string& def = "") const;
    int64_t Int(const std::string& key, int64_t def = 0) const;
    double Float(const std::string& key, double def = 0) const;
    // Architecture-scoped key, e.g. ArchInt("block_count") -> "qwen3next.block_count"
    int64_t ArchInt(const std::string& key, int64_t def = 0) const { return Int(Arch() + "." + key, def); }

    static GgufFile Read(const std::wstring& path);
};

const char* GgmlTypeName(uint32_t type);
const char* GgufFileTypeName(int64_t fileType);
