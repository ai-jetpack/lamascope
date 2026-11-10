#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>
#include "model/Gguf.h"

// Functional blocks of a transformer layer, in compute order.
enum class Block { Mixer, Router, Shared, Experts, DenseFfn, Norm, Other, Count };
inline constexpr int kBlocks = (int)Block::Count;
enum class MixerKind { None, Attention, Linear };   // Linear = Gated DeltaNet / SSM-style recurrent mixer

struct BlockTensor { std::string name; uint64_t bytes; uint32_t type; };

struct LayerInfo {
    int index = 0;
    MixerKind mixer = MixerKind::None;
    std::array<uint64_t, kBlocks> bytes{};
    std::array<std::vector<BlockTensor>, kBlocks> tensors;
    uint64_t Total() const { uint64_t s = 0; for (auto b : bytes) s += b; return s; }
};

struct ModelInfo {
    bool ok = false;
    std::string error, path, fileName, name, arch, quant, sizeLabel;
    uint64_t fileSize = 0, totalBytes = 0, nParams = 0;
    int nLayers = 0, nEmbd = 0, nHead = 0, nHeadKv = 0, keyLen = 0, valLen = 0;
    int nExpert = 0, nExpertUsed = 0, nExpertShared = 0, fullAttnInterval = 0, nVocab = 0, ctxTrain = 0;
    int ssmStateSize = 0, ssmInner = 0, ssmConvKernel = 0, ssmConvChannels = 0;
    uint64_t embdBytes = 0, outputBytes = 0;   // token_embd; output.weight + output_norm
    std::vector<LayerInfo> layers;

    int CountMixer(MixerKind k) const;
    static ModelInfo FromGguf(const GgufFile& g);
};

enum class Device { Gpu, Cpu };

struct KvType { const char* name; double bytesPerElement; };
const std::vector<KvType>& KvTypes();

struct LaunchParams {
    int ngl = 999;
    int nCpuMoe = 0;
    bool cpuMoe = false;    // --cpu-moe: experts of all layers on the CPU
    int nCtx = 16384;
    int nBatch = 2048;
    int nUbatch = 512;
    int nParallel = 1;
    int ctk = 0, ctv = 0;   // indices into KvTypes()
    std::vector<std::pair<std::string, std::string>> overrides;   // -ot / --override-tensor: regex -> buffer
    std::string cmdline;    // last pasted llama.cpp command line
};

// Parses a llama.cpp command line into p. Lists what was applied, what has no effect on placement, what affects
// placement but is not modeled, and what is unknown.
struct ParsedArgs {
    std::vector<std::string> applied, noEffect, notModeled, unknown;
    std::string model;      // -m / --model
};
ParsedArgs ParseLlamaArgs(const std::string& cmdline, LaunchParams& p);
// Splits a command line: whitespace separates, '...' and "..." group, ^ and ` (line continuations) are dropped.
std::vector<std::string> TokenizeArgs(const std::string& cmdline);

struct Placement {
    std::vector<std::array<Device, kBlocks>> dev;   // per layer, per block
    std::vector<Device> layerDev;
    bool outputOnGpu = true;
    uint64_t gpuWeights = 0, cpuWeights = 0;
    uint64_t kvBytes = 0, rsBytes = 0;          // KV cache (attention layers), recurrent state (linear layers)
    uint64_t computeEst = 0, cudaCtxEst = 0;    // estimates until a load log provides real numbers
    double cpuBytesPerToken = 0, gpuBytesPerToken = 0;
    int crossingsPerToken = 0;
    uint64_t crossingBytes = 0;                 // per crossing (one hidden-state vector)

    uint64_t GpuTotal() const { return gpuWeights + kvBytes + rsBytes + computeEst + cudaCtxEst; }
    static Placement Compute(const ModelInfo& m, const LaunchParams& p);
    // Smallest --n-cpu-moe whose GPU total fits in budget bytes, -1 if none.
    static int SmallestFittingCpuMoe(const ModelInfo& m, LaunchParams p, double budget);
};

const char* BlockName(Block b);
