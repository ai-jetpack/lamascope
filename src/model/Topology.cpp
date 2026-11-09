#include <algorithm>
#include <regex>
#include "model/Topology.h"

namespace {

Block Classify(const std::string& s) {
    if (s.find("_exps") != std::string::npos) return Block::Experts;
    if (s.find("_shexp") != std::string::npos) return Block::Shared;
    if (s.rfind("ffn_gate_inp", 0) == 0) return Block::Router;
    if (s.rfind("ffn_norm", 0) == 0 || s.rfind("attn_norm", 0) == 0 || s.rfind("post_attention_norm", 0) == 0 ||
        s.rfind("layer_output_norm", 0) == 0)
        return Block::Norm;
    if (s.rfind("ffn_", 0) == 0) return Block::DenseFfn;
    if (s.rfind("attn_", 0) == 0 || s.rfind("ssm_", 0) == 0 || s.rfind("time_mix", 0) == 0) return Block::Mixer;
    return Block::Other;
}

} // namespace

const char* BlockName(Block b) {
    switch (b) {
    case Block::Mixer: return "Mixer";
    case Block::Router: return "Router";
    case Block::Shared: return "Shared expert";
    case Block::Experts: return "Routed experts";
    case Block::DenseFfn: return "FFN";
    case Block::Norm: return "Norms";
    default: return "Other";
    }
}

const std::vector<KvType>& KvTypes() {
    static const std::vector<KvType> t = {{"f16", 2.0},       {"q8_0", 34.0 / 32}, {"q4_0", 18.0 / 32},
                                          {"bf16", 2.0},      {"f32", 4.0},        {"q5_0", 22.0 / 32},
                                          {"q5_1", 24.0 / 32}, {"q4_1", 20.0 / 32}, {"iq4_nl", 18.0 / 32}};
    return t;
}

int ModelInfo::CountMixer(MixerKind k) const {
    int n = 0;
    for (auto& l : layers) n += l.mixer == k;
    return n;
}

ModelInfo ModelInfo::FromGguf(const GgufFile& g) {
    ModelInfo m;
    m.path = g.path;
    m.fileName = g.path.substr(g.path.find_last_of("\\/") + 1);
    if (!g.ok) {
        m.error = g.error;
        return m;
    }
    m.ok = true;
    m.fileSize = g.fileSize;
    m.arch = g.Arch();
    m.name = g.Str("general.name", m.fileName);
    m.sizeLabel = g.Str("general.size_label");
    m.quant = GgufFileTypeName(g.Int("general.file_type", -1));
    m.nLayers = (int)g.ArchInt("block_count");
    m.nEmbd = (int)g.ArchInt("embedding_length");
    m.nHead = (int)g.ArchInt("attention.head_count");
    m.nHeadKv = (int)g.ArchInt("attention.head_count_kv", m.nHead);
    int headDim = m.nHead ? m.nEmbd / m.nHead : 0;
    m.keyLen = (int)g.ArchInt("attention.key_length", headDim);
    m.valLen = (int)g.ArchInt("attention.value_length", headDim);
    m.nExpert = (int)g.ArchInt("expert_count");
    m.nExpertUsed = (int)g.ArchInt("expert_used_count");
    m.nExpertShared = (int)g.ArchInt("expert_shared_count", g.meta.count(m.arch + ".expert_shared_feed_forward_length") ? 1 : 0);
    m.fullAttnInterval = (int)g.ArchInt("full_attention_interval");
    m.ctxTrain = (int)g.ArchInt("context_length");
    m.ssmStateSize = (int)g.ArchInt("ssm.state_size");
    m.ssmInner = (int)g.ArchInt("ssm.inner_size");
    m.ssmConvKernel = (int)g.ArchInt("ssm.conv_kernel");

    m.layers.resize(std::max(0, m.nLayers));
    for (int i = 0; i < m.nLayers; ++i) m.layers[i].index = i;
    static const std::regex blk("^blk\\.(\\d+)\\.(.+)$");
    std::smatch sm;
    for (const auto& t : g.tensors) {
        m.totalBytes += t.bytes;
        m.nParams += t.Elements();
        if (std::regex_match(t.name, sm, blk)) {
            int idx = std::stoi(sm[1].str());
            if (idx < 0 || idx >= m.nLayers) continue;
            std::string suffix = sm[2].str();
            Block b = Classify(suffix);
            auto& L = m.layers[idx];
            L.bytes[(int)b] += t.bytes;
            L.tensors[(int)b].push_back({suffix, t.bytes, t.type});
            if (suffix.rfind("ssm_", 0) == 0) L.mixer = MixerKind::Linear;
            else if (L.mixer == MixerKind::None && suffix.rfind("attn_", 0) == 0 && b == Block::Mixer)
                L.mixer = MixerKind::Attention;
            if (suffix.rfind("ssm_conv1d", 0) == 0 && t.dims.size() >= 2) m.ssmConvChannels = (int)t.dims[1];
        } else if (t.name.rfind("token_embd", 0) == 0) {
            m.embdBytes += t.bytes;
            if (t.dims.size() >= 2) m.nVocab = (int)t.dims[1];
        } else if (t.name.rfind("output", 0) == 0) {
            m.outputBytes += t.bytes;
            if (t.name == "output.weight" && t.dims.size() >= 2) m.nVocab = (int)t.dims[1];
        }
    }
    // A layer marked Attention earlier may also hold ssm_* tensors processed later: Linear wins.
    for (auto& L : m.layers)
        for (auto& t : L.tensors[(int)Block::Mixer])
            if (t.name.rfind("ssm_", 0) == 0) L.mixer = MixerKind::Linear;
    // Tied embeddings: llama.cpp puts a copy of token_embd on the GPU as the output layer.
    bool hasOutput = std::any_of(g.tensors.begin(), g.tensors.end(), [](auto& t) { return t.name == "output.weight"; });
    if (!hasOutput) m.outputBytes += m.embdBytes;
    return m;
}

Placement Placement::Compute(const ModelInfo& m, const LaunchParams& p) {
    Placement r;
    const int L = m.nLayers;
    const int ngl = std::clamp(p.ngl, 0, L + 1);
    const int firstGpu = L - std::min(ngl, L);
    r.outputOnGpu = ngl > L;
    r.dev.resize(L);
    r.layerDev.resize(L);
    const double expertFrac = m.nExpert > 0 ? (double)m.nExpertUsed / m.nExpert : 1.0;

    for (int i = 0; i < L; ++i) {
        Device ld = i >= firstGpu ? Device::Gpu : Device::Cpu;
        r.layerDev[i] = ld;
        for (int b = 0; b < kBlocks; ++b) r.dev[i][b] = ld;
        if (i < p.nCpuMoe) r.dev[i][(int)Block::Experts] = Device::Cpu;   // --n-cpu-moe: experts of first N layers
        for (int b = 0; b < kBlocks; ++b) {
            uint64_t bytes = m.layers[i].bytes[b];
            double active = b == (int)Block::Experts ? bytes * expertFrac : (double)bytes;
            if (r.dev[i][b] == Device::Gpu) { r.gpuWeights += bytes; r.gpuBytesPerToken += active; }
            else { r.cpuWeights += bytes; r.cpuBytesPerToken += active; }
        }
    }
    // Input embeddings always stay on the CPU (one row read per token); output follows -ngl.
    r.cpuWeights += m.embdBytes;
    if (r.outputOnGpu) { r.gpuWeights += m.outputBytes; r.gpuBytesPerToken += m.outputBytes; }
    else { r.cpuWeights += m.outputBytes; r.cpuBytesPerToken += m.outputBytes; }

    // KV cache: attention layers only. Recurrent state: linear-mixer layers, per parallel sequence.
    const auto& kv = KvTypes();
    double kb = kv[std::clamp(p.ctk, 0, (int)kv.size() - 1)].bytesPerElement;
    double vb = kv[std::clamp(p.ctv, 0, (int)kv.size() - 1)].bytesPerElement;
    int attn = m.CountMixer(MixerKind::Attention), lin = m.CountMixer(MixerKind::Linear);
    r.kvBytes = (uint64_t)((double)attn * p.nCtx * m.nHeadKv * (m.keyLen * kb + m.valLen * vb));
    uint64_t rsPerLayer = (uint64_t)m.ssmStateSize * m.ssmInner * 4 +
                          (uint64_t)std::max(0, m.ssmConvKernel - 1) * m.ssmConvChannels * 4;
    r.rsBytes = rsPerLayer * lin * std::max(1, p.nParallel);
    // Rough until a load log is available: logits for one ubatch plus graph scratch; CUDA context.
    r.computeEst = (uint64_t)p.nUbatch * m.nVocab * 4 + (350ull << 20);
    r.cudaCtxEst = 300ull << 20;

    // PCIe crossings for one generated token: every device change along the compute order.
    r.crossingBytes = (uint64_t)m.nEmbd * 4;
    Device cur = Device::Cpu;   // token embedding lookup
    auto hop = [&](Device d) { if (d != cur) { ++r.crossingsPerToken; cur = d; } };
    for (int i = 0; i < L; ++i) {
        hop(r.dev[i][(int)Block::Mixer]);
        if (m.layers[i].bytes[(int)Block::Experts]) {
            hop(r.dev[i][(int)Block::Experts]);
            hop(r.layerDev[i]);   // expert outputs are combined with the shared expert on the layer device
        }
    }
    hop(r.outputOnGpu ? Device::Gpu : Device::Cpu);
    return r;
}

int Placement::SmallestFittingCpuMoe(const ModelInfo& m, LaunchParams p, double budget) {
    for (int n = 0; n <= m.nLayers; ++n) {
        p.nCpuMoe = n;
        if ((double)Compute(m, p).GpuTotal() <= budget) return n;
    }
    return -1;
}
