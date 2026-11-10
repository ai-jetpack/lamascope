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

namespace {

// -ot assignments per (layer, block): bytes forced to the CPU and to the GPU. llama.cpp matches each tensor name
// against the patterns in order (std::regex_search) and uses the first match. Cached: Compute runs many times per frame.
struct OverrideBytes { uint64_t cpu = 0, gpu = 0; };
const std::vector<std::array<OverrideBytes, kBlocks>>& OverrideMap(const ModelInfo& m, const LaunchParams& p) {
    static std::string key;
    static std::vector<std::array<OverrideBytes, kBlocks>> map;
    std::string k = m.path + "|";
    for (const auto& [pat, buf] : p.overrides) k += pat + "=" + buf + ";";
    if (k == key) return map;
    key = k;
    map.assign(m.layers.size(), {});
    std::vector<std::pair<std::regex, bool>> res;   // bool: CPU
    for (const auto& [pat, buf] : p.overrides) {
        try {
            res.push_back({std::regex(pat), buf.rfind("CPU", 0) == 0});
        } catch (const std::regex_error&) {
        }
    }
    if (res.empty()) return map;
    for (size_t i = 0; i < m.layers.size(); ++i)
        for (int b = 0; b < kBlocks; ++b)
            for (const auto& t : m.layers[i].tensors[b]) {
                const std::string name = "blk." + std::to_string(i) + "." + t.name;
                for (const auto& [re, cpu] : res)
                    if (std::regex_search(name, re)) {
                        (cpu ? map[i][b].cpu : map[i][b].gpu) += t.bytes;
                        break;
                    }
            }
    return map;
}

} // namespace

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
        if (p.cpuMoe || i < p.nCpuMoe) r.dev[i][(int)Block::Experts] = Device::Cpu;   // --cpu-moe / --n-cpu-moe
        if (!p.overrides.empty()) {
            // -ot: a block goes where most of its bytes are forced (rest follows the layer)
            const auto& ov = OverrideMap(m, p)[i];
            for (int b = 0; b < kBlocks; ++b) {
                const uint64_t total = m.layers[i].bytes[b];
                if (!total || (ov[b].cpu == 0 && ov[b].gpu == 0)) continue;
                const uint64_t rest = total - std::min(total, ov[b].cpu + ov[b].gpu);
                const uint64_t onCpu = ov[b].cpu + (r.dev[i][b] == Device::Cpu ? rest : 0);
                r.dev[i][b] = onCpu * 2 > total ? Device::Cpu : Device::Gpu;
            }
        }
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

std::vector<std::string> TokenizeArgs(const std::string& cmdline) {
    std::vector<std::string> tok;
    std::string cur;
    char quote = 0;
    bool in = false;
    for (char c : cmdline) {
        if (quote) {
            if (c == quote) quote = 0;
            else cur += c;
            continue;
        }
        if (c == '"' || c == '\'') { quote = c; in = true; continue; }
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '^' || c == '`') {
            if (in) { tok.push_back(cur); cur.clear(); in = false; }
            continue;
        }
        cur += c;
        in = true;
    }
    if (in) tok.push_back(cur);
    return tok;
}

ParsedArgs ParseLlamaArgs(const std::string& cmdline, LaunchParams& p) {
    ParsedArgs out;
    std::vector<std::string> tok = TokenizeArgs(cmdline);
    if (!tok.empty() && tok[0].find("llama") != std::string::npos && tok[0].find(".gguf") == std::string::npos)
        tok.erase(tok.begin());   // the executable

    auto num = [](const std::string& s, int& v) {
        try { v = std::stoi(s); return true; } catch (...) { return false; }
    };
    auto kvIndex = [](const std::string& s) {
        const auto& kv = KvTypes();
        for (int i = 0; i < (int)kv.size(); ++i)
            if (s == kv[i].name) return i;
        return -1;
    };
    static const std::vector<std::string> noEffect = {
        "-t", "--threads", "-tb", "--threads-batch", "-C", "--cpu-mask", "-Cr", "--cpu-range", "--cpu-strict", "--prio",
        "--poll", "-Cb", "--cpu-mask-batch", "-Crb", "--cpu-range-batch", "--cpu-strict-batch", "--prio-batch",
        "--poll-batch", "-fa", "--flash-attn", "--load-mode", "--mlock", "--no-mmap", "--mmap", "-n", "--n-predict",
        "-p", "--prompt", "-f", "--file", "-sys", "--system-prompt", "--temp", "--top-k", "--top-p", "--min-p",
        "--repeat-penalty", "--seed", "-s", "--jinja", "--chat-template", "--host", "--port", "-a", "--alias",
        "--log-file", "--verbose", "-v", "--color", "-cnv", "-no-cnv", "-st", "--single-turn", "-i", "--interactive",
        "--no-warmup", "--keep", "--reasoning-budget", "--reasoning-format", "-r", "--reverse-prompt", "--numa"};
    static const std::vector<std::string> noValue = {
        "--cpu-moe", "-cmoe", "--mlock", "--no-mmap", "--mmap", "--jinja", "--color", "-cnv", "-no-cnv", "-st",
        "--single-turn", "-i", "--interactive", "--no-warmup", "--verbose", "-v", "--cont-batching", "-cb",
        "--no-kv-offload", "-nkvo", "--no-op-offload"};
    static const std::vector<std::string> notModeled = {"-ts", "--tensor-split", "-sm", "--split-mode", "-mg",
                                                        "--main-gpu", "-dev", "--device", "--fit", "-nkvo",
                                                        "--no-kv-offload", "--no-op-offload", "-md", "--model-draft",
                                                        "-ngld", "--gpu-layers-draft", "--rpc"};
    auto has = [](const std::vector<std::string>& v, const std::string& s) { return std::find(v.begin(), v.end(), s) != v.end(); };

    for (size_t i = 0; i < tok.size(); ++i) {
        std::string f = tok[i], val;
        const size_t eq = f.find('=');
        if (f.rfind("--", 0) == 0 && eq != std::string::npos) { val = f.substr(eq + 1); f = f.substr(0, eq); }
        else if (!has(noValue, f) && i + 1 < tok.size() && (tok[i + 1].empty() || tok[i + 1][0] != '-' || (tok[i + 1].size() > 1 && isdigit((unsigned char)tok[i + 1][1])))) val = tok[++i];
        if (f.empty() || f[0] != '-') { out.unknown.push_back(f); continue; }
        int v = 0;
        const std::string shown = val.empty() ? f : f + " " + val;
        if (f == "-m" || f == "--model") { out.model = val; out.applied.push_back(shown); }
        else if ((f == "-ngl" || f == "--n-gpu-layers" || f == "--gpu-layers") && (val == "all" || val == "auto" || num(val, v))) {
            p.ngl = val == "all" || val == "auto" ? 999 : v; out.applied.push_back(shown);
        }
        else if ((f == "--n-cpu-moe" || f == "-ncmoe") && num(val, v)) { p.nCpuMoe = v; p.cpuMoe = false; out.applied.push_back(shown); }
        else if (f == "--cpu-moe" || f == "-cmoe") { p.cpuMoe = true; out.applied.push_back(shown); }
        else if ((f == "-c" || f == "--ctx-size") && num(val, v)) { p.nCtx = v; out.applied.push_back(shown); }
        else if ((f == "-b" || f == "--batch-size") && num(val, v)) { p.nBatch = v; out.applied.push_back(shown); }
        else if ((f == "-ub" || f == "--ubatch-size") && num(val, v)) { p.nUbatch = v; out.applied.push_back(shown); }
        else if ((f == "-np" || f == "--parallel") && num(val, v)) { p.nParallel = v; out.applied.push_back(shown); }
        else if ((f == "-ctk" || f == "--cache-type-k") && kvIndex(val) >= 0) { p.ctk = kvIndex(val); out.applied.push_back(shown); }
        else if ((f == "-ctv" || f == "--cache-type-v") && kvIndex(val) >= 0) { p.ctv = kvIndex(val); out.applied.push_back(shown); }
        else if (f == "-ot" || f == "--override-tensor") {
            // comma-separated pattern=buffer pairs; repeated flags add more
            size_t a = 0;
            while (a <= val.size()) {
                size_t b = val.find(',', a);
                std::string pair = val.substr(a, b == std::string::npos ? std::string::npos : b - a);
                size_t e = pair.rfind('=');
                if (e != std::string::npos) p.overrides.push_back({pair.substr(0, e), pair.substr(e + 1)});
                if (b == std::string::npos) break;
                a = b + 1;
            }
            out.applied.push_back(shown);
        }
        else if (has(notModeled, f)) out.notModeled.push_back(shown);
        else if (has(noEffect, f) || has(noValue, f)) out.noEffect.push_back(shown);
        else out.unknown.push_back(shown);
    }
    return out;
}
