#pragma once
#include <string>
#include "model/Topology.h"

// Persistent user settings in %LOCALAPPDATA%\LlamaScope\settings.json.
struct Settings {
    std::string modelPath;
    LaunchParams launch;
    std::string profBin;            // bin folder of the profiler-enabled llama.cpp build (default: %LLAMACPP_PROF_BIN%)
    std::string cudaBin;            // CUDA runtime DLLs, if not on PATH (default: %CUDA_BIN%)
    std::string profMode = "block"; // GGML_PROFILE: block | op

    void Load();
    void Save() const;
};
