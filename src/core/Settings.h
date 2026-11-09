#pragma once
#include <string>
#include "model/Topology.h"

// Persistent user settings in %LOCALAPPDATA%\LlamaScope\settings.json.
struct Settings {
    std::string modelPath;
    LaunchParams launch;

    void Load();
    void Save() const;
};
