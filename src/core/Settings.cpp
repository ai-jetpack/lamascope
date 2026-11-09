#include <fstream>
#include "core/Log.h"
#include "core/Settings.h"
#include "core/Util.h"
#include "json.hpp"

using nlohmann::json;

namespace {
std::wstring SettingsPath() { return LocalDataDir() + L"\\settings.json"; }
}

void Settings::Load() {
    std::ifstream f(SettingsPath());
    if (!f) return;
    try {
        json j = json::parse(f);
        modelPath = j.value("modelPath", "");
        auto l = j.value("launch", json::object());
        launch.ngl = l.value("ngl", launch.ngl);
        launch.nCpuMoe = l.value("nCpuMoe", launch.nCpuMoe);
        launch.nCtx = l.value("nCtx", launch.nCtx);
        launch.nUbatch = l.value("nUbatch", launch.nUbatch);
        launch.nParallel = l.value("nParallel", launch.nParallel);
        launch.ctk = l.value("ctk", launch.ctk);
        launch.ctv = l.value("ctv", launch.ctv);
    } catch (std::exception& e) {
        Log::W(std::string("settings.json ignored: ") + e.what());
    }
}

void Settings::Save() const {
    json j;
    j["modelPath"] = modelPath;
    j["launch"] = {{"ngl", launch.ngl},         {"nCpuMoe", launch.nCpuMoe}, {"nCtx", launch.nCtx},
                   {"nUbatch", launch.nUbatch}, {"nParallel", launch.nParallel}, {"ctk", launch.ctk},
                   {"ctv", launch.ctv}};
    std::ofstream f(SettingsPath());
    f << j.dump(2);
}
