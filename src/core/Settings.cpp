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
    struct EnvDefaults {
        Settings* s;
        ~EnvDefaults() {
            auto env = [](const char* n) { char* v = nullptr; size_t len = 0; std::string r; if (_dupenv_s(&v, &len, n) == 0 && v) { r = v; free(v); } return r; };
            if (s->profBin.empty()) s->profBin = env("LLAMACPP_PROF_BIN");
            if (s->cudaBin.empty()) s->cudaBin = env("CUDA_BIN");
        }
    } envDefaults{this};
    std::ifstream f(SettingsPath());
    if (!f) return;
    try {
        json j = json::parse(f);
        modelPath = j.value("modelPath", "");
        profBin = j.value("profBin", "");
        cudaBin = j.value("cudaBin", "");
        profMode = j.value("profMode", profMode);
        auto l = j.value("launch", json::object());
        launch.ngl = l.value("ngl", launch.ngl);
        launch.nCpuMoe = l.value("nCpuMoe", launch.nCpuMoe);
        launch.nCtx = l.value("nCtx", launch.nCtx);
        launch.nUbatch = l.value("nUbatch", launch.nUbatch);
        launch.nParallel = l.value("nParallel", launch.nParallel);
        launch.ctk = l.value("ctk", launch.ctk);
        launch.ctv = l.value("ctv", launch.ctv);
        launch.cpuMoe = l.value("cpuMoe", launch.cpuMoe);
        launch.nBatch = l.value("nBatch", launch.nBatch);
        launch.cmdline = l.value("cmdline", launch.cmdline);
        launch.overrides.clear();
        for (const auto& o : l.value("overrides", json::array()))
            if (o.is_array() && o.size() == 2) launch.overrides.push_back({o[0].get<std::string>(), o[1].get<std::string>()});
    } catch (std::exception& e) {
        Log::W(std::string("settings.json ignored: ") + e.what());
    }
}

void Settings::Save() const {
    json j;
    j["modelPath"] = modelPath;
    j["profBin"] = profBin;
    j["cudaBin"] = cudaBin;
    j["profMode"] = profMode;
    j["launch"] = {{"ngl", launch.ngl},         {"nCpuMoe", launch.nCpuMoe}, {"nCtx", launch.nCtx},
                   {"nUbatch", launch.nUbatch}, {"nParallel", launch.nParallel}, {"ctk", launch.ctk},
                   {"ctv", launch.ctv},         {"cpuMoe", launch.cpuMoe},   {"nBatch", launch.nBatch},
                   {"cmdline", launch.cmdline}, {"overrides", launch.overrides}};
    std::ofstream f(SettingsPath());
    f << j.dump(2);
}
