/// @file athenea_probe_runner.cpp
/// @brief Standalone runner for the Athenea NPU probe daily KPI check.
///
/// Usage: athenea_probe_runner.exe <path_to_gguf>
///
/// Initializes CerberusRuntime, executes npu:athenea-probe, captures JSON output.

#include "hq/cerberus_runtime.hpp"
#include "hq/cerberus_local_maintenance_db.hpp"
#include "hq/cerberus_user_security.hpp"
#include "hq/cxx26_features.hpp"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <filesystem>

extern "C" std::size_t hq_safe_write(int fd, const char* data, std::size_t len);

namespace {
    inline void hq_print(std::string_view msg) {
        hq_safe_write(1, msg.data(), msg.size());
    }
    inline void hq_println(std::string_view msg) {
        hq_safe_write(1, msg.data(), msg.size());
        hq_safe_write(1, "\n", 1);
    }
}

int main(int argc, char** argv) {
    std::string gguf_path;
    if (argc > 1) {
        gguf_path = argv[1];
    } else {
        gguf_path = "C:\\McMaker Projects\\Projects\\Athenea\\GGUF\\athenea-4b-coding-IQ4_NL.gguf";
    }

    if (!std::filesystem::exists(gguf_path)) {
        hq_println("ERROR: GGUF file not found: " + gguf_path);
        return 1;
    }

    hq::cerberus::CerberusRuntime::Config cfg;
    cfg.preferred_backend = "native";
    cfg.enable_fusion = true;
    cfg.enable_quantization = true;

    // Initialize LCMD for probe recording
    auto lcmd = std::make_shared<hq::cerberus::privacy::LocalMaintenanceDB>();
    std::filesystem::path lcmd_path = std::filesystem::temp_directory_path() / "cerberus_athenea_probe_lcmd.db";
    std::vector<uint8_t> key(32, 0xAB);
    if (lcmd->initialize(lcmd_path.string(), key)) {
        cfg.lcmd = lcmd;
    }

    hq::cerberus::CerberusRuntime rt(cfg);
    rt.setLcmdForDiagnostics(lcmd);

    std::string cmd = "cerberus://npu:athenea-probe::path:" + gguf_path + ";";
    hq_println("Executing: " + cmd);

    std::string result = rt.execute_command(cmd);
    hq_println(result);

    return 0;
}
