/// @file c_api_propup_runner.cpp
/// @brief Standalone runner for the 4 C API propup tests.

#include "hq/david_propup_engine.hpp"
#include "hq/cerberus_api_wrapper.hpp"

#include <cstdio>
#include <cstdlib>

extern "C" std::size_t hq_safe_write(int fd, const char* data, std::size_t len);

namespace {
    inline void hq_print(std::string msg) {
        hq_safe_write(1, msg.data(), msg.size());
    }
    inline void hq_println(std::string msg) {
        hq_safe_write(1, msg.data(), msg.size());
        hq_safe_write(1, "\n", 1);
    }
}

int main() {
    int passed = 0;
    int failed = 0;

    auto run = [&](const char* name, auto fn) {
        hq_println(std::string("[RUN] ") + name);
        try {
            auto r = fn(nullptr);
            if (r.passed) {
                ++passed;
                hq_println(std::string("[PASS] ") + name + " — " + std::to_string(r.elapsed_ms) + " ms");
            } else if (r.skipped) {
                hq_println(std::string("[SKIP] ") + name + " — " + r.diagnostic);
            } else {
                ++failed;
                hq_println(std::string("[FAIL] ") + name + " — " + r.diagnostic);
            }
        } catch (const std::exception& e) {
            ++failed;
            hq_println(std::string("[FAIL] ") + name + " — exception: " + e.what());
        } catch (...) {
            ++failed;
            hq_println(std::string("[FAIL] ") + name + " — unknown exception");
        }
    };

    run("propup_c_api_init_shutdown_cycle", hq::propup::propup_c_api_init_shutdown_cycle);
    run("propup_c_api_load_model_rejects_invalid_path", hq::propup::propup_c_api_load_model_rejects_invalid_path);
    run("propup_c_api_run_inference_rejects_null_handle", hq::propup::propup_c_api_run_inference_rejects_null_handle);
    run("propup_c_api_get_last_error_consistent", hq::propup::propup_c_api_get_last_error_consistent);

    hq_println("");
    hq_println(std::string("=== C API Propup Results: ") + std::to_string(passed) + " passed, " + std::to_string(failed) + " failed ===");
    return failed > 0 ? 1 : 0;
}
