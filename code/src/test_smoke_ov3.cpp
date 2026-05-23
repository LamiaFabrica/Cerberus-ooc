#include <fstream>
#include "hq/npu_backend_unified.hpp"
#include <vector>
#include <cmath>

int main() {
    std::ofstream f("C:\\McMaker Projects\\Projects\\Cerberus - Copy\\code\\build\\smoke_debug_log3.txt", std::ios::trunc);
    auto wl = [&](const char* s) {
        if (f) f << s << "\n", f.flush();
    };
    wl("A_start");
    {
        hq::npu::IntelOpenVinoBackend intel;
        wl("B_ctor");
        if (!intel.is_available()) {
            wl("C_not_avail");
            wl(intel.unavailable_reason().c_str());
            return 1;
        }
        wl("D_avail");

        hq::npu::KernelGraph graph;
        graph.source_path = "smoke_test.onnx";
        graph.entry_point = "smoke_test";
        hq::npu::TargetConfig cfg;
        cfg.target_name = "intel_npu";
        cfg.output_dir = ".";

        wl("E_compile");
        auto ck = intel.compile(graph, cfg);
        if (!ck) {
            wl("F_compile_fail");
            wl(ck.error().c_str());
            return 1;
        }
        wl("G_compile_ok");

        if (ck->inputs.empty() || ck->outputs.empty()) {
            wl("H_empty_desc");
            return 1;
        }
        wl("H_desc_ok");

        std::vector<float> in_data(ck->inputs[0].size_bytes()/sizeof(float), 1.0f);
        std::vector<float> out_data(ck->outputs[0].size_bytes()/sizeof(float), 0.0f);
        const std::byte* ins[] = {reinterpret_cast<const std::byte*>(in_data.data())};
        std::byte* outs[] = {reinterpret_cast<std::byte*>(out_data.data())};

        wl("I_execute");
        auto er = intel.execute(*ck, std::span<const std::byte*>(ins), std::span<std::byte*>(outs));
        if (!er) {
            wl("J_exec_fail");
            wl(er.error().c_str());
            return 1;
        }
        wl("K_exec_ok");

        float expc[] = {3.0f,5.0f,7.0f,9.0f};
        for (size_t i=0;i<out_data.size();++i) {
            if (std::fabs(out_data[i]-expc[i]) > 1e-4f) {
                wl("L_verify_fail");
                return 1;
            }
        }
        wl("PASS");
        return 0;
    }
}
