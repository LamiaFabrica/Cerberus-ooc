#include <cstdio>
#include "hq/npu_backend_unified.hpp"

int main() {
    std::fprintf(stderr, "BEFORE_CTOR\n"); std::fflush(stderr);
    hq::npu::IntelOpenVinoBackend intel;
    std::fprintf(stderr, "AFTER_CTOR\n"); std::fflush(stderr);
    std::fprintf(stderr, "AVAILABLE=%d\n", intel.is_available()); std::fflush(stderr);
    if (!intel.is_available()) {
        std::fprintf(stderr, "REASON=%s\n", intel.unavailable_reason().c_str());
        std::fflush(stderr);
        return 1;
    }
    std::fprintf(stderr, "DO_COMPILE\n"); std::fflush(stderr);

    hq::npu::KernelGraph graph;
    graph.source_path = "smoke_test.onnx";
    graph.entry_point = "smoke_test";
    hq::npu::TargetConfig cfg;
    cfg.target_name = "intel_npu";
    cfg.output_dir = ".";

    auto ck = intel.compile(graph, cfg);
    if (!ck) {
        std::fprintf(stderr, "COMPILE_FAIL=%s\n", ck.error().c_str());
        std::fflush(stderr);
        return 1;
    }
    std::fprintf(stderr, "COMPILE_OK in=%zu out=%zu\n", ck->inputs.size(), ck->outputs.size());
    std::fflush(stderr);
    return 0;
}
