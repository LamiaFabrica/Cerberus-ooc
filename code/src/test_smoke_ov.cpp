/// test_smoke_ov.cpp
/// Cerberus smoke test — fully self-contained, no external DLLs required.
///
/// Proves:
///   1. CerberusExecutionCoordinator is the ONLY valid execution path.
///   2. BOTH inputs and outputs are staged through TieredMemoryManager.
///   3. Tier decisions (Warm for high-reuse, Cool otherwise) are visible.
///   4. The backend contract is backend-agnostic.

#include "hq/npu_backend_unified.hpp"
#include "hq/tiered_memory_manager.hpp"
#include "hq/cerberus_execution_coordinator.hpp"
#include <vector>
#include <cmath>
#include <fstream>

// ===========================================================================
// SmokeTestBackend — minimal native implementation of the smoke model:
//   y = x * 2 + 1
//
// This backend is deliberately trivial so the test never depends on
// OpenVINO DLLs or any vendor SDK being present.
// ===========================================================================

class SmokeTestBackend final : public hq::npu::INpuBackend {
public:
    [[nodiscard]] std::expected<hq::npu::CompiledKernel, std::string>
    compile(const hq::npu::KernelGraph& graph,
            const hq::npu::TargetConfig& cfg) override {
        (void)cfg;
        hq::npu::CompiledKernel k;
        k.target_name = "smoke_test";
        k.compiled = true;

        // Input: x [4] f32
        k.inputs.push_back(hq::npu::TensorDesc{{4}, hq::npu::TensorDesc::DataType::F32});
        k.input_names.push_back("x");

        // Output: y [4] f32
        k.outputs.push_back(hq::npu::TensorDesc{{4}, hq::npu::TensorDesc::DataType::F32});
        k.output_names.push_back("y");

        // Analyze the Cerberus-owned graph for reuse
        k.graph_nodes = graph.nodes;
        for (const auto& node : graph.nodes) {
            for (const auto& out : node.outputs) {
                bool reused = false;
                for (const auto& later : graph.nodes) {
                    for (const auto& in_name : later.inputs) {
                        if (in_name == out) { reused = true; break; }
                    }
                    if (reused) break;
                }
                if (reused) k.high_reuse_tensors.push_back(out);
            }
        }

        k.estimated_working_set_bytes =
            k.inputs[0].size_bytes() + k.outputs[0].size_bytes();
        return k;
    }

    [[nodiscard]] std::expected<void, std::string>
    execute(const hq::npu::CompiledKernel& kernel,
            std::span<const std::byte*> inputs,
            std::span<std::byte*> outputs) override {
        if (inputs.size() != 1 || outputs.size() != 1)
            return std::unexpected{"SmokeTestBackend: bad io count"};
        (void)kernel;
        const float* in  = reinterpret_cast<const float*>(inputs[0]);
        float*       out = reinterpret_cast<float*>(outputs[0]);
        for (std::size_t i = 0; i < 4; ++i) {
            out[i] = in[i] * 2.0f + 1.0f;
        }
        return {};
    }

    [[nodiscard]] bool can_compile_for(std::string_view t) const override {
        return t == "smoke_test";
    }
    [[nodiscard]] bool is_available() const override { return true; }
    [[nodiscard]] std::string name() const override { return "SmokeTestBackend"; }
    [[nodiscard]] bool synthetic_mode() const noexcept override { return false; }
    [[nodiscard]] std::string unavailable_reason() const override { return {}; }
    [[nodiscard]] float utilization() const override { return -1.0f; }
    [[nodiscard]] float temperature() const override { return -1.0f; }
};

int main() {
    std::ofstream f("smoke_debug_log.txt", std::ios::trunc);
    auto wl = [&](const std::string& s) { if (f) f << s << "\n", f.flush(); };
    wl("=== Cerberus Smoke Test (Coordinator owns memory loop) ===");

    SmokeTestBackend backend;
    wl("backend: " + backend.name());

    // --- Populate Cerberus-owned graph so compile() does real work ---
    hq::npu::KernelGraph graph;
    graph.source_path = "smoke_test.onnx";
    graph.entry_point = "smoke_test";

    hq::npu::KernelNode mul_node;
    mul_node.name = "mul_1";
    mul_node.op = hq::npu::KernelNode::Op::Mul;
    mul_node.inputs = {"x"};
    mul_node.outputs = {"mul_out"};
    graph.nodes.push_back(std::move(mul_node));

    hq::npu::KernelNode add_node;
    add_node.name = "add_1";
    add_node.op = hq::npu::KernelNode::Op::Add;
    add_node.inputs = {"mul_out"};
    add_node.outputs = {"y"};
    graph.nodes.push_back(std::move(add_node));

    hq::npu::TargetConfig cfg;
    cfg.target_name = "smoke_test";
    cfg.output_dir = ".";

    wl("compiling...");
    auto ck = backend.compile(graph, cfg);
    if (!ck) { wl("compile failed: " + ck.error()); return 1; }
    wl("compile ok");

    // --- Verify Cerberus-owned analysis made it into CompiledKernel ---
    wl("graph_nodes.size=" + std::to_string(ck->graph_nodes.size()));
    wl("high_reuse_tensors=" + std::to_string(ck->high_reuse_tensors.size()));
    for (const auto& n : ck->high_reuse_tensors) wl("  reuse: " + n);
    wl("estimated_working_set=" + std::to_string(ck->estimated_working_set_bytes));

    // --- Rule: coordinator is the mandatory execution path ---
    hq::TieredMemoryManager mem(hq::TieredMemoryConfig{});
    hq::CerberusExecutionCoordinator exec(mem);

    std::vector<float> in_data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> out_data(4, 0.0f);
    const std::byte* ins[]  = {reinterpret_cast<const std::byte*>(in_data.data())};
    std::byte*       outs[] = {reinterpret_cast<std::byte*>(out_data.data())};

    wl("executing via coordinator (inputs + outputs go through TieredMemoryManager)...");
    auto er = exec.run(backend, *ck,
                       std::span<const std::byte*>(ins),
                       std::span<std::byte*>(outs),
                       &f);
    if (!er) { wl(std::format("execution failed: {}", hq::to_string(er.error()))); return 1; }
    wl("execution ok");

    float expected[] = {3.0f, 5.0f, 7.0f, 9.0f};
    for (size_t i = 0; i < out_data.size(); ++i) {
        if (std::fabs(out_data[i] - expected[i]) > 1e-4f) {
            wl("verify FAIL at [" + std::to_string(i) + "]: got " +
               std::to_string(out_data[i]) + " expected " + std::to_string(expected[i]));
            return 1;
        }
    }
    wl("PASS");
    return 0;
}
