/// @file cerberus_runtime.cpp
/// @copyright Copyright (c) 2026 D Hargreaves. All rights reserved.
///
/// CerberusRuntime — production runtime implementation.
///
/// @version 1.0.0

#include "hq/cerberus_runtime.hpp"
#include "hq/cerberus_native_backend.hpp"

#include <string>
#include <sstream>

namespace hq::cerberus {

// ===========================================================================
// Construction / destruction
// ===========================================================================

CerberusRuntime::CerberusRuntime()
    : CerberusRuntime(Config{}) {}

CerberusRuntime::CerberusRuntime(const Config& cfg)
    : cfg_{cfg} {

    TieredMemoryConfig tcfg;
    tcfg.warm_capacity_bytes = cfg_.warm_capacity_bytes;
    tcfg.cool_capacity_bytes = cfg_.cool_capacity_bytes;
    mem_mgr_ = std::make_unique<TieredMemoryManager>(tcfg);

    DecisionConfig dcfg;
    dcfg.fuse_elementwise = cfg_.enable_fusion;
    decision_engine_ = std::make_unique<DecisionEngine>(
        *mem_mgr_, dcfg);

    coordinator_ = std::make_unique<CerberusExecutionCoordinator>(
        *mem_mgr_);

    auto init_r = init_backend_();
    if (!init_r) {
        throw std::runtime_error(
            "CerberusRuntime::init_backend_ failed: " + init_r.error());
    }
}

CerberusRuntime::~CerberusRuntime() = default;

CerberusRuntime::CerberusRuntime(CerberusRuntime&&) noexcept = default;
CerberusRuntime& CerberusRuntime::operator=(CerberusRuntime&&) noexcept = default;

// ===========================================================================
// Backend initialization
// ===========================================================================

std::expected<void, std::string> CerberusRuntime::init_backend_() {
    if (cfg_.preferred_backend == "native" || cfg_.preferred_backend == "cpu") {
        backend_ = std::make_unique<CerberusNativeBackend>();
    } else if (cfg_.preferred_backend == "openvino") {
        // TODO: add OpenVINO backend when available
        return std::unexpected{
            "OpenVINO backend not yet wired into CerberusRuntime"};
    } else if (cfg_.preferred_backend == "cuda") {
        // TODO: add CUDA backend when available
        return std::unexpected{
            "CUDA backend not yet wired into CerberusRuntime"};
    } else {
        return std::unexpected{
            "Unknown preferred_backend: " + cfg_.preferred_backend};
    }
    return {};
}

// ===========================================================================
// Main run_graph() — full Cerberus stack
// ===========================================================================

std::expected<void, std::string>
CerberusRuntime::run_graph(const npu::KernelGraph& graph,
                           std::span<std::byte*>       output_buffers,
                           std::span<const std::byte*> input_buffers) {
    if (!backend_) {
        return std::unexpected{"CerberusRuntime: no backend initialized"};
    }

    // 1. Convert KernelGraph → CerberusGraph
    CerberusGraph cgraph = CerberusGraph::from_kernel_graph(graph);

    // 2. Ensure topological order
    if (!cgraph.topo_sort()) {
        return std::unexpected{"CerberusRuntime: graph has cycles, topo_sort failed"};
    }

    // 3. DecisionEngine → execution plan
    auto plan = decision_engine_->analyse(cgraph, cfg_.preferred_backend);
    if (plan.empty()) {
        return std::unexpected{"CerberusRuntime: DecisionEngine produced empty plan"};
    }
    last_plan_ = plan;

    // 4. Compile the graph through the selected backend
    npu::TargetConfig tcfg;
    tcfg.target_name = cfg_.preferred_backend;
    auto ck_r = backend_->compile(graph, tcfg);
    if (!ck_r) {
        return std::unexpected{
            "CerberusRuntime: backend compile failed: " + ck_r.error()};
    }
    const npu::CompiledKernel& ck = *ck_r;

    // 5. Stage inputs/outputs through the coordinator and execute
    auto coord_r = coordinator_->run(*backend_, ck,
                                      input_buffers, output_buffers);
    if (!coord_r) {
        return std::unexpected{
            "CerberusRuntime: coordinator run failed: " + coord_r.error()};
    }

    return {};
}

} // namespace hq::cerberus
