/// @file cerberus_runtime.cpp
/// @copyright Copyright © 2026, all rights reserved | David Hargreaves aka Roylepython trading as The Medusa Initiative & Lamia Fabrica
/// Support: https://www.patreon.com/TheMedusaInitiative — £25/month removes ads from all software at 200 subscribers
///
/// CerberusRuntime — production runtime implementation.
///
/// @version 1.0.0

#include "hq/cerberus_runtime.hpp"
#include "hq/cerberus_native_backend.hpp"
#include "hq/cerberus_command_executor.hpp"
#include "hq/cerberus_glow_engine.hpp"
#include "hq/tiered_memory_manager.hpp"
#include "hq/cerberus_execution_coordinator.hpp"
#include "hq/cerberus_psiforcedb_security.hpp"

#include <string>
#include <sstream>
#include <filesystem>
#include <cstdlib>

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
    lcmd_diagnostic_ = cfg_.lcmd;
    if (!lcmd_diagnostic_) {
        try {
            auto default_lcmd = std::make_shared<hq::cerberus::privacy::LocalMaintenanceDB>();
            std::filesystem::path db_path;
            if (const char* local_appdata = std::getenv("LOCALAPPDATA")) {
                db_path = std::filesystem::path(local_appdata) / "Cerberus" / "local_maintenance.db";
            } else {
                db_path = std::filesystem::temp_directory_path() / "cerberus_local_maintenance.db";
            }
            std::filesystem::create_directories(db_path.parent_path());
            auto db_key = hq::cerberus::security::CryptoBridge::sha256(
                "cerberus_runtime_default_lcmd_key_v1");
            if (db_key.size() == 32 && default_lcmd->initialize(db_path, db_key)) {
                lcmd_diagnostic_ = std::move(default_lcmd);
            }
        } catch (...) {
            // leave lcmd_diagnostic_ null if default setup fails
        }
    }
    glow_engine_ = std::make_unique<GlowEngine>();
    executor_ = std::make_unique<cli::CerberusCommandExecutor>(*this);
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
    } else {
        // Try factory-registered backends (OpenVINO, CUDA, CPU-fallback)
        npu::NpuBackendFactory::initialize();
        npu::INpuBackend* found = npu::NpuBackendFactory::by_name(cfg_.preferred_backend);
        if (!found) {
            return std::unexpected{
                cfg_.preferred_backend + " backend not registered in NpuBackendFactory"};
        }
        if (!found->is_available()) {
            return std::unexpected{
                cfg_.preferred_backend + " unavailable: " + found->unavailable_reason()};
        }
        // Wrap in a delegating backend (lifetime managed by factory singleton)
        delegating_backend_ = found;
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
    npu::INpuBackend* active = backend_.get() ? backend_.get() : delegating_backend_;
    if (!active) {
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
    auto ck_r = active->compile(graph, tcfg);
    if (!ck_r) {
        return std::unexpected{
            "CerberusRuntime: backend compile failed: " + ck_r.error()};
    }
    const npu::CompiledKernel& ck = *ck_r;

    // 5. Stage inputs/outputs through the coordinator and execute
    auto coord_r = coordinator_->run(*active, ck,
                                      input_buffers, output_buffers);
    if (!coord_r) {
        return std::unexpected{
            "CerberusRuntime: coordinator run failed: " + coord_r.error()};
    }

    // 6. Learn from this execution via the GlowEngine
    if (glow_engine_) {
        std::vector<std::int32_t> node_path;
        node_path.reserve(cgraph.nodes.size());
        for (const auto& node : cgraph.nodes) {
            node_path.push_back(node.id);
        }
        glow_engine_->record_execution(node_path);
    }

    return {};
}

std::string CerberusRuntime::execute_command(const std::string& raw_command) {
    if (!executor_) {
        return "{\"success\":false,\"error\":\"Command executor not initialized\"}";
    }
    auto result = executor_->execute(raw_command);
    std::ostringstream oss;
    oss << "{\"success\":" << (result.success ? "true" : "false");
    if (result.success) {
        oss << ",\"output\":\"";
        // Escape double quotes in output to keep JSON valid
        for (char c : result.output) {
            if (c == '"') oss << "\\\"";
            else if (c == '\\') oss << "\\\\";
            else if (c == '\n') oss << "\\n";
            else oss << c;
        }
        oss << "\"";
    } else {
        oss << ",\"error\":\"";
        for (char c : result.error) {
            if (c == '"') oss << "\\\"";
            else if (c == '\\') oss << "\\\\";
            else if (c == '\n') oss << "\\n";
            else oss << c;
        }
        oss << "\"";
    }
    oss << ",\"exit_code\":" << result.exit_code;
    oss << ",\"execution_time_ms\":" << result.execution_time_ms << "}";
    return oss.str();
}

// Diagnostic accessors (must be inside namespace hq::cerberus)
TieredMemoryManager* CerberusRuntime::getMemoryManagerForDiagnostics() const {
    return mem_mgr_.get();
}

CerberusExecutionCoordinator* CerberusRuntime::getExecutionCoordinatorForDiagnostics() const {
    return coordinator_.get();
}

hq::cerberus::privacy::LocalMaintenanceDB* CerberusRuntime::getLcmdForDiagnostics() const {
    return lcmd_diagnostic_.get();
}

void CerberusRuntime::setLcmdForDiagnostics(std::shared_ptr<hq::cerberus::privacy::LocalMaintenanceDB> lcmd) {
    lcmd_diagnostic_ = std::move(lcmd);
}

} // namespace hq::cerberus
