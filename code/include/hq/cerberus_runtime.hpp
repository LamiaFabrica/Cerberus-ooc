#pragma once
/// @file cerberus_runtime.hpp
/// @copyright Copyright (c) 2026 D Hargreaves. All rights reserved.
///
/// CerberusRuntime — production runtime entry point for the full engine stack.
///
/// This is the bridge between the C API and the Cerberus compiler+runtime.
/// It owns the full execution path:
///   KernelGraph → CerberusGraph → DecisionEngine → ExecutionCoordinator → Backend
///
/// @version 1.0.0

#include "hq/npu_backend_unified.hpp"
#include "hq/tiered_memory_manager.hpp"
#include "hq/cerberus_execution_coordinator.hpp"
#include "hq/cerberus_graph_engine.hpp"
#include "hq/cerberus_decision_engine.hpp"
#include "hq/cerberus_glow_engine.hpp"
#include "hq/cerberus_local_maintenance_db.hpp"
#include "hq/cerberus_error.hpp"

#include <expected>
#include <string>
#include <memory>
#include <vector>
#include <span>

namespace hq::cerberus::cli {
class CerberusCommandExecutor;
}

namespace hq::cerberus {

// ===========================================================================
// CerberusRuntime — production runtime wrapper
// ===========================================================================

class CerberusRuntime {
public:
    struct Config {
        std::string preferred_backend{"native"}; ///< "native", "openvino", "cuda", "cpu"
        bool        enable_fusion{true};         ///< enable Mul+Add → FMA fusion
        bool        enable_quantization{false};  ///< enable INT8 path where possible
        std::size_t warm_capacity_bytes{1ULL << 30}; ///< 1 GiB
        std::size_t cool_capacity_bytes{512ULL << 20}; ///< 512 MiB
        /// Optional LocalMaintenanceDB (LCMD) — when provided, the runtime owns
        /// the reference and diagnostic accessors return the real instance.
        /// Required for Athenea probe endurance path and inference audit endpoints.
        std::shared_ptr<hq::cerberus::privacy::LocalMaintenanceDB> lcmd;
    };

    explicit CerberusRuntime();
    explicit CerberusRuntime(const Config& cfg);
    ~CerberusRuntime();

    // Non-copyable, movable
    CerberusRuntime(const CerberusRuntime&) = delete;
    CerberusRuntime& operator=(const CerberusRuntime&) = delete;
    CerberusRuntime(CerberusRuntime&&) noexcept;
    CerberusRuntime& operator=(CerberusRuntime&&) noexcept;

    /// Compile and execute a KernelGraph through the full Cerberus stack.
    /// @param graph  Input computational graph.
    /// @param input_buffers  Pointers to input host buffers (size = graph.graph_inputs.size()).
    /// @param output_buffers Pointers to output host buffers (size = graph.graph_outputs.size()).
    /// @return CERBERUS_OK on success, or an error string.
    [[nodiscard]] hq::ExpectedVoid
    run_graph(const npu::KernelGraph& graph,
              std::span<std::byte*>       output_buffers,
              std::span<const std::byte*> input_buffers);

    /// Return last execution plan (for diagnostics / profiling).
    [[nodiscard]] const std::vector<ExecutionStep>& last_plan() const noexcept {
        return last_plan_;
    }

    /// Return runtime name for C API compatibility.
    [[nodiscard]] static const char* name() noexcept { return "CerberusRuntime-v1.0"; }

    /// Execute a Cerberus command string through the command layer.
    /// @param raw_command  A command in protocol form (cerberus:// or cbr://) or ergonomic shorthand.
    /// @return JSON-like output string, or an error message.
    [[nodiscard]] std::string execute_command(const std::string& raw_command);

    /// Access the GlowEngine for hot-path learning and query.
    [[nodiscard]] GlowEngine* glow_engine() noexcept { return glow_engine_.get(); }
    [[nodiscard]] const GlowEngine* glow_engine() const noexcept { return glow_engine_.get(); }

private:
    Config cfg_;
    std::unique_ptr<TieredMemoryManager>    mem_mgr_;
    std::unique_ptr<DecisionEngine>       decision_engine_;
    std::unique_ptr<CerberusExecutionCoordinator> coordinator_;
    std::unique_ptr<npu::INpuBackend>       backend_;
    npu::INpuBackend*                       delegating_backend_{nullptr};
    std::unique_ptr<GlowEngine>             glow_engine_;

public:
    // Diagnostic accessors (for npu:athenea-probe and similar production-grade tools only)
    // These allow the Athenea endurance tool to exercise the *real* runtime memory loop
    // instead of a throwaway local TMM.
    [[nodiscard]] TieredMemoryManager* getMemoryManagerForDiagnostics() const;
    [[nodiscard]] CerberusExecutionCoordinator* getExecutionCoordinatorForDiagnostics() const;
    // Real LCMD (never throwaway/hardcoded path in handlers; passed via runtime for innovative audit path)
    [[nodiscard]] hq::cerberus::privacy::LocalMaintenanceDB* getLcmdForDiagnostics() const;
    void setLcmdForDiagnostics(std::shared_ptr<hq::cerberus::privacy::LocalMaintenanceDB> lcmd);

    std::vector<ExecutionStep> last_plan_;
    std::unique_ptr<cli::CerberusCommandExecutor> executor_;
    std::shared_ptr<hq::cerberus::privacy::LocalMaintenanceDB> lcmd_diagnostic_;  // real LCMD passed in (no throwaway creation in athenea-probe handler)

    [[nodiscard]] hq::ExpectedVoid init_backend_();
};

} // namespace hq::cerberus
