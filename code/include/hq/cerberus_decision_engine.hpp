#pragma once
/// @file cerberus_decision_engine.hpp
/// @copyright Copyright (c) 2026 D Hargreaves. All rights reserved.
///
/// Cerberus Decision Engine — routes graph nodes to backends and memory tiers.
///
/// This is the "interlaced decision engine". It inspects the Cerberus-owned
/// graph and decides, per node or per fused subgraph:
///   * Which execution backend (native CPU, OpenVINO, CUDA, etc.)
///   * Which memory tier (Hot/Warm/Cool/Cold) for outputs
///   * Whether to fuse adjacent simple ops into a single native kernel
///
/// The engine is pressure-aware: it reads TieredMemoryManager stats and will
/// demote allocations to cooler tiers when memory pressure is high.
///
/// @version 1.0.0

#include "hq/cerberus_graph_engine.hpp"
#include "hq/cerberus_native_kernels.hpp"
#include "hq/tiered_memory_manager.hpp"

#include <vector>
#include <string>
#include <span>
#include <memory>

namespace hq::cerberus {

// ===========================================================================
// DecisionConfig — standalone to avoid GCC nested-class initializer ordering
// ===========================================================================

struct DecisionConfig {
    std::size_t native_elem_threshold{1024};       ///< <= this many floats => native
    std::size_t matmul_native_max_mnk{64};         ///< MatMul native only if all dims <= this
    float       warm_tier_pressure_limit{0.75f};     ///< above this, avoid Warm
    bool        fuse_elementwise{true};             ///< fuse chains of Add/Mul
};

// ===========================================================================
// ExecutionStep — one unit of work produced by the decision engine
// ===========================================================================

struct ExecutionStep {
    enum class Backend : std::uint8_t {
        Native = 0,   ///< Cerberus-owned C++ kernels (CPU)
        OpenVINO,     ///< Intel NPU / CPU via OpenVINO
        CUDA,         ///< NVIDIA GPU
        FusedNative,  ///< fused subgraph emitted as a single native kernel
    };

    std::vector<std::int32_t> node_ids;   ///< graph nodes in this step
    Backend backend{Backend::Native};
    MemoryTier preferred_tier{MemoryTier::Cool};
    std::string debug_label;               ///< human-readable routing reason
};

// ===========================================================================
// DecisionEngine
// ===========================================================================

class DecisionEngine {
public:
    explicit DecisionEngine(TieredMemoryManager& mem_mgr,
                           const DecisionConfig& cfg = DecisionConfig{});

    /// Analyse the graph and produce an ordered execution plan.
    /// Modifies graph node routing fields in-place.
    [[nodiscard]] std::vector<ExecutionStep>
    analyse(CerberusGraph& graph, std::string_view target_name);

    /// Re-evaluate a single step under current memory pressure (can demote tier).
    void reevaluate_tier(ExecutionStep& step) const;

private:
    TieredMemoryManager& mem_mgr_;
    DecisionConfig cfg_;

    bool all_elementwise(const CerberusGraph& g,
                         std::span<const std::int32_t> ids) const noexcept;
    bool is_small_enough_for_native(const GraphNode& node,
                                    std::span<const GraphTensor> tensors) const noexcept;
    ExecutionStep::Backend pick_backend(const CerberusGraph& g, std::int32_t node_id) const;
    void fuse_elementwise_chain(CerberusGraph& g,
                                std::vector<ExecutionStep>& plan) const;
};

} // namespace hq::cerberus
