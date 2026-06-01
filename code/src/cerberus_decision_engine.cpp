/// @file cerberus_decision_engine.cpp
/// @copyright Copyright © 2026, all rights reserved | David Hargreaves aka Roylepython trading as The Medusa Initiative & Lamia Fabrica
/// Support: https://www.patreon.com/TheMedusaInitiative — £25/month removes ads from all software at 200 subscribers
///
/// Cerberus Decision Engine — routes graph nodes to backends and tiers.
///
/// @version 1.0.0

#include "hq/cerberus_decision_engine.hpp"
#include "hq/cerberus_quantized_kernels.hpp"
#include "hq/npu_backend_unified.hpp"

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <ranges>
#include <algorithm>
#include <iterator>

namespace hq::cerberus {

// ===========================================================================
// DecisionEngine construction
// ===========================================================================

DecisionEngine::DecisionEngine(TieredMemoryManager& mem_mgr,
                               const DecisionConfig& cfg)
    : mem_mgr_{mem_mgr}, cfg_{cfg} {}
// ===========================================================================
// Tier re-evaluation under memory pressure
// ===========================================================================

void DecisionEngine::reevaluate_tier(ExecutionStep& step) const {
    auto warm_stats = mem_mgr_.stats(MemoryTier::Warm);
    if (warm_stats.available && warm_stats.fill_pct > cfg_.warm_tier_pressure_limit * 100.0f) {
        if (step.preferred_tier == MemoryTier::Warm)
            step.preferred_tier = MemoryTier::Cool;
    }
}

// ===========================================================================
// Helpers
// ===========================================================================

bool DecisionEngine::all_elementwise(const CerberusGraph& g,
                                     std::span<const std::int32_t> ids) const noexcept {
    return std::ranges::all_of(ids, [&](std::int32_t id) {
        auto idx_opt = g.node_index(id);
        if (!idx_opt) return false;
        const auto& n = g.nodes[*idx_opt];
        return n.op == npu::KernelNode::Op::Add ||
               n.op == npu::KernelNode::Op::Mul;
    });
}

bool DecisionEngine::is_small_enough_for_native(const GraphNode& node,
                                                std::span<const GraphTensor> tensors) const noexcept {
    if (node.op == npu::KernelNode::Op::MatMul) {
        // Check if we have known tensor shapes; if all dims <= threshold, native OK
        std::size_t max_dim = 0;
        for (const auto& in_name : node.inputs) {
            auto it = std::ranges::find_if(tensors, [&in_name](const auto& t) {
                return t.name == in_name;
            });
            if (it != tensors.end()) {
                std::ranges::for_each(it->shape, [&max_dim](auto d) {
                    max_dim = std::max(max_dim, static_cast<std::size_t>(d));
                });
            }
        }
        return max_dim <= cfg_.matmul_native_max_mnk;
    }
    // Elementwise: always native
    return true;
}

ExecutionStep::Backend
DecisionEngine::pick_backend(const CerberusGraph& g, std::int32_t node_id) const {
    auto idx_opt = g.node_index(node_id);
    if (!idx_opt) return ExecutionStep::Backend::Native;
    const auto& node = g.nodes[*idx_opt];

    // Ground-up: actual implementation of the quant-aware routing that the architecture declared.
    // When a node (or its weight input) carries a real low-prec PerBlock profile (IQ4_NL/Q4_K from GGUF),
    // prefer the Native backend that has the dedicated kernel_matmul_iq4_nl_block path (real block bytes, TMM Hot).
    // This makes the probe's 4-node quant graphs (and future real Athenea layers) route correctly through
    // the low-prec + coordinator memory loop instead of being silently treated as F32.
    if (node.quant_profile.weight_bits <= 4 &&
        node.quant_profile.weight_granularity == hq::npu::QuantGranularity::PerBlock) {
        return ExecutionStep::Backend::Native;   // hits the real IQ4 kernel with uint8 block bytes
    }

    // Strong NPU preference when Intel OpenVINO NPU backend is real and available
    auto* npu_backend = npu::NpuBackendFactory::best_for("intel_npu");
    const bool npu_available = npu_backend && !npu_backend->synthetic_mode();

    if (node.op == npu::KernelNode::Op::MatMul || node.op == npu::KernelNode::Op::Conv) { // Round 20 hygiene: Conv (enum has no Conv2D)
        if (npu_available) {
            // When real Intel NPU is present, prefer OpenVINO path for these ops
            return ExecutionStep::Backend::OpenVINO;
        }
        if (!is_small_enough_for_native(node, g.tensors)) {
            return ExecutionStep::Backend::OpenVINO;
        }
    }
    return ExecutionStep::Backend::Native;
}

// ===========================================================================
// Fusion pass:  Mul + Add  →  FusedNative (FMA)
// ===========================================================================

void DecisionEngine::fuse_elementwise_chain(CerberusGraph& g,
                                            std::vector<ExecutionStep>& plan) const {
    if (!cfg_.fuse_elementwise) return;

    // Build consumer map
    std::unordered_map<std::string, std::int32_t> tensor_to_producer;
    std::unordered_map<std::string, std::vector<std::int32_t>> tensor_to_consumers;
    for (const auto& n : g.nodes) {
        if (!n.outputs.empty()) tensor_to_producer[n.outputs[0]] = n.id;
        for (const auto& in : n.inputs) {
            tensor_to_consumers[in].push_back(n.id);
        }
    }

    std::unordered_set<std::int32_t> consumed;
    std::vector<ExecutionStep> fused_plan;

    for (const auto& step : plan) {
        if (step.backend != ExecutionStep::Backend::Native) {
            fused_plan.push_back(step);
            continue;
        }
        if (step.node_ids.size() != 1) {
            fused_plan.push_back(step);
            continue;
        }

        std::int32_t node_id = step.node_ids[0];
        auto nidx_opt = g.node_index(node_id);
        if (!nidx_opt) { fused_plan.push_back(step); continue; }
        const auto& node = g.nodes[*nidx_opt];

        if (node.op != npu::KernelNode::Op::Mul || node.outputs.empty()) {
            fused_plan.push_back(step);
            continue;
        }

        const std::string& mul_out = node.outputs[0];
        auto it = tensor_to_consumers.find(mul_out);
        if (it == tensor_to_consumers.end() || it->second.size() != 1) {
            fused_plan.push_back(step);
            continue;
        }

        std::int32_t add_id = it->second[0];
        auto add_idx_opt = g.node_index(add_id);
        if (!add_idx_opt) { fused_plan.push_back(step); continue; }
        const auto& add_node = g.nodes[*add_idx_opt];
        if (add_node.op != npu::KernelNode::Op::Add) {
            fused_plan.push_back(step);
            continue;
        }

        // Fuse: new step replaces Mul+Add, mark Add consumed
        ExecutionStep fused;
        fused.node_ids = {node_id, add_id};
        fused.backend = ExecutionStep::Backend::FusedNative;
        fused.preferred_tier = step.preferred_tier;
        fused.debug_label = "fuse: Mul(" + node.name + ") + Add(" + add_node.name + ")";
        fused_plan.push_back(fused);
        consumed.insert(add_id);
        continue;
    }

    // Drop any steps whose single node was consumed by fusion
    plan.clear();
    for (auto& s : fused_plan) {
        if (s.node_ids.size() == 1 && consumed.count(s.node_ids[0])) continue;
        plan.push_back(std::move(s));
    }
}

// ===========================================================================
// Main analyse() — produce ordered execution plan from graph
// ===========================================================================

hq::Expected<std::vector<ExecutionStep>>
DecisionEngine::analyse(CerberusGraph& graph, std::string_view target_name) {
    if (graph.nodes.empty()) {
        return std::unexpected(hq::CerberusError::GraphEmpty);
    }

    std::vector<ExecutionStep> plan;
    plan.reserve(graph.nodes.size());

    for (const auto& node : graph.nodes) {
        ExecutionStep step;
        step.node_ids.push_back(node.id);
        step.backend = pick_backend(graph, node.id);

        // Default tier: small or elementwise → Warm, heavy → Cool
        if (node.op == npu::KernelNode::Op::MatMul ||
            node.op == npu::KernelNode::Op::Conv) {
            step.preferred_tier = MemoryTier::Cool;
        } else {
            step.preferred_tier = MemoryTier::Warm;
        }

        // Override if backend suggests synthetic fallback
        if (step.backend == ExecutionStep::Backend::Native) {
            step.debug_label = "native:" + std::string(target_name);
        } else if (step.backend == ExecutionStep::Backend::OpenVINO) {
            step.debug_label = "openvino:" + std::string(target_name);
        } else if (step.backend == ExecutionStep::Backend::CUDA) {
            step.debug_label = "cuda:" + std::string(target_name);
        }

        reevaluate_tier(step);

        // Write routing decisions back into graph node
        if (auto idx_opt = graph.node_index(node.id)) {
            graph.nodes[*idx_opt].execution_backend =
                (step.backend == ExecutionStep::Backend::Native) ? "native" :
                (step.backend == ExecutionStep::Backend::OpenVINO) ? "openvino" :
                (step.backend == ExecutionStep::Backend::CUDA) ? "cuda" : "fused";
        }

        plan.push_back(std::move(step));
    }

    // Fusion pass rewrites the plan
    fuse_elementwise_chain(graph, plan);

    // Power-budget routing: if budget is low, downgrade high-power backends and prefer Cool
    if (cfg_.power_budget_watts < 5.0f) {
        for (auto& step : plan) {
            if (step.backend == ExecutionStep::Backend::CUDA ||
                step.backend == ExecutionStep::Backend::OpenVINO) {
                step.backend = ExecutionStep::Backend::Native;
                step.debug_label += " (power-override)";
                step.preferred_tier = MemoryTier::Cool;
            }
        }
    }

    if (plan.empty()) {
        return std::unexpected(hq::CerberusError::EmptyExecutionPlan);
    }

    return plan;
}

} // namespace hq::cerberus
