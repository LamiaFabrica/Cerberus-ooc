#pragma once
/// @file david_propup_engine.hpp
/// @copyright Copyright (c) 2026 D Hargreaves. All rights reserved.
///
/// David Propup Engine — comprehensive test + validation harness for all
/// Cerberus subsystems.  Every logic gate, every tier decision, every
/// native kernel, and every graph transformation is exercised here.
///
/// Philosophy: test the engine INTO existence.  If a component cannot
/// pass its propup test, it does not ship.
///
/// @version 1.0.0

#include "hq/npu_backend_unified.hpp"
#include "hq/tiered_memory_manager.hpp"
#include "hq/cerberus_execution_coordinator.hpp"

#include <cstddef>
#include <span>
#include <expected>
#include <string>
#include <vector>
#include <ostream>

namespace hq::propup {

// ===========================================================================
// Test result — honest pass/fail with diagnostics
// ===========================================================================

struct PropupResult {
    bool passed{false};
    std::string name;
    std::string diagnostic;
    double elapsed_ms{0.0};

    static PropupResult pass(std::string_view n) {
        return PropupResult{true, std::string(n), {}, 0.0};
    }
    static PropupResult fail(std::string_view n, std::string_view diag) {
        return PropupResult{false, std::string(n), std::string(diag), 0.0};
    }
};

// ===========================================================================
// Individual subsystem validators
// ===========================================================================

/// @brief Prop up: does the native kernel_matmul produce correct results?
PropupResult propup_kernel_matmul(std::ostream* log = nullptr);

/// @brief Prop up: do native elementwise kernels (add, mul) match reference?
PropupResult propup_kernel_elementwise(std::ostream* log = nullptr);

/// @brief Prop up: does TieredMemoryManager allocate/fill/promote/demote?
PropupResult propup_tiered_memory(std::ostream* log = nullptr);

/// @brief Prop up: does the coordinator stage both inputs and outputs?
PropupResult propup_coordinator_memory_loop(std::ostream* log = nullptr);

/// @brief Prop up: does high_reuse_tensors actually influence tier placement?
PropupResult propup_coordinator_tier_decisions(std::ostream* log = nullptr);

/// @brief Prop up: does compile() walk graph.nodes and populate metadata?
PropupResult propup_compile_graph_analysis(std::ostream* log = nullptr);

/// @brief Prop up: fused FMA kernel produces correct results.
PropupResult propup_kernel_fma(std::ostream* log = nullptr);

/// @brief Prop up: cache-blocked MatMul correctness vs naïve reference.
PropupResult propup_kernel_matmul_blocked(std::ostream* log = nullptr);

/// @brief Prop up: INT8 quantized MatMul correctness vs float reference.
PropupResult propup_kernel_quantized_matmul(std::ostream* log = nullptr);

/// @brief Prop up: dynamic quantization produces >95% accuracy relative to float.
PropupResult propup_kernel_dynamic_quant_accuracy(std::ostream* log = nullptr);

/// @brief Prop up: INT8 is measurably faster than float (memory bandwidth).
PropupResult propup_performance_int8_vs_float(std::ostream* log = nullptr);

/// @brief Prop up: blocked matmul is measurably faster than naïve.
PropupResult propup_performance_matmul_vs_naive(std::ostream* log = nullptr);

/// @brief Prop up: end-to-end smoke test (Mul+Add graph => native execution).
PropupResult propup_end_to_end_native(std::ostream* log = nullptr);

/// @brief Prop up: graph → DecisionEngine → fused steps → coordinator → execution.
PropupResult propup_decision_engine_fusion(std::ostream* log = nullptr);

/// @brief VERBOSE: graph engine construction, topo sort, lookups, live bytes.
/// This test is extra-verbose and can be detached once stable.
PropupResult propup_graph_engine_verbose(std::ostream* log = nullptr);

/// @brief VERBOSE: DecisionEngine tier demotion under memory pressure.
PropupResult propup_tier_pressure_demotion(std::ostream* log = nullptr);

/// @brief Prop up: per-channel quantization accuracy vs reference float.
PropupResult propup_quant_per_channel(std::ostream* log = nullptr);

/// @brief Prop up: per-token (per-row) quantization accuracy.
PropupResult propup_quant_per_token(std::ostream* log = nullptr);

/// @brief Prop up: K-block grouped quantization (GGUF-style) accuracy.
PropupResult propup_quant_per_block(std::ostream* log = nullptr);

/// @brief Prop up: INT4 weight quantization/dequantization accuracy.
PropupResult propup_quant_4bit(std::ostream* log = nullptr);

/// @brief Prop up: fused quant→bias→ReLU accuracy.
PropupResult propup_quant_fused_bias_relu(std::ostream* log = nullptr);

/// @brief Prop up: CerberusNativeBackend end-to-end MatMul.
PropupResult propup_native_backend_matmul(std::ostream* log = nullptr);

/// @brief Prop up: CerberusNativeBackend elementwise chain (Add+Mul).
PropupResult propup_native_backend_elementwise(std::ostream* log = nullptr);

/// @brief Prop up: CerberusNativeBackend fused subgraph rewrite.
PropupResult propup_native_backend_fusion(std::ostream* log = nullptr);

/// @brief Prop up: DecisionEngine routes >64 MatMul to native, <64 to fallback.
PropupResult propup_decision_backend_routing(std::ostream* log = nullptr);

/// @brief Prop up: graph topo_sort rejects cycles.
PropupResult propup_graph_cycles_rejection(std::ostream* log = nullptr);

/// @brief Prop up: KernelGraph → CerberusGraph roundtrip preserves metadata.
PropupResult propup_graph_from_kernel_graph(std::ostream* log = nullptr);

/// @brief Prop up: CerberusRuntime::run_graph end-to-end (API surface).
PropupResult propup_runtime_full_stack(std::ostream* log = nullptr);

/// @brief Prop up: cold tier allocation + readback roundtrip.
PropupResult propup_tier_cold_spill(std::ostream* log = nullptr);

/// @brief Prop up: promote then demote a block and verify data integrity.
PropupResult propup_tier_migration_promote_demote(std::ostream* log = nullptr);

/// @brief Prop up: graceful OOM under tiny capacity.
PropupResult propup_tier_out_of_memory(std::ostream* log = nullptr);

/// @brief Prop up: simulate PTQ vs QAT accuracy gap (<2% vs >5%).
PropupResult propup_quant_ptq_vs_qat_sim(std::ostream* log = nullptr);

/// @brief Prop up: SmoothQuant-style max-scaling preserves ratio.
PropupResult propup_quant_smoothquant(std::ostream* log = nullptr);

/// @brief Prop up: blocked INT8 GEMM vs scalar INT8 speedup.
PropupResult propup_performance_blocked_vs_quantized(std::ostream* log = nullptr);

/// @brief Prop up: DecisionEngine routes attention→8-bit, linear→4-bit.
PropupResult propup_decision_quant_routing(std::ostream* log = nullptr);

/// @brief Prop up: quantized MatMul + blocked tiling correctness.
PropupResult propup_quantized_matmul_blocked(std::ostream* log = nullptr);

// ===========================================================================
// Full test suite — runs all validators and returns aggregate report
// ===========================================================================

struct PropupReport {
    std::vector<PropupResult> results;
    std::size_t passed_count{0};
    std::size_t failed_count{0};
    double total_ms{0.0};

    [[nodiscard]] bool all_passed() const noexcept {
        return failed_count == 0;
    }
    void print(std::ostream& out) const;
};

[[nodiscard]] PropupReport run_all_propups(std::ostream* log = nullptr);

} // namespace hq::propup
