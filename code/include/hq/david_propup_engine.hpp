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
    bool skipped{false};
    std::string name;
    std::string diagnostic;
    double elapsed_ms{0.0};

    static PropupResult pass(std::string_view n) {
        return PropupResult{true, false, std::string(n), {}, 0.0};
    }
    static PropupResult fail(std::string_view n, std::string_view diag) {
        return PropupResult{false, false, std::string(n), std::string(diag), 0.0};
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

/// @brief Prop up: blocked matmul is measurably faster than naïve.
PropupResult propup_performance_matmul_vs_naive(std::ostream* log = nullptr);

/// @brief Prop up: end-to-end smoke test (Mul+Add graph => native execution).
PropupResult propup_end_to_end_native(std::ostream* log = nullptr);

/// @brief Prop up: graph → DecisionEngine → fused steps → coordinator → execution.
PropupResult propup_decision_engine_fusion(std::ostream* log = nullptr);

// ===========================================================================
// MEGA SUITE (next 20 — pushes total past 50)
// ===========================================================================

/// @brief Prop up: AVX2 blocked MatMul numerical correctness on 256×256.
PropupResult propup_kernel_matmul_avx2(std::ostream* log = nullptr);

// ===========================================================================
// ADVERSARIAL ROBUSTNESS SUITE (deliberately breaks the engine)
// ===========================================================================

// ===========================================================================
// GLOW ENGINE SUITE (ported from PsiForceDB Nemadic v3)
// ===========================================================================

// ===========================================================================
// ADVERSARIAL ROBUSTNESS EXTENSIONS
// ===========================================================================

// ===========================================================================
// GGUF PARSER SUITE (synthetic — validates parser code, not external assets)
// ===========================================================================

// ===========================================================================
// PSIFORCEDB EXTENSION INTEGRATION
// ===========================================================================

// ===========================================================================
// AVX-512 DISPATCH SUITE (synthetic — validates dispatch logic, not host hw)
// ===========================================================================

/// @brief Prop up: AVX-512 feature detection returns consistent result.
PropupResult propup_kernel_avx512_detect(std::ostream* log = nullptr);

// ===========================================================================
// Additional native kernel propups
// ===========================================================================

/// @brief Prop up: kernel_sigmoid numerical correctness.
PropupResult propup_kernel_sigmoid(std::ostream* log = nullptr);

/// @brief Prop up: kernel_relu numerical correctness.
PropupResult propup_kernel_relu(std::ostream* log = nullptr);

// ===========================================================================
// Privacy / RBPC / Local Maintenance DB (carbon copy of PsiForceDB security surface)
// ===========================================================================

// ===========================================================================
// LCMD EDGE BEHAVIOR (fill remaining surface)
// ===========================================================================

// ===========================================================================
// JWT NEGATIVE PATHS
// ===========================================================================

// ===========================================================================
// DLL PRIMITIVES EXPANSION
// ===========================================================================

// ===========================================================================
// GLOW ENGINE EDGE CASES
// ===========================================================================

// ===========================================================================
// COMMAND / ANBP / METRO / SLIPSTREAM EDGE CASES
// ===========================================================================

// ===========================================================================
// Full test suite — runs all validators and returns aggregate report
// ===========================================================================
// COMMAND / ANBP / METRO / SLIPSTREAM EDGE CASES
// ===========================================================================

// ===========================================================================
// EXTENSION EDGE NEGATIVES
// ===========================================================================

// ===========================================================================
// FINAL 12 — reach 200+
// ===========================================================================

/// @brief Prop up: LCMD offline sync queue count matches push count.
PropupResult propup_lcmd_offline_sync_count(std::ostream* log = nullptr);

// ===========================================================================
// E2E DETECTABLE TESTBED — 25 additional propups (target 260+)
// ===========================================================================

// ===========================================================================
// INFRASTRUCTURE / MISSING GROUPS — 35 new propups to reach 300+
// ===========================================================================

/// @brief StagingManager operates correctly after TieredMemoryManager promote/demote activity.
///        Guards cross-test heap interaction between heavy TMM migration and subsequent staging.
PropupResult propup_staging_after_tier_migration(std::ostream* log = nullptr);

// ===========================================================================
// Full test suite — runs all validators and returns aggregate report
// ===========================================================================

struct PropupReport {
    std::vector<PropupResult> results;
    std::size_t passed_count{0};
    std::size_t failed_count{0};
    double total_ms{0.0};
    std::size_t skipped_verbose_count{0};

    [[nodiscard]] bool all_passed() const noexcept {
        return failed_count == 0;
    }
    void print(std::ostream& out) const;
};

[[nodiscard]] PropupReport run_all_propups(std::ostream* log = nullptr);

// ===========================================================================
// Round 22 propups (12 tests — FMA blend, telemetry cache, TMM hot, LCMD, denoise, endurance, quality, NPU metrics, coordination, documentation)
// ===========================================================================

/// @brief Prop up: FMA blend stability.
PropupResult propup_round22_fma_blend_stability(std::ostream* log = nullptr);

/// @brief Prop up: telemetry cache benefit.
PropupResult propup_round22_telemetry_cache_benefit(std::ostream* log = nullptr);

/// @brief Prop up: reduced sampling utilization.
PropupResult propup_round22_reduced_sampling_util(std::ostream* log = nullptr);

/// @brief Prop up: TMM hot during optimized burst.
PropupResult propup_round22_tmm_hot_during_optimized_burst(std::ostream* log = nullptr);

/// @brief Prop up: LCMD via runtime only.
PropupResult propup_round22_lcmd_via_runtime_only(std::ostream* log = nullptr);

/// @brief Prop up: FMA in denoise path.
PropupResult propup_round22_fma_in_denoise_path(std::ostream* log = nullptr);

/// @brief Prop up: cache in Intel telemetry.
PropupResult propup_round22_cache_in_intel_telemetry(std::ostream* log = nullptr);

/// @brief Prop up: endurance with reduced sync.
PropupResult propup_round22_endurance_with_reduced_sync(std::ostream* log = nullptr);

/// @brief Prop up: quality FMA vs naive.
PropupResult propup_round22_quality_fma_vs_naive(std::ostream* log = nullptr);

/// @brief Prop up: NPU utilization metrics in report.
PropupResult propup_round22_npu_util_metrics_in_report(std::ostream* log = nullptr);

/// @brief Prop up: TMM coordinator interaction.
PropupResult propup_round22_tmm_coordinator_interaction(std::ostream* log = nullptr);

/// @brief Prop up: all stages documented.
PropupResult propup_round22_all_stages_documented(std::ostream* log = nullptr);

/// @brief Prop up: Athenea 60s endurance cold→hot.
PropupResult propup_round24_athenea_60s_endurance_cold_hot(std::ostream* log = nullptr);

/// @brief Prop up: NPU memory loop readiness score.
PropupResult propup_round24_npu_memory_loop_readiness_score(std::ostream* log = nullptr);

/// @brief Prop up: Athenea cold vs hot burst.
PropupResult propup_round24_athenea_cold_vs_hot_burst(std::ostream* log = nullptr);

/// @brief Prop up: NPU memory loop full Athenea pressure.
PropupResult propup_round24_npu_memory_loop_full_athenea_pressure(std::ostream* log = nullptr);

/// @brief Prop up: Athenea probe readiness LCMD.
PropupResult propup_round24_athenea_probe_readiness_lcmd(std::ostream* log = nullptr);

/// @brief Prop up: NPU memory loop cold→hot delta LCMD.
PropupResult propup_round24_npu_memory_loop_cold_hot_delta_lcmd(std::ostream* log = nullptr);

/// @brief Prop up: Athenea 30s endurance cold→hot.
PropupResult propup_round24_athenea_30s_endurance_cold_hot(std::ostream* log = nullptr);

/// @brief Prop up: NPU memory loop sustained telemetry.
PropupResult propup_round24_npu_memory_loop_sustained_telemetry(std::ostream* log = nullptr);

// ===========================================================================
// Intel NPU Telemetry validation suite (graceful degradation on non-NPU hosts)
// ===========================================================================

/// @brief Prop up: IntelNpuTelemetry construction succeeds and produces non-empty description.
PropupResult propup_intel_npu_telemetry_construction(std::ostream* log = nullptr);

/// @brief Prop up: IntelNpuTelemetry returns -1.0f and false availability on non-NPU Windows.
PropupResult propup_intel_npu_telemetry_graceful_unavailable(std::ostream* log = nullptr);

/// @brief Prop up: IntelNpuTelemetry source_description contains platform info.
PropupResult propup_intel_npu_telemetry_source_description(std::ostream* log = nullptr);

/// @brief Prop up: IntelNpuTelemetry repeated calls are safe and consistent.
PropupResult propup_intel_npu_telemetry_repeated_calls_safe(std::ostream* log = nullptr);

/// @brief Prop up: IntelNpuTelemetry coexists with NpuBackendFactory without interference.
PropupResult propup_intel_npu_telemetry_backend_integration(std::ostream* log = nullptr);

/// @brief Prop up: IntelNpuTelemetry PDH discovery does not crash when no NPU driver present.
PropupResult propup_intel_npu_telemetry_discovery_does_not_crash(std::ostream* log = nullptr);

/// @brief Prop up: IntelNpuTelemetry real_source_available flag stays consistent across samples.
PropupResult propup_intel_npu_telemetry_real_source_flag_consistent(std::ostream* log = nullptr);

/// @brief Prop up: IntelNpuTelemetry + TieredMemoryManager Athenea shape coexist safely.
PropupResult propup_intel_npu_telemetry_with_tmm_athenea_shape(std::ostream* log = nullptr);

/// @brief Prop up: IntelNpuTelemetry samples remain valid during tier migration activity.
PropupResult propup_intel_npu_telemetry_during_tier_migration(std::ostream* log = nullptr);

/// @brief Prop up: IntelNpuTelemetry sustained sampling respects cache and stays in bounds.
PropupResult propup_intel_npu_telemetry_sustained_sampling(std::ostream* log = nullptr);

} // namespace hq::propup
