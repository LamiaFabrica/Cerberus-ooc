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
// MEGA SUITE (next 20 — pushes total past 50)
// ===========================================================================

/// @brief Prop up: 3×3 Conv2D kernel correctness on 4×4 input.
PropupResult propup_kernel_conv2d(std::ostream* log = nullptr);

/// @brief Prop up: GELU activation accuracy vs tanh reference.
PropupResult propup_kernel_gelu(std::ostream* log = nullptr);

/// @brief Prop up: softmax numerical stability (sums to 1.0, monotonic).
PropupResult propup_kernel_softmax(std::ostream* log = nullptr);

/// @brief Prop up: layer normalization (mean≈0, variance≈1).
PropupResult propup_kernel_layernorm(std::ostream* log = nullptr);

/// @brief Prop up: graph detects and preserves unreachable (dead) nodes.
PropupResult propup_graph_dead_code_elim(std::ostream* log = nullptr);

/// @brief Prop up: single node with multiple outputs.
PropupResult propup_graph_multi_output(std::ostream* log = nullptr);

/// @brief Prop up: constant node folding preserves constant_data.
PropupResult propup_graph_constant_folding(std::ostream* log = nullptr);

/// @brief Prop up: rewrite rule fuses MatMul→Add→ReLU into FusedMatMulBiasRelu.
PropupResult propup_graph_rewrite_fusion(std::ostream* log = nullptr);

/// @brief Prop up: AVX2 blocked MatMul numerical correctness on 256×256.
PropupResult propup_kernel_matmul_avx2(std::ostream* log = nullptr);

/// @brief Prop up: DecisionEngine demotes tier under memory pressure.
PropupResult propup_decision_memory_pressure(std::ostream* log = nullptr);

/// @brief Prop up: Mul→Add→Mul chain fusion detection.
PropupResult propup_decision_fuse_longer_chain(std::ostream* log = nullptr);

/// @brief Prop up: symmetric INT8 quantization roundtrip accuracy.
PropupResult propup_quant_symmetric_int8(std::ostream* log = nullptr);

/// @brief Prop up: BRECQ-style L2-minimizing scale search simulation.
PropupResult propup_quant_brecq(std::ostream* log = nullptr);

/// @brief Prop up: AdaRound-style grid-search rounding simulation.
PropupResult propup_quant_adaround(std::ostream* log = nullptr);

/// @brief Prop up: unavailable backend triggers CpuFallback path.
PropupResult propup_backend_cpu_fallback(std::ostream* log = nullptr);

/// @brief Prop up: compile() rejects empty graph with diagnostic.
PropupResult propup_backend_compile_error(std::ostream* log = nullptr);

/// @brief Prop up: warm→cool demotion preserves written bytes.
PropupResult propup_tier_demote_with_data(std::ostream* log = nullptr);

/// @brief Prop up: LRU eviction path under micro-capacity.
PropupResult propup_tier_eviction_lru(std::ostream* log = nullptr);

/// @brief Prop up: allocation respects tier alignment.
PropupResult propup_tier_alignment(std::ostream* log = nullptr);

/// @brief Prop up: 3-step graph (Mul→Add→Mul) through CerberusRuntime.
PropupResult propup_runtime_multi_step(std::ostream* log = nullptr);

/// @brief Prop up: runtime propagates compile errors upward.
PropupResult propup_runtime_error_propagation(std::ostream* log = nullptr);

/// @brief Prop up: 512×512 FP32 MatMul stress (must complete).
PropupResult propup_stress_matmul_512(std::ostream* log = nullptr);

// ===========================================================================
// ADVERSARIAL ROBUSTNESS SUITE (deliberately breaks the engine)
// ===========================================================================

/// @brief Prop up: null graph pointer → non-empty diagnostic.
PropupResult propup_robust_null_graph(std::ostream* log = nullptr);

/// @brief Prop up: null input buffer → non-empty diagnostic.
PropupResult propup_robust_null_input(std::ostream* log = nullptr);

/// @brief Prop up: M=0 dimension in MatMul → error, not crash.
PropupResult propup_robust_zero_dimension(std::ostream* log = nullptr);

/// @brief Prop up: misaligned tensor names (input "x" vs node "a").
PropupResult propup_robust_name_mismatch(std::ostream* log = nullptr);

/// @brief Prop up: cycle graph → topo_sort rejects.
PropupResult propup_robust_cycle_rejection(std::ostream* log = nullptr);

/// @brief Prop up: backend.execute after backend goes unavailable.
PropupResult propup_robust_backend_unavailable(std::ostream* log = nullptr);

/// @brief Prop up: duplicate tensor names in graph.
PropupResult propup_robust_duplicate_names(std::ostream* log = nullptr);

/// @brief Prop up: input buffer count mismatch.
PropupResult propup_robust_input_count_mismatch(std::ostream* log = nullptr);

/// @brief Prop up: allocate with size=0 → handled gracefully.
PropupResult propup_robust_zero_size_alloc(std::ostream* log = nullptr);

/// @brief Prop up: compile graph with unsupported op → error.
PropupResult propup_robust_unsupported_op(std::ostream* log = nullptr);

/// @brief Prop up: rapid promote/demote/pressure_evict pathological churn.
PropupResult propup_robust_tier_thrashing(std::ostream* log = nullptr);

/// @brief Prop up: register and verify dequantization during promote.
PropupResult propup_dequant_during_migration(std::ostream* log = nullptr);

/// @brief Prop up: register and verify in-flight compute during promote/demote.
PropupResult propup_compute_during_migration(std::ostream* log = nullptr);

/// @brief Prop up: FNV-1a hash consistency (identical/different data).
PropupResult propup_execution_integrity(std::ostream* log = nullptr);

/// @brief Prop up: ExecutionPredictor caches and returns consistent snapshots.
PropupResult propup_predictor_match(std::ostream* log = nullptr);

/// @brief Prop up: shadow state compress/restore produces bounded-quantization error.
PropupResult propup_shadow_rollback(std::ostream* log = nullptr);

/// @brief Prop up: DecisionEngine respects power budget and downgrades precision.
PropupResult propup_decision_power_budget(std::ostream* log = nullptr);

/// @brief Prop up: feedback loop from predictor hit rate → policy adjustment.
PropupResult propup_predictor_feedback(std::ostream* log = nullptr);

/// @brief Prop up: federated backend splits graph across two backends.
PropupResult propup_federation_split(std::ostream* log = nullptr);

/// @brief Prop up: command layer — system:status via cerberus:// protocol.
PropupResult propup_command_layer_status(std::ostream* log = nullptr);

/// @brief Prop up: command layer — graph:compile via cbr:compile ergonomic shortcut.
PropupResult propup_command_layer_compile(std::ostream* log = nullptr);

/// @brief Prop up: command layer — malformed commands produce structured error.
PropupResult propup_command_layer_malformed(std::ostream* log = nullptr);

/// @brief Prop up: command layer — ergonomic fallback parsing.
PropupResult propup_command_layer_ergonomic(std::ostream* log = nullptr);

/// @brief Prop up: ANBP gateway handshake with session establishment.
PropupResult propup_anbp_gateway_handshake(std::ostream* log = nullptr);

/// @brief Prop up: human operator safety filter — ACT mode blocks EXECUTE ops.
PropupResult propup_api_gateway_human_safety_filter(std::ostream* log = nullptr);

/// @brief Prop up: slipstream tensor roundtrip via depot exchange.
PropupResult propup_slipstream_tensor_exchange(std::ostream* log = nullptr);

/// @brief Prop up: metro audit trail captures packet stats + glowing string trace.
PropupResult propup_metro_audit_trail(std::ostream* log = nullptr);

// ===========================================================================
// GLOW ENGINE SUITE (ported from PsiForceDB Nemadic v3)
// ===========================================================================

/// @brief Prop up: record_execution creates bonds between traversed nodes.
PropupResult propup_glow_bond_creation(std::ostream* log = nullptr);

/// @brief Prop up: reinforcement strengthens bond learned_weight.
PropupResult propup_glow_reinforcement(std::ostream* log = nullptr);

/// @brief Prop up: decay weakens bonds and prunes zero-weight entries.
PropupResult propup_glow_decay(std::ostream* log = nullptr);

/// @brief Prop up: query_hot_paths returns paths sorted by amplitude desc.
PropupResult propup_glow_hot_path_query(std::ostream* log = nullptr);

/// @brief Prop up: best_next_hop returns strongest outgoing neighbor.
PropupResult propup_glow_best_next_hop(std::ostream* log = nullptr);

/// @brief Prop up: exact catchphrase resolves to correct node id.
PropupResult propup_glow_catchphrase_exact(std::ostream* log = nullptr);

/// @brief Prop up: fuzzy catchphrase matching finds nearest registered phrase.
PropupResult propup_glow_catchphrase_fuzzy(std::ostream* log = nullptr);

/// @brief Prop up: GlowStats reflect accurate active bond and reinforcement counts.
PropupResult propup_glow_stats_integrity(std::ostream* log = nullptr);

/// @brief Prop up: reset() clears all bonds, paths, and stats to zero.
PropupResult propup_glow_reset(std::ostream* log = nullptr);

/// @brief Prop up: bond attenuation decays amplitude across hops.
PropupResult propup_glow_attenuation(std::ostream* log = nullptr);

// ===========================================================================
// ADVERSARIAL ROBUSTNESS EXTENSIONS
// ===========================================================================

/// @brief Prop up: malformed ANBP header (wrong magic) rejected without crash.
PropupResult propup_adversarial_malformed_anbp(std::ostream* log = nullptr);

/// @brief Prop up: invalid session token in ANBP request returns ERROR_SESSION.
PropupResult propup_adversarial_invalid_token(std::ostream* log = nullptr);

/// @brief Prop up: permission escalation attempt (ACT -> AGENTIC) blocked.
PropupResult propup_adversarial_permission_escalation(std::ostream* log = nullptr);

/// @brief Prop up: slipstream depot drop under capacity overflow.
PropupResult propup_adversarial_slipstream_overflow(std::ostream* log = nullptr);

/// @brief Prop up: metro processIncoming with empty payload handled gracefully.
PropupResult propup_adversarial_metro_empty_payload(std::ostream* log = nullptr);

// ===========================================================================
// GGUF PARSER SUITE (synthetic — validates parser code, not external assets)
// ===========================================================================

/// @brief Prop up: synthetic GGUF header magic validates parser endianness logic.
PropupResult propup_gguf_synthetic_header(std::ostream* log = nullptr);

/// @brief Prop up: synthetic tensor info roundtrip validates data structure integrity.
PropupResult propup_gguf_synthetic_tensor_info(std::ostream* log = nullptr);

// ===========================================================================
// PSIFORCEDB EXTENSION INTEGRATION
// ===========================================================================

/// @brief Prop up: CerberusExtension initializes with PsiForceDB ExtensionConfig.
PropupResult propup_psiforcedb_extension_init(std::ostream* log = nullptr);

/// @brief Prop up: CerberusExtension load/unload lifecycle works.
PropupResult propup_psiforcedb_extension_load_unload(std::ostream* log = nullptr);

/// @brief Prop up: CerberusExtension executeQuery INFERENCE routes to CerberusRuntime.
PropupResult propup_psiforcedb_extension_inference_query(std::ostream* log = nullptr);

/// @brief Prop up: CerberusExtension executeQuery STATUS returns telemetry.
PropupResult propup_psiforcedb_extension_status_query(std::ostream* log = nullptr);

/// @brief Prop up: CerberusExtension getStatistics reports query counts.
PropupResult propup_psiforcedb_extension_stats(std::ostream* log = nullptr);

/// @brief Prop up: CerberusExtension validateQuery accepts cerberus:// and cbr: prefixes.
PropupResult propup_psiforcedb_extension_validate(std::ostream* log = nullptr);

/// @brief Prop up: CerberusExtension routes PFQL-style inference queries.
PropupResult propup_psiforcedb_extension_pfql_routing(std::ostream* log = nullptr);

/// @brief Prop up: CerberusExtension GGUF_LOADER query returns synthetic Athenea tensor metadata.
PropupResult propup_psiforcedb_extension_gguf_loader(std::ostream* log = nullptr);

/// @brief Prop up: CerberusExtension TELEMETRY query packages STATUS + GLOW rows for PsiForceDB web tier.
PropupResult propup_psiforcedb_extension_telemetry(std::ostream* log = nullptr);

/// @brief Prop up: CerberusExtension health check returns correct state after load/unload.
PropupResult propup_psiforcedb_extension_health_check(std::ostream* log = nullptr);

/// @brief Prop up: CerberusExtension rejects transaction begin because supportsTransaction=false.
PropupResult propup_psiforcedb_extension_transaction_reject(std::ostream* log = nullptr);

/// @brief Prop up: Synthetic PsiForceDB MultiModelCoordinator routes INFERENCE to CerberusExtension.
PropupResult propup_psiforcedb_extension_coordinator_routing(std::ostream* log = nullptr);

/// @brief Prop up: C factory cerberus_create_extension produces valid extension.
PropupResult propup_psiforcedb_extension_factory(std::ostream* log = nullptr);

/// @brief Prop up: Extension with missing dependency fails to load.
PropupResult propup_psiforcedb_extension_dependencies(std::ostream* log = nullptr);

/// @brief Prop up: getMetadata() roundtrip returns correct name, version, model_type, supported_queries.
PropupResult propup_psiforcedb_extension_metadata(std::ostream* log = nullptr);

/// @brief Prop up: ModelTopologyMapper produces valid PsiForceDB graph topology.
PropupResult propup_psiforcedb_graph_bridge_topology(std::ostream* log = nullptr);

/// @brief Prop up: GraphNode/GraphEdge PFQL row export produces correct fields.
PropupResult propup_psiforcedb_graph_bridge_pfql_rows(std::ostream* log = nullptr);

/// @brief Prop up: Extension validateQuery rejects unbalanced quotes and oversized strings.
PropupResult propup_psiforcedb_extension_validate_edge_cases(std::ostream* log = nullptr);

/// @brief Prop up: Extension error_count increments on failed init/load/unload.
PropupResult propup_psiforcedb_extension_error_counting(std::ostream* log = nullptr);

/// @brief Prop up: Extension parameter_or and model_from_type helper utilities work.
PropupResult propup_psiforcedb_extension_detail_helpers(std::ostream* log = nullptr);

/// @brief Prop up: Extension GlowEngine hot-path recording is visible through GLOW query.
PropupResult propup_psiforcedb_extension_glow_integration(std::ostream* log = nullptr);

// ===========================================================================
// AVX-512 DISPATCH SUITE (synthetic — validates dispatch logic, not host hw)
// ===========================================================================

/// @brief Prop up: AVX-512 feature detection returns consistent result.
PropupResult propup_kernel_avx512_detect(std::ostream* log = nullptr);

/// @brief Prop up: AVX-512 blocked MatMul synthetic dispatch (no real AVX-512 hw required).
PropupResult propup_kernel_matmul_avx512_dispatch(std::ostream* log = nullptr);

// ===========================================================================
// Additional native kernel propups
// ===========================================================================

/// @brief Prop up: kernel_sigmoid numerical correctness.
PropupResult propup_kernel_sigmoid(std::ostream* log = nullptr);

/// @brief Prop up: kernel_relu numerical correctness.
PropupResult propup_kernel_relu(std::ostream* log = nullptr);

/// @brief Prop up: CryptoBridge SHA256 determinism and digest size.
PropupResult propup_security_sha256(std::ostream* log = nullptr);

/// @brief Prop up: CryptoBridge HMAC-SHA256 integrity and uniqueness.
PropupResult propup_security_hmac_sha256(std::ostream* log = nullptr);

/// @brief Prop up: CryptoBridge PBKDF2-SHA256 determinism and sensitivity.
PropupResult propup_security_pbkdf2_sha256(std::ostream* log = nullptr);

/// @brief Prop up: LfsslSentinel correctly reports AES-256-GCM as delegated to PsiForceDB.
PropupResult propup_security_aes256_gcm_sentinel(std::ostream* log = nullptr);

/// @brief Prop up: LfsslSentinel correctly reports Kyber/Dilithium as delegated to PsiForceDB.
PropupResult propup_security_pqc_sentinel(std::ostream* log = nullptr);

/// @brief Prop up: Cerberus compiles and links against real PsiForceDB extension_interface.hpp.
PropupResult propup_psiforcedb_extension_real_header_compile(std::ostream* log = nullptr);

// ===========================================================================
// Privacy / RBPC / Local Maintenance DB (carbon copy of PsiForceDB security surface)
// ===========================================================================

/// @brief Prop up: LocalMaintenanceDB initializes, stores encrypted records, syncs queue.
PropupResult propup_privacy_local_maintenance_db(std::ostream* log = nullptr);

/// @brief Prop up: RBPC PIN generation system-issued, commitment-only storage.
PropupResult propup_privacy_pin_generation(std::ostream* log = nullptr);

/// @brief Prop up: RBPC PIN verification with burn-after-3-attempts policy.
PropupResult propup_privacy_pin_burn_policy(std::ostream* log = nullptr);

/// @brief Prop up: Memorable word validation and commitment derivation.
PropupResult propup_privacy_word_commitment(std::ostream* log = nullptr);

/// @brief Prop up: Dual-factor confirmation (PIN + Word) offline/online.
PropupResult propup_privacy_dual_factor_confirmation(std::ostream* log = nullptr);

/// @brief Prop up: JWT session creation, validation, refresh, revocation (CSF/BFD/InjectionProof/Sentry).
PropupResult propup_privacy_jwt_session(std::ostream* log = nullptr);

/// @brief Prop up: extension entry store/search/load in LocalMaintenanceDB.
PropupResult propup_lcmd_extension_entry(std::ostream* log = nullptr);

/// @brief Prop up: revenue share record store/load in LocalMaintenanceDB.
PropupResult propup_lcmd_revenue_share(std::ostream* log = nullptr);

/// @brief Prop up: VIP key store/exists/get_all in LocalMaintenanceDB.
PropupResult propup_lcmd_vip_keys(std::ostream* log = nullptr);

/// @brief Prop up: onboarding grant store/consume/load in LocalMaintenanceDB.
PropupResult propup_lcmd_onboarding_grant(std::ostream* log = nullptr);

/// @brief Prop up: concurrent JWT validation from multiple threads.
PropupResult propup_privacy_jwt_concurrent(std::ostream* log = nullptr);

/// @brief Prop up: cerberus_lfssl.dll loads, exports resolve, and primitives pass smoke.
PropupResult propup_lfssl_dll_smoke(std::ostream* log = nullptr);

/// @brief Prop up: SHA-256 digest determinism via DLL.
PropupResult propup_lfssl_dll_sha256(std::ostream* log = nullptr);

/// @brief Prop up: HMAC-SHA256 correctness via DLL.
PropupResult propup_lfssl_dll_hmac(std::ostream* log = nullptr);

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

} // namespace hq::propup
