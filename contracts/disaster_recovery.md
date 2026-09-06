# Cerberus Disaster Recovery Plan

**Goal:** Restore all 58 disabled propup tests to fully working, passing state. Fix all compilation errors in `david_propup_engine.cpp`. Return Cerberus to production-ready build state.

**Architecture:** Fix build errors first (Phase 0), then reconstruct 58 missing propup tests in thematic groups (Phase 1-5), then verify (Phase 6). Each test must be real, not synthetic, using actual production APIs.

**Tech Stack:** C++26, GCC 15.2.0, CMake, Custom propup framework (not gtest).

**Project Base:** C:\McMaker Projects\Projects\Cerberus - Main
**Build Dir:** C:\McMaker Projects\Projects\Cerberus - Main\build
**Build Cmd:** cmake -B build code -DCMAKE_BUILD_TYPE=Release -G "MinGW Makefiles"
**Build Target:** cmake --build build --target david_propup_engine -j4
**Run Cmd:** ./build/david_propup_engine.exe (NO REDIRECTS - MinGW CRT pipe bug)

---

## P0 — Fix Pre-existing Crashes (BLOCKERS preventing full suite execution)

These crashes stop execution before reaching the 58 missing tests. Must fix first.

- P0.1: Fix `propup_staging_manager_lifecycle` segfault in `EmbeddingStagingManager` constructor
- P0.2: Fix `propup_hip_graph_denoiser_*` segfaults when HIP not available
- P0.3: Fix C API NVML re-init crash (version_string, load_model, run_inference, get_last_error)
- P0.4: Fix `propup_glow_engine_tensor_create` FAIL
- P0.5: Fix `propup_graph_engine_dtype_mismatch` FAIL

---

## Phase 0: Fix Active Build Errors

### T0.1: Fix `PropupResult` brace-initialization order in late-round tests
**Files:** `code/src/david_propup_engine.cpp` lines ~2440-2600
**Fix:** Swap diagnostic and elapsed_ms in `return {bool, name, elapsed, msg}` to `return {bool, name, msg, elapsed}`.

### T0.2: Fix `AtheneaProbeReport` namespace / scope errors
**Files:** 
- Create: `code/include/hq/athenea_probe_report.hpp`
- Modify: `code/src/cerberus_command_executor.cpp`
- Modify: `code/src/david_propup_engine.cpp`
**Fix:** Extract `AtheneaProbeReport` from local scopes to `namespace hq` header.

### T0.3: Fix missing `using PropupResult` declarations in late scopes
**Files:** `code/src/david_propup_engine.cpp` lines ~2580-2610
**Fix:** Add `using PropupResult = hq::propup::PropupResult;` to late-round test functions.

### T0.4: Deal with `-Werror` if remaining warnings block build
**Files:** `code/CMakeLists.txt` (if needed)
**Fix:** Add `-Wno-error` for dep warnings or scope `-Werror` to project code only.

---

## Phase 1: Reconstruct LCMD Inference / Audit Propups (Group 1, ~9 tests)

**APIs:** `hq::cerberus::privacy::LocalMaintenanceDB` — store_inference_record, query_inference_records, load_inference_record, export_inference_json, inference_stats, store_audit_event, load_audit_events, save_rbpc_state, load_rbpc_state, increment_rbpc_failed_attempts.

**Tests:**
1. T1.1: `propup_lcmd_inference_record_roundtrip`
2. T1.2: `propup_lcmd_inference_query_and_stats`
3. T1.3: `propup_lcmd_inference_export_json`
4. T1.4: `propup_user_security_rbpc_burn_policy`
5. T1.5: `propup_inference_audit_rbpc_gate`
6. T1.6: `propup_anbp_inference_stats_and_query`
7. T1.7: `propup_lcmd_inference_failure_recording`
8. T1.8: `propup_lcmd_full_audit_trail`
9. T1.9: `propup_server_lcmd_fresh_auto_rbpc`

**Files:** `code/include/hq/david_propup_engine.hpp`, `code/src/david_propup_engine.cpp`

---

## Phase 2: Reconstruct Intel NPU Telemetry Propups (Group 2, ~11 tests)

**APIs:** `hq::npu::IntelNpuTelemetry` — current_utilization_percent(), is_real_source_available(), source_description()

**Key Design Note:** Tests must PASS when NPU is absent (verifying graceful degradation), not fail.

**Tests:**
1. T2.1: `propup_intel_openvino_npu_telemetry`
2. T2.2: `propup_intel_npu_telemetry_construction`
3. T2.3: `propup_intel_npu_telemetry_graceful_unavailable`
4. T2.4: `propup_intel_npu_telemetry_source_description`
5. T2.5: `propup_intel_npu_telemetry_repeated_calls_safe`
6. T2.6: `propup_intel_npu_telemetry_backend_integration`
7. T2.7: `propup_intel_npu_telemetry_discovery_does_not_crash`
8. T2.8: `propup_intel_npu_telemetry_real_source_flag_consistent`
9. T2.9: `propup_intel_npu_telemetry_with_tmm_athenea_shape`
10. T2.10: `propup_intel_npu_telemetry_during_tier_migration`
11. T2.11: `propup_intel_npu_telemetry_sustained_sampling`

**Files:** `code/include/hq/david_propup_engine.hpp`, `code/src/david_propup_engine.cpp`

---

## Phase 3: Reconstruct Athenea Sustained Endurance Propups (Group 3, ~29 tests)

**APIs:** `hq::tiered_memory::TieredMemoryManager`, `hq::cerberus::ExecutionCoordinator`, `hq::npu::IntelNpuTelemetry`, `hq::cerberus::privacy::LocalMaintenanceDB`, `hq::AtheneaProbeReport`

**Key Design Note:** These tests do NOT require real NPU hardware. They verify loop integrity, telemetry structure, LCMD records, readiness scoring math, and cold/hot tier deltas.

**Tests:**
1. T3.1: `propup_npu_memory_loop_athenea_sustained`
2. T3.2: `propup_tmm_athenea_hot_tier_pressure`
3. T3.3: `propup_npu_telemetry_under_athenea_hot_load`
4. T3.4: `propup_memory_loop_hot_tier_sustained_telemetry`
5. T3.5: `propup_athenea_multi_layer_hot_telemetry`
6. T3.6: `propup_athenea_lcmd_record_hot_path`
7. T3.7: `propup_npu_memory_loop_full_athenea_pressure`
8. T3.8: `propup_athenea_timed_burst_lcmd_record`
9. T3.9: `propup_athenea_multi_layer_hot_lcmd`
10. T3.10: `propup_npu_burst_hot_tier_telemetry_lcmd`
11. T3.11: `propup_athenea_15s_chained_burst_lcmd`
12. T3.12: `propup_npu_15s_hot_burst_telemetry_lcmd`
13. T3.13: `propup_gguf_weight_slice_tmm_burst`
14. T3.14: `propup_athenea_real_weight_slice_lcmd`
15. T3.15: `propup_athenea_cold_vs_hot_burst`
16. T3.16: `propup_npu_memory_loop_cold_hot_delta_lcmd`
17. T3.17: `propup_npu_memory_loop_readiness_score`
18. T3.18: `propup_athenea_probe_readiness_lcmd`
19. T3.19: `propup_athenea_30s_endurance_cold_hot`
20. T3.20: `propup_npu_readiness_score_endurance`
21. T3.21: `propup_athenea_probe_30s_lcmd_score`
22. T3.22: `propup_athenea_45s_endurance`
23. T3.23: `propup_cold_hot_45s_delta`
24. T3.24: `propup_readiness_45s_scoring`
25. T3.25: `propup_athenea_60s_endurance_cold_hot`
26. T3.26: `propup_npu_60s_apples_to_apples_delta`
27. T3.27: `propup_readiness_60s_scoring`
28. T3.28: `propup_runtime_tmm_athenea_endurance`
29. T3.29: `propup_coordinator_athenea_burst`

**Files:** `code/include/hq/david_propup_engine.hpp`, `code/src/david_propup_engine.cpp`

---

## Phase 4: Reconstruct Code Hygiene Propups (Group 4, ~8 tests)

**Context:** Source-level audits using `std::ifstream` to read source files and assert patterns.

**Tests:**
1. T4.1: `propup_npu_no_ov_opaque_forward_decls` — scan `intel_npu_telemetry.cpp` for `class.*;` forward declarations without bodies
2. T4.2: `propup_npu_telemetry_language_hygiene` — scan for forbidden terms or bad patterns
3. T4.3: `propup_athenea_probe_final_outer_hygiene` — verify no outer-scope variables in athenea probe
4. T4.4: `propup_athenea_probe_readiness_decl_hoist` — verify declaration hoisting patterns
5. T4.5: `propup_athenea_probe_dead_var_elim` — verify no obviously dead variables
6. T4.6: `propup_athenea_probe_control_flow_avg_streak` — verify control flow pattern
7. T4.7: `propup_athenea_probe_hygiene_timing_init` — verify timing variables are initialized
8. T4.8: `propup_athenea_probe_hygiene_avg_init` — verify avg variables are initialized

**Files:** `code/include/hq/david_propup_engine.hpp`, `code/src/david_propup_engine.cpp`

---

## Phase 5: Reconstruct Staging / Migration Propups (Group 5, ~1 test)

**Tests:**
1. T5.1: `propup_staging_after_tier_migration`

**APIs:** `hq::staging::StagingManager` or coordinator staging APIs

**Files:** `code/include/hq/david_propup_engine.hpp`, `code/src/david_propup_engine.cpp`

---

## Phase 6: Build, Run, Verify

### T6.1: Full build
**Cmd:** `cmake --build build --target david_propup_engine -j4`
**Expected:** Zero errors, zero warnings treated as errors.

### T6.2: Run propup engine
**Cmd:** `./build/david_propup_engine.exe`
**Expected:** All ~200 tests pass (104 existing + ~58 restored + ~40 re-wired).

### T6.3: Regression test other targets
**Cmd:** `cmake --build build --target um790_test -j4`
**Cmd:** `cmake --build build --target cerberus_server -j4`
**Expected:** All targets compile.

---

## Implementation Strategy for Subagents

**Wave 1 (Parallel):**
- Subagent A: Phase 0 build fixes (Tasks 0.1 – 0.4)
- Subagent B: Phase 1 LCMD propups + Phase 5 staging

**Wave 2 (Parallel, after Wave 1):**
- Subagent C: Phase 2 NPU telemetry propups
- Subagent D: Phase 4 hygiene propups

**Wave 3 (Parallel, after Wave 2):**
- Subagent E: Phase 3 Athenea endurance propups (biggest group)

**Wave 4 (Sequential):**
- Subagent F: Phase 6 build & run verification

**Wave 5:**
- Orchestrator (me) does two-stage review on all changes.

---

## Principles

- **NO STUBS.** Every function body must be complete.
- **NO FAKE DATA** where real APIs exist. Use actual `LocalMaintenanceDB`, actual `IntelNpuTelemetry`, actual `TieredMemoryManager`.
- **Graceful degradation is PASS.** Tests must not fail just because NPU hardware is absent. They verify the code path works.
- **Use `PropupResult::pass()` and `PropupResult::fail()`** factory methods consistently.
- **Match existing code style.** Look at `propup_privacy_local_maintenance_db` and `propup_kernel_matmul` as templates.
