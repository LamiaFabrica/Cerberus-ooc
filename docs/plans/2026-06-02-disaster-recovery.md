# Cerberus Disaster Recovery — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task. Each task is bite-sized (2-5 min of focused work).

**Goal:** Restore the Cerberus Propup Engine to full production readiness — 258 passing tests, zero segfaults, zero stubs, zero skipped tests without honest hardware justification.

**Architecture:** Fix the segfault first (binary stability), then implement missing tests in parallel swarms, then harden existing tests with real validation. All changes must survive direct execution without MSYS pipe redirects.

**Tech Stack:** C++26, MinGW-W64 GCC 15.2.0, CMake, ONNX Runtime 1.17.1, Windows 10

---

## Phase 0: CRITICAL — Fix Binary Segfault (Blocks Everything)

### Task 0.1: Diagnose Segfault Location

**Objective:** Identify exactly which test/function causes the crash.

**Files:**
- Read: `code/src/david_propup_engine.cpp` (find `run_all_propups()`)
- Read: `code/include/hq/hip_graph_denoiser.hpp`
- Read: `code/src/runtime/hip_graph_denoiser.cpp`

**Step 1:** Add debug prints before/after each `run_one()` call in `run_all_propups()`:
```cpp
std::cerr << "[DEBUG] About to run: " << name << std::endl;
run_one(propup_foo, "propup_foo");
std::cerr << "[DEBUG] Completed: " << name << std::endl;
```

**Step 2:** Rebuild and run:
```bash
cd "/c/McMaker Projects/Projects/Cerberus - Main"
mingw32-make -C build um790_pipeline
mingw32-make -C build david_propup_engine/fast
./build/david_propup_engine.exe
```

**Step 3:** Identify the last "About to run" print before crash. That test is the culprit.

**Verification:** The last printed test name is the segfault location.

---

### Task 0.2: Fix Segfault Root Cause

**Objective:** Fix the identified crash (likely HIPGraphDenoiser destructor or StagingManager).

**Files:**
- Modify: `code/src/runtime/hip_graph_denoiser.cpp` or `code/src/runtime/staging_manager.cpp`

**Common fixes:**
- Null pointer dereference in destructor
- Double-free of device buffers
- ODR violation in DdimDeviceParams (header vs .cpp alignment mismatch)
- Missing null check before `free_device_buffers_()`

**Step 1:** Add null guards:
```cpp
void free_device_buffers_() noexcept {
    if (d_latents_) { /* free */ d_latents_ = nullptr; }
    if (d_noise_pred_) { /* free */ d_noise_pred_ = nullptr; }
    if (d_ddim_params_) { /* free */ d_ddim_params_ = nullptr; }
    if (h_pinned_params_) { /* free */ h_pinned_params_ = nullptr; }
}
```

**Step 2:** Rebuild and verify:
```bash
mingw32-make -C build um790_pipeline
mingw32-make -C build david_propup_engine/fast
./build/david_propup_engine.exe
```

**Verification:** All 104 registered tests execute without crash. Exit code 0.

---

## Phase 1: Fix ORT API Version Mismatch (Cosmetic but Indicative)

### Task 1.1: Align ORT API Version to Runtime

**Objective:** Eliminate "API version [21] not available" warnings.

**Files:**
- Modify: `code/include/hq/npu_pipeline.hpp` (line 80, before `#include <onnxruntime_cxx_api.h>`)
- Modify: Any other file that includes ORT headers first

**Step 1:** Add version define BEFORE any ORT include:
```cpp
#define ORT_API_VERSION 17  // Match runtime v1.17.1
#include <onnxruntime_cxx_api.h>
```

**Step 2:** Check CMakeLists.txt for `ONNXRUNTIME_ROOT` and verify headers match runtime.

**Step 3:** Rebuild and verify warnings disappear.

**Verification:** No "API version [21]" messages in output.

---

## Phase 2: Implement 5 Skipped Tests (Real Code, No Stubs)

### Task 2.1: Implement GGUF Parser Tests

**Objective:** Replace `.skip("GGUF parser not implemented")` with real validation.

**Files:**
- Read: `code/include/hq/cerberus_gguf_parser.hpp`
- Read: `code/src/cerberus_gguf_parser.cpp`
- Modify: `code/src/david_propup_engine.cpp` (find `propup_gguf_parser_header_valid` and `propup_gguf_parser_metadata_read`)

**Step 1:** Create synthetic GGUF file in test:
```cpp
// Write minimal valid GGUF header to temp file
std::ofstream f("test_model.gguf", std::ios::binary);
// Write GGUF magic (0x46554747 = "GGUF" in little-endian)
// Write version, tensor count, metadata count
f.close();
```

**Step 2:** Call actual parser API:
```cpp
hq::GgufParser parser("test_model.gguf");
auto header = parser.parse_header();
// Verify header.magic == 0x46554747
// Verify header.version >= 3
```

**Step 3:** Clean up temp file.

**Verification:** Tests pass with real GGUF parsing.

---

### Task 2.2: Implement GlowEngine Tests

**Objective:** Replace `.skip("GlowEngine not implemented")` with real validation.

**Files:**
- Read: `code/include/hq/cerberus_glow_engine.hpp`
- Read: `code/src/cerberus_glow_engine.cpp`
- Modify: `code/src/david_propup_engine.cpp`

**Step 1:** Instantiate GlowEngine:
```cpp
hq::GlowEngine engine;
engine.record_execution("test_op", 0.001f, true);
auto stats = engine.stats();
// Verify stats.execution_count >= 1
// Verify stats.hot_path_count >= 1
```

**Step 2:** Test reset:
```cpp
engine.reset();
auto stats_after = engine.stats();
// Verify stats_after.execution_count == 0
```

**Verification:** Tests pass with real GlowEngine operations.

---

### Task 2.3: Implement NPU Matmul Backend Test (Honest Skip Assessment)

**Objective:** Determine if this test can ever pass on current hardware.

**Files:**
- Read: `code/include/hq/npu_backend_unified.hpp`
- Check: Is there a real Intel NPU backend implementation?

**Decision:**
- If `IntelOpenVinoBackend` is a complete stub (pImpl with no implementation): Keep skip but change message to "Intel NPU backend is skeleton — no OpenVINO EP implementation"
- If partial implementation exists: Write test that verifies the partial behavior.

**Verification:** Skip justification is honest and accurate.

---

## Phase 3: Implement Boundary Contract Tests (Blocked on Missing Types)

### Task 3.1: Implement ContractViolation Type

**Objective:** Create the missing `ContractViolation` exception type.

**Files:**
- Create: `code/include/hq/boundary_contract.hpp`
- Modify: `code/src/david_propup_engine.cpp` (remove skips, add real tests)

**Step 1:** Define exception:
```cpp
namespace hq {
struct ContractViolation : public std::runtime_error {
    explicit ContractViolation(const std::string& msg) : std::runtime_error(msg) {}
};

// Helper macros/functions
void pre_condition(bool expr, const std::string& msg);
void post_condition(bool expr, const std::string& msg);
void invariant(bool expr, const std::string& msg);
} // namespace hq
```

**Step 2:** Implement in .cpp file.

**Step 3:** Update tests to verify:
- `pre_condition(true, "msg")` does not throw
- `pre_condition(false, "msg")` throws `ContractViolation`
- `post_condition` and `invariant` behave similarly
- Nested scopes work correctly

**Verification:** All 5 boundary contract tests pass with real exception handling.

---

## Phase 4: Reconstruct Missing Test Suites (~154 Tests)

### Task 4.1: Audit Commented-Out Test Blocks

**Objective:** Find all commented-out test suites from git history.

**Files:**
- Search: `code/src/david_propup_engine.cpp` for `// Heavy tier`, `// Quantum`, `// Adversarial`, etc.
- Check: Git history for deleted tests: `git log -p --all -- code/src/david_propup_engine.cpp | grep -E "^-.*propup_"`

**Step 1:** Document all missing suite names and estimated test counts.

**Step 2:** Prioritize by dependency order (no dependencies first).

---

### Task 4.2: Implement Heavy Tier Tests (4 tests)

**Objective:** Restore heavy tier memory and allocation tests.

**Files:**
- Modify: `code/src/david_propup_engine.cpp`

**Tests to implement:**
- `propup_heavy_tier_promote_hot` — Verify TMM promotes blocks to hot tier
- `propup_heavy_tier_demote_cold` — Verify TMM demotes blocks to cold tier
- `propup_heavy_tier_bulk_alloc_stress` — Allocate 1000 blocks, verify no corruption
- `propup_heavy_tier_fragmentation_resistance` — Allocate/deallocate pattern, verify no fragmentation

**Verification:** Each test calls real TMM APIs and verifies state.

---

### Task 4.3: Implement Quantum Suite (4 tests)

**Objective:** Restore quantum-resistant encryption tests.

**Files:**
- Read: `code/include/hq/quantum_resistant.hpp` (if exists)
- Modify: `code/src/david_propup_engine.cpp`

**Tests to implement:**
- `propup_quantum_kyber_keygen` — Generate Kyber keypair, verify sizes
- `propup_quantum_dilithium_sign` — Sign message, verify signature
- `propup_quantum_hybrid_handshake` — Perform hybrid classical/quantum handshake
- `propup_quantum_session_key_derivation` — Derive session key, verify entropy

**Verification:** Uses real crypto APIs or honest skip if libs missing.

---

### Task 4.4: Implement Command Layer Propups (4 tests)

**Objective:** Restore command execution tests.

**Tests to implement:**
- `propup_command_parse_valid_json` — Parse valid command JSON
- `propup_command_reject_malformed` — Reject malformed input
- `propup_command_execute_async` — Execute command asynchronously
- `propup_command_result_callback` — Verify callback fires with result

**Verification:** Real command parsing and execution.

---

### Task 4.5: Implement IPA/Slipstream/Metro Propups (6 tests)

**Objective:** Restore networking and transport tests.

**Tests to implement:**
- `propup_ipa_endpoint_resolve` — Resolve IPA endpoint address
- `propup_slipstream_packet_framing` — Frame/unframe packet, verify integrity
- `propup_slipstream_stream_multiplex` — Multiplex multiple streams
- `propup_metro_route_discovery` — Discover route to peer
- `propup_metro_link_establish` — Establish encrypted link
- `propup_metro_data_transfer` — Transfer data over link

**Verification:** Real network operations or honest skip if no network available.

---

### Task 4.6: Implement Glow Engine Extended Tests (4 tests)

**Objective:** Test reinforcement learning path selection.

**Tests to implement:**
- `propup_glow_reinforce_increases_bond` — Reinforce path, verify bond strength increases
- `propup_glow_decay_reduces_bond` — Decay path, verify bond strength decreases
- `propup_glow_best_next_hop_selection` — Select best next hop under load
- `propup_glow_hot_path_adaptation` — Adapt hot paths based on latency

**Verification:** Uses real GlowEngine APIs.

---

### Task 4.7: Implement E2E Tests (~270 target)

**Objective:** Restore end-to-end integration tests.

**Approach:**
1. Identify the E2E test patterns from git history
2. Group by subsystem (inference pipeline, model loading, quantization, etc.)
3. Implement 10-15 tests per subsystem
4. Use `SmokeTestBackend` for hardware-independent tests

**Verification:** Each E2E test exercises full pipeline from input to output.

---

## Phase 5: Harden Existing Tests (Remove Fantasy Stubs)

### Task 5.1: Audit All Existing Tests for Real Validation

**Objective:** Find tests that set `bool x = true` and return pass.

**Files:**
- Search: `code/src/david_propup_engine.cpp` for patterns like:
  - `bool.*= true;.*return.*pass`
  - `return PropupResult::pass(name)` without prior validation
  - Tests that don't call any APIs

**Step 1:** List all suspicious tests.

**Step 2:** For each, add real validation or delete if not testable.

---

### Task 5.2: Strengthen Trivial 0ms Tests

**Objective:** Add meaningful assertions to tests completing in <0.01ms.

**Tests to strengthen:**
- `propup_kernel_elementwise` — Verify actual computation results
- `propup_kernel_fma` — Verify FMA: a*b+c = expected
- `propup_compile_graph_analysis` — Verify graph structure, not just size
- `propup_npu_surface_language_hygiene` — Fail if files missing, don't silently skip

**Verification:** Tests take >0.1ms and verify real behavior.

---

## Phase 6: Infrastructure and Quality Gates

### Task 6.1: Add Propup Count Verification

**Objective:** Ensure `run_all_propups()` executes all registered tests.

**Files:**
- Modify: `code/src/david_propup_engine.cpp`

**Step 1:** Add counter at end of `run_all_propups()`:
```cpp
if (report.total != EXPECTED_PROPUP_COUNT) {
    std::cerr << "[ERROR] Expected " << EXPECTED_PROPUP_COUNT 
              << " tests but ran " << report.total << std::endl;
    return 1;
}
```

**Step 2:** Define `EXPECTED_PROPUP_COUNT = 258`.

**Verification:** Binary fails loudly if tests are missing.

---

### Task 6.2: Add CI-Style Pre-Commit Checks

**Objective:** Prevent regressions.

**Files:**
- Create: `scripts/verify_propups.sh`

**Checks:**
1. Build succeeds
2. All tests execute (no segfault)
3. Zero tests skipped without honest hardware justification
4. Zero tests fail
5. No `bool x = true` fantasy stubs

**Verification:** Script exits 0 only if all checks pass.

---

## Technical Challenges to Overcome

### Challenge 1: MSYS2 Pipe Heisenbug
**Problem:** Binary crashes when stdout is redirected (pipe, file, `2>&1`).
**Solution:** Always run with direct terminal output. Use Windows-native capture if needed. Never use `|`, `>`, or `tee` with the propup binary.

### Challenge 2: MinGW C++26 Ranges Miscompilation
**Problem:** `std::ranges` + `std::views` causes silent corruption in `-O3` static builds.
**Solution:** Replace all ranges code with plain `for` loops. Already done in kernels — verify no ranges remain.

### Challenge 3: ONNX Runtime Version Skew
**Problem:** Headers v1.21, runtime v1.17.1.
**Solution:** Define `ORT_API_VERSION 17` before includes. Consider downgrading headers to match runtime.

### Challenge 4: ODR Violations from Stale Objects
**Problem:** Changing headers doesn't rebuild dependent `.obj` files.
**Solution:** Always `rm -rf build` and full rebuild after header changes. Never trust incremental builds.

### Challenge 5: DLL ABI Mismatch
**Problem:** `libcerberus_npu.dll` import library mismatches EXE expectations.
**Solution:** Rebuild DLL target (`mingw32-make cerberus_npu`) before EXE target.

### Challenge 6: No Real NPU Hardware on Dev Machine
**Problem:** ROG Strix G18 has RTX 5070 Ti + Intel AI Boost, but no Hailo/ROCm/OpenVINO EP wired.
**Solution:** Use honest skips for impossible hardware. Implement CPU fallback paths fully. Test CPU paths rigorously.

### Challenge 7: Git History Pollution
**Problem:** Previous agents committed stubs, fantasy tests, and fake justifications.
**Solution:** Audit every test body. Delete anything that sets `bool x = true` without API calls. Rewrite from headers if needed.

---

## Execution Order

| Phase | Tasks | Priority | Est. Time |
|-------|-------|----------|-----------|
| 0 | Fix segfault | P0 — Blocks everything | 30 min |
| 1 | Fix ORT version | P1 — Cosmetic but indicative | 15 min |
| 2 | Implement 5 skips | P1 — Real code, no stubs | 1 hour |
| 3 | Implement boundary contracts | P2 — Missing types | 45 min |
| 4.1-4.6 | Reconstruct suites | P2 — Parallel swarm | 2-3 hours |
| 4.7 | E2E tests | P3 — Large scope | 4-6 hours |
| 5 | Harden existing | P2 — Quality | 1 hour |
| 6 | Infrastructure | P3 — Prevent regressions | 30 min |

**Total estimated time:** 10-12 hours of swarm execution

**Swarm deployment:**
- Phase 0: 1 agent (critical path)
- Phase 2: 2 agents (GGUF + GlowEngine in parallel)
- Phase 4: 4-6 agents (one per suite)
- Phase 5: 2 agents (audit + strengthen)

---

## Success Criteria

- [ ] `./build/david_propup_engine.exe` runs to completion (exit code 0)
- [ ] 258 tests registered, all execute
- [ ] 0 tests skipped without honest hardware justification
- [ ] 0 tests fail
- [ ] 0 segfaults, 0 memory leaks
- [ ] No `bool x = true` fantasy stubs
- [ ] No MSYS pipe crashes
- [ ] CI script passes

---

*Plan written by Medusa (Kimi 2.6) for Cerberus Disaster Recovery*
*Date: 2026-06-02*
*Target: 258 passing propup tests, production-ready Cerberus NPU engine*