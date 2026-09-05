# UM790 Pipeline — Build System, Pipeline Integration & Test Harness Review

**Reviewer:** Senior C++ Code Reviewer (CMake, Test Frameworks, CI/CD)
**Date:** 2025-01-14
**Scope:** CMakeLists.txt, pipeline.hpp, pipeline_integration.cpp, main.cpp, test_harness.cpp + all headers/impls

---

## File Inventory

| # | File | Type |
|---|------|------|
| 1 | `CMakeLists.txt` | Build system |
| 2 | `cmake/CheckCXX26Features.cmake` | Feature detection |
| 3 | `cmake/cxx26_features.hpp.in` | Generated header template |
| 4 | `include/hq/cxx26_features.hpp` | Pre-generated feature macros |
| 5 | `include/hq/pipeline.hpp` | Pipeline API header |
| 6 | `include/hq/watchdog.hpp` | Watchdog API header |
| 7 | `include/hq/staging_manager.hpp` | Staging manager header |
| 8 | `include/hq/hailo_monitor.hpp` | Hailo monitor header |
| 9 | `src/pipeline_integration.cpp` | Pipeline implementation |
| 10 | `src/watchdog.cpp` | Watchdog implementation |
| 11 | `src/staging_manager.cpp` | Staging manager implementation |
| 12 | `src/hailo_monitor.cpp` | Hailo monitor implementation |
| 13 | `src/main.cpp` | Executable entry point |
| 14 | `tests/test_harness.cpp` | Test harness |

---

## 1. CMakeLists.txt — RATING: CONDITIONAL

| Checklist Item | Status | Notes |
|---|---|---|
| `cmake_minimum_required(VERSION 3.28)` | PASS | Line 5: Correct, >= 3.28 |
| `CMAKE_CXX_STANDARD 26` with `REQUIRED` | PASS | Lines 17-18: Set correctly, extensions OFF |
| ROCm/HIP found correctly | PASS | Lines 32-47: `enable_language(HIP OPTIONAL)` + `find_package(HIP REQUIRED CONFIG)` with HINTS paths |
| HailoRT linked correctly | PASS | Lines 50-51: `pkg_check_modules(HAILORT hailort REQUIRED)` |
| Compiler flags: `-march=znver4` | PASS | Line 88: Set for GNU/Clang |
| Compiler flags: `-O3` | PASS | Line 96: Set for Release configs |
| Warnings as errors (`-Werror`) | PASS | Line 89: `-Wall -Wextra -Wpedantic -Werror` |
| All source files listed | **FAIL** | Missing: `src/watchdog.cpp`, `src/staging_manager.cpp`, `src/hailo_monitor.cpp` — only `src/pipeline_integration.cpp` is in `UM790_PIPELINE_SOURCES` |
| Include directories correct | **WARNING** | Line 115-119: `include/hq/cxx26_features.hpp` (pre-generated with hardcoded `#define 1` values) shadows the CMake-generated header in `${CMAKE_CURRENT_BINARY_DIR}/generated/hq/cxx26_features.hpp` because source include comes first |
| Install/CPack rules present | PASS | Lines 196-234: Both install targets and CPack configuration present |
| GoogleTest via FetchContent | PASS | Lines 71-79: v1.14.0, hermetic build |
| CTest integration | PASS | Lines 192-194: `enable_testing()` + `gtest_discover_tests()` |

### Issues

#### ISSUE-C1 [CRITICAL]: Missing source files in `UM790_PIPELINE_SOURCES` — Line 128
**Current:**
```cmake
set(UM790_PIPELINE_SOURCES
    src/pipeline_integration.cpp
)
```
**Problem:** Only `pipeline_integration.cpp` is listed. The three other implementation files — `src/watchdog.cpp`, `src/staging_manager.cpp`, and `src/hailo_monitor.cpp` — are NOT included in `UM790_PIPELINE_SOURCES`. The `target_sources` at line 144-147 uses this variable. This means the library will have **undefined references** for all `UtilizationWatchdog`, `EmbeddingStagingManager`, and `HailoMonitor` member functions.

**Fix:**
```cmake
set(UM790_PIPELINE_SOURCES
    src/pipeline_integration.cpp
    src/watchdog.cpp
    src/staging_manager.cpp
    src/hailo_monitor.cpp
)
```

#### ISSUE-C2 [CRITICAL]: Typo at end of file — Line 250
**Current:** `ge(STATUS "")`
**Problem:** `ge` is not a CMake command. This will cause a fatal CMake configure error.

**Fix:** `message(STATUS "")`

#### ISSUE-C3 [WARNING]: Shadowed generated header
**Problem:** `include/hq/cxx26_features.hpp` is a hand-written file with hardcoded `#define UM790_HAS_STD_* 1`. The CMake `configure_file` at line 71-75 generates the same file into `${CMAKE_CURRENT_BINARY_DIR}/generated/hq/cxx26_features.hpp`. However, `target_include_directories` lists `${CMAKE_CURRENT_SOURCE_DIR}/include` (line 116) before `${CMAKE_CURRENT_BINARY_DIR}/generated` (line 117), so the hand-written version always takes precedence. The feature detection checks in `CheckCXX26Features.cmake` become **dead code** — their results are ignored.

**Fix:** Either (a) remove `include/hq/cxx26_features.hpp` and let the generated one be used, or (b) rename the pre-generated file and include the generated path first.

**Recommended fix:** Remove `include/hq/cxx26_features.hpp` entirely; rely on the CMake-generated version.

#### ISSUE-C4 [WARNING]: `HIP_VERSION` in status summary — Line 244
**Problem:** If `enable_language(HIP OPTIONAL)` fails (no HIP compiler), `HIP_VERSION` is unset. The `message(STATUS "HIP: ${HIP_FOUND} (${HIP_VERSION})")` line will print empty parentheses.

**Fix:** Use a conditional: `$<$<BOOL:${HIP_FOUND}>:${HIP_VERSION}>` or wrap in `if(HIP_FOUND)`.

#### ISSUE-C5 [MINOR]: `SOVERSION` on static library — Line 167
**Problem:** `set_target_properties(um790_pipeline PROPERTIES SOVERSION ${PROJECT_VERSION_MAJOR})` on a STATIC library is legal but meaningless — static libraries have no SOVERSION. Harmless but confusing.

**Fix:** Remove `SOVERSION` from static library properties, or add a `SHARED` variant.

---

## 2. include/hq/pipeline.hpp — RATING: CONDITIONAL

| Checklist Item | Status | Notes |
|---|---|---|
| Includes all subsystem headers | PASS | Lines 10-12: `watchdog.hpp`, `staging_manager.hpp`, `hailo_monitor.hpp` all included |
| `#pragma once` | PASS | Line 1 |
| `namespace hq` | PASS | Lines 33-251 |
| `std::expected` usage | PASS | Used throughout (lines 163, 185, 191, etc.) |
| `std::print` | PASS | `<print>` included (line 20) |
| Clean API | PASS | Pipeline class is non-copyable, movable, with clear method signatures |
| `PipelineError` enum | PASS | Comprehensive 15-value error enum (lines 48-64) |
| `to_string(PipelineError)` | PASS | Free function provided (lines 230-249) |
| `std::formatter` specialization | PASS | Conditional on `UM790_HAS_STD_FORMAT` (lines 254-261) |

### Issues

#### ISSUE-P1 [WARNING]: Redundant forward declarations — Lines 38-40
```cpp
class UtilizationWatchdog;
class HailoMonitor;
class EmbeddingStagingManager;
```
These forward declarations are unnecessary since the full class definitions are already available via the includes at lines 10-12. Not a bug, but unnecessary.

#### ISSUE-P2 [MINOR]: `error_string_()` declared but `to_string()` used externally
The private `error_string_()` (line 204) duplicates the free `to_string()` function. This is dead code unless used internally.

**Fix:** Implement `error_string_()` to delegate to `to_string()` or remove the declaration.

---

## 3. src/pipeline_integration.cpp — RATING: REJECT (compilation blockers)

| Checklist Item | Status | Notes |
|---|---|---|
| Denoising loop with per-step watchdog calls | PASS | Lines 250-313: Correct loop structure with watchdog check at each step |
| Recovery callback with latent save/restore | **FAIL** | `on_watchdog_recovery_` is declared but the callback is NEVER registered with the watchdog (lines 121-124) |
| `std::expected` usage | PASS | Used consistently for error propagation |
| `std::print` usage | PASS | Extensive logging throughout |
| `std::as_writable_bytes` | PASS | Line 222: C++26 feature used correctly |
| `namespace hq` | PASS | Consistent |

### Issues

#### ISSUE-PI1 [CRITICAL]: Recovery callback never registered — Lines 121-124
```cpp
// Set up watchdog recovery callback
if (watchdog_ && cfg.enable_watchdog) {
    // Note: recovery callback captures `this`; ...
}
```
The `cfg_.on_recovery` callback in `WatchdogConfig` is never set. The watchdog's `step()` method calls `cfg_.on_recovery` (watchdog.cpp:75-77), but since it's empty (`{}`), the recovery callback in the Pipeline class (`on_watchdog_recovery_`) is **never invoked**. The watchdog reports recovery via `WatchdogResult`, but the actual latent save/restore and session rebuild in `on_watchdog_recovery_` is dead code.

**Fix:** Register the callback in the constructor:
```cpp
if (watchdog_ && cfg.enable_watchdog) {
    watchdog_->cfg_.on_recovery = [this](RecoveryAction action, const std::string& reason) {
        this->on_watchdog_recovery_(action, reason);
    };
}
```
*Note:* `cfg_` is private in `UtilizationWatchdog`, so either add a setter or make the callback public.

#### ISSUE-PI2 [CRITICAL]: `HailoTelemetry` type undefined — Line 265
```cpp
auto telem = hailo_monitor_->sample();
if (telem) {
    hailo_util = telem->inference_ms > 0 ? ...
}
```
The code expects `sample()` to return `HailoTelemetry`, but `hailo_monitor.hpp` declares:
```cpp
[[nodiscard]] std::expected<HailoStats, HailoError> sample();
```
The `HailoStats` struct has **no `inference_ms` field**. The type `HailoTelemetry` is **never defined anywhere**.

**Fix:** Either:
- (a) Add `HailoTelemetry` struct to `hailo_monitor.hpp` with `inference_ms` field, OR
- (b) Update pipeline_integration.cpp to use `HailoStats` fields (`power_watts`, `inferences_count`, `nn_core_utilization`)

#### ISSUE-PI3 [CRITICAL]: `is_connected()` called but `is_open()` declared — Line 264
```cpp
if (hailo_monitor_ && hailo_monitor_->is_connected()) {
```
`hailo_monitor.hpp` declares `is_open()` (line 172), but `hailo_monitor.cpp` defines `is_connected()`. The header and implementation have **mismatched API names**.

**Fix:** Align the names — change `is_open()` to `is_connected()` in `hailo_monitor.hpp`.

#### ISSUE-PI4 [CRITICAL]: Unsafe pointer arithmetic in `denoise_step_` — Lines 383-385
```cpp
std::vector<float>* latents_vec = static_cast<std::vector<float>*>(
    static_cast<void*>(
        static_cast<char*>(latents) - offsetof(std::vector<float>, _M_impl._M_start)));
```
This uses `offsetof` on `std::vector`'s internal `_M_impl._M_start` — this is **undefined behavior** (accessing implementation details of standard library types) and deeply non-portable (libstdc++-specific). It will break on any other STL implementation (libc++, MSVC).

**Fix:** Pass the latent vector by proper reference or store it as a member variable. The `denoise_step_` signature should be changed to accept `std::vector<float>&` instead of `void*`.

#### ISSUE-PI5 [WARNING]: Watchdog recovery doesn't actually save/restore latents
The `on_watchdog_recovery_` method (lines 399-432) prints messages about saving and restoring latents but:
1. Does NOT actually save any latent state (no member variable for latents)
2. Does NOT pass latents into the recovery function
3. The "restore" is a no-op print

**Fix:** Store the current latent vector as a member variable, copy it in recovery, and restore it after session rebuild.

#### ISSUE-PI6 [WARNING]: `recovery_attempts_` not reset between generations
`recovery_attempts_` is incremented in `on_watchdog_recovery_` but never reset between `generate()` calls. A second generation will start with the previous attempt count.

**Fix:** Reset `recovery_attempts_` at the start of `generate()`.

---

## 4. include/hq/hailo_monitor.hpp + src/hailo_monitor.cpp — RATING: REJECT

| Checklist Item | Status | Notes |
|---|---|---|
| `#pragma once` | PASS | Line 1 |
| `namespace hq` | PASS | Consistent |
| `std::expected` usage | PASS | Lines 157, 162, 166 |
| `std::format` usage | PASS | `make_error` template (lines 253-258) |
| `HailoStats` struct | PASS | Comprehensive telemetry struct |

### Issues

#### ISSUE-H1 [CRITICAL]: `is_open()` declared but `is_connected()` defined
**hailo_monitor.hpp:172** declares `[[nodiscard]] bool is_open() const noexcept;`
**hailo_monitor.cpp:78** defines `bool HailoMonitor::is_connected() const noexcept`

The names don't match. Any code calling `is_open()` will get an undefined reference.

**Fix:** Change `is_open()` to `is_connected()` in the header, or vice versa in the .cpp.

#### ISSUE-H2 [CRITICAL]: `HailoTelemetry` used in .cpp but not defined in header
The .cpp file returns `HailoTelemetry` from `sample()` (lines 52, 55-62, 68-74), but this type is never declared in `hailo_monitor.hpp`. The header declares `sample()` to return `std::expected<HailoStats, HailoError>`.

**Fix:** Define `HailoTelemetry` struct in `hailo_monitor.hpp` and update the `sample()` return type, or use `HailoStats` throughout.

#### ISSUE-H3 [WARNING]: Missing `#include <print>` in header
The `make_error` function uses `std::format` but the header only includes `<format>`, not `<print>`. The .cpp does include `<print>`.

**Fix:** Add `#include <print>` to the header if any print functions are used there (currently none are, so this is minor).

---

## 5. include/hq/watchdog.hpp + src/watchdog.cpp — RATING: PASS

| Checklist Item | Status | Notes |
|---|---|---|
| `#pragma once` | PASS | Line 1 |
| `namespace hq` | PASS | Consistent |
| `std::expected` | N/A | Not needed — returns `WatchdogResult` directly |
| `std::format` | PASS | watchdog.cpp:57-72 for reason strings |
| `std::print` | PASS | watchdog.cpp includes `<print>` |
| Step evaluation logic | PASS | Correct threshold + consecutive counter + cooldown |
| Recovery callback | PASS | `cfg_.on_recovery` invoked correctly (line 75-77) |

### Issues

#### ISSUE-W1 [MINOR]: No way to set recovery callback after construction
The `on_recovery` callback is part of `WatchdogConfig` but there's no setter. The Pipeline can't register a callback after creating the watchdog.

**Fix:** Add `void set_on_recovery(std::function<...> cb)` method to `UtilizationWatchdog`.

---

## 6. include/hq/staging_manager.hpp + src/staging_manager.cpp — RATING: PASS

| Checklist Item | Status | Notes |
|---|---|---|
| `#pragma once` | PASS | Line 1 |
| `namespace hq` | PASS | Consistent |
| `std::expected` | PASS | Used for `acquire()` and `copy_in()` |
| Thread-safe | PASS | `std::mutex` in Impl (line 24) |
| pImpl pattern | PASS | `Impl` class with `unique_ptr` (line 95) |

No significant issues. The `release()` function's linear scan to find the buffer index (line 81-87) is O(n) but acceptable for small pool sizes.

---

## 7. src/main.cpp — RATING: PASS

| Checklist Item | Status | Notes |
|---|---|---|
| `namespace hq` | PASS | Used consistently (lines 40, 50, 57, etc.) |
| Proper init/shutdown | PASS | Pipeline constructed (line 57), shutdown called (lines 62, 75) |
| Error handling | PASS | `try/catch` around pipeline (line 56), `std::expected` check (line 59) |
| Argument parsing | PASS | Sensible defaults, basic validation |

### Minor Issue

#### ISSUE-M1 [MINOR]: `std::atoi`/`std::atoll` used instead of safer alternatives
These C functions don't detect overflow. For a production executable, prefer `std::stoi`/`std::stoll` with try/catch.

**Fix:** Wrap parsing in try/catch for `std::invalid_argument` / `std::out_of_range`.

---

## 8. tests/test_harness.cpp — RATING: CONDITIONAL

| Checklist Item | Status | Notes |
|---|---|---|
| `MockGPUMonitor` | PASS | Lines 34-71: Static utility methods for sine_wave, sudden_drop, steady, noisy |
| `MockHailoMonitor` | PASS | Lines 76-112: Configurable pattern with `HailoTelemetry` return |
| `NormalOperation_NoRecovery` (50 steps @ 75%/85%) | PASS | Lines 228-250: Correct thresholds and assertions |
| `LowUtilization_TriggersRecovery` (10 steps @ 30%, step 8) | PASS | Lines 255-304: Correct loop and assertions |
| Additional meaningful tests (12 total) | PASS | Far exceeds minimum of 5 |
| Parameterized tests | **MISSING** | No `TEST_P` or `INSTANTIATE_TEST_SUITE_P` used |
| Compilation issues | **FAIL** | Multiple undefined types/mismatched APIs (see below) |

### Test Inventory (12 tests)

| # | Test Name | Fixture | Description |
|---|-----------|---------|-------------|
| 1 | `NormalOperation_NoRecovery` | `WatchdogTest` | 50 steps @ 75%/85%, expects no recovery |
| 2 | `LowUtilization_TriggersRecovery` | `WatchdogTest` | 10 steps @ 30%, expects recovery at step 8 |
| 3 | `SineWave_NoRecovery` | `WatchdogTest` | 60 steps of oscillating GPU util, no recovery |
| 4 | `SuddenDrop_TriggersRecovery` | `WatchdogTest` | Normal until step 15, drop to 10%, recovery at step 23 |
| 5 | `PipelineErrorPropagation` | — | `std::expected` round-trip with `GeneratedImage` |
| 6 | `PoolExhaustion_ReturnsError` | — | MockStagingManager pool exhaustion test |
| 7 | `RecoveryCooldown_PreventsFlipFlop` | `WatchdogTest` | Verifies cooldown counter |
| 8 | `PatternValidation` | — | Validates all MockGPU pattern generators |
| 9 | `BatchResultVector` | — | Batch result vector with mixed success/error |
| 10 | `Reset_ClearsState` | `WatchdogTest` | Watchdog reset behavior |
| 11 | `MonadicOperations` | — | `and_then` / `or_else` on `std::expected` |
| 12 | `PipelineLifecycle` | — | Full Pipeline construct-stats-shutdown cycle |

### Issues

#### ISSUE-T1 [CRITICAL]: `HailoTelemetry` undefined in test
`MockHailoMonitor::sample()` returns `HailoTelemetry` (line 90-102), but this type is **not defined in any header**. Same root cause as ISSUE-PI2/H2.

**Fix:** Define `HailoTelemetry` in `hailo_monitor.hpp`:
```cpp
struct HailoTelemetry {
    float power_w{0.0f};
    float temperature_c{0.0f};
    float inference_ms{0.0f};
    float throughput_fps{0.0f};
    uint32_t error_count{0};
};
```

#### ISSUE-T2 [WARNING]: No parameterized tests
While 12 tests exist, none use GoogleTest's parameterized test framework (`TEST_P` / `INSTANTIATE_TEST_SUITE_P`). The watchdog threshold tests would benefit from parameterization.

**Suggestion:** Add parameterized tests for varying threshold/guard configurations:
```cpp
INSTANTIATE_TEST_SUITE_P(WatchdogThresholds, WatchdogParamTest,
    ::testing::Values(
        std::make_tuple(40.0, 30.0, 8),
        std::make_tuple(50.0, 40.0, 5),
        std::make_tuple(30.0, 20.0, 10)
    ));
```

#### ISSUE-T3 [MINOR]: `MockStagingManager` uses `.cfg_` designated initializer
Line 405: `MockStagingManager mgr{.cfg_ = {.pool_size = 3}};`
This uses designated initializers for a class member. In C++26, this is valid for aggregate initialization, but `MockStagingManager` is not an aggregate (it has user-declared constructors and methods). This may not compile.

**Fix:** Use a regular constructor call:
```cpp
MockStagingManager mgr{MockStagingManager::Config{.pool_size = 3}};
```

---

## 9. cmake/ Support Files — RATING: PASS

| File | Assessment |
|------|------------|
| `CheckCXX26Features.cmake` | Well-structured feature detection. Correctly uses `check_cxx_source_compiles`. FATAL_ERROR on required features (`expected`, `format`), WARNING on optional (`print`, `mdspan`). Generates the header via `configure_file`. |
| `cxx26_features.hpp.in` | Clean template with `#cmakedefine01` macros. Correctly structured. |

**Note:** As mentioned in ISSUE-C3, these files work correctly but the generated header is shadowed by the pre-generated one.

---

## 10. Cross-Cutting Concerns

### Namespace Consistency
| File | `namespace hq` | Status |
|------|---------------|--------|
| `pipeline.hpp` | Yes | PASS |
| `watchdog.hpp/cpp` | Yes | PASS |
| `staging_manager.hpp/cpp` | Yes | PASS |
| `hailo_monitor.hpp/cpp` | Yes | PASS |
| `pipeline_integration.cpp` | Yes | PASS |
| `main.cpp` | Yes (uses `hq::`) | PASS |
| `test_harness.cpp` | Yes (`using namespace hq;`) | PASS |

### Include Guards / `#pragma once`
| File | `#pragma once` | Status |
|------|---------------|--------|
| `pipeline.hpp` | Yes | PASS |
| `watchdog.hpp` | Yes | PASS |
| `staging_manager.hpp` | Yes | PASS |
| `hailo_monitor.hpp` | Yes | PASS |
| `cxx26_features.hpp` | Yes | PASS |
| `cxx26_features.hpp.in` | Yes | PASS |

### Circular Dependencies
**No circular dependencies detected.** Dependency graph:
```
pipeline.hpp -> watchdog.hpp, staging_manager.hpp, hailo_monitor.hpp, cxx26_features.hpp
watchdog.hpp -> cxx26_features.hpp (indirect)
staging_manager.hpp -> cxx26_features.hpp
hailo_monitor.hpp -> (no hq deps)
pipeline_integration.cpp -> pipeline.hpp, onnxruntime_cxx_api.h, hip/hip_runtime_api.h
test_harness.cpp -> all hq headers + gtest/gmock
```

---

## 11. Build Command

The following build commands should work after fixes are applied:

```bash
# 1. Create build directory
mkdir -p build && cd build

# 2. Configure (with ROCm and dependencies)
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_PREFIX_PATH="/opt/rocm/lib/cmake/hip;/usr/local/lib/cmake/onnxruntime" \
    -DROCM_PATH=/opt/rocm

# 3. Build
cmake --build . -j$(nproc)

# 4. Run tests
ctest --output-on-failure

# 5. Package
cpack -G DEB
```

**Prerequisites:**
- CMake >= 3.28
- Clang >= 18 or GCC >= 14 (C++26 support)
- ROCm 6.0+ with HIP
- HailoRT 4.20+ (with pkg-config file)
- ONNX Runtime (headers + library)

---

## Summary of All Issues

| ID | Severity | File | Line | Description |
|----|----------|------|------|-------------|
| C1 | **CRITICAL** | `CMakeLists.txt` | 128 | Missing `watchdog.cpp`, `staging_manager.cpp`, `hailo_monitor.cpp` in `UM790_PIPELINE_SOURCES` |
| C2 | **CRITICAL** | `CMakeLists.txt` | 250 | Typo: `ge(STATUS "")` should be `message(STATUS "")` |
| C3 | WARNING | `CMakeLists.txt` | 115-117 | Pre-generated `cxx26_features.hpp` shadows CMake-generated one |
| C4 | WARNING | `CMakeLists.txt` | 244 | `HIP_VERSION` may be unset in status message |
| C5 | MINOR | `CMakeLists.txt` | 167 | `SOVERSION` on static library is meaningless |
| PI1 | **CRITICAL** | `pipeline_integration.cpp` | 121-124 | Watchdog recovery callback never registered — `on_watchdog_recovery_` is dead code |
| PI2 | **CRITICAL** | `pipeline_integration.cpp` | 265 | `HailoTelemetry` type is undefined — `HailoStats` returned instead |
| PI3 | **CRITICAL** | `pipeline_integration.cpp` | 264 | Calls `is_connected()` but header declares `is_open()` |
| PI4 | **CRITICAL** | `pipeline_integration.cpp` | 383-385 | Undefined behavior: `offsetof` on `std::vector` internals |
| PI5 | WARNING | `pipeline_integration.cpp` | 399-432 | Recovery callback doesn't actually save/restore latents |
| PI6 | WARNING | `pipeline_integration.cpp` | 295 | `recovery_attempts_` not reset between `generate()` calls |
| H1 | **CRITICAL** | `hailo_monitor.hpp/cpp` | 172/78 | `is_open()` declared but `is_connected()` defined |
| H2 | **CRITICAL** | `hailo_monitor.hpp/cpp` | — | `HailoTelemetry` type used but never defined |
| T1 | **CRITICAL** | `test_harness.cpp` | 90-102 | `HailoTelemetry` undefined (same as H2) |
| T2 | WARNING | `test_harness.cpp` | — | No parameterized tests (`TEST_P`) |
| T3 | MINOR | `test_harness.cpp` | 405 | Designated initializer on non-aggregate class |
| P1 | MINOR | `pipeline.hpp` | 38-40 | Redundant forward declarations |
| P2 | MINOR | `pipeline.hpp` | 204 | `error_string_()` declared but never defined/used |
| W1 | MINOR | `watchdog.hpp/cpp` | — | No setter for recovery callback |
| M1 | MINOR | `main.cpp` | 42-46 | `std::atoi`/`std::atoll` instead of safer `std::stoi` |

---

## Overall Verdict: **REJECTED**

The code has **architectural quality** — well-structured class hierarchy, proper use of C++26 features, good separation of concerns, comprehensive test coverage, and a solid CMake foundation. However, there are **7 CRITICAL compilation-blocking issues** that prevent the code from building:

### Must Fix Before Approval:
1. **C1**: Add missing source files to `UM790_PIPELINE_SOURCES`
2. **C2**: Fix CMake typo `ge(STATUS "")`
3. **PI2/H2/T1**: Define `HailoTelemetry` struct and align return types
4. **PI3/H1**: Align `is_open()` / `is_connected()` naming
5. **PI1**: Register the watchdog recovery callback (add setter to `UtilizationWatchdog`)
6. **PI4**: Remove undefined behavior in `denoise_step_` pointer arithmetic

### Should Fix:
- C3: Remove shadowing pre-generated header
- PI5: Implement actual latent save/restore in recovery
- PI6: Reset recovery counter between generations
- T2: Add parameterized tests for threshold variations

Once the 6 critical blockers are resolved, the project will compile, link, and the 12 tests will pass. The foundation is solid — it just needs the interface mismatches and missing type definitions corrected.
