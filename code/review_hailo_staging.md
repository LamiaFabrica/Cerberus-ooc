# Code Review Report: HailoMonitor & PinnedStagingBuffer Implementation

**Review Date:** 2025-01-14
**Reviewer:** Senior C++ Code Reviewer (AI Accelerator / HIP / HailoRT Specialization)
**Scope:** `hailo_monitor.hpp`, `hailo_monitor.cpp`, `staging_manager.hpp`, `staging_manager.cpp`, `pipeline.hpp`, `pipeline_integration.cpp`, `CMakeLists.txt`, `test_harness.cpp`
**Overall Verdict:** **REJECTED** — Critical header/implementation mismatches, missing files, and unimplemented core functionality prevent compilation and correct operation.

---

## Executive Summary

The codebase has a **severe structural problem**: the header files declare one API, but the implementation files implement an entirely different, incompatible API. Additionally, two files (`pinned_staging.hpp`, `pinned_staging.cpp`) are **completely missing**. The staging manager that exists does **not** use HIP pinned memory at all, defeating the stated purpose of replacing XNACK zero-copy. The CMake build file contains a typo that will cause configuration to fail.

**Blocker count:** 7 critical, 8 major, 12 minor/warnings.

---

## File-by-File Review

### 1. `include/hq/hailo_monitor.hpp` — Rating: **CONDITIONAL**

| Aspect | Status | Notes |
|--------|--------|-------|
| Dual-indicator constants | OK | `HAILO8L_FUSED_WEIGHT_POWER = 0.5f`, `HAILO8L_FUSED_WEIGHT_INFERENCE = 0.5f` correctly defined |
| Power indicator formula | Needs fix | Declared `(power - 0.5) / 5.5 * 100` logic is documented but **not implemented** in `.cpp` |
| DMA stall thresholds | OK | `70%` power + `<30%` inference thresholds correctly declared |
| Sensor mismatch detection | OK | `SensorMismatch` error code present; `detect_sensor_mismatch()` declared |
| `std::expected` usage | OK | All fallible operations return `std::expected<T, HailoError>` |
| RAII (destructor) | OK | `~HailoMonitor() noexcept` declared; move semantics correct |
| HailoRT forward decl | OK | `namespace hailort { class Device; }` properly forward-declared |
| `HailoStats` struct | OK | Well-structured with `nn_core_utilization`, dual raw indicators, timestamps |

**Issues found:**

| Line | Severity | Issue | Fix |
|------|----------|-------|-----|
| 44-53 | Minor | Magic numbers for power thresholds should be `constexpr` validated at compile time (e.g., `static_assert(HAILO8L_IDLE_POWER_W < HAILO8L_ACTIVE_POWER_W)`) | Add `static_assert` validation block |
| 53 | Minor | `HAILO8L_EXPECTED_INFERENCES_PER_SEC = 60` is a coarse average; should be model-configurable | Add comment noting this is a conservative default; consider making it a runtime parameter with per-model profiles |
| 132 | Info | `DeviceDeleter` uses `noexcept` but actual `hailort::Device` destructor may throw; consider `noexcept(false)` or swallowing | Document assumption that HailoRT cleanup doesn't throw |
| 236 | Minor | `have_prev_inferences_` bool flag is redundant; can use `prev_timestamp_.time_since_epoch().count() == 0` | Simplify by removing `have_prev_inferences_` or keep for clarity — **style preference** |

**Verdict:** The header is well-designed and declares the correct dual-indicator API. It cannot compile because no matching implementation exists.

---

### 2. `src/hailo_monitor.cpp` — Rating: **REJECT**

| Aspect | Status | Notes |
|--------|--------|-------|
| Header match | **FAIL** | Implements a completely different API than the header declares |
| Dual-indicator fusion | **FAIL** | Not implemented at all |
| `HailoRT` API calls | **FAIL** | Only stub comments; no actual HailoRT calls |
| `std::expected` | Partial | Uses `std::expected` but with wrong types |
| RAII | Partial | `Impl::~Impl()` calls `close()` which is correct, but `HailoMonitor` destructor uses `=default` with pimpl — OK pattern |

**Critical Issues:**

| Line | Severity | Issue | Fix |
|------|----------|-------|-----|
| 41 | **CRITICAL** | Constructor signature `HailoMonitor(int device_index)` does **NOT** match header which declares `HailoMonitor()` (default ctor). Header also declares `open(const std::string&)` for late binding. | **Rewrite constructor** to match header: `HailoMonitor::HailoMonitor() = default;` Implement `open(const std::string&)` as the connection method. |
| 52 | **CRITICAL** | Return type `HailoTelemetry` does **NOT** exist anywhere in the codebase. Header declares `std::expected<HailoStats, HailoError>`. | Replace `HailoTelemetry` with `HailoStats`. Implement the full dual-indicator fusion logic. |
| 52 | **CRITICAL** | `sample()` is declared `const` in implementation but **non-const** in header (line 162). Also mutates `prev_inferences_` which should be `mutable` if `const`. | Match header signature exactly. Mark `prev_inferences_`, `have_prev_inferences_`, `prev_timestamp_` as `mutable` if `sample()` remains `const`, OR remove `const` from `sample()`. |
| 78 | **CRITICAL** | `is_connected()` does **NOT** match header which declares `is_open() const noexcept`. | Rename to `is_open()`. Return `bool(device_)` or track open state. |
| 14-35 | **CRITICAL** | `Impl` class uses stub HailoRT integration (just `connected_` bool). No actual `hailort::Device` usage, no `Device::scan_pcie()`, no `get_power_measurement()`. | Implement real HailoRT calls behind `#ifdef UM790_HAS_HAILORT` / `#else` skeleton blocks. |
| 41-49 | **CRITICAL** | `HailoMonitor` declares move ctor/assignment as `=default` but header declares them as user-declared; implementation file defines them as `=default` which is fine, BUT the pimpl `Impl` class makes this work only if `Impl` is complete at the point of definition. Here `Impl` is complete before the `=default` — OK. | No change needed for move operations. |
| 55-74 | **CRITICAL** | `sample()` returns synthetic hardcoded `HailoTelemetry{}` with no actual sensor reading. The dual-indicator fusion formula `(power - 0.5) / 5.5 * 100` is **not implemented**. | Implement the full fusion pipeline: `read_power_watts()` -> normalize to %, `read_inference_count()` -> compute delta -> normalize to %, `fuse_indicators()` -> weighted average, `detect_sensor_mismatch()` -> sanity check. |
| 80 | **CRITICAL** | Missing all declared public API methods: `open()`, `hard_reset()`, `close()`, `device_id()`, all tunable threshold getters/setters. | Implement all declared methods. |
| 93 | **CRITICAL** | `HailoError` comparison in tests uses `HailoTelemetry` which doesn't exist in header. | Define `HailoStats` in header (already done); use it consistently. |

**The two implementations conflict:** The header declares a sophisticated dual-indicator monitoring class; the `.cpp` implements a trivial stub with a different interface. **The header is canonical** — it represents the intended design. The `.cpp` must be rewritten.

**Recommended canonical version:** Keep `hailo_monitor.hpp` as the canonical header. Rewrite `hailo_monitor.cpp` from scratch to match.

---

### 3. `include/hq/staging_manager.hpp` — Rating: **CONDITIONAL**

| Aspect | Status | Notes |
|--------|--------|-------|
| `std::expected` | OK | `acquire()` and `copy_in()` return `std::expected` |
| `StagingConfig` | Partial | Has `.pinned = true` and `.alignment = 256` fields |
| `std::mdspan` alias | Present but unused | `staging_view<T>` defined but never used in code |
| RAII | OK | Destructor declared; move semantics declared |
| Thread-safety intent | OK | `acquire()`/`release()` designed for mutex protection |

**Issues found:**

| Line | Severity | Issue | Fix |
|------|----------|-------|-----|
| 46 | Major | `StagingConfig::pinned{true}` is a **dead field** — the implementation completely ignores it. No HIP pinned memory is ever allocated. | Either implement pinned memory in `Impl` or remove the field to avoid misleading API users. |
| 49 | Major | `StagingConfig::alignment{256}` is **ignored** — buffers are allocated via `std::vector` with no alignment guarantee. HIP requires 256-byte alignment for `hipMemcpyAsync`. | Use `std::aligned_alloc` or `hipHostMalloc` with alignment; OR validate `hipHostMalloc` already returns aligned memory. |
| 60-96 | Major | Class is named `EmbeddingStagingManager` but manages **generic** byte buffers with no embedding-specific logic. The name is misleading. | Rename to `StagingBufferPool` or add embedding-specific methods (e.g., `stage_tensor()`, `get_mdspan_view()`). |
| 74 | Minor | `acquire()` returns `StagingBuffer` by value — contains two `std::span`s which are trivially copyable, OK. But the span points into `Impl::buffers_` which may reallocate if `buffers_` is mutated. Since `buffers_` never grows after construction, this is safe. | Add `static_assert` that `StagingBuffer` is trivially relocatable, or document the non-owning reference semantics. |
| 78 | Minor | `release(const StagingBuffer& buf) noexcept` takes `const&` but the implementation does a linear search over all buffers. If buffer not found, silently returns — no error indication. | Consider returning `bool` indicating success/failure, or at minimum `assert`/log when buffer not found. |
| 84-85 | Minor | `copy_in()` does not validate that `dst` was acquired from this pool — a `StagingBuffer` from another pool could be passed. | Add a pool-ownership validation (e.g., check pointer range or tag). |
| 93 | Minor | `total_capacity()` does not account for buffers currently in use — name implies it returns remaining capacity, not total. | Rename to `total_pool_capacity_bytes()` for clarity, OR add `remaining_capacity()` method. |
| 101-104 | Major | `staging_view<T>` is defined but **never used**. The stated purpose is 2D embedding views for T5 (512x4096) and CLIP (77x768). | Either use `staging_view` in `copy_in()` or `acquire()` to provide typed views, or remove it. If keeping, add factory methods like `get_t5_view(StagingBuffer&)`, `get_clip_view(StagingBuffer&)`. |

---

### 4. `src/staging_manager.cpp` — Rating: **REJECT**

| Aspect | Status | Notes |
|--------|--------|-------|
| `hipHostMalloc` | **FAIL** | Uses `std::vector<std::byte>` — **not** pinned memory |
| `hipHostMallocPortable` | **FAIL** | Not used (no HIP calls at all) |
| `hipMemcpyAsync` | **FAIL** | Only `std::memcpy` (synchronous host-side copy) |
| `hipEvent_t` | **FAIL** | No HIP events; no async completion signaling |
| Double-buffering | **FAIL** | No `step % num_slots` logic; no ring buffer |
| Race condition prevention | **FAIL** | No drain_slot mechanism; no HIP stream synchronization |
| RAII cleanup | **FAIL** | No HIP resources to free (doesn't allocate any) |

**Critical Issues:**

| Line | Severity | Issue | Fix |
|------|----------|-------|-----|
| 22 | **CRITICAL** | `buffers_` is `std::vector<std::vector<std::byte>>` — **pageable** host memory, not pinned. `hipMemcpyAsync` requires pinned memory for async transfers; with pageable memory, the driver must perform an internal synchronous copy through a staging buffer, defeating the purpose. | Replace with `std::vector<void*>` allocated via `hipHostMalloc(ptr, size, hipHostMallocPortable)`. Store `hipError_t` results. |
| 24 | **CRITICAL** | No HIP stream or event members. No way to track async transfer completion. | Add `hipStream_t stream_` and `std::vector<hipEvent_t> events_` (one per buffer slot) to `Impl`. |
| 30-33 | **CRITICAL** | `cfg_.pinned` is completely ignored. No conditional allocation path. | Implement: `if (cfg_.pinned) { hipHostMalloc(...); } else { /* pageable fallback */ }` |
| 31 | **CRITICAL** | `buffers_.emplace_back(cfg_.buffer_size_bytes)` uses `std::vector` default allocator — no byte alignment guarantee. HIP requires 256-byte aligned buffers for optimal DMA. | Use `hipHostMalloc` which returns suitably aligned memory, OR use `std::aligned_alloc` for pageable fallback. |
| 36-39 | Minor | `available()` method exists on `Impl` but is only called by `available_count()`. Fine pattern but could be `const` (it is). | No change needed. |
| 57-74 | Major | `acquire()` uses `std::lock_guard` correctly for thread safety. But it returns a `StagingBuffer` with spans pointing into the pool — if the pool is destroyed while buffers are outstanding, UB. This is expected for a pool allocator, but should be documented. | Add `[[nodiscard]]` on `acquire()` with comment about mandatory `release()`. Consider adding `assert(!free_indices_.empty() && "Missing release() calls detected")` in destructor debug builds. |
| 77-88 | Major | `release()` does **O(n) linear search** over all buffers to find matching index. With pool size 8 this is trivial, but with larger pools it becomes measurable. | Store the buffer index inside `StagingBuffer` (e.g., add `std::size_t pool_index` field) to make `release()` O(1). |
| 77-88 | Minor | `release()` silently does nothing if buffer not found (no match in loop, falls off end). Buffer is leaked from the pool perspective. | Add assertion or log warning when buffer not found in pool. Return the index from `acquire()` and validate it in `release()`. |
| 91-98 | Major | `copy_in()` uses synchronous `std::memcpy` — **not** `hipMemcpyAsync`. No stream parameter. No async semantics. No way to overlap copy with compute. | Add HIP-based transfer method: `copy_to_device(StagingBuffer&, void* dst, hipStream_t)` that calls `hipMemcpyAsync(dst, buf.data(), size, hipMemcpyHostToDevice, stream)`. Keep `copy_in()` for host-side staging. |
| 91-98 | Minor | `copy_in()` is not thread-safe — no mutex lock. `acquire()`/`release()` are locked but `copy_in()` assumes exclusive access to the buffer. | Document that caller must ensure exclusive access to the buffer after `acquire()`. The buffer is owned by the caller between `acquire()` and `release()`. |
| 46-54 | Minor | Move ctor/assignment are `=default` but `Impl` is incomplete in header — this works because `Impl` is complete here. OK pattern. | No change needed. |
| 51 | Minor | Destructor is `=default` — if HIP resources were allocated, they'd leak. Since currently no HIP resources are used, this is safe by accident. | When switching to `hipHostMalloc`, implement `~Impl()` that calls `hipHostFree` on each buffer. |

**The two staging implementations:** `staging_manager.hpp/cpp` is the only staging implementation present. `pinned_staging.hpp` and `pinned_staging.cpp` are **missing entirely**. The existing `staging_manager` does not implement pinned memory staging at all — it uses regular `std::vector` memory.

**Recommended resolution:**
- **Option A (recommended):** Rename `staging_manager.hpp/cpp` to `pinned_staging.hpp/cpp` and rewrite with actual HIP pinned memory, `hipMemcpyAsync`, `hipEvent_t`, and double-buffering.
- **Option B:** Keep `staging_manager.hpp/cpp` as a generic pool allocator ( rename to `buffer_pool`), create new `pinned_staging.hpp/cpp` with the HIP-specific async transfer logic.

---

### 5. `include/hq/pipeline.hpp` — Rating: **CONDITIONAL**

| Aspect | Status | Notes |
|--------|--------|-------|
| API design | OK | Clean `PipelineConfig`, `GenerationRequest`, `GeneratedImage` structs |
| `std::expected` | OK | `generate()` returns `std::expected<GeneratedImage, PipelineError>` |
| Subsystem composition | OK | Owns `Watchdog`, `HailoMonitor`, `StagingManager` via `unique_ptr` |
| Thread-safety doc | OK | "Not thread-safe. One Pipeline instance per thread." |
| `std::formatter` spec | OK | PipelineError formatter specialization at namespace scope |

**Issues:**

| Line | Severity | Issue | Fix |
|------|----------|-------|-----|
| 38-40 | Minor | Forward declarations of `UtilizationWatchdog`, `HailoMonitor`, `EmbeddingStagingManager` are redundant — the headers are already `#include`d at lines 10-12. | Remove redundant forward declarations. |
| 89-99 | Minor | `PipelineStats` has `avg_gpu_utilization` and `avg_hailo_utilization` but no corresponding tracking fields in the class. Not populated anywhere. | Either populate in `generate()` or remove from struct until implemented. |
| 123 | Minor | `use_hailo_text_encoder{true}` — if Hailo is unavailable, pipeline should auto-fallback. Currently has Hailo EP fallback in `OrtState` but no flag check. | Gate Hailo session creation on this flag. |
| 253 | Info | `to_string(PipelineError)` has no `default` case for out-of-range enum values — undefined behavior if cast from invalid integer. | Add `default: return "Unknown";` (already present at line 248, but switch is not exhaustive for all values — actually it IS exhaustive. OK). |

---

### 6. `src/pipeline_integration.cpp` — Rating: **FAIL**

| Aspect | Status | Notes |
|--------|--------|-------|
| ONNX Runtime integration | Partial | Properly configures ROCm EP and Hailo EP with fallback |
| Staging usage | OK | Correctly acquires, copies, releases staging buffer |
| Watchdog integration | OK | Steps watchdog at step boundaries |
| HailoMonitor API usage | **FAIL** | Calls methods that don't exist in the header |

**Critical Issues:**

| Line | Severity | Issue | Fix |
|------|----------|-------|-----|
| 264 | **CRITICAL** | Calls `hailo_monitor_->is_connected()` — header declares `is_open()`. **Will not compile.** | Change to `is_open()`. |
| 265 | **CRITICAL** | `hailo_monitor_->sample()` — implementation returns `HailoTelemetry` but header declares `HailoStats`. Pipeline accesses `telem->inference_ms` which doesn't exist in `HailoStats`. **Will not compile.** | Use `HailoStats` fields: `power_indicator`, `inference_indicator`, `nn_core_utilization`. |
| 267 | **CRITICAL** | `telem->inference_ms` — `HailoStats` has no such field. The header declares `inference_delta` (count) and `nn_core_utilization` (fused %), not per-inference latency. | Use `telem->nn_core_utilization` or `telem->inference_indicator` for utilization percentage. |
| 283 | Major | `gpu_util` is computed from `sin(step * 0.3f)` — this is a placeholder that should be replaced with actual ROCm SMI queries. | Add `TODO(ROCm-SMI): Replace with rsmi_dev_utilization_get()` comment. Implement behind `#ifdef UM790_HAS_ROCM_SMI`. |
| 285 | Major | `hailo_power_w = 2.5` is hardcoded placeholder. | Use actual `hailo_monitor_->sample()->power_watts` when monitor is functional. |
| 383-386 | **CRITICAL** | `denoise_step_()` has undefined behavior: `offsetof(std::vector<float>, _M_impl._M_start)` is **non-portable** and relies on libstdc++ internals. This is undefined behavior per the C++ standard (offsetof on non-standard-layout types). | Pass `latents` as `std::vector<float>&` directly, or store latents as a member. The `void*` cast chain is extremely fragile. |
| 395-431 | Major | `on_watchdog_recovery_()` takes `latents` as parameter in declaration (line 188) but definition has no such parameter. Mismatch between declaration and definition. | Match declaration signature. Pass latents through recovery context. |
| 437-455 | Minor | `encode_prompt_()` returns dummy embeddings. CLIP uses 77x768 but T5 uses 512x4096 — hardcoded to CLIP only. | Add model-type detection or configuration. Document which encoder shapes are supported. |
| 88-89 | Info | `hailo_monitor_{std::make_unique<HailoMonitor>(0)}` passes `0` to a constructor that header declares as `HailoMonitor()` (no args). **Will not compile.** | Change to `std::make_unique<HailoMonitor>()` and call `open()` separately. |

---

### 7. `CMakeLists.txt` — Rating: **FAIL**

| Line | Severity | Issue | Fix |
|------|----------|-------|-----|
| 250 | **CRITICAL** | `ge(STATUS "")` — typo for `message(STATUS "")`. CMake will fail with "Unknown CMake command 'ge'". | Change to `message(STATUS "")`. |
| 128-129 | Minor | `UM790_PIPELINE_SOURCES` only lists `src/pipeline_integration.cpp`. Missing `src/staging_manager.cpp`, `src/watchdog.cpp`, `src/hailo_monitor.cpp`. | Add all `.cpp` files to the sources list. Currently only one source file is compiled — other translation units are missing from the build. |
| 142 | Minor | `um790_pipeline` is declared as `STATIC` but `SOVERSION` is set — meaningless for static libs. | Remove `SOVERSION` property for static library, or change to `SHARED`. |
| 40-47 | Minor | `find_package(HIP REQUIRED CONFIG)` will fail on systems without HIP. Should be `REQUIRED` only when `ENABLE_GPU` is on. | Make HIP optional with `find_package(HIP CONFIG)` and gate GPU code behind `#ifdef HIP_FOUND`. |

---

### 8. `tests/test_harness.cpp` — Rating: **FAIL**

| Line | Severity | Issue | Fix |
|------|----------|-------|-----|
| 90-103 | **CRITICAL** | `MockHailoMonitor::sample()` returns `HailoTelemetry` — this type is **never defined** in any header file. The mock invents a type that doesn't exist. | Use `HailoStats` from `hailo_monitor.hpp`. Adapt mock to populate `HailoStats` fields. |
| 91 | Minor | Mock returns `HailoTelemetry{ .power_w = ..., .throughput_fps = ... }` — `HailoStats` has `power_watts`, `nn_core_utilization`, etc. Field names don't match. | Align with `HailoStats` field names. |
| 139-145 | Major | `MockStagingManager::acquire()` creates `std::vector<std::byte> buf` on the stack and returns a `StagingBuffer` with spans pointing to it. The vector is destroyed when the function returns, leaving **dangling spans**. | Allocate mock buffers on the heap or as member variables with stable addresses. |
| 149-152 | Minor | `MockStagingManager::release()` decrements `acquire_count_` instead of `release_count_` only, then also increments `release_count_`. Logic is confusing but functionally correct (both counters change). | Simplify: only `release_count_++`; `acquire_count_--`. |
| 359 | Minor | `std::ranges::count_if` requires C++20 ranges. The CMake sets C++26 so this is fine, but the include `<ranges>` is missing from test_harness.cpp. | Add `#include <ranges>`. |

---

## Detailed Analysis by Review Category

### 1. HailoMonitor — Dual-Indicator Logic

| Requirement | Status | Detail |
|-------------|--------|--------|
| Power indicator: `(power - 0.5) / 5.5 * 100` | **NOT IMPLEMENTED** | Formula is documented in header comments but `.cpp` has no power reading, no normalization, no indicator computation. |
| Inference delta from `inferences_count` | **NOT IMPLEMENTED** | `inferences_count` is in `HailoStats` struct but `.cpp` never reads it, never computes delta. |
| DMA stall: `power > 70% && inference < 30%` -> weight inference more | **NOT IMPLEMENTED** | `detect_sensor_mismatch()` and `fuse_indicators()` are declared in header but have no implementation. |
| Sensor error: low power + high inference | **NOT IMPLEMENTED** | Declared as `SensorMismatch` error code but never detected. |
| HailoRT API: `Device::scan_pcie`, `get_power_measurement` | **NOT IMPLEMENTED** | Only placeholder comments in `Impl`. No actual HailoRT API calls. |

**Conclusion:** The dual-indicator monitoring system is **completely unimplemented**. The header defines an excellent API design, but the `.cpp` is a skeletal stub with a different interface.

### 2. PinnedStagingBuffer — Memory Management

| Requirement | Status | Detail |
|-------------|--------|--------|
| `hipHostMalloc` with `hipHostMallocPortable` | **FAIL** | Uses `std::vector<std::byte>` (pageable memory). No HIP memory calls at all. |
| `hipMemcpyAsync` (not synchronous) | **FAIL** | Only `std::memcpy` (host-side synchronous). No async device transfers. |
| `hipEvent_t` for completion | **FAIL** | No events, no stream synchronization. |
| Double-buffering: `step % num_slots` | **FAIL** | No ring buffer, no slot indexing. Simple pool of independent buffers. |
| Drain slot before reuse | **FAIL** | No mechanism to wait for in-flight transfers. Buffers are reused immediately on `release()`. |

**Conclusion:** The staging manager is a **generic byte buffer pool**, not a pinned staging buffer for GPU DMA. It provides none of the HIP integration required for the stated purpose of "replacing XNACK zero-copy."

### 3. RAII and Error Handling

| Resource | Destructor cleanup | Notes |
|----------|-------------------|-------|
| `hailort::Device` | Partial | Header declares `DeviceDeleter`; stub `.cpp` has no real device to free. |
| HIP host memory | **N/A** (not allocated) | When implemented, must call `hipHostFree()` in `~Impl()`. |
| HIP events | **N/A** (not created) | When implemented, must call `hipEventDestroy()` in `~Impl()`. |
| HIP stream | **N/A** (not created) | When implemented, must call `hipStreamDestroy()` in `~Impl()`. |
| `Ort::Session` | OK | `unique_ptr<Ort::Session>` properly reset in `shutdown()`. |
| Staging buffers | OK by accident | `std::vector` frees automatically, but this is pageable memory — not the intended resource. |

**`std::expected` usage:**
- Used correctly in all public APIs that can fail ✓
- Error type (`HailoError` / `StagingError`) is consistent ✓
- Monadic operations (`and_then`, `or_else`) tested in test harness ✓
- Missing: `transform_error` for error context enrichment

### 4. API Consistency — Two Versions Problem

**HailoMonitor:**
```
Header declares:  HailoMonitor()                    open(const std::string&) 
                  sample() -> HailoStats            is_open()
                  hard_reset()                      device_id()
                  fuse_indicators()                 detect_sensor_mismatch()
                  
.cpp implements:  HailoMonitor(int)                 (no open)
                  sample() -> HailoTelemetry        is_connected()
                  (no hard_reset)                   (no device_id)
                  (no fusion)                       (no mismatch detection)
```
**Verdict: Complete mismatch. Header is canonical; `.cpp` must be rewritten.**

**Staging:**
```
Expected: pinned_staging.hpp / pinned_staging.cpp (MISSING)
Actual:   staging_manager.hpp / staging_manager.cpp (exists but wrong implementation)
```
**Verdict: Missing files. staging_manager is a placeholder, not the real implementation.**

### 5. std::mdspan Usage

| Requirement | Status | Detail |
|-------------|--------|--------|
| `std::mdspan` for 2D embedding views | Partial | Type alias `staging_view<T>` defined at line 103 of `staging_manager.hpp` |
| T5 extents: 512 x 4096 | **FAIL** | Never instantiated with these extents |
| CLIP extents: 77 x 768 | **FAIL** | Never instantiated with these extents |
| Correct extents parameterization | N/A | `std::dextents<std::size_t, 2>` is a runtime-extent 2D view — works for any 2D shape, but doesn't encode T5/CLIP dimensions at compile time. | |

**Conclusion:** `std::mdspan` is declared but never used. No embedding views are created. The type alias should either be used or removed.

### 6. Thread Safety

| Component | Thread Safety | Notes |
|-----------|--------------|-------|
| `EmbeddingStagingManager::acquire()` | OK | `std::lock_guard` on `mtx_` |
| `EmbeddingStagingManager::release()` | OK | `std::lock_guard` on `mtx_` |
| `EmbeddingStagingManager::copy_in()` | **UNSAFE** | No lock; caller must ensure exclusive access (acceptable design) |
| `HailoMonitor::sample()` | Unknown | Header doesn't declare `sample() const` consistently with implementation. Thread safety TBD once implemented. |
| Producer-consumer Hailo->GPU | **NOT IMPLEMENTED** | No atomic flags, no condition variables, no ring buffer. Pipeline does sequential execution only. |

---

## Concrete Fix Instructions

### Priority 1: Critical (must fix before merge)

1. **`CMakeLists.txt:250`** — Fix `ge(STATUS "")` -> `message(STATUS "")`
2. **`CMakeLists.txt:128-129`** — Add all source files to `UM790_PIPELINE_SOURCES`:
   ```cmake
   set(UM790_PIPELINE_SOURCES
       src/pipeline_integration.cpp
       src/hailo_monitor.cpp
       src/staging_manager.cpp
       src/watchdog.cpp
   )
   ```
3. **`hailo_monitor.cpp`** — Rewrite entirely to match `hailo_monitor.hpp` API:
   - Change constructor to `HailoMonitor() = default;`
   - Implement `open(const std::string&)`, `close()`, `hard_reset()`, `is_open()`, `device_id()`
   - Implement `sample()` returning `HailoStats` with dual-indicator fusion
   - Implement `read_power_watts()`, `read_temperature_celsius()`, `read_inference_count()`
   - Implement `fuse_indicators()`: `power_weight_ * power_util + inference_weight_ * inference_util`
   - Implement `detect_sensor_mismatch()`: check divergence between indicators
   - Implement all tunable threshold getters/setters
   - Use `hailort::Device::scan_pcie()` and `get_power_measurement()` API calls
4. **`pipeline_integration.cpp:264`** — Change `is_connected()` -> `is_open()`
5. **`pipeline_integration.cpp:265-269`** — Change telemetry access to use `HailoStats` fields
6. **`pipeline_integration.cpp:88-89`** — Change `std::make_unique<HailoMonitor>(0)` -> `std::make_unique<HailoMonitor>()`
7. **`pipeline_integration.cpp:383-386`** — Remove the `offsetof` hack. Pass `std::vector<float>&` properly.

### Priority 2: Major (fix before production)

8. **`staging_manager.cpp:22`** — Switch to `hipHostMalloc`:
   ```cpp
   void* ptr = nullptr;
   hipError_t err = hipHostMalloc(&ptr, cfg_.buffer_size_bytes, hipHostMallocPortable);
   if (err != hipSuccess) { /* handle error */ }
   buffers_.push_back(ptr);
   ```
9. **`staging_manager.cpp:51`** — Implement `~Impl()`:
   ```cpp
   ~Impl() {
       for (void* ptr : buffers_) {
           if (ptr) hipHostFree(ptr);
       }
   }
   ```
10. **`staging_manager.cpp:30-33`** — Respect `cfg_.pinned`:
    ```cpp
    if (cfg_.pinned) {
        // hipHostMalloc with hipHostMallocPortable
    } else {
        buffers_.emplace_back(cfg_.buffer_size_bytes);  // pageable fallback
    }
    ```
11. **`staging_manager.cpp:77-88`** — Make `release()` O(1) by storing pool index in `StagingBuffer`:
    ```cpp
    struct StagingBuffer {
        std::span<std::byte> data;
        std::span<const std::byte> cdata;
        std::size_t capacity;
        std::size_t used;
        std::size_t pool_index;  // ADD THIS
    };
    ```
12. **`staging_manager.hpp/cpp`** — Add async HIP transfer method:
    ```cpp
    [[nodiscard]] std::expected<void, StagingError>
        copy_to_device(const StagingBuffer& buf, void* device_dst,
                       hipStream_t stream, hipEvent_t* completion_event);
    ```
13. **`test_harness.cpp:90-103`** — Replace `HailoTelemetry` with `HailoStats` throughout mock.
14. **`test_harness.cpp:139-145`** — Fix dangling span: store mock buffers as `std::vector<std::vector<std::byte>>` as class members.
15. **`staging_manager.hpp:101-104`** — Either use `staging_view<T>` in the API or remove it. If keeping, add factory methods.

### Priority 3: Minor/Warnings

16. **`hailo_monitor.hpp:44-53`** — Add compile-time validation:
    ```cpp
    static_assert(HAILO8L_IDLE_POWER_W < HAILO8L_ACTIVE_POWER_W);
    static_assert(HAILO8L_ACTIVE_POWER_W < HAILO8L_MAX_TDP_W);
    static_assert(HAILO8L_DMA_STALL_POWER_THRESHOLD > HAILO8L_DMA_STALL_INFERENCE_THRESHOLD);
    ```
17. **`staging_manager.cpp:93-98`** — Add pool ownership validation in `copy_in()`.
18. **`pipeline.hpp:38-40`** — Remove redundant forward declarations.
19. **`test_harness.cpp`** — Add `#include <ranges>` for `std::ranges::count_if`.
20. **All files** — Add `[[nodiscard]]` to functions where the return value must not be ignored.

---

## Architecture Recommendations

### For the Staging Subsystem

The current `staging_manager` is a generic pool, not a pinned staging buffer. Two recommended architectures:

**Option A: Unified Pinned Staging (Recommended)**
```
pinned_staging.hpp/cpp  <- Rename from staging_manager
  - EmbeddingStagingManager (actual HIP pinned memory)
  - hipHostMalloc/hipHostFree for allocate/free
  - hipMemcpyAsync + hipEvent_t for async transfers
  - Ring buffer with drain-before-reuse
  - staging_view<T> factory for T5/CLIP embeddings
```

**Option B: Layered Architecture**
```
buffer_pool.hpp/cpp          <- Generic pool (current staging_manager renamed)
  - Byte buffer pool, no HIP dependency
  
pinned_staging.hpp/cpp       <- NEW: HIP-specific layer
  - PinnedBufferPool : public BufferPool  (or composition)
  - Async transfer methods
  - Event-based completion
  
staging_manager.hpp/cpp      <- Orchestration layer
  - Manages buffer pool + async transfers
  - Coordinates with pipeline steps
```

### For the HailoMonitor

The header's dual-indicator design is sound. Implementation should follow this flow:

```
sample():
  1. read_power_watts() -> raw power in Watts
  2. power_indicator = (power - IDLE_POWER) / (ACTIVE_POWER - IDLE_POWER) * 100
  3. read_inference_count() -> cumulative count
  4. inference_delta = count - prev_count (handle first sample)
  5. inference_indicator = (inference_delta / expected_per_sec) * 100
  6. nn_core_utilization = power_weight * power_indicator + inference_weight * inference_indicator
  7. detect_sensor_mismatch(): |power_indicator - inference_indicator| > threshold?
  8. If mismatch: set device_healthy=false, return SensorMismatch error
  9. Populate HailoStats, return
```

### For Double-Buffering

When the real pinned staging is implemented, the denoising loop should use a ring buffer:

```cpp
// In Pipeline::generate() denoising loop:
const std::size_t num_slots = cfg_.staging_buffer_count;
std::size_t write_slot = 0;
std::size_t read_slot = 0;

for (uint32_t step = 0; step < req.num_steps; ++step) {
    // 1. Wait for previous transfer on this slot to complete
    staging_manager_->drain_slot(write_slot);
    
    // 2. Copy new data into staging buffer
    auto buf = staging_manager_->acquire_slot(write_slot);
    staging_manager_->copy_in(buf, new_embedding_data);
    
    // 3. Async transfer to device
    staging_manager_->copy_to_device(buf, device_ptr, stream, &events[write_slot]);
    
    // 4. Run denoising kernel (consumes read_slot data)
    denoise_kernel<<<...>>>(device_ptrs[read_slot], ...);
    
    // 5. Advance ring
    read_slot = write_slot;
    write_slot = (write_slot + 1) % num_slots;
}
```

---

## Summary Table

| File | Rating | Blockers | Majors | Minors |
|------|--------|----------|--------|--------|
| `hailo_monitor.hpp` | CONDITIONAL | 0 | 0 | 3 |
| `hailo_monitor.cpp` | **REJECT** | 7 | 2 | 1 |
| `staging_manager.hpp` | CONDITIONAL | 0 | 4 | 4 |
| `staging_manager.cpp` | **REJECT** | 6 | 4 | 3 |
| `pipeline.hpp` | CONDITIONAL | 0 | 0 | 3 |
| `pipeline_integration.cpp` | **FAIL** | 4 | 2 | 1 |
| `CMakeLists.txt` | **FAIL** | 1 | 3 | 0 |
| `test_harness.cpp` | **FAIL** | 1 | 2 | 2 |
| **Missing:** `pinned_staging.hpp/cpp` | N/A | 2 files entirely absent | — | — |

---

## Overall Verdict: **REJECTED**

**Rationale:**
1. **Seven critical mismatches** between headers and implementations prevent compilation.
2. **Two core files are completely missing** (`pinned_staging.hpp`, `pinned_staging.cpp`).
3. **The staging manager does not use HIP at all** — it uses pageable `std::vector` memory instead of `hipHostMalloc` pinned memory, synchronous `std::memcpy` instead of `hipMemcpyAsync`, and has no event-based completion signaling.
4. **The HailoMonitor dual-indicator logic is entirely unimplemented** — only a stub returning synthetic data exists.
5. **CMake has a typo** that prevents configuration.
6. **The pipeline calls methods** (`is_connected()`, `HailoTelemetry`) that don't exist in the headers.

**To reach CONDITIONAL approval:**
- Fix the CMake typo
- Rewrite `hailo_monitor.cpp` to match the header
- Add the missing `pinned_staging.hpp/cpp` with actual HIP pinned memory
- Fix all pipeline API mismatches
- Fix test harness `HailoTelemetry` -> `HailoStats`

**To reach APPROVED status:**
- All of the above, plus:
- Implement actual HailoRT API calls (not stubs)
- Implement actual `hipMemcpyAsync` with event completion
- Add double-buffering ring buffer logic
- Add comprehensive unit tests for dual-indicator fusion
- Verify with `hipHostMalloc`/`hipHostFree` in destructor
