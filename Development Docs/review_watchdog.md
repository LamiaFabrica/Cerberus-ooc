# UtilizationWatchdog Code Review Report

**Reviewer:** Senior C++ Code Reviewer (Real-Time Systems / GPU / C++26)
**Date:** 2025-01-20
**Scope:** Per-step inline sampling watchdog replacing 500ms async thread
**Files Reviewed:**
- `include/hq/watchdog.hpp`
- `src/watchdog.cpp`
- `include/hq/pipeline.hpp`
- `src/pipeline_integration.cpp`
- `include/hq/cxx26_features.hpp` (supporting)
- `include/hq/hailo_monitor.hpp` (supporting)
- `include/hq/staging_manager.hpp` (supporting)

---

## Overall Verdict: **REJECTED**

The implementation has **4 critical bugs** that would cause compilation failures, undefined behavior, or broken recovery semantics. Several design requirements (WARNING state, exponential backoff) are not implemented. The missing `utilization_watchdog.hpp`/`utilization_watchdog.cpp` files indicate incomplete deliverables.

---

## File 1: `include/hq/watchdog.hpp`

### Rating: **Pass with Warnings**

| Check | Status | Notes |
|-------|--------|-------|
| C++26 `std::expected` usage | N/A | Header defines types; no error paths here |
| `std::print` for logging | N/A | Not used in header |
| `[[nodiscard]]` attributes | PASS | On `step()` (line 80) and `recovery_count()` (line 83) |
| Strong enum with underlying type | PASS | `RecoveryAction : std::uint8_t` (line 30) |
| Raw pointers in public API | PASS | None; uses `std::function` for callback |
| RAII | PASS | All members are value types or `std::function` |
| Thread safety design | PASS | No mutable shared state in header; single-threaded |

### Issues Found

#### W1 — `utilization_watchdog.hpp` MISSING (Structural Gap)
**Severity: HIGH**

The mission specifies two watchdog headers:
- `include/hq/utilization_watchdog.hpp` (from CoreImplementer_Watchdog) — **MISSING**
- `include/hq/watchdog.hpp` (from CoreImplementer_BuildSystem) — **PRESENT**

Only `watchdog.hpp` exists in the codebase. The `UtilizationWatchdog` class is defined in `watchdog.hpp`, but the expected second header is absent. This means:
- No separate interface from the "CoreImplementer_Watchdog" contributor
- Cannot evaluate API conflicts between the two headers as specified in the checklist
- Deliverable is incomplete

**Fix:** Either merge was intended (document it) or the missing file must be provided.

---

#### W2 — `WatchdogConfig` callback lifetime hazard acknowledged but unaddressed
**File: `watchdog.hpp:61,76` | Severity: MEDIUM**

The `on_recovery` callback in `WatchdogConfig` (line 61) is a `std::function` that the `UtilizationWatchdog` stores by value. If this callback captures `this` from `Pipeline`, moving the `Pipeline` object (which has a defaulted move constructor) would leave the callback with a dangling `this` pointer.

The `pipeline_integration.cpp` comment at line 122-123 acknowledges this: "Note: recovery callback captures `this`; Pipeline is non-movable after init." But the callback is **never actually registered**, making this a latent bug waiting to be activated.

**Fix:** Use a `std::weak_ptr<Pipeline>` or an opaque handle ID in the callback. Or remove the callback from `WatchdogConfig` entirely since the pipeline doesn't use it (see P3 below).

---

## File 2: `src/watchdog.cpp`

### Rating: **Fail — 3 Issues (1 Medium, 2 Design Gaps)**

| Check | Status | Notes |
|-------|--------|-------|
| `std::print` usage | WARNING | `<print>` included (line 8) but **never used** — only `std::format` is used. Dead include. |
| No async thread | PASS | Pure per-step inline sampling |
| 8-step threshold | PASS | `max_consecutive_low{8}` enforced at line 50 |
| 40% critical threshold | PASS | `gpu_threshold{40.0}` enforced at line 39 |
| WARNING -> CRITICAL | **FAIL** | No WARNING state exists; only `None` and `Restart` |
| Exponential backoff | **FAIL** | Fixed cooldown; no exponential delay between successive recoveries |
| Raw pointers / exceptions | PASS | None |
| Thread safety | PASS | Runs on caller (pipeline) thread; no mutex needed |

### Issues Found

#### C1 — No WARNING State: Direct None -> Restart Escalation
**File: `watchdog.cpp:23-91` | Severity: HIGH (Design Gap)**

The checklist requires "WARNING -> CRITICAL escalation." The current implementation has only two states:
- `RecoveryAction::None` — everything fine
- `RecoveryAction::Restart` — recovery triggered after 8 consecutive low steps

There is **no `Warning` action**. The watchdog should emit a WARNING after some intermediate threshold (e.g., 4 consecutive low steps, 50% of the critical threshold) before escalating to CRITICAL/Restart at 8 steps. This gives the pipeline operator early notice that utilization is trending downward.

**Fix:** Add a `Warning` recovery action and a `warning_threshold` (e.g., `max_consecutive_low / 2 = 4`):

```cpp
enum class RecoveryAction : std::uint8_t {
    None     = 0,
    Warning  = 1,  // ADD: early warning at N/2 consecutive low steps
    Throttle = 2,  // already defined but never used
    Restart  = 3,
    Abort    = 4,
};
```

In `step()`:
```cpp
if (consecutive_low_ >= cfg_.max_consecutive_low) {
    // CRITICAL: trigger restart
    // ...
} else if (consecutive_low_ == cfg_.max_consecutive_low / 2) {
    return WatchdogResult{.action = RecoveryAction::Warning, ...};
}
```

---

#### C2 — No Exponential Backoff for Recovery
**File: `watchdog.cpp:51-53` | Severity: HIGH (Design Gap)**

The checklist asks for "recovery backoff (exponential delay)." The current cooldown is a **fixed** number of steps (`cfg_.recovery_cooldown_steps`, default 3). After each recovery, the same fixed cooldown is applied. If the first recovery fails to restore utilization, subsequent recoveries will fire at the same rate, potentially causing a recovery storm.

Expected behavior: After recovery N, cooldown should be `base * 2^N` steps (capped at some max).

**Fix:** Track `recovery_count_` and compute exponential cooldown:

```cpp
// In UtilizationWatchdog class, add:
static constexpr uint32_t MAX_COOLDOWN_STEPS = 64;

// In step(), when triggering recovery:
uint32_t cooldown = cfg_.recovery_cooldown_steps * (1u << std::min(recovery_count_ - 1, 5u));
cooldown_remaining_ = std::min(cooldown, MAX_COOLDOWN_STEPS);
```

---

#### M1 — `<print>` header included but never used
**File: `watchdog.cpp:8` | Severity: LOW**

The `#include <print>` is unused. The file only uses `std::format` (which requires `<format>`, already transitively included via header). Remove the dead include.

**Fix:** Remove line 8.

---

#### M2 — Recovery reason only reports first matching condition
**File: `watchdog.cpp:56-72` | Severity: LOW**

The `if/else if` chain for constructing the recovery reason only captures the first true condition. If both GPU is low (35%) AND step timed out (250ms), only the GPU message is reported. The timeout condition is silently omitted from diagnostics.

**Fix:** Build a composite reason string:

```cpp
std::vector<std::string> reasons;
if (gpu_low)  reasons.push_back(std::format("GPU {:.1f}%", snap.gpu_utilization));
if (hailo_low)reasons.push_back(std::format("Hailo {:.1f}%", snap.hailo_utilization));
if (timeout)  reasons.push_back(std::format("timeout {:.1f}ms", snap.inference_ms));
std::string reason = std::format("Low utilization: {}", fmt::join(reasons, ", "));
```

---

## File 3: `include/hq/pipeline.hpp`

### Rating: **Pass with Warnings**

| Check | Status | Notes |
|-------|--------|-------|
| `std::expected` usage | PASS | Error handling for all fallible operations |
| `std::print` | PASS | Used throughout |
| `[[nodiscard]]` | PASS | On generate(), generate_batch(), get_stats() |
| Strong enums | PASS | `PipelineError : std::uint32_t` |

### Issues Found

#### W3 — `error_string_()` declared but never defined
**File: `pipeline.hpp:204` | Severity: MEDIUM**

`error_string_(PipelineError e)` is declared as a private static method but the codebase has `to_string(PipelineError e)` as a free function instead. Either remove the declaration or implement the method.

**Fix:** Remove the declaration at line 204, or rename `to_string` to `Pipeline::error_string_` and make it static private.

---

#### W4 — `UtilizationWatchdog` forward-declared but also fully included
**File: `pipeline.hpp:38,210` | Severity: LOW**

Line 38 forward-declares `class UtilizationWatchdog;` but line 10 `#include "hq/watchdog.hpp"` already provides the full definition. The forward declaration is redundant.

**Fix:** Remove line 38.

---

## File 4: `src/pipeline_integration.cpp`

### Rating: **Fail — 5 Critical Bugs**

| Check | Status | Notes |
|-------|--------|-------|
| Watchdog called at step boundaries | PASS | Lines 279-288 |
| Recovery callback registered | **FAIL** | Empty setup block — callback never attached |
| Latent save/restore | **FAIL** | Prints messages only — no actual tensor save/restore |
| `std::print` usage | PASS | Used throughout |
| `std::expected` error handling | PASS | Proper error propagation |

### Issues Found

#### C3 — COMPILATION ERROR: `hailo_monitor_->is_connected()` doesn't exist
**File: `pipeline_integration.cpp:264` | Severity: CRITICAL**

```cpp
if (hailo_monitor_ && hailo_monitor_->is_connected()) {
```

The `HailoMonitor` class (defined in `hailo_monitor.hpp:172`) provides `is_open()`, not `is_connected()`. This is a **compile-time error**.

**Fix:** Change to `is_open()`:
```cpp
if (hailo_monitor_ && hailo_monitor_->is_open()) {
```

---

#### C4 — UB: `denoise_step_` uses `offsetof` on non-standard-layout type
**File: `pipeline_integration.cpp:383-386` | Severity: CRITICAL**

```cpp
std::vector<float>* latents_vec = static_cast<std::vector<float>*>(
    static_cast<void*>(
        static_cast<char*>(latents) - offsetof(std::vector<float>, _M_impl._M_start)));
```

Three layers of undefined behavior:
1. `offsetof` on `std::vector<float>` is UB — `std::vector` is not standard-layout
2. `_M_impl._M_start` is a **libstdc++ internal detail** — won't compile on libc++ or MSVC
3. The caller passes `latents.data()` (a `float*`) at line 254, and this code attempts to reverse-engineer the `std::vector*` from a pointer to its internal heap data using implementation-specific layout knowledge

**Fix:** Pass the `std::vector<float>&` directly:
```cpp
// In pipeline.hpp, change declaration:
[[nodiscard]] std::expected<void*, PipelineError>
    denoise_step_(uint32_t step, std::vector<float>& latents);

// In pipeline_integration.cpp line 254:
auto denoise_result = denoise_step_(step, latents);

// In denoise_step_ implementation:
std::expected<void*, PipelineError>
Pipeline::denoise_step_(uint32_t step, std::vector<float>& latents) {
    // Use latents directly — no pointer arithmetic needed
    (void)step;
    return latents.data();
}
```

---

#### C5 — Recovery callback NEVER registered with watchdog
**File: `pipeline_integration.cpp:120-124` | Severity: CRITICAL**

```cpp
    // Set up watchdog recovery callback
    if (watchdog_ && cfg.enable_watchdog) {
        // Note: recovery callback captures `this`; Pipeline is non-movable after init
        // In production, use a weak_ptr or explicit handle to avoid lifetime issues
    }
```

The callback setup block is **completely empty**. The `WatchdogConfig::on_recovery` `std::function` remains default-constructed (empty). The watchdog's `step()` method checks `if (cfg_.on_recovery)` before calling it, so no crash occurs, but:

1. The callback mechanism in `WatchdogConfig` is dead code from the pipeline's perspective
2. The pipeline manually checks `wd_result.action` and calls `on_watchdog_recovery_()` separately (lines 290-303), which is a different code path
3. This creates a dual-path design where the callback architecture exists but is bypassed

**Fix:** Either:
- **Option A (recommended):** Register the callback properly:
  ```cpp
  if (watchdog_ && cfg.enable_watchdog) {
      watchdog_.reset();  // Re-create with callback
      auto cfg_copy = cfg;  // capture recovery config
      watchdog_ = std::make_unique<UtilizationWatchdog>(WatchdogConfig{
          // ... copy thresholds ...
          .on_recovery = [this](RecoveryAction a, const std::string& r) {
              this->on_watchdog_recovery_(a, r);
          },
      });
  }
  ```
- **Option B:** Remove the callback from `WatchdogConfig` entirely and rely solely on the return-value check pattern.

---

#### C6 — `on_watchdog_recovery_` does NOT save/restore latent tensors
**File: `pipeline_integration.cpp:399-432` | Severity: CRITICAL**

```cpp
void Pipeline::on_watchdog_recovery_(RecoveryAction action, const std::string& reason) {
    // ...
    // Save current latents (deep copy)
    // In the real implementation, latents would be passed in or stored as member
    std::print("[Pipeline]  -> Saving latent checkpoint\n");
    // ...
    // Restore latents from checkpoint
    std::print("[Pipeline]  -> Restoring latents\n");
    // ...
}
```

The function prints "Saving latent checkpoint" and "Restoring latents" but performs **no actual work**. The `latents` vector is a local variable in `generate()` (line 238) and is not accessible from this member function. The comment acknowledges the gap: "In the real implementation, latents would be passed in or stored as member."

After session rebuild, the pipeline continues with the same `latents` vector (since it's on the caller's stack frame and wasn't modified). This happens to work by accident — the latents aren't corrupted — but there is no deep-copy checkpoint/restore as the design requires.

More critically: if the ONNX session rebuild fails, the function returns early without any error propagation. The next denoising step will fail because `ort_state_->gpu_session` is null.

**Fix:** Store latents as a member variable or pass by reference:

```cpp
// In Pipeline class, add member:
std::vector<float> latent_checkpoint_;

// In on_watchdog_recovery_:
void Pipeline::on_watchdog_recovery_(RecoveryAction action, const std::string& reason,
                                     std::vector<float>& current_latents) {
    // Deep copy save
    latent_checkpoint_ = current_latents;  // copy
    std::print("[Pipeline]  -> Saved {} latent floats\n", latent_checkpoint_.size());
    
    if (action == RecoveryAction::Restart) {
        // ... rebuild sessions ...
    }
    
    // Restore
    current_latents = latent_checkpoint_;  // copy back
    std::print("[Pipeline]  -> Restored {} latent floats\n", current_latents.size());
}
```

---

#### C7 — `recovery_attempts_` never reset between `generate()` calls
**File: `pipeline_integration.cpp:248,295,404` | Severity: HIGH**

`recovery_attempts_` is incremented in `on_watchdog_recovery_()` (line 404) and checked against `cfg_.max_recovery_attempts` (line 295). However, it is **never reset** at the start of a new `generate()` call. Only `recovery_in_progress_` is set to `false` (line 248).

Impact: If generation A triggers 2 recoveries, generation B starts with `recovery_attempts_ == 2`. Generation B can only tolerate 1 more recovery before failing. This creates cross-call state pollution.

**Fix:** Reset in `generate()`:
```cpp
// At line 248, change:
recovery_in_progress_ = false;
recovery_attempts_ = 0;  // ADD this line
```

---

#### M3 — Watchdog setup doesn't configure `on_recovery` callback
**File: `pipeline_integration.cpp:78-88` | Severity: MEDIUM**

The `WatchdogConfig` is constructed with designated initializers at lines 81-87, but the `on_recovery` field is omitted (left default-empty). Since the pipeline doesn't use the callback (see C5), this is consistent but represents incomplete integration.

**Fix:** If using Option B from C5 (remove callback), this is fine. If using Option A, populate the callback here.

---

#### M4 — `shutdown_` flag not checked after recovery in denoising loop
**File: `pipeline_integration.cpp:250-313` | Severity: LOW**

The `shutdown_` flag is checked at the start of `generate()` (line 191) but not inside the denoising loop. If `shutdown()` is called from another thread during generation, the loop won't notice until the current generation completes. (Though the header says "Not thread-safe. One Pipeline instance per thread," so this may be acceptable.)

**Fix:** Add a periodic check inside the loop:
```cpp
for (uint32_t step = 0; step < req.num_steps; ++step) {
    if (shutdown_) break;  // cooperative cancellation
    // ...
}
```

---

## File 5: `include/hq/cxx26_features.hpp`

### Rating: **Pass**

Simple feature-detection header. All macros properly `#ifndef` guarded. No issues.

---

## Summary: Prioritized Fix List

### P0 — Must Fix Before Merge (Compilation / Crash)

| Priority | Issue | File | Line | Fix |
|----------|-------|------|------|-----|
| P0 | `is_connected()` -> `is_open()` | `pipeline_integration.cpp` | 264 | Rename method call |
| P0 | Remove UB `offsetof` on `std::vector` | `pipeline_integration.cpp` | 383-386 | Pass `std::vector<float>&` instead of `void*` |
| P0 | Reset `recovery_attempts_` per call | `pipeline_integration.cpp` | 248 | Add `recovery_attempts_ = 0;` |

### P1 — Must Fix (Broken Recovery Semantics)

| Priority | Issue | File | Line | Fix |
|----------|-------|------|------|-----|
| P1 | Implement actual latent save/restore | `pipeline_integration.cpp` | 399-432 | Store checkpoint as member; pass latents by ref |
| P1 | Add WARNING state to watchdog | `watchdog.hpp` + `watchdog.cpp` | 30-35, 49-91 | Add `RecoveryAction::Warning` at N/2 threshold |
| P1 | Implement exponential backoff | `watchdog.cpp` | 51-53 | Use `recovery_count_` to scale cooldown exponentially |
| P1 | Resolve callback registration | `pipeline_integration.cpp` | 120-124 | Either register callback or remove from config |

### P2 — Should Fix (Code Quality)

| Priority | Issue | File | Line | Fix |
|----------|-------|------|------|-----|
| P2 | Remove unused `<print>` include | `watchdog.cpp` | 8 | Delete line |
| P2 | Remove redundant forward declaration | `pipeline.hpp` | 38 | Delete line |
| P2 | Fix `error_string_` declaration | `pipeline.hpp` | 204 | Remove or implement |
| P2 | Composite recovery reason | `watchdog.cpp` | 56-72 | Build reason from all matching conditions |

### P3 — Documentation / Process

| Priority | Issue | Notes |
|----------|-------|-------|
| P3 | Missing `utilization_watchdog.hpp` | Document merge or provide missing file |

---

## Cross-File Consistency Assessment

| Aspect | Status | Notes |
|--------|--------|-------|
| Header conflict | N/A | Only one watchdog header exists |
| Naming conventions | PASS | `UtilizationWatchdog` in `watchdog.hpp` — acceptable but inconsistent with filename |
| Recovery callback signature | PASS | `void(RecoveryAction, const std::string&)` — consistent between header and pipeline |
| Error enum consistency | PASS | `PipelineError` used consistently; `RecoveryAction` used in both watchdog and pipeline |
| C++26 feature usage | PASS | `std::expected`, `std::print`, `[[nodiscard]]`, designated initializers all used correctly |
| Thread safety model | PASS | Single-threaded by design; no mutex needed; documented in header |

---

## Design Review Notes

1. **Per-step inline sampling architecture:** Correct. No async thread, no polling loop. The watchdog is called synchronously at each denoising step boundary. This eliminates all race conditions and keeps the design simple.

2. **State machine correctness (with fixes):** After adding WARNING state and exponential backoff, the state machine will be:
   - `None` -> `Warning` (at 4 consecutive low steps)
   - `Warning` -> `Restart` (at 8 consecutive low steps)
   - `Restart` -> cooldown with exponential delay
   - Cooldown -> `None` (reset counter, resume monitoring)

3. **No async thread:** The design correctly replaces the 500ms async polling thread with inline per-step evaluation. All state mutations happen on the pipeline thread.

4. **Memory safety:** No heap allocations in the hot path (watchdog `step()` uses only stack locals). RAII throughout. No exceptions for control flow.

---

*Report generated by senior code reviewer. All line numbers reference the current file versions as of review date.*
