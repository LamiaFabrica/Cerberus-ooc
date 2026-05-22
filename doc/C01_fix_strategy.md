# C01 Fix Strategy: Data Race in UtilizationWatchdog

## Race Inventory

`evaluate_device()` (called from `step()`, single-threaded) writes these without `mutex_`:

| Field | Write site(s) | Read by |
|---|---|---|
| `stats_.gpu_state` | evaluate_device:143,157,209 (via `current_state` ref) | `get_gpu_state()`, `get_stats()` |
| `stats_.hailo_state` | (same, via device branch) | `get_hailo_state()`, `get_stats()` |
| `stats_.gpu_recovery_count` | evaluate_device:198-199 | `get_stats()` |
| `stats_.hailo_recovery_count` | evaluate_device:247-248 | `get_stats()` |

These 4 fields have concurrent reads from `get_*()` (telemetry thread) and concurrent writes from `evaluate_device()` (step thread) — UB.

Fields already OK (written under `scoped_lock` in `step()`):
- `stats_.total_steps`, `stats_.last_gpu_util`, `stats_.last_hailo_util`
- `stats_.gpu_consecutive_low`, `stats_.hailo_consecutive_low`

Internal counters (`gpu_consecutive_`, `hailo_consecutive_`, `gpu_in_recovery_`,
`hailo_in_recovery_`, `gpu_recovery_count_`, `hailo_recovery_count_`) are
single-threaded within `step()` — no race from `get_*()` readers.

---

## Strategy A: Minimally-invasive Atomics

### Philosophy
Fix only the 4 data-race fields. Everything else stays under `mutex_`.
Smallest diff, lowest risk.

### Member declarations — what changes

**Become `std::atomic` (pulled OUT of `WatchdogStatistics`):**
```cpp
std::atomic<WatchdogState>    gpu_state_{WatchdogState::NORMAL};
std::atomic<WatchdogState>    hailo_state_{WatchdogState::NORMAL};
std::atomic<std::uint32_t>    gpu_recovery_count_{0};
std::atomic<std::uint32_t>    hailo_recovery_count_{0};
```

**Stay under `mutex_`:**
```cpp
mutable std::mutex mutex_;
WatchdogStatistics stats_{};   // now minus gpu_state, hailo_state,
                               // gpu_recovery_count, hailo_recovery_count
```
(The `WatchdogStatistics` struct itself gets those 4 fields removed.)

### Refactored `step()` — pseudocode

```cpp
std::optional<RecoveryAction> step(uint32_t step_num,
                                    const UtilizationSnapshot& gpu_snap,
                                    const UtilizationSnapshot& hailo_snap)
{
    // ── GPU evaluation (single-threaded, no lock needed for internal counters) ──
    auto gpu_action = evaluate_device(
        ComputeUnit::GPU_780M, gpu_snap.utilization, gpu_snap.temperature,
        cfg_.gpu_low_threshold, cfg_.gpu_critical_threshold,
        gpu_consecutive_, gpu_in_recovery_, gpu_recovery_count_, step_num);

    // ── Hailo evaluation ──
    auto hailo_action = evaluate_device(
        ComputeUnit::HAILO_8L, hailo_snap.utilization, hailo_snap.temperature,
        cfg_.hailo_low_threshold, cfg_.hailo_critical_threshold,
        hailo_consecutive_, hailo_in_recovery_, hailo_recovery_count_, step_num);

    // ── Snap stats (mutex-guarded fields only) ──
    {
        const std::scoped_lock lock{mutex_};
        stats_.total_steps++;
        stats_.last_gpu_util  = gpu_snap.utilization;
        stats_.last_hailo_util = hailo_snap.utilization;
        stats_.gpu_consecutive_low   = gpu_consecutive_;   // from internal counters
        stats_.hailo_consecutive_low = hailo_consecutive_;
    }

    if (gpu_action)  return gpu_action;
    if (hailo_action) return hailo_action;
    return nullopt;
}
```

### Refactored `evaluate_device()` — pseudocode (critical sections only)

```cpp
optional<RecoveryAction> evaluate_device(ComputeUnit unit, float util,
    float temp, float thresh, float crit_thresh,
    uint32_t& consecutive, bool& in_recovery,
    uint32_t& recovery_count, uint32_t step)
{
    // ── Pick the right atomic ref ──
    atomic<WatchdogState>& current_state =
        (unit == GPU_780M) ? gpu_state_ : hailo_state_;

    if (util >= thresh) {
        WatchdogState old = current_state.load(relaxed);
        if (consecutive != 0)
            log_state_change(unit, old, NORMAL, util, step);
        if (old != NORMAL) {
            current_state.store(NORMAL, release);   // ★ publish to readers
            log_state_change(unit, old, NORMAL, util, step);
        }
        consecutive = 0;
        in_recovery = false;
        return nullopt;
    }

    consecutive++;

    if (util >= crit_thresh) {
        WatchdogState old = current_state.load(relaxed);
        if (old != WARNING) {
            current_state.store(WARNING, release);   // ★ publish
            log_state_change(unit, old, WARNING, util, step);
        }
        // ... thermal guard (unchanged) ...
        if (consecutive >= cfg_.consecutive_threshold && !in_recovery) {
            in_recovery = true;
            // ... max_recoveries check (unchanged) ...
            auto action = trigger_recovery(unit, step, util);
            in_recovery = false;
            // ★ Atomically bump the exported recovery count
            if (unit == GPU_780M)
                gpu_recovery_count_.fetch_add(1, release);   // ★
            else
                hailo_recovery_count_.fetch_add(1, release);  // ★
            // ... build RecoveryAction, return ...
        }
        return nullopt;
    }

    // ── CRITICAL zone ──
    WatchdogState old = current_state.load(relaxed);
    current_state.store(CRITICAL, release);   // ★ publish
    if (old != CRITICAL)
        log_state_change(unit, old, CRITICAL, util, step);
    // ... thermal guard (unchanged) ...
    if (!in_recovery) {
        in_recovery = true;
        // ... max_recoveries check ...
        auto action = trigger_recovery(unit, step, util);
        in_recovery = false;
        if (unit == GPU_780M)
            gpu_recovery_count_.fetch_add(1, release);   // ★
        else
            hailo_recovery_count_.fetch_add(1, release);  // ★
        // ... build RecoveryAction, return ...
    }
    return nullopt;
}
```

### Refactored `get_*()` — pseudocode

```cpp
WatchdogState get_gpu_state() const noexcept {
    return gpu_state_.load(acquire);        // ★ paired with store(release)
}
WatchdogState get_hailo_state() const noexcept {
    return hailo_state_.load(acquire);
}
WatchdogStatistics get_stats() const noexcept {
    // Mutex-guarded fields:
    const scoped_lock lock{mutex_};
    WatchdogStatistics s = stats_;           // copy the non-atomic part
    // Atomic fields (snapshot individually):
    s.gpu_state          = gpu_state_.load(acquire);
    s.hailo_state        = hailo_state_.load(acquire);
    s.gpu_recovery_count   = gpu_recovery_count_.load(acquire);
    s.hailo_recovery_count = hailo_recovery_count_.load(acquire);
    return s;
}
```

### Memory ordering requirements

| Operation | Ordering | Rationale |
|---|---|---|
| `current_state.store(NORMAL/WARNING/CRITICAL, release)` | `release` | Before: all prior writes (consecutive, in_recovery logic) happen-before the state update. After: paired with `get_*().load(acquire)`. |
| `recovery_count_.fetch_add(1, release)` | `release` | Makes increment visible to `get_stats()` reader. |
| `get_*().load(acquire)` | `acquire` | Paired with the writer's `release`. Guarantees reader sees the state that was true *at or after* the write. |
| `current_state.load(relaxed)` inside `evaluate_device()` | `relaxed` | This is the step thread itself reading its own last write — no concurrent writer, so `relaxed` is sufficient. |

### Risk Assessment

- **`get_stats()` sees a partially-incoherent snapshot**: The mutex-guarded fields (total_steps, last_util, consecutive_low) are read at a slightly different moment than the atomic fields (state, recovery_count). This was already true before (the entire stats_ was copied under one lock, but the underlying data raced). Now it's explicitly a staggered snapshot — two consecutive snapshots might show recovery_count incremented but state still WARNING. Acceptable for monitoring; the next call to get_stats() will catch up.

- **`gpu_recovery_count_` vs `stats_.gpu_recovery_count` double-track**: `trigger_recovery()` still increments `gpu_recovery_count_` (internal, for backoff/max check). `stats_.gpu_recovery_count` was previously a separate variable. Now `gpu_recovery_count_` stays non-atomic (single-threaded) and the exported atomic is `gpu_recovery_count_`'s atomic replacement — wait, actually no. Currently there are TWO counters: `gpu_recovery_count_` (member, used in internal logic) and `stats_.gpu_recovery_count` (exported). In our design, we replace `stats_.gpu_recovery_count` with the atomic `gpu_recovery_count_`. But `gpu_recovery_count_` is also used internally. We need to either:
  1. Keep `gpu_recovery_count_` as the internal counter AND have a separate atomic for export, OR
  2. Combine them into one atomic.
  
  Option (2) is cleanest: remove the separate `gpu_recovery_count_` member and use the atomic for both. `trigger_recovery()` does `fetch_add(1)` on the atomic. `evaluate_device()` reads it with `.load(relaxed)` for max_recoveries checks. The `++` on lines 198-199 and 247-248 are removed (already incremented in `trigger_recovery`). This changes the increment count: currently `gpu_recovery_count_` is incremented in `trigger_recovery` and `stats_.gpu_recovery_count` is incremented separately after. If we unify, it's incremented once. **This is actually correct** — the double increment was arguably a bug (recovery_count should increment once per recovery attempt).

- **Move semantics**: `UtilizationWatchdog` currently has `= default` move constructor. `std::atomic<T>` is not movable. This is a **breaking API change** for Strategy A. Workaround: define a custom move constructor that does atomic loads/stores. Or accept that the class becomes non-movable.

  **Mitigation for Strategy A**: The class is used via `unique_ptr<UtilizationWatchdog>` in Pipeline (see pipeline.hpp:241), so the move constructor is never exercised. The `= default` move can be replaced with `= delete` without breaking the codebase. The class should be declared non-movable.

- **Thermal guard `cfg_.thermal_throttle_threshold_c`**: `cfg_` is const after construction and read-only — no race.

---

## Strategy B: Full Lock-Free

### Philosophy
Remove `mutex_` entirely. All shared state becomes `std::atomic`. No locks anywhere.
Higher complexity, cleaner conceptual model.

### Member declarations

```cpp
// Configuration (read-only after construction)
WatchdogConfig   cfg_;
RecoveryCallback on_recovery_;
AlertCallback    on_alert_;

// ── All shared state is atomic ──
std::atomic<std::uint32_t> total_steps_{0};
std::atomic<float>         last_gpu_util_{0.0f};
std::atomic<float>         last_hailo_util_{0.0f};

std::atomic<std::uint32_t> gpu_consecutive_{0};
std::atomic<std::uint32_t> hailo_consecutive_{0};
std::atomic<bool>          gpu_in_recovery_{false};
std::atomic<bool>          hailo_in_recovery_{false};
std::atomic<std::uint32_t> gpu_recovery_count_{0};
std::atomic<std::uint32_t> hailo_recovery_count_{0};

std::atomic<WatchdogState> gpu_state_{WatchdogState::NORMAL};
std::atomic<WatchdogState> hailo_state_{WatchdogState::NORMAL};

// mutex_ — REMOVED
```

The `WatchdogStatistics` struct is no longer a live member. `get_stats()` builds a
`WatchdogStatistics` on-the-fly by loading each atomic individually.

### Core challenge: making `trigger_recovery()` "atomic"

`trigger_recovery()` must:
1. Increment `recovery_count`
2. Sleep for backoff
3. Call user callback (may sleep 30s)

This cannot be truly atomic. But correctness only requires that **no two callers**

invoke recovery for the **same device** simultaneously.

**Solution: `compare_exchange_strong` on `in_recovery_` flag.**

The `in_recovery_` flag acts as a spinlock/try-lock:
- Before calling `trigger_recovery`, do `compare_exchange_strong(false → true)`.
- If CAS succeeds, we "own" the recovery slot. Call the (sleeping) callback.
- If CAS fails, someone else is already in recovery. Skip (return nullopt).

But wait — in current usage, `step()` is single-threaded. This CAS is defense-in-depth
for the case where someone later calls `step()` concurrently, OR where `get_*()` is
called during recovery (which is fine — they just read atomics and don't trigger).

Actually, the real design for Strategy B is: `step()` IS called from one thread, but
the atomic ordering guarantees mean `get_*()` from any thread reads coherent snapshots.

For `trigger_recovery()` under lock-free design, the callback invocation itself does
NOT need to be atomic with the state change. What matters is:
- The state transition to CRITICAL/WARNING is visible to observers
- The recovery_count increment is visible
- The in_recovery flag prevents re-entrant recovery for the same device

Here's how it works in practice:

### Refactored `step()` — pseudocode

```cpp
std::optional<RecoveryAction> step(uint32_t step_num,
                                    const UtilizationSnapshot& gpu_snap,
                                    const UtilizationSnapshot& hailo_snap)
{
    // ── Snapshot-read stats (release the writer's data) ──
    total_steps_.fetch_add(1, release);
    last_gpu_util_.store(gpu_snap.utilization, relaxed);
    last_hailo_util_.store(hailo_snap.utilization, relaxed);

    auto gpu_action = evaluate_device(ComputeUnit::GPU_780M,
        gpu_snap.utilization, gpu_snap.temperature, step_num);
    auto hailo_action = evaluate_device(ComputeUnit::HAILO_8L,
        hailo_snap.utilization, hailo_snap.temperature, step_num);

    if (gpu_action)  return gpu_action;
    if (hailo_action) return hailo_action;
    return nullopt;
}
```

### Refactored `evaluate_device()` — pseudocode

```cpp
optional<RecoveryAction> evaluate_device(ComputeUnit unit,
    float util, float temp, uint32_t step)
{
    // ── Select the right atomics ──
    atomic<uint32_t>&  consecutive     = (unit == GPU) ? gpu_consecutive_     : hailo_consecutive_;
    atomic<bool>&      in_recovery     = (unit == GPU) ? gpu_in_recovery_     : hailo_in_recovery_;
    atomic<uint32_t>&  recovery_count  = (unit == GPU) ? gpu_recovery_count_  : hailo_recovery_count_;
    atomic<WatchdogState>& state       = (unit == GPU) ? gpu_state_           : hailo_state_;
    float low_thresh   = (unit == GPU) ? cfg_.gpu_low_threshold    : cfg_.hailo_low_threshold;
    float crit_thresh  = (unit == GPU) ? cfg_.gpu_critical_threshold : cfg_.hailo_critical_threshold;

    // ── Fast path: NORMAL ──
    if (util >= low_thresh) {
        WatchdogState cur = state.load(relaxed);
        if (consecutive.load(relaxed) != 0)
            log_state_change(unit, cur, NORMAL, util, step);
        state.store(NORMAL, release);
        consecutive.store(0, relaxed);
        in_recovery.store(false, relaxed);
        return nullopt;
    }

    // ── Below low threshold ──
    uint32_t c = consecutive.fetch_add(1, relaxed) + 1;

    if (util >= crit_thresh) {
        // WARNING zone
        WatchdogState cur = state.load(relaxed);
        if (cur != WARNING) {
            state.store(WARNING, release);
            log_state_change(unit, cur, WARNING, util, step);
        }

        // ── Thermal guard ──
        if (temp > cfg_.thermal_throttle_threshold_c) {
            /* alert; return nullopt; */
        }

        // ── Threshold check ──
        if (c >= cfg_.consecutive_threshold) {
            // ★ Try to claim the recovery slot
            bool expected = false;
            if (in_recovery.compare_exchange_strong(expected, true, acquire, relaxed)) {
                // ── We own the slot ──
                uint32_t rc = recovery_count.load(relaxed);
                if (rc >= cfg_.max_recoveries) {
                    state.store(NORMAL, release);  // back to baseline
                    consecutive.store(0, relaxed);
                    in_recovery.store(false, release);
                    return RecoveryAction{ .result = FATAL, .device = unit,
                        .step = step, .util_at_fault = util, .reason = "max exhausted" };
                }
                // ★ Increment recovery_count AND call callback — we hold the slot
                recovery_count.fetch_add(1, release);
                auto result = trigger_recovery(unit, step, util);
                // ★ Release the slot
                in_recovery.store(false, release);
                return RecoveryAction{ .result = result.value_or(FATAL),
                    .device = unit, .step = step, .util_at_fault = util, .reason = "..." };
            }
            // else: another caller already in recovery — skip
        }
        return nullopt;
    }

    // ── CRITICAL zone ──
    WatchdogState cur = state.load(relaxed);
    state.store(CRITICAL, release);
    if (cur != CRITICAL)
        log_state_change(unit, cur, CRITICAL, util, step);

    if (temp > cfg_.thermal_throttle_threshold_c) {
        /* alert; return nullopt; */
    }

    bool expected = false;
    if (in_recovery.compare_exchange_strong(expected, true, acquire, relaxed)) {
        uint32_t rc = recovery_count.load(relaxed);
        if (rc >= cfg_.max_recoveries) {
            state.store(NORMAL, release);
            consecutive.store(0, relaxed);
            in_recovery.store(false, release);
            return RecoveryAction{ .result = FATAL, .device = unit, ... };
        }
        recovery_count.fetch_add(1, release);
        auto result = trigger_recovery(unit, step, util);
        in_recovery.store(false, release);
        return RecoveryAction{ .result = result.value_or(FATAL), ... };
    }
    return nullopt;
}
```

### Refactored `trigger_recovery()` — pseudocode

```cpp
expected<RecoveryResult, string> trigger_recovery(ComputeUnit unit,
    uint32_t step, float util)
{
    // recovery_count already incremented by caller (fetched before calling us)
    atomic<uint32_t>& rc = (unit == GPU) ? gpu_recovery_count_ : hailo_recovery_count_;
    uint32_t current_count = rc.load(relaxed);   // already includes our increment

    float delay_ms = compute_backoff_ms(current_count);
    print("[watchdog] [{0}] step={1} util={2:.1f}% — recovery #{3}/{4} with {5:.0f}ms backoff\n",
          device_name(unit), step, util, current_count, cfg_.max_recoveries, delay_ms);

    if (delay_ms > 0.0f)
        this_thread::sleep_for(milliseconds(static_cast<int64_t>(delay_ms)));

    if (!on_recovery_)
        return unexpected("No recovery callback");

    auto result = on_recovery_(unit, step, util);
    // ... logging ...
    return result;
}
```

### Memory ordering requirements

| Operation | Ordering | Rationale |
|---|---|---|
| `state.store(ANY, release)` | `release` | All prior metadata writes (consecutive reset, log_state_change side effects) happen-before the state becomes visible to readers. Paired with get_*() acquire loads. |
| `get_*().load(acquire)` | `acquire` | Paired with writer's release. Reader sees consistent post-update state. |
| `consecutive.fetch_add(1, relaxed)` | `relaxed` | Counter — no ordering dependency. Only this thread reads it next. |
| `in_recovery.CAS(false→true, acquire, relaxed)` | `acquire` on success | If we win the CAS, we "acquire" the slot — loads after CAS (recovery_count load, callback invocation) see correct state. |
| `in_recovery.store(false, release)` | `release` | Releasing the slot — all recovery side-effects (stats updates, log lines) happen-before another thread can CAS back in. |
| `recovery_count.fetch_add(1, release)` | `release` | Makes increment visible to observers (get_stats()). |
| `recovery_count.load(relaxed)` in evaluate_device | `relaxed` | Only read here to check max_recoveries — ordering not critical since we hold in_recovery slot. |
| `total_steps_.fetch_add(1, release)` | `release` | Paired with get_stats(). |
| `last_*_util_.store(val, relaxed)` | `relaxed` | Single scalar — no ordering dependency. get_stats() acquire catches up. |

### How the CAS achieves "atomic" increment+callback

The key insight: the user callback is invoked from `trigger_recovery()`, which is called
AFTER `in_recovery_` has been atomically set to `true` via CAS. While `in_recovery_` is
`true`, NO other caller (even from another thread) can enter recovery for the same device.
The callback may sleep 30s — but during that sleep, `in_recovery_` is `true`, so any
concurrent `evaluate_device()` call will fail the CAS and skip. When the callback returns,
`in_recovery_` is set back to `false`, releasing the slot.

This is NOT making the callback atomic — it's making the **decision to call the callback**
exclusive. This is exactly what a spinlock does, but without blocking (non-contending CASs
are lock-free).

### Risk Assessment

- **Move constructor broken**: Same as Strategy A — `std::atomic` not movable. Must
  delete move or write custom move. Non-issue for current Pipeline usage (unique_ptr).

- **`get_stats()` snapshot incoherence**: Each field is loaded individually. Between loads,
  another `step()` call may advance state. Result: `get_stats()` can return a "Frankenstein"
  snapshot where, e.g., `total_steps=100` but `gpu_state=WARNING` from step 101. This is
  acceptable for monitoring/metrics — the next poll catches up. If strict snapshot
  consistency is required, a seqlock would be needed (adds complexity).

- **ABA on `in_recovery_` CAS**: Two threads could observe in_recovery=false, both CAS,
  one wins, one loses (fine). After the winner releases, the loser's CAS already failed —
  no ABA issue because the loser doesn't retry. If we ever add retry logic, ABA could
  be a problem (the recovery flag goes false→true→false between retries). Mitigation:
  don't retry.

- **Callback invocation while `in_recovery_` is true means Pipeline stalls**: During the
  30s callback sleep, `step()` returns `nullopt` (since `in_recovery_` is already true,
  the CAS in the next step fails). This means pipeline continues WITHOUT recovery for
  that step — if the device is genuinely broken, the next step hits CRITICAL again, fails
  CAS (still in recovery), skips again. This is acceptable: recovery is in-flight. Once
  it completes, the next low-util step triggers a fresh attempt.

  **BUT**: if the callback sleeps 30s on step N, and step N+1 is NORMAL (util >=
  threshold), `in_recovery_` gets cleared at the top of `evaluate_device()` (the "Fast
  path: NORMAL" section sets it to false). This is correct — the device is healthy again,
  recovery was unnecessary but harmless.

- **`trigger_recovery()` no longer owns the increment**: In the original code,
  `trigger_recovery()` both increments recovery_count and calls the callback. In Strategy B,
  the caller (`evaluate_device`) does the increment before calling `trigger_recovery`.
  This means `trigger_recovery`'s `compute_backoff_ms` sees the new count — correct.
  But it also means if `trigger_recovery` is called from anywhere else without the
  CAS guard, the count would be wrong. Risk is low since `trigger_recovery` is private
  and only called from `evaluate_device`.

---

## Comparison Matrix

| Dimension | Strategy A (Minimally-invasive atomics) | Strategy B (Full lock-free) |
|---|---|---|
| Lines changed | ~30 | ~80 |
| New atomics | 4 fields | 12 fields |
| Mutex | Kept | Removed |
| Move semantics | Broken (must delete) | Broken (must delete) |
| `get_stats()` coherence | Staggered snapshot (was UB before; now well-defined but staggered) | Frankenstein snapshot (worse than A) |
| Risk of regression | Very low — only 4 fields change | Moderate — entire state machine refactored |
| Understandability | Easy — same structure, different storage | Harder — CAS logic replaces simple bool flag |
| Extensibility (multi-threaded step) | Not safe — step() would need its own lock | Safe — CAS prevents concurrent recovery |
| Test impact | Minimal — same behavior, same test suite passes | Significant — CAS semantics change edge cases |

---

## Recommendation

**Recommend Strategy A.**

Rationale:

1. **The data race is real but narrow**. Only 4 fields cross the read/write boundary
   between the step thread and telemetry readers. Fixing just those 4 eliminates UB
   without touching the well-understood state machine logic.

2. **Strategy B solves a problem that doesn't exist**. The watchdog is called
   synchronously from Pipeline's step loop. There is no concurrent `step()` caller.
   The CAS logic in Strategy B is dead code — it protects against multiple writers
   that will never exist. It adds complexity with no benefit.

3. **The "Frankenstein snapshot" in Strategy B is worse than A's staggered snapshot**.
   In Strategy A, 7 of 9 fields in WatchdogStatistics are read under one lock (atomic
   snapshot of the non-racing fields). In Strategy B, all 12 fields are loaded
   individually with no coherence guarantee.

4. **`in_recovery_` as CAS is less robust than as a simple bool**. Under Strategy A,
   `in_recovery_` is a simple flag set/clear in `evaluate_device()` — its semantics
   are "we've already decided to recover, don't trigger again." Under Strategy B, it
   becomes a synchronization primitive whose acquire/release semantics interact with
   the callback's side effects in subtle ways. A thread that invokes the callback
   holds the "lock" for 30s — this is effectively a blocking mutex disguised as
   lock-free code.

5. **Strategy B's "removed mutex" is misleading**. The CAS on `in_recovery_` IS a
   mutual exclusion mechanism. You've traded `std::mutex::lock()` for
   `compare_exchange_strong` — same semantics, different implementation, harder
   to audit.

### If Strategy B is ever needed

Strategy B becomes the right answer ONLY if:
- `step()` is called from multiple threads concurrently (unlikely for a Pipeline)
- OR `trigger_recovery()` needs lock-free progress guarantees (hard real-time)

In that scenario, keep the CAS but add a **seqlock** for `get_stats()` to provide
coherent snapshots, and ensure the callback timeout is bounded (not user-controlled).

---

## Implementation Plan (Strategy A)

1. Remove 4 fields from `WatchdogStatistics` struct
2. Add 4 `std::atomic` members to class (see member declarations above)
3. Rewrite `evaluate_device()` to use atomic loads/stores for state and
   `fetch_add` for recovery_count (with correct ordering)
4. Unify `gpu_recovery_count_` with the atomic: remove the non-atomic
   `gpu_recovery_count_` member, use the atomic for all internal checks
5. Rewrite `get_stats()`, `get_gpu_state()`, `get_hailo_state()` to use
   `acquire` loads (remove mutex from these — they become lock-free)
6. Delete move constructor/assignment
7. Run full test suite to verify no behavioral changes
