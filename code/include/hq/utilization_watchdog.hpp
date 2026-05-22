#pragma once
/// @file utilization_watchdog.hpp
/// @copyright Copyright (c) 2026 D Hargreaves (AKA Roylepython). LamiaFabrica. All rights reserved.
/// Utilization watchdog: monitors GPU and Hailo step-level performance
/// with a 3-state machine (NORMAL / WARNING / CRITICAL) and per-device
/// recovery backed by exponential backoff.
///
/// @code{.cpp}
///   hq::UtilizationWatchdog wdog(cfg,
///       [](hq::ComputeUnit u, uint32_t s, float util) {
///           rebuild_session(u); return hq::RecoveryResult::SUCCESS;
///       },
///       [](hq::ComputeUnit u, uint32_t s, float util,
///          const std::string& msg) { std::print("ALERT: {}\n", msg); }
///   );
///   auto action = wdog.step(step_num, gpu_snap, hailo_snap);
///   if (action) { /* recovery was triggered */ }
/// @endcode

#include "hq/cxx26_features.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#if UM790_HAS_STD_EXPECTED
#  include <expected>
#else
#  error "utilization_watchdog.hpp requires std::expected (<expected>) — GCC >= 14 or Clang >= 18 with C++26 enabled"
#endif
#if UM790_HAS_STD_FORMAT
#  include <format>
#endif
#include <functional>
#include <mutex>
#include <optional>
#if UM790_HAS_STD_PRINT
#  include <print>
#endif
#include <string>

namespace hq {

// ---------------------------------------------------------------------------
/// @brief Accelerator types tracked by the watchdog.
// ---------------------------------------------------------------------------
enum class ComputeUnit : std::uint8_t { GPU_780M = 0, HAILO_8L = 1 };

// ---------------------------------------------------------------------------
/// @brief Watchdog health state per device.
///
/// - NORMAL  — utilization is above the low threshold.
/// - WARNING — utilization dipped below the low threshold; counting steps.
/// - CRITICAL — utilization fell below the critical threshold; immediate
///              recovery is triggered.
// ---------------------------------------------------------------------------
enum class WatchdogState : std::uint8_t { NORMAL = 0, WARNING = 1, CRITICAL = 2 };

// ---------------------------------------------------------------------------
/// @brief Result reported by a recovery callback.
///
/// - SUCCESS — device is healthy again.
/// - PARTIAL — recovery helped but may need another attempt.
/// - FATAL   — device cannot be recovered; pipeline should abort.
// ---------------------------------------------------------------------------
enum class RecoveryResult : std::uint8_t { SUCCESS = 0, PARTIAL = 1, FATAL = 2 };

// ---------------------------------------------------------------------------
/// @brief Per-device utilization snapshot captured at a single denoising step.
// ---------------------------------------------------------------------------
struct UtilizationSnapshot {
    ComputeUnit   device{ComputeUnit::GPU_780M}; ///< Which accelerator
    std::uint32_t step{0};            ///< Denoising step index
    float         utilization{0.0f};     ///< Utilization percentage [0, 100]
    float         temperature{0.0f};     ///< Die temperature (Celsius)
    float         power_watts{0.0f};     ///< Power draw in watts
    bool          device_healthy{true};  ///< False if the driver reported an error
};

// ---------------------------------------------------------------------------
/// @brief Action returned by UtilizationWatchdog::step() when recovery fires.
// ---------------------------------------------------------------------------
struct RecoveryAction {
    RecoveryResult result{RecoveryResult::SUCCESS}; ///< Outcome of the recovery attempt
    ComputeUnit    device{ComputeUnit::GPU_780M};       ///< Which accelerator was recovered
    std::uint32_t  step{0};         ///< Step number where recovery triggered
    float          util_at_fault{0.0f};///< Utilization that caused the trigger
    std::string    reason;       ///< Human-readable description
};

// ---------------------------------------------------------------------------
/// @brief Configuration for the utilization watchdog.
///
/// Each device has a low threshold (enters WARNING) and a critical threshold
/// (enters CRITICAL and triggers immediate recovery).
// ---------------------------------------------------------------------------
struct WatchdogConfig {
    float         gpu_low_threshold = 60.0f;        ///< % below => WARNING
    float         gpu_critical_threshold = 40.0f;   ///< % below => CRITICAL
    float         hailo_low_threshold = 60.0f;      ///< % below => WARNING
    float         hailo_critical_threshold = 40.0f; ///< % below => CRITICAL
    std::uint32_t consecutive_threshold = 8;        ///< steps before recovery
    std::uint32_t max_recoveries = 10;              ///< max before giving up
    float         backoff_base_ms = 100.0f;         ///< base recovery delay
    float         backoff_max_ms = 30000.0f;        ///< cap at 30 s
    float         thermal_throttle_threshold_c = 85.0f; ///< temp above => skip recovery
};

// ---------------------------------------------------------------------------
/// @brief Read-only statistics exported by the watchdog.
///
/// This is a pure POD — get_stats() populates it from the mutex-guarded
/// stats_ fields and the lock-free atomic members.
// ---------------------------------------------------------------------------
struct WatchdogStatistics {
    std::uint32_t total_steps = 0;           ///< How many step() calls served
    std::uint32_t gpu_recovery_count = 0;    ///< GPU recoveries attempted
    std::uint32_t hailo_recovery_count = 0;  ///< Hailo recoveries attempted
    std::uint32_t gpu_consecutive_low = 0;   ///< Current GPU warning streak
    std::uint32_t hailo_consecutive_low = 0; ///< Current Hailo warning streak
    WatchdogState gpu_state = WatchdogState::NORMAL;   ///< Current GPU state
    WatchdogState hailo_state = WatchdogState::NORMAL; ///< Current Hailo state
    float         last_gpu_util = 0.0f;      ///< Last GPU utilization seen
    float         last_hailo_util = 0.0f;    ///< Last Hailo utilization seen
};

// ---------------------------------------------------------------------------
/// @brief Signature for user-supplied recovery callbacks.
///
/// Called when the watchdog decides a device needs recovery.  The callback
/// should attempt to restore the device (e.g. rebuild the ONNX session) and
/// report whether it succeeded.
// ---------------------------------------------------------------------------
using RecoveryCallback =
    std::function<std::expected<RecoveryResult, std::string>(
        ComputeUnit unit, std::uint32_t step, float util_at_fault)>;

// ---------------------------------------------------------------------------
/// @brief Signature for optional alert / logging callbacks.
///
/// Invoked on every state transition so that callers can forward alerts to
/// their own telemetry system.
// ---------------------------------------------------------------------------
using AlertCallback =
    std::function<void(ComputeUnit unit, std::uint32_t step, float util,
                       const std::string& message)>;

// ---------------------------------------------------------------------------
/// @class UtilizationWatchdog
/// @brief Step-level utilization monitor with automatic recovery triggering.
///
/// Watches GPU and Hailo utilization independently at each denoising step.
///
/// State machine per device:
/// @verbatim
///   NORMAL
///     |
///     | util < low_threshold
///     v
///   WARNING  ----(consecutive >= 8 & not in_recovery)----> trigger_recovery()
///     |                                                        |
///     | util < critical_threshold                              | callback returns
///     v                                                        v
///   CRITICAL --(immediate, not in_recovery)--> trigger_recovery()
/// @endverbatim
///
/// After a recovery callback returns (regardless of result) the device exits
/// the @c in_recovery flag so that the next low-utilization step can trigger
/// another recovery if needed.
///
/// Thread-safety (C01 fix — mixed atomics + mutex):
///   - evaluate_device() writes state-machine fields via std::atomic
///     WITHOUT holding mutex_.  Readers load the same atomics, so no
///     data-race on the hot path.
///   - Low-contention POD fields (total_steps, last_gpu_util, last_hailo_util,
///     gpu_consecutive_low, hailo_consecutive_low) remain under mutex_
///     because they are updated once per step and read infrequently.
///   - trigger_recovery() does NOT hold mutex_ while sleeping — it only
///     touches atomic recovery_count_.
// ---------------------------------------------------------------------------
class UtilizationWatchdog {
public:
    /// @brief Construct the watchdog with configuration and callbacks.
    /// @param cfg          Configuration thresholds and limits.
    /// @param on_recovery  Mandatory recovery callback.
    /// @param on_alert     Optional alert callback for state transitions.
    explicit UtilizationWatchdog(WatchdogConfig cfg,
                                 RecoveryCallback on_recovery,
                                 AlertCallback on_alert = nullptr);

    /// Destructor — trivial, all members clean themselves up.
    ~UtilizationWatchdog() = default;

    // Movable but not copyable (mutex + atomics members)
    UtilizationWatchdog(const UtilizationWatchdog&) = delete;
    UtilizationWatchdog& operator=(const UtilizationWatchdog&) = delete;
    UtilizationWatchdog(UtilizationWatchdog&&) = default;
    UtilizationWatchdog& operator=(UtilizationWatchdog&&) = default;

    /// @brief Evaluate one denoising step.
    ///
    /// Called synchronously at the end of each denoising step.  Both snapshots
    /// are evaluated independently.  If either device triggers recovery the
    /// corresponding RecoveryAction is returned.
    ///
    /// @param step_num   Current denoising step index.
    /// @param gpu_snap   GPU utilization snapshot.
    /// @param hailo_snap Hailo utilization snapshot.
    /// @return std::optional<RecoveryAction> populated if recovery fired.
    [[nodiscard("recovery action must be checked")]]
    std::optional<RecoveryAction> step(std::uint32_t step_num,
                                       const UtilizationSnapshot& gpu_snap,
                                       const UtilizationSnapshot& hailo_snap);

    /// @return Current GPU health state (lock-free atomic load).
    [[nodiscard]] WatchdogState get_gpu_state() const noexcept;

    /// @return Current Hailo health state (lock-free atomic load).
    [[nodiscard]] WatchdogState get_hailo_state() const noexcept;

    /// @return Copy of current statistics (mutex + atomic loads).
    [[nodiscard]] WatchdogStatistics get_stats() const noexcept;

    /// @brief Reset only the consecutive-low counters (lock-free).
    void reset_counters() noexcept;

    /// @brief Reset all internal state — counters, recovery flags, and
    ///        per-device recovery counts.  Atomically resets atomics;
    ///        mutex-guarded reset for stats_ POD.
    void reset_all() noexcept;

private:
    WatchdogConfig   cfg_;         ///< Thresholds and limits
    RecoveryCallback on_recovery_; ///< User-supplied recovery handler
    AlertCallback    on_alert_;    ///< Optional state-transition logger

    // -----------------------------------------------------------------------
    // Mutex-guarded POD fields (low-contention, updated once per step)
    // -----------------------------------------------------------------------
    mutable std::mutex mutex_;     ///< Protects the five fields below only
    struct {
        std::uint32_t total_steps           = 0;
        float         last_gpu_util         = 0.0f;
        float         last_hailo_util       = 0.0f;
        std::uint32_t gpu_consecutive_low   = 0;
        std::uint32_t hailo_consecutive_low = 0;
    } stats_;  ///< The five fields still under mutex_ (C01: reduced scope)

    // -----------------------------------------------------------------------
    // Lock-free atomic state (C01: what evaluate_device writes / get_* reads)
    // -----------------------------------------------------------------------
    std::atomic<std::uint32_t> gpu_consecutive_{0};
    std::atomic<std::uint32_t> hailo_consecutive_{0};
    std::atomic<bool>          gpu_in_recovery_{false};
    std::atomic<bool>          hailo_in_recovery_{false};
    std::atomic<std::uint32_t> gpu_recovery_count_{0};
    std::atomic<std::uint32_t> hailo_recovery_count_{0};
    std::atomic<WatchdogState> gpu_state_{WatchdogState::NORMAL};
    std::atomic<WatchdogState> hailo_state_{WatchdogState::NORMAL};

    /// @brief Evaluate a single device against its thresholds.
    ///
    /// Implements the 3-state machine on lock-free atomics:
    ///   - util >= low_threshold        -> NORMAL,  reset counter, no action
    ///   - util >= critical_threshold   -> WARNING, increment counter, recover
    ///                                     if threshold reached
    ///   - util <  critical_threshold   -> CRITICAL, immediate recovery
    ///
    /// @param unit              Which accelerator.
    /// @param util              Current utilization [0,100].
    /// @param temperature       Die temperature (Celsius).
    /// @param threshold         Low threshold (WARNING boundary).
    /// @param critical_threshold Critical threshold (CRITICAL boundary).
    /// @param consecutive       std::atomic<uint32_t>& counter for consecutive
    ///                          low steps.
    /// @param in_recovery       std::atomic<bool>& — true while recovery runs.
    /// @param recovery_count    std::atomic<uint32_t>& total recovery attempts.
    /// @param current_state     std::atomic<WatchdogState>& current health.
    /// @param step_num          Current denoising step.
    /// @return std::optional<RecoveryAction> if recovery was triggered.
    [[nodiscard]] std::optional<RecoveryAction> evaluate_device(
        ComputeUnit unit, float util, float temperature,
        float threshold, float critical_threshold,
        std::atomic<std::uint32_t>& consecutive,
        std::atomic<bool>&          in_recovery,
        std::atomic<std::uint32_t>& recovery_count,
        std::atomic<WatchdogState>& current_state,
        std::uint32_t step_num);

    /// @brief Invoke the user recovery callback with exponential backoff delay.
    ///
    /// Sleeps for backoff_ms (computed from recovery_count) before calling
    /// @c on_recovery_.  Atomically increments the per-device recovery_count.
    ///
    /// @param unit  Which accelerator needs recovery.
    /// @param step  Step number that triggered recovery.
    /// @param util  Utilization that caused the fault.
    /// @return Expected recovery result or an error string.
    [[nodiscard]] std::expected<RecoveryResult, std::string>
    trigger_recovery(ComputeUnit unit, std::uint32_t step, float util);

    /// @brief Compute backoff delay: base * 2^(count-1), capped at max.
    /// @param recovery_count  Number of recovery attempts so far (plain int).
    /// @return Delay in milliseconds.
    [[nodiscard]] float compute_backoff_ms(
        std::uint32_t recovery_count) const noexcept;

    /// @brief Check thermal throttle and skip recovery if needed.
    /// @return true if recovery should be skipped due to thermal throttling.
    [[nodiscard]] bool thermal_skip_recovery_(ComputeUnit unit,
                                              float temperature,
                                              float util,
                                              std::uint32_t step,
                                              bool is_critical);

    /// @brief Emit a state-transition log line via std::print.
    /// @param unit     Which accelerator changed state.
    /// @param old_state Previous state.
    /// @param new_state New state.
    /// @param util     Current utilization.
    /// @param step     Current step number.
    void log_state_change(ComputeUnit unit, WatchdogState old_state,
                          WatchdogState new_state, float util,
                          std::uint32_t step);

    /// @brief Human-readable name for a WatchdogState.
    [[nodiscard]] static constexpr const char* state_name(
        WatchdogState s) noexcept;

    /// @brief Human-readable name for a ComputeUnit.
    [[nodiscard]] static constexpr const char* device_name(
        ComputeUnit u) noexcept;
};

/// @brief Convert WatchdogState to human-readable string.
[[nodiscard]] inline std::string to_string(WatchdogState s) {
    switch (s) {
        case WatchdogState::NORMAL:   return "NORMAL";
        case WatchdogState::WARNING:  return "WARNING";
        case WatchdogState::CRITICAL: return "CRITICAL";
    }
    return "UNKNOWN";
}

/// @brief Convert RecoveryResult to human-readable string.
[[nodiscard]] inline std::string to_string(RecoveryResult r) {
    switch (r) {
        case RecoveryResult::SUCCESS: return "SUCCESS";
        case RecoveryResult::PARTIAL: return "PARTIAL";
        case RecoveryResult::FATAL:   return "FATAL";
    }
    return "UNKNOWN";
}

/// @brief Convert ComputeUnit to human-readable string.
[[nodiscard]] inline std::string to_string(ComputeUnit u) {
    switch (u) {
        case ComputeUnit::GPU_780M:  return "GPU_780M";
        case ComputeUnit::HAILO_8L:  return "HAILO_8L";
    }
    return "UNKNOWN";
}

} // namespace hq
