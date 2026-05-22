/// @file utilization_watchdog.cpp
/// Full implementation of the UtilizationWatchdog 3-state machine.
///
/// Per-device state transitions:
///   NORMAL  <-(util >= low_threshold)-  reset counter
///   WARNING <-(low_threshold > util >= critical_threshold)- count steps
///   CRITICAL <-(util < critical_threshold)- immediate recovery
///
/// Recovery is debounced: once @c in_recovery is set, further low-utilization
/// steps on the same device are ignored until the callback returns and clears
/// the flag.  This prevents re-entrant recovery storms.

#include "hq/utilization_watchdog.hpp"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <thread>

namespace hq {

// ============================================================================
//  Public interface
// ============================================================================

// ---------------------------------------------------------------------------
UtilizationWatchdog::UtilizationWatchdog(WatchdogConfig cfg,
                                         RecoveryCallback on_recovery,
                                         AlertCallback on_alert)
    : cfg_{std::move(cfg)}
    , on_recovery_{std::move(on_recovery)}
    , on_alert_{std::move(on_alert)}
{
    // Validate configuration invariants at construction time.
    if (cfg_.gpu_critical_threshold >= cfg_.gpu_low_threshold) {
        std::print(stderr,
                   "[watchdog] WARN: gpu_critical_threshold ({:.1f}) should be "
                   "below gpu_low_threshold ({:.1f})\n",
                   cfg_.gpu_critical_threshold, cfg_.gpu_low_threshold);
    }
    if (cfg_.hailo_critical_threshold >= cfg_.hailo_low_threshold) {
        std::print(stderr,
                   "[watchdog] WARN: hailo_critical_threshold ({:.1f}) should be "
                   "below hailo_low_threshold ({:.1f})\n",
                   cfg_.hailo_critical_threshold, cfg_.hailo_low_threshold);
    }

    // Validate thermal throttle threshold (physical silicon limit: 0-125 C)
    if (cfg_.thermal_throttle_threshold_c < 0.0f ||
        cfg_.thermal_throttle_threshold_c > 125.0f) {
        std::print(
            "[watchdog] WARNING: thermal threshold {:.1f}C out of range [0, 125], "
            "clamping\n",
            cfg_.thermal_throttle_threshold_c);
        cfg_.thermal_throttle_threshold_c =
            std::clamp(cfg_.thermal_throttle_threshold_c, 0.0f, 125.0f);
    }
}

// ---------------------------------------------------------------------------
std::optional<RecoveryAction>
UtilizationWatchdog::step(std::uint32_t step_num,
                          const UtilizationSnapshot& gpu_snap,
                          const UtilizationSnapshot& hailo_snap)
{
    // --- Evaluate GPU (may trigger recovery) ---------------------------------
    auto gpu_action = evaluate_device(
        ComputeUnit::GPU_780M,
        gpu_snap.utilization,
        gpu_snap.temperature,
        cfg_.gpu_low_threshold,
        cfg_.gpu_critical_threshold,
        gpu_consecutive_,
        gpu_in_recovery_,
        gpu_recovery_count_,
        step_num);

    // --- Evaluate Hailo (may trigger recovery) -------------------------------
    auto hailo_action = evaluate_device(
        ComputeUnit::HAILO_8L,
        hailo_snap.utilization,
        hailo_snap.temperature,
        cfg_.hailo_low_threshold,
        cfg_.hailo_critical_threshold,
        hailo_consecutive_,
        hailo_in_recovery_,
        hailo_recovery_count_,
        step_num);

    // --- Export stats (under lock) -------------------------------------------
    {
        const std::scoped_lock lock{mutex_};
        stats_.total_steps++;
        stats_.last_gpu_util = gpu_snap.utilization;
        stats_.last_hailo_util = hailo_snap.utilization;
        stats_.gpu_consecutive_low = gpu_consecutive_;
        stats_.hailo_consecutive_low = hailo_consecutive_;
    }

    // Return the first action that fired.  If both fire in the same step
    // (rare — both devices simultaneously critical) we prefer the GPU action
    // because it is the primary compute unit.
    if (gpu_action) {
        return gpu_action;
    }
    if (hailo_action) {
        return hailo_action;
    }
    return std::nullopt;
}

// ---------------------------------------------------------------------------
WatchdogState UtilizationWatchdog::get_gpu_state() const noexcept
{
    const std::scoped_lock lock{mutex_};
    return stats_.gpu_state;
}

// ---------------------------------------------------------------------------
WatchdogState UtilizationWatchdog::get_hailo_state() const noexcept
{
    const std::scoped_lock lock{mutex_};
    return stats_.hailo_state;
}

// ---------------------------------------------------------------------------
WatchdogStatistics UtilizationWatchdog::get_stats() const noexcept
{
    const std::scoped_lock lock{mutex_};
    return stats_;
}

// ---------------------------------------------------------------------------
void UtilizationWatchdog::reset_counters() noexcept
{
    const std::scoped_lock lock{mutex_};
    gpu_consecutive_   = 0;
    hailo_consecutive_ = 0;
}

// ---------------------------------------------------------------------------
void UtilizationWatchdog::reset_all() noexcept
{
    const std::scoped_lock lock{mutex_};
    gpu_consecutive_    = 0;
    hailo_consecutive_  = 0;
    gpu_in_recovery_    = false;
    hailo_in_recovery_  = false;
    gpu_recovery_count_ = 0;
    hailo_recovery_count_ = 0;

    stats_.total_steps         = 0;
    stats_.gpu_recovery_count  = 0;
    stats_.hailo_recovery_count= 0;
    stats_.gpu_consecutive_low = 0;
    stats_.hailo_consecutive_low= 0;
    stats_.gpu_state           = WatchdogState::NORMAL;
    stats_.hailo_state         = WatchdogState::NORMAL;
    stats_.last_gpu_util       = 0.0f;
    stats_.last_hailo_util     = 0.0f;
}

// ============================================================================
//  Private: per-device 3-state evaluator
// ============================================================================

std::optional<RecoveryAction>
UtilizationWatchdog::evaluate_device(ComputeUnit unit,
                                     float util,
                                     float temperature,
                                     float threshold,
                                     float critical_threshold,
                                     std::uint32_t& consecutive,
                                     bool& in_recovery,
                                     std::uint32_t& recovery_count,
                                     std::uint32_t step_num)
{
    WatchdogState& current_state = (unit == ComputeUnit::GPU_780M)
                                       ? stats_.gpu_state
                                       : stats_.hailo_state;

    // ------------------------------------------------------------------
    //  NORMAL: utilization is healthy — reset the warning counter.
    // ------------------------------------------------------------------
    if (util >= threshold) {
        if (consecutive != 0) {
            log_state_change(unit, current_state, WatchdogState::NORMAL,
                             util, step_num);
        }
        if (current_state != WatchdogState::NORMAL) {
            WatchdogState old = current_state;
            current_state = WatchdogState::NORMAL;
            log_state_change(unit, old, current_state, util, step_num);
        }
        consecutive = 0;
        // Clear the recovery flag once utilization is healthy again so that
        // the next dip can trigger a fresh recovery attempt.
        in_recovery = false;
        return std::nullopt;
    }

    // Below threshold — increment consecutive counter BEFORE any branch logic.
    // This ensures thermal skips still count toward the degradation window.
    consecutive++;

    // ------------------------------------------------------------------
    //  WARNING: utilization is below the low threshold but above critical.
    // ------------------------------------------------------------------
    if (util >= critical_threshold) {
        WatchdogState old_state = current_state;
        if (current_state != WatchdogState::WARNING) {
            current_state = WatchdogState::WARNING;
            log_state_change(unit, old_state, current_state, util, step_num);
        }

        // Thermal guard: skip recovery trigger during thermal throttling,
        // but counter has already been incremented above.
        if (temperature > cfg_.thermal_throttle_threshold_c) {
            std::string msg = std::format(
                "[{}] step={} util={:.1f}% — THERMAL THROTTLING detected "
                "(temp={:.1f}C > threshold={:.1f}C), skipping recovery",
                device_name(unit), step_num, util,
                temperature, cfg_.thermal_throttle_threshold_c);
            std::print("[watchdog] {}\n", msg);
            if (on_alert_) {
                on_alert_(unit, step_num, util, msg);
            }
            return std::nullopt;
        }

        if (consecutive >= cfg_.consecutive_threshold && !in_recovery) {
            in_recovery = true;

            if (recovery_count >= cfg_.max_recoveries) {
                std::string reason = std::format(
                    "[{}] step={} util={:.1f}% — max_recoveries ({}) exhausted, "
                    "treating as FATAL",
                    device_name(unit), step_num, util, cfg_.max_recoveries);

                std::print("[watchdog] FATAL {}\n", reason);

                if (on_alert_) {
                    on_alert_(unit, step_num, util, reason);
                }

                consecutive = 0;
                return RecoveryAction{
                    .result       = RecoveryResult::FATAL,
                    .device       = unit,
                    .step         = step_num,
                    .util_at_fault= util,
                    .reason       = reason,
                };
            }

            // CRITICAL FIX (C2): trigger_recovery sleeps — do NOT hold mutex.
            auto action = trigger_recovery(unit, step_num, util);
            in_recovery = false;  // Allow next trigger after this one completes

            std::string reason = std::format(
                "[{}] step={} util={:.1f}% — WARNING consecutive={}/{} → recovery {}",
                device_name(unit), step_num, util,
                consecutive, cfg_.consecutive_threshold,
                action.has_value()
                    ? (action.value() == RecoveryResult::SUCCESS   ? "SUCCESS"
                       : action.value() == RecoveryResult::PARTIAL ? "PARTIAL"
                                                                   : "FATAL")
                    : "FAILED_TO_INVOKE");

            // Update stats
            if (unit == ComputeUnit::GPU_780M) {
                stats_.gpu_recovery_count++;
            } else {
                stats_.hailo_recovery_count++;
            }

            RecoveryResult r = action.value_or(RecoveryResult::FATAL);
            return RecoveryAction{
                .result        = r,
                .device        = unit,
                .step          = step_num,
                .util_at_fault = util,
                .reason        = reason,
            };
        }

        // Below threshold but not yet enough consecutive steps.
        return std::nullopt;
    }

    // ------------------------------------------------------------------
    //  CRITICAL: utilization fell below the critical threshold.
    //  Immediate recovery — no counting required.
    // ------------------------------------------------------------------
    {
        WatchdogState old_state = current_state;
        current_state = WatchdogState::CRITICAL;
        if (old_state != WatchdogState::CRITICAL) {
            log_state_change(unit, old_state, current_state, util, step_num);
        }

        // Thermal guard inside CRITICAL as well (C3 fix)
        if (temperature > cfg_.thermal_throttle_threshold_c) {
            std::string msg = std::format(
                "[{}] step={} util={:.1f}% — CRITICAL but THERMAL THROTTLING "
                "(temp={:.1f}C), skipping recovery",
                device_name(unit), step_num, util, temperature);
            std::print("[watchdog] {}\n", msg);
            if (on_alert_) {
                on_alert_(unit, step_num, util, msg);
            }
            return std::nullopt;
        }

        if (!in_recovery) {
            in_recovery = true;

            if (recovery_count >= cfg_.max_recoveries) {
                std::string reason = std::format(
                    "[{}] step={} util={:.1f}% — CRITICAL zone, max_recoveries "
                    "({}) exhausted, treating as FATAL",
                    device_name(unit), step_num, util, cfg_.max_recoveries);

                std::print("[watchdog] FATAL {}\n", reason);

                if (on_alert_) {
                    on_alert_(unit, step_num, util, reason);
                }

                return RecoveryAction{
                    .result        = RecoveryResult::FATAL,
                    .device        = unit,
                    .step          = step_num,
                    .util_at_fault = util,
                    .reason        = reason,
                };
            }

            // CRITICAL FIX (C2): trigger_recovery sleeps — do NOT hold mutex.
            auto action = trigger_recovery(unit, step_num, util);
            in_recovery = false;

            std::string reason = std::format(
                "[{}] step={} util={:.1f}% — CRITICAL immediate recovery {}",
                device_name(unit), step_num, util,
                action.has_value()
                    ? (action.value() == RecoveryResult::SUCCESS   ? "SUCCESS"
                       : action.value() == RecoveryResult::PARTIAL ? "PARTIAL"
                                                                   : "FATAL")
                    : "FAILED_TO_INVOKE");

            // Update stats
            if (unit == ComputeUnit::GPU_780M) {
                stats_.gpu_recovery_count++;
            } else {
                stats_.hailo_recovery_count++;
            }

            RecoveryResult r = action.value_or(RecoveryResult::FATAL);
            return RecoveryAction{
                .result        = r,
                .device        = unit,
                .step          = step_num,
                .util_at_fault = util,
                .reason        = reason,
            };
        }

        // Already in recovery — suppress duplicate triggers.
        return std::nullopt;
    }
}

// ============================================================================
//  Private: recovery with exponential backoff
// ============================================================================

std::expected<RecoveryResult, std::string>
UtilizationWatchdog::trigger_recovery(ComputeUnit unit,
                                      std::uint32_t step,
                                      float util)
{
    // Choose the correct per-device recovery counter.
    std::uint32_t& recovery_count = (unit == ComputeUnit::GPU_780M)
                                        ? gpu_recovery_count_
                                        : hailo_recovery_count_;

    recovery_count++;

    const float delay_ms = compute_backoff_ms(recovery_count);

    std::print("[watchdog] [{}] step={} util={:.1f}% — recovery #{}/{} with "
               "{:.0f}ms backoff\n",
               device_name(unit), step, util,
               recovery_count, cfg_.max_recoveries, delay_ms);

    if (delay_ms > 0.0f) {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(static_cast<std::int64_t>(delay_ms)));
    }

    if (!on_recovery_) {
        return std::unexpected<std::string>(
            "No recovery callback registered");
    }

    auto result = on_recovery_(unit, step, util);

    if (result) {
        std::print("[watchdog] [{}] step={} — recovery result: {}\n",
                   device_name(unit), step,
                   result.value() == RecoveryResult::SUCCESS   ? "SUCCESS"
                   : result.value() == RecoveryResult::PARTIAL ? "PARTIAL"
                                                               : "FATAL");
    } else {
        std::print("[watchdog] [{}] step={} — recovery FAILED: {}\n",
                   device_name(unit), step, result.error());
    }

    return result;
}

// ============================================================================
//  Private: helpers
// ============================================================================

float UtilizationWatchdog::compute_backoff_ms(
    std::uint32_t recovery_count) const noexcept
{
    if (recovery_count == 0) {
        return 0.0f;
    }

    // exponential: base * 2^(count-1)
    const float factor = std::exp2f(static_cast<float>(recovery_count - 1));
    float       delay  = cfg_.backoff_base_ms * factor;

    if (delay > cfg_.backoff_max_ms) {
        delay = cfg_.backoff_max_ms;
    }
    return delay;
}

// ---------------------------------------------------------------------------

void UtilizationWatchdog::log_state_change(ComputeUnit unit,
                                           WatchdogState old_state,
                                           WatchdogState new_state,
                                           float util,
                                           std::uint32_t step)
{
    if (old_state == new_state) {
        return;
    }

    // ISO-8601 timestamp with millisecond precision using system_clock.
    const auto now = std::chrono::system_clock::now();
    const auto time_t_now = std::chrono::system_clock::to_time_t(now);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now.time_since_epoch())
                        .count() % 1000;

    struct std::tm utc{};
    gmtime_r(&time_t_now, &utc);

    std::print(
        "[watchdog] {:04d}-{:02d}-{:02d}T{:02d}:{:02d}:{:02d}.{:03d}Z "
        "[{}] step={} state_change: {} -> {} util={:.1f}%\n",
        utc.tm_year + 1900, utc.tm_mon + 1, utc.tm_mday,
        utc.tm_hour, utc.tm_min, utc.tm_sec, ms,
        device_name(unit), step, state_name(old_state), state_name(new_state),
        util);

    if (on_alert_) {
        std::string msg = std::format(
            "[{}] step={} state_change: {} -> {} util={:.1f}%",
            device_name(unit), step,
            state_name(old_state), state_name(new_state), util);
        on_alert_(unit, step, util, std::move(msg));
    }
}

// ---------------------------------------------------------------------------

constexpr const char* UtilizationWatchdog::state_name(WatchdogState s) noexcept
{
    switch (s) {
    case WatchdogState::NORMAL:   return "NORMAL";
    case WatchdogState::WARNING:  return "WARNING";
    case WatchdogState::CRITICAL: return "CRITICAL";
    }
    return "UNKNOWN";
}

// ---------------------------------------------------------------------------

constexpr const char* UtilizationWatchdog::device_name(ComputeUnit u) noexcept
{
    switch (u) {
    case ComputeUnit::GPU_780M:  return "GPU_780M";
    case ComputeUnit::HAILO_8L:  return "HAILO_8L";
    }
    return "UNKNOWN";
}

} // namespace hq
