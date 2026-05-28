/// @file utilization_watchdog.cpp
/// @copyright Copyright © 2026, all rights reserved | David Hargreaves aka Roylepython trading as The Medusa Initiative & Lamia Fabrica
/// Full implementation of the UtilizationWatchdog 3-state machine.
/// C01 fix: state-machine fields are std::atomic — no data race between
/// evaluate_device() and get_*_state().  Mutex_ guards only the five
/// low-contention diagnostics fields.

#include "hq/utilization_watchdog.hpp"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <thread>

namespace hq {

UtilizationWatchdog::UtilizationWatchdog(WatchdogConfig cfg,
                                         RecoveryCallback on_recovery,
                                         AlertCallback on_alert)
    : cfg_{std::move(cfg)}
    , on_recovery_{std::move(on_recovery)}
    , on_alert_{std::move(on_alert)}
{
    if (cfg_.gpu_critical_threshold >= cfg_.gpu_low_threshold) {
        std::fputs(std::format("[watchdog] WARN: gpu_critical_threshold ({:.1f}) should be "
                   "below gpu_low_threshold ({:.1f})\n",
                   cfg_.gpu_critical_threshold, cfg_.gpu_low_threshold).c_str(), stderr);
    }
    if (cfg_.hailo_critical_threshold >= cfg_.hailo_low_threshold) {
        std::fputs(std::format("[watchdog] WARN: hailo_critical_threshold ({:.1f}) should be "
                   "below hailo_low_threshold ({:.1f})\n",
                   cfg_.hailo_critical_threshold, cfg_.hailo_low_threshold).c_str(), stderr);
    }
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

std::optional<RecoveryAction>
UtilizationWatchdog::step(std::uint32_t step_num,
                          const UtilizationSnapshot& gpu_snap,
                          const UtilizationSnapshot& hailo_snap)
{
    auto gpu_action = evaluate_device(
        ComputeUnit::GPU_780M,
        gpu_snap.utilization,
        gpu_snap.temperature,
        cfg_.gpu_low_threshold,
        cfg_.gpu_critical_threshold,
        gpu_consecutive_,
        gpu_in_recovery_,
        gpu_recovery_count_,
        gpu_state_,
        step_num);

    auto hailo_action = evaluate_device(
        ComputeUnit::HAILO_8L,
        hailo_snap.utilization,
        hailo_snap.temperature,
        cfg_.hailo_low_threshold,
        cfg_.hailo_critical_threshold,
        hailo_consecutive_,
        hailo_in_recovery_,
        hailo_recovery_count_,
        hailo_state_,
        step_num);

    {
        const std::scoped_lock lock{mutex_};
        stats_.total_steps++;
        stats_.last_gpu_util         = gpu_snap.utilization;
        stats_.last_hailo_util       = hailo_snap.utilization;
        stats_.gpu_consecutive_low   = gpu_consecutive_.load(std::memory_order_relaxed);
        stats_.hailo_consecutive_low = hailo_consecutive_.load(std::memory_order_relaxed);
    }

    if (gpu_action) return gpu_action;
    if (hailo_action) return hailo_action;
    return std::nullopt;
}

WatchdogState UtilizationWatchdog::get_gpu_state() const noexcept
{
    return gpu_state_.load(std::memory_order_acquire);
}

WatchdogState UtilizationWatchdog::get_hailo_state() const noexcept
{
    return hailo_state_.load(std::memory_order_acquire);
}

WatchdogStatistics UtilizationWatchdog::get_stats() const noexcept
{
    WatchdogStatistics s;
    {
        const std::scoped_lock lock{mutex_};
        s.total_steps           = stats_.total_steps;
        s.last_gpu_util         = stats_.last_gpu_util;
        s.last_hailo_util       = stats_.last_hailo_util;
        s.gpu_consecutive_low   = stats_.gpu_consecutive_low;
        s.hailo_consecutive_low = stats_.hailo_consecutive_low;
    }
    s.gpu_state            = gpu_state_.load(std::memory_order_acquire);
    s.hailo_state          = hailo_state_.load(std::memory_order_acquire);
    s.gpu_recovery_count   = gpu_recovery_count_.load(std::memory_order_relaxed);
    s.hailo_recovery_count = hailo_recovery_count_.load(std::memory_order_relaxed);
    return s;
}

void UtilizationWatchdog::reset_counters() noexcept
{
    gpu_consecutive_.store(0, std::memory_order_relaxed);
    hailo_consecutive_.store(0, std::memory_order_relaxed);
}

void UtilizationWatchdog::reset_all() noexcept
{
    gpu_consecutive_.store(0, std::memory_order_relaxed);
    hailo_consecutive_.store(0, std::memory_order_relaxed);
    gpu_in_recovery_.store(false, std::memory_order_relaxed);
    hailo_in_recovery_.store(false, std::memory_order_relaxed);
    gpu_recovery_count_.store(0, std::memory_order_relaxed);
    hailo_recovery_count_.store(0, std::memory_order_relaxed);
    gpu_state_.store(WatchdogState::NORMAL, std::memory_order_release);
    hailo_state_.store(WatchdogState::NORMAL, std::memory_order_release);

    const std::scoped_lock lock{mutex_};
    stats_ = {};
}

std::optional<RecoveryAction>
UtilizationWatchdog::evaluate_device(ComputeUnit unit,
                                     float util,
                                     float temperature,
                                     float threshold,
                                     float critical_threshold,
                                     std::atomic<std::uint32_t>& consecutive,
                                     std::atomic<bool>&          in_recovery,
                                     std::atomic<std::uint32_t>& recovery_count,
                                     std::atomic<WatchdogState>& current_state,
                                     std::uint32_t step_num)
{
    if (util >= threshold) {
        if (consecutive.load(std::memory_order_relaxed) != 0) {
            log_state_change(unit, current_state.load(std::memory_order_acquire),
                             WatchdogState::NORMAL, util, step_num);
        }
        if (current_state.load(std::memory_order_acquire) != WatchdogState::NORMAL) {
            const WatchdogState old = current_state.load(std::memory_order_acquire);
            current_state.store(WatchdogState::NORMAL, std::memory_order_release);
            log_state_change(unit, old, WatchdogState::NORMAL, util, step_num);
        }
        consecutive.store(0, std::memory_order_relaxed);
        in_recovery.store(false, std::memory_order_relaxed);
        return std::nullopt;
    }

    const std::uint32_t cons =
        consecutive.fetch_add(1, std::memory_order_relaxed) + 1;

    if (util >= critical_threshold) {
        const WatchdogState old_state = current_state.load(std::memory_order_acquire);
        if (old_state != WatchdogState::WARNING) {
            current_state.store(WatchdogState::WARNING, std::memory_order_release);
            log_state_change(unit, old_state, WatchdogState::WARNING, util, step_num);
        }

        if (thermal_skip_recovery_(unit, temperature, util, step_num, false)) {
            return std::nullopt;
        }

        if (cons >= cfg_.consecutive_threshold &&
            !in_recovery.load(std::memory_order_acquire)) {
            in_recovery.store(true, std::memory_order_release);
            if (recovery_count.load(std::memory_order_relaxed) >= cfg_.max_recoveries) {
                std::string reason = std::format(
                    "[{}] step={} util={:.1f}% — max_recoveries ({}) exhausted, "
                    "treating as FATAL",
                    device_name(unit), step_num, util, cfg_.max_recoveries);
                std::print("[watchdog] FATAL {}\n", reason);
                if (on_alert_) on_alert_(unit, step_num, util, reason);
                consecutive.store(0, std::memory_order_relaxed);
                return RecoveryAction{.result=RecoveryResult::FATAL, .device=unit,
                                      .step=step_num, .util_at_fault=util, .reason=reason};
            }
            auto action = trigger_recovery(unit, step_num, util);
            in_recovery.store(false, std::memory_order_release);
            std::string reason = std::format(
                "[{}] step={} util={:.1f}% — WARNING consecutive={}/{} → recovery {}",
                device_name(unit), step_num, util,
                cons, cfg_.consecutive_threshold,
                action.has_value()
                    ? (action.value() == RecoveryResult::SUCCESS   ? "SUCCESS"
                       : action.value() == RecoveryResult::PARTIAL ? "PARTIAL"
                                                                   : "FATAL")
                    : "FAILED_TO_INVOKE");
            RecoveryResult r = action.value_or(RecoveryResult::FATAL);
            return RecoveryAction{.result=r, .device=unit, .step=step_num,
                                  .util_at_fault=util, .reason=reason};
        }
        return std::nullopt;
    }

    {
        const WatchdogState old_state = current_state.load(std::memory_order_acquire);
        current_state.store(WatchdogState::CRITICAL, std::memory_order_release);
        if (old_state != WatchdogState::CRITICAL) {
            log_state_change(unit, old_state, WatchdogState::CRITICAL, util, step_num);
        }

        if (thermal_skip_recovery_(unit, temperature, util, step_num, true)) {
            return std::nullopt;
        }

        if (!in_recovery.load(std::memory_order_acquire)) {
            in_recovery.store(true, std::memory_order_release);
            if (recovery_count.load(std::memory_order_relaxed) >= cfg_.max_recoveries) {
                std::string reason = std::format(
                    "[{}] step={} util={:.1f}% — CRITICAL zone, max_recoveries "
                    "({}) exhausted, treating as FATAL",
                    device_name(unit), step_num, util, cfg_.max_recoveries);
                std::print("[watchdog] FATAL {}\n", reason);
                if (on_alert_) on_alert_(unit, step_num, util, reason);
                return RecoveryAction{.result=RecoveryResult::FATAL, .device=unit,
                                      .step=step_num, .util_at_fault=util, .reason=reason};
            }
            auto action = trigger_recovery(unit, step_num, util);
            in_recovery.store(false, std::memory_order_release);
            std::string reason = std::format(
                "[{}] step={} util={:.1f}% — CRITICAL immediate recovery {}",
                device_name(unit), step_num, util,
                action.has_value()
                    ? (action.value() == RecoveryResult::SUCCESS   ? "SUCCESS"
                       : action.value() == RecoveryResult::PARTIAL ? "PARTIAL"
                                                                   : "FATAL")
                    : "FAILED_TO_INVOKE");
            RecoveryResult r = action.value_or(RecoveryResult::FATAL);
            return RecoveryAction{.result=r, .device=unit, .step=step_num,
                                  .util_at_fault=util, .reason=reason};
        }
        return std::nullopt;
    }
}

std::expected<RecoveryResult, std::string>
UtilizationWatchdog::trigger_recovery(ComputeUnit unit,
                                      std::uint32_t step,
                                      float util)
{
    std::atomic<std::uint32_t>& recovery_count =
        (unit == ComputeUnit::GPU_780M) ? gpu_recovery_count_ : hailo_recovery_count_;
    const std::uint32_t new_count =
        recovery_count.fetch_add(1, std::memory_order_relaxed) + 1;
    const float delay_ms = compute_backoff_ms(new_count);
    std::print("[watchdog] [{}] step={} util={:.1f}% — recovery #{}/{} with "
               "{:.0f}ms backoff\n",
               device_name(unit), step, util,
               new_count, cfg_.max_recoveries, delay_ms);
    if (delay_ms > 0.0f) {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(static_cast<std::int64_t>(delay_ms)));
    }
    if (!on_recovery_) {
        return std::unexpected<std::string>("No recovery callback registered");
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

float UtilizationWatchdog::compute_backoff_ms(
    std::uint32_t recovery_count) const noexcept
{
    if (recovery_count == 0) return 0.0f;
    const float factor = std::exp2f(static_cast<float>(recovery_count - 1));
    float delay = cfg_.backoff_base_ms * factor;
    if (delay > cfg_.backoff_max_ms) delay = cfg_.backoff_max_ms;
    return delay;
}

bool UtilizationWatchdog::thermal_skip_recovery_(ComputeUnit unit,
                                                 float temperature,
                                                 float util,
                                                 std::uint32_t step,
                                                 bool is_critical) {
    if (temperature > cfg_.thermal_throttle_threshold_c) {
        std::string msg = std::format(
            "[{}] step={} util={:.1f}% — {} THERMAL THROTTLING "
            "(temp={:.1f}C > threshold={:.1f}C), skipping recovery",
            device_name(unit), step, util,
            is_critical ? "CRITICAL but" : "",
            temperature, cfg_.thermal_throttle_threshold_c);
        std::print("[watchdog] {}\n", msg);
        if (on_alert_) on_alert_(unit, step, util, msg);
        return true;
    }
    return false;
}

void UtilizationWatchdog::log_state_change(ComputeUnit unit,
                                           WatchdogState old_state,
                                           WatchdogState new_state,
                                           float util,
                                           std::uint32_t step)
{
    if (old_state == new_state) return;
    const auto now = std::chrono::system_clock::now();
    const auto time_t_now = std::chrono::system_clock::to_time_t(now);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now.time_since_epoch()).count() % 1000;
    struct std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &time_t_now);
#else
    gmtime_r(&time_t_now, &utc);
#endif
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

constexpr const char* UtilizationWatchdog::state_name(WatchdogState s) noexcept
{
    switch (s) {
    case WatchdogState::NORMAL:   return "NORMAL";
    case WatchdogState::WARNING:  return "WARNING";
    case WatchdogState::CRITICAL: return "CRITICAL";
    }
    return "UNKNOWN";
}

constexpr const char* UtilizationWatchdog::device_name(ComputeUnit u) noexcept
{
    switch (u) {
    case ComputeUnit::GPU_780M:  return "GPU_780M";
    case ComputeUnit::HAILO_8L:  return "HAILO_8L";
    }
    return "UNKNOWN";
}

} // namespace hq
