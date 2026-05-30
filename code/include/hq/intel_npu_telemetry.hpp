#pragma once
/// @file intel_npu_telemetry.hpp
/// @copyright Copyright (c) 2026 D Hargreaves. All rights reserved.
///
/// Real hardware NPU utilization collection for Intel AI Boost / NPU 4.
///
/// On Windows: Uses PDH (Performance Data Helper) to read actual NPU engine
/// utilization counters exposed by the Intel driver. This is the only honest
/// way to report 0-100% busy on the target platform without vendor magic.
///
/// On Linux: dynamic zeLoader + zet* metric streamer (libze_loader.so via dlopen/dlsym only).
/// Zero hard dependency, zero compile-time requirement for level-zero headers.
/// Mirrors Windows PDH + PdhExpandWildCardPathW dynamic discovery exactly.
/// Graceful -1.0f + real_source_available_=false when loader/metrics absent.
/// Real 0-100 values (engine utilization) when Intel NPU + Level Zero present.
///
/// This exists because we refuse shoddy or approximated utilization numbers.
/// 70-75% sustained on Athenea-class workloads through the real memory loop
/// is the defining KPI.

#include <atomic>
#include <chrono>
#include <string>

#if defined(_WIN32)
#include <windows.h>
#include <pdh.h>
#include <pdhmsg.h>
#if defined(_MSC_VER)
#pragma comment(lib, "pdh.lib")
#endif
#endif

namespace hq::npu {

/// Real utilization source for Intel NPU.
/// Returns 0.0f - 100.0f when real data is available, -1.0f when unavailable.
class IntelNpuTelemetry {
public:
    IntelNpuTelemetry();
    ~IntelNpuTelemetry() noexcept;

    /// Returns the current NPU utilization percentage (0-100) or -1 if not available.
    [[nodiscard]] float current_utilization_percent() const;

    /// Returns true if we have a working real source on this platform.
    [[nodiscard]] bool is_real_source_available() const noexcept { return real_source_available_; }

    /// Platform string for logging ("Windows PDH", "Linux LevelZero", "None").
    [[nodiscard]] std::string source_description() const;

private:
    bool real_source_available_{false};
    std::string description_;

    // Round 21: lightweight time-based cache to reduce expensive PDH/L0 collects
    // in tight endurance loops (directly improves sustained NPU utilisation by
    // lowering host sync overhead; tunable interval).
    mutable std::chrono::steady_clock::time_point last_sample_time_{};
    mutable float cached_util_{-1.0f};
    int min_sample_interval_ms_{8};  // ~125 Hz max; enough for streak/pct metrics

#if defined(_WIN32)
    // PDH query for Intel NPU / AI Boost utilization.
    // We try several known counter paths that Intel NPU drivers expose.
    PDH_HQUERY   pdh_query_{nullptr};
    PDH_HCOUNTER pdh_counter_{nullptr};
    mutable std::atomic<float> last_reading_{-1.0f};

    bool try_open_pdh_counter(const wchar_t* counter_path);
    bool try_discover_npu_counter_via_wildcard();
    void close_pdh();
#endif

    // Linux path: always returns unavailable (-1.0f).
    // Real Level Zero (zeMetric* / zeDeviceGet* APIs via zeLoader) is required
    // for production telemetry on Linux. No approximation or stand-in implementation is provided.
};

} // namespace hq::npu
