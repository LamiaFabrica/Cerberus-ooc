#pragma once
/// @file gpu_monitor.hpp
/// @copyright Copyright (c) 2026 D Hargreaves (AKA Roylepython). LamiaFabrica. All rights reserved.
/// GPU telemetry monitor with multi-backend support.
///
/// Backends (selected at compile time, probed at runtime):
///   - NVIDIA NVML (UM790_HAS_CUDA): RTX GPUs, uses nvmlDeviceGet* APIs
///   - AMD ROCm SMI (UM790_HAS_HIP): Radeon GPUs, uses rsmi_dev_* APIs
///   - Fallback: returns zeros when no GPU telemetry library is available
///
/// @author LamiaFabrica Team
/// @version 2.1.0

#include "hq/cxx26_features.hpp"

#include <cstdint>
#if UM790_HAS_STD_EXPECTED
#  include <expected>
#else
#  error "gpu_monitor.hpp requires std::expected (<expected>) — GCC >= 14 or Clang >= 18 with C++26 enabled"
#endif
#if UM790_HAS_STD_FORMAT
#  include <format>
#endif
#include <string>
#include <utility>

namespace hq {

// ---------------------------------------------------------------------------
/// @brief Default thermal throttling threshold.
// ---------------------------------------------------------------------------
inline constexpr float kDefaultThrottleThresholdC = 85.0f;

// ---------------------------------------------------------------------------
/// @brief Error conditions for GPU telemetry queries.
// ---------------------------------------------------------------------------
enum class GPUError : std::uint8_t {
    Ok = 0,
    NotInitialized,
    DeviceNotFound,
    UtilizationQueryFailed,
    TemperatureQueryFailed,
    PowerQueryFailed,
    MemoryQueryFailed,
    ThrottlingDetected,
};

// ---------------------------------------------------------------------------
/// @brief Detailed error information for a failed GPU query.
// ---------------------------------------------------------------------------
struct GPUErrorInfo {
    GPUError code{GPUError::Ok};
    std::string message;
    int raw_status{0}; ///< raw rsmi_status_t if applicable
};

// ---------------------------------------------------------------------------
/// @brief Complete GPU telemetry snapshot from a single query_all() call.
// ---------------------------------------------------------------------------
struct GPUTelemetry {
    float utilization_percent{0.0f};      ///< [0, 100] from rsmi_dev_gpu_busy_percent_get
    float temperature_celsius{0.0f};      ///< from rsmi_dev_temp_metric_get (EDGE)
    float junction_temperature_c{0.0f};   ///< from rsmi_dev_temp_metric_get (JUNCTION)
    float power_watts{0.0f};              ///< from rsmi_dev_power_ave_get
    float memory_used_mb{0.0f};           ///< from rsmi_dev_memory_usage_get (VRAM)
    float memory_total_mb{0.0f};          ///< total VRAM
    bool  is_throttling{false};           ///< true if temp > 85C or throttling status active
    std::uint64_t timestamp_ms{0};        ///< steady_clock time since epoch
};

// ---------------------------------------------------------------------------
/// @class GPUMonitor
/// @brief Standalone GPU telemetry monitor wrapping ROCm SMI.
///
/// Provides RAII-managed access to AMD GPU hardware telemetry via ROCm SMI.
/// All queries return std::expected to cleanly communicate failures without
/// throwing exceptions.
///
/// Thread-safety: Not thread-safe. One GPUMonitor per thread.
///
/// Usage:
/// @code
///   GPUMonitor mon{0};          // monitor GPU 0
///   auto init = mon.initialize();
///   if (!init) { /* handle error */ }
///
///   auto telem = mon.query_all();
///   if (telem) {
///       std::print("GPU {:.0f}%  {:.0f}C  {:.1f}W\n",
///                  telem->utilization_percent,
///                  telem->temperature_celsius,
///                  telem->power_watts);
///   }
/// @endcode
// ---------------------------------------------------------------------------
class GPUMonitor {
public:
    /// @brief Construct a GPUMonitor for a specific GPU device.
    /// @param device_index ROCm device index (default 0).
    explicit GPUMonitor(std::uint32_t device_index = 0);
    ~GPUMonitor() noexcept;

    // Non-copyable, movable
    GPUMonitor(const GPUMonitor&) = delete;
    GPUMonitor& operator=(const GPUMonitor&) = delete;
    GPUMonitor(GPUMonitor&&) noexcept;
    GPUMonitor& operator=(GPUMonitor&&) noexcept;

    /// Initialize GPU telemetry backend (NVML prioritized, then ROCm SMI, then none).
    /// Must be called before any queries.
    /// @return void on success, GPUErrorInfo on failure.
    [[nodiscard]] std::expected<void, GPUErrorInfo> initialize();

    /// Query all GPU telemetry in one call.
    /// @return GPUTelemetry snapshot on success, GPUErrorInfo on failure.
    [[nodiscard]] std::expected<GPUTelemetry, GPUErrorInfo> query_all();

    /// @brief Query GPU utilization percentage.
    /// @return Percentage [0, 100] on success.
    [[nodiscard]] std::expected<float, GPUErrorInfo> query_utilization();

    /// @brief Query GPU edge temperature.
    /// @return Temperature in Celsius on success.
    [[nodiscard]] std::expected<float, GPUErrorInfo> query_temperature_edge();

    /// @brief Query GPU junction temperature.
    /// @return Temperature in Celsius on success.
    [[nodiscard]] std::expected<float, GPUErrorInfo> query_temperature_junction();

    /// @brief Query GPU average power draw.
    /// @return Power in watts on success.
    [[nodiscard]] std::expected<float, GPUErrorInfo> query_power();

    /// @brief Query GPU VRAM usage.
    /// @return Pair of {used_MiB, total_MiB} on success.
    [[nodiscard]] std::expected<std::pair<float, float>, GPUErrorInfo> query_memory();

    /// Check if the monitor has been successfully initialized.
    [[nodiscard]] bool is_initialized() const noexcept { return initialized_; }

    /// Get the monitored device index.
    [[nodiscard]] std::uint32_t device_index() const noexcept { return device_index_; }

    /// Check if GPU is currently thermal throttling.
    /// @param threshold_c Temperature threshold in Celsius (default 85.0f).
    /// @return true if throttling detected, false otherwise.
    [[nodiscard]] std::expected<bool, GPUErrorInfo> is_throttling(float threshold_c = kDefaultThrottleThresholdC);

    /// @brief Which backend is active (if any) after initialize().
    enum class Backend : std::uint8_t { None = 0, NVML, ROCM_SMI };

    [[nodiscard]] Backend backend() const noexcept { return backend_; }

private:
    std::uint32_t device_index_{0};
    bool initialized_{false};
    Backend backend_{Backend::None};

#if defined(UM790_HAS_CUDA) && __has_include(<nvml.h>)
    void* nvml_device_{nullptr};
#endif
};

// ---------------------------------------------------------------------------
/// @brief Convenience free function for error formatting.
// ---------------------------------------------------------------------------
[[nodiscard]] inline std::string to_string(GPUError e) {
    switch (e) {
        case GPUError::Ok:                        return "Ok";
        case GPUError::NotInitialized:            return "NotInitialized";
        case GPUError::DeviceNotFound:            return "DeviceNotFound";
        case GPUError::UtilizationQueryFailed:    return "UtilizationQueryFailed";
        case GPUError::TemperatureQueryFailed:    return "TemperatureQueryFailed";
        case GPUError::PowerQueryFailed:          return "PowerQueryFailed";
        case GPUError::MemoryQueryFailed:         return "MemoryQueryFailed";
        case GPUError::ThrottlingDetected:        return "ThrottlingDetected";
    }
    return "Unknown";
}

[[nodiscard]] inline std::string to_string(GPUMonitor::Backend b) {
    switch (b) {
        case GPUMonitor::Backend::None:     return "None";
        case GPUMonitor::Backend::NVML:     return "NVML";
        case GPUMonitor::Backend::ROCM_SMI: return "ROCM_SMI";
    }
    return "Unknown";
}

} // namespace hq

// std::formatter specialization for GPUError
#if UM790_HAS_STD_FORMAT
template<>
struct std::formatter<hq::GPUError> : std::formatter<std::string_view> {
    auto format(hq::GPUError e, std::format_context& ctx) const {
        return std::formatter<std::string_view>::format(hq::to_string(e), ctx);
    }
};

template<>
struct std::formatter<hq::GPUMonitor::Backend> : std::formatter<std::string_view> {
    auto format(hq::GPUMonitor::Backend b, std::format_context& ctx) const {
        return std::formatter<std::string_view>::format(hq::to_string(b), ctx);
    }
};
#endif
