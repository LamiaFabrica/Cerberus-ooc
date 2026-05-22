/// @file gpu_monitor.cpp
/// @copyright Copyright (c) 2026 D Hargreaves (AKA Roylepython). LamiaFabrica. All rights reserved.
/// GPUMonitor implementation: ROCm SMI telemetry queries with fallback.
///
/// @version 2.0.0

#include "hq/gpu_monitor.hpp"

#include <chrono>
#if UM790_HAS_STD_PRINT
#  include <print>
#endif
#include <utility>

// ROCm SMI (for GPU telemetry queries)
#if defined(UM790_HAS_HIP) && __has_include(<rocm_smi/rocm_smi.h>)
#include <rocm_smi/rocm_smi.h>
#define GPUMONITOR_HAS_ROCM_SMI 1
#else
#define GPUMONITOR_HAS_ROCM_SMI 0
#endif

namespace hq {

// ---------------------------------------------------------------------------
// Construction / Destruction / Move semantics
// ---------------------------------------------------------------------------

GPUMonitor::GPUMonitor(std::uint32_t device_index)
    : device_index_{device_index}
    , initialized_{false}
    , rsmi_initialized_{false}
{
}

GPUMonitor::~GPUMonitor() noexcept {
    if (rsmi_initialized_) {
#if GPUMONITOR_HAS_ROCM_SMI
        rsmi_shut_down();
#endif
        rsmi_initialized_ = false;
    }
    initialized_ = false;
}

GPUMonitor::GPUMonitor(GPUMonitor&& other) noexcept
    : device_index_{other.device_index_}
    , initialized_{other.initialized_}
    , rsmi_initialized_{other.rsmi_initialized_}
{
    other.device_index_    = 0;
    other.initialized_     = false;
    other.rsmi_initialized_ = false;
}

GPUMonitor& GPUMonitor::operator=(GPUMonitor&& other) noexcept {
    if (this != &other) {
        // Clean up our own ROCm SMI state
        if (rsmi_initialized_) {
#if GPUMONITOR_HAS_ROCM_SMI
            rsmi_shut_down();
#endif
        }

        device_index_      = other.device_index_;
        initialized_       = other.initialized_;
        rsmi_initialized_  = other.rsmi_initialized_;

        other.device_index_    = 0;
        other.initialized_     = false;
        other.rsmi_initialized_ = false;
    }
    return *this;
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

std::expected<void, GPUErrorInfo> GPUMonitor::initialize() {
    if (initialized_) {
        return {}; // already initialized
    }

#if GPUMONITOR_HAS_ROCM_SMI
    rsmi_status_t st = rsmi_init(0);
    if (st != RSMI_STATUS_SUCCESS) {
        std::print("[GPUMonitor] rsmi_init failed (status={})\n", static_cast<int>(st));
        return std::unexpected{GPUErrorInfo{
            .code       = GPUError::NotInitialized,
            .message    = std::format("rsmi_init failed with status {}", static_cast<int>(st)),
            .raw_status = static_cast<int>(st),
        }};
    }
    rsmi_initialized_ = true;

    // Validate device index exists
    uint32_t device_count = 0;
    st = rsmi_num_monitor_devices(&device_count);
    if (st != RSMI_STATUS_SUCCESS) {
        std::print("[GPUMonitor] rsmi_num_monitor_devices failed (status={})\n",
                   static_cast<int>(st));
        return std::unexpected{GPUErrorInfo{
            .code       = GPUError::DeviceNotFound,
            .message    = std::format("Failed to enumerate ROCm devices (status {})",
                                       static_cast<int>(st)),
            .raw_status = static_cast<int>(st),
        }};
    }

    if (device_index_ >= device_count) {
        std::print("[GPUMonitor] Device index {} out of range ({} devices available)\n",
                   device_index_, device_count);
        return std::unexpected{GPUErrorInfo{
            .code       = GPUError::DeviceNotFound,
            .message    = std::format("Device index {} >= device count {}",
                                       device_index_, device_count),
            .raw_status = static_cast<int>(st),
        }};
    }
#else
    std::print("[GPUMonitor] ROCm SMI not available -- GPU telemetry is unavailable\n");
    return std::unexpected{GPUErrorInfo{
        .code       = GPUError::DeviceNotFound,
        .message    = "ROCm SMI library not available; cannot query GPU. "
                       "Install rocm-smi-lib or compile with ROCm support.",
        .raw_status = -1,
    }};
#endif

    initialized_ = true;
    return {};
}

// ---------------------------------------------------------------------------
// Individual queries
// ---------------------------------------------------------------------------

std::expected<float, GPUErrorInfo> GPUMonitor::query_utilization() {
    if (!initialized_) {
        return std::unexpected{GPUErrorInfo{
            .code       = GPUError::NotInitialized,
            .message    = "GPUMonitor not initialized; call initialize() first",
            .raw_status = -1,
        }};
    }

#if GPUMONITOR_HAS_ROCM_SMI
    std::uint32_t pct = 0;
    rsmi_status_t st = rsmi_dev_gpu_busy_percent_get(device_index_, &pct);
    if (st != RSMI_STATUS_SUCCESS) {
        return std::unexpected{GPUErrorInfo{
            .code       = GPUError::UtilizationQueryFailed,
            .message    = std::format("rsmi_dev_gpu_busy_percent_get failed (status {})",
                                       static_cast<int>(st)),
            .raw_status = static_cast<int>(st),
        }};
    }
    return static_cast<float>(pct);
#else
    // CI/TEST PATH: Returns 0.0f when ROCm SMI library is not installed.
    // Install rocm-smi-lib for real GPU telemetry.
    return 0.0f;
#endif
}

std::expected<float, GPUErrorInfo> GPUMonitor::query_temperature_edge() {
    if (!initialized_) {
        return std::unexpected{GPUErrorInfo{
            .code       = GPUError::NotInitialized,
            .message    = "GPUMonitor not initialized; call initialize() first",
            .raw_status = -1,
        }};
    }

#if GPUMONITOR_HAS_ROCM_SMI
    std::int64_t temp_mc = 0; // millidegrees Celsius
    rsmi_status_t st = rsmi_dev_temp_metric_get(
        device_index_, RSMI_TEMP_TYPE_EDGE, RSMI_TEMP_CURRENT, &temp_mc);
    if (st != RSMI_STATUS_SUCCESS) {
        return std::unexpected{GPUErrorInfo{
            .code       = GPUError::TemperatureQueryFailed,
            .message    = std::format("rsmi_dev_temp_metric_get(EDGE) failed (status {})",
                                       static_cast<int>(st)),
            .raw_status = static_cast<int>(st),
        }};
    }
    return static_cast<float>(temp_mc) / 1000.0f;
#else
    // CI/TEST PATH: Returns 0.0f when ROCm SMI library is not installed.
    // Install rocm-smi-lib for real GPU telemetry.
    return 0.0f;
#endif
}

std::expected<float, GPUErrorInfo> GPUMonitor::query_temperature_junction() {
    if (!initialized_) {
        return std::unexpected{GPUErrorInfo{
            .code       = GPUError::NotInitialized,
            .message    = "GPUMonitor not initialized; call initialize() first",
            .raw_status = -1,
        }};
    }

#if GPUMONITOR_HAS_ROCM_SMI
    std::int64_t temp_mc = 0; // millidegrees Celsius
    rsmi_status_t st = rsmi_dev_temp_metric_get(
        device_index_, RSMI_TEMP_TYPE_JUNCTION, RSMI_TEMP_CURRENT, &temp_mc);
    if (st != RSMI_STATUS_SUCCESS) {
        return std::unexpected{GPUErrorInfo{
            .code       = GPUError::TemperatureQueryFailed,
            .message    = std::format("rsmi_dev_temp_metric_get(JUNCTION) failed (status {})",
                                       static_cast<int>(st)),
            .raw_status = static_cast<int>(st),
        }};
    }
    return static_cast<float>(temp_mc) / 1000.0f;
#else
    // CI/TEST PATH: Returns 0.0f when ROCm SMI library is not installed.
    // Install rocm-smi-lib for real GPU telemetry.
    return 0.0f;
#endif
}

std::expected<float, GPUErrorInfo> GPUMonitor::query_power() {
    if (!initialized_) {
        return std::unexpected{GPUErrorInfo{
            .code       = GPUError::NotInitialized,
            .message    = "GPUMonitor not initialized; call initialize() first",
            .raw_status = -1,
        }};
    }

#if GPUMONITOR_HAS_ROCM_SMI
    std::uint64_t power_uw = 0; // microwatts
    rsmi_status_t st = rsmi_dev_power_ave_get(device_index_, 0, &power_uw);
    if (st != RSMI_STATUS_SUCCESS) {
        return std::unexpected{GPUErrorInfo{
            .code       = GPUError::PowerQueryFailed,
            .message    = std::format("rsmi_dev_power_ave_get failed (status {})",
                                       static_cast<int>(st)),
            .raw_status = static_cast<int>(st),
        }};
    }
    return static_cast<float>(power_uw) / 1'000'000.0f; // uW -> W
#else
    // CI/TEST PATH: Returns 0.0f when ROCm SMI library is not installed.
    // Install rocm-smi-lib for real GPU telemetry.
    return 0.0f;
#endif
}

std::expected<std::pair<float, float>, GPUErrorInfo> GPUMonitor::query_memory() {
    if (!initialized_) {
        return std::unexpected{GPUErrorInfo{
            .code       = GPUError::NotInitialized,
            .message    = "GPUMonitor not initialized; call initialize() first",
            .raw_status = -1,
        }};
    }

#if GPUMONITOR_HAS_ROCM_SMI
    std::uint64_t mem_used_bytes = 0;
    std::uint64_t mem_total_bytes = 0;

    rsmi_status_t st_used = rsmi_dev_memory_usage_get(
        device_index_, RSMI_MEM_TYPE_VRAM, &mem_used_bytes);
    if (st_used != RSMI_STATUS_SUCCESS) {
        return std::unexpected{GPUErrorInfo{
            .code       = GPUError::MemoryQueryFailed,
            .message    = std::format("rsmi_dev_memory_usage_get failed (status {})",
                                       static_cast<int>(st_used)),
            .raw_status = static_cast<int>(st_used),
        }};
    }

    rsmi_status_t st_total = rsmi_dev_memory_total_get(
        device_index_, RSMI_MEM_TYPE_VRAM, &mem_total_bytes);
    if (st_total != RSMI_STATUS_SUCCESS) {
        return std::unexpected{GPUErrorInfo{
            .code       = GPUError::MemoryQueryFailed,
            .message    = std::format("rsmi_dev_memory_total_get failed (status {})",
                                       static_cast<int>(st_total)),
            .raw_status = static_cast<int>(st_total),
        }};
    }

    const float used_mib  = static_cast<float>(mem_used_bytes)  / (1024.0f * 1024.0f);
    const float total_mib = static_cast<float>(mem_total_bytes) / (1024.0f * 1024.0f);
    return std::pair{used_mib, total_mib};
#else
    // CI/TEST PATH: Returns {0.0f, 0.0f} when ROCm SMI library is not
    // installed. Install rocm-smi-lib for real GPU telemetry.
    return std::pair{0.0f, 0.0f};
#endif
}

// ---------------------------------------------------------------------------
// Combined query
// ---------------------------------------------------------------------------

std::expected<GPUTelemetry, GPUErrorInfo> GPUMonitor::query_all() {
    if (!initialized_) {
        return std::unexpected{GPUErrorInfo{
            .code       = GPUError::NotInitialized,
            .message    = "GPUMonitor not initialized; call initialize() first",
            .raw_status = -1,
        }};
    }

    GPUTelemetry telem{};

    // Timestamp
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    telem.timestamp_ms = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(now).count());

    // Utilization
    auto util = query_utilization();
    if (!util) {
        // Partial failure: log and continue with 0
        std::print("[GPUMonitor] Utilization query failed: {}\n", util.error().message);
        telem.utilization_percent = 0.0f;
    } else {
        telem.utilization_percent = *util;
    }

    // Edge temperature
    auto temp_edge = query_temperature_edge();
    if (!temp_edge) {
        std::print("[GPUMonitor] Edge temperature query failed: {}\n", temp_edge.error().message);
        telem.temperature_celsius = 0.0f;
    } else {
        telem.temperature_celsius = *temp_edge;
    }

    // Junction temperature
    auto temp_junc = query_temperature_junction();
    if (!temp_junc) {
        std::print("[GPUMonitor] Junction temperature query failed: {}\n", temp_junc.error().message);
        telem.junction_temperature_c = 0.0f;
    } else {
        telem.junction_temperature_c = *temp_junc;
    }

    // Power
    auto power = query_power();
    if (!power) {
        std::print("[GPUMonitor] Power query failed: {}\n", power.error().message);
        telem.power_watts = 0.0f;
    } else {
        telem.power_watts = *power;
    }

    // Memory
    auto mem = query_memory();
    if (!mem) {
        std::print("[GPUMonitor] Memory query failed: {}\n", mem.error().message);
        telem.memory_used_mb  = 0.0f;
        telem.memory_total_mb = 0.0f;
    } else {
        telem.memory_used_mb  = mem->first;
        telem.memory_total_mb = mem->second;
    }

    // Throttling check (use junction temp for throttling detection)
    if (temp_junc) {
        telem.is_throttling = (*temp_junc > kDefaultThrottleThresholdC);
    } else {
        telem.is_throttling = false;
    }

    return telem;
}

// ---------------------------------------------------------------------------
// Throttling detection
// ---------------------------------------------------------------------------

std::expected<bool, GPUErrorInfo> GPUMonitor::is_throttling(float threshold_c) {
    auto temp = query_temperature_junction();
    if (!temp) {
        return std::unexpected{std::move(temp.error())};
    }
    return (*temp > threshold_c);
}

} // namespace hq
