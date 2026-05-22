/// @file gpu_monitor.cpp
/// @copyright Copyright (c) 2026 D Hargreaves (AKA Roylepython). LamiaFabrica. All rights reserved.
/// GPUMonitor implementation: NVML + ROCm SMI telemetry queries with fallback.
///
/// Backend priority: NVML (NVIDIA) > ROCm SMI (AMD) > none (returns zeros).
///
/// @version 2.1.0

#include "hq/gpu_monitor.hpp"

#include <chrono>
#if UM790_HAS_STD_PRINT
#  include <print>
#endif
#include <utility>

// NVML (NVIDIA Management Library) — queried before ROCm SMI
#if defined(UM790_HAS_CUDA) && __has_include(<nvml.h>)
#include <nvml.h>
#define GPUMONITOR_HAS_NVML 1
// Suppress NVML deprecation warnings (nvmlDeviceGetTemperature deprecated in CUDA 13.0)
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#else
#define GPUMONITOR_HAS_NVML 0
#endif

// ROCm SMI (AMD GPU telemetry)
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
    , backend_{Backend::None}
{
}

GPUMonitor::~GPUMonitor() noexcept {
    if (initialized_) {
#if GPUMONITOR_HAS_NVML
        if (backend_ == Backend::NVML) {
            nvmlShutdown();
        }
#endif
#if GPUMONITOR_HAS_ROCM_SMI
        if (backend_ == Backend::ROCM_SMI) {
            rsmi_shut_down();
        }
#endif
    }
    initialized_ = false;
    backend_ = Backend::None;
}

GPUMonitor::GPUMonitor(GPUMonitor&& other) noexcept
    : device_index_{other.device_index_}
    , initialized_{other.initialized_}
    , backend_{other.backend_}
#if defined(UM790_HAS_CUDA) && __has_include(<nvml.h>)
    , nvml_device_{other.nvml_device_}
#endif
{
    other.device_index_ = 0;
    other.initialized_   = false;
    other.backend_       = Backend::None;
#if defined(UM790_HAS_CUDA) && __has_include(<nvml.h>)
    other.nvml_device_  = nullptr;
#endif
}

GPUMonitor& GPUMonitor::operator=(GPUMonitor&& other) noexcept {
    if (this != &other) {
        if (initialized_) {
#if GPUMONITOR_HAS_NVML
            if (backend_ == Backend::NVML) {
                nvmlShutdown();
            }
#endif
#if GPUMONITOR_HAS_ROCM_SMI
            if (backend_ == Backend::ROCM_SMI) {
                rsmi_shut_down();
            }
#endif
        }

        device_index_ = other.device_index_;
        initialized_  = other.initialized_;
        backend_      = other.backend_;
#if defined(UM790_HAS_CUDA) && __has_include(<nvml.h>)
        nvml_device_  = other.nvml_device_;
#endif

        other.device_index_ = 0;
        other.initialized_  = false;
        other.backend_      = Backend::None;
#if defined(UM790_HAS_CUDA) && __has_include(<nvml.h>)
        other.nvml_device_  = nullptr;
#endif
    }
    return *this;
}

// ---------------------------------------------------------------------------
// Initialization — tries NVML first, then ROCm SMI, then fails
// ---------------------------------------------------------------------------

std::expected<void, GPUErrorInfo> GPUMonitor::initialize() {
    if (initialized_) {
        return {};
    }

    // --- Try NVML first (NVIDIA GPUs) ---
#if GPUMONITOR_HAS_NVML
    {
        nvmlReturn_t nr = nvmlInit();
        if (nr == NVML_SUCCESS) {
            unsigned int device_count = 0;
            nr = nvmlDeviceGetCount(&device_count);
            if (nr == NVML_SUCCESS && device_count > 0 && device_index_ < device_count) {
                nvmlDevice_t dev = nullptr;
                nr = nvmlDeviceGetHandleByIndex(device_index_, &dev);
                if (nr == NVML_SUCCESS) {
#if defined(UM790_HAS_CUDA) && __has_include(<nvml.h>)
                    nvml_device_ = dev;
#endif
                    backend_     = Backend::NVML;
                    initialized_ = true;
                    std::print("[GPUMonitor] Initialized with NVML backend (device {})\n",
                               device_index_);
                    return {};
                }
                std::print("[GPUMonitor] NVML device {} handle failed (status={})\n",
                           device_index_, static_cast<int>(nr));
            } else {
                std::print("[GPUMonitor] NVML device count query failed or no devices (status={}, count={})\n",
                           static_cast<int>(nr), device_count);
            }
            nvmlShutdown();
        } else {
            std::print("[GPUMonitor] NVML init failed (status={})\n", static_cast<int>(nr));
        }
    }
#endif

    // --- Try ROCm SMI next (AMD GPUs) ---
#if GPUMONITOR_HAS_ROCM_SMI
    {
        rsmi_status_t st = rsmi_init(0);
        if (st == RSMI_STATUS_SUCCESS) {
            uint32_t device_count = 0;
            st = rsmi_num_monitor_devices(&device_count);
            if (st == RSMI_STATUS_SUCCESS && device_count > 0 && device_index_ < device_count) {
                backend_     = Backend::ROCM_SMI;
                initialized_ = true;
                std::print("[GPUMonitor] Initialized with ROCm SMI backend (device {})\n",
                           device_index_);
                return {};
            }
            std::print("[GPUMonitor] ROCm SMI device count query failed or no devices (status={}, count={})\n",
                       static_cast<int>(st), device_count);
            rsmi_shut_down();
        } else {
            std::print("[GPUMonitor] ROCm SMI init failed (status={})\n", static_cast<int>(st));
        }
    }
#endif

    // --- No backend available ---
#if !GPUMONITOR_HAS_NVML && !GPUMONITOR_HAS_ROCM_SMI
    std::print("[GPUMonitor] No GPU telemetry backend available (neither NVML nor ROCm SMI)\n");
#else
    std::print("[GPUMonitor] All GPU telemetry backends failed to initialize\n");
#endif
    return std::unexpected{GPUErrorInfo{
        .code       = GPUError::DeviceNotFound,
        .message    = "No GPU telemetry backend available; install NVIDIA driver (nvml.dll) "
                      "or ROCm SMI library (rocm_smi64).",
        .raw_status = -1,
    }};
}

// ---------------------------------------------------------------------------
// Individual queries — dispatch to active backend
// ---------------------------------------------------------------------------

std::expected<float, GPUErrorInfo> GPUMonitor::query_utilization() {
    if (!initialized_) {
        return std::unexpected{GPUErrorInfo{
            .code       = GPUError::NotInitialized,
            .message    = "GPUMonitor not initialized; call initialize() first",
            .raw_status = -1,
        }};
    }

    switch (backend_) {
#if GPUMONITOR_HAS_NVML
    case Backend::NVML: {
        nvmlUtilization_t util{};
        auto* dev = static_cast<nvmlDevice_t>(
#if defined(UM790_HAS_CUDA) && __has_include(<nvml.h>)
            nvml_device_
#else
            nullptr
#endif
        );
        if (!dev) {
            return std::unexpected{GPUErrorInfo{
                .code       = GPUError::DeviceNotFound,
                .message    = "NVML device handle is null",
                .raw_status = -1,
            }};
        }
        nvmlReturn_t nr = nvmlDeviceGetUtilizationRates(dev, &util);
        if (nr != NVML_SUCCESS) {
            return std::unexpected{GPUErrorInfo{
                .code       = GPUError::UtilizationQueryFailed,
                .message    = std::format("nvmlDeviceGetUtilizationRates failed (status {})",
                                           static_cast<int>(nr)),
                .raw_status = static_cast<int>(nr),
            }};
        }
        return static_cast<float>(util.gpu);
    }
#endif

#if GPUMONITOR_HAS_ROCM_SMI
    case Backend::ROCM_SMI: {
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
    }
#endif

    default:
        return 0.0f;
    }
}

std::expected<float, GPUErrorInfo> GPUMonitor::query_temperature_edge() {
    if (!initialized_) {
        return std::unexpected{GPUErrorInfo{
            .code       = GPUError::NotInitialized,
            .message    = "GPUMonitor not initialized; call initialize() first",
            .raw_status = -1,
        }};
    }

    switch (backend_) {
#if GPUMONITOR_HAS_NVML
    case Backend::NVML: {
        auto* dev = static_cast<nvmlDevice_t>(
#if defined(UM790_HAS_CUDA) && __has_include(<nvml.h>)
            nvml_device_
#else
            nullptr
#endif
        );
        if (!dev) {
            return std::unexpected{GPUErrorInfo{
                .code       = GPUError::DeviceNotFound,
                .message    = "NVML device handle is null",
                .raw_status = -1,
            }};
        }
        unsigned int temp = 0;
        nvmlReturn_t nr = nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &temp);
        if (nr != NVML_SUCCESS) {
            return std::unexpected{GPUErrorInfo{
                .code       = GPUError::TemperatureQueryFailed,
                .message    = std::format("nvmlDeviceGetTemperature(GPU) failed (status {})",
                                           static_cast<int>(nr)),
                .raw_status = static_cast<int>(nr),
            }};
        }
        return static_cast<float>(temp);
    }
#endif

#if GPUMONITOR_HAS_ROCM_SMI
    case Backend::ROCM_SMI: {
        std::int64_t temp_mc = 0;
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
    }
#endif

    default:
        return 0.0f;
    }
}

std::expected<float, GPUErrorInfo> GPUMonitor::query_temperature_junction() {
    if (!initialized_) {
        return std::unexpected{GPUErrorInfo{
            .code       = GPUError::NotInitialized,
            .message    = "GPUMonitor not initialized; call initialize() first",
            .raw_status = -1,
        }};
    }

    switch (backend_) {
#if GPUMONITOR_HAS_NVML
    case Backend::NVML: {
        auto* dev = static_cast<nvmlDevice_t>(
#if defined(UM790_HAS_CUDA) && __has_include(<nvml.h>)
            nvml_device_
#else
            nullptr
#endif
        );
        if (!dev) {
            return std::unexpected{GPUErrorInfo{
                .code       = GPUError::DeviceNotFound,
                .message    = "NVML device handle is null",
                .raw_status = -1,
            }};
        }
        // NVIDIA GPUs expose GPU die temperature via NVML_TEMPERATURE_GPU.
        // There is no separate "junction" or "memory" temperature sensor in the
        // public NVML enum — NVML_TEMPERATURE_GPU is the best available proxy
        // for hotspot/junction temperature on NVIDIA hardware.
        unsigned int temp = 0;
        nvmlReturn_t nr = nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &temp);
        if (nr != NVML_SUCCESS) {
            return std::unexpected{GPUErrorInfo{
                .code       = GPUError::TemperatureQueryFailed,
                .message    = std::format("nvmlDeviceGetTemperature(GPU) failed (status {})",
                                           static_cast<int>(nr)),
                .raw_status = static_cast<int>(nr),
            }};
        }
        return static_cast<float>(temp);
    }
#endif

#if GPUMONITOR_HAS_ROCM_SMI
    case Backend::ROCM_SMI: {
        std::int64_t temp_mc = 0;
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
    }
#endif

    default:
        return 0.0f;
    }
}

std::expected<float, GPUErrorInfo> GPUMonitor::query_power() {
    if (!initialized_) {
        return std::unexpected{GPUErrorInfo{
            .code       = GPUError::NotInitialized,
            .message    = "GPUMonitor not initialized; call initialize() first",
            .raw_status = -1,
        }};
    }

    switch (backend_) {
#if GPUMONITOR_HAS_NVML
    case Backend::NVML: {
        auto* dev = static_cast<nvmlDevice_t>(
#if defined(UM790_HAS_CUDA) && __has_include(<nvml.h>)
            nvml_device_
#else
            nullptr
#endif
        );
        if (!dev) {
            return std::unexpected{GPUErrorInfo{
                .code       = GPUError::DeviceNotFound,
                .message    = "NVML device handle is null",
                .raw_status = -1,
            }};
        }
        unsigned int power_mw = 0;
        nvmlReturn_t nr = nvmlDeviceGetPowerUsage(dev, &power_mw);
        if (nr != NVML_SUCCESS) {
            return std::unexpected{GPUErrorInfo{
                .code       = GPUError::PowerQueryFailed,
                .message    = std::format("nvmlDeviceGetPowerUsage failed (status {})",
                                           static_cast<int>(nr)),
                .raw_status = static_cast<int>(nr),
            }};
        }
        return static_cast<float>(power_mw) / 1000.0f; // mW -> W
    }
#endif

#if GPUMONITOR_HAS_ROCM_SMI
    case Backend::ROCM_SMI: {
        std::uint64_t power_uw = 0;
        rsmi_status_t st = rsmi_dev_power_ave_get(device_index_, 0, &power_uw);
        if (st != RSMI_STATUS_SUCCESS) {
            return std::unexpected{GPUErrorInfo{
                .code       = GPUError::PowerQueryFailed,
                .message    = std::format("rsmi_dev_power_ave_get failed (status {})",
                                           static_cast<int>(st)),
                .raw_status = static_cast<int>(st),
            }};
        }
        return static_cast<float>(power_uw) / 1'000'000.0f;
    }
#endif

    default:
        return 0.0f;
    }
}

std::expected<std::pair<float, float>, GPUErrorInfo> GPUMonitor::query_memory() {
    if (!initialized_) {
        return std::unexpected{GPUErrorInfo{
            .code       = GPUError::NotInitialized,
            .message    = "GPUMonitor not initialized; call initialize() first",
            .raw_status = -1,
        }};
    }

    switch (backend_) {
#if GPUMONITOR_HAS_NVML
    case Backend::NVML: {
        auto* dev = static_cast<nvmlDevice_t>(
#if defined(UM790_HAS_CUDA) && __has_include(<nvml.h>)
            nvml_device_
#else
            nullptr
#endif
        );
        if (!dev) {
            return std::unexpected{GPUErrorInfo{
                .code       = GPUError::DeviceNotFound,
                .message    = "NVML device handle is null",
                .raw_status = -1,
            }};
        }
        nvmlMemory_t mem{};
        nvmlReturn_t nr = nvmlDeviceGetMemoryInfo(dev, &mem);
        if (nr != NVML_SUCCESS) {
            return std::unexpected{GPUErrorInfo{
                .code       = GPUError::MemoryQueryFailed,
                .message    = std::format("nvmlDeviceGetMemoryInfo failed (status {})",
                                           static_cast<int>(nr)),
                .raw_status = static_cast<int>(nr),
            }};
        }
        const float used_mib  = static_cast<float>(mem.used)  / (1024.0f * 1024.0f);
        const float total_mib = static_cast<float>(mem.total) / (1024.0f * 1024.0f);
        return std::pair{used_mib, total_mib};
    }
#endif

#if GPUMONITOR_HAS_ROCM_SMI
    case Backend::ROCM_SMI: {
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
    }
#endif

    default:
        return std::pair{0.0f, 0.0f};
    }
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

    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    telem.timestamp_ms = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(now).count());

    auto util = query_utilization();
    if (!util) {
        std::print("[GPUMonitor] Utilization query failed: {}\n", util.error().message);
        telem.utilization_percent = 0.0f;
    } else {
        telem.utilization_percent = *util;
    }

    auto temp_edge = query_temperature_edge();
    if (!temp_edge) {
        std::print("[GPUMonitor] Edge temperature query failed: {}\n", temp_edge.error().message);
        telem.temperature_celsius = 0.0f;
    } else {
        telem.temperature_celsius = *temp_edge;
    }

    auto temp_junc = query_temperature_junction();
    if (!temp_junc) {
        std::print("[GPUMonitor] Junction temperature query failed: {}\n", temp_junc.error().message);
        telem.junction_temperature_c = 0.0f;
    } else {
        telem.junction_temperature_c = *temp_junc;
    }

    auto power = query_power();
    if (!power) {
        std::print("[GPUMonitor] Power query failed: {}\n", power.error().message);
        telem.power_watts = 0.0f;
    } else {
        telem.power_watts = *power;
    }

    auto mem = query_memory();
    if (!mem) {
        std::print("[GPUMonitor] Memory query failed: {}\n", mem.error().message);
        telem.memory_used_mb  = 0.0f;
        telem.memory_total_mb = 0.0f;
    } else {
        telem.memory_used_mb  = mem->first;
        telem.memory_total_mb = mem->second;
    }

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

// Restore deprecation warnings
#if GPUMONITOR_HAS_NVML && (defined(__GNUC__) || defined(__clang__))
#pragma GCC diagnostic pop
#endif

} // namespace hq