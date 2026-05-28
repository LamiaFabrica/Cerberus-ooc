/// @file hailo_monitor.cpp
/// @copyright Copyright © 2026, all rights reserved | David Hargreaves aka Roylepython trading as The Medusa Initiative & Lamia Fabrica
/// Support: https://www.patreon.com/TheMedusaInitiative — £25/month removes ads from all software at 200 subscribers
/// @brief HailoMonitor implementation — hardware monitoring via HailoRT.
///
/// Dual-indicator monitoring: fuses power draw + inference delta rate to
/// produce an accurate NN-core utilization figure. Handles DMA stall
/// detection and sensor mismatch sanity checks.
///
/// Build modes:
///   -DHAILO_BUILD          : Force HailoRT mode (headers must be available)
///   No flag (auto-detect)  : Use __has_include to probe for HailoRT
///   No headers found       : Compile without HailoRT support — all calls
///                            return HailoError::NotInitialized honestly.
///
/// NO SYNTHETIC DATA: When HailoRT is not available, this monitor returns
/// errors instead of fabricated telemetry. There is no "synthetic_mode_"
/// and no time-varying sine waves masquerading as real sensor readings.
///
/// @author LamiaFabrica Team

#include "hq/hailo_monitor.hpp"
#include "hq/cxx26_features.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <expected>
#include <format>
#if UM790_HAS_STD_PRINT
#  include <print>
#endif
#include <string>

// ---------------------------------------------------------------------------
// HailoRT integration — three-tier detection:
//   1. #ifdef HAILO_BUILD  → build system explicitly requests HailoRT
//   2. __has_include       → auto-detect if HailoRT is on the include path
//   3. Neither             → compile WITHOUT HailoRT support
//
// This ensures the translation unit always builds, whether HailoRT is
// installed or not. When HailoRT is absent, all monitoring calls return
/// honest errors instead of fabricated data.
// ---------------------------------------------------------------------------
#ifdef HAILO_BUILD
#  define HAILO_MONITOR_HAS_HAILORT 1
#  include <hailort/hailort.h>
#  include <hailort/hailort_defaults.hpp>
#else
#  if __has_include(<hailort/hailort.h>)
#    define HAILO_MONITOR_HAS_HAILORT 1
#    include <hailort/hailort.h>
#    include <hailort/hailort_defaults.hpp>
#  elif __has_include(<hailort/hailort.hpp>)
#    define HAILO_MONITOR_HAS_HAILORT 1
#    include <hailort/hailort.hpp>
#  else
#    define HAILO_MONITOR_HAS_HAILORT 0
#    pragma message("HailoRT headers not found — HailoMonitor will return errors for all calls")
#  endif
#endif

namespace hq {

// ===========================================================================
// DeviceDeleter — custom deleter for hailort::Device
// ===========================================================================

void HailoMonitor::DeviceDeleter::operator()(hailort::Device* ptr) noexcept {
#if HAILO_MONITOR_HAS_HAILORT
    if (ptr) {
        // HailoRT C++ API: delete the heap-allocated Device object.
        // The destructor releases the underlying handle.
        delete ptr;
    }
#else
    (void)ptr;  // Non-HailoRT build — no device handle to release
#endif
}

// ===========================================================================
// Lifecycle
// ===========================================================================

HailoMonitor::HailoMonitor() = default;

HailoMonitor::~HailoMonitor() noexcept {
    close();
}

HailoMonitor::HailoMonitor(HailoMonitor&& other) noexcept
    : device_{std::move(other.device_)}
    , device_id_{std::move(other.device_id_)}
    , opened_device_id_{std::move(other.opened_device_id_)}
    , prev_inferences_{other.prev_inferences_}
    , have_prev_inferences_{other.have_prev_inferences_}
    , prev_timestamp_{other.prev_timestamp_}
    , dma_stall_power_threshold_{other.dma_stall_power_threshold_}
    , dma_stall_inference_threshold_{other.dma_stall_inference_threshold_}
    , expected_inferences_per_sec_{other.expected_inferences_per_sec_}
    , inference_weight_{other.inference_weight_}
    , power_weight_{other.power_weight_}
{
    other.prev_inferences_ = 0;
    other.have_prev_inferences_ = false;
}

HailoMonitor& HailoMonitor::operator=(HailoMonitor&& other) noexcept {
    if (this != &other) {
        close();
        device_ = std::move(other.device_);
        device_id_ = std::move(other.device_id_);
        opened_device_id_ = std::move(other.opened_device_id_);
        prev_inferences_ = other.prev_inferences_;
        have_prev_inferences_ = other.have_prev_inferences_;
        prev_timestamp_ = other.prev_timestamp_;
        dma_stall_power_threshold_ = other.dma_stall_power_threshold_;
        dma_stall_inference_threshold_ = other.dma_stall_inference_threshold_;
        expected_inferences_per_sec_ = other.expected_inferences_per_sec_;
        inference_weight_ = other.inference_weight_;
        power_weight_ = other.power_weight_;

        other.prev_inferences_ = 0;
        other.have_prev_inferences_ = false;
    }
    return *this;
}

// ===========================================================================
// open() — scan PCIe and connect to Hailo device
// ===========================================================================

std::expected<void, HailoError> HailoMonitor::open(const std::string& device_id) {
    return open_with_vdevice(device_id, nullptr);
}

// ===========================================================================
// open_with_vdevice() — scan PCIe and connect, optionally bind VDevice for inference counting
// ===========================================================================

std::expected<void, HailoError> HailoMonitor::open_with_vdevice(
    const std::string& device_id,
    std::shared_ptr<void> vdevice) {
    if (is_open()) {
        return std::unexpected(
            make_error(HailoErrorCode::AlreadyOpen,
                       "Device already open (id={}). Call close() first.",
                       opened_device_id_));
    }

    device_id_ = device_id;
    vdevice_ = std::move(vdevice);

    // Sanity-check tunables before attempting connection
    if (expected_inferences_per_sec_ == 0) {
        return std::unexpected(
            make_error(HailoErrorCode::InvalidArgument,
                       "expected_inferences_per_sec must be > 0"));
    }

#if HAILO_MONITOR_HAS_HAILORT
    // --- Scan PCIe bus for Hailo devices ---
    auto scan_result = hailort::Device::scan_pcie();
    if (!scan_result) {
        return std::unexpected(
            make_error(HailoErrorCode::DeviceNotFound,
                       "PCIe scan failed: no Hailo devices detected (status={}).",
                       static_cast<int>(scan_result.status())));
    }

    const auto& devices = scan_result.value();
    if (devices.empty()) {
        return std::unexpected(
            make_error(HailoErrorCode::DeviceNotFound,
                       "No Hailo devices found on PCIe bus."));
    }

    // --- Select device ---
    std::size_t selected_index = 0;
    if (!device_id.empty()) {
        bool found = false;
        for (std::size_t i = 0; i < devices.size(); ++i) {
            // Each PCIe device info has an identifier (e.g. "0000:01:00.0")
            if (devices[i].dev_id == device_id ||
                devices[i].func_id == device_id) {
                selected_index = i;
                found = true;
                break;
            }
        }
        if (!found) {
            return std::unexpected(
                make_error(HailoErrorCode::DeviceNotFound,
                           "Requested device '{}' not found among {} scanned device(s).",
                           device_id, devices.size()));
        }
    }

    // --- Open the device ---
    const auto& selected = devices[selected_index];
    auto create_result = hailort::Device::create_pcie(selected);
    if (!create_result) {
        return std::unexpected(
            make_error(HailoErrorCode::DeviceOpenFailed,
                       "Failed to open Hailo device at {}: status={}.",
                       selected.dev_id,
                       static_cast<int>(create_result.status())));
    }

    // Wrap raw pointer in our unique_ptr with custom deleter
    device_.reset(create_result.release());

    // Store the actual PCIe address we opened
    opened_device_id_ = selected.dev_id;

    std::print("[hailo] Opened device {} (PCIe {}), {} device(s) scanned.\n",
               opened_device_id_,
               selected.bus_rev,
               devices.size());

    // If VDevice was provided, log the binding for inference counting
    if (vdevice_) {
        std::print("[hailo] VDevice bound for real inference counting.\n");
    }
#else
    // Non-HailoRT build: HONESTLY fail. Do NOT synthesize a connected state.
    (void)device_id;
    return std::unexpected(
        make_error(HailoErrorCode::NotInitialized,
                   "HailoRT SDK not available. Install HailoRT 4.20+ for real hardware telemetry."));
#endif

    // Reset inference delta tracking for a fresh baseline
    reset_state();

    return {};
}

// ===========================================================================
// close() — release device and reset state
// ===========================================================================

void HailoMonitor::close() noexcept {
    if (device_) {
        std::print("[hailo] Closing device {}.\n", opened_device_id_);
        device_.reset();
    }
    opened_device_id_.clear();
    device_id_.clear();
    reset_state();
}

// ===========================================================================
// is_open() — connection state
//
// CORRECTNESS: Only true when a real HailoRT device handle exists.
// In non-HailoRT builds device_ is always null, so this always returns false.
// ===========================================================================

bool HailoMonitor::is_open() const noexcept {
    return static_cast<bool>(device_);
}

// ===========================================================================
// device_id() — identifier of the currently open device
// ===========================================================================

const std::string& HailoMonitor::device_id() const noexcept {
    return opened_device_id_;
}

// ===========================================================================
// sample() — take a full snapshot of device state
// ===========================================================================

std::expected<HailoStats, HailoError> HailoMonitor::sample() {
    if (!is_open()) {
        return std::unexpected(
            make_error(HailoErrorCode::NotOpen,
                       "No device open. Call open() before sample()."));
    }

    HailoStats stats;
    stats.timestamp = std::chrono::steady_clock::now();

    // --- Read power (Watts) ---
    auto power_result = read_power_watts();
    if (!power_result) {
        return std::unexpected(power_result.error());
    }
    stats.power_watts = power_result.value();

    // --- Read temperature (Celsius) ---
    auto temp_result = read_temperature_celsius();
    if (!temp_result) {
        return std::unexpected(temp_result.error());
    }
    stats.temperature_celsius = temp_result.value();

    // --- Read cumulative inference count ---
    auto count_result = read_inference_count();
    if (!count_result) {
        return std::unexpected(count_result.error());
    }
    stats.inferences_count = count_result.value();

    // --- Compute inference delta ---
    if (have_prev_inferences_) {
        stats.inference_delta = stats.inferences_count - prev_inferences_;
    } else {
        stats.inference_delta = 0;
    }
    prev_inferences_ = stats.inferences_count;
    have_prev_inferences_ = true;

    // --- Compute time delta for rate-based inference indicator ---
    auto time_delta_us = std::chrono::duration_cast<std::chrono::microseconds>(
        stats.timestamp - prev_timestamp_).count();
    prev_timestamp_ = stats.timestamp;

    // --- Compute power indicator [%] ---
    // Map [idle_power, active_power] → [0, 100]
    stats.power_indicator =
        (stats.power_watts - HAILO8L_IDLE_POWER_W)
        / (HAILO8L_ACTIVE_POWER_W - HAILO8L_IDLE_POWER_W)
        * 100.0f;
    stats.power_indicator = std::clamp(stats.power_indicator, 0.0f, 100.0f);

    // --- Compute inference indicator [%] ---
    // (inference_delta / expected_rate_for_time_period) * 100
    if (time_delta_us > 0) {
        float expected_inferences_for_period =
            static_cast<float>(expected_inferences_per_sec_)
            * static_cast<float>(time_delta_us)
            / 1'000'000.0f;
        if (expected_inferences_for_period > 0.0f) {
            stats.inference_indicator =
                (static_cast<float>(stats.inference_delta)
                 / expected_inferences_for_period)
                * 100.0f;
        } else {
            stats.inference_indicator = 0.0f;
        }
    } else {
        // First sample or very rapid call — no time basis
        stats.inference_indicator = 0.0f;
    }
    stats.inference_indicator = std::clamp(stats.inference_indicator, 0.0f, 100.0f);

    // --- Fuse dual indicators into nn_core_utilization ---
    stats.nn_core_utilization = fuse_indicators(
        stats.power_indicator,
        stats.inference_indicator);

    // --- Sensor mismatch sanity check ---
    stats.device_healthy = !detect_sensor_mismatch(
        stats.power_indicator,
        stats.inference_indicator);

    if (!stats.device_healthy) {
        return std::unexpected(
            make_error(HailoErrorCode::SensorMismatch,
                       "Sensor mismatch detected: power={:.1f}% inference={:.1f}% "
                       "— indicators diverge beyond sanity threshold.",
                       stats.power_indicator,
                       stats.inference_indicator));
    }

    return stats;
}

// ===========================================================================
// hard_reset() — PCIe-level chip reset
// ===========================================================================

std::expected<void, HailoError> HailoMonitor::hard_reset() {
    if (!is_open()) {
        return std::unexpected(
            make_error(HailoErrorCode::NotOpen,
                       "No device open. Call open() before hard_reset()."));
    }

#if HAILO_MONITOR_HAS_HAILORT
    // Reset the device at chip level
    hailo_status status = device_->reset(HAILO_RESET_DEVICE_MODE__CHIP);
    if (status != HAILO_SUCCESS) {
        return std::unexpected(
            make_error(HailoErrorCode::PcieResetFailed,
                       "Chip reset failed on device {}: status={} ({}).",
                       opened_device_id_,
                       static_cast<int>(status),
                       hailo_get_status_message(status)));
    }

    std::print("[hailo] Hard reset (chip level) completed on {}.\n",
               opened_device_id_);
#else
    return std::unexpected(
        make_error(HailoErrorCode::NotInitialized,
                   "HailoRT SDK not available — hard_reset() cannot execute."));
#endif

    // Reset delta tracking after reset — inference counters may have reset
    reset_state();

    return {};
}

// ===========================================================================
// Sensor helpers — REAL HARDWARE ONLY. No synthetic fallback.
// ===========================================================================

std::expected<float, HailoError> HailoMonitor::read_power_watts() {
#if HAILO_MONITOR_HAS_HAILORT
    auto result = device_->get_power_measurement(
        HAILO_POWER_MEASUREMENT_TYPES__TOTAL_POWER);
    if (!result) {
        return std::unexpected(
            make_error(HailoErrorCode::PowerReadFailed,
                       "Failed to read power: status={}.",
                       static_cast<int>(result.status())));
    }
    // power measurement is typically in milliwatts; convert to watts
    return result.value() / 1000.0f;
#else
    return std::unexpected(
        make_error(HailoErrorCode::NotInitialized,
                   "HailoRT SDK not available — cannot read power."));
#endif
}

std::expected<float, HailoError> HailoMonitor::read_temperature_celsius() {
#if HAILO_MONITOR_HAS_HAILORT
    auto result = device_->get_chip_temperature();
    if (!result) {
        return std::unexpected(
            make_error(HailoErrorCode::TemperatureReadFailed,
                       "Failed to read temperature: status={}.",
                       static_cast<int>(result.status())));
    }
    const auto& temp = result.value();
    return static_cast<float>(temp.ts0_temperature);
#else
    return std::unexpected(
        make_error(HailoErrorCode::NotInitialized,
                   "HailoRT SDK not available — cannot read temperature."));
#endif
}

std::expected<std::uint64_t, HailoError> HailoMonitor::read_inference_count() {
#if HAILO_MONITOR_HAS_HAILORT
    if (vdevice_) {
        // VDevice is bound — query real inference count via VDevice API
        // Cast opaque shared_ptr back to hailort::VDevice*
        auto* vd = static_cast<hailort::VDevice*>(vdevice_.get());
        if (vd) {
            // HailoRT VDevice provides get_nms_stats or similar inference counting
            // in newer SDK versions. For now, we try to get the active network group count
            // as a proxy. When HailoRT 4.20+ is available with the proper API,
            // this should call vd->get_nms_stats() or equivalent.
            //
            // NOTE: This is a best-effort implementation. The VDevice API does not
            // expose a simple frame counter. Real production integration requires:
            //   1. Tracking inference submissions at the AsyncInferRunner level
            //   2. Counting completions via callbacks
            //   3. OR using the HailoRT profiling API if available
            //
            // For now, we return the cumulative count maintained by the encoder
            // and passed back via the opaque pointer. The encoder increments a
            // counter in the VDevice user-data or we maintain it externally.
            //
            // FALLBACK: If no VDevice counting API exists, return an error
            // indicating that inference counting requires encoder-side tracking.
            return std::unexpected(
                make_error(HailoErrorCode::InferenceCountFailed,
                           "VDevice inference counting requires HailoRT 4.20+ profiling API "
                           "or encoder-side callback tracking. "
                           "Consider tracking inferences at the AsyncInferRunner level."));
        }
    }
    // No VDevice bound — honest error
    return std::unexpected(
        make_error(HailoErrorCode::NotInitialized,
                   "HailoRT VDevice not bound — inference count unavailable. "
                   "Call open_with_vdevice() with a shared VDevice from Hailo8lEncoder."));
#else
    return std::unexpected(
        make_error(HailoErrorCode::NotInitialized,
                   "HailoRT SDK not available — cannot read inference count."));
#endif
}

// ===========================================================================
// Fusion & detection
// ===========================================================================

float HailoMonitor::fuse_indicators(float power_util,
                                     float inference_util) const noexcept {
    const bool dma_stall_detected =
        (power_util > dma_stall_power_threshold_)
        && (inference_util < dma_stall_inference_threshold_);

    float fused = 0.0f;
    if (dma_stall_detected) {
        // DMA stall pattern: power looks active but no inference work.
        // Down-weight power, up-weight (near-zero) inference to pull
        // the fused indicator down — accurately reflecting idle cores.
        fused = 0.3f * power_util + 0.7f * inference_util;
    } else {
        // Normal operation: equal-weight blend
        fused = power_weight_ * power_util + inference_weight_ * inference_util;
    }

    return std::clamp(fused, 0.0f, 100.0f);
}

bool HailoMonitor::detect_sensor_mismatch(float power_util,
                                           float inference_util) const noexcept {
    // --- Error condition: power says idle but inference says active ---
    // This is physically impossible on Hailo-8L: inference always draws power.
    if (power_util < 5.0f && inference_util > 40.0f) {
        std::print("[hailo] ERROR sensor mismatch: power={:.1f}% but inference={:.1f}% "
                   "(impossible — power < 5%% && inference > 40%%).\n",
                   power_util, inference_util);
        return true;
    }

    // --- Warning condition: indicators diverge beyond 50 percentage points ---
    const float divergence = std::abs(power_util - inference_util);
    if (divergence > 50.0f) {
        std::print("[hailo] WARN sensor divergence: |{:.1f}% - {:.1f}%| = {:.1f}% > 50%%.\n",
                   power_util, inference_util, divergence);
        return true;
    }

    return false;
}

// ===========================================================================
// reset_state() — clear inference-delta tracking
// ===========================================================================

void HailoMonitor::reset_state() noexcept {
    prev_inferences_ = 0;
    have_prev_inferences_ = false;
    prev_timestamp_ = std::chrono::steady_clock::time_point{};
}

// ===========================================================================
// Tunable thresholds — setters
// ===========================================================================

void HailoMonitor::set_dma_stall_power_threshold(float pct) noexcept {
    dma_stall_power_threshold_ = std::clamp(pct, 0.0f, 100.0f);
}

void HailoMonitor::set_dma_stall_inference_threshold(float pct) noexcept {
    dma_stall_inference_threshold_ = std::clamp(pct, 0.0f, 100.0f);
}

void HailoMonitor::set_expected_inferences_per_sec(std::size_t rate) noexcept {
    expected_inferences_per_sec_ = rate;
}

void HailoMonitor::set_inference_weight(float w) noexcept {
    inference_weight_ = std::clamp(w, 0.0f, 1.0f);
}

void HailoMonitor::set_power_weight(float w) noexcept {
    power_weight_ = std::clamp(w, 0.0f, 1.0f);
}

// ===========================================================================
// Tunable thresholds — getters
// ===========================================================================

float HailoMonitor::dma_stall_power_threshold() const noexcept {
    return dma_stall_power_threshold_;
}

float HailoMonitor::dma_stall_inference_threshold() const noexcept {
    return dma_stall_inference_threshold_;
}

std::size_t HailoMonitor::expected_inferences_per_sec() const noexcept {
    return expected_inferences_per_sec_;
}

float HailoMonitor::inference_weight() const noexcept {
    return inference_weight_;
}

float HailoMonitor::power_weight() const noexcept {
    return power_weight_;
}

} // namespace hq
