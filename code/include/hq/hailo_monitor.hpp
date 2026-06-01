#pragma once

/// @file hailo_monitor.hpp
/// @copyright Copyright (c) 2026 D Hargreaves (AKA Roylepython). LamiaFabrica. All rights reserved.
/// @brief Dual-Indicator Hailo-8L Monitor — power + inference delta fusion.
///
/// Rationale: Single-indicator (power-only) monitoring conflates DMA stalls
/// with actual inference work. This monitor fuses power draw + inference
/// delta rate to produce an accurate utilization figure.
///
/// Hardware: Hailo-8L M.2 (13 TOPS, PCIe Gen3 x2)
///   Idle ~0.5W, active ~6W, max TDP 8.65W
///
/// NO SYNTHETIC DATA: When HailoRT is not available, all calls return
/// HailoError honestly. There is no fabricated telemetry.
///
/// C++26 features used:
///   - std::expected<T,E> for error handling (no exceptions for control flow)
///   - std::format for error messages
///   - std::print for diagnostic logging
///   - std::chrono for sample timestamps
///
/// @author LamiaFabrica Team
/// @version 2.0.0

#include "hq/cxx26_features.hpp"
#include "hq/concepts.hpp"

#include <chrono>
#include <cstdint>
#if UM790_HAS_STD_EXPECTED
#  include <expected>
#else
#  error "hailo_monitor.hpp requires std::expected (<expected>) — GCC >= 14 or Clang >= 18 with C++26 enabled"
#endif
#if UM790_HAS_STD_FORMAT
#  include <format>
#endif
#include <memory>
#include <string>

// ---------------------------------------------------------------------------
// Forward declarations (HailoRT)
// ---------------------------------------------------------------------------
namespace hailort {
class Device;
} // namespace hailort

// ---------------------------------------------------------------------------
// Namespace: hq
// ---------------------------------------------------------------------------
namespace hq {

// ===========================================================================
// Constants — Hailo-8L M.2 power characteristics
// ===========================================================================

inline constexpr float HAILO8L_IDLE_POWER_W = 0.5f;          ///< Idle power draw
inline constexpr float HAILO8L_ACTIVE_POWER_W = 6.0f;        ///< Typical active power
inline constexpr float HAILO8L_MAX_TDP_W = 8.65f;            ///< Maximum TDP
inline constexpr float HAILO8L_FUSED_WEIGHT_POWER = 0.5f;    ///< Fusion weight for power
inline constexpr float HAILO8L_FUSED_WEIGHT_INFERENCE = 0.5f;///< Fusion weight for inference
inline constexpr float HAILO8L_DMA_STALL_POWER_THRESHOLD = 70.0f;     ///< % power threshold
inline constexpr float HAILO8L_DMA_STALL_INFERENCE_THRESHOLD = 30.0f; ///< % inference threshold

/// Expected peak inferences per second (conservative average across models).
inline constexpr std::size_t HAILO8L_EXPECTED_INFERENCES_PER_SEC = 60;

// ===========================================================================
// HailoErrorCode — Strongly-typed error codes
// ===========================================================================

enum class HailoErrorCode : int {
    Ok = 0,
    DeviceNotFound,
    DeviceOpenFailed,
    AlreadyOpen,
    NotOpen,
    PowerReadFailed,
    TemperatureReadFailed,
    InfoReadFailed,
    InferenceCountFailed,
    NotInitialized,       ///< HailoRT SDK not available
    SensorMismatch,       ///< Dual indicators diverge beyond sanity threshold
    PcieResetFailed,
    InvalidArgument,
    InternalError,
};

// ===========================================================================
// HailoError — Human-readable error carrying code + message
// ===========================================================================

struct HailoError {
    HailoErrorCode code{HailoErrorCode::Ok};
    std::string message;

    HailoError() = default;
    HailoError(HailoErrorCode c, std::string msg)
        : code(c), message(std::move(msg)) {}

    [[nodiscard]] constexpr bool is_ok() const noexcept {
        return code == HailoErrorCode::Ok;
    }

    [[nodiscard]] HailoErrorCode error_code() const noexcept { return code; }
    [[nodiscard]] const std::string& what() const noexcept { return message; }
};

// ===========================================================================
// HailoStats — Snapshot of device state at a sample point
// ===========================================================================

struct HailoStats {
    /// Fused NN-core utilization [%] 0.0–100.0.
    /// Sentinel: -1.0f when no real sample taken.
    float nn_core_utilization{-1.0f};

    /// Raw power indicator [%] (for diagnostics).
    /// Sentinel: -1.0f when no real sample taken.
    float power_indicator{-1.0f};

    /// Raw inference indicator [%] (for diagnostics).
    /// Sentinel: -1.0f when no real sample taken.
    float inference_indicator{-1.0f};

    /// Power draw in Watts.
    /// Sentinel: -1.0f when no real sample taken.
    float power_watts{-1.0f};

    /// Chip temperature in Celsius.
    /// Sentinel: -1.0f when no real sample taken.
    float temperature_celsius{-1.0f};

    /// Cumulative inference count (monotonically increasing).
    std::uint64_t inferences_count{0};

    /// Inference delta since previous sample.
    std::uint64_t inference_delta{0};

    /// True if device is healthy (no sensor mismatch).
    bool device_healthy{false};

    /// Timestamp when this sample was taken.
    std::chrono::steady_clock::time_point timestamp;
};

// ===========================================================================
// HailoMonitor — Dual-indicator monitor for Hailo-8L
// ===========================================================================

class HailoMonitor {
public:
    // -----------------------------------------------------------------------
    // Lifecycle
    // -----------------------------------------------------------------------

    HailoMonitor();
    ~HailoMonitor() noexcept;

    // Non-copyable (unique device handle).
    HailoMonitor(const HailoMonitor&) = delete;
    HailoMonitor& operator=(const HailoMonitor&) = delete;

    // Movable.
    HailoMonitor(HailoMonitor&& other) noexcept;
    HailoMonitor& operator=(HailoMonitor&& other) noexcept;

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// @brief Open a Hailo device for monitoring (separate from inference).
    /// @param device_id  PCIe identifier, e.g. "0000:01:00.0".
    ///                   Empty string opens the first available device.
    /// @return std::expected<void, HailoError>
    [[nodiscard]] std::expected<void, HailoError> open(const std::string& device_id = "");

    /// @brief Open a Hailo device and associate a VDevice for inference counting.
    /// @param device_id  PCIe identifier.
    /// @param vdevice    Shared VDevice handle from Hailo8lEncoder or other inference context.
    /// @return std::expected<void, HailoError>
    [[nodiscard]] std::expected<void, HailoError> open_with_vdevice(
        const std::string& device_id,
        std::shared_ptr<void> vdevice);  // opaque shared_ptr to avoid HailoRT header leak

    /// @brief Take a sample: read power, temperature, inference count,
    ///        and compute dual-indicator fused utilization.
    /// @return std::expected<HailoStats, HailoError>
    [[nodiscard]] std::expected<HailoStats, HailoError> sample();

    /// @brief Perform a hard PCIe-level reset of the device.
    /// @return std::expected<void, HailoError>
    [[nodiscard]] std::expected<void, HailoError> hard_reset();

    /// @brief Close the device and release all resources.
    void close() noexcept;

    /// @brief Check whether a real HailoRT device is currently open.
    [[nodiscard]] bool is_open() const noexcept;

    /// @brief Get the identifier of the currently open device.
    [[nodiscard]] const std::string& device_id() const noexcept;

    // -----------------------------------------------------------------------
    // Tunable thresholds (runtime adjustable)
    // -----------------------------------------------------------------------

    void set_dma_stall_power_threshold(float pct) noexcept;
    void set_dma_stall_inference_threshold(float pct) noexcept;
    void set_expected_inferences_per_sec(std::size_t rate) noexcept;
    void set_inference_weight(float w) noexcept;
    void set_power_weight(float w) noexcept;

    [[nodiscard]] float dma_stall_power_threshold() const noexcept;
    [[nodiscard]] float dma_stall_inference_threshold() const noexcept;
    [[nodiscard]] std::size_t expected_inferences_per_sec() const noexcept;
    [[nodiscard]] float inference_weight() const noexcept;
    [[nodiscard]] float power_weight() const noexcept;

private:
    // -----------------------------------------------------------------------
    // Internal types
    // -----------------------------------------------------------------------

    struct DeviceDeleter {
        void operator()(hailort::Device* ptr) noexcept;
    };
    using DevicePtr = std::unique_ptr<hailort::Device, DeviceDeleter>;

    // -----------------------------------------------------------------------
    // Sensor helpers
    // -----------------------------------------------------------------------

    [[nodiscard]] std::expected<float, HailoError> read_power_watts();
    [[nodiscard]] std::expected<float, HailoError> read_temperature_celsius();
    [[nodiscard]] std::expected<std::uint64_t, HailoError> read_inference_count();

    // -----------------------------------------------------------------------
    // Fusion & detection
    // -----------------------------------------------------------------------

    /// Compute fused utilization from dual indicators.
    [[nodiscard]] float fuse_indicators(float power_util,
                                         float inference_util) const noexcept;

    /// Detect sensor mismatch (indicators diverge beyond sanity).
    [[nodiscard]] bool detect_sensor_mismatch(float power_util,
                                               float inference_util) const noexcept;

    /// Reset internal inference-delta tracking state.
    void reset_state() noexcept;

    // -----------------------------------------------------------------------
    // Data members
    // -----------------------------------------------------------------------

    DevicePtr device_;                ///< HailoRT device handle (null when SDK absent)
    std::string device_id_;           ///< User-supplied device identifier
    std::string opened_device_id_;    ///< Actual PCIe address from scan

    // Inference delta tracking
    std::uint64_t prev_inferences_{0};   ///< Previous sample inference count
    bool have_prev_inferences_{false};   ///< True after first sample

    // VDevice integration for real inference counting
    std::shared_ptr<void> vdevice_;     ///< Opaque shared_ptr<hailort::VDevice> from encoder

    // Timestamp tracking
    std::chrono::steady_clock::time_point prev_timestamp_;

    // Tunable thresholds
    float dma_stall_power_threshold_{HAILO8L_DMA_STALL_POWER_THRESHOLD};
    float dma_stall_inference_threshold_{HAILO8L_DMA_STALL_INFERENCE_THRESHOLD};
    std::size_t expected_inferences_per_sec_{HAILO8L_EXPECTED_INFERENCES_PER_SEC};
    float inference_weight_{HAILO8L_FUSED_WEIGHT_INFERENCE};
    float power_weight_{HAILO8L_FUSED_WEIGHT_POWER};
};

// ===========================================================================
// make_error — Factory for HailoError with formatted message
// ===========================================================================

template<typename... Args>
    requires hq::HqFormattableArgs<Args...>
[[nodiscard]] inline HailoError make_error(HailoErrorCode code,
                                           std::format_string<Args...> fmt,
                                           Args&&... args) {
    return HailoError{code, std::format(fmt, std::forward<Args>(args)...) };
}

// ===========================================================================
/// @brief Convenience free function for error formatting.
// ===========================================================================
[[nodiscard]] inline std::string to_string(HailoErrorCode e) {
    switch (e) {
        case HailoErrorCode::Ok:                  return "Ok";
        case HailoErrorCode::DeviceNotFound:      return "DeviceNotFound";
        case HailoErrorCode::DeviceOpenFailed:    return "DeviceOpenFailed";
        case HailoErrorCode::AlreadyOpen:         return "AlreadyOpen";
        case HailoErrorCode::NotOpen:             return "NotOpen";
        case HailoErrorCode::PowerReadFailed:     return "PowerReadFailed";
        case HailoErrorCode::TemperatureReadFailed: return "TemperatureReadFailed";
        case HailoErrorCode::InfoReadFailed:      return "InfoReadFailed";
        case HailoErrorCode::InferenceCountFailed: return "InferenceCountFailed";
        case HailoErrorCode::NotInitialized:      return "NotInitialized";
        case HailoErrorCode::SensorMismatch:      return "SensorMismatch";
        case HailoErrorCode::PcieResetFailed:     return "PcieResetFailed";
        case HailoErrorCode::InvalidArgument:     return "InvalidArgument";
        case HailoErrorCode::InternalError:       return "InternalError";
    }
    return "Unknown";
}

} // namespace hq
