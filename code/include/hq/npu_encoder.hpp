#pragma once
/// @file npu_encoder.hpp
/// @copyright Copyright (c) 2026 D Hargreaves (AKA Roylepython). LamiaFabrica. All rights reserved.
/// NPU Encoder abstraction layer for Cerberus.
///
/// Pluggable encoder interface enabling drop-in replacement of:
///   - Hailo-8L hardware encoding (production, requires HailoRT SDK + compiled HEF)
///   - ONNX Runtime CPU execution-provider fallback encoding
///
/// NO SYNTHETIC ENCODERS: The pipeline must have real model files and real
/// hardware to encode text. When neither Hailo-8L nor an ORT session is
/// available, the factory returns nullptr and the caller must fall back to
/// direct ORT tokenizer path or fail gracefully.
///
/// @version 2.1.0 — production HailoRT path with HEF loading

#include "hq/clip_tokenizer.hpp"
#include "hq/npu_pipeline.hpp"
#include "hq/cxx26_features.hpp"

#if UM790_HAS_STD_EXPECTED
#  include <expected>
#else
#  error "npu_encoder.hpp requires std::expected — C++26 or GCC14+/Clang18+"
#endif

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Forward-declare ONNX Runtime types so callers don't need the full header.
// ---------------------------------------------------------------------------
namespace Ort {
class Session;
class MemoryInfo;
} // namespace Ort

// ===========================================================================
// Namespace: hq::npu
// ===========================================================================
namespace hq::npu {

// ===========================================================================
// INpuEncoder — pure virtual interface for NPU text encoding
// ===========================================================================

/// @brief Abstract NPU encoder.
///
/// Every encoder implementation (hardware, CPU-fallback) exposes
/// the same interface so NpuDmaPipeline never needs to care which backend
/// is active.
class INpuEncoder {
public:
    virtual ~INpuEncoder() = default;

    /// @brief Encode a prompt into CLIP-style embeddings.
    /// @param req  Prompt + generation parameters.
    /// @return Embeddings + telemetry on success, error string on failure.
    [[nodiscard]] virtual std::expected<NpuEncodeResult, std::string>
        encode(const NpuEncodeRequest& req) = 0;

    /// Current NPU utilization [0.0, 100.0].
    [[nodiscard]] virtual float utilization() const = 0;

    /// Current NPU temperature in degrees Celsius.
    [[nodiscard]] virtual float temperature() const = 0;

    /// Human-readable encoder name (e.g. "Hailo-8L", "ONNX-CPU").
    [[nodiscard]] virtual std::string name() const = 0;

    /// True when this encoder is usable: hardware present, HEF loaded,
    /// driver ready, ORT session valid, etc.
    [[nodiscard]] virtual bool is_available() const = 0;

    // --- NEW: Honesty markers ---
    /// @brief Returns true if this encoder operates without real NPU/GPU hardware.
    ///        CPU fallback, stub, or synthetic path = true. Real silicon = false.
    ///        ALL concrete implementations MUST override this.
    [[nodiscard]] virtual bool synthetic_mode() const noexcept { return false; }

    /// @brief Diagnostic: why is_available() returned false.
    ///        Returns empty string when is_available() == true.
    ///        ALL concrete implementations MUST override this.
    [[nodiscard]] virtual std::string unavailable_reason() const { return {}; }
};

// ===========================================================================
// Hailo8lEncoder — real Hailo-8L inference with HEF loading
// ===========================================================================

/// @brief Hailo-8L hardware encoder via HailoRT async inference API.
///
/// Construction probes for the Hailo-8L device on PCIe and attempts to load
/// the compiled HEF (Hailo Executable File). If both succeed, is_available()
/// returns true and encode() submits real async inference to the NPU. If the
/// device is present but the HEF is missing or invalid, is_available() returns
/// false — the factory will fall back to CpuFallbackEncoder, but HailoMonitor
/// can still open the same device for telemetry.
///
/// Requires HailoRT >= 4.20. On SDK-absent builds is_available() is always false.
class Hailo8lEncoder : public INpuEncoder {
public:
    /// @param pcie_address  PCIe BDF, e.g. "0000:01:00.0". Empty = auto-detect.
    /// @param hef_path      Path to compiled HEF for CLIP text encoder.
    ///                      Empty string skips HEF loading (encoder unavailable).
    explicit Hailo8lEncoder(const std::string& pcie_address = "",
                            const std::filesystem::path& hef_path = "");

    ~Hailo8lEncoder() override;

    // ---- INpuEncoder -------------------------------------------------------
    [[nodiscard]] std::expected<NpuEncodeResult, std::string>
        encode(const NpuEncodeRequest& req) override;
    [[nodiscard]] float       utilization()   const override;
    [[nodiscard]] float       temperature()   const override;
    [[nodiscard]] std::string name()          const override;
    [[nodiscard]] bool        is_available()  const override;

    /// @brief Diagnostic: why is_available() returned false.
    [[nodiscard]] std::string unavailable_reason() const override;

    /// @brief Returns true when no real Hailo hardware + HEF is loaded.
    [[nodiscard]] bool        synthetic_mode() const noexcept override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ===========================================================================
// CpuFallbackEncoder — ONNX Runtime CPU EP for text encoding
// ===========================================================================

/// @brief Runs the CLIP text encoder on CPU via ONNX Runtime when the NPU
///        is unavailable (e.g. Hailo-8L not connected, HailoRT not installed,
///        or HEF not yet compiled).
///
/// Accepts a pre-loaded Ort::Session* and Ort::MemoryInfo* for CPU allocation.
/// The session is expected to wrap a CPU-execution-provider CLIP text model.
class CpuFallbackEncoder : public INpuEncoder {
public:
    /// @param session      CPU-EP ONNX Runtime session for CLIP text.
    /// @param memory_info  CPU Ort::MemoryInfo for input/output tensor allocation.
    CpuFallbackEncoder(Ort::Session* session, Ort::MemoryInfo* memory_info);

    // ---- INpuEncoder -------------------------------------------------------
    [[nodiscard]] std::expected<NpuEncodeResult, std::string>
        encode(const NpuEncodeRequest& req) override;
    [[nodiscard]] float       utilization()   const override;
    [[nodiscard]] float       temperature()   const override;
    [[nodiscard]] std::string name()          const override;
    [[nodiscard]] bool        is_available()  const override;

    /// @brief Diagnostic: why is_available() returned false.
    [[nodiscard]] std::string unavailable_reason() const override;

    /// @brief Returns true — CPU fallback never uses real NPU/GPU hardware.
    [[nodiscard]] bool        synthetic_mode() const noexcept override;

private:
    Ort::Session*    session_{nullptr};
    Ort::MemoryInfo* memory_info_{nullptr};
    CLIPTokenizer    tokenizer_{};
};

// ===========================================================================
// NpuEncoderFactory — probes hardware, returns best available encoder
// ===========================================================================

/// @brief Static factory that probes the system and constructs the
///        highest-priority encoder that is actually available.
///
/// Priority order:
///   1. Hailo-8L  (Hailo8lEncoder)       — if HailoRT present + device found + HEF loaded
///   2. ONNX CPU   (CpuFallbackEncoder)   — if ORT session provided
///   3. nullptr    — honest failure when no encoder is available
///
/// Callers MUST check the return value for nullptr. There is no synthetic
/// fallback because fabricated embeddings and telemetry are unacceptable.
class NpuEncoderFactory {
public:
    /// @param ort_session     Optional ORT session for CPU-fallback priority 2.
    /// @param ort_memory_info Optional CPU memory info for ORT tensor alloc.
    /// @param hef_path        Optional path to Hailo HEF for priority 1.
    /// @return Owning pointer to the best available encoder, or nullptr if none.
    [[nodiscard]] static std::unique_ptr<INpuEncoder> create_best_available(
        Ort::Session*    ort_session     = nullptr,
        Ort::MemoryInfo* ort_memory_info = nullptr,
        const std::filesystem::path& hef_path = "");

    /// @brief Probe the system and return descriptions of all available Hailo devices.
    ///
    /// Each entry contains the PCIe address and availability status. This enables
    /// multi-node clustering where the coordinator selects devices across workers.
    struct DeviceInfo {
        std::string pcie_address;      ///< e.g. "0000:01:00.0"
        bool        hef_loaded{false};   ///< True if HEF is currently loaded on this device
        std::string status;            ///< Human-readable status ("ready", "no HEF", "busy")
    };

    /// @return Vector of detected Hailo devices. Empty if none found or SDK absent.
    [[nodiscard]] static std::vector<DeviceInfo> enumerate_devices(
        const std::filesystem::path& hef_path = "");
};

} // namespace hq::npu
