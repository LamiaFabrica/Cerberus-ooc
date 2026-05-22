#pragma once
/// @file npu_encoder.hpp
/// @copyright Copyright (c) 2026 D Hargreaves (AKA Roylepython). LamiaFabrica. All rights reserved.
/// NPU Encoder abstraction layer for Cerberus.
///
/// Pluggable encoder interface enabling drop-in replacement of:
///   - Hailo-8L hardware encoding (production, requires HailoRT SDK)
///   - Synthetic/XOR-based deterministic encoding (CI/testing)
///   - ONNX Runtime CPU execution-provider fallback encoding
///
/// The NpuDmaPipeline delegates all encoding to an INpuEncoder*,
/// eliminating hard-coded simulate_encode() paths.  This directly
/// addresses P0-1 (real NPU encoding) and P0-2 (CPU fallback).
///
/// @version 1.0.0

#include "hq/clip_tokenizer.hpp"   // CLIPTokenizer
#include "hq/npu_pipeline.hpp"     // NpuEncodeRequest, NpuEncodeResult, PinnedTensor<float>
#include "hq/cxx26_features.hpp"

#if UM790_HAS_STD_EXPECTED
#  include <expected>
#else
#  error "npu_encoder.hpp requires std::expected — C++26 or GCC14+/Clang18+"
#endif

#include <cstdint>
#include <memory>
#include <string>

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

/// @brief Abstract NPU encoder — drop-in replacement for simulate_encode().
///
/// Every encoder implementation (hardware, synthetic, CPU-fallback) exposes
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

    /// Human-readable encoder name (e.g. "Hailo-8L", "Synthetic-XOR").
    [[nodiscard]] virtual std::string name() const = 0;

    /// True when this encoder is usable: hardware present, driver loaded,
    /// ORT session valid, etc.
    [[nodiscard]] virtual bool is_available() const = 0;
};

// ===========================================================================
// Hailo8lEncoder — real Hailo-8L inference (skeleton)
// ===========================================================================

/// @brief Hailo-8L hardware encoder via HailoRT async inference API.
///
/// Requires HailoRT >= 4.20 and a compiled HEF (Hailo Executable File)
/// for the CLIP text encoder on the target hardware.
///
/// In SDK-absent builds (no HailoRT SDK), `is_available()` returns false and
/// `encode()` logs a diagnostic + delegates to a private Synthetic fallback.
class Hailo8lEncoder : public INpuEncoder {
public:
    /// @param pcie_address  PCIe BDF, e.g. "0000:01:00.0".
    ///                      Empty string = auto-detect first device.
    explicit Hailo8lEncoder(const std::string& pcie_address = "");

    ~Hailo8lEncoder() override;

    // ---- INpuEncoder -------------------------------------------------------
    [[nodiscard]] std::expected<NpuEncodeResult, std::string>
        encode(const NpuEncodeRequest& req) override;
    [[nodiscard]] float       utilization()   const override;
    [[nodiscard]] float       temperature()   const override;
    [[nodiscard]] std::string name()          const override;
    [[nodiscard]] bool        is_available()  const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ===========================================================================
// SyntheticNpuEncoder — deterministic XOR embeddings (deterministic CI/test behavior)
// ===========================================================================

/// @brief Deterministic synthetic encoder for CI and offline testing.
///
/// Produces the same interface as Hailo8lEncoder so they're drop-in
/// replacements.  Embeddings are generated via a repeatable XOR-based
/// function seeded from the prompt and encode counter.
class SyntheticNpuEncoder : public INpuEncoder {
public:
    SyntheticNpuEncoder();

    // ---- INpuEncoder -------------------------------------------------------
    [[nodiscard]] std::expected<NpuEncodeResult, std::string>
        encode(const NpuEncodeRequest& req) override;
    [[nodiscard]] float       utilization()   const override;
    [[nodiscard]] float       temperature()   const override;
    [[nodiscard]] std::string name()          const override;
    [[nodiscard]] bool        is_available()  const override;

private:
    std::uint64_t encode_count_{0};
    float         last_utilization_{0.0f};
    float         last_temperature_{0.0f};
};

// ===========================================================================
// CpuFallbackEncoder — ONNX Runtime CPU EP for text encoding
// ===========================================================================

/// @brief Runs the CLIP text encoder on CPU via ONNX Runtime when the NPU
///        is unavailable (e.g. Hailo-8L not connected or HailoRT not installed).
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
///   1. Hailo-8L  (Hailo8lEncoder)       — if HailoRT present + device found
///   2. ONNX CPU   (CpuFallbackEncoder)   — if ORT session provided
///   3. Synthetic  (SyntheticNpuEncoder)  — always available (CI/testing)
class NpuEncoderFactory {
public:
    /// @param ort_session     Optional ORT session for CPU-fallback priority 2.
    /// @param ort_memory_info Optional CPU memory info for ORT tensor alloc.
    /// @return Owning pointer to the best available encoder.
    [[nodiscard]] static std::unique_ptr<INpuEncoder> create_best_available(
        Ort::Session*    ort_session     = nullptr,
        Ort::MemoryInfo* ort_memory_info = nullptr);
};

} // namespace hq::npu
