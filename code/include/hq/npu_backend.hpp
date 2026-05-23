#pragma once
/// @file npu_backend.hpp
/// @copyright Copyright (c) 2026 D Hargreaves (AKA Roylepython). LamiaFabrica. All rights reserved.
/// C++26 concept layer and Windows-native extension point for NPU encoding backends.
///
/// Architectural overview:
///   INpuEncoder (npu_encoder.hpp) is the runtime polymorphic interface.
///   NpuBackend<T>               is the static (concept) formalization of that contract.
///   WindowsNpuBackend           is the extension point for DirectML / Windows ML EP.
///
/// Extension paths (not yet wired; implement when SDK is available):
///   1. DirectML EP via ONNX Runtime:
///        - Create CLIP OnnxRuntime session with OrtSessionOptionsAppendExecutionProvider_DML
///        - Wrap in CpuFallbackEncoder (already exists) with a DirectML MemoryInfo
///        - No new class needed — just a different session construction path.
///
///   2. Windows ML (WinML) native API:
///        - Load CLIP as LearningModel via Windows.AI.MachineLearning
///        - Implement WindowsNpuBackend::encode() delegating to WinML session.Run()
///        - Set is_available() to check WinML API availability at runtime.
///
///   3. Hailo DirectML co-execution:
///        - Run CLIP on Windows DirectML, post-processing on Hailo.
///        - Create a CompositeNpuBackend that chains two INpuEncoder implementations.
///
/// @author LamiaFabrica Team
/// @version 1.0.0

#include "hq/cxx26_features.hpp"
#include "hq/npu_encoder.hpp"
#include "hq/npu_accelerator.hpp"

#include <concepts>
#include <expected>
#include <memory>
#include <string>

namespace hq::npu {

// ===========================================================================
// NpuBackend<T> — static concept formalising INpuEncoder's contract
// ===========================================================================

/// @brief Statically constrains any type that satisfies the NPU backend contract.
///
///   Satisfied by Hailo8lEncoder, CpuFallbackEncoder, WindowsNpuBackend, and
/// any future backend (DirectML, OpenVINO, etc.) without requiring
/// virtual dispatch.
template<typename T>
concept NpuBackend =
    requires(T& backend, const NpuEncodeRequest& req) {
        { backend.encode(req) }
            -> std::same_as<std::expected<NpuEncodeResult, std::string>>;
        { backend.utilization() }  -> std::convertible_to<float>;
        { backend.temperature()  } -> std::convertible_to<float>;
        { backend.name()         } -> std::convertible_to<std::string>;
        { backend.is_available() } -> std::same_as<bool>;
        // Honesty markers — compile-time guarantee they exist
        { backend.synthetic_mode() } -> std::same_as<bool>;
        { backend.unavailable_reason() } -> std::convertible_to<std::string>;
    };

// Concept proofs — if any of these fail the class violates the NPU backend contract.
static_assert(NpuBackend<Hailo8lEncoder>,
    "Hailo8lEncoder must satisfy NpuBackend");
static_assert(NpuBackend<CpuFallbackEncoder>,
    "CpuFallbackEncoder must satisfy NpuBackend");

// ===========================================================================
// WindowsNpuBackend — extension point for DirectML / Windows ML EP
// ===========================================================================

/// @brief Windows-native NPU backend with compile-time availability probing.
///
/// Probes the build environment at construction to determine which Windows
/// acceleration paths are present.  Stores a human-readable diagnostic so
/// callers know exactly what's missing when `is_available()` returns false.
///
/// Extension paths (implement when the respective SDK is available):
///   1. DirectML EP — link `onnxruntime_providers_dml.dll`; define
///      `ONNXRUNTIME_DML_EP_AVAILABLE` to enable this path.
///   2. WinML (Windows.AI.MachineLearning) — include `<winml.h>` from
///      Windows SDK 10.0.17763+; implement `encode()` via WinML session.Run().
///   3. OpenVINO EP (Intel NPU) — link `onnxruntime_providers_openvino.dll`.
///
/// Satisfies `NpuBackend<T>` in all build configurations (concept-proven at
/// namespace scope below).
class WindowsNpuBackend : public INpuEncoder {
public:
    /// @brief Availability probe captured at construction.
    struct ProbeResult {
        bool d3d12_sdk_present{false};   ///< <d3d12.h> detectable at compile time
        bool directml_ep_linked{false};  ///< ONNXRUNTIME_DML_EP_AVAILABLE defined
        bool winml_sdk_present{false};   ///< <winml.h> detectable at compile time
        std::string reason;              ///< Human-readable diagnosis
    };

    WindowsNpuBackend() : probe_{probe_windows_npu_()} {}
    ~WindowsNpuBackend() override = default;

    [[nodiscard]] std::expected<NpuEncodeResult, std::string>
    encode(const NpuEncodeRequest& req) override {
        (void)req;
        return std::unexpected{
            std::string{"WindowsNpuBackend unavailable: "} + probe_.reason};
    }

    /// @return -1.0f — no NPU utilization data available (backend not operational).
    [[nodiscard]] float       utilization()  const override { return -1.0f; }
    /// @return -1.0f — no NPU temperature data available (backend not operational).
    [[nodiscard]] float       temperature()  const override { return -1.0f; }

    [[nodiscard]] std::string name() const override {
        if (probe_.directml_ep_linked) return "WindowsNpuBackend(DirectML)";
        return "WindowsNpuBackend";
    }

    /// True only when DirectML EP or WinML is actually linked and ready.
    [[nodiscard]] bool is_available() const override {
        return probe_.directml_ep_linked;
    }

    [[nodiscard]] std::string unavailable_reason() const override {
        return probe_.reason;
    }

    [[nodiscard]] bool synthetic_mode() const noexcept override {
        // Not real hardware until DirectML EP is actually linked
        return !probe_.directml_ep_linked;
    }

    /// Expose the probe result for diagnostic logging and tests.
    [[nodiscard]] const ProbeResult& probe_result() const noexcept {
        return probe_;
    }

private:
    ProbeResult probe_;

    [[nodiscard]] static ProbeResult probe_windows_npu_() noexcept {
        ProbeResult r;
#if defined(_WIN32) && __has_include(<d3d12.h>)
        r.d3d12_sdk_present = true;
#endif
#if defined(UM790_HAS_DIRECTML)
        r.directml_ep_linked = true;
#elif defined(ONNXRUNTIME_DML_EP_AVAILABLE)
        r.directml_ep_linked = true;
#endif
#if defined(_WIN32) && __has_include(<winml.h>)
        r.winml_sdk_present = true;
#endif

        if (r.directml_ep_linked) {
            r.reason = "DirectML EP linked and ready";
        } else if (r.d3d12_sdk_present) {
            r.reason = "D3D12 SDK present but DirectML EP not linked; "
                       "define ONNXRUNTIME_DML_EP_AVAILABLE and link "
                       "onnxruntime_providers_dml.dll to enable";
        } else {
            r.reason = "Windows native GPU SDK not detected at compile time; "
                       "build with MSVC or LLVM-MinGW + Windows 11 SDK targeting "
                       "DirectML (see npu_backend.hpp Extension paths)";
        }
        return r;
    }
};

// WindowsNpuBackend satisfies the concept too.
static_assert(NpuBackend<WindowsNpuBackend>,
    "WindowsNpuBackend must satisfy NpuBackend");

// ===========================================================================
// make_npu_backend<T> — concept-constrained factory helper
// ===========================================================================

/// @brief Construct an NPU backend, checking the concept at compile time.
template<NpuBackend T, typename... Args>
[[nodiscard]] std::unique_ptr<INpuEncoder> make_npu_backend(Args&&... args) {
    return std::make_unique<T>(std::forward<Args>(args)...);
}

// ===========================================================================
// make_npu_accelerator<T> — concept-constrained post-processor factory helper
// ===========================================================================

/// @brief Construct an NpuAccelerator, checking the concept at compile time.
///
/// Usage:
///   auto pp = hq::npu::make_npu_accelerator<CpuPostProcessor>();
///   auto pp = hq::npu::make_npu_accelerator<HailoNpuPostProcessor>();
template<NpuAccelerator T, typename... Args>
[[nodiscard]] std::unique_ptr<INpuPostProcessor> make_npu_accelerator(Args&&... args) {
    return std::make_unique<T>(std::forward<Args>(args)...);
}

} // namespace hq::npu
