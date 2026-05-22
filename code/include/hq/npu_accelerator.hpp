#pragma once
/// @file npu_accelerator.hpp
/// @copyright Copyright (c) 2026 D Hargreaves (AKA Roylepython). LamiaFabrica. All rights reserved.
/// NPU post-processing accelerator interface, concept, and factory.
///
/// Extends the NPU abstraction layer beyond text encoding:
///   INpuPostProcessor    — virtual interface for image post-processing tasks
///   NpuPostProcessorFactory — probes hardware, returns best available backend
///   NpuAccelerator<T>    — C++26 concept for full-pipeline NPU accelerators
///
/// Architecture:
///   NpuBackend<T> (npu_backend.hpp)  — text encoding contract
///   NpuAccelerator<T>                — post-processing contract (this file)
///
/// Current implementation status (2026-05-22):
///   SyntheticNpuPostProcessor — always available, CPU pass-through (no real NPU)
///   HailoNpuPostProcessor     — skeleton: delegates to synthetic (HailoRT not installed)
///
/// Extension path (production):
///   1. Install HailoRT SDK on Ubuntu 22.04 / 24.04 (not Windows)
///   2. Compile Hailo Executable Format (HEF) for the desired post-processing network
///   3. Implement HailoNpuPostProcessor::post_process() via hailo_async_infer_runner
///   4. Set is_available() = true when device found via hailort::Device::scan_pcie()
///
/// @author LamiaFabrica Team
/// @version 1.0.0

#include "hq/cxx26_features.hpp"

#include <concepts>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace hq::npu {

// ===========================================================================
// NpuTaskType — identifies the work type a post-processor handles
// ===========================================================================

/// @brief Tag enum for the kinds of work an NPU accelerator can handle.
///
/// Used to query NpuAccelerator::can_handle() before dispatching.
enum class NpuTaskType : std::uint32_t {
    TextEncode   = 0,  ///< CLIP / text encoder (INpuEncoder handles this)
    PostProcess  = 1,  ///< Image post-processing (sharpening, noise reduction, etc.)
    SafetyFilter = 2,  ///< NSFW / content safety classifier
    LatentRefine = 3,  ///< Latent-space upscaling or refinement
};

// ===========================================================================
// NpuPostProcessRequest / NpuPostProcessResult
// ===========================================================================

/// @brief Input to a post-processing pass on the NPU.
struct NpuPostProcessRequest {
    std::span<const std::uint8_t> pixels;  ///< RGBA8 pixel data (caller-owned)
    std::uint32_t width{512};
    std::uint32_t height{512};
    NpuTaskType   task{NpuTaskType::PostProcess};
};

/// @brief Output from a completed post-processing pass.
struct NpuPostProcessResult {
    std::vector<std::uint8_t> pixels;      ///< RGBA8 output (same size as input)
    std::uint32_t width{0};
    std::uint32_t height{0};
    float processing_time_us{0.0f};        ///< Wall-clock duration of the NPU call
    float npu_utilization{0.0f};           ///< NN-core utilization % during this pass
    bool  was_npu_accelerated{false};      ///< True only when real NPU ran (not synthetic)
};

// ===========================================================================
// INpuPostProcessor — pure virtual post-processing interface
// ===========================================================================

/// @brief Runtime-polymorphic interface for image post-processing on the NPU.
///
/// Mirrors the INpuEncoder pattern: a single virtual table, factory-selected
/// at pipeline init, with a synthetic fallback that is always available.
class INpuPostProcessor {
public:
    virtual ~INpuPostProcessor() = default;

    /// @brief Execute a post-processing task.
    /// @returns NpuPostProcessResult on success, error string on failure.
    [[nodiscard]] virtual std::expected<NpuPostProcessResult, std::string>
    post_process(const NpuPostProcessRequest& req) = 0;

    /// @brief Blend conditional + unconditional noise predictions for CFG.
    ///
    /// Computes in-place: noise_out[i] = noise_uncond[i] + scale*(noise_out[i] - noise_uncond[i])
    ///
    /// Called inside denoise_step_() at every denoising step when guidance_scale > 1.0.
    /// noise_out initially holds the conditional UNet prediction; on return it holds the
    /// CFG-blended noise prediction fed to the scheduler.
    ///
    /// Production (HailoNpuPostProcessor): submitted as SAXPY on Hailo-8L NN core.
    /// Current (SyntheticNpuPostProcessor): CPU scalar SAXPY, measured and logged.
    ///
    /// For SD 1.5 at 512×512: 16,384 floats per step × 20 steps = 327,680 ops/generation.
    [[nodiscard]] virtual std::expected<void, std::string>
    blend_noise_cfg(std::span<float>       noise_out,
                    std::span<const float> noise_uncond,
                    float                  guidance_scale) noexcept = 0;

    /// @brief Returns true if this backend can run the given task type.
    [[nodiscard]] virtual bool can_handle(NpuTaskType task) const = 0;

    /// @brief Returns true if underlying hardware / SDK is actually available.
    [[nodiscard]] virtual bool is_available() const = 0;

    /// @brief Human-readable backend identifier for logging and diagnostics.
    [[nodiscard]] virtual std::string name() const = 0;

    /// @brief Instantaneous hardware utilization (0.0–100.0%).
    [[nodiscard]] virtual float utilization() const = 0;
};

// ===========================================================================
// SyntheticNpuPostProcessor — CPU pass-through, always available
// ===========================================================================

/// @brief CPU pass-through post-processor.
///
/// Copies pixels unchanged.  Used when HailoRT is not installed or the
/// requested task type is unsupported.  Always available; sets
/// was_npu_accelerated = false.
class SyntheticNpuPostProcessor final : public INpuPostProcessor {
public:
    SyntheticNpuPostProcessor() = default;
    ~SyntheticNpuPostProcessor() override = default;

    [[nodiscard]] std::expected<NpuPostProcessResult, std::string>
    post_process(const NpuPostProcessRequest& req) override;

    [[nodiscard]] std::expected<void, std::string>
    blend_noise_cfg(std::span<float>       noise_out,
                    std::span<const float> noise_uncond,
                    float                  guidance_scale) noexcept override;

    [[nodiscard]] bool can_handle(NpuTaskType task) const override;
    [[nodiscard]] bool is_available() const override { return true; }
    [[nodiscard]] std::string name() const override { return "Synthetic-PassThrough"; }
    [[nodiscard]] float utilization() const override { return 0.0f; }
};

// ===========================================================================
// HailoNpuPostProcessor — skeleton (HailoRT not yet installed)
// ===========================================================================

/// @brief Hailo-8L post-processing backend (skeleton).
///
/// Production path: compile a post-processing network (noise reduction, ESRGAN
/// upscaler, etc.) to a Hailo Executable Format (HEF) and invoke via
/// hailo_async_infer_runner.  Currently delegates to SyntheticNpuPostProcessor
/// because HailoRT is not installed on the build host.
///
/// @experimental
///   HailoNpuPostProcessor is always unavailable on this build:
///     - HailoRT SDK not installed (pkg_check_modules(HAILORT hailort) fails)
///     - No HEF compiled for any post-processing network
///     - Windows PCIe enumeration blocked (HailoRT Linux-only)
///   is_available() returns false; all calls delegate to SyntheticNpuPostProcessor.
class HailoNpuPostProcessor final : public INpuPostProcessor {
public:
    HailoNpuPostProcessor();
    ~HailoNpuPostProcessor() override = default;

    [[nodiscard]] std::expected<NpuPostProcessResult, std::string>
    post_process(const NpuPostProcessRequest& req) override;

    [[nodiscard]] std::expected<void, std::string>
    blend_noise_cfg(std::span<float>       noise_out,
                    std::span<const float> noise_uncond,
                    float                  guidance_scale) noexcept override;

    [[nodiscard]] bool can_handle(NpuTaskType task) const override;
    [[nodiscard]] bool is_available() const override { return false; }
    [[nodiscard]] std::string name() const override { return "Hailo-8L PostProcessor (skeleton)"; }
    [[nodiscard]] float utilization() const override { return 0.0f; }

private:
    SyntheticNpuPostProcessor synthetic_fallback_;
};

// ===========================================================================
// NpuPostProcessorFactory — probe hardware, return best available
// ===========================================================================

/// @brief Selects the highest-capability post-processor available at runtime.
///
/// Priority:
///   1. HailoNpuPostProcessor (is_available() = false until HailoRT installed)
///   2. SyntheticNpuPostProcessor (always available)
class NpuPostProcessorFactory {
public:
    /// @brief Construct and return the best post-processor for this hardware.
    [[nodiscard]] static std::unique_ptr<INpuPostProcessor>
    create_best_available();
};

// ===========================================================================
// NpuAccelerator<T> — C++26 concept for full post-processing NPU types
// ===========================================================================

/// @brief Statically constrains types that implement the NPU post-processing contract.
///
/// Distinct from NpuBackend<T> (which covers text encoding) — post-processors
/// do not perform text encoding, so the two contracts are kept separate.
///
/// Satisfied by SyntheticNpuPostProcessor, HailoNpuPostProcessor, and any
/// future class that implements post_process() + can_handle().
template<typename T>
concept NpuAccelerator =
    requires(T& a, const NpuPostProcessRequest& req, NpuTaskType task,
             std::span<float> noise_out, std::span<const float> noise_uncond,
             float scale) {
        { a.post_process(req) }
            -> std::same_as<std::expected<NpuPostProcessResult, std::string>>;
        { a.blend_noise_cfg(noise_out, noise_uncond, scale) }
            -> std::same_as<std::expected<void, std::string>>;
        { a.can_handle(task) } -> std::same_as<bool>;
        { a.name()           } -> std::convertible_to<std::string>;
        { a.is_available()   } -> std::same_as<bool>;
        { a.utilization()    } -> std::convertible_to<float>;
    };

// Concept proofs — compilation fails if any class violates the contract.
static_assert(NpuAccelerator<SyntheticNpuPostProcessor>,
    "SyntheticNpuPostProcessor must satisfy NpuAccelerator");
static_assert(NpuAccelerator<HailoNpuPostProcessor>,
    "HailoNpuPostProcessor must satisfy NpuAccelerator");

} // namespace hq::npu
