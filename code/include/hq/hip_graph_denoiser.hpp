#pragma once
/// @file hip_graph_denoiser.hpp
/// @copyright Copyright (c) 2026 D Hargreaves (AKA Roylepython). LamiaFabrica. All rights reserved.
///
/// TRADE SECRET — Proprietary and Confidential.
/// This file contains trade secrets of LamiaFabrica / D Hargreaves.
/// Unauthorized use, reproduction, or disclosure is strictly prohibited.
/// See LICENSE for terms.
///
/// HIP Graph-captured denoising loop for zero-CPU-overhead replay.
///
/// Records the full denoising step (H2D memcpy + UNet kernel dispatch via
/// ONNX Runtime ROCm EP + device-side DDIM scheduler kernel + D2H memcpy)
/// as a HIP graph, then replays it num_steps times with a single
/// hipGraphLaunch per step.
///
/// This class is integrated into Pipeline::generate() via execute_full().
/// When capture or replay fails, execution falls back transparently to
/// per-step ONNX Runtime inference (mirroring Pipeline::denoise_step_)
/// with full CFG support. The fallback path uses the same DEISScheduler
/// coefficients and guidance blending as the manual loop.
///
/// @author LamiaFabrica Team
/// @version 2.1.0

#include "hq/cxx26_features.hpp"
#include "hq/tensor_view.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#if UM790_HAS_STD_EXPECTED
#  include <expected>
#else
#  error "hip_graph_denoiser.hpp requires std::expected (<expected>) — GCC >= 14 or Clang >= 18 with C++26 enabled"
#endif
#include <memory>
#include <span>
#include <string>
#include <vector>

// ONNX Runtime forward declarations
namespace Ort {
class Session;
class MemoryInfo;
class RunOptions;
} // namespace Ort

namespace hq {

class DEISScheduler;

// ---------------------------------------------------------------------------
/// @brief Error conditions for HIP graph operations.
// ---------------------------------------------------------------------------
enum class GraphError : std::uint8_t {
    Ok = 0,
    HipError,
    NotCaptured,
    AlreadyCaptured,
    CaptureFailed,
    ReplayFailed,
    InvalidStepCount,
    ONNXError,
};

// ---------------------------------------------------------------------------
/// @brief Detailed error information with optional HIP error code.
// ---------------------------------------------------------------------------
struct GraphErrorInfo {
    GraphError code{GraphError::Ok};
    std::string message;
    int hip_error_code{0};
};

// ---------------------------------------------------------------------------
/// @brief Configuration for graph-captured denoising.
// ---------------------------------------------------------------------------
struct GraphConfig {
    std::uint32_t num_steps{20};       ///< Total denoising iterations
    std::size_t latent_count{1 * 4 * 64 * 64}; ///< batch * channels * H * W
    bool enable_capture{true};         ///< false => always use CPU fallback
};

// ---------------------------------------------------------------------------
/// @brief HIP Graph-captured denoising executor.
///
/// Two usage patterns:
///
///   1. Explicit capture + replay (Pipeline controls the loop):
///      @code
///      HIPGraphDenoiser denoiser(cfg);
///      denoiser.set_scheduler(&scheduler);
///      denoiser.capture(latents, embeddings, session, mem_info, shape);
///      for (uint32_t s = 1; s < num_steps; ++s) {
///          denoiser.replay(latents, embeddings, session, mem_info);
///      }
///      @endcode
///
///   2. Single-call convenience (denoiser controls the loop):
///      @code
///      HIPGraphDenoiser denoiser(cfg);
///      denoiser.set_scheduler(&scheduler);
///      denoiser.execute_full(latents, embeddings, session, mem_info, shape,
///                            7.5f, &uncond_embeddings);
///      @endcode
///
/// Thread-safety: Not thread-safe.  One instance per thread / per stream.
// ---------------------------------------------------------------------------
class HIPGraphDenoiser {
public:
    explicit HIPGraphDenoiser(const GraphConfig& cfg);
    ~HIPGraphDenoiser() noexcept;

    // Non-copyable, movable
    HIPGraphDenoiser(const HIPGraphDenoiser&) = delete;
    HIPGraphDenoiser& operator=(const HIPGraphDenoiser&) = delete;
    HIPGraphDenoiser(HIPGraphDenoiser&&) noexcept;
    HIPGraphDenoiser& operator=(HIPGraphDenoiser&&) noexcept;

    // -----------------------------------------------------------------------
    /// @brief Attach a DEIS scheduler for proper DDIM coefficients.
    ///
    /// The scheduler is borrowed (caller retains ownership). Without it,
    /// a simple linear schedule is used for coefficient computation.
    ///
    /// @param scheduler  Pointer to an initialized DEISScheduler instance
    // -----------------------------------------------------------------------
    void set_scheduler(DEISScheduler* scheduler);

    // -----------------------------------------------------------------------
    /// @brief Capture the denoising step as a HIP graph.
    ///
    /// Performs step 0 normally while recording all GPU operations into a
    /// replayable HIP graph.  After successful capture, replay() executes
    /// subsequent steps with zero CPU overhead.
    ///
    /// @param latents         Host latent buffer [latent_count floats], in/out
    /// @param embeddings      Host embeddings (copied to device during capture)
    /// @param onnx_session    ONNX Runtime UNet session (ROCm EP)
    /// @param memory_info     ONNX Runtime memory info (CPU default)
    /// @param latent_shape    {batch, channels, H, W}
    /// @param guidance_scale  CFG scale; <=1.0 disables CFG (default: 1.0)
    /// @param uncond_embeddings Unconditional embeddings for CFG; nullptr
    ///                          or empty => no CFG
    /// @return void on success, GraphErrorInfo on failure
    // -----------------------------------------------------------------------
    [[nodiscard]] std::expected<void, GraphErrorInfo>
    capture(hq::tensor::FloatTensor4D latents,
            std::span<const float> embeddings,
            Ort::Session* onnx_session,
            Ort::MemoryInfo* memory_info,
            const std::array<std::int64_t, 4>& latent_shape,
            float guidance_scale = 1.0f,
            std::span<const float> uncond_embeddings = {});

    // -----------------------------------------------------------------------
    /// @brief Replay the captured graph for one denoising step.
    ///
    /// Runs UNet inference (outside graph) with the current latents and
    /// timestep, uploads per-step DDIM scalars, then replays the captured
    /// DDIM kernel + D2H graph.  One hipGraphLaunch replays the entire
    /// device-side step.
    ///
    /// @param latents         Host latent buffer (current state), in/out
    /// @param embeddings      Host embeddings for UNet inference
    /// @param onnx_session    ONNX Runtime UNet session
    /// @param memory_info     ONNX Runtime memory info
    /// @pre capture() succeeded.
    /// @return void on success, GraphErrorInfo on failure
    // -----------------------------------------------------------------------
    [[nodiscard]] std::expected<void, GraphErrorInfo>
    replay(hq::tensor::FloatTensor4D latents,
           std::span<const float> embeddings,
           Ort::Session* onnx_session,
           Ort::MemoryInfo* memory_info);

    // -----------------------------------------------------------------------
    /// @brief Execute the full denoising loop (convenience method).
    ///
    /// Main Pipeline entry point.  Attempts graph capture for step 0;
    /// replays for steps 1..N-1; falls back to CPU execution at any point.
    ///
    /// @param latents         Host latent buffer, in/out
    /// @param embeddings      Host embeddings
    /// @param onnx_session    ONNX Runtime UNet session
    /// @param memory_info     ONNX Runtime memory info
    /// @param latent_shape    {batch, channels, H, W}
    /// @param guidance_scale  CFG scale; <=1.0 disables CFG (default: 1.0)
    /// @param uncond_embeddings Unconditional embeddings for CFG; nullptr
    ///                          or empty => no CFG
    /// @return void on success, GraphErrorInfo on failure
    // -----------------------------------------------------------------------
    [[nodiscard]] std::expected<void, GraphErrorInfo>
    execute_full(hq::tensor::FloatTensor4D latents,
                 std::span<const float> embeddings,
                 Ort::Session* onnx_session,
                 Ort::MemoryInfo* memory_info,
                 const std::array<std::int64_t, 4>& latent_shape,
                 float guidance_scale = 1.0f,
                 std::span<const float> uncond_embeddings = {});

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------
    [[nodiscard]] bool is_captured() const noexcept { return captured_; }
    [[nodiscard]] std::uint32_t steps_replayed() const noexcept {
        return steps_replayed_;
    }
    [[nodiscard]] bool is_available() const noexcept;

private:
    // ========================================================================
    // Opaque HIP state (pImpl)
    // ========================================================================
    struct HipGraphState;
    std::unique_ptr<HipGraphState> hip_state_;

    // ========================================================================
    // Persistent device buffers (allocated once, reused for all steps)
    // ========================================================================
    float* d_latents_{nullptr};          ///< Device latent buffer [latent_count]
    float* d_noise_pred_{nullptr};       ///< Device noise-pred buffer [latent_count]
    void*  d_ddim_params_{nullptr};      ///< Device DDIM params (DdimDeviceParams)

    // ========================================================================
    // Persistent pinned host buffer for step scalars (C12 fix)
    // ========================================================================
    struct DdimDeviceParams;
    DdimDeviceParams* h_pinned_params_{nullptr};

    // ========================================================================
    // Host-side state
    // ========================================================================
    GraphConfig cfg_;
    bool captured_{false};
    bool available_{false};
    std::uint32_t steps_replayed_{0};
    std::int64_t h_timestep_{0};

    // Borrowed DEIS scheduler for proper DDIM coefficients
    DEISScheduler* scheduler_{nullptr};

    // CFG state
    float cfg_scale_{1.0f};
    std::vector<float> uncond_embeddings_;

    // Cached tensor shapes (set during capture)
    std::array<std::int64_t, 4> latent_shape_{1, 4, 64, 64};
    std::size_t latent_count_{1 * 4 * 64 * 64};
    std::int64_t hidden_dim_{768};

    // ========================================================================
    // Private helpers
    // ========================================================================

    [[nodiscard]] std::expected<void, GraphErrorInfo>
    allocate_device_buffers_();

    void free_device_buffers_() noexcept;

    [[nodiscard]] std::expected<void, GraphErrorInfo>
    upload_step_scalars_(std::uint32_t step);

    /// Run UNet inference (always outside HIP graph).
    [[nodiscard]] std::expected<std::vector<float>, GraphErrorInfo>
    run_unet(std::span<const float> latents,
             std::span<const float> embeddings,
             Ort::Session* onnx_session,
             Ort::MemoryInfo* memory_info);

    [[nodiscard]] std::expected<void, GraphErrorInfo>
    execute_step_fallback_(float* latents,
                           std::span<const float> embeddings,
                           Ort::Session* onnx_session,
                           Ort::MemoryInfo* memory_info,
                           const std::array<std::int64_t, 4>& latent_shape,
                           std::uint32_t step,
                           float cfg_scale = 1.0f,
                           std::span<const float> uncond_embeddings = {});

    void compute_linear_coeffs_(std::uint32_t step,
                                float& coeff_x0, float& coeff_eps) const noexcept;
};

} // namespace hq
