#pragma once
/// @file deis_scheduler.hpp
/// @copyright Copyright (c) 2026 D Hargreaves (AKA Roylepython). LamiaFabrica. All rights reserved.
///
/// TRADE SECRET — Proprietary and Confidential.
/// This file contains trade secrets of LamiaFabrica / D Hargreaves.
/// Unauthorized use, reproduction, or disclosure is strictly prohibited.
/// See LICENSE for terms.
///
/// DEIS (Differential Equation Improved Scheduler) with AVX-512 precompute.
/// Precomputes all alpha_t, sigma_t, coefficients at construction.
/// Per-step cost: O(1) with AVX-512 vectorized apply.
///
/// DEIS uses a first-order approximation of the probability-flow ODE:
///   dx/dt = f(t)*x + 0.5*g(t)^2 * score(x,t)
/// where the discretization follows the exponential integrator form.
///
/// @version 1.0.0

#include "hq/tensor_view.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

namespace hq {

/// @brief Error conditions for DEISScheduler::step().
enum class SchedulerError : std::uint8_t {
    NotPrecomputed = 0,  ///< precompute() has not been called
    StepOutOfRange,      ///< step >= num_inference_steps
};

[[nodiscard]] inline std::string to_string(SchedulerError e) {
    switch (e) {
        case SchedulerError::NotPrecomputed: return "NotPrecomputed";
        case SchedulerError::StepOutOfRange: return "StepOutOfRange";
    }
    return "Unknown";
}

/// @brief Supported diffusion noise schedulers.
enum class SchedulerType : std::uint8_t {
    DDIM = 0,
    DEIS = 1,
    EULER = 2,
};

/// @brief Configuration for the DEIS scheduler.
struct SchedulerConfig {
    SchedulerType type{SchedulerType::DEIS};
    std::uint32_t num_train_timesteps{1000};
    float beta_start{0.0001f};
    float beta_end{0.02f};
    std::string beta_schedule{"linear"};  // "linear" or "scaled_linear"
    int prediction_type{0};               // 0=epsilon, 1=v_prediction
};

// ============================================================================
// DEISScheduler
// ============================================================================
/// @brief Production-ready DEIS scheduler with full AVX-512 precomputation.
///
/// All per-step math (alpha, sigma, coefficients) is precomputed at
/// construction / precompute() time.  The hot denoising loop pays only
/// O(latent_count/16) for the AVX-512 vectorized update.
class DEISScheduler {
public:
    /// @param cfg                Scheduler configuration (betas, schedule, etc.)
    /// @param num_inference_steps Number of denoising steps at inference
    explicit DEISScheduler(const SchedulerConfig& cfg,
                           std::uint32_t num_inference_steps);
    ~DEISScheduler() = default;

    // Non-copyable, movable
    DEISScheduler(const DEISScheduler&) = delete;
    DEISScheduler& operator=(const DEISScheduler&) = delete;
    DEISScheduler(DEISScheduler&&) noexcept = default;
    DEISScheduler& operator=(DEISScheduler&&) noexcept = default;

    /// Precompute all timesteps, alphas, sigmas, and per-step coefficients.
    /// Call once before the denoising loop.  O(num_train_timesteps).
    void precompute();

    /// Set the inference step count (may differ from constructor value).
    /// Automatically re-runs precompute().
    void set_inference_steps(std::uint32_t steps);

    /// Apply one scheduler step: update latents in-place.
    /// @param latents      Current latent tensor [1, C, H, W] (IN/OUT)
    /// @param model_output Noise prediction from UNet (flat, same element count)
    /// @param step         Current denoising step index (0-based)
    /// @return void on success, SchedulerError if not precomputed or step OOB
    [[nodiscard]] std::expected<void, SchedulerError>
    step(hq::tensor::FloatTensor4D latents,
         hq::tensor::Tensor1D<const float> model_output,
         std::uint32_t step);

    /// Get the timestep value for a given denoising step.
    [[nodiscard]] std::int64_t timestep(std::uint32_t step) const;

    /// Get precomputed alphas (for debugging / inspection).
    [[nodiscard]] std::span<const float> alphas_cumprod() const;

    /// Get precomputed sigmas (for debugging / inspection).
    [[nodiscard]] std::span<const float> sigmas() const;

    /// Number of inference steps.
    [[nodiscard]] std::uint32_t num_steps() const noexcept {
        return num_inference_steps_;
    }

    /// Precomputed DDIM coefficient for x0 direction at step.
    [[nodiscard]] float step_coeff_x0(std::uint32_t step) const;

    /// Precomputed DDIM coefficient for epsilon direction at step.
    [[nodiscard]] float step_coeff_eps(std::uint32_t step) const;

    /// Retrieve both per-step coefficients at once (for HIPGraphDenoiser).
    void get_step_coeffs(std::uint32_t step, float* coeff_x0,
                         float* coeff_eps) const;

private:
    SchedulerConfig cfg_;
    std::uint32_t num_inference_steps_{20};

    // ---- Precomputed arrays (size = num_train_timesteps) ------------------
    std::vector<float> alphas_cumprod_;   // alpha_bar_t = prod_{s=0}^t alpha_s
    std::vector<float> sigmas_;           // sqrt(1 - alpha_bar_t)
    std::vector<std::int64_t> timesteps_;   // Mapped timesteps for inference

    // ---- Precomputed per-step coefficients (size = num_inference_steps) ---
    //  These are the *only* values touched in the hot loop.
    //  NOTE: step_alpha_*, step_sqrt_alpha_*, etc. were removed (M02 audit) —
    //        only step_coeff_x0_ and step_coeff_eps_ are consumed in the hot path.
    std::vector<float> step_coeff_x0_;       // DEIS coefficient for predicted x0
    std::vector<float> step_coeff_eps_;      // DEIS coefficient for eps direction

    bool precomputed_{false};

    // ---- Internal helpers -------------------------------------------------

    /// Compute beta schedule, alphas_cumprod, and sigmas.
    /// Operates on full training timestep count.
    void compute_beta_schedule_();

    /// Precompute per-step coefficients after beta schedule is known.
    void compute_step_coefficients_();

    /// AVX-512: vectorized in-place update (16 floats / iteration).
    void apply_step_avx512_(float* latents, const float* model_output,
                            float coeff_x0, float coeff_eps,
                            std::size_t latent_count);

    /// Scalar fallback for small arrays or when AVX-512 unavailable.
    void apply_step_scalar_(float* latents, const float* model_output,
                            float coeff_x0, float coeff_eps,
                            std::size_t latent_count);
};

} // namespace hq
