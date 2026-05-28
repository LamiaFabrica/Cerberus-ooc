/// @file deis_scheduler.cpp
/// @copyright Copyright © 2026, all rights reserved | David Hargreaves aka Roylepython trading as The Medusa Initiative & Lamia Fabrica
/// DEIS scheduler implementation with AVX-512 precomputation.
///
/// @version 1.0.0

#include "hq/deis_scheduler.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

#include <immintrin.h>

namespace hq {

// ============================================================================
// Constants
// ============================================================================
namespace {

/// Clamp small alpha values to avoid division by zero in sqrt(alpha).
[[nodiscard]] constexpr float clamp_alpha(float a) noexcept {
    constexpr float k_min_alpha = 1e-6f;
    return (a < k_min_alpha) ? k_min_alpha : a;
}

} // anonymous namespace

// ============================================================================
// Construction / lifetime
// ============================================================================

DEISScheduler::DEISScheduler(const SchedulerConfig& cfg,
                             std::uint32_t num_inference_steps)
    : cfg_(cfg)
    , num_inference_steps_(num_inference_steps)
{
    precompute();
}

// ============================================================================
// Precomputation
// ============================================================================

void DEISScheduler::set_inference_steps(std::uint32_t steps) {
    num_inference_steps_ = steps;
    precomputed_ = false;
    precompute();
}

void DEISScheduler::precompute() {
    // 1. Full training-time beta schedule -> alphas_cumprod_ + sigmas_
    compute_beta_schedule_();

    // 2. Map inference steps -> training timesteps (evenly spaced, descending)
    timesteps_.resize(num_inference_steps_);
    const std::uint32_t n_train = cfg_.num_train_timesteps;
    const float step_size = static_cast<float>(n_train) /
                            static_cast<float>(num_inference_steps_);

    for (std::uint32_t i = 0; i < num_inference_steps_; ++i) {
        // Descending: start near num_train_timesteps-1, go down to 0
        const float raw = static_cast<float>(n_train - 1) -
                          step_size * static_cast<float>(i);
        const std::uint32_t idx = static_cast<std::uint32_t>(
            std::clamp(std::floor(raw), 0.0f,
                       static_cast<float>(n_train - 1)));
        timesteps_[i] = static_cast<std::int64_t>(idx);
    }

    // 3. Per-step coefficients (hot-loop constants)
    compute_step_coefficients_();

    precomputed_ = true;
}

// ============================================================================
// Beta schedule (training-time)
// ============================================================================

void DEISScheduler::compute_beta_schedule_() {
    const std::uint32_t n = cfg_.num_train_timesteps;
    alphas_cumprod_.resize(n);
    sigmas_.resize(n);

    float alpha_cumprod = 1.0f;
    const float beta_range = cfg_.beta_end - cfg_.beta_start;
    const float inv_n_minus_1 =
        (n > 1) ? 1.0f / static_cast<float>(n - 1) : 0.0f;

    for (std::uint32_t i = 0; i < n; ++i) {
        const float t = static_cast<float>(i) * inv_n_minus_1;

        // Beta schedule: linear or scaled_linear (sqrt-interpolated then squared)
        float beta;
        if (cfg_.beta_schedule == "scaled_linear") {
            const float sqrt_start = std::sqrt(cfg_.beta_start);
            const float sqrt_end   = std::sqrt(cfg_.beta_end);
            const float sqrt_beta  = sqrt_start + (sqrt_end - sqrt_start) * t;
            beta = sqrt_beta * sqrt_beta;
        } else {
            beta = cfg_.beta_start + beta_range * t;
        }

        // Clamp beta to valid range (0, 1)
        beta = std::clamp(beta, 0.0f, 0.999f);

        const float alpha = 1.0f - beta;
        alpha_cumprod *= alpha;

        alphas_cumprod_[i] = alpha_cumprod;
        sigmas_[i] = std::sqrt(1.0f - alpha_cumprod);
    }
}

// ============================================================================
// Per-step coefficients (inference-time constants)
// ============================================================================

void DEISScheduler::compute_step_coefficients_() {
    // step_alpha_, step_alpha_prev_, step_sqrt_alpha_, step_sqrt_one_minus_
    // were removed (M02 audit) — only these two are consumed in the hot path.
    step_coeff_x0_.resize(num_inference_steps_);
    step_coeff_eps_.resize(num_inference_steps_);

    const auto& acp = alphas_cumprod_;
    const std::uint32_t n_train = cfg_.num_train_timesteps;

    for (std::uint32_t i = 0; i < num_inference_steps_; ++i) {
        const std::uint32_t t_idx = static_cast<std::uint32_t>(timesteps_[i]);

        // Clamp timestep index to valid range
        const std::uint32_t safe_idx = std::min(t_idx, n_train - 1);

        const float alpha_t = acp[safe_idx];
        const float alpha_t_clamped = clamp_alpha(alpha_t);

        // Previous alpha: use the previous inference step's timestep,
        // or 1.0f for the first step (clean image prior)
        const float alpha_prev = (i > 0)
                                     ? acp[static_cast<std::uint32_t>(
                                           timesteps_[i - 1])]
                                     : 1.0f;
        const float alpha_prev_clamped = clamp_alpha(alpha_prev);

        const float sqrt_alpha_t = std::sqrt(alpha_t_clamped);
        const float sqrt_one_minus_alpha_t =
            std::sqrt(1.0f - alpha_t_clamped);
        const float sqrt_alpha_prev = std::sqrt(alpha_prev_clamped);
        const float sqrt_one_minus_alpha_prev =
            std::sqrt(1.0f - alpha_prev_clamped);

        // step_alpha_, step_alpha_prev_, step_sqrt_alpha_, step_sqrt_one_minus_
        // were removed (M02 audit) — computed locally, never stored.

        // ---- DEIS first-order coefficients --------------------------------
        //
        // The DDIM/DEIS update for epsilon prediction:
        //   pred_x0 = (latent - sqrt(1-alpha_t) * eps) / sqrt(alpha_t)
        //   latent_{t-1} = sqrt(alpha_prev) * pred_x0
        //                + sqrt(1 - alpha_prev) * eps
        //
        // Substituting pred_x0:
        //   latent_{t-1} = sqrt(alpha_prev)/sqrt(alpha_t) * latent
        //                + [sqrt(1-alpha_prev)
        //                   - sqrt(alpha_prev)*sqrt(1-alpha_t)/sqrt(alpha_t)]
        //                  * eps
        //
        // = coeff_x0 * latent + coeff_eps * eps

        const float coeff_x0 = sqrt_alpha_prev / sqrt_alpha_t;
        const float coeff_eps =
            sqrt_one_minus_alpha_prev - coeff_x0 * sqrt_one_minus_alpha_t;

        step_coeff_x0_[i] = coeff_x0;
        step_coeff_eps_[i] = coeff_eps;
    }
}

// ============================================================================
// Public API: timestep query
// ============================================================================

std::int64_t DEISScheduler::timestep(std::uint32_t step) const {
    if (!precomputed_) [[unlikely]] return -1;
    if (step >= num_inference_steps_) {
        return -1;
    }
    return timesteps_[step];
}

std::span<const float> DEISScheduler::alphas_cumprod() const {
    return std::span<const float>(alphas_cumprod_.data(),
                                  alphas_cumprod_.size());
}

std::span<const float> DEISScheduler::sigmas() const {
    return std::span<const float>(sigmas_.data(), sigmas_.size());
}

float DEISScheduler::step_coeff_x0(std::uint32_t step) const {
    if (step >= num_inference_steps_) return 0.0f;
    return step_coeff_x0_[step];
}

float DEISScheduler::step_coeff_eps(std::uint32_t step) const {
    if (step >= num_inference_steps_) return 0.0f;
    return step_coeff_eps_[step];
}

void DEISScheduler::get_step_coeffs(std::uint32_t step, float* coeff_x0,
                                    float* coeff_eps) const {
    if (step >= num_inference_steps_) return;
    *coeff_x0 = step_coeff_x0_[step];
    *coeff_eps = step_coeff_eps_[step];
}

// ============================================================================
// Step: the hot function
// ============================================================================

std::expected<void, SchedulerError>
DEISScheduler::step(hq::tensor::FloatTensor4D latents,
                    hq::tensor::Tensor1D<const float> model_output,
                    std::uint32_t step) {
    if (!precomputed_) [[unlikely]]
        return std::unexpected{SchedulerError::NotPrecomputed};

    if (step >= num_inference_steps_) [[unlikely]]
        return std::unexpected{SchedulerError::StepOutOfRange};

    float* latents_raw             = latents.data();
    const float* model_raw         = model_output.data();
    const std::size_t latent_count = std::min(latents.num_elements(),
                                              model_output.num_elements());

    // Runtime CPU feature detection: resolve AVX-512 support once
    static bool s_has_avx512f = []{
#if defined(__AVX512F__)
        return true;
#elif defined(__GNUC__) || defined(__clang__)
        return __builtin_cpu_supports("avx512f");
#else
        return false;
#endif
    }();

    const float coeff_x0 = step_coeff_x0_[step];
    const float coeff_eps = step_coeff_eps_[step];

    // Dispatch to fastest available path
    if (s_has_avx512f && latent_count >= 16) {
        apply_step_avx512_(latents_raw, model_raw, coeff_x0, coeff_eps,
                           latent_count);
    } else {
        apply_step_scalar_(latents_raw, model_raw, coeff_x0, coeff_eps,
                           latent_count);
    }

    return {};
}

// ============================================================================
// AVX-512 vectorized path
// ============================================================================

__attribute__((target("avx512f")))
void DEISScheduler::apply_step_avx512_(float* latents,
                                        const float* model_output,
                                        float coeff_x0, float coeff_eps,
                                        std::size_t latent_count) {
    const __m512 vc0 = _mm512_set1_ps(coeff_x0);
    const __m512 vc1 = _mm512_set1_ps(coeff_eps);

    const std::size_t simd_count = latent_count & ~std::size_t{15};

    // Main vectorized loop: 16 floats per iteration
    for (std::size_t i = 0; i < simd_count; i += 16) {
        const __m512 eps = _mm512_loadu_ps(model_output + i);
        const __m512 lat = _mm512_loadu_ps(latents + i);

        // result = coeff_x0 * latent + coeff_eps * model_output
        // Use two-lane FMA to maintain precision
        __m512 result = _mm512_mul_ps(vc0, lat);
        result = _mm512_fmadd_ps(vc1, eps, result);

        _mm512_storeu_ps(latents + i, result);
    }

    // Scalar tail: handle remaining elements
    for (std::size_t i = simd_count; i < latent_count; ++i) {
        latents[i] = coeff_x0 * latents[i] + coeff_eps * model_output[i];
    }
}

// ============================================================================
// Scalar fallback
// ============================================================================

void DEISScheduler::apply_step_scalar_(float* latents,
                                       const float* model_output,
                                       float coeff_x0, float coeff_eps,
                                       std::size_t latent_count) {
    for (std::size_t i = 0; i < latent_count; ++i) {
        latents[i] = coeff_x0 * latents[i] + coeff_eps * model_output[i];
    }
}

} // namespace hq
