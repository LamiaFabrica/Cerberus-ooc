/// @file health_score.cpp
/// @copyright Copyright (c) 2026 D Hargreaves (AKA Roylepython). LamiaFabrica. All rights reserved.
/// Pipeline Health Score calculator implementation.

#include "hq/health_score.hpp"
#include "hq/cxx26_features.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <sstream>
#if UM790_HAS_STD_PRINT
#  include <print>
#endif
#include <stdexcept>

namespace hq {

// ------------------------------------------------------------------
// Constructor
// ------------------------------------------------------------------

/// Constructs the health score calculator with the given weights.
/// Validates that weights sum to approximately 1.0.
/// @param weights Weight distribution across the 7 sub-scores.
PipelineHealthScore::PipelineHealthScore(const HealthWeights& weights)
    : weights_(weights)
{
    const float sum = weights_.gpu_utilization
                    + weights_.hailo_utilization
                    + weights_.npu_utilization
                    + weights_.latency
                    + weights_.memory
                    + weights_.recovery
                    + weights_.thermal
                    + weights_.stability;

    constexpr float epsilon = 0.001f;
    if (std::abs(sum - 1.0f) > epsilon) {
        // NOTE: std::format with float args SEGFAULTs on MinGW GCC 15.
        std::ostringstream oss;
        oss << "[PipelineHealthScore] WARNING: weights sum to " << sum
            << ", expected 1.0\n";
        std::fputs(oss.str().c_str(), stderr);
        // Normalise weights to sum to 1.0 so the composite remains valid
        if (sum > 0.0f) {
            weights_.gpu_utilization    /= sum;
            weights_.hailo_utilization  /= sum;
            weights_.npu_utilization    /= sum;
            weights_.latency            /= sum;
            weights_.memory             /= sum;
            weights_.recovery           /= sum;
            weights_.thermal            /= sum;
            weights_.stability          /= sum;
        }
    }
}

PipelineHealthScore::PipelineHealthScore(PipelineHealthScore&& other) noexcept
    : weights_(other.weights_)
    , gpu_util_{other.gpu_util_.load(std::memory_order_relaxed)}
    , gpu_temp_{other.gpu_temp_.load(std::memory_order_relaxed)}
    , hailo_util_{other.hailo_util_.load(std::memory_order_relaxed)}
    , hailo_temp_{other.hailo_temp_.load(std::memory_order_relaxed)}
    , npu_util_{other.npu_util_.load(std::memory_order_relaxed)}
    , tbt_ms_{other.tbt_ms_.load(std::memory_order_relaxed)}
    , memory_bw_{other.memory_bw_.load(std::memory_order_relaxed)}
    , recovery_attempts_{other.recovery_attempts_.load(std::memory_order_relaxed)}
    , recovery_successes_{other.recovery_successes_.load(std::memory_order_relaxed)}
    , latency_drift_{other.latency_drift_.load(std::memory_order_relaxed)}
{}

PipelineHealthScore& PipelineHealthScore::operator=(PipelineHealthScore&& other) noexcept
{
    if (this != &other) {
        weights_ = other.weights_;
        gpu_util_.store(other.gpu_util_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        gpu_temp_.store(other.gpu_temp_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        hailo_util_.store(other.hailo_util_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        hailo_temp_.store(other.hailo_temp_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        npu_util_.store(other.npu_util_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        tbt_ms_.store(other.tbt_ms_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        memory_bw_.store(other.memory_bw_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        recovery_attempts_.store(other.recovery_attempts_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        recovery_successes_.store(other.recovery_successes_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        latency_drift_.store(other.latency_drift_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    }
    return *this;
}

// ------------------------------------------------------------------
// Metric update methods (thread-safe, called from pipeline thread)
// ------------------------------------------------------------------

/// Atomically store the latest GPU telemetry.
/// @param utilization_percent GPU utilization in [0, 100].
/// @param temperature_c       GPU junction temperature in Celsius.
void PipelineHealthScore::update_gpu(float utilization_percent, float temperature_c)
{
    gpu_util_.store(utilization_percent, std::memory_order_release);
    gpu_temp_.store(temperature_c, std::memory_order_release);
}

/// Atomically store the latest Hailo NPU telemetry.
/// @param utilization_percent Hailo utilization in [0, 100].
/// @param temperature_c       Hailo junction temperature in Celsius.
void PipelineHealthScore::update_hailo(float utilization_percent, float temperature_c)
{
    hailo_util_.store(utilization_percent, std::memory_order_release);
    hailo_temp_.store(temperature_c, std::memory_order_release);
}

/// Atomically store the latest per-step latency.
/// @param time_between_steps_ms Time between consecutive pipeline steps in ms.
void PipelineHealthScore::update_latency(float time_between_steps_ms)
{
    tbt_ms_.store(time_between_steps_ms, std::memory_order_release);
}

/// Atomically store the latest memory bandwidth utilization.
/// @param bandwidth_utilization_percent Memory BW utilisation in [0, 100].
void PipelineHealthScore::update_memory(float bandwidth_utilization_percent)
{
    memory_bw_.store(bandwidth_utilization_percent, std::memory_order_release);
}

/// Record a recovery attempt outcome.
/// @param recovery_succeeded true if the recovery attempt succeeded.
void PipelineHealthScore::update_recovery(bool recovery_succeeded)
{
    recovery_attempts_.fetch_add(1, std::memory_order_acq_rel);
    if (recovery_succeeded) {
        recovery_successes_.fetch_add(1, std::memory_order_acq_rel);
    }
}

/// Atomically store the latest latency drift measurement.
/// @param latency_drift_percent Latency drift relative to baseline in [0, +inf).
void PipelineHealthScore::update_stability(float latency_drift_percent)
{
    latency_drift_.store(latency_drift_percent, std::memory_order_release);
}

void PipelineHealthScore::update_npu_utilization(float util_pct)
{
    npu_util_.store(util_pct, std::memory_order_release);
}

// ------------------------------------------------------------------
// Score computation
// ------------------------------------------------------------------

/// Normalise a "higher-is-better" metric.
/// At the target value the score is 100; at half the target it is 50;
/// at zero it is 0.  Values beyond double the target are clamped to 100.
/// @param value  Raw metric value.
/// @param target Desired target value.
/// @return Normalised score in [0, 100].
float PipelineHealthScore::normalize_linear(float value, float target) noexcept
{
    if (target <= 0.0f) {
        return 0.0f;
    }
    const float score = (value / target) * 100.0f;
    return std::clamp(score, 0.0f, 100.0f);
}

/// Normalise a "lower-is-better" metric.
/// At zero the score is 100; at half the max it is 50;
/// at the max (or beyond) it is 0.
/// @param value        Raw metric value.
/// @param max_acceptable Maximum acceptable value.
/// @return Normalised score in [0, 100].
float PipelineHealthScore::normalize_inverse(float value, float max_acceptable) noexcept
{
    if (max_acceptable <= 0.0f) {
        return (value <= 0.0f) ? 100.0f : 0.0f;
    }
    const float score = (1.0f - (value / max_acceptable)) * 100.0f;
    return std::clamp(score, 0.0f, 100.0f);
}

/// Compute the composite health report from all current metrics.
/// @return A HealthReport containing sub-scores, overall score, grade and summary.
HealthReport PipelineHealthScore::compute() const
{
    SubScores sub;
    RawHealthMetrics raw{
        .gpu_utilization_percent   = gpu_util_.load(std::memory_order_acquire),
        .gpu_temperature_c         = gpu_temp_.load(std::memory_order_acquire),
        .hailo_utilization_percent = hailo_util_.load(std::memory_order_acquire),
        .hailo_temperature_c       = hailo_temp_.load(std::memory_order_acquire),
        .npu_utilization_percent   = npu_util_.load(std::memory_order_acquire),
        .time_between_steps_ms     = tbt_ms_.load(std::memory_order_acquire),
        .memory_bandwidth_percent  = memory_bw_.load(std::memory_order_acquire),
        .recovery_success_rate_percent = 100.0f,
        .latency_drift_percent     = latency_drift_.load(std::memory_order_acquire),
    };

    // GPU utilisation — higher is better (target ~72.5%)
    sub.gpu_utilization = normalize_linear(raw.gpu_utilization_percent,
                                           target_gpu_util());

    // Hailo utilisation — higher is better (target ~84%)
    sub.hailo_utilization = normalize_linear(raw.hailo_utilization_percent,
                                              target_hailo_util());

    // NPU pipeline utilisation — higher is better (target ~84%)
    sub.npu_utilization = normalize_linear(raw.npu_utilization_percent,
                                            target_hailo_util());

    // Latency (TBT) — lower is better (budget 45 ms)
    sub.latency = normalize_inverse(raw.time_between_steps_ms,
                                    target_tbt_ms());

    // Memory bandwidth — lower is better (budget 70%)
    sub.memory = normalize_inverse(raw.memory_bandwidth_percent,
                                   target_memory_bw_pct());

    // Recovery rate — higher is better (target 98%)
    const std::uint32_t attempts  = recovery_attempts_.load(std::memory_order_acquire);
    const std::uint32_t successes = recovery_successes_.load(std::memory_order_acquire);
    raw.recovery_success_rate_percent =
        (attempts > 0) ? (100.0f * successes / attempts) : 100.0f;
    sub.recovery = normalize_linear(raw.recovery_success_rate_percent, target_recovery_rate());

    // Thermal — lower is better (budget 85C); worst of GPU/Hailo junction temps
    const float worst_temp = std::max(raw.gpu_temperature_c, raw.hailo_temperature_c);
    sub.thermal = normalize_inverse(worst_temp, target_junction_temp_c());

    // NPU thermal throttling is more severe when Hailo is hotter.
    // We redistribute from stability weight so the total remains 1.0.
    const bool hailo_is_hotter = raw.hailo_temperature_c > raw.gpu_temperature_c;
    const float thermal_boost  = hailo_is_hotter ? weights_.thermal * 0.5f : 0.0f;
    const float thermal_weight = weights_.thermal + thermal_boost;
    const float stability_weight = weights_.stability - thermal_boost;

    // Stability (latency drift) — lower is better (budget +10%)
    sub.stability = normalize_inverse(raw.latency_drift_percent,
                                      max_latency_drift_pct());

    // Weighted composite (sum of effective weights == 1.0 by construction)
    const float overall =
          weights_.gpu_utilization   * sub.gpu_utilization
        + weights_.hailo_utilization * sub.hailo_utilization
        + weights_.npu_utilization   * sub.npu_utilization
        + weights_.latency           * sub.latency
        + weights_.memory            * sub.memory
        + weights_.recovery          * sub.recovery
        + thermal_weight             * sub.thermal
        + stability_weight           * sub.stability;

    const float overall_clamped = std::clamp(overall, 0.0f, 100.0f);
    const HealthGrade grade = score_to_grade(overall_clamped);

    // Current timestamp in milliseconds since epoch
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    const std::uint64_t timestamp_ms =
        static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(now).count());

    return HealthReport{
        overall_clamped,
        grade,
        sub,
        raw,
        build_summary(sub, raw, overall_clamped),
        timestamp_ms
    };
}

// ------------------------------------------------------------------
// Grade helpers
// ------------------------------------------------------------------

/// Convert a numeric score to a letter grade.
/// @param score Score in [0, 100].
/// @return Corresponding HealthGrade.
HealthGrade PipelineHealthScore::score_to_grade(float score) noexcept
{
    if (score >= 90.0f) return HealthGrade::A;
    if (score >= 75.0f) return HealthGrade::B;
    if (score >= 60.0f) return HealthGrade::C;
    if (score >= 40.0f) return HealthGrade::D;
    return HealthGrade::F;
}

/// Get the short name for a grade.
/// @param g Grade to name.
/// @return "A", "B", "C", "D", or "F".
const char* PipelineHealthScore::grade_name(HealthGrade g) noexcept
{
    switch (g) {
        case HealthGrade::A: return "A";
        case HealthGrade::B: return "B";
        case HealthGrade::C: return "C";
        case HealthGrade::D: return "D";
        case HealthGrade::F: return "F";
    }
    return "?";
}

/// Get a human-readable description for a grade.
/// @param g Grade to describe.
/// @return "Excellent", "Good", "Acceptable", "Poor", or "Critical".
const char* PipelineHealthScore::grade_description(HealthGrade g) noexcept
{
    switch (g) {
        case HealthGrade::A: return "Excellent";
        case HealthGrade::B: return "Good";
        case HealthGrade::C: return "Acceptable";
        case HealthGrade::D: return "Poor";
        case HealthGrade::F: return "Critical";
    }
    return "Unknown";
}

// ------------------------------------------------------------------
// Summary builder
// ------------------------------------------------------------------

/// Build a concise human-readable summary from raw values plus sub-score grades.
/// @param sub     The normalized sub-scores structure.
/// @param raw     The raw metrics behind the normalized scores.
/// @param overall The overall composite score.
/// @return Formatted summary string.
std::string PipelineHealthScore::build_summary(const SubScores& sub,
                                               const RawHealthMetrics& raw,
                                               float overall)
{
    const float worst_temp = std::max(raw.gpu_temperature_c, raw.gpu_temperature_c);
    const bool hailo_is_hotter = raw.hailo_temperature_c > raw.gpu_temperature_c;
    std::string_view g_gpu    = grade_name(score_to_grade(sub.gpu_utilization));
    std::string_view g_hailo  = grade_name(score_to_grade(sub.hailo_utilization));
    std::string_view g_npu    = grade_name(score_to_grade(sub.npu_utilization));
    std::string_view g_lat    = grade_name(score_to_grade(sub.latency));
    std::string_view g_mem    = grade_name(score_to_grade(sub.memory));
    std::string_view g_rec    = grade_name(score_to_grade(sub.recovery));
    std::string_view therm_src = hailo_is_hotter ? "/H" : "/G";
    std::string_view g_therm  = grade_name(score_to_grade(sub.thermal));
    std::string_view g_stab   = grade_name(score_to_grade(sub.stability));
    std::string_view g_all    = grade_name(score_to_grade(overall));

    std::ostringstream oss;
    oss << "GPU " << static_cast<int>(raw.gpu_utilization_percent) << "%(" << g_gpu << "), "
        << "Hailo " << static_cast<int>(raw.hailo_utilization_percent) << "%(" << g_hailo << "), "
        << "NPU " << static_cast<int>(raw.npu_utilization_percent) << "%(" << g_npu << "), "
        << "TBT " << raw.time_between_steps_ms << "ms(" << g_lat << "), "
        << "Mem " << static_cast<int>(raw.memory_bandwidth_percent) << "%(" << g_mem << "), "
        << "Rec " << static_cast<int>(raw.recovery_success_rate_percent) << "%(" << g_rec << "), "
        << "Therm " << worst_temp << "C" << therm_src << "(" << g_therm << "), "
        << "Drift " << raw.latency_drift_percent << "%(" << g_stab << "), "
        << "Overall " << overall << "(" << g_all << ")";
    return oss.str();
}

// ------------------------------------------------------------------
// Reset
// ------------------------------------------------------------------

/// Reset all internal metrics to zero.  Useful between test runs
/// or when re-initialising the pipeline.
void PipelineHealthScore::reset() noexcept
{
    gpu_util_.store(0.0f, std::memory_order_relaxed);
    gpu_temp_.store(0.0f, std::memory_order_relaxed);
    hailo_util_.store(0.0f, std::memory_order_relaxed);
    hailo_temp_.store(0.0f, std::memory_order_relaxed);
    npu_util_.store(0.0f, std::memory_order_relaxed);
    tbt_ms_.store(0.0f, std::memory_order_relaxed);
    memory_bw_.store(0.0f, std::memory_order_relaxed);
    recovery_attempts_.store(0, std::memory_order_relaxed);
    recovery_successes_.store(0, std::memory_order_relaxed);
    latency_drift_.store(0.0f, std::memory_order_relaxed);
}

} // namespace hq
