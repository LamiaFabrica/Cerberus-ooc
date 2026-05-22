#pragma once
/// @file health_score.hpp
/// Pipeline Health Score calculator.
/// Computes a real-time composite health score from 7 weighted sub-scores.
/// Thread-safe for metric updates from the pipeline thread.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>

namespace hq {

// Forward declarations
struct GPUTelemetry;
struct HailoStats;
struct WatchdogStatistics;

enum class HealthGrade : std::uint8_t {
    A = 0,   // 90-100
    B = 1,   // 75-89
    C = 2,   // 60-74
    D = 3,   // 40-59
    F = 4,   // 0-39
};

struct HealthWeights {
    float gpu_utilization{0.20f};
    float hailo_utilization{0.15f};
    float latency{0.20f};
    float memory{0.10f};
    float recovery{0.15f};
    float thermal{0.10f};
    float stability{0.10f};
};

struct SubScores {
    float gpu_utilization{0.0f};    // [0, 100] — higher is better
    float hailo_utilization{0.0f};  // [0, 100]
    float latency{0.0f};            // [0, 100] — normalized from ms budget
    float memory{0.0f};             // [0, 100]
    float recovery{0.0f};           // [0, 100]
    float thermal{0.0f};            // [0, 100]
    float stability{0.0f};          // [0, 100]
};

struct HealthReport {
    float overall_score{0.0f};      // [0, 100] weighted composite
    HealthGrade grade{HealthGrade::F};
    SubScores sub_scores;
    std::string summary;            // e.g. "GPU util low (45%), thermal OK (72C)"
    std::uint64_t timestamp_ms{0};
};

class PipelineHealthScore {
public:
    explicit PipelineHealthScore(const HealthWeights& weights = {});
    ~PipelineHealthScore() = default;

    PipelineHealthScore(const PipelineHealthScore&) = delete;
    PipelineHealthScore& operator=(const PipelineHealthScore&) = delete;
    PipelineHealthScore(PipelineHealthScore&& other) noexcept;
    PipelineHealthScore& operator=(PipelineHealthScore&& other) noexcept;

    // Update raw metrics (called per-step from pipeline thread)
    void update_gpu(float utilization_percent, float temperature_c);
    void update_hailo(float utilization_percent, float temperature_c);
    void update_latency(float time_between_steps_ms);
    void update_memory(float bandwidth_utilization_percent);
    void update_recovery(bool recovery_succeeded);
    void update_stability(float latency_drift_percent);

    // Compute health score from current metrics
    [[nodiscard]] HealthReport compute() const;

    // Grade conversion
    [[nodiscard]] static HealthGrade score_to_grade(float score) noexcept;
    [[nodiscard]] static const char* grade_name(HealthGrade g) noexcept;
    [[nodiscard]] static const char* grade_description(HealthGrade g) noexcept;

    // Reset all metrics (e.g., between test runs)
    void reset() noexcept;

    // Get target thresholds
    [[nodiscard]] static constexpr float target_gpu_util() noexcept { return 72.5f; }  // midpoint of 70-75
    [[nodiscard]] static constexpr float target_hailo_util() noexcept { return 84.0f; } // midpoint of 80-88
    [[nodiscard]] static constexpr float target_tbt_ms() noexcept { return 45.0f; }
    [[nodiscard]] static constexpr float target_memory_bw_pct() noexcept { return 70.0f; }
    [[nodiscard]] static constexpr float target_recovery_rate() noexcept { return 98.0f; }
    [[nodiscard]] static constexpr float target_junction_temp_c() noexcept { return 85.0f; }
    [[nodiscard]] static constexpr float max_latency_drift_pct() noexcept { return 10.0f; }

private:
    HealthWeights weights_;

    // Running statistics (updated per-step)
    std::atomic<float> gpu_util_{0.0f};
    std::atomic<float> gpu_temp_{0.0f};
    std::atomic<float> hailo_util_{0.0f};
    std::atomic<float> hailo_temp_{0.0f};
    std::atomic<float> tbt_ms_{0.0f};
    std::atomic<float> memory_bw_{0.0f};
    std::atomic<std::uint32_t> recovery_attempts_{0};
    std::atomic<std::uint32_t> recovery_successes_{0};
    std::atomic<float> latency_drift_{0.0f};

    // Score normalization functions (target = 100, half-target = 50, zero = 0)
    [[nodiscard]] static float normalize_linear(float value, float target) noexcept;
    [[nodiscard]] static float normalize_inverse(float value, float max_acceptable) noexcept;

    /// Build a human-readable summary string from sub-scores.
    [[nodiscard]] static std::string build_summary(const SubScores& sub, float overall);
};

} // namespace hq
