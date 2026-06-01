#pragma once
/// @file athenea_probe_report.hpp
/// @copyright Copyright (c) 2026 D Hargreaves. All rights reserved.
///
/// AtheneaProbeReport — owning telemetry + readiness struct for Athenea NPU probe.
/// Moved from local lambda scope in cerberus_command_executor.cpp to global header
/// so propup tests and production code share one canonical definition.
///
/// @version 1.0.0

#include <string>
#include <sstream>

namespace hq {

struct AtheneaProbeReport {
    AtheneaProbeReport()
        : readiness_score(0), campaign_runs(0), campaign_best_sustained(0.0f), campaign_avg(0.0f)
        , pct_time_above_65(0.0f), pct_time_above_70(0.0f), longest_70_streak_sec(0.0f)
        , total_bench_us(0.0), completed(0), hot_avg_util(0.0), cold_avg_util(0.0)
        , peak_util(0.0f), avg_util(0.0f), exec_time_us(0.0)
        , used_hot(false), ran_cold_comparison(false), has_real_hw_source(false)
        , using_real_runtime_tmm(false), longest_65_streak(0.0)
        , total_telemetry_time(0.0), time_above_65(0.0), time_above_70(0.0)
        , longest_70_streak(0.0), current_65_streak(0.0), current_70_streak(0.0)
        , sum_util(0.0f), util_samples(0), cold_sum_util(0.0f), cold_samples(0)
        , hot_sum_util(0.0f), hot_samples(0), extra_sum_util(0.0f), extra_samples(0)
        , cold_completed(0), hot_completed_in_phase(0)
    {}

    int readiness_score; int campaign_runs; float campaign_best_sustained; float campaign_avg;
    float pct_time_above_65; float pct_time_above_70; float longest_70_streak_sec;
    double total_bench_us; int completed; double hot_avg_util; double cold_avg_util;
    float peak_util; float avg_util; double exec_time_us;
    bool used_hot; bool ran_cold_comparison; bool has_real_hw_source; bool using_real_runtime_tmm;
    double longest_65_streak;

    // Owning all telemetry accumulation state (no raw parallel vars anywhere in handler)
    double total_telemetry_time;
    double time_above_65;
    double time_above_70;
    double longest_70_streak;
    double current_65_streak;
    double current_70_streak;

    // Parallel accumulators eliminated: everything accumulated directly on these report.* members
    float sum_util;
    int util_samples;
    float cold_sum_util;
    int cold_samples;
    float hot_sum_util;
    int hot_samples;
    float extra_sum_util;
    int extra_samples;
    int cold_completed;
    int hot_completed_in_phase;

    void finalize_readiness() {
        readiness_score = 15;
        if (has_real_hw_source) readiness_score += 15;
        if (used_hot) readiness_score += 20;
        if (ran_cold_comparison) readiness_score += 18;
        if (hot_avg_util > 50) readiness_score += 8;
        if (has_real_hw_source) readiness_score += 10;
        if (using_real_runtime_tmm) readiness_score += 18;
        if (pct_time_above_65 > 80) readiness_score += 10;
        if (pct_time_above_70 > 50) readiness_score += 12;
        if (longest_70_streak_sec > 15) { readiness_score += 8; }
        readiness_score += 12;  // Round 22 hygiene: explicit, no misleading-indent warning
        if (readiness_score > 100) readiness_score = 100;
        if (readiness_score < 0) readiness_score = 0;
    }

    std::string build_lcmd_blob() const {
        std::ostringstream oss;
        oss << "athenea_probe_report_v1:"
            << "readiness=" << readiness_score
            << ";campaign_runs=" << campaign_runs
            << ";campaign_best_sustained=" << campaign_best_sustained
            << ";campaign_avg=" << campaign_avg
            << ";pct_time_above_65=" << pct_time_above_65
            << ";pct_time_above_70=" << pct_time_above_70
            << ";longest_70_streak_sec=" << longest_70_streak_sec
            << ";total_bench_us=" << total_bench_us
            << ";completed=" << completed
            << ";hot_avg_util=" << hot_avg_util
            << ";cold_avg_util=" << cold_avg_util
            << ";peak_util=" << peak_util
            << ";avg_util=" << avg_util
            << ";exec_time_us=" << exec_time_us
            << ";used_hot=" << (used_hot ? "true" : "false")
            << ";ran_cold_comparison=" << (ran_cold_comparison ? "true" : "false")
            << ";has_real_hw_source=" << (has_real_hw_source ? "true" : "false")
            << ";using_real_runtime_tmm=" << (using_real_runtime_tmm ? "true" : "false")
            << ";longest_65_streak=" << longest_65_streak;
        return oss.str();
    }
};

} // namespace hq
