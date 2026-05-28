/// @file benchmark_logger.hpp
/// @copyright Copyright © 2026, all rights reserved | David Hargreaves aka Roylepython trading as The Medusa Initiative & Lamia Fabrica
/// Support: https://www.patreon.com/TheMedusaInitiative — £25/month removes ads from all software at 200 subscribers
///
/// PROPRIETARY AND CONFIDENTIAL
/// This file contains trade secrets of LamiaFabrica / D Hargreaves.
/// Unauthorised copying, disclosure or use is strictly prohibited.
/// See SECURITY.md for the responsible-disclosure policy.
///
/// Binary ring-buffer event logger for real pipeline measurement campaigns.
/// All benchmark data recorded through this logger derives from actual compiled
/// binary execution; no synthetic values are ever injected.
///
/// Thread safety: record() is safe for concurrent writers (atomic slot claim).
/// stats_for_phase() and export_*() must not be called concurrently with record().
///
/// @author LamiaFabrica Team

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <span>
#include <string>
#include <system_error>
#include <vector>

#include "hq/cxx26_features.hpp"
#if UM790_HAS_STD_EXPECTED
#  include <expected>
#else
#  error "benchmark_logger.hpp requires std::expected — GCC >= 14 with C++26 enabled"
#endif

namespace hq {

// ---------------------------------------------------------------------------
// BenchPhase — identifies the pipeline phase being timed
// ---------------------------------------------------------------------------
enum class BenchPhase : std::uint8_t {
    CAMPAIGN_START       =  0,
    CAMPAIGN_END         =  1,
    ENCODE_START         =  2,
    ENCODE_END           =  3,
    DENOISE_STEP_START   =  4,
    DENOISE_STEP_END     =  5,
    VAE_START            =  6,
    VAE_END              =  7,
    RECOVERY_START       =  8,
    RECOVERY_END         =  9,
    TIER_MIGRATE         = 10,
    OVERHEAD_PROBE       = 11,
    ITER_START           = 12,
    ITER_END             = 13,
};

[[nodiscard]] inline const char* bench_phase_name(BenchPhase p) noexcept {
    switch (p) {
        case BenchPhase::CAMPAIGN_START:     return "CAMPAIGN_START";
        case BenchPhase::CAMPAIGN_END:       return "CAMPAIGN_END";
        case BenchPhase::ENCODE_START:       return "ENCODE_START";
        case BenchPhase::ENCODE_END:         return "ENCODE_END";
        case BenchPhase::DENOISE_STEP_START: return "DENOISE_STEP_START";
        case BenchPhase::DENOISE_STEP_END:   return "DENOISE_STEP_END";
        case BenchPhase::VAE_START:          return "VAE_START";
        case BenchPhase::VAE_END:            return "VAE_END";
        case BenchPhase::RECOVERY_START:     return "RECOVERY_START";
        case BenchPhase::RECOVERY_END:       return "RECOVERY_END";
        case BenchPhase::TIER_MIGRATE:       return "TIER_MIGRATE";
        case BenchPhase::OVERHEAD_PROBE:     return "OVERHEAD_PROBE";
        case BenchPhase::ITER_START:         return "ITER_START";
        case BenchPhase::ITER_END:           return "ITER_END";
        default:                             return "UNKNOWN";
    }
}

// ---------------------------------------------------------------------------
// BenchEvent — 32 bytes, fits two per 64-byte cache line
// ---------------------------------------------------------------------------
struct BenchEvent {
    std::uint64_t timestamp_ns = 0;  // ns since epoch (steady_clock)
    std::uint64_t duration_ns  = 0;  // 0 for START events; wall-clock span for END/point
    std::uint64_t metadata     = 0;  // phase-specific (workload_id, bytes transferred, etc.)
    BenchPhase    phase        = BenchPhase::CAMPAIGN_START;
    std::uint8_t  _pad[3]     {};
    std::uint32_t step_index   = 0;
};
static_assert(sizeof(BenchEvent) == 32, "BenchEvent must be 32 bytes");

// ---------------------------------------------------------------------------
// LatencyStats — computed from a sample set; all times in milliseconds
// ---------------------------------------------------------------------------
struct LatencyStats {
    double      p50_ms    = 0.0;
    double      p95_ms    = 0.0;
    double      p99_ms    = 0.0;
    double      mean_ms   = 0.0;
    double      stddev_ms = 0.0;
    double      min_ms    = 0.0;
    double      max_ms    = 0.0;
    double      cv_pct    = 0.0;  // coefficient of variation = stddev/mean * 100
    std::size_t count     = 0;

    // Build from raw nanosecond durations (moved in — sorts in place)
    [[nodiscard]] static LatencyStats from_ns(std::vector<std::uint64_t> durations_ns);

    // Build from millisecond samples (moved in — sorts in place)
    [[nodiscard]] static LatencyStats from_ms(std::vector<double> samples_ms);
};

// ---------------------------------------------------------------------------
// BenchmarkLogger — lock-free single-producer or low-contention ring buffer
// ---------------------------------------------------------------------------
class BenchmarkLogger {
public:
    static constexpr std::size_t kDefaultCapacity = 65536;

    explicit BenchmarkLogger(std::size_t capacity = kDefaultCapacity);

    // Record an event (thread-safe for concurrent writers; noexcept)
    void record(BenchPhase    phase,
                std::uint32_t step_index  = 0,
                std::uint64_t duration_ns = 0,
                std::uint64_t metadata    = 0) noexcept;

    // Compute statistics over all events with matching phase that have duration_ns > 0
    [[nodiscard]] LatencyStats stats_for_phase(BenchPhase phase) const;

    // Export all recorded events; returns error_code on I/O failure
    [[nodiscard]] std::expected<void, std::error_code>
        export_json    (const std::filesystem::path& path) const;
    [[nodiscard]] std::expected<void, std::error_code>
        export_csv     (const std::filesystem::path& path) const;
    [[nodiscard]] std::expected<void, std::error_code>
        export_markdown(const std::filesystem::path& path) const;

    // Time N record() calls; returns average nanoseconds per call.
    // The probe events are written to the ring and can be cleared afterwards.
    [[nodiscard]] double measure_overhead_ns(std::size_t probe_count = 10000) noexcept;

    // Reset the ring (does not free memory)
    void clear() noexcept;

    [[nodiscard]] std::size_t event_count() const noexcept;
    [[nodiscard]] std::size_t capacity()    const noexcept { return ring_.size(); }

private:
    std::vector<BenchEvent>           ring_;
    alignas(64) std::atomic<std::size_t> head_{0};

    [[nodiscard]] static std::uint64_t now_ns() noexcept;
};

// ---------------------------------------------------------------------------
// ScopedPhaseTimer — RAII scope timer; records phase + duration on destruction
// ---------------------------------------------------------------------------
class ScopedPhaseTimer {
public:
    ScopedPhaseTimer(BenchmarkLogger& logger, BenchPhase phase,
                     std::uint32_t step = 0, std::uint64_t metadata = 0) noexcept
        : logger_(logger), phase_(phase), step_(step), meta_(metadata),
          t0_(std::chrono::steady_clock::now()) {}

    ~ScopedPhaseTimer() noexcept {
        const auto dur_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t0_).count());
        logger_.record(phase_, step_, dur_ns, meta_);
    }

    ScopedPhaseTimer(const ScopedPhaseTimer&)            = delete;
    ScopedPhaseTimer& operator=(const ScopedPhaseTimer&) = delete;

private:
    BenchmarkLogger&                       logger_;
    BenchPhase                             phase_;
    std::uint32_t                          step_;
    std::uint64_t                          meta_;
    std::chrono::steady_clock::time_point  t0_;
};

} // namespace hq
