/// @file benchmark_logger.cpp
/// @copyright Copyright © 2026, all rights reserved | David Hargreaves aka Roylepython trading as The Medusa Initiative & Lamia Fabrica
/// @author LamiaFabrica Team

#include "hq/benchmark_logger.hpp"

#include <format>
#include <stdexcept>

namespace hq {

// ---------------------------------------------------------------------------
// LatencyStats
// ---------------------------------------------------------------------------

LatencyStats LatencyStats::from_ns(std::vector<std::uint64_t> durations_ns) {
    std::vector<double> ms;
    ms.reserve(durations_ns.size());
    for (auto d : durations_ns)
        ms.push_back(static_cast<double>(d) / 1.0e6);
    return from_ms(std::move(ms));
}

LatencyStats LatencyStats::from_ms(std::vector<double> samples) {
    LatencyStats s;
    s.count = samples.size();
    if (s.count == 0) return s;

    std::sort(samples.begin(), samples.end());

    s.min_ms = samples.front();
    s.max_ms = samples.back();

    auto percentile = [&](double p) -> double {
        const double idx = p * (static_cast<double>(s.count) - 1.0);
        const auto   lo  = static_cast<std::size_t>(idx);
        const auto   hi  = lo + 1;
        if (hi >= s.count) return samples.back();
        return samples[lo] + (idx - static_cast<double>(lo))
                           * (samples[hi] - samples[lo]);
    };

    s.p50_ms = percentile(0.50);
    s.p95_ms = percentile(0.95);
    s.p99_ms = percentile(0.99);

    s.mean_ms = std::accumulate(samples.begin(), samples.end(), 0.0)
                / static_cast<double>(s.count);

    double sq = 0.0;
    for (double v : samples) { double d = v - s.mean_ms; sq += d * d; }
    s.stddev_ms = std::sqrt(sq / static_cast<double>(s.count));

    s.cv_pct = (s.mean_ms > 0.0)
               ? (s.stddev_ms / s.mean_ms * 100.0)
               : 0.0;

    return s;
}

// ---------------------------------------------------------------------------
// BenchmarkLogger
// ---------------------------------------------------------------------------

BenchmarkLogger::BenchmarkLogger(std::size_t capacity)
    : ring_(capacity), head_(0) {
    if (capacity == 0)
        throw std::invalid_argument("BenchmarkLogger: capacity must be > 0");
}

std::uint64_t BenchmarkLogger::now_ns() noexcept {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

void BenchmarkLogger::record(BenchPhase phase, std::uint32_t step_index,
                              std::uint64_t duration_ns,
                              std::uint64_t metadata) noexcept {
    const std::size_t cap  = ring_.size();
    // Atomic claim: overwrites oldest entry on wrap (best-effort ring)
    const std::size_t slot = head_.fetch_add(1, std::memory_order_relaxed) % cap;

    BenchEvent& ev  = ring_[slot];
    ev.timestamp_ns = now_ns();
    ev.phase        = phase;
    ev.step_index   = step_index;
    ev.duration_ns  = duration_ns;
    ev.metadata     = metadata;
}

LatencyStats BenchmarkLogger::stats_for_phase(BenchPhase target) const {
    const std::size_t total = std::min(
        head_.load(std::memory_order_acquire), ring_.size());

    std::vector<std::uint64_t> durs;
    durs.reserve(64);

    for (std::size_t i = 0; i < total; ++i) {
        const BenchEvent& ev = ring_[i];
        if (ev.phase == target && ev.duration_ns > 0)
            durs.push_back(ev.duration_ns);
    }

    return LatencyStats::from_ns(std::move(durs));
}

std::expected<void, std::error_code>
BenchmarkLogger::export_json(const std::filesystem::path& path) const {
    std::ofstream ofs(path);
    if (!ofs) {
        return std::unexpected{
            std::make_error_code(std::errc::no_such_file_or_directory)};
    }

    const std::size_t total = std::min(
        head_.load(std::memory_order_acquire), ring_.size());

    ofs << "[\n";
    for (std::size_t i = 0; i < total; ++i) {
        const BenchEvent& ev = ring_[i];
        ofs << std::format(
            "  {{\"seq\":{},\"phase\":\"{}\",\"step\":{}"
            ",\"ts_ns\":{},\"dur_ns\":{},\"dur_ms\":{:.4f},\"meta\":{}}}",
            i,
            bench_phase_name(ev.phase),
            ev.step_index,
            ev.timestamp_ns,
            ev.duration_ns,
            static_cast<double>(ev.duration_ns) / 1.0e6,
            ev.metadata);
        if (i + 1 < total) ofs << ",";
        ofs << "\n";
    }
    ofs << "]\n";
    if (!ofs.good()) {
        return std::unexpected{
            std::make_error_code(std::errc::io_error)};
    }
    return {};
}

std::expected<void, std::error_code>
BenchmarkLogger::export_csv(const std::filesystem::path& path) const {
    std::ofstream ofs(path);
    if (!ofs) {
        return std::unexpected{
            std::make_error_code(std::errc::no_such_file_or_directory)};
    }

    ofs << "seq,phase,step_index,timestamp_ns,duration_ns,duration_ms,metadata\n";

    const std::size_t total = std::min(
        head_.load(std::memory_order_acquire), ring_.size());

    for (std::size_t i = 0; i < total; ++i) {
        const BenchEvent& ev = ring_[i];
        ofs << std::format("{},{},{},{},{},{:.4f},{}\n",
            i,
            bench_phase_name(ev.phase),
            ev.step_index,
            ev.timestamp_ns,
            ev.duration_ns,
            static_cast<double>(ev.duration_ns) / 1.0e6,
            ev.metadata);
    }
    if (!ofs.good()) {
        return std::unexpected{
            std::make_error_code(std::errc::io_error)};
    }
    return {};
}

std::expected<void, std::error_code>
BenchmarkLogger::export_markdown(const std::filesystem::path& path) const {
    std::ofstream ofs(path);
    if (!ofs) {
        return std::unexpected{
            std::make_error_code(std::errc::no_such_file_or_directory)};
    }

    const std::size_t total = std::min(
        head_.load(std::memory_order_acquire), ring_.size());

    ofs << "# Cerberus Benchmark Events\n\n";
    ofs << "| Seq | Phase | Step | Duration (ms) | Metadata |\n";
    ofs << "|-----|-------|------|--------------|----------|\n";

    for (std::size_t i = 0; i < total; ++i) {
        const BenchEvent& ev = ring_[i];
        ofs << std::format("| {} | {} | {} | {:.4f} | {} |\n",
            i,
            bench_phase_name(ev.phase),
            ev.step_index,
            static_cast<double>(ev.duration_ns) / 1.0e6,
            ev.metadata);
    }

    // Summary statistics for ITER_END phase
    const auto s = stats_for_phase(BenchPhase::ITER_END);
    if (s.count > 0) {
        ofs << "\n## ITER_END Statistics\n\n";
        ofs << "| Metric | Value |\n";
        ofs << "|--------|-------|\n";
        ofs << std::format("| Count    | {} |\n",   s.count);
        ofs << std::format("| P50 ms   | {:.4f} |\n", s.p50_ms);
        ofs << std::format("| P95 ms   | {:.4f} |\n", s.p95_ms);
        ofs << std::format("| P99 ms   | {:.4f} |\n", s.p99_ms);
        ofs << std::format("| Mean ms  | {:.4f} |\n", s.mean_ms);
        ofs << std::format("| Stddev ms| {:.4f} |\n", s.stddev_ms);
        ofs << std::format("| Min ms   | {:.4f} |\n", s.min_ms);
        ofs << std::format("| Max ms   | {:.4f} |\n", s.max_ms);
        ofs << std::format("| CV%      | {:.2f} |\n", s.cv_pct);
    }

    if (!ofs.good()) {
        return std::unexpected{
            std::make_error_code(std::errc::io_error)};
    }
    return {};
}

double BenchmarkLogger::measure_overhead_ns(std::size_t probe_count) noexcept {
    if (probe_count == 0) return 0.0;
    const auto t0 = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < probe_count; ++i)
        record(BenchPhase::OVERHEAD_PROBE, 0, 0, i);
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::nano>(t1 - t0).count()
           / static_cast<double>(probe_count);
}

void BenchmarkLogger::clear() noexcept {
    head_.store(0, std::memory_order_release);
}

std::size_t BenchmarkLogger::event_count() const noexcept {
    return std::min(head_.load(std::memory_order_acquire), ring_.size());
}

} // namespace hq
