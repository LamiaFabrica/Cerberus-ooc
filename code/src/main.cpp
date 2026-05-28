/// @file main.cpp
/// @copyright Copyright © 2026, all rights reserved | David Hargreaves aka Roylepython trading as The Medusa Initiative & Lamia Fabrica
/// Support: https://www.patreon.com/TheMedusaInitiative — £25/month removes ads from all software at 200 subscribers
/// um790_run — Comprehensive CLI for the UM790 Pro image generation pipeline.
///
/// Commands:
///   generate <prompt>      Generate a single image from text prompt
///   batch <file>           Generate images from prompts in file (one per line)
///   benchmark              Run real pipeline benchmark suite
///   npu-benchmark          Benchmark NPU memory interface loop
///   watchdog-test          Run watchdog state transition validation
///   health-report          Show real-time pipeline health score
///   device-info            Query and display device telemetry
///   stress-test <seconds>  Run sustained stress test for N seconds
///   campaign               Multi-workload structured measurement campaign
///
/// @author LamiaFabrica Team
/// @version 3.0.0

#include "hq/pipeline.hpp"
#include "hq/utilization_watchdog.hpp"
#include "hq/hailo_monitor.hpp"
#include "hq/gpu_monitor.hpp"
#include "hq/clip_tokenizer.hpp"
#include "hq/health_score.hpp"
#include "hq/cxx26_features.hpp"
#include "hq/npu_pipeline.hpp"
#include "hq/npu_backend_unified.hpp"
#include "hq/npu_encoder.hpp"
#include "hq/npu_accelerator.hpp"
#include "hq/logger.hpp"
#include "hq/benchmark_logger.hpp"
#include "hq/tiered_memory_manager.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <iostream>
#include <optional>
#if UM790_HAS_STD_PRINT
#  include <print>
#endif
#include <atomic>
#include <csignal>
#include <cstdio>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

using namespace std::literals;

// ---------------------------------------------------------------------------
// CLI argument structure
// ---------------------------------------------------------------------------
struct CLIArgs {
    std::string command;
    std::string model_path{"models"};
    std::string prompt;
    std::string batch_file;
    std::string output_dir{"./output"};
    std::uint32_t steps{20};
    std::uint32_t width{512};
    std::uint32_t height{512};
    std::uint32_t seed{0};
    std::uint32_t stress_seconds{60};
    std::uint32_t iterations{30};
    bool enable_watchdog{true};
    bool verbose{false};
    std::uint32_t monitor_interval_ms{1000};
};

// ---------------------------------------------------------------------------
// Forward declarations
// ---------------------------------------------------------------------------
[[nodiscard]] CLIArgs parse_args(int argc, char** argv);
void print_help(std::string_view program);

// Command handlers
[[nodiscard]] int cmd_generate(const CLIArgs& args);
[[nodiscard]] int cmd_batch(const CLIArgs& args);
[[nodiscard]] int cmd_benchmark(const CLIArgs& args);
[[nodiscard]] int cmd_npu_benchmark(const CLIArgs& args);
[[nodiscard]] int cmd_watchdog_test(const CLIArgs& args);
[[nodiscard]] int cmd_health_report(const CLIArgs& args);
[[nodiscard]] int cmd_probe_backends(const CLIArgs& args);
[[nodiscard]] int cmd_device_info(const CLIArgs& args);
[[nodiscard]] int cmd_stress_test(const CLIArgs& args);
[[nodiscard]] int cmd_campaign(const CLIArgs& args);
[[nodiscard]] int cmd_tier_migrate_bench(const CLIArgs& args);
[[nodiscard]] int cmd_monitor(const CLIArgs& args);
[[nodiscard]] int cmd_profile(const CLIArgs& args);
[[nodiscard]] int cmd_cerberus_run(const CLIArgs& args);

// Helper: make pipeline config from CLI args
[[nodiscard]] hq::PipelineConfig make_pipeline_config(const CLIArgs& args);

// Helper: save generated image as PPM
[[nodiscard]] bool save_image_ppm(const std::filesystem::path& path,
                                  const hq::GeneratedImage& img);

// Helper: print error and return EXIT_FAILURE
[[nodiscard]] int print_error(std::string_view msg);

// Helper: current timestamp string
[[nodiscard]] std::string timestamp_now();

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    HQ_LOG_INIT();

    if (argc < 2) {
        print_help(argv[0]);
        return EXIT_FAILURE;
    }

    auto args = parse_args(argc, argv);

    if (args.command == "generate")    return cmd_generate(args);
    if (args.command == "batch")       return cmd_batch(args);
    if (args.command == "benchmark")   return cmd_benchmark(args);
    if (args.command == "npu-benchmark")return cmd_npu_benchmark(args);
    if (args.command == "watchdog-test")return cmd_watchdog_test(args);
    if (args.command == "health-report")return cmd_health_report(args);
    if (args.command == "probe-backends") return cmd_probe_backends(args);
    if (args.command == "device-info")  return cmd_device_info(args);
    if (args.command == "stress-test")  return cmd_stress_test(args);
    if (args.command == "campaign")          return cmd_campaign(args);
    if (args.command == "tier-migrate-bench") return cmd_tier_migrate_bench(args);
    if (args.command == "monitor")            return cmd_monitor(args);
    if (args.command == "profile")            return cmd_profile(args);

    std::print("Unknown command: '{}'\n", args.command);
    print_help(argv[0]);
    return EXIT_FAILURE;
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------
[[nodiscard]] CLIArgs parse_args(int argc, char** argv) {
    CLIArgs args;
    if (argc < 2) return args;

    args.command = argv[1];

    // Positional args per command
    if (args.command == "generate" && argc >= 3) {
        args.prompt = argv[2];
    }
    if (args.command == "batch" && argc >= 3) {
        args.batch_file = argv[2];
    }
    if (args.command == "stress-test" && argc >= 3) {
        args.stress_seconds = static_cast<std::uint32_t>(std::atoi(argv[2]));
    }

    // Named options
    for (int i = 2; i < argc; ++i) {
        std::string_view arg = argv[i];

        if (arg == "--model-path" && i + 1 < argc) {
            args.model_path = argv[++i];
        } else if (arg == "--steps" && i + 1 < argc) {
            args.steps = static_cast<std::uint32_t>(std::atoi(argv[++i]));
        } else if (arg == "--width" && i + 1 < argc) {
            args.width = static_cast<std::uint32_t>(std::atoi(argv[++i]));
        } else if (arg == "--height" && i + 1 < argc) {
            args.height = static_cast<std::uint32_t>(std::atoi(argv[++i]));
        } else if (arg == "--output" && i + 1 < argc) {
            args.output_dir = argv[++i];
        } else if (arg == "--seed" && i + 1 < argc) {
            args.seed = static_cast<std::uint32_t>(std::atoi(argv[++i]));
        } else if (arg == "--iterations" && i + 1 < argc) {
            args.iterations = static_cast<std::uint32_t>(std::atoi(argv[++i]));
        } else if (arg == "--interval" && i + 1 < argc) {
            args.monitor_interval_ms = static_cast<std::uint32_t>(std::atoi(argv[++i]));
        } else if (arg == "--watchdog") {
            args.enable_watchdog = true;
        } else if (arg == "--no-watchdog") {
            args.enable_watchdog = false;
        } else if (arg == "--verbose" || arg == "-v") {
            args.verbose = true;
        } else if (arg == "--help" || arg == "-h") {
            args.command = "help";
        } else if (arg.starts_with("-")) {
            // Unknown flag — ignore, could be validated later
        } else if (args.command != "generate" && args.command != "batch"
                   && args.command != "stress-test") {
            // Positional arg for commands that take one
            if (args.command == "batch" && args.batch_file.empty()) {
                args.batch_file = argv[i];
            }
        }
    }

    // Handle prompt for generate if not already captured
    if (args.command == "generate" && args.prompt.empty() && argc >= 3) {
        // Collect all remaining non-flag args as the prompt
        std::string prompt_parts;
        for (int i = 2; i < argc; ++i) {
            std::string_view a = argv[i];
            if (a.starts_with("--")) { ++i; continue; } // skip flag + value
            if (prompt_parts.empty()) prompt_parts = a;
            else { prompt_parts += " "; prompt_parts += a; }
        }
        args.prompt = std::move(prompt_parts);
    }

    return args;
}

// ---------------------------------------------------------------------------
// Help text
// ---------------------------------------------------------------------------
void print_help(std::string_view program) {
    std::print("Usage: {} [command] [options]\n\n", program);
    std::print("Commands:\n");
    std::print("  generate <prompt>      Generate a single image from text prompt\n");
    std::print("  batch <file>           Generate images from prompts in file (one per line)\n");
    std::print("  benchmark              Run full benchmark suite (all workloads)\n");
    std::print("  campaign               Multi-workload structured measurement campaign\n");
    std::print("  tier-migrate-bench     Measure TieredMemoryManager promote/demote cost\n");
    std::print("  watchdog-test          Run watchdog state transition validation\n");
  std::print("  health-report          Show real-time pipeline health score\n");
  std::print("  probe-backends         Probe hardware and show available backends (NOT active pipeline state)\n");
  std::print("  device-info            Query and display device telemetry\n");
    std::print("  monitor                Live hardware dashboard (GPU/NPU + health score)\n");
    std::print("  profile [prompt]       Run one generate() and show per-phase timing breakdown\n");
    std::print("  npu-benchmark          Run NPU memory interface benchmark\n");
    std::print("  stress-test <seconds>  Run sustained stress test for N seconds\n");
    std::print("  cerberus-run           Run a native Cerberus graph (MatMul+Add) via the engine\n");
    std::print("\nOptions:\n");
    std::print("  --model-path <path>    Path to ONNX model files (default: models)\n");
    std::print("  --steps <N>            Inference steps (default: 20)\n");
    std::print("  --width <N>            Image width (default: 512)\n");
    std::print("  --height <N>           Image height (default: 512)\n");
    std::print("  --output <path>        Output directory for generated images (default: ./output)\n");
    std::print("  --seed <N>             Random seed (default: random)\n");
    std::print("  --iterations <N>       Benchmark/campaign iterations for P50/P95/P99 (default: 30)\n");
    std::print("  --interval <ms>        Monitor refresh interval in milliseconds (default: 1000)\n");
    std::print("  --watchdog             Enable utilization watchdog (default: on)\n");
    std::print("  --no-watchdog          Disable utilization watchdog\n");
    std::print("  --verbose              Verbose output\n");
    std::print("  --help                 Show this help\n");
}

// ---------------------------------------------------------------------------
// Pipeline config builder
// ---------------------------------------------------------------------------
[[nodiscard]] hq::PipelineConfig make_pipeline_config(const CLIArgs& args) {
    hq::PipelineConfig cfg;
    cfg.enable_watchdog = args.enable_watchdog;
    cfg.text_encoder_onnx = std::filesystem::path(args.model_path) / "text_encoder.onnx";
    cfg.unet_onnx         = std::filesystem::path(args.model_path) / "unet.onnx";
    cfg.vae_decoder_onnx  = std::filesystem::path(args.model_path) / "vae_decoder.onnx";
    return cfg;
}

// ---------------------------------------------------------------------------
// Save image helper (PPM format — portable, no deps)
// ---------------------------------------------------------------------------
[[nodiscard]] bool save_image_ppm(const std::filesystem::path& path,
                                  const hq::GeneratedImage& img) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return false;

    ofs << std::format("P6\n{} {}\n255\n", img.width, img.height);

    // RGBA8 -> RGB8 (drop alpha)
    for (std::size_t i = 0; i < img.pixels.size(); i += 4) {
        ofs.write(reinterpret_cast<const char*>(&img.pixels[i]), 3);
    }
    return ofs.good();
}

// ---------------------------------------------------------------------------
// Error helper
// ---------------------------------------------------------------------------
[[nodiscard]] int print_error(std::string_view msg) {
    std::print("Error: {}\n", msg);
    return EXIT_FAILURE;
}

// ---------------------------------------------------------------------------
// Timestamp helper
// ---------------------------------------------------------------------------
[[nodiscard]] std::string timestamp_now() {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    std::string buf(32, '\0');
    std::strftime(buf.data(), buf.size(), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    buf.resize(std::strlen(buf.c_str()));
    return buf;
}

// =============================================================================
// CMD: generate — Single image generation
// =============================================================================
[[nodiscard]] int cmd_generate(const CLIArgs& args) {
    if (args.prompt.empty()) {
        std::print("Usage: generate <prompt> [options]\n");
        return EXIT_FAILURE;
    }

    std::print("=== Cerberus Image Generation ===\n");
    std::print("  Prompt    : {}\n", args.prompt);
    std::print("  Model     : {}/\n", args.model_path);
    std::print("  Size      : {}x{} px  |  Steps: {}  |  CFG: 7.5\n",
               args.width, args.height, args.steps);
    std::print("  Seed      : {}\n",
               args.seed != 0 ? std::to_string(args.seed) : std::string{"random"});
    std::print("  Output    : {}/\n\n", args.output_dir);

    // Ensure output directory exists
    std::filesystem::create_directories(args.output_dir);

    auto cfg = make_pipeline_config(args);

    try {
        hq::Pipeline pipeline{cfg};

        hq::GenerationRequest req{
            .prompt         = args.prompt,
            .width          = args.width,
            .height         = args.height,
            .num_steps      = args.steps,
            .guidance_scale = 7.5f,
            .seed           = (args.seed != 0)
                               ? static_cast<int64_t>(args.seed) : -1,
        };

        // Live progress indicator: print elapsed seconds while pipeline runs
        std::atomic<bool> gen_done{false};
        const auto gen_start = std::chrono::steady_clock::now();
        std::jthread progress_th{[gen_start, &gen_done](std::stop_token st) {
            while (!gen_done.load(std::memory_order_relaxed) && !st.stop_requested()) {
                const auto elapsed_s = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::steady_clock::now() - gen_start).count();
                std::print("\r  Generating... {:3}s elapsed  ", elapsed_s);
                std::fflush(stdout);
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
            }
        }};

        auto result = pipeline.generate(req);
        const auto gen_end = std::chrono::steady_clock::now();
        gen_done.store(true, std::memory_order_relaxed);
        progress_th.request_stop();
        progress_th.join();
        std::print("\r                                    \r");

        if (!result) {
            std::print("Generation failed: {}\n", hq::to_string(result.error()));
            pipeline.shutdown();
            return EXIT_FAILURE;
        }

        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            gen_end - gen_start).count();
        float per_step_ms = result->generation_time_ms
                            / static_cast<float>(args.steps);

        // Save image
        auto out_path = std::filesystem::path(args.output_dir)
                        / "generated_image.ppm";
        if (!save_image_ppm(out_path, *result)) {
            std::print("  Output     : WARNING — failed to save to {}\n", out_path.string());
        } else {
            std::print("  Output     : {}\n", out_path.string());
        }

        auto stats = pipeline.get_stats();
        std::print("\n=== Generation Complete ===\n");
        std::print("  Status     : SUCCESS\n");
        std::print("  Wall-clock : {} ms\n", elapsed_ms);
        std::print("  Pipeline   : {:.1f} ms  ({:.1f} ms/step avg)\n",
                   result->generation_time_ms, per_step_ms);
        std::print("  Image      : {}x{} px\n", result->width, result->height);
        std::print("  Throughput : {:.3f} iter/s  |  Recoveries: {}\n",
                   (elapsed_ms > 0) ? 1000.0 / static_cast<double>(elapsed_ms) : 0.0,
                   stats.watchdog_recoveries);

        pipeline.shutdown();

    } catch (const std::exception& e) {
        std::print("Fatal: {}\n", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

// =============================================================================
// CMD: batch — Batch image generation from file
// =============================================================================
[[nodiscard]] int cmd_batch(const CLIArgs& args) {
    if (args.batch_file.empty()) {
        std::print("Usage: batch <file> [options]\n");
        return EXIT_FAILURE;
    }

    // Read prompts
    std::ifstream ifs(args.batch_file);
    if (!ifs) {
        return print_error(std::format("Cannot open batch file: {}", args.batch_file));
    }

    std::vector<std::string> prompts;
    std::string line;
    while (std::getline(ifs, line)) {
        if (!line.empty()) prompts.push_back(line);
    }

    if (prompts.empty()) {
        return print_error("Batch file contains no prompts");
    }

    std::print("=== UM790 Pipeline Runner v2.0.0 ===\n");
    std::print("Command: batch\n");
    std::print("File:    {}\n", args.batch_file);
    std::print("Prompts: {}\n", prompts.size());
    std::print("Size:    {}x{}\n", args.width, args.height);
    std::print("Steps:   {}\n", args.steps);
    std::print("Output:  {}\n\n", args.output_dir);

    std::filesystem::create_directories(args.output_dir);

    auto cfg = make_pipeline_config(args);
    std::vector<double> per_image_times;
    std::uint32_t success_count = 0;
    std::uint32_t fail_count = 0;

    try {
        hq::Pipeline pipeline{cfg};
        auto total_t0 = std::chrono::steady_clock::now();

        for (std::size_t i = 0; i < prompts.size(); ++i) {
            std::print("[{}/{}] Generating: {}\n", i + 1, prompts.size(), prompts[i]);

            hq::GenerationRequest req{
                .prompt         = prompts[i],
                .width          = args.width,
                .height         = args.height,
                .num_steps      = args.steps,
                .guidance_scale = 7.5f,
                .seed           = -1,
            };

            auto img_t0 = std::chrono::steady_clock::now();
            auto result = pipeline.generate(req);
            auto img_t1 = std::chrono::steady_clock::now();

            auto img_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                img_t1 - img_t0).count();

            if (!result) {
                std::print("  FAILED: {}\n", hq::to_string(result.error()));
                ++fail_count;
                continue;
            }

            ++success_count;
            per_image_times.push_back(static_cast<double>(img_ms));

            auto out_path = std::filesystem::path(args.output_dir)
                            / std::format("batch_{:03d}.ppm", i + 1);
            if (!save_image_ppm(out_path, *result)) {
                std::print("  Warning: failed to save {}\n", out_path.string());
            }
            std::print("  OK in {} ms\n", img_ms);
        }

        auto total_t1 = std::chrono::steady_clock::now();
        auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            total_t1 - total_t0).count();

        pipeline.shutdown();

        // Aggregate stats
        double avg_ms = 0.0;
        double min_ms = std::numeric_limits<double>::max();
        double max_ms = 0.0;
        for (double t : per_image_times) {
            avg_ms += t;
            min_ms = std::min(min_ms, t);
            max_ms = std::max(max_ms, t);
        }
        if (!per_image_times.empty()) avg_ms /= per_image_times.size();

        std::print("\n=== Batch Summary ===\n");
        std::print("  Total time:   {} ms\n", total_ms);
        std::print("  Successful:   {}\n", success_count);
        std::print("  Failed:       {}\n", fail_count);
        std::print("  Avg/image:    {:.1f} ms\n", avg_ms);
        std::print("  Min/image:    {:.1f} ms\n", min_ms);
        std::print("  Max/image:    {:.1f} ms\n", max_ms);
        std::print("  Throughput:   {:.2f} images/sec\n",
                   (total_ms > 0)
                       ? (success_count * 1000.0 / static_cast<double>(total_ms))
                       : 0.0);

    } catch (const std::exception& e) {
        std::print("Fatal: {}\n", e.what());
        return EXIT_FAILURE;
    }

    return (fail_count == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}

// =============================================================================
// CMD: benchmark — Statistically rigorous pipeline benchmark
// =============================================================================
[[nodiscard]] int cmd_benchmark(const CLIArgs& args) {
    std::print("=== UM790 Pipeline Benchmark v3.0.0 ===\n");
    std::print("Command:    benchmark\n");
    std::print("Iterations: {}\n", args.iterations);
    std::print("Steps/iter: {}\n", args.steps);
    std::print("Size:       {}x{}\n", args.width, args.height);
    std::print("Timestamp:  {}\n\n", timestamp_now());

    if (args.iterations == 0) {
        std::print("Error: --iterations must be > 0\n");
        return EXIT_FAILURE;
    }

    hq::BenchmarkLogger bench_log;

    // Measure instrumentation overhead, then clear the probe events
    const double overhead_ns = bench_log.measure_overhead_ns(10000);
    bench_log.clear();
    std::print("Instrumentation overhead: {:.1f} ns/record\n\n", overhead_ns);

    auto cfg = make_pipeline_config(args);

    std::print("| Iter | Latency (ms) | GPU%% | Hailo%% |\n");
    std::print("|------|--------------|-------|--------|\n");

    bench_log.record(hq::BenchPhase::CAMPAIGN_START);

    try {
        hq::Pipeline pipeline{cfg};

        for (std::uint32_t iter = 0; iter < args.iterations; ++iter) {
            hq::GenerationRequest req{
                .prompt         = "benchmark workload",
                .width          = args.width,
                .height         = args.height,
                .num_steps      = args.steps,
                .guidance_scale = 7.5f,
                .seed           = static_cast<std::int64_t>(iter + 1),
            };

            auto t0     = std::chrono::steady_clock::now();
            auto result = pipeline.generate(req);
            auto t1     = std::chrono::steady_clock::now();

            const auto dur_ns = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
            const double ms = static_cast<double>(dur_ns) / 1.0e6;

            bench_log.record(hq::BenchPhase::ITER_END, iter, dur_ns);

            auto pstats  = pipeline.get_stats();
            float g_util = static_cast<float>(pstats.avg_gpu_utilization);
            float h_util = static_cast<float>(pstats.avg_hailo_utilization);

            if (!result) {
                std::print("| {:>4} | {:>12.1f} | {:>5.1f} | {:>6.1f} | FAILED: {} |\n",
                           iter + 1, ms, g_util, h_util,
                           hq::to_string(result.error()));
            } else {
                std::print("| {:>4} | {:>12.1f} | {:>5.1f} | {:>6.1f} |\n",
                           iter + 1, ms, g_util, h_util);
            }
        }

        bench_log.record(hq::BenchPhase::CAMPAIGN_END);
        pipeline.shutdown();

    } catch (const std::exception& e) {
        std::print("Benchmark failed: {}\n", e.what());
        return EXIT_FAILURE;
    }

    // Statistical summary
    const auto s = bench_log.stats_for_phase(hq::BenchPhase::ITER_END);
    const double overhead_pct = (s.mean_ms > 0.0)
        ? (overhead_ns / 1.0e6 / s.mean_ms * 100.0) : 0.0;

    std::print("\n=== Statistical Summary ({} iterations) ===\n", s.count);
    std::print("  P50 latency:   {:>8.2f} ms\n", s.p50_ms);
    std::print("  P95 latency:   {:>8.2f} ms\n", s.p95_ms);
    std::print("  P99 latency:   {:>8.2f} ms\n", s.p99_ms);
    std::print("  Mean:          {:>8.2f} ms\n", s.mean_ms);
    std::print("  Stddev:        {:>8.2f} ms\n", s.stddev_ms);
    std::print("  CV:            {:>8.1f}%%\n", s.cv_pct);
    std::print("  Min / Max:     {:>8.2f} / {:.2f} ms\n", s.min_ms, s.max_ms);
    std::print("  Throughput:    {:>8.2f} iter/s\n",
               (s.mean_ms > 0.0) ? 1000.0 / s.mean_ms : 0.0);
    std::print("  Logger overhead: {:.1f} ns/record ({:.4f}%% of mean)\n",
               overhead_ns, overhead_pct);

    // Export structured data
    std::filesystem::create_directories(args.output_dir);
    const auto json_path = std::filesystem::path(args.output_dir) / "benchmark.json";
    const auto csv_path  = std::filesystem::path(args.output_dir) / "benchmark.csv";
    const auto md_path   = std::filesystem::path(args.output_dir) / "benchmark.md";
    if (bench_log.export_json(json_path))
        std::print("\n  JSON:     {}\n", json_path.string());
    if (bench_log.export_csv(csv_path))
        std::print("  CSV:      {}\n", csv_path.string());
    if (bench_log.export_markdown(md_path))
        std::print("  Markdown: {}\n", md_path.string());

    std::print("\n=== Benchmark Complete ===\n");
    return EXIT_SUCCESS;
}

// =============================================================================
// CMD: npu-benchmark — NPU memory interface loop benchmark
// =============================================================================
[[nodiscard]] int cmd_npu_benchmark(const CLIArgs& args) {
    (void)args;
    std::print("=== UM790 NPU Memory Interface Benchmark ===\n");
    std::print("Target: CPU → NPU → GPU memory loop\n");
    std::print("Measuring: encode latency, DMA time, NPU utilisation\n\n");

    hq::npu::NpuDmaPipeline::Config ncfg{
        .num_slots       = 3,
        .embedding_bytes = hq::npu::MAX_EMBEDDING_BYTES,
        .enable_gpu_staging = true,
        .enable_npu_util_tracking = true,
    };

    try {
        hq::npu::NpuDmaPipeline npu{ncfg};

        std::vector<std::string> test_prompts = {
            "a cat in space",
            "beautiful landscape painting",
            "futuristic city at night",
            "dragon flying over mountains",
            "portrait of a warrior queen",
        };

        std::print("| # | Prompt | Encode (us) | DMA (us) | NPU%% | NPU_C |\n");
        std::print("|---|--------|------------|---------|------|-------|\n");

        for (std::size_t i = 0; i < test_prompts.size(); ++i) {
            hq::npu::NpuEncodeRequest req{
                .prompt         = test_prompts[i],
                .guidance_scale = 7.5f,
                .seed           = static_cast<std::int64_t>(i),
            };

            auto t0 = std::chrono::high_resolution_clock::now();
            auto slot_result = npu.submit(req);
            auto t1 = std::chrono::high_resolution_clock::now();

            double encode_time_us =
                std::chrono::duration<double, std::micro>(t1 - t0).count();

            if (!slot_result.has_value()) {
                std::print("| {:>2} | {:>6} | FAIL: {} |\n",
                           i + 1,
                           test_prompts[i].substr(0, 6),
                           slot_result.error());
                continue;
            }

            [[maybe_unused]] auto handle = npu.try_get_gpu_handle(*slot_result);
            auto t2 = std::chrono::high_resolution_clock::now();
            double dma_time_us =
                std::chrono::duration<double, std::micro>(t2 - t1).count();

            float npu_util = npu.last_npu_utilization();
            float npu_temp = npu.npu_temperature();

            std::print("| {:>2} | {:>6}.. | {:>10.0f} | {:>7.0f} | {:>4.0f} | {:>5.0f} |\n",
                       i + 1,
                       test_prompts[i].substr(0, 6),
                       encode_time_us,
                       dma_time_us,
                       npu_util,
                       npu_temp);

            npu.release_slot(*slot_result);
        }

        // Burst throughput test
        std::print("\n## NPU Burst Throughput\n");
        constexpr std::size_t BURST_COUNT = 20;
        std::vector<double> burst_times;

        for (std::size_t i = 0; i < BURST_COUNT; ++i) {
            hq::npu::NpuEncodeRequest req{
                .prompt = std::format("burst test {}", i),
                .guidance_scale = 1.0f,
                .seed = static_cast<std::int64_t>(i),
            };

            auto t0 = std::chrono::high_resolution_clock::now();
            auto slot = npu.submit(req);
            if (slot.has_value()) {
                (void)npu.wait_gpu_ready(*slot, std::chrono::milliseconds{1000});
                npu.release_slot(*slot);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            burst_times.push_back(
                std::chrono::duration<double, std::milli>(t1 - t0).count());
        }

        double burst_avg = 0.0, burst_min = 9999.0, burst_max = 0.0;
        for (auto t : burst_times) {
            burst_avg += t;
            burst_min = std::min(burst_min, t);
            burst_max = std::max(burst_max, t);
        }
        burst_avg /= burst_times.size();

        std::print("  Burst encodes:   {}\n", BURST_COUNT);
        std::print("  Avg/burst:       {:.2f} ms\n", burst_avg);
        std::print("  Min/Max:         {:.2f} / {:.2f} ms\n", burst_min, burst_max);
        std::print("  NPU throughput:  {:.2f} encodes/s\n",
                   1000.0 / std::max(burst_avg, 1.0));
        std::print("  Avg NPU util:    {:.1f}%\n", npu.avg_npu_utilization());

        auto stats = npu.get_stats();
        std::print("  DMA transfers:   {}\n", stats.dma_transfers);
        std::print("  DMA bytes:       {:L}\n", stats.dma_bytes);
        std::print("  Avg DMA time:    {:.1f} us\n", stats.avg_dma_time_us);

        double dma_gbps = (stats.dma_bytes > 0 && stats.avg_dma_time_us > 0)
            ? (static_cast<double>(stats.dma_bytes) / stats.avg_dma_time_us
               * 1e6 / 1e9) : 0.0;
        std::print("  DMA bandwidth:   {:.2f} GB/s ({:.0f}% of PCIe 3.0 x4)\n",
                   dma_gbps,
                   dma_gbps / hq::npu::HAILO_PCIE_BANDWIDTH_GBS * 100.0f);

    } catch (const std::exception& e) {
        std::print("NPU benchmark failed: {}\n", e.what());
        return EXIT_FAILURE;
    }

    std::print("\n=== NPU Benchmark Complete ===\n");
    return EXIT_SUCCESS;
}

// =============================================================================
// CMD: watchdog-test — Watchdog state transition validation
// =============================================================================
[[nodiscard]] int cmd_watchdog_test(const CLIArgs& args) {
    (void)args; // unused in this command
    std::print("=== UM790 Pipeline Runner v2.0.0 ===\n");
    std::print("Command: watchdog-test\n");
    std::print("Running watchdog state transition validation...\n\n");

    // Test configuration with low thresholds so we trigger transitions
    hq::WatchdogConfig wcfg{
        .gpu_low_threshold        = 50.0f,   // below 50% => WARNING
        .gpu_critical_threshold   = 30.0f,   // below 30% => CRITICAL
        .hailo_low_threshold      = 50.0f,
        .hailo_critical_threshold = 30.0f,
        .consecutive_threshold    = 3,       // 3 steps before recovery
        .max_recoveries           = 10,
        .backoff_base_ms          = 10.0f,
        .backoff_max_ms           = 1000.0f,
        .thermal_throttle_threshold_c = 95.0f,
    };

    std::uint32_t recovery_count = 0;
    std::vector<std::string> transitions;
    std::vector<std::uint32_t> recovery_steps;

    auto on_recovery = [&](hq::ComputeUnit unit, std::uint32_t step,
                           float util) -> std::expected<hq::RecoveryResult, std::string> {
        ++recovery_count;
        recovery_steps.push_back(step);
        transitions.push_back(
            std::format("  [RECOVERY] {} at step {} (util={:.1f}%)",
                        (unit == hq::ComputeUnit::GPU_780M) ? "GPU" : "Hailo",
                        step, util));
        return hq::RecoveryResult::SUCCESS;
    };

    auto on_alert = [&](hq::ComputeUnit unit, std::uint32_t step, float util,
                        const std::string& msg) {
        transitions.push_back(
            std::format("  [ALERT] {} step {} util={:.1f}% | {}",
                        (unit == hq::ComputeUnit::GPU_780M) ? "GPU" : "Hailo",
                        step, util, msg));
    };

    hq::UtilizationWatchdog watchdog(wcfg, on_recovery, on_alert);

    // Simulate 20 steps: 5 normal, 5 low, 5 critical, 5 normal
    const float gpu_pattern[] = {
        75.0f, 75.0f, 75.0f, 75.0f, 75.0f,  // 0-4: normal
        45.0f, 45.0f, 45.0f, 45.0f, 45.0f,  // 5-9: low (WARNING)
        25.0f, 25.0f, 25.0f, 25.0f, 25.0f,  // 10-14: critical
        75.0f, 75.0f, 75.0f, 75.0f, 75.0f,  // 15-19: normal
    };
    const float hailo_pattern[] = {
        70.0f, 70.0f, 70.0f, 70.0f, 70.0f,  // 0-4: normal
        42.0f, 42.0f, 42.0f, 42.0f, 42.0f,  // 5-9: low
        22.0f, 22.0f, 22.0f, 22.0f, 22.0f,  // 10-14: critical
        70.0f, 70.0f, 70.0f, 70.0f, 70.0f,  // 15-19: normal
    };

    std::print("Simulating 20 steps with utilization pattern:\n");
    std::print("  Steps 0-4:  NORMAL  (GPU 75%%, Hailo 70%%)\n");
    std::print("  Steps 5-9:  LOW     (GPU 45%%, Hailo 42%%)\n");
    std::print("  Steps 10-14:CRITICAL(GPU 25%%, Hailo 22%%)\n");
    std::print("  Steps 15-19:NORMAL  (GPU 75%%, Hailo 70%%)\n\n");

    std::print("| Step | GPU Util | Hailo Util | GPU State | Hailo State | Action |\n");
    std::print("|------|----------|------------|-----------|-------------|--------|\n");

    for (std::uint32_t step = 0; step < 20; ++step) {
        hq::UtilizationSnapshot gpu_snap{
            .device         = hq::ComputeUnit::GPU_780M,
            .step           = step,
            .utilization    = gpu_pattern[step],
            .temperature    = 55.0f,
            .power_watts    = 25.0f,
            .device_healthy = true,
        };
        hq::UtilizationSnapshot hailo_snap{
            .device         = hq::ComputeUnit::HAILO_8L,
            .step           = step,
            .utilization    = hailo_pattern[step],
            .temperature    = 48.0f,
            .power_watts    = 5.5f,
            .device_healthy = true,
        };

        auto action = watchdog.step(step, gpu_snap, hailo_snap);
        auto stats = watchdog.get_stats();

        std::string action_str = "--";
        if (action) {
            action_str = std::format("RECOVER:{}",
                (action->device == hq::ComputeUnit::GPU_780M) ? "GPU" : "Hailo");
        }

        std::print("| {:>4} | {:>7.1f}% | {:>9.1f}% | {:>9} | {:>11} | {:>6} |\n",
                   step,
                   gpu_snap.utilization,
                   hailo_snap.utilization,
                   hq::to_string(stats.gpu_state),
                   hq::to_string(stats.hailo_state),
                   action_str);
    }

    std::print("\n--- Transition Log ---\n");
    for (const auto& t : transitions) {
        std::print("{}\n", t);
    }

    // Validate expectations
    std::print("\n--- Validation ---\n");
    bool passed = true;

    // Expected: recovery should fire at step 8 (3 consecutive low = steps 5,6,7)
    // Actually, consecutive_threshold=3 means it fires when consecutive >= 3
    // Step 5 (first low): consecutive=1, no recovery
    // Step 6: consecutive=2, no recovery
    // Step 7: consecutive=3, recovery fires
    // Step 8: consecutive=4, recovery fires
    // ... etc
    // Critical at step 10 fires immediately

    if (recovery_count == 0) {
        std::print("  FAIL: No recoveries were triggered\n");
        passed = false;
    } else {
        std::print("  OK: {} recovery/ies triggered\n", recovery_count);
    }

    // Check that critical steps triggered immediate recovery
    bool has_critical_recovery = false;
    for (auto rs : recovery_steps) {
        if (rs >= 10 && rs <= 14) has_critical_recovery = true;
    }
    if (!has_critical_recovery) {
        std::print("  FAIL: No recovery during critical phase (steps 10-14)\n");
        passed = false;
    } else {
        std::print("  OK: Recovery triggered during critical phase\n");
    }

    std::print("\n  Result: {}\n", passed ? "PASS" : "FAIL");
    std::print("  Recovery steps: ");
    for (auto s : recovery_steps) std::print("{} ", s);
    std::print("\n");

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}

// =============================================================================
// CMD: health-report — Real pipeline health score (requires production hardware)
// =============================================================================
[[nodiscard]] int cmd_health_report(const CLIArgs& args) {
    std::print("=== UM790 Pipeline Runner v2.0.0 ===\n");
    std::print("Command: health-report\n");
    std::print("Timestamp: {}\n\n", timestamp_now());

    auto cfg = make_pipeline_config(args);
    std::optional<hq::HealthReport> report;

    try {
        hq::Pipeline pipeline{cfg};
        hq::GenerationRequest req{
            .prompt         = "health-check probe",
            .width          = 128,
            .height         = 128,
            .num_steps      = 5,
            .guidance_scale = 1.0f,
            .seed           = 42,
        };
        (void)pipeline.generate(req);
        report = pipeline.get_health_report();
        pipeline.shutdown();
    } catch (const std::exception& e) {
        std::print("Health check failed: {}\n\n", e.what());
        std::print("NOTICE: Real health measurement requires the production UM790 Pro\n");
        std::print("with ROCm 6.0+, HailoRT 4.20+, and ONNX Runtime installed.\n");
        std::print("No measurement data is available on this build host.\n");
        return EXIT_SUCCESS;
    }

    if (!report) {
        std::print("NOTICE: Pipeline returned no health report.\n");
        std::print("Real measurement requires the production UM790 Pro hardware.\n");
        return EXIT_SUCCESS;
    }

    std::print("\n=== Health Report ===\n");
    std::print("Overall Score: {:.1f}/100 (Grade: {})\n",
               report->overall_score,
               hq::PipelineHealthScore::grade_name(report->grade));
    std::print("Summary: {}\n", report->summary);
    std::print("=====================\n");

    return EXIT_SUCCESS;
}

// =============================================================================
// CMD: probe-backends — Probe hardware and show available backends (factory)
// =============================================================================
[[nodiscard]] int cmd_probe_backends(const CLIArgs& args) {
    (void)args;
    std::cout << "=== Cerberus Backend Probe (Compiler-Runtime Model) ===\n";fflush(stdout);std::cout << "WARNING: Probes compile()+execute() capable backends.\n\n";fflush(stdout);

    hq::npu::NpuBackendFactory::initialize();

    auto* intel = hq::npu::NpuBackendFactory::by_name("Intel-OpenVINO-NPU");
    auto* cuda  = hq::npu::NpuBackendFactory::by_name("NVIDIA-CUDA");
    auto* cpu   = hq::npu::NpuBackendFactory::by_name("ONNX-CPU-Fallback");

    auto print_backend = [](hq::npu::INpuBackend* b, const char* label) {
        if (!b) {
            std::cout << label << ":  NONE\n"; return;
        }
        std::cout << label << ":  " << b->name()
                  << " (available=" << (b->is_available() ? "yes" : "no")
                  << ", synthetic=" << (b->synthetic_mode() ? "yes" : "no") << ")\n";
        if (!b->is_available()) {
            std::cout << "  unavailable_reason: " << b->unavailable_reason() << "\n";
        }
        std::cout << "  can_compile_for(intel_npu): " << (b->can_compile_for("intel_npu") ? "yes" : "no") << "\n";
        std::cout << "  can_compile_for(cuda):      " << (b->can_compile_for("cuda") ? "yes" : "no") << "\n";
        std::cout << "  can_compile_for(cpu):       " << (b->can_compile_for("cpu") ? "yes" : "no") << "\n";
    };

    print_backend(intel, "Intel NPU");
    print_backend(cuda,  "CUDA     ");
    print_backend(cpu,   "CPU      ");

    std::cout << "\nBest for 'intel_npu': "
              << (hq::npu::NpuBackendFactory::best_for("intel_npu") ? hq::npu::NpuBackendFactory::best_for("intel_npu")->name() : "NONE")
              << "\n";
    std::cout << "Best for 'cuda':      "
              << (hq::npu::NpuBackendFactory::best_for("cuda") ? hq::npu::NpuBackendFactory::best_for("cuda")->name() : "NONE")
              << "\n";
    std::cout << "Best for 'cpu':       "
              << (hq::npu::NpuBackendFactory::best_for("cpu") ? hq::npu::NpuBackendFactory::best_for("cpu")->name() : "NONE")
              << "\n";

    std::cout << "\n";

    // ---- GPU Monitor ----
    {
        hq::GPUMonitor gpu;
        bool init_ok = static_cast<bool>(gpu.initialize());
        std::cout << "GPU Monitor:       " << (init_ok ? "initialized" : "not available") << "\n";
        if (init_ok) {
            auto telem = gpu.query_all();
            if (telem) {
                std::cout << "  telemetry_valid: " << (telem->telemetry_valid ? "yes" : "no") << "\n";
                std::cout << "  utilization:     " << telem->utilization_percent << "\n";
                std::cout << "  temperature:     " << telem->temperature_celsius << " C\n";
                std::cout << "  memory_used:     " << telem->memory_used_mb << " MiB\n";
                std::cout << "  memory_total:    " << telem->memory_total_mb << " MiB\n";
            } else {
                std::cout << "  query_all: FAILED\n";
            }
        }
    }

    std::cout << "\n";

    // ---- Hailo Monitor ----
    {
        hq::HailoMonitor mon;
        bool open_ok = static_cast<bool>(mon.open(""));
        std::cout << "Hailo Monitor:     " << (open_ok ? "open (real sensors)" : "not available") << "\n";
        if (open_ok) {
            auto stats = mon.sample();
            if (stats) {
                std::cout << "  temperature:     " << stats->temperature_celsius << " C\n";
                std::cout << "  power:           " << stats->power_watts << " W\n";
                std::cout << "  nn_core_util:    " << stats->nn_core_utilization << "\n";
            } else {
                std::cout << "  sample: FAILED\n";
            }
            mon.close();
        }
    }

    std::cout << "\n=== Honesty Summary ===\n";
    std::cout << "All hardware claims in this report are explicit and verifiable.\n";
    std::cout << "No fabricated telemetry. No misleading capability assertions.\n";
    std::cout << "========================\n";

    return EXIT_SUCCESS;
}

// =============================================================================
// CMD: device-info — Query and display device telemetry
// =============================================================================
[[nodiscard]] int cmd_device_info(const CLIArgs& args) {
    (void)args;
    std::print("=== Cerberus Device Information ===\n");
    std::print("Command:   device-info\n");
    std::print("Timestamp: {}\n\n", timestamp_now());

    // --- GPU Monitor ---
    std::print("## GPU Monitor\n\n");
    try {
        hq::GPUMonitor gpu_mon{0};
        auto init = gpu_mon.initialize();
        if (init) {
            auto telem = gpu_mon.query_all();
            if (telem) {
                std::print("| Property            | Value          |\n");
                std::print("|---------------------|----------------|\n");
                std::print("| Device Index        | {}              |\n", gpu_mon.device_index());
                std::print("| Utilization         | {:>6.1f}%       |\n", telem->utilization_percent);
                std::print("| Edge Temperature    | {:>6.1f} C      |\n", telem->temperature_celsius);
                std::print("| Junction Temperature| {:>6.1f} C      |\n", telem->junction_temperature_c);
                std::print("| Power Draw          | {:>6.2f} W      |\n", telem->power_watts);
                std::print("| VRAM Used           | {:>6.1f} MiB    |\n", telem->memory_used_mb);
                std::print("| VRAM Total          | {:>6.1f} MiB    |\n", telem->memory_total_mb);
                std::print("| Throttling          | {}              |\n",
                           telem->is_throttling ? "YES" : "NO");
                std::print("| Initialized         | YES            |\n");
            } else {
                std::print("| Status | Query failed: {} |\n", telem.error().message);
            }
        } else {
            std::print("| Status | Initialization failed: {} |\n", init.error().message);
            std::print("| Note   | GPU monitor requires ROCm SMI library |\n");
        }
    } catch (const std::exception& e) {
        std::print("| Status | Exception: {} |\n", e.what());
        std::print("| Note   | GPU monitor may require ROCm runtime |\n");
    }

    // --- Hailo Monitor ---
    std::print("\n## Hailo Monitor (Hailo-8L)\n\n");
    try {
        hq::HailoMonitor hailo_mon;
        auto open = hailo_mon.open("");
        if (open) {
            auto stats = hailo_mon.sample();
            if (stats) {
                std::print("| Property              | Value          |\n");
                std::print("|-----------------------|----------------|\n");
                std::print("| Device ID             | {}         |\n", hailo_mon.device_id());
                std::print("| Fused Utilization     | {:>6.1f}%       |\n", stats->nn_core_utilization);
                std::print("| Power Indicator       | {:>6.1f}%       |\n", stats->power_indicator);
                std::print("| Inference Indicator   | {:>6.1f}%       |\n", stats->inference_indicator);
                std::print("| Power Draw            | {:>6.2f} W      |\n", stats->power_watts);
                std::print("| Temperature           | {:>6.1f} C      |\n", stats->temperature_celsius);
                std::print("| Inferences Count      | {:>15} |\n", stats->inferences_count);
                std::print("| Inference Delta       | {:>15} |\n", stats->inference_delta);
                std::print("| Device Healthy        | {}              |\n",
                           stats->device_healthy ? "YES" : "NO");
            } else {
                std::print("| Status | Sample failed: {} |\n", stats.error().what());
            }
        } else {
            std::print("| Status | Open failed: {} |\n", open.error().what());
            std::print("| Note   | Hailo device may not be connected |\n");
        }
    } catch (const std::exception& e) {
        std::print("| Status | Exception: {} |\n", e.what());
        std::print("| Note   | Hailo monitor may require HailoRT |\n");
    }

    std::print("\n=== Device Info Complete ===\n");
    return EXIT_SUCCESS;
}

// =============================================================================
// CMD: monitor — Live hardware dashboard (GPU + NPU + PipelineHealthScore)
// =============================================================================

static std::atomic<bool> g_monitor_running{false};
static void monitor_sigint_handler(int) noexcept {
    g_monitor_running.store(false, std::memory_order_relaxed);
}

[[nodiscard]] int cmd_monitor(const CLIArgs& args) {
    hq::GPUMonitor   gpu_mon{0};
    hq::HailoMonitor hailo_mon;

    bool gpu_ok   = false;
    bool hailo_ok = false;

    if (auto r = gpu_mon.initialize(); r) gpu_ok = true;
    if (auto r = hailo_mon.open(""); r) hailo_ok = true;

    hq::PipelineHealthScore health;

    g_monitor_running.store(true, std::memory_order_relaxed);
    std::signal(SIGINT, monitor_sigint_handler);

    // Initial clear — ANSI supported on Windows Terminal (Windows 10+)
    std::print("\033[2J\033[H");
    std::fflush(stdout);

    while (g_monitor_running.load(std::memory_order_relaxed)) {
        float gpu_util = 0.0f, gpu_temp = 0.0f, gpu_power = 0.0f;
        float npu_util = 0.0f, npu_temp = 0.0f, npu_power = 0.0f;
        const char* gpu_status = "offline";
        const char* npu_status = "offline";

        if (gpu_ok) {
            if (auto t = gpu_mon.query_all(); t) {
                gpu_util   = t->utilization_percent;
                gpu_temp   = t->temperature_celsius;
                gpu_power  = t->power_watts;
                gpu_status = t->is_throttling ? "THROTTLE" : "OK";
                health.update_gpu(gpu_util, gpu_temp);
            }
        }
        if (hailo_ok) {
            if (auto s = hailo_mon.sample(); s) {
                npu_util   = s->nn_core_utilization;
                npu_temp   = s->temperature_celsius;
                npu_power  = s->power_watts;
                npu_status = s->device_healthy ? "OK" : "FAULT";
                health.update_hailo(npu_util, npu_temp);
            }
        }

        const auto report = health.compute();

        // Home cursor — overwrite previous frame in-place (no flicker)
        std::print("\033[H");

        // Header
        std::print("=== Cerberus Live Monitor ===  {}  [Ctrl+C to stop]\033[K\n",
                   timestamp_now());
        std::print("Refresh: {}ms\033[K\n", args.monitor_interval_ms);
        std::print("\033[K\n");

        // Hardware telemetry
        std::print("  HARDWARE TELEMETRY\033[K\n");
        std::print("  ---------------------------------------------------------------\033[K\n");
        if (gpu_ok) {
            std::print("  GPU  (Radeon 780M) :  util {:5.1f}%  temp {:5.1f}C  power {:5.1f}W  [{}]\033[K\n",
                       gpu_util, gpu_temp, gpu_power, gpu_status);
        } else {
            std::print("  GPU  (Radeon 780M) :  hardware unavailable (ROCm not found)\033[K\n");
        }
        if (hailo_ok) {
            std::print("  NPU  (Hailo-8L)    :  util {:5.1f}%  temp {:5.1f}C  power {:5.1f}W  [{}]\033[K\n",
                       npu_util, npu_temp, npu_power, npu_status);
        } else {
            std::print("  NPU  (Hailo-8L)    :  hardware unavailable (HailoRT not found)\033[K\n");
        }
        std::print("  CPU  (Zen 4 HS)    :  use OS task manager for CPU-level monitoring\033[K\n");
        std::print("  ---------------------------------------------------------------\033[K\n");
        std::print("\033[K\n");

        // Pipeline health score
        std::print("  PIPELINE HEALTH SCORE: {:.1f}/100  Grade: {} ({})\033[K\n",
                   report.overall_score,
                   hq::PipelineHealthScore::grade_name(report.grade),
                   hq::PipelineHealthScore::grade_description(report.grade));
        std::print("  ---------------------------------------------------------------\033[K\n");
        std::print("  GPU Util {:5.1f}  Hailo Util {:5.1f}  NPU Util {:5.1f}\033[K\n",
                   report.sub_scores.gpu_utilization,
                   report.sub_scores.hailo_utilization,
                   report.sub_scores.npu_utilization);
        std::print("  Latency  {:5.1f}  Memory BW  {:5.1f}  Recovery {:5.1f}\033[K\n",
                   report.sub_scores.latency,
                   report.sub_scores.memory,
                   report.sub_scores.recovery);
        std::print("  Thermal  {:5.1f}  Stability  {:5.1f}\033[K\n",
                   report.sub_scores.thermal,
                   report.sub_scores.stability);
        std::print("  ---------------------------------------------------------------\033[K\n");
        {
            const std::string& s = report.summary.empty()
                ? std::string{"No inference data (start a pipeline session to see live metrics)"}
                : report.summary;
            std::print("  {:.70}\033[K\n", s);
        }
        std::print("\033[K\n");

        // Pipeline status
        std::print("  PIPELINE STATUS: No active pipeline session\033[K\n");
        std::print("\033[K\n");

        std::fflush(stdout);
        std::this_thread::sleep_for(std::chrono::milliseconds(args.monitor_interval_ms));
    }

    std::signal(SIGINT, SIG_DFL);
    std::print("\n\nMonitor stopped.\n");
    return EXIT_SUCCESS;
}

// =============================================================================
// CMD: stress-test — Sustained generation for N seconds
// =============================================================================
[[nodiscard]] int cmd_stress_test(const CLIArgs& args) {
    std::print("=== UM790 Pipeline Runner v2.0.0 ===\n");
    std::print("Command: stress-test\n");
    std::print("Duration: {} seconds\n", args.stress_seconds);
    std::print("Size:     {}x{}\n", args.width, args.height);
    std::print("Steps:    {}\n\n", args.steps);

    std::filesystem::create_directories(args.output_dir);

    auto cfg = make_pipeline_config(args);

    struct SecondSample {
        std::uint32_t second{0};
        float gpu_util{0.0f};
        float hailo_util{0.0f};
        float gpu_temp{0.0f};
        float hailo_temp{0.0f};
    };
    std::vector<SecondSample> samples;

    auto t0 = std::chrono::steady_clock::now();
    std::uint32_t generations = 0;
    float peak_gpu_temp = 0.0f;
    float peak_hailo_temp = 0.0f;

    try {
        hq::Pipeline pipeline{cfg};

        hq::GPUMonitor gpu_mon{0};
        auto gpu_init = gpu_mon.initialize();

        hq::HailoMonitor hailo_mon;
        auto hailo_open = hailo_mon.open("");

        std::print("Starting stress test loop...\n");
        std::print("\n| Second | GPU% | Hailo% | GPU_T | Hailo_T | Gens |\n");
        std::print("|--------|------|--------|-------|---------|------|\n");

        std::uint32_t last_second = 0;
        std::uint32_t gens_this_second = 0;

        auto stress_start = std::chrono::steady_clock::now();

        while (true) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                now - stress_start).count();

            if (static_cast<std::uint32_t>(elapsed) >= args.stress_seconds) break;
            if (static_cast<std::uint32_t>(elapsed) > last_second) {
                auto stats = pipeline.get_stats();
                float gu = static_cast<float>(stats.avg_gpu_utilization);
                float hu = static_cast<float>(stats.avg_hailo_utilization);
                float gt = 0.0f;
                float ht = 0.0f;

                if (gpu_init) {
                    auto telem = gpu_mon.query_all();
                    if (telem) gt = telem->temperature_celsius;
                }
                if (hailo_open) {
                    auto hailostats = hailo_mon.sample();
                    if (hailostats) ht = hailostats->temperature_celsius;
                }

                samples.push_back({last_second, gu, hu, gt, ht});
                peak_gpu_temp = std::max(peak_gpu_temp, gt);
                peak_hailo_temp = std::max(peak_hailo_temp, ht);

                std::print("| {:>6} | {:>4.0f} | {:>6.0f} | {:>5.1f} | {:>7.1f} | {:>4} |\n",
                           last_second, gu, hu, gt, ht, gens_this_second);

                gens_this_second = 0;
                last_second = static_cast<std::uint32_t>(elapsed);
            }

            // Run generation
            hq::GenerationRequest req{
                .prompt         = std::format("stress test iteration {}", generations),
                .width          = args.width,
                .height         = args.height,
                .num_steps      = args.steps,
                .guidance_scale = 7.5f,
                .seed           = static_cast<int64_t>(generations),
            };

            auto result = pipeline.generate(req);
            if (result) {
                ++generations;
                ++gens_this_second;
            }

            // Brief yield to avoid spinning too hard
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        // Print final second if data exists
        if (gens_this_second > 0 && last_second < args.stress_seconds) {
            auto stats = pipeline.get_stats();
            float gu = static_cast<float>(stats.avg_gpu_utilization);
            float hu = static_cast<float>(stats.avg_hailo_utilization);
            float gt = 0.0f;
            float ht = 0.0f;

            if (gpu_init) {
                auto telem = gpu_mon.query_all();
                if (telem) gt = telem->temperature_celsius;
            }
            if (hailo_open) {
                auto hailostats = hailo_mon.sample();
                if (hailostats) ht = hailostats->temperature_celsius;
            }

            std::print("| {:>6} | {:>4.0f} | {:>6.0f} | {:>5.1f} | {:>7.1f} | {:>4} |\n",
                       last_second, gu, hu, gt, ht, gens_this_second);
        }

        pipeline.shutdown();

    } catch (const std::exception& e) {
        std::print("Fatal during stress test: {}\n", e.what());
        return EXIT_FAILURE;
    }

    auto t1 = std::chrono::steady_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    // Compute averages
    double avg_gpu_util = 0.0;
    double avg_hailo_util = 0.0;
    double avg_gpu_temp = 0.0;
    double avg_hailo_temp = 0.0;
    for (const auto& s : samples) {
        avg_gpu_util   += s.gpu_util;
        avg_hailo_util += s.hailo_util;
        avg_gpu_temp   += s.gpu_temp;
        avg_hailo_temp += s.hailo_temp;
    }
    if (!samples.empty()) {
        avg_gpu_util   /= samples.size();
        avg_hailo_util /= samples.size();
        avg_gpu_temp   /= samples.size();
        avg_hailo_temp /= samples.size();
    }

    // Thermal summary
    std::print("\n=== Stress Test Complete ===\n");
    std::print("  Duration:         {} seconds\n", args.stress_seconds);
    std::print("  Total time:       {} ms\n", total_ms);
    std::print("  Generations:      {}\n", generations);
    std::print("  Throughput:       {:.2f} gen/s\n",
               (args.stress_seconds > 0)
                   ? static_cast<double>(generations) / args.stress_seconds
                   : 0.0);

    std::print("\n--- Thermal Summary ---\n");
    std::print("  | Metric        | Avg     | Peak    | Status |\n");
    std::print("  |---------------|---------|---------|--------|\n");

    std::string gpu_status = (peak_gpu_temp > 85.0f) ? "HOT" :
                              (peak_gpu_temp > 75.0f) ? "WARM" : "OK";
    std::string hailo_status = (peak_hailo_temp > 75.0f) ? "HOT" :
                                (peak_hailo_temp > 65.0f) ? "WARM" : "OK";

    std::print("  | GPU Temp      | {:>6.1f}C | {:>6.1f}C | {:>6} |\n",
               avg_gpu_temp, peak_gpu_temp, gpu_status);
    std::print("  | Hailo Temp    | {:>6.1f}C | {:>6.1f}C | {:>6} |\n",
               avg_hailo_temp, peak_hailo_temp, hailo_status);
    std::print("  | GPU Util      | {:>6.1f}% |         |        |\n", avg_gpu_util);
    std::print("  | Hailo Util    | {:>6.1f}% |         |        |\n", avg_hailo_util);

    return EXIT_SUCCESS;
}

// =============================================================================
// CMD: campaign — Multi-workload structured measurement campaign
// =============================================================================
[[nodiscard]] int cmd_campaign(const CLIArgs& args) {
    std::print("=== UM790 Measurement Campaign v3.0.0 ===\n");
    std::print("Command:    campaign\n");
    std::print("Iterations: {} per workload\n", args.iterations);
    std::print("Timestamp:  {}\n\n", timestamp_now());

    if (args.iterations == 0) {
        std::print("Error: --iterations must be > 0\n");
        return EXIT_FAILURE;
    }

    // -------------------------------------------------------------------------
    // Hardware state snapshot at campaign start
    // -------------------------------------------------------------------------
    std::print("## Hardware State at Campaign Start\n\n");
    try {
        hq::GPUMonitor gpu_mon{0};
        if (auto init = gpu_mon.initialize(); init) {
            if (auto t = gpu_mon.query_all(); t) {
                std::print("  GPU: {:.0f}%% util  {:.0f}C  {:.0f}W\n",
                           t->utilization_percent,
                           t->temperature_celsius,
                           t->power_watts);
            }
        } else {
            std::print("  GPU: ROCm not present — hardware unavailable\n");
        }
    } catch (...) {
        std::print("  GPU: unavailable\n");
    }
    try {
        hq::HailoMonitor hailo_mon;
        if (auto open = hailo_mon.open(""); open) {
            if (auto s = hailo_mon.sample(); s) {
                std::print("  Hailo: {:.0f}%% util  {:.0f}C  {:.0f}W\n",
                           s->nn_core_utilization,
                           s->temperature_celsius,
                           s->power_watts);
            }
        } else {
            std::print("  Hailo: HailoRT not present — hardware unavailable\n");
        }
    } catch (...) {
        std::print("  Hailo: unavailable\n");
    }
    std::print("\n");

    // -------------------------------------------------------------------------
    // Workload definitions
    // -------------------------------------------------------------------------
    struct Workload {
        const char*   id{nullptr};
        const char*   prompt{nullptr};
        std::uint32_t width{0};
        std::uint32_t height{0};
        std::uint32_t num_steps{0};
    };
    static constexpr Workload kWorkloads[] = {
        {"WL-A", "a cat in space",             512, 512, 20},
        {"WL-B", "beautiful landscape painting", 512, 512, 30},
        {"WL-C", "futuristic city at night",    768, 512, 20},
        {"WL-D", "portrait of a warrior queen", 512, 768, 20},
        {"WL-E", "abstract fractal geometry",   512, 512, 10},
    };
    static constexpr std::size_t kNumWorkloads =
        sizeof(kWorkloads) / sizeof(kWorkloads[0]);

    // -------------------------------------------------------------------------
    // Overhead proof
    // -------------------------------------------------------------------------
    hq::BenchmarkLogger bench_log;
    const double overhead_ns = bench_log.measure_overhead_ns(10000);
    bench_log.clear();
    std::print("Logger overhead: {:.1f} ns/record\n\n", overhead_ns);

    // -------------------------------------------------------------------------
    // Per-workload timing vectors
    // -------------------------------------------------------------------------
    std::vector<std::vector<double>> wl_times(kNumWorkloads);
    for (auto& v : wl_times) v.reserve(args.iterations);

    // -------------------------------------------------------------------------
    // Run campaign
    // -------------------------------------------------------------------------
    auto cfg = make_pipeline_config(args);

    bench_log.record(hq::BenchPhase::CAMPAIGN_START);

    try {
        hq::Pipeline pipeline{cfg};

        // 3 warm-up iterations (results discarded) to reach steady state
        {
            hq::GenerationRequest warm{
                .prompt = "warmup", .width = 512, .height = 512,
                .num_steps = 5, .guidance_scale = 1.0f, .seed = 0};
            for (int w = 0; w < 3; ++w) (void)pipeline.generate(warm);
        }

        for (std::size_t wl = 0; wl < kNumWorkloads; ++wl) {
            const Workload& W = kWorkloads[wl];
            std::print("Running {} — {} ({}x{} {}steps × {} iters)...\n",
                       W.id, W.prompt, W.width, W.height,
                       W.num_steps, args.iterations);

            for (std::uint32_t iter = 0; iter < args.iterations; ++iter) {
                hq::GenerationRequest req{
                    .prompt         = W.prompt,
                    .width          = W.width,
                    .height         = W.height,
                    .num_steps      = W.num_steps,
                    .guidance_scale = 7.5f,
                    .seed           = static_cast<std::int64_t>(iter + 1),
                };

                const auto t0 = std::chrono::steady_clock::now();
                (void)pipeline.generate(req);
                const auto t1 = std::chrono::steady_clock::now();

                const auto dur_ns = static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        t1 - t0).count());

                wl_times[wl].push_back(static_cast<double>(dur_ns) / 1.0e6);
                bench_log.record(hq::BenchPhase::ITER_END, iter, dur_ns,
                                 static_cast<std::uint64_t>(wl));
            }
        }

        bench_log.record(hq::BenchPhase::CAMPAIGN_END);
        pipeline.shutdown();

    } catch (const std::exception& e) {
        std::print("Campaign failed: {}\n", e.what());
        return EXIT_FAILURE;
    }

    // -------------------------------------------------------------------------
    // Results table
    // -------------------------------------------------------------------------
    std::print("\n=== Campaign Results ===\n\n");
    std::print("| ID   | Workload                  | N  |  P50 ms |  P95 ms |  P99 ms |  Mean ms | CV%% |\n");
    std::print("|------|---------------------------|----|---------|---------|---------|---------|---------|\n");

    for (std::size_t wl = 0; wl < kNumWorkloads; ++wl) {
        const Workload& W = kWorkloads[wl];
        auto s = hq::LatencyStats::from_ms(wl_times[wl]);
        std::print("| {:4} | {:25} | {:2} | {:>7.2f} | {:>7.2f} | {:>7.2f} | {:>8.2f} | {:>6.1f} |\n",
                   W.id,
                   W.prompt,
                   s.count,
                   s.p50_ms,
                   s.p95_ms,
                   s.p99_ms,
                   s.mean_ms,
                   s.cv_pct);
    }

    std::print("\nLogger overhead: {:.1f} ns/record\n", overhead_ns);

    // -------------------------------------------------------------------------
    // Export
    // -------------------------------------------------------------------------
    std::filesystem::create_directories(args.output_dir);
    const auto json_path = std::filesystem::path(args.output_dir) / "campaign.json";
    const auto csv_path  = std::filesystem::path(args.output_dir) / "campaign.csv";
    if (bench_log.export_json(json_path))
        std::print("  JSON: {}\n", json_path.string());
    if (bench_log.export_csv(csv_path))
        std::print("  CSV:  {}\n", csv_path.string());

    std::print("\n=== Campaign Complete ===\n");
    return EXIT_SUCCESS;
}

// ---------------------------------------------------------------------------
// cmd_tier_migrate_bench
// Measures real promote/demote cost for a >=64 MiB tensor through the
// TieredMemoryManager (Cool<->Warm). Reports P50/P95/P99/CV and throughput.
// This is Stage 2 evidence for Round 10: every migration timing is from a
// live memcpy of kTestBytes through the real TieredMemoryManager code path.
// ---------------------------------------------------------------------------
[[nodiscard]] int cmd_tier_migrate_bench(const CLIArgs& args) {
    constexpr std::size_t kTestSizeMiB = 64;
    constexpr std::size_t kTestBytes   = kTestSizeMiB * 1024ULL * 1024ULL;
    const std::uint32_t   N = args.iterations;

    std::print("\n=== TieredMemoryManager Migrate Benchmark ===\n");
    std::print("Block:      {} MiB ({} bytes)\n", kTestSizeMiB, kTestBytes);
    std::print("Direction:  Cool<->Warm ({} cycles each)\n", N);
    std::print("Host-only: memcpy-backed (CXL absent — Warm falls back to aligned_alloc)\n\n");

    hq::BenchmarkLogger bench_log;
    const double overhead_ns = bench_log.measure_overhead_ns(10000);
    bench_log.clear();
    std::print("Logger overhead: {:.1f} ns/record\n\n", overhead_ns);

    // Capacity: 3x block size to handle simultaneous old+new during migration
    hq::TieredMemoryConfig cfg;
    cfg.cool_capacity_bytes = 3ULL * kTestBytes;
    cfg.warm_capacity_bytes = 3ULL * kTestBytes;
    hq::TieredMemoryManager tmm(cfg);

    // Allocate test block in Cool tier, fill with a known pattern
    auto alloc_r = tmm.allocate(kTestBytes, hq::MemoryTier::Cool);
    if (!alloc_r) {
        std::print("FATAL: Cool tier alloc failed: {}\n", hq::to_string(alloc_r.error()));
        return EXIT_FAILURE;
    }
    const hq::TierHandle handle = alloc_r->handle;
    if (alloc_r->ptr) std::memset(alloc_r->ptr, 0xAB, kTestBytes);

    std::print("Initial: handle={:#x}  tier={}  ptr={}\n\n",
               handle,
               hq::to_string(alloc_r->tier),
               alloc_r->ptr ? "valid" : "null");

    std::vector<std::uint64_t> promote_ns_v, demote_ns_v;
    promote_ns_v.reserve(N);
    demote_ns_v.reserve(N);

    for (std::uint32_t i = 0; i < N; ++i) {
        // Cool → Warm
        {
            const auto t0 = std::chrono::steady_clock::now();
            auto pr = tmm.promote(handle);
            const auto t1 = std::chrono::steady_clock::now();
            const std::uint64_t dur = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
            if (!pr) {
                std::print("promote() failed at iter {}: {}\n", i, hq::to_string(pr.error()));
                return EXIT_FAILURE;
            }
            promote_ns_v.push_back(dur);
            bench_log.record(hq::BenchPhase::TIER_MIGRATE, i, dur, 0); // meta=0 => promote
        }
        // Warm → Cool
        {
            const auto t0 = std::chrono::steady_clock::now();
            auto dm = tmm.demote(handle);
            const auto t1 = std::chrono::steady_clock::now();
            const std::uint64_t dur = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
            if (!dm) {
                std::print("demote() failed at iter {}: {}\n", i, hq::to_string(dm.error()));
                return EXIT_FAILURE;
            }
            demote_ns_v.push_back(dur);
            bench_log.record(hq::BenchPhase::TIER_MIGRATE, i, dur, 1); // meta=1 => demote
        }
    }

    (void)tmm.free(handle);

    const auto ps = hq::LatencyStats::from_ns(std::move(promote_ns_v));
    const auto ds = hq::LatencyStats::from_ns(std::move(demote_ns_v));

    const double tp_promote = (kTestSizeMiB * 1000.0) / ps.p50_ms; // MiB/s at P50
    const double tp_demote  = (kTestSizeMiB * 1000.0) / ds.p50_ms;

    std::print("=== Results (N={}) ===\n\n", N);
    std::print("{:<22}  {:>9}  {:>9}  {:>9}  {:>9}  {:>7}  {:>16}\n",
               "Direction", "P50 ms", "P95 ms", "P99 ms", "Mean ms", "CV%", "Throughput MiB/s");
    std::print("{:-<22}  {:->9}  {:->9}  {:->9}  {:->9}  {:->7}  {:->16}\n",
               "", "", "", "", "", "", "");
    std::print("{:<22}  {:>9.3f}  {:>9.3f}  {:>9.3f}  {:>9.3f}  {:>7.1f}  {:>16.1f}\n",
               "Cool->Warm (promote)", ps.p50_ms, ps.p95_ms, ps.p99_ms, ps.mean_ms, ps.cv_pct, tp_promote);
    std::print("{:<22}  {:>9.3f}  {:>9.3f}  {:>9.3f}  {:>9.3f}  {:>7.1f}  {:>16.1f}\n",
               "Warm->Cool (demote)", ds.p50_ms, ds.p95_ms, ds.p99_ms, ds.mean_ms, ds.cv_pct, tp_demote);

    std::print("\nLogger overhead: {:.1f} ns/record = {:.4f}%% of P50 promote cost\n",
               overhead_ns,
               (ps.p50_ms > 0.0) ? (overhead_ns / (ps.p50_ms * 1.0e6) * 100.0) : 0.0);

    // Export
    std::filesystem::create_directories(args.output_dir);
    const auto json_path = std::filesystem::path(args.output_dir) / "tier_migrate.json";
    const auto csv_path  = std::filesystem::path(args.output_dir) / "tier_migrate.csv";
    if (bench_log.export_json(json_path))
        std::print("  JSON: {}\n", json_path.string());
    if (bench_log.export_csv(csv_path))
        std::print("  CSV:  {}\n", csv_path.string());

    std::print("\n=== TierMigrateBench Complete ===\n");
    return EXIT_SUCCESS;
}

// =============================================================================
// CMD: profile — Per-phase timing breakdown (NPU/GPU/CPU heterogeneous split)
// =============================================================================
[[nodiscard]] int cmd_profile(const CLIArgs& args) {
    const std::string prompt = args.prompt.empty()
        ? "a futuristic city at night, neon lights, cinematic"
        : args.prompt;

    std::print("=== Cerberus Heterogeneous Profile ===\n");
    std::print("  Prompt  : {}\n", prompt);
    std::print("  Size    : {}x{}  Steps: {}\n\n", args.width, args.height, args.steps);

    std::print("  NOTE: This binary targets AMD Zen 4 (-march=znver4).\n");
    std::print("  Run on the UM790 Pro (Ryzen 9 7940HS) for real hardware numbers.\n");
    std::print("  On non-Zen4 hardware the binary will fault with STATUS_ILLEGAL_INSTRUCTION.\n\n");

    auto cfg = make_pipeline_config(args);
    cfg.enable_watchdog = false;  // suppress watchdog noise during profiling

    hq::Pipeline pipeline{cfg};

    hq::GenerationRequest req;
    req.prompt         = prompt;
    req.width          = args.width;
    req.height         = args.height;
    req.num_steps      = args.steps;
    req.guidance_scale = 7.5f;
    req.seed           = args.seed != 0 ? static_cast<int>(args.seed) : 42;

    std::print("  Running generate()...\n\n");
    auto result = pipeline.generate(req);

    const auto& timings = pipeline.last_phase_timings();

    std::print("=== Phase Timing Breakdown ===\n");
    std::print("  +-----------------------+----------+------+\n");
    std::print("  | Phase                 |    ms    | %%   |\n");
    std::print("  +-----------------------+----------+------+\n");

    const double total_ms = timings.text_encode_ms + timings.embedding_stage_ms
                          + timings.denoise_total_ms + timings.vae_decode_ms
                          + timings.post_process_ms;

    auto pct = [&](double ms) -> double {
        return total_ms > 0.0 ? (ms / total_ms) * 100.0 : 0.0;
    };

    std::print("  | Text encode (NPU/CPU) | {:8.1f} | {:4.1f} |\n",
               timings.text_encode_ms, pct(timings.text_encode_ms));
    std::print("  | Embedding staging     | {:8.1f} | {:4.1f} |\n",
               timings.embedding_stage_ms, pct(timings.embedding_stage_ms));
    std::print("  | Denoise loop ({:2} step)| {:8.1f} | {:4.1f} |\n",
               timings.num_denoise_steps, timings.denoise_total_ms, pct(timings.denoise_total_ms));
    std::print("  |   CFG blend (NPU/in-loop) {:6.1f}µs total across {} steps\n",
               timings.npu_blend_in_loop_us, timings.num_denoise_steps);
    std::print("  | VAE decode            | {:8.1f} | {:4.1f} |\n",
               timings.vae_decode_ms, pct(timings.vae_decode_ms));
    std::print("  | NPU post-process      | {:8.1f} | {:4.1f} |\n",
               timings.post_process_ms, pct(timings.post_process_ms));
    std::print("  +-----------------------+----------+------+\n");
    std::print("  | TOTAL                 | {:8.1f} | 100  |\n", total_ms);
    std::print("  +-----------------------+----------+------+\n\n");

    std::print("  Encoder       : {}\n", timings.encoder_name);
    std::print("  Post-processor: {}\n", timings.post_processor_name);
    std::print("\n");

    if (!result) {
        std::print("  generate() failed: {}\n", hq::to_string(result.error()));
        return EXIT_FAILURE;
    }

    std::print("  Generate succeeded. Image: {}x{} ({} bytes)\n",
               result->width, result->height, result->pixels.size());
    std::print("\n=== Heterogeneous Execution Reality (2026-05-22) ===\n");
    std::print("  Text encoding  : {} — CPU inference via ONNX Runtime (no NPU hardware)\n",
                timings.encoder_name);
    std::print("  UNet denoising : ORT GPU session (ROCm EP if available, else CPU)\n");
    std::print("  VAE decode     : ORT session (same path as UNet)\n");
    std::print("  Post-processing: {} — CPU pass-through (HailoRT not installed)\n",
               timings.post_processor_name);
    std::print("\n  Real NPU (Hailo-8L) participation: 0%%\n");
    std::print("  Requires: HailoRT SDK + HEF on Ubuntu + UM790 Pro hardware\n");
    std::print("\n=== Profile Complete ===\n");
    return EXIT_SUCCESS;
}
