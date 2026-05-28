/// @file cerberus_server_main.cpp
/// @copyright Copyright © 2026, all rights reserved | David Hargreaves aka Roylepython trading as The Medusa Initiative & Lamia Fabrica
/// Entry point for the Cerberus Inference Server.
///
/// Parses command-line arguments, registers POSIX signal handlers,
/// creates the InferenceServer, and blocks until SIGINT/SIGTERM.
///
/// Usage:
///   cerberus_server --port 8080 --model-dir ./models --verbose

#include "hq/inference_server.hpp"
#include "hq/cxx26_features.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#if UM790_HAS_STD_PRINT
#  include <print>
#endif
#include <string>
#include <string_view>
#include <thread>

#include <csignal>

using namespace std::literals;

// ---------------------------------------------------------------------------
// Global pointer — the only thing signal handlers can safely touch.
// We use a raw pointer (not smart) because the signal handler runs in
// a special context and we only need to call shutdown().
// ---------------------------------------------------------------------------

namespace {
std::atomic<hq::InferenceServer*> g_server{nullptr};
}

// ---------------------------------------------------------------------------
// Signal handler — async-signal-safe
// ---------------------------------------------------------------------------

extern "C" {

static void on_signal(int sig) {
    (void)sig;
    auto* srv = g_server.load(std::memory_order_acquire);
    if (srv) srv->shutdown();
}

} // extern "C"

// ---------------------------------------------------------------------------
// register_signals
// ---------------------------------------------------------------------------

static void register_signals() {
    ::signal(SIGINT,  on_signal);
    ::signal(SIGTERM, on_signal);
#ifndef _WIN32
    // Ignore SIGPIPE — we use MSG_NOSIGNAL on send() calls anyway
    ::signal(SIGPIPE, SIG_IGN);
#endif
}

// ---------------------------------------------------------------------------
// print_stats
// ---------------------------------------------------------------------------

static void print_stats(const hq::ServerStats& s,
                        std::chrono::steady_clock::time_point end) {
    auto uptime = std::chrono::duration_cast<std::chrono::seconds>(
        end - s.start_time).count();

    uint64_t reqs  = s.requests_served.load();
    uint64_t chats = s.chat_completions.load();
    uint64_t healths = s.health_checks.load();
    uint64_t models  = s.model_lists.load();
    uint64_t errs    = s.errors.load();

    std::print("\n=== Cerberus Inference Server Stats ===\n");
    std::print("  Uptime:           {} s\n", uptime);
    std::print("  Requests served:  {}\n", reqs);
    std::print("  Chat completions: {}\n", chats);
    std::print("  Health checks:    {}\n", healths);
    std::print("  Model lists:      {}\n", models);
    std::print("  Errors:           {}\n", errs);

    double rps = (uptime > 0) ? static_cast<double>(reqs) / uptime : 0.0;
    std::print("  Throughput:       {:.2f} req/s\n", rps);
    std::print("========================================\n");
}

// ---------------------------------------------------------------------------
// parse_args
// ---------------------------------------------------------------------------

struct CLIArgs {
    uint16_t    port{8080};
    std::string model_dir{"./models"};
    bool        verbose{false};
};

[[nodiscard]] CLIArgs parse_args(int argc, char** argv) {
    CLIArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (arg == "--port" && i + 1 < argc) {
            int p = std::atoi(argv[++i]);
            if (p > 0 && p <= 65535) {
                args.port = static_cast<uint16_t>(p);
            }
        } else if (arg == "--model-dir" && i + 1 < argc) {
            args.model_dir = argv[++i];
        } else if (arg == "--verbose" || arg == "-v") {
            args.verbose = true;
        } else if (arg == "--help" || arg == "-h") {
            std::print("Usage: cerberus_server [options]\n\n");
            std::print("Options:\n");
            std::print("  --port <N>        Listen port (default 8080)\n");
            std::print("  --model-dir <dir> Model directory (default ./models)\n");
            std::print("  --verbose, -v     Verbose request logging\n");
            std::print("  --help, -h        Show this help\n");
            std::exit(EXIT_SUCCESS);
        } else {
            std::print("Unknown option: {}\n", arg);
            std::exit(EXIT_FAILURE);
        }
    }
    return args;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    auto args = parse_args(argc, argv);

    std::print("=== Cerberus Inference Server v1.0.0 ===\n");
    std::print("Port:      {}\n", args.port);
    std::print("Model dir: {}\n", args.model_dir);
    std::print("Verbose:   {}\n\n", args.verbose ? "yes" : "no");

    hq::ServerConfig cfg{
        .port             = args.port,
        .bind_address     = "127.0.0.1",
        .model_dir        = args.model_dir,
        .verbose_logging  = args.verbose,
    };

    try {
        hq::InferenceServer server{cfg};
        g_server.store(&server, std::memory_order_release);

        register_signals();

        (void)server.start();

        g_server.store(nullptr, std::memory_order_release);

        auto end = std::chrono::steady_clock::now();
        print_stats(server.get_stats(), end);

    } catch (const std::exception& e) {
        std::print("[cerberus] fatal: {}\n", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
