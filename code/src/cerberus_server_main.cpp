/// @file cerberus_server_main.cpp
/// @copyright Copyright © 2026, all rights reserved | David Hargreaves aka Roylepython trading as The Medusa Initiative & Lamia Fabrica
/// Support: https://www.patreon.com/TheMedusaInitiative — £25/month removes ads from all software at 200 subscribers
/// Entry point for the Cerberus Inference Server.
///
/// Parses command-line arguments, registers POSIX signal handlers,
/// creates the InferenceServer, and blocks until SIGINT/SIGTERM.
///
/// Usage:
///   cerberus_server --port 8080 --model-dir ./models --verbose

#include "hq/inference_server.hpp"
#include "hq/cxx26_features.hpp"
#include "hq/cerberus_local_maintenance_db.hpp"
#include "hq/cerberus_user_security.hpp"

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
#include <memory>
#include <vector>

#include <csignal>
#include <format>
#include <iostream>

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

    std::cout << std::format("\n=== Cerberus Inference Server Stats ===\n");
    std::cout << std::format("  Uptime:           {} s\n", uptime);
    std::cout << std::format("  Requests served:  {}\n", reqs);
    std::cout << std::format("  Chat completions: {}\n", chats);
    std::cout << std::format("  Health checks:    {}\n", healths);
    std::cout << std::format("  Model lists:      {}\n", models);
    std::cout << std::format("  Errors:           {}\n", errs);

    double rps = (uptime > 0) ? static_cast<double>(reqs) / uptime : 0.0;
    std::cout << std::format("  Throughput:       {:.2f} req/s\n", rps);
    std::cout << std::format("========================================\n");
}

// ---------------------------------------------------------------------------
// parse_args
// ---------------------------------------------------------------------------

struct CLIArgs {
    uint16_t    port{8080};
    std::string model_dir{"./models"};
    bool        verbose{false};
    // LCMD / privacy surface (makes /v1/inference/* audit endpoints fully functional)
    std::string lcmd_path{};
    std::vector<uint8_t> lcmd_key;   // 32 bytes when provided
    bool        enable_audit{true};
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
        } else if (arg == "--lcmd" && i + 1 < argc) {
            args.lcmd_path = argv[++i];
        } else if (arg == "--lcmd-key" && i + 1 < argc) {
            // Accept 64 hex chars or raw 32-byte file path for simplicity
            std::string keystr = argv[++i];
            if (keystr.size() == 64) {
                args.lcmd_key.resize(32);
                for (int j = 0; j < 32; ++j) {
                    unsigned int byte;
                    sscanf(keystr.c_str() + j*2, "%02x", &byte);
                    args.lcmd_key[j] = static_cast<uint8_t>(byte);
                }
            }
        } else if (arg == "--disable-audit") {
            args.enable_audit = false;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << std::format("Usage: cerberus_server [options]\n\n");
            std::cout << std::format("Options:\n");
            std::cout << std::format("  --port <N>           Listen port (default 8080)\n");
            std::cout << std::format("  --model-dir <dir>    Model directory (default ./models)\n");
            std::cout << std::format("  --verbose, -v        Verbose request logging\n");
            std::cout << std::format("  --lcmd <path>        Path to LCMD file (enables full inference audit)\n");
            std::cout << std::format("  --lcmd-key <hex64>   32-byte hex key for LCMD\n");
            std::cout << std::format("  --disable-audit      Disable inference history endpoints\n");
            std::cout << std::format("  --help, -h           Show this help\n");
            std::exit(EXIT_SUCCESS);
        } else {
            std::cout << std::format("Unknown option: {}\n", arg);
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

    std::cout << std::format("=== Cerberus Inference Server v1.0.0 ===\n");
    std::cout << std::format("Port:      {}\n", args.port);
    std::cout << std::format("Model dir: {}\n", args.model_dir);
    std::cout << std::format("Verbose:   {}\n\n", args.verbose ? "yes" : "no");

    hq::ServerConfig cfg;
    cfg.port            = args.port;
    cfg.bind_address    = "127.0.0.1";
    cfg.model_dir       = args.model_dir;
    cfg.verbose_logging = args.verbose;

    // === LCMD + RBPC wiring (makes /v1/inference/* audit surface fully functional) ===
    std::shared_ptr<hq::cerberus::privacy::LocalMaintenanceDB> lcmd;
    std::shared_ptr<hq::cerberus::privacy::UserSecurity>       user_sec;

    if (args.enable_audit && !args.lcmd_path.empty() && args.lcmd_key.size() == 32) {
        lcmd = std::make_shared<hq::cerberus::privacy::LocalMaintenanceDB>();

        // Production robustness: create parent directories if they don't exist
        try {
            std::filesystem::create_directories(std::filesystem::path(args.lcmd_path).parent_path());
        } catch (const std::exception& e) {
            std::cout << std::format("[server] WARNING: Could not create LCMD parent directory: {}\n", e.what());
        }

        if (lcmd->initialize(args.lcmd_path, args.lcmd_key)) {
            std::cout << std::format("[server] LCMD initialized successfully (inference audit enabled)\n");

            user_sec = std::make_shared<hq::cerberus::privacy::UserSecurity>();

            // Detect fresh DB (no RBPC state yet) and give clear operator guidance
            auto rbpc = lcmd->load_rbpc_state("server-default");
            bool fresh = !rbpc.has_value();

            if (fresh) {
                std::cout << std::format("[server] NOTICE: This appears to be a new or empty LCMD.\n");

                // Auto-create minimal RBPC state for immediate audit usability
                // Use a server-derived master secret from the LCMD key itself.
                // This allows basic protected audit without forcing external registration first.
                std::vector<uint8_t> server_master = args.lcmd_key;
                // Mix in a stable server identifier
                const std::string server_tag = "cerberus-server-default";
                server_master.insert(server_master.end(), server_tag.begin(), server_tag.end());

                auto pin_opt = user_sec->generate_pin("server-default", server_master);
                if (pin_opt) {
                    std::cout << std::format("[server]         Auto-generated one-time PIN for this server node: %s\n", pin_opt->c_str());
                    std::cout << std::format("[server]         !!! WRITE THIS DOWN NOW - it will not be shown again !!!\n");
                    std::cout << std::format("[server]         Use this PIN + a memorable word (register one via cerberus_register) for export/clear.\n");

                    // Persist the newly generated RBPC state into the LCMD immediately
                    // so it survives server restart without requiring external registration.
                    auto current_state = lcmd->load_rbpc_state("server-default");
                    if (current_state.has_value()) {
                        (void)lcmd->save_rbpc_state(*current_state);
                    }
                }

                std::cout << std::format("[server]         For full RBPC setup (recommended for production), run:\n");
                std::cout << std::format("[server]           cerberus_register --lcmd \"%s\" --lcmd-key <64-hex-chars>\n", args.lcmd_path.c_str());
            }
        } else {
            std::cout << std::format("[server] WARNING: Failed to open LCMD — audit endpoints will be limited\n");
            lcmd.reset();
        }
    } else if (args.enable_audit) {
        std::cout << std::format("[server] LCMD not configured — inference history endpoints will return 503\n");
        std::cout << std::format("[server] Use --lcmd <path> --lcmd-key <64-hex> to enable full audit\n");
    }

    if (lcmd) {
        cfg.lcmd = lcmd;
        cfg.user_security = user_sec;
        cfg.rbpc_node_id = "server-default";
    }

    try {
        hq::InferenceServer server{cfg};
        g_server.store(&server, std::memory_order_release);

        register_signals();

        (void)server.start();

        g_server.store(nullptr, std::memory_order_release);

        auto end = std::chrono::steady_clock::now();
        print_stats(server.get_stats(), end);

    } catch (const std::exception& e) {
        std::cout << std::format("[cerberus] fatal: {}\n", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
