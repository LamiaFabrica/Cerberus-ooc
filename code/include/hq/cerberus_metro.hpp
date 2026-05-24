#pragma once
/// @file cerberus_metro.hpp
/// @copyright Copyright (c) 2026 D Hargreaves. All rights reserved.
///
/// Cerberus Metro — Monitoring station with trace capture and audit.
/// Ported from PsiForceDB Metro Station.
/// "1 Pipe In / 1 Pipe Out" — every inference command is traced.
///
/// @version 1.0.0

#include <cstdint>
#include <string>
#include <vector>
#include <chrono>
#include <optional>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <shared_mutex>
#include <cstring>

namespace hq::cerberus::metro {

// ===========================================================================
// Glowing String Trace (simplified from PFT)
// ===========================================================================

struct MetroTrace {
    uint64_t trace_id{0};
    std::string origin;
    std::string destination;
    double resonance{0.0};
    uint8_t hop_count{0};
    bool persistent{true};
    std::vector<std::string> waypoints;
    std::chrono::steady_clock::time_point birth_timestamp;

    [[nodiscard]] double glowStrength() const noexcept;
};

// ===========================================================================
// Session (per-inference-session tracking)
// ===========================================================================

struct MetroSession {
    uint64_t session_id{0};
    std::string client_identity;
    std::chrono::steady_clock::time_point created_at;
    std::chrono::steady_clock::time_point last_activity;
    bool authenticated{false};
    uint32_t permissions_mask{0};
    uint64_t packets_processed{0};
    uint64_t bytes_processed{0};

    void touch() { last_activity = std::chrono::steady_clock::now(); }
};

// ===========================================================================
// Station Config
// ===========================================================================

struct MetroConfig {
    std::string station_id{"cerberus_gate_01"};
    std::string station_name{"Cerberus Inference Fortress Entry/Exit"};
    bool require_encryption{false};
    bool require_authentication{true};
    bool trace_all_packets{true};
    double min_glow_resonance{0.1};
    std::chrono::nanoseconds routing_timeout{std::chrono::seconds{30}};
};

// ===========================================================================
// Metro Station
// ===========================================================================

class MetroStation {
public:
    explicit MetroStation(MetroConfig cfg);
    ~MetroStation();

    bool open();
    void close();
    [[nodiscard]] bool isOpen() const noexcept;

    // Main entry: process incoming command packet
    [[nodiscard]] std::vector<uint8_t> processIncoming(
        const std::vector<uint8_t>& request,
        const std::string& client_identity);

    // Session management
    [[nodiscard]] uint64_t createSession(const std::string& client_identity);
    bool authenticateSession(uint64_t session_id, uint32_t permissions_mask);
    void terminateSession(uint64_t session_id);
    [[nodiscard]] std::size_t activeSessions() const;

    // Trace management
    void storeTrace(const MetroTrace& trace);
    [[nodiscard]] std::vector<MetroTrace> activeTraces() const;

    // Stats
    [[nodiscard]] uint64_t packetsIn() const { return packets_in_.load(); }
    [[nodiscard]] uint64_t packetsOut() const { return packets_out_.load(); }
    [[nodiscard]] uint64_t bytesIn() const { return bytes_in_.load(); }
    [[nodiscard]] uint64_t bytesOut() const { return bytes_out_.load(); }

private:
    MetroConfig config_;

    // Pipes
    struct Pipe {
        int read_fd{-1};
        int write_fd{-1};
        bool is_open{false};
    };
    Pipe inbound_pipe_;
    Pipe outbound_pipe_;

    // Sessions
    std::unordered_map<uint64_t, MetroSession> sessions_;
    mutable std::shared_mutex sessions_mutex_;
    std::atomic<uint64_t> next_session_id_{1};

    // Traces
    std::vector<MetroTrace> active_traces_;
    mutable std::shared_mutex traces_mutex_;

    // Stats
    std::atomic<uint64_t> packets_in_{0};
    std::atomic<uint64_t> packets_out_{0};
    std::atomic<uint64_t> bytes_in_{0};
    std::atomic<uint64_t> bytes_out_{0};
};

// Global accessor
MetroStation& get_global_metro_station();

} // namespace hq::cerberus::metro
