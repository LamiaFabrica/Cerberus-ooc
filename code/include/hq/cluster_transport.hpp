#pragma once
/// @file cluster_transport.hpp
/// @copyright Copyright (c) 2026 D Hargreaves (AKA Roylepython). LamiaFabrica. All rights reserved.
///
/// TRADE SECRET — Proprietary and Confidential.
/// This file contains trade secrets of LamiaFabrica / D Hargreaves.
/// Unauthorized use, reproduction, or disclosure is strictly prohibited.
/// See LICENSE for terms.
///
/// Multi-node clustering foundation for Cerberus pipeline.
///
/// @experimental
///   ClusterTransport contains real TCP socket code (bind/listen/accept/connect,
///   per-worker jthread pump, health-score load balancer) but has NEVER been
///   tested with real multi-node hardware.  Known limitations:
///     - collect_telemetry() synchronous recv on the same fd as per-worker pumps
///       creates a read-race; no mutex guards the TCP receive path.
///     - No TLS: all inter-node traffic is plaintext TCP.
///     - No reconnect logic: a dropped connection is a permanent failure.
///     - LoopbackUnix path succeeds without real socket activity (intra-host only).
///   Do not deploy in production until these issues are addressed.
///
/// ClusterTransport provides the wire layer for coordinating multiple
/// UM790 nodes over Thunderbolt 5 (40 Gb/s) or 10 GbE.  Key design
/// decisions:
///
///   - Zero-copy embedding transfers use PinnedStagingPool<float> buffers
///     so the CPU never touches payload bytes on the critical path.
///   - Coordinator/worker protocol uses a simple length-prefixed binary
///     framing over TCP sockets (POSIX) or AF_UNIX for intra-host IPC.
///   - The load balancer queries real PipelineHealthScore + watchdog
///     data to distribute work; when no real node is reachable it falls back
///     to round-robin simulation.
///   - std::expected<T, ClusterError> propagates errors without exceptions.
///
/// @version 1.0.0

#include "hq/cxx26_features.hpp"
#include "hq/health_score.hpp"
#include "hq/pinned_staging.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#if UM790_HAS_STD_EXPECTED
#  include <expected>
#else
#  error "cluster_transport.hpp requires std::expected (C++26 / GCC 14+)"
#endif
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace hq::cluster {

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

enum class ClusterError : std::uint8_t {
    NotConnected    = 0,
    SendFailed      = 1,
    RecvFailed      = 2,
    Timeout         = 3,
    InvalidMessage  = 4,
    NoWorkers       = 5,
    WorkerBusy      = 6,
    ProtocolError   = 7,
};

[[nodiscard]] inline std::string to_string(ClusterError e) noexcept {
    switch (e) {
        case ClusterError::NotConnected:   return "NotConnected";
        case ClusterError::SendFailed:     return "SendFailed";
        case ClusterError::RecvFailed:     return "RecvFailed";
        case ClusterError::Timeout:        return "Timeout";
        case ClusterError::InvalidMessage: return "InvalidMessage";
        case ClusterError::NoWorkers:      return "NoWorkers";
        case ClusterError::WorkerBusy:     return "WorkerBusy";
        case ClusterError::ProtocolError:  return "ProtocolError";
    }
    return "Unknown";
}

// ---------------------------------------------------------------------------
// Link type (physical transport)
// ---------------------------------------------------------------------------

enum class LinkType : std::uint8_t {
    Thunderbolt5 = 0,  ///< TB5 at up to 40 Gb/s (PCIe tunnelled)
    Ethernet10G  = 1,  ///< 10 GbE
    LoopbackUnix = 2,  ///< AF_UNIX socket for intra-host IPC / testing
};

// ---------------------------------------------------------------------------
// Node descriptor
// ---------------------------------------------------------------------------

struct ClusterNode {
    std::uint32_t node_id{0};
    std::string   address{};
    std::uint16_t port{0};
    LinkType      link{LinkType::LoopbackUnix};
    float         health_score{0.0f};
    bool          is_coordinator{false};
    bool          reachable{false};
};

// ---------------------------------------------------------------------------
// Message framing (wire protocol)
//
// Every message on the wire:
//   [magic:4 bytes][type:1 byte][payload_len:4 bytes][payload:N bytes]
// ---------------------------------------------------------------------------

enum class MsgType : std::uint8_t {
    // Coordinator → Worker
    GenerateRequest  = 0x01,  ///< Submit a GenerationRequest payload
    Heartbeat        = 0x02,  ///< Keep-alive + load query
    Shutdown         = 0x03,  ///< Worker should gracefully exit

    // Worker → Coordinator
    GenerateResult   = 0x10,  ///< GeneratedImage bytes
    HeartbeatAck     = 0x11,  ///< Health score + utilization telemetry
    WorkerReady      = 0x12,  ///< Worker finished init, ready for work
    ErrorReport      = 0x1F,  ///< Worker-side error
};

struct MessageHeader {
    std::uint32_t magic{0xCEBE5721};
    MsgType       type{MsgType::Heartbeat};
    std::uint8_t  _pad[3]{};
    std::uint32_t payload_len{0};
};
static_assert(sizeof(MessageHeader) == 12, "MessageHeader must be 12 bytes");

// ---------------------------------------------------------------------------
// Telemetry payload carried in HeartbeatAck
// ---------------------------------------------------------------------------

struct WorkerTelemetry {
    std::uint32_t node_id{0};
    float         composite_health{0.0f};
    float         gpu_utilization{0.0f};
    float         hailo_utilization{0.0f};
    float         gpu_temperature_c{0.0f};
    float         queue_depth{0.0f};        ///< Pending generate() requests
    std::uint64_t generations_completed{0};
};

// ---------------------------------------------------------------------------
// Load balancer result
// ---------------------------------------------------------------------------

struct DispatchDecision {
    std::uint32_t target_node_id;  ///< Which worker to route to
    float         expected_score;  ///< Predicted composite health after dispatch
    std::string   rationale;       ///< Human-readable reason (for logging)
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct TransportConfig {
    std::uint32_t this_node_id{0};
    bool          is_coordinator{true};
    std::string   bind_address{"0.0.0.0"};
    std::uint16_t bind_port{9501};
    LinkType      preferred_link{LinkType::Ethernet10G};
    std::chrono::milliseconds heartbeat_interval{5000};
    std::chrono::milliseconds connect_timeout{10000};
    std::chrono::milliseconds recv_timeout{30000};
    std::size_t   max_workers{16};
    bool          enable_zero_copy{true};  ///< Use PinnedStagingPool for payloads
};

// ---------------------------------------------------------------------------
// ClusterTransport
// ---------------------------------------------------------------------------

class ClusterTransport {
public:
    explicit ClusterTransport(TransportConfig cfg);
    ~ClusterTransport() noexcept;

    ClusterTransport(const ClusterTransport&) = delete;
    ClusterTransport& operator=(const ClusterTransport&) = delete;
    ClusterTransport(ClusterTransport&&) noexcept;
    ClusterTransport& operator=(ClusterTransport&&) noexcept;

    // ------------------------------------------------------------------
    // Lifecycle
    // ------------------------------------------------------------------

    /// Start listening (coordinator) or begin readiness loop (worker).
    [[nodiscard]] std::expected<void, ClusterError> start() noexcept;

    /// Worker only: connect to a running coordinator and send WorkerReady.
    /// Returns NotConnected if this node is a coordinator or connection fails.
    [[nodiscard]] std::expected<void, ClusterError>
    connect_to_coordinator(const std::string& host, std::uint16_t port) noexcept;

    /// Gracefully stop all connections and background threads.
    void stop() noexcept;

    [[nodiscard]] bool is_running() const noexcept;

    // ------------------------------------------------------------------
    // Worker registration (coordinator side)
    // ------------------------------------------------------------------

    /// Register a known worker node (pre-configured roster).
    /// @return false if the node_id is already registered.
    [[nodiscard]] bool register_worker(ClusterNode node) noexcept;

    /// Get a snapshot of all known worker nodes.
    [[nodiscard]] std::vector<ClusterNode> workers() const;

    // ------------------------------------------------------------------
    // Message transport
    // ------------------------------------------------------------------

    /// Send raw payload to a specific node.
    [[nodiscard]] std::expected<void, ClusterError>
    send(std::uint32_t target_node_id,
         MsgType type,
         std::span<const std::byte> payload) noexcept;

    /// Receive next message (blocking up to recv_timeout).
    /// Returns {source_node_id, type, payload}.
    struct ReceivedMsg {
        std::uint32_t       from_node_id;
        MsgType             type;
        std::vector<std::byte> payload;
    };
    [[nodiscard]] std::expected<ReceivedMsg, ClusterError>
    recv(std::chrono::milliseconds timeout = {}) noexcept;

    // ------------------------------------------------------------------
    // Zero-copy path (coordinator → worker embedding transfer)
    // ------------------------------------------------------------------

    /// Acquire a pinned staging slot and copy embeddings into it.
    /// The slot is released automatically once send() drains it.
    [[nodiscard]] std::expected<void, ClusterError>
    send_embeddings(std::uint32_t target_node_id,
                    std::span<const float> embeddings) noexcept;

    // ------------------------------------------------------------------
    // Heartbeat / telemetry
    // ------------------------------------------------------------------

    /// Broadcast a heartbeat to all workers and collect telemetry.
    [[nodiscard]] std::expected<std::vector<WorkerTelemetry>, ClusterError>
    collect_telemetry() noexcept;

    /// Update local telemetry (called by the pipeline after each step).
    void update_local_telemetry(const WorkerTelemetry& telem) noexcept;

    // ------------------------------------------------------------------
    // Load balancer
    // ------------------------------------------------------------------

    /// Select the best worker for the next generation request.
    /// Uses composite health scores + queue depth as primary signal.
    /// Falls back to round-robin when no telemetry is available.
    [[nodiscard]] std::expected<DispatchDecision, ClusterError>
    select_worker() noexcept;

    // ------------------------------------------------------------------
    // Statistics
    // ------------------------------------------------------------------

    struct TransportStats {
        std::uint64_t messages_sent{0};
        std::uint64_t messages_received{0};
        std::uint64_t bytes_sent{0};
        std::uint64_t bytes_received{0};
        std::uint64_t send_errors{0};
        std::uint64_t recv_errors{0};
        std::uint64_t heartbeats_sent{0};
        std::uint32_t active_workers{0};
    };

    [[nodiscard]] TransportStats stats() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hq::cluster
