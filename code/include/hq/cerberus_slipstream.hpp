#pragma once
/// @file cerberus_slipstream.hpp
/// @copyright Copyright (c) 2026 D Hargreaves. All rights reserved.
///
/// Cerberus Slipstream — Parallel AI-native command/data superhighway.
/// Ported from PsiForceDB AI Slipstream Core.
/// Separate from human command path; depot-based buffering.
///
/// @version 1.0.0

#include <cstdint>
#include <vector>
#include <memory>
#include <functional>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <optional>
#include <thread>

namespace hq::cerberus::slipstream {

// ===========================================================================
// Message Types
// ===========================================================================

enum class SlipstreamMessageType : uint16_t {
    HANDSHAKE      = 0x0001,
    HANDSHAKE_ACK  = 0x0002,
    HEARTBEAT      = 0x0003,
    DISCONNECT     = 0x0004,

    CMD_EXECUTE    = 0x0200,
    CMD_RESPONSE   = 0x0201,
    CMD_STREAM_START = 0x0202,
    CMD_STREAM_DATA  = 0x0203,
    CMD_STREAM_END   = 0x0204,

    SYS_STATS      = 0x0500,
    SYS_SHUTDOWN   = 0x0502,

    ERROR_GENERIC  = 0xFF00,
    ERROR_AUTH     = 0xFF01,
    ERROR_TIMEOUT  = 0xFF04,
};

// ===========================================================================
// Message Header (fixed 32 bytes)
// ===========================================================================

#pragma pack(push, 1)
struct SlipstreamMessageHeader {
    uint32_t magic{0x534C4950}; // "SLIP"
    uint16_t version{1};
    uint16_t type{0x0001};
    uint16_t flags{0};
    uint32_t payload_length{0};
    uint32_t sequence_number{0};
    uint32_t timestamp_us{0};
    uint8_t session_id[8]{};
    uint8_t padding[2]{};

    [[nodiscard]] bool isValid() const noexcept { return magic == 0x534C4950; }
};
static_assert(sizeof(SlipstreamMessageHeader) == 32, "Header must be exactly 32 bytes");
#pragma pack(pop)

// ===========================================================================
// Message
// ===========================================================================

struct SlipstreamMessage {
    SlipstreamMessageHeader header{};
    std::vector<uint8_t> payload;

    [[nodiscard]] SlipstreamMessageType getType() const {
        return static_cast<SlipstreamMessageType>(header.type);
    }

    [[nodiscard]] std::vector<uint8_t> serialize() const;
    [[nodiscard]] static std::optional<SlipstreamMessage> deserialize(const uint8_t* data, std::size_t len);

    static SlipstreamMessage create(SlipstreamMessageType type,
                                     const std::vector<uint8_t>& data = {},
                                     uint32_t seq = 0);
};

// ===========================================================================
// Depot — lock-free-ish buffer at each end
// ===========================================================================

class SlipstreamDepot {
public:
    explicit SlipstreamDepot(std::size_t capacity = 10000);
    ~SlipstreamDepot() = default;

    bool deposit(SlipstreamMessage message);
    std::optional<SlipstreamMessage> collect(std::chrono::milliseconds timeout = std::chrono::milliseconds{1});
    std::optional<SlipstreamMessage> tryCollect();
    std::optional<SlipstreamMessage> peek() const;

    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] bool empty() const;
    [[nodiscard]] bool full() const;
    [[nodiscard]] std::size_t capacity() const { return capacity_; }
    void clear();

    struct Stats {
        uint64_t deposited{0};
        uint64_t collected{0};
        uint64_t dropped{0};
        uint64_t timeouts{0};
        uint64_t peeked{0};
    };
    [[nodiscard]] Stats getStats() const;

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<SlipstreamMessage> queue_;
    std::size_t capacity_{0};
    mutable Stats stats_;
};

// ===========================================================================
// TUI Monitor — read-only human window into slipstream
// ===========================================================================

class SlipstreamMonitor {
public:
    static SlipstreamMonitor& instance();

    void initialize(const SlipstreamDepot* ingress, const SlipstreamDepot* egress);
    void start();
    void stop();

    struct Snapshot {
        uint64_t messages_in{0};
        uint64_t messages_out{0};
        uint64_t bytes_per_sec{0};
        std::size_t queue_depth_in{0};
        std::size_t queue_depth_out{0};
        std::string last_command_preview;
        std::chrono::system_clock::time_point timestamp;
    };
    [[nodiscard]] Snapshot getSnapshot() const;
    [[nodiscard]] bool isRunning() const noexcept { return running_.load(); }

private:
    SlipstreamMonitor() = default;
    ~SlipstreamMonitor();

    void monitorLoop();
    void updateSnapshot();  // Added declaration

    const SlipstreamDepot* ingress_depot_{nullptr};
    const SlipstreamDepot* egress_depot_{nullptr};

    std::atomic<bool> running_{false};
    std::atomic<bool> stop_requested_{false};
    std::thread monitor_thread_;

    mutable std::mutex snapshot_mutex_;
    Snapshot current_snapshot_;
};

// ===========================================================================
// Worker — executes commands from depot
// ===========================================================================

class SlipstreamWorker {
public:
    using CommandHandler = std::function<void(const SlipstreamMessage& cmd, SlipstreamDepot& response_depot)>;

    SlipstreamWorker(uint32_t worker_id, SlipstreamDepot& command_depot,
                     SlipstreamDepot& response_depot, CommandHandler handler);
    ~SlipstreamWorker();

    void start();
    void stop();
    [[nodiscard]] bool isRunning() const noexcept { return running_.load(); }
    [[nodiscard]] uint32_t id() const noexcept { return worker_id_; }

private:
    void workerLoop();

    uint32_t worker_id_{0};
    SlipstreamDepot& command_depot_;
    SlipstreamDepot& response_depot_;
    CommandHandler handler_;

    std::atomic<bool> running_{false};
    std::atomic<bool> stop_requested_{false};
    std::thread thread_;
};

// ===========================================================================
// Engine — core parallel execution system
// ===========================================================================

class CerberusSlipstreamEngine {
public:
    CerberusSlipstreamEngine();
    ~CerberusSlipstreamEngine();

    bool initialize(std::size_t worker_count = 4);
    bool start();
    void stop();
    [[nodiscard]] bool isRunning() const noexcept { return running_.load(); }

    void registerHandler(SlipstreamMessageType type, SlipstreamWorker::CommandHandler handler);
    SlipstreamDepot& getIngressDepot() { return *ingress_depot_; }
    SlipstreamDepot& getEgressDepot() { return *egress_depot_; }
    SlipstreamMonitor& getMonitor() { return SlipstreamMonitor::instance(); }

    struct Stats {
        uint64_t messages_processed{0};
        uint64_t commands_executed{0};
        uint64_t bytes_transferred{0};
        double avg_latency_ms{0.0};
        std::size_t active_workers{0};
        std::size_t queue_depth{0};
    };
    [[nodiscard]] Stats getStats() const;

private:
    void dispatcherLoop();

    std::atomic<bool> running_{false};
    std::atomic<bool> stop_requested_{false};

    std::unique_ptr<SlipstreamDepot> ingress_depot_;
    std::unique_ptr<SlipstreamDepot> egress_depot_;
    std::vector<std::unique_ptr<SlipstreamWorker>> workers_;

    std::unordered_map<SlipstreamMessageType, SlipstreamWorker::CommandHandler> handlers_;
    mutable std::mutex handlers_mutex_;

    mutable std::mutex stats_mutex_;
    Stats stats_;
};

} // namespace hq::cerberus::slipstream
