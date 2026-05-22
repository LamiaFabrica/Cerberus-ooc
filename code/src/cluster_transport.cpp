/// @file cluster_transport.cpp
/// @copyright Copyright (c) 2026 D Hargreaves (AKA Roylepython). LamiaFabrica. All rights reserved.
/// Multi-node clustering transport — implementation.
///
/// Socket layer: POSIX TCP (AF_INET) or AF_UNIX for loopback.
/// The socket I/O runs in background jthreads so the pipeline thread
/// is never blocked on network I/O.

#include "hq/cluster_transport.hpp"
#include "hq/logger.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <format>
#if UM790_HAS_STD_PRINT
#  include <print>
#endif
#include <queue>
#include <stop_token>
#include <thread>
#include <unordered_map>

// POSIX socket headers (Linux / macOS / MinGW on Windows)
#if defined(_WIN32)
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  ifdef _MSC_VER
#    pragma comment(lib, "Ws2_32.lib")
#  endif
   using SockFd = SOCKET;
   static constexpr SockFd kInvalidSock = INVALID_SOCKET;
   inline bool sock_valid(SockFd s) { return s != INVALID_SOCKET; }
   inline void sock_close(SockFd s) { closesocket(s); }
#else
#  include <arpa/inet.h>
#  include <netinet/tcp.h>
#  include <sys/socket.h>
#  include <sys/un.h>
#  include <unistd.h>
   using SockFd = int;
   static constexpr SockFd kInvalidSock = -1;
   inline bool sock_valid(SockFd s) { return s >= 0; }
   inline void sock_close(SockFd s) { ::close(s); }
#endif

namespace hq::cluster {

// ---------------------------------------------------------------------------
// Wire helpers
// ---------------------------------------------------------------------------

namespace {

constexpr std::uint32_t kMagic = 0xCEBE5721;

bool write_all(SockFd fd, const void* buf, std::size_t len) noexcept {
    const auto* p = static_cast<const char*>(buf);
    std::size_t remaining = len;
    while (remaining > 0) {
#if defined(_WIN32)
        auto n = send(fd, p, static_cast<int>(remaining), 0);
#else
        auto n = ::write(fd, p, remaining);
#endif
        if (n <= 0) return false;
        p += n;
        remaining -= static_cast<std::size_t>(n);
    }
    return true;
}

bool read_all(SockFd fd, void* buf, std::size_t len) noexcept {
    auto* p = static_cast<char*>(buf);
    std::size_t remaining = len;
    while (remaining > 0) {
#if defined(_WIN32)
        auto n = recv(fd, p, static_cast<int>(remaining), 0);
#else
        auto n = ::read(fd, p, remaining);
#endif
        if (n <= 0) return false;
        p += n;
        remaining -= static_cast<std::size_t>(n);
    }
    return true;
}

[[nodiscard]] bool send_msg(SockFd fd, MsgType type,
                            std::span<const std::byte> payload) noexcept {
    MessageHeader hdr{};
    hdr.magic       = kMagic;
    hdr.type        = type;
    hdr.payload_len = static_cast<std::uint32_t>(payload.size());

    if (!write_all(fd, &hdr, sizeof(hdr))) return false;
    if (!payload.empty())
        if (!write_all(fd, payload.data(), payload.size())) return false;
    return true;
}

[[nodiscard]] std::optional<ClusterTransport::ReceivedMsg>
recv_msg(SockFd fd) noexcept {
    MessageHeader hdr{};
    if (!read_all(fd, &hdr, sizeof(hdr))) return std::nullopt;
    if (hdr.magic != kMagic) return std::nullopt;
    // Guard against pathological payload sizes (> 256 MiB)
    if (hdr.payload_len > 256u * 1024u * 1024u) return std::nullopt;

    ClusterTransport::ReceivedMsg msg{};
    msg.type = hdr.type;
    if (hdr.payload_len > 0) {
        msg.payload.resize(hdr.payload_len);
        if (!read_all(fd, msg.payload.data(), hdr.payload_len)) return std::nullopt;
    }
    return msg;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------

struct ClusterTransport::Impl {
    TransportConfig cfg;

    // Atomic state
    std::atomic<bool> running{false};

    // Worker registry (coordinator side)
    mutable std::mutex workers_mutex;
    std::unordered_map<std::uint32_t, ClusterNode>       worker_map;
    std::unordered_map<std::uint32_t, SockFd>            worker_sockets;
    std::unordered_map<std::uint32_t, WorkerTelemetry>   worker_telem;
    std::atomic<std::uint32_t> rr_index{0}; // round-robin cursor

    // Inbound message queue (fed by accept loop + recv loops)
    mutable std::mutex recv_mutex;
    std::queue<ReceivedMsg> recv_queue;
    std::condition_variable recv_cv;

    // Local telemetry (updated by the pipeline each step)
    mutable std::mutex local_telem_mutex;
    WorkerTelemetry local_telem{};

    // Listen socket (coordinator) / connect socket (worker)
    SockFd listen_fd{kInvalidSock};
    SockFd coord_fd{kInvalidSock}; // worker → coordinator connection

    // Background threads
    std::unique_ptr<std::jthread> accept_thread;
    std::unique_ptr<std::jthread> heartbeat_thread;
    std::unique_ptr<std::jthread> worker_recv_thread; // worker → pump incoming msgs

    // Stats
    std::atomic<std::uint64_t> stat_sent{0};
    std::atomic<std::uint64_t> stat_recvd{0};
    std::atomic<std::uint64_t> stat_bytes_sent{0};
    std::atomic<std::uint64_t> stat_bytes_recvd{0};
    std::atomic<std::uint64_t> stat_send_err{0};
    std::atomic<std::uint64_t> stat_recv_err{0};
    std::atomic<std::uint64_t> stat_heartbeats{0};

    // Pinned staging pool for zero-copy embedding transfers
    std::unique_ptr<PinnedStagingPool<float>> staging_pool;

    explicit Impl(TransportConfig c) : cfg{std::move(c)} {}
    ~Impl() noexcept { cleanup(); }

    void cleanup() noexcept {
        if (sock_valid(listen_fd)) { sock_close(listen_fd); listen_fd = kInvalidSock; }
        if (sock_valid(coord_fd))  { sock_close(coord_fd);  coord_fd  = kInvalidSock; }
        std::unique_lock lock{workers_mutex};
        for (auto& [id, fd] : worker_sockets) {
            if (sock_valid(fd)) sock_close(fd);
        }
        worker_sockets.clear();
    }

    /// Push a received message onto the inbound queue and notify waiters.
    void push_recv(ReceivedMsg msg, std::uint32_t from_node_id) noexcept {
        msg.from_node_id = from_node_id;
        std::unique_lock lock{recv_mutex};
        stat_bytes_recvd.fetch_add(sizeof(MessageHeader) + msg.payload.size(),
                                   std::memory_order_relaxed);
        recv_queue.push(std::move(msg));
        recv_cv.notify_one();
    }

    /// Spawn a per-worker receive pump for a newly-connected worker socket.
    /// The pump reads messages and forwards them to the inbound queue.
    void spawn_worker_pump(SockFd fd, std::uint32_t worker_id) {
        // Detached jthread — stopped when fd is closed or running goes false
        std::jthread pump{[this, fd, worker_id](std::stop_token st) {
            while (!st.stop_requested() && running.load()) {
                auto msg = recv_msg(fd);
                if (!msg) {
                    // Socket closed or protocol error — worker disconnected
                    HQ_LOG_WARN("ClusterTransport: worker {} disconnected",
                                worker_id);
                    std::unique_lock lock{workers_mutex};
                    if (auto it = worker_sockets.find(worker_id);
                        it != worker_sockets.end()) {
                        sock_close(it->second);
                        it->second = kInvalidSock;
                    }
                    if (auto nit = worker_map.find(worker_id);
                        nit != worker_map.end()) {
                        nit->second.reachable = false;
                    }
                    break;
                }
                stat_recvd.fetch_add(1, std::memory_order_relaxed);

                // Handle HeartbeatAck inline: update worker telemetry
                if (msg->type == MsgType::HeartbeatAck &&
                    msg->payload.size() >= sizeof(WorkerTelemetry)) {
                    WorkerTelemetry telem{};
                    std::memcpy(&telem, msg->payload.data(), sizeof(telem));
                    std::unique_lock lock{workers_mutex};
                    worker_telem[worker_id] = telem;
                    if (auto nit = worker_map.find(worker_id);
                        nit != worker_map.end()) {
                        nit->second.health_score = telem.composite_health;
                    }
                } else {
                    push_recv(std::move(*msg), worker_id);
                }
            }
        }};
        pump.detach();
    }
};

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------

ClusterTransport::ClusterTransport(TransportConfig cfg)
    : impl_{std::make_unique<Impl>(std::move(cfg))} {
    auto& m = *impl_;

    // Set up zero-copy staging pool for embedding transfers
    if (m.cfg.enable_zero_copy) {
        constexpr std::size_t kEmbeddingFloats = 77 * 768; // CLIP SD1.x
        m.staging_pool = std::make_unique<PinnedStagingPool<float>>(
            kEmbeddingFloats * sizeof(float), 4);
        if (!m.staging_pool->initialized()) {
            HQ_LOG_WARN("ClusterTransport: PinnedStagingPool init failed, "
                        "using CPU copy fallback");
            m.staging_pool.reset();
        }
    }

#if defined(_WIN32)
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#endif

    HQ_LOG_INFO("ClusterTransport: node={} role={} addr={}:{}",
                m.cfg.this_node_id,
                m.cfg.is_coordinator ? "coordinator" : "worker",
                m.cfg.bind_address, m.cfg.bind_port);
}

ClusterTransport::~ClusterTransport() noexcept {
    stop();
#if defined(_WIN32)
    WSACleanup();
#endif
}

ClusterTransport::ClusterTransport(ClusterTransport&&) noexcept = default;
ClusterTransport& ClusterTransport::operator=(ClusterTransport&&) noexcept = default;

// ---------------------------------------------------------------------------
// start()
// ---------------------------------------------------------------------------

std::expected<void, ClusterError> ClusterTransport::start() noexcept {
    auto& m = *impl_;
    if (m.running.load()) return {};

    if (m.cfg.preferred_link == LinkType::LoopbackUnix) {
        // AF_UNIX loopback — used for intra-host IPC and unit tests.
        // No real socket is opened; send() returns success immediately.
        HQ_LOG_INFO("ClusterTransport: LoopbackUnix mode (intra-host IPC / testing)");
        m.running.store(true);
        return {};
    }

    // TCP socket
    m.listen_fd = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (!sock_valid(m.listen_fd))
        return std::unexpected{ClusterError::NotConnected};

    // Reuse address to avoid TIME_WAIT stalls on restart
    int opt = 1;
    ::setsockopt(m.listen_fd, SOL_SOCKET, SO_REUSEADDR,
                 reinterpret_cast<const char*>(&opt), sizeof(opt));

    // Disable Nagle for latency-sensitive embedding transfers
    ::setsockopt(m.listen_fd, IPPROTO_TCP, TCP_NODELAY,
                 reinterpret_cast<const char*>(&opt), sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(m.cfg.bind_port);
    ::inet_pton(AF_INET, m.cfg.bind_address.c_str(), &addr.sin_addr);

    if (::bind(m.listen_fd,
               reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        sock_close(m.listen_fd);
        m.listen_fd = kInvalidSock;
        return std::unexpected{ClusterError::NotConnected};
    }

    if (::listen(m.listen_fd, static_cast<int>(m.cfg.max_workers)) < 0) {
        sock_close(m.listen_fd);
        m.listen_fd = kInvalidSock;
        return std::unexpected{ClusterError::NotConnected};
    }

    m.running.store(true);

    // Accept loop (coordinator) in background jthread
    if (m.cfg.is_coordinator) {
        m.accept_thread = std::make_unique<std::jthread>([&m](std::stop_token stoken) {
            HQ_LOG_INFO("ClusterTransport: accept loop started");
            while (!stoken.stop_requested() && m.running.load()) {
                sockaddr_in peer_addr{};
                socklen_t peer_len = sizeof(peer_addr);
                SockFd client = ::accept(m.listen_fd,
                                         reinterpret_cast<sockaddr*>(&peer_addr),
                                         &peer_len);
                if (!sock_valid(client)) {
                    if (m.running.load())
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    continue;
                }

                // Read WorkerReady handshake
                auto msg = recv_msg(client);
                if (!msg || msg->type != MsgType::WorkerReady
                    || msg->payload.size() < sizeof(std::uint32_t)) {
                    sock_close(client);
                    continue;
                }

                std::uint32_t worker_id = 0;
                std::memcpy(&worker_id, msg->payload.data(), sizeof(worker_id));

                char ip_buf[INET_ADDRSTRLEN]{};
                ::inet_ntop(AF_INET, &peer_addr.sin_addr, ip_buf, sizeof(ip_buf));

                {
                    std::unique_lock lock{m.workers_mutex};
                    m.worker_sockets[worker_id] = client;
                    auto& node = m.worker_map[worker_id];
                    node.node_id        = worker_id;
                    node.address        = ip_buf;
                    node.port           = ntohs(peer_addr.sin_port);
                    node.is_coordinator = false;
                    node.reachable      = true;
                    node.health_score   = 0.0f;
                }

                HQ_LOG_INFO("ClusterTransport: worker {} connected from {}:{}",
                            worker_id, ip_buf, ntohs(peer_addr.sin_port));

                // Spawn per-worker receive pump so inbound results land in recv_queue
                m.spawn_worker_pump(client, worker_id);
            }
            HQ_LOG_INFO("ClusterTransport: accept loop exited");
        });
    }

    // Heartbeat loop in background jthread
    m.heartbeat_thread = std::make_unique<std::jthread>([&m](std::stop_token stoken) {
        while (!stoken.stop_requested() && m.running.load()) {
            std::this_thread::sleep_for(m.cfg.heartbeat_interval);
            if (!m.cfg.is_coordinator) continue;

            std::unique_lock lock{m.workers_mutex};
            for (auto& [id, fd] : m.worker_sockets) {
                if (!sock_valid(fd)) continue;
                MessageHeader hdr{kMagic, MsgType::Heartbeat, {}, 0};
                write_all(fd, &hdr, sizeof(hdr));
                m.stat_heartbeats.fetch_add(1, std::memory_order_relaxed);
            }
        }
    });

    HQ_LOG_INFO("ClusterTransport: TCP listener started on {}:{}",
                m.cfg.bind_address, m.cfg.bind_port);
    return {};
}

// ---------------------------------------------------------------------------
// connect_to_coordinator() — worker-side outbound connect
//
// The worker opens a TCP connection to the coordinator, sends WorkerReady
// with its own node_id, then spawns a receive pump to handle incoming
// GenerateRequest and Heartbeat messages.
// ---------------------------------------------------------------------------

std::expected<void, ClusterError>
ClusterTransport::connect_to_coordinator(const std::string& host,
                                          std::uint16_t port) noexcept {
    auto& m = *impl_;

    if (m.cfg.is_coordinator) {
        HQ_LOG_WARN("ClusterTransport: connect_to_coordinator() called on coordinator node");
        return std::unexpected{ClusterError::NotConnected};
    }
    if (m.cfg.preferred_link == LinkType::LoopbackUnix) {
        // LoopbackUnix workers are co-located — no real socket needed.
        HQ_LOG_INFO("ClusterTransport: LoopbackUnix worker, coordinator connection skipped");
        m.running.store(true);
        return {};
    }

    m.coord_fd = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (!sock_valid(m.coord_fd))
        return std::unexpected{ClusterError::NotConnected};

    int opt = 1;
    ::setsockopt(m.coord_fd, IPPROTO_TCP, TCP_NODELAY,
                 reinterpret_cast<const char*>(&opt), sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);
    if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) <= 0) {
        sock_close(m.coord_fd);
        m.coord_fd = kInvalidSock;
        return std::unexpected{ClusterError::NotConnected};
    }

    if (::connect(m.coord_fd,
                  reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        sock_close(m.coord_fd);
        m.coord_fd = kInvalidSock;
        HQ_LOG_WARN("ClusterTransport: connect to coordinator {}:{} failed", host, port);
        return std::unexpected{ClusterError::NotConnected};
    }

    // Send WorkerReady with our node_id
    const std::uint32_t my_id = m.cfg.this_node_id;
    std::array<std::byte, sizeof(std::uint32_t)> id_payload{};
    std::memcpy(id_payload.data(), &my_id, sizeof(my_id));
    if (!send_msg(m.coord_fd, MsgType::WorkerReady,
                  std::span<const std::byte>{id_payload})) {
        sock_close(m.coord_fd);
        m.coord_fd = kInvalidSock;
        return std::unexpected{ClusterError::SendFailed};
    }

    m.running.store(true);
    HQ_LOG_INFO("ClusterTransport: worker {} connected to coordinator {}:{}",
                m.cfg.this_node_id, host, port);

    // Spawn receive pump for coordinator → worker messages
    m.worker_recv_thread = std::make_unique<std::jthread>(
        [&m](std::stop_token st) {
            while (!st.stop_requested() && m.running.load()) {
                auto msg = recv_msg(m.coord_fd);
                if (!msg) {
                    HQ_LOG_WARN("ClusterTransport: coordinator connection lost");
                    m.running.store(false);
                    break;
                }
                m.stat_recvd.fetch_add(1, std::memory_order_relaxed);
                m.push_recv(std::move(*msg), 0); // from_node_id=0 = coordinator
            }
        });

    return {};
}

// ---------------------------------------------------------------------------
// stop()
// ---------------------------------------------------------------------------

void ClusterTransport::stop() noexcept {
    auto& m = *impl_;
    m.running.store(false);
    if (m.accept_thread)     { m.accept_thread->request_stop();     m.accept_thread.reset(); }
    if (m.heartbeat_thread)  { m.heartbeat_thread->request_stop();  m.heartbeat_thread.reset(); }
    if (m.worker_recv_thread){ m.worker_recv_thread->request_stop();m.worker_recv_thread.reset(); }
    m.cleanup();
    HQ_LOG_INFO("ClusterTransport: stopped");
}

bool ClusterTransport::is_running() const noexcept {
    return impl_->running.load();
}

// ---------------------------------------------------------------------------
// register_worker() / workers()
// ---------------------------------------------------------------------------

bool ClusterTransport::register_worker(ClusterNode node) noexcept {
    auto& m = *impl_;
    std::unique_lock lock{m.workers_mutex};
    if (m.worker_map.count(node.node_id)) return false;
    m.worker_map.emplace(node.node_id, std::move(node));
    return true;
}

std::vector<ClusterNode> ClusterTransport::workers() const {
    auto& m = *impl_;
    std::unique_lock lock{m.workers_mutex};
    std::vector<ClusterNode> out;
    out.reserve(m.worker_map.size());
    for (const auto& [id, node] : m.worker_map) out.push_back(node);
    return out;
}

// ---------------------------------------------------------------------------
// send() / recv()
// ---------------------------------------------------------------------------

std::expected<void, ClusterError>
ClusterTransport::send(std::uint32_t target_node_id,
                       MsgType type,
                       std::span<const std::byte> payload) noexcept {
    auto& m = *impl_;
    SockFd fd = kInvalidSock;
    {
        std::unique_lock lock{m.workers_mutex};
        auto it = m.worker_sockets.find(target_node_id);
        if (it == m.worker_sockets.end()) {
            if (m.cfg.preferred_link == LinkType::LoopbackUnix) {
                // LoopbackUnix path: no real socket, record stats and succeed.
                m.stat_sent.fetch_add(1, std::memory_order_relaxed);
                m.stat_bytes_sent.fetch_add(sizeof(MessageHeader) + payload.size(),
                                            std::memory_order_relaxed);
                return {};
            }
            return std::unexpected{ClusterError::NotConnected};
        }
        fd = it->second;
    }

    if (!send_msg(fd, type, payload)) {
        m.stat_send_err.fetch_add(1, std::memory_order_relaxed);
        return std::unexpected{ClusterError::SendFailed};
    }
    m.stat_sent.fetch_add(1, std::memory_order_relaxed);
    m.stat_bytes_sent.fetch_add(sizeof(MessageHeader) + payload.size(),
                                std::memory_order_relaxed);
    return {};
}

std::expected<ClusterTransport::ReceivedMsg, ClusterError>
ClusterTransport::recv(std::chrono::milliseconds timeout) noexcept {
    auto& m = *impl_;
    std::unique_lock lock{m.recv_mutex};
    const auto deadline = (timeout.count() > 0)
                            ? std::chrono::steady_clock::now() + timeout
                            : std::chrono::steady_clock::time_point::max();
    while (m.recv_queue.empty()) {
        if (std::chrono::steady_clock::now() >= deadline)
            return std::unexpected{ClusterError::Timeout};
        m.recv_cv.wait_for(lock, std::chrono::milliseconds(50));
    }
    ReceivedMsg msg = std::move(m.recv_queue.front());
    m.recv_queue.pop();
    return msg;
}

// ---------------------------------------------------------------------------
// send_embeddings() — zero-copy path
// ---------------------------------------------------------------------------

std::expected<void, ClusterError>
ClusterTransport::send_embeddings(std::uint32_t target_node_id,
                                   std::span<const float> embeddings) noexcept {
    auto& m = *impl_;
    const auto bytes = std::as_bytes(embeddings);

    if (m.staging_pool && m.staging_pool->initialized()) {
        // Acquire a pinned slot, copy once into it, then send from pinned memory
        auto host_buf = m.staging_pool->acquire_host_buffer(0);
        if (host_buf && host_buf->size_bytes() >= bytes.size()) {
            std::memcpy(host_buf->data(), embeddings.data(), bytes.size());
            const auto pinned_span = std::span<const std::byte>{
                reinterpret_cast<const std::byte*>(host_buf->data()),
                bytes.size()};
            return send(target_node_id, MsgType::GenerateRequest, pinned_span);
        }
    }

    // Fallback: regular send from caller's buffer
    return send(target_node_id, MsgType::GenerateRequest, bytes);
}

// ---------------------------------------------------------------------------
// collect_telemetry()
// ---------------------------------------------------------------------------

std::expected<std::vector<WorkerTelemetry>, ClusterError>
ClusterTransport::collect_telemetry() noexcept {
    auto& m = *impl_;

    if (m.cfg.preferred_link == LinkType::LoopbackUnix) {
        // LoopbackUnix: return whatever telemetry has been registered for workers.
        std::unique_lock lock{m.workers_mutex};
        std::vector<WorkerTelemetry> result;
        for (const auto& [id, node] : m.worker_map) {
            auto it = m.worker_telem.find(id);
            if (it != m.worker_telem.end()) {
                result.push_back(it->second);
            } else {
                WorkerTelemetry t{};
                t.node_id         = id;
                t.composite_health = 75.0f; // neutral default for unknown nodes
                result.push_back(t);
            }
        }
        return result;
    }

    // TCP path: send Heartbeat to each connected worker; parse HeartbeatAck inline.
    // (Per-worker pumps handle async messages; this is a synchronous best-effort scan.)
    std::vector<WorkerTelemetry> result;
    std::unique_lock lock{m.workers_mutex};
    for (auto& [id, fd] : m.worker_sockets) {
        if (!sock_valid(fd)) continue;
        MessageHeader hdr{kMagic, MsgType::Heartbeat, {}, 0};
        if (!write_all(fd, &hdr, sizeof(hdr))) continue;

        auto msg = recv_msg(fd);
        if (!msg || msg->type != MsgType::HeartbeatAck) continue;
        if (msg->payload.size() < sizeof(WorkerTelemetry)) continue;

        WorkerTelemetry telem{};
        std::memcpy(&telem, msg->payload.data(), sizeof(telem));
        m.worker_telem[id] = telem;
        if (auto nit = m.worker_map.find(id); nit != m.worker_map.end())
            nit->second.health_score = telem.composite_health;
        result.push_back(telem);
    }
    return result;
}

// ---------------------------------------------------------------------------
// update_local_telemetry()
// ---------------------------------------------------------------------------

void ClusterTransport::update_local_telemetry(const WorkerTelemetry& telem) noexcept {
    auto& m = *impl_;
    std::unique_lock lock{m.local_telem_mutex};
    m.local_telem = telem;
}

// ---------------------------------------------------------------------------
// select_worker() — health-score-aware load balancer
//
// Priority order:
//   1. Highest effective_score = health - kQueuePenalty * queue_depth
//   2. Round-robin over all registered nodes when none are reachable
// ---------------------------------------------------------------------------

std::expected<DispatchDecision, ClusterError>
ClusterTransport::select_worker() noexcept {
    auto& m = *impl_;

    std::unique_lock lock{m.workers_mutex};
    if (m.worker_map.empty())
        return std::unexpected{ClusterError::NoWorkers};

    constexpr float kQueuePenalty = 5.0f;
    constexpr float kDefaultHealth = 50.0f;

    std::uint32_t best_id    = std::numeric_limits<std::uint32_t>::max();
    float         best_score = std::numeric_limits<float>::lowest();

    for (const auto& [id, node] : m.worker_map) {
        if (!node.reachable) [[unlikely]] continue;
        const float health = (node.health_score > 0.0f)
                             ? node.health_score : kDefaultHealth;
        float queue = 0.0f;
        if (auto it = m.worker_telem.find(id); it != m.worker_telem.end())
            queue = it->second.queue_depth;
        const float effective_score = health - kQueuePenalty * queue;
        if (effective_score > best_score) {
            best_score = effective_score;
            best_id    = id;
        }
    }

    if (best_id == std::numeric_limits<std::uint32_t>::max()) [[unlikely]] {
        // All unreachable — round-robin over the registered roster
        auto it = m.worker_map.begin();
        const auto sz = static_cast<std::uint32_t>(m.worker_map.size());
        const std::uint32_t rr = m.rr_index.fetch_add(1, std::memory_order_relaxed) % sz;
        std::advance(it, static_cast<std::ptrdiff_t>(rr));
        return DispatchDecision{
            .target_node_id = it->first,
            .expected_score = 0.0f,
            .rationale      = std::format("round-robin fallback (node {} of {})",
                                          it->first, sz),
        };
    }

    return DispatchDecision{
        .target_node_id = best_id,
        .expected_score = best_score,
        .rationale      = std::format("health-score={:.1f} queue-adjusted (node {})",
                                       best_score, best_id),
    };
}

// ---------------------------------------------------------------------------
// stats()
// ---------------------------------------------------------------------------

ClusterTransport::TransportStats ClusterTransport::stats() const noexcept {
    auto& m = *impl_;
    std::unique_lock lock{m.workers_mutex};
    return {
        m.stat_sent.load(),
        m.stat_recvd.load(),
        m.stat_bytes_sent.load(),
        m.stat_bytes_recvd.load(),
        m.stat_send_err.load(),
        m.stat_recv_err.load(),
        m.stat_heartbeats.load(),
        static_cast<std::uint32_t>(m.worker_sockets.size()),
    };
}

} // namespace hq::cluster
