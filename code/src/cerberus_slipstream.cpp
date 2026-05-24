/// @file cerberus_slipstream.cpp
/// @copyright Copyright (c) 2026 D Hargreaves. All rights reserved.
///
/// Cerberus Slipstream — depot-based parallel execution bus.
/// Ported from PsiForceDB ai_slipstream_core.cpp.
///
/// @version 1.0.0

#include "hq/cerberus_slipstream.hpp"

#include <algorithm>
#include <cstring>
#include <string>

namespace hq::cerberus::slipstream {

// ===========================================================================
// Message
// ===========================================================================

std::vector<uint8_t> SlipstreamMessage::serialize() const {
    std::vector<uint8_t> result;
    result.reserve(sizeof(header) + payload.size());
    result.resize(sizeof(header));
    std::memcpy(result.data(), &header, sizeof(header));
    if (!payload.empty()) {
        result.insert(result.end(), payload.begin(), payload.end());
    }
    return result;
}

std::optional<SlipstreamMessage> SlipstreamMessage::deserialize(const uint8_t* data, std::size_t len) {
    if (data == nullptr || len < sizeof(SlipstreamMessageHeader)) return std::nullopt;
    SlipstreamMessage msg;
    std::memcpy(&msg.header, data, sizeof(SlipstreamMessageHeader));
    if (!msg.header.isValid()) return std::nullopt;
    if (msg.header.payload_length > len - sizeof(SlipstreamMessageHeader)) return std::nullopt;
    if (msg.header.payload_length > 0) {
        msg.payload.resize(msg.header.payload_length);
        std::memcpy(msg.payload.data(), data + sizeof(SlipstreamMessageHeader), msg.header.payload_length);
    }
    return msg;
}

SlipstreamMessage SlipstreamMessage::create(SlipstreamMessageType type,
                                            const std::vector<uint8_t>& data,
                                            uint32_t seq) {
    SlipstreamMessage msg;
    msg.header.magic = 0x534C4950;
    msg.header.version = 1;
    msg.header.type = static_cast<uint16_t>(type);
    msg.header.flags = 0;
    msg.header.payload_length = static_cast<uint32_t>(data.size());
    msg.header.sequence_number = seq;
    auto timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    msg.header.timestamp_us = static_cast<uint32_t>(timestamp_us & 0xFFFFFFFFu);
    msg.payload = data;
    return msg;
}

// ===========================================================================
// Depot
// ===========================================================================

SlipstreamDepot::SlipstreamDepot(std::size_t capacity) : capacity_(capacity) {}

bool SlipstreamDepot::deposit(SlipstreamMessage message) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.size() >= capacity_) {
        ++stats_.dropped;
        return false;
    }
    queue_.push(std::move(message));
    ++stats_.deposited;
    cv_.notify_one();
    return true;
}

std::optional<SlipstreamMessage> SlipstreamDepot::collect(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    bool collected = cv_.wait_for(lock, timeout, [this] { return !queue_.empty(); });
    if (!collected) {
        ++stats_.timeouts;
        return std::nullopt;
    }
    SlipstreamMessage msg = std::move(queue_.front());
    queue_.pop();
    ++stats_.collected;
    return msg;
}

std::optional<SlipstreamMessage> SlipstreamDepot::tryCollect() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) return std::nullopt;
    SlipstreamMessage msg = std::move(queue_.front());
    queue_.pop();
    ++stats_.collected;
    return msg;
}

std::optional<SlipstreamMessage> SlipstreamDepot::peek() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) return std::nullopt;
    ++stats_.peeked;
    return queue_.front();
}

std::size_t SlipstreamDepot::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

bool SlipstreamDepot::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
}

bool SlipstreamDepot::full() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size() >= capacity_;
}

void SlipstreamDepot::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    while (!queue_.empty()) queue_.pop();
}

SlipstreamDepot::Stats SlipstreamDepot::getStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

// ===========================================================================
// Monitor
// ===========================================================================

SlipstreamMonitor& SlipstreamMonitor::instance() {
    static SlipstreamMonitor mon;
    return mon;
}

SlipstreamMonitor::~SlipstreamMonitor() {
    if (running_.load()) {
        stop();
    }
}

void SlipstreamMonitor::initialize(const SlipstreamDepot* ingress, const SlipstreamDepot* egress) {
    ingress_depot_ = ingress;
    egress_depot_ = egress;
}

void SlipstreamMonitor::start() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) return;
    stop_requested_.store(false);
    monitor_thread_ = std::thread(&SlipstreamMonitor::monitorLoop, this);
}

void SlipstreamMonitor::stop() {
    stop_requested_.store(true);
    if (monitor_thread_.joinable()) {
        monitor_thread_.join();
    }
    running_.store(false);
}

void SlipstreamMonitor::monitorLoop() {
    while (!stop_requested_.load()) {
        updateSnapshot();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

void SlipstreamMonitor::updateSnapshot() {
    Snapshot snap;
    snap.timestamp = std::chrono::system_clock::now();
    if (ingress_depot_) {
        auto istats = ingress_depot_->getStats();
        snap.messages_in = istats.deposited;
        snap.queue_depth_in = ingress_depot_->size();
    }
    if (egress_depot_) {
        auto ostats = egress_depot_->getStats();
        snap.messages_out = ostats.collected;
        snap.queue_depth_out = egress_depot_->size();
    }
    {
        std::lock_guard<std::mutex> lock(snapshot_mutex_);
        current_snapshot_ = snap;
    }
}

SlipstreamMonitor::Snapshot SlipstreamMonitor::getSnapshot() const {
    std::lock_guard<std::mutex> lock(snapshot_mutex_);
    return current_snapshot_;
}

// ===========================================================================
// Worker
// ===========================================================================

SlipstreamWorker::SlipstreamWorker(uint32_t worker_id, SlipstreamDepot& command_depot,
                                     SlipstreamDepot& response_depot, CommandHandler handler)
    : worker_id_(worker_id), command_depot_(command_depot), response_depot_(response_depot),
      handler_(std::move(handler)) {}

SlipstreamWorker::~SlipstreamWorker() {
    if (running_.load()) stop();
}

void SlipstreamWorker::start() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) return;
    stop_requested_.store(false);
    thread_ = std::thread(&SlipstreamWorker::workerLoop, this);
}

void SlipstreamWorker::stop() {
    stop_requested_.store(true);
    if (thread_.joinable()) thread_.join();
    running_.store(false);
}

void SlipstreamWorker::workerLoop() {
    while (!stop_requested_.load()) {
        auto msg = command_depot_.collect(std::chrono::milliseconds(10));
        if (msg.has_value() && handler_) {
            handler_(*msg, response_depot_);
        }
    }
}

// ===========================================================================
// Engine
// ===========================================================================

CerberusSlipstreamEngine::CerberusSlipstreamEngine() = default;
CerberusSlipstreamEngine::~CerberusSlipstreamEngine() {
    if (running_.load()) stop();
}

bool CerberusSlipstreamEngine::initialize(std::size_t worker_count) {
    if (ingress_depot_) return true;
    ingress_depot_ = std::make_unique<SlipstreamDepot>();
    egress_depot_ = std::make_unique<SlipstreamDepot>();
    SlipstreamMonitor::instance().initialize(ingress_depot_.get(), egress_depot_.get());

    for (std::size_t i = 0; i < worker_count; ++i) {
        auto handler = [this](const SlipstreamMessage& cmd, SlipstreamDepot& resp) {
            SlipstreamMessageType type = cmd.getType();
            std::lock_guard<std::mutex> lock(handlers_mutex_);
            auto it = handlers_.find(type);
            if (it != handlers_.end()) {
                it->second(cmd, resp);
            } else {
                // Default echo response
                auto response = SlipstreamMessage::create(SlipstreamMessageType::CMD_RESPONSE,
                                                           cmd.payload, cmd.header.sequence_number);
                resp.deposit(std::move(response));
            }
        };
        workers_.push_back(std::make_unique<SlipstreamWorker>(
            static_cast<uint32_t>(i), *ingress_depot_, *egress_depot_, std::move(handler)));
    }
    return true;
}

bool CerberusSlipstreamEngine::start() {
    if (running_.load()) return true;
    if (!ingress_depot_ && !initialize()) return false;
    running_.store(true);
    stop_requested_.store(false);
    for (auto& worker : workers_) {
        worker->start();
    }
    SlipstreamMonitor::instance().start();
    return true;
}

void CerberusSlipstreamEngine::stop() {
    stop_requested_.store(true);
    for (auto& worker : workers_) {
        worker->stop();
    }
    SlipstreamMonitor::instance().stop();
    running_.store(false);
}

void CerberusSlipstreamEngine::registerHandler(SlipstreamMessageType type,
                                                SlipstreamWorker::CommandHandler handler) {
    std::lock_guard<std::mutex> lock(handlers_mutex_);
    handlers_[type] = std::move(handler);
}

CerberusSlipstreamEngine::Stats CerberusSlipstreamEngine::getStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

} // namespace hq::cerberus::slipstream
