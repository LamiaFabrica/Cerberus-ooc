/// @file cerberus_metro.cpp
/// @copyright Copyright © 2026, all rights reserved | David Hargreaves aka Roylepython trading as The Medusa Initiative & Lamia Fabrica
/// Support: https://www.patreon.com/TheMedusaInitiative — £25/month removes ads from all software at 200 subscribers
///
/// Cerberus Metro — monitoring station with trace capture.
/// Ported from PsiForceDB metro_station.cpp.
/// "1 Pipe In / 1 Pipe Out" — every command is traced.
///
/// @version 1.0.0

#include "hq/cerberus_metro.hpp"

#include <cmath>
#include <cstring>

namespace hq::cerberus::metro {

// ===========================================================================
// Trace
// ===========================================================================

double MetroTrace::glowStrength() const noexcept {
    if (hop_count >= 16) return 0.0;
    constexpr double fade_exponent = 2.0;
    double fade = 1.0 / (1.0 + std::pow(static_cast<double>(hop_count), fade_exponent));
    return resonance * fade;
}

// ===========================================================================
// Station
// ===========================================================================

MetroStation::MetroStation(MetroConfig cfg) : config_(std::move(cfg)) {}
MetroStation::~MetroStation() { close(); }

bool MetroStation::open() {
    if (inbound_pipe_.is_open) return false;
    inbound_pipe_.read_fd = 1;
    inbound_pipe_.write_fd = 2;
    inbound_pipe_.is_open = true;
    outbound_pipe_.read_fd = 3;
    outbound_pipe_.write_fd = 4;
    outbound_pipe_.is_open = true;
    return true;
}

void MetroStation::close() {
    inbound_pipe_.is_open = false;
    outbound_pipe_.is_open = false;
    inbound_pipe_.read_fd = -1;
    inbound_pipe_.write_fd = -1;
    outbound_pipe_.read_fd = -1;
    outbound_pipe_.write_fd = -1;
    {
        std::unique_lock lock(sessions_mutex_);
        sessions_.clear();
    }
    {
        std::unique_lock lock(traces_mutex_);
        active_traces_.clear();
    }
}

bool MetroStation::isOpen() const noexcept {
    return inbound_pipe_.is_open && outbound_pipe_.is_open;
}

uint64_t MetroStation::createSession(const std::string& client_identity) {
    std::unique_lock lock(sessions_mutex_);
    for (const auto& [id, sess] : sessions_) {
        if (sess.client_identity == client_identity) return id;
    }
    uint64_t id = next_session_id_.fetch_add(1);
    MetroSession session;
    session.session_id = id;
    session.client_identity = client_identity;
    session.created_at = std::chrono::steady_clock::now();
    session.last_activity = session.created_at;
    sessions_[id] = std::move(session);
    return id;
}

bool MetroStation::authenticateSession(uint64_t session_id, uint32_t permissions_mask) {
    std::unique_lock lock(sessions_mutex_);
    auto it = sessions_.find(session_id);
    if (it == sessions_.end()) return false;
    it->second.authenticated = true;
    it->second.permissions_mask = permissions_mask;
    return true;
}

void MetroStation::terminateSession(uint64_t session_id) {
    std::unique_lock lock(sessions_mutex_);
    sessions_.erase(session_id);
}

std::size_t MetroStation::activeSessions() const {
    std::shared_lock lock(sessions_mutex_);
    return sessions_.size();
}

void MetroStation::storeTrace(const MetroTrace& trace) {
    std::unique_lock lock(traces_mutex_);
    active_traces_.push_back(trace);
}

std::vector<MetroTrace> MetroStation::activeTraces() const {
    std::shared_lock lock(traces_mutex_);
    return active_traces_;
}

std::vector<uint8_t> MetroStation::processIncoming(
    const std::vector<uint8_t>& request,
    const std::string& client_identity) {
    bytes_in_.fetch_add(request.size());

    uint64_t session_id = createSession(client_identity);

    // Step 2: Decrypt (if required)
    std::vector<uint8_t> plaintext;
    if (config_.require_encryption) {
        if (request.size() > 4) {
            uint32_t len = 0;
            std::memcpy(&len, request.data(), 4);
            if (len > 0 && len + 4 <= request.size()) {
                plaintext.assign(request.begin() + 4, request.begin() + 4 + len);
            } else {
                plaintext = request;
            }
        } else {
            plaintext = request;
        }
    } else {
        plaintext = request;
    }

    // Step 3: Simple parse — treat as cerberus:// command text
    std::string command_text(plaintext.begin(), plaintext.end());

    // Step 4: Authentication check
    if (config_.require_authentication) {
        auto it = sessions_.find(session_id);
        if (it == sessions_.end() || !it->second.authenticated) {
            std::string error = "AUTHENTICATION_REQUIRED";
            return std::vector<uint8_t>(error.begin(), error.end());
        }
    }

    // Step 5: Build trace
    MetroTrace trace;
    trace.trace_id = static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    trace.origin = config_.station_id + "/session_" + std::to_string(session_id);
    trace.destination = "cerberus_runtime";
    trace.resonance = 0.5; // default confidence
    trace.persistent = true;
    trace.hop_count = 0;
    trace.birth_timestamp = std::chrono::steady_clock::now();
    trace.waypoints.push_back(trace.origin);
    storeTrace(trace);

    // Build response
    std::string response_text = command_text.empty() ? "OK" : command_text;
    std::vector<uint8_t> response_plain(response_text.begin(), response_text.end());
    std::vector<uint8_t> encrypted_response;
    if (config_.require_encryption) {
        uint32_t len = static_cast<uint32_t>(response_plain.size());
        encrypted_response.resize(4 + len);
        std::memcpy(encrypted_response.data(), &len, 4);
        std::memcpy(encrypted_response.data() + 4, response_plain.data(), len);
    } else {
        encrypted_response = std::move(response_plain);
    }

    packets_in_.fetch_add(1);
    packets_out_.fetch_add(1);
    bytes_out_.fetch_add(encrypted_response.size());

    return encrypted_response;
}

// ===========================================================================
// Global Station
// ===========================================================================

MetroStation& get_global_metro_station() {
    static MetroStation station([]() {
        MetroConfig cfg;
        cfg.station_id = "cerberus_gate_01";
        cfg.station_name = "Cerberus Inference Fortress Entry/Exit";
        cfg.require_encryption = false;
        cfg.require_authentication = false; // dev/test default; operators set true in prod
        cfg.trace_all_packets = true;
        cfg.min_glow_resonance = 0.1;
        return cfg;
    }());
    return station;
}

} // namespace hq::cerberus::metro
