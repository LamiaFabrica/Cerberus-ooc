/// @file cerberus_api_gateway.cpp
/// @copyright Copyright © 2026, all rights reserved | David Hargreaves aka Roylepython trading as The Medusa Initiative & Lamia Fabrica
/// Support: https://www.patreon.com/TheMedusaInitiative — £25/month removes ads from all software at 200 subscribers
///
/// Cerberus API Gateway — ANBP protocol implementation.
/// Ported from PsiForceDB anbp_dome_protocol.cpp.
///
/// @version 1.0.0

#include "hq/cerberus_api_gateway.hpp"
#include "hq/cerberus_runtime.hpp"

#include <random>
#include <limits>
#include <cstring>
#include <filesystem>
#include <chrono>
#include <ctime>

namespace hq::cerberus::gateway {

// ===========================================================================
// Handshake
// ===========================================================================

void HandshakeInitRequest::generateNonce() {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dis;
    for (std::size_t i = 0; i < 16; ++i) {
        client_nonce[i] = static_cast<uint8_t>(dis(gen) & 0xFFu);
    }
}

// ===========================================================================
// Protocol Helpers
// ===========================================================================

std::vector<uint8_t> ProtocolHelper::serializeHeader(const ANBPHeader& header) {
    std::vector<uint8_t> bytes(sizeof(ANBPHeader));
    std::memcpy(bytes.data(), &header, sizeof(ANBPHeader));
    return bytes;
}

bool ProtocolHelper::deserializeHeader(const uint8_t* data, std::size_t len, ANBPHeader& header) {
    if (data == nullptr || len < sizeof(ANBPHeader)) return false;
    ANBPHeader candidate;
    std::memcpy(&candidate, data, sizeof(ANBPHeader));
    bool magic_valid = candidate.magic == CERBERUS_ANBP_MAGIC;
    bool version_valid = candidate.version == CERBERUS_ANBP_VERSION;
    bool payload_ok = candidate.payload_length <= MAX_PAYLOAD_SIZE
                       && len >= sizeof(ANBPHeader) + candidate.payload_length;
    if (magic_valid && version_valid && payload_ok) {
        header = candidate;
        return true;
    }
    return false;
}

std::vector<uint8_t> ProtocolHelper::buildMessage(CerberusOpcode opcode,
                                                   uint16_t session_token,
                                                   uint32_t sequence,
                                                   const std::vector<uint8_t>& payload,
                                                   bool encrypt,
                                                   const uint8_t* key) {
    std::vector<uint8_t> message;
    if (payload.size() > MAX_PAYLOAD_SIZE || (encrypt && key == nullptr)) {
        return message;
    }

    ANBPHeader header;
    header.magic = CERBERUS_ANBP_MAGIC;
    header.version = CERBERUS_ANBP_VERSION;
    header.opcode = static_cast<uint16_t>(opcode);
    header.payload_length = static_cast<uint32_t>(payload.size());
    header.sequence_id = sequence;
    header.session_token = session_token;
    header.flags = encrypt ? ANBPHeader::FLAG_ENCRYPTED : 0;
    header.timestamp_us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());

    message.resize(sizeof(ANBPHeader) + payload.size());
    std::memcpy(message.data(), &header, sizeof(ANBPHeader));
    if (!payload.empty()) {
        std::memcpy(message.data() + sizeof(ANBPHeader), payload.data(), payload.size());
        if (encrypt) {
            for (std::size_t i = 0; i < payload.size(); ++i) {
                message[sizeof(ANBPHeader) + i] ^= key[i % SHARED_SECRET_SIZE];
            }
        }
    }
    return message;
}

std::array<uint8_t, 32> ProtocolHelper::computeAuthProof(
    const std::array<uint8_t, SHARED_SECRET_SIZE>& shared_secret,
    const std::string& context) {
    std::array<uint8_t, 32> proof{};
    uint64_t state = 1469598103934665603ull; // FNV-1a offset basis
    for (uint8_t byte : shared_secret) {
        state ^= byte;
        state *= 1099511628211ull;
    }
    for (unsigned char ch : context) {
        state ^= ch;
        state *= 1099511628211ull;
    }
    for (std::size_t i = 0; i < proof.size(); ++i) {
        state ^= (state >> 33);
        state *= 0xff51afd7ed558ccdull;
        state ^= (state >> 33);
        proof[i] = static_cast<uint8_t>((state >> ((i % 8) * 8)) & 0xFFu);
    }
    return proof;
}

bool ProtocolHelper::validateMessage(const ANBPHeader& header,
                                      const uint8_t* payload,
                                      std::size_t payload_len) {
    bool header_ok = header.magic == CERBERUS_ANBP_MAGIC
                  && header.version == CERBERUS_ANBP_VERSION
                  && header.payload_length <= MAX_PAYLOAD_SIZE;
    bool payload_ok = header.payload_length == payload_len
                   && (payload_len == 0 || payload != nullptr);
    return header_ok && payload_ok;
}

uint16_t ProtocolHelper::generateSessionToken() {
    static std::atomic<uint32_t> counter{1};
    static std::random_device rd;
    uint16_t token = 0;
    while (token == 0) {
        uint32_t mixed = counter.fetch_add(1, std::memory_order_relaxed)
            ^ (static_cast<uint32_t>(rd()) << 1);
        token = static_cast<uint16_t>(mixed & 0xFFFFu);
    }
    return token;
}

const char* ProtocolHelper::opcodeToString(CerberusOpcode opcode) {
    switch (opcode) {
        case CerberusOpcode::RUN_GRAPH:       return "RUN_GRAPH";
        case CerberusOpcode::COMPILE_GRAPH:   return "COMPILE_GRAPH";
        case CerberusOpcode::GET_STATUS:      return "GET_STATUS";
        case CerberusOpcode::LIST_BACKENDS:   return "LIST_BACKENDS";
        case CerberusOpcode::SET_BACKEND:     return "SET_BACKEND";
        case CerberusOpcode::PROMOTE_TENSOR:  return "PROMOTE_TENSOR";
        case CerberusOpcode::DEMOTE_TENSOR:   return "DEMOTE_TENSOR";
        case CerberusOpcode::BENCHMARK:       return "BENCHMARK";
        case CerberusOpcode::SLIPSTREAM_WRITE: return "SLIPSTREAM_WRITE";
        case CerberusOpcode::SLIPSTREAM_FLUSH: return "SLIPSTREAM_FLUSH";
        case CerberusOpcode::SLIPSTREAM_STATUS: return "SLIPSTREAM_STATUS";
        case CerberusOpcode::INFERENCE_QUERY:   return "INFERENCE_QUERY";
        case CerberusOpcode::INFERENCE_EXPORT:  return "INFERENCE_EXPORT";
        case CerberusOpcode::INFERENCE_CLEAR:   return "INFERENCE_CLEAR";
        case CerberusOpcode::INFERENCE_STATS:   return "INFERENCE_STATS";
        case CerberusOpcode::HANDSHAKE_INIT:  return "HANDSHAKE_INIT";
        case CerberusOpcode::HANDSHAKE_COMP:  return "HANDSHAKE_COMP";
        case CerberusOpcode::SESSION_AUTH:    return "SESSION_AUTH";
        case CerberusOpcode::SESSION_CLOSE:   return "SESSION_CLOSE";
        case CerberusOpcode::ERROR_GENERAL:   return "ERROR_GENERAL";
        case CerberusOpcode::ERROR_AUTH:      return "ERROR_AUTH";
        case CerberusOpcode::ERROR_PERMISSION: return "ERROR_PERMISSION";
        case CerberusOpcode::ERROR_NOT_FOUND:  return "ERROR_NOT_FOUND";
        case CerberusOpcode::ERROR_INVALID:    return "ERROR_INVALID";
        case CerberusOpcode::ERROR_SESSION:    return "ERROR_SESSION";
        case CerberusOpcode::SYS_SHUTDOWN:     return "SYS_SHUTDOWN";
    }
    return "UNKNOWN";
}

// ===========================================================================
// API Gateway Engine
// ===========================================================================

CerberusApiGateway::CerberusApiGateway() = default;
CerberusApiGateway::~CerberusApiGateway() { shutdown(); }

bool CerberusApiGateway::initialize() {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    if (initialized_) return true;
    next_session_token_ = std::max<uint16_t>(next_session_token_, 1);
    initialized_ = true;
    return true;
}

void CerberusApiGateway::shutdown() {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    sessions_.clear();
    initialized_ = false;
}

uint16_t CerberusApiGateway::createSession() {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    uint16_t token = next_session_token_;
    while (token == 0 || sessions_.contains(token)) {
        token = ProtocolHelper::generateSessionToken();
    }
    GatewaySession session;
    session.token = token;
    session.created_at = std::chrono::steady_clock::now();
    session.last_activity = session.created_at;
    session.authenticated = false;
    session.mode = PermissionMode::NONE;
    sessions_[token] = std::move(session);
    next_session_token_ = static_cast<uint16_t>(token + 1);
    if (next_session_token_ == 0) next_session_token_ = 1;
    return token;
}

bool CerberusApiGateway::closeSession(uint16_t token) {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    return sessions_.erase(token) > 0;
}

GatewaySession* CerberusApiGateway::getSession(uint16_t token) {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    auto it = sessions_.find(token);
    if (it != sessions_.end() && !it->second.isExpired()) {
        it->second.updateActivity();
        return &it->second;
    }
    return nullptr;
}

void CerberusApiGateway::cleanupExpiredSessions() {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    for (auto it = sessions_.begin(); it != sessions_.end();) {
        if (it->second.isExpired()) {
            it = sessions_.erase(it);
        } else {
            ++it;
        }
    }
}

std::size_t CerberusApiGateway::sessionCount() const {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    return sessions_.size();
}

bool CerberusApiGateway::setPermissionMode(uint16_t token, PermissionMode mode) {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    auto it = sessions_.find(token);
    if (it == sessions_.end()) return false;
    it->second.mode = mode;
    return true;
}

std::vector<uint8_t> CerberusApiGateway::encodeResponse(CerberusOpcode opcode,
                                                       uint16_t session_token,
                                                       uint32_t sequence_id,
                                                       const std::vector<uint8_t>& payload) {
    return ProtocolHelper::buildMessage(opcode, session_token, sequence_id, payload);
}

std::vector<uint8_t> CerberusApiGateway::encodeError(CerberusOpcode original_opcode,
                                                    uint32_t sequence_id,
                                                    uint16_t session_token,
                                                    CerberusOpcode /* unused error_opcode */,
                                                    const std::string& message) {
    // Build error payload: status (4) + msg_len (4) + msg
    std::vector<uint8_t> payload(8 + message.size());
    uint32_t status = static_cast<uint32_t>(1); // error status
    uint32_t msg_len = static_cast<uint32_t>(message.size());
    std::memcpy(payload.data(), &status, 4);
    std::memcpy(payload.data() + 4, &msg_len, 4);
    if (!message.empty()) {
        std::memcpy(payload.data() + 8, message.data(), message.size());
    }
    return ProtocolHelper::buildMessage(original_opcode, session_token, sequence_id, payload);
}

std::vector<uint8_t> CerberusApiGateway::handleRequest(const uint8_t* data, std::size_t len) {
    if (!initialized_) {
        return encodeError(CerberusOpcode::ERROR_GENERAL, 0, 0, CerberusOpcode::ERROR_GENERAL,
                           "Gateway not initialized");
    }

    ANBPHeader header;
    bool parsed = ProtocolHelper::deserializeHeader(data, len, header);
    if (!parsed) {
        return encodeError(CerberusOpcode::ERROR_GENERAL, 0, 0, CerberusOpcode::ERROR_GENERAL,
                           "Invalid ANBP header");
    }

    const uint8_t* payload = data + sizeof(ANBPHeader);
    std::size_t payload_len = header.payload_length;

    cleanupExpiredSessions();

    CerberusOpcode opcode = static_cast<CerberusOpcode>(header.opcode);

    // HANDSHAKE_INIT
    if (opcode == CerberusOpcode::HANDSHAKE_INIT) {
        uint16_t token = createSession();
        HandshakeInitResponse resp{};
        resp.server_version = CERBERUS_ANBP_VERSION;
        resp.session_token = token;
        resp.session_timeout_sec = static_cast<uint32_t>(SESSION_TIMEOUT.count());
        // Fill shared_secret with deterministic placeholder for testing
        for (std::size_t i = 0; i < SHARED_SECRET_SIZE; ++i) resp.shared_secret[i] = static_cast<uint8_t>(i);
        std::vector<uint8_t> resp_bytes(sizeof(resp));
        std::memcpy(resp_bytes.data(), &resp, sizeof(resp));
        return encodeResponse(CerberusOpcode::HANDSHAKE_INIT, token, header.sequence_id, resp_bytes);
    }

    // Validate session for non-handshake
    if (opcode != CerberusOpcode::HANDSHAKE_INIT && opcode != CerberusOpcode::SESSION_AUTH
        && header.session_token != 0) {
        auto* sess = getSession(header.session_token);
        if (!sess) {
            return encodeError(opcode, header.sequence_id, header.session_token,
                               CerberusOpcode::ERROR_SESSION, "Session unknown or expired");
        }
        if (!sess->authenticated && opcode != CerberusOpcode::SESSION_AUTH) {
            return encodeError(opcode, header.sequence_id, header.session_token,
                               CerberusOpcode::ERROR_AUTH, "Authentication required");
        }
    }

    // HANDSHAKE_COMP / SESSION_AUTH
    if (opcode == CerberusOpcode::HANDSHAKE_COMP || opcode == CerberusOpcode::SESSION_AUTH) {
        auto* sess = getSession(header.session_token);
        if (!sess) {
            return encodeError(opcode, header.sequence_id, header.session_token,
                               CerberusOpcode::ERROR_SESSION, "Session unknown");
        }
        sess->authenticated = true;
        sess->mode = PermissionMode::ACT;

        HandshakeCompleteResponse resp{};
        resp.session_token = header.session_token;
        resp.capabilities = HandshakeCompleteResponse::CAP_INFERENCE
                          | HandshakeCompleteResponse::CAP_SLIPSTREAM
                          | HandshakeCompleteResponse::CAP_TIERING
                          | HandshakeCompleteResponse::CAP_ENCRYPTION;
        resp.auth_timestamp_us = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
        std::vector<uint8_t> resp_bytes(sizeof(resp));
        std::memcpy(resp_bytes.data(), &resp, sizeof(resp));
        return encodeResponse(CerberusOpcode::HANDSHAKE_COMP, header.session_token, header.sequence_id, resp_bytes);
    }

    // SESSION_CLOSE
    if (opcode == CerberusOpcode::SESSION_CLOSE) {
        bool closed = closeSession(header.session_token);
        uint32_t status = closed ? 0u : 1u;
        std::vector<uint8_t> resp(4);
        std::memcpy(resp.data(), &status, 4);
        return encodeResponse(CerberusOpcode::SESSION_CLOSE, header.session_token, header.sequence_id, resp);
    }

    // Permission gate: map opcode → required mode
    PermissionMode required = PermissionMode::NONE;
    switch (opcode) {
        case CerberusOpcode::GET_STATUS:
        case CerberusOpcode::LIST_BACKENDS:
        case CerberusOpcode::BENCHMARK:
            required = PermissionMode::ACT; break;
        case CerberusOpcode::COMPILE_GRAPH:
        case CerberusOpcode::SET_BACKEND:
            required = PermissionMode::PLAN; break;
        case CerberusOpcode::RUN_GRAPH:
        case CerberusOpcode::PROMOTE_TENSOR:
        case CerberusOpcode::DEMOTE_TENSOR:
        case CerberusOpcode::SLIPSTREAM_WRITE:
        case CerberusOpcode::SLIPSTREAM_FLUSH:
            required = PermissionMode::EXECUTE; break;
        case CerberusOpcode::INFERENCE_QUERY:
        case CerberusOpcode::INFERENCE_STATS:
            required = PermissionMode::ACT; break;
        case CerberusOpcode::INFERENCE_EXPORT:
        case CerberusOpcode::INFERENCE_CLEAR:
            required = PermissionMode::EXECUTE; break;
        case CerberusOpcode::SYS_SHUTDOWN:
            required = PermissionMode::AGENTIC; break;
        default:
            required = PermissionMode::ACT; break;
    }

    auto* sess = getSession(header.session_token);
    if (!sess || static_cast<uint8_t>(sess->mode) < static_cast<uint8_t>(required)) {
        return encodeError(opcode, header.sequence_id, header.session_token,
                           CerberusOpcode::ERROR_PERMISSION,
                           std::string("Permission denied: requires mode ") +
                           std::to_string(static_cast<int>(required)) +
                           " but session has mode " +
                           std::to_string(sess ? static_cast<int>(sess->mode) : -1));
    }

    // =======================================================================
    // Real inference audit opcode handling (LCMD + RBPC)
    // No echo stubs. Full symmetry with the HTTP surface.
    // =======================================================================
    if (opcode == CerberusOpcode::INFERENCE_QUERY || opcode == CerberusOpcode::INFERENCE_STATS) {
        if (!lcmd_) {
            return encodeError(opcode, header.sequence_id, header.session_token,
                               CerberusOpcode::ERROR_INTERNAL, "inference audit not configured");
        }
        if (opcode == CerberusOpcode::INFERENCE_STATS) {
            auto m = lcmd_->inference_stats();
            std::string s = "{";
            size_t i = 0;
            for (const auto& [k, v] : m) {
                if (i++) s += ",";
                s += "\"" + k + "\":\"" + v + "\"";
            }
            s += "}";
            std::vector<uint8_t> pl(s.begin(), s.end());
            return encodeResponse(opcode, header.session_token, header.sequence_id, pl);
        } else {
            // QUERY: return simple count + last few (binary: 4-byte count + JSON-ish records)
            auto recs = lcmd_->query_inference_records(0, time(nullptr) + 86400, 5);
            std::string s = "[";
            for (size_t i = 0; i < recs.size(); ++i) {
                if (i) s += ",";
                s += "{\"id\":\"" + recs[i].inference_id + "\",\"status\":\"" + recs[i].status + "\"}";
            }
            s += "]";
            std::vector<uint8_t> pl(s.begin(), s.end());
            return encodeResponse(opcode, header.session_token, header.sequence_id, pl);
        }
    }

    if (opcode == CerberusOpcode::INFERENCE_EXPORT || opcode == CerberusOpcode::INFERENCE_CLEAR) {
        if (!lcmd_) {
            return encodeError(opcode, header.sequence_id, header.session_token,
                               CerberusOpcode::ERROR_INTERNAL, "inference audit not configured");
        }

        // Minimal binary payload: [u16 pin_len][pin][u16 word_len][word]
        // (production callers use the same format on both HTTP JSON and ANBP)
        auto parse_len_str = [](const uint8_t* p, size_t& off, size_t total) -> std::pair<std::string, bool> {
            if (off + 2 > total) return {{}, false};
            uint16_t len = 0;
            memcpy(&len, p + off, 2); off += 2;
            if (off + len > total) return {{}, false};
            std::string str(reinterpret_cast<const char*>(p + off), len);
            off += len;
            return {str, true};
        };

        size_t off = 0;
        auto [pin, pin_ok] = parse_len_str(payload, off, payload_len);
        auto [word, word_ok] = parse_len_str(payload, off, payload_len);

        if (user_security_) {
            if (!pin_ok || !word_ok || pin.empty() || word.empty()) {
                return encodeError(opcode, header.sequence_id, header.session_token,
                                   CerberusOpcode::ERROR_PERMISSION, "RBPC pin+word required (binary len+data)");
            }
            std::string rbpc_res = user_security_->verify_confirmation(rbpc_node_id_, pin, word);
            if (!rbpc_res.empty()) {
                bool burned = rbpc_res.find("burned") != std::string::npos || rbpc_res.find("LOCKED") != std::string::npos;
                std::string msg = rbpc_res + (burned ? " [BURNED]" : "");
                return encodeError(opcode, header.sequence_id, header.session_token,
                                   CerberusOpcode::ERROR_PERMISSION, msg);
            }
        }

        if (opcode == CerberusOpcode::INFERENCE_EXPORT) {
            auto ts = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
            std::filesystem::path p = std::filesystem::temp_directory_path() / ("cerberus_anbp_export_" + std::to_string(ts) + ".json");
            bool ok = lcmd_->export_inference_json(p);
            std::string resp = ok ? ("{\"exported\":true,\"path\":\"" + p.string() + "\"}") : "{\"exported\":false}";
            std::vector<uint8_t> pl(resp.begin(), resp.end());
            return encodeResponse(opcode, header.session_token, header.sequence_id, pl);
        } else {
            size_t before = 0;
            auto sm = lcmd_->inference_stats();
            if (auto it = sm.find("count"); it != sm.end()) { try { before = std::stoull(it->second); } catch (...) {} }
            bool cleared = lcmd_->clear_inference_records();
            std::string resp = "{\"cleared\":" + std::string(cleared ? "true" : "false") + ",\"before\":" + std::to_string(before) + "}";
            std::vector<uint8_t> pl(resp.begin(), resp.end());
            return encodeResponse(opcode, header.session_token, header.sequence_id, pl);
        }
    }

    // Fallback for all other opcodes (existing behavior for compile/run_graph etc.)
    std::string command_text(reinterpret_cast<const char*>(payload), payload_len);
    std::vector<uint8_t> response_payload;
    response_payload.resize(payload_len);
    std::memcpy(response_payload.data(), payload, payload_len);

    return encodeResponse(opcode, header.session_token, header.sequence_id, response_payload);
}

void CerberusApiGateway::setPrivacyContext(std::shared_ptr<hq::cerberus::LocalMaintenanceDB> lcmd,
                                           std::shared_ptr<hq::cerberus::UserSecurity> us,
                                           std::string node_id) {
    lcmd_ = std::move(lcmd);
    user_security_ = std::move(us);
    rbpc_node_id_ = node_id.empty() ? "local" : std::move(node_id);
}

} // namespace hq::cerberus::gateway
