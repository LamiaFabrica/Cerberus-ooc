#pragma once
/// @file cerberus_api_gateway.hpp
/// @copyright Copyright (c) 2026 D Hargreaves. All rights reserved.
///
/// Cerberus API Gateway — ANBP (AI-Native Binary Protocol) entry point.
/// Ported from PsiForceDB ANBP Dome Protocol.
/// Human operator issues commands; AI executes via binary wire.
/// Safety enforced by permission mode gate (ACT / PLAN / EXECUTE / AGENTIC).
///
/// @version 1.0.0

#include <cstdint>
#include <vector>
#include <array>
#include <string>
#include <memory>
#include <chrono>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <optional>
#include <expected>

// Optional production privacy/audit injection (LCMD + RBPC) for the four inference opcodes.
// Mirrors the pattern used in InferenceServer for full symmetry. Caller owns lifetime.
#include "hq/cerberus_local_maintenance_db.hpp"
#include "hq/cerberus_user_security.hpp"

namespace hq::cerberus::gateway {

// ===========================================================================
// Constants
// ===========================================================================

inline constexpr uint32_t CERBERUS_ANBP_MAGIC = 0x414E4250; // "ANBP"
inline constexpr uint16_t CERBERUS_ANBP_VERSION = 0x0001;
inline constexpr uint16_t PORT_STEALTH  = 70;
inline constexpr uint16_t PORT_SLIPSTREAM = 75;
inline constexpr uint16_t PORT_NATIVE   = 9742;
inline constexpr std::size_t MAX_PAYLOAD_SIZE = 64 * 1024 * 1024; // 64 MB
inline constexpr std::size_t SESSION_TOKEN_SIZE = 16;
inline constexpr std::size_t SHARED_SECRET_SIZE = 32;
inline constexpr std::chrono::seconds SESSION_TIMEOUT{300}; // 5 min

// ===========================================================================
// Permission Modes (human-operator enforced)
// ===========================================================================

enum class PermissionMode : uint8_t {
    NONE    = 0, ///< No access
    ACT     = 1, ///< Read-only: status, list, benchmark
    PLAN    = 2, ///< Design: compile, set-backend, enable-fusion
    EXECUTE = 3, ///< Operations: run graph, promote, demote
    AGENTIC = 4  ///< Full control: shutdown, reconfigure, tier thrashing
};

// ===========================================================================
// Binary Header (packed, 28 bytes)
// ===========================================================================

#pragma pack(push, 1)
struct ANBPHeader {
    uint32_t magic{CERBERUS_ANBP_MAGIC};
    uint16_t version{CERBERUS_ANBP_VERSION};
    uint16_t opcode{0};
    uint32_t payload_length{0};
    uint32_t sequence_id{0};
    uint16_t session_token{0};
    uint16_t flags{0};
    uint64_t timestamp_us{0};

    static constexpr uint16_t FLAG_ENCRYPTED  = 0x0001;
    static constexpr uint16_t FLAG_COMPRESSED = 0x0002;
    static constexpr uint16_t FLAG_URGENT     = 0x0004;

    [[nodiscard]] bool isValid() const noexcept { return magic == CERBERUS_ANBP_MAGIC; }
    [[nodiscard]] bool isEncrypted() const noexcept { return flags & FLAG_ENCRYPTED; }
};
static_assert(sizeof(ANBPHeader) == 28, "ANBPHeader must be exactly 28 bytes");
#pragma pack(pop)

// ===========================================================================
// Handshake Messages (packed)
// ===========================================================================

#pragma pack(push, 1)
struct HandshakeInitRequest {
    uint32_t client_version{CERBERUS_ANBP_VERSION};
    uint8_t client_nonce[16]{};
    void generateNonce();
};

struct HandshakeInitResponse {
    uint32_t server_version{CERBERUS_ANBP_VERSION};
    uint8_t server_nonce[16]{};
    uint8_t shared_secret[SHARED_SECRET_SIZE]{}; // simulated Kyber encaps
    uint16_t session_token{0};
    uint32_t session_timeout_sec{300};
    uint8_t challenge[32]{};
    uint64_t challenge_timestamp_us{0};
};

struct HandshakeCompleteRequest {
    uint16_t session_token{0};
    uint8_t auth_proof[32]{};
    uint64_t auth_timestamp_us{0};
};

struct HandshakeCompleteResponse {
    uint16_t session_token{0};
    uint32_t capabilities{0};
    uint64_t auth_timestamp_us{0};

    static constexpr uint32_t CAP_INFERENCE  = 0x0001;
    static constexpr uint32_t CAP_SLIPSTREAM = 0x0002;
    static constexpr uint32_t CAP_TIERING    = 0x0004;
    static constexpr uint32_t CAP_ENCRYPTION = 0x0008;
};
#pragma pack(pop)

// ===========================================================================
// Command Opcodes (Cerberus-specific)
// ===========================================================================

// MinGW headers define ERROR_NOT_FOUND via winerror.h macros; guard enum body.
#ifdef ERROR_NOT_FOUND
#undef ERROR_NOT_FOUND
#endif
#ifdef ERROR
#undef ERROR
#endif

enum class CerberusOpcode : uint16_t {
    RUN_GRAPH       = 0x1000,
    COMPILE_GRAPH   = 0x1001,
    GET_STATUS      = 0x1002,
    LIST_BACKENDS   = 0x1003,
    SET_BACKEND     = 0x1004,
    PROMOTE_TENSOR  = 0x1005,
    DEMOTE_TENSOR   = 0x1006,
    BENCHMARK       = 0x1007,

    SLIPSTREAM_WRITE = 0x2000,
    SLIPSTREAM_FLUSH = 0x2001,
    SLIPSTREAM_STATUS = 0x2002,

    INFERENCE_QUERY  = 0x3000,
    INFERENCE_EXPORT = 0x3001,
    INFERENCE_CLEAR  = 0x3002,
    INFERENCE_STATS  = 0x3003,

    HANDSHAKE_INIT  = 0xF000,
    HANDSHAKE_COMP  = 0xF001,
    SESSION_AUTH    = 0xF002,
    SESSION_CLOSE   = 0xF003,

    ERR_GENERAL   = 0xFF00,
    ERR_AUTH      = 0xFF01,
    ERR_PERMISSION= 0xFF02,
    ERR_NOT_FOUND = 0xFF03,
    ERR_INVALID   = 0xFF04,
    ERR_SESSION   = 0xFF05,
    SYS_SHUTDOWN    = 0xFF06,
};

// ===========================================================================
// Session
// ===========================================================================

struct GatewaySession {
    uint16_t token{0};
    std::array<uint8_t, SHARED_SECRET_SIZE> shared_secret{};
    std::chrono::steady_clock::time_point created_at;
    std::chrono::steady_clock::time_point last_activity;
    PermissionMode mode{PermissionMode::NONE};
    bool authenticated{false};
    std::string client_identity;

    [[nodiscard]] bool isExpired() const {
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - last_activity);
        return elapsed > SESSION_TIMEOUT;
    }
    void updateActivity() { last_activity = std::chrono::steady_clock::now(); }
};

// ===========================================================================
// Protocol Helpers
// ===========================================================================

class ProtocolHelper {
public:
    [[nodiscard]] static std::vector<uint8_t> serializeHeader(const ANBPHeader& header);
    [[nodiscard]] static bool deserializeHeader(const uint8_t* data, std::size_t len, ANBPHeader& header);
    [[nodiscard]] static std::vector<uint8_t> buildMessage(CerberusOpcode opcode,
                                                            uint16_t session_token,
                                                            uint32_t sequence,
                                                            const std::vector<uint8_t>& payload,
                                                            bool encrypt = false,
                                                            const uint8_t* key = nullptr);
    [[nodiscard]] static std::array<uint8_t, 32> computeAuthProof(
        const std::array<uint8_t, SHARED_SECRET_SIZE>& shared_secret,
        const std::string& context);
    [[nodiscard]] static bool validateMessage(const ANBPHeader& header,
                                                const uint8_t* payload,
                                                std::size_t payload_len);
    [[nodiscard]] static uint16_t generateSessionToken();
    [[nodiscard]] static const char* opcodeToString(CerberusOpcode opcode);
};

// ===========================================================================
// API Gateway Engine
// ===========================================================================

class CerberusApiGateway {
public:
    CerberusApiGateway();
    ~CerberusApiGateway();

    // Initialize/shutdown
    bool initialize();
    void shutdown();
    [[nodiscard]] bool isInitialized() const noexcept { return initialized_; }

    // Handle incoming ANBP request → outgoing ANBP response
    [[nodiscard]] std::vector<uint8_t> handleRequest(const uint8_t* data, std::size_t len);

    // Session management
    [[nodiscard]] uint16_t createSession();
    bool closeSession(uint16_t token);
    GatewaySession* getSession(uint16_t token);
    void cleanupExpiredSessions();
    [[nodiscard]] std::size_t sessionCount() const;

    // Human operator: set permission mode for a session
    bool setPermissionMode(uint16_t token, PermissionMode mode);

    // Encode helpers
    [[nodiscard]] std::vector<uint8_t> encodeResponse(CerberusOpcode opcode,
                                                         uint16_t session_token,
                                                         uint32_t sequence_id,
                                                         const std::vector<uint8_t>& payload);
    [[nodiscard]] std::vector<uint8_t> encodeError(CerberusOpcode original_opcode,
                                                    uint32_t sequence_id,
                                                    uint16_t session_token,
                                                    CerberusOpcode error_opcode,
                                                    const std::string& message);

    // Production privacy injection for inference opcodes (LCMD + RBPC).
    // Call before handling traffic if you want real audit + RBPC on EXPORT/CLEAR.
    void setPrivacyContext(std::shared_ptr<hq::cerberus::privacy::LocalMaintenanceDB> lcmd,
                           std::shared_ptr<hq::cerberus::privacy::UserSecurity> us,
                           std::string node_id = "local");

private:
    bool initialized_{false};
    uint16_t next_session_token_{1};
    std::unordered_map<uint16_t, GatewaySession> sessions_;
    mutable std::mutex sessions_mutex_;

    // Privacy surface (optional)
    std::shared_ptr<hq::cerberus::privacy::LocalMaintenanceDB> lcmd_;
    std::shared_ptr<hq::cerberus::privacy::UserSecurity> user_security_;
    std::string rbpc_node_id_{"local"};
};

} // namespace hq::cerberus::gateway
