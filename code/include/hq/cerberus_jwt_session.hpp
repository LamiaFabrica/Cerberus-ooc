#pragma once
/// @file cerberus_jwt_session.hpp
/// @copyright Copyright (c) 2026 D Hargreaves. All rights reserved.
///
/// Cerberus JWT Session — Persistent Authentication Without Passwords
/// ============================================================
///
/// JWT (JSON Web Token) session management for Cerberus.
///
/// Architecture (copied from PsiForceDB encrypted_jwt.cpp):
///   - Session tokens are HMAC-SHA256 signed, optionally PQC-compatible.
///   - Token expires after configurable lifetime (default: 24 hours).
///   - Token carries claims: sub (user/node), iss (issuer), aud (audience),
///     iat (issued at), exp (expiration), jti (unique token ID).
///   - On each request: validate signature, check expiration, verify audience.
///   - On token refresh: issue new token with extended expiration, invalidate old.
///
/// BOUNDARY:
///   - Real PQC signing (Dilithium-5) is delegated to LFSSL.
///   - This class handles HMAC-SHA256 local tokens and PQC delegation.
///   - No plaintext secrets stored. Only HMAC key material in the Local DB.
///
/// © 2026 D Hargreaves | LamiaFabrica Software

#include <string>
#include <vector>
#include <cstdint>
#include <chrono>
#include <optional>
#include <map>

namespace hq::cerberus::privacy {

// ============================================================================
// JWTHeader
// ============================================================================
struct JWTHeader {
    std::string alg{"HS256"};    // "HS256" or "PQC-DILITHIUM-5"
    std::string typ{"JWT"};
    std::string kid;              // key ID
    std::string enc;              // encryption type (if encrypted)
    std::uint64_t iat{0};
    std::uint64_t exp{0};

    [[nodiscard]] bool is_valid() const noexcept { return !alg.empty() && typ == "JWT" && exp > iat; }
};

// ============================================================================
// JWTPayload
// ============================================================================
struct JWTPayload {
    std::string sub;              // subject (node_id / user_id)
    std::string iss{"cerberus"};  // issuer
    std::vector<std::string> aud; // audience
    std::uint64_t iat{0};         // issued at (unix seconds)
    std::uint64_t exp{0};         // expiration (unix seconds)
    std::string jti;              // unique token ID
    std::map<std::string, std::string> claims; // extra claims

    [[nodiscard]] bool is_expired() const noexcept;
    [[nodiscard]] bool is_audience_allowed(const std::vector<std::string>& allowed) const;
};

// ============================================================================
// JWTParts
// ============================================================================
struct JWTParts {
    std::string encoded_header;
    std::string encoded_payload;
    std::string encoded_signature;
    std::string signing_input;
    bool complete{false};
};

// ============================================================================
// SessionConfig — complete replica of PsiForceDB::Security::SessionConfig
// ============================================================================
struct SessionConfig {
    // HMAC
    std::string jwt_secret;                       // primary HMAC signing secret (from Local DB)
    std::string jwt_secret_key;                   // key-ID label for the secret
    std::string issuer = "cerberus";              // token issuer claim
    std::chrono::seconds access_token_ttl{3600};  // 1 hour
    std::chrono::seconds refresh_token_ttl{604800}; // 7 days
    std::chrono::seconds session_cleanup_interval{300}; // 5 min
    std::chrono::seconds session_duration{3600};  // synonym / default
    std::chrono::seconds token_lifetime{86400}; // 24 hours (Cerberus legacy default)

    // Feature toggles (all defaulted to false until LFSSL links)
    bool enable_refresh_tokens{false};
    bool enable_session_fingerprinting{false};
    bool enable_concurrent_session_limits{false};
    bool enable_hybrid_signatures{false};
    bool enable_pqc_signatures{false};
    bool enable_token_rotation{false};
    bool require_hardware_fingerprint{false};
    bool require_pqc{false};                       // legacy alias

    // Limits
    std::size_t max_concurrent_sessions{0};        // 0 = unlimited / sentinel

    // Persistence
    std::string database_path = "data/sessions.db"; // local session store path

    // PQC key material (delegated to LFSSL)
    std::vector<std::uint8_t> dilithium_public_key;
    std::vector<std::uint8_t> dilithium_private_key;

    // Audience / issuer policy
    std::vector<std::string> allowed_issuers{"cerberus"};
    std::vector<std::string> allowed_audiences{"cerberus", "psiforcedb"};

    // JWKS / revocation (delegated to PsiForceDB MedusaServ)
    std::vector<std::string> jwks_endpoints;
    std::vector<std::string> public_keys;
    std::vector<std::string> revocation_endpoints;

    /// Sentinel: every non-zero, non-empty field is consumed; zeros/empties indicate delegation.
    [[nodiscard]] static std::string unavailable_reason() noexcept {
        return "SessionConfig: Cerberus JWT session parameters. "
               "HMAC-SHA256 is inline. "
               "PQC (Dilithium-5), hardware fingerprinting, JWKS, revocation endpoints, "
               "refresh-token rotation, concurrent-session limits, and token-DB persistence "
               "are all delegated to LFSSL/PsiForceDB at runtime. "
               "On this build host they remain unlinked (sentinel mode).";
    }
};

// ============================================================================
// JWTSession — creates, validates, and refreshes session tokens
// ============================================================================
class JWTSession {
public:
    explicit JWTSession(const SessionConfig& config);

    /// Create a new JWT for a subject (node/user).
    [[nodiscard]] std::string create_token(const std::string& sub,
                                             const std::vector<std::string>& aud = {});

    /// Validate a JWT string. Returns payload if valid, nullopt + error if invalid.
    [[nodiscard]] std::pair<std::optional<JWTPayload>, std::string> validate_token(
        const std::string& jwt);

    /// Refresh a token (issue new with extended expiry, return old token jti for invalidation).
    [[nodiscard]] std::pair<std::string, std::string> refresh_token(const std::string& jwt);

    /// Invalidate a token by jti (record in local revocation set).
    void invalidate_token(const std::string& jti);

    /// Check if a token jti has been revoked.
    [[nodiscard]] bool is_revoked(const std::string& jti) const;

    // MANDATORY unavailable_reason per AGENTS.md
    [[nodiscard]] static std::string unavailable_reason() noexcept;

private:
    SessionConfig config_;
    std::map<std::string, std::chrono::system_clock::time_point> revoked_jtis_;

    JWTParts split_jwt_(const std::string& jwt) const;
    std::string base64_url_encode_(const std::vector<std::uint8_t>& data) const;
    std::vector<std::uint8_t> base64_url_decode_(const std::string& str) const;
    bool verify_hmac_signature_(const JWTParts& parts) const;
    std::string generate_jti_() const;
    std::uint64_t now_seconds_() const;
};

} // namespace hq::cerberus::privacy
