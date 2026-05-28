#pragma once
/// @file cerberus_psiforcedb_security.hpp
/// @copyright Copyright (c) 2026 D Hargreaves (AKA Roylepython). LamiaFabrica. All rights reserved.
///
/// Cerberus — Thin LFSSL_Native_Crypto Bridge (SHA256/HMAC/PBKDF2 only)
///
/// THIS IS NOT A STANDALONE SECURITY LAYER.
/// Cerberus is a PsiForceDB MultiModelExtension (type "EXT", model "inference").
/// Authentication, authorization, audit logging, and session management are
/// provided BY PsiForceDB:
///   - FortressAuth         → RBAC, password hashing, account lifecycle
///   - JWTSessionValidator  → token validation, session fingerprinting
///   - TamperProofAuditLog  → cryptographic audit chain
///
/// This header provides ONLY thin wrappers around the truly inline portions of
/// LFSSL_Native_Crypto.hpp:
///   - SHA256         (fully inline, FIPS 180-4 compliant)
///   - HMAC-SHA256    (fully inline, RFC 2104 compliant)
///   - PBKDF2-SHA256  (fully inline, RFC 2898 / PKCS#5 compliant)
///
/// AES-256-GCM, SHA-384, SHA-512, Kyber, and Dilithium are NOT inline in
/// LFSSL_Native_Crypto.hpp; they live in PsiForceDB's LFSSL library.
/// Cerberus delegates those to PsiForceDB at runtime.
///
/// @version 1.0.0

#include <cstdint>
#include <string>
#include <vector>
#include <string>
#include <optional>

// LFSSL Native Crypto — SHA256/HMAC/PBKDF2 are inline; AES/GCM is NOT.
// On Linux/WSL, LFSSL may not be present in the include path — guard the include.
#if __has_include(<lfssl/LFSSL_Native_Crypto.hpp>)
#  include <lfssl/LFSSL_Native_Crypto.hpp>
#  define CERBERUS_HAS_LFSSL_NATIVE_CRYPTO 1
#else
#  pragma message("LFSSL_Native_Crypto.hpp not found — CryptoBridge uses runtime DLL fallback only")
#  define CERBERUS_HAS_LFSSL_NATIVE_CRYPTO 0
#endif

namespace hq::cerberus::security {

// ============================================================================
// LFSSL Native Crypto Wrappers — Inline-safe subset only
// ============================================================================

struct CryptoBridge {
    // SHA256 — returns 32-byte digest (FIPS 180-4, fully inline)
    static std::vector<std::uint8_t> sha256(const std::string& data);

    // HMAC-SHA256 — returns 32-byte MAC (RFC 2104, fully inline)
    static std::vector<std::uint8_t> hmac_sha256(const std::vector<std::uint8_t>& key,
                                                 const std::string& message);

    // PBKDF2-SHA256 (RFC 2898 / PKCS#5, fully inline)
    static std::vector<std::uint8_t> pbkdf2_sha256(const std::string& password,
                                                   const std::vector<std::uint8_t>& salt,
                                                   std::uint32_t iterations,
                                                   std::size_t key_len);
};

// ============================================================================
// Sentinels for non-inline LFSSL primitives
// ============================================================================
// These functions are declared in LFSSL_Native_Crypto.hpp but NOT defined
// inline. The real implementations live in PsiForceDB's LFSSL library.
// Cerberus surfaces the gap with documented sentinels per AGENTS.md.
// ============================================================================

// Alias for backward compat and for new privacy code that names the full class
using LFSSL_Native_Crypto = CryptoBridge;

struct LfsslSentinel {
    // AES-256-GCM: declared but not inline in LFSSL_Native_Crypto.hpp
    static bool aes256_gcm_available() noexcept;
    static std::string aes256_gcm_unavailable_reason() noexcept;

    // Kyber KEM: declared but not inline (PQC requires LFSSL Linux .so)
    static bool kyber_available() noexcept;
    static std::string kyber_unavailable_reason() noexcept;

    // Dilithium: declared but not inline (PQC requires LFSSL Linux .so)
    static bool dilithium_available() noexcept;
    static std::string dilithium_unavailable_reason() noexcept;
};

} // namespace hq::cerberus::security
