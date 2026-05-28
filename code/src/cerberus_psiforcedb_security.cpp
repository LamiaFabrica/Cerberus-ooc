/// @file cerberus_psiforcedb_security.cpp
/// @copyright Copyright © 2026, all rights reserved | David Hargreaves aka Roylepython trading as The Medusa Initiative & Lamia Fabrica
/// Support: https://www.patreon.com/TheMedusaInitiative — £25/month removes ads from all software at 200 subscribers
///
/// Thin LFSSL_Native_Crypto bridge — SHA256/HMAC/PBKDF2 inline-safe.
/// AES-256-GCM, Kyber, Dilithium backed by cerberus_lfssl.dll when present;
/// sentinel reflects runtime load state.
///
/// @version 2.0.0

#include "hq/cerberus_psiforcedb_security.hpp"

// Minimal Windows API forward declarations (avoid <windows.h> macro pollution)
using HMODULE = void*;
extern "C" __declspec(dllimport) HMODULE LoadLibraryA(const char*);
extern "C" __declspec(dllimport) void*   GetProcAddress(HMODULE, const char*);
extern "C" __declspec(dllimport) int     FreeLibrary(HMODULE);

namespace hq::cerberus::security {

// ============================================================================
// CryptoBridge — Inline-safe subset only
// ============================================================================

std::vector<std::uint8_t> CryptoBridge::sha256(const std::string& data) {
    return LFSSL::Crypto::SHA256::hash(
        reinterpret_cast<const uint8_t*>(data.data()), data.size());
}

std::vector<std::uint8_t> CryptoBridge::hmac_sha256(
    const std::vector<std::uint8_t>& key,
    const std::string& message) {
    std::vector<std::uint8_t> digest(32);
    LFSSL::Crypto::HMAC_SHA256::compute(
        key.data(), key.size(),
        reinterpret_cast<const uint8_t*>(message.data()), message.size(),
        digest.data());
    return digest;
}

std::vector<std::uint8_t> CryptoBridge::pbkdf2_sha256(
    const std::string& password,
    const std::vector<std::uint8_t>& salt,
    std::uint32_t iterations,
    std::size_t key_len) {
    std::vector<std::uint8_t> derived(key_len);
    LFSSL::Crypto::PBKDF2_SHA256::derive(
        reinterpret_cast<const uint8_t*>(password.data()), password.size(),
        salt.data(), salt.size(),
        static_cast<int>(iterations), key_len, derived.data());
    return derived;
}

// ============================================================================
// Sentinels — Non-inline LFSSL primitives
// ============================================================================

static bool check_lfssl_dll_present() noexcept {
    HMODULE h = LoadLibraryA("cerberus_lfssl.dll");
    if (!h) h = LoadLibraryA("../../lfssl_bridge/cerberus_lfssl.dll");
    if (h) {
        FreeLibrary(h);
        return true;
    }
    return false;
}

bool LfsslSentinel::aes256_gcm_available() noexcept {
    static bool present = check_lfssl_dll_present();
    return present;
}

std::string LfsslSentinel::aes256_gcm_unavailable_reason() noexcept {
    if (aes256_gcm_available()) {
        return "AES-256-GCM available via cerberus_lfssl.dll (LFSSL runtime linked).";
    }
    return "AES-256-GCM implementation is in PsiForceDB LFSSL library "
           "(not inline in LFSSL_Native_Crypto.hpp). "
           "Cerberus delegates encryption to PsiForceDB at runtime.";
}

bool LfsslSentinel::kyber_available() noexcept {
    static bool present = check_lfssl_dll_present();
    return present;
}

std::string LfsslSentinel::kyber_unavailable_reason() noexcept {
    if (kyber_available()) {
        return "KyberKEM available via cerberus_lfssl.dll (LFSSL runtime linked).";
    }
    return "KyberKEM requires LFSSL compiled library. "
           "Cerberus delegates PQC to PsiForceDB at runtime.";
}

bool LfsslSentinel::dilithium_available() noexcept {
    static bool present = check_lfssl_dll_present();
    return present;
}

std::string LfsslSentinel::dilithium_unavailable_reason() noexcept {
    if (dilithium_available()) {
        return "Dilithium available via cerberus_lfssl.dll (LFSSL runtime linked).";
    }
    return "Dilithium requires LFSSL compiled library. "
           "Cerberus delegates PQC to PsiForceDB at runtime.";
}

} // namespace hq::cerberus::security
