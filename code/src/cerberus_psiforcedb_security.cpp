/// @file cerberus_psiforcedb_security.cpp
/// @copyright Copyright © 2026, all rights reserved | David Hargreaves aka Roylepython trading as The Medusa Initiative & Lamia Fabrica
/// Support: https://www.patreon.com/TheMedusaInitiative — £25/month removes ads from all software at 200 subscribers
///
/// Thin LFSSL_Native_Crypto bridge — SHA256/HMAC/PBKDF2 inline-safe.
/// AES-256-GCM, Kyber, Dilithium backed by cerberus_lfssl DLL/.so when present;
/// sentinel reflects runtime load state.
///
/// @version 2.1.0

#include "hq/cerberus_psiforcedb_security.hpp"

// Platform-specific dynamic library loading
#ifdef _WIN32
using HMOD = void*;
extern "C" __declspec(dllimport) HMOD LoadLibraryA(const char*);
extern "C" __declspec(dllimport) void* GetProcAddress(HMOD, const char*);
extern "C" __declspec(dllimport) int   FreeLibrary(HMOD);
#else
#  include <dlfcn.h>
using HMOD = void*;
#  define LoadLibraryA(p) dlopen(p, RTLD_NOW)
#  define GetProcAddress(h, n) dlsym(h, n)
#  define FreeLibrary(h) dlclose(h)
#endif

namespace hq::cerberus::security {

// ============================================================================
// CryptoBridge — Inline-safe subset only
// ============================================================================

#if CERBERUS_HAS_LFSSL_NATIVE_CRYPTO
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
#else
// Fallback: when LFSSL_Native_Crypto.hpp is unavailable (Linux/WSL without LFSSL)
// the inline primitives are unreachable. We load them from the LFSSL .dll/.so at runtime.
// ============================================================================
// LFSSL Runtime Loader Helper (shared by sha256, hmac_sha256, pbkdf2_sha256)
// ============================================================================
static HMOD load_lfssl_once() {
    static HMOD h = []() -> HMOD {
#ifdef _WIN32
        const char* paths[] = {
            "cerberus_lfssl.dll",
            "../../lfssl_bridge/cerberus_lfssl.dll",
            "../lfssl_bridge/cerberus_lfssl.dll",
            "lfssl_bridge/cerberus_lfssl.dll",
            nullptr
        };
#else
        const char* paths[] = {
            "./libcerberus_lfssl.so",
            "../../lfssl_bridge/libcerberus_lfssl.so",
            "../lfssl_bridge/libcerberus_lfssl.so",
            "lfssl_bridge/libcerberus_lfssl.so",
            nullptr
        };
#endif
        for (auto** p = paths; *p; ++p) {
            HMOD lib = reinterpret_cast<HMOD>(LoadLibraryA(*p));
            if (lib) return lib;
        }
        return nullptr;
    }();
    return h;
}

std::vector<std::uint8_t> CryptoBridge::sha256(const std::string& data) {
    HMOD h = load_lfssl_once();
    if (!h) return {};
    using fn_t = void (*)(const uint8_t*, size_t, uint8_t[32]);
    auto fn = reinterpret_cast<fn_t>(GetProcAddress(h, "cerberus_lfssl_sha256"));
    if (!fn) return {};
    std::vector<std::uint8_t> out(32);
    fn(reinterpret_cast<const uint8_t*>(data.data()), data.size(), out.data());
    return out;
}

std::vector<std::uint8_t> CryptoBridge::hmac_sha256(
    const std::vector<std::uint8_t>& key,
    const std::string& message) {
    HMOD h = load_lfssl_once();
    if (!h) return {};
    using fn_t = void (*)(const uint8_t*, size_t, const uint8_t*, size_t, uint8_t[32]);
    auto fn = reinterpret_cast<fn_t>(GetProcAddress(h, "cerberus_lfssl_hmac_sha256"));
    if (!fn) return {};
    std::vector<std::uint8_t> out(32);
    fn(key.data(), key.size(),
       reinterpret_cast<const uint8_t*>(message.data()), message.size(),
       out.data());
    return out;
}

std::vector<std::uint8_t> CryptoBridge::pbkdf2_sha256(
    const std::string& password,
    const std::vector<std::uint8_t>& salt,
    std::uint32_t iterations,
    std::size_t key_len) {
    HMOD h = load_lfssl_once();
    if (!h) return {};
    using fn_t = int (*)(const uint8_t*, size_t, const uint8_t*, size_t, int, size_t, uint8_t*);
    auto fn = reinterpret_cast<fn_t>(GetProcAddress(h, "cerberus_lfssl_pbkdf2_sha256"));
    if (!fn) return {};
    std::vector<std::uint8_t> out(key_len);
    int rc = fn(reinterpret_cast<const uint8_t*>(password.data()), password.size(),
                salt.data(), salt.size(), static_cast<int>(iterations), key_len, out.data());
    if (rc != 0) return {};
    return out;
}
#endif

// ============================================================================
// Argon2id — runtime LFSSL loading (NOT inline; requires .dll/.so)
// ============================================================================

std::vector<std::uint8_t> CryptoBridge::argon2id(
    std::string_view password,
    const std::vector<std::uint8_t>& salt,
    std::size_t hash_len,
    uint32_t t_cost,
    uint32_t m_cost,
    uint32_t parallelism) {

    using fn_t = int (*)(
        uint32_t, uint32_t, uint32_t,
        const uint8_t*, size_t,
        const uint8_t*, size_t,
        uint8_t*, size_t);

#ifdef _WIN32
    const char* paths[] = {
        "cerberus_lfssl.dll",
        "../../lfssl_bridge/cerberus_lfssl.dll",
        "../lfssl_bridge/cerberus_lfssl.dll",
        "lfssl_bridge/cerberus_lfssl.dll",
        nullptr
    };
#else
    const char* paths[] = {
        "./libcerberus_lfssl.so",
        "../../lfssl_bridge/libcerberus_lfssl.so",
        "../lfssl_bridge/libcerberus_lfssl.so",
        "lfssl_bridge/libcerberus_lfssl.so",
        nullptr
    };
#endif
    for (auto** p = paths; *p; ++p) {
        HMOD h = reinterpret_cast<HMOD>(LoadLibraryA(*p));
        if (!h) continue;
        auto fn = reinterpret_cast<fn_t>(GetProcAddress(h, "cerberus_lfssl_argon2id"));
        if (fn) {
            std::vector<std::uint8_t> hash(hash_len);
            int r = fn(t_cost, m_cost, parallelism,
                       reinterpret_cast<const uint8_t*>(password.data()), password.size(),
                       salt.data(), salt.size(),
                       hash.data(), hash_len);
            FreeLibrary(h);
            if (r == 0) return hash;
        }
        FreeLibrary(h);
    }
    return {}; // LFSSL not found or Argon2id export missing
}

// ============================================================================
// Sentinels — Non-inline LFSSL primitives
// ============================================================================

static bool check_lfssl_dll_present() noexcept {
#ifdef _WIN32
    const char* paths[] = {
        "cerberus_lfssl.dll",
        "../../lfssl_bridge/cerberus_lfssl.dll",
        "../lfssl_bridge/cerberus_lfssl.dll",
        "lfssl_bridge/cerberus_lfssl.dll",
        nullptr
    };
#else
    const char* paths[] = {
        "./libcerberus_lfssl.so",
        "../../lfssl_bridge/libcerberus_lfssl.so",
        "../lfssl_bridge/libcerberus_lfssl.so",
        "lfssl_bridge/libcerberus_lfssl.so",
        nullptr
    };
#endif
    for (auto** p = paths; *p; ++p) {
        HMOD h = reinterpret_cast<HMOD>(LoadLibraryA(*p));
        if (h) {
            FreeLibrary(h);
            return true;
        }
    }
    return false;
}

bool LfsslSentinel::aes256_gcm_available() noexcept {
    static bool present = check_lfssl_dll_present();
    return present;
}

std::string LfsslSentinel::aes256_gcm_unavailable_reason() noexcept {
    if (aes256_gcm_available()) {
        return "AES-256-GCM available via LFSSL runtime library.";
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

bool LfsslSentinel::argon2id_available() noexcept {
    static bool present = check_lfssl_dll_present();
    return present;
}

std::string LfsslSentinel::argon2id_unavailable_reason() noexcept {
    if (argon2id_available()) {
        return "Argon2id available via cerberus_lfssl.dll (LFSSL runtime linked).";
    }
    return "Argon2id requires LFSSL compiled library. "
           "Cerberus delegates memory-hard hashing to PsiForceDB at runtime.";
}

} // namespace hq::cerberus::security
