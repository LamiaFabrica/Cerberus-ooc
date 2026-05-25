/**
 * Crypto Audit Log Shim
 * =======================
 * Minimal compatibility header for PsiForceDB extension_interface.hpp.
 *
 * The real implementation (TamperProofAuditLog, CryptoAuditLog,
 * AuditLoggedSecureRandom) lives in PsiForceDB's security subsystem.
 * This shim provides enough declarations so that
 * extension_interface.hpp compiles in the Cerberus build tree.
 *
 * Cerberus does NOT implement its own audit logging — it consumes
 * PsiForceDB's TamperProofAuditLog at runtime.
 *
 * © 2026 D Hargreaves | Yorkshire Champion Standards
 */

#ifndef PSIFORCEDB_SECURITY_CRYPTO_AUDIT_LOG_HPP
#define PSIFORCEDB_SECURITY_CRYPTO_AUDIT_LOG_HPP

#include <vector>
#include <cstdint>
#include <cstddef>
#include <string>
#include <source_location>

namespace PsiForceDB {
namespace Security {

// Minimal AuditLoggedSecureRandom — delegates to LFSSL::Crypto::SecureRandom
class AuditLoggedSecureRandom {
public:
    static std::vector<uint8_t> bytes(size_t length,
                                      const std::string& purpose = "shim",
                                      std::source_location loc = std::source_location::current()) {
        (void)purpose;
        (void)loc;
        return LFSSL::Crypto::secure_random_bytes(length);
    }
    static bool fill(void* buffer, size_t length,
                     const std::string& purpose = "shim",
                     std::source_location loc = std::source_location::current()) {
        (void)purpose;
        (void)loc;
        return LFSSL::Crypto::secure_random_bytes(buffer, length);
    }
    static std::string hexString(size_t byte_length,
                               const std::string& purpose = "shim",
                               std::source_location loc = std::source_location::current()) {
        (void)purpose;
        (void)loc;
        auto data = bytes(byte_length);
        std::string out;
        out.reserve(byte_length * 2);
        for (auto b : data) {
            char hex[3];
            std::snprintf(hex, sizeof(hex), "%02x", static_cast<int>(b));
            out += hex;
        }
        return out;
    }
    static uint64_t uint64(const std::string& purpose = "shim",
                           std::source_location loc = std::source_location::current()) {
        (void)purpose;
        (void)loc;
        auto data = bytes(sizeof(uint64_t));
        uint64_t v = 0;
        for (size_t i = 0; i < sizeof(uint64_t); ++i) {
            v |= static_cast<uint64_t>(data[i]) << (8 * i);
        }
        return v;
    }
    static uint32_t uint32(const std::string& purpose = "shim",
                           std::source_location loc = std::source_location::current()) {
        (void)purpose;
        (void)loc;
        auto data = bytes(sizeof(uint32_t));
        uint32_t v = 0;
        for (size_t i = 0; i < sizeof(uint32_t); ++i) {
            v |= static_cast<uint32_t>(data[i]) << (8 * i);
        }
        return v;
    }
};

// Minimal CryptoAuditLog stub
class CryptoAuditLog {
public:
    static CryptoAuditLog& instance() {
        static CryptoAuditLog inst;
        return inst;
    }
    void logRandomBytes(size_t byte_count,
                        const std::string& purpose,
                        std::source_location loc = std::source_location::current()) {
        (void)byte_count; (void)purpose; (void)loc;
    }
    void logFailure(const std::string& operation,
                    const std::string& reason,
                    std::source_location loc = std::source_location::current()) {
        (void)operation; (void)reason; (void)loc;
    }
    void logHealthCheck(bool passed, const std::string& detail) {
        (void)passed; (void)detail;
    }
    bool verifyIntegrity() const { return true; }
private:
    CryptoAuditLog() = default;
    ~CryptoAuditLog() = default;
    CryptoAuditLog(const CryptoAuditLog&) = delete;
    CryptoAuditLog& operator=(const CryptoAuditLog&) = delete;
};

} // namespace Security
} // namespace PsiForceDB

#endif // PSIFORCEDB_SECURITY_CRYPTO_AUDIT_LOG_HPP
