/**
 * LFSSL Random Shim
 * ==================
 * Documented placeholder for PsiForceDB header compatibility on Windows.
 *
 * The real LFSSL SecureRandom implementation lives in PsiForceDB's compiled
 * LFSSL library (Linux .so). This shim provides the minimal declarations
 * required so that PsiForceDB headers (e.g., extension_interface.hpp) compile
 * in the Cerberus build tree. It is NOT a production-grade RNG.
 *
 * Once the LFSSL Windows build is available, this shim will be replaced.
 *
 * © 2026 D Hargreaves | Yorkshire Champion Standards
 */

#ifndef LFSSL_CRYPTO_RANDOM_HPP
#define LFSSL_CRYPTO_RANDOM_HPP

#include <vector>
#include <cstdint>
#include <cstddef>
#include <memory>
#include <random>
#include <stdexcept>

namespace LFSSL {
namespace Crypto {

class SecureRandom {
public:
    static SecureRandom& instance() {
        static SecureRandom inst;
        return inst;
    }
    std::vector<uint8_t> bytes(size_t length) {
        std::vector<uint8_t> result(length);
        fill(result.data(), length);
        return result;
    }
    bool fill(void* buffer, size_t length) {
        if (!buffer || length == 0) return true;
        try {
            std::random_device rd;
            uint8_t* out = static_cast<uint8_t*>(buffer);
            for (size_t i = 0; i < length; ++i) {
                out[i] = static_cast<uint8_t>(rd() & 0xFF);
            }
            return true;
        } catch (...) {
            return false;
        }
    }
    uint32_t u32() {
        std::vector<uint8_t> b = bytes(4);
        uint32_t v = 0;
        for (size_t i = 0; i < 4; ++i) v |= static_cast<uint32_t>(b[i]) << (8 * i);
        return v;
    }
    uint64_t u64() {
        std::vector<uint8_t> b = bytes(8);
        uint64_t v = 0;
        for (size_t i = 0; i < 8; ++i) v |= static_cast<uint64_t>(b[i]) << (8 * i);
        return v;
    }
    double f64() {
        return static_cast<double>(u64()) / static_cast<double>(std::numeric_limits<uint64_t>::max());
    }
private:
    SecureRandom() = default;
    ~SecureRandom() = default;
    SecureRandom(const SecureRandom&) = delete;
    SecureRandom& operator=(const SecureRandom&) = delete;
};

inline bool secure_random_bytes(void* buffer, size_t length) {
    return SecureRandom::instance().fill(buffer, length);
}

inline std::vector<uint8_t> secure_random_bytes(size_t length) {
    return SecureRandom::instance().bytes(length);
}

inline uint32_t secure_random_u32() { return SecureRandom::instance().u32(); }
inline uint64_t secure_random_u64() { return SecureRandom::instance().u64(); }
inline double secure_random_double() { return SecureRandom::instance().f64(); }

inline void secure_zero(void* buffer, size_t length) {
    if (!buffer || length == 0) return;
    volatile uint8_t* p = static_cast<volatile uint8_t*>(buffer);
    for (size_t i = 0; i < length; ++i) p[i] = 0;
}

} // namespace Crypto
} // namespace LFSSL

#endif // LFSSL_CRYPTO_RANDOM_HPP
