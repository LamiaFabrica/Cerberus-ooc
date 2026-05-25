/**
 * cerberus_lfssl_wrapper.cpp
 * Thin C-export wrapper around verified LFSSL individual source files.
 * Build with -DBUILD_DLL to get __declspec(dllexport).
 */

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <vector>

#include <lfssl/crypto/sha256.hpp>
#include <lfssl/crypto/hmac.hpp>
#include <lfssl/crypto/pbkdf2.hpp>
#include <lfssl/crypto/aes256.hpp>
#include <lfssl/crypto/aes256_gcm.hpp>
#include <lfssl/crypto/random.hpp>

#ifdef BUILD_DLL
    #ifdef _WIN32
        #define CERBERUS_LFSSL_API __declspec(dllexport)
    #else
        #define CERBERUS_LFSSL_API __attribute__((visibility("default")))
    #endif
#else
    #ifdef _WIN32
        #define CERBERUS_LFSSL_API __declspec(dllimport)
    #else
        #define CERBERUS_LFSSL_API
    #endif
#endif

extern "C" {

CERBERUS_LFSSL_API const char* cerberus_lfssl_version(void) {
    return "cerberus_lfssl 1.0.0";
}

CERBERUS_LFSSL_API void cerberus_lfssl_sha256(const uint8_t* data, size_t len, uint8_t out[32]) {
    std::vector<uint8_t> digest = LFSSL::Crypto::SHA256::hash(data, len);
    std::memcpy(out, digest.data(), 32);
}

CERBERUS_LFSSL_API void cerberus_lfssl_hmac_sha256(
    const uint8_t* key, size_t key_len,
    const uint8_t* data, size_t data_len,
    uint8_t out[32]
) {
    std::vector<uint8_t> mac = LFSSL::Crypto::HMAC<LFSSL::Crypto::SHA256>::mac(
        key, key_len, data, data_len
    );
    std::memcpy(out, mac.data(), 32);
}

CERBERUS_LFSSL_API int cerberus_lfssl_pbkdf2_sha256(
    const char* password,
    const uint8_t* salt, size_t salt_len,
    uint32_t iterations,
    uint8_t* out, size_t out_len
) {
    if (!password || !salt || !out || out_len == 0) return -1;
    LFSSL::Crypto::PBKDF2_SHA256::derive(
        reinterpret_cast<const uint8_t*>(password), std::strlen(password),
        salt, salt_len,
        iterations, out_len, out
    );
    return 0;
}

CERBERUS_LFSSL_API void cerberus_lfssl_aes256_encrypt_block(
    const uint8_t key[32],
    const uint8_t in[16],
    uint8_t out[16]
) {
    LFSSL::Crypto::AES256 aes(key);
    aes.encrypt_block(in, out);
}

CERBERUS_LFSSL_API void cerberus_lfssl_aes256_decrypt_block(
    const uint8_t key[32],
    const uint8_t in[16],
    uint8_t out[16]
) {
    LFSSL::Crypto::AES256 aes(key);
    aes.decrypt_block(in, out);
}

CERBERUS_LFSSL_API int cerberus_lfssl_aes256gcm_encrypt(
    const uint8_t key[32],
    const uint8_t* nonce, size_t nonce_len,
    const uint8_t* pt, size_t pt_len,
    const uint8_t* aad, size_t aad_len,
    uint8_t* out, size_t out_capacity, size_t* out_written
) {
    if (!key || !nonce || !pt || !out || !out_written) return -1;
    if (nonce_len != 12) return -2;
    if (out_capacity < pt_len + 16) return -3;

    try {
        LFSSL::Crypto::AES256GCM gcm(key);
        std::vector<uint8_t> aad_vec(aad ? aad : nullptr, aad ? aad + aad_len : nullptr);
        std::vector<uint8_t> ct = gcm.encrypt(
            nonce, nonce_len,
            pt,   pt_len,
            aad_vec.data(), aad_vec.size()
        );
        std::memcpy(out, ct.data(), ct.size());
        *out_written = ct.size();
        return 0;
    } catch (...) {
        return -4;
    }
}

CERBERUS_LFSSL_API int cerberus_lfssl_aes256gcm_decrypt(
    const uint8_t key[32],
    const uint8_t* nonce, size_t nonce_len,
    const uint8_t* ct_tag, size_t ct_tag_len,
    const uint8_t* aad, size_t aad_len,
    uint8_t* out, size_t out_capacity, size_t* out_written
) {
    if (!key || !nonce || !ct_tag || !out || !out_written) return -1;
    if (nonce_len != 12) return -2;
    if (ct_tag_len < 16) return -3;
    size_t ct_len = ct_tag_len - 16;
    if (out_capacity < ct_len) return -4;

    try {
        LFSSL::Crypto::AES256GCM gcm(key);
        std::vector<uint8_t> nonce_vec(nonce, nonce + nonce_len);
        std::vector<uint8_t> ct_tag_vec(ct_tag, ct_tag + ct_tag_len);
        std::vector<uint8_t> aad_vec(aad ? aad : nullptr, aad ? aad + aad_len : nullptr);

        std::vector<uint8_t> pt = gcm.decrypt(
            nonce_vec, ct_tag_vec, aad_vec
        );
        if (pt.empty()) return -5; // auth failure
        std::memcpy(out, pt.data(), pt.size());
        *out_written = pt.size();
        return 0;
    } catch (...) {
        return -6;
    }
}

CERBERUS_LFSSL_API void cerberus_lfssl_random_bytes(uint8_t* out, size_t len) {
    if (!out || len == 0) return;
    std::vector<uint8_t> buf = LFSSL::Crypto::SecureRandom::instance().bytes(len);
    std::memcpy(out, buf.data(), len);
}

} // extern "C"
