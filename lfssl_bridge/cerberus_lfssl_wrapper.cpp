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
#include <lfssl/crypto/kyber_kem.hpp>
#include <lfssl/crypto/dilithium.hpp>

// Argon2id from LFSSL phc-argon2 (public domain)
extern "C" {
#include <argon2.h>
}

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

// ============================================================================
// PQC — Kyber KEM (all three levels)
// ============================================================================

CERBERUS_LFSSL_API int cerberus_lfssl_kyber_keypair(
    uint32_t level,
    uint8_t* public_key, size_t public_key_len,
    uint8_t* private_key, size_t private_key_len
) {
    LFSSL::Crypto::KyberKEM::SecurityLevel lvl;
    size_t pk_sz = 0, sk_sz = 0;
    switch (level) {
        case 512: lvl = LFSSL::Crypto::KyberKEM::SecurityLevel::KYBER_512; pk_sz = 800; sk_sz = 1632; break;
        case 768: lvl = LFSSL::Crypto::KyberKEM::SecurityLevel::KYBER_768; pk_sz = 1184; sk_sz = 2400; break;
        case 1024: lvl = LFSSL::Crypto::KyberKEM::SecurityLevel::KYBER_1024; pk_sz = 1568; sk_sz = 3168; break;
        default: return -1;
    }
    if (public_key_len < pk_sz || private_key_len < sk_sz) return -2;
    std::vector<uint8_t> pk, sk;
    if (!LFSSL::Crypto::KyberKEM::generate_keypair(lvl, pk, sk)) return -3;
    std::memcpy(public_key, pk.data(), pk.size());
    std::memcpy(private_key, sk.data(), sk.size());
    return 0;
}

CERBERUS_LFSSL_API int cerberus_lfssl_kyber_encapsulate(
    uint32_t level,
    const uint8_t* public_key, size_t public_key_len,
    uint8_t* ciphertext, size_t ciphertext_len,
    uint8_t* shared_secret, size_t shared_secret_len
) {
    LFSSL::Crypto::KyberKEM::SecurityLevel lvl;
    size_t ct_sz = 0, ss_sz = 0;
    switch (level) {
        case 512: lvl = LFSSL::Crypto::KyberKEM::SecurityLevel::KYBER_512; ct_sz = 768; ss_sz = 32; break;
        case 768: lvl = LFSSL::Crypto::KyberKEM::SecurityLevel::KYBER_768; ct_sz = 1088; ss_sz = 32; break;
        case 1024: lvl = LFSSL::Crypto::KyberKEM::SecurityLevel::KYBER_1024; ct_sz = 1568; ss_sz = 32; break;
        default: return -1;
    }
    if (ciphertext_len < ct_sz || shared_secret_len < ss_sz) return -2;
    std::vector<uint8_t> ct, ss;
    std::vector<uint8_t> pk(public_key, public_key + public_key_len);
    if (!LFSSL::Crypto::KyberKEM::encapsulate(lvl, pk, ct, ss)) return -3;
    std::memcpy(ciphertext, ct.data(), ct.size());
    std::memcpy(shared_secret, ss.data(), ss.size());
    return 0;
}

CERBERUS_LFSSL_API int cerberus_lfssl_kyber_decapsulate(
    uint32_t level,
    const uint8_t* private_key, size_t private_key_len,
    const uint8_t* ciphertext, size_t ciphertext_len,
    uint8_t* shared_secret, size_t shared_secret_len
) {
    LFSSL::Crypto::KyberKEM::SecurityLevel lvl;
    size_t sk_sz = 0, ct_sz = 0, ss_sz = 32;
    switch (level) {
        case 512: lvl = LFSSL::Crypto::KyberKEM::SecurityLevel::KYBER_512; sk_sz = 1632; ct_sz = 768; break;
        case 768: lvl = LFSSL::Crypto::KyberKEM::SecurityLevel::KYBER_768; sk_sz = 2400; ct_sz = 1088; break;
        case 1024: lvl = LFSSL::Crypto::KyberKEM::SecurityLevel::KYBER_1024; sk_sz = 3168; ct_sz = 1568; break;
        default: return -1;
    }
    if (private_key_len < sk_sz || ciphertext_len < ct_sz || shared_secret_len < ss_sz) return -2;
    std::vector<uint8_t> sk(private_key, private_key + private_key_len);
    std::vector<uint8_t> ct(ciphertext, ciphertext + ciphertext_len);
    std::vector<uint8_t> ss;
    if (!LFSSL::Crypto::KyberKEM::decapsulate(lvl, sk, ct, ss)) return -3;
    std::memcpy(shared_secret, ss.data(), ss.size());
    return 0;
}

// ============================================================================
// PQC — Dilithium signatures (all three levels)
// ============================================================================

CERBERUS_LFSSL_API int cerberus_lfssl_dilithium_keypair(
    uint32_t level,
    uint8_t* public_key, size_t public_key_len,
    uint8_t* private_key, size_t private_key_len
) {
    LFSSL::Crypto::Dilithium::SecurityLevel lvl;
    size_t pk_sz = 0, sk_sz = 0;
    switch (level) {
        // Sizes from dilithium reference api.h/params.h (not dilithium.hpp which has incorrect SK sizes)
        case 2: lvl = LFSSL::Crypto::Dilithium::SecurityLevel::DILITHIUM2; pk_sz = 1312; sk_sz = 2560; break;
        case 3: lvl = LFSSL::Crypto::Dilithium::SecurityLevel::DILITHIUM3; pk_sz = 1952; sk_sz = 4032; break;
        case 5: lvl = LFSSL::Crypto::Dilithium::SecurityLevel::DILITHIUM5; pk_sz = 2592; sk_sz = 4896; break;
        default: return -1;
    }
    if (public_key_len < pk_sz || private_key_len < sk_sz) return -2;
    std::vector<uint8_t> pk, sk;
    if (!LFSSL::Crypto::Dilithium::generate_keypair(lvl, pk, sk)) return -3;
    std::memcpy(public_key, pk.data(), pk.size());
    std::memcpy(private_key, sk.data(), sk.size());
    return 0;
}

CERBERUS_LFSSL_API int cerberus_lfssl_dilithium_sign(
    uint32_t level,
    const uint8_t* data, size_t data_len,
    const uint8_t* private_key, size_t private_key_len,
    uint8_t* signature, size_t signature_len,
    size_t* sig_out_len
) {
    LFSSL::Crypto::Dilithium::SecurityLevel lvl;
    size_t sk_sz = 0, sig_sz = 0;
    switch (level) {
        // Reference SK sizes: DILITHIUM_MODE 2=2560, 3=4032, 5=4896 (dilithium.hpp has incorrect values)
        case 2: lvl = LFSSL::Crypto::Dilithium::SecurityLevel::DILITHIUM2; sk_sz = 2560; sig_sz = 2420; break;
        case 3: lvl = LFSSL::Crypto::Dilithium::SecurityLevel::DILITHIUM3; sk_sz = 4032; sig_sz = 3293; break;
        case 5: lvl = LFSSL::Crypto::Dilithium::SecurityLevel::DILITHIUM5; sk_sz = 4896; sig_sz = 4595; break;
        default: return -1;
    }
    if (private_key_len < sk_sz || signature_len < sig_sz) return -2;
    std::vector<uint8_t> sk(private_key, private_key + private_key_len);
    std::vector<uint8_t> msg(data, data + data_len);
    std::vector<uint8_t> sig;
    if (!LFSSL::Crypto::Dilithium::sign(lvl, msg, sk, sig)) return -3;
    if (sig.size() > signature_len) return -4;
    std::memcpy(signature, sig.data(), sig.size());
    if (sig_out_len) *sig_out_len = sig.size();
    return 0;
}

CERBERUS_LFSSL_API int cerberus_lfssl_dilithium_verify(
    uint32_t level,
    const uint8_t* data, size_t data_len,
    const uint8_t* signature, size_t signature_len,
    const uint8_t* public_key, size_t public_key_len
) {
    LFSSL::Crypto::Dilithium::SecurityLevel lvl;
    size_t pk_sz = 0;
    switch (level) {
        case 2: lvl = LFSSL::Crypto::Dilithium::SecurityLevel::DILITHIUM2; pk_sz = 1312; break;
        case 3: lvl = LFSSL::Crypto::Dilithium::SecurityLevel::DILITHIUM3; pk_sz = 1952; break;
        case 5: lvl = LFSSL::Crypto::Dilithium::SecurityLevel::DILITHIUM5; pk_sz = 2592; break;
        default: return -1;
    }
    if (public_key_len < pk_sz) return -2;
    std::vector<uint8_t> pk(public_key, public_key + public_key_len);
    std::vector<uint8_t> msg(data, data + data_len);
    std::vector<uint8_t> sig(signature, signature + signature_len);
    bool ok = LFSSL::Crypto::Dilithium::verify(lvl, msg, sig, pk);
    return ok ? 0 : -3;
}

// ============================================================================
// Argon2id — memory-hard password hashing
// ============================================================================

CERBERUS_LFSSL_API int cerberus_lfssl_argon2id(
    uint32_t t_cost,
    uint32_t m_cost,
    uint32_t parallelism,
    const uint8_t* password, size_t password_len,
    const uint8_t* salt, size_t salt_len,
    uint8_t* hash, size_t hash_len
) {
    if (!password || !salt || !hash) return -1;
    int r = argon2id_hash_raw(t_cost, m_cost, parallelism,
                              password, password_len,
                              salt, salt_len,
                              hash, hash_len);
    return r;
}

CERBERUS_LFSSL_API int cerberus_lfssl_argon2id_verify(
    uint32_t t_cost,
    uint32_t m_cost,
    uint32_t parallelism,
    const uint8_t* password, size_t password_len,
    const uint8_t* salt, size_t salt_len,
    const uint8_t* expected_hash, size_t hash_len
) {
    if (!password || !salt || !expected_hash) return -1;
    uint8_t* computed = static_cast<uint8_t*>(::malloc(hash_len));
    if (!computed) return -2;
    int r = argon2id_hash_raw(t_cost, m_cost, parallelism,
                              password, password_len,
                              salt, salt_len,
                              computed, hash_len);
    if (r != 0) { ::free(computed); return -3; }
    volatile uint8_t diff = 0;
    for (size_t i = 0; i < hash_len; ++i) diff |= (computed[i] ^ expected_hash[i]);
    ::free(computed);
    return (diff == 0) ? 0 : -4;
}

} // extern "C"
