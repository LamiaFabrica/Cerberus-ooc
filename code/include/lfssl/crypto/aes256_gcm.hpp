/**
 * LFSSL AES-256-GCM Shim
 * ========================
 * Declares AES256GCM class for PsiForceDB header compatibility.
 *
 * The real implementation lives in PsiForceDB's compiled LFSSL library.
 * This header provides the class signature so PsiForceDB headers compile.
 * Cerberus delegates AES-256-GCM operations to PsiForceDB at runtime.
 *
 * © 2026 D Hargreaves | Yorkshire Champion Standards
 */

#ifndef LFSSL_CRYPTO_AES256_GCM_HPP
#define LFSSL_CRYPTO_AES256_GCM_HPP

#include <vector>
#include <cstdint>
#include <cstddef>

namespace LFSSL {
namespace Crypto {

class AES256GCM {
public:
    static constexpr size_t KEY_SIZE = 32;
    static constexpr size_t NONCE_SIZE = 12;
    static constexpr size_t TAG_SIZE = 16;

    explicit AES256GCM(const uint8_t key[KEY_SIZE]);
    explicit AES256GCM(const std::vector<uint8_t>& key);
    ~AES256GCM();

    // Raw C-style encrypt/decrypt (PsiForceDB internal API)
    void encrypt(const uint8_t* plaintext, size_t plaintext_len,
                 const uint8_t* aad, size_t aad_len,
                 const uint8_t nonce[NONCE_SIZE],
                 uint8_t* ciphertext,
                 uint8_t tag[TAG_SIZE]);

    bool decrypt(const uint8_t* ciphertext, size_t ciphertext_len,
                 const uint8_t* aad, size_t aad_len,
                 const uint8_t nonce[NONCE_SIZE],
                 const uint8_t tag[TAG_SIZE],
                 uint8_t* plaintext);

    // Vector-based convenience (returns ciphertext with tag appended)
    std::vector<uint8_t> encrypt(const uint8_t* nonce_data, size_t nonce_len,
                                   const uint8_t* plaintext_data, size_t plaintext_len,
                                   const uint8_t* aad_data, size_t aad_len);

    std::vector<uint8_t> decrypt(const uint8_t* nonce_data, size_t nonce_len,
                                   const uint8_t* ciphertext_data, size_t ciphertext_len,
                                   const uint8_t* tag_data, size_t tag_len,
                                   const uint8_t* aad_data, size_t aad_len);

    std::vector<uint8_t> encrypt(const std::vector<uint8_t>& nonce,
                                   const std::vector<uint8_t>& plaintext,
                                   const std::vector<uint8_t>& aad = {});

    std::vector<uint8_t> decrypt(const std::vector<uint8_t>& nonce,
                                   const std::vector<uint8_t>& ciphertext,
                                   const std::vector<uint8_t>& aad = {});

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace Crypto
} // namespace LFSSL

#endif // LFSSL_CRYPTO_AES256_GCM_HPP
