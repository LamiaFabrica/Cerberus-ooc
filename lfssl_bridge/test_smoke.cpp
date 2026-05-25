#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <windows.h>

// C-export function pointer types
typedef const char* (*fp_version)(void);
typedef void (*fp_sha256)(const uint8_t*, size_t, uint8_t[32]);
typedef void (*fp_hmac_sha256)(const uint8_t*,size_t,const uint8_t*,size_t,uint8_t[32]);
typedef void (*fp_random_bytes)(uint8_t*,size_t);
typedef void (*fp_aes256_enc)(const uint8_t[32],const uint8_t[16],uint8_t[16]);
typedef void (*fp_aes256_dec)(const uint8_t[32],const uint8_t[16],uint8_t[16]);

static void print_hex(const char* label, const uint8_t* data, size_t len) {
    printf("%s: ", label);
    for (size_t i = 0; i < len; ++i) printf("%02x", data[i]);
    printf("\n");
}

int main() {
    HMODULE h = LoadLibraryA("cerberus_lfssl.dll");
    if (!h) {
        printf("FAILED to load cerberus_lfssl.dll, error=%lu\n", GetLastError());
        return 1;
    }

    auto version = (fp_version)GetProcAddress(h, "cerberus_lfssl_version");
    auto sha256  = (fp_sha256)GetProcAddress(h, "cerberus_lfssl_sha256");
    auto hmac    = (fp_hmac_sha256)GetProcAddress(h, "cerberus_lfssl_hmac_sha256");
    auto randb   = (fp_random_bytes)GetProcAddress(h, "cerberus_lfssl_random_bytes");
    auto aesenc  = (fp_aes256_enc)GetProcAddress(h, "cerberus_lfssl_aes256_encrypt_block");
    auto aesdec  = (fp_aes256_dec)GetProcAddress(h, "cerberus_lfssl_aes256_decrypt_block");

    if (!version || !sha256 || !hmac || !randb || !aesenc || !aesdec) {
        printf("FAILED to resolve some exports\n");
        FreeLibrary(h);
        return 1;
    }

    printf("Version: %s\n", version());

    // SHA-256 smoke: empty string -> e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    uint8_t hash[32] = {0};
    sha256((const uint8_t*)"", 0, hash);
    print_hex("SHA256(\"\")", hash, 32);
    const uint8_t empty_expected[32] = {
        0xe3,0xb0,0xc4,0x42,0x98,0xfc,0x1c,0x14,0x9a,0xfb,0xf4,0xc8,0x99,0x6f,0xb9,0x24,
        0x27,0xae,0x41,0xe4,0x64,0x9b,0x93,0x4c,0xa4,0x95,0x99,0x1b,0x78,0x52,0xb8,0x55
    };
    if (std::memcmp(hash, empty_expected, 32) != 0) {
        printf("SHA256 FAIL: mismatch\n");
        return 1;
    }
    printf("SHA256 PASS\n");

    // HMAC-SHA256 smoke (RFC 4231 test case 1 key = 20 bytes of 0x0b)
    uint8_t key[20] = {0x0b,0x0b,0x0b,0x0b,0x0b,0x0b,0x0b,0x0b,0x0b,0x0b,0x0b,0x0b,0x0b,0x0b,0x0b,0x0b,0x0b,0x0b,0x0b,0x0b};
    uint8_t data[8] = {0x48,0x69,0x20,0x54,0x68,0x65,0x72,0x65}; // "Hi There"
    uint8_t hmac_out[32] = {0};
    hmac(key,20,data,8,hmac_out);
    print_hex("HMAC-SHA256", hmac_out, 32);
    const uint8_t hmac_expected[32] = {
        0xb0,0x34,0x4c,0x61,0xd8,0xdb,0x38,0x53,0x5c,0xa8,0xaf,0xce,0xaf,0x0b,0xf1,0x2b,
        0x88,0x1d,0xc2,0x00,0xc9,0x83,0x3d,0xa7,0x26,0xe9,0x37,0x6c,0x2e,0x32,0xcf,0xf7
    };
    if (std::memcmp(hmac_out, hmac_expected, 32) != 0) {
        printf("HMAC FAIL: mismatch\n");
        return 1;
    }
    printf("HMAC PASS\n");

    // AES-256 block encrypt/decrypt round-trip (ECB, zero key)
    uint8_t k[32] = {0};
    uint8_t pt[16] = {0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,0x10};
    uint8_t ct[16] = {0};
    uint8_t dec[16] = {0};
    aesenc(k, pt, ct);
    aesdec(k, ct, dec);
    print_hex("AES-256 CT", ct, 16);
    print_hex("AES-256 DEC", dec, 16);
    if (std::memcmp(dec, pt, 16) != 0) {
        printf("AES-256 block FAIL: mismatch\n");
        return 1;
    }
    printf("AES-256 block PASS\n");

    // Random bytes smoke
    uint8_t rnd[16] = {0};
    randb(rnd, 16);
    print_hex("Random", rnd, 16);
    bool any_nonzero = false;
    for (size_t i = 0; i < 16; ++i) if (rnd[i]) { any_nonzero = true; break; }
    if (!any_nonzero) {
        printf("Random FAIL: all zeros\n");
        return 1;
    }
    printf("Random PASS\n");

    FreeLibrary(h);
    printf("\n=== ALL SMOKE TESTS PASSED ===\n");
    return 0;
}
