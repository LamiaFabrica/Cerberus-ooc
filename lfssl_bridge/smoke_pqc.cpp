/**
 * smoke_pqc.cpp — standalone smoke test for Kyber + Dilithium via cerberus_lfssl.dll
 */
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cstdio>
#include <windows.h>

typedef int (*fp_kyber_keypair)(uint32_t,uint8_t*,size_t,uint8_t*,size_t);
typedef int (*fp_kyber_enc)(uint32_t,const uint8_t*,size_t,uint8_t*,size_t,uint8_t*,size_t);
typedef int (*fp_kyber_dec)(uint32_t,const uint8_t*,size_t,const uint8_t*,size_t,uint8_t*,size_t);

typedef int (*fp_dil_keypair)(uint32_t,uint8_t*,size_t,uint8_t*,size_t);
typedef int (*fp_dil_sign)(uint32_t,const uint8_t*,size_t,const uint8_t*,size_t,uint8_t*,size_t,size_t*);
typedef int (*fp_dil_verify)(uint32_t,const uint8_t*,size_t,const uint8_t*,size_t,const uint8_t*,size_t);

int main() {
    HMODULE h = LoadLibraryA("cerberus_lfssl.dll");
    if (!h) { printf("FAIL load DLL: %lu\n", GetLastError()); return 1; }

    auto kyber_kp  = (fp_kyber_keypair)GetProcAddress(h, "cerberus_lfssl_kyber_keypair");
    auto kyber_enc = (fp_kyber_enc)GetProcAddress(h, "cerberus_lfssl_kyber_encapsulate");
    auto kyber_dec = (fp_kyber_dec)GetProcAddress(h, "cerberus_lfssl_kyber_decapsulate");
    auto dil_kp    = (fp_dil_keypair)GetProcAddress(h, "cerberus_lfssl_dilithium_keypair");
    auto dil_sign  = (fp_dil_sign)GetProcAddress(h, "cerberus_lfssl_dilithium_sign");
    auto dil_verify= (fp_dil_verify)GetProcAddress(h, "cerberus_lfssl_dilithium_verify");

    if (!kyber_kp || !kyber_enc || !kyber_dec || !dil_kp || !dil_sign || !dil_verify) {
        printf("FAIL resolve PQC exports\n"); FreeLibrary(h); return 1;
    }

    // ---- Kyber round-trip (Kyber512, smallest) ----
    uint8_t pk[800] = {0}, sk[1632] = {0};
    uint8_t ct[768] = {0}, ss_enc[32] = {0}, ss_dec[32] = {0};

    int r = kyber_kp(512, pk, sizeof(pk), sk, sizeof(sk));
    if (r != 0) { printf("FAIL kyber_keypair: %d\n", r); FreeLibrary(h); return 1; }
    printf("Kyber512 keypair OK\n");

    r = kyber_enc(512, pk, sizeof(pk), ct, sizeof(ct), ss_enc, sizeof(ss_enc));
    if (r != 0) { printf("FAIL kyber_enc: %d\n", r); FreeLibrary(h); return 1; }
    printf("Kyber512 encapsulate OK\n");

    r = kyber_dec(512, sk, sizeof(sk), ct, sizeof(ct), ss_dec, sizeof(ss_dec));
    if (r != 0) { printf("FAIL kyber_dec: %d\n", r); FreeLibrary(h); return 1; }
    printf("Kyber512 decapsulate OK\n");

    if (std::memcmp(ss_enc, ss_dec, 32) != 0) {
        printf("FAIL shared secret mismatch\n"); FreeLibrary(h); return 1;
    }
    printf("Kyber512 shared secret MATCH\n");

    // ---- Dilithium sign/verify (mode 2, smallest) ----
    // Reference sizes from api.h: PK=1312, SK=2560 (dilithium.hpp lists 2528, incorrect)
    uint8_t dil_pk[1312] = {0}, dil_sk[2560] = {0};
    uint8_t msg[16] = {0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,
                       0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,0x10};
    uint8_t sig[2420] = {0};
    size_t sig_len = 0;

    r = dil_kp(2, dil_pk, sizeof(dil_pk), dil_sk, sizeof(dil_sk));
    if (r != 0) { printf("FAIL dilithium_keypair: %d\n", r); FreeLibrary(h); return 1; }
    printf("Dilithium2 keypair OK\n");

    r = dil_sign(2, msg, sizeof(msg), dil_sk, sizeof(dil_sk), sig, sizeof(sig), &sig_len);
    if (r != 0) { printf("FAIL dilithium_sign: %d\n", r); FreeLibrary(h); return 1; }
    printf("Dilithium2 sign OK (len=%zu)\n", sig_len);

    r = dil_verify(2, msg, sizeof(msg), sig, sig_len, dil_pk, sizeof(dil_pk));
    if (r != 0) { printf("FAIL dilithium_verify: %d\n", r); FreeLibrary(h); return 1; }
    printf("Dilithium2 verify OK\n");

    FreeLibrary(h);
    printf("\n=== ALL PQC SMOKE TESTS PASSED ===\n");
    return 0;
}
