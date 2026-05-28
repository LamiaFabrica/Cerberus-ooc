#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cstdio>
#include <windows.h>

typedef const char* (*fp_v)(void);
typedef void (*fp_sha256)(const uint8_t*,size_t,uint8_t[32]);
typedef void (*fp_hmac)(const uint8_t*,size_t,const uint8_t*,size_t,uint8_t[32]);
typedef void (*fp_rand)(uint8_t*,size_t);
typedef void (*fp_aesenc)(const uint8_t[32],const uint8_t[16],uint8_t[16]);
typedef void (*fp_aesdec)(const uint8_t[32],const uint8_t[16],uint8_t[16]);
typedef int (*fp_gcm_enc)(const uint8_t[32],const uint8_t*,size_t,const uint8_t*,size_t,const uint8_t*,size_t,uint8_t*,size_t,size_t*);
typedef int (*fp_gcm_dec)(const uint8_t[32],const uint8_t*,size_t,const uint8_t*,size_t,const uint8_t*,size_t,uint8_t*,size_t,size_t*);

int main(){
    HMODULE h=LoadLibraryA("cerberus_lfssl.dll");
    if(!h){printf("FAIL load\n");return 1;}
    auto v=(fp_v)GetProcAddress(h,"cerberus_lfssl_version");
    auto s=(fp_sha256)GetProcAddress(h,"cerberus_lfssl_sha256");
    auto m=(fp_hmac)GetProcAddress(h,"cerberus_lfssl_hmac_sha256");
    auto r=(fp_rand)GetProcAddress(h,"cerberus_lfssl_random_bytes");
    auto e=(fp_aesenc)GetProcAddress(h,"cerberus_lfssl_aes256_encrypt_block");
    auto d=(fp_aesdec)GetProcAddress(h,"cerberus_lfssl_aes256_decrypt_block");
    auto ge=(fp_gcm_enc)GetProcAddress(h,"cerberus_lfssl_aes256gcm_encrypt");
    auto gd=(fp_gcm_dec)GetProcAddress(h,"cerberus_lfssl_aes256gcm_decrypt");
    if(!v||!s||!m||!r||!e||!d||!ge||!gd){printf("FAIL resolve\n");return 1;}
    printf("Version: %s\n",v());

    uint8_t hash[32]={0};
    s((const uint8_t*)"",0,hash);
    const uint8_t exp1[32]={0xe3,0xb0,0xc4,0x42,0x98,0xfc,0x1c,0x14,0x9a,0xfb,0xf4,0xc8,0x99,0x6f,0xb9,0x24,0x27,0xae,0x41,0xe4,0x64,0x9b,0x93,0x4c,0xa4,0x95,0x99,0x1b,0x78,0x52,0xb8,0x55};
    if(std::memcmp(hash,exp1,32)!=0){printf("SHA FAIL\n");return 1;} printf("SHA PASS\n");

    uint8_t key[20]={0x0b,0x0b,0x0b,0x0b,0x0b,0x0b,0x0b,0x0b,0x0b,0x0b,0x0b,0x0b,0x0b,0x0b,0x0b,0x0b,0x0b,0x0b,0x0b,0x0b};
    uint8_t data[8]={0x48,0x69,0x20,0x54,0x68,0x65,0x72,0x65};
    uint8_t hm[32]={0};
    m(key,20,data,8,hm);
    const uint8_t exp2[32]={0xb0,0x34,0x4c,0x61,0xd8,0xdb,0x38,0x53,0x5c,0xa8,0xaf,0xce,0xaf,0x0b,0xf1,0x2b,0x88,0x1d,0xc2,0x00,0xc9,0x83,0x3d,0xa7,0x26,0xe9,0x37,0x6c,0x2e,0x32,0xcf,0xf7};
    if(std::memcmp(hm,exp2,32)!=0){printf("HMAC FAIL\n");return 1;} printf("HMAC PASS\n");

    uint8_t k[32]={0},pt[16]={0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,0x10},ct[16]={0},dc[16]={0};
    e(k,pt,ct); d(k,ct,dc);
    if(std::memcmp(dc,pt,16)!=0){printf("AES BLOCK FAIL\n");return 1;} printf("AES BLOCK PASS\n");

    uint8_t rnd[16]={0}; r(rnd,16);
    bool nz=false; for(int i=0;i<16;++i) if(rnd[i]){nz=true;break;}
    if(!nz){printf("RAND FAIL\n");return 1;} printf("RAND PASS\n");

    // AES-256-GCM round-trip
    uint8_t nonce[12]={0};
    uint8_t gpt[32]={0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,0x18,0x19,0x1a,0x1b,0x1c,0x1d,0x1e,0x1f,0x20};
    uint8_t aad[8]={0xde,0xad,0xbe,0xef,0xca,0xfe,0xba,0xbe};
    uint8_t gct[64]={0}; size_t gct_len=0;
    int rc=ge(k,nonce,12,gpt,32,aad,8,gct,sizeof(gct),&gct_len);
    if(rc!=0){printf("GCM ENC FAIL rc=%d\n",rc);return 1;}
    uint8_t gdec[64]={0}; size_t gdec_len=0;
    rc=gd(k,nonce,12,gct,gct_len,aad,8,gdec,sizeof(gdec),&gdec_len);
    if(rc!=0){printf("GCM DEC FAIL rc=%d\n",rc);return 1;}
    if(gdec_len!=32||std::memcmp(gdec,gpt,32)!=0){printf("GCM ROUNDTRIP FAIL\n");return 1;}
    printf("GCM PASS (len=%zu)\n",gct_len);

    FreeLibrary(h);
    printf("\n=== ALL PASSED ===\n");
    return 0;
}
