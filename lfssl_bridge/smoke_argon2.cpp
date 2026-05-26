#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <windows.h>
#include <string.h>

int main() {
    HMODULE h = LoadLibraryA("cerberus_lfssl.dll");
    if(!h){printf("load fail\n");return 1;}
    using fp = int (*)(uint32_t,uint32_t,uint32_t,const uint8_t*,size_t,const uint8_t*,size_t,uint8_t*,size_t);
    auto hash=(fp)GetProcAddress(h,"cerberus_lfssl_argon2id");
    if(!hash){printf("missing\n");return 1;}
    uint8_t pwd[9]="password";
    uint8_t salt[16]={0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,0x10};
    uint8_t out1[32]={0},out2[32]={0};
    if(hash(2,65536,1,pwd,8,salt,16,out1,32)!=0){printf("hash1 fail\n");return 1;}
    if(hash(2,65536,1,pwd,8,salt,16,out2,32)!=0){printf("hash2 fail\n");return 1;}
    if(memcmp(out1,out2,32)!=0){printf("non-deterministic\n");return 1;}
    printf("Argon2id OK\n"); return 0;
}
