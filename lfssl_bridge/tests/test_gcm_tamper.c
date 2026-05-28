#include <cstdint.h>
#include <cstddef.h>
#include <stdio.h>
#include <windows.h>
#include <string.h>
int main() {
    HMODULE h = LoadLibraryA("cerberus_lfssl.dll");
    if(!h){printf("load fail\n");return 1;}
    using fp_enc = int (*)(const uint8_t[32],const uint8_t*,size_t,const uint8_t*,size_t,const uint8_t*,size_t,uint8_t*,size_t,size_t*);
    using fp_dec = int (*)(const uint8_t[32],const uint8_t*,size_t,const uint8_t*,size_t,const uint8_t*,size_t,uint8_t*,size_t,size_t*);
    auto enc=(fp_enc)GetProcAddress(h,"cerberus_lfssl_aes256gcm_encrypt");
    auto dec=(fp_dec)GetProcAddress(h,"cerberus_lfssl_aes256gcm_decrypt");
    if(!enc||!dec){printf("missing\n");return 1;}
    uint8_t key[32]={0},nonce[12]={1,2,3,4,5,6,7,8,9,10,11,12};
    uint8_t pt[32]={0x48,0x65,0x6c,0x6c,0x6f},aad[8]={0x61,0x62};
    uint8_t ct[64]={0},rec[64]={0};size_t ct_len=0,rec_len=0;
    if(enc(key,nonce,12,pt,5,aad,2,ct,sizeof(ct),&ct_len)!=0){printf("enc fail\n");return 1;}
    if(dec(key,nonce,12,ct,ct_len,aad,2,rec,sizeof(rec),&rec_len)!=0){printf("dec fail\n");return 1;}
    if(rec_len!=5||memcmp(pt,rec,5)!=0){printf("data mismatch\n");return 1;}
    ct[0]^=1; printf("Testing tamper...\n");
    int r=dec(key,nonce,12,ct,ct_len,aad,2,rec,sizeof(rec),&rec_len);
    if(r==0){printf("TAMPER ACCEPTED -- BUG\n");return 1;}
    printf("Tamper rejected OK (r=%d)\n",r);
    printf("AES-256-GCM OK\n"); return 0;
}
