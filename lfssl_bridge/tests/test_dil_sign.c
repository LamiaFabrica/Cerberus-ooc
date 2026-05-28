#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
extern int pqcrystals_dilithium2_ref_keypair(uint8_t *pk, uint8_t *sk);
extern int pqcrystals_dilithium2_ref_signature(uint8_t *sig, size_t *siglen, const uint8_t *m, size_t mlen, const uint8_t *ctx, size_t ctxlen, const uint8_t *sk);
extern int pqcrystals_dilithium2_ref_verify(const uint8_t *sig, size_t siglen, const uint8_t *m, size_t mlen, const uint8_t *ctx, size_t ctxlen, const uint8_t *pk);
int main() {
    uint8_t pk[1312]={0}, sk[2560]={0}, sig[2420]={0}, msg[16]={0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,0x10};
    size_t siglen=0;
    if(pqcrystals_dilithium2_ref_keypair(pk,sk)!=0){printf("kp fail\n");return 1;}
    if(pqcrystals_dilithium2_ref_signature(sig,&siglen,msg,sizeof(msg),NULL,0,sk)!=0){printf("sign fail\n");return 1;}
    if(pqcrystals_dilithium2_ref_verify(sig,siglen,msg,sizeof(msg),NULL,0,pk)!=0){printf("verify fail\n");return 1;}
    printf("OK\n"); return 0;
}
