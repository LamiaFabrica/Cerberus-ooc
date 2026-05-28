#include <lfssl/crypto/hmac.hpp>
#include <lfssl/crypto/sha256.hpp>
using namespace LFSSL::Crypto;
int main() {
    uint8_t key[32]={0}; uint8_t data[16]={0};
    auto r = HMAC<SHA256>::mac(key,32,data,16);
    return r.size()==32?0:1;
}
