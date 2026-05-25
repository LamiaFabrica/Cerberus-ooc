@echo off
g++ -shared -o cerberus_lfssl.dll sha256.o sha384.o sha512.o aes256.o pbkdf2.o hkdf.o random.o aes256_gcm.o wrapper.o -Wl^,--out-implib^,cerberus_lfssl.lib -lbcrypt
