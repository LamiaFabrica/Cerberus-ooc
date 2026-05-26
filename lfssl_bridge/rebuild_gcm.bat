@echo off
setlocal

cd /d "C:\McMaker Projects\Projects\Cerberus - Copy\lfssl_bridge"
"C:\Users\david\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.MCF.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin\g++.exe" -std=c++23 -c "C:\McMaker Projects\Projects\LFSSL - Lamia Fabrica SSL\src\crypto\aes256_gcm_hardware.cpp" -o lfssl_aes256_gcm_hardware.o -O2 -Wall -DBUILD_DLL -maes -msse2 -msse4.1 -I"C:\McMaker Projects\Projects\LFSSL - Lamia Fabrica SSL\include"
if errorlevel 1 exit /b 1

g++ -shared -o cerberus_lfssl.dll wrapper.o kyber_k2_kem.o kyber_k2_indcpa.o kyber_k2_poly.o kyber_k2_polyvec.o kyber_k2_cbd.o kyber_k2_reduce.o kyber_k2_ntt.o kyber_k2_verify.o kyber_k2_symmetric-shake.o kyber_k3_kem.o kyber_k3_indcpa.o kyber_k3_poly.o kyber_k3_polyvec.o kyber_k3_cbd.o kyber_k3_reduce.o kyber_k3_ntt.o kyber_k3_verify.o kyber_k3_symmetric-shake.o kyber_k4_kem.o kyber_k4_indcpa.o kyber_k4_poly.o kyber_k4_polyvec.o kyber_k4_cbd.o kyber_k4_reduce.o kyber_k4_ntt.o kyber_k4_verify.o kyber_k4_symmetric-shake.o kyber_fips202.o dil_m2_sign.o dil_m2_packing.o dil_m2_polyvec.o dil_m2_poly.o dil_m2_ntt.o dil_m2_reduce.o dil_m2_rounding.o dil_m2_symmetric-shake.o dil_m3_sign.o dil_m3_packing.o dil_m3_polyvec.o dil_m3_poly.o dil_m3_ntt.o dil_m3_reduce.o dil_m3_rounding.o dil_m3_symmetric-shake.o dil_m5_sign.o dil_m5_packing.o dil_m5_polyvec.o dil_m5_poly.o dil_m5_ntt.o dil_m5_reduce.o dil_m5_rounding.o dil_m5_symmetric-shake.o dil_fips202.o lfssl_sha256.o lfssl_sha384.o lfssl_sha512.o lfssl_aes256.o lfssl_aes256_gcm_hardware.o lfssl_pbkdf2.o lfssl_hkdf.o lfssl_random.o lfssl_kyber_kem_wrapper.o lfssl_dilithium_wrapper.o lfssl_pqc_randombytes.o -Wl,--out-implib,cerberus_lfssl.lib -lbcrypt
if errorlevel 1 exit /b 1

echo DLL relinked OK.
