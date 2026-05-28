#!/bin/bash
# build_lfssl_linux.sh — Build libcerberus_lfssl.so for Linux (Ubuntu/WSL)
# Copyright © 2026, all rights reserved | David Hargreaves aka Roylepython
# Support: https://www.patreon.com/TheMedusaInitiative

set -e

BUILD_DIR="$(cd "$(dirname "$0")" && pwd)"
LFSSL="/mnt/c/McMaker Projects/Projects/LFSSL - Lamia Fabrica SSL"
INCLUDE="$LFSSL/include"
LFSSLSRC="$LFSSL/src"
KYBER_REF="$LFSSL/lib/kyber-main/ref"
DIL_REF="$LFSSL/lib/dilithium-master/ref"
ARGON_SRC="$LFSSL/phc-argon2"
ARGON_INC="$ARGON_SRC/include"

OUT="$BUILD_DIR/libcerberus_lfssl.so"

echo "=== Building libcerberus_lfssl.so ==="
echo "  LFSSL root: $LFSSL"
echo "  Output:     $OUT"

mkdir -p "$BUILD_DIR/obj"

OBJS=()

# ---- Kyber ref for each K ----
for K in 2 3 4; do
    for S in kem indcpa poly polyvec cbd reduce ntt verify symmetric-shake; do
        SRC="$KYBER_REF/$S.c"
        OBJ="$BUILD_DIR/obj/kyber_k${K}_${S}.o"
        echo "  CC  $S.c for K=$K"
        gcc -c "$SRC" -o "$OBJ" -O2 -fPIC -Wall -DKYBER_K=$K -I"$KYBER_REF"
        OBJS+=("$OBJ")
    done
done

# Kyber fips202 (once)
OBJ="$BUILD_DIR/obj/kyber_fips202.o"
echo "  CC  fips202.c (Kyber)"
gcc -c "$KYBER_REF/fips202.c" -o "$OBJ" -O2 -fPIC -Wall -I"$KYBER_REF"
OBJS+=("$OBJ")

# ---- Dilithium ref for each mode ----
for M in 2 3 5; do
    for S in sign packing polyvec poly ntt reduce rounding symmetric-shake; do
        SRC="$DIL_REF/$S.c"
        OBJ="$BUILD_DIR/obj/dil_m${M}_${S}.o"
        echo "  CC  $S.c for mode=$M"
        gcc -c "$SRC" -o "$OBJ" -O2 -fPIC -Wall -DDILITHIUM_MODE=$M -I"$DIL_REF"
        OBJS+=("$OBJ")
    done
done

# Dilithium fips202 (once)
OBJ="$BUILD_DIR/obj/dil_fips202.o"
echo "  CC  fips202.c (Dilithium)"
gcc -c "$DIL_REF/fips202.c" -o "$OBJ" -O2 -fPIC -Wall -I"$DIL_REF"
OBJS+=("$OBJ")

# Use GCC 15 if available, otherwise fall back to system g++
GXX=g++
if command -v g++-15 &> /dev/null; then
    GXX=g++-15
fi
echo "  Using C++ compiler: $GXX"

# ---- LFSSL individual .cpp files ----
for F in crypto/sha256 crypto/sha384 crypto/sha512 crypto/aes256 crypto/aes256_gcm_hardware crypto/pbkdf2 crypto/hkdf crypto/random kyber_kem_wrapper dilithium_wrapper pqc_randombytes; do
    CPP="$LFSSLSRC/$F.cpp"
    OBJ="$BUILD_DIR/obj/lfssl_$(basename $F).o"
    echo "  CXX $(basename $F).cpp"
    $GXX -std=c++2c -c "$CPP" -o "$OBJ" -O2 -fPIC -Wall -DBUILD_DLL -maes -msse2 -msse4.1 -I"$INCLUDE" -I"$KYBER_REF" -I"$DIL_REF"
    OBJS+=("$OBJ")
done

# ---- Wrapper ----
OBJ="$BUILD_DIR/obj/wrapper.o"
echo "  CXX cerberus_lfssl_wrapper.cpp"
$GXX -std=c++2c -c "$BUILD_DIR/cerberus_lfssl_wrapper.cpp" -o "$OBJ" -O2 -fPIC -Wall -DBUILD_DLL -maes -msse2 -msse4.1 -I"$INCLUDE" -I"$KYBER_REF" -I"$DIL_REF" -I"$ARGON_INC"
OBJS+=("$OBJ")
done

# ---- Wrapper ----
OBJ="$BUILD_DIR/obj/wrapper.o"
echo "  CXX cerberus_lfssl_wrapper.cpp"
g++ -std=c\+\+2c -c "$BUILD_DIR/cerberus_lfssl_wrapper.cpp" -o "$OBJ" -O2 -fPIC -Wall -DBUILD_DLL -maes -msse2 -msse4.1 -I"$INCLUDE" -I"$KYBER_REF" -I"$DIL_REF" -I"$ARGON_INC"
OBJS+=("$OBJ")

# ---- Argon2 reference source files ----
for A in src/argon2.c src/core.c src/blake2/blake2b.c src/thread.c src/encoding.c src/opt.c; do
    SRC="$ARGON_SRC/$A"
    OBJ="$BUILD_DIR/obj/argon2_$(basename $A .c).o"
    echo "  CC  $A"
    gcc -c "$SRC" -o "$OBJ" -O2 -fPIC -Wall -I"$ARGON_INC" -I"$ARGON_SRC/src"
    OBJS+=("$OBJ")
done

# Link shared library
echo "  LD  -> $OUT"
g++ -shared -fPIC -o "$OUT" "${OBJS[@]}"

echo "=== libcerberus_lfssl.so built: $OUT ==="
ls -lh "$OUT"
