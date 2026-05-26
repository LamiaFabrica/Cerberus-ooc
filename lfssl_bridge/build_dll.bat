@echo off
setlocal EnableDelayedExpansion

REM Build cerberus_lfssl.dll with PQC (Kyber + Dilithium) + existing crypto

set "BUILD_DIR=%~dp0"
set "LFSSL=%BUILD_DIR%..\..\LFSSL - Lamia Fabrica SSL"
set "INCLUDE=%LFSSL%\include"
set "LFSSLSRC=%LFSSL%\src"
set "KYBER_REF=%LFSSL%\lib\kyber-main\ref"
set "DIL_REF=%LFSSL%\lib\dilithium-master\ref"
set "ARGON_SRC=%LFSSL%\phc-argon2"
set "ARGON_INC=%ARGON_SRC%\include"
set "OUT=%BUILD_DIR%cerberus_lfssl.dll"
set "LIB=%BUILD_DIR%cerberus_lfssl.lib"

echo === Building cerberus_lfssl.dll with PQC ===

set OBJS=

REM ---- Kyber ref for each K ----
for %%K in (2 3 4) do (
    for %%S in (kem indcpa poly polyvec cbd reduce ntt verify symmetric-shake) do (
        set SRC=%KYBER_REF%\%%S.c
        set OBJ=%BUILD_DIR%kyber_k%%K_%%S.o
        echo   CC  %%S.c for K=%%K
        gcc -c "!SRC!" -o "!OBJ!" -O2 -Wall -DKYBER_K=%%K -I"%KYBER_REF%"
        if errorlevel 1 exit /b 1
        set OBJS=!OBJS! "!OBJ!"
    )
)

REM Kyber fips202 (once)
set OBJ=%BUILD_DIR%kyber_fips202.o
gcc -c "%KYBER_REF%\fips202.c" -o "%OBJ%" -O2 -Wall -I"%KYBER_REF%"
if errorlevel 1 exit /b 1
set OBJS=%OBJS% "%OBJ%"

REM ---- Dilithium ref for each mode ----
for %%M in (2 3 5) do (
    for %%S in (sign packing polyvec poly ntt reduce rounding symmetric-shake) do (
        set SRC=%DIL_REF%\%%S.c
        set OBJ=%BUILD_DIR%dil_m%%M_%%S.o
        echo   CC  %%S.c for mode=%%M
        gcc -c "!SRC!" -o "!OBJ!" -O2 -Wall -DDILITHIUM_MODE=%%M -I"%DIL_REF%"
        if errorlevel 1 exit /b 1
        set OBJS=!OBJS! "!OBJ!"
    )
)

REM Dilithium fips202 (once)
set OBJ=%BUILD_DIR%dil_fips202.o
gcc -c "%DIL_REF%\fips202.c" -o "%OBJ%" -O2 -Wall -I"%DIL_REF%"
if errorlevel 1 exit /b 1
set OBJS=%OBJS% "%OBJ%"

REM ---- LFSSL individual .cpp files ----
for %%F in (
    crypto\sha256
    crypto\sha384
    crypto\sha512
    crypto\aes256
    crypto\aes256_gcm_hardware
    crypto\pbkdf2
    crypto\hkdf
    crypto\random
    kyber_kem_wrapper
    dilithium_wrapper
    pqc_randombytes
) do (
    set CPP=%LFSSLSRC%\%%F.cpp
    set OBJ=%BUILD_DIR%lfssl_%%~nF.o
    echo   CXX %%~nF.cpp
    g++ -std=c++23 -c "!CPP!" -o "!OBJ!" -O2 -Wall -DBUILD_DLL -maes -msse2 -msse4.1 -I"%INCLUDE%" -I"%KYBER_REF%" -I"%DIL_REF%"
    if errorlevel 1 exit /b 1
    set OBJS=!OBJS! "!OBJ!"
)

REM ---- Wrapper ----
set OBJ=%BUILD_DIR%wrapper.o
echo   CXX cerberus_lfssl_wrapper.cpp
g++ -std=c++23 -c "%BUILD_DIR%cerberus_lfssl_wrapper.cpp" -o "%OBJ%" -O2 -Wall -DBUILD_DLL -maes -msse2 -msse4.1 -I"%INCLUDE%" -I"%KYBER_REF%" -I"%DIL_REF%" -I"%ARGON_INC%"
if errorlevel 1 exit /b 1
set OBJS=%OBJS% "%OBJ%"

REM ---- Argon2 reference source files ----
for %%A in (src/argon2.c src/core.c src/blake2/blake2b.c src/thread.c src/encoding.c src/opt.c) do (
    set SRC=%ARGON_SRC%\%%A
    set OBJ=%BUILD_DIR%argon2_%%~nA.o
    echo   CC  %%A
    gcc -c "!SRC!" -o "!OBJ!" -O2 -Wall -I"%ARGON_INC%" -I"%ARGON_SRC%\src"
    if errorlevel 1 exit /b 1
    set OBJS=!OBJS! "!OBJ!"
)

REM Link (after all objects collected)
echo   LD  -^> %OUT%
g++ -shared -o "%OUT%" %OBJS% -Wl,--out-implib,"%LIB%" -lbcrypt
if errorlevel 1 exit /b 1

echo === DLL built: %OUT% ===
endlocal
