# build_dll.ps1
# Rebuild cerberus_lfssl.dll with PQC (Kyber + Dilithium), plus existing crypto
param([string]$BuildDir = $PSScriptRoot)

$ErrorActionPreference = "Stop"

# LFSSL root relative to lfssl_bridge
$LFSSL     = Join-Path $BuildDir "..\..\LFSSL - Lamia Fabrica SSL"
$INCLUDE   = Join-Path $LFSSL "include"
$LFSSLSRC  = Join-Path $LFSSL "src"
$KYBER_REF = Join-Path $LFSSL "lib\kyber-main\ref"
$DIL_REF   = Join-Path $LFSSL "lib\dilithium-master\ref"

$OUT = Join-Path $BuildDir "cerberus_lfssl.dll"
$LIB = Join-Path $BuildDir "cerberus_lfssl.lib"

function Compile-CFile($file, $obj, $defs, $incs) {
    $cmd = "gcc -c `"$file`" -o `"$obj`" -O2 -Wall $defs $incs"
    Write-Host "  CC  $file -> $obj"
    Invoke-Expression $cmd | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "FAILED: $cmd" }
}

function Compile-CppFile($file, $obj, $defs, $incs) {
    $cmd = "g++ -std=c++23 -c `"$file`" -o `"$obj`" -O2 -Wall -DBUILD_DLL -mavx2 $defs $incs"
    Write-Host "  CXX $file -> $obj"
    Invoke-Expression $cmd | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "FAILED: $cmd" }
}

Write-Host "=== Building cerberus_lfssl.dll with PQC ==="

$objects = @()

# ---- Kyber ref for each K ----
foreach ($k in @(2,3,4)) {
    $kyber_srcs = @("kem.c","indcpa.c","poly.c","polyvec.c","cbd.c","reduce.c","ntt.c","verify.c","symmetric-shake.c")
    foreach ($s in $kyber_srcs) {
        $file = Join-Path $KYBER_REF $s
        $obj  = Join-Path $BuildDir ("kyber_k{0}_{1}.o" -f $k, ($s -replace '\.c$',''))
        Compile-CFile $file $obj "-DKYBER_K=$k" "-I`"$KYBER_REF`""
        $objects += $obj
    }
}

# Kyber fips202 (once)
$kyber_fips202_obj = Join-Path $BuildDir "kyber_fips202.o"
Compile-CFile (Join-Path $KYBER_REF "fips202.c") $kyber_fips202_obj "" "-I`"$KYBER_REF`""
$objects += $kyber_fips202_obj

# ---- Dilithium ref for each mode ----
foreach ($m in @(2,3,5)) {
    $dil_srcs = @("sign.c","packing.c","polyvec.c","poly.c","ntt.c","reduce.c","rounding.c","symmetric-shake.c")
    foreach ($s in $dil_srcs) {
        $file = Join-Path $DIL_REF $s
        $obj  = Join-Path $BuildDir ("dil_m{0}_{1}.o" -f $m, ($s -replace '\.c$',''))
        Compile-CFile $file $obj "-DDILITHIUM_MODE=$m" "-I`"$DIL_REF`""
        $objects += $obj
    }
}

# Dilithium fips202 (once)
$dil_fips202_obj = Join-Path $BuildDir "dil_fips202.o"
Compile-CFile (Join-Path $DIL_REF "fips202.c") $dil_fips202_obj "" "-I`"$DIL_REF`""
$objects += $dil_fips202_obj

# LFSSL individual .cpp files
$lfssl_cpp = @(
    (Join-Path $LFSSLSRC "crypto\sha256.cpp"),
    (Join-Path $LFSSLSRC "crypto\sha384.cpp"),
    (Join-Path $LFSSLSRC "crypto\sha512.cpp"),
    (Join-Path $LFSSLSRC "crypto\aes256.cpp"),
    (Join-Path $LFSSLSRC "crypto\aes256_gcm_hardware.cpp"),
    (Join-Path $LFSSLSRC "crypto\pbkdf2.cpp"),
    (Join-Path $LFSSLSRC "crypto\hkdf.cpp"),
    (Join-Path $LFSSLSRC "crypto\random.cpp"),
    (Join-Path $LFSSLSRC "kyber_kem_wrapper.cpp"),
    (Join-Path $LFSSLSRC "dilithium_wrapper.cpp"),
    (Join-Path $LFSSLSRC "pqc_randombytes.cpp")
)
foreach ($cpp in $lfssl_cpp) {
    $name = [System.IO.Path]::GetFileNameWithoutExtension($cpp)
    $obj  = Join-Path $BuildDir ("lfssl_{0}.o" -f $name)
    Compile-CppFile $cpp $obj "" "-I`"$INCLUDE`" -I`"$KYBER_REF`" -I`"$DIL_REF`""
    $objects += $obj
}

# Wrapper .cpp
$wrapper     = Join-Path $BuildDir "cerberus_lfssl_wrapper.cpp"
$wrapper_obj = Join-Path $BuildDir "wrapper.o"
Compile-CppFile $wrapper $wrapper_obj "" "-I`"$INCLUDE`" -I`"$KYBER_REF`" -I`"$DIL_REF`""
$objects += $wrapper_obj

# Link
$objs_str = ($objects | ForEach-Object { '"{0}"' -f $_ }) -join ' '
$linkcmd = "g++ -shared -o `"$OUT`" $objs_str -Wl,--out-implib,`"$LIB`" -lbcrypt"
Write-Host "  LD  -> $OUT"
Invoke-Expression $linkcmd | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Link failed" }

Write-Host "=== DLL built: $OUT ==="

# Verify exports
Write-Host "  Exports:"
$dumpbin = Get-Command dumpbin -ErrorAction SilentlyContinue
if ($dumpbin) {
    dumpbin /exports "$OUT" 2>$null | Select-String "cerberus_lfssl_"
} else {
    nm "$OUT" | Select-String "cerberus_lfssl_"
}
