#requires -Version 5.1
# =============================================================================
# OpenVINO 2024.4 Windows ZIP Setup (Intel Core Ultra 9 275HX + AI Boost NPU)
# =============================================================================
# Run this from PowerShell as Administrator if you need to modify system PATH.
#
# Usage:
#   .\setup-openvino.zip.ps1
#
# What it does:
#   1. Downloads OpenVINO 2024.4 Windows ZIP (if missing)
#   2. Extracts to C:\openvino-2024.4
#   3. Sets OPENVINO_ROOT environment variable (user scope)
#   4. Adds runtime DLLs to user PATH
#   5. Prints diagnostic commands to verify NPU device detection
# =============================================================================

$ErrorActionPreference = "Stop"

$OpenVinoVersion = "2024.4.0"
$ZipUrl = "https://storage.openvinotoolkit.org/repositories/openvino/packages/2024.4/windows/w_openvino_toolkit_windows_2024.4.0.16579.c3152d32c9c_x86_64.zip"
$TargetDir = "C:\openvino-2024.4"
$ZipFile = "$env:TEMP\openvino_2024.4_windows.zip"

# --- 1. Download ---
if (-not (Test-Path $TargetDir)) {
    Write-Host "Downloading OpenVINO $OpenVinoVersion Windows ZIP..." -ForegroundColor Cyan
    if (-not (Test-Path $ZipFile)) {
        Invoke-WebRequest -Uri $ZipUrl -OutFile $ZipFile -UseBasicParsing
        Write-Host "Saved to $ZipFile" -ForegroundColor Green
    }

    Write-Host "Extracting to $TargetDir ..." -ForegroundColor Cyan
    Expand-Archive -LiteralPath $ZipFile -DestinationPath $TargetDir -Force
    Write-Host "Extraction complete." -ForegroundColor Green
} else {
    Write-Host "OpenVINO already exists at $TargetDir" -ForegroundColor Yellow
}

# --- 2. Environment setup ---
$RuntimeBin = "$TargetDir\runtime\bin\intel64\Release"
$RuntimeBinDebug = "$TargetDir\runtime\bin\intel64\Debug"
$RuntimeLib = "$TargetDir\runtime\lib\intel64\Release"
$RuntimeLibDebug = "$TargetDir\runtime\lib\intel64\Debug"

# OPENVINO_ROOT (user environment variable)
[System.Environment]::SetEnvironmentVariable("OPENVINO_ROOT", $TargetDir, "User")
Write-Host "Set OPENVINO_ROOT = $TargetDir" -ForegroundColor Green

# PATH: add runtime DLL directories
$CurrentPath = [System.Environment]::GetEnvironmentVariable("PATH", "User")
if ($CurrentPath -notlike "*$RuntimeBin*") {
    $NewPath = "$RuntimeBin;$RuntimeBinDebug;$RuntimeLib;$RuntimeLibDebug;$CurrentPath"
    [System.Environment]::SetEnvironmentVariable("PATH", $NewPath, "User")
    Write-Host "Added OpenVINO runtime to user PATH" -ForegroundColor Green
} else {
    Write-Host "PATH already contains OpenVINO runtime" -ForegroundColor Yellow
}

# --- 3. Optional: copy openvino.dll side-by-side with build output ---
# This makes LoadLibraryW(L"openvino.dll") succeed even if PATH is not set
# in the current PowerShell session.
$RepoRoot = (Get-Item $PSScriptRoot).Parent.Parent.FullName
$BuildOut = "$RepoRoot\build\Release"   # adjust to your CMake build tree
if (Test-Path $BuildOut) {
    Copy-Item "$RuntimeBin\openvino.dll" $BuildOut -ErrorAction SilentlyContinue
    Copy-Item "$RuntimeBin\openvino_c.dll" $BuildOut -ErrorAction SilentlyContinue
    Write-Host "Copied openvino.dll / openvino_c.dll to $BuildOut" -ForegroundColor Green
}

# --- 4. Diagnostics ---
Write-Host "`n=== OpenVINO NPU Diagnostics ===" -ForegroundColor Cyan
Write-Host "Run these commands in a NEW PowerShell window (so env vars are loaded):" -ForegroundColor White
Write-Host "  cd $TargetDir" -ForegroundColor Gray
Write-Host "  .\setupvars.ps1                     # Sets up the runtime environment" -ForegroundColor Gray
Write-Host "  cd $TargetDir\samples\cpp" -ForegroundColor Gray
Write-Host "  cmake -S . -B build && cmake --build build --config Release" -ForegroundColor Gray
Write-Host "  .\build\Release\hello_query_device.exe   # Should list NPU among devices" -ForegroundColor Gray
Write-Host ""
Write-Host "Device name for NPU in OpenVINO API: 'NPU'" -ForegroundColor Yellow
Write-Host ""
Write-Host "=== Intel NPU Driver Check ===" -ForegroundColor Cyan
Write-Host "  Get-PnpDevice -Class System | Where-Object { `$_.FriendlyName -like '*NPU*' }" -ForegroundColor Gray
Write-Host "  Ensure driver version >= 32.0.100.xxxx (Intel NPU Driver v32+)" -ForegroundColor Gray
Write-Host "================================" -ForegroundColor Cyan
