# Download ONNX Runtime for Windows (ROG G18 dev laptop)
# ========================================================
param(
    [string]$Version = "1.26.0",
    [string]$Arch = "win-x64-gpu"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$OrtDir = Join-Path $ProjectRoot "third_party\onnxruntime"
$ZipName = "onnxruntime-${Arch}-${Version}.zip"
$Url = "https://github.com/microsoft/onnxruntime/releases/download/v${Version}/${ZipName}"

Write-Host "[download-onnxruntime] Target: ${Arch} v${Version}"
Write-Host "[download-onnxruntime] Destination: ${OrtDir}"

New-Item -ItemType Directory -Force -Path $OrtDir | Out-Null

$MarkerFile = Join-Path $OrtDir ".download-marker"
if (Test-Path $MarkerFile) {
    $MarkerVersion = Get-Content $MarkerFile
    if ($MarkerVersion -eq $Version) {
        Write-Host "[download-onnxruntime] Already downloaded v${Version}"
        exit 0
    }
}

Write-Host "[download-onnxruntime] Downloading from GitHub releases..."
$ZipPath = Join-Path $OrtDir $ZipName
Invoke-WebRequest -Uri $Url -OutFile $ZipPath -ProgressAction Continue

Write-Host "[download-onnxruntime] Extracting..."
Expand-Archive -Path $ZipPath -DestinationPath $OrtDir -Force
Remove-Item $ZipPath

$Version | Set-Content $MarkerFile
Write-Host "[download-onnxruntime] Done. Version ${Version} ready."
