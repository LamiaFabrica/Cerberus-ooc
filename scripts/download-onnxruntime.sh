#!/usr/bin/env bash
# Download ONNX Runtime for Linux (UM790 Pro)
# =============================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ORT_DIR="${PROJECT_ROOT}/third_party/onnxruntime"
VERSION="1.26.0"
ARCH="linux-x64-gpu"
ZIP_NAME="onnxruntime-${ARCH}-${VERSION}.tgz"
URL="https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/${ZIP_NAME}"

echo "[download-onnxruntime] Target: ${ARCH} v${VERSION}"
echo "[download-onnxruntime] Destination: ${ORT_DIR}"

mkdir -p "${ORT_DIR}"

if [ -f "${ORT_DIR}/.download-marker" ]; then
    MARKER_VERSION=$(cat "${ORT_DIR}/.download-marker")
    if [ "${MARKER_VERSION}" = "${VERSION}" ]; then
        echo "[download-onnxruntime] Already downloaded v${VERSION}"
        exit 0
    fi
fi

echo "[download-onnxruntime] Downloading from GitHub releases..."
curl -L --progress-bar "${URL}" -o "${ORT_DIR}/${ZIP_NAME}"

echo "[download-onnxruntime] Extracting..."
tar -xzf "${ORT_DIR}/${ZIP_NAME}" -C "${ORT_DIR}"
rm "${ORT_DIR}/${ZIP_NAME}"

echo "${VERSION}" > "${ORT_DIR}/.download-marker"
echo "[download-onnxruntime] Done. Version ${VERSION} ready."
