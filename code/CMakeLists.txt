# =============================================================================
# Top-level CMake for Cerberus Heterogeneous AI Inference Runtime
# Dev target:  ROG Strix G18 (Intel Ultra 9 275HX + RTX 5070 Ti + Intel AI Boost NPU)
# Prod target: MinisForum UM790 Pro (Zen 4, AVX-512, ROCm 6.0+, HailoRT 4.20+)
# -march=native auto-detects the host CPU on every build machine.
# =============================================================================
cmake_minimum_required(VERSION 3.28)

# Platform-conditional toolchain selection
# Windows: Force the known-good GCC 15.2.0 toolchain (WinLibs MCF UCRT)
# Linux (Ubuntu/WSL): Require GCC 15+ for full C++26 support
if(WIN32)
    set(CMAKE_C_COMPILER   "C:/gcc-15.2.0/mingw64/bin/gcc.exe"  CACHE FILEPATH "C compiler" FORCE)
    set(CMAKE_CXX_COMPILER "C:/gcc-15.2.0/mingw64/bin/g++.exe" CACHE FILEPATH "C++ compiler" FORCE)
else()
    # Ubuntu: GCC 15+ is required for full C++26 feature completeness.
    # GCC 14 is insufficient — we need designated initializers, std::print,
    # and other P2300/P2537 features that are only stable in GCC 15+.
    find_program(GXX_15 g++-15)
    find_program(GCC_15 gcc-15)
    if(GXX_15 AND GCC_15)
        set(CMAKE_C_COMPILER   "${GCC_15}" CACHE FILEPATH "C compiler" FORCE)
        set(CMAKE_CXX_COMPILER "${GXX_15}" CACHE FILEPATH "C++ compiler" FORCE)
        message(STATUS "Linux toolchain: GCC 15 (${GXX_15})")
    else()
        message(FATAL_ERROR
            "GCC 15+ is required on Linux but was not found.\n"
            "  Install GCC 15 from the ubuntu-toolchain-r PPA:\n"
            "    sudo add-apt-repository ppa:ubuntu-toolchain-r/test\n"
            "    sudo apt-get update\n"
            "    sudo apt-get install g++-15 gcc-15\n"
            "  Or build GCC 15 from source:\n"
            "    wget https://ftp.gnu.org/gnu/gcc/gcc-15.2.0/gcc-15.2.0.tar.gz\n"
            "    # ... standard GCC build procedure\n")
    endif()
endif()

project(um790_pipeline
    VERSION 1.0.0
    DESCRIPTION "AI Image Generation Pipeline for UM790 Pro"
    HOMEPAGE_URL "https://github.com/hybridquant/um790-pipeline"
    LANGUAGES CXX
)

# ---------------------------------------------------------------------------
# C++26 Standard Configuration
# ---------------------------------------------------------------------------
# CMake 3.28 does not recognise "CXX26" as a dialect name (added in 3.30).
# GCC 14+ supports -std=c++26 natively. We set the flag manually when CMake
# is too old to handle cxx_std_26.
if(CMAKE_VERSION VERSION_LESS "3.30")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++26" CACHE STRING "C++ flags" FORCE)
    message(STATUS "CMake ${CMAKE_VERSION}: C++26 enabled via -std=c++26 flag")
    # Do NOT set CMAKE_CXX_STANDARD to 26 here — it leaks into try_compile
    # subprojects and CMake 3.28 fails because it doesn't recognise CXX26.
    set(CERBERUS_CXX_STANDARD_SET_BY_FLAGS TRUE)
else()
    set(CMAKE_CXX_STANDARD 26)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Feature Detection: Verify the compiler actually supports requested C++26
# ---------------------------------------------------------------------------
# CMake 3.28 does not recognise "CXX26" as a dialect name. When
# CMAKE_CXX_STANDARD is set to anything (even 20), CMake appends -std=c++20
# AFTER CMAKE_REQUIRED_FLAGS, overriding -std=c++26. We must CLEAR
# CMAKE_CXX_STANDARD entirely before the feature-check module.
set(_SAVED_CXX_STANDARD ${CMAKE_CXX_STANDARD})
unset(CMAKE_CXX_STANDARD)
set(CMAKE_CXX_STANDARD_REQUIRED OFF)
include(cmake/CheckCXX26Features.cmake)
set(CMAKE_CXX_STANDARD ${_SAVED_CXX_STANDARD})
unset(_SAVED_CXX_STANDARD)
if(CERBERUS_CXX_STANDARD_SET_BY_FLAGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++26" CACHE STRING "C++ flags" FORCE)
endif()

# ---------------------------------------------------------------------------
# Threading support (required by std::thread, std::jthread, std::mutex)
# ---------------------------------------------------------------------------
find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# Toolchain & Vendor Settings
# ---------------------------------------------------------------------------
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ROCm/HIP toolchain — we do NOT enable_language(HIP) because all sources are
# plain C++ that optionally calls HIP APIs via #ifdef guards.
# The hip::host target is found via find_package(HIP) when available.

# ---------------------------------------------------------------------------
# Find Dependencies
# ---------------------------------------------------------------------------
include(FindPkgConfig)

# --- ROCm/HIP ---
find_package(HIP CONFIG QUIET
    HINTS
        /opt/rocm/lib/cmake/hip
        $ENV{ROCM_PATH}/lib/cmake/hip
)
if(TARGET hip::host)
    message(STATUS "HIP found: ${HIP_VERSION}")
else()
    message(STATUS "HIP not found — GPU compute disabled")
endif()

# ROCm SMI (GPU telemetry)
if(TARGET hip::host)
    find_package(rocm_smi64 QUIET
        HINTS /opt/rocm/lib/cmake/rocm_smi64
              $ENV{ROCM_PATH}/lib/cmake/rocm_smi64)
endif()

# --- CUDA (NVIDIA GPU) ---
# Primary GPU backend for NVIDIA RTX GPUs (replaces ROCm/HIP on Windows)
# Required for real UNet/VAE inference on RTX 5070 Ti and similar hardware
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA found: ${CUDAToolkit_VERSION} at ${CUDAToolkit_BIN_DIR}")
    # NVML (NVIDIA Management Library) for GPU telemetry — always available with CUDA toolkit
    find_library(NVML_LIB NAMES nvml nvidia-ml nvml_linux
        HINTS
            ${CUDAToolkit_BIN_DIR}/../lib/x64
            ${CUDAToolkit_BIN_DIR}/../lib
            ${CUDAToolkit_ROOT}/lib/x64
            ${CUDAToolkit_ROOT}/lib
        DOC "NVIDIA Management Library (nvml.lib / libnvidia-ml.so)"
    )
    find_path(NVML_INCLUDE_DIR NAMES nvml.h
        HINTS
            ${CUDAToolkit_BIN_DIR}/../include
            ${CUDAToolkit_ROOT}/include
        DOC "NVML header directory"
    )
    if(NVML_LIB)
        message(STATUS "NVML found: ${NVML_LIB}")
    else()
        message(STATUS "NVML not found — NVIDIA GPU telemetry disabled")
    endif()
    if(NVML_INCLUDE_DIR)
        message(STATUS "NVML headers: ${NVML_INCLUDE_DIR}")
    else()
        message(STATUS "NVML headers not found — will rely on __has_include probe")
    endif()
else()
    message(STATUS "CUDA not found — GPU compute disabled (HIP fallback if available)")
endif()

# --- DirectML (Intel AI Boost NPU / NVIDIA DML) ---
# Intel AI Boost NPU (PCI VEN_8086 DEV_AD1D) and NVIDIA DML EP
# require DirectML headers + onnxruntime_providers_dml.lib
# On Windows 11, DirectML is available via the Windows SDK and ORT DML EP
if(WIN32)
    set(UM790_HAS_DIRECTML ON CACHE BOOL "DirectML available on Windows via SDK")
else()
    set(UM790_HAS_DIRECTML OFF CACHE BOOL "DirectML not available on non-Windows")
endif()
if(UM790_HAS_DIRECTML)
    message(STATUS "DirectML: available (Windows SDK) — Intel AI Boost NPU + NVIDIA DML EP enabled")
else()
    message(STATUS "DirectML: not available — NPU/DML inference paths disabled")
endif()

if(PkgConfig_FOUND)
    pkg_check_modules(HAILORT hailort)
    if(NOT HAILORT_FOUND)
        message(STATUS "HailoRT not found via pkg-config — Hailo offload disabled")
    endif()
else()
    message(STATUS "pkg-config not found — HailoRT detection skipped")
endif()

# --- ONNX Runtime ---
# Use the staged ONNX Runtime SDK (GPU package with CUDA/DML EP support).
# The SDK was downloaded and staged by parent task t_31fcdbae to:
#   <project-root>/third_party/onnxruntime/onnxruntime-win-x64-gpu-1.26.0
# On Linux, set -DONNXRUNTIME_ROOT=... to point at the extracted tarball.
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/onnxruntime/onnxruntime-win-x64-gpu-1.26.0" CACHE PATH "ONNX Runtime SDK root directory")

find_library(ONNXRUNTIME_LIB NAMES onnxruntime
    HINTS
        ${ONNXRUNTIME_ROOT}/lib
    DOC "ONNX Runtime shared library"
)
find_path(ONNXRUNTIME_INCLUDE NAMES onnxruntime_cxx_api.h onnxruntime_cxx_inline.h
    HINTS
        ${ONNXRUNTIME_ROOT}/include
    DOC "ONNX Runtime C++ API headers"
)
if(ONNXRUNTIME_LIB AND ONNXRUNTIME_INCLUDE)
    message(STATUS "ONNX Runtime: found (lib=${ONNXRUNTIME_LIB}, include=${ONNXRUNTIME_INCLUDE})")
    set(ONNXRUNTIME_LIBRARIES ${ONNXRUNTIME_LIB})
    set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_INCLUDE})
else()
    message(STATUS "ONNX Runtime: NOT found — inference will be unavailable")
    message(STATUS "  Set ONNXRUNTIME_ROOT to point to the ORT SDK directory")
endif()

# Find ORT CUDA provider library (optional)
find_library(ORT_CUDA_PROVIDER_LIB NAMES onnxruntime_providers_cuda
    HINTS
        ${ONNXRUNTIME_ROOT}/lib
    DOC "ONNX Runtime CUDA provider library"
)
if(ORT_CUDA_PROVIDER_LIB)
    message(STATUS "ORT CUDA provider: found (${ORT_CUDA_PROVIDER_LIB})")
else()
    message(STATUS "ORT CUDA provider: not found — CUDA EP will use CPU fallback")
endif()

# Find ORT TensorRT provider library (optional)
find_library(ORT_TENSORRT_PROVIDER_LIB NAMES onnxruntime_providers_tensorrt
    HINTS
        ${ONNXRUNTIME_ROOT}/lib
    DOC "ONNX Runtime TensorRT provider library"
)
if(ORT_TENSORRT_PROVIDER_LIB)
    message(STATUS "ORT TensorRT provider: found (${ORT_TENSORRT_PROVIDER_LIB})")
else()
    message(STATUS "ORT TensorRT provider: not found — TensorRT EP disabled")
endif()

# Find ORT shared provider library (required by CUDA/TensorRT EPs)
find_library(ORT_SHARED_PROVIDER_LIB NAMES onnxruntime_providers_shared
    HINTS
        ${ONNXRUNTIME_ROOT}/lib
    DOC "ONNX Runtime shared provider library"
)
if(ORT_SHARED_PROVIDER_LIB)
    message(STATUS "ORT shared provider: found (${ORT_SHARED_PROVIDER_LIB})")
else()
    message(STATUS "ORT shared provider: not found — provider EPs may fail at runtime")
endif()

# --- Runtime DLL deployment (Windows) ---
# Copy onnxruntime.dll and provider DLLs next to every executable so tests
# can run without modifying PATH.
if(WIN32 AND ONNXRUNTIME_LIB)
    set(ONNXRUNTIME_DLLS)
    if(EXISTS "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll")
        list(APPEND ONNXRUNTIME_DLLS "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll")
    endif()
    if(ORT_CUDA_PROVIDER_LIB AND EXISTS "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_cuda.dll")
        list(APPEND ONNXRUNTIME_DLLS "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_cuda.dll")
    endif()
    if(ORT_TENSORRT_PROVIDER_LIB AND EXISTS "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_tensorrt.dll")
        list(APPEND ONNXRUNTIME_DLLS "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_tensorrt.dll")
    endif()
    if(ORT_SHARED_PROVIDER_LIB AND EXISTS "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.dll")
        list(APPEND ONNXRUNTIME_DLLS "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.dll")
    endif()

    if(ONNXRUNTIME_DLLS)
        message(STATUS "ONNX Runtime DLLs to copy: ${ONNXRUNTIME_DLLS}")
    endif()
endif()

# --- Intel OpenVINO (NPU backend) ---
# NOTE: OpenVINO Windows binaries are MSVC-built. Because we compile with MinGW,
# we CANNOT link the C++ API (ABI mismatch). We use the C API and dynamically
# load openvino.dll at runtime. CMake only needs the header path.
set(OPENVINO_ROOT "$ENV{OPENVINO_ROOT}" CACHE PATH "OpenVINO runtime root directory")
if(NOT OPENVINO_ROOT)
    set(OPENVINO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../openvino" CACHE PATH "OpenVINO runtime root directory")
endif()

find_path(OPENVINO_INCLUDE NAMES openvino/c/openvino.h
    HINTS
        ${OPENVINO_ROOT}/runtime/include
        ${OPENVINO_ROOT}/include
    DOC "OpenVINO C API headers"
)

if(OPENVINO_INCLUDE)
    message(STATUS "OpenVINO C API: found (include=${OPENVINO_INCLUDE})")
else()
    message(STATUS "OpenVINO: NOT found — Intel NPU backend disabled")
    message(STATUS "  Set OPENVINO_ROOT to the extracted ZIP directory")
endif()

# --- GoogleTest (FetchContent preferred for hermetic builds) ---
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
    GIT_SHALLOW    TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# --- Standalone mdspan (upgrade path note) ---
# UM790_HAS_STD_MDSPAN=0 means the MinGW-W64 sysroot headers predate C++23
# <mdspan> (GCC 14 compiler, older sysroot). TensorView is guarded by
# #if UM790_HAS_STD_MDSPAN and compiles out cleanly.
#
# Upgrade paths to enable native <mdspan>:
#   1. Replace MinGW-W64 with winlibs.com GCC 14.2.0 (ships full C++23 headers)
#      https://winlibs.com/ → UCRT runtime, GCC 14.2.0, LLVM/Clang
#   2. Use w64devkit (Andrea Corallo's distribution with C++23 headers)
#   3. On Ubuntu 24.04: system GCC 14 ships <mdspan> natively.
#
# If MDSPAN_INCLUDE_DIR is set to an external mdspan header directory
# (e.g., Kokkos reference impl), that path is added to the include dirs
# and the detection is re-run automatically.
if(DEFINED MDSPAN_INCLUDE_DIR AND NOT UM790_HAS_STD_MDSPAN)
    message(STATUS "  mdspan: retrying detection with MDSPAN_INCLUDE_DIR=${MDSPAN_INCLUDE_DIR}")
    set(CMAKE_REQUIRED_INCLUDES_SAVED ${CMAKE_REQUIRED_INCLUDES})
    list(APPEND CMAKE_REQUIRED_INCLUDES "${MDSPAN_INCLUDE_DIR}")
    unset(UM790_HAS_STD_MDSPAN CACHE)
    check_cxx_source_compiles("
        #include <mdspan>
        int main() {
            float buf[12] = {};
            std::mdspan<float, std::dextents<std::size_t, 2>> m(buf, 3, 4);
            return static_cast<int>(m.extent(0)) - 3;
        }
    " UM790_HAS_STD_MDSPAN)
    set(CMAKE_REQUIRED_INCLUDES ${CMAKE_REQUIRED_INCLUDES_SAVED})
    if(UM790_HAS_STD_MDSPAN)
        target_include_directories(um790_includes INTERFACE
            $<BUILD_INTERFACE:${MDSPAN_INCLUDE_DIR}>)
        message(STATUS "  mdspan: enabled via MDSPAN_INCLUDE_DIR")
    else()
        message(STATUS "  mdspan: still unavailable after MDSPAN_INCLUDE_DIR retry")
    endif()
elseif(NOT UM790_HAS_STD_MDSPAN)
    message(STATUS "  mdspan: unavailable (sysroot gap) — TensorView compiled out")
    message(STATUS "  mdspan: set -DMDSPAN_INCLUDE_DIR=<path> to enable via external headers")
else()
    message(STATUS "  mdspan: native <mdspan> available (UM790_HAS_STD_MDSPAN=1)")
endif()

# --- PsiForceDB ecosystem headers (optional, for extension integration) ---
# Navigate from .../Cerberus - Copy/code up to Projects/, then into PsiForceDB
get_filename_component(PSIFORCEDB_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../PsiForceDB" ABSOLUTE)
set(PSIFORCEDB_INCLUDE_DIRS
    "${PSIFORCEDB_ROOT}/Src/Inc/Intake/include"
    "${PSIFORCEDB_ROOT}/Src/Database/Intake/include"
    "${PSIFORCEDB_ROOT}/Src/Unclassified/Intake/include"
    "${PSIFORCEDB_ROOT}/Src/Inc/Concurrency/include"
)
# NOTE: actual target_include_directories() calls are deferred until um790_includes
# is defined (see below after add_library(um790_includes INTERFACE)).

# Detect whether the real PsiForceDB extension_interface.hpp is present
set(PSIFORCEDB_HEADERS_FOUND FALSE)
foreach(dir IN LISTS PSIFORCEDB_INCLUDE_DIRS)
    if(IS_DIRECTORY "${dir}")
        list(APPEND PSIFORCEDB_VALID_INCLUDE_DIRS "${dir}")
        if(EXISTS "${dir}/multimodel/extension_interface.hpp")
            set(PSIFORCEDB_HEADERS_FOUND TRUE)
            message(STATUS "PsiForceDB extension_interface.hpp FOUND: ${dir}/multimodel/extension_interface.hpp")
        endif()
    endif()
endforeach()

if(PSIFORCEDB_HEADERS_FOUND)
    message(STATUS "PsiForceDB real headers available — CERBERUS_USE_REAL_PSIFORCEDB_HEADERS will be defined")
else()
    message(STATUS "PsiForceDB real headers NOT available — using standalone replica")
endif()

# ---------------------------------------------------------------------------
# Compiler Flags (target-based, no globals)
# ---------------------------------------------------------------------------
# Base flags applied to all targets via interface library
# C++26 coroutine support requires GCC 14+ or Clang 18+
add_library(um790_compile_options INTERFACE)
target_compile_options(um790_compile_options INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
        -march=native        # Detect host CPU features (Zen 4 on UM790, Arrow Lake on G18 dev)
        -Wall -Wextra -Wpedantic
        -std=c++26
        -fPIC               # Required for static libs linked into .so on Linux

        # --- Maximum diagnostic verbosity ---
        # Show the source line with a caret (^) pointing at the exact character
        -fdiagnostics-show-caret
        # Show which -Wfoo flag controls each warning so we know exactly what to fix
        -fdiagnostics-show-option
        # Show labels on fix-it hints inline in the source listing
        -fdiagnostics-show-labels
        # Expand templates fully in diagnostics (catches template-induced bugs)
        -fdiagnostics-show-template-tree
        # Report every error — do NOT stop at the first N (we want the full picture)
        -fmax-errors=0
        # Increase template backtrace depth for C++26 concept diagnostics
        -fconcepts-diagnostics-depth=4
        # Show full paths in diagnostics (not just basename)
        -fdiagnostics-path-format=inline-events
        # Do NOT use colour in the compiler output — the split script handles that
        -fno-diagnostics-color
    >
    $<$<CONFIG:Debug>:
        -O0 -g3 -fno-omit-frame-pointer
    >
    $<$<CONFIG:Release,RelWithDebInfo,MinSizeRel>:
        -O3
    >
    $<$<CXX_COMPILER_ID:GNU>:-fcoroutines>
    $<$<CXX_COMPILER_ID:Clang,AppleClang>:-stdlib=libc++ -fcoroutines>
)
target_compile_features(um790_compile_options INTERFACE
    $<$<NOT:$<VERSION_LESS:${CMAKE_VERSION},3.30>>:cxx_std_26>
)

# ---------------------------------------------------------------------------
# Link Flags
# ---------------------------------------------------------------------------
add_library(um790_link_options INTERFACE)
target_link_options(um790_link_options INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
        -Wl,--as-needed
    >
)

# ---------------------------------------------------------------------------
# Include Directories
# ---------------------------------------------------------------------------
add_library(um790_includes INTERFACE)
target_include_directories(um790_includes INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/openvino/include>
    $<INSTALL_INTERFACE:include>
)
# ORT headers are treated as SYSTEM includes to suppress their warnings
target_include_directories(um790_includes SYSTEM INTERFACE
    $<$<BOOL:${HAILORT_FOUND}>:${HAILORT_INCLUDE_DIRS}>
    $<$<BOOL:${ONNXRUNTIME_INCLUDE}>:${ONNXRUNTIME_INCLUDE}>
    $<$<BOOL:${NVML_INCLUDE_DIR}>:${NVML_INCLUDE_DIR}>
    $<$<BOOL:${OPENVINO_INCLUDE}>:${OPENVINO_INCLUDE}>
)

# --- PsiForceDB ecosystem includes (deferred until um790_includes exists) ---
foreach(dir IN LISTS PSIFORCEDB_INCLUDE_DIRS)
    if(IS_DIRECTORY "${dir}")
        target_include_directories(um790_includes SYSTEM INTERFACE "${dir}")
        message(STATUS "PsiForceDB include: ${dir}")
    else()
        message(WARNING "PsiForceDB include NOT FOUND: ${dir}")
    endif()
endforeach()

# --- LFSSL ecosystem includes (sibling project, added AFTER PsiForceDB so that
#     PsiForceDB's monolithic LFSSL_Native_Crypto.hpp takes precedence for the
#     inline SHA256/HMAC/PBKDF2 subset Cerberus already consumes) ---
get_filename_component(LFSSL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../LFSSL - Lamia Fabrica SSL" ABSOLUTE)
if(IS_DIRECTORY "${LFSSL_ROOT}/include")
    target_include_directories(um790_includes SYSTEM INTERFACE "${LFSSL_ROOT}/include")
    message(STATUS "LFSSL include: ${LFSSL_ROOT}/include")
    set(LFSSL_HEADERS_FOUND TRUE)
else()
    message(STATUS "LFSSL include NOT FOUND")
    set(LFSSL_HEADERS_FOUND FALSE)
endif()

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
set(UM790_PIPELINE_SOURCES
    src/logger.cpp
    src/benchmark_logger.cpp
    src/safe_write.cpp
    src/pipeline_integration.cpp
    src/hailo_monitor.cpp
    src/utilization_watchdog.cpp
    src/staging_manager.cpp
    src/gpu_monitor.cpp
    src/clip_tokenizer.cpp
    src/deis_scheduler.cpp
    src/health_score.cpp
    src/npu_pipeline.cpp
    src/npu_encoder.cpp
    src/npu_accelerator.cpp
    src/npu_backend_unified.cpp
    src/async_pipeline.cpp
    src/tiered_memory_manager.cpp
    src/cluster_transport.cpp
    src/cerberus_execution_coordinator.cpp
    src/david_propup_engine.cpp
    src/cerberus_native_kernels.cpp
    src/cerberus_native_backend.cpp
    src/cerberus_fused_kernels.cpp
    src/cerberus_quantized_kernels.cpp
    src/cerberus_graph_engine.cpp
    src/cerberus_decision_engine.cpp
    src/cerberus_runtime.cpp
    src/cerberus_predictor.cpp
    src/cerberus_shadow_state.cpp
    src/cerberus_command_parser.cpp
    src/cerberus_command_executor.cpp
    src/cerberus_api_gateway.cpp
    src/cerberus_slipstream.cpp
    src/cerberus_metro.cpp
    src/cerberus_glow_engine.cpp
    src/cerberus_gguf_parser.cpp
    src/cerberus_psiforcedb_extension.cpp
    src/cerberus_psiforcedb_graph_bridge.cpp
    src/cerberus_psiforcedb_security.cpp
    src/cerberus_psiforcedb_real_integration.cpp
    src/cerberus_smdi.cpp
    src/cerberus_local_maintenance_db.cpp
    src/cerberus_user_security.cpp
    src/cerberus_jwt_session.cpp
    src/cerberus_first_run.cpp
    src/intel_npu_telemetry.cpp
)

list(APPEND UM790_PIPELINE_SOURCES src/hip_graph_denoiser.cpp)

set(UM790_PIPELINE_HEADERS
    include/hq/pipeline.hpp
    include/hq/staging_manager.hpp
    include/hq/utilization_watchdog.hpp
    include/hq/pinned_staging.hpp
    include/hq/hailo_monitor.hpp
    include/hq/clip_tokenizer.hpp
    include/hq/deis_scheduler.hpp
    include/hq/gpu_monitor.hpp
    include/hq/hip_graph_denoiser.hpp
    include/hq/health_score.hpp
    include/hq/tensor_view.hpp
    include/hq/npu_pipeline.hpp
    include/hq/async_pipeline.hpp
    include/hq/cerberus_api.h
    include/hq/inference_server.hpp
    include/hq/tiered_memory_manager.hpp
    include/hq/cluster_transport.hpp
    include/hq/benchmark_logger.hpp
    include/hq/npu_backend_unified.hpp
    include/hq/cerberus_psiforcedb_security.hpp
)

# ---------------------------------------------------------------------------
# Target: um790_pipeline (static library)
# ---------------------------------------------------------------------------
add_library(um790_pipeline STATIC)

target_sources(um790_pipeline PRIVATE
    ${UM790_PIPELINE_SOURCES}
    ${UM790_PIPELINE_HEADERS}
)

target_link_libraries(um790_pipeline
    PUBLIC
        um790_includes
        um790_compile_options
        $<$<TARGET_EXISTS:hip::host>:hip::host>
        $<$<BOOL:${HAILORT_FOUND}>:${HAILORT_LIBRARIES}>
        $<$<TARGET_EXISTS:rocm_smi64>:rocm_smi64>
        $<$<TARGET_EXISTS:CUDA::cudart>:CUDA::cudart>
        $<$<BOOL:${NVML_LIB}>:${NVML_LIB}>
    PRIVATE
        $<$<BOOL:${ONNXRUNTIME_LIB}>:${ONNXRUNTIME_LIB}>
        $<$<BOOL:${ORT_CUDA_PROVIDER_LIB}>:${ORT_CUDA_PROVIDER_LIB}>
        $<$<BOOL:${ORT_TENSORRT_PROVIDER_LIB}>:${ORT_TENSORRT_PROVIDER_LIB}>
        $<$<BOOL:${ORT_SHARED_PROVIDER_LIB}>:${ORT_SHARED_PROVIDER_LIB}>
        um790_link_options
        Threads::Threads
        # ClusterTransport requires Winsock on Windows
        $<$<PLATFORM_ID:Windows>:ws2_32>
        # Intel NPU telemetry requires PDH (Performance Data Helper) on Windows
        $<$<PLATFORM_ID:Windows>:pdh>
)

target_compile_definitions(um790_pipeline PUBLIC
    ${HAILORT_CFLAGS_OTHER}
    $<$<BOOL:${HIP_FOUND}>:UM790_HAS_HIP>
    $<$<BOOL:${CUDAToolkit_FOUND}>:UM790_HAS_CUDA>
    $<$<BOOL:${UM790_HAS_DIRECTML}>:UM790_HAS_DIRECTML>
    $<$<BOOL:${ONNXRUNTIME_LIB}>:CERBERUS_HAS_ONNXRUNTIME=1>
    $<$<BOOL:${OPENVINO_INCLUDE}>:CERBERUS_HAS_OPENVINO=1>
    $<$<BOOL:${CERBERUS_OPENVINO_FOUND}>:CERBERUS_HAS_OPENVINO=1>
    $<$<BOOL:${CERBERUS_OPENVINO_NOTFOUND}>:CERBERUS_HAS_OPENVINO=0>
    # Enable real PsiForceDB header compilation when headers are present
    $<$<BOOL:${PSIFORCEDB_HEADERS_FOUND}>:CERBERUS_USE_REAL_PSIFORCEDB_HEADERS=1>
)

set_target_properties(um790_pipeline PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER "${UM790_PIPELINE_HEADERS}"
)

# ---------------------------------------------------------------------------
# Target: um790_run (executable)
# ---------------------------------------------------------------------------
add_executable(um790_run src/main.cpp)
target_link_libraries(um790_run PRIVATE um790_pipeline)

# ---------------------------------------------------------------------------
# Target: um790_test (test executable)
# ---------------------------------------------------------------------------
add_executable(um790_test)
target_sources(um790_test PRIVATE
    tests/test_all.cpp
)
target_link_libraries(um790_test
    PRIVATE
        um790_pipeline
        GTest::gtest
        Threads::Threads
)

# Note: test_all.cpp provides its own main() — do NOT link gtest_main

# Register with CTest (deferred discovery so AVX-512 binaries don't crash
# during build on CI machines that lack the instruction set).
enable_testing()
include(GoogleTest)
gtest_discover_tests(um790_test DISCOVERY_MODE PRE_TEST)

# Target: test_smoke_ov
add_executable(test_smoke_ov src/test_smoke_ov.cpp)
target_link_libraries(test_smoke_ov PRIVATE um790_pipeline um790_compile_options um790_includes um790_link_options Threads::Threads)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND WIN32)
    target_link_options(test_smoke_ov PRIVATE -static-libgcc -static-libstdc++)
endif()

add_executable(david_propup_engine src/david_propup_engine_main.cpp)
target_link_libraries(david_propup_engine PRIVATE um790_pipeline cerberus_npu um790_compile_options um790_includes um790_link_options Threads::Threads)
target_compile_definitions(david_propup_engine PRIVATE CERBERUS_TESTING=1)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND WIN32)
    # NOTE: -static-libstdc++ is INTENTIONALLY OMITTED here.
    # Under MinGW UCRT, static libstdc++ causes deterministic segfault (exit 139)
    # during CRT TLS teardown when stdout is redirected to a pipe/file.
    # Dynamic libstdc++ (libstdc++-6.dll present in project root) avoids this.
    target_link_options(david_propup_engine PRIVATE -static-libgcc)
endif()

# Registration CLI — creates encrypted Local Maintenance DB
add_executable(cerberus_register src/cerberus_register_main.cpp)
target_link_libraries(cerberus_register PRIVATE um790_pipeline um790_compile_options um790_includes um790_link_options Threads::Threads)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND WIN32)
    target_link_options(cerberus_register PRIVATE -static-libgcc -static-libstdc++)
endif()

# Standalone LCMD end-to-end validation (no ONNX Runtime dependency)
add_executable(test_lcmd_e2e tests/test_lcmd_e2e.cpp)
target_link_libraries(test_lcmd_e2e PRIVATE um790_pipeline um790_compile_options um790_includes um790_link_options Threads::Threads)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND WIN32)
    target_link_options(test_lcmd_e2e PRIVATE -static-libgcc -static-libstdc++)
endif()

# Standalone fresh auto RBPC test (isolated propup for server LCMD)
add_executable(test_fresh_auto_rbpc tests/test_fresh_auto_rbpc.cpp)
target_link_libraries(test_fresh_auto_rbpc PRIVATE um790_pipeline cerberus_npu um790_compile_options um790_includes um790_link_options Threads::Threads)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND WIN32)
    target_link_options(test_fresh_auto_rbpc PRIVATE -static-libgcc -static-libstdc++)
endif()

# Build cerberus_npu as STATIC to avoid MinGW duplicate-symbol segfault.
# When SHARED, um790_pipeline (static) is linked into BOTH the DLL and
# the executable, causing ABI mismatch crashes (0xc0000139) on Windows.
add_library(cerberus_npu STATIC src/cerberus_api.cpp)

target_sources(cerberus_npu PRIVATE
    include/hq/cerberus_api.h
)

target_link_libraries(cerberus_npu PRIVATE um790_pipeline Threads::Threads)

set_target_properties(cerberus_npu PROPERTIES
    VERSION 1.0.0
    SOVERSION 1
    PUBLIC_HEADER "include/hq/cerberus_api.h"
)

# ---------------------------------------------------------------------------
# Target: cerberus_server (executable)
# ---------------------------------------------------------------------------
add_executable(cerberus_server
    src/inference_server.cpp
    src/cerberus_server_main.cpp
)

target_sources(cerberus_server PRIVATE
    include/hq/inference_server.hpp
)

target_link_libraries(cerberus_server PRIVATE
    um790_pipeline
    um790_compile_options
    $<$<PLATFORM_ID:Windows>:ws2_32>
    $<$<PLATFORM_ID:Windows>:pdh>
)

# ---------------------------------------------------------------------------
# Runtime DLL deployment (Windows) — copy ONNX Runtime DLLs next to exes
# MUST appear AFTER all target definitions (cerberus_server is last).
# ---------------------------------------------------------------------------
if(WIN32 AND ONNXRUNTIME_DLLS)
    set(_ORT_EXE_TARGETS
        um790_run
        um790_test
        test_smoke_ov
        david_propup_engine
        cerberus_register
        test_lcmd_e2e
        test_fresh_auto_rbpc
        cerberus_server
    )
    foreach(_ort_tgt IN LISTS _ORT_EXE_TARGETS)
        if(TARGET ${_ort_tgt})
            add_custom_command(TARGET ${_ort_tgt} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${ONNXRUNTIME_DLLS}
                    $<TARGET_FILE_DIR:${_ort_tgt}>
                COMMENT "Copying ONNX Runtime DLLs for ${_ort_tgt}"
            )
        endif()
    endforeach()
endif()

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------
include(GNUInstallDirs)

install(TARGETS um790_pipeline um790_run cerberus_npu cerberus_server um790_compile_options um790_includes um790_link_options
    EXPORT um790_pipeline-targets
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hq
)

install(EXPORT um790_pipeline-targets
    FILE um790_pipeline-targets.cmake
    NAMESPACE hq::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/um790_pipeline
)

# ---------------------------------------------------------------------------
# CPack Packaging
# ---------------------------------------------------------------------------
set(CPACK_PACKAGE_NAME "um790-pipeline")
set(CPACK_PACKAGE_VENDOR "HybridQuant")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY ${PROJECT_DESCRIPTION})
set(CPACK_PACKAGE_DESCRIPTION
    "High-performance AI image generation pipeline for MinisForum UM790 Pro. \
     Integrates ROCm/HIP GPU compute, HailoRT edge inference, and ONNX Runtime.")
set(CPACK_PACKAGE_CONTACT "https://github.com/hybridquant/um790-pipeline")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "rocm-hip, hailort, libonnxruntime")
set(CPACK_RPM_PACKAGE_REQUIRES "rocm-hip, hailort, onnxruntime")
set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
set(CPACK_RPM_PACKAGE_ARCHITECTURE "x86_64")
set(CPACK_GENERATOR "DEB;RPM;TGZ")

include(CPack)

# =============================================================================
# Status Summary
# =============================================================================
message(STATUS "")
message(STATUS "=== Cerberus Pipeline Configuration ===")
message(STATUS "  Project:    ${PROJECT_NAME} v${PROJECT_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Compiler:   ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  HIP:        ${HIP_FOUND} (${HIP_VERSION})")
message(STATUS "  CUDA:       ${CUDAToolkit_FOUND} (${CUDAToolkit_VERSION})")
message(STATUS "  NVML:       ${NVML_LIB}")
    message(STATUS "  DirectML:  ${UM790_HAS_DIRECTML}")
    message(STATUS "  HailoRT:    ${HAILORT_FOUND} (${HAILORT_VERSION})")
    message(STATUS "  ONNX RT:    ${ONNXRUNTIME_LIB}")
    message(STATUS "  OpenVINO:   ${OPENVINO_INCLUDE}")
    message(STATUS "  Install:    ${CMAKE_INSTALL_PREFIX}")
    message(STATUS "===================================")
    message(STATUS "")
message(STATUS "")
