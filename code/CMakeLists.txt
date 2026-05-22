# =============================================================================
# Top-level CMake for Cerberus Heterogeneous AI Inference Runtime
# Dev target:  ROG Strix G18 (Intel Ultra 9 275HX + RTX 5070 Ti + Intel AI Boost NPU)
# Prod target: MinisForum UM790 Pro (Zen 4, AVX-512, ROCm 6.0+, HailoRT 4.20+)
# -march=native auto-detects the host CPU on every build machine.
# =============================================================================
cmake_minimum_required(VERSION 3.28)

project(um790_pipeline
    VERSION 1.0.0
    DESCRIPTION "AI Image Generation Pipeline for UM790 Pro"
    HOMEPAGE_URL "https://github.com/hybridquant/um790-pipeline"
    LANGUAGES CXX
)

# ---------------------------------------------------------------------------
# C++26 Standard Configuration
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Feature Detection: Verify the compiler actually supports requested C++26
# ---------------------------------------------------------------------------
include(cmake/CheckCXX26Features.cmake)

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
# Use the local ORT SDK (GPU package with CUDA/DML EP support)
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../ort" CACHE PATH "ONNX Runtime SDK root directory")
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

# ---------------------------------------------------------------------------
# Compiler Flags (target-based, no globals)
# ---------------------------------------------------------------------------
# Base flags applied to all targets via interface library
# C++26 coroutine support requires GCC 14+ or Clang 18+
add_library(um790_compile_options INTERFACE)
target_compile_options(um790_compile_options INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
        -march=native        # Detect host CPU features (Zen 4 on UM790, Arrow Lake on G18 dev)
        -Wall -Wextra -Wpedantic -Werror

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
target_compile_features(um790_compile_options INTERFACE cxx_std_26)

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
    $<INSTALL_INTERFACE:include>
)
# ORT headers are treated as SYSTEM includes to suppress their warnings
target_include_directories(um790_includes SYSTEM INTERFACE
    $<$<BOOL:${HAILORT_FOUND}>:${HAILORT_INCLUDE_DIRS}>
    $<$<BOOL:${ONNXRUNTIME_INCLUDE}>:${ONNXRUNTIME_INCLUDE}>
    $<$<BOOL:${NVML_INCLUDE_DIR}>:${NVML_INCLUDE_DIR}>
)

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
    src/async_pipeline.cpp
    src/tiered_memory_manager.cpp
    src/cluster_transport.cpp
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
    include/hq/npu_backend.hpp
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
        um790_link_options
        Threads::Threads
        # ClusterTransport requires Winsock on Windows
        $<$<PLATFORM_ID:Windows>:ws2_32>
)

target_compile_definitions(um790_pipeline PUBLIC
    ${HAILORT_CFLAGS_OTHER}
    $<$<BOOL:${HIP_FOUND}>:UM790_HAS_HIP>
    $<$<BOOL:${CUDAToolkit_FOUND}>:UM790_HAS_CUDA>
    $<$<BOOL:${UM790_HAS_DIRECTML}>:UM790_HAS_DIRECTML>
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

# ---------------------------------------------------------------------------
# Target: cerberus_npu (shared library)
# ---------------------------------------------------------------------------
add_library(cerberus_npu SHARED src/cerberus_api.cpp)

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
)

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
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
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
message(STATUS "  Install:    ${CMAKE_INSTALL_PREFIX}")
message(STATUS "===================================")
message(STATUS "")
message(STATUS "")
