# Cerberus — Heterogeneous AI Inference Runtime for Consumer Hardware

> **⚠️ Work in Progress — Passion Project.** This software is under heavy active
> development and is not yet production-ready. It is a personal passion project
> exploring the bleeding edge of C++26 on consumer-grade heterogeneous hardware.
> **Please do not open issues complaining it doesn't work — we already know.**
> Contributions, ideas, and constructive feedback are welcome.

AI inference runtime spanning CPU, GPU, and NPU accelerators on the MinisForum
UM790 Pro (Zen 4, AVX-512, ROCm 6.0+, HailoRT 4.20+).

## Architecture

| Accelerator  | Hardware                  | Role                                     |
|-------------|---------------------------|------------------------------------------|
| CPU         | AMD Ryzen 9 7940HS (Zen 4) | Tokenization, DEIS scheduling, orchestration |
| GPU         | AMD Radeon 780M (RDNA 3)   | UNet denoising, VAE decoding, HIP compute |
| NPU         | Hailo-8L M.2               | CLIP text encoding, edge inference         |

## Component Inventory

| Component               | Header                             | Purpose                                                       |
|-------------------------|------------------------------------|---------------------------------------------------------------|
| Pipeline                | `include/hq/pipeline.hpp`          | End-to-end generation orchestration: encode → denoise → decode |
| DEISScheduler           | `include/hq/deis_scheduler.hpp`    | Precomputed DEIS/DDIM/Euler diffusion scheduler with AVX-512   |
| UtilizationWatchdog     | `include/hq/utilization_watchdog.hpp` | 3-state per-step HW monitor with exponential-backoff recovery |
| GPUMonitor              | `include/hq/gpu_monitor.hpp`       | ROCm SMI GPU telemetry (utilization, temperature, power)       |
| HailoMonitor            | `include/hq/hailo_monitor.hpp`     | Dual-indicator Hailo-8L monitoring (power + inference fusion)  |
| CLIPTokenizer           | `include/hq/clip_tokenizer.hpp`    | CLIP text tokenizer with byte-pair encoding                    |
| PinnedStagingPool       | `include/hq/pinned_staging.hpp`    | Double-buffered pinned-host staging for async CPU-to-GPU DMA    |
| EmbeddingStagingManager | `include/hq/staging_manager.hpp`   | Host-side pinned buffer pool for embedding transport            |
| HIPGraphDenoiser        | `include/hq/hip_graph_denoiser.hpp` | HIP-graph-capture denoiser with CPU fallback (experimental)    |
| PipelineHealthScore     | `include/hq/health_score.hpp`      | Composite health scoring from 7 weighted sub-metrics            |
| TensorView              | `include/hq/tensor_view.hpp`       | Zero-overhead tensor views via `std::mdspan` + ONNX interop     |
| NpuDmaPipeline          | `include/hq/npu_pipeline.hpp`      | NPU↔GPU DMA pipeline: pinned tensors, double-buffered encoding  |
| NpuEncoder              | `include/hq/npu_encoder.hpp`       | Pluggable encoder family — `INpuEncoder`, `SyntheticNpuEncoder`, `Hailo8lEncoder`, `CpuFallbackEncoder` |
| AsyncPipeline           | `include/hq/async_pipeline.hpp`    | C++26 coroutine pipeline — `task<T>`, `Generator<T>`, hip/DMA awaiters |
| InferenceServer         | `include/hq/inference_server.hpp`  | HTTP inference server with OpenAI-compatible `/v1/chat/completions` |
| Cerberus C API          | `include/hq/cerberus_api.h`        | Pure C FFI for `libcerberus_npu.so` — device discovery, inference, telemetry |

## Build Targets

| Target             | Type       | Description                                  |
|--------------------|------------|----------------------------------------------|
| `um790_pipeline`   | Static lib | Core pipeline library                        |
| `um790_run`        | Executable | Standalone generation runner                 |
| `um790_test`       | Executable | Test harness (CTest-discovered)              |
| `cerberus_npu`     | Shared lib | `libcerberus_npu.so` — C-ABI inference engine |
| `cerberus_server`  | Executable | Inference server with OpenAI-compatible API  |

## C++26 Features

- **`std::expected`** — error handling across all component contracts
- **`std::print`** — type-safe formatted output
- **`std::mdspan`** — zero-abstraction tensor views with compile-time layouts
- **`std::format`** — compile-time-checked string formatting
- **Coroutines** (`co_await`/`co_return`) — async pipeline with GPU event awaiters

## Dependencies

| Library       | Min Version | Purpose                          |
|---------------|-------------|----------------------------------|
| ROCm/HIP      | 6.0         | GPU compute                      |
| HailoRT       | 4.20        | NPU edge inference               |
| ONNX Runtime  | 1.17+       | Model inference, multi-EP        |
| GoogleTest    | 1.14        | Testing (FetchContent)           |
| CMake         | 3.28        | Build system                     |
| GCC/Clang     | 14+/18+     | C++26 compiler                   |

## Build

```bash
export ROCM_PATH=/opt/rocm

cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="/opt/rocm;/usr/local/lib/onnxruntime"

cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

## Test Coverage

~99 tests across 12 component test groups: UtilizationWatchdog (18), TensorView (14),
DEISScheduler (12), NpuDmaPipeline (9), Integration (8), HailoMonitor (7),
CLIPTokenizer (7), SyntheticNpuEncoder (6), PinnedStagingPool (5), GPUMonitor (4),
PipelineHealthScore (2), and encoding/misc (7).

Tests compile and run without HIP, HailoRT, or ROCm SMI (stub modes active).

## Quick Start — Inference Server

```bash
./build/cerberus_server --port 8080 --models /path/to/models
```

Endpoints:

| Method | Path                    | Description                |
|--------|-------------------------|----------------------------|
| POST   | `/v1/chat/completions`  | OpenAI-compatible chat     |
| GET    | `/v1/models`            | List available models      |
| GET    | `/health`               | Device utilization stats   |

## Project Structure

```
.
├── CMakeLists.txt
├── cmake/
│   ├── CheckCXX26Features.cmake
│   └── cxx26_features.hpp.in
├── include/hq/
│   ├── async_pipeline.hpp
│   ├── cerberus_api.h
│   ├── clip_tokenizer.hpp
│   ├── deis_scheduler.hpp
│   ├── gpu_monitor.hpp
│   ├── hailo_monitor.hpp
│   ├── health_score.hpp
│   ├── hip_graph_denoiser.hpp
│   ├── inference_server.hpp
│   ├── npu_encoder.hpp
│   ├── npu_pipeline.hpp
│   ├── pinned_staging.hpp
│   ├── pipeline.hpp
│   ├── staging_manager.hpp
│   ├── tensor_view.hpp
│   └── utilization_watchdog.hpp
├── src/
│   ├── async_pipeline.cpp
│   ├── cerberus_api.cpp
│   ├── cerberus_server_main.cpp
│   ├── clip_tokenizer.cpp
│   ├── deis_scheduler.cpp
│   ├── gpu_monitor.cpp
│   ├── hailo_monitor.cpp
│   ├── health_score.cpp
│   ├── hip_graph_denoiser.cpp
│   ├── inference_server.cpp
│   ├── main.cpp
│   ├── npu_encoder.cpp
│   ├── npu_pipeline.cpp
│   ├── pinned_staging.cpp
│   ├── pipeline_integration.cpp
│   ├── staging_manager.cpp
│   └── utilization_watchdog.cpp
└── tests/
    └── test_all.cpp
```

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.
