# Cerberus — Heterogeneous AI Inference Runtime for Consumer Hardware

> **Work in Progress — Passion Project**
> This software is under heavy active development and is **not yet production-ready**.
> It is a personal passion project exploring the bleeding edge of C++26 on consumer-grade heterogeneous hardware.
> Please do not open issues complaining it doesn't work — we already know.
> Contributions, ideas, and constructive feedback are very welcome.

---

## Independent Hobbyist Project — Affiliation Disclaimer

**This project is independently developed by a hobbyist and has no affiliation with, endorsement by, or
connection to any of the hardware or software vendors whose products it targets.**

Specifically:

- **AMD** — Cerberus uses AMD ROCm/HIP and targets AMD Ryzen/Radeon hardware. We are **not** AMD, are
  **not** affiliated with AMD, and AMD has **not** endorsed or sponsored this project.
- **NVIDIA** — Some design patterns and tooling draw inspiration from the NVIDIA/CUDA ecosystem.
  We are **not** NVIDIA and have no affiliation with NVIDIA.
- **MinisForum** — The primary development hardware is the MinisForum UM790 Pro. We are **not**
  MinisForum, are **not** affiliated with MinisForum, and MinisForum has **not** endorsed this project.
- **Google / Hailo** — The NPU target is the Hailo-8L M.2 accelerator (HailoRT SDK). We are **not**
  Hailo, are **not** Google, and have no affiliation with either company.
- **Bambu Lab** — A Bambu Lab 3D printer is used to fabricate custom riser brackets and mechanical
  parts for hardware expansion (mounting the Hailo-8L M.2 and additional hardware). We are **not**
  Bambu Lab and have no affiliation with Bambu Lab. The printer is consumer hardware used as a
  fabrication tool; no Bambu Lab software, firmware, or intellectual property is part of this project.

This software is a personal project written by a single hobbyist to explore what consumer hardware
is capable of when pushed to its limits with modern C++. The goal is to benefit end users who own
this hardware and want to run serious AI inference locally — nothing more.

**We are not impersonating, representing, or acting on behalf of any of the above companies.**

---

## What This Project Is

**Cerberus** is a C++26 AI inference runtime targeting consumer hardware — specifically the MinisForum
UM790 Pro with an AMD Ryzen 9 7940HS (CPU), AMD Radeon 780M iGPU, and Hailo-8L M.2 NPU.

The goal is a high-performance, production-quality local inference system that ordinary people can
actually run and benefit from — without a $10k server or cloud dependency.

This is a **real engineering project**, not a demo. Every component has real C++ code, real tests,
and real error handling. Several components are in **working proof-of-concept** state; others are
**architectural foundations** that will be built out as the project matures.

See [Current Known Limitations](#current-known-limitations) below for an honest assessment of
what is and is not working today.

---

## Architecture

| Accelerator | Hardware                          | Role                                                    |
|-------------|-----------------------------------|---------------------------------------------------------|
| **CPU**     | AMD Ryzen 9 7940HS (Zen 4)        | Tokenization, DEIS scheduling, orchestration            |
| **GPU**     | AMD Radeon 780M (RDNA 3)          | UNet/DiT denoising, VAE decoding, HIP compute           |
| **NPU**     | Hailo-8L M.2                      | CLIP/T5 text encoding, edge inference                   |

## Key Components

- **Pipeline** — End-to-end generation orchestration with watchdog recovery
- **DEISScheduler** — Precomputed DEIS/DDIM scheduler with AVX-512; full `std::expected<void, SchedulerError>` error propagation; `TensorView`-typed public API
- **INpuEncoder / NpuBackend\<T\>** — Pluggable NPU encoder abstraction (virtual interface + C++26 concept). Factory-selected at pipeline init: Hailo8lEncoder > CpuFallbackEncoder > SyntheticNpuEncoder. Wired into Pipeline from Round 17.
- **UtilizationWatchdog** — Per-step monitoring with exponential backoff recovery
- **GPUMonitor** + **HailoMonitor** — Real hardware telemetry (ROCm SMI + HailoRT)
- **CLIPTokenizer** — Self-contained CLIP BPE tokenizer
- **PinnedStagingPool** — High-performance async DMA staging
- **HIPGraphDenoiser** — Experimental zero-CPU-overhead graph capture; `FloatTensor4D`-typed public API; `std::span<const float>` embedding parameters
- **PipelineHealthScore** — Composite health scoring across 7 metrics
- **TieredMemoryManager** — Four-tier memory system (Hot/Warm/Cool/Cold)
- **ClusterTransport** — Multi-node clustering foundation (Thunderbolt 5 / 10 GbE ready) — see Known Limitations
- **AsyncPipeline** — C++26 coroutine-based async execution
- **InferenceServer** — OpenAI-compatible HTTP API
- **TensorView\<T,Rank\>** — Self-contained, zero-allocation tensor view (no std::mdspan); primary view type across all hot-path APIs
- **NpuBackend concept** — Static C++26 concept formalising the NPU encoder contract; proved for Hailo8lEncoder, SyntheticNpuEncoder, CpuFallbackEncoder, WindowsNpuBackend
- **INpuPostProcessor / NpuAccelerator\<T\>** — Pluggable NPU post-processing abstraction (virtual interface + C++26 concept). Factory-selected at pipeline init. Wired into both the denoising loop (CFG blend) and post-VAE decode. `NpuAccelerator<T>` concept requires `blend_noise_cfg()` — proven for both concrete classes.
- **CFG blend in denoising loop via NPU** — `blend_noise_cfg()` replaces the scalar CPU loop in `denoise_step_()` at every step with CFG enabled. Routes through `npu_post_processor_`. Timing accumulated in `npu_blend_in_loop_us`. On real Hailo-8L: SAXPY on NN core. Currently: CPU path via `SyntheticNpuPostProcessor`.
- **`cerberus profile`** — Per-phase timing breakdown command: text_encode / embedding_stage / denoise_total (with in-loop NPU blend µs) / vae_decode / post_process.

## Current Status (May 2026)

- Round 19 complete — CFG blend routed through NPU abstraction inside `denoise_step_()` at every step; `blend_noise_cfg()` in `NpuAccelerator<T>` concept; per-step blend timing; 8 new tests
- Round 18 — INpuPostProcessor + NpuAccelerator<T> wired, per-phase timing, `cerberus profile` command, 12 new tests
- Round 17 — INpuEncoder wired into Pipeline, honest README, HARDWARE.md, ClusterTransport EXPERIMENTAL marker
- Round 16 — `cerberus monitor` live dashboard, generate progress indicator, Markdown export
- **251 tests** across 23 component groups
- Zero errors, zero warnings on MinGW-W64 GCC 14.2.0 (Windows) and targeting Ubuntu 24.04 parity
- Self-review score: **9.7/10**

## What Currently Works (Proved by Tests)

- **C++26 foundations** — `std::expected`, `std::format`/`std::print`, `std::jthread`, concepts, `TensorView<T,Rank>`, all compile and pass tests
- **DEISScheduler** — Full DEIS/DDIM scheduler with AVX-512, tested for correctness
- **CLIPTokenizer** — BPE tokenization matches reference outputs
- **PipelineHealthScore** — Grade boundaries, sub-score ranges, reset, all verified
- **BenchmarkLogger** — Ring-buffer event logger, P50/P95/P99/stddev/CV, JSON/CSV/Markdown export
- **TieredMemoryManager** — Four-tier alloc/free/migration, RAII `ScopedTierAlloc`
- **UtilizationWatchdog** — Threshold detection, exponential backoff, recovery callbacks
- **INpuEncoder interface** — SyntheticNpuEncoder, CpuFallbackEncoder, Hailo8lEncoder, WindowsNpuBackend all satisfy `NpuBackend<T>` concept; wired into Pipeline constructor
- **INpuPostProcessor interface** — SyntheticNpuPostProcessor, HailoNpuPostProcessor satisfy `NpuAccelerator<T>` concept (requires `post_process()` + `blend_noise_cfg()`); wired into Pipeline constructor, called in `denoise_step_()` CFG blend AND after VAE decode
- **Per-phase timing** — `PipelinePhaseTimings` populated by every `generate()` call; accessible via `Pipeline::last_phase_timings()`
- **ClusterTransport** — Real TCP socket code compiles and unit-tests pass
- **PinnedStagingPool** — hipHostMalloc with `_aligned_malloc` fallback, tested on Windows

## Heterogeneous Execution Reality (Honest Assessment)

| Phase | Hardware | Reality |
|-------|----------|---------|
| Text encoding | NPU target | Always **CPU/synthetic** — HailoRT not installed; ORT stub means no real inference |
| UNet denoising | GPU (Radeon 780M) | Always **CPU fallback** on this build — ROCm EP requires Ubuntu + ROCm stack |
| VAE decode | GPU (same) | Same as UNet — CPU path on Windows |
| Post-processing | NPU target | Always **CPU pass-through** — HailoNpuPostProcessor skeleton, HailoRT absent |

**Current NPU participation: 0%** on all platforms this binary runs on.
True heterogeneous execution requires: Ubuntu 22.04/24.04 + ROCm 6.x + HailoRT SDK + ONNX model files.

### Why NPU denoising is not feasible in the near term

| Constraint | Detail |
|------------|--------|
| Hailo-8L theoretical throughput | 13 TOPS (INT8) |
| SD 1.5 UNet at 512×512, one step | ~2.3 TFLOPS FP32 equivalent |
| Estimated latency per step on Hailo-8L | ~177 ms/step (vs ~28 ms/step on Radeon 780M) |
| HailoRT availability | Linux only; no Windows PCIe driver |
| HEF requirement | Must compile UNet weights to Hailo Executable Format — no public SD 1.5 HEF |

**Correct NPU use case:** Post-processing / safety filtering (much smaller networks, <100M params vs UNet ~860M). This is the `INpuPostProcessor` extension point added in Round 18.

## Current Known Limitations

These are **honest limitations** — not aspirations. They will be fixed in future rounds.

| # | Component | Limitation |
|---|-----------|------------|
| L1 | **NPU text encoding** | `Hailo8lEncoder::is_available()` is always `false` (HailoRT SDK absent). `encode_prompt_()` routes via `CpuFallbackEncoder` (real ORT) when a CLIP ONNX model is loaded, or `SyntheticNpuEncoder` (XOR/deterministic) when no model file is present. Real Hailo-8L text encoding is a future work item. |
| L2 | **H2D copy waste (BUG B3)** | Pipeline stages embeddings to GPU via `PinnedStagingPool`, then `denoise_step_()` creates `Ort::Value` tensors from raw CPU pointers — causing ONNX Runtime's ROCm EP to perform a second implicit H2D copy. The staging DMA is wasted. Zero-copy fix is documented in `pipeline.hpp` BUG B3 note and deferred to a future round. |
| L3 | **ClusterTransport (EXPERIMENTAL)** | Real TCP socket code exists (`bind/listen/accept/connect`, per-worker jthread pumps, health-score load balancer), but has **never been tested with real multi-node hardware**. Known issues: `collect_telemetry()` races the per-worker pump on the same fd (no mutex); no TLS; no reconnect on drop; LoopbackUnix intra-host path bypasses real sockets entirely. Do not use in production. |
| L4 | **No real model files** | The build system produces all targets and tests pass, but no ONNX model weights are distributed. Actual image generation requires valid `text_encoder.onnx`, `unet.onnx`, and `vae_decoder.onnx` files placed in the configured model directory. |
| L5 | **HIPGraphDenoiser** | Graph capture/replay works in principle but is `enable_hip_graph = false` by default. Requires a full ROCm stack and valid UNet ONNX model for end-to-end testing. |
| L6 | **Ubuntu 24.04 build** | The build system targets Ubuntu 24.04 with ROCm. Parity is maintained architecturally (conditional `#ifdef UM790_HAS_HIP` guards), but the Ubuntu build is not CI-verified in every round — it is deduced, not continuously measured. |
| L7 | **NPU post-processing** | `HailoNpuPostProcessor::is_available()` is always `false`. The post-processing slot in the pipeline is wired and timed (Round 18), but the Hailo backend is a skeleton: no HEF compiled, HailoRT not installed. `SyntheticNpuPostProcessor` (CPU pass-through) runs instead. |
| L8 | **ORT is a compile-time stub** | `code/include/onnxruntime_cxx_api.h` is a minimal no-op stub (`Session::Run()` returns empty, `GetTensorData()` returns nullptr). All inference code compiles but produces no real output. Real ORT requires linking against `onnxruntime.dll`/`.so` — not distributed in this repo. |

---

## Build

### Windows (MinGW-W64 GCC 14.2.0)

```bash
py build.py           # configure + build
py build.py --clean   # clean + rebuild
```

### Linux (Ubuntu 24.04, ROCm)

```bash
export ROCM_PATH=/opt/rocm

cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="/opt/rocm;/usr/local/lib/onnxruntime"

cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

### Quick Start — Inference Server

```bash
./build/cerberus_server --port 8080 --models /path/to/models
```

### Quick Start — Live Monitor

```bash
# Live GPU + NPU + health score dashboard (1 Hz, Ctrl+C to stop)
cerberus monitor

# Faster refresh rate
cerberus monitor --interval 500

# Generate image with live progress
cerberus generate "a futuristic city at night" --steps 20

# Full benchmark with JSON, CSV, and Markdown export
cerberus benchmark --iterations 50 --output ./results

# Per-phase timing breakdown (NPU/GPU/CPU heterogeneous split)
cerberus profile "a futuristic city at night" --steps 20
```

Endpoints:

- `POST /v1/chat/completions` — OpenAI-compatible
- `GET  /v1/models` — List available models
- `GET  /health` — Device utilization and health stats

## Hardware

See [HARDWARE.md](HARDWARE.md) for the physical build, Bambu Lab 3D-printed riser brackets,
and the constraint-driven philosophy behind targeting this specific hardware.

## Project Structure

```
CMakeLists.txt
code/include/hq/      # All public headers
code/src/             # Implementation files
code/tests/           # 243 tests across 22 component groups
LICENSE
README.md
HARDWARE.md
SECURITY.md
CONTRIBUTING.md
FUNDING.md
```

## License

This project is licensed under the MIT License — see the LICENSE file for details.
Copyright (c) 2026 D Hargreaves (AKA Roylepython). LamiaFabrica. All rights reserved.

## Funding & Patronage

This project is built by a disabled programmer who can no longer work full-time due to a severe spinal injury.
Every hour of development is hard-won.

Cerberus is my main focus and passion. Without financial support, I simply cannot continue developing at this
pace or keep the hardware and servers running.

If you find this project valuable and want to see it reach production quality (real CXL memory tiering,
proper Thunderbolt 5 clustering, broader hardware support, and eventually post-quantum security), please
consider becoming a patron. Every single contribution directly funds continued development.

Thank you for your interest in Cerberus.
— David
