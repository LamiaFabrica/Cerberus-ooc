# Cerberus — Heterogeneous AI Inference Runtime

> **WORK IN PROGRESS — Not Functional Yet**
>
> This project is a **work-in-progress passion project** exploring the design space for
> high-performance AI inference on heterogeneous consumer hardware (CPU + GPU + NPU).
>
> **It does not currently work.** Many components are stubbed, forward-declared, or
> implemented against interfaces that lack real GPU/NPU hardware on the current build host.
>
> If you are looking for a working inference runtime today, please use
> [ONNX Runtime](https://github.com/microsoft/onnxruntime),
> [Llama.cpp](https://github.com/ggerganov/llama.cpp), or
> [ComfyUI](https://github.com/comfyanonymous/ComfyUI) instead.
>
> Pull requests, design ideas, and constructive feedback are very welcome.

---

## License

**MIT License** — see [LICENSE](LICENSE)  
Copyright (c) 2026 D Hargreaves (AKA Roylepython)  
All rights reserved.

---

## What is Cerberus?

Cerberus is a **C++26 AI inference runtime** targeting heterogeneous consumer hardware across **Windows (MinGW-W64) and Ubuntu (24.04 LTS)**. The eventual deployment platforms are Windows and Linux — the two most approachable operating systems for developers and end users.

We are currently adapting the **ROCm/HIP execution provider** for AMD GPU compute on Windows (via the DirectML fallback path) alongside the existing ROCm/HIP paths on Ubuntu. For the Radeon 780M iGPU specifically, ONNX Runtime will use the CPU execution provider or DirectML depending on driver availability.

Target hardware:

- **CPU** — AMD Ryzen 9 7940HS (Zen 4)
- **GPU** — AMD Radeon 780M (RDNA 3)
- **NPU** — Hailo-8L M.2 accelerator

The architecture is designed around:
- **Tiered memory management** — Hot (GPU VRAM), Warm (CXL/RAM), Cool (CPU RAM), Cold (NVMe)
- **Honest hardware probing** — Every backend reports `unavailable_reason()` when hardware is missing
- **Zero false claims** — No empty feature stubs; if a component requires unavailable hardware, it honestly says so
- **Pipeline health scoring** — Per-step watchdog, recovery, and composite health metrics
- **End-to-end security** — LFSSL bridge with Kyber + Argon2id + AES-256-GCM (PQC-ready)

---

## Current Status: Honestly Not Working

### What Works (CPU-only on this build host)

| Component        | Status | Notes |
|------------------|--------|-------|
| Core pipeline architecture | Working | `Pipeline::generate()` orchestrates encode → denoise → decode |
| CLIP text tokenization | Working | Self-contained BPE tokenizer, no external model files |
| DEIS/DDIM scheduler | Working | Precomputed coefficients, AVX-512 dispatch (hardware-gated) |
| TieredMemoryManager | Working | Cool tier (CPU RAM) fully operational; Hot/Warm/Cold are stubs |
| Security (LFSSL/JWT/RBPC) | Working | Full cryptographic suite compiled and tested |
| Watchdog + health scoring | Working | Per-step inline monitoring with exponential backoff |
| Async pipeline (coroutines) | Working | `co_await`/`co_return` pipeline with proper cancellation |
| Cluster dispatch (serialization) | Working | Request serialization, worker selection, timeout logic |
| 347/347 E2E tests | Passing | All propup tests pass (~1771 ms, zero warnings) |
| BUG B3 embedding staging (Round 20) | Improved | Unnecessary host memcpy for embeddings before ONNX input eliminated (direct vector data to Ort::Value); TMM only for latents. Zero/min-copy path for CPU→ORT (ORT still does H2D if GPU EP). |
| Round 20: NpuAccelerator / INpuPostProcessor SafetyFilter extension | Complete | Added NpuSafetyFilterRequest/Result + safety_filter() virtual to the abstraction (NpuTaskType::SafetyFilter). Honest CPU luminance/variance heuristic (always functional, synthetic_mode=true, was_npu=false). Wired post-VAE in Pipeline generate with dedicated PipelinePhaseTimings.safety_filter_ms + HardwareAccelerationReport safety_* fields (now 4 cheap components in npu_cheap_ops_percent). +12 evidence tests in dedicated Round20EvidenceTest Section 23 (honesty, heuristic, virtual dispatch, concept, error paths, timing, factory). Multiple pre-existing -Werror debt items fixed during iteration for hygiene. Full clean test run blocked on this host by remaining legacy surface (see Known Limitations). |

### What Does NOT Work (Hardware Blocked)

| Component | Blocker | Honest Reason |
|-----------|---------|---------------|
| GPU inference (UNet/VAE) | ROCm not available on Windows build host | Requires Ubuntu + ROCm 6.x stack |
| NPU text encoding | HailoRT not installed | HailoRT SDK is Linux-only; no Windows PCIe driver |
| NPU post-processing | No compiled HEF models | Hailo-8L requires HEF; none exist for SD 1.5 UNet at 512x512 |
| GPU zero-copy staging | Same as GPU inference | HIP pinned memory requires `hipHostMalloc` |
| CXL memory tier | No CXL hardware | `detect_cxl()` returns false; falls back to `aligned_alloc` |
| AVX-512 kernels | Host lacks AVX-512F | Dispatch exists but routes to AVX2/ scalar |

### The Core Truth

> **The expensive 90% of inference (UNet denoising) currently runs on CPU via ONNX Runtime CPU execution provider.**
>
> The GPU/NPU code paths are all designed, typed, and unit-tested, but the actual hardware-accelerated execution requires:
> 1. Ubuntu 22.04/24.04 with ROCm 6.x
> 2. HailoRT SDK 4.20+ installed
> 3. Compiled ONNX model files (text_encoder, unet, vae_decoder)
>
> None of these are present on the current Windows/MinGW build host. The code honestly reports this at runtime.

---

## Architecture

```
Prompt ──► Pipeline::generate()
            │
            ├── encode_prompt_() ──► CLIP tokenizer ──► ORT text encoder (CPU EP)
            │
            ├── denoise_step_() ──► UNet ──► ORT GPU/CPU EP (currently CPU)
            │       └── CFG blend via NPU abstraction (currently CPU scalar loop)
            │
            └── decode_latents_() ──► VAE decoder ──► ORT (currently CPU EP)
                        │
                        ▼
               Post-processing via NPU abstraction (currently CPU pass-through)
                        │
                        ▼
            GeneratedImage (RGBA)
```

### Memory Flow per Generation

1. **Embeddings** — TMM Cool tier (`aligned_alloc`); copied from ORT output buffer
2. **Latents** — TMM Cool tier; random noise, updated in-place by scheduler
3. **Checkpoints** — TMM Cool tier; latent save/restore for watchdog recovery
4. **VAE output** — `std::vector<uint8_t>` on heap; moved out as `GeneratedImage`

---

## Build

### Prerequisites

- CMake 3.28+
- GCC 15+ or Clang 18+ (C++26 required: `std::expected`, `std::format`, designated initializers)
- ONNX Runtime 1.17+ (headers + runtime library)

### Windows (MinGW-W64)

```powershell
cd code/
cmake -B build -DCMAKE_BUILD_TYPE=Release -G "MinGW Makefiles"
cmake --build build --target david_propup_engine
.\build\david_propup_engine.exe
```

Expected output:
```
TOTAL: 347/347 passed in ~1771 ms
STATUS: ALL CLEAR
```

### Linux (Ubuntu 24.04, ROCm)

```bash
export ROCM_PATH=/opt/rocm
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="/opt/rocm;/usr/local/lib/onnxruntime"
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

---

## Project Structure

```
Cerberus/
├── code/                  # All source code
│   ├── include/hq/        # Public headers (*.hpp)
│   ├── src/               # Implementation (*.cpp)
│   ├── tests/             # Test harness
│   ├── CMakeLists.txt     # Build configuration
│   └── build/             # Build outputs (git-ignored)
├── lfssl_bridge/          # LFSSL DLL build (PQC + AES + Argon2id)
├── ort/                   # ONNX Runtime libraries
├── Development docs/      # Internal docs, logs, research (git-ignored)
├── LICENSE                # MIT License
└── README.md              # You are here
```

All internal documentation, build logs, scratch files, and research reports live in
`Development docs/` which is **git-ignored** and not part of the public repository.

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

- **Bug reports** — Open a GitHub Issue with OS, hardware, build environment, and reproduction steps
- **Feature ideas** — Open a GitHub Discussion before opening a PR
- **Security vulnerabilities** — Follow the process in `CONTRIBUTING.md`

---

## Honest Disclaimer

**I am not affiliated with any of the following companies:**

- AMD / ROCm
- NVIDIA / CUDA
- Hailo / Google
- MinisForum
- Bambu Lab
- Microsoft / ONNX Runtime

This is an **independent engineering project** run as a professional commitment to build open infrastructure for local AI inference. It is not a side project. It is written and maintained by a single developer (me, David) committed to pushing consumer hardware to its limits with modern C++.

**I am not impersonating, representing, or acting on behalf of any company.**

The project name "Cerberus" and all original code are my own work. Third-party libraries
(ONNX Runtime, LFSSL, etc.) are used under their respective open-source licenses, which are
attributed in the source.

---

## About the Author

I am a **systems analyst, network engineer, programmer, and web developer** committed to building open infrastructure for local AI inference. This is not a spare-time project. It is a strategic commitment of all available resources toward building an A-Team of neurodiverse developers plus AI, to overcome physical and mental limitations and produce production-quality results.

I do not leave my house. I do not have a social network. The workday is solitary. It is a deliberate choice to redirect all available capacity into this infrastructure. Every hour of development is hard-won. If you find this project valuable and want to see it reach production quality — real CXL memory tiering, proper Thunderbolt 5 clustering, broader hardware support, and post-quantum security — support is welcome.

Every contribution, code or financial, directly funds continued development and keeps the hardware running.

- **Code & design help**: Open a PR or issue — expertise is worth more than money
- **Patreon**: [https://www.patreon.com/TheMedusaInitiative](https://www.patreon.com/TheMedusaInitiative) — £25/month removes ads from all software at 200 subscribers

— David Hargreaves (Roylepython), May 2026
