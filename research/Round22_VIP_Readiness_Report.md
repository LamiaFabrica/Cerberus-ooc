# Round 22 — VIP Readiness Report

**Date:** 2026-05-22  
**Author:** LamiaFabrica / opencode assessment  
**Purpose:** Honest evaluation of whether Cerberus is ready to show to respected technical people for feedback.

---

## 1. Stage 1 — Brutal Gap Analysis

### 1.1 Real NPU Participation During `generate()`

During a normal `Pipeline::generate()` call, **zero percent of the critical path uses real NPU hardware**:

| Stage | Function | Hardware Used | Evidence |
|-------|----------|---------------|----------|
| Text encode | `CpuFallbackEncoder::encode()` or `SyntheticNpuEncoder::encode()` | CPU (ORT) or XOR synthesis | `NpuEncoderFactory` always selects CpuFallback or Synthetic because `Hailo8lEncoder::is_available()` is hardcoded `false` (`npu_encoder.cpp:54`) |
| CFG blend | `SyntheticNpuPostProcessor::blend_noise_cfg()` | CPU scalar loop | `npu_accelerator.cpp:74-76` — inline SAXPY on CPU |
| Post-process | `SyntheticNpuPostProcessor::post_process()` | CPU memcpy | `npu_accelerator.cpp:41-42` — `memcpy`, `was_npu_accelerated = false` |
| Hailo telemetry | `HailoMonitor::sample()` | **Never called** — monitor was never opened | `pipeline_integration.cpp:196` — `HailoMonitor` constructed but `open()` never called. With fix: synthetic time-varying data (`hailo_monitor.cpp:426-490`) |
| GPU telemetry | `GPUMonitor::query_all()` | REAL if NVML available | `gpu_monitor.cpp:209-236` — actual `nvmlDeviceGetUtilizationRates()` etc. |
| UNet denoise | `denoise_step_()` → `ort.gpu_session->Run()` | REAL ORT inference (CPU only due to CUDA DLL mismatch) | `pipeline_integration.cpp:95-153` — CUDA EP registration fails at runtime; falls back to CPU |
| VAE decode | `decode_latents_()` → `ort.vae_session->Run()` | REAL ORT inference (CPU only, same reason) | Same DLL mismatch |
| DMA staging | `PinnedStagingPool::stage_to_gpu()` or `EmbeddingStagingManager` | REAL pinned memory allocation but **wasted H2D copy** (`pipeline_integration.cpp:585-593`) | BUG B3: staged data never consumed by `denoise_step_()` |

**Bottom line: The only real hardware acceleration currently working is NVML GPU telemetry queries. All inference runs on CPU. All NPU participation is simulated.**

### 1.2 Top 3 Issues That Would Make a Serious Reviewer Question the Project

1. **Synthetic data masquerading as real telemetry**: `HailoMonitor` returns time-varying sinusoidal data (`55 + 3*sin(t)` for temperature, phased power cycling, synthetic inference counts) that looks realistic but is completely fabricated. Until this round, there was no `synthetic_mode()` flag and no runtime warnings on individual reads. A reviewer running the pipeline on the bench would see plausible-looking Hailo utilization graphs that are pure fiction.

2. **`is_available() = true` on a component that does nothing**: `SyntheticNpuPostProcessor::is_available()` returned `true` despite being a memcpy passthrough with `was_npu_accelerated = false`. This made factory selection silently choose a no-op component. The name "NpuAccelerator" concept was satisfied by a class that performs zero acceleration.

3. **HailoMonitor never opened**: A fully constructed `HailoMonitor` exists in the Pipeline but `open()` was never called. This means `hailo_monitor_->is_open()` always returns `false`, so `hailo_util = 0.0f` every denoise step, triggering perpetual false watchdog recovery actions for the Hailo device.

### 1.3 Parts That Give a False Impression of NPU Acceleration

- **`NpuDmaPipeline`**: Named "DMA" but non-HIP builds use `_aligned_malloc` instead of `hipHostMalloc` — not actually DMA-capable.
- **`PinnedStagingPool`**: Named "Pinned" but non-HIP builds use `std::malloc` — not actually page-locked.
- **`Hailo8lEncoder`**: Named after real silicon, always delegates to `SyntheticNpuEncoder`.
- **`NpuEncoderFactory::create_best_available()`**: Always returns SyntheticNpuEncoder or CpuFallbackEncoder — the "probe" never finds real hardware.
- **`NpuPostProcessorFactory::create_best_available()`**: Always returns SyntheticNpuPostProcessor.
- **`CpuFallbackEncoder` reports `npu_utilization = 0.0f`**: Indistinguishable from "NPU measured at 0% utilization" vs "no NPU present".

---

## 2. Stage 2 — Concrete Changes Made

### 2a. `HardwareAccelerationReport` in `GeneratedImage`

**File:** `code/include/hq/pipeline.hpp`

Added `HardwareAccelerationReport` struct to `GeneratedImage` with explicit boolean fields:
- `text_encode_used_npu`, `text_encode_used_gpu`, `denoise_used_gpu`, `vae_decode_used_gpu`, `post_process_used_npu`, `cfg_blend_used_npu`, `hailo_telemetry_real`, `gpu_telemetry_real`
- Plus `encoder_name`, `post_processor_name`, `gpu_backend_name` for diagnostics

Every `generate()` call now populates this report from ground truth (component `is_available()` and `name()` checks, monitor `synthetic_mode()` and `is_initialized()`). No inference or guessing.

### 2b. HailoMonitor `open()` Bug Fix

**File:** `code/src/pipeline_integration.cpp`

Added `hailo_monitor_->open()` call in Pipeline constructor (after GPUMonitor init). Without this, `is_open()` always returned `false` and Hailo telemetry was permanently zeroed. With the call, `open()` either connects to real Hailo hardware or enters `synthetic_mode_` with explicit WARNING logs.

### 2c. `SyntheticNpuPostProcessor::is_available() → false`

**File:** `code/include/hq/npu_accelerator.hpp`

Changed from `return true` to `return false`. This is the single most damaging silent lie in the codebase. A component that performs memcpy and returns `was_npu_accelerated = false` must not claim it's "available" as an NPU accelerator. The factory now explicitly logs that no NPU hardware is available and it's falling back to CPU pass-through.

Renamed from `"Synthetic-PassThrough"` to `"CPU-PassThrough"` to eliminate the misleading "Synthetic" prefix (which incorrectly implied some form of NPU emulation).

### 2d. Sentinel Values for NPU Metrics

**File:** `code/src/npu_encoder.cpp`, `code/include/hq/npu_pipeline.hpp`

- `CpuFallbackEncoder`: `npu_utilization = -1.0f`, `npu_temperature = -1.0f` (sentinel: "no NPU hardware present")
- `SyntheticNpuEncoder`: `npu_utilization = -2.0f`, `npu_temperature = -2.0f` (sentinel: "synthetic fabricated data")
- Documented sentinel convention in `NpuEncodeResult` struct

Previously both returned `0.0f`, which is indistinguishable from "NPU measured at 0% utilization" — a real and dangerous ambiguity.

### 2e. Honest NpuPostProcessorFactory Logging

**File:** `code/src/npu_accelerator.cpp`

Factory log message changed from "Hailo-8L unavailable, using Synthetic pass-through" to explicit: "No NPU hardware available — using CPU pass-through (no acceleration). post_process() will copy pixels unchanged; blend_noise_cfg() will use CPU scalar math."

### 2f. (From previous round) Runtime WARNING logs on synthetic telemetry

**Files:** `hailo_monitor.cpp`, `gpu_monitor.cpp`, `cerberus_api.cpp`

Every synthetic telemetry path now prints a `WARNING` on every call. Every GPUMonitor `default:` switch case (from `Backend::None`) prints a `WARNING`. CPU utilization on non-Linux prints a `WARNING`. These are per-call, not per-init.

---

## 3. Current Honest State of NPU Participation

| Component | Participation | Mechanism | Truthful? |
|-----------|--------------|-----------|-----------|
| Text encoding | **CPU via ONNX Runtime** | CpuFallbackEncoder (ORT CPU EP) or SyntheticNpuEncoder (XOR hash) | Yes — sentinel metrics, `is_available()` reflects reality, `name()` is honest |
| CFG blend | **CPU scalar SAXPY** | SyntheticNpuPostProcessor::blend_noise_cfg | Now honest — `is_available() = false`, `name() = "CPU-PassThrough"` |
| Post-processing | **CPU memcpy** | SyntheticNpuPostProcessor::post_process | Now honest — `is_available() = false`, `was_npu_accelerated = false` |
| Hailo telemetry | **Synthetic time-varying data** or **zeros if not opened** | HailoMonitor in synthetic_mode | Now detectable — `synthetic_mode()` and `is_open()` are queryable; per-call WARNING logs |
| GPU telemetry | **REAL via NVML** (if NVIDIA driver present) | GPUMonitor with Backend::NVML | Yes — real hardware queries; `Backend::None` case returns 0 with WARNING |
| UNet inference | **CPU via ORT** (CUDA EP fails at runtime) | `ort.gpu_session->Run()` with CPU fallback | Implicit — no explicit surface yet; `HardwareAccelerationReport.denoise_used_gpu` now populated |
| VAE decode | **CPU via ORT** | `ort.vae_session->Run()` with CPU fallback | Same as UNet |
| DMA staging | **Wasted H2D copy** (bug B3) | PinnedStagingPool | Documented but not yet fixed |

---

## 4. Remaining Issues That Would Damage Credibility

### Critical (would cause a reviewer to dismiss the project)

1. **No real model files**: ORT sessions are created but no ONNX models are loaded. The 5 integration tests all `GTEST_SKIP` due to missing model files. The pipeline constructs sessions but never loads CLIP, UNet, or VAE weights. Without real models, inference is impossible — this is the single biggest gap.

2. **CUDA DLL version mismatch**: `onnxruntime_providers_cuda.dll` requires CUDA 12 runtime libraries (`cublas64_12.dll`, `cudart64_12.dll`) but the system has CUDA 13.2. GPU inference silently falls back to CPU. This means even if model files existed, inference would be orders of magnitude slower than claimed.

3. **BUG B3: Staged GPU copy is wasted**: PinnedStagingPool performs an H2D DMA copy of embeddings, then `denoise_step_()` creates Ort::Value tensors from CPU pointers, forcing ORT to do a second implicit H2D copy. The pinned copy is completely wasted.

### Important (would cause a reviewer to question engineering quality)

4. **`NpuDmaPipeline` / `PinnedStagingPool` names imply DMA**: On non-HIP builds, these use heap allocation instead of `hipHostMalloc`. The "DMA" and "Pinned" names are misleading when no HIP runtime is present.

5. **`TieredMemoryManager::WarmPmrResource`**: Documented as "CXL coherent pool" but `detect_cxl()` always returns false. It allocates regular RAM. The configuration defaults mention "128 GiB CXL" which is aspirational, not real.

6. **`ClusterTransport` loopback mode**: When running in single-node mode, `send()` returns success without transmitting data, and `collect_telemetry()` returns fabricated worker health scores. A reviewer inspecting cluster code would find this misleading.

7. **Watchdog recovery callback is a no-op**: The lambda passed to `UtilizationWatchdog` (lines 181-188) always returns `RecoveryResult::PARTIAL`. The actual recovery logic in `on_watchdog_recovery_()` works, but the callback itself does nothing — a reviewer would wonder why it exists.

### Minor (would not block feedback but should be noted)

8. **Test binary crashes on MinGW pipe redirect**: The `LowUtilization_TriggersRecovery` test crashes with ACCESS_VIOLATION when output is piped. The `std::print` shim using `hq_safe_write()` was supposed to fix this, but the `std::fputs` calls in `utilization_watchdog.cpp` constructor bypass the shim.

9. **`read_inference_count()` is always synthetic**: Even with HailoRT hardware connected, `read_inference_count()` falls through to the time-based synthetic simulation because the HailoRT VDevice API integration is not yet implemented. The `#if HAILO_MONITOR_HAS_HAILORT` block is empty except for a comment.

---

## 5. Recommendation

### Conditional — show to a small group of trusted engineers under specific conditions.

**What to show:**
- The architecture, the abstractions (`INpuEncoder`, `INpuPostProcessor`, `HardwareAccelerationReport`), the honest documentation, and the `synthetic_mode()` / sentinel value / `is_available()` truth-telling mechanisms
- The ORT session wiring, the watchdog system, the DEIS scheduler, the NVML telemetry (which is real)
- The HealthScore dashboard, the PipelinePhaseTimings, and the `HardwareAccelerationReport`

**What to explicitly disclose upfront:**
- No real model files are loaded — all inference is a session-creation-then-fail currently
- GPU acceleration is blocked by the CUDA 13.2 / ORT CUDA 12 DLL mismatch
- NPU (Hailo-8L) is entirely simulated — no real inference, no real telemetry
- The Hailo-8L hardware is not installed on the development machine
- This is an architecture-and-structure review, not a working-demo review

**What NOT to claim:**
- "The pipeline runs inference on GPU/NPU" — it runs on CPU, and even that requires model files
- "Hailo-8L telemetry shows real utilization" — it's synthetic time-varying data
- "The DMA path is zero-copy" — BUG B3 means the H2D copy is wasted
- "NPU acceleration is operational" — it's CPU fallback and memcpy pass-through

**Who should review this:**
- Engineers experienced in ONNX Runtime integration, heterogeneous compute, or inference server architecture
- People who can evaluate the quality of the abstractions and error handling, not just the output
- People who understand that a v0.x runtime without real hardware is a scaffold, not a product

**When NOT to show this:**
- To anyone expecting a working demo or visual output
- To anyone who would conflate "synthetic data" with "fraud" rather than "honest placeholder"
- To a broad audience before the CUDA DLL mismatch is resolved (it's the fastest path to a working demo)