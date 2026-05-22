# Production-Grade Test Workloads: UM790 Pro + Hailo-8L Heterogeneous Inference Pipeline

**Document Version:** 1.0  
**Target Platform:** MinisForum UM790 Pro (Ryzen 9 7940HS, Radeon 780M, Hailo-8L)  
**Date:** 2025-06-10  

---

## Table of Contents

1. [Hardware Baseline & Computational Budget](#1-hardware-baseline)
2. [Workload 1: FLUX Large-Scale Generation](#workload-1)
3. [Workload 2: Qwen-Edit Complex Instruction Editing](#workload-2)
4. [Workload 3: Hybrid VLM -> Diffusion Pipeline](#workload-3)
5. [Workload 4: Long-Running Autonomous Style Session](#workload-4)
6. [Workload 5: Sustained Stress / Thermal Boundary Test](#workload-5)
7. [Cross-Workload Summary & Scheduling Matrix](#cross-workload-summary)
8. [Instrumentation Probe Reference](#instrumentation-reference)

---

## 1. Hardware Baseline & Computational Budget <a name="1-hardware-baseline"></a>

### 1.1 Platform Specifications

| Component | Specification | TOPS | Memory | Peak BW |
|---|---|---|---|---|
| **CPU** | AMD Ryzen 9 7940HS (Zen 4, 8C/16T, AVX-512) | ~12 | 64 GB LPDDR5X-7500 (shared) | — |
| **iGPU** | AMD Radeon 780M (gfx1103, RDNA3, 12 CUs) | ~16–17 | 4 GB carve-out from unified pool | ~83 GB/s (system) |
| **NPU** | AMD XDNA1 (Ryzen AI) | ~10 | Shared | — |
| **Accelerator** | Hailo-8L (PCIe Gen3 x2, M.2 2230 A+E) | 13 | On-chip SRAM + PCIe DMA | ~4 GB/s (PCIe) |
| **System RAM** | 64 GB LPDDR5X-7500 dual-channel | — | — | ~83 GB/s theoretical |

### 1.2 Active Compute Budget

```
Total active TOPS (encoding + denoising): ~51 TOPS
    Hailo-8L (encoding):    13 TOPS  -> T5-XXL, CLIP-L, Qwen text encoders
    Radeon 780M (denoising): 16-17 TOPS -> FLUX DiT, Qwen MMDiT diffusion loop
    CPU (VAE + scheduler):   12 TOPS  -> VAE decode, DEIS/Flow scheduler math
```

### 1.3 Memory Budget Per Device

| Device | Available | Model Weights | Activation Scratch | Safety Margin |
|---|---|---|---|---|
| Radeon 780M | 4.0 GB VRAM | 2.6–3.0 GB (quantized) | 0.8–1.2 GB | 0.2 GB |
| Hailo-8L | 8 MB on-chip SRAM | HEF-loaded at boot | — | — |
| System RAM | 64 GB (unified) | VAE: ~500 MB, LoRA: ~100 MB | OS + buffers: ~4 GB | ~55 GB free |

### 1.4 PCIe & Memory Bandwidth Budget

| Link | Theoretical | Realistic (70%) | Latency |
|---|---|---|---|
| PCIe Gen3 x2 (Hailo) | ~2 GB/s | ~1.4 GB/s | ~1 us DMA setup |
| LPDDR5X-7500 (system) | ~83 GB/s | ~58 GB/s | ~10 ns |
| Hailo -> Host DMA (512-token T5 embedding) | — | ~8 MB / 1.4 GB/s = **~6 ms** | — |
| Host -> GPU (embedding upload) | — | ~8 MB / 58 GB/s = **~0.14 ms** | — |

### 1.5 Utilization Metric Definitions

| Device | Metric | Sampling | Burst Profile | Target Sustained |
|---|---|---|---|---|
| Radeon 780M | `gpu_busy_percent` (rsmi) | ~1 ms firmware | 40–95% spikes | **75–80%** |
| Hailo-8L | Power-proxy utilization | 500 ms watchdog | Immediate 90-100% | **90–100%** |

### 1.6 Watchdog Thresholds (from Architecture Spec)

```
NORMAL:   utilization >= 60%                -> no action
WARNING:  40% <= utilization < 60%, 1-7 steps -> log, countdown
CRITICAL: utilization < 40% OR >= 8 steps below 60% -> trigger recovery

Sample interval: 500 ms
Consecutive threshold: 8 steps
Recovery: EP reset + device reset + HEF reload (Hailo) / session rebuild (GPU)
```

---

## Workload 1: FLUX Large-Scale Generation <a name="workload-1"></a>

### Purpose

Tests the **core diffusion generation pipeline** under maximum quality and scale configurations. Validates the end-to-end flow: Hailo text encoding -> GPU denoising -> CPU VAE decode, across resolution and step-count extremes. Measures the pipeline's ability to maintain target utilization at the edge of the VRAM envelope (4 GB). This is the **foundational correctness workload** — all other workloads assume this one passes.

**Primary Stress Targets:** GPU VRAM limit, per-step latency consistency, Hailo encoding throughput at maximum token length, scheduler CPU overhead.

---

### Input Specification

#### Model Versions

| Model | Repository / Checkpoint | Hash / Tag | Format |
|---|---|---|---|
| FLUX.1-dev | `black-forest-labs/FLUX.1-dev` | `dev` | ONNX bf16 + int8 MLP |
| FLUX.1-schnell | `black-forest-labs/FLUX.1-schnell` | `schnell` | ONNX bf16 + int8 MLP |
| T5-XXL Encoder | `google/t5-v1_1-xxl` | `encoder_only` | HEF (Hailo) |
| CLIP-L Encoder | `openai/clip-vit-large-patch14` | `text_encoder` | HEF (Hailo) |
| VAE | `black-forest-labs/FLUX.1-dev` | `ae.safetensors` | ONNX fp16 (CPU) |

#### Test Matrix

| Config ID | Model | Resolution | Steps | Batch | Precision | Token Length |
|---|---|---|---|---|---|---|
| W1-A | FLUX.1-schnell | 1024 x 1024 | 4 | 1 | bf16+int8 | 20 |
| W1-B | FLUX.1-schnell | 1024 x 1024 | 8 | 1 | bf16+int8 | 100 |
| W1-C | FLUX.1-dev | 1024 x 1024 | 20 | 1 | bf16+int8 | 20 |
| W1-D | FLUX.1-dev | 1024 x 1024 | 20 | 1 | bf16+int8 | 100 |
| W1-E | FLUX.1-dev | 1024 x 1024 | 20 | 1 | bf16+int8 | **512** |
| W1-F | FLUX.1-dev | 1024 x 1024 | 30 | 1 | bf16+int8 | 100 |
| W1-G | FLUX.1-dev | 1024 x 1024 | 50 | 1 | bf16+int8 | 100 |
| W1-H | FLUX.1-dev | **2048 x 2048** | 20 | 1 | bf16+int8 | 100 |
| W1-I | FLUX.1-dev | 1024 x 1024 | 20 | **2** | bf16+int8 | 100 |

#### Prompt Catalog

```
Short (20 tokens):  "A red sports car on a mountain road, golden hour"
Medium (100 tokens): "A highly detailed digital painting of a futuristic cityscape at 
dusk, flying vehicles between neon-lit skyscrapers, reflective wet streets, cyberpunk 
aesthetic, volumetric fog, 8k resolution, art by Simon Stalenhag and Syd Mead, 
cinematic composition, dramatic lighting"
Long (512 tokens):  "An ultra-realistic portrait photograph of an elderly fisherman 
with weathered skin and deep-set blue eyes, standing on the deck of a wooden trawler 
boat at dawn. He wears a yellow oilskin jacket covered in salt spray and fish scales, 
a woolen cap pulled low over his forehead. Behind him, the North Atlantic ocean 
stretches to the horizon with gentle rolling swells catching the first golden rays of 
sunrise. Seagulls circle overhead. The fishing nets are piled on the deck beside him, 
glistening with moisture. Shot on medium format film, shallow depth of field, natural 
lighting, documentary photography style, National Geographic quality, visible pores 
and skin texture, authentic emotional expression, storytelling composition with leading 
lines from the boat railing drawing the eye to the subject's face, slight film grain, 
warm color temperature, highlights clipped gently in the sky creating a soft halo 
effect around the subject's silhouette..."
```

#### Scheduler Configuration

| Parameter | Value | Notes |
|---|---|---|
| Scheduler | FlowMatchEulerDiscrete (for dev) / Euler (for schnell) | Matches model training |
| Guidance Scale | 3.5 (dev) / 0.0 (schnell) | Schnell is guidance-distilled |
| Sigma Schedule | karras | Standard for FLUX |
| CPU Math | AVX-512 precomputed | All timesteps + sigmas computed before loop |

---

### Pipeline Flow

```
PHASE 0: PREPARATION (CPU, ~50-200 ms)
    [CPU] Tokenize prompt -> input_ids (T5: 512 max, CLIP: 77 max)
    [CPU] Allocate latent buffer: 1 x 16 x 128 x 128 (1024x1024) or 1 x 16 x 256 x 256 (2048x2048)
    [CPU] Precompute all timesteps and sigma values for scheduler
          -> Store in AVX-512 aligned buffer for zero-allocation lookup during loop

PHASE 1: TEXT ENCODING (Hailo-8L, parallel where possible)
    [Hailo] Load input_ids into HEF input buffer via PCIe DMA
    [Hailo] T5-XXL encoder forward pass
            -> Output: text_embeddings_t5 [seq_len, 4096] fp16
    [Hailo] CLIP-L encoder forward pass (parallel with T5 DMA out)
            -> Output: text_embeddings_clip [77, 768] fp16
    [Hailo] DMA embeddings back to host unified memory
    [Host]  Concatenate embeddings, prepare for GPU upload

PHASE 2: DENOISING LOOP (Radeon 780M, dominant phase)
    [GPU] Upload embeddings to VRAM (via zero-copy from unified memory)
    FOR step = 0 to num_steps - 1:
        [GPU] FLUX DiT forward pass:
              - AdaLN modulation from timestep embedding
              - 19 transformer blocks (self-attention + MLP)
              - Sequence: ~4,096 image patches + 512 T5 tokens + 77 CLIP tokens
              - Output: noise prediction
        [CPU] Scheduler math (AVX-512, ~2ms):
              - Compute x_{t-1} from x_t, noise_pred, sigma_t, sigma_{t-1}
              - Overlapped with GPU DMA for next step's embeddings if batch>1
        [Watchdog] Sample GPU % and Hailo % (every 500ms)
    END FOR

PHASE 3: VAE DECODE (CPU, single-threaded -> multi-threaded)
    [CPU] Copy final latent from GPU to host buffer
    [CPU] Run VAE decoder (ONNX fp16, CPU EP, 4 threads)
          -> 1024x1024: ~800-1200ms
          -> 2048x2048: ~3,500-4,500ms
    [CPU] Post-process: clamp to [0,1], convert to uint8

PHASE 4: SAVE (CPU, async)
    [CPU] Encode to PNG/JPEG (libspng / libjpeg-turbo, async IO thread)
    [CPU] Write to disk (NVMe, ~3 GB/s)
```

#### Timing Diagram (W1-D: 1024x1024, 20 steps, 100 tokens)

```
Time (ms)   0     30    50   750  1500  2250  ...  15750  16550  17350
            |      |     |    |     |     |        |      |      |
Hailo-8L:   [====ENCODE====]  IDLE ...........................  IDLE
            |<-- ~25ms -->|    |<------- ~15,700ms --------->|
GPU 780M:   IDLE  [==== STEP 0 ====][==== STEP 1 ====] ... [=== S19 ====]
            |      |<---- ~720ms ---->|<---- ~720ms ---->|     |<-~720ms->|
CPU (sched):IDLE  [2ms][2ms] .................................. [2ms][VAE]
            |      |    |                                         |<1.0s>|
Watchdog:   .  [S].  [S].  [S] ... (every 500ms, ~31 samples/job)
            |  |  |   |   |   |
            S = sample point (gpu_busy_percent + hailo power proxy)
```

---

### Memory Footprint

#### Phase-by-Phase Breakdown (W1-D: 1024x1024, bs=1, FLUX.1-dev)

| Phase | Component | Size | Location | Notes |
|---|---|---|---|---|
| **Prep** | Tokenized input_ids | 2 x 512 x int64 = 8 KB | CPU heap | T5 + CLIP |
| **Prep** | Precomputed sigmas | 20 x float32 = 80 B | CPU (AVX-512 aligned) | Negligible |
| **Prep** | Initial latent (noise) | 1 x 16 x 128 x 128 x bf16 = 4.2 MB | GPU VRAM | Random init |
| **Encode** | T5-XXL HEF weights | ~1.2 GB (INT8) | Hailo on-chip | Loaded once at boot |
| **Encode** | CLIP-L HEF weights | ~150 MB (INT8) | Hailo on-chip | Loaded once at boot |
| **Encode** | T5 output embeddings | 512 x 4096 x fp16 = **4.0 MB** | Host unified | DMA from Hailo |
| **Encode** | CLIP output embeddings | 77 x 768 x fp16 = **118 KB** | Host unified | DMA from Hailo |
| **Denoise** | FLUX DiT weights (int8 MLP + bf16 attn) | **2,850 MB** | GPU VRAM | Dominant allocation |
| **Denoise** | Activation buffer (x_t) | 4,096+589 x 3,072 x bf16 = **27 MB** | GPU VRAM | Per-layer scratch |
| **Denoise** | Attention K/V cache | 4,685 x 3,072 x 19 layers x bf16 = **519 MB** | GPU VRAM | Gradient-checkpointed |
| **Denoise** | Noise prediction output | 1 x 16 x 128 x 128 x bf16 = 4.2 MB | GPU VRAM | Per-step |
| **VAE** | VAE decoder weights | ~470 MB (fp16) | CPU RAM | ONNX CPU EP |
| **VAE** | VAE output buffer | 3 x 1024 x 1024 x uint8 = **3.0 MB** | CPU RAM | Final image |
| **VAE** | Decode scratch | ~50 MB | CPU RAM | Intermediate activations |
| **Overhead** | ONNX Runtime arenas | ~150 MB | GPU VRAM + CPU | Session overhead |
| **Total GPU** | | **~3,550 MB / 4,096 MB** | | **87% VRAM utilization** |
| **Total Hailo** | | **~1,350 MB SRAM** | | HEF weights |
| **Total CPU** | | **~600 MB** | | VAE + buffers + OS |

#### VRAM Pressure Analysis

```
Radeon 780M VRAM Budget (4,096 MB):
    [|||||||||||||||||||||||||||||||....]  3,550 MB used (87%)
    Model weights:     2,850 MB ████████████████████████████
    Attention K/V:       519 MB █████▌
    Activations/scratch:  80 MB ▊
    Latent buffers:       17 MB ▏
    ORT overhead:        150 MB █▌
    Safety margin:       480 MB ███▌ (required for OS/driver transient)

WARNING: At 2048x2048 (W1-H), latent is 1 x 16 x 256 x 256 = 16.8 MB
         Sequence length = 16,384 patches + 589 text = 16,973 tokens
         Attention K/V scales to ~1,880 MB
         Total GPU: ~4,400 MB -> EXCEEDS 4,096 MB budget
         
         MITIGATION: Tiled denoising (4 tiles of 1024x1024 each, ~25% overlap)
         Effective VRAM: ~3,800 MB (within budget)
         Penalty: ~3.5x step time increase
```

---

### Expected Utilization Profile

#### Time-Series (W1-D: 1024x1024, 20 steps, ~15.5s total)

```
GPU Busy %
100% |                                          ____      ____
 90% |                              ____       /    \    /    \
 80% |                 ____        /    \     /      \  /      \
 75% |    (target)====/    \======/      \===/        \/        \===
 60% |===/                  \                                     \==
 40% | (threshold)
 20% |
  0% +----+----+----+----+----+----+----+----+----+----+----+----+----+
     0   0.5   1    2    3    4    5    ...   14   15   16  (seconds)
     [Enc][=========== DENOISING LOOP (20 x ~720ms) ==========][VAE]

Hailo NN Core % (power-proxy)
100% |====
 90% |    (encoding complete -> idle for rest of job)
 60% |
 40% |
  0% +----+----+----+----+----+----+----+----+----+----+----+----+----+
     0   0.5   1    2    3    4    5    ...   14   15   16  (seconds)
     [ENCODE][IDLE.....................................][IDLE]

System Memory BW %
 30% |              .  .      .  .       .  .       .  .
 20% |    .        / \/ \    / \/ \     / \/ \     / \/ \
 15% |===/ \======/      \==/      \===/      \===/      \===========
 10% |                                                                  [VAE]
  5% |                                                                   \___/
  0% +----+----+----+----+----+----+----+----+----+----+----+----+----+
     0    1    2    3    4    5    ...   14   15   16   17   18 (sec)

Key observations:
- GPU: Burst pattern (40% valley during CPU scheduler, 92% peak during attention)
- Hailo: 100% for ~25ms encoding, then idle (this is expected — no work assigned)
- Memory BW: Spikes at step boundaries (~15% of 83 GB/s = ~12 GB/s transient)
```

#### Single-Step GPU Micro-Profile (one ~720ms denoising step)

```
Time within one step (0 to 720ms):
    0-50ms:   AdaLN embedding + patch embedding   (GPU: 60%,  BW: high)
    50-150ms: Transformer blocks 0-6 (attention)    (GPU: 88%,  BW: med)
    150-250ms: Transformer blocks 7-12 (attention)   (GPU: 92%,  BW: med)
    250-350ms: Transformer blocks 13-18 (attention)  (GPU: 90%,  BW: med)
    350-400ms: Final projection + noise pred         (GPU: 65%,  BW: high)
    400-420ms: CPU scheduler math (AVX-512)          (GPU: 40%,  BW: low)
    420-720ms: Next step prep / overlap              (GPU: 70%,  BW: med)

    Average GPU busy within step: ~78%
    Valley at 400-420ms: CPU math gap (target: reduce to <5ms via prefetch)
```

---

### Target Metrics (Pass/Fail)

#### Latency Targets

| Config ID | Resolution | Steps | Batch | Target Total | Target Per-Step | Hailo Encode | VAE Decode |
|---|---|---|---|---|---|---|---|
| W1-A | 1024x1024 | 4 | 1 | **< 6.5 s** | < 1,200 ms | < 10 ms | < 1,200 ms |
| W1-B | 1024x1024 | 8 | 1 | **< 12.0 s** | < 1,200 ms | < 15 ms | < 1,200 ms |
| W1-C | 1024x1024 | 20 | 1 | **< 28.0 s** | < 1,250 ms | < 10 ms | < 1,200 ms |
| W1-D | 1024x1024 | 20 | 1 | **< 30.0 s** | < 1,300 ms | < 20 ms | < 1,200 ms |
| W1-E | 1024x1024 | 20 | 1 | **< 32.0 s** | < 1,300 ms | < 45 ms | < 1,200 ms |
| W1-F | 1024x1024 | 30 | 1 | **< 42.0 s** | < 1,300 ms | < 20 ms | < 1,200 ms |
| W1-G | 1024x1024 | 50 | 1 | **< 68.0 s** | < 1,300 ms | < 20 ms | < 1,200 ms |
| W1-H | 2048x2048 | 20 | 1 | **< 105 s** | < 4,800 ms (tiled) | < 20 ms | < 4,500 ms |
| W1-I | 1024x1024 | 20 | 2 | **< 52.0 s** | < 2,400 ms (bs=2) | < 20 ms | < 2,400 ms (2 images) |

#### Utilization Targets

| Metric | Target | Fail Threshold | Measurement |
|---|---|---|---|
| GPU sustained avg (denoising only) | 75-80% | < 60% for >= 8 steps | rsmi, 500ms samples |
| GPU peak (within step) | 85-95% | < 70% consistently | rsmi, 500ms samples |
| GPU valley (scheduler gap) | 35-45% | > 50% (indicates CPU bound) | rsmi, 500ms samples |
| Hailo encoding utilization | 90-100% | < 80% | Power proxy, 500ms |
| Per-step latency stddev | < 8% of mean | > 15% (indicates jitter) | Timed probes |
| End-to-end latency stddev (5 runs) | < 5% of mean | > 10% | Wall-clock |

#### Memory Targets

| Metric | Target | Fail Threshold |
|---|---|---|
| GPU VRAM peak | < 3,850 MB | > 4,000 MB (OOM risk) |
| GPU VRAM at end (no leak) | within 50 MB of start | growth > 100 MB across 20 steps |
| Host memory peak | < 1,500 MB | > 2,500 MB |
| Embedding DMA latency | < 10 ms | > 20 ms |

---

### Risk Scenarios

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| **OOM at 2048x2048** | High | Job crash, GPU session loss | Tiled denoising with overlap; automatic fallback to 1024-tile mode |
| **OOM at bs=2** | High | Job crash | Activation checkpointing (trade 30% speed for memory); auto fallback to bs=1 |
| **VRAM fragmentation after 10+ steps** | Medium | Gradual VRAM growth, eventual OOM | Defragmentation pass every N steps; ORT memory pattern optimization |
| **Hailo DMA timeout on 512-token prompt** | Low | Encoder hang, watchdog triggers | DMA timeout 100ms; fallback to CPU T5 encoder |
| **GPU utilization valley too deep (>55%)** | Medium | Watchdog false-positive recovery | CPU scheduler prefetch; AVX-512 precompute all sigmas |
| **Step time jitter >15%** | Medium | Inconsistent UX; watchdog confusion | Lock CPU governor to performance; isolate inference to cores 0-7 |
| **Thermal throttling at step 15+** | Medium | Step time increases 20-40% | iGPU thermal limit monitoring; reduce clock if Tj > 95C |
| **VAE decode memory spike** | Low | Transient 500MB CPU allocation | Pre-allocated output buffer; avoid malloc during decode |

---

### Instrumentation Points

| Probe ID | Location | What | Frequency | Method |
|---|---|---|---|---|
| `W1-GPU-UTIL` | Watchdog thread | `rsmi_dev_gpu_busy_percent_get()` | 500 ms | ROCm SMI C API |
| `W1-GPU-VRAM` | Watchdog thread | `rsmi_dev_memory_usage_get(VRAM)` | 500 ms | ROCm SMI C API |
| `W1-GPU-TEMP` | Watchdog thread | `rsmi_dev_temp_metric_get(JUNCTION)` | 500 ms | ROCm SMI C API |
| `W1-HAILO-PWR` | Watchdog thread | `device_->get_power_measurement()` | 500 ms | HailoRT C++ API |
| `W1-HAILO-TEMP`| Watchdog thread | `device_->get_chip_temperature()` | 500 ms | HailoRT C++ API |
| `W1-STEP-T0` | Top of denoising loop | Step start timestamp | Per step | `std::chrono::high_resolution_clock` |
| `W1-STEP-T1` | Bottom of denoising loop | Step end timestamp + duration | Per step | `std::chrono::high_resolution_clock` |
| `W1-SCHED-T` | Scheduler math function | CPU scheduler latency | Per step | `rdtsc` or chrono |
| `W1-ENC-T0` | Pre-Hailo inference | Encoding start | Per job | chrono |
| `W1-ENC-T1` | Post-Hailo DMA out | Encoding end (including DMA) | Per job | chrono |
| `W1-VAE-T0` | Pre-VAE decode | Decode start | Per job | chrono |
| `W1-VAE-T1` | Post-VAE decode | Decode end | Per job | chrono |
| `W1-MEM-PEAK`| Post-job | Peak GPU + host memory | Per job | rsmi + `getrusage(RUSAGE_SELF)` |
| `W1-EMBED-Q` | Post-encoding | Embedding cosine similarity vs. reference | Per job | CPU dot product check |
| `W1-RECOVERY` | Recovery callback | Recovery trigger count + outcome | Per event | Watchdog log |

---

## Workload 2: Qwen-Edit Complex Instruction Editing <a name="workload-2"></a>

### Purpose

Tests **multi-modal instruction-following editing** via the Qwen-Edit pipeline. Unlike FLUX generation (text-to-image from scratch), Qwen-Edit takes a source image + complex natural-language editing instruction and produces a modified image. This exercises a different model architecture (Qwen MMDiT vs FLUX DiT) with **dual-stream conditioning** (image + text simultaneously), stresses the Hailo Qwen text encoder pathway, and validates the pipeline's ability to handle **layered compositional instructions** with multiple editing operations per job.

**Primary Stress Targets:** Qwen MMDiT attention over dual-modality sequences, instruction parsing overhead on CPU, Hailo Qwen encoder throughput, masked-region consistency across denoising steps.

---

### Input Specification

#### Model Versions

| Model | Repository / Checkpoint | Format | Role |
|---|---|---|---|
| Qwen-Edit-2509-abliterated | `Qwen/Qwen-Edit-2509-abliterated` | ONNX bf16 (GPU) | MMDiT denoising |
| Qwen2.5-VL Text Encoder | `Qwen/Qwen2.5-VL` | HEF (Hailo) | Text encoding for instructions |
| Qwen2.5-VL Vision Tower | `Qwen/Qwen2.5-VL` | ONNX bf16 (CPU) | Image patch encoding (vision features) |
| VAE (SD3-compatible) | `stabilityai/sd-vae-ft-mse` | ONNX fp16 (CPU) | Latent <-> pixel decode/encode |

#### Test Matrix

| Config ID | Operation | Source Res | Steps | Mask Complexity | Instruction Tokens |
|---|---|---|---|---|---|
| W2-A | Style transfer (oil painting) | 1024 x 1024 | 20 | Full frame (none) | 50 |
| W2-B | Object replacement | 1024 x 1024 | 25 | Single bounding box | 120 |
| W2-C | Background extension | 1024 x 1024 | 30 | Inverted region mask | 200 |
| W2-D | Multi-mask compositing (3 objects) | 1024 x 1024 | 30 | 3 separate masks | 350 |
| W2-E | Layered edit (style + object + bg) | 1024 x 1024 | **40** | Multi-layer composite | **500** |
| W2-F | Fine-grained texture edit | 1024 x 1024 | 20 | Pixel-precise mask (PNG) | 80 |
| W2-G | Batch edit (4 images, same instruction) | 1024 x 1024 | 20 | Per-image mask | 100 |

#### Instruction Catalog

```
W2-A (50 tokens):
"Convert this photograph into a Renaissance oil painting style. Use chiaroscuro 
lighting, visible brushstrokes, and a warm amber palette."

W2-B (120 tokens):
"Replace the car in the center of the image with a vintage red Vespa scooter. 
The Vespa should have chrome detailing, a leather seat, and be positioned at the 
same angle as the original car. Preserve the surrounding environment: the cobblestone 
street, the cafe tables, and the pedestrians. Match the lighting and shadows to the 
original scene so the replacement looks photorealistic."

W2-C (200 tokens):
"Extend the background of this portrait to include a full scene. The subject is 
a musician sitting on a wooden stool. Extend the frame upward to show a cozy jazz 
club interior with exposed brick walls, warm pendant lights, and a piano in the 
background. Extend left to show a small round table with a whiskey glass and 
cigarette smoke curling upward. Keep the lighting consistent: warm tungsten from 
above, cool blue rim light from stage left. The musician's original pose and 
expression must remain unchanged."

W2-D (350 tokens):
"Edit this kitchen scene with three changes: (1) Replace the stainless steel 
refrigerator with a vintage Smeg refrigerator in pastel green. (2) Change the 
marble countertop to a dark walnut butcher block surface with visible wood grain 
and knife marks. (3) Replace the pendant lights above the island with three 
basket-weave rattan pendants of decreasing size. Each edit region is masked: 
[MASK_1] covers the refrigerator area from x=50,y=100 to x=300,y=600; [MASK_2] 
covers the countertop from y=450 to y=512 across full width; [MASK_3] covers 
the ceiling light fixtures in the top third. Preserve all other elements: the 
white subway tile backsplash, the brass faucet, the bowl of lemons, and the 
window light from the right side. Ensure color temperature consistency across 
all three edited regions."

W2-E (500 tokens):
"Comprehensive editorial redesign of this living room photograph. Step 1: Apply 
a mid-century modern style transformation with warm teak wood tones, mustard 
accent colors, and geometric patterns. Step 2: Replace the sofa with a low-profile 
Hans Wegner-style leather sofa in cognac brown. Step 3: Add a large Persian rug 
with deep reds and blues under the coffee table area. Step 4: Change the wall 
color from white to a soft sage green. Step 5: Replace the modern floor lamp 
with an Arco floor lamp casting dramatic shadows. Step 6: Add bookshelves to 
the left wall filled with colorful books and small plants. Step 7: Change the 
window curtains to sheer linen in natural white. Preserve: the hardwood floor, 
the ceiling beams, the fireplace mantel, and the natural window light direction. 
All new elements must cast physically correct shadows and respect the existing 
lighting direction from camera-right."
```

#### Mask Specifications

| Mask Type | Format | Size | Generation |
|---|---|---|---|
| Full-frame | None (implicit) | 1024x1024 | All ones |
| Bounding box | Binary PNG | 1024x1024 | User-defined ROI |
| Inverted region | Binary PNG | 1024x1024 | Inverted ROI (edit everything EXCEPT) |
| Multi-mask | 3-channel PNG | 1024x1024 x 3 | Each channel = one object's mask |
| Pixel-precise | 8-bit grayscale PNG | 1024x1024 | Per-pixel blend weight (0-255) |
| Layered composite | Multi-channel + JSON | 1024x1024 | Layer metadata with per-layer strength |

---

### Pipeline Flow

```
PHASE 0: INPUT PREPARATION (CPU, ~100-300 ms)
    [CPU] Load source image -> resize to 1024x1024 -> normalize to [-1, 1]
    [CPU] Load mask(s) -> resize to latent resolution (128x128) -> binarize
    [CPU] Tokenize editing instruction -> input_ids (Qwen tokenizer, 512 max)
    [CPU] Encode source image to latent via VAE encoder
          -> source_latent: 1 x 16 x 128 x 128
    [CPU] Precompute timestep schedule for editing (different from generation)
          -> Qwen-Edit uses inversion-based editing: encode -> add noise -> denoise with guidance

PHASE 1: TEXT ENCODING (Hailo-8L, ~20-60 ms)
    [Hailo] Qwen2.5-VL text encoder HEF forward pass
            -> Output: instruction_embeddings [seq_len, 3584] fp16
    [Hailo] DMA embeddings to host unified memory
    [CPU] (Parallel) Run Qwen2.5-VL vision tower on source image
            -> Output: image_features [num_patches, 3584] fp16
    [CPU] Merge text + vision embeddings into MMDiT conditioning vector

PHASE 2: INVERSION + DENOISING (Radeon 780M, dominant phase)
    [GPU] Upload: source_latent + instruction_embeddings + mask(s)
    
    SUB-PHASE 2a: INVERSION (if using DDIM inversion for fidelity)
        FOR step = num_steps-1 down to 0:
            [GPU] MMDiT forward (conditioned on source image + instruction)
            [GPU] Invert latent: x_t = sqrt(alpha_t)*x_0 + sqrt(1-alpha_t)*noise
        END FOR
        -> Inverted latent trajectory stored (or recomputed on-the-fly)
    
    SUB-PHASE 2b: EDITING DENOISING
        FOR step = 0 to num_steps - 1:
            [GPU] MMDiT forward:
                  - Dual-stream: image patches + text tokens cross-attend
                  - Masked regions: apply per-pixel blend weights
                  - 24 MMDiT blocks (vs 19 in FLUX — larger model)
                  - Sequence: ~4,096 image patches + 512 text + 256 vision
            [CPU] Scheduler math (AVX-512): compute edited x_{t-1}
            [CPU] Apply mask compositing: edited_region * mask + original * (1-mask)
            [Watchdog] Sample GPU % and Hailo % (every 500ms)
        END FOR

PHASE 3: VAE DECODE (CPU)
    [CPU] Copy edited latent from GPU
    [CPU] VAE decode -> edited image 1024x1024
    [CPU] Post-process + save
```

#### Timing Diagram (W2-E: 500 tokens, 40 steps, multi-layer edit)

```
Time (ms)   0     60   200  1000  1800  ...  50000  50800  51600
            |      |     |    |     |         |      |      |
Hailo-8L:   [========ENCODE========]
            |<------ ~55ms ------>|
CPU Vision: [====VISION TOWER====]  (parallel with Hailo, ~150ms)
            |<----- ~150ms ---->|
GPU 780M:          [INV][================ 40 x ~1200ms ===============]
            |      |<-~200ms->|<----------- ~48,000ms ------------->|
CPU (sched+mask):  [..] [3ms][mask][3ms][mask] ... [3ms][mask][VAE]
            |      |    |     |    |     |    |        |    |  |<-1.2s->|
Watchdog:   .     [S].  [S].  [S] ... (every 500ms, ~100 samples/job)
```

---

### Memory Footprint

#### Phase-by-Phase (W2-E: most demanding config)

| Phase | Component | Size | Location |
|---|---|---|---|
| **Prep** | Source image | 3 x 1024 x 1024 x float32 = 12 MB | CPU RAM |
| **Prep** | Source latent (VAE encoded) | 1 x 16 x 128 x 128 x bf16 = 4.2 MB | GPU VRAM |
| **Prep** | Mask composite (4 layers) | 4 x 128 x 128 x float32 = 256 KB | GPU VRAM |
| **Prep** | Tokenized instruction | 512 x int64 = 4 KB | CPU heap |
| **Encode** | Qwen text encoder HEF | ~1.5 GB (INT8) | Hailo SRAM |
| **Encode** | Text embeddings | 512 x 3584 x fp16 = 3.5 MB | Host unified |
| **Vision** | Vision tower weights | ~890 MB (bf16) | CPU RAM |
| **Vision** | Image patch features | 1,024 x 3584 x fp16 = 7.0 MB | CPU RAM |
| **Denoise** | Qwen MMDiT weights | **3,200 MB** (int8 MLP + bf16 attn) | GPU VRAM |
| **Denoise** | Dual-stream K/V cache | ~680 MB (24 layers, dual modality) | GPU VRAM |
| **Denoise** | Inversion trajectory (if stored) | 40 x 4.2 MB = 168 MB | GPU VRAM |
| **Denoise** | Masked region buffers | 3 x 128 x 128 x 16 x bf16 = 12.6 MB | GPU VRAM |
| **VAE** | VAE decoder weights | ~470 MB | CPU RAM |
| **VAE** | Output image buffer | 3 x 1024 x 1024 = 3 MB | CPU RAM |
| **Total GPU** | | **~4,070 MB / 4,096 MB** | |
| **Total Hailo** | | **~1,500 MB** | |
| **Total CPU** | | **~1,400 MB** | |

```
CRITICAL: Qwen MMDiT is LARGER than FLUX DiT (24 blocks vs 19, 3584 dim vs 3072)
VRAM utilization at W2-E: 4,070 / 4,096 = 99.4% — AT ABSOLUTE LIMIT
Mitigation: Inversion trajectory must be computed on-the-fly (recompute, don't store)
With on-the-fly inversion: GPU drops to ~3,900 MB (95% — still very tight)
```

---

### Expected Utilization Profile

#### Time-Series (W2-E: 40 steps, ~51s active denoising)

```
GPU Busy %
100% |                                                  ____
 90% |                                    ____         /    \
 80% |               ____        ____  /    \       /      \
 75% |==============/    \======/    \/      \=====\/        \=========
 60% |                                                           (VAE)
 40% |
  0% +----+----+----+----+----+----+----+----+----+----+----+----+----+
     0   0.5   1    2   ...  10   20   30   40   50   51   52   53
     [Hailo][Vision][INV][======== DENOISING (40 x ~1,200ms) =====][VAE]

Key difference from FLUX:
- Step time is ~50% longer (1,200ms vs 720ms) due to larger MMDiT + dual-stream attention
- Inversion phase adds ~200ms overhead at start
- GPU valleys are WIDER due to CPU mask compositing between steps
- Target sustained: 70-78% (slightly lower than FLUX due to CPU mask overhead)
```

---

### Target Metrics (Pass/Fail)

#### Latency Targets

| Config ID | Operation | Steps | Target Total | Target Per-Step | Hailo Encode | Vision Tower |
|---|---|---|---|---|---|---|
| W2-A | Style transfer | 20 | **< 28 s** | < 1,250 ms | < 15 ms | < 120 ms |
| W2-B | Object replace | 25 | **< 36 s** | < 1,350 ms | < 25 ms | < 120 ms |
| W2-C | Background ext | 30 | **< 44 s** | < 1,350 ms | < 35 ms | < 120 ms |
| W2-D | Multi-mask (3) | 30 | **< 46 s** | < 1,450 ms | < 45 ms | < 120 ms |
| W2-E | Layered (500tk) | 40 | **< 56 s** | < 1,300 ms | < 55 ms | < 120 ms |
| W2-F | Texture edit | 20 | **< 30 s** | < 1,350 ms | < 20 ms | < 120 ms |
| W2-G | Batch (4 imgs) | 20 x 4 | **< 100 s** | < 1,200 ms each | < 25 ms each | < 120 ms each |

#### Edit Quality Targets

| Metric | Target | Fail Threshold | Method |
|---|---|---|---|
| Mask edge adherence (PSNR in masked region) | > 32 dB | < 28 dB | Pixel-level diff vs. ground-truth mask |
| Style consistency (CLIP cosine similarity) | > 0.82 | < 0.75 | CLIP embedding of edited region vs. style reference |
| Structural preservation (SSIM non-edit region) | > 0.92 | < 0.85 | Structural similarity in unmasked pixels |
| Instruction fidelity (text-image alignment) | > 0.78 | < 0.70 | CLIP directional similarity (text -> image change) |
| Cross-step mask consistency | < 3% variance | > 8% variance | Per-pixel variance of mask blend weights across steps |

#### Utilization Targets

| Metric | Target | Fail Threshold |
|---|---|---|
| GPU sustained avg (editing) | 70-78% | < 58% for >= 8 steps |
| Hailo encoding utilization | 90-100% | < 80% |
| Per-step latency stddev | < 10% of mean | > 18% |
| CPU mask compositing time | < 5 ms per step | > 10 ms (widens GPU valley) |

---

### Risk Scenarios

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| **OOM with Qwen MMDiT** | Very High | Job crash (99.4% VRAM at W2-E) | On-the-fly inversion; activation checkpointing; int8 attention weights |
| **CPU mask compositing bottleneck** | Medium | GPU valleys widen to 20-30ms | Move mask blending to GPU shader; pre-compute mask pyramid |
| **Hailo Qwen encoder failure** | Medium | Qwen encoder not yet validated on Hailo-8L | Fallback to CPU encoder (10x slower, ~500ms); validate HEF before production |
| **Vision tower CPU bottleneck** | Medium | 150ms serial dependency before GPU can start | Pipeline: start vision tower in parallel with Hailo encoding |
| **Dual-stream attention OOM** | High | MMDiT dual attention exceeds 4GB | Split attention computation across steps (process image stream, then text stream) |
| **Edit inconsistency at mask boundaries** | Medium | Visible seams at mask edges | Feathered mask blending (gaussian blur on mask edges); 8-16 pixel blend zone |

---

### Instrumentation Points

| Probe ID | Location | What | Frequency |
|---|---|---|---|
| `W2-GPU-UTIL` | Watchdog thread | GPU busy % | 500 ms |
| `W2-HAILO-PWR` | Watchdog thread | Hailo power-proxy % | 500 ms |
| `W2-STEP-T` | Denoising loop | Step duration + breakdown | Per step |
| `W2-ENC-T` | Hailo encoding | Qwen encoder latency | Per job |
| `W2-VISION-T` | Vision tower | Image feature extraction time | Per job |
| `W2-INVERT-T` | Inversion phase | DDIM inversion duration | Per job |
| `W2-MASK-T` | Mask compositing | CPU mask blend time | Per step |
| `W2-MMDIT-MEM` | Post-step | GPU VRAM snapshot | Per step |
| `W2-EDIT-Q` | Post-decode | CLIP directional similarity | Per job |
| `W2-RECOVERY` | Recovery callback | Recovery count + trigger reason | Per event |



---

## Workload 3: Hybrid VLM -> Diffusion Pipeline <a name="workload-3"></a>

### Purpose

Tests **simultaneous dual-accelerator operation** with different model types on each device. This workload chains Qwen2.5-VL (vision-language model) for image understanding with FLUX.1-dev for image generation — the VLM interprets a source image and produces a detailed generation prompt, which is then fed into the diffusion pipeline. This is the **most realistic production scenario** for a creative AI workstation: vision understanding + generation in a single user request.

**Primary Stress Targets:** Concurrent accelerator utilization (both devices active simultaneously with different workloads), pipeline handoff latency between VLM and diffusion phases, memory bandwidth contention when both devices access unified memory, scheduler orchestration complexity, watchdog behavior under mixed-device load.

---

### Input Specification

#### Model Versions

| Model | Repository | Format | Device | Role |
|---|---|---|---|---|
| Qwen2.5-VL-3B-Instruct | `Qwen/Qwen2.5-VL-3B-Instruct` | ONNX bf16 | CPU (vision) + Hailo (text) | Image understanding + prompt generation |
| Qwen2.5-VL Vision Encoder | `Qwen/Qwen2.5-VL-3B-Instruct` | ONNX bf16 | CPU | ViT image patch encoding |
| Qwen2.5-VL Text Decoder | `Qwen/Qwen2.5-VL-3B-Instruct` | HEF | Hailo-8L | Text generation (autoregressive) |
| T5-XXL Encoder | `google/t5-v1_1-xxl` | HEF | Hailo-8L | FLUX text encoding |
| CLIP-L Encoder | `openai/clip-vit-large-patch14` | HEF | Hailo-8L | FLUX text encoding |
| FLUX.1-dev | `black-forest-labs/FLUX.1-dev` | ONNX bf16+int8 | Radeon 780M | Image generation |
| VAE | `black-forest-labs/FLUX.1-dev` | ONNX fp16 | CPU | Decode final latent |

**Note on Qwen2.5-VL-3B:** The 3B parameter variant is chosen specifically to fit within the platform constraints. The full 7B or 72B variants would not fit. The 3B variant provides sufficient vision understanding quality for prompt generation.

#### Test Matrix

| Config ID | Source Image | VLM Task | VLM Output Tokens | FLUX Steps | FLUX Resolution |
|---|---|---|---|---|---|
| W3-A | 512x512 photo | "Describe for generation" | 100 | 20 | 1024x1024 |
| W3-B | 1024x1024 artwork | "Style analysis + gen prompt" | 200 | 20 | 1024x1024 |
| W3-C | 1024x1024 interior | "Redesign this room" | 300 | 30 | 1024x1024 |
| W3-D | 1024x1024 portrait | "Create variation prompt" | 150 | 20 | 1024x1024 |
| W3-E | 512x512 product | "Marketing image generation" | 250 | 25 | 1024x1024 |
| W3-F | 1024x1024 landscape | "Season change + mood shift" | 350 | 30 | 1024x1024 |
| W3-G | 1024x1024 composite | "Multi-element scene prompt" | 400 | 20 | 1024x1024 |

#### VLM Prompt Catalog

```
W3-A: "Look at this image and write a detailed text-to-image generation prompt 
that would recreate it. Include lighting, composition, style, colors, mood, and 
all visible objects. Format as a single paragraph."

W3-B: "Analyze the artistic style of this image. Then write a prompt that 
captures: (1) the medium and technique, (2) the color palette and lighting, 
(3) the compositional structure, (4) the emotional mood. End with a complete 
generation-ready prompt."

W3-C: "You are an interior designer. Analyze this room photo and write a 
prompt for an AI image generator to redesign it in Scandinavian minimalist 
style. Specify: furniture, color scheme, lighting, materials, and decorative 
elements. Keep the room layout and dimensions the same."

W3-D: "Create a variation of this portrait by: changing the background to 
a dramatic studio setting, altering the lighting to Rembrandt style, keeping 
the subject's face identical. Write a detailed generation prompt."

W3-E: "This is a product photo. Write a prompt to generate a professional 
e-commerce marketing image: clean white background, soft shadow underneath, 
slight reflection, studio lighting from top-left, 45-degree product angle, 
8k commercial photography quality."

W3-F: "Transform this summer landscape into a winter scene. Write a prompt 
that changes: season to deep winter with snow, time of day to blue hour, 
mood to melancholic and serene. Keep the same geographic features: mountains, 
lake, trees. Specify color temperature, atmospheric conditions, and lighting."

W3-G: "This composite image has multiple elements. Write a prompt that: 
describes the main subject in detail, specifies the relationship between 
foreground and background elements, sets the lighting direction and quality, 
defines the depth of field, and establishes the overall narrative mood. 
Minimum 300 words."
```

#### Pipeline Orchestration Parameters

| Parameter | Value | Rationale |
|---|---|---|
| VLM max new tokens | 100-400 (config-dependent) | Autoregressive generation on Hailo |
| VLM temperature | 0.7 | Balance creativity and coherence |
| VLM top-p | 0.9 | Nucleus sampling |
| FLUX guidance scale | 3.5 | Standard for FLUX.1-dev |
| FLUX scheduler | FlowMatchEulerDiscrete | Matches model |
| Pipeline overlap mode | `VLM_STREAM` | Start FLUX encoding while VLM still generating |

---

### Pipeline Flow

```
PHASE 0: VLM IMAGE ENCODING (CPU, ~200-400 ms)
    [CPU] Load source image -> resize to Qwen-VL input size (448x448 or native)
    [CPU] ViT vision encoder forward pass (ONNX CPU EP, AVX-512)
            -> image_embeds: [num_patches, 2048] bf16
    [CPU] Prepare initial text prompt tokens for VLM
            -> system prompt: "You are a helpful visual assistant..."
            -> user prompt: [task-specific question]
            -> input_ids: [seq_len,] int64

PHASE 1: VLM TEXT GENERATION (Hailo-8L + CPU, ~2-8 seconds)
    
    ITERATIVE AUTOREGRESSIVE GENERATION:
    FOR token_position = 0 to max_new_tokens - 1:
        
        [Hailo] Qwen2.5-VL text decoder HEF forward pass:
                Input: current input_ids + image_embeds cross-attention
                Output: logits for next token [vocab_size]
                Latency per token: ~15-25ms (Hailo autoregressive)
        
        [CPU] Sample next token (temperature scaling + top-p):
                -> Greedy or nucleus sampling
                -> Append to input_ids for next iteration
        
        [CPU] (OVERLAP OPTIMIZATION) When token_position >= 20:
                -> Feed completed sentence fragments to T5-XXL tokenizer
                -> Begin pre-encoding on Hailo (overlapped with VLM generation)
                -> This hides T5 encoding latency inside VLM generation
        
        [Watchdog] Sample BOTH devices:
                Hailo: text decoder active (high utilization expected)
                GPU: idle during VLM phase (expected 0-5%)
                
    END FOR
    
    -> Generated prompt text: 100-400 tokens

PHASE 2: FLUX ENCODING (Hailo-8L, ~15-50 ms — mostly overlapped)
    
    If overlap mode enabled:
        [Hailo] T5-XXL encoding of generated prompt (mostly COMPLETE from overlap)
        [Hailo] CLIP-L encoding of truncated prompt (77 tokens)
        [Hailo] DMA embeddings to host
        -> Actual latency: < 10ms (remaining work)
    
    If overlap disabled (baseline):
        [Hailo] Full T5-XXL + CLIP-L encoding: ~20-50ms

PHASE 3: FLUX DENOISING (Radeon 780M, ~25-40 seconds)
    [GPU] Identical to Workload 1 Phase 2
    -> 20-30 steps of FLUX DiT at 1024x1024
    -> Hailo is IDLE during this phase (expected)

PHASE 4: VAE DECODE + SAVE (CPU, ~1-2 seconds)
    [CPU] Identical to Workload 1 Phase 3
```

#### Timing Diagram (W3-C: 300 VLM tokens, 30 FLUX steps)

```
Time (s)    0    0.3   0.5    3.5    5.0    5.1   5.2   40.2   41.0   42.0
            |     |     |      |      |      |     |      |      |      |
CPU Vision: [====ViT====]
            |<-~300ms->|
Hailo-8L:         [========= VLM GENERATION (300 x ~15ms = ~4.5s) =====]
                  |<- - - T5 OVERLAP BEGINS at token 20 - - - ->|<T5+CLIP>|<-IDLE->
                  |<=========== ~5.0s total active ============>|<~10ms> |
GPU 780M:  IDLE  [IDLE during VLM]                              [====30 x ~1.2s====]
            |     |                                              |<--- ~36s --->|
CPU sched:  [prep][tok][tok] ... [tok][overlap_enc]             [s][s][s]...[s][VAE]
                                                              |<~36s >|<~1s>|
Watchdog:   [S]  [S]  [S]   [S]   [S]   [S]   [S]  [S]   [S]  [S] [S] ... [S] [S]
                  ^     ^     ^     ^     ^     ^    ^     ^
                  |     |     |     |     |     |    |     |
                  +-----+-----+-----+-----+-----+----+-----+ (every 500ms)

CRITICAL OBSERVATION:
    Hailo is 100% active during VLM generation (~5s)
    GPU is 0-5% during VLM generation
    GPU jumps to 75-80% during FLUX denoising
    Hailo drops to 0% during FLUX denoising
    This is BY DESIGN — the test validates the handoff, not concurrent same-device use
    
    Memory bandwidth contention point: at T=5.0s, Hailo DMAs embeddings 
    WHILE GPU begins loading its first denoising step. Both hit LPDDR5X.
    Expected BW spike: ~15-20 GB/s transient (well under 83 GB/s budget)
```

#### Accelerator Handoff Detail

```
HANDOFF from Hailo (encoding) to GPU (denoising) at T ~ 5.0s:

    T=4,980ms: Hailo completes final DMA of T5 embeddings to host buffer
    T=4,985ms: Host memcpy (unified memory) embeddings -> GPU visible heap
    T=4,990ms: GPU kernel launch for first denoising step
    T=4,991ms: GPU begins first AdaLN computation
    
    Handoff latency: ~11ms (target: < 15ms)
    
    If handoff > 30ms: GPU valley at step 0 widens, watchdog may flag
    If handoff > 50ms: First step latency increases, triggers WARNING state
```

---

### Memory Footprint

#### Phase-by-Phase (W3-C: 1024x1024 source, 300 VLM tokens, 30 FLUX steps)

| Phase | Component | Size | Location |
|---|---|---|---|
| **VLM ViT** | Vision encoder weights | ~650 MB (bf16) | CPU RAM |
| **VLM ViT** | Image embeddings | 256 x 2048 x bf16 = 1 MB | CPU RAM |
| **VLM ViT** | ViT activation scratch | ~80 MB | CPU RAM |
| **VLM Decode** | Qwen text decoder HEF | ~800 MB (INT8) | Hailo SRAM |
| **VLM Decode** | KV cache (300 tokens x 24 layers) | 300 x 24 x 2048 x bf16 = 5.5 MB | Hailo SRAM |
| **VLM Decode** | Generated text (300 tokens) | ~1,200 bytes | CPU heap |
| **FLUX Encode** | T5-XXL HEF | ~1.2 GB | Hailo SRAM (same as W1) |
| **FLUX Encode** | CLIP-L HEF | ~150 MB | Hailo SRAM (same as W1) |
| **FLUX Encode** | T5 output embeddings | 512 x 4096 x fp16 = 4.0 MB | Host unified |
| **FLUX Encode** | CLIP output embeddings | 77 x 768 x fp16 = 118 KB | Host unified |
| **FLUX Denoise** | FLUX DiT weights | 2,850 MB | GPU VRAM |
| **FLUX Denoise** | Activation + K/V cache | ~580 MB | GPU VRAM |
| **FLUX Denoise** | Latent buffer | 4.2 MB | GPU VRAM |
| **VAE** | VAE decoder weights | ~470 MB | CPU RAM |
| **VAE** | Output buffer | 3 MB | CPU RAM |
| **Total GPU (peak)** | | **~3,440 MB** | |
| **Total Hailo (peak)** | | **~2,000 MB** (switching between VLM dec and T5) | |
| **Total CPU (peak)** | | **~1,900 MB** | |

```
IMPORTANT: Hailo cannot run VLM decoder AND T5 encoder simultaneously.
They share the same 8MB on-chip SRAM (HEF weights loaded sequentially).
The handoff requires:
    1. Unload Qwen decoder HEF (~10ms)
    2. Load T5-XXL HEF (~15ms)
    3. Run T5 encoding (~20ms)
Total switch time: ~45ms (included in handoff latency budget)

If using dual-HEF mode (both loaded, context-switching):
    Context switch time: ~5ms (preferred)
    Total Hailo SRAM required: 800MB (Qwen) + 1,200MB (T5) + 150MB (CLIP) = ~2.15 GB
    Hailo-8L has sufficient SRAM for dual-HEF if compiled correctly.
```

---

### Expected Utilization Profile

#### Time-Series (W3-C: full pipeline ~42 seconds)

```
GPU Busy %
100% |
 80% |
 75% |                                          ____________ sustained
 60% |                                         /            \
 40% |                                        /              \
 20% |                                       /                \
  0% +----+----+----+----+----+----+----+----+----+----+----+----+----+
     0    1    2    3    4    5    6   ...  20   30   40   41   42
     [IDLE.........][======= 30 x ~1.2s FLUX denoising ========][VAE]

Hailo NN Core %
100% |         [========================================]
 90% |         [                                        ]
 60% |         [                                        ] [ENC][IDLE]
 40% |
  0% +----+----+----+----+----+----+----+----+----+----+----+----+----+
     0    1    2    3    4    5    6    7   ...  20   30   40   41   42
     [IDLE][======= VLM autoregressive gen (~5s) ======][T5][IDLE]

System Memory BW %
 25% |    .        .       .  .        .        .  .      .  .
 20% |   / \      / \     / \/ \      / \      / \/ \    / \/ \
 15% |==/   \====/   \===/      \====/   \====/      \==/      \=====
 10% |
  0% +----+----+----+----+----+----+----+----+----+----+----+----+----+
     0    1    2    3    4    5    6    7   ...  20   30   40   41   42
     [ViT][==== VLM gen (modest BW) ====][BURST][==== FLUX (high BW) ===]
                                         ^
                                         |
                              Handoff BW spike at T~5s
                              (Hailo DMA out + GPU DMA in overlap)

Key Test: At T=5s, memory BW should NOT exceed 30% (~25 GB/s)
If it does: indicates contention, pipeline should stagger by 100ms
```

#### Device Concurrency Analysis

```
Time Region        GPU       Hailo-8L      Memory BW       Concurrent?
--------------------------------------------------------------------------------
T=0-0.3s          0%        0%            5% (ViT load)   No
T=0.3-5.0s        0-5%      90-100%       8% (token gen)  PARTIAL (Hailo only)
T=5.0-5.1s        5-60%     90-100%       22% (handoff)   YES (both active!)
T=5.1-41s         75-80%    0%            15% (denoising) No (GPU only)
T=41-42s          5%        0%            10% (VAE)       No (CPU+GPU)

Concurrent overlap window: ~100ms at handoff (T=5.0-5.1s)
This is the ONLY period where both devices compete for memory BW.
The test validates that this 100ms window does not cause:
    - GPU underrun (step 0 latency spike)
    - Hailo DMA timeout
    - Watchdog false trigger
```

---

### Target Metrics (Pass/Fail)

#### End-to-End Latency Targets

| Config ID | VLM Tokens | FLUX Steps | VLM Time | FLUX Time | Handoff | **Total E2E** |
|---|---|---|---|---|---|---|
| W3-A | 100 | 20 | < 2.5 s | < 28 s | < 15 ms | **< 31.0 s** |
| W3-B | 200 | 20 | < 4.0 s | < 28 s | < 15 ms | **< 32.5 s** |
| W3-C | 300 | 30 | < 5.5 s | < 42 s | < 15 ms | **< 48.0 s** |
| W3-D | 150 | 20 | < 3.0 s | < 28 s | < 15 ms | **< 31.5 s** |
| W3-E | 250 | 25 | < 4.5 s | < 35 s | < 15 ms | **< 40.0 s** |
| W3-F | 350 | 30 | < 6.0 s | < 42 s | < 15 ms | **< 48.5 s** |
| W3-G | 400 | 20 | < 7.0 s | < 28 s | < 15 ms | **< 35.5 s** |

#### Handoff-Specific Targets

| Metric | Target | Fail Threshold | Measurement |
|---|---|---|---|
| Hailo->GPU handoff latency | < 15 ms | > 30 ms | Chrono between DMA complete and GPU kernel launch |
| Handoff BW spike | < 25% of memory BW | > 40% | Instrumented memory BW counter |
| Step 0 latency (post-handoff) | within 10% of steady-state step | > 20% slower | Per-step timer |
| GPU utilization at step 0 | > 60% within 200ms | < 40% after 500ms | rsmi sample |

#### VLM Quality Targets

| Metric | Target | Fail Threshold |
|---|---|---|
| Prompt relevance (BERTScore vs. reference) | > 0.78 | < 0.65 |
| Prompt length adherence | within 10% of target | > 25% off |
| Prompt coherence (perplexity) | < 45 | > 65 |
| Format compliance (% following requested structure) | > 90% | < 70% |

#### Utilization Targets

| Metric | Target | Fail Threshold |
|---|---|---|
| Hailo during VLM generation | 90-100% | < 75% for > 2s |
| GPU during FLUX denoising | 75-80% | < 60% for >= 8 steps |
| GPU idle during VLM (acceptable) | 0-10% | > 20% (leak suspicion) |
| Memory BW at handoff | < 25% | > 40% |

---

### Risk Scenarios

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| **Hailo HEF switch delay** | Medium | 45ms unload/load adds to handoff | Dual-HEF mode; context switch instead of full reload |
| **VLM generation too slow** | Medium | 300 tokens x 25ms = 7.5s exceeds budget | KV cache quantization (int8); reduce to 1.8B variant if needed |
| **VLM output quality poor** | Low | Generated prompt produces bad image | Temperature tuning; fallback to template-based prompts |
| **Memory BW contention at handoff** | Medium | Both devices DMA simultaneously | Stagger: delay GPU kernel launch by 50ms after Hailo DMA complete |
| **GPU idle detection false positive** | Medium | Watchdog sees 0% GPU during VLM phase | Phase-aware watchdog: disable GPU monitoring during known idle phases |
| **Long VLM prompts overflow T5** | Low | 400 VLM tokens -> T5 truncate to 512 | T5 handles 512 natively; ok up to 512. Beyond: split into chunks |
| **Qwen decoder KV cache exceeds Hailo SRAM** | Medium | 400 tokens x 24 layers x 2048 x fp16 = 7.8 MB | Quantize KV to int8; 3.9 MB fits |

---

### Instrumentation Points

| Probe ID | Location | What | Frequency |
|---|---|---|---|
| `W3-GPU-UTIL` | Watchdog thread | GPU busy % | 500 ms |
| `W3-HAILO-PWR` | Watchdog thread | Hailo power-proxy % | 500 ms |
| `W3-VLM-T0` | Pre-ViT | Vision encoding start | Per job |
| `W3-VLM-T1` | Post-ViT | Vision encoding end | Per job |
| `W3-VLM-GEN-T0` | First token generation | VLM text gen start | Per job |
| `W3-VLM-GEN-T1` | Last token generation | VLM text gen end | Per job |
| `W3-VLM-TOK` | Per-token | Token generation latency | Per token |
| `W3-HANDOFF-T0` | Hailo DMA complete | Handoff start marker | Per job |
| `W3-HANDOFF-T1` | GPU kernel launch | Handoff end marker | Per job |
| `W3-BW-SPIKE` | Memory controller | Peak BW during handoff | Once per handoff |
| `W3-FLUX-STEP0` | Step 0 of denoising | First step latency | Per job |
| `W3-RECOVERY` | Recovery callback | Recovery triggers + reason | Per event |



---

## Workload 4: Long-Running Autonomous Style Session <a name="workload-4"></a>

### Purpose

Tests **sustained correctness and thermal stability** over an extended session mimicking a real-world autonomous creative workflow. Unlike short single-shot workloads, this test runs 50+ consecutive image generations with enforced style consistency (same LoRA/style embedding) across varied prompts over a target duration of **2+ hours**. It is designed to expose: memory fragmentation in the ORT allocator, thermal throttling drift, VRAM leak accumulation, utilization decay as caches warm/cool, and watchdog stability under continuous operation.

**Primary Stress Targets:** Thermal envelope sustainability (iGPU TDP vs. cooling), memory fragmentation over 50+ allocations, ORT session memory leak detection, LoRA loading/unloading stability, utilization consistency over time, watchdog false-positive rate under sustained load.

---

### Input Specification

#### Model Versions

| Model | Repository | Format | Device |
|---|---|---|---|
| FLUX.1-dev | `black-forest-labs/FLUX.1-dev` | ONNX bf16+int8 | Radeon 780M |
| T5-XXL Encoder | `google/t5-v1_1-xxl` | HEF | Hailo-8L |
| CLIP-L Encoder | `openai/clip-vit-large-patch14` | HEF | Hailo-8L |
| VAE | `black-forest-labs/FLUX.1-dev` | ONNX fp16 | CPU |
| Style LoRA | `strangerzonehf/Flux-Sketch-LoRA` or custom | ONNX adapters | GPU (loaded per-session) |

#### Session Configuration

| Parameter | Value | Notes |
|---|---|---|
| **Session duration target** | 2.0 - 2.5 hours | Monitored, not enforced hard limit |
| **Images to generate** | 60 (minimum 50 required for pass) | One every ~2 minutes |
| **Resolution** | 1024 x 1024 | Fixed for consistency |
| **Steps per image** | 25 (FLUX.1-dev) | Standard quality setting |
| **Batch size** | 1 | Memory stability priority |
| **Token length** | 80-150 tokens per prompt | Varied (see prompt catalog) |
| **Guidance scale** | 3.5 | Fixed |
| **LoRA** | Single style LoRA, weight 0.8 | Fixed throughout session |
| **Scheduler** | FlowMatchEulerDiscrete | Fixed |
| **Inter-image delay** | 5-15 seconds (randomized) | Simulates real user think-time |
| **VRAM defrag interval** | Every 10 images | Explicit `Ort::Allocator` trim call |

#### Style LoRA Specification

```
LoRA: "Renaissance Oil Painting" style
    - File: renaissance_oil_painting_flux_lora_16rank.onnx
    - Rank: 16
    - Alpha: 16
    - Applied weight: 0.8 (80% style strength)
    - Loaded: Once at session start, remains in GPU VRAM for all 60 images
    - VRAM footprint: ~105 MB (16 rank x ~6.5M DiT params)
    
Style embedding: Pre-computed at session start via reference image
    - Reference: "mona_lisa_style_reference_1024.jpg"
    - Embedding: 1 x 4096 x fp16 = 8 KB (negligible)
    - Injected into T5 output embeddings at position 0 for all generations
```

#### Prompt Catalog (60 Varied Prompts)

The session cycles through 60 distinct prompts while keeping the **style token fixed**. Prompts are drawn from 6 categories (10 prompts each):

```
CATEGORY A: Portraits (prompts 1-10)
    "A portrait of a young woman with auburn hair, [STYLE]"
    "An old fisherman with weathered skin and deep eyes, [STYLE]"
    "A child playing with a puppy in a garden, [STYLE]"
    "A royal king on his throne wearing a crown, [STYLE]"
    "A jazz musician playing saxophone at midnight, [STYLE]"
    "A ballerina in mid-leap on stage, [STYLE]"
    "A scientist examining a specimen through microscope, [STYLE]"
    "A warrior in ornate armor holding a sword, [STYLE]"
    "A mother holding her newborn baby, [STYLE]"
    "A street vendor selling flowers in a market, [STYLE]"

CATEGORY B: Landscapes (prompts 11-20)
    "A misty mountain valley at sunrise, [STYLE]"
    "A stormy seascape with crashing waves on cliffs, [STYLE]"
    "A bamboo forest with a winding stone path, [STYLE]"
    "An aurora borealis over a frozen lake, [STYLE]"
    "A desert oasis with palm trees and a tent, [STYLE]"
    "A rolling vineyard in Tuscany at harvest time, [STYLE]"
    "A volcanic eruption seen from a safe distance, [STYLE]"
    "A coral reef underwater with tropical fish, [STYLE]"
    "A cherry blossom park in full bloom with a pagoda, [STYLE]"
    "A canyon with a river at golden hour, [STYLE]"

CATEGORY C: Architecture (prompts 21-30)
    "A Gothic cathedral interior with stained glass, [STYLE]"
    "A futuristic space station control room, [STYLE]"
    "A traditional Japanese tea house in a garden, [STYLE]"
    "An Art Deco ballroom with chandeliers, [STYLE]"
    "A ruined castle on a hilltop at dusk, [STYLE]"
    "A modern glass skyscraper reflecting clouds, [STYLE]"
    "An ancient Egyptian temple with hieroglyphics, [STYLE]"
    "A cozy log cabin in a snowy forest, [STYLE]"
    "A Venetian canal scene with gondolas, [STYLE]"
    "A Moroccan riad courtyard with mosaic tiles, [STYLE]"

CATEGORY D: Objects / Still Life (prompts 31-40)
    "An arrangement of antique books and quill pens, [STYLE]"
    "A bowl of fresh fruit on a wooden table, [STYLE]"
    "A vintage pocket watch on velvet cloth, [STYLE]"
    "A collection of seashells and starfish, [STYLE]"
    "A glass perfume bottle with roses, [STYLE]"
    "A copper kettle steaming on a stove, [STYLE]"
    "A crystal chandelier against a dark ceiling, [STYLE]"
    "A violin and sheet music on a windowsill, [STYLE]"
    "A ceramic vase with wildflowers, [STYLE]"
    "A chess board mid-game with dramatic lighting, [STYLE]"

CATEGORY E: Animals / Nature (prompts 41-50)
    "A white stallion running through a meadow, [STYLE]"
    "An eagle soaring above mountain peaks, [STYLE]"
    "A family of deer in an autumn forest, [STYLE]"
    "A peacock with feathers fully fanned, [STYLE]"
    "A pod of dolphins leaping from ocean waves, [STYLE]"
    "A butterfly emerging from its chrysalis, [STYLE]"
    "A wolf howling at a full moon, [STYLE]"
    "A hummingbird hovering near red flowers, [STYLE]"
    "An elephant herd walking across savanna, [STYLE]"
    "A polar bear on an ice floe at sunset, [STYLE]"

CATEGORY F: Abstract / Fantasy (prompts 51-60)
    "A dragon curled around a mountain of gold, [STYLE]"
    "A floating island with waterfalls in the sky, [STYLE]"
    "A phoenix rising from flames and ashes, [STYLE]"
    "A crystal cave with bioluminescent plants, [STYLE]"
    "A mermaid sitting on a rock by the sea, [STYLE]"
    "A clock tower in a steampunk city, [STYLE]"
    "A unicorn in an enchanted moonlit forest, [STYLE]"
    "A giant tree with a village built in its branches, [STYLE]"
    "A celestial map painted on a domed ceiling, [STYLE]"
    "A library that extends infinitely into darkness, [STYLE]"

[STYLE] = "Renaissance oil painting style, visible brushstrokes, 
           chiaroscuro lighting, warm amber palette, rich textures, 
           classical composition, museum quality fine art"
```

#### Thermal Monitoring Parameters

| Parameter | Sample Interval | Alert Threshold | Critical Threshold |
|---|---|---|---|
| iGPU junction temperature | 500 ms (watchdog) | > 90 deg C | > 100 deg C |
| iGPU clock frequency | 500 ms | Drop > 10% from nominal | Drop > 25% from nominal |
| iGPU power (proxy from util) | 500 ms | Sustained drop > 15% | Sustained drop > 30% |
| Hailo temperature | 500 ms | > 75 deg C | > 85 deg C |
| Host CPU temperature | 5 s | > 85 deg C | > 95 deg C |
| Host power (ACPI) | 5 s | > 65W sustained | > 80W sustained |

---

### Pipeline Flow

```
SESSION INITIALIZATION (~30 seconds, once)
    [CPU] Load FLUX DiT model -> GPU VRAM (2,850 MB)
    [CPU] Load style LoRA adapters -> GPU VRAM (+105 MB = 2,955 MB total)
    [CPU] Compute style embedding from reference image
    [Hailo] Load T5-XXL HEF (if not already loaded)
    [Hailo] Load CLIP-L HEF (if not already loaded)
    [CPU] Pre-allocate all persistent buffers (latents, embeddings, output)
    [CPU] Start watchdog monitoring thread
    [CPU] Initialize session log: timestamp, baseline metrics
    [Watchdog] Baseline sample: record GPU temp, Hailo temp, ambient util

PER-IMAGE LOOP (executed 60 times, ~2 min/image = ~2 hours total)

    FOR image_id = 1 to 60:
        
        [CPU] Select prompt from catalog[image_id]
        [CPU] Tokenize: T5 (512 max) + CLIP (77 max)
        [CPU] Inject style embedding into T5 output position 0
        
        PHASE 1: ENCODE (Hailo-8L, ~20-30ms)
            [Hailo] T5-XXL forward pass with style embedding injection
            [Hailo] CLIP-L forward pass
            [Hailo] DMA embeddings to host
        
        PHASE 2: DENOISE (GPU, ~32s per image = 25 steps x ~1,280ms)
            [GPU] Upload embeddings + apply LoRA adapters
            FOR step = 0 to 24:
                [GPU] FLUX DiT forward with LoRA injection at each layer
                      - LoRA weights fused at runtime: W_eff = W_base + alpha * B * A
                      - 19 blocks, each with LoRA on Q/K/V/O projections + MLP
                [CPU] Scheduler math (AVX-512, ~2ms)
                [Watchdog] Sample GPU %, Hailo %, temperatures
            END FOR
        
        PHASE 3: DECODE (CPU, ~1.0s)
            [CPU] VAE decode latent -> 1024x1024 image
            [CPU] Save to disk: ./session_output/img_{image_id:03d}.png
        
        PHASE 4: INTER-IMAGE (CPU, 5-15s randomized)
            [CPU] Log per-image metrics to session CSV
            [CPU] If image_id % 10 == 0:
                      - Run ORT allocator trim (release unused arenas)
                      - Take full memory snapshot (GPU + host)
                      - Compare to baseline, check for growth > 50 MB
            [CPU] If image_id % 20 == 0:
                      - Compute style consistency score vs. image 1
                      - Alert if CLIP style drift > 15%
            [CPU] Random sleep 5-15s (thermal recovery window)
        
    END FOR

SESSION TEARDOWN (~5 seconds)
    [CPU] Compute session summary statistics
    [CPU] Save session report: ./session_output/session_report.json
    [CPU] Unload LoRA adapters from GPU
    [Watchdog] Final sample + shutdown
```

#### Timing Diagram (Session Overview)

```
Time    0      30s     2m      4m      6m     ...     1h58m   2h0m
        |       |       |       |       |              |       |
Image:  [INIT]  [01]    [02]    [03]    [04]    ...    [59]    [60]
GPU:    [load]  [====denoise====][====denoise====] ... [====denoise====]
Hailo:  [HEF]   [enc][enc]      [enc][enc]       ... [enc][enc]
CPU:    [setup] [VAE][log][~10s][VAE][log][~10s]... [VAE][log][report]
        |       |<- img 1 (~35s)->|<- img 2 (~35s)->|  |<- img 60 ->|
Watchdog: [base][S][S][S]...[S]   [S][S][S]...[S]    ...  [S]...[S][final]

Every 10th image (~20 min intervals):
        [DEFRAG][mem_snap][style_check]
        |<-- ~500ms overhead -->|
```

---

### Memory Footprint

#### Per-Image (Stable State, Post-Image-1)

| Component | Size | Location | Notes |
|---|---|---|---|
| FLUX DiT base weights | 2,850 MB | GPU VRAM | Constant throughout session |
| LoRA adapter weights (rank-16) | 105 MB | GPU VRAM | Constant (loaded at init) |
| Style embedding | 8 KB | GPU VRAM | Injected per-generation |
| Latent buffer | 4.2 MB | GPU VRAM | Reused per-image |
| Activation scratch | ~580 MB | GPU VRAM | Reused per-step |
| T5 embeddings (with style) | 4.0 MB | Host unified | Recomputed per-image |
| CLIP embeddings | 118 KB | Host unified | Recomputed per-image |
| VAE output buffer | 3 MB | CPU RAM | Reused per-image |
| Session log buffer | ~10 KB | CPU RAM | Append-only CSV |
| **Total GPU (steady state)** | **~3,540 MB** | | **86.4% of budget — tight but stable** |

#### Memory Drift Tracking

```
Expected memory pattern over 60 images:

GPU VRAM (MB)
4000 |                                    (defrag)      (defrag)
3800 |                                          \       /
3600 |  ____                                    \_____/
3540 |_/    \________________________________________\________
3500 |  img1   img5   img10   img20   img30   img40   img50   img60
     |
     +-- Baseline: 3,540 MB at img1
     +-- Expected drift: +0-20 MB over 60 images (ORT allocator fragmentation)
     +-- Defrag pass at img10, 20, 30, 40, 50, 60 should return to baseline
     +-- FAIL if post-defrag > 3,600 MB (growth > 60 MB)
     +-- FAIL if any point exceeds 3,900 MB (OOM risk)

Host RAM (MB)
2000 |
1800 |  /
1600 | /
1500 |/________________________________________________________
     |  img1                                        img60
     +-- Baseline: ~1,500 MB (OS + ORT CPU sessions + VAE)
     +-- Expected drift: +0-100 MB over 2 hours (acceptable)
     +-- FAIL if growth > 300 MB (indicates memory leak)
```

---

### Expected Utilization Profile

#### Sustained Session Profile (images 1, 15, 30, 45, 60 compared)

```
GPU Busy % (averaged over each image's denoising phase)
100% |
 85% | img1  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ (fresh, cool)
 80% | img15 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  (warmed up, optimal)
 75% | img30 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~   (stable sustained)
 70% | img45 ~~~~~~~~~~~~~~~~~~~~~~~~~~~    (possible thermal dip)
 65% | img60 ~~~~~~~~~~~~~~~~~~~~~~~~~~     (thermal equilibrium)
 60% |___________________________________________________________
     |  img1    img15    img30    img45    img60
     
Expected thermal drift:
    Images 1-10:   78-82% GPU (cool start)
    Images 11-30:  75-80% GPU (thermal equilibrium)
    Images 31-50:  72-78% GPU (possible mild throttling)
    Images 51-60:  70-76% GPU (fully warm system)
    
    ACCEPTABLE drift: -8% from baseline (img1 vs img60)
    FAIL threshold: -15% from baseline
    RECOVERY action: If drift > 12%, increase inter-image delay to 30s

Hailo utilization (per-image encoding spikes):
    Consistent 90-100% for ~20-30ms per image
    No thermal drift expected (Hailo runs cool at 2.5-8W)
    
Per-image latency drift:
    Image 1:  ~35s (baseline)
    Image 30: ~36s (+3% acceptable)
    Image 60: ~38s (+8% acceptable)
    FAIL: > 42s (+20%) at any image after img30
```

#### Thermal Profile (2-hour session projection)

```
Temperature (deg C)
105  |                                                   (throttle)
100  |                                              ___________
 95  |                                         ____/
 90  |                                    ____/
 85  |                               ____/
 80  |                          ____/
 75  |_________________________/
 70  |
     +----+----+----+----+----+----+----+----+----+----+----+----+
     0   10   20   30   40   50   60   70   80   90  100  120 min
     
     iGPU Junction Temp projection:
     - T+0 min:   55 C (ambient + idle)
     - T+10 min:  78 C (warming)
     - T+20 min:  88 C (approaching equilibrium)
     - T+30 min:  93 C (thermal equilibrium, fans at max)
     - T+60 min:  95 C (stable sustained)
     - T+90 min:  96 C (very slow creep)
     - T+120 min: 97 C (maximum sustained)
     
     Throttle threshold: 100 C (iGPU will downclock)
     Critical threshold: 105 C (emergency shutdown)
     
     Hailo-8L Temp projection:
     - T+0:  40 C
     - T+120: 62 C (very low thermal stress — 2.5W TDP)
```

---

### Target Metrics (Pass/Fail)

#### Session-Level Targets

| Metric | Target | Fail Threshold | Measurement |
|---|---|---|---|
| **Session completion** | 60/60 images | < 50 images completed | Job counter |
| **Session duration** | 2.0-2.5 hours | > 3.0 hours (too slow) or < 1.5 hours (too fast = not sustained) | Wall clock |
| **Per-image latency (image 1)** | < 35 s | > 42 s | Chrono per image |
| **Per-image latency (image 60)** | < 38 s (+8%) | > 42 s (+20%) | Chrono per image |
| **Latency drift (img1 vs img60)** | < +10% | > +20% | Percentage change |
| **GPU util img1** | 78-82% | < 65% | rsmi average over denoise |
| **GPU util img60** | 70-76% | < 58% | rsmi average over denoise |
| **Utilization drift** | -8% acceptable | -15% | Percentage change |
| **iGPU peak temperature** | < 97 C | > 100 C (throttle) | rsmi temperature |
| **Thermal throttle events** | 0 | > 2 | Clock frequency monitoring |
| **Watchdog recovery triggers** | < 3 per session | > 8 | Watchdog log |
| **VRAM post-defrag (img60)** | within 60 MB of img1 | > 100 MB growth | rsmi VRAM used |
| **Host RAM growth** | < 150 MB over session | > 300 MB | `getrusage` |

#### Style Consistency Targets

| Metric | Target | Fail Threshold | Method |
|---|---|---|---|
| CLIP style embedding cosine similarity (img N vs img 1) | > 0.90 | < 0.80 | CLIP image embedding comparison |
| Perceptual hash difference (img N vs img 1) | < 15% | > 25% | pHash Hamming distance |
| Color histogram KL divergence | < 0.15 | > 0.30 | HSV histogram comparison |
| LoRA weight checksum | Constant | Any change | SHA-256 of loaded LoRA buffer |

#### Long-Term Stability Targets

| Metric | Target | Fail Threshold |
|---|---|---|
| Zero OOM events | 0 OOM | Any OOM |
| Zero GPU session crashes | 0 crashes | Any session rebuild |
| Zero Hailo firmware hangs | 0 hangs | Any Hailo hard_reset |
| Image save success rate | 100% | < 98% |
| Disk space consumption | ~180 MB (60 x 3MB PNG) | > 500 MB |

---

### Risk Scenarios

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| **Thermal throttling after 45+ min** | High | Step time increases 15-25% | Increase inter-image delay dynamically; monitor Tj every image |
| **VRAM fragmentation after 30+ images** | Medium | Slow OOM or allocation failures | ORT allocator trim every 10 images; pre-allocate all buffers |
| **LoRA weight corruption** | Low | Style drift, inconsistent outputs | Checksum verification every 20 images; reload if mismatch |
| **Hailo firmware fatigue** | Low | Hailo hangs after 2+ hours continuous | Hailo hard_reset every 30 images (preventive) |
| **OS scheduler interference** | Medium | Background processes steal CPU cycles | Isolate to cores 0-5; `taskset` + `nice -20` |
| **Disk I/O bottleneck on save** | Low | 60 x 3MB PNG writes = 180MB total | Async IO thread; write to tmpfs then batch flush |
| **Watchdog false positive accumulation** | Medium | Recovery triggers increase over time | Adaptive thresholds: widen by 5% after img30 if no real faults |
| **Power supply sag under sustained load** | Low | System instability, random crashes | Monitor ACPI power; abort if voltage droop detected |
| **LPDDR5X thermal throttling** | Medium | Memory bandwidth reduction | Monitor memory controller temp; reduce BW if needed |

---

### Instrumentation Points

| Probe ID | Location | What | Frequency |
|---|---|---|---|
| `W4-SESSION-START` | Session init | Baseline metrics snapshot | Once |
| `W4-GPU-UTIL` | Watchdog | GPU busy % + temperature | 500 ms |
| `W4-HAILO-PWR` | Watchdog | Hailo power-proxy % + temp | 500 ms |
| `W4-IMAGE-T0` | Per-image start | Image generation start time | Per image |
| `W4-IMAGE-T1` | Per-image end | Image generation end time | Per image |
| `W4-IMAGE-DUR` | Per-image | Total latency + phase breakdown | Per image |
| `W4-GPU-CLOCK` | Watchdog | iGPU clock frequency (MHz) | 500 ms |
| `W4-GPU-VRAM` | Post-image | Peak VRAM for this image | Per image |
| `W4-DEFRAG-T` | Every 10 images | ORT allocator trim duration + VRAM delta | Every 10 images |
| `W4-STYLE-DRIFT` | Every 20 images | CLIP cosine similarity vs. img1 | Every 20 images |
| `W4-THROTTLE` | Watchdog | Clock frequency drop detection | Every sample |
| `W4-RECOVERY` | Recovery callback | Recovery trigger + outcome | Per event |
| `W4-SESSION-END` | Session teardown | Final report generation | Once |
| `W4-POWER-ACPI` | Background thread | System power consumption (W) | 5 s |



---

## Workload 5: Sustained Stress / Thermal Boundary Test <a name="workload-5"></a>

### Purpose

Tests the **absolute thermal and stability limits** of the UM790 Pro under maximum sustained inference load. This workload alternates rapidly between FLUX generation (GPU-heavy) and Qwen-Edit (GPU+CPU-heavy) with **minimal inter-job delay**, creating the worst-case thermal scenario. It runs until either (a) thermal throttling is detected, (b) a device fault occurs, (c) the watchdog triggers recovery 10+ times, or (d) 2 hours elapse — whichever comes first. This is the **burn-in validation workload** that proves the system can survive worst-case thermal and scheduling stress.

**Primary Stress Targets:** Thermal ceiling (iGPU Tj max, Hailo Tj max), sustained vs. peak utilization gap, power envelope sustainability, watchdog behavior under fault pressure, recovery chain stability (recovery triggering another recovery), code path switching overhead between FLUX and Qwen-Edit pipelines.

---

### Input Specification

#### Test Mode: Rapid-Fire Alternation

```
ALTERNATION PATTERN (continuous loop):
    Job 1:  FLUX generation (Workload 1, W1-D config: 1024x1024, 20 steps, 100 tokens)
            -> ~30s job duration
    Job 2:  Qwen-Edit (Workload 2, W2-D config: multi-mask, 30 steps, 350 tokens)
            -> ~46s job duration  
    Job 3:  FLUX generation (W1-E config: 1024x1024, 20 steps, 512 tokens)
            -> ~32s job duration
    Job 4:  Qwen-Edit (W2-B config: object replacement, 25 steps, 120 tokens)
            -> ~36s job duration
    REPEAT until termination condition
    
    Expected cycle time: ~144s per 4-job cycle
    Expected jobs in 2 hours: ~50 cycles = ~200 individual jobs
    Inter-job delay: 2 seconds (absolute minimum — just enough for pipeline teardown/setup)
```

#### Termination Conditions (first to trigger ends test)

| Condition | Detection Method | Action |
|---|---|---|
| **Thermal throttling** | iGPU clock < 90% of nominal for > 60s | Log + continue for 10 more jobs + terminate |
| **GPU watchdog recovery >= 10** | Recovery counter in watchdog log | Log + continue for 5 more jobs + terminate |
| **Hailo watchdog recovery >= 10** | Recovery counter in watchdog log | Log + continue for 5 more jobs + terminate |
| **GPU critical (0% util for 30s)** | `rsmi` returns 0 for 60 consecutive samples | Immediate terminate |
| **Hailo critical (device not present)** | `hailortcli scan` fails | Immediate terminate |
| **System power > 80W sustained** | ACPI power meter | Log + continue + terminate if > 90W |
| **CPU junction > 100 C** | k10temp sensor | Immediate terminate |
| **2 hours elapsed** | Wall clock | Normal termination |
| **200 jobs completed** | Job counter | Normal termination |

#### Model Loadout

| Model | Device | Format | Load Strategy |
|---|---|---|---|
| FLUX.1-dev DiT | Radeon 780M | ONNX bf16+int8 | **Keep loaded** (dominant GPU model) |
| Qwen MMDiT | Radeon 780M | ONNX bf16+int8 | **Swap in/out** per Qwen-Edit job |
| T5-XXL Encoder | Hailo-8L | HEF | Keep loaded (shared between both pipelines) |
| CLIP-L Encoder | Hailo-8L | HEF | Keep loaded (shared) |
| Qwen Text Encoder | Hailo-8L | HEF | **Swap in/out** per Qwen-Edit job |
| Qwen Vision Tower | CPU | ONNX bf16 | **Swap in/out** per Qwen-Edit job |
| VAE (FLUX) | CPU | ONNX fp16 | Keep loaded |
| VAE (SD3) | CPU | ONNX fp16 | **Swap in/out** per Qwen-Edit job |

#### Pipeline Switching Overhead Budget

```
FLUX -> Qwen-Edit switch:
    [GPU] Unload FLUX DiT:     ~50ms (ORT session release)
    [GPU] Load Qwen MMDiT:     ~800ms (3,200 MB model from disk/cache)
    [Hailo] Context to Qwen encoder: ~5ms
    [CPU] Load Qwen vision tower: ~200ms
    [CPU] Load SD3 VAE: ~100ms
    TOTAL SWITCH: ~1,155ms (target: < 1,500ms)

Qwen-Edit -> FLUX switch:
    [GPU] Unload Qwen MMDiT:   ~50ms
    [GPU] Load FLUX DiT:       ~600ms (2,850 MB from cache — likely still warm)
    [Hailo] Context to T5+CLIP: ~5ms
    [CPU] Unload Qwen vision: ~50ms
    [CPU] Unload SD3 VAE: ~50ms
    TOTAL SWITCH: ~755ms (target: < 1,000ms)

AVERAGE SWITCH: ~955ms
With 2s inter-job delay: 1,045ms buffer for unexpected overhead
```

---

### Pipeline Flow

```
INITIAL LOAD (all models into memory/cache):
    [GPU] Load FLUX DiT -> VRAM (2,850 MB)
    [Hailo] Load T5-XXL + CLIP-L HEFs
    [CPU] Load both VAEs into RAM
    [CPU] Cache Qwen MMDiT ONNX to NVMe read buffer (3,200 MB file)
    [CPU] Cache Qwen vision tower (890 MB file)
    [Watchdog] Start with AGGRESSIVE thresholds:
               - Sample interval: 250ms (2x normal)
               - Consecutive threshold: 6 steps (vs 8 normal)
               - Critical threshold: 45% (vs 40% normal)

MAIN STRESS LOOP (until termination):
    job_count = 0
    WHILE NOT terminated:
        
        job_count += 1
        pipeline = (job_count % 4)  // Alternates FLUX/Qwen/FLUX/Qwen
        
        // ==== FLUX JOB (job_count odd: 1, 3, 5, ...) ====
        IF pipeline == 1 OR pipeline == 3:
            [GPU] Ensure FLUX DiT loaded (if switch: ~600ms load)
            [Hailo] Ensure T5+CLIP context active
            RUN Workload 1 pipeline (20 steps, 1024x1024)
            [CPU] Log: job_count, pipeline_type, latency, peak_temp, peak_util
        
        // ==== QWEN-EDIT JOB (job_count even: 2, 4, 6, ...) ====
        ELSE:
            [GPU] Switch: unload FLUX, load Qwen MMDiT (~1,155ms)
            [Hailo] Switch to Qwen encoder context
            [CPU] Load Qwen vision tower + SD3 VAE
            RUN Workload 2 pipeline (25-30 steps, 1024x1024, multi-mask)
            [CPU] Log: job_count, pipeline_type, latency, peak_temp, peak_util
            [GPU] Switch: unload Qwen, load FLUX (~755ms)
        
        // ==== INTER-JOB ====
        [CPU] 2-second mandatory delay (thermal breathing room)
        [CPU] Check all termination conditions
        [Watchdog] Report: recovery count this job, max temp, min util
        
        // ==== PER-CYCLE CHECK (every 4 jobs) ====
        IF job_count % 4 == 0:
            [CPU] Full thermal snapshot: GPU temp, Hailo temp, CPU temp, power
            [CPU] Full memory snapshot: GPU VRAM, host RAM
            [CPU] Compute rolling averages (last 4, 8, 16 jobs)
            [CPU] If rolling avg latency increases > 15%: ALERT
            [CPU] If recovery count in last 16 jobs > 5: CRITICAL ALERT
    
    END WHILE

TEARDOWN:
    [CPU] Generate stress test report
    [CPU] Archive logs to ./stress_test_output/
    [Watchdog] Shutdown
```

#### Timing Diagram (4-Job Cycle, ~144s)

```
Time (s)   0     30    32    33   78    80    81   112  114  115  144
            |      |     |     |     |     |     |     |     |     |
Job:       [J1: FLUX       ] [d][J2: Qwen-Edit       ] [d][J3: FLUX   ] [d][J4: Qwen]
GPU:       [====FLUX DiT===] [sw][====Qwen MMDiT=====] [sw][===FLUX====] [sw][=Qwen=]
Hailo:     [T5][CLIP]       [ctx][QwenEnc][T5][CLIP]  [ctx][T5][CLIP]  [ctx][QwenEnc]
CPU:       [VAE][log]       [sw][vision][VAE][log]    [sw][VAE][log]   [sw][vision][V]
Temp:      rising ~~~~~~~~~~~~~ peak ~~~~~~~~~~~~~~~~ falling ~~~~~~~~ rising ~~~~~~~~~

Legend:
    [====] = active inference
    [d]    = 2s inter-job delay
    [sw]   = model switch (~1s)
    [ctx]  = Hailo context switch (~5ms)
    [T5]   = T5-XXL encoding
    [CLIP] = CLIP-L encoding
    [VAE]  = VAE decode
    [log]  = logging + metrics
    [vision]= Qwen vision tower
    ~~~~~~ = thermal trend
```

---

### Memory Footprint

#### Peak Concurrent Memory (worst case: mid-switch Qwen->FLUX)

```
Radeon 780M VRAM:
    FLUX DiT loaded:           2,850 MB
    Qwen MMDiT loading (overlap): +800 MB (partial)
    LoRA/adapter scratch:         50 MB
    Latent buffers:               17 MB
    Activation scratch:          580 MB
    TOTAL PEAK:                 ~4,297 MB
    
    WARNING: This exceeds 4,096 MB budget by ~200 MB!
    MITIGATION: Ensure complete unload of Qwen before FLUX load begins.
    Sequential (not overlapping) switch keeps peak at ~3,200 MB.
    
STEADY-STATE VRAM:
    During FLUX job:     2,850 + 580 + 17 = 3,447 MB
    During Qwen job:     3,200 + 680 + 26 = 3,906 MB (very tight!)

Hailo-8L SRAM:
    During FLUX job:     1,200 MB (T5 + CLIP)
    During Qwen job:     1,500 MB (Qwen + T5 + CLIP dual-HEF)
    Context switch:      5ms unload/load
    
Host RAM:
    Both VAEs loaded:    ~940 MB
    Vision tower loaded: ~890 MB (only during Qwen)
    ORT arenas:          ~300 MB
    OS + buffers:        ~2,000 MB
    TOTAL PEAK:          ~4,130 MB
```

#### Thermal Impact on Memory

```
LPDDR5X operates at rated speed up to 85 C die temperature.
Above 85 C: memory controller may reduce frequency (BW drop ~10%).
Above 95 C: emergency downclock (BW drop ~25%).

Expected memory temp trajectory:
    0-30 min:   45-65 C (safe)
    30-60 min:  65-78 C (safe)
    60-90 min:  78-85 C (approaching limit)
    90-120 min: 85-90 C (possible BW reduction)

MONITOR: Memory bandwidth via `perf` or `mbw` every 10 minutes.
FAIL if measured BW drops > 10% from baseline.
```

---

### Expected Utilization Profile

#### Time-Series (4-Job Cycle, ~144s)

```
GPU Busy %
100% |                              ____
 85% |     ____________            /    \           __________
 75% |    /            \          /      \         /          \
 60% |   /              \________/        \_______/            \_____
 40% |  /  (VAE+switch)                                                   (VAE+sw)
 20% |
  0% +----+----+----+----+----+----+----+----+----+----+----+----+----+----+
     0    5   10   15   20   25   30   35   40   45   50   55   60  (x2.4)
     [====FLUX: 75% avg===][sw][==Qwen: 72% avg===][sw][=FLUX=][sw][=Qwen=]

Hailo NN Core %
100% |==[enc]==        ===[enc]==      ==[enc]==        ===[enc]==
 90% |         (idle)            (idle)        (idle)           (idle)
 60% |
  0% +----+----+----+----+----+----+----+----+----+----+----+----+----+----+
     0    5   10   15   20   25   30   35   40   45   50   55   60  (x2.4)

System Power (W) — from ACPI
 80W |
 70W |     ___________           ___________          ___________
 60W |    /           \         /           \        /           \
 50W |___/             \_______/             \______/             \_____
 40W |
 30W |
     +----+----+----+----+----+----+----+----+----+----+----+----+----+----+
     0    5   10   15   20   25   30   35   40   45   50   55   60  (x2.4)
     [~~~FLUX job~~~~][dly][~~~Qwen job~~~~][dly][~FLUX~][dly][~Qwen~]
     
     Power envelope:
     - FLUX denoising:   ~55W (GPU + memory intensive)
     - Qwen denoising:   ~62W (GPU + CPU both intensive)
     - Switch + delay:   ~38W (idle)
     - Average per cycle: ~52W
     - 2-hour total energy: ~52W x 2h = ~104 Wh
     - UM790 Pro power adapter: 120W (sufficient headroom)
```

#### Sustained vs. Peak Gap Analysis

```
Utilization metrics across 200 jobs (2 hours):

                    Peak (best job)  Sustained (rolling avg)  Gap
GPU utilization:    82%              74%                      -8%
Hailo utilization:  100%             95% (during encode)      -5%
Per-job latency:    28s (FLUX best)  34s (rolling avg)        +21%
Peak temperature:   97 C (iGPU)      94 C (rolling avg)       -3 C
Recovery events:    0 (best window)  0.8 per 16-job window    variable

FAIL if sustained GPU util < 65% (indicates thermal or scheduling degradation)
FAIL if latency gap > 35% (indicates cumulative thermal throttling)
```

---

### Target Metrics (Pass/Fail)

#### Stress Test Pass Criteria

| Metric | Target | Fail Threshold | Measurement |
|---|---|---|---|
| **Jobs completed** | 200 | < 150 | Job counter |
| **Test duration** | 2 hours | < 90 min (early termination) | Wall clock |
| **GPU recovery events** | < 5 | > 10 | Watchdog log |
| **Hailo recovery events** | < 3 | > 10 | Watchdog log |
| **GPU sustained utilization** | 72-78% | < 60% | rsmi rolling 16-job avg |
| **iGPU peak temperature** | < 97 C | > 100 C (throttle line) | rsmi |
| **iGPU throttle events** | 0 | > 2 (clock drop > 10% for > 60s) | rsmi clock monitor |
| **Hailo peak temperature** | < 75 C | > 85 C | HailoRT temp API |
| **System peak power** | < 75W | > 85W sustained | ACPI power meter |
| **Per-job latency drift (job 1 vs job 100 vs job 200)** | < +20% | > +35% | Chrono per job |
| **Model switch latency** | < 1,500ms | > 2,500ms | Chrono switch time |
| **Memory BW degradation** | < 5% from baseline | > 10% | `mbw` benchmark every 10 min |
| **Zero fatal device faults** | 0 fatal | Any fatal (device offline) | Watchdog health flag |
| **Zero OOM events** | 0 | Any OOM | System log |

#### Rolling Window Targets

| Window Size | Metric | Target | Fail Threshold |
|---|---|---|---|
| 4 jobs (1 cycle) | Avg latency | within 15% of baseline | > 25% |
| 8 jobs (2 cycles) | Recovery count | < 2 | > 4 |
| 16 jobs (4 cycles) | Utilization | > 68% | < 58% |
| 32 jobs (8 cycles) | Temperature | stable (delta < 3 C) | rising trend > 5 C |
| 64 jobs (16 cycles) | Latency drift | < +15% from job 1 | > +30% |

---

### Risk Scenarios

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| **iGPU thermal throttle within 30 min** | Very High | 20-40% latency increase | Dynamic clock management; reduce to 18 steps if throttle detected |
| **Qwen MMDiT OOM during load** | High | 3,200MB model load into ~4GB VRAM with fragmentation | Pre-clean VRAM before switch; `hipDeviceSynchronize` + ORT arena clear |
| **Hailo context switch failure** | Medium | Hailo firmware hang | Full Hailo reset + HEF reload on context switch failure |
| **Model switch latency > 2s** | Medium | Breaches inter-job delay budget | Pre-load models to NVMe cache; use `mmap` for fast loading |
| **Watchdog aggressive mode false positive** | High | Shortened consecutive threshold (6 vs 8) triggers more recoveries | Accept higher recovery count; validate each recovery succeeds |
| **Recovery cascade** (one recovery triggers another) | Medium | System enters unstable oscillation | Circuit breaker: if 3 recoveries within 5 jobs, force 60s cooldown |
| **Power supply overload** | Low | 120W adapter cannot sustain 80W+ CPU+GPU+Hailo | Monitor ACPI; throttle if power > 85W for > 5 min |
| **OS process scheduler starvation** | Medium | Background kernel threads interfere | `isolcpus` for inference threads; `SCHED_FIFO` for watchdog |
| **LPDDR5X thermal derating** | Medium | Memory BW drops 10-15% at 85C+ | Reduce batch prefetch depth; tolerate longer step times |
| **FLUX/Qwen code path divergence causing state corruption** | Low | Shared ORT session state gets corrupted | Separate session pools for each pipeline; full state isolation |

---

### Instrumentation Points

| Probe ID | Location | What | Frequency |
|---|---|---|---|
| `W5-GPU-UTIL` | Watchdog (aggressive) | GPU busy % | **250 ms** (2x normal) |
| `W5-GPU-TEMP` | Watchdog | iGPU junction temperature | 250 ms |
| `W5-GPU-CLOCK` | Watchdog | iGPU clock frequency (MHz) | 250 ms |
| `W5-HAILO-PWR` | Watchdog | Hailo power-proxy % | 250 ms |
| `W5-HAILO-TEMP`| Watchdog | Hailo chip temperature | 250 ms |
| `W5-JOB-T0` | Job start | Job start timestamp + pipeline type | Per job |
| `W5-JOB-T1` | Job end | Job end timestamp + total latency | Per job |
| `W5-JOB-BREAKDOWN` | Per-phase | Encode / denoise / decode / switch latency | Per job |
| `W5-SWITCH-T` | Model switch | Switch duration per model | Per switch |
| `W5-RECOVERY` | Recovery callback | Recovery trigger + device + step | Per event |
| `W5-POWER` | Background thread | System power (W) from ACPI | 1 s |
| `W5-CPU-TEMP` | Background thread | CPU junction temperature | 1 s |
| `W5-MEM-BW` | Background thread | Memory bandwidth benchmark | Every 10 min |
| `W5-VRAM` | Per-job | GPU VRAM used at job end | Per job |
| `W5-HOSTMEM` | Per-job | Host RSS at job end | Per job |
| `W5-TERMINATION` | Termination check | Which termination condition triggered | Once |
| `W5-ROLLING-4` | Every 4 jobs | 4-job rolling avg latency + util | Every cycle |
| `W5-ROLLING-16`| Every 16 jobs | 16-job rolling avg + recovery count | Every 4 cycles |



---

## Cross-Workload Summary & Scheduling Matrix <a name="cross-workload-summary"></a>

### Summary Comparison

| Dimension | W1: FLUX Generation | W2: Qwen-Edit | W3: VLM->Diffusion | W4: Style Session | W5: Thermal Stress |
|---|---|---|---|---|---|
| **Primary Device** | GPU (denoising) | GPU (MMDiT) | Hailo then GPU | GPU (sustained) | GPU + Hailo (alternating) |
| **Secondary Device** | Hailo (encoding) | Hailo + CPU | Hailo (VLM) + CPU | Hailo (encoding) | Hailo (encoding + switch) |
| **Dominant Phase** | Denoising 20-50 steps | Denoising 20-40 steps | VLM gen + Denoising | Denoising x60 images | Rapid alternation |
| **Peak GPU VRAM** | 3,550 MB | 4,070 MB | 3,440 MB | 3,540 MB | 3,906 MB (Qwen) |
| **VRAM % of Budget** | 86.7% | **99.4%** | 84.0% | 86.4% | **95.4%** |
| **Hailo Active Time** | ~25ms per job | ~20-55ms per job | ~5,000ms per job | ~25ms x 60 = 1.5s total | ~25ms x 200 = 5s total |
| **Job Duration** | 6-105s | 28-100s | 31-49s | ~35s x 60 images | ~30-46s x 200 jobs |
| **Total Test Time** | ~10 min (all configs) | ~15 min (all configs) | ~15 min (all configs) | **2-2.5 hours** | **2 hours** |
| **Concurrent Devices** | No (sequential) | No (sequential) | **Yes (100ms overlap)** | No (sequential) | **Alternating** |
| **Thermal Stress** | Low-Medium | Medium | Low-Medium | **High (sustained)** | **Very High (cyclic)** |
| **Memory Stress** | High (OOM risk at 2048) | **Very High (OOM risk)** | Medium | High (fragmentation) | **Very High (switching)** |
| **Watchdog Aggressiveness** | Normal | Normal | Normal | Normal | **Aggressive (250ms, 6-step)** |
| **Recovery Tolerance** | 0-2 per config | 0-2 per config | 0-2 per config | < 3 per session | **< 5 per 16-job window** |

### Stress Coverage Matrix

```
                    W1   W2   W3   W4   W5
                    ---  ---  ---  ---  ---
GPU Compute         ***  ***  **   ***  ***
GPU VRAM            **   ***  **   **   ***
GPU Thermal         *    **   *    ***  ****
Hailo Compute       **   **   ***  **   **
Hailo Throughput    **   **   ***  *    **
Hailo Thermal       *    *    *    *    *
CPU Scheduler       **   ***  **   *    **
CPU VAE Decode      *    *    *    *    *
Memory Bandwidth    *    **   ***  *    ***
PCIe DMA            *    *    **   *    *
ORT Session Mgmt    *    **   *    ***  ****
LoRA/Adapters       —    —    —    ***  —
Watchdog Correctness **   **   **   ***  ****
Recovery Logic      *    *    **   **   ****
Pipeline Handoff    —    —    ***  —    ***
Style Consistency   —    —    —    ***  —

Legend:  * = light coverage    ** = moderate    *** = heavy    **** = maximum    — = not tested
```

### Recommended Execution Order

```
DAY 1 — Functional Validation:
    1. Workload 1 (W1-A through W1-E):  ~30 min
       -> Validates core pipeline: encode -> denoise -> decode
       -> Must PASS before any other workload
    
    2. Workload 2 (W2-A, W2-D):         ~15 min
       -> Validates Qwen-Edit with masks
       -> Tests OOM boundary with MMDiT
    
    3. Workload 3 (W3-A, W3-C):         ~15 min
       -> Validates dual-accelerator pipeline
       -> Tests handoff latency

DAY 2 — Sustained Validation:
    4. Workload 4:                        ~2.5 hours
       -> Validates 2-hour thermal stability
       -> Tests memory fragmentation and drift
    
    5. Workload 5:                        ~2 hours
       -> Validates absolute thermal limits
       -> Tests recovery system under stress

TOTAL VALIDATION TIME: ~5.5 hours per platform revision
```

### Resource Conflict Analysis

```
MODEL STORAGE REQUIREMENTS (all workloads):
    GPU models:
        FLUX.1-dev (quantized):           ~2,850 MB ONNX
        Qwen MMDiT (quantized):           ~3,200 MB ONNX
        Subtotal GPU:                      ~6,050 MB (loaded sequentially)
    
    Hailo HEFs:
        T5-XXL:                            ~1,200 MB
        CLIP-L:                            ~150 MB
        Qwen text encoder:                 ~800 MB
        Subtotal Hailo:                    ~2,150 MB (context-switched)
    
    CPU models:
        Qwen vision tower:                 ~890 MB
        FLUX VAE:                          ~470 MB
        SD3 VAE:                           ~470 MB
        Subtotal CPU:                      ~1,830 MB
    
    TOTAL ONNX/HEF STORAGE:                ~10,030 MB (~10 GB)
    Required NVMe space:                   ~15 GB (with caches + output)

VRAM CONFLICT TABLE (which models coexist in GPU memory):
    W1 (FLUX):       FLUX DiT only (2,850 MB) -> SAFE
    W2 (Qwen-Edit):  Qwen MMDiT only (3,200 MB) -> TIGHT but fits
    W3 (VLM+FLUX):   FLUX DiT only (2,850 MB) -> SAFE (VLM on CPU+Hailo)
    W4 (Style):      FLUX DiT + LoRA (2,955 MB) -> SAFE
    W5 (Stress):     FLUX OR Qwen at any moment -> SAFE if sequential load
                     Peak during switch: ~3,200 MB (if no overlap) -> FITS
```

---

## Instrumentation Probe Reference <a name="instrumentation-reference"></a>

### Unified Probe Schema

All probes emit structured JSON records to a shared telemetry sink:

```json
{
    "timestamp": "2025-06-10T14:32:15.123456Z",
    "workload": "W1",
    "config": "W1-D",
    "job_id": 1,
    "probe_id": "W1-STEP-T0",
    "device": "GPU_780M",
    "value": 0.720,
    "unit": "seconds",
    "metadata": {
        "step": 5,
        "resolution": "1024x1024",
        "model": "FLUX.1-dev"
    }
}
```

### Probe Registry

| Probe ID | Workloads | Device | Type | Value | Unit |
|---|---|---|---|---|---|
| `*-GPU-UTIL` | W1-W5 | Radeon 780M | Gauge | gpu_busy_percent | % |
| `*-GPU-VRAM` | W1-W5 | Radeon 780M | Gauge | VRAM used | bytes |
| `*-GPU-TEMP` | W1-W5 | Radeon 780M | Gauge | Junction temperature | deg C |
| `*-GPU-CLOCK` | W4-W5 | Radeon 780M | Gauge | Clock frequency | MHz |
| `*-HAILO-PWR` | W1-W5 | Hailo-8L | Gauge | Power-proxy utilization | % |
| `*-HAILO-TEMP`| W1-W5 | Hailo-8L | Gauge | Chip temperature | deg C |
| `*-STEP-T*` | W1-W2 | Radeon 780M | Timer | Per-step latency | seconds |
| `*-STEP-DUR` | W1-W2 | Radeon 780M | Timer | Step duration | seconds |
| `*-SCHED-T` | W1-W2 | CPU | Timer | Scheduler math latency | seconds |
| `*-ENC-T*` | W1-W2 | Hailo-8L | Timer | Encoding latency | seconds |
| `*-VAE-T*` | W1-W5 | CPU | Timer | VAE decode latency | seconds |
| `*-RECOVERY` | W1-W5 | System | Counter | Recovery event | count + enum |
| `W3-VLM-*` | W3 | Hailo+CPU | Timer | VLM phase metrics | mixed |
| `W3-HANDOFF-*`| W3 | System | Timer | Accelerator handoff | seconds |
| `W3-BW-SPIKE` | W3 | System | Gauge | Memory BW at handoff | % of max |
| `W4-STYLE-DRIFT` | W4 | CPU | Gauge | CLIP style similarity | cosine |
| `W4-DEFRAG-*` | W4 | GPU | Timer | Allocator trim metrics | seconds + bytes |
| `W4-THROTTLE` | W4-W5 | GPU | Counter | Throttle event | count + duration |
| `W5-SWITCH-T` | W5 | GPU | Timer | Model switch latency | seconds |
| `W5-POWER` | W5 | System | Gauge | System power draw | watts |
| `W5-MEM-BW` | W5 | System | Gauge | Memory bandwidth | GB/s |
| `W5-ROLLING-*` | W5 | System | Gauge | Rolling window averages | mixed |

### Telemetry Sink Configuration

```
OUTPUT: ./telemetry/workloads_{timestamp}.jsonl
        (JSON Lines format — one JSON object per line, append-only)

ROTATION: Rotate every 100 MB or every 30 minutes, whichever comes first.
          Compressed with zstd after rotation.

FIELDS (every record):
    - timestamp (ISO 8601 with microseconds)
    - workload (W1-W5)
    - config (config ID within workload)
    - job_id (monotonically increasing across session)
    - probe_id (from registry above)
    - device (GPU_780M, HAILO_8L, CPU, SYSTEM)
    - value (numeric)
    - unit (% | seconds | bytes | deg_C | MHz | watts | GB/s | count)
    - metadata (JSON object, probe-specific)

EXAMPLE OUTPUT:
    {"timestamp":"2025-06-10T14:32:15.123456Z","workload":"W1","config":"W1-D","job_id":1,"probe_id":"W1-GPU-UTIL","device":"GPU_780M","value":78.5,"unit":"%","metadata":{"step":5,"rsmi_sample_ms":500}}
    {"timestamp":"2025-06-10T14:32:15.123456Z","workload":"W1","config":"W1-D","job_id":1,"probe_id":"W1-HAILO-PWR","device":"HAILO_8L","value":0.0,"unit":"%","metadata":{"phase":"denoising"}}
    {"timestamp":"2025-06-10T14:32:15.623456Z","workload":"W1","config":"W1-D","job_id":1,"probe_id":"W1-STEP-T1","device":"GPU_780M","value":0.718,"unit":"seconds","metadata":{"step":5,"model":"FLUX.1-dev","resolution":"1024x1024"}}
```

### Analysis Scripts

Recommended post-processing:

```python
# tools/analyze_workload_telemetry.py
"""
Consumes JSONL telemetry and produces:
  1. Per-workload summary (latency P50, P95, P99)
  2. Utilization time-series plots (GPU + Hailo)
  3. Thermal trajectory overlay
  4. Recovery event timeline
  5. Pass/fail verdict per target metric
  6. Cross-workload regression comparison
"""

import json
import pandas as pd

def load_telemetry(path: str) -> pd.DataFrame:
    """Load JSONL telemetry into DataFrame."""
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)

def compute_latency_summary(df: pd.DataFrame, workload: str) -> dict:
    """Compute P50/P95/P99 for per-step and per-job latency."""
    step_times = df[(df.workload == workload) & (df.probe_id.str.contains("STEP"))]
    return {
        "p50": step_times.value.quantile(0.50),
        "p95": step_times.value.quantile(0.95),
        "p99": step_times.value.quantile(0.99),
        "mean": step_times.value.mean(),
        "std": step_times.value.std(),
        "cv": step_times.value.std() / step_times.value.mean(),
    }

def check_utilization_target(df, workload, device, target_low, target_high):
    """Check if sustained utilization stays within target band."""
    util = df[(df.workload == workload) & (df.device == device) & 
              (df.probe_id.str.contains("UTIL|PWR"))]
    sustained = util.value.mean()
    below_threshold = (util.value < 60).sum()
    return {
        "sustained_pct": sustained,
        "target_band": f"{target_low}-{target_high}%",
        "samples_below_60": below_threshold,
        "verdict": "PASS" if target_low <= sustained <= target_high 
                   and below_threshold < 8 else "FAIL"
    }

def generate_report(telemetry_path: str, output_path: str):
    df = load_telemetry(telemetry_path)
    report = {}
    
    for wl in ["W1", "W2", "W3", "W4", "W5"]:
        report[wl] = {
            "latency": compute_latency_summary(df, wl),
            "gpu_util": check_utilization_target(df, wl, "GPU_780M", 70, 85),
            "hailo_util": check_utilization_target(df, wl, "HAILO_8L", 85, 100),
            "recoveries": len(df[(df.workload == wl) & (df.probe_id == "*-RECOVERY")]),
        }
    
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
```

---

## Appendix A: Memory Footprint Calculator

Use this formula to compute expected GPU VRAM for any configuration:

```
GPU_VRAM (MB) = model_weights + attention_kv_cache + activation_scratch 
                + latent_buffers + lora_adapters + ort_overhead

Where:
    model_weights (FLUX DiT) = ~2,850 MB (int8 MLP + bf16 attention)
    model_weights (Qwen MMDiT) = ~3,200 MB (int8 MLP + bf16 attention)
    
    attention_kv_cache = seq_len x hidden_dim x num_layers x 2 bytes / 1e6
        FLUX:  4,685 x 3,072 x 19 x 2 / 1e6 = ~547 MB (with gradient checkpointing: ~137 MB)
        Qwen:  4,685 x 3,584 x 24 x 2 / 1e6 = ~805 MB (with gradient checkpointing: ~201 MB)
    
    activation_scratch = ~80 MB (per-layer, reused)
    
    latent_buffers = batch x 16 x (W/8) x (H/8) x 2 bytes / 1e6
        1024x1024, bs=1: 1 x 16 x 128 x 128 x 2 / 1e6 = 0.52 MB
        2048x2048, bs=1: 1 x 16 x 256 x 256 x 2 / 1e6 = 2.1 MB
        1024x1024, bs=2: 2 x 16 x 128 x 128 x 2 / 1e6 = 1.05 MB
    
    lora_adapters = rank x (num_params / rank_divisor) x 2 bytes / 1e6
        rank-16 FLUX LoRA: ~105 MB
    
    ort_overhead = ~150 MB (ONNX Runtime allocator arenas + session state)
```

## Appendix B: Utilization Quick Reference

| Scenario | GPU Expected | Hailo Expected | Action if Below |
|---|---|---|---|
| FLUX denoising (step 5+) | 75-80% avg | 0% (idle) | Check scheduler prefetch; check CPU math time |
| Qwen-Edit denoising | 70-78% avg | 0% (idle) | Check mask compositing CPU time; check MMDiT size |
| Hailo encoding (T5 512 tokens) | 0-5% (idle) | 90-100% | Check PCIe DMA queue depth; check HEF integrity |
| Hailo encoding (Qwen VLM decode) | 0-5% (idle) | 90-100% | Check autoregressive token loop overhead |
| VLM->FLUX handoff | 60% ramping | 90% -> 0% | Check DMA completion signal; check GPU kernel launch |
| Model switch (W5) | 0% during load | Context switching | Check NVMe read speed; check ORT session cleanup |
| Idle between jobs | 0-5% | 0% | Expected; > 20% indicates GPU workload leak |

## Appendix C: Recovery Decision Tree

```
WATCHDOG SAMPLE (every 500ms):
    |
    +-- Device healthy?
    |       |
    |       +-- NO -> IMMEDIATE RECOVERY
    |       |           GPU: soft_reset -> hipDeviceReset -> rebuild session
    |       |           Hailo: hard_reset -> reload HEF -> rebuild session
    |       |           Log: CRITICAL + device fault reason
    |       |
    |       +-- YES -> Check utilization
    |                   |
    |                   +-- util < 40% (CRITICAL)
    |                   |       |
    |                   |       +-- GPU: IMMEDIATE RECOVERY
    |                   |       +-- Hailo: IMMEDIATE RECOVERY + CPU fallback
    |                   |       Log: CRITICAL + util value
    |                   |
    |                   +-- 40% <= util < 60% (WARNING)
    |                   |       |
    |                   |       +-- Increment consecutive counter
    |                   |       +-- If counter >= 8: TRIGGER RECOVERY
    |                   |       +-- Log: WARN + counter progress
    |                   |
    |                   +-- util >= 60% (NORMAL)
    |                           |
    |                           +-- If was recovering: Log INFO "recovered"
    |                           +-- Reset consecutive counter to 0
    |                           +-- If 75% <= util <= 80%: Log DEBUG "on target"
    |
    +-- Check thermal:
            GPU > 100 C: Log WARNING "thermal throttling likely"
            Hailo > 85 C: Log WARNING "Hailo overheating"
```

## Appendix D: Hardware Monitoring Cheat Sheet

```bash
# Quick hardware status checks during test execution

# GPU utilization (continuous)
watch -n 0.5 rocm-smi --showbus --showtemp --showclk --showuse

# GPU memory
rocm-smi --showmeminfo vram

# Hailo device status
hailortcli scan
hailortcli monitor --duration 60

# Hailo power measurement
hailortcli fw-control read-power

# System power (UM790 Pro)
cat /sys/class/power_supply/BAT0/power_now 2>/dev/null || \
cat /sys/class/hwmon/hwmon*/power1_input 2>/dev/null

# CPU temperature
sensors k10temp-pci-*

# Memory bandwidth (quick test)
mbw 1024  # 1 GB memory bandwidth test

# PCIe link status for Hailo
lspci -vv -s 01:00.0 | grep LnkSta
# Expected: LnkSta: Speed 8 GT/s, Width x2 (PCIe Gen3 x2)

# Check for thermal throttling in kernel log
dmesg | grep -i "throttl\|thermal"

# Process isolation
taskset -cp 0-7 $PID  # Pin inference to cores 0-7
chrt -f -p 99 $PID    # SCHED_FIFO highest priority

# Watchdog log tail
tail -f ./logs/watchdog.log
```

---

*End of Document — 5 Production-Grade Test Workloads for UM790 Pro + Hailo-8L*
