# UM790 Pro Heterogeneous Inference Pipeline — Quantitative Metrics & KPI Framework
## Battle-Testing Framework for Radeon 780M + Hailo-8L Co-processing

**Version:** 1.0  
**Hardware Target:** Minisforum UM790 Pro (Ryzen 9 7940HS, Radeon 780M gfx1103, Hailo-8L PCIe Gen3 x2)  
**Memory:** 64 GB LPDDR5X-7500 (Unified, HSA_XNACK=1)  
**System TDP:** 65W (7940HS configured)  
**Total Active Compute:** ~51 TOPS  

---

## Document Conventions

| Symbol | Meaning |
|---|---|
| **Target** | The design-center goal; achieving this = "excellent" |
| **Acceptable** | Pass/fail boundary; below this = pipeline needs tuning |
| **Critical** | Hard stop; exceeding this triggers immediate intervention |
| **σ** | Standard deviation of a sample distribution |
| **μ** | Arithmetic mean |
| **P99** | 99th percentile (worst 1% of observations) |
| **P95** | 95th percentile |

**Measurement Time Bases:**
- **Per-step:** Every denoising step (typically 4–50 steps per job)
- **Per-job:** One complete end-to-end inference (encode → denoise → decode)
- **Per-session:** A 30-minute sustained stress run (≥100 consecutive jobs)
- **Continuous:** Background sampling at watchdog frequency (500 ms)

---

## 1. Utilization Metrics (Primary KPIs)

### 1.1 Radeon 780M — GPU Busy Percent

The 780M runs FLUX DiT / Qwen MMDiT denoising. Because attention heads execute sequentially within a step, utilization shows **burst-sawtooth** behavior: spikes to ~95% during GEMM/conv kernels, valleys to ~40% during attention reduction and scheduler handoff. The scheduler's job is to compress the valleys.

| # | Metric | Measurement Method | Formula | Target | Acceptable Range | Critical | Frequency |
|---|---|---|---|---|---|---|---|
| U-G1 | **Sustained Average Utilization** | `rsmi_dev_gpu_busy_percent_get()` sampled every 500 ms; arithmetic mean over job duration | μ = Σ(utilᵢ) / N, where N = job_duration_ms / 500 | **75–80%** | ≥ 65% | < 60% | Per-job |
| U-G2 | **Per-Step Utilization Variance** | Compute σ over all 500 ms samples within a single denoising step | σ = √(Σ(utilᵢ − μ_step)² / (N−1)) | **σ < 12%** | σ < 15% | σ ≥ 20% | Per-step |
| U-G3 | **Valley Frequency** | Count of 500 ms samples where util < 60%; divide by total samples | VF = 100 × |{utilᵢ < 60}| / N | **< 2%** | < 5% | ≥ 10% | Per-job |
| U-G4 | **Burst-to-Sustained Ratio** | Ratio of peak step-utilization to job-mean utilization | BSR = P95_step(util) / μ_job(util) | **1.15–1.25** | 1.10–1.35 | > 1.50 or < 1.05 | Per-job |
| U-G5 | **Idle Gap Duration** | Maximum contiguous span of samples below 40% | max_gap = max duration of consecutive utilᵢ < 40% | **< 1.0 s** | < 2.0 s | ≥ 4.0 s | Per-job |
| U-G6 | **Inter-Step Dead Time** | Wall-clock time between `hipLaunchKernel` end of step N and start of step N+1 | measured via `hipEventElapsedTime` around step boundaries | **< 3 ms** | < 8 ms | ≥ 20 ms | Per-step |

**Measurement API Details:**

```cpp
// U-G1: Sustained Average — accumulate over job
struct JobUtilSummary {
    float mean_util;           // μ over all 500ms samples
    float std_dev;             // σ (U-G2)
    float valley_pct;          // % samples < 60% (U-G3)
    float burst_sustained_ratio; // U-G4
    float max_idle_gap_ms;     // U-G5
};

// Sampling loop (called from watchdog thread)
std::vector<float> gpu_samples;  // cleared at job start
void on_gpu_sample(float util) {
    gpu_samples.push_back(util);
}

JobUtilSummary summarize_gpu_job() {
    JobUtilSummary s{};
    s.mean_util = mean(gpu_samples);
    s.std_dev = std_dev(gpu_samples);
    s.valley_pct = 100.0f * count_below(gpu_samples, 60.0f) / gpu_samples.size();
    s.burst_sustained_ratio = percentile(gpu_samples, 95) / s.mean_util;
    s.max_idle_gap_ms = max_consecutive_below(gpu_samples, 40.0f) * 500.0f;
    return s;
}
```

**Target Derivation for 780M:**
- Theoretical max: ~95% (never 100% due to unavoidable scheduler handoff)
- Realistic well-pipelined ceiling: 80% ( burst peaks 95%, valleys 40% → average ~75%)
- Target band 75–80% = "scheduler is successfully hiding CPU math and memory latency"
- Below 65% = "GPU is starved; either scheduler too slow or memory-bandwidth saturated"

---

### 1.2 Hailo-8L — NN Core Utilization (Power-Proxy)

The Hailo-8L uses dataflow architecture: once a HEF is loaded and input streaming begins, the NN Core pipelines internally with fixed latency. Utilization should sustain near 100%. Drops below 90% are **almost exclusively** PCIe DMA starvation (host not feeding inputs fast enough).

| # | Metric | Measurement Method | Formula | Target | Acceptable Range | Critical | Frequency |
|---|---|---|---|---|---|---|---|
| U-H1 | **Sustained nn_core_utilization** | Power-proxy: `(power_watts − 0.5) / (6.0 − 0.5) × 100` from `device_->get_power_measurement()` | μ_H = Σ(util_Hᵢ) / N | **85–95%** | ≥ 80% | < 60% | Per-job |
| U-H2 | **Utilization Drop Events** | Count contiguous runs of ≥2 samples below 80% | DE = |{runs of util_H < 80% lasting ≥ 1 s}| | **0** | ≤ 1 per job | ≥ 3 per job | Per-job |
| U-H3 | **PCIe DMA Efficiency** | Ratio of embedding transfer time to encoding compute time | η_DMA = T_transfer / T_compute | **< 0.15** | < 0.25 | ≥ 0.40 | Per-encode |
| U-H4 | **Queue Depth Adequacy** | Instantaneous input FIFO depth reported by HailoRT | QD = `hailo_input_queue_depth_` (read via EP) | **4–8** | ≥ 2 | 0 (empty) | Per-step |
| U-H5 | **Encode Batch Throughput** | Prompt tokens encoded per second end-to-end | T_enc = tokens_prompt / T_elapsed_encode | **> 800 tok/s** | > 500 tok/s | < 300 tok/s | Per-encode |
| U-H6 | **Power Stability** | Coefficient of variation of Hailo power draw during active encoding | CV_P = σ_P / μ_P | **< 0.08** | < 0.15 | ≥ 0.25 | Per-encode |

**Measurement API Details:**

```cpp
// U-H1, U-H6: Power-proxy utilization
constexpr float HAILO_IDLE_W  = 0.5f;
constexpr float HAILO_ACTIVE_W = 6.0f;

float hailo_util_from_power(float power_watts) {
    float norm = (power_watts - HAILO_IDLE_W) / (HAILO_ACTIVE_W - HAILO_IDLE_W);
    return std::clamp(norm * 100.0f, 0.0f, 100.0f);
}

// U-H3: PCIe DMA Efficiency
// T_compute = time from first input token submission to last output ready
// T_transfer = time spent in PCIe memcpy (H2D + D2H)
// Both measured via cudaEvent/hipEvent equivalents or std::chrono around DMA calls

// U-H4: Queue depth — instrumented in pipeline
struct HailoQueueStatus {
    int current_depth;     // slots filled in input FIFO
    int max_depth;         // configured capacity
    bool underrun;         // queue went to 0 during step
};
```

**Target Derivation for Hailo-8L:**
- Idle power: 0.5W (firmware running, NN Core clock-gated)
- Active power: 5.5–6.5W at 100% utilization (all clusters streaming)
- Max TDP: 8.65W (never hit in normal encoding; only seen during HEF load)
- Power-proxy normalization maps 0.5W → 0%, 6.0W → 100%
- Target 85–95% = "NN Core is saturated; PCIe is not the bottleneck"
- Below 80% = "investigate PCIe DMA scheduling or host CPU contention"

---

## 2. Latency Metrics

### 2.1 First-Token & Step Latency

| # | Metric | Measurement Method | Target | Acceptable | Critical | Frequency |
|---|---|---|---|---|---|---|
| L-1 | **TTFT — Time To First Token** (encoder) | `std::chrono::steady_clock` from prompt submission to first embedding output ready in host RAM | **< 35 ms** | < 60 ms | ≥ 100 ms | Per-encode |
| L-2 | **TBT — Time Between Steps** (denoising) | Wall-clock time between start of step N and start of step N+1 | **< 45 ms** | < 65 ms | ≥ 100 ms | Per-step |
| L-3 | **TTE — Time To Encode** (end-to-end) | Hailo T5-XXL + CLIP-L + any CPU post-processing | **< 120 ms** | < 200 ms | ≥ 350 ms | Per-encode |
| L-4 | **VAE Decode Latency** | Time from final latent to RGB tensor (780M via ROCm) | **< 80 ms** | < 120 ms | ≥ 200 ms | Per-job |
| L-5 | **Pipeline Overhead Ratio** | (scheduler + copy + sync) / pure_compute | **< 0.15** | < 0.25 | ≥ 0.40 | Per-job |
| L-6 | **End-to-End Job Latency** (512 px, 20 steps) | Total wall time from prompt to image | **< 1.2 s** | < 2.0 s | ≥ 3.5 s | Per-job |
| L-7 | **End-to-End Job Latency** (1024 px, 28 steps) | Total wall time from prompt to image | **< 2.5 s** | < 4.0 s | ≥ 6.0 s | Per-job |
| L-8 | **P99 Step Latency** | 99th percentile of per-step latencies in a job | **< 60 ms** | < 90 ms | ≥ 150 ms | Per-job |
| L-9 | **Step Latency Jitter** | σ of per-step latencies / μ | **< 0.10** | < 0.20 | ≥ 0.35 | Per-job |
| L-10 | **Memory-Copy Overhead** | Time in `hipMemcpy` + `hailo_dma_copy` per step | **< 5 ms** | < 10 ms | ≥ 20 ms | Per-step |

**Measurement Probe Locations:**

```cpp
// L-1: TTFT
auto t0 = std::chrono::steady_clock::now();
hailo_session_->Run(...);  // Submit encode
hailo_session_->Run(...);  // Get output
auto t1 = std::chrono::steady_clock::now();
float ttft_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

// L-2: TBT (Time Between Steps)
hipEventRecord(start, stream);
// ... denoising kernel for step N ...
hipEventRecord(stop, stream);
hipEventSynchronize(stop);
float step_ms;
hipEventElapsedTime(&step_ms, start, stop);  // pure GPU compute
// TBT = step_ms + overhead_ms (measured separately)

// L-5: Overhead Ratio
float pure_compute_ms = sum(step_kernel_times);
float total_job_ms = end_to_end_ms;
float overhead_ratio = (total_job_ms - pure_compute_ms) / pure_compute_ms;
```

**Target Derivation:**
- 780M @ 16 TOPS, BF16 FLUX DiT: ~2.5–3 TOPS effective → 1024px step ≈ 35–45 ms kernel time
- Hailo-8L T5-XXL INT8: ~13 TOPS, 512-token sequence → ~25–30 ms compute
- Add 2 ms PCIe DMA per encode (8 MB over Gen3 x2)
- TBT target 45 ms = 35 ms kernel + 5 ms scheduler + 5 ms copy/sync
- Total 512px/20-step job: encode 120 ms + 20×45 ms decode + 80 ms VAE = ~1.1 s

---

### 2.2 Resolution-Scaled Latency Bounds

| Resolution | Steps | Target Total | Acceptable | Critical |
|---|---|---|---|---|
| 512 × 512 | 20 | < 1.2 s | < 2.0 s | ≥ 3.5 s |
| 768 × 768 | 25 | < 1.8 s | < 3.0 s | ≥ 5.0 s |
| 1024 × 1024 | 28 | < 2.5 s | < 4.0 s | ≥ 6.0 s |
| 1024 × 1024 | 50 (high-quality) | < 4.0 s | < 6.5 s | ≥ 10.0 s |

Scaling law: Total latency ∝ (pixels × steps) / effective_TOPS + fixed_overhead

---

## 3. Memory Metrics

The UM790 Pro uses **unified memory** (64 GB LPDDR5X-7500, HSA_XNACK=1). The GPU allocates VRAM from this pool via `hipMalloc` (4 GB visible to gfx1103). Page migration occurs transparently on first touch / page fault.

| # | Metric | Measurement Method | Formula | Target | Acceptable | Critical | Frequency |
|---|---|---|---|---|---|---|---|
| M-1 | **Peak Memory Footprint** | `rsmi_dev_memory_usage_get(RSMI_MEM_TYPE_VRAM)` + `getrusage(RUSAGE_SELF).ru_maxrss` | Peak bytes allocated during job | **< 6 GB** | < 10 GB | ≥ 14 GB | Per-job |
| M-2 | **Unified Memory Bandwidth Utilization** | Derived from `perf` / `rocprof` memory counters or kernel PMU | BW_pct = measured_GB/s / 83 GB/s × 100 | **< 70%** | < 85% | ≥ 95% | Per-step |
| M-3 | **Page Migration Rate** | Count HSA_XNACK page-fault events via `dmesg` / ROCm trace | PMR = faults / job_duration_sec | **< 50 faults/s** | < 200 faults/s | ≥ 500 faults/s | Per-job |
| M-4 | **GPU VRAM Saturation** | `rsmi_dev_memory_usage_get` / 4 GB × 100 | VRAM_pct = used / 4 GB × 100 | **< 75%** | < 90% | ≥ 98% | Continuous |
| M-5 | **Memory Fragmentation Index** | After N jobs, measure largest allocatable contiguous block vs total free | FI = 1 − (largest_free / total_free) | **< 0.20** | < 0.35 | ≥ 0.50 | Per-session |
| M-6 | **Allocator Retry Rate** | Count of `hipMalloc` calls that returned `hipErrorMemoryAllocation` and required retry | ARR = retries / total_allocs | **0%** | < 1% | ≥ 5% | Per-job |
| M-7 | **Cross-Device Copy Bandwidth** | Measure H2D + D2H throughput during active inference | CBW = bytes_copied / copy_duration_ms × 1000 / 1e9 | **> 3.5 GB/s** | > 2.5 GB/s | < 1.5 GB/s | Per-job |

**Measurement API Details:**

```cpp
// M-2: Bandwidth utilization via rocprof
// Run: rocprof --stats -i mem_counters.txt ./pipeline
// Counter: FETCH_SIZE + WRITE_SIZE (bytes) / kernel_duration (ns)
// Or via PM4 perf counters on gfx1103:
//   GRBM_PERF_SEL: TA_BUSY, TCC_EA0_WRREQ_sum, TCC_EA0_RDREQ_sum

// M-3: Page migration via kernel trace
// echo 1 > /sys/kernel/debug/tracing/events/kfd/hsa_xnack_page_fault/enable
// cat /sys/kernel/debug/tracing/trace_pipe | grep xnack

// M-5: Fragmentation index
size_t total_free = 0, largest_free = 0;
hipMemGetInfo(&total_free, &largest_free);  // total_free = largest in this API
// Actually: use hipMalloc/hipFree probing to find largest allocatable
// Or use rsmi_dev_memory_total - rsmi_dev_memory_usage_get for free
float fragmentation = 1.0f - (float)largest_contig / (float)total_free;
```

**Target Derivation:**
- Theoretical LPDDR5X-7500 bandwidth: 83 GB/s (dual-channel, 7500 MT/s, 64-bit/channel)
- GPU + Hailo + CPU contending: practical ceiling ~75 GB/s before saturation
- Target < 70% = "headroom for burst traffic; no QoS degradation"
- ≥ 95% = "memory bandwidth is the bottleneck; stalls will cascade to GPU and Hailo"
- 4 GB GPU carve-out: FLUX DiT 1024px needs ~2.5 GB weights + ~1 GB activations → target < 75%

---

## 4. Recovery & Reliability Metrics

The watchdog triggers recovery after 8 consecutive 500 ms samples below 60% utilization, or immediately on < 40%. Recovery teardown destroys the ONNX session, resets the device, and rebuilds.

| # | Metric | Measurement Method | Formula | Target | Acceptable | Critical | Frequency |
|---|---|---|---|---|---|---|---|
| R-1 | **Recovery Success Rate** | `#SUCCESS / (#SUCCESS + #PARTIAL + #FATAL)` over session | RSR = successes / total_recoveries × 100 | **> 98%** | > 95% | < 90% | Per-session |
| R-2 | **State Preservation Accuracy** | MSE between latent tensor before and after recovery | MSE = mean((latent_pre − latent_post)²) | **< 1×10⁻⁶** | < 1×10⁻⁵ | ≥ 1×10⁻⁴ | Per-recovery |
| R-3 | **Time-to-Recover (GPU)** | Wall time from recovery trigger to `hipLaunchKernel` resuming | TTR_GPU = t_resume − t_trigger | **< 1.5 s** | < 3.0 s | ≥ 5.0 s | Per-recovery |
| R-4 | **Time-to-Recover (Hailo)** | Wall time from trigger to first post-recovery encode complete | TTR_Hailo = t_encode_done − t_trigger | **< 2.0 s** | < 4.0 s | ≥ 6.0 s | Per-recovery |
| R-5 | **False Positive Rate** | Recoveries triggered where post-hoc analysis shows no actual fault | FPR = false_positives / total_recoveries × 100 | **< 2%** | < 5% | ≥ 10% | Per-session |
| R-6 | **Recovery Cascade Events** | One recovery triggering a secondary recovery on the other device | RCE = count(cascade_recoveries) | **0** | ≤ 1 | ≥ 3 | Per-session |
| R-7 | **Session Abort Rate** | Jobs that could not complete due to unrecoverable fault | SAR = aborted_jobs / total_jobs × 100 | **0%** | < 0.5% | ≥ 2% | Per-session |
| R-8 | **Recovery-Induced Latency Penalty** | Extra wall time added to a job that experienced recovery | RLP = (job_with_recovery_ms − baseline_job_ms) / baseline_job_ms × 100 | **< 15%** | < 25% | ≥ 50% | Per-recovery |
| R-9 | **Partial Recovery Rate** | Recoveries returning `PARTIAL` (reduced capacity) | PRR = partials / total_recoveries × 100 | **< 2%** | < 5% | ≥ 15% | Per-session |
| R-10 | **Mean Steps Between Recovery** | Average inference steps between recovery events | MTBR = total_steps / recovery_count | **> 1000** | > 500 | < 200 | Per-session |

**Measurement API Details:**

```cpp
// R-2: State preservation accuracy
// Before recovery: snapshot latents to CPU pinned memory
// After recovery: compare restored latents to snapshot
float compute_mse(const float* a, const float* b, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double diff = double(a[i]) - double(b[i]);
        sum += diff * diff;
    }
    return float(sum / double(n));
}
// Expected: < 1e-6 for FP16/BF16 latents (numerical noise floor)

// R-3 / R-4: Time-to-recover
std::chrono::steady_clock::time_point t_trigger, t_resume;
// t_trigger = time when trigger_recovery() called
// t_resume = time when first post-recovery Run() returns

// R-5: False positive detection (post-hoc analysis)
// After recovery, examine preceding 8 steps:
//   - If all kernel traces show normal execution, and
//   - If power/temp were nominal, and
//   - If utilization drop was transient (< 2s)
//   → classify as false positive
```

**Target Derivation:**
- GPU recovery: ~500 ms session destroy + 500 ms `hipDeviceReset` + 500 ms context rebuild + 300 ms session recreate = ~1.8 s worst case; target < 1.5 s requires aggressive parallelization
- Hailo recovery: ~300 ms hard reset + 200 ms firmware boot + 500 ms HEF reload + 300 ms session rebuild = ~1.3 s; target < 2.0 s is achievable
- State preservation: FP16 latents have ~1e-4 quantization noise; recovery should not add more than 1e-6 additional MSE
- Cascade events are the most dangerous: a GPU recovery that triggers Hailo recovery (or vice versa) indicates resource-lock bug

---

## 5. Thermal & Power Metrics

The 7940HS has a 65W configurable TDP. The 780M is a large iGPU that can throttle aggressively. The Hailo-8L is low-power but temperature-sensitive.

| # | Metric | Measurement Method | Formula | Target | Acceptable | Critical | Frequency |
|---|---|---|---|---|---|---|---|
| T-1 | **780M Junction Temperature** | `rsmi_dev_temp_metric_get(RSMI_TEMP_TYPE_JUNCTION, RSMI_TEMP_CURRENT)` | °C | **< 85°C** | < 95°C | ≥ 100°C | Continuous |
| T-2 | **Hailo-8L Chip Temperature** | `device_->get_chip_temperature().ts0_temperature` | °C | **< 70°C** | < 80°C | ≥ 85°C | Continuous |
| T-3 | **Thermal Throttling Events** | Count `rsmi_dev_perf_level` transitions below max perf level | TTE = count(throttle_events) | **0** | ≤ 2 per session | ≥ 5 per session | Per-session |
| T-4 | **Throttling Impact on Utilization** | Utilization delta in 5 s window after throttle vs 5 s before | ΔU = μ_after − μ_before | **> −5%** | > −10% | ≤ −20% | Per-throttle |
| T-5 | **System Power Envelope** | Total SoC power via `cat /sys/class/powercap/intel-rapl` equivalent or `ryzenadj` | P_sys = CPU + GPU + SoC (W) | **< 55W sustained** | < 65W | ≥ 75W | Continuous |
| T-6 | **GPU Power Draw** | Derived from `rsmi_dev_power_cap_get` / `rsmi_dev_power_avg_get` | P_gpu (W) | **< 25W** | < 30W | ≥ 35W | Continuous |
| T-7 | **Power-Performance Efficiency** | TOPS per watt for the full inference job | EFF = (GPU_TOPS × util + Hailo_TOPS × util) / P_total | **> 0.40 TOPS/W** | > 0.30 | < 0.20 | Per-job |
| T-8 | **Temperature Rise Rate** | Δ°C / s during sustained load ramp | TRR = (T_peak − T_idle) / ramp_time_s | **< 2°C/s** | < 4°C/s | ≥ 6°C/s | Per-session |
| T-9 | **Cooldown Recovery Time** | Time from job end for junction temp to drop below 70°C | T_cool = t(T < 70°C) − t(job_end) | **< 30 s** | < 60 s | ≥ 120 s | Per-job |
| T-10 | **Hailo Power Variance During Encode** | σ of Hailo power during a single encode | σ_P_encode | **< 0.3W** | < 0.5W | ≥ 1.0W | Per-encode |

**Measurement API Details:**

```cpp
// T-1: GPU junction temperature
int64_t temp_milli = 0;
rsmi_dev_temp_metric_get(0, RSMI_TEMP_TYPE_JUNCTION, RSMI_TEMP_CURRENT, &temp_milli);
float gpu_junction_c = temp_milli / 1000.0f;

// T-3: Throttling events
// Watch for perf level dropping from RSMI_DEV_PERF_LEVEL_AUTO (max) to
// RSMI_DEV_PERF_LEVEL_LOW or RSMI_DEV_PERF_LEVEL_MANUAL with reduced clocks
uint32_t perf_level;
rsmi_dev_perf_level_get(0, &perf_level);
// Log transition when perf_level != previous_perf_level

// T-5: System power envelope
// Using ryzenadj (must run as root)
// ryzenadj -i | grep "PPT LIMIT" | grep "FAST" → gives current PPT
// Or read MSR 0xC0010299 (PkgPwrLimit) if accessible
// Fallback: estimate from wall power meter or battery discharge rate

// T-7: Power-performance efficiency
float gpu_effective_tops = 16.5f * (gpu_util / 100.0f);
float hailo_effective_tops = 13.0f * (hailo_util / 100.0f);
float total_power = cpu_power + gpu_power + hailo_power + dram_power;
float eff = (gpu_effective_tops + hailo_effective_tops) / total_power;
```

**Target Derivation:**
- 7940HS thermal design: 65W TDP at 95°C junction; throttling begins at ~100°C
- 780M junction target < 85°C = "comfortable headroom before throttling; consistent clocks"
- 780M junction acceptable < 95°C = "near throttling threshold but not yet hitting it"
- Hailo-8L: max ambient + junction rise = ~80°C acceptable; above 85°C may cause PCIe link errors
- System power 55W sustained = "fits within 65W TDP with 10W headroom for CPU burst and DRAM"

---

## 6. Composite Pipeline Health Score

A single 0–100 score that aggregates all primary metrics into an interpretable health index.

### 6.1 Score Formula

```
HealthScore = Σ (wᵢ × scoreᵢ)
```

where each `scoreᵢ` is a per-metric normalized sub-score [0, 100], and `wᵢ` is the weight.

| Sub-Score | Weight | Metric Source | Normalization Formula |
|---|---|---|---|
| **S_gpu_util** | 20% | U-G1 (sustained GPU util) | `clamp((μ − 50) / 30 × 100, 0, 100)` → 50%→0, 80%→100 |
| **S_hailo_util** | 15% | U-H1 (sustained Hailo util) | `clamp((μ − 60) / 35 × 100, 0, 100)` → 60%→0, 95%→100 |
| **S_latency** | 20% | L-6 / L-7 (job latency) | `clamp((2.0 − job_sec) / 1.5 × 100, 0, 100)` for 512px; 1024px scaled |
| **S_memory** | 10% | M-2 (BW util) + M-1 (footprint) | `100 − BW_pct` (lower is better) capped at 70 |
| **S_recovery** | 15% | R-1 (success rate) + R-3/R-4 (TTR) | `RSR × 0.8 + (1 − mean(TTR) / 3.0) × 20` |
| **S_thermal** | 10% | T-1 (GPU temp) + T-3 (throttle events) | `100 − max(0, (T − 75) × 2) − TTE × 10` |
| **S_variance** | 10% | U-G2 (σ) + L-9 (jitter) | `100 − σ × 200 − jitter × 200` |

### 6.2 Detailed Sub-Score Definitions

#### S_gpu_util — GPU Utilization Score (weight: 20%)
```
S_gpu_util = clamp((μ_util − 50.0) / 30.0 × 100, 0, 100)
```
- μ = 50% → 0 points
- μ = 65% → 50 points
- μ = 80% → 100 points
- μ > 80% → 100 points (diminishing returns above target)

#### S_hailo_util — Hailo Utilization Score (weight: 15%)
```
S_hailo_util = clamp((μ_util − 60.0) / 35.0 × 100, 0, 100)
```
- μ = 60% → 0 points
- μ = 85% → 71 points
- μ = 95% → 100 points

#### S_latency — End-to-End Latency Score (weight: 20%)
```
// For 512px/20-step jobs:
S_latency_512 = clamp((2.0 − job_sec) / 1.5 × 100, 0, 100)
// For 1024px/28-step jobs:
S_latency_1024 = clamp((4.0 − job_sec) / 2.5 × 100, 0, 100)
```
- 512px: 2.0 s → 0, 1.2 s → 53, 0.5 s → 100
- 1024px: 4.0 s → 0, 2.5 s → 60, 1.5 s → 100

Use the resolution-appropriate formula and weight by job mix.

#### S_memory — Memory Health Score (weight: 10%)
```
S_memory_bw = clamp(100 − M-2, 0, 100)    // lower BW% is better
S_memory_fp = clamp(100 − (M-1 / 6.0) × 100, 0, 100)  // < 6GB = 100
S_memory = 0.6 × S_memory_bw + 0.4 × S_memory_fp
```

#### S_recovery — Recovery Reliability Score (weight: 15%)
```
S_recovery_sr  = clamp(RSR, 0, 100)          // R-1: success rate
S_recovery_ttr = clamp((3.0 − mean_TTR) / 3.0 × 100, 0, 100)
S_recovery = 0.7 × S_recovery_sr + 0.3 × S_recovery_ttr
```
- Success rate dominates; fast recovery is secondary

#### S_thermal — Thermal Health Score (weight: 10%)
```
S_thermal_temp = clamp(100 − max(0, (T_junction − 75) × 2.5), 0, 100)
S_thermal_throt = clamp(100 − TTE × 15, 0, 100)
S_thermal = 0.6 × S_thermal_temp + 0.4 × S_thermal_throt
```
- 75°C → 100 points, 95°C → 50 points, 115°C → 0 points
- Each throttle event deducts 15 points

#### S_variance — Stability / Consistency Score (weight: 10%)
```
S_variance = clamp(100 − U_G2 × 400 − L_9 × 200, 0, 100)
```
- σ = 12% → 52 points (moderate)
- σ = 15% → 40 points (acceptable boundary)
- jitter = 0.10 → 80 points; jitter = 0.20 → 60 points

### 6.3 Final Composite Formula

```
PipelineHealthScore =
    0.20 × S_gpu_util
  + 0.15 × S_hailo_util
  + 0.20 × S_latency
  + 0.10 × S_memory
  + 0.15 × S_recovery
  + 0.10 × S_thermal
  + 0.10 × S_variance
```

### 6.4 Interpretation Bands

| Score | Grade | Interpretation | Action |
|---|---|---|---|
| 90–100 | **A — Excellent** | Pipeline is fully optimized. All metrics at or above target. | None; monitor for regression |
| 75–89 | **B — Good** | Pipeline operates within acceptable bounds. Minor tuning possible. | Optional: review variance metrics |
| 60–74 | **C — Fair** | One or more metrics below acceptable. Pipeline functional but not efficient. | **Required:** Identify low sub-score and tune |
| 40–59 | **D — Poor** | Multiple metrics critical or below acceptable. Recovery events likely. | **Urgent:** Full profiling and scheduler review |
| 0–39 | **F — Failing** | System is not viable for production. Frequent aborts or throttling. | **Stop:** Hardware or architecture change required |

### 6.5 Score Computation Example

Scenario: A 1024px/28-step FLUX job completes in 2.8 s.

| Metric | Raw Value | Sub-Score | Weighted |
|---|---|---|---|
| GPU util μ | 72% | S_gpu_util = 73 | 20% → 14.6 |
| Hailo util μ | 88% | S_hailo_util = 80 | 15% → 12.0 |
| Job latency | 2.8 s | S_latency = 48 | 20% → 9.6 |
| BW util | 65% | S_memory = 61 | 10% → 6.1 |
| Recovery SR | 100% (no recovery) | S_recovery = 100 | 15% → 15.0 |
| GPU temp | 82°C | S_thermal = 83 | 10% → 8.3 |
| σ util | 14% | S_variance = 44 | 10% → 4.4 |
| **TOTAL** | | | **70.0** |

**Result: Grade C — Fair.** The job is functional but latency (2.8 s vs target 2.5 s) and utilization variance are dragging the score down. Action: tune scheduler prefetch and reduce step-to-step dead time.

---

## 7. Measurement Methodology & Instrumentation

### 7.1 Required Instrumentation Hooks

Add the following probes to `pipeline.cpp` and `watchdog.cpp`:

```cpp
// === include/hq/metrics_collector.hpp ===
#pragma once
#include <chrono>
#include <vector>
#include <atomic>

namespace hq {

struct StepMetrics {
    int step_number;
    float gpu_util;              // U-G1 per-step
    float hailo_util;            // U-H1 per-step
    float step_latency_ms;       // L-2
    float memory_used_mb;        // M-1
    float gpu_temp_c;            // T-1
    float hailo_temp_c;          // T-2
    float hailo_power_w;         // T-6 proxy
    std::chrono::steady_clock::time_point timestamp;
};

struct JobMetrics {
    int job_id;
    int resolution;
    int num_steps;
    float total_latency_ms;      // L-6 / L-7
    float encode_latency_ms;     // L-3
    float vae_latency_ms;        // L-4
    float overhead_ratio;        // L-5
    float gpu_mean_util;         // U-G1
    float gpu_std_dev;           // U-G2
    float gpu_valley_pct;        // U-G3
    float hailo_mean_util;       // U-H1
    int recovery_events;         // R-1 source
    bool aborted;              // R-7
    std::vector<StepMetrics> steps;
};

class MetricsCollector {
public:
    void record_step(const StepMetrics& m);
    void finalize_job(JobMetrics& m);  // computes aggregates
    void export_json(const std::string& path) const;
    void export_csv(const std::string& path) const;
    
    // Compute composite score for last N jobs
    float compute_health_score(int n_jobs = 10) const;
    
private:
    std::vector<JobMetrics> jobs_;
    mutable std::mutex mutex_;
};

} // namespace hq
```

### 7.2 Sampling Frequencies Summary

| Metric Category | Primary Sample Rate | Aggregation Window | Storage |
|---|---|---|---|
| Utilization (GPU + Hailo) | 500 ms (watchdog) | Per-step mean + per-job mean | In-memory ring buffer (last 10k samples) |
| Latency (per-step) | Every step boundary | Per-job P50/P95/P99 | JobMetrics vector |
| Latency (end-to-end) | Per-job | Per-session mean | Session log |
| Memory | Every 4 steps (via watchdog) | Per-job peak | JobMetrics |
| Thermal | 500 ms (watchdog) | Per-job peak + sustained mean | JobMetrics |
| Recovery | Event-triggered | Per-session statistics | Recovery log |
| Power | 500 ms (Hailo) / ROCm SMI (GPU) | Per-job mean | JobMetrics |

### 7.3 Benchmark Run Protocol

To establish baseline metrics for a configuration, run this protocol:

```
Phase 1: Warm-up (discard)
  - 5 dummy jobs at target resolution
  - Let caches warm, let thermals stabilize

Phase 2: Baseline Measurement
  - 20 consecutive jobs at 512px / 20 steps
  - 20 consecutive jobs at 1024px / 28 steps
  - Single-job mode (no batching)
  - Record all metrics per job

Phase 3: Stress Measurement
  - 100 consecutive jobs at 1024px / 28 steps
  - No cooldown between jobs
  - Record thermal ramp, throttling events, recovery events
  - Measure fragmentation index at end

Phase 4: Recovery Stress (optional)
  - Induce artificial fault (e.g., kill Hailo EP process)
  - Verify recovery path executes successfully
  - Measure R-2 through R-10

Phase 5: Analysis
  - Compute all per-job metrics
  - Compute per-session aggregates
  - Compute composite health score
  - Generate pass/fail report
```

### 7.4 Pass/Fail Report Template

```
=== UM790 Pro Pipeline Benchmark Report ===
Date: YYYY-MM-DD HH:MM:SS
Configuration: FLUX-schnell + T5-XXL (Hailo) + 1024px/28step

[UTILIZATION]
  GPU sustained util:    73.2%  [TARGET: 75-80%]  [RESULT: ACCEPTABLE]
  Hailo sustained util:  91.5%  [TARGET: 85-95%]  [RESULT: PASS]
  GPU valley frequency:  3.1%   [TARGET: <2%]     [RESULT: FAIL]

[LATENCY]
  TTFT:                  28 ms  [TARGET: <35ms]   [RESULT: PASS]
  TBT (mean):            42 ms  [TARGET: <45ms]   [RESULT: PASS]
  E2E 1024px/28:         2.62 s [TARGET: <2.5s]   [RESULT: ACCEPTABLE]

[MEMORY]
  Peak footprint:        5.8 GB [TARGET: <6GB]    [RESULT: PASS]
  BW utilization:        68%    [TARGET: <70%]    [RESULT: PASS]

[RECOVERY]
  Success rate:          N/A    (no events)       [RESULT: PASS]

[THERMAL]
  GPU junction peak:     87°C   [TARGET: <85°C]   [RESULT: ACCEPTABLE]
  Throttle events:       0      [TARGET: 0]       [RESULT: PASS]

[COMPOSITE SCORE]
  Health Score:          74.3   [GRADE: C — Fair]
  Primary concern:       GPU valley frequency (3.1% > 2% target)
                         → Tune scheduler prefetch aggressiveness
```

---

## 8. Metric Quick Reference Table

| ID | Metric | Unit | Target | Acceptable | Critical | Measure Frequency | Where Measured |
|---|---|---|---|---|---|---|---|
| U-G1 | GPU sustained util | % | 75–80 | ≥ 65 | < 60 | Per-job | Watchdog thread |
| U-G2 | GPU util variance | % σ | < 12 | < 15 | ≥ 20 | Per-step | Watchdog samples |
| U-G3 | GPU valley frequency | % steps | < 2 | < 5 | ≥ 10 | Per-job | Watchdog samples |
| U-G4 | Burst-to-sustained ratio | ratio | 1.15–1.25 | 1.10–1.35 | > 1.50 | Per-job | Post-analysis |
| U-G5 | Idle gap duration | s | < 1.0 | < 2.0 | ≥ 4.0 | Per-job | Watchdog samples |
| U-G6 | Inter-step dead time | ms | < 3 | < 8 | ≥ 20 | Per-step | hipEvent |
| U-H1 | Hailo sustained util | % | 85–95 | ≥ 80 | < 60 | Per-job | Power-proxy |
| U-H2 | Hailo drop events | count | 0 | ≤ 1 | ≥ 3 | Per-job | Event detection |
| U-H3 | PCIe DMA efficiency | ratio | < 0.15 | < 0.25 | ≥ 0.40 | Per-encode | Event timing |
| U-H4 | Queue depth | slots | 4–8 | ≥ 2 | 0 | Per-step | HailoRT API |
| U-H5 | Encode throughput | tok/s | > 800 | > 500 | < 300 | Per-encode | Timer |
| U-H6 | Hailo power stability | CV | < 0.08 | < 0.15 | ≥ 0.25 | Per-encode | Power samples |
| L-1 | TTFT | ms | < 35 | < 60 | ≥ 100 | Per-encode | chrono timer |
| L-2 | TBT | ms | < 45 | < 65 | ≥ 100 | Per-step | chrono timer |
| L-3 | TTE | ms | < 120 | < 200 | ≥ 350 | Per-encode | chrono timer |
| L-4 | VAE decode latency | ms | < 80 | < 120 | ≥ 200 | Per-job | chrono timer |
| L-5 | Overhead ratio | ratio | < 0.15 | < 0.25 | ≥ 0.40 | Per-job | Post-analysis |
| L-6 | E2E latency (512/20) | s | < 1.2 | < 2.0 | ≥ 3.5 | Per-job | chrono timer |
| L-7 | E2E latency (1024/28) | s | < 2.5 | < 4.0 | ≥ 6.0 | Per-job | chrono timer |
| L-8 | P99 step latency | ms | < 60 | < 90 | ≥ 150 | Per-job | Step distribution |
| L-9 | Step latency jitter | ratio | < 0.10 | < 0.20 | ≥ 0.35 | Per-job | σ/μ |
| L-10 | Mem-copy overhead | ms | < 5 | < 10 | ≥ 20 | Per-step | chrono timer |
| M-1 | Peak memory footprint | GB | < 6 | < 10 | ≥ 14 | Per-job | RSMI + getrusage |
| M-2 | Memory BW utilization | % | < 70 | < 85 | ≥ 95 | Per-step | rocprof / PMU |
| M-3 | Page migration rate | faults/s | < 50 | < 200 | ≥ 500 | Per-job | Kernel trace |
| M-4 | GPU VRAM saturation | % | < 75 | < 90 | ≥ 98 | Continuous | RSMI |
| M-5 | Fragmentation index | ratio | < 0.20 | < 0.35 | ≥ 0.50 | Per-session | Allocation probe |
| M-6 | Allocator retry rate | % | 0 | < 1 | ≥ 5 | Per-job | Error counter |
| M-7 | Cross-device copy BW | GB/s | > 3.5 | > 2.5 | < 1.5 | Per-job | memcpy timer |
| R-1 | Recovery success rate | % | > 98 | > 95 | < 90 | Per-session | Recovery log |
| R-2 | State preservation MSE | float | < 1e-6 | < 1e-5 | ≥ 1e-4 | Per-recovery | Tensor compare |
| R-3 | Time-to-recover (GPU) | s | < 1.5 | < 3.0 | ≥ 5.0 | Per-recovery | chrono timer |
| R-4 | Time-to-recover (Hailo) | s | < 2.0 | < 4.0 | ≥ 6.0 | Per-recovery | chrono timer |
| R-5 | False positive rate | % | < 2 | < 5 | ≥ 10 | Per-session | Post-analysis |
| R-6 | Recovery cascade events | count | 0 | ≤ 1 | ≥ 3 | Per-session | Event correlation |
| R-7 | Session abort rate | % | 0 | < 0.5 | ≥ 2 | Per-session | Job completion log |
| R-8 | Recovery latency penalty | % | < 15 | < 25 | ≥ 50 | Per-recovery | Timer compare |
| R-9 | Partial recovery rate | % | < 2 | < 5 | ≥ 15 | Per-session | Recovery log |
| R-10 | Mean steps between recovery | steps | > 1000 | > 500 | < 200 | Per-session | Job log |
| T-1 | GPU junction temp | °C | < 85 | < 95 | ≥ 100 | Continuous | RSMI |
| T-2 | Hailo chip temp | °C | < 70 | < 80 | ≥ 85 | Continuous | HailoRT |
| T-3 | Throttle events | count | 0 | ≤ 2 | ≥ 5 | Per-session | Perf level log |
| T-4 | Throttle util impact | % Δ | > −5 | > −10 | ≤ −20 | Per-throttle | Pre/post compare |
| T-5 | System power envelope | W | < 55 | < 65 | ≥ 75 | Continuous | ryzenadj / powercap |
| T-6 | GPU power draw | W | < 25 | < 30 | ≥ 35 | Continuous | RSMI |
| T-7 | Power efficiency | TOPS/W | > 0.40 | > 0.30 | < 0.20 | Per-job | Derived |
| T-8 | Temp rise rate | °C/s | < 2 | < 4 | ≥ 6 | Per-session | Derived |
| T-9 | Cooldown recovery time | s | < 30 | < 60 | ≥ 120 | Per-job | Timer |
| T-10 | Hailo power variance | W σ | < 0.3 | < 0.5 | ≥ 1.0 | Per-encode | Power samples |

---

## 9. Appendices

### Appendix A: Hardware-Specific Measurement Notes

**Radeon 780M (gfx1103, RDNA3):**
- `rsmi_dev_gpu_busy_percent_get()` samples the SMU's accumulated busy counter. The SMU firmware samples at ~1 ms intervals internally and returns the fraction of intervals where any shader block (WGP) was active.
- This is **not** a cycle-accurate occupancy metric. It measures "was the GPU doing anything" not "were all SMs fully occupied." This is appropriate for our purposes: we want to detect idle gaps.
- Counter wraparound: The SMU counter is 32-bit and wraps at ~71 minutes of continuous 100% utilization. For our 500 ms sampling, this is irrelevant.
- Temperature: `RSMI_TEMP_TYPE_JUNCTION` is the hot-spot (hottest point on die). This is more representative than edge temperature for throttling behavior. Target < 85°C keeps us comfortably below the 100°C throttling threshold.

**Hailo-8L (PCIe Gen3 x2):**
- The power-proxy utilization metric is a **derived** value, not a native counter. The Hailo-8L does not expose NN Core cycle counters to the host. The power measurement is the most reliable proxy.
- Power measurement accuracy: The on-chip power sensor has ±5% accuracy at typical load. This means the utilization proxy has ±5% error at the 6W operating point. Do not over-interpret single-sample variations.
- PCIe Gen3 x2 theoretical bandwidth: 2 lanes × 8 GT/s × 128b/130b encoding = ~1.97 GB/s per direction. Effective bandwidth with DMA overhead is ~1.5–1.7 GB/s. An 8 MB embedding transfer therefore takes ~5 ms worst case, ~2 ms typical.
- Temperature sensor: `ts0_temperature` is the primary on-die sensor. The Hailo-8L has no active cooling on the UM790 Pro (it relies on case airflow). Monitor carefully in sustained runs.

### Appendix B: ROCm SMI Counter Reference

| Counter | API Call | Unit | Relevant Metrics |
|---|---|---|---|
| GPU Busy % | `rsmi_dev_gpu_busy_percent_get()` | % | U-G1, U-G2, U-G3, U-G4, U-G5 |
| VRAM Used | `rsmi_dev_memory_usage_get(VRAM)` | bytes | M-1, M-4 |
| Junction Temp | `rsmi_dev_temp_metric_get(JUNCTION, CURRENT)` | °C × 1000 | T-1 |
| Edge Temp | `rsmi_dev_temp_metric_get(EDGE, CURRENT)` | °C × 1000 | Diagnostic |
| Power Draw | `rsmi_dev_power_ave_get()` | μW | T-6 |
| Power Cap | `rsmi_dev_power_cap_get()` | μW | T-5 (as component) |
| Perf Level | `rsmi_dev_perf_level_get()` | enum | T-3 (throttle detection) |
| Clock (GFX) | `rsmi_dev_gpu_clk_freq_get(CLK_TYPE_DF)` | MHz | Diagnostic |
| PCIe Throughput | `rsmi_dev_pci_throughput_get()` | B/s | M-7 (indirect) |

### Appendix C: rocprof / HW Perf Counter Selection (gfx1103)

For deeper analysis of memory bandwidth and GPU efficiency, use these PM4 perf counters:

```
# Counter group for memory bandwidth
GRBM_PERF_SEL: TA_BUSY                    # Texture addresser busy
GRBM_PERF_SEL: TCC_EA0_WRREQ_sum          # Write requests to L2
GRBM_PERF_SEL: TCC_EA0_RDREQ_sum          # Read requests from L2
SQ_PERF_SEL: SQ_WAVES                      # Wavefronts dispatched
SQ_PERF_SEL: SQ_INSTS_VALU                 # VALU instructions issued

# Compute bandwidth from counter deltas:
# BW_write = (TCC_EA0_WRREQ_delta × 64B) / time_delta_ns × 1e9 / 1e9 [GB/s]
# BW_read  = (TCC_EA0_RDREQ_delta × 64B) / time_delta_ns × 1e9 / 1e9 [GB/s]
```

Run with:
```bash
rocprof --stats -i counters.txt ./pipeline_benchmark
```

### Appendix D: Formulas for Non-Linear Metric Combinations

Some metrics interact non-linearly. Use these derived formulas for advanced analysis:

**Effective Compute Throughput (ECT):**
```
ECT (GOP/s) = (GPU_T_eff + Hailo_T_eff + CPU_T_eff)
where:
  GPU_T_eff   = 16.5 TOPS × (U-G1 / 100) × 0.6   // 60% of theoretical due to memory
  Hailo_T_eff = 13.0 TOPS × (U-H1 / 100) × 0.85  // 85% of theoretical (dataflow efficient)
  CPU_T_eff   = 12.0 TOPS × CPU_util × 0.30        // 30% of theoretical (scheduler-bound)
```

**Pipeline Efficiency Factor (PEF):**
```
PEF = ECT / (P_system × 1.0)   // GOP/s per watt
Target PEF > 0.40 for competitive performance-per-watt
```

**Quality-Adjusted Latency (QAL):**
```
QAL = L-7 (total latency) × (1 + R-7 (abort rate)) × (1 + T-3 (throttle events / 10))
// Penalizes latency by reliability and thermal stability
```

---

*End of Metrics Framework v1.0*
