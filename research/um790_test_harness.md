# UM790 Pro Test Harness Specification
## Complete Benchmark & Watchdog Validation Suite v1.0

**Target Hardware:** Ryzen 9 7940HS (Zen 4, AVX-512) + Radeon 780M (gfx1103, RDNA3) + Hailo-8L (13 TOPS, PCIe Gen3 x2)
**Memory:** 64GB LPDDR5X-7500 (~83 GB/s), unified with HSA_XNACK=1
**Watchdog:** C++20 std::jthread, 500ms sampling, 8-step consecutive threshold
**Document Version:** 1.0

---

## Table of Contents

1. [Harness Architecture](#1-harness-architecture)
2. [Instrumentation Points](#2-instrumentation-points)
3. [Warm-up / Measurement / Cool-down Protocol](#3-warm-up--measurement--cool-down-protocol)
4. [Configuration Schema](#4-configuration-schema)
5. [Pseudo-code for Core Harness](#5-pseudo-code-for-core-harness)
6. [Reproducibility Requirements](#6-reproducibility-requirements)

---

## 1. Harness Architecture

### 1.1 Component Overview

```
+------------------------------------------------------------------+
|                    BenchmarkRunner                                 |
|  +------------------------------------------------------------+  |
|  |                    run_suite()                               |  |
|  |  For each workload config:                                   |  |
|  |    run_workload() -> produces WorkloadResult                 |  |
|  +------------------------------------------------------------+  |
|                    |                                               |
|     +--------------+--------------+                                |
|     |                             |                                |
|  WorkloadAdapter              Instrumentor                         |
|  (5 workloads)                (probes at all IPs)                  |
|     |                             |                                |
|     +--------------+--------------+                                |
|                    |                                               |
|              BinaryDataLogger                                      |
|              (high-freq samples) + EventLogger (text)              |
|                    |                                               |
|              ReportGenerator                                       |
|              (aggregation + statistics)                            |
+------------------------------------------------------------------+
```

### 1.2 Class Hierarchy

#### 1.2.1 `BenchmarkRunner` — Orchestrator

```cpp
class BenchmarkRunner {
public:
    struct Config {
        SuiteConfig suite;          // TOML-parsed top-level config
        std::string output_dir;     // Where all artifacts go
        uint32_t random_seed;       // For reproducible prompts/noise
    };

    explicit BenchmarkRunner(Config cfg);
    ~BenchmarkRunner();

    // Run all workloads defined in suite config
    SuiteResult run_suite();

    // Run a single workload with full instrumentation
    WorkloadResult run_workload(const WorkloadConfig& wcfg, uint32_t run_id);

    // Generate final JSON + markdown report
    void generate_report(const SuiteResult& results);

private:
    Config cfg_;
    std::unique_ptr<Instrumentor> instrumentor_;
    std::unique_ptr<BinaryDataLogger> data_logger_;
    std::unique_ptr<EventLogger> event_logger_;
    std::unique_ptr<RecoveryValidator> recovery_validator_;

    // Pre-run system state capture
    SystemSnapshot capture_system_state();

    // Thermal stabilization
    bool wait_for_thermal_idle(float max_temp_c, std::chrono::seconds timeout);
};
```

#### 1.2.2 `IWorkloadAdapter` — Interface

```cpp
// Every workload implements this interface. The harness does NOT know
// workload internals — it only calls these methods and the workload
// reports phase boundaries via the Instrumentor handle passed in.

class IWorkloadAdapter {
public:
    virtual ~IWorkloadAdapter() = default;

    // Called once before any runs. Validate model files, warm caches.
    virtual void initialize(const WorkloadConfig& cfg,
                           Instrumentor* instr) = 0;

    // Called for warm-up iterations (results discarded)
    virtual void run_warmup(const WorkloadConfig& cfg,
                            Instrumentor* instr) = 0;

    // Called for measured iterations. Must call Instrumentor phase markers.
    virtual WorkloadResult run_measured(const WorkloadConfig& cfg,
                                         Instrumentor* instr,
                                         uint32_t iteration) = 0;

    // Called between runs. Release resources, allow thermal recovery.
    virtual void cooldown() = 0;

    // Human-readable name for reports
    virtual std::string name() const = 0;

    // Workload category for aggregation
    virtual WorkloadCategory category() const = 0;
};

// Factory
std::unique_ptr<IWorkloadAdapter> create_workload(WorkloadType type);
```

**Five Workload Implementations:**

| # | Adapter Class | Workload | Pipeline Phase |
|---|---|---|---|
| 1 | `FluxText2ImgAdapter` | FLUX.1-dev text-to-image (12-step DEIS) | Full pipeline |
| 2 | `FluxImg2ImgAdapter` | FLUX.1-dev image-to-image (8-step) | Full pipeline |
| 3 | `QwenEditAdapter` | Qwen2.5-VL edit (mask + prompt) | Full pipeline |
| 4 | `SanaSprintAdapter` | Sana Sprint 4-step (lightning-fast) | Full pipeline |
| 5 | `WatchdogStressAdapter` | Synthetic watchdog stress test | Watchdog validation |

#### 1.2.3 `Instrumentor` — Central Probe Hub

```cpp
// All timing and sampling flows through here.
// The Instrumentor owns the high-frequency sampler thread AND
// receives explicit phase markers from workload adapters.

class Instrumentor {
public:
    struct Config {
        uint32_t sample_interval_us = 1000;   // High-freq loop: 1kHz
        uint32_t thermal_interval_ms = 500;   // Thermal polling: 2Hz
        bool enable_gpu_sampling = true;
        bool enable_hailo_sampling = true;
        bool enable_memory_sampling = true;
    };

    explicit Instrumentor(Config cfg);
    ~Instrumentor();

    // ── Phase markers (called by workload adapters) ──
    void mark_phase(PhaseId phase, PhaseBoundary boundary, int64_t seq = 0);

    // Convenience wrappers
    void mark_encode_start() { mark_phase(PhaseId::HAILO_ENCODE, PhaseBoundary::START); }
    void mark_encode_end()   { mark_phase(PhaseId::HAILO_ENCODE, PhaseBoundary::END); }
    void mark_denoise_step_start(int step) { mark_phase(PhaseId::GPU_DENOISE, PhaseBoundary::START, step); }
    void mark_denoise_step_end(int step)   { mark_phase(PhaseId::GPU_DENOISE, PhaseBoundary::END, step); }
    void mark_scheduler_start(int step)    { mark_phase(PhaseId::CPU_SCHEDULER, PhaseBoundary::START, step); }
    void mark_scheduler_end(int step)      { mark_phase(PhaseId::CPU_SCHEDULER, PhaseBoundary::END, step); }
    void mark_vae_start()    { mark_phase(PhaseId::VAE_DECODE, PhaseBoundary::START); }
    void mark_vae_end()      { mark_phase(PhaseId::VAE_DECODE, PhaseBoundary::END); }
    void mark_recovery_start(ComputeUnit unit, int step, RecoveryTrigger reason);
    void mark_recovery_end(ComputeUnit unit, RecoveryResult result, int steps_lost);
    void mark_memory_alloc(size_t bytes, AllocType type);
    void mark_memory_free(size_t bytes, AllocType type);

    // ── Sampler control ──
    void start_sampling();   // Launch sampler threads
    void stop_sampling();    // Drain and stop
    void reset_buffers();    // Clear between iterations

    // ── Data retrieval ──
    std::vector<SampleRecord> get_samples(PhaseId phase) const;
    std::vector<PhaseInterval> get_phase_intervals(PhaseId phase) const;
    InstrumentationSummary get_summary() const;

private:
    Config cfg_;

    // High-frequency sampler thread (1kHz)
    std::jthread hf_sampler_;
    void hf_sampler_loop(std::stop_token st);

    // Thermal sampler thread (2Hz — aligned with watchdog)
    std::jthread thermal_sampler_;
    void thermal_sampler_loop(std::stop_token st);

    // Ring buffers for zero-allocation logging
    static constexpr size_t SAMPLE_RING_SIZE = 1 << 20;  // ~1M samples
    RingBuffer<SampleRecord> sample_ring_{SAMPLE_RING_SIZE};
    RingBuffer<PhaseMarker>  phase_ring_{SAMPLE_RING_SIZE >> 4};
    RingBuffer<EventRecord>  event_ring_{SAMPLE_RING_SIZE >> 8};

    // Per-device last-known values (for correlation)
    std::atomic<float> last_gpu_util_{0.0f};
    std::atomic<float> last_hailo_util_{0.0f};
    std::atomic<float> last_gpu_temp_{0.0f};
    std::atomic<float> last_hailo_temp_{0.0f};
    std::atomic<uint64_t> last_vram_used_{0};
    std::atomic<uint64_t> last_sys_ram_used_{0};
};
```

**Key Design Decision:** The Instrumentor uses a **ring buffer** for all high-frequency data. This avoids heap allocation during measurement windows, which would perturb the very memory metrics we're trying to capture. The ring buffer is sized to hold ~1M samples (~16 seconds at 1kHz), which exceeds the longest single workload (Sana Sprint completes in ~800ms, FLUX in ~8-12s).

#### 1.2.4 `BinaryDataLogger` — Zero-Overhead Sample Storage

```cpp
// Writes ring-buffer dumps to a memory-mapped binary file.
// Format is append-only, structured as:
//
//   [Header: 64 bytes]
//   [Chunk 1: PhaseMarkers]
//   [Chunk 2: SampleRecords]
//   [Chunk 3: EventRecords]
//   ... (one chunk per measurement window)

class BinaryDataLogger {
public:
    struct FileHeader {
        char magic[8] = "HQSAMP01";
        uint64_t version = 1;
        uint64_t chunk_count = 0;
        uint64_t total_samples = 0;
        uint64_t total_markers = 0;
        uint64_t total_events = 0;
        SystemSnapshot sys_state;
        char padding[8];
    };

    explicit BinaryDataLogger(const std::string& output_path);
    ~BinaryDataLogger();

    // Append a measurement window's data
    void write_window(const InstrumentationWindow& window);

    // Flush to disk
    void flush();

private:
    int fd_;
    size_t file_size_;
    uint8_t* mmap_base_;
    FileHeader* header_;
    std::mutex write_mutex_;
};
```

**Binary Record Formats:**

```cpp
// 32 bytes — one per high-frequency sample (1kHz during active phases)
struct SampleRecord {
    uint64_t timestamp_ns;      // std::chrono::high_resolution_clock nanos
    float    gpu_util_pct;      // rsmi_dev_gpu_busy_percent_get()
    float    hailo_util_pct;    // Power-proxy derived
    float    gpu_temp_c;        // Junction temperature
    float    hailo_temp_c;      // TS0 temperature
    float    gpu_power_w;       // If available via SMI
    float    hailo_power_w;     // get_power_measurement()
    uint64_t vram_used_bytes;   // RSMI VRAM
    uint64_t sys_ram_used_bytes;// /proc/meminfo or rsmi
};

// 32 bytes — one per phase transition (inserted by workload adapters)
struct PhaseMarker {
    uint64_t timestamp_ns;
    PhaseId  phase;             // 1 byte enum
    PhaseBoundary boundary;     // START / END
    int32_t  sequence;          // Step number for iterative phases
    uint32_t run_id;            // Which measured run
    uint32_t workload_id;       // Which workload
    char     padding[6];
};

// 64 bytes — one per discrete event (recovery, allocation, etc.)
struct EventRecord {
    uint64_t timestamp_ns;
    EventType type;             // RECOVERY, ALLOC, FREE, THERMAL_ALERT, etc.
    uint32_t run_id;
    uint32_t workload_id;
    union {
        struct { float util_at_fault; int step; RecoveryTrigger reason; } recovery;
        struct { size_t bytes; AllocType alloc_type; } alloc;
        struct { float temp_c; float threshold_c; } thermal;
        struct { int step; ComputeUnit unit; } watchdog;
    } detail;
    char padding[16];
};
```

#### 1.2.5 `ReportGenerator` — Aggregation Engine

```cpp
class ReportGenerator {
public:
    // Aggregate raw samples into final metrics
    static BenchmarkReport generate(
        const SuiteResult& suite,
        const std::vector<InstrumentationWindow>& windows
    );

    // Write JSON (machine-readable) + Markdown (human-readable)
    static void write_json(const BenchmarkReport& report,
                           const std::string& path);
    static void write_markdown(const BenchmarkReport& report,
                                const std::string& path);

private:
    // Per-workload statistics
    static WorkloadMetrics compute_metrics(
        const std::vector<PhaseInterval>& intervals,
        const std::vector<SampleRecord>& samples
    );

    // Cross-workload comparisons
    static ComparisonTable compare_workloads(
        const std::vector<WorkloadMetrics>& metrics
    );

    // Watchdog validation summary
    static WatchdogValidationReport validate_watchdog(
        const std::vector<EventRecord>& events,
        const std::vector<PhaseInterval>& intervals
    );
};
```

### 1.3 Data Flow Diagram

```
WorkloadAdapter::run_measured()
    |
    +-- [BEGIN] calls instrumentor_->mark_encode_start()
    |        |
    |        v
    |   Instrumentor writes PhaseMarker to ring buffer
    |   (timestamp_ns = now, phase=HAILO_ENCODE, boundary=START)
    |
    +-- Hailo does encoding work...
    |   (Watchdog samples independently at 500ms)
    |
    +-- [END] calls instrumentor_->mark_encode_end()
    |        |
    |        v
    |   Instrumentor writes PhaseMarker
    |   Sampler thread captures GPU/Hailo readings at 1kHz
    |        |
    |        v
    |   BinaryDataLogger.write_window()  <-- called after each iteration
    |   (appends ring buffer dump to mmap'd file)
    |
    +-- Workload completes, returns WorkloadResult

After all iterations:
    ReportGenerator::generate()
        reads binary data
        computes statistics (mean, p50, p99, stddev)
        writes JSON + Markdown
```

---

## 2. Instrumentation Points

### 2.1 Instrumentation Map

| ID | Name | Location | What to Measure | Sampling | Data Format |
|---|---|---|---|---|---|
| IP-01 | `PRE_ENCODE` | `Pipeline::Impl::run_generate()` line: "Encode prompt with Hailo" | Hailo power ramp, PCIe DMA setup latency | Phase marker + 1kHz samples | `PhaseMarker{HAILO_ENCODE, START}` + `SampleRecord` stream |
| IP-02 | `POST_ENCODE` | Same function, after `hailo_session_->Run()` returns | Total encode time, Hailo peak power, embedding output size | Phase marker | `PhaseMarker{HAILO_ENCODE, END}` |
| IP-03 | `PRE_DENOISE_STEP` | Top of denoising loop: `for (int step = 0; ...)` | GPU idle time before kernel launch | Phase marker per step | `PhaseMarker{GPU_DENOISE, START, step}` |
| IP-04 | `POST_DENOISE_STEP` | Bottom of denoising loop body | Per-step GPU time, peak utilization during step | Phase marker per step + peak sample | `PhaseMarker{GPU_DENOISE, END, step}` + max `gpu_util` in interval |
| IP-05 | `SCHEDULER_START` | Before `scheduler_->step()` call | CPU scheduling computation start | Phase marker | `PhaseMarker{CPU_SCHEDULER, START, step}` |
| IP-06 | `SCHEDULER_END` | After `scheduler_->step()` returns | Scheduler latency, AVX-512 throughput proxy | Phase marker + CPU cycle counter delta | `PhaseMarker{CPU_SCHEDULER, END, step}` |
| IP-07 | `VAE_START` | Before `vae_session_->Run()` | VRAM allocation for VAE, GPU context switch | Phase marker + `rsmi_dev_memory_usage_get()` | `PhaseMarker{VAE_DECODE, START}` |
| IP-08 | `VAE_POST_LATENT` | After latent decode, before image-to-pixel conversion | Latent decode time only (excluding pixel conversion) | Phase marker | `PhaseMarker{VAE_LATENT, END}` |
| IP-09 | `VAE_END` | After final image tensor is ready | Total VAE time, peak VRAM during decode | Phase marker + peak sample | `PhaseMarker{VAE_DECODE, END}` |
| IP-10 | `WATCHDOG_SAMPLE` | `UtilizationWatchdog::monitor_loop()` every 500ms | Aligned with harness timestamps for correlation | 2Hz (500ms fixed) | `EventRecord{WATCHDOG_SAMPLE}` + util values |
| IP-11 | `MEM_ALLOC` | `MemPool::allocate()` / `hipMalloc()` | Allocation size, type (GPU/CPU/unified), fragmentation signal | Event on every alloc > 1MB | `EventRecord{ALLOC, bytes, type}` |
| IP-12 | `MEM_FREE` | `MemPool::free()` / `hipFree()` | Free size, coalescing effectiveness | Event on every free > 1MB | `EventRecord{FREE, bytes, type}` |
| IP-13 | `RECOVERY_START` | `UtilizationWatchdog::trigger_recovery()` entry | Device, step, util at fault, consecutive count | Event + Phase marker | `EventRecord{RECOVERY, ...}` + `PhaseMarker{RECOVERY, START}` |
| IP-14 | `RECOVERY_END` | `trigger_recovery()` exit via result | Result (SUCCESS/PARTIAL/FATAL), time elapsed, steps lost | Event + Phase marker | `EventRecord{RECOVERY_END, ...}` + `PhaseMarker{RECOVERY, END}` |
| IP-15 | `THERMAL_SAMPLE` | `Instrumentor::thermal_sampler_loop()` every 500ms | GPU junction temp, Hailo TS0 temp, throttling status | 2Hz continuous | `SampleRecord{temps}` |
| IP-16 | `MEM_BW_SAMPLE` | `Instrumentor::hf_sampler_loop()` derived | Unified memory bandwidth utilization | 1Hz derived | See Section 2.3 |

### 2.2 Phase Definitions

```cpp
enum class PhaseId : uint8_t {
    HAILO_ENCODE   = 0x01,  // Text encoding on Hailo-8L (T5 + CLIP)
    GPU_DENOISE    = 0x02,  // Single denoising step on 780M (per-step)
    CPU_SCHEDULER  = 0x03,  // DEIS/DPMSolver scheduler computation
    VAE_DECODE     = 0x04,  // Full VAE decode (latent -> image)
    VAE_LATENT     = 0x05,  // Latent decode portion only
    RECOVERY       = 0x06,  // Watchdog recovery sequence
    WARMUP         = 0x07,  // Warm-up iterations (discarded)
    IDLE           = 0x08,  // Between phases (GPU idle)
};

enum class PhaseBoundary : uint8_t {
    START = 0x01,
    END   = 0x02,
};

enum class RecoveryTrigger : uint8_t {
    CONSECUTIVE_LOW = 0x01,  // 8 steps below 60%
    CRITICAL_UTIL   = 0x02,  // < 40% (immediate)
    DEVICE_FAULT    = 0x03,  // Device health bit false
    INJECTED_FAULT  = 0x04,  // Synthetic (watchdog stress test)
};

enum class EventType : uint8_t {
    RECOVERY        = 0x01,
    RECOVERY_END    = 0x02,
    ALLOC           = 0x03,
    FREE            = 0x04,
    WATCHDOG_SAMPLE = 0x05,
    THERMAL_ALERT   = 0x06,
    RECOVERY_FAULT  = 0x07,  // Recovery callback threw
    LATENT_SAVE     = 0x08,  // Latent state snapshot taken
    LATENT_RESTORE  = 0x09,  // Latent state restored post-recovery
};

enum class AllocType : uint8_t {
    GPU_VRAM     = 0x01,
    CPU_HOST     = 0x02,
    UNIFIED      = 0x03,  // hipMallocManaged / HSA_XNACK
    PINNED       = 0x04,
};
```

### 2.3 Memory Bandwidth Measurement Method

The unified LPDDR5X bus is shared across CPU, GPU, and Hailo. Direct bandwidth measurement requires hardware PMU counters that may not be accessible. The harness uses a **proxy method**:

```cpp
// Method: Read /sys/class/drm/card0/gt_cur_freq_mhz and
// /sys/class/hwmon/hwmon*/power1_average (if available)
// plus derive bandwidth from transfer sizes + timing.

// Primary approach: Instrumented transfer timing
struct BandwidthProbe {
    // For known-size transfers, compute achieved bandwidth
    static float measure_transfer_bw(size_t bytes, float seconds) {
        return static_cast<float>(bytes) / seconds / (1024.0f * 1024.0f * 1024.0f); // GB/s
    }

    // For the Hailo DMA transfer specifically:
    // Embedding size = tokens * hidden_dim * sizeof(float32)
    // T5-XXL: 512 tokens * 4096 * 4 = 8,388,608 bytes (~8MB)
    // PCIe Gen3 x2 theoretical: ~2 GB/s → expect ~4ms transfer
    // Measured > 6ms → bandwidth contention detected
    static float embedding_transfer_bw(int num_tokens, int hidden_dim, float transfer_seconds) {
        size_t bytes = num_tokens * hidden_dim * sizeof(float);
        return measure_transfer_bw(bytes, transfer_seconds);
    }
};

// Secondary approach: System-level memory pressure via vmstat
// Read /proc/vmstat before and after measurement window:
//   pgpgin, pgpgout, pswpin, pswpout, pgalloc, pgfree
// Delta over window gives page-level memory activity.
struct SystemMemProbe {
    struct VmStat {
        uint64_t pgpgin, pgpgout;   // Pages paged in/out
        uint64_t pgalloc, pgfree;   // Pages allocated/freed
        uint64_t pswpin, pswpout;   // Swap activity (should be 0)
    };

    static VmStat read_vmstat();
    static float derive_pressure_index(const VmStat& before, const VmStat& after, float seconds);
};
```

### 2.4 Watchdog-Harness Timestamp Alignment

The watchdog and instrumentor run on different threads with different sampling rates. To correlate events:

```cpp
// Both systems use std::chrono::steady_clock (monotonic, nanosecond precision).
// At harness start, capture a synchronization point:

struct TimeSync {
    std::chrono::steady_clock::time_point steady;
    std::chrono::system_clock::time_point system;
    uint64_t steady_ns;  // Redundant for speed
};

// The watchdog's 500ms samples and the instrumentor's 1kHz samples
// are merged during report generation by nearest-neighbor timestamp matching.
// Tolerance: 1ms (samples within 1ms are considered "same instant").

// Recovery events from the watchdog (IP-13, IP-14) are matched to
// harness phase intervals by searching for the overlapping time range.
```

---

## 3. Warm-up / Measurement / Cool-down Protocol

### 3.1 Rationale

Cold-start effects on the UM790 Pro include:
1. **GPU kernel compilation** (HIP runtime caches compiled kernels after first use)
2. **Hailo HEF loading** (HEF is loaded into Hailo SRAM on first inference)
3. **Memory page migration** (HSA_XNACK pages migrate to GPU on first touch)
4. **ROC pipeline warmup** (shader caches, texture caches, descriptor caches)
5. **Thermal throttling** (first run may boost higher, subsequent runs throttle)

### 3.2 Protocol Definition

```
For each workload:

    ┌─────────────────────────────────────────────────────────────┐
    │  PHASE 0: System State Verification                         │
    │  ├── Verify CPU governor = 'performance'                    │
    │  ├── Verify GPU power profile = 'PROFILE_COMPUTE'           │
    │  ├── Verify HSA_XNACK = 1                                 │
    │  ├── Verify no other GPU processes running                  │
    │  └── Record baseline temperature (must be < 55C junction)  │
    └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  PHASE 1: Thermal Idle                                      │
    │  ├── Wait until GPU junction < 55C                          │
    │  ├── Wait until Hailo TS0 < 50C                             │
    │  ├── Timeout: 120 seconds (fail if not reached)             │
    │  └── Log idle temperatures                                  │
    └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  PHASE 2: Warm-up                                           │
    │  ├── Run 3 throwaway iterations of the FULL workload        │
    │  ├── No instrumentation (avoid probe overhead)              │
    │  ├── Purpose:                                               │
    │  │   - Warm GPU caches (shader, texture, descriptor)        │
    │  │   - Load Hailo HEF into SRAM                             │
    │  │   - Trigger HSA_XNACK page migration                     │
    │  │   - Stabilize thermal profile                            │
    │  └── Between warm-ups: 5 second gap                         │
    └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  PHASE 3: Measurement Window                                │
    │  ├── Run N measured iterations (default N=10)               │
    │  ├── Full instrumentation active (1kHz + phase markers)     │
    │  ├── Stabilization criteria (MUST pass before accepting):   │
    │  │   - Coefficient of variation (CV) < 5% across last 5     │
    │  │   - No thermal throttling detected during any iteration  │
    │  │   - No recovery events during last 3 iterations          │
    │  │   - If criteria fail after N=10, extend to N=20          │
    │  │   - If criteria still fail, flag in report               │
    │  └── Between iterations: 10 second gap (see 3.3)            │
    └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  PHASE 4: Cool-down                                         │
    │  ├── Wait until GPU junction < 55C (or within 5C of idle)   │
    │  ├── Wait until Hailo TS0 < 50C                             │
    │  ├── Timeout: 180 seconds                                   │
    │  └── Log recovery temperatures                              │
    └─────────────────────────────────────────────────────────────┘
```

### 3.3 Inter-Iteration Timing

```cpp
// Between measured iterations, a fixed gap prevents thermal carryover.
// The gap duration depends on the previous workload's thermal output:

struct InterIterationGap {
    static std::chrono::seconds calculate(
        float peak_gpu_temp_c,      // Peak junction temp during iteration
        float target_idle_temp_c,   // From Phase 1 baseline
        WorkloadCategory category   // LIGHT / MEDIUM / HEAVY
    ) {
        float delta = peak_gpu_temp_c - target_idle_temp_c;

        // Base gap by workload thermal class
        int base_seconds;
        switch (category) {
            case WorkloadCategory::LIGHT:   base_seconds = 5;  break;  // Sana Sprint
            case WorkloadCategory::MEDIUM:  base_seconds = 10; break;  // img2img
            case WorkloadCategory::HEAVY:   base_seconds = 15; break;  // FLUX txt2img
        }

        // Additional time proportional to overheating
        int thermal_penalty = static_cast<int>(delta / 2.0f);  // +1s per 2C over target

        return std::chrono::seconds(base_seconds + thermal_penalty);
    }
};
```

### 3.4 Stabilization Criteria Detail

```cpp
bool check_stabilization(const std::vector<WorkloadResult>& results) {
    // Need at least 5 results to check
    if (results.size() < 5) return false;

    // Get last 5
    auto last5 = std::vector<WorkloadResult>(results.end() - 5, results.end());

    // 1. Coefficient of variation on end-to-end time < 5%
    std::vector<float> times;
    for (const auto& r : last5) times.push_back(r.total_time_ms);
    float mean = compute_mean(times);
    float stddev = compute_stddev(times, mean);
    float cv = (mean > 0) ? (stddev / mean) : 0;
    if (cv > 0.05f) return false;

    // 2. No thermal throttling in last 3
    for (size_t i = last5.size() - 3; i < last5.size(); ++i) {
        if (last5[i].gpu_peak_temp_c > 95.0f) return false;  // Throttle threshold
    }

    // 3. No recovery events in last 3
    for (size_t i = last5.size() - 3; i < last5.size(); ++i) {
        if (last5[i].recovery_count > 0) return false;
    }

    return true;
}
```

### 3.5 Watchdog Stress Test Protocol (Workload #5)

The watchdog stress test uses synthetic fault injection to validate recovery:

```
Stress Test Sequence:

    Test A: GPU Utilization Drop
    ├── Run 1: Normal workload (baseline)
    ├── Run 2: Inject GPU idle by inserting 100ms sleep between denoise steps at step 4
    │   └── Expected: Watchdog triggers after 8 consecutive low steps (steps 4-11)
    │   └── Validate: Recovery fires, latents preserved, denoising resumes
    ├── Run 3: Inject GPU idle at step 20 (later in pipeline)
    └── Run 4: Inject critical drop (< 40%) at step 8
        └── Expected: Immediate recovery (bypasses 8-step counter)

    Test B: Hailo Utilization Drop
    ├── Run 5: Stall PCIe DMA by allocating large CPU buffer during encode
    │   └── Expected: Hailo watchdog triggers
    └── Run 6: Simulate Hailo device fault

    Test C: Rapid Alternation
    ├── Run 7: Alternate high/low utilization every 3 steps
    │   └── Expected: Counter resets between highs, no false recovery

    Test D: Concurrent Drop
    ├── Run 8: Drop both GPU and Hailo simultaneously
    │   └── Expected: Both recoveries fire, ordered by criticality

    Validation Criteria for ALL stress tests:
    1. Recovery latency < 5 seconds (from fault injection to resumed inference)
    2. No latent corruption (output image hash matches non-recovery baseline)
    3. Recovery counter increments correctly
    4. Phase markers show continuous denoising (no gaps > 500ms post-recovery)
```

---

## 4. Configuration Schema

### 4.1 TOML Configuration (Primary)

```toml
# ============================================================
# UM790 Pro Benchmark Suite Configuration
# ============================================================

[suite]
name = "UM790Pro_Full_Benchmark_v1"
description = "Complete benchmark of all 5 workloads with watchdog validation"
output_dir = "./benchmark_results"
random_seed = 42                    # Fixed seed for reproducible prompts
runs_per_workload = 10              # Default measured iterations
max_runs_per_workload = 20          # Extended limit if CV not met
cv_threshold = 0.05                 # Coefficient of variation target (< 5%)
parallel_workloads = false          # Run sequentially (thermal control)

# ============================================================
# System State Requirements (enforced before suite start)
# ============================================================

[system]
cpu_governor = "performance"        # Enforce: cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
gpu_power_profile = "PROFILE_COMPUTE"  # ROCm SMI profile
gpu_performance_level = "HIGH"      # rsmi_dev_perf_level_set
max_background_processes = 2        # Fail if > N non-system processes use GPU
required_env_vars = [               # All must be present
    "HSA_XNACK=1",
    "HSA_ENABLE_SDMA=1",
    "HIP_VISIBLE_DEVICES=0",
]

# Pre-run temperature limits
max_idle_gpu_temp_c = 55.0
max_idle_hailo_temp_c = 50.0

# ============================================================
# Thermal Protocol Parameters
# ============================================================

[thermal]
idle_timeout_sec = 120              # Max wait for thermal idle before suite start
cooldown_timeout_sec = 180          # Max wait for cooldown between workloads
throttle_detect_temp_c = 95.0       # GPU junction temp indicating throttling
emergency_shutdown_temp_c = 100.0   # Abort benchmark if exceeded

# ============================================================
# Measurement Parameters
# ============================================================

[measurement]
sample_interval_us = 1000           # High-freq sampler: 1kHz
timestamp_clock = "steady"          # Use steady_clock (monotonic)
enable_phase_markers = true
enable_gpu_sampling = true
enable_hailo_sampling = true
enable_memory_sampling = true
enable_bandwidth_proxy = true       # Enable derived bandwidth estimates

# Ring buffer sizing
ring_buffer_samples = 1048576       # 1M samples (~16s at 1kHz)
ring_buffer_markers = 65536         # 64K markers
events_buffer_size = 8192           # 8K events

# Output formats
output_binary = true                # .hqsamp binary files
output_json = true                  # Per-run JSON dumps
output_csv = false                  # Optional CSV export

# ============================================================
# Watchdog Parameters
# ============================================================

[watchdog]
sample_interval_ms = 500            # Must match pipeline watchdog
consecutive_threshold = 8           # Steps below threshold before recovery
low_threshold = 60.0                # Warning/recovery trigger (%)
critical_threshold = 40.0           # Immediate recovery trigger (%)
target_low = 75.0                   # Sweet-spot lower bound
target_high = 80.0                  # Sweet-spot upper bound
auto_recover = true                 # Enable automatic recovery
log_path = "./benchmark_results/watchdog.log"

# ============================================================
# Workload Definitions
# ============================================================

[[workload]]
name = "FLUX_text2img"
type = "flux_text2img"
category = "heavy"                  # For thermal classification
enabled = true
run_order = 1                       # Execution order

[workload.model]
flux_model_path = "./models/gpu/flux.1-dev.bf16.onnx"
t5_encoder_path = "./models/hailo/t5_encoder.hef"
clip_encoder_path = "./models/hailo/clip_encoder.hef"
vae_decoder_path = "./models/gpu/vae_decoder.bf16.onnx"
scheduler_type = "deis"
num_steps = 12                      # DEIS default
guidance_scale = 3.5
resolution = [1024, 1024]

[workload.prompt]
prompt = "a serene mountain landscape at golden hour, photorealistic, 8k uhd"
negative_prompt = ""
max_tokens = 512

[workload.measurement]
runs = 10
warmup_runs = 3
inter_iteration_gap_sec = 15        # Base gap (may be extended thermally)

# ---

[[workload]]
name = "FLUX_img2img"
type = "flux_img2img"
category = "medium"
enabled = true
run_order = 2

[workload.model]
flux_model_path = "./models/gpu/flux.1-dev.bf16.onnx"
t5_encoder_path = "./models/hailo/t5_encoder.hef"
clip_encoder_path = "./models/hailo/clip_encoder.hef"
vae_decoder_path = "./models/gpu/vae_decoder.bf16.onnx"
vae_encoder_path = "./models/gpu/vae_encoder.bf16.onnx"
scheduler_type = "deis"
num_steps = 8                       # Fewer steps for img2img
guidance_scale = 3.0
strength = 0.75                     # img2img denoising strength
resolution = [1024, 1024]

[workload.prompt]
prompt = "transform into a watercolor painting"
input_image = "./test_assets/photo_base_1024.png"

[workload.measurement]
runs = 10
warmup_runs = 3
inter_iteration_gap_sec = 10

# ---

[[workload]]
name = "Qwen2.5_VL_edit"
type = "qwen_edit"
category = "medium"
enabled = true
run_order = 3

[workload.model]
qwen_mmdit_path = "./models/gpu/qwen2.5vl_mmdit.onnx"
qwen_text_encoder_path = "./models/hailo/qwen_text_encoder.hef"
qwen_vision_path = "./models/gpu/qwen_vision_tower.onnx"
vae_path = "./models/gpu/qwen_vae.onnx"
num_steps = 20
guidance_scale = 4.0

[workload.prompt]
image = "./test_assets/portrait_1024.png"
edit_prompt = "Change the hair color to red and add sunglasses"
mask_prompt = "hair and face region"

[workload.measurement]
runs = 10
warmup_runs = 3
inter_iteration_gap_sec = 10

# ---

[[workload]]
name = "Sana_Sprint_4step"
type = "sana_sprint"
category = "light"
enabled = true
run_order = 4

[workload.model]
sana_model_path = "./models/gpu/sana_sprint_1.6b.onnx"
vae_path = "./models/gpu/sana_vae.onnx"
num_steps = 4                       # Lightning-fast
guidance_scale = 4.5
resolution = [1024, 1024]

[workload.prompt]
prompt = "a cute cat wearing a tiny hat, digital art"

[workload.measurement]
runs = 20                           # More runs due to speed
warmup_runs = 5
inter_iteration_gap_sec = 5

# ---

[[workload]]
name = "Watchdog_Stress_Test"
type = "watchdog_stress"
category = "heavy"                  # Stress tests run long
category = "heavy"
enabled = true
run_order = 5                       # Run last (may leave system in odd state)

[workload.model]
flux_model_path = "./models/gpu/flux.1-dev.bf16.onnx"
t5_encoder_path = "./models/hailo/t5_encoder.hef"
clip_encoder_path = "./models/hailo/clip_encoder.hef"
vae_decoder_path = "./models/gpu/vae_decoder.bf16.onnx"
num_steps = 28                      # Long enough to inject at multiple points

[workload.stress]
# Fault injection scenarios to enable
injections = [
    { type = "gpu_idle", step = 4, duration_ms = 100, test_id = "A1" },
    { type = "gpu_idle", step = 20, duration_ms = 100, test_id = "A2" },
    { type = "gpu_critical", step = 8, duration_ms = 500, test_id = "A3" },
    { type = "hailo_dma_stall", step = 0, duration_ms = 200, test_id = "B1" },
    { type = "hailo_device_fault", step = 0, simulate = true, test_id = "B2" },
    { type = "rapid_alternation", start_step = 6, period = 3, count = 6, test_id = "C1" },
    { type = "concurrent_drop", step = 10, duration_ms = 150, test_id = "D1" },
]

# Validation criteria
max_recovery_latency_ms = 5000
max_denoise_gap_ms = 500            # Max gap in denoising phase markers
latent_hash_tolerance = 0           # Output must match baseline exactly

[workload.measurement]
runs = 7                            # One per injection scenario
warmup_runs = 1
inter_iteration_gap_sec = 30        # Extended for post-recovery stabilization
```

### 4.2 JSON Configuration (Alternative / Override)

The harness also accepts a JSON config with identical structure for programmatic generation:

```json
{
  "suite": {
    "name": "UM790Pro_CI_Benchmark",
    "runs_per_workload": 5,
    "cv_threshold": 0.10
  },
  "system": {
    "cpu_governor": "performance",
    "required_env_vars": ["HSA_XNACK=1"]
  },
  "workloads": [
    {
      "name": "FLUX_text2img",
      "type": "flux_text2img",
      "model": { "num_steps": 12 },
      "measurement": { "runs": 5 }
    }
  ]
}
```

### 4.3 Environment Variable Overrides

All TOML values can be overridden via environment variables for CI integration:

```bash
# Override format: HARNESS_<Section>__<Key>
export HARNESS_SUITE__RUNS_PER_WORKLOAD=5
export HARNESS_SUITE__CV_THRESHOLD=0.10
export HARNESS_MEASUREMENT__SAMPLE_INTERVAL_US=500
export HARNESS_WATCHDOG__AUTO_RECOVER=false
export HARNESS_WORKLOAD__0__ENABLED=false   # Disable first workload
```

---

## 5. Pseudo-code for Core Harness

### 5.1 `BenchmarkRunner::run_suite()`

```cpp
// ============================================================
// BenchmarkRunner::run_suite()
// ============================================================
// Orchestrates the full benchmark suite across all configured
// workloads. Enforces thermal protocols between workloads.

SuiteResult BenchmarkRunner::run_suite() {
    SuiteResult suite_result;
    suite_result.config_name = cfg_.suite.name;
    suite_result.start_time = system_clock::now();

    // ── Phase 0: System State Verification ──
    auto sys_state = capture_system_state();
    if (!verify_system_state(sys_state)) {
        throw BenchmarkError("System state verification failed. "
                             "Run preflight checklist.");
    }
    suite_result.system_state = sys_state;

    // ── Phase 1: Thermal Idle ──
    event_logger_->log(EventType::SUITE_START, "Waiting for thermal idle...");
    if (!wait_for_thermal_idle(cfg_.system.max_idle_gpu_temp_c,
                                120s)) {
        throw BenchmarkError("Thermal idle timeout. System too hot to start.");
    }
    event_logger_->log(EventType::THERMAL_IDLE, "Thermal idle reached");

    // Sort workloads by run_order
    auto workloads = cfg_.suite.workloads;
    std::sort(workloads.begin(), workloads.end(),
              [](auto& a, auto& b) { return a.run_order < b.run_order; });

    // ── Run each workload ──
    for (size_t wi = 0; wi < workloads.size(); ++wi) {
        const auto& wcfg = workloads[wi];

        if (!wcfg.enabled) {
            event_logger_->log(EventType::WORKLOAD_SKIP,
                std::format("Workload '{}' disabled", wcfg.name));
            continue;
        }

        event_logger_->log(EventType::WORKLOAD_START,
            std::format("[{}] Starting workload '{}', category={}",
                        wi, wcfg.name, to_string(wcfg.category)));

        // Create workload adapter via factory
        auto adapter = create_workload(wcfg.type);

        // Run the workload with full instrumentation
        WorkloadResult wresult = run_workload(wcfg, adapter.get(), static_cast<uint32_t>(wi));

        suite_result.workloads.push_back(std::move(wresult));

        // ── Inter-workload cooldown (except last) ──
        if (wi < workloads.size() - 1) {
            event_logger_->log(EventType::COOLDOWN_START,
                std::format("Cooling down before next workload..."));

            bool cooled = wait_for_thermal_idle(
                cfg_.system.max_idle_gpu_temp_c,
                std::chrono::seconds(cfg_.thermal.cooldown_timeout_sec)
            );

            if (!cooled) {
                event_logger_->log(EventType::THERMAL_ALERT,
                    "WARNING: Cooldown incomplete. Proceeding with elevated temps.");
                suite_result.warnings.push_back("Incomplete cooldown after " + wcfg.name);
            }
        }
    }

    suite_result.end_time = system_clock::now();

    // ── Generate report ──
    event_logger_->log(EventType::SUITE_END, "All workloads complete. Generating report...");
    generate_report(suite_result);

    return suite_result;
}
```

### 5.2 `BenchmarkRunner::run_workload()`

```cpp
// ============================================================
// BenchmarkRunner::run_workload()
// ============================================================
// Runs a single workload through warm-up, measurement, and
// validation phases. Manages instrumentor lifecycle per workload.

WorkloadResult BenchmarkRunner::run_workload(
    const WorkloadConfig& wcfg,
    IWorkloadAdapter* adapter,
    uint32_t workload_id)
{
    WorkloadResult result;
    result.workload_name = wcfg.name;
    result.workload_id = workload_id;
    result.category = wcfg.category;

    // ── Initialize adapter ──
    adapter->initialize(wcfg, instrumentor_.get());

    // ── Phase 2: Warm-up ──
    event_logger_->log(EventType::WARMUP_START,
        std::format("Warm-up: {} throwaway iterations", wcfg.measurement.warmup_runs));

    instrumentor_->reset_buffers();
    instrumentor_->mark_phase(PhaseId::WARMUP, PhaseBoundary::START);

    for (uint32_t w = 0; w < wcfg.measurement.warmup_runs; ++w) {
        adapter->run_warmup(wcfg, instrumentor_.get());

        // Brief gap between warm-ups
        std::this_thread::sleep_for(5s);
    }

    instrumentor_->mark_phase(PhaseId::WARMUP, PhaseBoundary::END);
    event_logger_->log(EventType::WARMUP_END, "Warm-up complete");

    // ── Phase 3: Measurement Window ──
    std::vector<WorkloadResult> iteration_results;
    uint32_t target_runs = wcfg.measurement.runs;
    uint32_t max_runs = cfg_.suite.max_runs_per_workload;

    for (uint32_t run = 0; run < max_runs; ++run) {
        event_logger_->log(EventType::ITERATION_START,
            std::format("Run {}/{} (target={})", run + 1, max_runs, target_runs));

        // Reset instrumentor buffers for this iteration
        instrumentor_->reset_buffers();
        instrumentor_->start_sampling();

        // Run the measured workload
        auto iter_result = adapter->run_measured(wcfg, instrumentor_.get(), run);
        iter_result.iteration = run;
        iter_result.workload_id = workload_id;

        instrumentor_->stop_sampling();

        // Dump ring buffers to binary log
        InstrumentationWindow window;
        window.workload_id = workload_id;
        window.iteration = run;
        window.samples = instrumentor_->get_all_samples();
        window.markers = instrumentor_->get_all_markers();
        window.events = instrumentor_->get_all_events();
        data_logger_->write_window(window);

        iteration_results.push_back(iter_result);

        // Check stabilization (after we have enough samples)
        if (run >= target_runs - 1 && run >= 4) {
            if (check_stabilization(iteration_results)) {
                event_logger_->log(EventType::STABILIZATION,
                    std::format("Stabilized after {} runs (CV < {}%)",
                                run + 1, cfg_.suite.cv_threshold * 100));
                break;
            } else if (run == max_runs - 1) {
                event_logger_->log(EventType::STABILIZATION,
                    std::format("WARNING: Did not stabilize after {} runs", max_runs));
                result.warnings.push_back("Stabilization not achieved");
            }
        }

        // Inter-iteration gap (thermal)
        if (run < max_runs - 1) {
            auto gap = InterIterationGap::calculate(
                iter_result.gpu_peak_temp_c,
                sys_state_.idle_gpu_temp_c,
                wcfg.category
            );
            event_logger_->log(EventType::ITERATION_GAP,
                std::format("Inter-iteration gap: {}s", gap.count()));
            std::this_thread::sleep_for(gap);
        }
    }

    result.iterations = std::move(iteration_results);

    // Compute aggregate statistics
    result.aggregate = compute_aggregate(result.iterations);

    // ── Phase 4: Cooldown ──
    adapter->cooldown();

    return result;
}
```

### 5.3 `Instrumentor::sample_gpu()` — ROCm SMI Reader

```cpp
// ============================================================
// Instrumentor::sample_gpu()
// ============================================================
// Reads Radeon 780M utilization and temperature via ROCm SMI.
// Called from the high-frequency sampler thread at 1kHz.
// Must complete in < 50us to maintain sampling cadence.

void Instrumentor::sample_gpu(SampleRecord& out) {
    // GPU busy percentage [0, 100]
    // rsmi_dev_gpu_busy_percent_get reads the SMU's accumulated counter.
    // Firmware samples internally every ~1ms and returns the average
    // since the last call. We call it at 1kHz, so each sample
    // represents ~1ms of GPU activity.
    uint32_t busy_pct = 0;
    rsmi_dev_gpu_busy_percent_get(0, &busy_pct);
    out.gpu_util_pct = static_cast<float>(busy_pct);
    last_gpu_util_.store(out.gpu_util_pct);

    // Junction temperature (most representative for throttling)
    int64_t temp_milli = 0;
    rsmi_dev_temp_metric_get(0, RSMI_TEMP_TYPE_JUNCTION,
                              RSMI_TEMP_CURRENT, &temp_milli);
    out.gpu_temp_c = static_cast<float>(temp_milli) / 1000.0f;
    last_gpu_temp_.store(out.gpu_temp_c);

    // VRAM usage (unified memory on UM790)
    uint64_t vram_used = 0;
    rsmi_dev_memory_usage_get(0, RSMI_MEM_TYPE_VRAM, &vram_used);
    out.vram_used_bytes = vram_used;
    last_vram_used_.store(vram_used);

    // GPU power (if available via RSMI)
    // Not all firmware exposes this; default to 0 if unavailable
    out.gpu_power_w = 0.0f;
#ifdef RSMI_PWR_AVERAGE
    uint64_t power_micro = 0;
    if (rsmi_dev_power_ave_get(0, 0, &power_micro) == RSMI_STATUS_SUCCESS) {
        out.gpu_power_w = static_cast<float>(power_micro) / 1000000.0f;
    }
#endif
}
```

### 5.4 `Instrumentor::sample_hailo()` — HailoRT Power Reader

```cpp
// ============================================================
// Instrumentor::sample_hailo()
// ============================================================
// Reads Hailo-8L power and derives utilization proxy.
// Called from the high-frequency sampler thread.
// Note: HailoRT power measurement has ~1100us integration time,
// so 1kHz sampling slightly oversamples. We average 4 consecutive
// readings to reduce noise.

void Instrumentor::sample_hailo(SampleRecord& out, HailoMonitor* monitor) {
    // Read power measurement (requires setup/teardown per sample)
    // In practice, we keep measurement running and just read.
    auto power_meas = monitor->get_power_measurement(false);
    if (power_meas) {
        out.hailo_power_w = power_meas->average_value;
    } else {
        out.hailo_power_w = 0.0f;
    }

    // Read temperature
    auto temp_result = monitor->get_chip_temperature();
    if (temp_result) {
        out.hailo_temp_c = static_cast<float>(temp_result->ts0_temperature);
    } else {
        out.hailo_temp_c = 0.0f;
    }

    // Derive utilization proxy from power draw
    // Hailo-8L: idle ~0.5W, active ~6W (at 100% NN core utilization)
    constexpr float HAILO8L_ACTIVE_POWER = 6.0f;
    constexpr float HAILO8L_IDLE_POWER = 0.5f;

    if (out.hailo_power_w > HAILO8L_IDLE_POWER) {
        float normalized = (out.hailo_power_w - HAILO8L_IDLE_POWER)
                         / (HAILO8L_ACTIVE_POWER - HAILO8L_IDLE_POWER);
        out.hailo_util_pct = std::clamp(normalized * 100.0f, 0.0f, 100.0f);
    } else {
        out.hailo_util_pct = 0.0f;
    }

    last_hailo_util_.store(out.hailo_util_pct);
    last_hailo_temp_.store(out.hailo_temp_c);
}
```

### 5.5 `Instrumentor::sample_memory()` — Memory Bandwidth Proxy

```cpp
// ============================================================
// Instrumentor::sample_memory()
// ============================================================
// Captures memory state. Uses multiple sources:
//   1. RSMI VRAM counters (fast, always available)
//   2. /proc/meminfo for system RAM (moderate overhead)
//   3. /proc/vmstat for page activity (only read once per second)

void Instrumentor::sample_memory(SampleRecord& out) {
    // 1. VRAM (already read in sample_gpu, just copy)
    out.vram_used_bytes = last_vram_used_.load();

    // 2. System RAM — read /proc/meminfo
    // Format: MemTotal: 65922816 kB
    //         MemAvailable: 52428800 kB
    // We parse only the MemAvailable line for speed.
    static FILE* meminfo = nullptr;
    if (!meminfo) meminfo = fopen("/proc/meminfo", "r");
    else rewind(meminfo);

    if (meminfo) {
        char line[256];
        while (fgets(line, sizeof(line), meminfo)) {
            if (strncmp(line, "MemAvailable:", 13) == 0) {
                unsigned long kb;
                sscanf(line, "MemAvailable: %lu", &kb);
                // Compute used = total - available
                // Total is cached from first read
                static unsigned long total_kb = 0;
                if (total_kb == 0) {
                    // Parse total from same file
                    rewind(meminfo);
                    char l2[256];
                    while (fgets(l2, sizeof(l2), meminfo)) {
                        if (strncmp(l2, "MemTotal:", 9) == 0) {
                            sscanf(l2, "MemTotal: %lu", &total_kb);
                            break;
                        }
                    }
                }
                out.sys_ram_used_bytes = (total_kb - kb) * 1024;
                last_sys_ram_used_.store(out.sys_ram_used_bytes);
                break;
            }
        }
    }
}

// Extended memory probe: read once per second for bandwidth proxy
struct ExtendedMemProbe {
    // Read /sys/class/drm/card0/gt_cur_freq_mhz for GPU memory clock
    // (proxy for memory controller activity)
    static float read_gpu_mem_clock_mhz() {
        FILE* f = fopen("/sys/class/drm/card0/device/pp_dpm_mclk", "r");
        if (!f) return 0.0f;
        char line[64];
        float freq = 0;
        // Format: "0: 400Mhz *"  (asterisk marks active level)
        while (fgets(line, sizeof(line), f)) {
            if (strchr(line, '*')) {
                sscanf(line, "%*d: %fMhz", &freq);
                break;
            }
        }
        fclose(f);
        return freq;
    }

    // Derive a memory pressure index [0.0, 1.0] from multiple signals
    static float compute_pressure_index(
        float gpu_mem_clock_mhz,
        uint64_t vmstat_page_delta,
        uint64_t vram_delta_bytes)
    {
        // Normalize each signal and combine
        float clock_ratio = std::min(gpu_mem_clock_mhz / 3200.0f, 1.0f);  // LPDDR5X max
        float page_pressure = std::min(
            static_cast<float>(vmstat_page_delta) / 100000.0f, 1.0f);
        float vram_pressure = std::min(
            static_cast<float>(vram_delta_bytes) / (512ULL * 1024 * 1024), 1.0f);

        // Weighted combination
        return clock_ratio * 0.4f + page_pressure * 0.3f + vram_pressure * 0.3f;
    }
};
```

### 5.6 `RecoveryValidator::verify_state()` — Latent Preservation Check

```cpp
// ============================================================
// RecoveryValidator::verify_state()
// ============================================================
// Validates that recovery did not corrupt inference state.
// Checks:
//   1. Latent tensor preservation (hash comparison)
//   2. Phase continuity (no unaccounted gaps)
//   3. Recovery latency bounds
//   4. Step counter integrity

class RecoveryValidator {
public:
    struct ValidationResult {
        bool passed = true;
        std::vector<std::string> failures;

        // Timing
        float recovery_latency_ms = 0.0f;   // Fault to resume
        float max_denoise_gap_ms = 0.0f;    // Largest gap in denoise markers

        // State integrity
        bool latent_preserved = true;
        bool step_continuity = true;
        int steps_lost = 0;

        // Comparison with baseline (non-recovery run)
        float output_similarity = 1.0f;      // SSIM or hash comparison
        bool output_matches_baseline = true;
    };

    // Validate a single recovery event
    ValidationResult verify_recovery(
        const EventRecord& recovery_start,
        const EventRecord& recovery_end,
        const std::vector<PhaseMarker>& all_markers,
        const std::vector<SampleRecord>& all_samples)
    {
        ValidationResult result;

        // 1. Recovery latency
        result.recovery_latency_ms =
            static_cast<float>(recovery_end.timestamp_ns - recovery_start.timestamp_ns)
            / 1'000'000.0f;

        if (result.recovery_latency_ms > MAX_RECOVERY_LATENCY_MS) {
            result.passed = false;
            result.failures.push_back(
                std::format("Recovery latency {:.1f}ms exceeds limit {:.1f}ms",
                    result.recovery_latency_ms, MAX_RECOVERY_LATENCY_MS));
        }

        // 2. Phase continuity: find largest gap in GPU_DENOISE markers
        // within [recovery_start, recovery_end + 1s]
        auto denoise_markers = filter_markers(all_markers, PhaseId::GPU_DENOISE,
                                               recovery_start.timestamp_ns,
                                               recovery_end.timestamp_ns + 1'000'000'000);

        float max_gap = 0;
        for (size_t i = 1; i < denoise_markers.size(); ++i) {
            float gap_ms = static_cast<float>(
                denoise_markers[i].timestamp_ns - denoise_markers[i-1].timestamp_ns)
                / 1'000'000.0f;
            if (gap_ms > max_gap) max_gap = gap_ms;
        }
        result.max_denoise_gap_ms = max_gap;

        if (max_gap > MAX_DENOISE_GAP_MS) {
            result.passed = false;
            result.failures.push_back(
                std::format("Denoise gap {:.1f}ms exceeds limit {:.1f}ms",
                    max_gap, MAX_DENOISE_GAP_MS));
        }

        // 3. Step continuity: denoise steps should increment by 1
        // (except across recovery where we expect a jump)
        for (size_t i = 1; i < denoise_markers.size(); ++i) {
            if (denoise_markers[i].boundary == PhaseBoundary::START &&
                denoise_markers[i-1].boundary == PhaseBoundary::START) {
                int step_delta = denoise_markers[i].sequence - denoise_markers[i-1].sequence;
                if (step_delta < 0) {
                    // Step went backward — state corruption
                    result.step_continuity = false;
                    result.passed = false;
                    result.failures.push_back(
                        std::format("Step regression: {} -> {}",
                            denoise_markers[i-1].sequence,
                            denoise_markers[i].sequence));
                }
                // Steps lost = any positive delta > 1 (we resumed mid-pipeline)
                if (step_delta > 1) {
                    result.steps_lost += (step_delta - 1);
                }
            }
        }

        // 4. Latent preservation: compare hash pre and post recovery
        // (This requires the workload adapter to provide latent snapshots)
        // The EventRecord for LATENT_SAVE and LATENT_RESTORE carry hashes.
        auto pre_hash = find_latent_hash(all_markers, recovery_start.timestamp_ns, -1); // before
        auto post_hash = find_latent_hash(all_markers, recovery_end.timestamp_ns, +1);  // after

        if (pre_hash && post_hash) {
            if (*pre_hash != *post_hash) {
                result.latent_preserved = false;
                result.passed = false;
                result.failures.push_back("Latent hash mismatch across recovery");
            }
        }

        return result;
    }

    // Full validation: run this after the stress test workload
    StressValidationReport validate_stress_suite(
        const std::vector<WorkloadResult>& stress_results,
        const WorkloadResult& baseline_result)
    {
        StressValidationReport report;

        for (const auto& result : stress_results) {
            for (const auto& recovery : result.recovery_events) {
                auto val = verify_recovery(
                    recovery.start_event,
                    recovery.end_event,
                    result.all_markers,
                    result.all_samples
                );
                report.per_recovery.push_back(val);
                if (!val.passed) report.total_failures++;
            }
        }

        // Compare stress output to baseline
        report.output_similarity = compute_ssim(
            baseline_result.output_image_path,
            stress_results.back().output_image_path
        );
        report.output_matches_baseline = (report.output_similarity > 0.99f);

        report.all_passed = (report.total_failures == 0)
                         && report.output_matches_baseline;

        return report;
    }

private:
    static constexpr float MAX_RECOVERY_LATENCY_MS = 5000.0f;
    static constexpr float MAX_DENOISE_GAP_MS = 500.0f;

    std::optional<uint64_t> find_latent_hash(
        const std::vector<PhaseMarker>& markers,
        uint64_t near_timestamp,
        int direction);  // -1 = before, +1 = after
};
```

### 5.7 High-Frequency Sampler Thread

```cpp
// ============================================================
// Instrumentor::hf_sampler_loop()
// ============================================================
// Runs at 1kHz (1ms period) during active measurement windows.
// Uses spin-wait + yield for sub-millisecond precision.
// Must never allocate memory (pre-allocated ring buffer only).

void Instrumentor::hf_sampler_loop(std::stop_token st) {
    // Pre-allocate sample slot
    SampleRecord sample{};

    // Track timing to maintain 1kHz cadence
    auto next_sample = high_resolution_clock::now();
    constexpr auto INTERVAL = microseconds(1000);

    // Extended memory probe state (once per second)
    auto last_vmstat_time = high_resolution_clock::now();
    SystemMemProbe::VmStat last_vmstat = SystemMemProbe::read_vmstat();

    while (!st.stop_requested()) {
        next_sample += INTERVAL;

        // ── Fill sample record ──
        sample.timestamp_ns = duration_cast<nanoseconds>(
            steady_clock::now().time_since_epoch()).count();

        if (cfg_.enable_gpu_sampling) {
            sample_gpu(sample);
        }

        if (cfg_.enable_hailo_sampling && hailo_monitor_) {
            sample_hailo(sample, hailo_monitor_.get());
        }

        if (cfg_.enable_memory_sampling) {
            sample_memory(sample);
        }

        // ── Extended memory probe (1Hz) ──
        auto now = high_resolution_clock::now();
        if (duration_cast<seconds>(now - last_vmstat_time).count() >= 1) {
            auto current_vmstat = SystemMemProbe::read_vmstat();
            // Compute deltas, store as custom event
            float pressure = SystemMemProbe::derive_pressure_index(
                last_vmstat, current_vmstat, 1.0f);
            // Store pressure in a dedicated event (not in SampleRecord)
            EventRecord evt;
            evt.timestamp_ns = sample.timestamp_ns;
            evt.type = EventType::MEM_PRESSURE;
            evt.detail.pressure_index = pressure;
            event_ring_.push(evt);

            last_vmstat = current_vmstat;
            last_vmstat_time = now;
        }

        // ── Write to ring buffer ──
        sample_ring_.push(sample);

        // ── Spin-wait precision sleep ──
        while (high_resolution_clock::now() < next_sample) {
            // Busy-wait for last ~50us, yield before that
            auto remaining = duration_cast<microseconds>(
                next_sample - high_resolution_clock::now());
            if (remaining.count() > 100) {
                std::this_thread::yield();
            }
            // Spin for final microsecond-scale accuracy
        }
    }
}
```

### 5.8 `FLUXText2ImgAdapter::run_measured()` — Example Workload

```cpp
// ============================================================
// FLUXText2ImgAdapter::run_measured()
// ============================================================
// Example of how a workload adapter instruments its pipeline.
// This is the ONLY code that knows the pipeline structure.

WorkloadResult FLUXText2ImgAdapter::run_measured(
    const WorkloadConfig& cfg,
    Instrumentor* instr,
    uint32_t iteration)
{
    WorkloadResult result;
    result.iteration = iteration;

    auto t_start = steady_clock::now();

    // ── Phase: Text Encoding (Hailo-8L) ──
    instr->mark_encode_start();

    // Run T5-XXL encoder on Hailo
    Ort::Value t5_embeddings = hailo_session_->Run(...);
    // Run CLIP-L encoder on Hailo
    Ort::Value clip_embeddings = hailo_session_->Run(...);

    instr->mark_encode_end();

    // ── Phase: VAE Encode (if img2img) or prepare latent ──
    // For text2img: initialize latent from noise
    Ort::Value latent = initialize_latent(cfg.random_seed + iteration);

    // Prepare timesteps
    auto timesteps = scheduler_->get_timesteps(cfg.model.num_steps);

    // ── Phase: Denoising Loop (GPU 780M) ──
    for (int step = 0; step < cfg.model.num_steps; ++step) {

        // Report step to watchdog (aligns with pipeline.cpp)
        watchdog_->report_step(step);

        // Scheduler computation (CPU, AVX-512)
        instr->mark_scheduler_start(step);
        float t = timesteps[step];
        auto sigma_params = scheduler_->compute_step_params(t);
        instr->mark_scheduler_end(step);

        // GPU denoising step
        instr->mark_denoise_step_start(step);

        Ort::Value noise_pred = gpu_session_->Run({
            latent, t5_embeddings, clip_embeddings, sigma_params
        });

        latent = scheduler_->step(noise_pred, latent, sigma_params);

        instr->mark_denoise_step_end(step);
    }

    // ── Phase: VAE Decode (GPU 780M) ──
    instr->mark_vae_start();

    // Decode latent to image
    Ort::Value image = vae_session_->Run({latent});

    instr->mark_vae_end();

    auto t_end = steady_clock::now();
    result.total_time_ms = duration_cast<microseconds>(t_end - t_start).count() / 1000.0f;

    // Save output image for validation
    std::string img_path = std::format("{}/flux_t2i_run_{:02d}_iter_{:02d}.png",
                                       output_dir_, workload_id_, iteration);
    save_image(image, img_path);
    result.output_image_path = img_path;

    // Capture peak values from instrumentor
    auto summary = instr->get_summary();
    result.gpu_peak_temp_c = summary.gpu_peak_temp;
    result.gpu_avg_util = summary.gpu_avg_util;
    result.hailo_avg_util = summary.hailo_avg_util;
    result.recovery_count = summary.recovery_events;

    return result;
}
```

---

## 6. Reproducibility Requirements

### 6.1 System State Requirements

| Parameter | Required Value | Verification Command | Failure Action |
|---|---|---|---|
| CPU Governor | `performance` | `cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor` | Set via `cpupower` |
| CPU Frequency | Fixed at max | `cpupower frequency-info` | Set via `cpupower frequency-set` |
| GPU Power Profile | `PROFILE_COMPUTE` | `rocm-smi --showprofile` | Set via `rocm-smi --setprofile` |
| GPU Performance Level | `HIGH` | `rocm-smi --showperflevel` | Set via `rocm-smi --setperflevel` |
| ROCm Version | `>= 6.0` | `rocm-smi --showdriverversion` | Log warning |
| HailoRT Version | `== 4.20.0` | `hailortcli --version` | Fail |
| HSA_XNACK | `1` | `echo $HSA_XNACK` | Fail |
| HSA_ENABLE_SDMA | `1` | `echo $HSA_ENABLE_SDMA` | Set and warn |
| GPU Processes | None except harness | `rocm-smi --showpidgpus` | Fail if > 0 |
| Swap | Disabled or unused | `free -m` (swap == 0) | Warn |
| GPU Junction Temp | < 55C at idle | `rocm-smi --showtemp` | Wait or fail |
| Hailo TS0 Temp | < 50C at idle | `hailortcli monitor` | Wait or fail |

### 6.2 Pre-Run Checklist

```bash
#!/bin/bash
# preflight.sh — Run before each benchmark suite

echo "=== UM790 Pro Benchmark Preflight ==="

# 1. CPU Governor
for cpu in /sys/devices/system/cpu/cpu[0-7]/cpufreq/scaling_governor; do
    echo performance > $cpu
done
echo "[OK] CPU governor set to performance"

# 2. GPU Power Profile
rocm-smi --setprofile 0  # 0 = COMPUTE on most firmware

# 3. Environment
test "$HSA_XNACK" = "1" || { echo "[FAIL] HSA_XNACK not set"; exit 1; }
test "$HSA_ENABLE_SDMA" = "1" || export HSA_ENABLE_SDMA=1

# 4. GPU idle
GPU_TEMP=$(rocm-smi --showtemp | grep "Temperature" | head -1 | awk '{print $2}')
if (( $(echo "$GPU_TEMP > 55" | bc -l) )); then
    echo "[WAIT] GPU temp ${GPU_TEMP}C, waiting for cooldown..."
    while (( $(echo "$GPU_TEMP > 55" | bc -l) )); do
        sleep 5
        GPU_TEMP=$(rocm-smi --showtemp | grep "Temperature" | head -1 | awk '{print $2}')
    done
fi
echo "[OK] GPU idle temperature: ${GPU_TEMP}C"

# 5. No competing GPU processes
if rocm-smi --showpidgpus | grep -q "[0-9]"; then
    echo "[WARN] Other GPU processes detected:"
    rocm-smi --showpidgpus
    read -p "Continue anyway? (y/N) " -n 1 -r
    [[ $REPLY =~ ^[Yy]$ ]] || exit 1
fi

# 6. Model files exist
for model in "$@"; do
    test -f "$model" || { echo "[FAIL] Model not found: $model"; exit 1; }
done

echo "[OK] Preflight passed. Starting benchmark."
```

### 6.3 Expected Variance Bounds

Based on the UM790 Pro hardware characteristics, the following variance bounds are expected for stabilized runs:

| Metric | Expected Mean | Acceptable StdDev | Max CV | Notes |
|---|---|---|---|---|
| End-to-end FLUX txt2img (12-step) | 8,000–12,000 ms | < 400 ms | < 5% | Depends on thermal state |
| End-to-end FLUX img2img (8-step) | 5,000–8,000 ms | < 300 ms | < 5% | |
| End-to-end Qwen edit (20-step) | 10,000–15,000 ms | < 500 ms | < 5% | Vision tower adds variance |
| End-to-end Sana Sprint (4-step) | 600–1,000 ms | < 60 ms | < 7% | Faster = more timer variance |
| Per-denoise-step GPU time | 500–800 ms | < 50 ms | < 7% | Per-step, averaged across steps |
| Hailo encode time | 30–80 ms | < 5 ms | < 7% | Very consistent (dataflow architecture) |
| Scheduler CPU time | 1–3 ms | < 0.5 ms | < 20% | Depends on AVX-512 clock |
| VAE decode time | 800–1,500 ms | < 100 ms | < 8% | Memory-bandwidth sensitive |
| Recovery latency | 2,000–4,000 ms | < 1,000 ms | < 30% | Wide range acceptable |
| GPU utilization (avg) | 65–80% | < 5% | < 7% | Target: 75–80% |
| Hailo utilization (avg) | 85–100% | < 5% | < 5% | Power-proxy derived |

**Reproducibility Criteria:**
- A benchmark run is considered **reproducible** if:
  - CV of end-to-end time across last 5 iterations is < 5%
  - No thermal throttling detected in any iteration
  - No recovery events in last 3 iterations
  - All system state requirements verified at start

- A benchmark **suite** is reproducible if all workloads meet individual criteria.

### 6.4 Variance Attribution Guide

If variance exceeds bounds, check in order:

1. **Thermal throttling** (> 95C junction): Increase inter-iteration gap
2. **Background processes**: Check `rocm-smi --showpidgpus`
3. **Memory bandwidth contention**: Check if other processes allocate large buffers
4. **HSA_XNACK page migration**: First few runs may be slower due to page migration
5. **HIP kernel cache misses**: Ensure warm-up runs are sufficient
6. **PCIe link quality**: Check `lspci -vv | grep Hailo` for link errors
7. **Watchdog false triggers**: Check if recovery events add latency variance

---

## Appendix A: Report Output Format

### A.1 JSON Report Structure

```json
{
  "suite": {
    "name": "UM790Pro_Full_Benchmark_v1",
    "timestamp": "2025-01-15T09:30:00Z",
    "duration_sec": 1847.3,
    "system": { "cpu": "Ryzen 9 7940HS", "rocm_version": "6.0.0", ... }
  },
  "workloads": [
    {
      "name": "FLUX_text2img",
      "category": "heavy",
      "aggregate": {
        "end_to_end_ms": { "mean": 9847.3, "p50": 9812.1, "p99": 10234.5, "stddev": 312.4, "cv": 0.032 },
        "encode_ms": { "mean": 45.2, "p50": 44.8, "stddev": 2.1 },
        "per_denoise_step_ms": { "mean": 743.1, "p50": 738.2, "stddev": 31.5 },
        "scheduler_ms": { "mean": 1.8, "p50": 1.7, "stddev": 0.3 },
        "vae_decode_ms": { "mean": 1123.4, "p50": 1118.2, "stddev": 45.6 },
        "gpu_util_avg": { "mean": 0.742, "p50": 0.751, "stddev": 0.032 },
        "hailo_util_avg": { "mean": 0.912, "p50": 0.918, "stddev": 0.021 },
        "gpu_peak_temp_c": { "mean": 87.3, "max": 91.2 },
        "recovery_events": 0
      },
      "iterations": [ { "iteration": 0, "total_time_ms": 9812.1, ... }, ... ]
    }
  ],
  "watchdog_validation": {
    "stress_tests_run": 7,
    "recoveries_validated": 7,
    "latent_preservation": true,
    "max_recovery_latency_ms": 3421.0,
    "avg_recovery_latency_ms": 2789.5,
    "all_passed": true
  },
  "comparisons": {
    "speedup_vs_baseline": { "Sana_Sprint": 11.2, "FLUX_img2img": 1.4, ... },
    "gpu_efficiency_ranking": [ "Sana_Sprint", "FLUX_img2img", "FLUX_text2img", "Qwen2.5_VL_edit" ]
  }
}
```

### A.2 Markdown Report Summary

The Markdown report provides a human-readable summary with tables and pass/fail indicators.

---

## Appendix B: Build & Run Instructions

```bash
# Build harness
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CXX_STANDARD=23 \
         -DENABLE_ROCM_SMI=ON \
         -DENABLE_HAILORT=ON
make -j$(nproc)

# Run full suite
./hq_benchmark --config ../configs/full_suite.toml

# Run single workload
./hq_benchmark --config ../configs/full_suite.toml --workload FLUX_text2img

# Run with fault injection (watchdog stress only)
./hq_benchmark --config ../configs/full_suite.toml --workload Watchdog_Stress_Test --inject

# Validate existing results (re-run report generation)
./hq_benchmark --validate ./benchmark_results/
```

---

## Appendix C: File Structure

```
hq_benchmark/
├── CMakeLists.txt
├── configs/
│   ├── full_suite.toml
│   ├── ci_suite.toml          # Shorter runs for CI
│   └── stress_only.toml
├── include/
│   ├── hq/
│   │   ├── benchmark_runner.hpp
│   │   ├── workload_adapter.hpp
│   │   ├── instrumentor.hpp
│   │   ├── binary_data_logger.hpp
│   │   ├── report_generator.hpp
│   │   ├── recovery_validator.hpp
│   │   ├── data_formats.hpp      # SampleRecord, PhaseMarker, etc.
│   │   ├── ring_buffer.hpp       # Lock-free ring buffer
│   │   ├── config_parser.hpp     # TOML/JSON parser
│   │   └── system_probes.hpp     # Thermal, memory, CPU probes
│   └── adapters/
│       ├── flux_text2img_adapter.hpp
│       ├── flux_img2img_adapter.hpp
│       ├── qwen_edit_adapter.hpp
│       ├── sana_sprint_adapter.hpp
│       └── watchdog_stress_adapter.hpp
├── src/
│   ├── main.cpp
│   ├── benchmark_runner.cpp
│   ├── instrumentor.cpp
│   ├── binary_data_logger.cpp
│   ├── report_generator.cpp
│   ├── recovery_validator.cpp
│   ├── config_parser.cpp
│   └── adapters/
│       ├── flux_text2img_adapter.cpp
│       ├── flux_img2img_adapter.cpp
│       ├── qwen_edit_adapter.cpp
│       ├── sana_sprint_adapter.cpp
│       └── watchdog_stress_adapter.cpp
└── scripts/
    ├── preflight.sh
    ├── parse_results.py       # Post-processing utility
    └── plot_timeline.py       # Generate utilization timeline plots
```

---

*Document Version: 1.0*
*Target Hardware: Minisforum UM790 Pro (Ryzen 9 7940HS + Radeon 780M + Hailo-8L)*
*Specification Date: 2025-01-15*
