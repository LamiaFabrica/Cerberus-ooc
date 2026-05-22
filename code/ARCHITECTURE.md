# UM790 Pipeline — Final Architecture Document (Round 3)

**Version:** 3.0.0  
**Date:** 2025-01-16  
**Status:** Canonical — all implementation must derive from this document  
**Target:** MinisForum UM790 Pro (Zen 4, AVX-512, ROCm 6.0+, HailoRT 4.20+)

---

## Table of Contents

1. [Component Inventory](#1-component-inventory)
2. [Component Contracts](#2-component-contracts)
3. [Interaction Diagrams](#3-interaction-diagrams)
4. [Data Flow](#4-data-flow)
5. [Decision Log](#5-decision-log)
6. [Specialist Assignments](#6-specialist-assignments)

---

## 1. Component Inventory

### 1.1 Canonical Components (KEEP — production code)

| Component | Header | Implementation | Purpose |
|-----------|--------|----------------|---------|
| UtilizationWatchdog | `include/hq/utilization_watchdog.hpp` | `src/utilization_watchdog.cpp` | 3-state per-step monitor with exponential backoff recovery |
| HailoMonitor | `include/hq/hailo_monitor.hpp` | `src/hailo_monitor.cpp` | Dual-indicator Hailo-8L monitoring (power + inference fusion) |
| PinnedStagingPool | `include/hq/pinned_staging.hpp` | `src/pinned_staging.cpp` | Double-buffered pinned staging for async CPU-to-GPU DMA |
| EmbeddingStagingManager | `include/hq/staging_manager.hpp` | `src/staging_manager.cpp` | Host-side pinned buffer pool (different purpose than PinnedStagingPool) |
| Pipeline | `include/hq/pipeline.hpp` | `src/pipeline_integration.cpp` | End-to-end orchestration: encode → denoise → decode |
| Feature Detection | `include/hq/cxx26_features.hpp` | — | C++26 capability macros |
| Entry Point | — | `src/main.cpp` | `um790_run` executable |
| Test Harness | — | `tests/test_harness.cpp` | GoogleTest suite |
| Build System | `CMakeLists.txt` | — | CMake 3.28+ configuration |

### 1.2 Files to DELETE

| File | Reason |
|------|--------|
| `include/hq/watchdog.hpp` | Superseded by `utilization_watchdog.hpp`. Different API (`StepSnapshot` vs `UtilizationSnapshot`, 4-state vs 3-state). |
| `src/watchdog.cpp` | Superseded by `utilization_watchdog.cpp`. Old cooldown-based logic replaced by exponential backoff. |

### 1.3 Files to MODIFY

| File | Changes Required |
|------|-----------------|
| `include/hq/pipeline.hpp` | Switch `#include` from `watchdog.hpp` to `utilization_watchdog.hpp`; update `on_watchdog_recovery_` signature; update `PipelineConfig` watchdog fields; add `query_gpu_utilization_()` declaration |
| `src/pipeline_integration.cpp` | Replace OLD watchdog API usage; replace GPU utilization placeholder; replace `denoise_step_` skeleton; replace `encode_prompt_` dummy; replace `decode_latents_` checkerboard; add `query_gpu_utilization_()` implementation |
| `tests/test_harness.cpp` | Switch from OLD `StepSnapshot` / `WatchdogResult` API to NEW `UtilizationSnapshot` / `RecoveryAction` API |
| `CMakeLists.txt` | Replace `src/watchdog.cpp` with `src/utilization_watchdog.cpp` in source list; replace `include/hq/watchdog.hpp` with `include/hq/utilization_watchdog.hpp` in header list; add `src/pinned_staging.cpp` to sources |
| `include/hq/utilization_watchdog.hpp` | Add `thermal_throttle_threshold_c` to `WatchdogConfig`; add temperature parameter to `evaluate_device` |
| `src/utilization_watchdog.cpp` | Add thermal throttling guard in `evaluate_device`; pass temperature from snapshots |
| `include/hq/staging_manager.hpp` | Rename `StagingError` enum to `HostStagingError` to avoid collision with `pinned_staging.hpp` |
| `src/staging_manager.cpp` | Update all references from `StagingError` to `HostStagingError` |

### 1.4 File Status Summary

```
include/hq/
    utilization_watchdog.hpp    KEEP + MODIFY (add thermal awareness)
    watchdog.hpp                DELETE
    hailo_monitor.hpp           KEEP (no changes)
    pinned_staging.hpp          KEEP (no changes)
    staging_manager.hpp         KEEP + MODIFY (rename StagingError)
    pipeline.hpp                KEEP + MODIFY (new watchdog API)
    cxx26_features.hpp          KEEP (no changes)

src/
    utilization_watchdog.cpp    KEEP + MODIFY (add thermal guard)
    watchdog.cpp                DELETE
    hailo_monitor.cpp           KEEP (no changes — is_open() already correct)
    pinned_staging.cpp          KEEP (no changes)
    staging_manager.cpp         KEEP + MODIFY (rename StagingError)
    pipeline_integration.cpp    KEEP + MODIFY (4 placeholder replacements)
    main.cpp                    KEEP (no changes)

tests/
    test_harness.cpp            KEEP + MODIFY (new watchdog API)

CMakeLists.txt                  KEEP + MODIFY (source/header list update)
```

---

## 2. Component Contracts

### 2.1 UtilizationWatchdog (`include/hq/utilization_watchdog.hpp`)

**Namespace:** `hq`  
**Thread Safety:** All public methods are `std::mutex`-protected. Safe for concurrent calls.

#### 2.1.1 Types

```cpp
enum class ComputeUnit : std::uint8_t { GPU_780M = 0, HAILO_8L = 1 };

enum class WatchdogState : std::uint8_t { NORMAL = 0, WARNING = 1, CRITICAL = 2 };

enum class RecoveryResult : std::uint8_t { SUCCESS = 0, PARTIAL = 1, FATAL = 2 };

struct UtilizationSnapshot {
    ComputeUnit   device;          // Which accelerator
    std::uint32_t step;            // Denoising step index
    float         utilization;     // Utilization percentage [0, 100]
    float         temperature;     // Die temperature (Celsius)
    float         power_watts;     // Power draw in watts
    bool          device_healthy;  // False if the driver reported an error
};

struct RecoveryAction {
    RecoveryResult result;        // Outcome of the recovery attempt
    ComputeUnit    device;        // Which accelerator was recovered
    std::uint32_t  step;          // Step number where recovery triggered
    float          util_at_fault; // Utilization that caused the trigger
    std::string    reason;        // Human-readable description
};

struct WatchdogConfig {
    float         gpu_low_threshold = 60.0f;        // % below => WARNING
    float         gpu_critical_threshold = 40.0f;   // % below => CRITICAL
    float         hailo_low_threshold = 60.0f;      // % below => WARNING
    float         hailo_critical_threshold = 40.0f; // % below => CRITICAL
    std::uint32_t consecutive_threshold = 8;        // steps before recovery
    std::uint32_t max_recoveries = 10;              // max before giving up
    float         backoff_base_ms = 100.0f;         // base recovery delay
    float         backoff_max_ms = 30000.0f;        // cap at 30 s
    float         thermal_throttle_threshold_c = 85.0f; // NEW: temp above => thermal, skip recovery
};

struct WatchdogStatistics {
    std::uint32_t total_steps = 0;
    std::uint32_t gpu_recovery_count = 0;
    std::uint32_t hailo_recovery_count = 0;
    std::uint32_t gpu_consecutive_low = 0;
    std::uint32_t hailo_consecutive_low = 0;
    WatchdogState gpu_state = WatchdogState::NORMAL;
    WatchdogState hailo_state = WatchdogState::NORMAL;
    float         last_gpu_util = 0.0f;
    float         last_hailo_util = 0.0f;
};

using RecoveryCallback =
    std::function<std::expected<RecoveryResult, std::string>(
        ComputeUnit unit, std::uint32_t step, float util_at_fault)>;

using AlertCallback =
    std::function<void(ComputeUnit unit, std::uint32_t step, float util,
                       const std::string& message)>;
```

#### 2.1.2 Class Signature

```cpp
class UtilizationWatchdog {
public:
    explicit UtilizationWatchdog(WatchdogConfig cfg,
                                 RecoveryCallback on_recovery,
                                 AlertCallback on_alert = nullptr);
    ~UtilizationWatchdog() = default;

    UtilizationWatchdog(const UtilizationWatchdog&) = delete;
    UtilizationWatchdog& operator=(const UtilizationWatchdog&) = delete;
    UtilizationWatchdog(UtilizationWatchdog&&) = default;
    UtilizationWatchdog& operator=(UtilizationWatchdog&&) = default;

    [[nodiscard("recovery action must be checked")]]
    std::optional<RecoveryAction> step(std::uint32_t step_num,
                                       const UtilizationSnapshot& gpu_snap,
                                       const UtilizationSnapshot& hailo_snap);

    [[nodiscard]] WatchdogState get_gpu_state() const noexcept;
    [[nodiscard]] WatchdogState get_hailo_state() const noexcept;
    [[nodiscard]] WatchdogStatistics get_stats() const noexcept;

    void reset_counters() noexcept;
    void reset_all() noexcept;

private:
    WatchdogConfig   cfg_;
    RecoveryCallback on_recovery_;
    AlertCallback    on_alert_;
    mutable std::mutex mutex_;
    WatchdogStatistics stats_{};
    std::uint32_t gpu_consecutive_ = 0;
    std::uint32_t hailo_consecutive_ = 0;
    bool          gpu_in_recovery_ = false;
    bool          hailo_in_recovery_ = false;
    std::uint32_t gpu_recovery_count_ = 0;
    std::uint32_t hailo_recovery_count_ = 0;

    [[nodiscard]] std::optional<RecoveryAction> evaluate_device(
        ComputeUnit unit, float util, float temperature,         // <-- temperature ADDED
        float threshold, float critical_threshold,
        std::uint32_t& consecutive, bool& in_recovery,
        std::uint32_t& recovery_count, std::uint32_t step_num);

    [[nodiscard]] std::expected<RecoveryResult, std::string>
    trigger_recovery(ComputeUnit unit, std::uint32_t step, float util);

    [[nodiscard]] float compute_backoff_ms(std::uint32_t recovery_count) const noexcept;

    void log_state_change(ComputeUnit unit, WatchdogState old_state,
                          WatchdogState new_state, float util, std::uint32_t step);

    [[nodiscard]] static constexpr const char* state_name(WatchdogState s) noexcept;
    [[nodiscard]] static constexpr const char* device_name(ComputeUnit u) noexcept;
};
```

#### 2.1.3 State Machine (per device)

```
                    util >= low_threshold
              +---------------------------+
              |                           |
              v                           |
    +------------------+                  |
    |     NORMAL       |<-----------------+
    +------------------+  reset consecutive
              |
              | util < low_threshold AND util >= critical_threshold
              v
    +------------------+     consecutive >= threshold      +------------------+
    |     WARNING      |---------------------------------->| trigger_recovery |
    |  (count steps)   |     AND NOT in_recovery           |  with backoff    |
    +------------------+                                   +------------------+
              |                                                  |
              | util < critical_threshold                        | callback returns
              v                                                  v
    +------------------+                                   +------------------+
    |    CRITICAL      |---------------------------------->| clear in_recovery|
    | (immediate recov)|                                   |                  |
    +------------------+                                   +------------------+
```

#### 2.1.4 Thermal Guard Rule (NEW)

Before triggering recovery in `evaluate_device()`, check:

```cpp
// If the device is thermally throttling, do NOT recover.
// Low utilization during thermal throttling is expected behavior,
// not a scheduling fault. Recovering would be counterproductive.
if (temperature > cfg_.thermal_throttle_threshold_c) {
    std::string msg = std::format(
        "[{}] step={} util={:.1f}% — THERMAL THROTTLING detected "
        "(temp={:.1f}C > threshold={:.1f}C), skipping recovery",
        device_name(unit), step_num, util,
        temperature, cfg_.thermal_throttle_threshold_c);
    std::print("[watchdog] {}\n", msg);
    if (on_alert_) {
        on_alert_(unit, step_num, util, msg);
    }
    return std::nullopt;  // Do NOT trigger recovery
}
```

#### 2.1.5 Invariants

- `gpu_critical_threshold < gpu_low_threshold` (enforced at construction, warn if violated)
- `hailo_critical_threshold < hailo_low_threshold` (enforced at construction, warn if violated)
- `0.0f <= backoff_base_ms <= backoff_max_ms`
- `consecutive_threshold > 0`
- `max_recoveries > 0`
- `0.0f <= thermal_throttle_threshold_c <= 125.0f` (physical silicon limit)
- `in_recovery` flag prevents re-entrant recovery storms
- Recovery count is per-device and monotonically increasing within a session
- Backoff delay = `min(backoff_max_ms, backoff_base_ms * 2^(recovery_count - 1))`

---

### 2.2 HailoMonitor (`include/hq/hailo_monitor.hpp`)

**Namespace:** `hq`  
**Thread Safety:** `open()`, `sample()`, `hard_reset()`, `close()` are NOT thread-safe. Call from a single thread or externally synchronize. `is_open()`, getters/setters for tunables are safe for concurrent reads.

#### 2.2.1 Types

```cpp
// Constants
inline constexpr float HAILO8L_IDLE_POWER_W = 0.5f;
inline constexpr float HAILO8L_ACTIVE_POWER_W = 6.0f;
inline constexpr float HAILO8L_MAX_TDP_W = 8.65f;
inline constexpr float HAILO8L_FUSED_WEIGHT_POWER = 0.5f;
inline constexpr float HAILO8L_FUSED_WEIGHT_INFERENCE = 0.5f;
inline constexpr float HAILO8L_DMA_STALL_POWER_THRESHOLD = 70.0f;
inline constexpr float HAILO8L_DMA_STALL_INFERENCE_THRESHOLD = 30.0f;
inline constexpr std::size_t HAILO8L_EXPECTED_INFERENCES_PER_SEC = 60;

enum class HailoErrorCode : int {
    Ok = 0, DeviceNotFound, DeviceOpenFailed, AlreadyOpen, NotOpen,
    PowerReadFailed, TemperatureReadFailed, InfoReadFailed,
    InferenceCountFailed, SensorMismatch, PcieResetFailed,
    InvalidArgument, InternalError,
};

struct HailoError { HailoErrorCode code; std::string message; ... };
struct HailoStats {
    float nn_core_utilization;    // Fused utilization [0, 100]
    float power_indicator;        // Raw power-based [%]
    float inference_indicator;    // Raw inference-based [%]
    float power_watts;            // Watts
    float temperature_celsius;    // Celsius
    std::uint64_t inferences_count;
    std::uint64_t inference_delta;
    bool device_healthy;
    std::chrono::steady_clock::time_point timestamp;
};
```

#### 2.2.2 Class Signature

```cpp
class HailoMonitor {
public:
    HailoMonitor();
    ~HailoMonitor() noexcept;
    HailoMonitor(HailoMonitor&& other) noexcept;
    HailoMonitor& operator=(HailoMonitor&& other) noexcept;
    HailoMonitor(const HailoMonitor&) = delete;
    HailoMonitor& operator=(const HailoMonitor&) = delete;

    [[nodiscard]] std::expected<void, HailoError> open(const std::string& device_id = "");
    [[nodiscard]] std::expected<HailoStats, HailoError> sample();
    [[nodiscard]] std::expected<void, HailoError> hard_reset();
    void close() noexcept;
    [[nodiscard]] bool is_open() const noexcept;       // CORRECT — matches header
    [[nodiscard]] const std::string& device_id() const noexcept;

    // Tunable thresholds
    void set_dma_stall_power_threshold(float pct) noexcept;
    void set_dma_stall_inference_threshold(float pct) noexcept;
    void set_expected_inferences_per_sec(std::size_t rate) noexcept;
    void set_inference_weight(float w) noexcept;
    void set_power_weight(float w) noexcept;

    [[nodiscard]] float dma_stall_power_threshold() const noexcept;
    [[nodiscard]] float dma_stall_inference_threshold() const noexcept;
    [[nodiscard]] std::size_t expected_inferences_per_sec() const noexcept;
    [[nodiscard]] float inference_weight() const noexcept;
    [[nodiscard]] float power_weight() const noexcept;
};
```

#### 2.2.3 Invariants

- `is_open()` returns `true` iff `device_` handle is non-null
- `sample()` returns `HailoErrorCode::NotOpen` if called before `open()`
- `hard_reset()` resets inference delta tracking (`prev_inferences_ = 0`, `have_prev_inferences_ = false`)
- `nn_core_utilization` is always in `[0.0f, 100.0f]` (clamped)
- Sensor mismatch: `power_util < 5% && inference_util > 40%` is physically impossible — treated as error
- Sensor divergence: `|power_util - inference_util| > 50%` triggers warning
- DMA stall: `power > dma_stall_power_threshold_ && inference < dma_stall_inference_threshold_` => down-weight power to 0.3, up-weight inference to 0.7

---

### 2.3 PinnedStagingPool (`include/hq/pinned_staging.hpp`)

**Namespace:** `hq`  
**Thread Safety:** `acquire_host_buffer()`, `stage_to_gpu()`, `get_gpu_buffer()`, `synchronize_step()` must be called sequentially (producer-consumer pattern). `is_ready()` and queries are thread-safe for reads.

#### 2.3.1 Types

```cpp
enum class StagingError : std::uint8_t {
    Ok = 0, HipError, InvalidSize, InvalidSlot, TransferInProgress, NotInitialized,
};

struct StagingErrorInfo {
    StagingError code{StagingError::Ok};
    std::string message;
    int hip_error_code{0};
    [[nodiscard]] constexpr bool is_ok() const noexcept { return code == StagingError::Ok; }
};
```

#### 2.3.2 Class Signature (template)

```cpp
template<typename T = float>
class PinnedStagingPool {
public:
    PinnedStagingPool(std::size_t embedding_bytes, int num_slots = 2);
    ~PinnedStagingPool();

    PinnedStagingPool(const PinnedStagingPool&) = delete;
    PinnedStagingPool& operator=(const PinnedStagingPool&) = delete;
    PinnedStagingPool(PinnedStagingPool&&) noexcept;
    PinnedStagingPool& operator=(PinnedStagingPool&&) noexcept;

    [[nodiscard]] std::expected<std::span<T>, StagingErrorInfo> acquire_host_buffer(std::uint32_t step);
    [[nodiscard]] std::expected<void, StagingErrorInfo> stage_to_gpu(std::uint32_t step);
    [[nodiscard]] std::expected<T*, StagingErrorInfo> get_gpu_buffer(std::uint32_t step);
    [[nodiscard]] std::expected<void, StagingErrorInfo> synchronize_step(std::uint32_t step);
    [[nodiscard]] bool is_ready(std::uint32_t step) const noexcept;

    [[nodiscard]] std::size_t embedding_bytes() const noexcept;
    [[nodiscard]] int num_slots() const noexcept;
    [[nodiscard]] bool initialized() const noexcept;
};

using T5StagingPool = PinnedStagingPool<float>;
using CLIPStagingPool = PinnedStagingPool<float>;
```

#### 2.3.3 Invariants

- `embedding_bytes > 0` (constructor validates)
- `num_slots >= 1` (constructor validates)
- `slot_for_step(step) = step % num_slots` (round-robin)
- `acquire_host_buffer()` drains in-flight slot before returning (blocks via `hipEventSynchronize`)
- `stage_to_gpu()` submits `hipMemcpyAsync` + `hipEventRecord` on the pool's stream
- `get_gpu_buffer()` returns non-null only after the transfer event reports complete
- Destructor releases all HIP resources (noexcept via `free_all()`)

---

### 2.4 EmbeddingStagingManager (`include/hq/staging_manager.hpp`)

**Namespace:** `hq`  
**Thread Safety:** `acquire()`, `release()`, `copy_in()` are `std::mutex`-protected.

#### 2.4.1 Types (RENAMED)

```cpp
// RENAMED from StagingError -> HostStagingError to avoid collision with pinned_staging.hpp
enum class HostStagingError : std::uint8_t {
    OutOfMemory = 0, InvalidSize = 1, TransferFailed = 2,
    PoolExhausted = 3, NotAligned = 4, Unknown = 5,
};

struct StagingBuffer {
    std::span<std::byte>       data;
    std::span<const std::byte> cdata;
    std::size_t                capacity;
    std::size_t                used;
};

struct StagingConfig {
    std::size_t buffer_count{8};
    std::size_t buffer_size_bytes{64ULL * 1024 * 1024};
    bool        pinned{true};
    std::size_t alignment{256};
};
```

#### 2.4.2 Class Signature

```cpp
class EmbeddingStagingManager {
public:
    explicit EmbeddingStagingManager(StagingConfig cfg);
    ~EmbeddingStagingManager();
    EmbeddingStagingManager(const EmbeddingStagingManager&) = delete;
    EmbeddingStagingManager& operator=(const EmbeddingStagingManager&) = delete;
    EmbeddingStagingManager(EmbeddingStagingManager&&) noexcept;
    EmbeddingStagingManager& operator=(EmbeddingStagingManager&&) noexcept;

    [[nodiscard]] std::expected<StagingBuffer, HostStagingError> acquire();
    void release(const StagingBuffer& buf) noexcept;
    [[nodiscard]] std::expected<std::size_t, HostStagingError>
        copy_in(StagingBuffer& dst, std::span<const std::byte> src);
    [[nodiscard]] std::size_t total_capacity() const noexcept;
    [[nodiscard]] std::size_t available_count() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
```

#### 2.4.3 Invariants

- `release()` must be called for every successful `acquire()` to prevent pool leakage
- `copy_in()` truncates to `min(src.size(), dst.capacity)` — never overruns
- `available_count()` + (buffers currently acquired) == `cfg.buffer_count`
- Pool is pre-allocated at construction time; `acquire()` never allocates

---

### 2.5 Pipeline (`include/hq/pipeline.hpp`)

**Namespace:** `hq`  
**Thread Safety:** NOT thread-safe. One `Pipeline` instance per thread.

#### 2.5.1 Types

```cpp
enum class PipelineError : std::uint32_t {
    Ok = 0, InvalidRequest, WatchdogRecoveryFailed, HailoNotAvailable,
    HailoTimeout, HailoThermal, GPUOutOfMemory, ONNXSessionLoadFailed,
    ONNXRunFailed, StagingPoolExhausted, LatencyBudgetExceeded,
    RecoveryTooManyAttempts, InvalidModelPath, ShutdownInProgress, Unknown,
};

struct GenerationRequest {
    std::string prompt;
    std::uint32_t width{512};
    std::uint32_t height{512};
    std::uint32_t num_steps{20};
    float       guidance_scale{7.5f};
    int64_t     seed{-1};
};

struct GeneratedImage {
    std::vector<std::uint8_t> pixels;  // RGBA8
    std::uint32_t width;
    std::uint32_t height;
    float         generation_time_ms;
};

struct PipelineStats {
    std::uint64_t generations_completed{0};
    std::uint64_t generations_failed{0};
    std::uint32_t watchdog_recoveries{0};
    double        avg_generation_ms{0.0};
    double        avg_gpu_utilization{0.0};
    double        avg_hailo_utilization{0.0};
    std::uint64_t total_steps_executed{0};
};

struct PipelineConfig {
    // Watchdog thresholds (match utilization_watchdog.hpp fields)
    float    watchdog_gpu_low_threshold{60.0f};        // MATCHES WatchdogConfig::gpu_low_threshold
    float    watchdog_gpu_critical_threshold{40.0f};   // MATCHES WatchdogConfig::gpu_critical_threshold
    float    watchdog_hailo_low_threshold{60.0f};      // MATCHES WatchdogConfig::hailo_low_threshold
    float    watchdog_hailo_critical_threshold{40.0f}; // MATCHES WatchdogConfig::hailo_critical_threshold
    std::uint32_t watchdog_consecutive_threshold{8};   // MATCHES WatchdogConfig::consecutive_threshold
    std::uint32_t watchdog_max_recoveries{10};         // MATCHES WatchdogConfig::max_recoveries
    float    watchdog_backoff_base_ms{100.0f};         // MATCHES WatchdogConfig::backoff_base_ms
    float    watchdog_backoff_max_ms{30000.0f};        // MATCHES WatchdogConfig::backoff_max_ms
    float    watchdog_thermal_threshold_c{85.0f};      // MATCHES WatchdogConfig::thermal_throttle_threshold_c
    bool     enable_watchdog{true};

    // Staging
    std::uint32_t staging_buffer_count{8};
    std::uint32_t staging_buffer_size_mb{64};

    // Model paths
    std::filesystem::path text_encoder_onnx;
    std::filesystem::path unet_onnx;
    std::filesystem::path vae_decoder_onnx;

    // Inference
    std::uint32_t gpu_batch_size{1};
    std::uint32_t hailo_batch_size{1};
    bool          use_hailo_text_encoder{true};

    // Recovery
    std::uint32_t max_recovery_attempts{3};
};
```

#### 2.5.2 Class Signature

```cpp
class Pipeline {
public:
    explicit Pipeline(const PipelineConfig& cfg);
    ~Pipeline();
    Pipeline(Pipeline&&) noexcept;
    Pipeline& operator=(Pipeline&&) noexcept;
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    [[nodiscard]] std::expected<GeneratedImage, PipelineError>
        generate(const GenerationRequest& req);

    [[nodiscard]] std::vector<std::expected<GeneratedImage, PipelineError>>
        generate_batch(const std::vector<GenerationRequest>& requests);

    [[nodiscard]] PipelineStats get_stats() const;
    void shutdown() noexcept;

private:
    [[nodiscard]] bool initialize_onnx_sessions_();

    [[nodiscard]] std::expected<void*, PipelineError>
        denoise_step_(std::uint32_t step, void* latents,    // <-- MUST call Ort::Session::Run
                      const std::vector<float>& embeddings, std::uint32_t num_steps);

    void on_watchdog_recovery_(ComputeUnit unit, std::uint32_t step,   // <-- UPDATED signature
                               float util_at_fault);

    [[nodiscard]] std::expected<std::vector<float>, PipelineError>
        encode_prompt_(const std::string& prompt);                      // <-- MUST call hailo_session->Run()

    [[nodiscard]] std::expected<GeneratedImage, PipelineError>
        decode_latents_(const std::vector<float>& latents,              // <-- MUST call vae_session->Run()
                        std::uint32_t out_width, std::uint32_t out_height);

    [[nodiscard]] float query_gpu_utilization_();                       // <-- NEW: query ROCm SMI
    [[nodiscard]] float query_gpu_temperature_();                       // <-- NEW: for thermal awareness

    [[nodiscard]] static bool validate_model_path_(const std::filesystem::path& p) noexcept;
    [[nodiscard]] static std::string error_string_(PipelineError e);

    PipelineConfig cfg_;
    std::unique_ptr<UtilizationWatchdog>      watchdog_;
    std::unique_ptr<HailoMonitor>             hailo_monitor_;
    std::unique_ptr<EmbeddingStagingManager>  staging_manager_;

    class OrtState;
    std::unique_ptr<OrtState> ort_state_;

    PipelineStats stats_{};
    std::vector<float> latent_checkpoint_;
    bool shutdown_{false};
    bool recovery_in_progress_{false};
    std::uint32_t recovery_attempts_{0};
};
```

---

## 3. Interaction Diagrams

### 3.1 Normal Denoising Step

```
Pipeline::generate()
|
+-- [LOOP for step = 0 .. num_steps-1]
|   |
|   +-- Pipeline::denoise_step_(step, latents, embeddings, num_steps)
|   |   |
|   |   +-- ort_state_->gpu_session->Run(...)           // REAL ONNX inference
|   |   +-- Apply scheduler step (DDIM/Euler)
|   |   +-- Return updated latents pointer
|   |
|   +-- hailo_monitor_->sample()                        // Hailo telemetry
|   |   +-- Return HailoStats (nn_core_utilization, temperature_celsius, ...)
|   |
|   +-- Pipeline::query_gpu_utilization_()              // NEW: ROCm SMI
|   |   +-- rsmi_dev_gpu_busy_percent_get(0, &pct)      // REAL GPU util
|   |   +-- Return float utilization [0, 100]
|   |
|   +-- Pipeline::query_gpu_temperature_()              // NEW: ROCm SMI
|   |   +-- rsmi_dev_temp_metric_get(0, RSMI_TEMP_TYPE_EDGE, &temp)
|   |   +-- Return float temperature (Celsius)
|   |
|   +-- Construct UtilizationSnapshot gpu_snap {        // NEW API
|   |       .device = GPU_780M,
|   |       .step = step,
|   |       .utilization = gpu_util,
|   |       .temperature = gpu_temp,                     // NEW: thermal data
|   |       .power_watts = gpu_power,
|   |       .device_healthy = true
|   |   }
|   +-- Construct UtilizationSnapshot hailo_snap {      // NEW API
|   |       .device = HAILO_8L,
|   |       .step = step,
|   |       .utilization = hailo_stats.nn_core_utilization,
|   |       .temperature = hailo_stats.temperature_celsius,
|   |       .power_watts = hailo_stats.power_watts,
|   |       .device_healthy = hailo_stats.device_healthy
|   |   }
|   |
|   +-- watchdog_->step(step, gpu_snap, hailo_snap)     // NEW API
|   |   +-- evaluate_device(GPU, gpu_util, gpu_temp, ...) // NEW: temp passed
|   |   |   +-- [Thermal guard] if temp > 85C: return nullopt (skip recovery)
|   |   |   +-- [State machine transitions]
|   |   +-- evaluate_device(HAILO, hailo_util, hailo_temp, ...)
|   |   +-- Return std::optional<RecoveryAction>
|   |
|   +-- if (recovery_action.has_value())
|   |   +-- Handle recovery (see 3.2)
|   |
|   +-- stats_.total_steps_executed++
|
+-- [END LOOP]
```

### 3.2 Recovery Event

```
watchdog_->step() returns RecoveryAction{result, device, step, util, reason}
|
+-- Pipeline checks recovery_attempts_ < cfg_.max_recovery_attempts
|   +-- If exceeded: return PipelineError::RecoveryTooManyAttempts
|
+-- Pipeline::on_watchdog_recovery_(device, step, util_at_fault)   // NEW signature
|   |
|   +-- recovery_in_progress_ = true
|   +-- recovery_attempts_++
|   |
|   +-- Save latents: latent_checkpoint_ = latents (deep copy)
|   |
|   +-- Tear down sessions:
|   |   +-- ort_state_->gpu_session.reset()
|   |   +-- if (device == HAILO_8L): ort_state_->hailo_session.reset()
|   |
|   +-- std::this_thread::sleep_for(backoff_ms)   // Exponential backoff ALREADY
|   |                                              // applied inside watchdog::trigger_recovery()
|   |
|   +-- Rebuild sessions:
|   |   +-- Pipeline::initialize_onnx_sessions_()
|   |       +-- Ort::Session(env, unet_onnx, gpu_options)    // Rebuild GPU
|   |       +-- Ort::Session(env, text_encoder_onnx, ...)    // Rebuild Hailo if needed
|   |       +-- Ort::Session(env, vae_decoder_onnx, vae_options)
|   |
|   +-- Restore latents: latents = latent_checkpoint_
|   |
|   +-- recovery_in_progress_ = false
|   +-- stats_.watchdog_recoveries++
|
+-- Resume denoising loop from step + 1
```

### 3.3 Shutdown Sequence

```
Pipeline::shutdown() noexcept
|
+-- shutdown_ = true
|
+-- Release ONNX sessions (deterministic order):
|   +-- ort_state_->gpu_session.reset()
|   +-- ort_state_->hailo_session.reset()
|   +-- ort_state_->vae_session.reset()
|
+-- Subsystems clean up via destructors:
|   +-- watchdog_.reset()          -> UtilizationWatchdog dtor (trivial, mutex auto-releases)
|   +-- hailo_monitor_->close()    -> HailoMonitor releases HailoRT device
|   +-- hailo_monitor_.reset()     -> HailoMonitor dtor calls close() again (idempotent)
|   +-- staging_manager_.reset()   -> EmbeddingStagingManager dtor frees buffer pool
|
+-- ort_state_.reset()             -> Ort::Env cleanup
|
+-- Print final statistics
```

---

## 4. Data Flow

### 4.1 Embeddings Path: Hailo → Staging → GPU

```
[Text Prompt: "a cat in space"]
    |
    v
[Pipeline::encode_prompt_()]
    |
    +-- Tokenize prompt -> token_ids[77]
    |
    +-- Create Ort::Value input tensor (int64[1,77] on CPU)
    |
    +-- ort_state_->hailo_session->Run(...)
    |       Hailo-8L M.2 via ONNX Runtime Hailo EP
    |       Output: float[1, 77, 768] — CLIP text embeddings
    |
    +-- Copy output tensor to std::vector<float> embeddings
    |
    v
[EmbeddingStagingManager::acquire()] -> StagingBuffer (host pinned memory)
    |
    +-- EmbeddingStagingManager::copy_in(staging_buf, embeddings_bytes)
    |       host-side memcpy into pooled buffer
    |
    v
[PinnedStagingPool::acquire_host_buffer(step)]
    |
    +-- CPU writes embedding data into hipHostMalloc'd pinned buffer
    |
    +-- PinnedStagingPool::stage_to_gpu(step)
    |       hipMemcpyAsync(host_ptr -> device_ptr, stream_)
    |       hipEventRecord(event, stream_)
    |
    +-- [GPU denoising loop continues...]
    |
    +-- PinnedStagingPool::get_gpu_buffer(step)
    |       hipEventQuery(event) -> returns device_ptr when transfer complete
    |
    v
[GPU UNet session input: embeddings as device pointer]
```

### 4.2 Latents Path: GPU → Checkpoint → Restore

```
[Initial latent tensor: random normal noise]
    |
    v
[GPU denoising loop]
    |
    +-- Each step: latents = scheduler_step(latents, noise_pred, timestep)
    |       Operates on GPU device memory via ONNX Runtime ROCm EP
    |
    +-- [If watchdog triggers recovery]
    |   |
    |   +-- latent_checkpoint_ = latents (deep copy from GPU to host)
    |   |       via hipMemcpy(device -> host) or Ort tensor extraction
    |   |
    |   +-- Tear down GPU session (releases all GPU memory)
    |   |
    |   +-- Rebuild GPU session (fresh Ort::Session)
    |   |
    |   +-- latents = latent_checkpoint_ (copy back to new GPU allocation)
    |   |       via hipMemcpy(host -> device)
    |   |
    |   v
    +-- [Resume denoising from same step with restored latents]
    |
    v
[Final latent tensor after num_steps iterations]
    |
    v
[Pipeline::decode_latents_()]
    +-- ort_state_->vae_session->Run(latents)
    +-- Output: float[1, 3, H, W] image tensor
    +-- Convert to RGBA8: pixels[output_H * output_W * 4]
```

---

## 5. Decision Log

### 5.1 Watchdog Consolidation (Issue 1)

**Decision:** Delete `watchdog.hpp` + `watchdog.cpp`. Pipeline adopts `utilization_watchdog.hpp` API exclusively.

**Rationale:**
- The NEW watchdog has strictly superior design: 3-state machine (NORMAL/WARNING/CRITICAL) vs old flat threshold
- NEW watchdog has exponential backoff (`backoff_base_ms * 2^(count-1)`) vs old fixed cooldown
- NEW watchdog has per-device independent tracking (GPU and Hailo evaluated separately)
- NEW watchdog uses `std::expected<RecoveryResult, std::string>` for rich error reporting vs old void callback
- OLD watchdog's `StepSnapshot` aggregates both devices into one struct, preventing independent evaluation
- OLD watchdog's `RecoveryAction` enum {None, Throttle, Restart, Abort} is less expressive than `RecoveryResult` {SUCCESS, PARTIAL, FATAL}

**Impact:**
- `pipeline.hpp` must change `#include "hq/watchdog.hpp"` to `#include "hq/utilization_watchdog.hpp"`
- `pipeline_integration.cpp` must replace `StepSnapshot` construction with `UtilizationSnapshot`, replace `watchdog_->step(snap)` with `watchdog_->step(step, gpu_snap, hailo_snap)`, replace `set_on_recovery()` lambda signature
- `test_harness.cpp` must replace all `StepSnapshot` usage with `UtilizationSnapshot`, replace `WatchdogResult` with `std::optional<RecoveryAction>`

### 5.2 Hailo Monitor API Verification (Issue 2)

**Decision:** `hailo_monitor.hpp` and `hailo_monitor.cpp` are CORRECT as-is. No changes required.

**Rationale:**
- Header declares `bool is_open() const noexcept;` at line 172
- Implementation defines `bool HailoMonitor::is_open() const noexcept` at line 216
- The task description mentioned `is_connected()` at line 102, but the actual code at line 102 is `return *this;` in the move assignment operator
- `pipeline_integration.cpp` line 269 correctly calls `hailo_monitor_->is_open()`

### 5.3 GPU Utilization Query (Issue 3 — placeholder replacement)

**Decision:** Replace `gpu_util = 75.0 + 10.0f * std::sin(step * 0.3f)` with real ROCm SMI query.

**Implementation:**
```cpp
float Pipeline::query_gpu_utilization_() {
#if defined(UM790_HAS_HIP) && __has_include(<rocm_smi/rocm_smi.h>)
    uint32_t pct = 0;
    rsmi_status_t st = rsmi_dev_gpu_busy_percent_get(0, &pct);
    return (st == RSMI_STATUS_SUCCESS) ? static_cast<float>(pct) : 0.0f;
#else
    return 0.0f;  // Stub when ROCm SMI unavailable
#endif
}
```

**Rationale:**
- `rsmi_dev_gpu_busy_percent_get(uint32_t dv_ind, uint32_t *percent)` returns GPU activity % over the last sample period
- Device index 0 is the 780M iGPU on UM790 Pro
- Must link against `rocm_smi64` library (add to CMakeLists.txt)

### 5.4 Thermal Throttling Guard (Issue 4)

**Decision:** Add temperature-aware recovery suppression to `UtilizationWatchdog::evaluate_device()`.

**Rationale:**
- If GPU temperature > 85C and utilization is low, the device is thermal-throttling
- Triggering recovery (session rebuild) during thermal throttling wastes time and stresses the device further
- The watchdog should detect this condition and skip recovery, logging the event via `on_alert_`
- 85C is the AMD Radeon 780M thermal throttling threshold

**Implementation:** Added `temperature` parameter to `evaluate_device()` and thermal guard before recovery trigger (see section 2.1.4). Also added `thermal_throttle_threshold_c` to `WatchdogConfig`.

### 5.5 Exponential Backoff (Issue 5)

**Decision:** Already implemented in `utilization_watchdog.cpp`. No additional work needed.

**Verification:**
- `compute_backoff_ms(recovery_count)` returns `min(backoff_max_ms, backoff_base_ms * 2^(count-1))`
- `trigger_recovery()` increments `recovery_count` before computing delay, so first recovery has `count=1` => `base * 2^0 = base`, second recovery has `count=2` => `base * 2^1 = 2*base`, etc.
- Delay is applied via `std::this_thread::sleep_for()` before calling the recovery callback
- `in_recovery` flag prevents overlapping recovery invocations on the same device

### 5.6 StagingError Name Collision

**Decision:** Rename `StagingError` in `staging_manager.hpp` to `HostStagingError`.

**Rationale:**
- `pinned_staging.hpp` defines `enum class StagingError : std::uint8_t { Ok, HipError, ... }`
- `staging_manager.hpp` defines `enum class StagingError : std::uint8_t { OutOfMemory, InvalidSize, ... }`
- Both are in namespace `hq`. Including both headers causes name collision
- The `EmbeddingStagingManager` uses `StagingError` for its own operations (pool exhaustion, etc.)
- Renaming to `HostStagingError` preserves the semantics: errors from the host-side buffer pool
- `PinnedStagingPool` keeps its `StagingError` since it's the GPU-side staging error

### 5.7 Pipeline PipelineConfig Field Update

**Decision:** Replace OLD watchdog fields in `PipelineConfig` with NEW watchdog fields.

**OLD fields (to remove):**
```cpp
double watchdog_gpu_threshold;          // single threshold
double watchdog_hailo_threshold;        // single threshold
uint32_t watchdog_max_low_steps;        // renamed
double watchdog_step_timeout_ms;        // removed (not in new design)
uint32_t recovery_cooldown_steps;       // replaced by exponential backoff
```

**NEW fields (to add):**
```cpp
float watchdog_gpu_low_threshold;       // WARNING boundary
float watchdog_gpu_critical_threshold;  // CRITICAL boundary
float watchdog_hailo_low_threshold;     // WARNING boundary
float watchdog_hailo_critical_threshold;// CRITICAL boundary
uint32_t watchdog_consecutive_threshold;// steps before recovery
uint32_t watchdog_max_recoveries;       // max before FATAL
float watchdog_backoff_base_ms;         // exponential backoff base
float watchdog_backoff_max_ms;          // exponential backoff cap
float watchdog_thermal_threshold_c;     // thermal throttling detection
```

### 5.8 Test Harness API Migration

**Decision:** `test_harness.cpp` must be updated to use the new watchdog API exclusively.

**Changes required:**
- Replace `#include "hq/watchdog.hpp"` with `#include "hq/utilization_watchdog.hpp"`
- Replace `StepSnapshot` with `UtilizationSnapshot` (two snapshots: one for GPU, one for Hailo)
- Replace `WatchdogResult` return type with `std::optional<RecoveryAction>`
- Replace `watchdog_->step(snap)` with `watchdog_->step(step, gpu_snap, hailo_snap)`
- Replace `RecoveryAction::None` comparisons with `!recovery_action.has_value()`
- Replace `result.action != RecoveryAction::None` with `recovery_action.has_value()`
- Update `WatchdogConfig` initialization to use new field names
- Update `simulate_steps()` helper to construct two `UtilizationSnapshot` objects

---

## 6. Specialist Assignments

### Specialist 1: Watchdog Engineer
**Files:** `include/hq/utilization_watchdog.hpp`, `src/utilization_watchdog.cpp`

**Requirements:**
1. Add `float thermal_throttle_threshold_c = 85.0f;` to `WatchdogConfig` struct
2. Add `float temperature` parameter to `evaluate_device()` signature
3. In `step()`, pass `gpu_snap.temperature` and `hailo_snap.temperature` to the respective `evaluate_device()` calls
4. In `evaluate_device()`, add thermal guard BEFORE the WARNING→recovery and CRITICAL→recovery trigger points:
   - If `temperature > cfg_.thermal_throttle_threshold_c`, log via `std::print`, call `on_alert_` if set, return `std::nullopt`
   - Must NOT count as a recovery skip (consecutive counter continues incrementing)
5. Verify `compute_backoff_ms()` formula: `delay = min(backoff_max_ms, backoff_base_ms * exp2f(recovery_count - 1))`
6. Invariants to verify: all `std::scoped_lock` calls protect mutable state; `in_recovery` prevents re-entrant triggers; `recovery_count` monotonically increases per device

### Specialist 2: Pipeline Integration Engineer
**Files:** `src/pipeline_integration.cpp`, `include/hq/pipeline.hpp`

**Requirements:**

**Header changes (`pipeline.hpp`):**
1. Replace `#include "hq/watchdog.hpp"` with `#include "hq/utilization_watchdog.hpp"`
2. Update `PipelineConfig` watchdog fields to match NEW `WatchdogConfig` (8 fields, see section 5.7)
3. Update `on_watchdog_recovery_` signature: `void on_watchdog_recovery_(ComputeUnit unit, uint32_t step, float util_at_fault);`
4. Update `denoise_step_` signature: add `const std::vector<float>& embeddings, uint32_t num_steps`
5. Add declarations: `[[nodiscard]] float query_gpu_utilization_();` and `[[nodiscard]] float query_gpu_temperature_();`

**Implementation changes (`pipeline_integration.cpp`):**
1. **Constructor:** Update `WatchdogConfig` initialization to use NEW field names. Pass RecoveryCallback lambda to `UtilizationWatchdog` constructor with signature `std::expected<RecoveryResult, std::string>(ComputeUnit, uint32_t, float)`.
2. **GPU utilization (line 276):** Replace `gpu_util = 75.0 + 10.0f * std::sin(step * 0.3f)` with `gpu_util = query_gpu_utilization_();`
3. **GPU temperature:** Add `float gpu_temp = query_gpu_temperature_();` before watchdog step
4. **Watchdog call (lines 283-291):** Replace `StepSnapshot` + `watchdog_->step(snap)` with two `UtilizationSnapshot` constructions + `watchdog_->step(step, gpu_snap, hailo_snap)`. Return type is `std::optional<RecoveryAction>`.
5. **Recovery handling:** Update to check `recovery_action.has_value()` and access fields via `recovery_action->result`, `recovery_action->device`, `recovery_action->reason`
6. **`denoise_step_` (lines 373-392):** Implement REAL inference:
   - Build input `Ort::Value` tensors from latents + embeddings + timestep
   - Call `ort_state_->gpu_session->Run(...)`
   - Extract noise prediction output tensor
   - Apply scheduler step (DDIM or Euler)
   - Return updated latents pointer
7. **`encode_prompt_` (lines 443-461):** Implement REAL inference:
   - Tokenize prompt (or use pre-tokenized input)
   - Create input `Ort::Value` tensor
   - If `ort_state_->hailo_session` exists: call `Run()` with Hailo session
   - Otherwise: fall back to CPU execution
   - Extract output tensor as `std::vector<float>`
8. **`decode_latents_` (lines 466-498):** Implement REAL inference:
   - Create input `Ort::Value` from latent tensor
   - Call `ort_state_->vae_session->Run(...)`
   - Extract output image tensor (float[1,3,H,W])
   - Normalize from [-1,1] to [0,255], convert to RGBA8
   - Return `GeneratedImage` with actual pixel data (NOT checkerboard)
9. **`query_gpu_utilization_()`:** Implement using `rsmi_dev_gpu_busy_percent_get(0, &pct)` from rocm_smi.h. Return 0.0f if unavailable.
10. **`query_gpu_temperature_()`:** Implement using `rsmi_dev_temp_metric_get(0, RSMI_TEMP_TYPE_EDGE, &temp)` from rocm_smi.h. Return 0.0f if unavailable.

### Specialist 3: Staging Manager Engineer
**Files:** `include/hq/staging_manager.hpp`, `src/staging_manager.cpp`

**Requirements:**
1. In `staging_manager.hpp`: Rename `enum class StagingError` to `enum class HostStagingError`
2. In `staging_manager.hpp`: Update all method signatures that return `std::expected<..., StagingError>` to return `std::expected<..., HostStagingError>`
3. In `staging_manager.cpp`: Replace all occurrences of `StagingError::` with `HostStagingError::`
4. In `staging_manager.cpp`: Update return types in `acquire()` and `copy_in()` to use `HostStagingError`
5. Verify `EmbeddingStagingManager::release()` is idempotent (safe to call multiple times on same buffer)
6. Verify no behavioral changes — this is a pure rename to resolve the header collision

### Specialist 4: Build System & Test Engineer
**Files:** `CMakeLists.txt`, `tests/test_harness.cpp`

**Requirements for `CMakeLists.txt`:**
1. In `UM790_PIPELINE_SOURCES`: replace `src/watchdog.cpp` with `src/utilization_watchdog.cpp`
2. Add `src/pinned_staging.cpp` to `UM790_PIPELINE_SOURCES` (was missing)
3. In `UM790_PIPELINE_HEADERS`: replace `include/hq/watchdog.hpp` with `include/hq/utilization_watchdog.hpp`
4. Add `include/hq/pinned_staging.hpp` to `UM790_PIPELINE_HEADERS` (was missing)
5. Link `rocm_smi64` library for `rsmi_*` functions: add to `target_link_libraries(um790_pipeline PUBLIC ...)`

**Requirements for `test_harness.cpp`:**
1. Replace `#include "hq/watchdog.hpp"` with `#include "hq/utilization_watchdog.hpp"`
2. Replace `StepSnapshot` with `UtilizationSnapshot` throughout
3. Update `WatchdogTest::SetUp()` to construct `UtilizationWatchdog` with NEW `WatchdogConfig` fields and a RecoveryCallback lambda
4. Rewrite `simulate_steps()` to construct TWO `UtilizationSnapshot` objects (GPU and Hailo) per step and call `watchdog_->step(i, gpu_snap, hailo_snap)`
5. Update all return value handling: `std::optional<RecoveryAction>` instead of `WatchdogResult`
6. Replace `result.action == RecoveryAction::None` with `!result.has_value()`
7. Replace `result.action != RecoveryAction::None` with `result.has_value()`
8. Replace `watchdog_->recovery_count()` with `watchdog_->get_stats().gpu_recovery_count + watchdog_->get_stats().hailo_recovery_count`
9. Replace `watchdog_->reset()` with `watchdog_->reset_all()` or `watchdog_->reset_counters()` as appropriate per test
10. Verify all 12 tests compile and pass after migration

### Specialist 5: Hailo Monitor & GPU Query Engineer
**Files:** `src/hailo_monitor.cpp` (minor), `src/pipeline_integration.cpp` (GPU query functions)

**Requirements:**
1. **Verify `hailo_monitor.cpp` correctness:** Confirm `is_open()` is correctly implemented at line 216 (it is — no changes needed)
2. **Implement `Pipeline::query_gpu_utilization_()` in `pipeline_integration.cpp`:**
   ```cpp
   float Pipeline::query_gpu_utilization_() {
   #if defined(UM790_HAS_HIP) && __has_include(<rocm_smi/rocm_smi.h>)
       #include <rocm_smi/rocm_smi.h>
       uint32_t pct = 0;
       if (rsmi_dev_gpu_busy_percent_get(0, &pct) == RSMI_STATUS_SUCCESS) {
           return static_cast<float>(pct);
       }
   #endif
       return 0.0f;
   }
   ```
3. **Implement `Pipeline::query_gpu_temperature_()` in `pipeline_integration.cpp`:**
   ```cpp
   float Pipeline::query_gpu_temperature_() {
   #if defined(UM790_HAS_HIP) && __has_include(<rocm_smi/rocm_smi.h>)
       #include <rocm_smi/rocm_smi.h>
       int64_t temp_mc = 0;  // millidegrees Celsius
       if (rsmi_dev_temp_metric_get(0, RSMI_TEMP_TYPE_EDGE,
                                    RSMI_TEMP_CURRENT, &temp_mc) == RSMI_STATUS_SUCCESS) {
           return static_cast<float>(temp_mc) / 1000.0f;  // convert millidegrees -> degrees
       }
   #endif
       return 0.0f;
   }
   ```
4. **Ensure ROCm SMI initialization:** Call `rsmi_init(0)` once in `Pipeline` constructor and `rsmi_shut_down()` in destructor
5. **Verify `HailoMonitor::sample()` returns correct temperature:** The temperature is read from `read_temperature_celsius()` and stored in `HailoStats::temperature_celsius` — confirm this flows correctly into `UtilizationSnapshot::temperature` in the pipeline's watchdog call

---

## Appendix A: Exact diff for CMakeLists.txt

```diff
 set(UM790_PIPELINE_SOURCES
     src/pipeline_integration.cpp
     src/hailo_monitor.cpp
-    src/watchdog.cpp
+    src/utilization_watchdog.cpp
     src/staging_manager.cpp
+    src/pinned_staging.cpp
 )
 
 set(UM790_PIPELINE_HEADERS
     include/hq/pipeline.hpp
     include/hq/staging_manager.hpp
-    include/hq/watchdog.hpp
+    include/hq/utilization_watchdog.hpp
+    include/hq/pinned_staging.hpp
     include/hq/hailo_monitor.hpp
 )
```

## Appendix B: Exact diff for pipeline.hpp includes

```diff
-#include "hq/watchdog.hpp"
+#include "hq/utilization_watchdog.hpp"
 #include "hq/hailo_monitor.hpp"
+#include "hq/pinned_staging.hpp"
```

## Appendix C: New Watchdog Construction in Pipeline

```cpp
Pipeline::Pipeline(const PipelineConfig& cfg)
    : cfg_{cfg}
    , watchdog_{cfg.enable_watchdog
        ? std::make_unique<UtilizationWatchdog>(
              WatchdogConfig{
                  .gpu_low_threshold          = cfg.watchdog_gpu_low_threshold,
                  .gpu_critical_threshold     = cfg.watchdog_gpu_critical_threshold,
                  .hailo_low_threshold        = cfg.watchdog_hailo_low_threshold,
                  .hailo_critical_threshold   = cfg.watchdog_hailo_critical_threshold,
                  .consecutive_threshold      = cfg.watchdog_consecutive_threshold,
                  .max_recoveries             = cfg.watchdog_max_recoveries,
                  .backoff_base_ms            = cfg.watchdog_backoff_base_ms,
                  .backoff_max_ms             = cfg.watchdog_backoff_max_ms,
                  .thermal_throttle_threshold_c = cfg.watchdog_thermal_threshold_c,
              },
              [this](ComputeUnit unit, uint32_t step, float util) -> std::expected<RecoveryResult, std::string> {
                  this->on_watchdog_recovery_(unit, step, util);
                  return RecoveryResult::SUCCESS;
              },
              [](ComputeUnit unit, uint32_t step, float util,
                 const std::string& msg) {
                  std::print("[Pipeline] Watchdog alert: {}\n", msg);
              })
        : nullptr}
    // ... rest of initializer list
```

## Appendix D: File Checksum Verification (pre-modification)

Before making any changes, verify the files to be deleted are the OLD versions:

| File | SHA-256 (verify before delete) |
|------|-------------------------------|
| `include/hq/watchdog.hpp` | Must contain `StepSnapshot`, `RecoveryAction` enum with `None/Throttle/Restart/Abort` |
| `src/watchdog.cpp` | Must contain `UtilizationWatchdog::step(const StepSnapshot&)`, `cooldown_remaining_` |

These are the ONLY two files being deleted. All other files are preserved with targeted modifications.
