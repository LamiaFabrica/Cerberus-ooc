# UM790Pro Watchdog Failure Injection Scenarios
## Chaos Engineering Test Suite for AI Inference Watchdog Validation
### Target System: Radeon 780M + Hailo-8L on UM790 Pro (Ryzen 9 7940HS)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Test Infrastructure Requirements](#2-test-infrastructure-requirements)
3. [Scenario 1: GPU Scheduling Gap Injection](#scenario-1-gpu-scheduling-gap-injection)
4. [Scenario 2: Hailo PCIe DMA Starvation](#scenario-2-hailo-pcie-dma-starvation)
5. [Scenario 3: Thermal Throttling Gradation](#scenario-3-thermal-throttling-gradation)
6. [Scenario 4: Hailo Firmware Fault Simulation](#scenario-4-hailo-firmware-fault-simulation)
7. [Scenario 5: ROCm Session Corruption](#scenario-5-rocm-session-corruption)
8. [Scenario 6: Unified Memory Page Fault Storm](#scenario-6-unified-memory-page-fault-storm)
9. [Scenario 7: Sustained Low-Utilization Recovery Loop](#scenario-7-sustained-low-utilization-recovery-loop)
10. [Scenario 8: Concurrent Multi-Device Fault](#scenario-8-concurrent-multi-device-fault)
11. [Scenario 9: Memory Bandwidth Saturation (Bonus)](#scenario-9-memory-bandwidth-saturation-bonus)
12. [Scenario 10: Watchdog State Machine Edge Cases](#scenario-10-watchdog-state-machine-edge-cases)
13. [Scenario 11: Cross-Correlation Noise Injection](#scenario-11-cross-correlation-noise-injection)
14. [Test Execution Matrix](#test-execution-matrix)
15. [Instrumentation & Verification Framework](#instrumentation--verification-framework)

---

## 1. Executive Summary

This document defines **11 implementable failure injection scenarios** designed to validate the utilization watchdog, recovery mechanisms, and fault tolerance of the UM790 Pro AI inference pipeline. The scenarios cover the full spectrum of realistic failure modes across the Radeon 780M (GPU) and Hailo-8L (NPU) compute devices.

### Watchdog Architecture Under Test

| Parameter | Value |
|---|---|
| Sampling interval | 500ms |
| Warning threshold | < 60% utilization |
| Critical threshold | < 40% utilization |
| Recovery trigger | 8 consecutive steps below 60% OR immediate if critical/health fault |
| GPU recovery | `hipDeviceReset()` + ONNX session rebuild + prefetch tuning |
| Hailo recovery | `hard_reset()` + HEF reload + input queue depth increase |
| State preservation | Latent tensor snapshot before teardown, restore after recovery |
| CPU fallback | If Hailo FATAL, route encoders to CPU FP32, return `PARTIAL` |

### Failure Taxonomy

```
Root Causes:
  CPU-side:    Scheduler delays, memory pressure, thread contention
  GPU-side:    Session corruption, thermal throttling, XNACK storms
  Hailo-side:  Firmware crash, PCIe DMA starvation, HEF unload
  System-side: Power transients, bandwidth saturation, thermal emergency

Watchdog Behaviors to Validate:
  [OK]   Normal operation (>= 60%)
  [WARN] Low utilization (40-60%) — countdown to recovery
  [CRIT] Critical utilization (< 40%) — immediate recovery
  [FAULT] Device unhealthy — immediate recovery, bypass counter
  [LOOP] Repeated recovery — escalation to FATAL
  [MULTI] Concurrent multi-device fault — independent recovery paths
```

---

## 2. Test Infrastructure Requirements

### 2.1 Hardware Setup

```bash
# Verify test platform
lspci | grep -E "(Advanced Micro Devices|Hailo)"
cat /proc/cpuinfo | grep "model name" | head -1
# Expected: Ryzen 9 7940HS + Radeon 780M iGPU + Hailo-8L

# Verify ROCm SMI
rocm-smi --showbus --showtemp --showuse

# Verify HailoRT
hailortcli scan
hailortcli fw-control identify

# Temperature monitoring (external validation)
sensors  # lm-sensors for junction temp
# Expected: Tdie, Tctl, Tccd1, edge, junction temps available
```

### 2.2 Fault Injection Harness

```cpp
// include/hq/fault_injector.hpp — Test harness for chaos engineering
#pragma once
#include "hq/watchdog.hpp"
#include <functional>
#include <chrono>

namespace hq::test {

enum class FaultType {
    CPU_SCHEDULER_DELAY,       // Delay CPU scheduler computation
    HAILO_QUEUE_DEPTH_REDUCE,  // Reduce Hailo input queue depth
    GPU_INVALID_TENSOR_SHAPE,  // Inject bad tensor shapes
    MEMORY_PRESSURE,           // Allocate large host buffers
    THERMAL_CONSTRAINT,        // Block cooling
    HAILO_DEVICE_RESET,        // Trigger external Hailo reset
    BANDWIDTH_SATURATION,      // Run memory-intensive CPU workload
    SYSTEM_WIDE_STRESS,        // Max all compute units
};

struct FaultInjectionConfig {
    FaultType type;
    int       duration_ms;        // How long to sustain the fault
    int       start_step;         // Which denoising step to start at
    float     intensity;          // Severity scalar [0.0, 1.0]
    bool      auto_revert;        // Automatically revert after duration
};

class FaultInjector {
public:
    explicit FaultInjector(UtilizationWatchdog* watchdog);
    
    // Inject a fault according to configuration
    void inject(const FaultInjectionConfig& cfg);
    
    // Revert any active fault
    void revert();
    
    // Check if a fault is currently active
    bool is_fault_active() const;
    
    // Get fault statistics
    struct FaultStats {
        int total_injections;
        int total_recoveries_triggered;
        int false_positives;
        std::chrono::milliseconds total_fault_duration;
    };
    FaultStats stats() const;

private:
    UtilizationWatchdog* watchdog_;
    std::atomic<bool>    fault_active_{false};
    FaultStats           stats_{};
    
    // Per-fault-type injection implementations
    void inject_scheduler_delay(float intensity, int duration_ms);
    void inject_hailo_dma_starvation(float intensity, int duration_ms);
    void inject_thermal_constraint(float intensity, int duration_ms);
    void inject_memory_pressure(float intensity, int duration_ms);
    void inject_bandwidth_saturation(float intensity, int duration_ms);
};

} // namespace hq::test
```

### 2.3 State Preservation Verification

```cpp
// include/hq/state_verifier.hpp — Verify latent tensor integrity
#pragma once
#include <vector>
#include <cmath>

namespace hq::test {

struct LatentState {
    std::vector<float> tensor_data;
    std::vector<int64_t> shape;     // {batch, channels, height, width}
    int current_step;
    float scheduler_sigma;
};

class StateVerifier {
public:
    // Capture a snapshot of current latent state
    LatentState capture();
    
    // Compare two latent states
    struct Comparison {
        float mse;                  // Mean squared error
        float max_abs_diff;         // Maximum absolute difference
        float relative_error;       // MSE / mean_squared_value
        bool  equivalent;           // mse < 1e-6
    };
    static Comparison compare(const LatentState& before, const LatentState& after);
    
    // Verify embeddings match after recovery
    static bool verify_embeddings_consistent(
        const LatentState& state_pre_fault,
        const LatentState& state_post_recovery,
        int recovered_step
    );
};

} // namespace hq::test
```

---

## Scenario 1: GPU Scheduling Gap Injection

### Overview

| Field | Content |
|---|---|
| **Name** | GPU Scheduling Gap Injection |
| **Target Component** | Radeon 780M (GPU) + CPU scheduler |
| **Target Failure Mode** | CPU scheduler math too slow, causing GPU idle gaps between denoising steps |
| **Severity** | Medium — tests core watchdog + recovery path |
| **Duration** | ~30 seconds per variant |

### Background

The 780M runs a large transformer (FLUX DiT / Qwen MMDiT) with sequential attention heads, producing **burst utilization** (65-80% average with spikes to 95% and valleys to 40%). The CPU must compute `x_t-1 = f(x_t, noise, sigma)` between GPU steps via the DEIS scheduler. If this math is too slow, the GPU idles, `rsmi_dev_gpu_busy_percent_get()` drops, and the watchdog may trigger recovery.

### Injection Mechanism

```cpp
// Variant A: 5ms delay (subtle, may not trigger)
// Variant B: 10ms delay (moderate, should trigger after 8 steps)
// Variant C: 20ms delay (severe, should trigger critical path)

// Implementation: Hook into scheduler computation
void inject_scheduler_delay(float delay_ms, int duration_steps) {
    auto original_scheduler = scheduler_->get_compute_fn();
    
    scheduler_->set_compute_fn([original_scheduler, delay_ms](const Tensor& x_t,
                                                               float sigma,
                                                               float noise) {
        // Add artificial delay to simulate slow CPU math
        std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(delay_ms)));
        
        // Then run the actual scheduler computation
        return original_scheduler(x_t, sigma, noise);
    });
    
    // Revert after N steps or watchdog recovery
    std::thread([duration_steps, original_scheduler, this]() {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(duration_steps * 500)
        );
        scheduler_->set_compute_fn(original_scheduler);
    }).detach();
}
```

**Configuration change:**
```yaml
# test_config/scheduling_gap.yaml
fault_injection:
  type: CPU_SCHEDULER_DELAY
  variants:
    - name: "subtle_5ms"
      delay_ms: 5
      duration_steps: 16
      expected_behavior: "may not trigger recovery"
    - name: "moderate_10ms"
      delay_ms: 10
      duration_steps: 16
      expected_behavior: "triggers recovery after 8 steps"
    - name: "severe_20ms"
      delay_ms: 20
      duration_steps: 16
      expected_behavior: "triggers critical immediate recovery"
```

### Expected Device Behavior

| Metric | Normal | 5ms Gap | 10ms Gap | 20ms Gap |
|---|---|---|---|---|
| `gpu_busy_percent` | 65-80% | 55-70% | 40-60% | 25-45% |
| Burst-to-valley ratio | 3:1 | 2:1 | 1.5:1 | 1:1 |
| GPU temperature | 72-78C | 70-76C | 65-74C | 60-70C |
| Power draw | ~35W | ~32W | ~28W | ~22W |
| Watchdog step counter | 0 | 2-4 | 8+ (triggers) | 8+ (critical) |

### Watchdog Expected Response

**For 10ms variant (moderate):**
```
Step 0-6:  gpu_busy_percent ~55-60%  → WARNING zone, consecutive counter increments
Step 7:    gpu_busy_percent ~52%     → consecutive counter = 8 → RECOVERY TRIGGERED
Step 8+:   Recovery sequence executes:
           1. Latent tensor snapshot captured (MSE checkpoint)
           2. ONNX session destroyed
           3. hipDeviceReset() called
           4. 500ms settle wait
           5. hipSetDevice(0); hipInit(0)
           6. ONNX session rebuilt with fresh opts
           7. rebalance_workload(): scheduler_->reduce_step_size(0.05f)
           8. Latents restored from snapshot
           9. Resumes from step N
```

**For 20ms variant (severe):**
```
Step 0:    gpu_busy_percent ~35%     → CRITICAL zone (< 40%)
           → IMMEDIATE RECOVERY (bypasses 8-step counter)
           → rebalance: aggressive_prefetch enabled
```

### State Preservation Check

```cpp
// Verification procedure
TEST(Scenario1, StatePreservation) {
    // 1. Run inference to step 5 (latents have meaningful state)
    auto state_before = StateVerifier::capture();
    
    // 2. Inject 10ms scheduler delay
    FaultInjector injector(watchdog.get());
    injector.inject({
        .type = CPU_SCHEDULER_DELAY,
        .duration_ms = 10000,
        .start_step = 5,
        .intensity = 0.5f  // 10ms
    });
    
    // 3. Wait for recovery to complete
    wait_for_recovery(watchdog.get(), ComputeUnit::GPU_780M, 30s);
    
    // 4. Capture post-recovery state
    auto state_after = StateVerifier::capture();
    
    // 5. Compare
    auto cmp = StateVerifier::compare(state_before, state_after);
    
    EXPECT_TRUE(cmp.equivalent) << "MSE=" << cmp.mse
                                 << " max_diff=" << cmp.max_abs_diff;
    EXPECT_LT(cmp.mse, 1e-6f) << "Latent MSE exceeds tolerance";
    EXPECT_EQ(state_after.current_step, state_before.current_step)
        << "Resumed from wrong step";
}
```

### Success Criteria

| # | Criterion | Pass Condition |
|---|---|---|
| 1.1 | Watchdog detects low utilization | `consecutive_below` >= 1 after fault injection |
| 1.2 | Recovery triggers after 8 steps (moderate) | Recovery callback invoked with `unit=GPU_780M` |
| 1.3 | Recovery triggers immediately (severe) | Recovery callback invoked within 2 sampling periods |
| 1.4 | Session rebuilds successfully | `recover_gpu_session()` returns `SUCCESS` |
| 1.5 | Latents preserved | MSE before/after < 1e-6 |
| 1.6 | Resumes from correct step | `current_step` unchanged across recovery |
| 1.7 | Rebalancing applied | `aggressive_prefetch=true` OR `step_size` reduced |
| 1.8 | No memory leaks | ROCm SMI `vram_used` pre/post within 50MB |

### Risk of False Positive

**Moderate.** The 780M naturally shows burst utilization with valleys to 40%. A momentary scheduler delay (e.g., OS interrupt, AVX-512 thermal throttling on CPU) could look similar. The 8-step consecutive threshold mitigates this, but the scenario validates that the threshold is correctly tuned.

**Mitigation in test:** Run the 5ms variant first to establish a baseline for "benign scheduling jitter." If 5ms triggers recovery, the threshold is too aggressive.

---

## Scenario 2: Hailo PCIe DMA Starvation

### Overview

| Field | Content |
|---|---|
| **Name** | Hailo PCIe DMA Starvation |
| **Target Component** | Hailo-8L (NPU) + PCIe Gen3 x2 link |
| **Target Failure Mode** | Host doesn't feed encoder inputs fast enough; NN core starves |
| **Severity** | High — validates Hailo-specific recovery path |
| **Duration** | ~20 seconds |

### Background

The Hailo-8L is a dataflow architecture with fixed internal latency. Once a HEF is loaded, utilization immediately hits ~90-100% and stays there. The **only** reason Hailo drops below 60% is PCIe DMA starvation (host not feeding inputs) or a session fault. This scenario tests the DMA starvation path specifically.

The Hailo monitor derives utilization from power draw:
```cpp
normalized = (power_watts - 0.5) / (6.0 - 0.5) * 100%
// At idle:   0.5W → 0% utilization
// At active: 6.0W → 100% utilization
```

### Injection Mechanism

```cpp
// Method 1: Reduce input queue depth
void inject_hailo_dma_starvation_queue_reduce() {
    // Save original depth
    original_queue_depth_ = hailo_input_queue_depth_;
    
    // Reduce to minimum — forces Hailo to wait for each input individually
    hailo_input_queue_depth_ = 1;
    
    // Add artificial delay between batch submissions
    submit_delay_us_ = 5000;  // 5ms delay per batch
}

// Method 2: Add artificial stall in input preparation
void inject_hailo_dma_starvation_stall() {
    auto original_prepare = encoder_input_pipeline_->get_prepare_fn();
    
    encoder_input_pipeline_->set_prepare_fn(
        [original_prepare, this](const Batch& batch) {
            // Simulate slow tokenization / tensor packing
            std::this_thread::sleep_for(std::chrono::milliseconds(15));
            return original_prepare(batch);
        }
    );
}

// Revert
void revert_hailo_dma_starvation() {
    hailo_input_queue_depth_ = original_queue_depth_;  // Restore (typically 4)
    submit_delay_us_ = 0;
    encoder_input_pipeline_->reset_prepare_fn();
}
```

**Configuration change:**
```yaml
# test_config/hailo_dma_starvation.yaml
fault_injection:
  type: HAILO_QUEUE_DEPTH_REDUCE
  variants:
    - name: "queue_depth_1"
      queue_depth: 1
      submit_delay_us: 5000
      duration_steps: 12
      expected_behavior: "utilization drops, recovery triggers after 8 steps"
    - name: "input_stall_15ms"
      input_stall_ms: 15
      duration_steps: 12
      expected_behavior: "severe starvation, may trigger critical path"
```

### Expected Device Behavior

| Metric | Normal | Queue Depth 1 | Input Stall 15ms |
|---|---|---|---|
| Hailo power draw | 5.5-6.5W | 2.0-3.5W | 1.0-2.5W |
| `nn_core_utilization` (proxy) | 90-100% | 27-55% | 9-36% |
| `device_healthy` | true | true | true |
| PCIe link speed | Gen3 x2 | Gen3 x2 | Gen3 x2 |
| Inference throughput | ~800 infer/s | ~200 infer/s | ~80 infer/s |
| Temperature | 45-55C | 40-48C | 38-45C |

**Key observation:** `device_healthy` remains `true` — this is NOT a firmware fault. The Hailo is alive but starving. The watchdog must distinguish starvation from fault.

### Watchdog Expected Response

```
Step 0-1:  nn_core_utilization ~45%   → WARNING zone (40-60%), counter = 1-2
Step 2-6:  nn_core_utilization ~35%   → CRITICAL zone (< 40%), counter increments
Step 7:    nn_core_utilization ~30%   → counter = 8 → RECOVERY TRIGGERED

Recovery sequence (Hailo path):
  1. Hailo ONNX session destroyed (hailo_session_.reset())
  2. hard_reset() called — PCIe reset of Hailo device
  3. 300ms wait for firmware boot (~200ms typical)
  4. Hailo ONNX session rebuilt with make_hailo_session_opts()
  5. rebalance_workload(): queue depth increased (4 + 2 = 6)
  6. If was mid-encode: rerun_encoding() called
  7. Returns SUCCESS (or PARTIAL if HEF reload fails)

Post-recovery queue depth: 6 (was 4)
```

### State Preservation Check

```cpp
TEST(Scenario2, EmbeddingStatePreservation) {
    // Hailo runs encoders (T5, CLIP) — produces embeddings
    // These embeddings are consumed by the GPU denoising loop
    
    // 1. Capture embeddings after initial encoding
    auto embeddings_before = capture_text_embeddings();
    
    // 2. Inject Hailo DMA starvation during a generation step
    FaultInjector injector(watchdog.get());
    injector.inject({
        .type = HAILO_QUEUE_DEPTH_REDUCE,
        .duration_ms = 8000,
        .start_step = 0,  // During encoding phase
        .intensity = 1.0f  // Max starvation
    });
    
    // 3. Wait for recovery
    wait_for_recovery(watchdog.get(), ComputeUnit::HAILO_8L, 30s);
    
    // 4. Verify embeddings after recovery
    auto embeddings_after = capture_text_embeddings();
    
    // Embeddings should be regenerated (re-encode), so compare semantically
    float embedding_cosine_sim = cosine_similarity(embeddings_before, embeddings_after);
    EXPECT_GT(embedding_cosine_sim, 0.99f)
        << "Re-encoded embeddings diverge significantly";
    
    // Verify queue depth was increased
    EXPECT_GE(get_hailo_queue_depth(), 6)
        << "Queue depth not increased after recovery";
}
```

### Success Criteria

| # | Criterion | Pass Condition |
|---|---|---|
| 2.1 | Hailo utilization drops | `nn_core_utilization` < 60% for 3+ consecutive samples |
| 2.2 | Recovery triggers | `recover_hailo_session()` called after 8 steps |
| 2.3 | Hard reset succeeds | `hard_reset()` returns true |
| 2.4 | HEF reload succeeds | Hailo ONNX session rebuilt without exception |
| 2.5 | Queue depth increased | `hailo_input_queue_depth_` >= original + 2 |
| 2.6 | Device healthy after recovery | `device_healthy` == true |
| 2.7 | Re-encode succeeds | Text embeddings regenerated and valid |
| 2.8 | GPU unaffected | `gpu_busy_percent` remains in normal range |

### Risk of False Positive

**Low.** The Hailo-8L, once streaming, maintains near-constant 90-100% utilization. Any sustained drop below 60% is abnormal. The power-proxy measurement is inherently smoothed (1100us sampling x 256 averaging = ~280ms effective window), making transient drops unlikely to trigger falsely.

---

## Scenario 3: Thermal Throttling Gradation

### Overview

| Field | Content |
|---|---|
| **Name** | Thermal Throttling Gradation |
| **Target Component** | Radeon 780M (GPU) SMU thermal management |
| **Target Failure Mode** | Sustained load raises junction temp until SMU throttles clocks |
| **Severity** | High — tests false-positive prevention |
| **Duration** | 30+ minutes |

### Background

The Ryzen 9 7940HS has a 54W TDP. The Radeon 780M iGPU shares the thermal budget with the CPU cores. Under sustained load, the SMU (System Management Unit) progressively reduces GPU clock frequencies to maintain thermal limits. This causes gradual utilization decline that is **NOT a scheduling fault** — the watchdog should NOT recover for thermal throttling.

**SMU Behavior:**
```
T_junction < 85C:   Full clocks (2.8 GHz GPU)
T_junction 85-95C:  Progressive throttling (-50 MHz per degree)
T_junction > 95C:   Aggressive throttling (may drop to 400 MHz)
T_junction > 105C:  Emergency shutdown (hard thermal limit)
```

### Injection Mechanism

```cpp
// Method 1: Thermal constraint (physical)
// Place UM790 Pro in thermally constrained environment:
// - Enclosed space with minimal airflow
// - Block vents partially (use tape, 50% coverage)
// - Ambient temperature elevated to 35-40C

// Method 2: Synthetic thermal stress (software)
// Run CPU + GPU stress simultaneously to maximize heat generation
void inject_thermal_stress() {
    // Launch CPU stress on all 8 cores (Zen 4)
    for (int i = 0; i < 8; ++i) {
        cpu_stress_threads_.emplace_back([]() {
            // AVX-512 FMA loop — maximum power draw
            volatile double accumulator = 1.0;
            while (!stop_cpu_stress_.load()) {
                for (int j = 0; j < 1000000; ++j) {
                    accumulator = accumulator * 1.0000001 + 0.0000001;
                }
            }
        });
    }
    
    // GPU stress: run compute shader at maximum occupancy
    gpu_stress_handle_ = launch_gpu_compute_stress();
    
    // Run for 30 minutes minimum
}

// Method 3: Monitor-driven synthetic throttling simulation
// (For testing without physical thermal constraints)
void simulate_smu_throttling_in_software() {
    // Override the ROCm SMI busy percent to simulate throttling
    // This is a TEST-ONLY hook, not for production
    rsmi_override_busy_percent_fn_ = [](uint32_t* busy) {
        // Simulate gradual decline: 75% -> 50% over 10 minutes
        float elapsed_min = get_test_elapsed_minutes();
        float simulated_util = std::max(50.0f, 75.0f - 2.5f * elapsed_min);
        *busy = static_cast<uint32_t>(simulated_util);
        return RSMI_STATUS_SUCCESS;
    };
}
```

**Test protocol (physical method):**
```bash
# 1. Establish baseline in normal environment
# Run 50-step generation, record utilization profile
# Expected: 65-80% average, stable

# 2. Apply thermal constraint
# - Enclose UM790 Pro in insulated chamber
# - Set ambient to 40C
# - Block 50% of exhaust vents

# 3. Run continuous generation for 30 minutes
# Record: gpu_busy_percent, temperature, power draw, clock frequency

# 4. Analyze:
# - If utilization declines but temperature rises → THROTTLING (correct, no recovery)
# - If utilization declines and temperature normal → FAULT (should recover)
```

### Expected Device Behavior

| Phase | Time | T_junction | GPU Clock | gpu_busy_percent | Power | SMU State |
|---|---|---|---|---|---|---|
| Normal | 0-5 min | 72-80C | 2.8 GHz | 65-80% | ~35W | Nominal |
| Warm | 5-10 min | 80-88C | 2.5 GHz | 55-70% | ~32W | Light throttle |
| Hot | 10-20 min | 88-95C | 1.8 GHz | 50-60% | ~28W | Moderate throttle |
| Critical | 20-30 min | 95-100C | 1.2 GHz | 45-55% | ~24W | Heavy throttle |

### Watchdog Expected Response

```
CRITICAL REQUIREMENT: Watchdog MUST distinguish thermal throttling from actual faults.

Incorrect behavior (false positive):
  gpu_busy_percent drops to 52% for 8 consecutive steps
  → Watchdog triggers recovery
  → hipDeviceReset() + session rebuild
  → No improvement (root cause is thermal, not software)
  → Recovery loop ensues

Correct behavior:
  gpu_busy_percent drops to 52% for 8 consecutive steps
  → Watchdog checks: T_junction = 92C (above 85C threshold)
  → Watchdog recognizes THROTTLING, not fault
  → Logs WARNING: "GPU utilization low but temperature high — possible thermal throttling"
  → Does NOT trigger recovery
  → Optionally: alerts operator to thermal condition
  → Continues monitoring; if temp normalizes and util stays low → then recover
```

**Implementation note:** The watchdog must correlate utilization with temperature. Add thermal awareness:

```cpp
// Proposed enhancement to check_device()
constexpr float THERMAL_THROTTLE_TEMP_C = 85.0f;

bool is_thermal_throttling(const DeviceState& state, float util, float temp) {
    return (util < UTIL_THRESHOLD) && (temp > THERMAL_THROTTLE_TEMP_C);
}

// In check_device():
if (is_thermal_throttling(state, util, temp)) {
    log("WARN", std::format(
        "[{}] Low utilization ({:.1f}%) but high temperature ({:.1f}C). "
        "Possible thermal throttling. Skipping recovery.",
        device_name, util, temp
    ));
    // Reset consecutive counter to prevent false-positive recovery
    state.consecutive_below.store(0);
    return;
}
```

### State Preservation Check

No recovery triggered in correct behavior, so state preservation is moot. However, if the watchdog incorrectly triggers recovery, verify:

```cpp
TEST(Scenario3, NoFalsePositiveRecovery) {
    // 1. Simulate thermal throttling (high temp, low util)
    simulate_thermal_conditions(92.0f, 52.0f);  // temp, util
    
    // 2. Run for 10 sampling periods (5 seconds)
    for (int i = 0; i < 10; ++i) {
        std::this_thread::sleep_for(500ms);
    }
    
    // 3. Verify NO recovery was triggered
    EXPECT_EQ(watchdog->current().gpu_below_threshold_steps, 0)
        << "Consecutive counter should be reset due to thermal correlation";
    
    // 4. Verify session was NOT destroyed/rebuilt
    EXPECT_TRUE(original_session_handle == get_current_session_handle())
        << "Session should not be touched during thermal throttling";
}
```

### Success Criteria

| # | Criterion | Pass Condition |
|---|---|---|
| 3.1 | Thermal correlation detected | Watchdog reads temperature alongside utilization |
| 3.2 | No false-positive recovery | `trigger_recovery()` NOT called when temp > 85C |
| 3.3 | Consecutive counter reset | `consecutive_below` resets to 0 on thermal correlation |
| 3.4 | Warning logged | Log entry contains "thermal throttling" warning |
| 3.5 | Recovery DOES trigger if temp normal | When temp drops below 80C and util stays low, recovery fires |
| 3.6 | Operator alert | Alert callback fires for sustained thermal condition |

### Risk of False Positive

**VERY HIGH — this is the primary concern of this scenario.** Thermal throttling is the most common cause of gradual utilization decline in sustained workloads. The 8-step consecutive threshold is designed to catch scheduling faults, but it will also catch thermal throttling. The temperature correlation check is essential to prevent this.

**Validation approach:**
1. Run the thermal scenario with the correlation check **enabled** — verify no recovery triggers
2. Run the thermal scenario with the correlation check **disabled** — verify recovery DOES trigger (confirms the fault would have been caught without the thermal guard)
3. Both behaviors confirm the system works correctly

---

## Scenario 4: Hailo Firmware Fault Simulation

### Overview

| Field | Content |
|---|---|
| **Name** | Hailo Firmware Fault Simulation |
| **Target Component** | Hailo-8L firmware + PCIe link |
| **Target Failure Mode** | Hailo firmware crash or PCIe link instability |
| **Severity** | Critical — validates fault bypass and CPU fallback |
| **Duration** | ~10 seconds |

### Background

Hailo firmware can crash due to:
- PCIe link training errors (Gen3 x2 marginal signal integrity)
- SRAM corruption during inference
- Power rail instability
- Thermal emergency in Hailo die

When firmware crashes, `device_->get_device_information()` fails, `device_healthy` becomes `false`, and the watchdog must trigger **immediate recovery** (bypass the 8-step counter). If recovery fails, the system must fall back to CPU encoder execution.

### Injection Mechanism

```cpp
// Method 1: Trigger Hailo device reset via HailoRT API
void inject_hailo_firmware_fault_external() {
    // Use hailortcli to force a device reset
    std::system("hailortcli fw-control reset");
    // This simulates a firmware crash from the device's perspective
    // The inference pipeline sees: device_healthy = false
}

// Method 2: PCIe link retrain (more severe)
void inject_pcie_link_retrain() {
    // Force PCIe link retrain via sysfs
    std::system(
        "echo 1 > /sys/bus/pci/devices/0000:01:00.0/remove && "
        "echo 1 > /sys/bus/pci/rescan"
    );
    // This causes a brief PCIe disconnect/reconnect
    // HailoRT sees device disappear then reappear
}

// Method 3: Software fault injection (non-destructive, preferred for CI)
void inject_hailo_fault_software() {
    // Intercept the Hailo sample() call to simulate a fault
    hailo_test_inject_fault_ = true;
    // This causes HailoMonitor::sample() to throw and set device_healthy = false
}

// HailoMonitor modification for testability:
HailoStats HailoMonitor::sample() {
    #ifdef FAULT_INJECTION_BUILD
    if (hailo_test_inject_fault_) {
        HailoStats fake_stats{};
        fake_stats.device_healthy = false;
        fake_stats.nn_core_utilization = 0.0f;
        fake_stats.power_watts = 0.0f;
        hailo_test_inject_fault_ = false;  // One-shot
        throw std::runtime_error("Simulated Hailo firmware fault");
    }
    #endif
    // ... normal sample() implementation
}
```

**Test sequence:**
```yaml
# test_config/hailo_firmware_fault.yaml
fault_injection:
  type: HAILO_DEVICE_RESET
  variants:
    - name: "soft_fault"
      method: "software_inject"
      duration_ms: 0  # Instant fault
      expected_recovery: "immediate hard_reset + HEF reload"
      expected_result: SUCCESS
    - name: "hard_fault"
      method: "pcie_retrain"
      duration_ms: 0
      expected_recovery: "hard_reset + HEF reload"
      expected_result: "SUCCESS or PARTIAL"
    - name: "unrecoverable"
      method: "pcie_remove"
      duration_ms: 5000
      expected_recovery: "hard_reset fails → CPU fallback"
      expected_result: FATAL (but pipeline continues with CPU)
```

### Expected Device Behavior

| Metric | Normal | Fault | During Recovery | Post-Recovery |
|---|---|---|---|---|
| `nn_core_utilization` | 90-100% | 0% (immediate) | N/A (device resetting) | 90-100% |
| `power_watts` | 5.5-6.5W | 0.5W (idle) | 0.5W | 5.5-6.5W |
| `device_healthy` | true | false | false | true (if successful) |
| `hailo_session_` | valid | valid (stale) | nullptr | valid (new) |
| PCIe link | Gen3 x2 | May show errors | Disconnected/retraining | Gen3 x2 |

### Watchdog Expected Response

**Path A: Recovery succeeds**
```
T+0ms:   Hailo firmware fault injected
         → sample() throws / returns device_healthy=false
         → check_device(): !healthy → IMMEDIATE RECOVERY (bypass counter)

T+1ms:   trigger_recovery(HAILO_8L):
         1. needs_reencode = (step == 0)
         2. hailo_session_.reset()  — destroy ONNX session
         3. hard_reset() → true     — PCIe reset succeeds
         4. Sleep 300ms             — firmware boot
         5. Rebuild Hailo session   — make_hailo_session_opts()
         6. Returns SUCCESS

T+500ms: Watchdog resumes monitoring
         → nn_core_utilization back to 90-100%
         → device_healthy = true
```

**Path B: Recovery fails → CPU fallback**
```
T+0ms:   Hailo firmware fault injected
         → check_device(): !healthy → IMMEDIATE RECOVERY

T+1ms:   trigger_recovery(HAILO_8L):
         1. hailo_session_.reset()
         2. hard_reset() → true
         3. Sleep 300ms
         4. Rebuild Hailo session → Ort::Exception thrown
         5. catch block: route_encoders_to_cpu()
         6. Returns PARTIAL

T+500ms: Pipeline continues with CPU FP32 encoders
         → Generation quality may be slightly reduced but continues
         → Watchdog monitors GPU only (Hailo marked as FATAL)
```

### State Preservation Check

```cpp
TEST(Scenario4, CPUEncoderFallbackState) {
    // 1. Start generation (encoding phase uses Hailo)
    auto embeddings_before = capture_text_embeddings();
    
    // 2. Inject Hailo fault during encoding
    inject_hailo_fault_software();
    
    // 3. Wait for recovery
    auto result = wait_for_recovery_result(ComputeUnit::HAILO_8L, 15s);
    
    // 4. Verify fallback
    EXPECT_EQ(result, RecoveryResult::PARTIAL)
        << "Should return PARTIAL when Hailo session can't rebuild";
    
    // 5. Verify encoders are on CPU
    EXPECT_TRUE(encoders_routed_to_cpu())
        << "Encoders should be on CPU after Hailo FATAL";
    
    // 6. Verify generation continues
    EXPECT_TRUE(generation_in_progress())
        << "Generation should continue with CPU fallback";
    
    // 7. Verify embeddings are valid (may differ slightly from Hailo INT8)
    auto embeddings_after = capture_text_embeddings();
    float cos_sim = cosine_similarity(embeddings_before, embeddings_after);
    EXPECT_GT(cos_sim, 0.95f)
        << "CPU embeddings should be semantically similar to Hailo embeddings";
}
```

### Success Criteria

| # | Criterion | Pass Condition |
|---|---|---|
| 4.1 | Immediate detection | Recovery triggered within 1 sampling period (500ms) |
| 4.2 | Bypass counter | `consecutive_below` counter is irrelevant; recovery fires immediately |
| 4.3 | Hard reset executed | `hard_reset()` called before any other recovery action |
| 4.4 | HEF reload attempt | Hailo ONNX session rebuilt |
| 4.5a | Path A: Success | If rebuild succeeds, returns SUCCESS, device healthy |
| 4.5b | Path B: CPU fallback | If rebuild fails, returns PARTIAL, encoders on CPU |
| 4.6 | Generation continues | Pipeline does not abort; generation completes |
| 4.7 | GPU unaffected | Radeon 780M continues denoising normally |

### Risk of False Positive

**Very Low.** `device_healthy = false` only occurs when HailoRT reports a device-level failure. This is a binary signal, not a utilization threshold. The only false positive would be if `sample()` throws spuriously, which is extremely unlikely with proper HailoRT initialization.

---

## Scenario 5: ROCm Session Corruption

### Overview

| Field | Content |
|---|---|
| **Name** | ROCm Session Corruption |
| **Target Component** | Radeon 780M (GPU) + ONNX Runtime session |
| **Target Failure Mode** | ONNX Runtime session enters bad state (memory corruption, kernel hang) |
| **Severity** | High — validates GPU recovery path with stateful restart |
| **Duration** | ~15 seconds |

### Background

The ONNX Runtime session for FLUX/Qwen denoising can enter a bad state due to:
- Memory corruption in ROCm EP (race condition, use-after-free)
- Kernel hang in MI300/MI200 driver (infinite loop in shader)
- Invalid tensor shapes passed from scheduler
- HSA queue overflow from too many concurrent kernel submissions

When this happens, `gpu_busy_percent` drops because the GPU is either hung or producing garbage. The watchdog must detect the low utilization, trigger recovery, and restore the denoising state.

### Injection Mechanism

```cpp
// Method 1: Inject invalid tensor shapes
void inject_invalid_tensor_shape() {
    // Corrupt the latent tensor shape mid-denoising
    // This causes ONNX Runtime to fail shape inference
    auto& latents = mem_pool_->get_latents();
    
    // Swap height and width dimensions (valid shape but wrong for model)
    int64_t h = latents.shape[2];
    latents.shape[2] = latents.shape[3];
    latents.shape[3] = h;
    
    // The next Ort::Session::Run() will either:
    // - Throw Ort::Exception (caught, triggers visible error)
    // - Hang silently (GPU kernel loops, utilization drops)
}

// Method 2: Force HSA queue overflow
void inject_hsa_queue_overflow() {
    // Submit many small kernels rapidly to overflow the HSA queue
    for (int i = 0; i < 10000; ++i) {
        launch_nop_kernel();  // No-op kernels to fill queue
    }
    // Subsequent real kernels may stall or fail
}

// Method 3: Memory corruption via double-free simulation
void inject_memory_corruption() {
    // Corrupt a small region of device memory
    // This is done via a test-only ROCm hook
    hipMemset(reinterpret_cast<void*>(0xDEADBEEF), 0xFF, 1024);
    // The address is intentionally invalid for safety
    // In a real test: corrupt a known allocation
}

// Method 4: Software hang simulation (test-only)
void inject_gpu_hang_software() {
    #ifdef FAULT_INJECTION_BUILD
    // Override the ROCm SMI busy percent to simulate a hung GPU
    rsmi_override_busy_percent_fn_ = [](uint32_t* busy) {
        // Simulate hung GPU: stuck at 15-25%
        *busy = 18;  // Consistent low value
        return RSMI_STATUS_SUCCESS;
    };
    #endif
}
```

**Test sequence:**
```yaml
# test_config/rocm_session_corruption.yaml
fault_injection:
  type: GPU_INVALID_TENSOR_SHAPE
  variants:
    - name: "shape_corruption"
      method: "swap_dimensions"
      target_step: 5
      expected_behavior: "Ort exception or GPU hang, utilization drops"
      expected_recovery: "hipDeviceReset + session rebuild + latent restore"
    - name: "gpu_hang"
      method: "software_simulation"
      target_step: 5
      expected_util: "15-25% stuck"
      expected_recovery: "hipDeviceReset + session rebuild"
```

### Expected Device Behavior

| Metric | Normal | Shape Corruption | GPU Hang |
|---|---|---|---|
| `gpu_busy_percent` | 65-80% | 0-30% (after crash) | 15-25% (stuck) |
| ROCm error | None | `HSA_STATUS_ERROR` possible | None (silent hang) |
| Session state | Valid | Invalid (post-crash) | Hung |
| VRAM usage | ~3GB | May leak (unfreed allocations) | ~3GB (stuck) |
| Temperature | 72-78C | Drops rapidly | Stays elevated |

### Watchdog Expected Response

```
Path A: Shape corruption (Ort exception visible)
  Step 5:  Ort::Run() throws exception
           → Pipeline catches exception, reports error
           → Next sampling: gpu_busy_percent = 0%
           → CRITICAL zone → immediate recovery
           → recover_gpu_session():
               1. Snapshot latents (MSE checkpoint)
               2. flux_gpu_session_.reset()
               3. gpu_monitor_->soft_reset() → may fail
               4. hipDeviceReset()
               5. 500ms settle
               6. hipSetDevice(0); hipInit(0)
               7. Rebuild session
               8. rebalance_workload(): aggressive_prefetch=true
               9. Restore latents
               10. Resume from step 5

Path B: GPU hang (silent)
  Step 5:  GPU kernel hangs internally
           → gpu_busy_percent drops to 18% (stuck)
           → Consecutive counter increments each step
           → Step 12: counter = 8 → RECOVERY TRIGGERED
           → Same recovery sequence as Path A
           → hipDeviceReset() breaks the hang
           → Fresh session starts clean
```

### State Preservation Check

```cpp
TEST(Scenario5, LatentPreservationAfterSessionCorruption) {
    // 1. Run to step 5
    for (int step = 0; step < 5; ++step) {
        run_denoising_step(step);
    }
    auto state_before = StateVerifier::capture();
    
    // 2. Inject shape corruption
    inject_invalid_tensor_shape();
    
    // 3. Attempt next step (will fail)
    bool threw = false;
    try {
        run_denoising_step(5);
    } catch (const Ort::Exception& e) {
        threw = true;
    }
    EXPECT_TRUE(threw) << "Shape corruption should cause exception";
    
    // 4. Wait for watchdog recovery
    auto result = wait_for_recovery_result(ComputeUnit::GPU_780M, 30s);
    EXPECT_EQ(result, RecoveryResult::SUCCESS);
    
    // 5. Verify state preservation
    auto state_after = StateVerifier::capture();
    auto cmp = StateVerifier::compare(state_before, state_after);
    
    EXPECT_LT(cmp.mse, 1e-6f)
        << "Latent MSE after recovery exceeds tolerance: " << cmp.mse;
    EXPECT_LT(cmp.max_abs_diff, 1e-5f)
        << "Max absolute diff exceeds tolerance: " << cmp.max_abs_diff;
    EXPECT_EQ(state_after.current_step, 5)
        << "Should resume from step 5, not restart";
    
    // 6. Verify session is fresh
    EXPECT_TRUE(session_is_fresh())
        << "Session should be newly rebuilt after corruption";
}
```

### Success Criteria

| # | Criterion | Pass Condition |
|---|---|---|
| 5.1 | Fault detected | `gpu_busy_percent` drops below threshold |
| 5.2 | Recovery triggers | `recover_gpu_session()` called |
| 5.3 | Latent snapshot | `snapshot_latents()` called before session teardown |
| 5.4 | hipDeviceReset executed | GPU hardware reset performed |
| 5.5 | Session rebuilt | New ONNX session created successfully |
| 5.6 | Latent MSE < 1e-6 | State preservation verified |
| 5.7 | Resumes from saved step | `current_step` == saved_step |
| 5.8 | No VRAM leak | `vram_used` within 100MB of pre-fault value |

### Risk of False Positive

**Moderate.** A legitimate slow denoising step (e.g., first step after VAE decode, which has setup overhead) could temporarily drop utilization. The 8-step counter provides protection, but a true GPU hang that holds the GPU at ~20% for 8+ steps is indistinguishable from a slow legitimate workload. The `hipDeviceReset()` is the correct action either way — it breaks hangs without losing state.

---

## Scenario 6: Unified Memory Page Fault Storm

### Overview

| Field | Content |
|---|---|
| **Name** | Unified Memory Page Fault Storm |
| **Target Component** | HSA_XNACK unified memory + page migration |
| **Target Failure Mode** | Page migration overhead under memory pressure causes CPU stalls |
| **Severity** | Medium — tests false-positive behavior under memory pressure |
| **Duration** | ~30 seconds |

### Background

The UM790 Pro uses HSA_XNACK for unified memory between CPU and GPU. When the GPU accesses host-allocated memory, pages are migrated on-demand. Under memory pressure, this causes:
1. CPU stalls waiting for page fault resolution
2. GPU idle time while pages are being migrated
3. Apparent low `gpu_busy_percent` even though the workload is valid

This scenario tests whether the watchdog correctly interprets low utilization under memory pressure.

### Injection Mechanism

```cpp
void inject_memory_pressure() {
    // Allocate large host buffers to fragment unified memory
    // Target: consume 80-90% of system RAM (32GB total on UM790 Pro)
    const size_t TARGET_ALLOC = 26ULL * 1024 * 1024 * 1024;  // 26 GB
    const size_t BLOCK_SIZE = 256 * 1024 * 1024;  // 256 MB blocks
    
    memory_pressure_blocks_.clear();
    size_t allocated = 0;
    
    while (allocated < TARGET_ALLOC) {
        try {
            auto block = std::make_unique<char[]>(BLOCK_SIZE);
            // Touch every page to force allocation
            std::memset(block.get(), 0xAB, BLOCK_SIZE);
            allocated += BLOCK_SIZE;
            memory_pressure_blocks_.push_back(std::move(block));
        } catch (const std::bad_alloc&) {
            break;  // Can't allocate more
        }
    }
    
    // Now force page migration by having GPU access host memory
    // that hasn't been migrated yet
    trigger_cross_device_memory_access();
}

void release_memory_pressure() {
    memory_pressure_blocks_.clear();
    memory_pressure_blocks_.shrink_to_fit();
    // May need to drop caches
    std::system("echo 3 > /proc/sys/vm/drop_caches");
}

// Monitor XNACK page fault counters (ROCm profiling)
struct XNACKStats {
    uint64_t page_faults;       // From kfd2kgd interface
    uint64_t migration_time_us; // Time spent in page migration
    uint64_t cpu_stall_cycles;  // Approximate CPU stall
};

XNACKStats read_xnack_stats() {
    XNACKStats stats{};
    #ifdef ROCM_PROFILER_AVAILABLE
    // Read from /sys/class/kfd/kfd/topology/nodes/*/mem_banks/*/used
    // or ROCm profiling API if available
    #endif
    return stats;
}
```

**Configuration:**
```yaml
# test_config/xnack_page_fault.yaml
fault_injection:
  type: MEMORY_PRESSURE
  variants:
    - name: "moderate_pressure"
      target_ram_usage_gb: 24
      duration_steps: 16
      expected_behavior: "some utilization drop, may or may not trigger recovery"
    - name: "severe_pressure"
      target_ram_usage_gb: 28
      duration_steps: 16
      expected_behavior: "significant utilization drop, likely triggers recovery"
```

### Expected Device Behavior

| Metric | Normal | Moderate Pressure | Severe Pressure |
|---|---|---|---|
| `gpu_busy_percent` | 65-80% | 50-65% | 30-50% |
| Page faults/sec | < 100 | 1,000-5,000 | 10,000+ |
| CPU iowait | < 2% | 10-20% | 30-50% |
| System RAM used | 4-8 GB | 24 GB | 28 GB |
| Migration stalls | None | Periodic spikes | Continuous |
| Generation time/step | ~35ms | ~50ms | ~80ms |

### Watchdog Expected Response

This scenario tests a **design decision**: Should the watchdog recover when low utilization is caused by external memory pressure?

**Option A: Recovery triggers (current behavior)**
- Watchdog sees low utilization → triggers GPU recovery
- `hipDeviceReset()` clears GPU memory state
- Session rebuild allocates fresh memory
- **Result:** May briefly improve (freed memory reduces pressure) but root cause persists
- **Risk:** Recovery loop if memory pressure continues

**Option B: Recovery suppressed (enhanced behavior)**
- Watchdog checks system memory pressure before recovering
- If RAM usage > 80%, log warning instead of recovering
- **Result:** No recovery; wait for memory pressure to subside
- **Risk:** Misses genuine GPU faults that happen during memory pressure

**Recommended approach:** Trigger recovery but with a memory-aware rebalance. After recovery, if memory pressure is still high, reduce batch size or enable more aggressive offloading.

```cpp
// Enhanced rebalance for memory pressure
void rebalance_workload_memory_aware(ComputeUnit unit, float util) {
    // Check system memory
    struct sysinfo info;
    sysinfo(&info);
    float ram_usage_frac = 1.0f - (float)info.freeram / info.totalram;
    
    if (ram_usage_frac > 0.85f) {
        log("WARN", "High memory pressure detected. Reducing batch size.");
        reduce_batch_size(0.5f);  // Halve batch size
    }
    
    // Standard rebalance
    rebalance_workload(unit, util);
}
```

### State Preservation Check

```cpp
TEST(Scenario6, StateUnderMemoryPressure) {
    // 1. Capture state at step 3
    for (int step = 0; step < 3; ++step) run_denoising_step(step);
    auto state_before = StateVerifier::capture();
    
    // 2. Apply memory pressure
    inject_memory_pressure();
    
    // 3. Run several more steps
    for (int step = 3; step < 10; ++step) {
        run_denoising_step(step);
    }
    
    // 4. Release pressure
    release_memory_pressure();
    
    // 5. Capture state
    auto state_after = StateVerifier::capture();
    
    // 6. Verify generation completed (may have recovered)
    EXPECT_GE(state_after.current_step, 9)
        << "Generation should progress despite memory pressure";
    
    // 7. If recovery triggered, verify state preserved
    auto cmp = StateVerifier::compare(state_before, state_after);
    // Note: MSE check is only valid if recovery triggered mid-sequence
    // If no recovery, states won't match (different steps) — that's OK
}
```

### Success Criteria

| # | Criterion | Pass Condition |
|---|---|---|
| 6.1 | Utilization drop detected | `gpu_busy_percent` measurably lower under pressure |
| 6.2 | Page fault increase | XNACK page fault counters increase (if monitorable) |
| 6.3 | Generation completes | Full generation finishes despite pressure |
| 6.4 | Recovery behavior documented | Log shows whether recovery triggered or suppressed |
| 6.5 | No infinite recovery loop | Max 2 recoveries in 16-step window |
| 6.6 | Memory pressure recovery | After releasing pressure, utilization returns to normal |

### Risk of False Positive

**HIGH.** This is the most likely false-positive scenario. Memory pressure is a common condition in multi-tasking environments. The watchdog should be enhanced with memory-pressure awareness to avoid unnecessary recoveries.

**Recommendation:** Add `ram_usage_fraction` to the `DeviceState` check. If RAM > 85% and utilization is low, suppress recovery and log a memory-pressure warning instead.

---

## Scenario 7: Sustained Low-Utilization Recovery Loop

### Overview

| Field | Content |
|---|---|
| **Name** | Sustained Low-Utilization Recovery Loop |
| **Target Component** | Both Radeon 780M + Hailo-8L |
| **Target Failure Mode** | Systematic under-feeding of both devices (config error) |
| **Severity** | Critical — validates recovery loop prevention |
| **Duration** | ~60 seconds |

### Background

If the pipeline is globally misconfigured with too-small batch sizes and too-slow data loading, both devices will consistently run below the 60% threshold. The watchdog will repeatedly trigger recovery, potentially creating a cascade or livelock:

```
Step 0-7:  GPU 45%, Hailo 35%  → both below threshold
Step 8:    Recovery triggered for both
           → GPU: hipDeviceReset + session rebuild
           → Hailo: hard_reset + HEF reload
Step 9-16: GPU 45%, Hailo 35%  → still under-fed (root cause not fixed)
Step 17:   Recovery triggered again
           ...repeat...
```

This scenario tests that the watchdog **escalates to FATAL** after repeated failed recoveries, rather than looping indefinitely.

### Injection Mechanism

```cpp
void inject_sustained_underfeeding() {
    // Reduce batch size to force GPU under-utilization
    original_batch_size_ = config_.batch_size;
    config_.batch_size = 1;  // Minimum batch size
    
    // Add artificial delay in data loading
    original_dataloader_fn_ = dataloader_->get_next_fn();
    dataloader_->set_next_fn([this]() {
        // Simulate slow data loading (30ms per sample)
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        return original_dataloader_fn_();
    });
    
    // Reduce Hailo input queue depth
    original_hailo_queue_depth_ = hailo_input_queue_depth_;
    hailo_input_queue_depth_ = 1;
    
    // Add inter-batch delay for Hailo
    original_hailo_submit_interval_ = hailo_submit_interval_ms_;
    hailo_submit_interval_ms_ = 50;  // 50ms between submissions
}

void revert_sustained_underfeeding() {
    config_.batch_size = original_batch_size_;
    dataloader_->set_next_fn(original_dataloader_fn_);
    hailo_input_queue_depth_ = original_hailo_queue_depth_;
    hailo_submit_interval_ms_ = original_hailo_submit_interval_ms_;
}
```

**Configuration:**
```yaml
# test_config/recovery_loop.yaml
fault_injection:
  type: SUSTAINED_LOW_UTILIZATION
  variants:
    - name: "dual_device_starvation"
      batch_size: 1
      dataloader_delay_ms: 30
      hailo_queue_depth: 1
      hailo_submit_delay_ms: 50
      max_recoveries_before_fatal: 3
      duration_seconds: 60
      expected_behavior: "recoveries triggered repeatedly, escalate to FATAL"
```

### Expected Device Behavior

| Metric | Normal | Under-fed |
|---|---|---|
| GPU `gpu_busy_percent` | 65-80% | 30-45% |
| Hailo `nn_core_utilization` | 90-100% | 20-40% |
| Recovery frequency | None | Every 8 steps (~4 seconds) |
| Recovery success rate | N/A | 100% (recovers successfully) |
| Post-recovery util | N/A | Same low value (root cause persists) |
| Total recoveries | 0 | 10+ in 60 seconds |

### Watchdog Expected Response

**Current behavior (to verify):**
```
Recovery #1:  Both devices recovered successfully
              → Returns SUCCESS for both
              → Rebalancing applied (but doesn't fix root cause)

Recovery #2:  (8 steps later) Both devices recovered again
              → Returns SUCCESS for both
              → total_recoveries = 2

Recovery #3:  (8 steps later) Both devices recovered again
              → Returns SUCCESS for both
              → total_recoveries = 3
              ... loop continues indefinitely
```

**Expected enhanced behavior:**
```
Recovery #1:  GPU recovered, Hailo recovered
              → total_recoveries: GPU=1, Hailo=1

Recovery #2:  GPU recovered, Hailo recovered
              → total_recoveries: GPU=2, Hailo=2
              → Log: "WARNING: Repeated recoveries detected. Root cause may be underfeeding."

Recovery #3:  GPU recovered, Hailo recovered
              → total_recoveries: GPU=3, Hailo=3
              → FATAL escalation: "Recovery loop detected. Marking devices as FATAL."
              → state.healthy = false for both devices
              → Pipeline aborts or falls back to CPU-only mode

Post-FATAL:
  GPU:   state.healthy = false, no further recovery attempts
  Hailo: state.healthy = false, encoders routed to CPU
  Pipeline: Continues on CPU if possible, or aborts
```

### Recovery Loop Prevention Enhancement

```cpp
// Add to DeviceState:
constexpr int MAX_RECOVERIES_BEFORE_FATAL = 3;

// Add to trigger_recovery():
void UtilizationWatchdog::trigger_recovery(DeviceState& state, int step) {
    // ... existing code ...
    
    int recovery_count = ++state.total_recoveries;
    
    if (recovery_count >= MAX_RECOVERIES_BEFORE_FATAL) {
        log("FATAL", std::format(
            "[{}] Recovery loop detected: {} recoveries in rapid succession. "
            "Root cause is likely systematic underfeeding or hardware failure. "
            "Escalating to FATAL.",
            device_name, recovery_count
        ));
        state.healthy.store(false);
        state.in_recovery.store(false);
        return;  // Don't attempt further recovery
    }
    
    // ... existing recovery code ...
}
```

### State Preservation Check

```cpp
TEST(Scenario7, RecoveryLoopEscalation) {
    // 1. Inject sustained underfeeding
    inject_sustained_underfeeding();
    
    // 2. Start generation
    start_generation(50_steps);
    
    // 3. Wait for recoveries
    std::this_thread::sleep_for(std::chrono::seconds(30));
    
    // 4. Verify escalation
    auto util = watchdog->current();
    EXPECT_FALSE(util.gpu_healthy)
        << "GPU should be marked FATAL after repeated recoveries";
    EXPECT_FALSE(util.hailo_healthy)
        << "Hailo should be marked FATAL after repeated recoveries";
    
    // 5. Verify recovery count
    // (Need to expose total_recoveries via test interface)
    EXPECT_GE(get_gpu_recovery_count(), MAX_RECOVERIES_BEFORE_FATAL);
    EXPECT_GE(get_hailo_recovery_count(), MAX_RECOVERIES_BEFORE_FATAL);
    
    // 6. Verify no further recovery attempts
    int recovery_count_before = get_total_recovery_count();
    std::this_thread::sleep_for(std::chrono::seconds(10));
    EXPECT_EQ(get_total_recovery_count(), recovery_count_before)
        << "No further recoveries should be attempted after FATAL";
}
```

### Success Criteria

| # | Criterion | Pass Condition |
|---|---|---|
| 7.1 | Both devices under-fed | Both utilizations < 60% sustained |
| 7.2 | Multiple recoveries triggered | 3+ recoveries per device |
| 7.3 | Recovery loop detected | Watchdog recognizes repeated recoveries |
| 7.4 | FATAL escalation | Both devices marked FATAL after N recoveries |
| 7.5 | Recovery stops | No further recovery attempts after FATAL |
| 7.6 | Pipeline handling | Pipeline either aborts gracefully or falls back to CPU |
| 7.7 | Log analysis | Log shows clear escalation pattern |

### Risk of False Positive

**Very Low in test, HIGH in production.** In a controlled test, sustained underfeeding is clearly intentional. In production, this would indicate a serious configuration error. The FATAL escalation is the correct behavior — it prevents the system from spending all its time recovering instead of generating.

---

## Scenario 8: Concurrent Multi-Device Fault

### Overview

| Field | Content |
|---|---|
| **Name** | Concurrent Multi-Device Fault |
| **Target Component** | Both Radeon 780M + Hailo-8L simultaneously |
| **Target Failure Mode** | Both devices fault simultaneously (power transient, thermal emergency) |
| **Severity** | Critical — validates independent, non-interfering recovery |
| **Duration** | ~20 seconds |

### Background

A system-wide stress event (e.g., power supply transient, thermal emergency, VRM overload) can affect both the GPU and Hailo simultaneously. The watchdog must handle both recovery paths independently without:
- Deadlock (both recoveries waiting for each other)
- Cross-device interference (GPU recovery affecting Hailo state)
- Resource contention (both trying to reset PCIe at the same time)

### Injection Mechanism

```cpp
void inject_concurrent_fault() {
    // Inject both faults at the same time
    std::thread gpu_fault([this]() {
        inject_gpu_hang_software();
    });
    
    std::thread hailo_fault([this]() {
        inject_hailo_fault_software();
    });
    
    gpu_fault.join();
    hailo_fault.join();
    // Both faults are now active simultaneously
}

// Alternative: System-wide stress that affects both
void inject_system_wide_stress() {
    // Max out all compute units simultaneously
    
    // CPU: All 8 cores at 100%
    cpu_stress_threads_.clear();
    for (int i = 0; i < 8; ++i) {
        cpu_stress_threads_.emplace_back([]() {
            volatile double acc = 1.0;
            while (!stop_stress_.load()) {
                for (int j = 0; j < 10000000; ++j) {
                    acc = acc * 1.000001 + 0.000001;
                }
            }
        });
    }
    
    // GPU: Compute stress
    launch_gpu_compute_stress();
    
    // Hailo: Inference stress (run model at maximum rate)
    launch_hailo_inference_stress();
    
    // Memory: Allocate large buffers
    inject_memory_pressure();
    
    // Run for 10 seconds — this should cause both devices to fault
    std::this_thread::sleep_for(std::chrono::seconds(10));
    
    // Release all stress
    stop_stress_.store(true);
    for (auto& t : cpu_stress_threads_) t.join();
    release_memory_pressure();
}
```

**Configuration:**
```yaml
# test_config/concurrent_fault.yaml
fault_injection:
  type: SYSTEM_WIDE_STRESS
  variants:
    - name: "simultaneous_software_faults"
      gpu_fault_method: "hang_simulation"
      hailo_fault_method: "firmware_fault_simulation"
      expected_behavior: "both recoveries fire independently"
    - name: "thermal_emergency_stress"
      method: "max_all_compute"
      duration_seconds: 10
      expected_behavior: "both devices throttle/fault, independent recovery"
```

### Expected Device Behavior

| Metric | Normal | Concurrent Fault |
|---|---|---|
| GPU `gpu_busy_percent` | 65-80% | 0-25% |
| Hailo `nn_core_utilization` | 90-100% | 0-30% |
| GPU `device_healthy` | true | false or true (low util) |
| Hailo `device_healthy` | true | false or true (low util) |
| CPU utilization | 20-40% | 100% (stress) |
| System power | ~65W | ~95W (peak) |
| Temperature | 72-78C GPU / 45-55C Hailo | 85C+ GPU / 60C+ Hailo |

### Watchdog Expected Response

```
T+0ms:     Both faults injected

T+500ms:   Watchdog samples:
           GPU: gpu_busy_percent = 12%, healthy = true
           Hailo: nn_core_utilization = 0%, device_healthy = false
           
           → Hailo: !healthy → IMMEDIATE RECOVERY
             → trigger_recovery(HAILO_8L) starts on watchdog thread
             
           → GPU: util = 12% (< 40% critical)
             → CRITICAL zone → IMMEDIATE RECOVERY
             → trigger_recovery(GPU_780M) starts on watchdog thread

CRITICAL: Both recoveries run CONCURRENTLY (same thread, sequential)
           The monitor_loop() is single-threaded:
           
           check_device(gpu_state_, ...)   // triggers GPU recovery
           check_device(hailo_state_, ...) // triggers Hailo recovery
           
           Both call trigger_recovery() in sequence on the same thread.
           This is actually sequential, not parallel.
           
           However, the recovery callback runs on the watchdog thread,
           which may block for several seconds (session rebuild).
           
           The second trigger_recovery() will start after the first completes.

T+500ms:   GPU recovery begins:
           1. snapshot_latents()
           2. flux_gpu_session_.reset()
           3. hipDeviceReset()  ← blocks for ~200ms
           4. hipSetDevice(0); hipInit(0)
           5. Rebuild session ← ~500ms
           6. restore_latents()
           7. Returns SUCCESS
           
T+2500ms:  GPU recovery complete
           Hailo recovery begins:
           1. hailo_session_.reset()
           2. hard_reset()  ← blocks for ~300ms
           3. Rebuild session ← ~500ms
           4. Returns SUCCESS (or PARTIAL)
           
T+4500ms:  Both recoveries complete
           Watchdog resumes normal monitoring
```

**Potential issue:** The sequential recovery means the second device waits. If the GPU recovery takes 2 seconds, the Hailo recovery doesn't start until then. During this time, Hailo remains faulted.

**Mitigation:** The current implementation uses `state.in_recovery` to prevent concurrent recovery of the same device. For concurrent multi-device faults, the sequential handling is acceptable because:
1. Each recovery is relatively fast (< 3 seconds)
2. The pipeline is already paused during recovery
3. No deadlock risk since it's the same thread

### State Preservation Check

```cpp
TEST(Scenario8, IndependentRecoveryNoInterference) {
    // 1. Capture pre-fault state
    auto state_before = StateVerifier::capture();
    auto embeddings_before = capture_text_embeddings();
    
    // 2. Inject concurrent faults
    inject_concurrent_fault();
    
    // 3. Wait for both recoveries
    wait_for_recovery(ComputeUnit::GPU_780M, 30s);
    wait_for_recovery(ComputeUnit::HAILO_8L, 30s);
    
    // 4. Verify both devices healthy
    auto util = watchdog->current();
    EXPECT_TRUE(util.gpu_healthy) << "GPU should be healthy after recovery";
    EXPECT_TRUE(util.hailo_healthy) << "Hailo should be healthy after recovery";
    
    // 5. Verify GPU latents preserved
    auto state_after = StateVerifier::capture();
    auto cmp = StateVerifier::compare(state_before, state_after);
    EXPECT_LT(cmp.mse, 1e-6f) << "GPU latent MSE after concurrent recovery";
    
    // 6. Verify Hailo embeddings valid
    auto embeddings_after = capture_text_embeddings();
    float cos_sim = cosine_similarity(embeddings_before, embeddings_after);
    EXPECT_GT(cos_sim, 0.95f) << "Embeddings valid after concurrent recovery";
    
    // 7. Verify no cross-interference
    // GPU session should not reference Hailo memory
    // Hailo session should not reference GPU memory
    EXPECT_TRUE(memories_are_independent())
        << "GPU and Hailo memory should be independent";
}
```

### Success Criteria

| # | Criterion | Pass Condition |
|---|---|---|
| 8.1 | Both faults detected | Both devices report low utilization or unhealthy |
| 8.2 | Both recoveries triggered | `trigger_recovery()` called for both devices |
| 8.3 | No deadlock | Both recoveries complete within 10 seconds |
| 8.4 | GPU recovers independently | GPU session rebuilt, util returns to normal |
| 8.5 | Hailo recovers independently | Hailo session rebuilt, util returns to normal |
| 8.6 | No cross-interference | GPU recovery doesn't corrupt Hailo state, vice versa |
| 8.7 | Latents preserved | GPU latent MSE < 1e-6 |
| 8.8 | Embeddings valid | Hailo embeddings semantically consistent |

### Risk of False Positive

**Low.** Simultaneous faults on both independent devices are extremely rare in normal operation. The only realistic trigger is a system-wide event (power transient, thermal emergency), which should indeed trigger recovery.

---

## Scenario 9: Memory Bandwidth Saturation (Bonus)

### Overview

| Field | Content |
|---|---|
| **Name** | Memory Bandwidth Saturation |
| **Target Component** | LPDDR5X-7500 memory bus (both GPU + Hailo affected) |
| **Target Failure Mode** | LPDDR5X-7500 bus saturated, both GPU and Hailo stall |
| **Severity** | High — tests external-root-cause handling |
| **Duration** | ~20 seconds |

### Background

The UM790 Pro uses LPDDR5X-7500 memory providing ~83 GB/s bandwidth. This is shared between:
- CPU (Zen 4 cores)
- GPU (Radeon 780M via UMA)
- Hailo-8L (via PCIe DMA to host memory)

When the memory bus is saturated by an external workload, both GPU and Hailo stall waiting for memory. The watchdog sees low utilization on both devices and triggers recovery. However, recovery cannot fix the root cause (external bandwidth contention).

### Injection Mechanism

```cpp
void inject_bandwidth_saturation() {
    // Launch memory-intensive CPU workload alongside inference
    // This saturates the LPDDR5X bus
    
    // Method 1: Stream copy benchmark (memcpy at maximum bandwidth)
    bandwidth_stress_threads_.clear();
    for (int i = 0; i < 4; ++i) {  // 4 threads to maximize bandwidth
        bandwidth_stress_threads_.emplace_back([this, i]() {
            const size_t BUFFER_SIZE = 256 * 1024 * 1024;  // 256 MB per thread
            auto src = std::make_unique<char[]>(BUFFER_SIZE);
            auto dst = std::make_unique<char[]>(BUFFER_SIZE);
            
            // Initialize with random data
            std::random_device rd;
            std::mt19937 gen(rd());
            std::generate(src.get(), src.get() + BUFFER_SIZE, gen);
            
            while (!stop_bandwidth_stress_.load()) {
                // Saturate memory bus with copies
                std::memcpy(dst.get(), src.get(), BUFFER_SIZE);
                // Prevent cache residency — force DRAM access
                _mm_clflush(dst.get());  // x86 cache line flush
                _mm_mfence();            // Memory fence
            }
        });
    }
    
    // Method 2: Use STREAM benchmark binary
    // std::system("./stream_c.exe &");  // McCalpin STREAM
}

void release_bandwidth_saturation() {
    stop_bandwidth_stress_.store(true);
    for (auto& t : bandwidth_stress_threads_) {
        if (t.joinable()) t.join();
    }
    bandwidth_stress_threads_.clear();
}
```

**Monitor memory bandwidth usage:**
```bash
# Using pcm-memory (Intel Performance Counter Monitor)
# or custom AMD uProf counters
# Target: sustained >75 GB/s (out of 83 GB/s peak)

# Alternative: Monitor via perf
perf stat -e uncore_imc/clockticks/ -a sleep 1
```

**Configuration:**
```yaml
# test_config/bandwidth_saturation.yaml
fault_injection:
  type: BANDWIDTH_SATURATION
  variants:
    - name: "moderate_saturation"
      method: "stream_copy_4_threads"
      target_bandwidth_gb_s: 60
      duration_seconds: 15
      expected_behavior: "utilization drop on both devices"
    - name: "severe_saturation"
      method: "stream_copy_8_threads"
      target_bandwidth_gb_s: 80
      duration_seconds: 15
      expected_behavior: "severe utilization drop, recovery likely fails"
```

### Expected Device Behavior

| Metric | Normal | Moderate Saturation | Severe Saturation |
|---|---|---|---|
| Memory bandwidth | 20-40 GB/s | 55-65 GB/s | 75-83 GB/s |
| GPU `gpu_busy_percent` | 65-80% | 45-60% | 25-40% |
| Hailo `nn_core_utilization` | 90-100% | 50-70% | 20-40% |
| CPU iowait | < 2% | 15-25% | 40-60% |
| Generation time/step | ~35ms | ~55ms | ~90ms |
| Recovery success | N/A | 50% | 10% |

### Watchdog Expected Response

```
T+0s:    Bandwidth saturation begins
         → Memory bus at 75+ GB/s
         
T+2s:    Watchdog samples:
         GPU: gpu_busy_percent = 38% (CRITICAL)
         Hailo: nn_core_utilization = 35% (CRITICAL)
         
         → Both devices trigger IMMEDIATE RECOVERY
         
T+2.5s:  GPU recovery: hipDeviceReset() + session rebuild
         Hailo recovery: hard_reset() + HEF reload
         
         → Both recoveries complete successfully
         → But: memory bus still saturated
         
T+5s:    Watchdog samples:
         GPU: gpu_busy_percent = 40% (CRITICAL)
         Hailo: nn_core_utilization = 38% (CRITICAL)
         
         → RECOVERY LOOP BEGINS
         
T+5-15s: Repeated recoveries every 4-8 steps
         → Recovery #3: FATAL escalation (per Scenario 7)
         → Both devices marked FATAL
         → Pipeline falls back to CPU-only or aborts
         
T+15s:   Bandwidth stress released
         → Memory bus returns to 20-40 GB/s
         → But devices are already FATAL
         → Manual intervention required (or watchdog restart)
```

### Key Insight

Recovery **cannot** fix bandwidth saturation. The watchdog should ideally:
1. Detect that recovery didn't improve utilization
2. Check external system load (CPU iowait, memory bandwidth)
3. Log: "Recovery failed — external resource contention suspected"
4. Escalate to FATAL faster than for internal faults

### State Preservation Check

```cpp
TEST(Scenario9, BandwidthSaturationHandling) {
    // 1. Capture state
    auto state_before = StateVerifier::capture();
    
    // 2. Start bandwidth saturation
    inject_bandwidth_saturation();
    
    // 3. Run generation for 15 seconds
    std::this_thread::sleep_for(std::chrono::seconds(15));
    
    // 4. Stop bandwidth stress
    release_bandwidth_saturation();
    
    // 5. Check results
    auto util = watchdog->current();
    
    // Watchdog should have attempted recovery
    EXPECT_GE(get_total_recovery_count(), 1)
        << "Watchdog should attempt recovery for bandwidth-related low util";
    
    // If multiple recoveries occurred, should escalate
    if (get_total_recovery_count() >= 3) {
        EXPECT_FALSE(util.gpu_healthy || util.hailo_healthy)
            << "Should escalate to FATAL after repeated failed recoveries";
    }
    
    // 6. After releasing bandwidth, should return to normal
    // (may need manual watchdog restart if devices marked FATAL)
    std::this_thread::sleep_for(std::chrono::seconds(5));
    
    // If devices are still healthy, utilization should recover
    if (util.gpu_healthy) {
        EXPECT_GT(watchdog->current().gpu_percent, 55.0f)
            << "GPU utilization should recover after bandwidth release";
    }
}
```

### Success Criteria

| # | Criterion | Pass Condition |
|---|---|---|
| 9.1 | Bandwidth saturation achieved | Memory bus > 75% of peak |
| 9.2 | Both devices affected | Both utilizations drop below threshold |
| 9.3 | Recovery attempts made | Watchdog attempts recovery |
| 9.4 | Root cause persistence | Utilization remains low after recovery (external cause) |
| 9.5 | Escalation | After N failed attempts, escalate to FATAL |
| 9.6 | Post-release recovery | When bandwidth released, utilization recovers (if not FATAL) |

### Risk of False Positive

**HIGH.** Memory bandwidth saturation from other applications is a realistic multi-tasking scenario. The watchdog should ideally be enhanced with bandwidth-awareness (check CPU iowait / memory bandwidth utilization before recovering). Without this enhancement, the test documents the expected behavior (recovery loop → FATAL) and validates the escalation path.

---

## Scenario 10: Watchdog State Machine Edge Cases

### Overview

| Field | Content |
|---|---|
| **Name** | Watchdog State Machine Edge Cases |
| **Target Component** | UtilizationWatchdog state machine |
| **Target Failure Mode** | Boundary conditions in threshold logic, race conditions |
| **Severity** | Medium — validates correctness of core state machine |
| **Duration** | ~10 seconds per sub-test |

### Sub-Tests

#### 10a: Exact Threshold Boundary (60.0%)

```cpp
TEST(Scenario10a, ExactThresholdBoundary) {
    // Set utilization to exactly 60.0% — should be NORMAL
    // (util < low_threshold triggers warning, util >= low_threshold is normal)
    // low_threshold = 60.0f
    // Condition: util < cfg_.low_threshold → warning
    // So 60.0 is NOT < 60.0 → should be NORMAL
    
    simulate_gpu_utilization(60.0f);
    for (int i = 0; i < 16; ++i) {
        std::this_thread::sleep_for(500ms);
    }
    
    EXPECT_EQ(watchdog->current().gpu_below_threshold_steps, 0)
        << "Exactly 60% should be normal, not warning";
}
```

#### 10b: Oscillating Utilization (59% -> 61% -> 59%)

```cpp
TEST(Scenario10b, OscillatingUtilization) {
    // Utilization oscillates around threshold
    // Should NOT accumulate consecutive steps if it goes above threshold
    
    int step = 0;
    auto oscillation_timer = std::thread([this, &step]() {
        while (step < 20) {
            float util = (step % 2 == 0) ? 59.0f : 61.0f;
            simulate_gpu_utilization(util);
            std::this_thread::sleep_for(500ms);
            ++step;
        }
    });
    
    oscillation_timer.join();
    
    // Counter should never exceed 1 (resets on 61% sample)
    EXPECT_LE(watchdog->current().gpu_below_threshold_steps, 1)
        << "Oscillating utilization should not accumulate consecutive steps";
}
```

#### 10c: Immediate Recovery During Active Recovery

```cpp
TEST(Scenario10c, RecoveryDuringRecovery) {
    // Trigger recovery, then trigger another fault before first recovery completes
    // The in_recovery flag should prevent re-entry
    
    // 1. Inject GPU fault
    simulate_gpu_utilization(10.0f);
    std::this_thread::sleep_for(500ms);
    
    // 2. Recovery starts (in_recovery = true)
    // 3. While recovery is in progress, inject another fault
    simulate_gpu_utilization(5.0f);
    std::this_thread::sleep_for(500ms);
    
    // 4. Verify only ONE recovery callback was invoked
    EXPECT_EQ(get_gpu_recovery_count(), 1)
        << "Only one recovery should run at a time";
}
```

#### 10d: Step Number Synchronization

```cpp
TEST(Scenario10d, StepNumberSync) {
    // Verify watchdog step tracking matches pipeline step
    for (int step = 0; step < 50; ++step) {
        watchdog_->report_step(step);
        
        // Verify watchdog sees correct step
        EXPECT_EQ(get_watchdog_current_step(), step)
            << "Watchdog step should match reported step";
        
        run_denoising_step(step);
    }
}
```

#### 10e: Rapid Start/Stop Cycling

```cpp
TEST(Scenario10e, RapidStartStop) {
    // Create and destroy watchdog rapidly
    for (int i = 0; i < 10; ++i) {
        auto wd = std::make_unique<UtilizationWatchdog>(
            config, recovery_fn, alert_fn, sample_fn
        );
        std::this_thread::sleep_for(100ms);
        wd.reset();  // Destructor should clean up jthread
        // No crash, no memory leak
    }
    // Verify no thread leaks
    EXPECT_LE(count_threads_with_name("watchdog"), 0);
}
```

### Success Criteria

| # | Criterion | Pass Condition |
|---|---|---|
| 10.1 | Boundary at 60% | Exactly 60% → NORMAL, 59.99% → WARNING |
| 10.2 | Oscillation handling | Counter resets when utilization goes above threshold |
| 10.3 | Recovery re-entry prevention | `in_recovery` flag prevents concurrent recovery |
| 10.4 | Step sync | Watchdog step matches pipeline step |
| 10.5 | Clean shutdown | Destructor properly joins jthread |

---

## Scenario 11: Cross-Correlation Noise Injection

### Overview

| Field | Content |
|---|---|
| **Name** | Cross-Correlation Noise Injection |
| **Target Component** | Watchdog sampling + correlation detection |
| **Target Failure Mode** | Random noise in utilization samples confuses watchdog |
| **Severity** | Low — validates robustness against noisy samples |
| **Duration** | ~15 seconds |

### Background

ROCm SMI's `rsmi_dev_gpu_busy_percent_get()` samples the SMU's accumulated busy counter. Under certain conditions (driver interrupt, SMU firmware update, power state transition), the sample may be noisy or temporarily invalid. The watchdog must be robust to occasional noisy samples.

### Injection Mechanism

```cpp
void inject_utilization_noise() {
    // Inject random noise into utilization samples
    // 20% of samples are "noise" (wildly different from true value)
    
    noise_generator_ = std::make_unique<std::thread>([this]() {
        std::mt19937 gen(std::random_device{}());
        std::bernoulli_dist inject_noise(0.2);  // 20% noise rate
        std::uniform_real_dist<float> noise_util(0.0f, 100.0f);
        
        while (!stop_noise_.load()) {
            if (inject_noise(gen)) {
                // Inject a noisy sample
                float fake_util = noise_util(gen);
                rsmi_override_busy_percent_fn_ = [fake_util](uint32_t* busy) {
                    *busy = static_cast<uint32_t>(fake_util);
                    return RSMI_STATUS_SUCCESS;
                };
            } else {
                // Clear override
                rsmi_override_busy_percent_fn_ = nullptr;
            }
            std::this_thread::sleep_for(100ms);
        }
    });
}
```

### Watchdog Expected Response

```
With 20% noise rate and 500ms sampling:
- ~2 noisy samples per 10 samples
- Noise values: random 0-100%
- If noise value < 60%: consecutive counter increments
- If noise value >= 60%: counter resets

Expected: Due to the 8-step consecutive threshold, occasional noise spikes
below 60% should NOT trigger recovery. Only sustained noise (or sustained
actual low utilization) should trigger.

Probability analysis:
- Single noise sample < 60%: 60% chance (uniform 0-100)
- 8 consecutive noisy samples all < 60%: 0.6^8 = 1.7%
- With 20% noise rate, getting 8 consecutive noisy samples is extremely unlikely

Result: Watchdog should be robust to 20% noise. Should NOT trigger recovery.
```

### Success Criteria

| # | Criterion | Pass Condition |
|---|---|---|
| 11.1 | Noise tolerance | No false recovery with 20% noise rate |
| 11.2 | Real fault detection | Recovery still triggers for sustained real low util |
| 11.3 | No counter overflow | Consecutive counter handles noise correctly |

---

## Test Execution Matrix

### Priority Order

| Priority | Scenario | Duration | Hardware Risk | CI Suitable |
|---|---|---|---|---|
| P0 | Scenario 4: Hailo Firmware Fault | 10s | Low (software) | Yes |
| P0 | Scenario 1: GPU Scheduling Gap | 30s | None | Yes |
| P0 | Scenario 8: Concurrent Multi-Device | 20s | None | Yes |
| P1 | Scenario 5: ROCm Session Corruption | 15s | None | Yes |
| P1 | Scenario 2: Hailo PCIe DMA Starvation | 20s | None | Yes |
| P1 | Scenario 7: Recovery Loop | 60s | None | Yes |
| P2 | Scenario 3: Thermal Throttling | 30min | Low | No |
| P2 | Scenario 6: XNACK Page Fault | 30s | Low | Yes |
| P2 | Scenario 9: Bandwidth Saturation | 20s | None | Yes |
| P3 | Scenario 10: State Machine Edges | 60s | None | Yes |
| P3 | Scenario 11: Noise Injection | 15s | None | Yes |

### CI Test Suite (Fast)

```bash
#!/bin/bash
# run_ci_tests.sh — Fast test suite for CI (< 5 minutes)

./um790_watchdog_test \
    --gtest_filter="Scenario1*:Scenario4*:Scenario5*:Scenario8*:Scenario10*" \
    --test_duration_factor=0.5 \
    --use_software_faults_only=true

# Expected: ~3 minutes, all tests pass
```

### Full Validation Suite

```bash
#!/bin/bash
# run_full_tests.sh — Complete validation (< 45 minutes)

./um790_watchdog_test \
    --gtest_filter="*" \
    --test_duration_factor=1.0 \
    --use_software_faults_only=false \
    --thermal_test_enabled=true \
    --bandwidth_test_enabled=true

# Expected: ~40 minutes, comprehensive coverage
```

---

## Instrumentation & Verification Framework

### Required Metrics Collection

```cpp
// Full instrumentation for all scenarios
struct TestMetrics {
    // Per-step utilization log
    std::vector<UtilSnapshot> utilization_history;
    
    // Recovery events
    struct RecoveryEvent {
        ComputeUnit device;
        int step;
        float util_at_fault;
        RecoveryResult result;
        std::chrono::steady_clock::time_point start_time;
        std::chrono::steady_clock::time_point end_time;
        int duration_ms;
    };
    std::vector<RecoveryEvent> recovery_events;
    
    // State preservation
    struct StateCheck {
        int step;
        float latent_mse;
        float embedding_cosine_sim;
        bool passed;
    };
    std::vector<StateCheck> state_checks;
    
    // System metrics
    struct SystemSnapshot {
        float ram_usage_fraction;
        float cpu_iowait_percent;
        float memory_bandwidth_gb_s;
        float gpu_temperature;
        float hailo_temperature;
    };
    std::vector<SystemSnapshot> system_history;
};
```

### Log Analysis Verification

```python
# verify_test_results.py — Post-test log analysis
import re
import json

def analyze_watchdog_log(log_path):
    """Parse watchdog.log and verify expected behavior."""
    
    with open(log_path) as f:
        lines = f.readlines()
    
    # Count recovery triggers
    recovery_triggers = [l for l in lines if "RECOVERY TRIGGERED" in l]
    recovery_successes = [l for l in lines if "RECOVERY SUCCESS" in l]
    recovery_failures = [l for l in lines if "RECOVERY FAILED" in l]
    
    # Check for false positives
    thermal_warnings = [l for l in lines if "thermal throttling" in l.lower()]
    
    # Check for escalation
    fatal_escalations = [l for l in lines if "FATAL" in l]
    
    return {
        "recovery_triggers": len(recovery_triggers),
        "recovery_successes": len(recovery_successes),
        "recovery_failures": len(recovery_failures),
        "thermal_warnings": len(thermal_warnings),
        "fatal_escalations": len(fatal_escalations),
    }

def verify_scenario(scenario_name, log_path, expected):
    results = analyze_watchdog_log(log_path)
    
    for key, expected_value in expected.items():
        actual = results.get(key, 0)
        assert actual == expected_value, \
            f"{scenario_name}: Expected {key}={expected_value}, got {actual}"
    
    print(f"  {scenario_name}: PASS")
```

---

## Appendix A: Proposed Watchdog Enhancements

Based on the failure scenarios, the following enhancements are recommended:

### A1: Thermal-Aware Recovery Suppression

```cpp
// Prevent false-positive recovery during thermal throttling
constexpr float THERMAL_THROTTLE_TEMP_C = 85.0f;

if (is_thermal_throttling(state, util, temp)) {
    state.consecutive_below.store(0);  // Reset counter
    log("WARN", "Thermal throttling suspected — suppressing recovery");
    return;
}
```

### A2: Memory-Pressure Awareness

```cpp
// Check system memory before recovering
float get_ram_usage_fraction() {
    struct sysinfo info;
    sysinfo(&info);
    return 1.0f - (float)info.freeram / info.totalram;
}

if (get_ram_usage_fraction() > 0.85f && util < UTIL_THRESHOLD) {
    log("WARN", "High memory pressure — recovery may not help");
    // Still recover, but with reduced expectations
}
```

### A3: Recovery Loop Prevention

```cpp
constexpr int MAX_RECOVERIES = 3;

if (state.total_recoveries >= MAX_RECOVERIES) {
    log("FATAL", "Recovery loop detected — escalating to FATAL");
    state.healthy.store(false);
    return;
}
```

### A4: Utilization Sample Smoothing

```cpp
// Exponential moving average to reduce noise sensitivity
float smoothed_util = EMA_ALPHA * raw_util + (1 - EMA_ALPHA) * prev_smoothed;
// Use smoothed_util for threshold checks instead of raw
```

---

## Appendix B: Test-Only Build Flags

```cmake
# CMakeLists.txt — Test configuration
target_compile_definitions(um790_watchdog_test PRIVATE
    FAULT_INJECTION_BUILD=1
    WATCHDOG_TEST_HOOKS=1
)

# These flags enable:
# - rsmi_override_busy_percent_fn_ (GPU util override)
# - hailo_test_inject_fault_ (Hailo fault simulation)
# - simulate_thermal_conditions() (temperature override)
# - simulate_memory_pressure() (RAM pressure)
```

---

*Document version: 1.0*
*Target system: UM790 Pro (Ryzen 9 7940HS) + Radeon 780M + Hailo-8L*
*Test framework: Google Test + custom fault injection harness*
