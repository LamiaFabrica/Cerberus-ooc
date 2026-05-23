/// @file npu_accelerator.cpp
/// @copyright Copyright (c) 2026 D Hargreaves (AKA Roylepython). LamiaFabrica. All rights reserved.
/// NPU post-processing accelerator — implementation.
///
/// Provides concrete post-processor implementations and the factory that
/// probes hardware to select the best available backend.
///
/// @version 2.0.0 — synthetic data eradicated. No fabricated telemetry.
/// @author LamiaFabrica Team

#include "hq/npu_accelerator.hpp"
#include "hq/logger.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>

// ===========================================================================
// HailoRT three-tier detection
// ===========================================================================
#ifdef HAILO_BUILD
#  define HAILO_ACCEL_HAS_HAILORT 1
#  include <hailort/hailort.h>
#else
#  if __has_include(<hailort/hailort.h>)
#    define HAILO_ACCEL_HAS_HAILORT 1
#    include <hailort/hailort.h>
#  elif __has_include(<hailort/hailort.hpp>)
#    define HAILO_ACCEL_HAS_HAILORT 1
#    include <hailort/hailort.hpp>
#  else
#    define HAILO_ACCEL_HAS_HAILORT 0
#    pragma message("HailoRT headers not found — HailoNpuPostProcessor compiled without HailoRT support")
#  endif
#endif

namespace hq::npu {

// ===========================================================================
// CpuPostProcessor — honest CPU pass-through (no NPU acceleration)
// ===========================================================================

std::expected<NpuPostProcessResult, std::string>
CpuPostProcessor::post_process(const NpuPostProcessRequest& req) {
    if (req.pixels.empty()) {
        return std::unexpected{"CpuPostProcessor: empty pixel input"};
    }
    if (req.width == 0 || req.height == 0) {
        return std::unexpected{"CpuPostProcessor: zero dimensions"};
    }

    const auto t0 = std::chrono::high_resolution_clock::now();

    NpuPostProcessResult result;
    result.width  = req.width;
    result.height = req.height;

    // CPU pass-through: copy pixels unchanged.
    // When a real NPU post-processing network (e.g. ESRGAN on Hailo-8L) is
    // available, HailoNpuPostProcessor will perform the work. Until then,
    // the pipeline uses this honest CPU fallback.
    result.pixels.resize(req.pixels.size());
    std::memcpy(result.pixels.data(), req.pixels.data(), req.pixels.size());

    const auto t1 = std::chrono::high_resolution_clock::now();
    result.processing_time_us = static_cast<float>(
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    result.npu_utilization    = 0.0f;   // CPU path — no NPU activity
    result.was_npu_accelerated = false;

    return result;
}

std::expected<void, std::string>
CpuPostProcessor::blend_noise_cfg(
    std::span<float>       noise_out,
    std::span<const float> noise_uncond,
    float                  guidance_scale) noexcept {
    if (noise_out.empty()) {
        return std::unexpected{std::string{"blend_noise_cfg: empty noise_out"}};
    }
    if (noise_out.size() != noise_uncond.size()) {
        return std::unexpected{std::string{"blend_noise_cfg: size mismatch cond="}
            + std::to_string(noise_out.size())
            + " uncond=" + std::to_string(noise_uncond.size())};
    }

    // CFG blend: noise_out[i] = noise_uncond[i] + scale*(noise_out[i] - noise_uncond[i])
    //
    // This is pure CPU scalar math. A real Hailo-8L implementation would
    // submit this as a SAXPY kernel on the NN core, but that requires:
    //   1. HailoRT SDK installed (Linux only)
    //   2. A compiled SAXPY HEF for Hailo-8L
    //   3. PCIe DMA path for input/output tensors
    //
    // Until those conditions are met, this honest CPU fallback performs
    // the blend without claiming NPU acceleration.
    for (std::size_t i = 0; i < noise_out.size(); ++i) {
        noise_out[i] = noise_uncond[i] + guidance_scale * (noise_out[i] - noise_uncond[i]);
    }
    return {};
}

bool CpuPostProcessor::can_handle(NpuTaskType task) const {
    // CPU fallback handles all task types because it is a general-purpose
    // compute path. It does not claim to accelerate anything.
    (void)task;
    return true;
}

// ===========================================================================
// HailoNpuPostProcessor — production Hailo-8L post-processing
// ===========================================================================

HailoNpuPostProcessor::HailoNpuPostProcessor() {
#if HAILO_ACCEL_HAS_HAILORT
    // Probe for Hailo devices
    auto scan_result = hailort::Device::scan_pcie();
    if (scan_result && !scan_result.value().empty()) {
        HQ_LOG_INFO("[HailoNpuPostProcessor] Hailo-8L detected — post-processing available when HEF loaded");
    } else {
        HQ_LOG_INFO("[HailoNpuPostProcessor] No Hailo-8L device detected");
    }
#else
    HQ_LOG_INFO("[HailoNpuPostProcessor] HailoRT SDK not available — post-processing unavailable");
#endif
}

std::expected<NpuPostProcessResult, std::string>
HailoNpuPostProcessor::post_process(const NpuPostProcessRequest& req) {
    (void)req;
    return std::unexpected{"Hailo-8L post-processing not yet implemented — HEF not loaded"};
}

std::expected<void, std::string>
HailoNpuPostProcessor::blend_noise_cfg(
    std::span<float>       noise_out,
    std::span<const float> noise_uncond,
    float                  guidance_scale) noexcept {
    (void)noise_out;
    (void)noise_uncond;
    (void)guidance_scale;
    return std::unexpected{"Hailo-8L SAXPY not yet implemented — HEF not loaded"};
}

bool HailoNpuPostProcessor::can_handle(NpuTaskType task) const {
    (void)task;
    return false;  // Not yet implemented — honest false
}

// ===========================================================================
// NpuPostProcessorFactory
// ===========================================================================

std::unique_ptr<INpuPostProcessor>
NpuPostProcessorFactory::create_best_available() {
    // ---- Priority 1: Hailo-8L ----
    {
        auto hailo = std::make_unique<HailoNpuPostProcessor>();
        if (hailo->is_available()) {
            HQ_LOG_INFO("[NpuPostProcessorFactory] Using Hailo-8L post-processor");
            return hailo;
        }
    }

    // ---- Priority 2: CPU pass-through (no NPU acceleration) ----
    // CpuPostProcessor performs memcpy and scalar SAXPY on CPU.
    // It does NOT claim to be an NPU accelerator.
    // is_available() returns false because it provides zero NPU acceleration.
    HQ_LOG_INFO("[NpuPostProcessorFactory] No NPU hardware available — "
                "using CPU pass-through (no acceleration). "
                "post_process() will copy pixels unchanged; "
                "blend_noise_cfg() will use CPU scalar math.");
    return std::make_unique<CpuPostProcessor>();
}

} // namespace hq::npu
