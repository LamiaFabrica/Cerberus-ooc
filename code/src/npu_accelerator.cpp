/// @file npu_accelerator.cpp
/// @copyright Copyright (c) 2026 D Hargreaves (AKA Roylepython). LamiaFabrica. All rights reserved.
/// NPU post-processing accelerator — implementation.
///
/// Provides concrete post-processor implementations and the factory that
/// probes hardware to select the best available backend.
///
/// @author LamiaFabrica Team
/// @version 1.0.0

#include "hq/npu_accelerator.hpp"
#include "hq/logger.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>

namespace hq::npu {

// ===========================================================================
// SyntheticNpuPostProcessor
// ===========================================================================

std::expected<NpuPostProcessResult, std::string>
SyntheticNpuPostProcessor::post_process(const NpuPostProcessRequest& req) {
    if (req.pixels.empty()) {
        return std::unexpected{"SyntheticNpuPostProcessor: empty pixel input"};
    }
    if (req.width == 0 || req.height == 0) {
        return std::unexpected{"SyntheticNpuPostProcessor: zero dimensions"};
    }

    const auto t0 = std::chrono::high_resolution_clock::now();

    NpuPostProcessResult result;
    result.width  = req.width;
    result.height = req.height;

    // Pass-through: copy pixels unchanged.
    // In production, this slot would run ESRGAN / noise-reduction on Hailo-8L.
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
SyntheticNpuPostProcessor::blend_noise_cfg(
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

    // SAXPY in-place: noise_out[i] = noise_uncond[i] + scale*(noise_out[i] - noise_uncond[i])
    //
    // This is the CFG blend formerly at pipeline_integration.cpp:1065-1070.
    // Production path on Hailo-8L: submitted as a SAXPY kernel on the NN core.
    // The Hailo-8L can execute element-wise float32 ops at ~13 TOPS; for 16,384 floats
    // this is <<1µs compute (bandwidth-bound at ~2GB/s PCIe — ~33µs transfer).
    // Current path: CPU scalar loop, timed.
    for (std::size_t i = 0; i < noise_out.size(); ++i) {
        noise_out[i] = noise_uncond[i] + guidance_scale * (noise_out[i] - noise_uncond[i]);
    }
    return {};
}

bool SyntheticNpuPostProcessor::can_handle(NpuTaskType task) const {
    // Synthetic handles all task types as a CPU fallback.
    (void)task;
    return true;
}

// ===========================================================================
// HailoNpuPostProcessor — skeleton (HailoRT not installed)
// ===========================================================================

HailoNpuPostProcessor::HailoNpuPostProcessor() {
    // Production path (not yet implemented):
    //   1. Call hailort::Device::scan_pcie() to enumerate devices
    //   2. Load a compiled HEF (post-processing network)
    //   3. Set hailo_available_ = true
    //
    // Skeleton: always unavailable until HailoRT SDK + HEF present on Linux.
    HQ_LOG_INFO("[HailoNpuPostProcessor] Skeleton mode: HailoRT not installed, "
                "post-processing will delegate to synthetic fallback");
}

std::expected<NpuPostProcessResult, std::string>
HailoNpuPostProcessor::post_process(const NpuPostProcessRequest& req) {
    // Production path:
    //   Submit req.pixels as input tensor to Hailo async infer runner.
    //   Wait for callback, copy output to NpuPostProcessResult::pixels.
    //
    // Skeleton: delegate to synthetic (CPU pass-through).
    HQ_LOG_INFO("[HailoNpuPostProcessor] Skeleton — delegating to synthetic post-processor");
    return synthetic_fallback_.post_process(req);
}

std::expected<void, std::string>
HailoNpuPostProcessor::blend_noise_cfg(
    std::span<float>       noise_out,
    std::span<const float> noise_uncond,
    float                  guidance_scale) noexcept {
    // Production path (when HailoRT installed + SAXPY HEF compiled):
    //   1. Copy noise_out and noise_uncond to Hailo input buffers via PCIe DMA
    //   2. Submit SAXPY HEF with guidance_scale as a constant parameter
    //   3. Wait for async completion callback
    //   4. Copy Hailo output buffer back to noise_out
    //
    // Blockers (as of 2026-05-22):
    //   - HailoRT SDK not installed (Windows; Linux-only)
    //   - No SAXPY HEF compiled for Hailo-8L
    //   - PCIe latency (~33µs for 16,384 floats) dominates compute time at this tensor size
    //
    // Skeleton: delegate to SyntheticNpuPostProcessor (CPU SAXPY).
    return synthetic_fallback_.blend_noise_cfg(noise_out, noise_uncond, guidance_scale);
}

bool HailoNpuPostProcessor::can_handle(NpuTaskType task) const {
    // Skeleton reports false for all real tasks — synthetic fallback handles everything.
    (void)task;
    return false;
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

    // ---- Priority 2: Synthetic (always available) ----
    HQ_LOG_INFO("[NpuPostProcessorFactory] Hailo-8L unavailable, using Synthetic pass-through");
    return std::make_unique<SyntheticNpuPostProcessor>();
}

} // namespace hq::npu
