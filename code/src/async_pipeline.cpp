/// @file async_pipeline.cpp
/// @copyright Copyright © 2026, all rights reserved | David Hargreaves aka Roylepython trading as The Medusa Initiative & Lamia Fabrica
/// AsyncPipeline coroutine implementations for UM790 Pro async generation.
///
/// @version 1.0.0

#include "hq/async_pipeline.hpp"
#include "hq/cxx26_features.hpp"
#include "hq/deis_scheduler.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <future>
#include <memory>
#if UM790_HAS_STD_PRINT
#  include <print>
#endif
#include <random>
#include <thread>

#ifdef UM790_HAS_HIP
#include <hip/hip_runtime_api.h>
#endif

namespace hq::async {

using Clock = std::chrono::high_resolution_clock;

// ===========================================================================
// AsyncPipeline Construction / Destruction
// ===========================================================================

AsyncPipeline::AsyncPipeline(const PipelineConfig& cfg)
    : pipeline_{std::make_unique<hq::Pipeline>(cfg)}
    , npu_pipeline_{std::make_unique<hq::npu::NpuDmaPipeline>(
          hq::npu::NpuDmaPipeline::Config{
              .num_slots       = 3,
              .embedding_bytes = hq::npu::MAX_EMBEDDING_BYTES,
          })}
{
    std::print("[AsyncPipeline] Initialized async pipeline for UM790 Pro\n");
}

AsyncPipeline::~AsyncPipeline() noexcept {
    shutdown();
}

AsyncPipeline::AsyncPipeline(AsyncPipeline&&) noexcept = default;
AsyncPipeline& AsyncPipeline::operator=(AsyncPipeline&&) noexcept = default;

// ===========================================================================
// generate_async(): fire-and-forget coroutine — runs the full pipeline
// ===========================================================================

task<GeneratedImage> AsyncPipeline::generate_async(GenerationRequest req) {
    const auto t_start = Clock::now();

    if (req.width == 0 || req.height == 0 ||
        req.width > 2048 || req.height > 2048) {
        std::print("[AsyncPipeline] Invalid dimensions: {}x{}\n",
                   req.width, req.height);
        co_return GeneratedImage{};
    }
    if ((req.width % 8U) != 0U || (req.height % 8U) != 0U) {
        std::print("[AsyncPipeline] Dimensions must be multiples of 8: "
                   "{}x{}\n", req.width, req.height);
        co_return GeneratedImage{};
    }
    if (req.num_steps == 0 || req.num_steps > 150) {
        std::print("[AsyncPipeline] Invalid step count: {}\n",
                   req.num_steps);
        co_return GeneratedImage{};
    }

    std::print("[AsyncPipeline] Starting async generation: \"{}\" "
               "({}x{} {} steps)\n",
               req.prompt, req.width, req.height, req.num_steps);

    // Step 1: NPU text encoding via NpuDmaPipeline
    hq::npu::NpuEncodeRequest enc_req{
        .prompt         = req.prompt,
        .guidance_scale = req.guidance_scale,
        .seed           = req.seed,
        .width          = req.width,
        .height         = req.height,
        .num_steps      = req.num_steps,
    };

    auto slot_result = npu_pipeline_->submit(enc_req);
    if (!slot_result.has_value()) {
        std::print("[AsyncPipeline] NPU encode submit failed: {}\n",
                   slot_result.error());
        co_return GeneratedImage{};
    }
    std::size_t slot = *slot_result;

    auto cond_handle = co_await NpuReadyAwaiter{*npu_pipeline_, slot,
        std::chrono::milliseconds{5000}};
    if (!cond_handle) {
        std::print("[AsyncPipeline] NPU conditional encoding timed out\n");
        npu_pipeline_->release_slot(slot);
        co_return GeneratedImage{};
    }

    std::print("[AsyncPipeline] Conditional embeddings ready "
               "(elements={:L})\n", cond_handle.element_count);

    // Step 1b: NPU empty-prompt encoding for CFG
    hq::npu::NpuTensorHandle uncond_handle{};
    std::size_t uncond_slot = 0;
    bool uncond_valid = false;

    if (req.guidance_scale > 1.0f) {
        hq::npu::NpuEncodeRequest uncond_req{
            .prompt         = "",
            .guidance_scale = 1.0f,
            .seed           = req.seed,
            .width          = req.width,
            .height         = req.height,
            .num_steps      = req.num_steps,
        };
        auto uslot = npu_pipeline_->submit(uncond_req);
        if (uslot.has_value()) {
            uncond_slot = *uslot;
            uncond_handle = co_await NpuReadyAwaiter{*npu_pipeline_,
                uncond_slot, std::chrono::milliseconds{5000}};
            uncond_valid = static_cast<bool>(uncond_handle);
            if (uncond_valid) {
                std::print("[AsyncPipeline] Unconditional embeddings ready "
                           "(CFG scale={:.1f})\n", req.guidance_scale);
            }
        }
    }

    // Allow NPU cooldown between encode and denoise phases
    co_await SleepAwaiter{std::chrono::microseconds{100}};

    // Release NPU slots
    npu_pipeline_->release_slot(slot);
    if (uncond_valid) npu_pipeline_->release_slot(uncond_slot);

    const float npu_util_pct = npu_pipeline_->last_npu_utilization();
    const float npu_temp_c   = npu_pipeline_->npu_temperature();

    // Step 2: Run the full pipeline via inner hq::Pipeline
    auto gen_result = pipeline_->generate(req);

    const auto t_end = Clock::now();
    const double total_ms =
        std::chrono::duration<double, std::milli>(t_end - t_start).count();

    std::print("[AsyncPipeline] Done ({:.1f} ms) "
               "NPU: {:.0f}% ({:.0f}C)\n",
               total_ms, npu_util_pct, npu_temp_c);

    GeneratedImage result;
    if (gen_result.has_value()) {
        result = std::move(*gen_result);
        result.generation_time_ms = static_cast<float>(total_ms);
    } else {
        std::print("[AsyncPipeline] Generation failed: {}\n",
                   hq::to_string(gen_result.error()));
    }

    co_return result;
}

// ===========================================================================
// generate_streaming(): yields StepResult after each denoising step
// ===========================================================================

Generator<StepResult> AsyncPipeline::generate_streaming(
    GenerationRequest req) {

    if (req.width == 0 || req.height == 0 ||
        req.width > 2048 || req.height > 2048) {
        std::print("[AsyncPipeline:stream] Invalid dimensions\n");
        co_return;
    }
    if (req.num_steps == 0 || req.num_steps > 150) {
        std::print("[AsyncPipeline:stream] Invalid step count: {}\n",
                   req.num_steps);
        co_return;
    }

    std::print("[AsyncPipeline:stream] Starting streaming: \"{}\" "
               "({}x{} {} steps)\n",
               req.prompt, req.width, req.height, req.num_steps);

    // Submit NPU encode
    hq::npu::NpuEncodeRequest enc_req{
        .prompt         = req.prompt,
        .guidance_scale = req.guidance_scale,
        .seed           = req.seed,
        .width          = req.width,
        .height         = req.height,
        .num_steps      = req.num_steps,
    };

    auto slot_res = npu_pipeline_->submit(enc_req);
    if (!slot_res.has_value()) {
        std::print("[AsyncPipeline:stream] NPU submit failed: {}\n",
                   slot_res.error());
        co_return;
    }
    const std::size_t slot = *slot_res;

    // Wait on NPU encode (non-blocking via awaiter)
    auto handle = co_await NpuReadyAwaiter{*npu_pipeline_, slot,
        std::chrono::milliseconds{5000}};
    npu_pipeline_->release_slot(slot);

    if (!handle) {
        std::print("[AsyncPipeline:stream] NPU encode timed out\n");
        co_return;
    }

    const std::size_t latent_w = req.width / 8;
    const std::size_t latent_h = req.height / 8;
    const std::size_t latent_size = 4ULL * latent_h * latent_w;

    // Float preview buffer: populated each step from the decoded pixel output
    // (normalized to [0,1]). Provides callers a float view of the step result
    // for progressive display without exposing pipeline-internal TMM latents.
    std::vector<float> latents(latent_size, 0.0f);

    // Build single-step requests for step-by-step streaming
    hq::GenerationRequest single_step_req{
        .prompt         = req.prompt,
        .width          = req.width,
        .height         = req.height,
        .num_steps      = 1,
        .guidance_scale = req.guidance_scale,
        .seed           = req.seed,
    };

    for (uint32_t step = 0; step < req.num_steps; ++step) {
        const auto step_t0 = Clock::now();

        auto step_result = pipeline_->generate(single_step_req);

        const auto step_t1 = Clock::now();
        const double step_ms =
            std::chrono::duration<double, std::milli>(
                step_t1 - step_t0).count();

        const float gpu_util  = pipeline_->get_stats().avg_gpu_utilization;
        const float hailo_util = npu_pipeline_->last_npu_utilization();
        const float temp       = npu_pipeline_->npu_temperature();

        if (step_result.has_value()) {
            // Populate float preview buffer from decoded RGBA8 pixels, normalized
            // to [0,1]. Samples every 16th pixel to fit latent_size floats.
            const auto& img = *step_result;
            const std::size_t pixel_stride =
                std::max(std::size_t{1},
                         img.pixels.size() / latent_size);
            for (std::size_t i = 0; i < latent_size; ++i) {
                const std::size_t src = std::min(i * pixel_stride,
                                                  img.pixels.size() - 1u);
                latents[i] = static_cast<float>(img.pixels[src]) / 255.0f;
            }
        } else {
            std::print("[AsyncPipeline:stream] Step {} failed: {}\n",
                       step, hq::to_string(step_result.error()));
        }

        co_yield StepResult{
            .step_index  = step,
            .gpu_util    = gpu_util,
            .hailo_util  = hailo_util,
            .temperature = temp,
            .latency_ms  = step_ms,
            .latents_ptr = latents.data(),
        };
    }

    std::print("[AsyncPipeline:stream] Streaming generation complete\n");
}

// ===========================================================================
// AsyncPipeline utilities
// ===========================================================================

PipelineStats AsyncPipeline::get_stats() const {
    return pipeline_->get_stats();
}

void AsyncPipeline::shutdown() noexcept {
    if (pipeline_) {
        pipeline_->shutdown();
        pipeline_.reset();
    }
    npu_pipeline_.reset();
}

} // namespace hq::async
