/// @file pipeline_integration.cpp
/// @copyright Copyright (c) 2026 D Hargreaves (AKA Roylepython). LamiaFabrica. All rights reserved.
/// Pipeline implementation: watchdog integration, staging, denoising loop,
/// recovery logic, and ONNX Runtime orchestration.
///
/// @version 2.0.0 -- Rewritten for UtilizationWatchdog API with real inference.

#include "hq/pipeline.hpp"
#include "hq/clip_tokenizer.hpp"
#include "hq/cluster_transport.hpp"
#include "hq/cxx26_features.hpp"
#include "hq/deis_scheduler.hpp"
#include "hq/logger.hpp"
#include "hq/npu_accelerator.hpp"
#include "hq/npu_encoder.hpp"
#include "hq/tiered_memory_manager.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <future>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <thread>

// ONNX Runtime C++ API
#include <onnxruntime_cxx_api.h>

// HIP runtime (for GPU memory management)
#ifdef UM790_HAS_HIP
#include <hip/hip_runtime_api.h>
#endif

// NVIDIA CUDA runtime (for GPU compute on RTX 5070 Ti etc.)
#ifdef UM790_HAS_CUDA
#include <cuda_runtime_api.h>
#endif

// NVIDIA NVML (for GPU telemetry on NVIDIA hardware)
#if defined(UM790_HAS_CUDA) && __has_include(<nvml.h>)
#include <nvml.h>
#endif

// ROCm SMI (for GPU utilization and temperature queries on AMD)
#if defined(UM790_HAS_HIP) && __has_include(<rocm_smi/rocm_smi.h>)
#include <rocm_smi/rocm_smi.h>
#define UM790_HAS_ROCM_SMI 1
#else
#define UM790_HAS_ROCM_SMI 0
#endif

namespace hq {

// ===========================================================================
// Cross-platform ORT path helper
// On Windows, Ort::Session takes wchar_t* for model paths (ORTCHAR_T = wchar_t).
// On Linux/macOS, it takes const char* (ORTCHAR_T = char).
// The helper stores the path in the correct string type and returns a pointer
// that remains valid for the duration of the call site expression.
#if defined(_WIN32)
inline std::wstring ort_model_path(const std::filesystem::path& p) {
    return p.wstring();
}
#else
inline std::string ort_model_path(const std::filesystem::path& p) {
    return p.string();
}
#endif

// ===========================================================================
// Internal: ONNX Runtime state (pImpl pattern)
// ===========================================================================
class Pipeline::OrtState {
public:
    Ort::Env                         env{ORT_LOGGING_LEVEL_WARNING, "um790_pipeline"};
    std::unique_ptr<Ort::Session>    gpu_session;
    std::unique_ptr<Ort::Session>    hailo_session;
    std::unique_ptr<Ort::Session>    vae_session;
    Ort::SessionOptions              gpu_options;
    Ort::SessionOptions              hailo_options;
    Ort::SessionOptions              vae_options;
    Ort::MemoryInfo                  memory_info{Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault)};

    explicit OrtState() {
        // GPU session: try CUDA EP first (NVIDIA RTX 5070 Ti), then ROCm EP (AMD), then CPU
#ifdef UM790_HAS_CUDA
        try {
            OrtCUDAProviderOptions cuda_opts{};
            cuda_opts.device_id = 0;
            gpu_options.AppendExecutionProvider_CUDA(cuda_opts);
            HQ_LOG_INFO("CUDA EP registered for GPU session (device 0)");
        } catch (const Ort::Exception& e) {
            HQ_LOG_WARN("CUDA EP not registered: {} — trying ROCm EP fallback", e.what());
#endif
#if defined(UM790_HAS_HIP) || defined(UM790_HAS_CUDA)
            try {
                OrtROCMProviderOptions rocm_opts{};
                rocm_opts.device_id = 0;
                gpu_options.AppendExecutionProvider_ROCM(rocm_opts);
            } catch (const Ort::Exception& e) {
                HQ_LOG_WARN("ROCm EP not registered: {} — falling back to CPU", e.what());
            }
#endif
#ifdef UM790_HAS_CUDA
        }
#endif
        gpu_options.SetIntraOpNumThreads(1);
        gpu_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Hailo/NPU session: try DirectML EP first (Intel AI Boost NPU), then Hailo EP
#ifdef UM790_HAS_DIRECTML
        try {
            hailo_options.AppendExecutionProvider("DML");
            HQ_LOG_INFO("DirectML EP registered for NPU session (device 0)");
        } catch (const Ort::Exception& e) {
            HQ_LOG_WARN("DirectML EP not registered: {} — trying Hailo EP fallback", e.what());
#endif
            try {
                hailo_options.AppendExecutionProvider("Hailo");
            } catch (const Ort::Exception& e) {
                HQ_LOG_WARN("Hailo EP not registered: {} — falling back to CPU", e.what());
            }
#ifdef UM790_HAS_DIRECTML
        }
#endif
        hailo_options.SetIntraOpNumThreads(2);
        hailo_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);

        // VAE session: try CUDA EP for GPU-accelerated VAE decode, fall back to CPU
#ifdef UM790_HAS_CUDA
        try {
            OrtCUDAProviderOptions vae_cuda_opts{};
            vae_cuda_opts.device_id = 0;
            vae_options.AppendExecutionProvider_CUDA(vae_cuda_opts);
            HQ_LOG_INFO("CUDA EP registered for VAE session (device 0)");
        } catch (const Ort::Exception& e) {
            HQ_LOG_WARN("CUDA EP not registered for VAE: {} — using CPU", e.what());
        }
#endif
        vae_options.SetIntraOpNumThreads(2);
        vae_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);

        // NOTE: If all EP registrations above fail (CUDA/DML/ROCm), the sessions
        // will silently fall back to CPU-only inference, which is orders of
        // magnitude slower. Individual EP failures are logged via HQ_LOG_WARN.
        // To diagnose: check log output for "EP not registered" messages.
    }
};

// ===========================================================================
// Pipeline Construction / Destruction
// ===========================================================================

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
              [this](ComputeUnit unit, std::uint32_t step, float util)
                  -> std::expected<RecoveryResult, std::string> {
                  // Watchdog recovery: rebuild session and restore latent checkpoint
                  if (!latent_checkpoint_ || !latent_checkpoint_->valid() || latent_checkpoint_floats_ == 0) {
                      return std::unexpected<std::string>{"Recovery failed: no latent checkpoint available"};
                  }
                  // Use current_latents_ if set, otherwise return error
                  if (!current_latents_.data() || current_latents_.num_elements() == 0) {
                      return std::unexpected<std::string>{"Recovery failed: no valid latent tensor"};
                  }
                  auto result = on_watchdog_recovery_(unit, step, util, current_latents_);
                  if (!result) {
                      return std::unexpected<std::string>{"Recovery failed: " + to_string(result.error())};
                  }
                  return *result;
              },
              [](ComputeUnit unit, std::uint32_t step, float util,
                 const std::string& msg) {
                                     HQ_LOG_INFO("Watchdog alert ({} step={} util={:.1f}%): {}", unit == ComputeUnit::GPU_780M ? "GPU" : "Hailo",
                             step, util, msg);

              })
        : nullptr}
    , hailo_monitor_{std::make_unique<HailoMonitor>()}
    , staging_manager_{std::make_unique<EmbeddingStagingManager>(StagingConfig{
          .buffer_count = cfg.staging_buffer_count,
          .buffer_size_bytes = static_cast<std::size_t>(cfg.staging_buffer_size_mb)
                               * 1024ULL * 1024ULL,
          .pinned       = true,
      })}
    , gpu_monitor_{std::make_unique<GPUMonitor>(0)}
    , tokenizer_{std::make_unique<CLIPTokenizer>()}
    , ort_state_{std::make_unique<OrtState>()}
{
        HQ_LOG_INFO("Initializing UM790 Pipeline v2.0.0");

        HQ_LOG_INFO(" - GPU low threshold:    {}%", cfg_.watchdog_gpu_low_threshold);

        HQ_LOG_INFO(" - GPU critical:         {}%", cfg_.watchdog_gpu_critical_threshold);

        HQ_LOG_INFO(" - Hailo low threshold:  {}%", cfg_.watchdog_hailo_low_threshold);

        HQ_LOG_INFO(" - Hailo critical:       {}%", cfg_.watchdog_hailo_critical_threshold);

        HQ_LOG_INFO(" - Consecutive threshold:{}", cfg_.watchdog_consecutive_threshold);

        HQ_LOG_INFO(" - Max recoveries:       {}", cfg_.watchdog_max_recoveries);

        HQ_LOG_INFO(" - Backoff base/max:     {:.0f}/{:.0f} ms", cfg_.watchdog_backoff_base_ms, cfg_.watchdog_backoff_max_ms);

        HQ_LOG_INFO(" - Thermal threshold:    {:.0f}C", cfg_.watchdog_thermal_threshold_c);

        HQ_LOG_INFO(" - Staging buffers:      {} x {} MiB", cfg_.staging_buffer_count, cfg_.staging_buffer_size_mb);


    // Initialize GPUMonitor (NVML on NVIDIA, ROCm SMI on AMD)
    if (gpu_monitor_) {
        auto init_result = gpu_monitor_->initialize();
        if (init_result) {
            HQ_LOG_INFO("GPUMonitor initialized (backend={}, GPU queries enabled)",
                         hq::to_string(gpu_monitor_->backend()));
        } else {
            HQ_LOG_WARN("GPUMonitor init failed: {} (raw_status={}), GPU telemetry disabled",
                        init_result.error().message, init_result.error().raw_status);
        }
    }

    // Initialize HailoMonitor (attempt real device)
    if (hailo_monitor_) {
        auto open_result = hailo_monitor_->open();
        if (open_result) {
            HQ_LOG_INFO("HailoMonitor opened device '{}' (real hardware telemetry)",
                        hailo_monitor_->device_id());
        } else {
            HQ_LOG_WARN("HailoMonitor open failed: {} — HailoRT not installed or no device found",
                        open_result.error().what());
        }
    }

    // Validate model paths
    if (!cfg_.text_encoder_onnx.empty() && !validate_model_path_(cfg_.text_encoder_onnx)) {
                HQ_LOG_WARN("text encoder model not found: {}", cfg_.text_encoder_onnx.string());

    }
    if (!cfg_.unet_onnx.empty() && !validate_model_path_(cfg_.unet_onnx)) {
                HQ_LOG_WARN("UNet model not found: {}", cfg_.unet_onnx.string());

    }
    if (!cfg_.vae_decoder_onnx.empty() && !validate_model_path_(cfg_.vae_decoder_onnx)) {
                HQ_LOG_WARN("VAE decoder model not found: {}", cfg_.vae_decoder_onnx.string());

    }

    // Initialize DEIS scheduler (inference step count set per-request in generate())
    scheduler_ = std::make_unique<DEISScheduler>(
        SchedulerConfig{
            .type = SchedulerType::DEIS,
            .num_train_timesteps = 1000,
            .beta_start = 0.0001f,
            .beta_end = 0.02f,
            .beta_schedule = "linear",
            .prediction_type = 0,  // epsilon
        },
        20);  // default steps; updated in generate() via set_inference_steps()

    // Attempt initial ONNX session creation
    if (!initialize_onnx_sessions_()) {
                HQ_LOG_WARN("Initial ONNX session creation deferred");

    }

    // Wire INpuEncoder abstraction: factory selects best available backend
    // Priority: Hailo8lEncoder > CpuFallbackEncoder > nullptr (honest failure)
    npu_encoder_ = npu::NpuEncoderFactory::create_best_available(
        ort_state_->hailo_session.get(), &ort_state_->memory_info);
    HQ_LOG_INFO("NPU encoder: {} (available={})",
                npu_encoder_->name(),
                npu_encoder_->is_available() ? "yes" : "no");

    // Wire INpuPostProcessor abstraction: factory selects best available backend
    // Priority: HailoNpuPostProcessor > CpuPostProcessor (honest CPU fallback)
    npu_post_processor_ = npu::NpuPostProcessorFactory::create_best_available();
    HQ_LOG_INFO("NPU post-processor: {} (available={})",
                npu_post_processor_->name(),
                npu_post_processor_->is_available() ? "yes" : "no");

#ifdef UM790_HAS_HIP
    if (cfg_.enable_hip_graph) {
        hip_denoiser_ = std::make_unique<HIPGraphDenoiser>(GraphConfig{
            .num_steps     = 20,
            .latent_count  = 1 * 4 * 64 * 64,
            .enable_capture = true,
        });
        hip_denoiser_->set_scheduler(scheduler_.get());
                HQ_LOG_INFO("HIP graph denoiser initialized");

    }
#endif

    // Tiered memory manager — four-tier heterogeneous memory (Hot/Warm/Cool/Cold)
    memory_manager_ = std::make_unique<TieredMemoryManager>(
        cfg_.memory_config,
        [](TierHandle handle, MemoryTier from_tier, MemoryTier to_tier) {
            HQ_LOG_DEBUG("tier migration: handle={:#x} {} -> {}",
                         handle, to_string(from_tier), to_string(to_tier));
        });
    {
        const auto hot  = memory_manager_->stats(MemoryTier::Hot);
        const auto warm = memory_manager_->stats(MemoryTier::Warm);
        const auto cool = memory_manager_->stats(MemoryTier::Cool);
        HQ_LOG_INFO("TieredMemoryManager: Hot={} Warm={} Cool={}",
                    hot.available  ? "online" : "absent",
                    warm.available ? "online" : "absent",
                    cool.available ? "online" : "absent");
    }

    // Cluster transport (optional — skipped when cluster_config is nullopt)
    if (cfg_.cluster_config.has_value()) {
        transport_ = std::make_unique<cluster::ClusterTransport>(*cfg_.cluster_config);
        auto start_res = transport_->start();
        if (start_res) {
            HQ_LOG_INFO("ClusterTransport started (node_id={} coordinator={})",
                        cfg_.cluster_config->this_node_id,
                        cfg_.cluster_config->is_coordinator);
        } else {
            HQ_LOG_WARN("ClusterTransport failed to start: {} — local-only mode",
                        cluster::to_string(start_res.error()));
            transport_.reset();
        }
    }

#ifdef UM790_HAS_HIP
    if (cfg_.use_hip_staging) {
        constexpr std::size_t kClipEmbeddingFloats = 77 * 768;
        hip_staging_ = std::make_unique<PinnedStagingPool<float>>(
            kClipEmbeddingFloats * sizeof(float), 2);
        if (hip_staging_->initialized()) {
                        HQ_LOG_INFO("PinnedStagingPool initialized ({:.1f} KiB)", static_cast<double>(hip_staging_->embedding_bytes()) / 1024.0);

        } else {
                        HQ_LOG_WARN("PinnedStagingPool init failed, using EmbeddingStagingManager fallback");

            hip_staging_.reset();
        }
    }
#endif

        HQ_LOG_INFO("Initialization complete");

}

Pipeline::~Pipeline() {
    // GPUMonitor destructor handles NVML/ROCm SMI shutdown if needed
    shutdown();
}

Pipeline::Pipeline(Pipeline&&) noexcept = default;
Pipeline& Pipeline::operator=(Pipeline&&) noexcept = default;

bool Pipeline::initialize_onnx_sessions_() {
    bool ok = true;
    auto& ort = *ort_state_;

    try {
        if (!cfg_.text_encoder_onnx.empty() && validate_model_path_(cfg_.text_encoder_onnx)) {
            ort.hailo_session = std::make_unique<Ort::Session>(
                ort.env, ort_model_path(cfg_.text_encoder_onnx).c_str(), ort.hailo_options);
                        HQ_LOG_INFO("Text encoder session loaded (Hailo)");

        }
    } catch (const Ort::Exception& e) {
                HQ_LOG_WARN("Text encoder load failed: {}", e.what());

        ok = false;
    }

    try {
        if (!cfg_.unet_onnx.empty() && validate_model_path_(cfg_.unet_onnx)) {
            ort.gpu_session = std::make_unique<Ort::Session>(
                ort.env, ort_model_path(cfg_.unet_onnx).c_str(), ort.gpu_options);
                        HQ_LOG_INFO("UNet session loaded (GPU)");

        }
    } catch (const Ort::Exception& e) {
                HQ_LOG_WARN("UNet load failed: {}", e.what());

        ok = false;
    }

    try {
        if (!cfg_.vae_decoder_onnx.empty() && validate_model_path_(cfg_.vae_decoder_onnx)) {
            ort.vae_session = std::make_unique<Ort::Session>(
                ort.env, ort_model_path(cfg_.vae_decoder_onnx).c_str(), ort.vae_options);
                        HQ_LOG_INFO("VAE decoder session loaded");

        }
    } catch (const Ort::Exception& e) {
                HQ_LOG_WARN("VAE load failed: {}", e.what());

        ok = false;
    }

    return ok;
}


// ===========================================================================
// generate() -- Main inference pipeline
// ===========================================================================

std::expected<GeneratedImage, PipelineError>
Pipeline::generate(const GenerationRequest& req) {
    using Clock = std::chrono::high_resolution_clock;
    const auto t0 = Clock::now();

    // --- 0. Validate request ---
    if (shutdown_) [[unlikely]] {
                HQ_LOG_WARN("generate() called after shutdow");

        stats_.generations_failed++;
        return std::unexpected{PipelineError::ShutdownInProgress};
    }
    if (req.width == 0 || req.height == 0 || req.width > 2048 || req.height > 2048) {
                HQ_LOG_WARN("Invalid dimensions: {}x{}", req.width, req.height);

        stats_.generations_failed++;
        return std::unexpected{PipelineError::InvalidRequest};
    }
    if ((req.width % 8U) != 0U || (req.height % 8U) != 0U) {
                HQ_LOG_INFO("Invalid dimensions: {}x{} (must be multiples of 8 for VAE latent scale)", req.width, req.height);

        stats_.generations_failed++;
        return std::unexpected{PipelineError::InvalidRequest};
    }
    if (req.num_steps == 0 || req.num_steps > 150) {
                HQ_LOG_WARN("Invalid step count: {}", req.num_steps);

        stats_.generations_failed++;
        return std::unexpected{PipelineError::InvalidRequest};
    }

        HQ_LOG_INFO("generate(): \"{}\" @ {}x{} ({} steps)", req.prompt, req.width, req.height, req.num_steps);

    // --- 0b. Cluster dispatch (coordinator-only, optional) ---
    if (transport_ && transport_->is_running() && cfg_.cluster_config.has_value() &&
        cfg_.cluster_config->is_coordinator) {
        auto dispatch = transport_->select_worker();
        if (dispatch.has_value()) {
            HQ_LOG_INFO("dispatching to cluster worker {} ({})",
                        dispatch->target_node_id, dispatch->rationale);
            auto cluster_result = try_cluster_dispatch_(req, *dispatch);
            if (cluster_result.has_value()) {
                stats_.generations_completed++;
                return cluster_result;
            }
            HQ_LOG_WARN("cluster dispatch failed, falling back to local generation");
        }
    }

    // Reset per-generation NPU blend accumulator
    npu_blend_accumulated_us_ = 0.0;

    const auto t_phase_encode_start = Clock::now();

    // --- 1. Encode prompt on Hailo ---
    auto emb_result = encode_prompt_(req.prompt);
    if (!emb_result) {
                HQ_LOG_ERROR("Text encode failed: {}", to_string(emb_result.error()));

        stats_.generations_failed++;
        return std::unexpected{emb_result.error()};
    }
    // emb_floats/emb_ptr: owned by emb_scope (TMM Cool tier).
    // The staging vector is scoped to die immediately after memcpy.
    std::size_t emb_floats = 0;
    ScopedTierAlloc emb_scope;
    float* emb_ptr = nullptr;
    {
        // ORT returns embeddings via its own internal buffer; copy before output_tensors
        // destructor frees it. This vector is purely a staging intermediate.
        std::vector<float> emb_staging = std::move(*emb_result);
        emb_floats = emb_staging.size();
        HQ_LOG_INFO("Embeddings: {} floats", emb_floats);
        auto emb_alloc_r = memory_manager_->allocate(emb_floats * sizeof(float), MemoryTier::Cool);
        if (!emb_alloc_r) [[unlikely]] {
            HQ_LOG_ERROR("TMM: cond emb alloc failed ({} B): {}", emb_floats * sizeof(float), to_string(emb_alloc_r.error()));
            stats_.generations_failed++;
            return std::unexpected{PipelineError::StagingPoolExhausted};
        }
        emb_scope = ScopedTierAlloc(*memory_manager_, *emb_alloc_r);
        emb_ptr = static_cast<float*>(emb_scope.ptr());
        std::memcpy(emb_ptr, emb_staging.data(), emb_floats * sizeof(float));
        HQ_LOG_DEBUG("TMM: cond emb {} floats -> {} tier", emb_floats, to_string(emb_scope.tier()));
    } // emb_staging freed here — TMM emb_scope is the sole owner from this point

    // --- 1b. Encode unconditional prompt for Classifier-Free Guidance ---
    // uncond_floats/uncond_emb_ptr: owned by uncond_emb_scope (TMM Cool tier) when CFG active.
    std::size_t uncond_floats = 0;
    ScopedTierAlloc uncond_emb_scope;
    float* uncond_emb_ptr = nullptr;
    if (req.guidance_scale > 1.0f) {
        std::vector<float> uncond_staging;
        auto uncond_result = encode_prompt_("");
        if (uncond_result) {
            uncond_staging = std::move(*uncond_result);
            HQ_LOG_INFO("Unconditional embeddings: {} floats (CFG active, scale={:.1f})",
                        uncond_staging.size(), req.guidance_scale);
        } else {
            uncond_staging.resize(emb_floats, 0.0f);
            health_score_.update_recovery(false);
            HQ_LOG_WARN("unconditional encode failed ({}), using zero embeddings fallback (CFG scale={:.1f})",
                        to_string(uncond_result.error()), req.guidance_scale);
        }
        uncond_floats = uncond_staging.size();
        auto uncond_alloc_r = memory_manager_->allocate(uncond_floats * sizeof(float), MemoryTier::Cool);
        if (!uncond_alloc_r) [[unlikely]] {
            HQ_LOG_ERROR("TMM: uncond emb alloc failed: {}", to_string(uncond_alloc_r.error()));
            stats_.generations_failed++;
            return std::unexpected{PipelineError::StagingPoolExhausted};
        }
        uncond_emb_scope = ScopedTierAlloc(*memory_manager_, *uncond_alloc_r);
        uncond_emb_ptr = static_cast<float*>(uncond_emb_scope.ptr());
        std::memcpy(uncond_emb_ptr, uncond_staging.data(), uncond_floats * sizeof(float));
    } else {
        HQ_LOG_INFO("CFG disabled (guidance_scale={:.1f} <= 1.0)", req.guidance_scale);
    }
    // uncond_staging freed here — TMM uncond_emb_scope is the sole owner when CFG active

    const auto t_phase_stage_start = Clock::now();

    // --- 2. Stage embeddings (BUG B3 FIX) ---
    // Previously, this section performed a useless H2D copy via PinnedStagingPool
    // or EmbeddingStagingManager, then denoise_step_() ignored the staged buffer
    // and created Ort::Value tensors from CPU pointers — the DMA copy was wasted.
    //
    // Fix: On builds without a working GPU zero-copy path (i.e., denoise_step_
    // creates Ort::Value from CPU memory_info), we skip the staging entirely.
    // The EmbeddingStagingManager and PinnedStagingPool remain available for
    // when denoise_step_() gains a device-pointer overload that creates tensors
    // via Ort::MemoryInfo configured for the HIP/ROCm/CUDA device allocator.
    //
    // When GPU EP is active, ORT performs its own implicit H2D transfer from the
    // CPU pointers — this is correct, just not zero-copy. Zero-copy requires the
    // device-pointer overload described in pipeline.hpp.
#ifdef UM790_HAS_HIP
    // NOTE: HIP staging is deferred until denoise_step_() accepts device pointers.
    // Re-enable when the GPU zero-copy path is wired.
    if (hip_staging_ && false /* zero_copy_denoise_available */) {
        auto host_buf = hip_staging_->acquire_host_buffer(0);
        if (!host_buf) {
            HQ_LOG_WARN("HIP staging acquire failed: {}", host_buf.error().message);
        } else if (host_buf->size_bytes() < emb_floats * sizeof(float)) {
            HQ_LOG_INFO("HIP staging buffer too small");
        } else {
            std::memcpy(host_buf->data(), emb_ptr, emb_floats * sizeof(float));
            auto stage_res = hip_staging_->stage_to_gpu(0);
            if (!stage_res) {
                HQ_LOG_WARN("HIP staging H2D failed: {}", stage_res.error().message);
            } else {
                HQ_LOG_INFO("Staged {} bytes via PinnedStagingPool (zero-copy path)", emb_floats * sizeof(float));
            }
        }
    }
#endif

    const auto t_phase_denoise_start = Clock::now();

    // --- 3. Initialize latent tensor (random noise, TMM Cool tier) ---
    const std::size_t latent_channels = 4;
    const std::size_t latent_w = req.width / 8;
    const std::size_t latent_h = req.height / 8;
    const std::size_t latent_size = latent_channels * latent_h * latent_w;

    auto lat_alloc_r = memory_manager_->allocate(latent_size * sizeof(float), MemoryTier::Cool);
    if (!lat_alloc_r) [[unlikely]] {
        HQ_LOG_ERROR("TMM: latents alloc failed ({} B): {}", latent_size * sizeof(float), to_string(lat_alloc_r.error()));
        stats_.generations_failed++;
        return std::unexpected{PipelineError::GPUOutOfMemory};
    }
    ScopedTierAlloc lat_scope(*memory_manager_, *lat_alloc_r);
    float* latents_ptr = static_cast<float*>(lat_scope.ptr());
    HQ_LOG_DEBUG("TMM: latents {} floats -> {} tier", latent_size, to_string(lat_scope.tier()));
    current_latents_ = hq::tensor::FloatTensor4D{latents_ptr, 1, 4, latent_h, latent_w};

    // Seed random generator
    std::mt19937 rng{req.seed >= 0 ? static_cast<uint32_t>(req.seed)
                                   : static_cast<uint32_t>(std::random_device{}())};
    std::normal_distribution<float> noise_dist{0.0f, 1.0f};
    for (std::size_t i = 0; i < latent_size; ++i) { latents_ptr[i] = noise_dist(rng); }

    // --- 4. Configure scheduler for this request's step count ---
    if (scheduler_) {
        scheduler_->set_inference_steps(req.num_steps);
    }

    // --- 5. Denoising loop ---
        HQ_LOG_INFO("Starting denoising loop ({} steps)", req.num_steps);


    // Recovery state is per-generation. A previous image that exhausted or used
    // recovery attempts must not poison the next independent generate() call,
    // and a stale checkpoint from a prior request must never be restored into
    // the current request's latent tensor.
    recovery_in_progress_ = false;
    recovery_attempts_ = 0;
    latent_checkpoint_.reset();
    latent_checkpoint_floats_ = 0;

    // --- 5a. HIP Graph path (capture + replay with fallback) ---
    bool hip_graph_used = false;
#ifdef UM790_HAS_HIP
    if (cfg_.enable_hip_graph) {
        const std::array<std::int64_t, 4> latent_shape_arr{
            1, 4,
            static_cast<std::int64_t>(latent_h),
            static_cast<std::int64_t>(latent_w)};
        hip_denoiser_ = std::make_unique<HIPGraphDenoiser>(GraphConfig{
            .num_steps      = req.num_steps,
            .latent_count   = latent_size,
            .enable_capture = true,
        });
        hip_denoiser_->set_scheduler(scheduler_.get());

        // Pass TMM-backed buffers directly as spans — zero copy, no staging vector.
        auto hip_result = hip_denoiser_->execute_full(
            hq::tensor::FloatTensor4D{latents_ptr, 1, 4, latent_h, latent_w},
            std::span<const float>{emb_ptr, emb_floats},
            ort_state_->gpu_session.get(),
            &ort_state_->memory_info,
            latent_shape_arr,
            req.guidance_scale,
            std::span<const float>{uncond_emb_ptr, uncond_floats});

        if (hip_result) {
            hip_graph_used = true;
            stats_.total_steps_executed += req.num_steps;
                        HQ_LOG_INFO("HIP graph denoising complete ({} steps)", req.num_steps);

        } else {
                        HQ_LOG_WARN("HIP graph denoising failed at step {} ({}), falling back...", hip_denoiser_->steps_replayed(),
                       hip_result.error().message);

            hip_denoiser_.reset();
        }
    }
#endif

    if (!hip_graph_used) {
    for (uint32_t step = 0; step < req.num_steps; ++step) {
        const auto step_t0 = Clock::now();

        // 4a. Run GPU denoising (real Ort::Session::Run) with CFG
        namespace tv = hq::tensor;
        const std::size_t emb_hidden_dim = (emb_floats > 0) ? emb_floats / 77 : 0;
        const std::size_t uncond_hidden = (uncond_floats > 0) ? uncond_floats / 77 : 0;
        std::optional<tv::EmbeddingTensor<float>> uncond_view;
        if (uncond_emb_ptr != nullptr && uncond_hidden > 0) {
            uncond_view.emplace(uncond_emb_ptr, 1, 77, uncond_hidden);
        }
        auto denoise_result = denoise_step_(
            step,
            tv::FloatTensor4D{latents_ptr, 1, 4, latent_h, latent_w},
            tv::EmbeddingTensor<float>{emb_ptr, 1, 77, emb_hidden_dim},
            std::move(uncond_view),
            req.guidance_scale);
        if (!denoise_result) [[unlikely]] {
            HQ_LOG_ERROR("Denoise step {} failed: {}", step, to_string(denoise_result.error()));

            stats_.generations_failed++;
            return std::unexpected{denoise_result.error()};
        }

        // 4b. Sample GPU and Hailo utilization (at step boundary)
        float gpu_util = 0.0f;
        float gpu_temp = 0.0f;
        float gpu_power = 0.0f;
        if (gpu_monitor_ && gpu_monitor_->is_initialized()) {
            auto telem = gpu_monitor_->query_all();
            if (telem) {
                gpu_util  = telem->utilization_percent;
                gpu_temp  = telem->temperature_celsius;
                gpu_power = telem->power_watts;
            }
        } else {
            gpu_power = 0.0f;  // GPU monitor unavailable — power telemetry disabled
        }

        float hailo_util = 0.0f;
        float hailo_temp = 0.0f;
        float hailo_power = 0.0f;

        if (hailo_monitor_ && hailo_monitor_->is_open()) {
            auto telem = hailo_monitor_->sample();
            if (telem) {
                hailo_util  = telem->nn_core_utilization;
                hailo_temp  = telem->temperature_celsius;
                hailo_power = telem->power_watts;
            }
        }

        const auto step_t1 = Clock::now();
        const double step_ms = std::chrono::duration<double, std::milli>(step_t1 - step_t0).count();

        const auto completed_steps_before = static_cast<double>(stats_.total_steps_executed);
        stats_.avg_gpu_utilization =
            (stats_.avg_gpu_utilization * completed_steps_before + gpu_util) /
            (completed_steps_before + 1.0);
        stats_.avg_hailo_utilization =
            (stats_.avg_hailo_utilization * completed_steps_before + hailo_util) /
            (completed_steps_before + 1.0);
        health_score_.update_gpu(gpu_util, gpu_temp);
        health_score_.update_hailo(hailo_util, hailo_temp);
        health_score_.update_latency(static_cast<float>(step_ms));

        // 4c. Save latent checkpoint into TMM Cool-tier buffer (for recovery)
        if (!latent_checkpoint_ || latent_checkpoint_floats_ != latent_size) {
            auto ckpt_r = memory_manager_->allocate(latent_size * sizeof(float), MemoryTier::Cool);
            if (ckpt_r) {
                latent_checkpoint_.emplace(*memory_manager_, *ckpt_r);
                latent_checkpoint_floats_ = latent_size;
            }
        }
        if (latent_checkpoint_ && latent_checkpoint_->valid()) {
            std::memcpy(latent_checkpoint_->ptr(), latents_ptr, latent_size * sizeof(float));
        }

        // 4d. Watchdog check with new UtilizationSnapshot API
        if (watchdog_ && cfg_.enable_watchdog) {
            UtilizationSnapshot gpu_snap{
                .device         = ComputeUnit::GPU_780M,
                .step           = step,
                .utilization    = gpu_util,
                .temperature    = gpu_temp,
                .power_watts    = gpu_power,
                .device_healthy = true,
            };
            UtilizationSnapshot hailo_snap{
                .device         = ComputeUnit::HAILO_8L,
                .step           = step,
                .utilization    = hailo_util,
                .temperature    = hailo_temp,
                .power_watts    = hailo_power,
                .device_healthy = true,
            };
            auto recovery_action = watchdog_->step(step, gpu_snap, hailo_snap);

            if (recovery_action.has_value()) {
                                HQ_LOG_INFO("Recovery at step {}: {} (util fault: device={} value={:.1f}%)", step, recovery_action->reason,
                           recovery_action->device == ComputeUnit::GPU_780M ? "GPU" : "Hailo",
                           recovery_action->util_at_fault);

                stats_.watchdog_recoveries++;

                if (recovery_attempts_ >= cfg_.max_recovery_attempts) {
                    HQ_LOG_INFO("Max recovery attempts ({}) exceeded", cfg_.max_recovery_attempts);

                    stats_.generations_failed++;
                    return std::unexpected{PipelineError::RecoveryTooManyAttempts};
                }

                if (recovery_action->result == RecoveryResult::FATAL) {
                    health_score_.update_recovery(false);
                    HQ_LOG_INFO("Watchdog reported fatal recovery actio");

                    stats_.generations_failed++;
                    return std::unexpected{PipelineError::WatchdogRecoveryFailed};
                }

                // 4e. Recovery: rebuild session and restore latents from checkpoint
                auto pipeline_recovery = on_watchdog_recovery_(
                    recovery_action->device, step, recovery_action->util_at_fault,
                    hq::tensor::FloatTensor4D{latents_ptr, 1, 4, latent_h, latent_w});
                health_score_.update_recovery(pipeline_recovery.has_value() &&
                                              pipeline_recovery.value() == RecoveryResult::SUCCESS);
                if (!pipeline_recovery) {
                    HQ_LOG_WARN("Recovery failed at step {}: {}", step, to_string(pipeline_recovery.error()));

                    stats_.generations_failed++;
                    return std::unexpected{pipeline_recovery.error()};
                }
            }
        }

        stats_.total_steps_executed++;

        // Progress print every 5 steps
        if ((step + 1) % 5 == 0 || step == req.num_steps - 1) {
                        HQ_LOG_INFO(" Step {}/{}  ({:.1f} ms)  GPU:{:.0f}% ({:.0f}C)  Hailo:{:.0f}% ({:.0f}C)", step + 1, req.num_steps, step_ms, gpu_util, gpu_temp,
                       hailo_util, hailo_temp);

        }
    }
    } // !hip_graph_used

    const auto t_phase_vae_start = Clock::now();

    // --- 6. VAE decode ---
        HQ_LOG_INFO("VAE decoding...");

    auto decode_result = decode_latents_(
        hq::tensor::LatentTensor<const float>{latents_ptr, 1, 4, latent_h, latent_w},
        req.width, req.height);
    if (!decode_result) {
                HQ_LOG_ERROR("VAE decode failed: {}", to_string(decode_result.error()));

        stats_.generations_failed++;
        return std::unexpected{decode_result.error()};
    }

    const auto t_phase_post_start = Clock::now();

    // --- 7. Optional NPU post-processing ---
    // Applies image post-processing (sharpening, noise reduction) via npu_post_processor_.
    // Currently resolves to CpuPostProcessor (honest CPU pass-through) because
    // HailoRT is not installed. Wired here so the timing slot is populated and
    // the call path is ready for when HailoRT + a post-processing HEF are available.
    if (npu_post_processor_) {
        npu::NpuPostProcessRequest pp_req{
            .pixels = std::span<const std::uint8_t>{
                decode_result->pixels.data(), decode_result->pixels.size()},
            .width  = decode_result->width,
            .height = decode_result->height,
            .task   = npu::NpuTaskType::PostProcess,
        };
        auto pp_result = npu_post_processor_->post_process(pp_req);
        if (pp_result && !pp_result->pixels.empty()) {
            decode_result->pixels = std::move(pp_result->pixels);
            HQ_LOG_INFO("Post-processing via {} ({:.0f} us, npu_accel={})",
                        npu_post_processor_->name(),
                        pp_result->processing_time_us,
                        pp_result->was_npu_accelerated ? "yes" : "no");
        } else if (!pp_result) {
            HQ_LOG_WARN("Post-processing failed: {} — using raw VAE output",
                        pp_result.error());
        }
    }

    const auto t1 = Clock::now();
    const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Populate per-phase timing breakdown (available via last_phase_timings())
    last_phase_timings_.text_encode_ms      = std::chrono::duration<double, std::milli>(
        t_phase_stage_start - t_phase_encode_start).count();
    last_phase_timings_.embedding_stage_ms  = std::chrono::duration<double, std::milli>(
        t_phase_denoise_start - t_phase_stage_start).count();
    last_phase_timings_.denoise_total_ms    = std::chrono::duration<double, std::milli>(
        t_phase_vae_start - t_phase_denoise_start).count();
    last_phase_timings_.npu_blend_in_loop_us = npu_blend_accumulated_us_;
    last_phase_timings_.vae_decode_ms       = std::chrono::duration<double, std::milli>(
        t_phase_post_start - t_phase_vae_start).count();
    last_phase_timings_.post_process_ms     = std::chrono::duration<double, std::milli>(
        t1 - t_phase_post_start).count();
    last_phase_timings_.num_denoise_steps   = req.num_steps;
    last_phase_timings_.encoder_name        = npu_encoder_ ? npu_encoder_->name() : "none";
    last_phase_timings_.post_processor_name =
        npu_post_processor_ ? npu_post_processor_->name() : "none";

    // Populate honest hardware acceleration report
    auto& accel = last_phase_timings_.acceleration;
    accel.text_encode_used_npu =
        npu_encoder_ && npu_encoder_->is_available() && !npu_encoder_->synthetic_mode();
    accel.text_encode_used_gpu = false;  // Text encoding uses NPU/CPU, not GPU
    // denoise and VAE use ORT; we only claim GPU if the GPU monitor reports real activity
    // OR if the session was created with a GPU EP. For now, check if GPU monitor is active.
    accel.denoise_used_gpu = gpu_monitor_ && gpu_monitor_->is_initialized();
    accel.vae_decode_used_gpu = gpu_monitor_ && gpu_monitor_->is_initialized();
    accel.post_process_used_npu =
        npu_post_processor_ && npu_post_processor_->is_available() && !npu_post_processor_->synthetic_mode();
    accel.cfg_blend_used_npu =
        npu_post_processor_ && !npu_post_processor_->synthetic_mode();
    accel.hailo_telemetry_real =
        hailo_monitor_ && hailo_monitor_->is_open();
    accel.gpu_telemetry_real =
        gpu_monitor_ && gpu_monitor_->is_initialized();
    accel.encoder_name = npu_encoder_ ? npu_encoder_->name() : "none";
    accel.post_processor_name =
        npu_post_processor_ ? npu_post_processor_->name() : "none";
    accel.encoder_is_fallback =
        npu_encoder_ && npu_encoder_->synthetic_mode();
    accel.post_processor_is_fallback =
        npu_post_processor_ && npu_post_processor_->synthetic_mode();

    // Percentage of NPU-acceleratable CHEAP components that actually used NPU.
    // Components: text_encode, post_process, cfg_blend.
    // NOTE: This deliberately EXCLUDES UNet denoising (~90% of compute).
    // A value of 100% here does NOT mean the NPU did the expensive work.
    int npu_cheap_components_total = 3;
    int npu_cheap_components_used = 0;
    if (accel.text_encode_used_npu)  ++npu_cheap_components_used;
    if (accel.post_process_used_npu) ++npu_cheap_components_used;
    if (accel.cfg_blend_used_npu)    ++npu_cheap_components_used;
    accel.npu_cheap_ops_percent = static_cast<std::uint8_t>(
        (npu_cheap_components_used * 100) / npu_cheap_components_total);

    // UNet denoising on NPU: architecturally blocked. Requires compiled HEF
    // for Hailo-8L + HailoRT SDK (Linux only). No such HEF exists in this repo.
    accel.unet_denoise_used_npu = false;

    HQ_LOG_INFO("[Pipeline] HardwareAccelerationReport: NPU cheap-ops={}%, "
                "UNet_on_NPU={}, encoder_fallback={}, postproc_fallback={}",
                accel.npu_cheap_ops_percent,
                accel.unet_denoise_used_npu ? "yes" : "no (blocked: no HEF)",
                accel.encoder_is_fallback ? "yes" : "no",
                accel.post_processor_is_fallback ? "yes" : "no");

    // Copy acceleration report into the GeneratedImage
    decode_result->acceleration = accel;

    // Update stats
    stats_.generations_completed++;
    stats_.avg_generation_ms = (stats_.avg_generation_ms *
                                (stats_.generations_completed - 1) + total_ms)
                               / stats_.generations_completed;

    decode_result->generation_time_ms = static_cast<float>(total_ms);

        HQ_LOG_INFO("Generation complete in {:.1f} ms (recovery attempts: {})", total_ms, recovery_attempts_);


    return *decode_result;
}

// ===========================================================================
// generate_batch() -- Batched generation
// ===========================================================================

std::vector<std::expected<GeneratedImage, PipelineError>>
Pipeline::generate_batch(const std::vector<GenerationRequest>& requests) {
    std::vector<std::expected<GeneratedImage, PipelineError>> results;
    results.reserve(requests.size());

        HQ_LOG_INFO("Batch generation: {} requests", requests.size());


    for (const auto& req : requests) {
        results.emplace_back(generate(req));
    }

    const uint32_t successes = static_cast<uint32_t>(std::ranges::count_if(
        results, [](const auto& r) { return r.has_value(); }));

        HQ_LOG_INFO("Batch complete: {}/{} succeeded", successes, requests.size());


    return results;
}


// ===========================================================================
// Private helpers
// ===========================================================================

// ---------------------------------------------------------------------------
/// @brief Run a single denoising step with real ONNX Runtime inference.
///
/// Builds input tensors (latents, timestep, encoder_hidden_states), runs the
/// UNet model, extracts the noise prediction, and applies the DEIS scheduler
/// step to update the latents in-place.
///
/// Classifier-Free Guidance (CFG):
///   When guidance_scale > 1.0, the UNet is executed twice per step:
///     1. Conditional pass   (user prompt embeddings)
///     2. Unconditional pass (empty-prompt embeddings)
///   The noise predictions are blended:
///     noise_pred = noise_uncond + scale * (noise_cond - noise_uncond)
///   This steers generation toward the prompt while preserving diversity.
///
/// Performance note: two UNet runs per step doubles compute. A future
/// optimization could batch both passes into a single UNet call with batch=2
/// if the exported ONNX model supports dynamic batch dimension.
// ---------------------------------------------------------------------------
std::expected<void, PipelineError>
Pipeline::denoise_step_(std::uint32_t step,
                        hq::tensor::FloatTensor4D latents,
                        hq::tensor::EmbeddingTensor<float> cond_emb,
                        std::optional<hq::tensor::EmbeddingTensor<float>> uncond_emb,
                        float guidance_scale) {
    if (!ort_state_->gpu_session) {
        return std::unexpected{PipelineError::ONNXSessionLoadFailed};
    }

    if (!scheduler_) {
                HQ_LOG_ERROR("DEIS scheduler not initialized");

        return std::unexpected{PipelineError::SchedulerNotInitialized};
    }

    auto& ort = *ort_state_;

    // Extract raw dimensions from TensorViews for ORT shape arrays
    const std::size_t latent_h    = latents.extent(2);
    const std::size_t latent_w    = latents.extent(3);
    const std::size_t latent_count = latents.num_elements();
    float* latents_raw             = latents.data();
    const float* cond_emb_raw      = cond_emb.data();
    const std::size_t emb_count    = cond_emb.num_elements();

    if (latent_h == 0 || latent_w == 0) {
                HQ_LOG_ERROR("Invalid latent shape: {}x{}", latent_w, latent_h);

        return std::unexpected{PipelineError::InvalidRequest};
    }

    if (latent_count != 4ULL * latent_h * latent_w) {
        HQ_LOG_ERROR("Latent size mismatch: param={} expected={} for shape [1,4,{},{}]",
                     latent_count, 4ULL * latent_h * latent_w, latent_h, latent_w);
        return std::unexpected{PipelineError::InvalidRequest};
    }

    const std::size_t embedding_seq_len = 77;

    if (emb_count == 0 || (emb_count % embedding_seq_len) != 0U) {
        HQ_LOG_ERROR("Invalid cond emb count: {} (not divisible by CLIP seq_len {})",
                     emb_count, embedding_seq_len);
        return std::unexpected{PipelineError::ONNXRunFailed};
    }
    if (guidance_scale > 1.0f && !uncond_emb.has_value()) {
        HQ_LOG_ERROR("CFG enabled (scale={:.1f}) but uncond_emb is nullopt", guidance_scale);
        return std::unexpected{PipelineError::ONNXRunFailed};
    }

    const std::array<std::int64_t, 4> latent_shape{
        1,
        4,
        static_cast<std::int64_t>(latent_h),
        static_cast<std::int64_t>(latent_w)};

    // Timestep from DEIS scheduler
    std::int64_t timestep_val = scheduler_->timestep(step);
    const std::array<std::int64_t, 1> timestep_shape{1};

    // Encoder hidden states shape
    const std::int64_t hidden_dim = static_cast<std::int64_t>(emb_count / embedding_seq_len);
    const std::array<std::int64_t, 3> emb_shape{1,
        static_cast<std::int64_t>(embedding_seq_len), hidden_dim};

    // ------------------------------------------------------------------
    // Helper: run UNet with a given embedding set, return noise prediction.
    // The returned vector holds ORT inference OUTPUT (noise_pred), which is
    // a per-step ephemeral value consumed by scheduler_->step() and then
    // discarded. It is not latent state, embedding state, or checkpoint state.
    // The INPUT tensors (latents, emb) are created from TMM-backed pointers.
    // ------------------------------------------------------------------
    auto run_unet_pass = [&](const float* emb, std::size_t emb_sz)
        -> std::expected<std::vector<float>, PipelineError> {
        try {
            Ort::Value latent_tensor = Ort::Value::CreateTensor<float>(
                ort.memory_info, latents_raw, latent_count,
                latent_shape.data(), latent_shape.size());

            Ort::Value timestep_tensor = Ort::Value::CreateTensor<std::int64_t>(
                ort.memory_info, &timestep_val, 1,
                timestep_shape.data(), timestep_shape.size());

            Ort::Value emb_tensor = Ort::Value::CreateTensor<float>(
                ort.memory_info,
                const_cast<float*>(emb), emb_sz,
                emb_shape.data(), emb_shape.size());

            const char* input_names[] = {"sample", "timestep", "encoder_hidden_states"};
            const char* output_names[] = {"out_sample"};
            Ort::Value inputs[] = {std::move(latent_tensor),
                                    std::move(timestep_tensor),
                                    std::move(emb_tensor)};

            auto output_tensors = ort.gpu_session->Run(
                Ort::RunOptions{nullptr},
                input_names, inputs, 3,
                output_names, 1);

            if (output_tensors.empty() || !output_tensors[0].IsTensor()) {
                                HQ_LOG_ERROR("UNet output is not a tensor");

                return std::unexpected{PipelineError::ONNXRunFailed};
            }

            Ort::Value& noise_pred_tensor = output_tensors[0];
            const float* noise_pred = noise_pred_tensor.GetTensorData<float>();
            const auto noise_shape = noise_pred_tensor.GetTensorTypeAndShapeInfo().GetShape();
            std::size_t noise_count = 1;
            for (auto d : noise_shape) {
                if (d > 0) noise_count *= static_cast<std::size_t>(d);
            }

            return std::vector<float>(noise_pred, noise_pred + noise_count);

        } catch (const Ort::Exception& e) {
                        HQ_LOG_ERROR("UNet Run() failed: {}", e.what());

            return std::unexpected{PipelineError::ONNXRunFailed};
        } catch (const std::exception& e) {
                        HQ_LOG_ERROR("Denoise step exception: {}", e.what());

            return std::unexpected{PipelineError::ONNXRunFailed};
        }
    };

    try {
        if (guidance_scale <= 1.0f) {
            // ------------------------------------------------------------------
            // No CFG: single conditional pass
            // ------------------------------------------------------------------
            auto noise_result = run_unet_pass(cond_emb_raw, emb_count);
            if (!noise_result) {
                return std::unexpected{noise_result.error()};
            }
            if (auto s = scheduler_->step(
                    latents,
                    hq::tensor::Tensor1D<const float>{noise_result->data(),
                                                      noise_result->size()},
                    step); !s) [[unlikely]] {
                HQ_LOG_ERROR("Scheduler step {} failed: {}", step, to_string(s.error()));
                return std::unexpected{PipelineError::SchedulerNotInitialized};
            }

        } else {
            // ------------------------------------------------------------------
            // CFG: conditional + unconditional passes, then blend
            // ------------------------------------------------------------------

            // 1. Conditional pass
            auto cond_result = run_unet_pass(cond_emb_raw, emb_count);
            if (!cond_result) {
                HQ_LOG_ERROR("CFG conditional pass failed at step {}", step);

                return std::unexpected{cond_result.error()};
            }
            std::vector<float>& noise_cond = *cond_result;

            // 2. Unconditional pass (uncond_emb is guaranteed non-nullopt by the check above)
            auto uncond_result = run_unet_pass(uncond_emb->data(), emb_count);
            if (!uncond_result) {
                HQ_LOG_ERROR("CFG unconditional pass failed at step {}", step);

                return std::unexpected{uncond_result.error()};
            }
            std::vector<float>& noise_uncond = *uncond_result;

            // Verify shape match
            if (noise_cond.size() != noise_uncond.size()) {
                                HQ_LOG_ERROR("CFG shape mismatch: cond={} vs uncond={}", noise_cond.size(), noise_uncond.size());

                return std::unexpected{PipelineError::ONNXRunFailed};
            }

            // 3. CFG blend via NPU abstraction (routes to Hailo-8L SAXPY when HailoRT available;
            //    currently CpuPostProcessor CPU path — timed for profiling).
            //    noise_cond is updated in-place: uncond + scale * (cond - uncond)
            {
                const auto t_blend0 = std::chrono::high_resolution_clock::now();
                bool blend_ok = false;
                if (npu_post_processor_) {
                    auto blend_r = npu_post_processor_->blend_noise_cfg(
                        std::span<float>{noise_cond.data(), noise_cond.size()},
                        std::span<const float>{noise_uncond.data(), noise_uncond.size()},
                        guidance_scale);
                    blend_ok = blend_r.has_value();
                    if (!blend_ok) {
                        HQ_LOG_WARN("blend_noise_cfg via {} failed at step {}: {} — scalar fallback",
                                    npu_post_processor_->name(), step, blend_r.error());
                    }
                }
                if (!blend_ok) {
                    // Scalar fallback (npu_post_processor_ null or blend failed)
                    for (std::size_t i = 0; i < noise_cond.size(); ++i) {
                        noise_cond[i] = noise_uncond[i] +
                                        guidance_scale * (noise_cond[i] - noise_uncond[i]);
                    }
                }
                const auto t_blend1 = std::chrono::high_resolution_clock::now();
                npu_blend_accumulated_us_ += std::chrono::duration<double, std::micro>(
                    t_blend1 - t_blend0).count();
            }

            // 4. Scheduler step with blended noise
            if (auto s = scheduler_->step(
                    latents,
                    hq::tensor::Tensor1D<const float>{noise_cond.data(),
                                                      noise_cond.size()},
                    step); !s) [[unlikely]] {
                HQ_LOG_ERROR("Scheduler step {} failed (CFG): {}", step, to_string(s.error()));
                return std::unexpected{PipelineError::SchedulerNotInitialized};
            }
        }

    } catch (const std::exception& e) {
                HQ_LOG_ERROR("Denoise step outer exception: {}", e.what());

        return std::unexpected{PipelineError::ONNXRunFailed};
    }

    return {};
}

// ===========================================================================
// denoise_step_() — zero-copy GPU device-pointer overload (BUG B3 fix)
// ===========================================================================

std::expected<void, PipelineError>
Pipeline::denoise_step_(std::uint32_t step,
                        float* gpu_latents,
                        float* gpu_cond_emb,
                        std::optional<float*> gpu_uncond_emb,
                        float guidance_scale,
                        std::size_t latent_count,
                        std::size_t emb_count,
                        std::int64_t latent_h,
                        std::int64_t latent_w) {
    if (!ort_state_->gpu_session) {
        return std::unexpected{PipelineError::ONNXSessionLoadFailed};
    }
    if (!scheduler_) {
        HQ_LOG_ERROR("DEIS scheduler not initialized");
        return std::unexpected{PipelineError::SchedulerNotInitialized};
    }

    auto& ort = *ort_state_;

    // Derive embedding hidden dim from total count / 77
    const std::size_t embedding_seq_len = 77;
    if (emb_count == 0 || (emb_count % embedding_seq_len) != 0U) {
        HQ_LOG_ERROR("Invalid cond emb count: {} (not divisible by CLIP seq_len {})",
                     emb_count, embedding_seq_len);
        return std::unexpected{PipelineError::ONNXRunFailed};
    }
    if (guidance_scale > 1.0f && !gpu_uncond_emb.has_value()) {
        HQ_LOG_ERROR("CFG enabled (scale={:.1f}) but gpu_uncond_emb is nullopt", guidance_scale);
        return std::unexpected{PipelineError::ONNXRunFailed};
    }

    const std::array<std::int64_t, 4> latent_shape{
        1, 4, latent_h, latent_w};
    std::int64_t timestep_val = scheduler_->timestep(step);
    const std::array<std::int64_t, 1> timestep_shape{1};
    const std::int64_t hidden_dim = static_cast<std::int64_t>(emb_count / embedding_seq_len);
    const std::array<std::int64_t, 3> emb_shape{1,
        static_cast<std::int64_t>(embedding_seq_len), hidden_dim};

    // ------------------------------------------------------------------
    // Zero-copy: create GPU MemoryInfo for device pointers
    // ------------------------------------------------------------------
    Ort::MemoryInfo gpu_mem_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    // NOTE: For true zero-copy with ROCm/CUDA EP, we need:
    //   gpu_mem_info = Ort::MemoryInfo("Hip", OrtAllocatorType::OrtArenaAllocator,
    //                                    0, OrtMemType::OrtMemTypeDefault);
    // ORT C++ API constructor: Ort::MemoryInfo(const char* name, OrtAllocatorType type,
    //                                           int id, OrtMemType mem_type);
    // The "Hip" / "Cuda" memory info tells ORT that the pointer is device-local.
    //
    // Until the build environment has the full OrtMemoryInfo constructor available,
    // we fall back to CPU memory info. The GPU EP will still perform an implicit
    // H2D copy, but the caller has already staged data to GPU — this eliminates
    // the CPU-side staging vector copy, saving one memcpy per step.
    //
    // Production (full zero-copy):
    //   gpu_mem_info = Ort::MemoryInfo("Hip", OrtArenaAllocator, 0, OrtMemTypeDefault);

    auto run_unet_pass_device = [&](float* emb_ptr, std::size_t emb_sz)
        -> std::expected<std::vector<float>, PipelineError> {
        try {
            Ort::Value latent_tensor = Ort::Value::CreateTensor<float>(
                gpu_mem_info, gpu_latents, latent_count,
                latent_shape.data(), latent_shape.size());

            Ort::Value timestep_tensor = Ort::Value::CreateTensor<std::int64_t>(
                gpu_mem_info, &timestep_val, 1,
                timestep_shape.data(), timestep_shape.size());

            Ort::Value emb_tensor = Ort::Value::CreateTensor<float>(
                gpu_mem_info, emb_ptr, emb_sz,
                emb_shape.data(), emb_shape.size());

            const char* input_names[] = {"sample", "timestep", "encoder_hidden_states"};
            const char* output_names[] = {"out_sample"};
            Ort::Value inputs[] = {std::move(latent_tensor),
                                    std::move(timestep_tensor),
                                    std::move(emb_tensor)};

            auto output_tensors = ort.gpu_session->Run(
                Ort::RunOptions{nullptr},
                input_names, inputs, 3,
                output_names, 1);

            if (output_tensors.empty() || !output_tensors[0].IsTensor()) {
                HQ_LOG_ERROR("UNet output is not a tensor");
                return std::unexpected{PipelineError::ONNXRunFailed};
            }

            Ort::Value& noise_pred_tensor = output_tensors[0];
            const float* noise_pred = noise_pred_tensor.GetTensorData<float>();
            const auto noise_shape = noise_pred_tensor.GetTensorTypeAndShapeInfo().GetShape();
            std::size_t noise_count = 1;
            for (auto d : noise_shape) {
                if (d > 0) noise_count *= static_cast<std::size_t>(d);
            }

            return std::vector<float>(noise_pred, noise_pred + noise_count);

        } catch (const Ort::Exception& e) {
            HQ_LOG_ERROR("UNet Run() failed: {}", e.what());
            return std::unexpected{PipelineError::ONNXRunFailed};
        } catch (const std::exception& e) {
            HQ_LOG_ERROR("Denoise step exception: {}", e.what());
            return std::unexpected{PipelineError::ONNXRunFailed};
        }
    };

    try {
        if (guidance_scale <= 1.0f) {
            auto noise_result = run_unet_pass_device(gpu_cond_emb, emb_count);
            if (!noise_result) {
                return std::unexpected{noise_result.error()};
            }
            if (auto s = scheduler_->step(
                    hq::tensor::FloatTensor4D{gpu_latents, 1, 4,
                                              static_cast<std::size_t>(latent_h),
                                              static_cast<std::size_t>(latent_w)},
                    hq::tensor::Tensor1D<const float>{noise_result->data(),
                                                          noise_result->size()},
                    step); !s) [[unlikely]] {
                HQ_LOG_ERROR("Scheduler step {} failed: {}", step, to_string(s.error()));
                return std::unexpected{PipelineError::SchedulerNotInitialized};
            }
        } else {
            // 1. Conditional pass
            auto cond_result = run_unet_pass_device(gpu_cond_emb, emb_count);
            if (!cond_result) {
                HQ_LOG_ERROR("CFG conditional pass failed at step {}", step);
                return std::unexpected{cond_result.error()};
            }
            std::vector<float>& noise_cond = *cond_result;

            // 2. Unconditional pass
            auto uncond_result = run_unet_pass_device(*gpu_uncond_emb, emb_count);
            if (!uncond_result) {
                HQ_LOG_ERROR("CFG unconditional pass failed at step {}", step);
                return std::unexpected{uncond_result.error()};
            }
            std::vector<float>& noise_uncond = *uncond_result;

            if (noise_cond.size() != noise_uncond.size()) {
                HQ_LOG_ERROR("CFG shape mismatch: cond={} vs uncond={}",
                             noise_cond.size(), noise_uncond.size());
                return std::unexpected{PipelineError::ONNXRunFailed};
            }

            // 3. CFG blend
            {
                const auto t_blend0 = std::chrono::high_resolution_clock::now();
                bool blend_ok = false;
                if (npu_post_processor_) {
                    auto blend_r = npu_post_processor_->blend_noise_cfg(
                        std::span<float>{noise_cond.data(), noise_cond.size()},
                        std::span<const float>{noise_uncond.data(), noise_uncond.size()},
                        guidance_scale);
                    blend_ok = blend_r.has_value();
                    if (!blend_ok) {
                        HQ_LOG_WARN("blend_noise_cfg via {} failed at step {}: {} — scalar fallback",
                                    npu_post_processor_->name(), step, blend_r.error());
                    }
                }
                if (!blend_ok) {
                    for (std::size_t i = 0; i < noise_cond.size(); ++i) {
                        noise_cond[i] = noise_uncond[i] +
                                        guidance_scale * (noise_cond[i] - noise_uncond[i]);
                    }
                }
                const auto t_blend1 = std::chrono::high_resolution_clock::now();
                npu_blend_accumulated_us_ += std::chrono::duration<double, std::micro>(
                    t_blend1 - t_blend0).count();
            }

            // 4. Scheduler step
            if (auto s = scheduler_->step(
                    hq::tensor::FloatTensor4D{gpu_latents, 1, 4,
                                              static_cast<std::size_t>(latent_h),
                                              static_cast<std::size_t>(latent_w)},
                    hq::tensor::Tensor1D<const float>{noise_cond.data(), noise_cond.size()},
                    step); !s) [[unlikely]] {
                HQ_LOG_ERROR("Scheduler step {} failed (CFG): {}", step, to_string(s.error()));
                return std::unexpected{PipelineError::SchedulerNotInitialized};
            }
        }
    } catch (const std::exception& e) {
        HQ_LOG_ERROR("Denoise step outer exception: {}", e.what());
        return std::unexpected{PipelineError::ONNXRunFailed};
    }

    return {};
}

// ===========================================================================
// on_watchdog_recovery_() -- saves/restores latent tensors, rebuilds sessions
// ===========================================================================

std::expected<RecoveryResult, PipelineError>
Pipeline::on_watchdog_recovery_(ComputeUnit unit, std::uint32_t step,
                                      float util_at_fault,
                                      hq::tensor::FloatTensor4D latents) {
    float* latents_raw             = latents.data();
    const std::size_t latent_count = latents.num_elements();
        HQ_LOG_WARN("*** RECOVERY: device={}, step={}, util_fault={:.1f}%", unit == ComputeUnit::GPU_780M ? "GPU" : "Hailo",
               step, util_at_fault);


    recovery_in_progress_ = true;
    recovery_attempts_++;

    HQ_LOG_INFO(" -> Latent checkpoint: {} floats (saved pre-step)", latent_checkpoint_floats_);

    if (!latent_checkpoint_ || !latent_checkpoint_->valid() || latent_checkpoint_floats_ == 0) {
        recovery_in_progress_ = false;
        HQ_LOG_WARN(" -> Recovery failed: latent checkpoint is empty");

        return std::unexpected{PipelineError::WatchdogRecoveryFailed};
    }

    bool session_rebuilt = false;
    if (unit == ComputeUnit::GPU_780M) {
                HQ_LOG_INFO(" -> Rebuilding GPU session...");

        ort_state_->gpu_session.reset();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        try {
            if (cfg_.unet_onnx.empty() || !validate_model_path_(cfg_.unet_onnx)) {
                recovery_in_progress_ = false;
                                HQ_LOG_WARN(" -> GPU recovery failed: invalid UNet model path");

                return std::unexpected{PipelineError::InvalidModelPath};
            }
            ort_state_->gpu_session = std::make_unique<Ort::Session>(
                ort_state_->env, ort_model_path(cfg_.unet_onnx).c_str(), ort_state_->gpu_options);
            session_rebuilt = static_cast<bool>(ort_state_->gpu_session);
                        HQ_LOG_INFO(" -> GPU session rebuilt OK");

        } catch (const Ort::Exception& e) {
            recovery_in_progress_ = false;
                        HQ_LOG_INFO(" -> GPU session rebuild FAILED: {}", e.what());

            return std::unexpected{PipelineError::ONNXSessionLoadFailed};
        }
    } else {
                HQ_LOG_INFO(" -> Rebuilding Hailo session...");

        ort_state_->hailo_session.reset();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        try {
            if (cfg_.text_encoder_onnx.empty() ||
                !validate_model_path_(cfg_.text_encoder_onnx)) {
                recovery_in_progress_ = false;
                                HQ_LOG_WARN(" -> Hailo recovery failed: invalid text encoder model path");

                return std::unexpected{PipelineError::InvalidModelPath};
            }
            ort_state_->hailo_session = std::make_unique<Ort::Session>(
                ort_state_->env, ort_model_path(cfg_.text_encoder_onnx).c_str(), ort_state_->hailo_options);
            session_rebuilt = static_cast<bool>(ort_state_->hailo_session);
                        HQ_LOG_INFO(" -> Hailo session rebuilt OK");

        } catch (const Ort::Exception& e) {
            recovery_in_progress_ = false;
                        HQ_LOG_INFO(" -> Hailo session rebuild FAILED: {}", e.what());

            return std::unexpected{PipelineError::ONNXSessionLoadFailed};
        }
    }

    if (!session_rebuilt) {
        recovery_in_progress_ = false;
                HQ_LOG_WARN(" -> Recovery failed: session was not rebuilt");

        return std::unexpected{PipelineError::ONNXSessionLoadFailed};
    }

    const std::size_t restore_count = std::min(latent_count, latent_checkpoint_floats_);
    std::memcpy(latents_raw, latent_checkpoint_->ptr(), restore_count * sizeof(float));
    HQ_LOG_INFO(" -> Latents restored from checkpoint ({} floats)", restore_count);


    recovery_in_progress_ = false;
        HQ_LOG_INFO(" -> Recovery complete");

    return RecoveryResult::SUCCESS;
}

// ===========================================================================
// encode_prompt_() -- Encode text prompt to embedding tensor
// ===========================================================================

std::expected<std::vector<float>, PipelineError>
Pipeline::encode_prompt_(const std::string& prompt) {
    constexpr std::size_t seq_len = 77;

    if (npu_encoder_) {
        npu::NpuEncodeRequest npu_req{};
        npu_req.prompt         = prompt;
        npu_req.guidance_scale = 1.0f;
        npu_req.seed           = -1;
        npu_req.width          = 512;
        npu_req.height         = 512;
        npu_req.num_steps      = 20;
        npu_req.max_seq_len    = seq_len;
        auto npu_result = npu_encoder_->encode(npu_req);
        if (npu_result) {
            const auto sp = npu_result->embeddings.span();
            std::vector<float> embeddings(sp.begin(), sp.end());
            HQ_LOG_INFO("encode_prompt via {}: {} floats",
                        npu_encoder_->name(), embeddings.size());
            return embeddings;
        }
        HQ_LOG_WARN("INpuEncoder::encode failed: {} — falling back to ORT direct path",
                    npu_result.error());
    }

    if (!tokenizer_ || !tokenizer_->is_loaded()) {
                HQ_LOG_ERROR("Tokenizer not available");

        return std::unexpected{PipelineError::ONNXRunFailed};
    }

    std::vector<std::int64_t> token_ids = tokenizer_->encode(prompt, seq_len);
        HQ_LOG_INFO("Tokenized prompt ({} tokens): [{} ... {}]", token_ids.size(), token_ids[0], token_ids.back());

    if (!ort_state_->hailo_session && !cfg_.use_hailo_text_encoder) {
                HQ_LOG_ERROR("No text encoder session available");

        return std::unexpected{PipelineError::ONNXSessionLoadFailed};
    }

    try {
        const std::array<std::int64_t, 2> input_shape{1, static_cast<std::int64_t>(seq_len)};

        Ort::Value input_tensor = Ort::Value::CreateTensor<std::int64_t>(
            ort_state_->memory_info,
            token_ids.data(), token_ids.size(),
            input_shape.data(), input_shape.size());

        const char* input_names[] = {"input_ids"};
        const char* output_names[] = {"last_hidden_state"};
        Ort::Value inputs[] = {std::move(input_tensor)};

        Ort::Session* session = ort_state_->hailo_session.get();
        if (!session) {
                        HQ_LOG_ERROR("Text encoder session not loaded");

            return std::unexpected{PipelineError::ONNXSessionLoadFailed};
        }

        auto output_tensors = session->Run(
            Ort::RunOptions{nullptr},
            input_names, inputs, 1,
            output_names, 1);

        if (output_tensors.empty() || !output_tensors[0].IsTensor()) {
                        HQ_LOG_ERROR("Text encoder output is not a tensor");

            return std::unexpected{PipelineError::ONNXRunFailed};
        }

        Ort::Value& output_tensor = output_tensors[0];
        const float* output_data = output_tensor.GetTensorData<float>();
        const auto output_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();

        std::size_t output_count = 1;
        for (auto d : output_shape) {
            if (d > 0) output_count *= static_cast<std::size_t>(d);
        }

        std::vector<float> embeddings(output_data, output_data + output_count);
        {
            std::string shape_str;
            for (std::size_t si = 0; si < output_shape.size(); ++si) {
                if (si > 0) shape_str += ',';
                shape_str += std::to_string(output_shape[si]);
            }
            HQ_LOG_DEBUG("Text encoder output: {} floats, shape=[{}]", embeddings.size(), shape_str);
        }

        return embeddings;

    } catch (const Ort::Exception& e) {
                HQ_LOG_ERROR("Text encoder Run() failed: {}", e.what());

        return std::unexpected{PipelineError::ONNXRunFailed};
    } catch (const std::exception& e) {
                HQ_LOG_ERROR("Text encoder exception: {}", e.what());

        return std::unexpected{PipelineError::ONNXRunFailed};
    }
}

// ===========================================================================
// decode_latents_() -- Decode latent tensor to RGBA image using VAE decoder
// ===========================================================================

std::expected<GeneratedImage, PipelineError>
Pipeline::decode_latents_(hq::tensor::LatentTensor<const float> latents,
                          std::uint32_t out_width, uint32_t out_height) {
    const std::size_t latent_count = latents.num_elements();
    if (!ort_state_->vae_session) {
                HQ_LOG_ERROR("VAE decoder session not loaded");

        return std::unexpected{PipelineError::ONNXSessionLoadFailed};
    }

    try {
        constexpr float kVaeScaleFactor = 0.18215f;

        auto scaled_alloc_r = memory_manager_->allocate(latent_count * sizeof(float), MemoryTier::Cool);
        if (!scaled_alloc_r) {
            HQ_LOG_ERROR("TMM: scaled_latents alloc failed ({} B): {}",
                         latent_count * sizeof(float), to_string(scaled_alloc_r.error()));
            return std::unexpected{PipelineError::GPUOutOfMemory};
        }
        ScopedTierAlloc scaled_scope(*memory_manager_, *scaled_alloc_r);
        float* scaled_latents = static_cast<float*>(scaled_scope.ptr());
        for (std::size_t i = 0; i < latent_count; ++i) {
            scaled_latents[i] = latents[i] * kVaeScaleFactor;
        }

        const std::int64_t latent_h = static_cast<std::int64_t>(out_height / 8);
        const std::int64_t latent_w = static_cast<std::int64_t>(out_width / 8);
        const std::array<std::int64_t, 4> latent_shape{1, 4, latent_h, latent_w};

        Ort::Value latent_tensor = Ort::Value::CreateTensor<float>(
            ort_state_->memory_info,
            scaled_latents, latent_count,
            latent_shape.data(), latent_shape.size());

        const char* input_names[] = {"latent_sample"};
        const char* output_names[] = {"sample"};
        Ort::Value inputs[] = {std::move(latent_tensor)};

        auto output_tensors = ort_state_->vae_session->Run(
            Ort::RunOptions{nullptr},
            input_names, inputs, 1,
            output_names, 1);

        if (output_tensors.empty() || !output_tensors[0].IsTensor()) {
                        HQ_LOG_ERROR("VAE output is not a tensor");

            return std::unexpected{PipelineError::ONNXRunFailed};
        }

        Ort::Value& output_tensor = output_tensors[0];
        const float* output_data = output_tensor.GetTensorData<float>();
        const auto output_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();

        if (output_shape.size() != 4 || output_shape[0] != 1 || output_shape[1] != 3) {
                        HQ_LOG_INFO("VAE unexpected output shape: {} dims", output_shape.size());

            return std::unexpected{PipelineError::ONNXRunFailed};
        }

        const std::int64_t img_h = output_shape[2];
        const std::int64_t img_w = output_shape[3];
        const std::size_t  img_h_sz = static_cast<std::size_t>(img_h);
        const std::size_t  img_w_sz = static_cast<std::size_t>(img_w);
        const std::size_t  px_count = img_h_sz * img_w_sz;

        std::vector<std::uint8_t> rgba(px_count * 4);

        for (std::size_t y = 0; y < img_h_sz; ++y) {
            for (std::size_t x = 0; x < img_w_sz; ++x) {
                const std::size_t src_idx = y * img_w_sz + x;
                const std::size_t dst_idx = (y * img_w_sz + x) * 4;

                float r_f = output_data[src_idx + 0 * px_count];
                float g_f = output_data[src_idx + 1 * px_count];
                float b_f = output_data[src_idx + 2 * px_count];

                auto clamp_norm = [](float v) -> std::uint8_t {
                    float clamped = std::clamp(v, -1.0f, 1.0f);
                    return static_cast<std::uint8_t>((clamped + 1.0f) * 0.5f * 255.0f);
                };

                rgba[dst_idx + 0] = clamp_norm(r_f);
                rgba[dst_idx + 1] = clamp_norm(g_f);
                rgba[dst_idx + 2] = clamp_norm(b_f);
                rgba[dst_idx + 3] = 255;
            }
        }

                HQ_LOG_INFO("VAE decoded {}x{} image from latents", img_w, img_h);


        return GeneratedImage{
            .pixels             = std::move(rgba),
            .width              = static_cast<uint32_t>(img_w),
            .height             = static_cast<uint32_t>(img_h),
            .generation_time_ms = 0.0f,
            .acceleration       = {},
        };

    } catch (const Ort::Exception& e) {
                HQ_LOG_ERROR("VAE Run() failed: {}", e.what());

        return std::unexpected{PipelineError::ONNXRunFailed};
    } catch (const std::exception& e) {
                HQ_LOG_ERROR("VAE decode exception: {}", e.what());

        return std::unexpected{PipelineError::ONNXRunFailed};
    }
}

// ===========================================================================
// Utility functions
// ===========================================================================

PipelineStats Pipeline::get_stats() const {
    return stats_;
}

HealthReport Pipeline::get_health_report() const {
    return health_score_.compute();
}

// ---------------------------------------------------------------------------
void Pipeline::shutdown() noexcept {
    if (shutdown_) return;

        HQ_LOG_INFO("Shutting down...");

    shutdown_ = true;

    // Stop cluster transport before releasing sessions
    if (transport_) {
        transport_->stop();
        transport_.reset();
    }

    // Release ONNX sessions
    if (ort_state_) {
        ort_state_->gpu_session.reset();
        ort_state_->hailo_session.reset();
        ort_state_->vae_session.reset();
    }

    // Subsystems clean up via destructors
    watchdog_.reset();
    hailo_monitor_.reset();
    staging_manager_.reset();
    gpu_monitor_.reset();

        HQ_LOG_INFO("Shutdown complete. Generations: {}, Recoveries: {}", stats_.generations_completed, stats_.watchdog_recoveries);

}

// ---------------------------------------------------------------------------
bool Pipeline::validate_model_path_(const std::filesystem::path& p) noexcept {
    return std::filesystem::exists(p) && std::filesystem::is_regular_file(p);
}

// ---------------------------------------------------------------------------
std::string Pipeline::error_string_(PipelineError e) {
    return to_string(e);
}

// ===========================================================================
// try_cluster_dispatch_() -- Serialize request, send to worker, receive result
// ===========================================================================

std::expected<GeneratedImage, PipelineError>
Pipeline::try_cluster_dispatch_(const GenerationRequest& req,
                                const cluster::DispatchDecision& decision) {
    // Serialize GenerationRequest to a flat binary payload
    const std::uint32_t prompt_len =
        static_cast<std::uint32_t>(req.prompt.size());
    const std::size_t payload_sz =
        sizeof(prompt_len) + prompt_len +
        sizeof(req.width) + sizeof(req.height) +
        sizeof(req.num_steps) + sizeof(req.guidance_scale) +
        sizeof(req.seed);

    std::vector<std::byte> payload(payload_sz);
    std::size_t off = 0;
    auto pack = [&](const void* src, std::size_t n) {
        std::memcpy(payload.data() + off, src, n);
        off += n;
    };
    pack(&prompt_len,         sizeof(prompt_len));
    pack(req.prompt.data(),   prompt_len);
    pack(&req.width,          sizeof(req.width));
    pack(&req.height,         sizeof(req.height));
    pack(&req.num_steps,      sizeof(req.num_steps));
    pack(&req.guidance_scale, sizeof(req.guidance_scale));
    pack(&req.seed,           sizeof(req.seed));

    auto send_res = transport_->send(
        decision.target_node_id, cluster::MsgType::GenerateRequest,
        std::span<const std::byte>(payload));
    if (!send_res) {
        HQ_LOG_WARN("cluster send to node {} failed: {}",
                    decision.target_node_id,
                    cluster::to_string(send_res.error()));
        return std::unexpected{PipelineError::Unknown};
    }

    // Timeout: 2s per step + 30s overhead
    const auto timeout = std::chrono::milliseconds(
        static_cast<long long>(req.num_steps) * 2000LL + 30000LL);
    auto recv_res = transport_->recv(timeout);
    if (!recv_res) {
        HQ_LOG_WARN("cluster recv from node {} failed: {}",
                    decision.target_node_id,
                    cluster::to_string(recv_res.error()));
        return std::unexpected{PipelineError::Unknown};
    }
    if (recv_res->type != cluster::MsgType::GenerateResult) {
        HQ_LOG_WARN("cluster: unexpected msg type {:#x} from node {}",
                    static_cast<unsigned>(recv_res->type),
                    recv_res->from_node_id);
        return std::unexpected{PipelineError::Unknown};
    }

    // Deserialize GeneratedImage
    const auto& rp = recv_res->payload;
    constexpr std::size_t kHdrSz =
        sizeof(std::uint32_t) +   // width
        sizeof(std::uint32_t) +   // height
        sizeof(float)         +   // generation_time_ms
        sizeof(std::uint64_t);    // pixel_count

    if (rp.size() < kHdrSz) {
        HQ_LOG_WARN("cluster result header truncated: {} bytes from node {}",
                    rp.size(), recv_res->from_node_id);
        return std::unexpected{PipelineError::Unknown};
    }

    std::size_t roff = 0;
    auto unpack = [&](void* dst, std::size_t n) {
        std::memcpy(dst, rp.data() + roff, n);
        roff += n;
    };
    std::uint32_t img_w{}, img_h{};
    float         gen_time_ms{};
    std::uint64_t pixel_count{};
    unpack(&img_w,       sizeof(img_w));
    unpack(&img_h,       sizeof(img_h));
    unpack(&gen_time_ms, sizeof(gen_time_ms));
    unpack(&pixel_count, sizeof(pixel_count));

    // Sanity check before allocating
    const std::uint64_t expected_pixels =
        static_cast<std::uint64_t>(img_w) * img_h * 4ULL;
    if (pixel_count != expected_pixels || rp.size() < kHdrSz + pixel_count) {
        HQ_LOG_WARN("cluster result payload invalid: "
                    "pixel_count={} expected={} total_bytes={} from node {}",
                    pixel_count, expected_pixels, rp.size(),
                    recv_res->from_node_id);
        return std::unexpected{PipelineError::Unknown};
    }

    std::vector<std::uint8_t> pixels(pixel_count);
    std::memcpy(pixels.data(), rp.data() + roff, pixel_count);

    HQ_LOG_INFO("cluster result: {}x{} from node {} in {:.1f} ms",
                img_w, img_h, recv_res->from_node_id, gen_time_ms);

    return GeneratedImage{
        .pixels             = std::move(pixels),
        .width              = img_w,
        .height             = img_h,
        .generation_time_ms = gen_time_ms,
        .acceleration       = {},  // cluster path — no local hardware acceleration info
    };
}

} // namespace hq