#pragma once
/// @file pipeline.hpp
/// @copyright Copyright (c) 2026 D Hargreaves (AKA Roylepython). LamiaFabrica. All rights reserved.
/// Core pipeline integration header -- glue between watchdog, Hailo monitor,
/// staging buffers, and ONNX Runtime inference sessions.
///
/// @author LamiaFabrica Team
/// @version 2.0.0

#include "hq/clip_tokenizer.hpp"
#include "hq/cluster_transport.hpp"
#include "hq/cxx26_features.hpp"
#include "hq/staging_manager.hpp"
#include "hq/tensor_view.hpp"
#include "hq/tiered_memory_manager.hpp"
#include "hq/utilization_watchdog.hpp"
#include "hq/hailo_monitor.hpp"
#include "hq/gpu_monitor.hpp"
#include "hq/pinned_staging.hpp"
#include "hq/deis_scheduler.hpp"
#include "hq/hip_graph_denoiser.hpp"
#include "hq/health_score.hpp"

#include <cstdint>
#if UM790_HAS_STD_EXPECTED
#  include <expected>
#else
#  error "pipeline.hpp requires std::expected (<expected>) — GCC >= 14 or Clang >= 18 with C++26 enabled"
#endif
#include <filesystem>
#if UM790_HAS_STD_FORMAT
#  include <format>
#endif
#include <memory>
#include <optional>
#if UM790_HAS_STD_PRINT
#  include <print>
#endif
#include <span>
#include <string>
#include <vector>

// ONNX Runtime C++ API (forward declarations to avoid leaking)
namespace Ort {
class Env;
class Session;
class SessionOptions;
class MemoryInfo;
} // namespace Ort

// NPU encoder + post-processor abstractions (forward-declared to avoid header leaks)
namespace hq::npu { class INpuEncoder; }
namespace hq::npu { class INpuPostProcessor; }

namespace hq {

// ---------------------------------------------------------------------------
/// @brief All possible pipeline error conditions.
///
/// Used as the error type in std::expected<T, PipelineError> throughout
/// the public API. Each variant carries a human-readable message.
// ---------------------------------------------------------------------------
enum class PipelineError : std::uint32_t {
    Ok                          = 0,
    InvalidRequest              = 1,   ///< Request parameters out of bounds
    WatchdogRecoveryFailed      = 2,   ///< Recovery callback failed
    HailoNotAvailable           = 3,   ///< Hailo device unreachable  (Reserved for future hardware error detection paths)
    HailoTimeout                = 4,   ///< Hailo inference timed out  (Reserved for future hardware error detection paths)
    HailoThermal                = 5,   ///< Hailo thermal throttling active  (Reserved for future hardware error detection paths)
    GPUOutOfMemory              = 6,   ///< GPU allocation failed  (Reserved for future hardware error detection paths)
    ONNXSessionLoadFailed       = 7,   ///< Could not load model.onnx
    ONNXRunFailed               = 8,   ///< Inference call failed
    StagingPoolExhausted        = 9,   ///< No staging buffers available
    LatencyBudgetExceeded       = 10,  ///< Step too slow, watchdog limit  (Reserved for future hardware error detection paths)
    RecoveryTooManyAttempts     = 11,  ///< Max recoveries exceeded
    InvalidModelPath            = 12,  ///< Model file not found
    ShutdownInProgress          = 13,  ///< Pipeline shutting down
    SchedulerNotInitialized     = 14,  ///< DEIS scheduler not created before denoise
    Unknown                     = 15,  ///< Catch-all
};

// ---------------------------------------------------------------------------
/// @brief Single image generation request.
// ---------------------------------------------------------------------------
struct GenerationRequest {
    std::string prompt;            ///< Text prompt (e.g., "a cat in space")
    uint32_t    width{512};        ///< Output width in pixels
    uint32_t    height{512};       ///< Output height in pixels
    uint32_t    num_steps{20};     ///< Number of denoising steps
    float       guidance_scale{7.5f}; ///< Classifier-free guidance
    int64_t     seed{-1};          ///< Random seed (-1 = random)
};

// ---------------------------------------------------------------------------
/// @brief Honest hardware acceleration report for a single generate() call.
///
/// Every stage reports whether real hardware was used or a CPU/synthetic
/// fallback. Callers should treat any stage marked `false` as "no acceleration
/// — data was produced by CPU compute or deterministic synthesis."
// ---------------------------------------------------------------------------
struct HardwareAccelerationReport {
    bool  text_encode_used_npu{false};       ///< True only if real NPU hardware encoded text
    bool  text_encode_used_gpu{false};       ///< True only if GPU-accelerated text encoding
    bool  denoise_used_gpu{false};           ///< True only if UNet ran on GPU (CUDA/ROCm EP)
    bool  vae_decode_used_gpu{false};         ///< True only if VAE ran on GPU
    bool  post_process_used_npu{false};      ///< True only if real NPU post-processing
    bool  cfg_blend_used_npu{false};         ///< True only if CFG blend ran on NPU
    bool  hailo_telemetry_real{false};       ///< True only if HailoMonitor is reading real sensors
    bool  gpu_telemetry_real{false};         ///< True only if GPUMonitor is reading real sensors

    // --- NEW: Round 24 hostile-review hardening ---
    /// @brief Percentage [0–100] of NPU-acceleratable *cheap* components that used NPU.
    ///        Components counted: text_encode, post_process, cfg_blend.
    ///        IMPORTANT: This does NOT include UNet denoising or VAE decode,
    ///        which are ~90% of wall-clock compute. A value of 100% here
    ///        means the NPU handled 3 lightweight tasks while the expensive
    ///        work still ran on CPU/GPU. Do not misinterpret as total NPU share.
    std::uint8_t npu_cheap_ops_percent{0};

    /// @brief True only when UNet denoising ran on real NPU hardware.
    ///        Currently always false — UNet on Hailo-8L requires a compiled
    ///        HEF that does not exist and HailoRT (Linux only) is not installed.
    bool unet_denoise_used_npu{false};

    /// @brief True when the selected encoder is a fallback (synthetic_mode() == true).
    bool encoder_is_fallback{false};

    /// @brief True when the selected post-processor is a fallback (synthetic_mode() == true).
    bool post_processor_is_fallback{false};

    std::string encoder_name;                ///< npu_encoder_->name() — "Hailo-8L", "ONNX-CPU", "none"
    std::string post_processor_name;          ///< npu_post_processor_->name()
    std::string gpu_backend_name;            ///< "NVML", "ROCM_SMI", "None"
};

// ---------------------------------------------------------------------------
/// @brief Result of a successful image generation.
// ---------------------------------------------------------------------------
struct GeneratedImage {
    std::vector<std::uint8_t> pixels;  ///< RGBA8 pixel data
    uint32_t                  width{0};   ///< Image width
    uint32_t                  height{0};  ///< Image height
    float                     generation_time_ms{0.0f};  ///< Wall-clock time
    HardwareAccelerationReport acceleration;  ///< Honest hardware usage report
};

// ---------------------------------------------------------------------------
/// @brief Per-phase timing breakdown for one generate() call.
///
/// Populated by generate() and accessible via Pipeline::last_phase_timings().
/// Enables the `cerberus profile` command to show NPU/GPU/CPU time breakdown
/// without requiring hardware that can run Zen 4 binaries.
// ---------------------------------------------------------------------------
struct PipelinePhaseTimings {
    double   text_encode_ms{0.0};       ///< encode_prompt_() wall-clock time
    double   embedding_stage_ms{0.0};   ///< TMM alloc + DMA staging time
    double   denoise_total_ms{0.0};     ///< full denoising loop (all steps)
    double   npu_blend_in_loop_us{0.0}; ///< total npu_post_processor_->blend_noise_cfg() time (all steps, µs)
    double   vae_decode_ms{0.0};        ///< decode_latents_() time
    double   post_process_ms{0.0};      ///< npu_post_processor_->post_process() time
    uint32_t num_denoise_steps{0};      ///< actual steps executed
    std::string encoder_name;           ///< npu_encoder_->name() selected at init
    std::string post_processor_name;    ///< npu_post_processor_->name() selected at init
    HardwareAccelerationReport acceleration;  ///< hardware usage report from last generate()
};

// ---------------------------------------------------------------------------
/// @brief Runtime pipeline statistics.
// ---------------------------------------------------------------------------
struct PipelineStats {
    uint64_t generations_completed{0};
    uint64_t generations_failed{0};
    uint32_t watchdog_recoveries{0};
    double   avg_generation_ms{0.0};
    double   avg_gpu_utilization{0.0};
    double   avg_hailo_utilization{0.0};
    uint64_t total_steps_executed{0};
};

// ---------------------------------------------------------------------------
/// @brief Tunable pipeline configuration.
// ---------------------------------------------------------------------------
struct PipelineConfig {
    // --- Watchdog thresholds (match WatchdogConfig) ---
    float         watchdog_gpu_low_threshold{60.0f};        ///< % below => WARNING
    float         watchdog_gpu_critical_threshold{40.0f};   ///< % below => CRITICAL
    float         watchdog_hailo_low_threshold{60.0f};      ///< % below => WARNING
    float         watchdog_hailo_critical_threshold{40.0f}; ///< % below => CRITICAL
    std::uint32_t watchdog_consecutive_threshold{8};        ///< steps before recovery
    std::uint32_t watchdog_max_recoveries{10};              ///< max before giving up
    float         watchdog_backoff_base_ms{100.0f};         ///< base recovery delay
    float         watchdog_backoff_max_ms{30000.0f};        ///< cap at 30 s
    float         watchdog_thermal_threshold_c{85.0f};      ///< thermal throttling detect
    bool          enable_watchdog{true};                    ///< Enable utilization watchdog

    // --- Staging ---
    uint32_t staging_buffer_count{8};           ///< Pool size
    uint32_t staging_buffer_size_mb{64};        ///< Each buffer (MiB)

    // --- Model paths ---
    std::filesystem::path text_encoder_onnx;    ///< Prompt -> embeddings
    std::filesystem::path unet_onnx;            ///< Denoising UNet
    std::filesystem::path vae_decoder_onnx;     ///< Latent -> RGB

    // --- Inference ---
    uint32_t gpu_batch_size{1};                 ///< GPU work batch
    uint32_t hailo_batch_size{1};               ///< Hailo work batch
    bool     use_hailo_text_encoder{true};      ///< Offload text encoding?
    bool     use_hip_staging{true};             ///< Use PinnedStagingPool for GPU staging
    bool     enable_hip_graph{false};           ///< Use HIP graph capture/replay denoiser

    // --- Recovery ---
    uint32_t max_recovery_attempts{3};          ///< Fail after N recoveries

    // --- Tiered memory ---
    TieredMemoryConfig memory_config;           ///< Four-tier memory manager settings

    // --- Clustering (nullopt = local-only mode) ---
    std::optional<cluster::TransportConfig> cluster_config;
};

// ---------------------------------------------------------------------------
/// @class Pipeline
/// @brief End-to-end image generation pipeline for UM790 Pro.
///
/// Orchestrates the full inference graph:
///  1. Text encode (Hailo via ONNX Runtime + Hailo EP)
///  2. Embedding staging (pinned CPU -> GPU via EmbeddingStagingManager)
///  3. Denoising loop (GPU via ONNX Runtime ROCm EP)
///     a. Per-step utilization monitoring via UtilizationWatchdog
///     b. Automatic recovery on low utilization
///  4. VAE decode (CPU fallback or ROCm EP)
///  5. Return RGBA image
///
/// Thread-safety: Not thread-safe. One Pipeline instance per thread.
// ---------------------------------------------------------------------------
class Pipeline {
public:
    /// @brief Construct and initialize all subsystems.
    /// @param cfg Pipeline configuration.
    /// @throws std::runtime_error on unrecoverable initialization failure.
    explicit Pipeline(const PipelineConfig& cfg);
    ~Pipeline();

    // Non-copyable, movable
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;
    Pipeline(Pipeline&&) noexcept;
    Pipeline& operator=(Pipeline&&) noexcept;

    /// @brief Generate a single image.
    /// @param req Generation parameters.
    /// @return Generated image on success, PipelineError on failure.
    [[nodiscard]] std::expected<GeneratedImage, PipelineError>
        generate(const GenerationRequest& req);

    /// @brief Generate a batch of images.
    /// @param requests Vector of generation requests.
    /// @return Vector of results (success or error per request).
    [[nodiscard]] std::vector<std::expected<GeneratedImage, PipelineError>>
        generate_batch(const std::vector<GenerationRequest>& requests);

    /// @brief Get current runtime statistics.
    [[nodiscard]] PipelineStats get_stats() const;

    /// @brief Get per-phase timings from the most recent generate() call.
    [[nodiscard]] const PipelinePhaseTimings& last_phase_timings() const noexcept {
        return last_phase_timings_;
    }

    /// @brief Get the current composite health report.
    [[nodiscard]] HealthReport get_health_report() const;

    /// @brief Graceful shutdown -- releases all resources.
    /// After shutdown, generate() will return ShutdownInProgress.
    void shutdown() noexcept;

private:
    /// @brief Initialize ONNX Runtime sessions.
    [[nodiscard]] bool initialize_onnx_sessions_();

    /// @brief Run a single denoising step with real ONNX Runtime inference.
    /// @param step       Current denoising step index.
    /// @param latents    TensorView over TMM Cool-tier buffer [1,4,H/8,W/8] (in/out).
    /// @param cond_emb   TensorView over TMM-backed conditional embeddings [1,77,hidden].
    /// @param uncond_emb TensorView over unconditional embeddings; nullopt = CFG off.
    /// @param guidance_scale CFG scale. <=1.0 disables CFG (single UNet pass).
    [[nodiscard]] std::expected<void, PipelineError>
        denoise_step_(std::uint32_t step,
                      hq::tensor::FloatTensor4D latents,
                      hq::tensor::EmbeddingTensor<float> cond_emb,
                      std::optional<hq::tensor::EmbeddingTensor<float>> uncond_emb,
                      float guidance_scale);

    /// @brief Recovery callback -- saves/restores latent tensors, rebuilds sessions.
    /// @param unit          Which accelerator triggered the recovery.
    /// @param step          Step number where recovery was triggered.
    /// @param util_at_fault Utilization percentage that caused the trigger.
    /// @param latents       TensorView over TMM latent buffer [1,4,H/8,W/8] (in/out).
    [[nodiscard]] std::expected<RecoveryResult, PipelineError>
        on_watchdog_recovery_(ComputeUnit unit, std::uint32_t step,
                               float util_at_fault,
                               hq::tensor::FloatTensor4D latents);

    /// @brief Zero-copy denoising step using GPU device pointers (BUG B3 fix).
    ///
    /// When latents and embeddings already reside in GPU memory (e.g. from
    /// PinnedStagingPool), this overload creates Ort::Value tensors using
    /// the GPU allocator memory info so ONNX Runtime reads/writes directly
    /// in device memory with no implicit H2D round-trip.
    ///
    /// @param step       Current denoising step index.
    /// @param gpu_latents    Device pointer to latents [1,4,H/8,W/8] (in/out).
    /// @param gpu_cond_emb   Device pointer to conditional embeddings [1,77,hidden].
    /// @param gpu_uncond_emb Device pointer to unconditional embeddings; nullopt = CFG off.
    /// @param guidance_scale CFG scale. <=1.0 disables CFG.
    /// @param latent_count   Total float elements in gpu_latents.
    /// @param emb_count      Total float elements in gpu_cond_emb / gpu_uncond_emb.
    [[nodiscard]] std::expected<void, PipelineError>
        denoise_step_(std::uint32_t step,
                      float* gpu_latents,
                      float* gpu_cond_emb,
                      std::optional<float*> gpu_uncond_emb,
                      float guidance_scale,
                      std::size_t latent_count,
                      std::size_t emb_count,
                      std::int64_t latent_h,
                      std::int64_t latent_w);

    /// @brief Encode text prompt to embedding tensor using Hailo.
    [[nodiscard]] std::expected<std::vector<float>, PipelineError>
        encode_prompt_(const std::string& prompt);

    /// @brief Decode latent tensor to RGBA image using VAE.
    /// @param latents    TensorView over TMM latent buffer [1,4,H/8,W/8] (read-only).
    [[nodiscard]] std::expected<GeneratedImage, PipelineError>
        decode_latents_(hq::tensor::LatentTensor<const float> latents,
                        std::uint32_t out_width, std::uint32_t out_height);

    /// @brief Attempt to dispatch a generation request to a cluster worker.
    /// Returns the remote result on success, or empty on any failure (caller
    /// must fall back to local generation).
    [[nodiscard]] std::expected<GeneratedImage, PipelineError>
        try_cluster_dispatch_(const GenerationRequest& req,
                              const cluster::DispatchDecision& decision);

    /// @brief Check if a file exists and is readable.
    [[nodiscard]] static bool validate_model_path_(
        const std::filesystem::path& p) noexcept;

    /// @brief Convert PipelineError to human-readable string.
    [[nodiscard]] static std::string error_string_(PipelineError e);

    // --- Configuration ---
    PipelineConfig cfg_;

    // --- Subsystems ---
    std::unique_ptr<UtilizationWatchdog>       watchdog_;
    std::unique_ptr<HailoMonitor>              hailo_monitor_;
    std::unique_ptr<EmbeddingStagingManager>   staging_manager_;
    std::unique_ptr<PinnedStagingPool<float>>  hip_staging_;
    std::unique_ptr<GPUMonitor>                gpu_monitor_;
    std::unique_ptr<TieredMemoryManager>       memory_manager_;
    std::unique_ptr<cluster::ClusterTransport> transport_;

    // --- CLIP Tokenizer (for text prompt encoding) ---
    std::unique_ptr<CLIPTokenizer> tokenizer_;

    // --- ONNX Runtime (pImpl to avoid header leakage) ---
    class OrtState;
    std::unique_ptr<OrtState> ort_state_;

    // --- Statistics / health ---
    PipelineStats stats_{};
    PipelineHealthScore health_score_{};

    // --- NPU encoder (INpuEncoder abstraction — factory-selected at init) ---
    std::unique_ptr<hq::npu::INpuEncoder> npu_encoder_;

    // --- NPU post-processor (INpuPostProcessor abstraction — factory-selected at init) ---
    std::unique_ptr<hq::npu::INpuPostProcessor> npu_post_processor_;

    // --- Phase timings for the most recent generate() call ---
    PipelinePhaseTimings last_phase_timings_{};
    double npu_blend_accumulated_us_{0.0};  ///< accumulates blend_noise_cfg time across all denoise steps

    // --- DEIS Scheduler (replaces inline DDIM math) ---
    std::unique_ptr<DEISScheduler> scheduler_;

    // --- HIP Graph Denoiser (optional, capture/replay/fallback) ---
    std::unique_ptr<HIPGraphDenoiser> hip_denoiser_;

    // --- Latent checkpoint for recovery (TMM Cool-tier, RAII) ---
    std::optional<ScopedTierAlloc> latent_checkpoint_;
    std::size_t latent_checkpoint_floats_{0};

    // --- Current latents reference for watchdog recovery callback ---
    hq::tensor::FloatTensor4D current_latents_{};

    // --- Lifecycle ---
    bool shutdown_{false};
    bool recovery_in_progress_{false};
    uint32_t recovery_attempts_{0};
};

// ===========================================================================
// DESIGN NOTE — Future GPU Zero-Copy Denoising Path (BUG B3)
// ===========================================================================
//
// Current state:
//   Pipeline::generate() stages embeddings to GPU via PinnedStagingPool or
//   EmbeddingStagingManager (H2D DMA), then calls denoise_step_() which
//   ignores the staged buffer and creates Ort::Value tensors from raw CPU
//   std::vector<float> pointers. ONNX Runtime's ROCm EP implicitly performs
//   a second H2D copy under the hood, so the staging DMA is wasted.
//
// Target architecture (zero-copy):
//   1. denoise_step_() gains a device-pointer overload:
//        denoise_step_(step, void* gpu_latents, void* gpu_embeddings,
//                      void* gpu_uncond, ...)
//
//   2. This overload creates Ort::Value tensors using Ort::MemoryInfo
//      configured for the HIP/ROCm device allocator:
//        auto mem_info = Ort::MemoryInfo::Create("Hip", ...);
//        Ort::Value::CreateTensor(mem_info, gpu_ptr, count, shape, ...)
//
//   3. generate() would:
//      a. Run text encoder on Hailo → CPU embeddings vector
//      b. Stage embeddings H2D once via PinnedStagingPool (kept)
//      c. Initialize latents on GPU via hipMalloc + H2D
//      d. In the loop, call the device-pointer overload of denoise_step_()
//         with the GPU-staging handles — ONNX Runtime reads/writes directly
//         in GPU memory with no implicit H2D round-trip.
//      e. VAE decode reads final latents from GPU (or reads back once).
//
// Benefits:
//   - Eliminates the implicit per-step H2D copy of latents and embeddings.
//   - For 20-step denoising with 77×768 embeddings (~59K floats) and
//     4×64×64 latents (~16K floats), saves ~20 × (75K × 4B) ≈ 6 MB of
//     per-generation H2D traffic.
// ===========================================================================

// ---------------------------------------------------------------------------
/// @brief Convenience free function for error formatting.
// ---------------------------------------------------------------------------
inline std::string to_string(PipelineError e) {
    switch (e) {
        case PipelineError::Ok:                      return "Ok";
        case PipelineError::InvalidRequest:          return "InvalidRequest";
        case PipelineError::WatchdogRecoveryFailed:  return "WatchdogRecoveryFailed";
        case PipelineError::HailoNotAvailable:       return "HailoNotAvailable";
        case PipelineError::HailoTimeout:            return "HailoTimeout";
        case PipelineError::HailoThermal:            return "HailoThermal";
        case PipelineError::GPUOutOfMemory:          return "GPUOutOfMemory";
        case PipelineError::ONNXSessionLoadFailed:   return "ONNXSessionLoadFailed";
        case PipelineError::ONNXRunFailed:           return "ONNXRunFailed";
        case PipelineError::StagingPoolExhausted:    return "StagingPoolExhausted";
        case PipelineError::LatencyBudgetExceeded:   return "LatencyBudgetExceeded";
        case PipelineError::RecoveryTooManyAttempts: return "RecoveryTooManyAttempts";
        case PipelineError::InvalidModelPath:        return "InvalidModelPath";
        case PipelineError::ShutdownInProgress:      return "ShutdownInProgress";
        case PipelineError::SchedulerNotInitialized: return "SchedulerNotInitialized";
        case PipelineError::Unknown:                 return "Unknown";
    }
    return "Unknown";
}

} // namespace hq

// std::formatter specialization for PipelineError
#if UM790_HAS_STD_FORMAT
template<>
struct std::formatter<hq::PipelineError> : std::formatter<std::string_view> {
    auto format(hq::PipelineError e, std::format_context& ctx) const {
        return std::formatter<std::string_view>::format(hq::to_string(e), ctx);
    }
};
#endif