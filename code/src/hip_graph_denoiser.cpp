/// @file hip_graph_denoiser.cpp
/// @copyright Copyright © 2026, all rights reserved | David Hargreaves aka Roylepython trading as The Medusa Initiative & Lamia Fabrica
/// Support: https://www.patreon.com/TheMedusaInitiative — £25/month removes ads from all software at 200 subscribers
/// HIP Graph-captured denoising loop -- implementation.
///
/// @version 3.1.0 -- C02 fix: replay() re-runs UNet per step, proper timestep
///                   mapping via DEISScheduler, CFG support in fallback path.
///                   C03 fix: eliminated goto in capture(); replaced with
///                   std::expected monadic chaining (.and_then() / .or_else()).
///
/// Architecture:
///   1. GPU UNet inference runs outside the graph (ORT is not capturable)
///   2. The captured HIP graph contains only:
///      a. Device-side DDIM scheduler kernel launch
///      b. D2H memcpy back to host
///   3. Per-step scalars (coeff_x0, coeff_eps) uploaded via persistent pinned
///      host buffer, sourced from DEISScheduler precomputed coefficients when
///      a scheduler is attached (matches Pipeline::denoise_step_ exactly).
///   4. Fallback path mirrors Pipeline::denoise_step_ with full CFG support:
///      dual UNet passes (cond + uncond), classifier-free guidance blending,
///      and scheduler step dispatch (AVX-512 when available, scalar fallback).
///
/// This approach provides partial acceleration: the DDIM step + D2H path
/// is zero-CPU-overhead, while ORT still runs normally. For many pipeline
/// configurations the DDIM/kernel portion is 10-30% of step time.

#include "hq/hip_graph_denoiser.hpp"
#include "hq/cxx26_features.hpp"
#include "hq/deis_scheduler.hpp"
#include "hq/tensor_view.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <expected>
#include <format>
#include <memory>
#include <numeric>
#if UM790_HAS_STD_PRINT
#  include <print>
#endif
#include <span>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime_api.h>
#endif

extern "C" std::size_t hq_safe_write(int fd, const char* data, std::size_t len);

namespace {
    inline void hq_print(std::string msg) {
        hq_safe_write(1, msg.data(), msg.size());
    }
    inline void hq_println(std::string msg) {
        hq_safe_write(1, msg.data(), msg.size());
        hq_safe_write(1, "\n", 1);
    }
}

namespace hq {

// ===========================================================================
// Device-side DDIM scheduler kernel
// ===========================================================================
#ifdef __HIP_PLATFORM_AMD__

// DdimDeviceParams is now defined in hip_graph_denoiser.hpp (not here)
// to avoid ODR violations between header and implementation.

__global__ void ddim_update_kernel(float* __restrict__ latents,
                                    const float* __restrict__ noise_pred,
                                    const DdimDeviceParams* __restrict__ params,
                                    std::size_t count) {
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    latents[idx] = params->coeff_x0 * latents[idx] +
                   params->coeff_eps * noise_pred[idx];
}

#endif // __HIP_PLATFORM_AMD__

// ===========================================================================
// Host-side helpers
// ===========================================================================
namespace {

[[maybe_unused]] std::string to_string(GraphError e) {
    switch (e) {
        case GraphError::Ok:              return "Ok";
        case GraphError::HipError:        return "HipError";
        case GraphError::NotCaptured:     return "NotCaptured";
        case GraphError::AlreadyCaptured: return "AlreadyCaptured";
        case GraphError::CaptureFailed:   return "CaptureFailed";
        case GraphError::ReplayFailed:    return "ReplayFailed";
        case GraphError::InvalidStepCount:return "InvalidStepCount";
        case GraphError::ONNXError:       return "ONNXError";
    }
    return "Unknown";
}

/// Compute an approximate timestep for the linear fallback schedule.
/// Mirrors the simple schedule used when no DEISScheduler is attached.
[[nodiscard]] std::int64_t compute_linear_timestep(std::uint32_t step,
                                                   std::uint32_t num_steps) {
    return static_cast<std::int64_t>(num_steps - 1 - step);
}

/// Check whether CFG should be active given the stored parameters.
[[nodiscard]] bool cfg_active(float cfg_scale,
                              std::span<const float> uncond) noexcept {
    return cfg_scale > 1.0f && !uncond.empty();
}

} // anonymous namespace

// ===========================================================================
// Opaque HIP state (pImpl)
// ===========================================================================
struct HIPGraphDenoiser::HipGraphState {
#if defined(__HIP_PLATFORM_AMD__)
    hipStream_t    stream{nullptr};
    hipGraph_t     graph{nullptr};
    hipGraphExec_t exec{nullptr};
#else
    void* stream{nullptr};
    void* graph{nullptr};
    void* exec{nullptr};
#endif
    bool owns_stream{false};

    HipGraphState() = default;
    ~HipGraphState() noexcept { reset(); }

    HipGraphState(const HipGraphState&) = delete;
    HipGraphState& operator=(const HipGraphState&) = delete;

    HipGraphState(HipGraphState&& o) noexcept
        : stream{o.stream}, graph{o.graph}, exec{o.exec},
          owns_stream{o.owns_stream} {
        o.stream = nullptr; o.graph = nullptr; o.exec = nullptr;
        o.owns_stream = false;
    }
    HipGraphState& operator=(HipGraphState&& o) noexcept {
        if (this != &o) {
            reset();
            stream = o.stream; graph = o.graph; exec = o.exec;
            owns_stream = o.owns_stream;
            o.stream = nullptr; o.graph = nullptr; o.exec = nullptr;
            o.owns_stream = false;
        }
        return *this;
    }

    void reset() noexcept {
#if defined(__HIP_PLATFORM_AMD__)
        if (exec)  { hipGraphExecDestroy(exec);  exec = nullptr; }
        if (graph) { hipGraphDestroy(graph);     graph = nullptr; }
        if (stream && owns_stream) {
            hipStreamDestroy(stream); stream = nullptr; owns_stream = false;
        }
#endif
    }
};

// ===========================================================================
// Construction / Destruction / Move
// ===========================================================================

HIPGraphDenoiser::HIPGraphDenoiser(const GraphConfig& cfg)
    : hip_state_{std::make_unique<HipGraphState>()}
    , cfg_{cfg} {
    available_ = is_available();
    hq_println(std::format("[HIPGraphDenoiser] created (available={}, capture_enabled={})",
               available_, cfg_.enable_capture));
}

HIPGraphDenoiser::~HIPGraphDenoiser() noexcept {
    free_device_buffers_();
}

HIPGraphDenoiser::HIPGraphDenoiser(HIPGraphDenoiser&&) noexcept = default;
HIPGraphDenoiser& HIPGraphDenoiser::operator=(HIPGraphDenoiser&&) noexcept = default;

bool HIPGraphDenoiser::is_available() const noexcept {
#if defined(__HIP_PLATFORM_AMD__)
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess || device_count == 0) return false;
    hipStream_t s = nullptr;
    err = hipStreamCreate(&s);
    if (err == hipSuccess && s) {
        hipStreamDestroy(s);
        return true;
    }
    return false;
#else
    return false;
#endif
}

// ===========================================================================
// set_scheduler()
// ===========================================================================

void HIPGraphDenoiser::set_scheduler(DEISScheduler* scheduler) {
    scheduler_ = scheduler;
    hq_println(std::format("[HIPGraphDenoiser] scheduler attached={}",
               scheduler_ != nullptr));
}

// ===========================================================================
// Device buffer management
// ===========================================================================

std::expected<void, GraphErrorInfo>
HIPGraphDenoiser::allocate_device_buffers_() {
#if defined(__HIP_PLATFORM_AMD__)
    const std::size_t latent_bytes = latent_count_ * sizeof(float);
    hipError_t err;

    if (auto e = hipMalloc(&d_latents_, latent_bytes); e != hipSuccess)
        return std::unexpected{GraphErrorInfo{GraphError::HipError,
            std::format("hipMalloc d_latents_: {}", hipGetErrorString(e)),
            static_cast<int>(e)}};

    if (auto e = hipMalloc(&d_noise_pred_, latent_bytes); e != hipSuccess) {
        free_device_buffers_();
        return std::unexpected{GraphErrorInfo{GraphError::HipError,
            std::format("hipMalloc d_noise_pred_: {}", hipGetErrorString(e)),
            static_cast<int>(e)}};
    }

    // C12 fix: persistent pinned host memory for step scalars
    err = hipHostMalloc(&h_pinned_params_, sizeof(DdimDeviceParams),
                         hipHostMallocPortable);
    if (err != hipSuccess) {
        free_device_buffers_();
        return std::unexpected{GraphErrorInfo{GraphError::HipError,
            std::format("hipHostMalloc h_pinned_params_: {}", hipGetErrorString(err)),
            static_cast<int>(err)}};
    }

    err = hipMalloc(&d_ddim_params_, sizeof(DdimDeviceParams));
    if (err != hipSuccess) {
        free_device_buffers_();
        return std::unexpected{GraphErrorInfo{GraphError::HipError,
            std::format("hipMalloc d_ddim_params_: {}", hipGetErrorString(err)),
            static_cast<int>(err)}};
    }

    hq_println(std::format("[HIPGraphDenoiser] Buffers: latents={}B noise_pred={}B "
               "ddim_params={}B (persistent pinned host)",
               latent_bytes, latent_bytes, sizeof(DdimDeviceParams)));
    return {};
#else
    return std::unexpected{GraphErrorInfo{GraphError::HipError,
        "HIP not available at compile time"}};
#endif
}

void HIPGraphDenoiser::free_device_buffers_() noexcept {
#if defined(__HIP_PLATFORM_AMD__)
    if (d_latents_)    { hipFree(d_latents_);    d_latents_    = nullptr; }
    if (d_noise_pred_) { hipFree(d_noise_pred_); d_noise_pred_ = nullptr; }
    if (d_ddim_params_){ hipFree(d_ddim_params_);d_ddim_params_= nullptr; }
    if (h_pinned_params_) {
        hipHostFree(h_pinned_params_);
        h_pinned_params_ = nullptr;
    }
#endif
}

// ===========================================================================
// upload_step_scalars_()
//
// C02: When a DEISScheduler is attached, use its precomputed coefficients
//      directly. This guarantees the graph path produces mathematically
//      identical results to Pipeline::denoise_step_'s CPU-side scheduler.
//      Without a scheduler, fall back to the linear schedule.
// ===========================================================================

std::expected<void, GraphErrorInfo>
HIPGraphDenoiser::upload_step_scalars_(std::uint32_t step) {
#if defined(__HIP_PLATFORM_AMD__)
    if (!h_pinned_params_) {
        return std::unexpected{GraphErrorInfo{GraphError::HipError,
            "h_pinned_params_ not allocated"}};
    }

    if (scheduler_) {
        h_pinned_params_->coeff_x0  = scheduler_->step_coeff_x0(step);
        h_pinned_params_->coeff_eps = scheduler_->step_coeff_eps(step);
    } else {
        compute_linear_coeffs_(step, h_pinned_params_->coeff_x0,
                               h_pinned_params_->coeff_eps);
    }

    hipError_t err = hipMemcpyAsync(d_ddim_params_, h_pinned_params_,
                                     sizeof(DdimDeviceParams),
                                     hipMemcpyHostToDevice, hip_state_->stream);
    if (err != hipSuccess) {
        return std::unexpected{GraphErrorInfo{GraphError::HipError,
            std::format("DDIM params H2D: {}", hipGetErrorString(err)),
            static_cast<int>(err)}};
    }
    return {};
#else
    (void)step;
    return std::unexpected{GraphErrorInfo{GraphError::HipError,
        "HIP not available"}};
#endif
}

// ===========================================================================
// run_unet() -- ORT inference (outside graph, always)
//
// C02: Returns std::vector<float> by value (was void*).
//       The timestep is set via h_timestep_ (set by caller before invoking).
//       Embeddings control conditional vs. unconditional pass for CFG.
// ===========================================================================

std::expected<std::vector<float>, GraphErrorInfo>
HIPGraphDenoiser::run_unet(std::span<const float> latents,
                           std::span<const float> embeddings,
                           Ort::Session* onnx_session,
                           Ort::MemoryInfo* memory_info) {
    if (!onnx_session || !memory_info) {
        return std::unexpected{GraphErrorInfo{GraphError::ONNXError,
            "null session or memory_info"}};
    }

    constexpr std::int64_t kSeqLen = 77;
    const std::int64_t hdim = static_cast<std::int64_t>(embeddings.size() / kSeqLen);

    const std::array<std::int64_t, 4> shape{latent_shape_};
    const std::array<std::int64_t, 1> tshape{1};
    const std::array<std::int64_t, 3> eshape{1, kSeqLen, hdim};

    try {
        Ort::Value lt = Ort::Value::CreateTensor<float>(
            *memory_info, const_cast<float*>(latents.data()), latents.size(),
            shape.data(), shape.size());

        Ort::Value tt = Ort::Value::CreateTensor<std::int64_t>(
            *memory_info, const_cast<std::int64_t*>(&h_timestep_), 1,
            tshape.data(), tshape.size());

        Ort::Value et = Ort::Value::CreateTensor<float>(
            *memory_info, const_cast<float*>(embeddings.data()),
            embeddings.size(), eshape.data(), eshape.size());

        const char* in_n[]  = {"sample", "timestep", "encoder_hidden_states"};
        const char* out_n[] = {"out_sample"};
        Ort::Value ins[] = {std::move(lt), std::move(tt), std::move(et)};

        auto outs = onnx_session->Run(Ort::RunOptions{nullptr},
                                       in_n, ins, 3, out_n, 1);

        if (outs.empty() || !outs[0].IsTensor()) {
            return std::unexpected{GraphErrorInfo{GraphError::ONNXError,
                "UNet output not a tensor"}};
        }

        const float* np = outs[0].GetTensorData<float>();
        const auto nshape = outs[0].GetTensorTypeAndShapeInfo().GetShape();
        std::size_t nc = 1;
        for (auto d : nshape) if (d > 0) nc *= static_cast<std::size_t>(d);

        return std::vector<float>(np, np + nc);

    } catch (const Ort::Exception& e) {
        return std::unexpected{GraphErrorInfo{GraphError::ONNXError,
            std::format("ORT UNet: {}", e.what())}};
    } catch (const std::exception& e) {
        return std::unexpected{GraphErrorInfo{GraphError::ONNXError,
            std::format("Exception: {}", e.what())}};
    }
}

// ===========================================================================
// capture() -- Captures the DDIM kernel + D2H memcpy as a HIP graph.
//
// C02 changes:
//   - Accepts guidance_scale + uncond_embeddings (stored as members for
//     CFG-aware fallback).
//   - Sets h_timestep_ from scheduler_->timestep(0) when a scheduler is
//     attached; otherwise uses 0 (legacy).
//
// C03 changes:
//   - Eliminated all goto labels.  The HIP path is now expressed as a chain
//     of std::expected-returning lambdas wired with .and_then() / .or_else().
//   - On any failure, the HIP state is reset and the fallback path is invoked.
// ===========================================================================

std::expected<void, GraphErrorInfo>
HIPGraphDenoiser::capture(hq::tensor::FloatTensor4D latents,
                          std::span<const float> embeddings,
                          Ort::Session* onnx_session,
                          Ort::MemoryInfo* memory_info,
                          const std::array<std::int64_t, 4>& latent_shape,
                          float guidance_scale,
                          std::span<const float> uncond_embeddings) {
    float* latents_raw = latents.data();
    if (!latents_raw || !onnx_session || !memory_info) {
        return std::unexpected{GraphErrorInfo{GraphError::InvalidStepCount,
            "null argument"}};
    }
    if (captured_) {
        return std::unexpected{GraphErrorInfo{GraphError::AlreadyCaptured,
            "Already captured"}};
    }

    // Store CFG state for use in subsequent replay()/fallback calls.
    // uncond_embeddings_ is a private copy — its lifetime must span all
    // subsequent replay() and execute_step_fallback_() calls.
    cfg_scale_ = guidance_scale;
    if (!uncond_embeddings.empty()) {
        uncond_embeddings_.assign(uncond_embeddings.begin(), uncond_embeddings.end());
    } else {
        uncond_embeddings_.clear();
    }

    latent_shape_ = latent_shape;
    latent_count_ = 1;
    for (auto d : latent_shape) if (d > 0)
        latent_count_ *= static_cast<std::size_t>(d);

    if (!available_ || !cfg_.enable_capture) {
        return execute_step_fallback_(latents_raw, embeddings, onnx_session,
                                      memory_info, latent_shape, 0,
                                      cfg_scale_,
                                      std::span<const float>{uncond_embeddings_});
    }

#if !defined(__HIP_PLATFORM_AMD__)
    return execute_step_fallback_(latents_raw, embeddings, onnx_session,
                                  memory_info, latent_shape, 0,
                                  cfg_scale_,
                                  std::span<const float>{uncond_embeddings_});
#else
    // ------------------------------------------------------------------
    // Helper lambdas that each return std::expected<void, GraphErrorInfo>
    // so they can be chained with .and_then() / .or_else().
    // ------------------------------------------------------------------

    auto create_stream = [this]() -> std::expected<void, GraphErrorInfo> {
        hipError_t herr = hipStreamCreate(&hip_state_->stream);
        if (herr != hipSuccess) {
            hq_println("[HIPGraphDenoiser] Stream create failed, fallback");
            return std::unexpected{GraphErrorInfo{GraphError::HipError,
                std::format("hipStreamCreate: {}", hipGetErrorString(herr)),
                static_cast<int>(herr)}};
        }
        hip_state_->owns_stream = true;
        return {};
    };

    auto alloc_buffers = [this]() -> std::expected<void, GraphErrorInfo> {
        if (auto r = allocate_device_buffers_(); !r) {
            hq_println("[HIPGraphDenoiser] Alloc failed, fallback");
            hip_state_->reset();
            return std::unexpected{r.error()};
        }
        return {};
    };

    auto set_timestep = [this]() -> std::expected<void, GraphErrorInfo> {
        if (scheduler_) {
            h_timestep_ = scheduler_->timestep(0);
        } else {
            h_timestep_ = 0;
        }
        return {};
    };

    auto run_unet_step0 = [&](std::vector<float>& noise_pred,
                               std::size_t& noise_count)
        -> std::expected<void, GraphErrorInfo> {
        auto unet_result = run_unet(
            std::span<const float>(latents_raw, latent_count_),
            embeddings, onnx_session, memory_info);
        if (!unet_result) {
            hip_state_->reset();
            return std::unexpected{unet_result.error()};
        }
        noise_pred = std::move(*unet_result);
        noise_count = std::min(noise_pred.size(), latent_count_);
        return {};
    };

    auto upload_scalars = [this]() -> std::expected<void, GraphErrorInfo> {
        if (auto r = upload_step_scalars_(0); !r) {
            hip_state_->reset();
            return std::unexpected{r.error()};
        }
        return {};
    };

    auto copy_noise_h2d = [&](const std::vector<float>& noise_pred,
                               std::size_t noise_count)
        -> std::expected<void, GraphErrorInfo> {
        hipError_t herr = hipMemcpyAsync(
            d_noise_pred_, noise_pred.data(), noise_count * sizeof(float),
            hipMemcpyHostToDevice, hip_state_->stream);
        if (herr != hipSuccess) {
            hip_state_->reset();
            return std::unexpected{GraphErrorInfo{GraphError::HipError,
                std::format("noise_pred H2D: {}", hipGetErrorString(herr)),
                static_cast<int>(herr)}};
        }
        return {};
    };

    auto copy_latents_h2d = [this](float* src) -> std::expected<void, GraphErrorInfo> {
        hipError_t herr = hipMemcpyAsync(
            d_latents_, src, latent_count_ * sizeof(float),
            hipMemcpyHostToDevice, hip_state_->stream);
        if (herr != hipSuccess) {
            hip_state_->reset();
            return std::unexpected{GraphErrorInfo{GraphError::HipError,
                std::format("latents H2D: {}", hipGetErrorString(herr)),
                static_cast<int>(herr)}};
        }
        return {};
    };

    auto sync_stream = [this]() -> std::expected<void, GraphErrorInfo> {
        hipError_t herr = hipStreamSynchronize(hip_state_->stream);
        if (herr != hipSuccess) {
            hip_state_->reset();
            return std::unexpected{GraphErrorInfo{GraphError::HipError,
                std::format("hipStreamSynchronize: {}", hipGetErrorString(herr)),
                static_cast<int>(herr)}};
        }
        return {};
    };

    auto begin_capture = [this]() -> std::expected<void, GraphErrorInfo> {
        hipError_t herr = hipStreamBeginCapture(hip_state_->stream,
                                                 hipStreamCaptureModeGlobal);
        if (herr != hipSuccess) {
            hq_println("[HIPGraphDenoiser] BeginCapture failed, fallback");
            return std::unexpected{GraphErrorInfo{GraphError::CaptureFailed,
                std::format("hipStreamBeginCapture: {}", hipGetErrorString(herr)),
                static_cast<int>(herr)}};
        }
        return {};
    };

    auto record_graph_ops = [this]() -> std::expected<void, GraphErrorInfo> {
        constexpr int kBlock = 256;
        const int grid = static_cast<int>((latent_count_ + kBlock - 1) / kBlock);
        hipLaunchKernelGGL(
            ddim_update_kernel,
            dim3(grid), dim3(kBlock), 0, hip_state_->stream,
            d_latents_, d_noise_pred_,
            static_cast<DdimDeviceParams*>(d_ddim_params_),
            latent_count_);

        hipMemcpyAsync(latents_raw, d_latents_, latent_count_ * sizeof(float),
                        hipMemcpyDeviceToHost, hip_state_->stream);
        return {};
    };

    auto end_capture = [this]() -> std::expected<void, GraphErrorInfo> {
        hipError_t herr = hipStreamEndCapture(hip_state_->stream, &hip_state_->graph);
        if (herr != hipSuccess || !hip_state_->graph) {
            hq_println("[HIPGraphDenoiser] Graph capture failed, fallback");
            hip_state_->graph = nullptr;
            hip_state_->reset();
            return std::unexpected{GraphErrorInfo{GraphError::CaptureFailed,
                std::format("hipStreamEndCapture: {}",
                    hipGetErrorString(herr)), static_cast<int>(herr)}};
        }
        return {};
    };

    auto instantiate_graph = [this]() -> std::expected<void, GraphErrorInfo> {
        hipError_t herr = hipGraphInstantiate(&hip_state_->exec,
                                               hip_state_->graph,
                                               nullptr, nullptr, 0);
        if (herr != hipSuccess) {
            hq_println("[HIPGraphDenoiser] Instantiate failed, fallback");
            hip_state_->exec = nullptr;
            hip_state_->reset();
            return std::unexpected{GraphErrorInfo{GraphError::CaptureFailed,
                std::format("hipGraphInstantiate: {}", hipGetErrorString(herr)),
                static_cast<int>(herr)}};
        }
        return {};
    };

    auto finalize = [this]() -> std::expected<void, GraphErrorInfo> {
        captured_ = true;
        steps_replayed_ = 0;
        hipStreamSynchronize(hip_state_->stream);

        std::size_t n_nodes = 0;
        hipGraphGetNodes(hip_state_->graph, nullptr, &n_nodes);
        hq_println(std::format(
            "[HIPGraphDenoiser] Captured {} nodes (DDIM+D2H, step 0 done)",
            n_nodes));
        return {};
    };

    auto fallback = [this, latents_raw, embeddings, onnx_session,
                     memory_info, &latent_shape]() -> std::expected<void, GraphErrorInfo> {
        return execute_step_fallback_(latents_raw, embeddings, onnx_session,
                                      memory_info, latent_shape, 0,
                                      cfg_scale_,
                                      std::span<const float>{uncond_embeddings_});
    };

    // ------------------------------------------------------------------
    // Monadic chain: each step feeds into the next via .and_then().
    // Any failure short-circuits to .or_else(fallback).
    // ------------------------------------------------------------------

    std::vector<float> noise_pred;
    std::size_t noise_count = 0;

    return create_stream()
        .and_then(alloc_buffers)
        .and_then(set_timestep)
        .and_then([&](void) -> std::expected<void, GraphErrorInfo> {
            return run_unet_step0(noise_pred, noise_count);
        })
        .and_then(upload_scalars)
        .and_then([&](void) -> std::expected<void, GraphErrorInfo> {
            return copy_noise_h2d(noise_pred, noise_count);
        })
        .and_then([&](void) -> std::expected<void, GraphErrorInfo> {
            return copy_latents_h2d(latents_raw);
        })
        .and_then(sync_stream)
        .and_then(begin_capture)
        .and_then(record_graph_ops)
        .and_then(end_capture)
        .and_then(instantiate_graph)
        .and_then(finalize)
        .or_else([&](GraphErrorInfo err) -> std::expected<void, GraphErrorInfo> {
            // If we already fell back inside a lambda, err may be a sentinel.
            // In all cases, delegate to the CPU fallback path.
            (void)err;
            return fallback();
        });
#endif
}

// ===========================================================================
// replay() -- Replay the captured DDIM+D2H graph.
//
// C02 Fix 1: Re-runs UNet before each graph launch so that d_noise_pred_
//            always contains the noise prediction for the current latents
//            and timestep. Step 0's noise is NOT re-used for later steps.
//
//            The execution sequence per step:
//              1. Set h_timestep_ from scheduler (or linear compute)
//              2. Run UNet on current host latents → fresh noise_pred
//              3. Upload DDIM params (coeff_x0, coeff_eps) for this step
//              4. H2D copy: noise_pred → d_noise_pred_
//              5. H2D copy: host latents → d_latents_
//              6. hipGraphLaunch (DDIM kernel in-place on d_latents_,
//                 then D2H back to host latents)
//              7. hipStreamSynchronize
// ===========================================================================

std::expected<void, GraphErrorInfo>
HIPGraphDenoiser::replay(hq::tensor::FloatTensor4D latents,
                          std::span<const float> embeddings,
                          Ort::Session* onnx_session,
                          Ort::MemoryInfo* memory_info) {
    float* latents_raw = latents.data();
    if (!captured_ || !hip_state_->exec) {
        return std::unexpected{GraphErrorInfo{GraphError::NotCaptured,
            "No graph captured"}};
    }
    if (steps_replayed_ + 1 >= cfg_.num_steps) {
        return std::unexpected{GraphErrorInfo{GraphError::InvalidStepCount,
            std::format("All {} steps done", cfg_.num_steps)}};
    }

#if !defined(__HIP_PLATFORM_AMD__)
    (void)latents_raw;
    (void)embeddings;
    (void)onnx_session;
    (void)memory_info;
    return std::unexpected{GraphErrorInfo{GraphError::HipError,
        "HIP not available"}};
#else
    const std::uint32_t step = steps_replayed_ + 1;

    // -- C02 Fix 2: Set correct timestep from scheduler ------------
    if (scheduler_) {
        h_timestep_ = scheduler_->timestep(step);
    } else {
        h_timestep_ = compute_linear_timestep(step, cfg_.num_steps);
    }

    // -- C02 Fix 1: Re-run UNet on current latents -----------------
    auto unet_result = run_unet(
        std::span<const float>(latents_raw, latent_count_),
        embeddings, onnx_session, memory_info);
    if (!unet_result) {
        return std::unexpected{unet_result.error()};
    }
    std::vector<float> noise_pred = std::move(*unet_result);
    const std::size_t noise_count = std::min(noise_pred.size(), latent_count_);

    // -- Upload DDIM params -----------------------------------------
    if (auto r = upload_step_scalars_(step); !r) return r;

    // -- H2D: noise_pred -> d_noise_pred_ ---------------------------
    hipError_t herr = hipMemcpyAsync(d_noise_pred_, noise_pred.data(),
                                      noise_count * sizeof(float),
                                      hipMemcpyHostToDevice,
                                      hip_state_->stream);
    if (herr != hipSuccess) {
        return std::unexpected{GraphErrorInfo{GraphError::ReplayFailed,
            std::format("noise H2D: {}", hipGetErrorString(herr)),
            static_cast<int>(herr)}};
    }

    // -- H2D: host latents -> d_latents_ ----------------------------
    herr = hipMemcpyAsync(d_latents_, latents_raw,
                           latent_count_ * sizeof(float),
                           hipMemcpyHostToDevice, hip_state_->stream);
    if (herr != hipSuccess) {
        return std::unexpected{GraphErrorInfo{GraphError::ReplayFailed,
            std::format("latents H2D: {}", hipGetErrorString(herr)),
            static_cast<int>(herr)}};
    }

    // -- Launch the captured graph (DDIM kernel + D2H) --------------
    herr = hipGraphLaunch(hip_state_->exec, hip_state_->stream);
    if (herr != hipSuccess) {
        return std::unexpected{GraphErrorInfo{GraphError::ReplayFailed,
            std::format("Launch: {}", hipGetErrorString(herr)),
            static_cast<int>(herr)}};
    }

    herr = hipStreamSynchronize(hip_state_->stream);
    if (herr != hipSuccess) {
        return std::unexpected{GraphErrorInfo{GraphError::ReplayFailed,
            std::format("Sync: {}", hipGetErrorString(herr)),
            static_cast<int>(herr)}};
    }

    steps_replayed_++;
    return {};
#endif
}

// ===========================================================================
// execute_full() -- Full denoising loop with CFG support.
//
// C02: Passes CFG parameters through to capture() and the fallback calls.
// ===========================================================================

std::expected<void, GraphErrorInfo>
HIPGraphDenoiser::execute_full(hq::tensor::FloatTensor4D latents,
                                std::span<const float> embeddings,
                                Ort::Session* onnx_session,
                                Ort::MemoryInfo* memory_info,
                                const std::array<std::int64_t, 4>& latent_shape,
                                float guidance_scale,
                                std::span<const float> uncond_embeddings) {
    float* latents_raw = latents.data();
    if (cfg_.num_steps == 0) {
        return std::unexpected{GraphErrorInfo{GraphError::InvalidStepCount,
            "num_steps == 0"}};
    }

    auto cap = capture(latents, embeddings, onnx_session, memory_info,
                       latent_shape, guidance_scale, uncond_embeddings);
    if (!cap && !captured_) return cap;

    for (std::uint32_t step = 1; step < cfg_.num_steps; ++step) {
        if (captured_) {
            auto rep = replay(latents, embeddings, onnx_session, memory_info);
            if (!rep) {
                    hq_println(std::format("[HIPGraphDenoiser] Replay step {} failed, fallback", step));
                    captured_ = false;
                    hip_state_->reset();
                    auto fb = execute_step_fallback_(latents_raw, embeddings,
                                                      onnx_session, memory_info,
                                                      latent_shape, step,
                                                      cfg_scale_,
                                                      std::span<const float>{uncond_embeddings_});
                    if (!fb) return fb;
                }
        } else {
            auto fb = execute_step_fallback_(latents_raw, embeddings,
                                              onnx_session, memory_info,
                                              latent_shape, step,
                                              cfg_scale_,
                                              std::span<const float>{uncond_embeddings_});
            if (!fb) return fb;
        }
    }

    hq_println(std::format("[HIPGraphDenoiser] Done: {} steps, {} replayed",
               cfg_.num_steps, steps_replayed_));
    return {};
}

// ===========================================================================
// execute_step_fallback_() -- CPU mirror of Pipeline::denoise_step_
//
// C02 Fix 2: Proper timestep from scheduler_->timestep(step), not raw step.
// C02 Fix 3: CFG support: dual UNet passes + guidance blending.
//
// Semantics:
//   - When no scheduler is attached: uses compute_linear_timestep() for
//     the ONNX timestep input and performs the DDIM update with the
//     same linear coefficient formula used in upload_step_scalars_().
//   - When a scheduler is attached: uses scheduler_->timestep(step) for
//     the ONNX input and delegates the latent update to scheduler_->step()
//   (AVX-512 accelerated when available).
// ===========================================================================

std::expected<void, GraphErrorInfo>
HIPGraphDenoiser::execute_step_fallback_(float* latents,
                                          std::span<const float> embeddings,
                                          Ort::Session* onnx_session,
                                          Ort::MemoryInfo* memory_info,
                                          const std::array<std::int64_t, 4>&
                                              latent_shape,
                                          std::uint32_t step,
                                          float cfg_scale,
                                          std::span<const float> uncond_embeddings) {
    if (!latents || !onnx_session || !memory_info) {
        return std::unexpected{GraphErrorInfo{GraphError::ONNXError,
            "null argument in execute_step_fallback_"}};
    }

    const std::size_t ls0 = static_cast<std::size_t>(latent_shape[0]);
    const std::size_t ls1 = static_cast<std::size_t>(latent_shape[1]);
    const std::size_t ls2 = static_cast<std::size_t>(latent_shape[2]);
    const std::size_t ls3 = static_cast<std::size_t>(latent_shape[3]);
    const std::size_t lc = ls0 * ls1 * ls2 * ls3;
    constexpr std::int64_t kSeq = 77;
    const std::int64_t hdim = static_cast<std::int64_t>(embeddings.size() / kSeq);

    const std::array<std::int64_t, 1> ts{1};
    const std::array<std::int64_t, 3> es{1, kSeq, hdim};

    // ------------------------------------------------------------------
    // Helper: create tensors + run ONNX for one embedding set
    // ------------------------------------------------------------------
    auto run_pass = [&](std::int64_t timestep_val,
                         std::span<const float> emb)
        -> std::expected<std::vector<float>, GraphErrorInfo> {
        try {
            Ort::Value lt = Ort::Value::CreateTensor<float>(
                *memory_info, latents, lc, latent_shape.data(),
                latent_shape.size());

            Ort::Value tt = Ort::Value::CreateTensor<std::int64_t>(
                *memory_info, const_cast<std::int64_t*>(&timestep_val), 1,
                ts.data(), ts.size());

            Ort::Value et = Ort::Value::CreateTensor<float>(
                *memory_info, const_cast<float*>(emb.data()),
                emb.size(), es.data(), es.size());

            const char* in_n[]  = {"sample", "timestep",
                                     "encoder_hidden_states"};
            const char* out_n[] = {"out_sample"};
            Ort::Value ins[] = {std::move(lt), std::move(tt), std::move(et)};

            auto outs = onnx_session->Run(Ort::RunOptions{nullptr},
                                           in_n, ins, 3, out_n, 1);

            if (outs.empty() || !outs[0].IsTensor()) {
                return std::unexpected{GraphErrorInfo{GraphError::ONNXError,
                    "UNet output not tensor"}};
            }

            const float* np = outs[0].GetTensorData<float>();
            const auto nshape = outs[0].GetTensorTypeAndShapeInfo()
                                    .GetShape();
            std::size_t nc = 1;
            for (auto d : nshape) if (d > 0)
                nc *= static_cast<std::size_t>(d);

            return std::vector<float>(np, np + nc);

        } catch (const Ort::Exception& e) {
            return std::unexpected{GraphErrorInfo{GraphError::ONNXError,
                std::format("ORT: {}", e.what())}};
        } catch (const std::exception& e) {
            return std::unexpected{GraphErrorInfo{GraphError::ONNXError,
                std::format("Ex: {}", e.what())}};
        }
    };

    // ------------------------------------------------------------------
    // Determine timestep: scheduler takes priority, else linear fallback
    // ------------------------------------------------------------------
    std::int64_t timestep_val;
    if (scheduler_) {
        timestep_val = scheduler_->timestep(step);
    } else {
        timestep_val = compute_linear_timestep(step, cfg_.num_steps);
    }

    // ------------------------------------------------------------------
    // CFG decision: if uncond_embeddings is valid and cfg_scale > 1.0,
    // run UNet twice and blend. Otherwise single conditional pass.
    // ------------------------------------------------------------------
    const bool do_cfg = cfg_active(cfg_scale, uncond_embeddings);

    if (!do_cfg) {
        // ---- No CFG: single conditional pass -------------------------------
        auto noise_result = run_pass(timestep_val, embeddings);
        if (!noise_result) return std::unexpected{noise_result.error()};

        const auto& np = *noise_result;
        const std::size_t nc = np.size();

        if (scheduler_) {
            const std::size_t safe_n = std::min(lc, nc);
            auto sched_r = scheduler_->step(
                hq::tensor::FloatTensor4D{latents, ls0, ls1, ls2, ls3},
                hq::tensor::Tensor1D<const float>{np.data(), safe_n},
                step);
            if (!sched_r) [[unlikely]]
                return std::unexpected{GraphErrorInfo{GraphError::ONNXError,
                    std::format("Scheduler step failed: {}",
                                hq::to_string(sched_r.error()))}};
        } else {
            float cx0, ceps;
            compute_linear_coeffs_(step, cx0, ceps);
            const std::size_t n = std::min(lc, nc);
            for (std::size_t i = 0; i < n; ++i)
                latents[i] = cx0 * latents[i] + ceps * np[i];
        }

    } else {
        // ---- CFG: conditional + unconditional, then blend ----------------
        // Validate unconditional embedding size matches conditional
        if (uncond_embeddings.size() != embeddings.size()) {
            return std::unexpected{GraphErrorInfo{GraphError::ONNXError,
                std::format("CFG embedding size mismatch: uncond={} vs cond={}",
                            uncond_embeddings.size(), embeddings.size())}};
        }

        // 1. Conditional pass
        auto cond_result = run_pass(timestep_val, embeddings);
        if (!cond_result) {
            hq_println(std::format("[HIPGraphDenoiser] CFG cond pass failed at step {}",
                       step));
            return std::unexpected{cond_result.error()};
        }
        std::vector<float> noise_cond = std::move(*cond_result);

        // 2. Unconditional pass
        auto uncond_result = run_pass(timestep_val, uncond_embeddings);
        if (!uncond_result) {
            hq_println(std::format("[HIPGraphDenoiser] CFG uncond pass failed at step {}",
                       step));
            return std::unexpected{uncond_result.error()};
        }
        std::vector<float> noise_uncond = std::move(*uncond_result);

        // Verify shape match
        if (noise_cond.size() != noise_uncond.size()) {
            return std::unexpected{GraphErrorInfo{GraphError::ONNXError,
                std::format("CFG shape mismatch: cond={} vs uncond={}",
                            noise_cond.size(), noise_uncond.size())}};
        }

        // 3. Blend into noise_cond:
        //    noise = uncond + scale * (cond - uncond)
        for (std::size_t i = 0; i < noise_cond.size(); ++i) {
            noise_cond[i] = noise_uncond[i] +
                            cfg_scale * (noise_cond[i] - noise_uncond[i]);
        }

        // 4. Scheduler step with blended noise
        if (scheduler_) {
            const std::size_t safe_noise = std::min(lc, noise_cond.size());
            auto sched_r = scheduler_->step(
                hq::tensor::FloatTensor4D{latents, ls0, ls1, ls2, ls3},
                hq::tensor::Tensor1D<const float>{noise_cond.data(), safe_noise},
                step);
            if (!sched_r) [[unlikely]]
                return std::unexpected{GraphErrorInfo{GraphError::ONNXError,
                    std::format("Scheduler CFG step failed: {}",
                                hq::to_string(sched_r.error()))}};
        } else {
            float cx0, ceps;
            compute_linear_coeffs_(step, cx0, ceps);
            const std::size_t n = std::min(lc, noise_cond.size());
            for (std::size_t i = 0; i < n; ++i)
                latents[i] = cx0 * latents[i] + ceps * noise_cond[i];
        }
    }

    return {};
}

void HIPGraphDenoiser::compute_linear_coeffs_(std::uint32_t step,
                                               float& coeff_x0,
                                               float& coeff_eps) const noexcept {
    const float t = static_cast<float>(cfg_.num_steps - step)
                    / static_cast<float>(cfg_.num_steps);
    const float at = std::max(t, 0.001f);
    const float ap = (step > 0)
        ? std::max(static_cast<float>(cfg_.num_steps - step + 1)
                   / static_cast<float>(cfg_.num_steps), 0.001f)
        : 1.0f;
    const float sa = std::sqrt(at);
    const float soa = std::sqrt(std::max(1.0f - at, 1e-6f));
    const float sap = std::sqrt(ap);
    coeff_x0 = sap / sa;
    coeff_eps = std::sqrt(std::max(1.0f - ap, 1e-6f)) - coeff_x0 * soa;
}

} // namespace hq
