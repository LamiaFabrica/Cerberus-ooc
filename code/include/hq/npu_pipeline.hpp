#pragma once
/// @file npu_pipeline.hpp
/// @copyright Copyright (c) 2026 D Hargreaves (AKA Roylepython). LamiaFabrica. All rights reserved.
///
/// TRADE SECRET — Proprietary and Confidential.
/// This file contains trade secrets of LamiaFabrica / D Hargreaves.
/// Unauthorized use, reproduction, or disclosure is strictly prohibited.
/// See LICENSE for terms.
///
/// NPU-optimised memory pipeline for Hailo-8L ↔ GPU 780M tensor flow.
///
/// The bottleneck in heterogeneous AI pipelines is the memory interface loop:
///   CPU prompt → NPU text encode → NPU output → staging → GPU UNet input
///
/// This header provides the primitives to eliminate that bottleneck:
///
///   1. Page-locked (pinned) host memory for direct DMA
///   2. Double-buffered async encode submissions (overlap encode + denoise)
///   3. HIP-stream-aware NPU↔GPU tensor handoff (zero intermediate copy)
///   4. C++26 coroutine-ready awaitables for GPU/DMA completion
///   5. NPU-aware utilisation feedback loop
///
/// @version 1.0.0
/// @target MinisForum UM790 Pro (7940HS + Radeon 780M + Hailo-8L)

#include "hq/cxx26_features.hpp"
#include "hq/tensor_view.hpp"
#include "hq/hailo_monitor.hpp"
#include "hq/gpu_monitor.hpp"

// Forward-declare encoder interface to avoid circular includes.
namespace hq::npu { class INpuEncoder; }

#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#if defined(_WIN32)
#  include <malloc.h>
#endif
#include <concepts>
#include <condition_variable>
#if UM790_HAS_COROUTINES
#  include <coroutine>
#else
#  error "npu_pipeline.hpp requires C++26 coroutines (<coroutine>) — GCC >= 14 or Clang >= 18 with C++26 enabled"
#endif
#include <cstddef>
#include <cstdint>
#include <cstring>
#if UM790_HAS_STD_EXPECTED
#  include <expected>
#else
#  error "npu_pipeline.hpp requires std::expected (<expected>) — GCC >= 14 or Clang >= 18 with C++26 enabled"
#endif
#if UM790_HAS_STD_FORMAT
#  include <format>
#endif
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#if UM790_HAS_STD_PRINT
#  include <print>
#endif
#include <span>
#include <stop_token>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#ifdef UM790_HAS_HIP
#include <hip/hip_runtime_api.h>
#endif

#include <onnxruntime_cxx_api.h>

namespace hq::npu {

// ===========================================================================
// Constants tuned for Hailo-8L on UM790 Pro
// ===========================================================================

/// Hailo-8L PCIe 3.0 x4 theoretical bandwidth: ~3.94 GB/s
inline constexpr float HAILO_PCIE_BANDWIDTH_GBS = 3.94f;

/// Typical CLIP text encoder output: 77 * 768 * 4 = 236,544 bytes ≈ 231 KiB
/// This is small — the bottleneck is submission latency, not throughput.
inline constexpr std::size_t CLIP_EMBEDDING_BYTES  = 77 * 768 * 4;

/// Maximum embeddings we expect (SDXL: 77 * 2048 * 4 = 630,784 bytes)
inline constexpr std::size_t MAX_EMBEDDING_BYTES   = 77 * 2048 * 4;

/// DMA transfer time at PCIe 3.0 x4 for CLIP embeddings: ~59 µs
/// So the real bottleneck is kernel launch + PCIe transaction overhead (~10-50 µs)
inline constexpr float CLIP_DMA_LATENCY_US = 60.0f;

// ===========================================================================
// PinnedTensor: page-locked host tensor for zero-copy NPU↔GPU DMA
// ===========================================================================

/// @brief A page-locked (pinned) host memory region for DMA operations.
///
/// Unlike std::vector, this is allocated via hipHostMalloc (pinned/page-locked)
/// guaranteeing that the GPU/NPU DMA engine can access it without CPU
/// intervention. This eliminates the intermediate bounce-buffer copy that
/// plagues heterogeneous pipelines.
///
/// @tparam T Element type.
template<typename T>
class PinnedTensor {
public:
    using value_type      = T;
    using size_type       = std::size_t;
    using pointer         = T*;
    using const_pointer   = const T*;
    using span_type       = std::span<T>;
    using const_span_type = std::span<const T>;

    PinnedTensor() = default;

    /// Allocate size elements of pinned memory.
    explicit PinnedTensor(std::size_t count)
        : count_{count}
    {
        if (count == 0) return;
#if defined(UM790_HAS_HIP)
        hipError_t err = hipHostMalloc(&data_, count * sizeof(T),
                                        hipHostMallocPortable);
        if (err != hipSuccess) {
            std::print("[PinnedTensor] hipHostMalloc({} B) failed: {}, "
                       "falling back to malloc\n",
                       count * sizeof(T), hipGetErrorString(err));
#  if defined(_WIN32)
            data_ = static_cast<T*>(_aligned_malloc(count * sizeof(T), 256));
#  else
            data_ = static_cast<T*>(::aligned_alloc(256, count * sizeof(T)));
#  endif
            owns_ = true;
            pinned_ = false;
        } else {
            owns_ = true;
            pinned_ = true;
        }
#elif defined(_WIN32)
        data_ = static_cast<T*>(_aligned_malloc(count * sizeof(T), 256));
        owns_ = true;
        pinned_ = false;
#else
        data_ = static_cast<T*>(::aligned_alloc(256, count * sizeof(T)));
        owns_ = true;
        pinned_ = false;
#endif
    }

    ~PinnedTensor() noexcept { release_(); }

    PinnedTensor(const PinnedTensor&) = delete;
    PinnedTensor& operator=(const PinnedTensor&) = delete;

    PinnedTensor(PinnedTensor&& other) noexcept
        : data_{other.data_}, count_{other.count_},
          owns_{other.owns_}, pinned_{other.pinned_}
    { other.data_ = nullptr; other.count_ = 0; other.owns_ = false; }

    PinnedTensor& operator=(PinnedTensor&& other) noexcept {
        if (this != &other) {
            release_();
            data_ = other.data_; count_ = other.count_;
            owns_ = other.owns_; pinned_ = other.pinned_;
            other.data_ = nullptr; other.count_ = 0; other.owns_ = false;
        }
        return *this;
    }

    [[nodiscard]] pointer       data()       noexcept { return data_; }
    [[nodiscard]] const_pointer data() const noexcept { return data_; }
    [[nodiscard]] std::size_t   size()  const noexcept { return count_; }
    [[nodiscard]] std::size_t   bytes() const noexcept { return count_ * sizeof(T); }
    [[nodiscard]] bool          empty() const noexcept { return count_ == 0; }

    /// True if this memory is page-locked (GPU DMA-capable without bounce).
    [[nodiscard]] bool is_pinned() const noexcept { return pinned_; }

    [[nodiscard]] span_type       span()       noexcept { return {data_, count_}; }
    [[nodiscard]] const_span_type span() const noexcept { return {data_, count_}; }

    operator span_type()       noexcept { return span(); }
    operator const_span_type() const noexcept { return span(); }

    T& operator[](std::size_t i)             noexcept { return data_[i]; }
    const T& operator[](std::size_t i) const noexcept { return data_[i]; }

    pointer       begin()       noexcept { return data_; }
    pointer       end()         noexcept { return data_ + count_; }
    const_pointer begin() const noexcept { return data_; }
    const_pointer end()   const noexcept { return data_ + count_; }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
    bool owns_   = false;
    bool pinned_ = false;

    void release_() noexcept {
        if (!owns_ || !data_) return;
#if defined(UM790_HAS_HIP)
        if (pinned_) {
            hipHostFree(data_);
        } else {
#  if defined(_WIN32)
            _aligned_free(data_);
#  else
            std::free(data_);
#  endif
        }
#elif defined(_WIN32)
        _aligned_free(data_);
#else
        std::free(data_);
#endif
        data_ = nullptr;
        count_ = 0;
        owns_ = false;
    }
};

// ===========================================================================
// NpuDmaSlot: one slot in the double-buffer NPU→GPU DMA pipeline
// ===========================================================================

/// @brief One slot in the NPU double-buffer pipeline.
///
/// Each slot holds a pinned host buffer (for NPU output) and a corresponding
/// GPU device buffer. The lifecycle:
///
///   IDLE → ENCODING (NPU running) → STAGING (H2D DMA) → READY (GPU can read)
///
/// C++26 std::atomic for the state flag allows lock-free polling by the GPU
/// consumer while the NPU producer writes.
struct NpuDmaSlot {
    enum class State : std::uint8_t {
        IDLE,         ///< Available for encoding
        ENCODING,     ///< NPU is writing to host buffer
        STAGING,      ///< DMA H2D in flight
        READY,        ///< GPU buffer ready
        ERROR         ///< Something failed
    };

    std::atomic<State> state{State::IDLE};

    PinnedTensor<float> host_buffer;     ///< NPU writes here (pinned for DMA)
    struct NpuDmaSlotDevice {
#if defined(UM790_HAS_HIP)
        float*        device_ptr{nullptr};
        hipEvent_t    dma_done{nullptr};
        hipStream_t   dma_stream{nullptr};
#else
        float*        device_ptr{nullptr};
        void*         dma_done{nullptr};
        void*         dma_stream{nullptr};
#endif
    } dev;
    std::size_t         slot_index{0};
};

// ===========================================================================
// NpuEncodeRequest: what the NPU pipeline processes
// ===========================================================================

struct NpuEncodeRequest {
    std::string prompt;
    float       guidance_scale{7.5f};
    std::int64_t seed{42};
    std::uint32_t width{512};
    std::uint32_t height{512};
    std::uint32_t num_steps{20};
    std::size_t   max_seq_len{77};
};

// ===========================================================================
// NpuEncodeResult: NPU output with metadata
// ===========================================================================

struct NpuEncodeResult {
    PinnedTensor<float> embeddings;
    PinnedTensor<float> uncond_embeddings;
    std::size_t         embedding_count{0};
    std::size_t         hidden_dim{768};
    bool                cfg_enabled{false};
    // Sentinel values for npu_utilization and npu_temperature:
    //   >= 0.0f : Real measured value from NPU hardware
    //   -1.0f   : No NPU hardware present (CPU fallback encoder)
    //   -2.0f   : Synthetic/fabricated value (reserved)
    float               npu_utilization{-1.0f};
    float               npu_temperature{-1.0f};
    float               encode_time_us{0.0f};
    std::size_t         slot_index{0};
};

// ===========================================================================
// NpuTensorHandle: a lightweight handle to a GPU tensor with NPU provenance
// ===========================================================================

/// @brief Zero-copy handle for a tensor that originated from the NPU
/// and now resides in GPU memory ready for UNet consumption.
struct NpuTensorHandle {
    float*              device_ptr{nullptr};
    std::size_t         element_count{0};
    std::array<std::int64_t, 3> shape{1, 77, 768}; ///< [batch, seq, hidden]
    float               npu_util{-1.0f};
    bool                valid{false};

    [[nodiscard]] explicit operator bool() const noexcept {
        return valid && device_ptr != nullptr;
    }
};

// ===========================================================================
// NpuDmaPipeline: the core NPU ↔ GPU memory interface
// ===========================================================================

/// @brief Double-buffered NPU → GPU DMA pipeline.
///
/// Eliminates the serial CPU→NPU→CPU→GPU bottleneck by:
///
///   1. Submitting NPU encode async (non-blocking via submit_async)
///   2. Automatically staging NPU output → GPU via pinned DMA
///   3. Signalling GPU readiness via atomic state flags
///   4. Enabling overlap: while GPU denoises step N, NPU encodes next batch
///
/// Two submission paths are offered:
///   - submit_async(): true async — spawns a thread, returns immediately.
///   - submit(): synchronous fallback — blocks the caller until encode
///     and DMA staging complete. Use only when caller thread can block.
///
/// This directly addresses the "memory/NPU interface loop" that is the
/// primary bottleneck on heterogeneous consumer hardware.
class NpuDmaPipeline {
public:
    /// Configuration for the NPU DMA pipeline.
    struct Config {
        std::size_t num_slots{3};             ///< Triple-buffer for deep pipelining
        std::size_t embedding_bytes{hq::npu::MAX_EMBEDDING_BYTES};
        bool        enable_gpu_staging{true};  ///< Auto-DMA to GPU
        bool        enable_npu_util_tracking{true};
    };

    explicit NpuDmaPipeline(const Config& cfg,
                             INpuEncoder* encoder = nullptr);
    ~NpuDmaPipeline() noexcept;

    NpuDmaPipeline(const NpuDmaPipeline&) = delete;
    NpuDmaPipeline& operator=(const NpuDmaPipeline&) = delete;
    NpuDmaPipeline(NpuDmaPipeline&&) noexcept;
    NpuDmaPipeline& operator=(NpuDmaPipeline&&) noexcept;

    /// @brief Inject an encoder implementation.
    /// If not called (or nullptr passed), the pipeline will fail encode
    /// requests until a real encoder (Hailo8lEncoder or CpuFallbackEncoder) is set.
    void set_encoder(INpuEncoder* encoder);

    // ---- NPU encode submission ---------------------------------------------

    /// Submit a prompt for async NPU encoding. Returns immediately with
    /// the slot index. The caller polls `slot_state(slot)` or awaits
    /// `wait_gpu_ready(slot)`.
    ///
    /// This is the synchronous fallback path — blocks caller until encode
    /// and DMA staging are fully complete.
    [[nodiscard]] std::expected<std::size_t, std::string>
        submit(const NpuEncodeRequest& req);

    /// True async submission: acquires a slot (blocking only if the
    /// pipeline is saturated), spawns a std::jthread to run
    /// simulate_encode() in the background, then returns immediately
    /// with the slot index. The background thread sets the slot state
    /// to ENCODING immediately and READY when staging completes.
    ///
    /// In production, replace simulate_encode() with HailoRT
    /// async_infer() submission to the hardware scheduler.
    [[nodiscard]] std::expected<std::size_t, std::string>
        submit_async(const NpuEncodeRequest& req);

    /// Check the state of a slot.
    [[nodiscard]] NpuDmaSlot::State slot_state(std::size_t slot) const;

    /// Block until the GPU buffer for the given slot is ready.
    [[nodiscard]] std::expected<NpuTensorHandle, std::string>
        wait_gpu_ready(std::size_t slot,
                        std::chrono::milliseconds timeout = std::chrono::milliseconds{5000});

    /// Non-blocking try: if GPU buffer is ready, return the handle.
    [[nodiscard]] std::optional<NpuTensorHandle> try_get_gpu_handle(
        std::size_t slot);

    /// Release a slot back to IDLE so the NPU can reuse it.
    void release_slot(std::size_t slot);

    // ---- NPU utilisation feedback -----------------------------------------

    /// Return the last known NPU utilisation (from HailoMonitor).
    [[nodiscard]] float last_npu_utilization() const noexcept;

    /// Return the average NPU utilisation across the pipeline session.
    [[nodiscard]] float avg_npu_utilization() const noexcept;

    // ---- Pipeline stats ---------------------------------------------------

    struct PipelineStats {
        std::uint64_t encodes_submitted{0};
        std::uint64_t encodes_completed{0};
        std::uint64_t dma_transfers{0};
        std::uint64_t dma_bytes{0};
        double        avg_encode_time_us{0.0};
        double        avg_dma_time_us{0.0};
        double        avg_npu_utilization{0.0};
        std::size_t   active_slots{0};
    };
    [[nodiscard]] PipelineStats get_stats() const;

    // ---- Thermal / health -------------------------------------------------

    [[nodiscard]] float npu_temperature() const noexcept;

private:
    Config config_;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ===========================================================================
// EncodeResult: unified NPU encode outcome (conditional + unconditional)
// ===========================================================================

struct EncodeResult {
    NpuTensorHandle cond;       ///< Conditional embeddings on GPU
    NpuTensorHandle uncond;     ///< Unconditional (empty-prompt) on GPU
    bool            cfg_active{false};
    float           npu_utilization{-1.0f};
    float           npu_temp{-1.0f};
    float           total_time_us{0.0f};
};

} // namespace hq::npu
