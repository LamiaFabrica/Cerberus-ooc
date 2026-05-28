/// @file npu_pipeline.cpp
/// @copyright Copyright © 2026, all rights reserved | David Hargreaves aka Roylepython trading as The Medusa Initiative & Lamia Fabrica
/// NpuDmaPipeline implementation — NPU ↔ GPU memory interface.
///
/// Directly addresses the CPU→NPU→GPU memory bottleneck:
///   - Pinned host memory eliminates bounce buffers
///   - Double/triple-buffer enables NPU+GPU overlap
///   - Atomic state flags for lock-free producer-consumer
///   - HIP-stream-aware DMA submission
///
/// @version 1.0.0

#include "hq/npu_pipeline.hpp"
#include "hq/npu_encoder.hpp"

#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <format>
#include <mutex>
#if UM790_HAS_STD_PRINT
#  include <print>
#endif
#include <queue>
#include <stop_token>
#include <thread>

#ifdef UM790_HAS_HIP
#include <hip/hip_runtime_api.h>
#endif

namespace hq::npu {

// ===========================================================================
// Impl: internal state
// ===========================================================================

struct NpuDmaPipeline::Impl {
    Config cfg;
    std::vector<std::unique_ptr<NpuDmaSlot>> slots;
    INpuEncoder*           encoder{nullptr};
    std::unique_ptr<INpuEncoder> owned_encoder;

    // NPU telemetry
    float last_npu_util{-1.0f};  // No NPU measurement yet
    float last_npu_temp{0.0f};
    double accum_npu_util{0.0};
    std::uint64_t npu_samples{0};

    // Pipeline accounting
    std::uint64_t encodes_submitted{0};
    std::uint64_t encodes_completed{0};
    std::uint64_t dma_transfers{0};
    std::uint64_t dma_bytes{0};
    double accum_encode_time_us{0.0};
    double accum_dma_time_us{0.0};

    // Resource management
    std::queue<std::size_t> free_slots;
    mutable std::mutex queue_mutex;
    std::condition_variable queue_cv;

    explicit Impl(Config c, INpuEncoder* enc)
        : cfg{std::move(c)}
        , encoder{enc}
    {
        const std::size_t ns = cfg.num_slots > 0 ? cfg.num_slots : 3;
        slots.reserve(ns);

        for (std::size_t i = 0; i < ns; ++i) {
            auto slot = std::make_unique<NpuDmaSlot>();
            slot->slot_index = i;
            slot->host_buffer = PinnedTensor<float>{
                cfg.embedding_bytes / sizeof(float)};
            slot->state.store(NpuDmaSlot::State::IDLE, std::memory_order_release);

#if defined(UM790_HAS_HIP)
            if (cfg.enable_gpu_staging) {
                hipError_t err;

                err = hipMalloc(&slot->dev.device_ptr,
                                cfg.embedding_bytes);
                if (err != hipSuccess) {
                    std::print("[NpuDmaPipeline] slot {} hipMalloc failed: {}\n",
                               i, hipGetErrorString(err));
                    slot->dev.device_ptr = nullptr;
                }

                err = hipStreamCreate(&slot->dev.dma_stream);
                if (err != hipSuccess) {
                    std::print("[NpuDmaPipeline] slot {} hipStreamCreate failed: {}\n",
                               i, hipGetErrorString(err));
                    slot->dev.dma_stream = nullptr;
                }

                err = hipEventCreate(&slot->dev.dma_done);
                if (err != hipSuccess) {
                    std::print("[NpuDmaPipeline] slot {} hipEventCreate failed: {}\n",
                               i, hipGetErrorString(err));
                    slot->dev.dma_done = nullptr;
                }
            }
#endif
            slots.push_back(std::move(slot));
            free_slots.push(i);
        }

        if (!encoder) {
            std::print("[NpuDmaPipeline] WARNING: No encoder provided. "
                       "Pipeline will fail encode requests until an encoder is set.\n");
        } else {
            std::print("[NpuDmaPipeline] Active encoder: {}\n", encoder->name());
        }
    }

#if defined(UM790_HAS_HIP)
    ~Impl() {
        for (auto& slot : slots) {
            if (slot->dev.dma_done)   hipEventDestroy(slot->dev.dma_done);
            if (slot->dev.dma_stream) hipStreamDestroy(slot->dev.dma_stream);
            if (slot->dev.device_ptr) hipFree(slot->dev.device_ptr);
            slot->dev.dma_done   = nullptr;
            slot->dev.dma_stream = nullptr;
            slot->dev.device_ptr = nullptr;
        }
    }
#else
    ~Impl() = default;
#endif

    std::size_t acquire_slot() {
        std::unique_lock lock{queue_mutex};
        queue_cv.wait(lock, [this] { return !free_slots.empty(); });
        std::size_t idx = free_slots.front();
        free_slots.pop();
        return idx;
    }

    void return_slot(std::size_t idx) {
        if (idx >= slots.size()) return;
        slots[idx]->state.store(NpuDmaSlot::State::IDLE, std::memory_order_release);
        {
            std::lock_guard lock{queue_mutex};
            free_slots.push(idx);
        }
        queue_cv.notify_one();
    }

    void stage_to_gpu([[maybe_unused]] std::size_t slot_idx) {
#if defined(UM790_HAS_HIP)
        auto& slot = slots[slot_idx];
        if (!slot->dev.device_ptr || !slot->dev.dma_stream ||
            slot->host_buffer.empty()) return;

        slot->state.store(NpuDmaSlot::State::STAGING, std::memory_order_release);

        auto t0 = std::chrono::high_resolution_clock::now();

        hipError_t err = hipMemcpyAsync(
            slot->dev.device_ptr,
            slot->host_buffer.data(),
            slot->host_buffer.bytes(),
            hipMemcpyHostToDevice,
            slot->dev.dma_stream);

        if (err != hipSuccess) {
            slot->state.store(NpuDmaSlot::State::ERROR, std::memory_order_release);
            std::print("[NpuDmaPipeline] DMA H2D slot {} failed: {}\n",
                       slot_idx, hipGetErrorString(err));
            return;
        }

        err = hipEventRecord(slot->dev.dma_done, slot->dev.dma_stream);
        if (err != hipSuccess) {
            slot->state.store(NpuDmaSlot::State::ERROR, std::memory_order_release);
            std::print("[NpuDmaPipeline] DMA event record slot {} failed: {}\n",
                       slot_idx, hipGetErrorString(err));
            return;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        accum_dma_time_us += static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
        dma_transfers++;
        dma_bytes += slot->host_buffer.bytes();
#endif
    }

    void run_encode(std::size_t slot_idx, const NpuEncodeRequest& req) {
        auto& slot = slots[slot_idx];
        slot->state.store(NpuDmaSlot::State::ENCODING, std::memory_order_release);

        auto t0 = std::chrono::high_resolution_clock::now();

        auto encoded = encoder->encode(req);
        if (!encoded) {
            slot->state.store(NpuDmaSlot::State::ERROR, std::memory_order_release);
            std::print("[NpuDmaPipeline] Encode failed slot {}: {}\n",
                       slot_idx, encoded.error());
            return;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        accum_encode_time_us += static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());

        last_npu_util  = encoded->npu_utilization;
        last_npu_temp  = encoded->npu_temperature;
        accum_npu_util += static_cast<double>(last_npu_util);
        npu_samples++;
        encodes_completed++;

        if (!slot->host_buffer.empty() && !encoded->embeddings.empty()) {
            std::size_t n = std::min(slot->host_buffer.size(),
                                     encoded->embeddings.size());
            std::copy_n(encoded->embeddings.data(), n,
                        slot->host_buffer.data());
        }
    }
};

// ===========================================================================
// NpuDmaPipeline — public API
// ===========================================================================

NpuDmaPipeline::NpuDmaPipeline(const Config& cfg,
                                 INpuEncoder* encoder)
    : config_{cfg}
    , impl_{std::make_unique<Impl>(cfg, encoder)}
{
    std::print("[NpuDmaPipeline] Initialised: {} slots, {} bytes/slot\n",
               config_.num_slots, config_.embedding_bytes);
}

NpuDmaPipeline::~NpuDmaPipeline() noexcept = default;
NpuDmaPipeline::NpuDmaPipeline(NpuDmaPipeline&&) noexcept = default;
NpuDmaPipeline& NpuDmaPipeline::operator=(NpuDmaPipeline&&) noexcept = default;

void NpuDmaPipeline::set_encoder(INpuEncoder* encoder) {
    impl_->encoder = encoder;
    std::print("[NpuDmaPipeline] Encoder set to: {}\n",
               encoder ? encoder->name() : "nullptr");
}

std::expected<std::size_t, std::string>
NpuDmaPipeline::submit(const NpuEncodeRequest& req) {
    std::size_t slot_idx = impl_->acquire_slot();
    impl_->encodes_submitted++;

    impl_->run_encode(slot_idx, req);

    if (config_.enable_gpu_staging) {
        impl_->stage_to_gpu(slot_idx);

        auto& slot = impl_->slots[slot_idx];
        slot->state.store(NpuDmaSlot::State::READY, std::memory_order_release);

#if defined(UM790_HAS_HIP)
        if (slot->dev.dma_stream) {
            hipStreamSynchronize(slot->dev.dma_stream);
        }
#endif
    }

    return slot_idx;
}

std::expected<std::size_t, std::string>
NpuDmaPipeline::submit_async(const NpuEncodeRequest& req) {
    std::size_t slot_idx = impl_->acquire_slot();
    impl_->encodes_submitted++;

    std::jthread([this, slot_idx, req](std::stop_token) {
        impl_->run_encode(slot_idx, req);

        if (config_.enable_gpu_staging) {
            impl_->stage_to_gpu(slot_idx);

            auto& slot = impl_->slots[slot_idx];
            slot->state.store(NpuDmaSlot::State::READY, std::memory_order_release);

#if defined(UM790_HAS_HIP)
            if (slot->dev.dma_stream) {
                hipStreamSynchronize(slot->dev.dma_stream);
            }
#endif
        }
    }).detach();

    return slot_idx;
}

NpuDmaSlot::State NpuDmaPipeline::slot_state(std::size_t slot) const {
    if (slot >= impl_->slots.size()) return NpuDmaSlot::State::ERROR;
    return impl_->slots[slot]->state.load(std::memory_order_acquire);
}

std::expected<NpuTensorHandle, std::string>
NpuDmaPipeline::wait_gpu_ready(std::size_t slot,
                                std::chrono::milliseconds timeout) {
    if (slot >= impl_->slots.size()) {
        return std::unexpected{std::format("Slot {} out of range", slot)};
    }

    auto& s = impl_->slots[slot];
    const auto deadline = std::chrono::steady_clock::now() + timeout;

    while (std::chrono::steady_clock::now() < deadline) {
        auto state = s->state.load(std::memory_order_acquire);
        if (state == NpuDmaSlot::State::READY) {
            NpuTensorHandle hnd;
            hnd.device_ptr = s->dev.device_ptr;
            hnd.element_count = s->host_buffer.size();
            hnd.npu_util = last_npu_utilization();
            hnd.valid = true;
            return hnd;
        }
        if (state == NpuDmaSlot::State::ERROR) {
            return std::unexpected{std::format("Slot {} in ERROR state", slot)};
        }
        std::this_thread::sleep_for(std::chrono::microseconds{50});
    }
    return std::unexpected{std::format("Timeout waiting for slot {}", slot)};
}

std::optional<NpuTensorHandle>
NpuDmaPipeline::try_get_gpu_handle(std::size_t slot) {
    if (slot >= impl_->slots.size()) return std::nullopt;

    auto state = impl_->slots[slot]->state.load(std::memory_order_acquire);
    if (state != NpuDmaSlot::State::READY) return std::nullopt;

    NpuTensorHandle hnd;
    hnd.device_ptr = impl_->slots[slot]->dev.device_ptr;
    hnd.element_count = impl_->slots[slot]->host_buffer.size();
    hnd.npu_util = last_npu_utilization();
    hnd.valid = (hnd.device_ptr != nullptr);
    return hnd;
}

void NpuDmaPipeline::release_slot(std::size_t slot) {
    impl_->return_slot(slot);
}

float NpuDmaPipeline::last_npu_utilization() const noexcept {
    return impl_->last_npu_util;
}

float NpuDmaPipeline::avg_npu_utilization() const noexcept {
    if (impl_->npu_samples == 0) return -1.0f;  // No samples = unknown
    return static_cast<float>(impl_->accum_npu_util /
                               static_cast<double>(impl_->npu_samples));
}

NpuDmaPipeline::PipelineStats NpuDmaPipeline::get_stats() const {
    return {
        .encodes_submitted    = impl_->encodes_submitted,
        .encodes_completed    = impl_->encodes_completed,
        .dma_transfers        = impl_->dma_transfers,
        .dma_bytes            = impl_->dma_bytes,
        .avg_encode_time_us   = impl_->encodes_completed > 0
            ? impl_->accum_encode_time_us / impl_->encodes_completed : 0.0,
        .avg_dma_time_us      = impl_->dma_transfers > 0
            ? impl_->accum_dma_time_us / impl_->dma_transfers : 0.0,
        .avg_npu_utilization  = static_cast<double>(avg_npu_utilization()),
        .active_slots         = impl_->cfg.num_slots - impl_->free_slots.size(),
    };
}

float NpuDmaPipeline::npu_temperature() const noexcept {
    return impl_->last_npu_temp;
}

} // namespace hq::npu
