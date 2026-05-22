/// @file pinned_staging.cpp
/// @copyright Copyright (c) 2026 D Hargreaves (AKA Roylepython). LamiaFabrica. All rights reserved.
/// @brief PinnedStagingPool template implementation — HIP async DMA staging.
///
/// This file contains the full implementation of PinnedStagingPool<T>.
/// It is #included at the bottom of pinned_staging.hpp so that the template
/// body is visible in every translation unit that instantiates it.
///
/// Do NOT compile this file directly — include the header instead.
///
/// @author LamiaFabrica Team

#include <algorithm>
#include <chrono>
#include <cstring>
#if UM790_HAS_STD_PRINT
#  include <print>
#endif
#include <expected>
#include <thread>

// ---------------------------------------------------------------------------
// HIP headers (only when compiling with HIP / ROCm)
// ---------------------------------------------------------------------------
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__) || defined(USE_HIP)
#  include <hip/hip_runtime.h>
#  define PINNED_STAGING_HAS_HIP 1
#else
#  define PINNED_STAGING_HAS_HIP 0
#  pragma message("HIP not detected \u2014 PinnedStagingPool will use regular allocation")
#endif

namespace hq {

// ===========================================================================
// PinnedStagingPool — Construction / Destruction
// ===========================================================================

template<typename T>
PinnedStagingPool<T>::PinnedStagingPool(std::size_t embedding_bytes, int num_slots)
    : embedding_bytes_{embedding_bytes}
    , num_slots_{num_slots}
{
    // Validate parameters
    if (embedding_bytes == 0) {
        std::print("[staging] ERROR: embedding_bytes must be > 0\n");
        return;
    }
    if (num_slots < 1) {
        std::print("[staging] ERROR: num_slots must be >= 1\n");
        return;
    }

    slots_.resize(num_slots_);

#if PINNED_STAGING_HAS_HIP
    // Create a dedicated HIP stream for all staging operations
    hipError_t err = hipStreamCreate(&stream_);
    if (err != hipSuccess) {
        std::print("[staging] hipStreamCreate failed: {}\n",
                   hipGetErrorString(err));
        stream_ = nullptr;
        return;
    }

    // Allocate each slot: pinned host buffer + device buffer + event
    for (int i = 0; i < num_slots_; ++i) {
        Slot& s = slots_[i];

        // --- Pinned host memory (page-locked) ---
        err = hipHostMalloc(&s.host_ptr, embedding_bytes_, hipHostMallocDefault);
        if (err != hipSuccess) {
            std::print("[staging] hipHostMalloc slot {} failed: {}\n",
                       i, hipGetErrorString(err));
            free_all();
            return;
        }

        // --- Device memory ---
        err = hipMalloc(&s.device_ptr, embedding_bytes_);
        if (err != hipSuccess) {
            std::print("[staging] hipMalloc slot {} failed: {}\n",
                       i, hipGetErrorString(err));
            free_all();
            return;
        }

        // --- Completion event ---
        err = hipEventCreate(&s.event);
        if (err != hipSuccess) {
            std::print("[staging] hipEventCreate slot {} failed: {}\n",
                       i, hipGetErrorString(err));
            free_all();
            return;
        }
    }
#else
    // --- Stub build: use regular heap allocations ---
    for (int i = 0; i < num_slots_; ++i) {
        Slot& s = slots_[i];
        s.host_ptr = std::malloc(embedding_bytes_);
        if (!s.host_ptr) {
            std::print("[staging] std::malloc slot {} failed\n", i);
            free_all();
            return;
        }
        // No device memory without HIP (CPU-only build)
        s.device_ptr = nullptr;
        s.event = nullptr;
    }
#endif

    initialized_ = true;
    std::print("[staging] PinnedStagingPool: {} slots x {} bytes = {} total\n",
               num_slots_, embedding_bytes_,
               static_cast<std::size_t>(num_slots_) * embedding_bytes_);
}

template<typename T>
PinnedStagingPool<T>::~PinnedStagingPool() {
    free_all();
}

// ===========================================================================
// Move semantics
// ===========================================================================

template<typename T>
PinnedStagingPool<T>::PinnedStagingPool(PinnedStagingPool&& other) noexcept
    : embedding_bytes_{other.embedding_bytes_}
    , num_slots_{other.num_slots_}
    , slots_{std::move(other.slots_)}
    , stream_{other.stream_}
    , initialized_{other.initialized_}
{
    other.embedding_bytes_ = 0;
    other.num_slots_ = 0;
    other.stream_ = nullptr;
    other.initialized_ = false;
}

template<typename T>
PinnedStagingPool<T>& PinnedStagingPool<T>::operator=(PinnedStagingPool&& other) noexcept {
    if (this != &other) {
        free_all();

        embedding_bytes_ = other.embedding_bytes_;
        num_slots_ = other.num_slots_;
        slots_ = std::move(other.slots_);
        stream_ = other.stream_;
        initialized_ = other.initialized_;

        other.embedding_bytes_ = 0;
        other.num_slots_ = 0;
        other.stream_ = nullptr;
        other.initialized_ = false;
    }
    return *this;
}

// ===========================================================================
// acquire_host_buffer — get writable host buffer, drain if in flight
// ===========================================================================

template<typename T>
std::expected<std::span<T>, StagingErrorInfo>
PinnedStagingPool<T>::acquire_host_buffer(std::uint32_t step) {
    if (!initialized_) {
        return std::unexpected(
            StagingErrorInfo{StagingError::NotInitialized,
                             "Pool not initialized (constructor failed)"});
    }

    const int idx = slot_for_step(step);
    if (idx < 0 || idx >= num_slots_) {
        return std::unexpected(
            StagingErrorInfo{StagingError::InvalidSlot,
                             std::format("Slot index {} out of range [0, {})",
                                         idx, num_slots_)});
    }

    Slot& s = slots_[idx];

    // If this slot still has an in-flight transfer, block until it completes.
    // This ensures the CPU does not overwrite data the DMA engine is reading.
    if (s.in_flight) {
        drain_slot(idx);
    }

    // Clear ready flag — the CPU is about to (re)write this buffer.
    s.ready = false;

    const std::size_t num_elements = embedding_bytes_ / sizeof(T);
    return std::span<T>{static_cast<T*>(s.host_ptr), num_elements};
}

// ===========================================================================
// stage_to_gpu — start async host→device transfer
// ===========================================================================

template<typename T>
std::expected<void, StagingErrorInfo>
PinnedStagingPool<T>::stage_to_gpu(std::uint32_t step) {
    if (!initialized_) {
        return std::unexpected(
            StagingErrorInfo{StagingError::NotInitialized,
                             "Pool not initialized (constructor failed)"});
    }

    const int idx = slot_for_step(step);
    if (idx < 0 || idx >= num_slots_) {
        return std::unexpected(
            StagingErrorInfo{StagingError::InvalidSlot,
                             std::format("Slot index {} out of range [0, {})",
                                         idx, num_slots_)});
    }

    Slot& s = slots_[idx];

    if (!s.host_ptr) {
        return std::unexpected(
            StagingErrorInfo{StagingError::InvalidSlot,
                             std::format("Slot {} has no host buffer", idx)});
    }

#if PINNED_STAGING_HAS_HIP
    if (!s.device_ptr) {
        return std::unexpected(
            StagingErrorInfo{StagingError::InvalidSlot,
                             std::format("Slot {} has no device buffer", idx)});
    }

    // hipMemcpyAsync: host (pinned) → device, non-blocking on the stream
    hipError_t err = hipMemcpyAsync(
        s.device_ptr,          // dst
        s.host_ptr,            // src
        embedding_bytes_,      // sizeBytes
        hipMemcpyHostToDevice, // kind
        stream_);              // stream

    if (err != hipSuccess) {
        return std::unexpected(
            StagingErrorInfo{StagingError::HipError,
                             std::format("hipMemcpyAsync failed: {}",
                                         hipGetErrorString(err)),
                             static_cast<int>(err)});
    }

    // Record completion event on the stream
    err = hipEventRecord(s.event, stream_);
    if (err != hipSuccess) {
        return std::unexpected(
            StagingErrorInfo{StagingError::HipError,
                             std::format("hipEventRecord failed: {}",
                                         hipGetErrorString(err)),
                             static_cast<int>(err)});
    }
#endif

    s.in_flight = true;
    s.ready = false;

    return {};
}

// ===========================================================================
// get_gpu_buffer — get device pointer if transfer is complete
// ===========================================================================

template<typename T>
std::expected<T*, StagingErrorInfo>
PinnedStagingPool<T>::get_gpu_buffer(std::uint32_t step) {
    if (!initialized_) {
        return std::unexpected(
            StagingErrorInfo{StagingError::NotInitialized,
                             "Pool not initialized (constructor failed)"});
    }

    const int idx = slot_for_step(step);
    if (idx < 0 || idx >= num_slots_) {
        return std::unexpected(
            StagingErrorInfo{StagingError::InvalidSlot,
                             std::format("Slot index {} out of range [0, {})",
                                         idx, num_slots_)});
    }

    Slot& s = slots_[idx];

    // get_gpu_buffer() returns nullptr in two distinct cases:
    //   (1) No transfer was ever submitted for this slot via stage_to_gpu()
    //   (2) Transfer was submitted but is still in-flight (not yet complete)
    // Callers should use is_ready() to disambiguate, or call synchronize_step()
    // for guaranteed readiness.
    if (!s.in_flight) {
        // No transfer was submitted for this slot
        return static_cast<T*>(s.device_ptr);
    }

#if PINNED_STAGING_HAS_HIP
    // Lightweight poll — does NOT block
    hipError_t err = hipEventQuery(s.event);
    if (err == hipSuccess) {
        s.in_flight = false;
        s.ready = true;
        return static_cast<T*>(s.device_ptr);
    } else if (err == hipErrorNotReady) {
        // Still in flight — return nullptr (caller should poll again)
        return nullptr;
    } else {
        // Real HIP error
        return std::unexpected(
            StagingErrorInfo{StagingError::HipError,
                             std::format("hipEventQuery failed: {}",
                                         hipGetErrorString(err)),
                             static_cast<int>(err)});
    }
#else
    // Stub build: always "ready"
    s.in_flight = false;
    s.ready = true;
    return static_cast<T*>(s.device_ptr);
#endif
}

// ===========================================================================
// synchronize_step — block until GPU buffer is ready
// ===========================================================================

template<typename T>
std::expected<void, StagingErrorInfo>
PinnedStagingPool<T>::synchronize_step(std::uint32_t step) {
    if (!initialized_) {
        return std::unexpected(
            StagingErrorInfo{StagingError::NotInitialized,
                             "Pool not initialized (constructor failed)"});
    }

    const int idx = slot_for_step(step);
    if (idx < 0 || idx >= num_slots_) {
        return std::unexpected(
            StagingErrorInfo{StagingError::InvalidSlot,
                             std::format("Slot index {} out of range [0, {})",
                                         idx, num_slots_)});
    }

    Slot& s = slots_[idx];

    if (!s.in_flight) {
        // Nothing to synchronize
        return {};
    }

#if PINNED_STAGING_HAS_HIP
    hipError_t err = hipEventSynchronize(s.event);
    if (err != hipSuccess) {
        return std::unexpected(
            StagingErrorInfo{StagingError::HipError,
                             std::format("hipEventSynchronize failed: {}",
                                         hipGetErrorString(err)),
                             static_cast<int>(err)});
    }
#endif

    s.in_flight = false;
    s.ready = true;

    return {};
}

// ===========================================================================
// is_ready — lightweight poll without blocking
// ===========================================================================

template<typename T>
bool PinnedStagingPool<T>::is_ready(std::uint32_t step) noexcept {
    if (!initialized_) {
        return false;
    }

    const int idx = slot_for_step(step);
    if (idx < 0 || idx >= num_slots_) {
        return false;
    }

    Slot& s = slots_[idx];

    if (!s.in_flight) {
        return s.ready;  // true if transfer already completed
    }

#if PINNED_STAGING_HAS_HIP
    hipError_t err = hipEventQuery(s.event);
    if (err == hipSuccess) {
        s.in_flight = false;
        s.ready = true;
        return true;
    }
    // hipErrorNotReady or other error → not ready
    return false;
#else
    // Stub build: always ready
    s.in_flight = false;
    s.ready = true;
    return true;
#endif
}

// ===========================================================================
// drain_slot — block until in-flight transfer completes
// ===========================================================================

template<typename T>
void PinnedStagingPool<T>::drain_slot(int slot_idx) {
    if (slot_idx < 0 || slot_idx >= num_slots_) {
        return;
    }

    Slot& s = slots_[slot_idx];

    if (!s.in_flight) {
        return;
    }

#if PINNED_STAGING_HAS_HIP
    hipError_t err = hipEventSynchronize(s.event);
    if (err != hipSuccess) {
        std::print("[staging] WARN: hipEventSynchronize slot {} failed: {}\n",
                   slot_idx, hipGetErrorString(err));
    }
#endif

    s.in_flight = false;
    s.ready = true;
}

// ===========================================================================
// free_all — release all HIP resources (noexcept for destructor safety)
// ===========================================================================

template<typename T>
void PinnedStagingPool<T>::free_all() noexcept {
#if PINNED_STAGING_HAS_HIP
    for (Slot& s : slots_) {
        if (s.event) {
            hipEventDestroy(s.event);
            s.event = nullptr;
        }
        if (s.device_ptr) {
            hipFree(s.device_ptr);
            s.device_ptr = nullptr;
        }
        if (s.host_ptr) {
            hipHostFree(s.host_ptr);
            s.host_ptr = nullptr;
        }
        s.in_flight = false;
        s.ready = false;
    }

    if (stream_) {
        hipStreamDestroy(stream_);
        stream_ = nullptr;
    }
#else
    // Stub build: free heap allocations
    for (Slot& s : slots_) {
        if (s.host_ptr) {
            std::free(s.host_ptr);
            s.host_ptr = nullptr;
        }
        s.in_flight = false;
        s.ready = false;
    }
#endif

    slots_.clear();
    initialized_ = false;
}

} // namespace hq
