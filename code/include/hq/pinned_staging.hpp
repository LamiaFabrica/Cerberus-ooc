#pragma once

/// @file pinned_staging.hpp
/// @copyright Copyright (c) 2026 D Hargreaves (AKA Roylepython). LamiaFabrica. All rights reserved.
/// @brief Double-buffered pinned staging pool for async CPU-to-GPU transfers.
///
/// Manages page-locked (pinned) host buffers paired with device buffers,
/// coordinated via HIP events for async DMA.  The pool uses a round-robin
/// slot allocation keyed by step number, enabling producer-consumer
/// overlap: the CPU fills slot N while slot N-1 is in flight.
///
/// C++26 features used:
///   - std::expected<T,E> for error propagation
///   - std::span<T> for buffer views
///   - std::print for diagnostics
///
/// @author LamiaFabrica Team

#include "hq/cxx26_features.hpp"
#include "hq/concepts.hpp"

#include <cstddef>
#include <cstdint>
#if UM790_HAS_STD_EXPECTED
#  include <expected>
#else
#  error "pinned_staging.hpp requires std::expected (<expected>) — GCC >= 14 or Clang >= 18 with C++26 enabled"
#endif
#include <memory>
#include <span>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// HIP forward declarations (avoid including hip_runtime.h in header)
// ---------------------------------------------------------------------------
struct ihipEvent_t;
struct ihipStream_t;
typedef struct ihipEvent_t* hipEvent_t;
typedef struct ihipStream_t* hipStream_t;

namespace hq {

// ===========================================================================
// StagingError — operation result codes
// ===========================================================================

enum class StagingError : std::uint8_t {
    Ok = 0,
    HipError,
    InvalidSize,
    InvalidSlot,
    TransferInProgress,
    NotInitialized,
};

// ===========================================================================
// StagingErrorInfo — rich error carrying code + message + HIP status
// ===========================================================================

struct StagingErrorInfo {
    StagingError code{StagingError::Ok};
    std::string message;
    int hip_error_code{0};  ///< Raw HIP error (e.g. hipError_t cast to int)

    StagingErrorInfo() = default;
    StagingErrorInfo(StagingError c, std::string msg, int hip_err = 0)
        : code(c), message(std::move(msg)), hip_error_code(hip_err) {}

    [[nodiscard]] constexpr bool is_ok() const noexcept {
        return code == StagingError::Ok;
    }
};

// ===========================================================================
// PinnedStagingPool — double-buffered pinned staging for embeddings
// ===========================================================================

template<hq::HqScalar T = float>
class PinnedStagingPool {
public:
    // -----------------------------------------------------------------------
    // Lifecycle
    // -----------------------------------------------------------------------

    /// @brief Create a double-buffered pool for embeddings of given size.
    /// @param embedding_bytes Size of one embedding in bytes.
    /// @param num_slots Number of buffer slots (default 2 = double-buffer).
    /// @throw Never — errors are reported via the initialized() check.
    PinnedStagingPool(std::size_t embedding_bytes, int num_slots = 2);
    ~PinnedStagingPool();

    // Non-copyable (unique GPU resources).
    PinnedStagingPool(const PinnedStagingPool&) = delete;
    PinnedStagingPool& operator=(const PinnedStagingPool&) = delete;

    // Movable.
    PinnedStagingPool(PinnedStagingPool&&) noexcept;
    PinnedStagingPool& operator=(PinnedStagingPool&&) noexcept;

    // -----------------------------------------------------------------------
    // Buffer acquisition
    // -----------------------------------------------------------------------

    /// @brief Get a host-writable buffer slot for the given step.
    ///
    /// Blocks (synchronizes) if the slot is still in flight from a
    /// previous transfer.  After this call returns, the host buffer is
    /// exclusively owned by the caller until stage_to_gpu() is invoked.
    ///
    /// @param step The step number (determines slot via round-robin).
    /// @return Writable host buffer span, or error if uninitialized.
    [[nodiscard]] std::expected<std::span<T>, StagingErrorInfo>
    acquire_host_buffer(std::uint32_t step);

    // -----------------------------------------------------------------------
    // GPU staging
    // -----------------------------------------------------------------------

    /// @brief Start async transfer of a host buffer to GPU.
    ///
    /// The host buffer for the specified step is submitted to the HIP
    /// stream via hipMemcpyAsync.  Completion is tracked via a per-slot
    /// event.  After this call, the host buffer is considered "in flight"
    /// and must not be modified until the transfer completes.
    ///
    /// @param step The step number (must match the step passed to
    ///             acquire_host_buffer() for this data).
    /// @return void on success, error if no buffer was acquired or
    ///         transfer submission fails.
    [[nodiscard]] std::expected<void, StagingErrorInfo>
    stage_to_gpu(std::uint32_t step);

    // -----------------------------------------------------------------------
    // GPU buffer access
    // -----------------------------------------------------------------------

    /// @brief Get the GPU-ready buffer for consumption.
    ///
    /// Returns the device pointer if the transfer for this step has
    /// completed.  Returns nullptr (in the expected) if still in flight.
    /// Use synchronize_step() or is_ready() to poll/await completion.
    ///
    /// @param step The step number.
    /// @return Device pointer on success, nullptr if not ready, error
    ///         if uninitialized.
    [[nodiscard]] std::expected<T*, StagingErrorInfo>
    get_gpu_buffer(std::uint32_t step);

    /// @brief Block until GPU buffer for this step is ready.
    ///
    /// Calls hipEventSynchronize on the slot's completion event.
    /// This is the heavyweight "wait" — prefer is_ready() for polling.
    ///
    /// @param step The step number.
    /// @return void on success, error if synchronization fails.
    [[nodiscard]] std::expected<void, StagingErrorInfo>
    synchronize_step(std::uint32_t step);

    /// @brief Check if GPU buffer is ready without blocking.
    ///
    /// Queries the slot's completion event via hipEventQuery.
    ///
    /// @param step The step number.
    /// @return true if the transfer has completed, false otherwise.
    [[nodiscard]] bool is_ready(std::uint32_t step) noexcept;

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    [[nodiscard]] std::size_t embedding_bytes() const noexcept {
        return embedding_bytes_;
    }

    [[nodiscard]] int num_slots() const noexcept { return num_slots_; }

    /// @brief Check whether the pool initialized successfully.
    [[nodiscard]] bool initialized() const noexcept { return initialized_; }

private:
    // -----------------------------------------------------------------------
    // Internal slot descriptor
    // -----------------------------------------------------------------------

    struct Slot {
        void* host_ptr = nullptr;    ///< hipHostMalloc'd pinned buffer
        void* device_ptr = nullptr;  ///< hipMalloc'd device buffer
        hipEvent_t event = nullptr;  ///< hipEventCreate'd completion event
        bool in_flight = false;      ///< True while async copy is pending
        bool ready = false;          ///< True after event reports complete
    };

    // -----------------------------------------------------------------------
    // Data members
    // -----------------------------------------------------------------------

    std::size_t embedding_bytes_;  ///< Bytes per embedding
    int num_slots_;                ///< Total ring-buffer slots
    std::vector<Slot> slots_;      ///< Slot storage
    hipStream_t stream_ = nullptr; ///< HIP stream for all async ops
    bool initialized_ = false;     ///< True after successful construction

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Map step → slot index via round-robin.
    [[nodiscard]] int slot_for_step(std::uint32_t step) const noexcept {
        return static_cast<int>(step % static_cast<std::uint32_t>(num_slots_));
    }

    /// Block until a slot's in-flight transfer completes, then clear flags.
    void drain_slot(int slot_idx);

    /// Release all HIP resources — noexcept, called from destructor.
    void free_all() noexcept;
};

// ===========================================================================
// Type aliases for common embedding types
// ===========================================================================

using T5StagingPool = PinnedStagingPool<float>;
using CLIPStagingPool = PinnedStagingPool<float>;

// ===========================================================================
/// @brief Convenience free function for error formatting.
// ===========================================================================
[[nodiscard]] inline std::string to_string(StagingError e) {
    switch (e) {
        case StagingError::Ok:                return "Ok";
        case StagingError::HipError:          return "HipError";
        case StagingError::InvalidSize:       return "InvalidSize";
        case StagingError::InvalidSlot:       return "InvalidSlot";
        case StagingError::TransferInProgress: return "TransferInProgress";
        case StagingError::NotInitialized:    return "NotInitialized";
    }
    return "Unknown";
}

} // namespace hq

// ---------------------------------------------------------------------------
// Template implementation — must be visible to all translation units that
// instantiate PinnedStagingPool<T>.  Included here (rather than the header
// alone) because the full implementation requires HIP headers.
// ---------------------------------------------------------------------------
#include "../src/pinned_staging.cpp"   // template implementation (in src/)
