#pragma once
/// @file tiered_memory_manager.hpp
/// @copyright Copyright (c) 2026 D Hargreaves (AKA Roylepython). LamiaFabrica. All rights reserved.
///
/// TRADE SECRET — Proprietary and Confidential.
/// This file contains trade secrets of LamiaFabrica / D Hargreaves.
/// Unauthorized use, reproduction, or disclosure is strictly prohibited.
/// See LICENSE for terms.
///
/// Four-tier heterogeneous memory manager for UM790 Pro pipeline.
///
/// Tier hierarchy (hot → cold):
///   Hot  — NPU SRAM + GPU VRAM (device-local, fastest)
///   Warm — System RAM fallback (CXL would be 128–256 GiB coherent DRAM expander when available)
///   Cool — System RAM (64 GiB DDR5)
///   Cold — NVMe / external SSD (4 TB+ persistent tier)
///
/// Design principles:
///   - Every allocation carries a MemoryTier tag; promotion/demotion is
///     explicit (triggered by the watchdog or the pipeline scheduler).
///   - C++26 std::pmr allocators back each tier so STL containers can
///     live in any tier without copying logic into business code.
///   - std::expected<T, TierError> propagates errors without exceptions on
///     hot paths.
///   - All public methods are thread-safe via per-tier shared_mutexes.
///
/// @version 1.0.0

#include "hq/cxx26_features.hpp"

#include <atomic>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#if UM790_HAS_STD_EXPECTED
#  include <expected>
#else
#  error "tiered_memory_manager.hpp requires std::expected (C++26 / GCC 14+)"
#endif
#if UM790_HAS_STD_MDSPAN
#  include <mdspan>
#endif
#include <functional>
#include <memory>
#include <memory_resource>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <string>
#include <vector>

#if defined(UM790_HAS_HIP)
#  include <hip/hip_runtime_api.h>
#endif

namespace hq {

// ---------------------------------------------------------------------------
// Tier identifiers
// ---------------------------------------------------------------------------

enum class MemoryTier : std::uint8_t {
    Hot  = 0,   ///< GPU VRAM or NPU SRAM (device-local)
    Warm = 1,   ///< CXL coherent expander pool
    Cool = 2,   ///< Host system RAM (DDR5)
    Cold = 3,   ///< NVMe / external SSD
};

[[nodiscard]] inline std::string to_string(MemoryTier t) noexcept {
    switch (t) {
        case MemoryTier::Hot:  return "Hot";
        case MemoryTier::Warm: return "Warm";
        case MemoryTier::Cool: return "Cool";
        case MemoryTier::Cold: return "Cold";
    }
    return "Unknown";
}

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

enum class TierError : std::uint8_t {
    OutOfMemory      = 0,  ///< Requested tier exhausted
    InvalidHandle    = 1,  ///< Handle not found or already freed
    MigrationFailed  = 2,  ///< Could not move block between tiers
    TierUnavailable  = 3,  ///< Hardware tier not present (e.g. no CXL)
    AlignmentError   = 4,  ///< Requested alignment not satisfiable
    InvalidSize      = 5,  ///< Zero or overflow size
};

[[nodiscard]] inline std::string to_string(TierError e) noexcept {
    switch (e) {
        case TierError::OutOfMemory:     return "OutOfMemory";
        case TierError::InvalidHandle:   return "InvalidHandle";
        case TierError::MigrationFailed: return "MigrationFailed";
        case TierError::TierUnavailable: return "TierUnavailable";
        case TierError::AlignmentError:  return "AlignmentError";
        case TierError::InvalidSize:     return "InvalidSize";
    }
    return "Unknown";
}

// ---------------------------------------------------------------------------
// Handle type (opaque 64-bit ID issued per allocation)
// ---------------------------------------------------------------------------

using TierHandle = std::uint64_t;
inline constexpr TierHandle kInvalidTierHandle = 0;

// ---------------------------------------------------------------------------
// Allocation descriptor returned to callers
// ---------------------------------------------------------------------------

struct TierAllocation {
    TierHandle   handle{kInvalidTierHandle}; ///< Unique allocation ID
    MemoryTier   tier{MemoryTier::Cold};      ///< Current residence tier
    void*        ptr{nullptr};         ///< Host-accessible pointer (may be null for Hot/device-only)
    void*        device_ptr{nullptr};  ///< Device pointer (non-null on Hot/CXL tiers with HIP)
    std::size_t  size_bytes{0};  ///< Requested allocation size
    std::size_t  alignment{0};   ///< Actual alignment (>= requested)
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct TieredMemoryConfig {
    // Tier capacities (0 = tier disabled)
    std::size_t hot_capacity_bytes  = 0;              ///< GPU VRAM budget (0=auto from ROCm)
    std::size_t warm_capacity_bytes = 128ULL << 30;   ///< Warm tier (CXL if available, else RAM; 128 GiB default)
    std::size_t cool_capacity_bytes =  64ULL << 30;   ///< System RAM pool (64 GiB)
    std::size_t cold_capacity_bytes =   4ULL << 40;   ///< NVMe budget (4 TiB)

    // Migration policy
    float       hot_watermark_pct   = 0.85f;   ///< Evict Hot→Warm above this fill %
    float       warm_watermark_pct  = 0.80f;   ///< Evict Warm→Cool above this fill %
    float       cool_watermark_pct  = 0.75f;   ///< Evict Cool→Cold above this fill %

    // Alignment defaults per tier (bytes)
    std::size_t hot_alignment       = 256;
    std::size_t warm_alignment      = 64;
    std::size_t cool_alignment      = 64;
    std::size_t cold_alignment      = 512;  ///< NVMe sector alignment

    // NVMe spill path
    std::string cold_spill_dir      = "/tmp/cerberus_cold";
};

// ---------------------------------------------------------------------------
// Per-tier statistics snapshot
// ---------------------------------------------------------------------------

struct TierStats {
    MemoryTier  tier{MemoryTier::Cold};
    bool        available{false};          ///< Hardware tier present
    std::size_t capacity_bytes{0};
    std::size_t allocated_bytes{0};
    std::size_t peak_allocated_bytes{0};
    std::uint64_t alloc_count{0};
    std::uint64_t free_count{0};
    std::uint64_t migration_in{0};     ///< Blocks migrated into this tier
    std::uint64_t migration_out{0};    ///< Blocks migrated out of this tier
    float        fill_pct{0.0f};          ///< allocated / capacity * 100
};

// ---------------------------------------------------------------------------
// Migration hook signature (for watchdog integration)
// ---------------------------------------------------------------------------

/// Called when the memory manager initiates a tier migration.
/// @param handle    Allocation being moved.
/// @param from_tier Source tier.
/// @param to_tier   Destination tier.
using MigrationHook = std::function<void(TierHandle handle,
                                           MemoryTier from_tier,
                                           MemoryTier to_tier)>;

// ===========================================================================
// MigrationComputeHook — in-flight compute during tier movement
//
// Called after source data is staged but before destination writeback.
// Examples: partial dequantization, layout reshape, compression.
// If the hook throws, migration is aborted and source data untouched.
// ===========================================================================

using MigrationComputeHook = std::function<void(
    void* src_ptr,         ///< source tensor data (const-ish, don't free)
    void* dst_ptr,         ///< destination buffer (write here)
    std::size_t bytes,     ///< total bytes in the tensor
    MemoryTier from_tier,  ///< where it lived
    MemoryTier to_tier)    ///< where it's going
>;

// ---------------------------------------------------------------------------
// TieredMemoryManager
// ---------------------------------------------------------------------------

class TieredMemoryManager {
public:
    explicit TieredMemoryManager(const TieredMemoryConfig& cfg = {},
                                 MigrationHook on_migrate = nullptr);
    ~TieredMemoryManager() noexcept;

    // Non-copyable; moveable
    TieredMemoryManager(const TieredMemoryManager&) = delete;
    TieredMemoryManager& operator=(const TieredMemoryManager&) = delete;
    TieredMemoryManager(TieredMemoryManager&&) noexcept;
    TieredMemoryManager& operator=(TieredMemoryManager&&) noexcept;

    // ------------------------------------------------------------------
    // Core allocation API
    // ------------------------------------------------------------------

    /// Allocate @p size bytes at the preferred @p tier (or the next
    /// available warmer/cooler tier on failure).
    [[nodiscard]] std::expected<TierAllocation, TierError>
    allocate(std::size_t size_bytes,
             MemoryTier preferred_tier = MemoryTier::Cool,
             std::size_t alignment = 0);

    /// Release a previously allocated block (any tier).
    std::expected<void, TierError> free(TierHandle handle) noexcept;

    /// Look up the current descriptor for a handle.
    [[nodiscard]] std::expected<TierAllocation, TierError>
    query(TierHandle handle) const noexcept;

    // ------------------------------------------------------------------
    // Migration API (callable by watchdog on memory-pressure events)
    // ------------------------------------------------------------------

    /// Register an in-flight compute hook called during promote/demote.
    [[nodiscard]] std::expected<void, std::string>
    register_compute_hook(MigrationComputeHook hook);

    /// Predict whether preferred_tier allocation will succeed soon.
    /// Returns estimated free bytes in that tier.
    [[nodiscard]] std::size_t predict_pressure(MemoryTier tier) const noexcept;

    /// Promote a block to a hotter tier (Cool→Warm→Hot).
    [[nodiscard]] std::expected<TierAllocation, TierError>
    promote(TierHandle handle);

    /// Demote a block to a cooler tier (Hot→Warm→Cool→Cold).
    [[nodiscard]] std::expected<TierAllocation, TierError>
    demote(TierHandle handle);

    /// Evict least-recently-used blocks from @p tier until fill is below
    /// @p target_pct. Triggers on_migrate hook for each eviction.
    /// Returns number of blocks migrated.
    std::expected<std::size_t, TierError>
    pressure_evict(MemoryTier tier, float target_pct = 0.70f);

    // ------------------------------------------------------------------
    // Telemetry
    // ------------------------------------------------------------------

    [[nodiscard]] TierStats stats(MemoryTier tier) const noexcept;
    [[nodiscard]] std::vector<TierStats> all_stats() const;
    [[nodiscard]] bool tier_available(MemoryTier tier) const noexcept;

    // ------------------------------------------------------------------
    // PMR resource accessors (for STL container use)
    // ------------------------------------------------------------------

    /// std::pmr::memory_resource for the Cool (system RAM) tier.
    [[nodiscard]] std::pmr::memory_resource* cool_resource() noexcept;

    /// std::pmr::memory_resource for the Warm (CXL) tier.
    [[nodiscard]] std::pmr::memory_resource* warm_resource() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ---------------------------------------------------------------------------
// Convenience: RAII scoped allocation
// ---------------------------------------------------------------------------

class ScopedTierAlloc {
public:
    ScopedTierAlloc() = default;

    ScopedTierAlloc(TieredMemoryManager& mgr, TierAllocation alloc) noexcept
        : mgr_{&mgr}, alloc_{alloc} {}

    ~ScopedTierAlloc() noexcept {
        if (mgr_ && alloc_.handle != kInvalidTierHandle)
            (void)mgr_->free(alloc_.handle);
    }

    ScopedTierAlloc(const ScopedTierAlloc&) = delete;
    ScopedTierAlloc& operator=(const ScopedTierAlloc&) = delete;

    ScopedTierAlloc(ScopedTierAlloc&& o) noexcept
        : mgr_{o.mgr_}, alloc_{o.alloc_} {
        o.mgr_ = nullptr;
        o.alloc_.handle = kInvalidTierHandle;
    }

    ScopedTierAlloc& operator=(ScopedTierAlloc&& o) noexcept {
        if (this != &o) {
            if (mgr_ && alloc_.handle != kInvalidTierHandle)
                (void)mgr_->free(alloc_.handle);
            mgr_ = o.mgr_;
            alloc_ = o.alloc_;
            o.mgr_ = nullptr;
            o.alloc_.handle = kInvalidTierHandle;
        }
        return *this;
    }

    [[nodiscard]] void*       ptr()         const noexcept { return alloc_.ptr; }
    [[nodiscard]] void*       device_ptr()  const noexcept { return alloc_.device_ptr; }
    [[nodiscard]] std::size_t size_bytes()  const noexcept { return alloc_.size_bytes; }
    [[nodiscard]] MemoryTier  tier()        const noexcept { return alloc_.tier; }
    [[nodiscard]] TierHandle  handle()      const noexcept { return alloc_.handle; }
    [[nodiscard]] bool        valid()       const noexcept { return alloc_.handle != kInvalidTierHandle; }

    [[nodiscard]] const TierAllocation& get() const noexcept { return alloc_; }

private:
    TieredMemoryManager* mgr_{nullptr};
    TierAllocation       alloc_{};
};

// ===========================================================================
// C++26 Concepts for memory tier constraints
// ===========================================================================

/// Requires that T exposes the core TieredMemoryManager allocation surface:
///   allocate(size, tier) → expected<TierAllocation, TierError>
///   free(handle)         → expected<void, TierError>
///   tier_available(tier) → bool
template<typename T>
concept MemoryTierResource = requires(T& mgr,
                                       std::size_t   sz,
                                       MemoryTier    tier,
                                       TierHandle    h) {
    { mgr.allocate(sz, tier) }    -> std::same_as<std::expected<TierAllocation, TierError>>;
    { mgr.free(h) }               -> std::same_as<std::expected<void, TierError>>;
    { mgr.tier_available(tier) }  -> std::same_as<bool>;
};

static_assert(MemoryTierResource<TieredMemoryManager>,
              "TieredMemoryManager must satisfy MemoryTierResource");

} // namespace hq
