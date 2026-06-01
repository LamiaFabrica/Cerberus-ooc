/// @file tiered_memory_manager.cpp
/// @copyright Copyright © 2026, all rights reserved | David Hargreaves aka Roylepython trading as The Medusa Initiative & Lamia Fabrica
/// Support: https://www.patreon.com/TheMedusaInitiative — £25/month removes ads from all software at 200 subscribers
/// Four-tier heterogeneous memory manager — implementation.

#include "hq/tiered_memory_manager.hpp"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <list>
#include <sstream>
#if UM790_HAS_STD_PRINT
#  include <print>
#endif
#include <unordered_map>

#ifdef UM790_HAS_HIP
#  include <hip/hip_runtime_api.h>
#endif
#ifdef _WIN32
#  include <malloc.h>
#endif

namespace hq {

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

constexpr std::size_t align_up(std::size_t v, std::size_t a) noexcept {
    return (v + a - 1) & ~(a - 1);
}

inline void* alloc_aligned_(std::size_t align, std::size_t size) noexcept {
#ifdef _WIN32
    return _aligned_malloc(size, align);
#else
    return std::aligned_alloc(align, size);
#endif
}

inline void free_aligned_(void* p) noexcept {
#ifdef _WIN32
    _aligned_free(p);
#else
    std::free(p);
#endif
}

// ---------------------------------------------------------------------------
// CoolPmrResource — std::pmr::memory_resource backed by aligned_alloc
// ---------------------------------------------------------------------------

class CoolPmrResource final : public std::pmr::memory_resource {
public:
    explicit CoolPmrResource(std::size_t capacity_bytes,
                              std::size_t default_alignment) noexcept
        : capacity_{capacity_bytes}
        , default_align_{default_alignment} {}

    [[nodiscard]] std::size_t allocated() const noexcept { return allocated_; }
    [[nodiscard]] std::size_t default_alignment() const noexcept { return default_align_; }

protected:
    void* do_allocate(std::size_t bytes, std::size_t alignment) override {
        const std::size_t a = std::max(alignment, default_align_);
        const std::size_t actual = align_up(bytes, a);
        if (capacity_ > 0 && allocated_ + actual > capacity_)
            throw std::bad_alloc{};
        void* p = alloc_aligned_(a, actual);
        if (!p) throw std::bad_alloc{};
        allocated_ += actual;  // track actual allocated size for accurate accounting
        return p;
    }

    void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) override {
        const std::size_t a = std::max(alignment, default_align_);
        const std::size_t actual = align_up(bytes, a);
        free_aligned_(p);
        if (allocated_ >= actual) allocated_ -= actual;
        else allocated_ = 0; // defensive
    }

    bool do_is_equal(const memory_resource& o) const noexcept override {
        return this == &o;
    }

private:
    std::size_t capacity_;
    std::size_t default_align_;
    std::size_t allocated_{0};
};

// ---------------------------------------------------------------------------
// RamFallbackPmrResource — System RAM pool (aligned_alloc fallback).
//                   When CXL hardware is present, a real implementation would
//                   call the CXL memory-tiering ioctl or numactl-aware mmap.
//                   Currently detect_cxl() always returns false, so this is
//                   honest RAM-backed allocation.
// ---------------------------------------------------------------------------

class RamFallbackPmrResource final : public std::pmr::memory_resource {
public:
    explicit RamFallbackPmrResource(std::size_t capacity_bytes,
                              std::size_t default_alignment,
                              bool cxl_present) noexcept
        : capacity_{capacity_bytes}
        , default_align_{default_alignment}
        , cxl_present_{cxl_present} {}

    [[nodiscard]] bool cxl_present() const noexcept { return cxl_present_; }
    [[nodiscard]] std::size_t allocated() const noexcept { return allocated_; }
    [[nodiscard]] std::size_t default_alignment() const noexcept { return default_align_; }

protected:
    void* do_allocate(std::size_t bytes, std::size_t alignment) override {
        const std::size_t a = std::max(alignment, default_align_);
        const std::size_t actual = align_up(bytes, a);
        if (capacity_ > 0 && allocated_ + actual > capacity_)
            throw std::bad_alloc{};

        void* p = nullptr;
        // When CXL is present, bind to NUMA node 1 (CXL expander).
        // Without libnuma / CXL driver, fall back to aligned_alloc.
#if __has_include(<numa.h>)
        if (cxl_present_) {
            p = numa_alloc_onnode(actual, 1 /*CXL node*/);
        }
#endif
        if (!p) {
            p = alloc_aligned_(a, actual);
        }
        if (!p) throw std::bad_alloc{};
        allocated_ += actual;  // track actual allocated size (with alignment)
        return p;
    }

    void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) override {
        const std::size_t a = std::max(alignment, default_align_);
        const std::size_t actual = align_up(bytes, a);
#if __has_include(<numa.h>)
        if (cxl_present_) {
            numa_free(p, actual);
            if (allocated_ >= actual) allocated_ -= actual;
            else allocated_ = 0;
            return;
        }
#endif
        free_aligned_(p);
        if (allocated_ >= actual) allocated_ -= actual;
        else allocated_ = 0; // defensive against drift
    }

    bool do_is_equal(const memory_resource& o) const noexcept override {
        return this == &o;
    }

private:
    std::size_t capacity_;
    std::size_t default_align_;
    bool        cxl_present_;
    std::size_t allocated_{0};
};

} // anonymous namespace

// ---------------------------------------------------------------------------
// Impl (pImpl pattern)
// ---------------------------------------------------------------------------

struct TieredMemoryManager::Impl {
    TieredMemoryConfig cfg;
    MigrationHook      on_migrate;
    MigrationComputeHook compute_hook;

    // PMR resources
    std::unique_ptr<CoolPmrResource> cool_res;
    std::unique_ptr<RamFallbackPmrResource> warm_res;

    // Per-tier availability flags
    bool hot_available{false};   // GPU VRAM (requires HIP)
    bool warm_available{false};  // CXL or fallback RAM
    bool cool_available{true};   // Always available
    bool cold_available{false};  // NVMe (path exists and is writable)

    // Allocation registry (handle → descriptor)
    // Protected by registry_mutex_
    mutable std::shared_mutex registry_mutex;
    std::unordered_map<TierHandle, TierAllocation> registry;
    std::atomic<TierHandle> next_handle{1};

    // Per-tier accounting (lock-free for hot-path reads)
    struct TierAccounting {
        std::atomic<std::size_t>   allocated{0};
        std::atomic<std::size_t>   peak{0};
        std::atomic<std::uint64_t> alloc_count{0};
        std::atomic<std::uint64_t> free_count{0};
        std::atomic<std::uint64_t> migration_in{0};
        std::atomic<std::uint64_t> migration_out{0};
        std::size_t capacity{0};
        bool available{false};
    };
    std::array<TierAccounting, 4> acct; // indexed by MemoryTier

    // LRU list per tier for pressure eviction (protected by registry_mutex)
    std::array<std::list<TierHandle>, 4> lru;

    // Cold-tier spill: map handle → file path
    std::unordered_map<TierHandle, std::filesystem::path> cold_files;

    explicit Impl(const TieredMemoryConfig& c, MigrationHook hook)
        : cfg{c}, on_migrate{std::move(hook)} {}
};

// ---------------------------------------------------------------------------
// Helpers for Impl internals
// ---------------------------------------------------------------------------

namespace {

void* alloc_hot(std::size_t bytes, std::size_t alignment) {
#ifdef UM790_HAS_HIP
    void* p = nullptr;
    hipError_t err = hipMalloc(&p, bytes);
    if (err != hipSuccess || !p) return nullptr;
    (void)alignment; // HIP aligns to at least 256 B
    return p;
#else
    (void)bytes; (void)alignment;
    return nullptr;
#endif
}

void free_hot(void* p) {
#ifdef UM790_HAS_HIP
    if (p) hipFree(p);
#else
    (void)p;
#endif
}

bool detect_cxl() noexcept {
    // Probe for NUMA node 1 with CXL memory type. On real hardware this would
    // inspect /sys/bus/cxl/devices/ or use libnuma's numa_node_size().
    // For portability across dev machines we return false and rely on fallback.
#if __has_include(<numa.h>)
    // Would call: numa_available() >= 0 && numa_num_configured_nodes() > 1
#endif
    return false;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------

TieredMemoryManager::TieredMemoryManager(const TieredMemoryConfig& cfg,
                                          MigrationHook on_migrate)
    : impl_{std::make_unique<Impl>(cfg, std::move(on_migrate))} {
    auto& m = *impl_;

    // Cool tier — always available
    m.cool_res = std::make_unique<CoolPmrResource>(
        cfg.cool_capacity_bytes, cfg.cool_alignment);
    m.cool_available = true;
    m.acct[static_cast<int>(MemoryTier::Cool)].capacity  = cfg.cool_capacity_bytes;
    m.acct[static_cast<int>(MemoryTier::Cool)].available = true;

    // Warm tier (CXL or RAM fallback)
    const bool cxl = detect_cxl();
    m.warm_res = std::make_unique<RamFallbackPmrResource>(
        cfg.warm_capacity_bytes, cfg.warm_alignment, cxl);
    m.warm_available = true; // Always provide Warm (even without real CXL)
    m.acct[static_cast<int>(MemoryTier::Warm)].capacity  = cfg.warm_capacity_bytes;
    m.acct[static_cast<int>(MemoryTier::Warm)].available = true;

    // Hot tier — GPU VRAM via HIP
#ifdef UM790_HAS_HIP
    int dev_count = 0;
    if (hipGetDeviceCount(&dev_count) == hipSuccess && dev_count > 0) {
        m.hot_available = true;
        // Query VRAM size and use configured budget or available VRAM
        std::size_t free_vram = 0, total_vram = 0;
        if (hipMemGetInfo(&free_vram, &total_vram) == hipSuccess) {
            m.acct[static_cast<int>(MemoryTier::Hot)].capacity =
                (cfg.hot_capacity_bytes > 0)
                    ? std::min(cfg.hot_capacity_bytes, total_vram)
                    : total_vram;
        } else {
            m.acct[static_cast<int>(MemoryTier::Hot)].capacity =
                cfg.hot_capacity_bytes;
        }
    }
#endif
    m.acct[static_cast<int>(MemoryTier::Hot)].available = m.hot_available;

    // Cold tier — NVMe spill
    {
        std::error_code ec;
        std::filesystem::create_directories(cfg.cold_spill_dir, ec);
        if (!ec) {
            m.cold_available = true;
            m.acct[static_cast<int>(MemoryTier::Cold)].capacity =
                cfg.cold_capacity_bytes;
        }
    }
    m.acct[static_cast<int>(MemoryTier::Cold)].available = m.cold_available;

    // NOTE: std::format with std::string_view/char* args SEGFAULTs on MinGW GCC 15.
    std::ostringstream oss;
    oss << "[TieredMemory] Hot=" << (m.hot_available ? "GPU" : "off")
        << " Warm=" << (m.warm_available ? "on" : "off")
        << (cxl ? "(CXL)" : "(RAM-fallback)")
        << " Cool=" << (m.cool_available ? "on" : "off")
        << " Cold=" << (m.cold_available ? cfg.cold_spill_dir : "off")
        << "\n";
    std::fputs(oss.str().c_str(), stdout);
}

TieredMemoryManager::~TieredMemoryManager() noexcept {
    if (!impl_) return;
    auto& m = *impl_;

    // Collect all handles first, then free them via free() to avoid
    // double-free and keep PMR accounting symmetric.
    std::vector<TierHandle> handles;
    {
        std::unique_lock lock{m.registry_mutex};
        handles.reserve(m.registry.size());
        for (auto& [handle, alloc] : m.registry) {
            handles.push_back(handle);
        }
    }
    for (auto h : handles) {
        (void)free(h);
    }
}

void TieredMemoryManager::reset_for_testing() noexcept {
    if (!impl_) return;
    auto& m = *impl_;

    std::unique_lock lock{m.registry_mutex};

    // Drain and free everything (same logic as destructor)
    for (auto& [handle, alloc] : m.registry) {
        if (alloc.tier == MemoryTier::Hot) {
            free_hot(alloc.device_ptr);
        } else if (alloc.tier == MemoryTier::Warm) {
            if (alloc.ptr) m.warm_res->deallocate(alloc.ptr, alloc.size_bytes, alloc.alignment);
        } else if (alloc.tier == MemoryTier::Cool) {
            if (alloc.ptr) m.cool_res->deallocate(alloc.ptr, alloc.size_bytes, alloc.alignment);
        } else {
            auto it = m.cold_files.find(handle);
            if (it != m.cold_files.end()) {
                std::error_code ec;
                std::filesystem::remove(it->second, ec);
            }
        }
    }

    m.registry.clear();
    m.lru[0].clear();
    m.lru[1].clear();
    m.lru[2].clear();
    m.lru[3].clear();
    m.cold_files.clear();

    // Reset accounting
    for (auto& acc : m.acct) {
        acc.allocated.store(0, std::memory_order_relaxed);
        acc.peak.store(0, std::memory_order_relaxed);
        acc.alloc_count.store(0, std::memory_order_relaxed);
        acc.free_count.store(0, std::memory_order_relaxed);
        acc.migration_in.store(0, std::memory_order_relaxed);
        acc.migration_out.store(0, std::memory_order_relaxed);
    }
}

TieredMemoryManager::TieredMemoryManager(TieredMemoryManager&&) noexcept = default;
TieredMemoryManager& TieredMemoryManager::operator=(TieredMemoryManager&&) noexcept = default;

// ---------------------------------------------------------------------------
// allocate()
// ---------------------------------------------------------------------------

std::expected<TierAllocation, TierError>
TieredMemoryManager::allocate(std::size_t size_bytes,
                               MemoryTier preferred_tier,
                               std::size_t alignment) {
    if (size_bytes == 0)
        return std::unexpected{TierError::InvalidSize};

    auto& m = *impl_;

    // Walk from preferred tier toward Cold until one succeeds
    constexpr auto kTiers = std::array{
        MemoryTier::Hot, MemoryTier::Warm,
        MemoryTier::Cool, MemoryTier::Cold};

    const int start = static_cast<int>(preferred_tier);

    for (int ti = start; ti < static_cast<int>(kTiers.size()); ++ti) {
        const MemoryTier tier = kTiers[static_cast<std::size_t>(ti)];
        auto& acc = m.acct[ti];

        if (!acc.available) continue;

        // Compute effective alignment for this tier
        std::size_t effective_align = alignment == 0
                            ? [&]() -> std::size_t {
                                switch (tier) {
                                    case MemoryTier::Hot:  return m.cfg.hot_alignment;
                                    case MemoryTier::Warm: return m.cfg.warm_alignment;
                                    case MemoryTier::Cool: return m.cfg.cool_alignment;
                                    case MemoryTier::Cold: return m.cfg.cold_alignment;
                                }
                                return 64;
                              }()
                            : alignment;

        // PMR resources use max(requested, default) — mirror that for capacity check
        std::size_t default_align = 64;
        if (tier == MemoryTier::Warm) default_align = m.warm_res->default_alignment();
        else if (tier == MemoryTier::Cool) default_align = m.cool_res->default_alignment();
        const std::size_t aligned_size = align_up(size_bytes, std::max(effective_align, default_align));

        if (acc.capacity > 0 &&
            acc.allocated.load(std::memory_order_relaxed) + aligned_size > acc.capacity)
            continue;

        TierAllocation alloc{};
        alloc.tier       = tier;
        alloc.size_bytes = size_bytes;
        alloc.alignment  = effective_align;

        bool success = false;

        if (tier == MemoryTier::Hot) {
            alloc.device_ptr = alloc_hot(size_bytes, alloc.alignment);
            if (alloc.device_ptr) success = true;

        } else if (tier == MemoryTier::Warm) {
            try {
                alloc.ptr = m.warm_res->allocate(size_bytes, alloc.alignment);
                success = (alloc.ptr != nullptr);
            } catch (...) {}

        } else if (tier == MemoryTier::Cool) {
            try {
                alloc.ptr = m.cool_res->allocate(size_bytes, alloc.alignment);
                success = (alloc.ptr != nullptr);
            } catch (...) {}

        } else { // Cold
            if (!m.cold_available) continue;

            // Allocate the handle *first* atomically to avoid TOCTOU between
            // filename generation, cold_files key, and registry insertion.
            // This fixes a source of mismatched metadata that manifested as
            // cross-test heap corruption when Cold tier + promote/demote were exercised.
            TierHandle h = m.next_handle.fetch_add(1, std::memory_order_relaxed);

            // Spill to a temp file on NVMe
            // NOTE: std::format crashes GCC 15 with string_view/char* args.
            char hex_buf[24] = {};
            std::to_chars(std::begin(hex_buf), std::end(hex_buf), h, 16);
            auto path = std::filesystem::path{m.cfg.cold_spill_dir} /
                        ("cerberus_" + std::string(hex_buf) + ".spill");
            std::ofstream f{path, std::ios::binary};
            if (f) {
                // Reserve space with seek + write of one zero byte
                f.seekp(static_cast<std::streamoff>(size_bytes - 1));
                f.put('\0');
                f.flush();
                success = f.good();
                if (success) {
                    m.cold_files[h] = path;
                    alloc.ptr = nullptr; // NVMe is not directly addressable
                }
            }

            if (success) {
                alloc.handle = h;
            } else {
                // Roll back the handle we consumed so it can be reused.
                // (best-effort; under heavy load a gap is acceptable)
                m.next_handle.fetch_sub(1, std::memory_order_relaxed);
            }
        }

        if (!success) continue;

        if (alloc.handle == 0) {
            // For non-Cold tiers, assign handle here
            alloc.handle = m.next_handle.fetch_add(1, std::memory_order_relaxed);
        }

        acc.allocated.fetch_add(aligned_size, std::memory_order_relaxed);
        const std::size_t cur = acc.allocated.load(std::memory_order_relaxed);
        std::size_t old_peak = acc.peak.load(std::memory_order_relaxed);
        while (cur > old_peak &&
               !acc.peak.compare_exchange_weak(old_peak, cur,
                                               std::memory_order_relaxed)) {}
        acc.alloc_count.fetch_add(1, std::memory_order_relaxed);

        {
            std::unique_lock lock{m.registry_mutex};
            m.registry.emplace(alloc.handle, alloc);
            m.lru[ti].push_front(alloc.handle);
        }

        return alloc;
    }

    return std::unexpected{TierError::OutOfMemory};
}

// ---------------------------------------------------------------------------
// free()
// ---------------------------------------------------------------------------

std::expected<void, TierError>
TieredMemoryManager::free(TierHandle handle) noexcept {
    auto& m = *impl_;

    std::unique_lock lock{m.registry_mutex};
    auto it = m.registry.find(handle);
    if (it == m.registry.end())
        return std::unexpected{TierError::InvalidHandle};

    TierAllocation alloc = it->second;
    const int ti = static_cast<int>(alloc.tier);

    // Remove from LRU list
    auto& lru_list = m.lru[ti];
    lru_list.remove(handle);

    // Actually free the memory
    if (alloc.tier == MemoryTier::Hot) {
        free_hot(alloc.device_ptr);
    } else if (alloc.tier == MemoryTier::Warm) {
        if (alloc.ptr)
            m.warm_res->deallocate(alloc.ptr, alloc.size_bytes, alloc.alignment);
    } else if (alloc.tier == MemoryTier::Cool) {
        if (alloc.ptr)
            m.cool_res->deallocate(alloc.ptr, alloc.size_bytes, alloc.alignment);
    } else {
        auto cf = m.cold_files.find(handle);
        if (cf != m.cold_files.end()) {
            std::error_code ec;
            std::filesystem::remove(cf->second, ec);
            m.cold_files.erase(cf);
        }
    }

    m.registry.erase(it);
    lock.unlock();

    auto& acc = m.acct[ti];
    // Compute aligned size the same way allocate() did
    std::size_t default_align = 64;
    if (alloc.tier == MemoryTier::Warm && m.warm_res) default_align = m.warm_res->default_alignment();
    else if (alloc.tier == MemoryTier::Cool && m.cool_res) default_align = m.cool_res->default_alignment();
    const std::size_t aligned_size = align_up(alloc.size_bytes, std::max(alloc.alignment, default_align));

    const std::size_t cur = acc.allocated.load(std::memory_order_relaxed);
    if (cur >= aligned_size)
        acc.allocated.fetch_sub(aligned_size, std::memory_order_relaxed);
    else
        acc.allocated.store(0, std::memory_order_relaxed);
    acc.free_count.fetch_add(1, std::memory_order_relaxed);

    return {};
}

// ---------------------------------------------------------------------------
// query()
// ---------------------------------------------------------------------------

std::expected<TierAllocation, TierError>
TieredMemoryManager::query(TierHandle handle) const noexcept {
    auto& m = *impl_;
    std::shared_lock lock{m.registry_mutex};
    auto it = m.registry.find(handle);
    if (it == m.registry.end())
        return std::unexpected{TierError::InvalidHandle};
    return it->second;
}

// ---------------------------------------------------------------------------
// promote() / demote()
// ---------------------------------------------------------------------------

std::expected<TierAllocation, TierError>
TieredMemoryManager::promote(TierHandle handle) {
    auto& m = *impl_;

    std::shared_lock rlock{m.registry_mutex};
    auto it = m.registry.find(handle);
    if (it == m.registry.end())
        return std::unexpected{TierError::InvalidHandle};
    TierAllocation src = it->second;
    rlock.unlock();

    if (src.tier == MemoryTier::Hot) return src; // already hottest

    const MemoryTier target = static_cast<MemoryTier>(
        static_cast<int>(src.tier) - 1);

    auto new_alloc = allocate(src.size_bytes, target, src.alignment);
    if (!new_alloc) return std::unexpected{TierError::MigrationFailed};

    // Copy data with optional in-flight compute
    if (src.tier == MemoryTier::Warm || src.tier == MemoryTier::Cool) {
        if (src.ptr && new_alloc->ptr) {
            if (m.compute_hook) {
                m.compute_hook(src.ptr, new_alloc->ptr, src.size_bytes, src.tier, new_alloc->tier);
            } else {
                std::memcpy(new_alloc->ptr, src.ptr, src.size_bytes);
            }
        }
#ifdef UM790_HAS_HIP
        else if (new_alloc->device_ptr && src.ptr) {
            hipMemcpy(new_alloc->device_ptr, src.ptr, src.size_bytes,
                      hipMemcpyHostToDevice);
        }
#endif
    } else if (src.tier == MemoryTier::Cold) {
        // Load from NVMe spill file
        std::shared_lock lock2{m.registry_mutex};
        auto cf = m.cold_files.find(handle);
        if (cf != m.cold_files.end() && new_alloc->ptr) {
            std::ifstream f{cf->second, std::ios::binary};
            f.read(static_cast<char*>(new_alloc->ptr),
                   static_cast<std::streamsize>(src.size_bytes));
        }
    }

    if (m.on_migrate) m.on_migrate(handle, src.tier, new_alloc->tier);

    auto& old_acc = m.acct[static_cast<int>(src.tier)];
    auto& new_acc = m.acct[static_cast<int>(new_alloc->tier)];
    old_acc.migration_out.fetch_add(1, std::memory_order_relaxed);
    new_acc.migration_in.fetch_add(1, std::memory_order_relaxed);

    // Update registry: replace old entry with new handle pointing to hotter tier
    {
        std::unique_lock lock{m.registry_mutex};
        // Re-look up because lock was dropped
        auto& entry = m.registry[handle];
        const MemoryTier old_tier = entry.tier;

        // Free old physical backing without touching registry yet
        const TierAllocation old_alloc = entry;
        const int old_ti = static_cast<int>(old_tier);
        m.lru[old_ti].remove(handle);
        entry = *new_alloc;
        entry.handle = handle; // keep original handle

        // Free the temporary handle created by allocate()
        m.registry.erase(new_alloc->handle);
        m.lru[static_cast<int>(new_alloc->tier)].remove(new_alloc->handle);

        lock.unlock();

        // Now free old backing
        if (old_alloc.tier == MemoryTier::Hot)       free_hot(old_alloc.device_ptr);
        else if (old_alloc.tier == MemoryTier::Warm && old_alloc.ptr)
            m.warm_res->deallocate(old_alloc.ptr, old_alloc.size_bytes, old_alloc.alignment);
        else if (old_alloc.tier == MemoryTier::Cool && old_alloc.ptr)
            m.cool_res->deallocate(old_alloc.ptr, old_alloc.size_bytes, old_alloc.alignment);
        else {
            std::unique_lock l2{m.registry_mutex};
            auto cf = m.cold_files.find(handle);
            if (cf != m.cold_files.end()) {
                std::error_code ec;
                std::filesystem::remove(cf->second, ec);
                m.cold_files.erase(cf);
            }
        }

        // Compute aligned size for old tier accounting
        std::size_t old_default_align = 64;
        if (old_alloc.tier == MemoryTier::Warm && m.warm_res) old_default_align = m.warm_res->default_alignment();
        else if (old_alloc.tier == MemoryTier::Cool && m.cool_res) old_default_align = m.cool_res->default_alignment();
        const std::size_t old_aligned_size = align_up(old_alloc.size_bytes, std::max(old_alloc.alignment, old_default_align));

        old_acc.allocated.fetch_sub(std::min(old_acc.allocated.load(), old_aligned_size),
                                    std::memory_order_relaxed);
        old_acc.free_count.fetch_add(1, std::memory_order_relaxed);
    }

    return query(handle);
}

std::expected<TierAllocation, TierError>
TieredMemoryManager::demote(TierHandle handle) {
    auto& m = *impl_;

    std::shared_lock rlock{m.registry_mutex};
    auto it = m.registry.find(handle);
    if (it == m.registry.end())
        return std::unexpected{TierError::InvalidHandle};
    TierAllocation src = it->second;
    rlock.unlock();

    if (src.tier == MemoryTier::Cold) return src; // already coldest

    const MemoryTier target = static_cast<MemoryTier>(
        static_cast<int>(src.tier) + 1);

    if (!tier_available(target))
        return std::unexpected{TierError::TierUnavailable};

    auto new_alloc = allocate(src.size_bytes, target, src.alignment);
    if (!new_alloc) return std::unexpected{TierError::MigrationFailed};

    // Copy data to colder tier
    if (target == MemoryTier::Warm || target == MemoryTier::Cool) {
#ifdef UM790_HAS_HIP
        if (src.device_ptr && new_alloc->ptr) {
            hipMemcpy(new_alloc->ptr, src.device_ptr, src.size_bytes,
                      hipMemcpyDeviceToHost);
        } else
#endif
        if (src.ptr && new_alloc->ptr) {
            if (m.compute_hook) {
                m.compute_hook(src.ptr, new_alloc->ptr, src.size_bytes, src.tier, new_alloc->tier);
            } else {
                std::memcpy(new_alloc->ptr, src.ptr, src.size_bytes);
            }
        }
    } else if (target == MemoryTier::Cold && new_alloc->ptr == nullptr) {
        // Written to spill file — read from source and write
        void* src_data = src.ptr;
        std::unique_ptr<std::byte[]> buf;
#ifdef UM790_HAS_HIP
        if (!src_data && src.device_ptr) {
            buf = std::make_unique<std::byte[]>(src.size_bytes);
            hipMemcpy(buf.get(), src.device_ptr, src.size_bytes,
                      hipMemcpyDeviceToHost);
            src_data = buf.get();
        }
#endif
        std::unique_lock l2{m.registry_mutex};
        auto cf = m.cold_files.find(new_alloc->handle);
        if (cf != m.cold_files.end() && src_data) {
            std::ofstream f{cf->second, std::ios::binary | std::ios::trunc};
            f.write(static_cast<const char*>(src_data),
                    static_cast<std::streamsize>(src.size_bytes));
        }
    }

    if (m.on_migrate) m.on_migrate(handle, src.tier, new_alloc->tier);

    auto& old_acc = m.acct[static_cast<int>(src.tier)];
    auto& new_acc = m.acct[static_cast<int>(new_alloc->tier)];
    old_acc.migration_out.fetch_add(1, std::memory_order_relaxed);
    new_acc.migration_in.fetch_add(1, std::memory_order_relaxed);

    // Atomic registry swap (same pattern as promote())
    {
        std::unique_lock lock{m.registry_mutex};
        auto& entry = m.registry[handle];
        const TierAllocation old_alloc = entry;
        const int old_ti = static_cast<int>(old_alloc.tier);
        m.lru[old_ti].remove(handle);
        entry = *new_alloc;
        entry.handle = handle;
        m.registry.erase(new_alloc->handle);
        m.lru[static_cast<int>(new_alloc->tier)].remove(new_alloc->handle);
        lock.unlock();

        if (old_alloc.tier == MemoryTier::Hot) free_hot(old_alloc.device_ptr);
        else if (old_alloc.tier == MemoryTier::Warm && old_alloc.ptr)
            m.warm_res->deallocate(old_alloc.ptr, old_alloc.size_bytes, old_alloc.alignment);
        else if (old_alloc.tier == MemoryTier::Cool && old_alloc.ptr)
            m.cool_res->deallocate(old_alloc.ptr, old_alloc.size_bytes, old_alloc.alignment);

        // Compute aligned size for old tier accounting
        std::size_t old_default_align = 64;
        if (old_alloc.tier == MemoryTier::Warm && m.warm_res) old_default_align = m.warm_res->default_alignment();
        else if (old_alloc.tier == MemoryTier::Cool && m.cool_res) old_default_align = m.cool_res->default_alignment();
        const std::size_t old_aligned_size = align_up(old_alloc.size_bytes, std::max(old_alloc.alignment, old_default_align));

        old_acc.allocated.fetch_sub(std::min(old_acc.allocated.load(), old_aligned_size),
                                    std::memory_order_relaxed);
        old_acc.free_count.fetch_add(1, std::memory_order_relaxed);
    }

    return query(handle);
}

// ---------------------------------------------------------------------------
// pressure_evict()
// ---------------------------------------------------------------------------

std::expected<std::size_t, TierError>
TieredMemoryManager::pressure_evict(MemoryTier tier, float target_pct) {
    auto& m = *impl_;
    const int ti = static_cast<int>(tier);
    auto& acc = m.acct[ti];

    if (!acc.available) return std::unexpected{TierError::TierUnavailable};
    if (acc.capacity == 0) return 0; // unlimited — nothing to evict

    std::size_t evicted = 0;

    while (true) {
        const float fill = static_cast<float>(acc.allocated.load(std::memory_order_relaxed))
                         / static_cast<float>(acc.capacity);
        if (fill <= target_pct) break;

        // Pop LRU candidate
        TierHandle victim = kInvalidTierHandle;
        {
            std::unique_lock lock{m.registry_mutex};
            auto& lru_list = m.lru[ti];
            if (lru_list.empty()) break;
            victim = lru_list.back();
        }

        if (victim == kInvalidTierHandle) break;

        auto dem = demote(victim);
        if (!dem) break; // demote failed (e.g. Cold tier full)
        ++evicted;
    }

    return evicted;
}

// ---------------------------------------------------------------------------
// Telemetry
// ---------------------------------------------------------------------------

TierStats TieredMemoryManager::stats(MemoryTier tier) const noexcept {
    auto& m = *impl_;
    const int ti = static_cast<int>(tier);
    const auto& acc = m.acct[ti];
    const std::size_t alloc = acc.allocated.load(std::memory_order_relaxed);
    const std::size_t cap   = acc.capacity;
    return TierStats{
        .tier                 = tier,
        .available            = acc.available,
        .capacity_bytes       = cap,
        .allocated_bytes      = alloc,
        .peak_allocated_bytes = acc.peak.load(std::memory_order_relaxed),
        .alloc_count          = acc.alloc_count.load(std::memory_order_relaxed),
        .free_count           = acc.free_count.load(std::memory_order_relaxed),
        .migration_in         = acc.migration_in.load(std::memory_order_relaxed),
        .migration_out        = acc.migration_out.load(std::memory_order_relaxed),
        .fill_pct             = (cap > 0)
                                 ? (static_cast<float>(alloc) / static_cast<float>(cap) * 100.0f)
                                 : 0.0f,
    };
}

std::vector<TierStats> TieredMemoryManager::all_stats() const {
    return {
        stats(MemoryTier::Hot),
        stats(MemoryTier::Warm),
        stats(MemoryTier::Cool),
        stats(MemoryTier::Cold),
    };
}

bool TieredMemoryManager::tier_available(MemoryTier tier) const noexcept {
    return impl_->acct[static_cast<int>(tier)].available;
}

// ---------------------------------------------------------------------------
// PMR resource accessors
// ---------------------------------------------------------------------------

std::pmr::memory_resource* TieredMemoryManager::cool_resource() noexcept {
    return impl_->cool_res.get();
}

std::pmr::memory_resource* TieredMemoryManager::warm_resource() noexcept {
    return impl_->warm_res.get();
}

// ===========================================================================
// Migration compute hook + pressure prediction
// ===========================================================================

std::expected<void, std::string>
TieredMemoryManager::register_compute_hook(MigrationComputeHook hook) {
    if (!hook) return std::unexpected{"null hook"};
    impl_->compute_hook = std::move(hook);
    return {};
}

std::size_t TieredMemoryManager::predict_pressure(MemoryTier tier) const noexcept {
    auto& m = *impl_;
    const int ti = static_cast<int>(tier);
    if (ti < 0 || ti >= 4) return 0;
    std::size_t cap = m.acct[ti].capacity;
    std::size_t used = m.acct[ti].allocated.load(std::memory_order_relaxed);
    return used >= cap ? 0 : cap - used;
}

} // namespace hq
