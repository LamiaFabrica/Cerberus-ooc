/// @file staging_manager.cpp
/// EmbeddingStagingManager implementation — pinned buffer pool management.

#include "hq/staging_manager.hpp"
#include "hq/cxx26_features.hpp"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <print>
#include <queue>
#include <vector>

#if defined(__HIP_PLATFORM_AMD__) || defined(USE_HIP)
#  include <hip/hip_runtime.h>
#  define STAGING_HAS_HIP 1
#else
#  define STAGING_HAS_HIP 0
#endif

namespace hq {

// ===========================================================================
// HostBuffer — owns either std::vector memory or HIP pinned memory
// ===========================================================================
struct HostBuffer {
    std::vector<std::byte> vec;
    void* pinned_ptr = nullptr;
    std::size_t buf_size = 0;

    explicit HostBuffer(std::size_t size)
        : vec(size), buf_size(size) {}

    HostBuffer(void* hip_ptr, std::size_t size)
        : pinned_ptr(hip_ptr), buf_size(size) {}

    std::byte* data() const {
        return pinned_ptr ? static_cast<std::byte*>(pinned_ptr)
                          : const_cast<std::byte*>(vec.data());
    }
    std::size_t size() const { return buf_size; }
    bool empty() const { return data() == nullptr; }
};

// ===========================================================================
// Private implementation
// ===========================================================================
class EmbeddingStagingManager::Impl {
public:
    StagingConfig cfg_;
    std::vector<HostBuffer> buffers_;
    std::queue<std::size_t> free_indices_;
    mutable std::mutex mtx_;

    explicit Impl(StagingConfig cfg)
        : cfg_{std::move(cfg)}
    {
        buffers_.reserve(cfg_.buffer_count);
        for (std::size_t i = 0; i < cfg_.buffer_count; ++i) {
#if STAGING_HAS_HIP
            if (cfg_.pinned) {
                void* ptr = nullptr;
                hipError_t err = hipHostMalloc(&ptr, cfg_.buffer_size_bytes,
                                               hipHostMallocPortable);
                if (err == hipSuccess) {
                    buffers_.emplace_back(ptr, cfg_.buffer_size_bytes);
                    free_indices_.push(i);
                    continue;
                }
                std::print("[staging] WARNING: hipHostMalloc failed ({}), "
                           "falling back to regular memory\n",
                           hipGetErrorString(err));
            }
#else
            if (cfg_.pinned) {
                std::print("[staging] WARNING: pinned=true but HIP not available "
                           "at compile time, using regular memory\n");
            }
#endif
            buffers_.emplace_back(cfg_.buffer_size_bytes);
            free_indices_.push(i);
        }
    }

    ~Impl() {
#if STAGING_HAS_HIP
        for (auto& buf : buffers_) {
            if (buf.pinned_ptr) {
                hipHostFree(buf.pinned_ptr);
            }
        }
#endif
    }

    std::size_t available() const {
        std::lock_guard lock{mtx_};
        return free_indices_.size();
    }
};

// ===========================================================================
// EmbeddingStagingManager
// ===========================================================================

EmbeddingStagingManager::EmbeddingStagingManager(StagingConfig cfg)
    : impl_{std::make_unique<Impl>(std::move(cfg))}
{
}

EmbeddingStagingManager::~EmbeddingStagingManager() = default;

EmbeddingStagingManager::EmbeddingStagingManager(EmbeddingStagingManager&&) noexcept = default;
EmbeddingStagingManager& EmbeddingStagingManager::operator=(EmbeddingStagingManager&&) noexcept = default;

// ---------------------------------------------------------------------------
std::expected<StagingBuffer, HostStagingError> EmbeddingStagingManager::acquire() {
    std::lock_guard lock{impl_->mtx_};

    if (impl_->free_indices_.empty()) {
        return std::unexpected{HostStagingError::PoolExhausted};
    }

    std::size_t idx = impl_->free_indices_.front();
    impl_->free_indices_.pop();

    auto& buf = impl_->buffers_[idx];
    return StagingBuffer{
        .data     = std::span{buf.data(), buf.size()},
        .cdata    = std::span{buf.data(), buf.size()},
        .capacity = buf.size(),
        .used     = 0,
    };
}

// ---------------------------------------------------------------------------
void EmbeddingStagingManager::release(const StagingBuffer& buf) noexcept {
    std::lock_guard lock{impl_->mtx_};

    // Find which buffer index matches the span address
    for (std::size_t i = 0; i < impl_->buffers_.size(); ++i) {
        if (!impl_->buffers_[i].empty() &&
            impl_->buffers_[i].data() == buf.data.data()) {
            impl_->free_indices_.push(i);
            return;
        }
    }
}

// ---------------------------------------------------------------------------
std::expected<std::size_t, HostStagingError>
EmbeddingStagingManager::copy_in(StagingBuffer& dst,
                                  std::span<const std::byte> src) {
    const std::size_t to_copy = std::min(src.size(), dst.capacity);
    std::memcpy(dst.data.data(), src.data(), to_copy);
    dst.used = to_copy;
    return to_copy;
}

// ---------------------------------------------------------------------------
std::size_t EmbeddingStagingManager::total_capacity() const noexcept {
    return impl_->cfg_.buffer_count * impl_->cfg_.buffer_size_bytes;
}

// ---------------------------------------------------------------------------
std::size_t EmbeddingStagingManager::available_count() const noexcept {
    return impl_->available();
}

} // namespace hq
