/// @file staging_manager.cpp
/// @copyright Copyright © 2026, all rights reserved | David Hargreaves aka Roylepython trading as The Medusa Initiative & Lamia Fabrica
/// Support: https://www.patreon.com/TheMedusaInitiative — £25/month removes ads from all software at 200 subscribers
/// EmbeddingStagingManager implementation — pinned buffer pool management.

#include "hq/staging_manager.hpp"
#include "hq/cxx26_features.hpp"

#include <algorithm>
#include <cstring>
#include <mutex>
#if UM790_HAS_STD_PRINT
#  include <print>
#endif
#include <queue>
#include <vector>
#include <format>

extern "C" std::size_t hq_safe_write(int fd, const char* data, std::size_t len);

#if defined(UM790_HAS_HIP) || defined(__HIP_PLATFORM_AMD__) || defined(USE_HIP)
#  include <hip/hip_runtime.h>
#  define STAGING_HAS_HIP 1
#else
#  define STAGING_HAS_HIP 0
#endif

namespace hq {

struct HostBuffer {
    std::vector<std::byte> vec;
    void* pinned_ptr = nullptr;
    std::size_t buf_size = 0;

    explicit HostBuffer(std::size_t size)
        : vec(size), buf_size(size) {}

    HostBuffer(void* hip_ptr, std::size_t size)
        : pinned_ptr(hip_ptr), buf_size(size) {}

    std::byte* data() {
        return pinned_ptr ? static_cast<std::byte*>(pinned_ptr)
                          : vec.data();
    }
    std::size_t size() const { return buf_size; }
    bool empty() const { return pinned_ptr == nullptr && vec.empty(); }
};

class EmbeddingStagingManager::Impl {
public:
    StagingConfig cfg_;
    std::vector<HostBuffer> buffers_;
    std::queue<std::size_t> free_indices_;
    mutable std::mutex mtx_;

    explicit Impl(StagingConfig cfg);
    ~Impl();

    std::size_t available() const;
};

EmbeddingStagingManager::Impl::Impl(StagingConfig cfg)
    : cfg_{std::move(cfg)}
{
    std::string msg1 = "[DEBUG] Impl ctor start, buffer_count=" + std::to_string(cfg_.buffer_count) + "\n";
    hq_safe_write(1, msg1.data(), msg1.size());
    buffers_.reserve(cfg_.buffer_count);
    std::string msg2 = "[DEBUG] After reserve\n";
    hq_safe_write(1, msg2.data(), msg2.size());
#if STAGING_HAS_HIP
    if (cfg_.pinned) {
        for (std::size_t i = 0; i < cfg_.buffer_count; ++i) {
            void* ptr = nullptr;
            hipError_t err = hipHostMalloc(&ptr, cfg_.buffer_size_bytes,
                                           hipHostMallocPortable);
            if (err == hipSuccess && ptr) {
                buffers_.emplace_back(ptr, cfg_.buffer_size_bytes);
                free_indices_.push(i);
            } else {
                std::string msg = "[staging] WARNING: hipHostMalloc failed, falling back to regular memory\n";
                hq_safe_write(1, msg.data(), msg.size());
                for (auto& b : buffers_) {
                    hipHostFree(b.pinned_ptr);
                    b.pinned_ptr = nullptr;
                }
                buffers_.clear();
                while (!free_indices_.empty()) {
                    free_indices_.pop();
                }
                break;
            }
        }
        if (!buffers_.empty()) {
            return;
        }
    }
#else
    if (cfg_.pinned) {
        std::string msg = "[staging] WARNING: pinned=true but HIP not available at compile time, using regular memory\n";
        hq_safe_write(1, msg.data(), msg.size());
    }
#endif
    std::string msg3 = "[DEBUG] Before loop, buffer_count=" + std::to_string(cfg_.buffer_count) + "\n";
    hq_safe_write(1, msg3.data(), msg3.size());
    for (std::size_t i = 0; i < cfg_.buffer_count; ++i) {
        std::string msg4 = "[DEBUG] Loop i=" + std::to_string(i) + "\n";
        hq_safe_write(1, msg4.data(), msg4.size());
        buffers_.emplace_back(cfg_.buffer_size_bytes);
        std::string msg5 = "[DEBUG] After emplace i=" + std::to_string(i) + "\n";
        hq_safe_write(1, msg5.data(), msg5.size());
        free_indices_.push(i);
        std::string msg6 = "[DEBUG] After push i=" + std::to_string(i) + "\n";
        hq_safe_write(1, msg6.data(), msg6.size());
    }
    std::string msg7 = "[DEBUG] Impl ctor end\n";
    hq_safe_write(1, msg7.data(), msg7.size());
}

EmbeddingStagingManager::Impl::~Impl() {
#if STAGING_HAS_HIP
    for (auto& buf : buffers_) {
        if (buf.pinned_ptr) {
            hipHostFree(buf.pinned_ptr);
            buf.pinned_ptr = nullptr;
        }
    }
#endif
}

std::size_t EmbeddingStagingManager::Impl::available() const {
    std::lock_guard lock{mtx_};
    return free_indices_.size();
}

EmbeddingStagingManager::EmbeddingStagingManager(StagingConfig cfg)
{
    std::string msg = "[DEBUG] EmbeddingStagingManager ctor before make_unique\n";
    hq_safe_write(1, msg.data(), msg.size());
    impl_ = std::make_unique<Impl>(std::move(cfg));
    std::string msg2 = "[DEBUG] EmbeddingStagingManager ctor after make_unique\n";
    hq_safe_write(1, msg2.data(), msg2.size());
}

EmbeddingStagingManager::~EmbeddingStagingManager() = default;

EmbeddingStagingManager::EmbeddingStagingManager(EmbeddingStagingManager&&) noexcept = default;
EmbeddingStagingManager& EmbeddingStagingManager::operator=(EmbeddingStagingManager&&) noexcept = default;

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
        .capacity = buf.size(),
        .used     = 0,
    };
}

void EmbeddingStagingManager::release(const StagingBuffer& buf) noexcept {
    std::lock_guard lock{impl_->mtx_};
    for (std::size_t i = 0; i < impl_->buffers_.size(); ++i) {
        if (!impl_->buffers_[i].empty() &&
            impl_->buffers_[i].data() == buf.data.data()) {
            impl_->free_indices_.push(i);
            return;
        }
    }
    hq_safe_write(2, "[staging] WARNING: release() called with unrecognized buffer pointer\n", 66);
}

std::expected<std::size_t, HostStagingError>
EmbeddingStagingManager::copy_in(StagingBuffer& dst,
                                  std::span<const std::byte> src) {
    const std::size_t to_copy = std::min(src.size(), dst.capacity);
    std::memcpy(dst.data.data(), src.data(), to_copy);
    dst.used = to_copy;
    return to_copy;
}

std::size_t EmbeddingStagingManager::total_capacity() const noexcept {
    return impl_->cfg_.buffer_count * impl_->cfg_.buffer_size_bytes;
}

std::size_t EmbeddingStagingManager::available_count() const noexcept {
    return impl_->available();
}

} // namespace hq
