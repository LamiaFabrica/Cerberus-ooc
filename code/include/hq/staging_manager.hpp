#pragma once
/// @file staging_manager.hpp
/// @copyright Copyright (c) 2026 D Hargreaves (AKA Roylepython). LamiaFabrica. All rights reserved.
/// Pinned staging buffer management for zero-copy CPU↔GPU transfers.

#include "hq/cxx26_features.hpp"

#include <cstdint>
#if UM790_HAS_STD_EXPECTED
#  include <expected>
#else
#  error "staging_manager.hpp requires std::expected (<expected>) — GCC >= 14 or Clang >= 18 with C++26 enabled"
#endif
#include <memory>
#include <span>
#include <string>
#include <vector>

#if UM790_HAS_STD_MDSPAN
#  include <mdspan>
#endif

namespace hq {

// ---------------------------------------------------------------------------
/// @brief Staging buffer descriptor.
// ---------------------------------------------------------------------------
struct StagingBuffer {
    std::span<std::byte>    data;      ///< Writable view
    std::size_t             capacity{0};  ///< Total allocated bytes
    std::size_t             used{0};      ///< Bytes currently in use
};

// ---------------------------------------------------------------------------
/// @brief Staging manager errors.
// ---------------------------------------------------------------------------
enum class HostStagingError : std::uint8_t {
    OutOfMemory     = 0,
    InvalidSize     = 1,
    TransferFailed  = 2,
    PoolExhausted   = 3,
    NotAligned      = 4,
    Unknown         = 5,
};

// ---------------------------------------------------------------------------
/// @brief Configuration for the staging buffer pool.
// ---------------------------------------------------------------------------
struct StagingConfig {
    std::size_t buffer_count{8};       ///< Number of buffers in pool
    std::size_t buffer_size_bytes{64ULL * 1024 * 1024};  ///< 64 MiB each
    bool        pinned{true};          ///< Use page-locked (pinned) memory
    std::size_t alignment{256};        ///< Buffer alignment in bytes
};

// ---------------------------------------------------------------------------
/// @class EmbeddingStagingManager
/// @brief Manages pinned CPU staging buffers for embedding transfers.
///
/// Provides a pool of page-locked buffers used to stage embedding tensors
/// between the Hailo encoder output and the GPU denoising input. Zero-copy
/// semantics when used with hipHostRegister/hipMemcpyAsync.
// ---------------------------------------------------------------------------
class EmbeddingStagingManager {
public:
    explicit EmbeddingStagingManager(StagingConfig cfg);
    ~EmbeddingStagingManager();

    // Non-copyable
    EmbeddingStagingManager(const EmbeddingStagingManager&) = delete;
    EmbeddingStagingManager& operator=(const EmbeddingStagingManager&) = delete;

    EmbeddingStagingManager(EmbeddingStagingManager&&) noexcept;
    EmbeddingStagingManager& operator=(EmbeddingStagingManager&&) noexcept;

    /// @brief Acquire a free buffer from the pool.
    /// @return StagingBuffer descriptor or error if pool exhausted.
    [[nodiscard]] std::expected<StagingBuffer, HostStagingError> acquire();

    /// @brief Return a buffer to the pool.
    /// @param buf Buffer previously acquired via acquire().
    void release(const StagingBuffer& buf) noexcept;

    /// @brief Copy data into a staging buffer (host-side memcpy).
    /// @param dst Target staging buffer (must be acquired).
    /// @param src Source data to copy.
    /// @return Bytes actually copied or error.
    [[nodiscard]] std::expected<std::size_t, HostStagingError>
        copy_in(StagingBuffer& dst, std::span<const std::byte> src);

    /// @brief Get total pool capacity in bytes.
    [[nodiscard]] std::size_t total_capacity() const noexcept;

    /// @brief Get number of buffers currently available.
    [[nodiscard]] std::size_t available_count() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// ---------------------------------------------------------------------------
/// @brief A simplified mdspan view over a staging buffer (C++26).
// ---------------------------------------------------------------------------
#if UM790_HAS_STD_MDSPAN
template<typename T>
using staging_view = std::mdspan<T, std::dextents<std::size_t, 2>>;
#endif

// ---------------------------------------------------------------------------
/// @brief Convenience free function for error formatting.
// ---------------------------------------------------------------------------
[[nodiscard]] inline std::string to_string(HostStagingError e) {
    switch (e) {
        case HostStagingError::OutOfMemory:    return "OutOfMemory";
        case HostStagingError::InvalidSize:    return "InvalidSize";
        case HostStagingError::TransferFailed: return "TransferFailed";
        case HostStagingError::PoolExhausted:  return "PoolExhausted";
        case HostStagingError::NotAligned:     return "NotAligned";
        case HostStagingError::Unknown:        return "Unknown";
    }
    return "Unknown";
}

} // namespace hq
