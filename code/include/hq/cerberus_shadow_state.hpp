#pragma once
/// @file cerberus_shadow_state.hpp
/// @copyright Copyright (c) 2026 D Hargreaves. All rights reserved.
///
/// ShadowState — compressed u8 snapshots of Warm-tier tensors for rollback.
/// Stores quantized representation + FNV hash in Cold tier.
///
/// @version 1.0.0

#include "hq/tiered_memory_manager.hpp"

#include <cstdint>
#include <cstddef>
#include <vector>
#include <optional>

namespace hq::cerberus {

struct ShadowSnapshot {
    float scale{1.0f};
    std::int32_t zero_point{0};
    std::uint64_t hash{0};
    std::vector<std::uint8_t> compressed;
    std::size_t original_elems{0};

    [[nodiscard]] bool valid() const noexcept {
        return !compressed.empty() && hash != 0 && scale > 0.0f;
    }
};

/// Compress a float buffer to u8 and store metadata.
[[nodiscard]] ShadowSnapshot
compress_to_shadow(const float* src, std::size_t n) noexcept;

/// Decompress shadow back to float, writing into dst.
[[nodiscard]] bool
restore_from_shadow(const ShadowSnapshot& snap, float* dst, std::size_t n) noexcept;

} // namespace hq::cerberus
