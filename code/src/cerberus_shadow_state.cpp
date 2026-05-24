/// @file cerberus_shadow_state.cpp
/// @copyright Copyright (c) 2026 D Hargreaves. All rights reserved.
///
/// ShadowState implementation — dynamic quantization to u8.
///
/// @version 1.0.0

#include "hq/cerberus_shadow_state.hpp"
#include "hq/cerberus_quantized_kernels.hpp"
#include "hq/cerberus_execution_coordinator.hpp"

#include <algorithm>
#include <cmath>

namespace hq::cerberus {

ShadowSnapshot compress_to_shadow(const float* src, std::size_t n) noexcept {
    ShadowSnapshot snap;
    if (n == 0 || !src) return snap;

    float min_val = src[0];
    float max_val = src[0];
    for (std::size_t i = 1; i < n; ++i) {
        if (src[i] < min_val) min_val = src[i];
        if (src[i] > max_val) max_val = src[i];
    }
    if (min_val == max_val) {
        snap.scale = 1.0f;
        snap.zero_point = 0;
        snap.compressed.assign(n, 128);
        snap.hash = hq::fnv1a_bytes(snap.compressed.data(), snap.compressed.size());
        snap.original_elems = n;
        return snap;
    }

    snap.scale = native::compute_scale(min_val, max_val);
    snap.zero_point = native::compute_zero_point(min_val, snap.scale);
    snap.compressed.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        float q = std::round(src[i] / snap.scale) + static_cast<float>(snap.zero_point);
        if (q < 0.0f) q = 0.0f;
        if (q > 255.0f) q = 255.0f;
        snap.compressed[i] = static_cast<std::uint8_t>(static_cast<std::int32_t>(q));
    }
    snap.hash = hq::fnv1a_bytes(snap.compressed.data(), snap.compressed.size());
    snap.original_elems = n;
    return snap;
}

bool restore_from_shadow(const ShadowSnapshot& snap, float* dst, std::size_t n) noexcept {
    if (!snap.valid() || !dst || n != snap.original_elems || n != snap.compressed.size())
        return false;
    // Verify integrity
    auto rehash = hq::fnv1a_bytes(snap.compressed.data(), snap.compressed.size());
    if (rehash != snap.hash) return false;
    for (std::size_t i = 0; i < n; ++i) {
        dst[i] = (static_cast<float>(snap.compressed[i]) - static_cast<float>(snap.zero_point)) * snap.scale;
    }
    return true;
}

} // namespace hq::cerberus
