#pragma once
/// @file cerberus_fused_kernels.hpp
/// @copyright Copyright (c) 2026 D Hargreaves. All rights reserved.
///
/// Fused native kernels — key to high NPU utilization.
///
/// The most common pattern in neural networks is:
///   z = x * y + bias   (Mul followed by Add)
///
/// Fusing these into a single FMA loop eliminates intermediate memory
/// traffic and increases arithmetic intensity. This is the kind of
/// bespoke optimisation that separates Cerberus from wrapper frameworks.
///
/// @version 1.0.0

#include <cstddef>
#include <expected>
#include <string>

namespace hq::cerberus::native {

/// @brief Fused multiply-add: out[i] = a[i] * b[i] + c[i]
/// This is the backbone of affine transformations and scale+shift.
[[nodiscard]] std::expected<void, std::string> kernel_fma(
    const float* a, const float* b, const float* c, float* out,
    std::size_t elems);

/// @brief Cache-blocked MatMul with micro-kernel tiling.
/// Target: achieve >70% peak FP32 on modern CPUs (AVX-512 friendly).
[[nodiscard]] std::expected<void, std::string> kernel_matmul_blocked(
    const float* A, const float* B, float* C,
    std::size_t M, std::size_t N, std::size_t K);

} // namespace hq::cerberus::native
