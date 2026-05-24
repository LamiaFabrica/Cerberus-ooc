#pragma once
/// @file cerberus_quantized_kernels.hpp
/// @copyright Copyright (c) 2026 D Hargreaves. All rights reserved.
///
/// Quantized native kernels — INT8 / INT4 execution paths.
///
/// QuantProfile, QuantMethod, QuantGranularity are now in npu_backend_unified.hpp
/// so they are available to all graph nodes, decision engine, and kernels.
///
/// @version 1.1.0

#include "hq/npu_backend_unified.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>

namespace hq::cerberus::native {

// ===========================================================================
// QuantParams — per-invocation scale + zero-point
// ===========================================================================

struct QuantParams {
    float scale_a{1.0f};
    float scale_b{1.0f};
    float scale_c{1.0f};   ///< output scale = scale_a * scale_b / scale_c
    std::int32_t zero_point_a{0};
    std::int32_t zero_point_b{0};
};

// ===========================================================================
// INT8 quantized MatMul (uint8_t storage, asymmetric range 0..255)
// ===========================================================================

[[nodiscard]] std::expected<void, std::string> kernel_matmul_int8(
    const std::uint8_t* A, const std::uint8_t* B, float* C_out,
    std::size_t M, std::size_t N, std::size_t K,
    const QuantParams& q);

// ===========================================================================
// Dynamic quantization: inputs are float, internally quantized to INT8,
// accumulated in INT32, then dequantized back to float.
// ===========================================================================

[[nodiscard]] std::expected<void, std::string> kernel_matmul_dynamic_quant(
    const float* A, const float* B, float* C,
    std::size_t M, std::size_t N, std::size_t K);

// ===========================================================================
// Fused uint8_t MatMul + Bias + ReLU
// ===========================================================================

[[nodiscard]] std::expected<void, std::string> kernel_matmul_bias_relu_int8(
    const std::int8_t* A, const std::int8_t* B,
    const float* bias,
    float* C_out,
    std::size_t M, std::size_t N, std::size_t K,
    const QuantParams& q);

// ===========================================================================
// Quantization helpers (exposed for tests)
// ===========================================================================

std::pair<float, float> compute_minmax(const float* src, std::size_t n);
float compute_scale(float min_val, float max_val);
std::int32_t compute_zero_point(float min_val, float scale);

// --- per-channel ---
void quantize_per_channel(const float* weight, std::uint8_t* out,
                          int out_channels, int in_channels,
                          float* scales, std::int32_t* zero_points);

void dequantize_per_channel(const float* src, float* dst,
                            int out_channels, int in_channels,
                            const float* scales, const std::int32_t* zps);

// ===========================================================================
// Dequantization kernel — turns uint8_t storage back to float during migration
// ===========================================================================

[[nodiscard]] std::expected<void, std::string>
dequantize_u8_to_f32(const std::uint8_t* src, float* dst, std::size_t n,
                     float scale, std::int32_t zero_point);

} // namespace hq::cerberus::native
