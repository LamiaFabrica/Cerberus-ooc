#pragma once
/// @file cerberus_native_kernels.hpp
/// @copyright Copyright (c) 2026 D Hargreaves (AKA Roylepython). LamiaFabrica. All rights reserved.
///
/// Native reference kernels — Cerberus-owned implementations of core operators.
///
/// These are *not* wrappers around vendor BLAS.  They are plain C++26
/// implementations that serve three purposes:
///   1.  Honest CPU fallback when no NPU is present (synthetic_mode).
///   2.  Correctness baseline for numerical validation against vendor outputs.
///   3.  Execution target for the DecisionEngine when a subgraph is small
///       enough that native dispatch beats vendor launch overhead.
///
/// The kernels use <execution> parallel policies when available but always
/// compile and run with plain scalar loops as the universal fallback.
///
/// @version 1.0.0

#include <cstdint>
#include <cstddef>
#include <span>
#include <expected>
#include <string>
#include <vector>

namespace hq::cerberus::native {

// ===========================================================================
// Op identifiers
// ===========================================================================

enum class OpType : std::uint8_t {
    Unknown = 0,
    MatMul,
    Add,
    Mul,
    Relu,
    Sigmoid,
    Softmax,
    Transpose,
    Reshape,      ///< no-op for contiguous data, just metadata
};

// ===========================================================================
// Kernel dispatcher — the single entry point for native execution
// ===========================================================================

/// @brief Execute a native kernel.
/// @param op         Which operator to run.
/// @param inputs     Pointers to input buffers (all contiguous float).
/// @param outputs    Pointers to output buffers (pre-allocated by caller).
/// @param shapes     Flattened shape descriptions.  For MatMul three shapes.
/// @return Success or an error string.
[[nodiscard]] std::expected<void, std::string>
execute(OpType op,
        std::span<const float*>    inputs,
        std::span<float*>          outputs,
        std::span<const std::int64_t> shapes);

// ===========================================================================
// Individual kernels (exposed for testing / fused subgraphs)
// ===========================================================================

[[nodiscard]] std::expected<void, std::string> kernel_matmul(
    const float* A, const float* B, float* C,
    std::size_t M, std::size_t N, std::size_t K);

[[nodiscard]] std::expected<void, std::string> kernel_matmul_blocked(
    const float* A, const float* B, float* C,
    std::size_t M, std::size_t N, std::size_t K);

#if defined(__x86_64__) || defined(_M_X64) || defined(__amd64__) || defined(__AVX2__)
/// @brief AVX2 blocked MatMul (requires __AVX2__).
[[nodiscard]] std::expected<void, std::string> kernel_matmul_blocked_avx2(
    const float* A, const float* B, float* C,
    std::size_t M, std::size_t N, std::size_t K);
#endif

[[nodiscard]] std::expected<void, std::string> kernel_add(
    const float* a, const float* b, float* out,
    std::size_t elems);

[[nodiscard]] std::expected<void, std::string> kernel_mul(
    const float* a, const float* b, float* out,
    std::size_t elems);

// ===========================================================================
// Activation + normalization kernels
// ===========================================================================

[[nodiscard]] std::expected<void, std::string> kernel_relu(
    const float* in, float* out, std::size_t elems);

[[nodiscard]] std::expected<void, std::string> kernel_sigmoid(
    const float* in, float* out, std::size_t elems);

[[nodiscard]] std::expected<void, std::string> kernel_softmax(
    const float* in, float* out,
    std::size_t rows, std::size_t cols);

[[nodiscard]] std::expected<void, std::string> kernel_gelu(
    const float* in, float* out, std::size_t elems);

[[nodiscard]] std::expected<void, std::string> kernel_layernorm(
    const float* in, float* out,
    std::size_t rows, std::size_t cols,
    float eps = 1e-6f);

// ===========================================================================
// Conv2D (reference)
// ===========================================================================

/// @brief 2D convolution — no padding, stride=1.
[[nodiscard]] std::expected<void, std::string> kernel_conv2d(
    const float* input,   /// [H * W * C]
    const float* weight,  /// [KH * KW * C * OC]
    const float* bias,    /// [OC] (may be nullptr)
    float* output,        /// [OH * OW * OC]
    std::size_t H, std::size_t W, std::size_t C,
    std::size_t KH, std::size_t KW,
    std::size_t OC);

} // namespace hq::cerberus::native
