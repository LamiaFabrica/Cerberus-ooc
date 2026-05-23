/// @file cerberus_native_kernels.cpp
/// @copyright Copyright (c) 2026 D Hargreaves. All rights reserved.
///
/// Native reference kernels — honest CPU fallback + correctness baseline.
///
/// @version 1.0.0

#include "hq/cerberus_native_kernels.hpp"

#include <cstring>
#include <cmath>
#include <span>
#include <expected>
#include <string>
#include <vector>

namespace hq::cerberus::native {

// ===========================================================================
// MatMul: plain reference, row-major A[M×K] × B[K×N] = C[M×N]
// ===========================================================================

std::expected<void, std::string> kernel_matmul(
    const float* A, const float* B, float* C,
    std::size_t M, std::size_t N, std::size_t K) {
    if (!A || !B || !C) return std::unexpected{"null pointer"};
    if (M == 0 || N == 0 || K == 0) return std::unexpected{"empty dimension"};

    // Naïve O(M×N×K), sufficient for correctness validation
    for (std::size_t m = 0; m < M; ++m) {
        for (std::size_t n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (std::size_t k = 0; k < K; ++k) {
                acc += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = acc;
        }
    }
    return {};
}

// ===========================================================================
// Elementwise
// ===========================================================================

std::expected<void, std::string> kernel_add(
    const float* a, const float* b, float* out,
    std::size_t elems) {
    if (!a || !b || !out) return std::unexpected{"null pointer"};
    for (std::size_t i = 0; i < elems; ++i) out[i] = a[i] + b[i];
    return {};
}

std::expected<void, std::string> kernel_mul(
    const float* a, const float* b, float* out,
    std::size_t elems) {
    if (!a || !b || !out) return std::unexpected{"null pointer"};
    for (std::size_t i = 0; i < elems; ++i) out[i] = a[i] * b[i];
    return {};
}

// ===========================================================================
// Dispatcher
// ===========================================================================

std::expected<void, std::string> execute(
    OpType op,
    std::span<const float*> inputs,
    std::span<float*> outputs,
    std::span<const std::int64_t> shapes) {
    switch (op) {
        case OpType::MatMul: {
            if (inputs.size() < 2 || outputs.empty() || shapes.size() < 3)
                return std::unexpected{"MatMul: insufficient args"};
            return kernel_matmul(inputs[0], inputs[1], outputs[0],
                                 static_cast<std::size_t>(shapes[0]),
                                 static_cast<std::size_t>(shapes[1]),
                                 static_cast<std::size_t>(shapes[2]));
        }
        case OpType::Add: {
            if (inputs.size() < 2 || outputs.empty())
                return std::unexpected{"Add: insufficient args"};
            std::size_t sz = (shapes.empty() ? 0 : static_cast<std::size_t>(shapes[0]));
            if (sz == 0) return std::unexpected{"Add: zero size"};
            return kernel_add(inputs[0], inputs[1], outputs[0], sz);
        }
        case OpType::Mul: {
            if (inputs.size() < 2 || outputs.empty())
                return std::unexpected{"Mul: insufficient args"};
            std::size_t sz = (shapes.empty() ? 0 : static_cast<std::size_t>(shapes[0]));
            if (sz == 0) return std::unexpected{"Mul: zero size"};
            return kernel_mul(inputs[0], inputs[1], outputs[0], sz);
        }
        default:
            return std::unexpected{"unsupported OpType"};
    }
}

} // namespace hq::cerberus::native
