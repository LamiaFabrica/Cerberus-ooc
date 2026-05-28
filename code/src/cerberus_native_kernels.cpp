/// @file cerberus_native_kernels.cpp
/// @copyright Copyright © 2026, all rights reserved | David Hargreaves aka Roylepython trading as The Medusa Initiative & Lamia Fabrica
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

// ===========================================================================
// Activations
// ===========================================================================

std::expected<void, std::string> kernel_relu(
    const float* in, float* out, std::size_t elems) {
    if (!in || !out) return std::unexpected{"null pointer"};
    for (std::size_t i = 0; i < elems; ++i)
        out[i] = in[i] > 0.0f ? in[i] : 0.0f;
    return {};
}

std::expected<void, std::string> kernel_sigmoid(
    const float* in, float* out, std::size_t elems) {
    if (!in || !out) return std::unexpected{"null pointer"};
    for (std::size_t i = 0; i < elems; ++i)
        out[i] = 1.0f / (1.0f + std::exp(-in[i]));
    return {};
}

std::expected<void, std::string> kernel_softmax(
    const float* in, float* out,
    std::size_t rows, std::size_t cols) {
    if (!in || !out) return std::unexpected{"null pointer"};
    for (std::size_t r = 0; r < rows; ++r) {
        float mx = in[r * cols];
        for (std::size_t c = 1; c < cols; ++c)
            if (in[r*cols+c] > mx) mx = in[r*cols+c];
        float sum = 0;
        for (std::size_t c = 0; c < cols; ++c) {
            out[r*cols+c] = std::exp(in[r*cols+c] - mx);
            sum += out[r*cols+c];
        }
        for (std::size_t c = 0; c < cols; ++c)
            out[r*cols+c] /= sum;
    }
    return {};
}

// Approximate GELU with tanh approximation
std::expected<void, std::string> kernel_gelu(
    const float* in, float* out, std::size_t elems) {
    if (!in || !out) return std::unexpected{"null pointer"};
    constexpr float sqrt_2_over_pi = 0.7978845608f;
    constexpr float coef = 0.044715f;
    for (std::size_t i = 0; i < elems; ++i) {
        float x = in[i];
        float x3 = x * x * x;
        float t = sqrt_2_over_pi * (x + coef * x3);
        out[i] = 0.5f * x * (1.0f + std::tanh(t));
    }
    return {};
}

std::expected<void, std::string> kernel_layernorm(
    const float* in, float* out,
    std::size_t rows, std::size_t cols, float eps) {
    if (!in || !out) return std::unexpected{"null pointer"};
    for (std::size_t r = 0; r < rows; ++r) {
        float mean = 0;
        for (std::size_t c = 0; c < cols; ++c) mean += in[r*cols+c];
        mean /= static_cast<float>(cols);
        float var = 0;
        for (std::size_t c = 0; c < cols; ++c) {
            float d = in[r*cols+c] - mean;
            var += d * d;
        }
        var /= static_cast<float>(cols);
        float inv_std = 1.0f / std::sqrt(var + eps);
        for (std::size_t c = 0; c < cols; ++c)
            out[r*cols+c] = (in[r*cols+c] - mean) * inv_std;
    }
    return {};
}

// ===========================================================================
// Conv2D reference (no padding, stride=1)
// ===========================================================================

std::expected<void, std::string> kernel_conv2d(
    const float* input, const float* weight,
    const float* bias, float* output,
    std::size_t H, std::size_t W, std::size_t C,
    std::size_t KH, std::size_t KW,
    std::size_t OC) {
    if (!input || !weight || !output) return std::unexpected{"null pointer"};
    if (H < KH || W < KW) return std::unexpected{"input smaller than kernel"};

    std::size_t OH = H - KH + 1;
    std::size_t OW = W - KW + 1;

    for (std::size_t oc = 0; oc < OC; ++oc) {
        for (std::size_t oh = 0; oh < OH; ++oh) {
            for (std::size_t ow = 0; ow < OW; ++ow) {
                float acc = 0;
                for (std::size_t kh = 0; kh < KH; ++kh) {
                    for (std::size_t kw = 0; kw < KW; ++kw) {
                        for (std::size_t c = 0; c < C; ++c) {
                            float iv = input[(oh + kh) * W * C + (ow + kw) * C + c];
                            float wv = weight[kh * KW * C * OC + kw * C * OC + c * OC + oc];
                            acc += iv * wv;
                        }
                    }
                }
                if (bias) acc += bias[oc];
                output[oh * OW * OC + ow * OC + oc] = acc;
            }
        }
    }
    return {};
}

} // namespace hq::cerberus::native
