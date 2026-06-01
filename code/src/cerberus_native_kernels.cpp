/// @file cerberus_native_kernels.cpp
/// @copyright Copyright © 2026, all rights reserved | David Hargreaves aka Roylepython trading as The Medusa Initiative & Lamia Fabrica
/// Support: https://www.patreon.com/TheMedusaInitiative — £25/month removes ads from all software at 200 subscribers
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
#include <ranges>
#include <algorithm>

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
    std::ranges::for_each(
        std::views::cartesian_product(
            std::views::iota(std::size_t{0}, M),
            std::views::iota(std::size_t{0}, N)),
        [&](auto mn) {
            auto [m, n] = mn;
            float acc = 0.0f;
            std::ranges::for_each(
                std::views::iota(std::size_t{0}, K),
                [&](std::size_t k) {
                    acc += A[m * K + k] * B[k * N + n];
                });
            C[m * N + n] = acc;
        });
    return {};
}

// ===========================================================================
// Elementwise
// ===========================================================================

std::expected<void, std::string> kernel_add(
    const float* a, const float* b, float* out,
    std::size_t elems) {
    if (!a || !b || !out) return std::unexpected{"null pointer"};
    std::ranges::transform(
        std::views::zip(std::span(a, elems), std::span(b, elems)),
        out,
        [](auto p) {
            auto [x, y] = p;
            return x + y;
        });
    return {};
}

std::expected<void, std::string> kernel_mul(
    const float* a, const float* b, float* out,
    std::size_t elems) {
    if (!a || !b || !out) return std::unexpected{"null pointer"};
    std::ranges::transform(
        std::views::zip(std::span(a, elems), std::span(b, elems)),
        out,
        [](auto p) {
            auto [x, y] = p;
            return x * y;
        });
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
    std::ranges::transform(
        std::span(in, elems), out,
        [](float x) { return x > 0.0f ? x : 0.0f; });
    return {};
}

std::expected<void, std::string> kernel_sigmoid(
    const float* in, float* out, std::size_t elems) {
    if (!in || !out) return std::unexpected{"null pointer"};
    std::ranges::transform(
        std::span(in, elems), out,
        [](float x) { return 1.0f / (1.0f + std::exp(-x)); });
    return {};
}

std::expected<void, std::string> kernel_softmax(
    const float* in, float* out,
    std::size_t rows, std::size_t cols) {
    if (!in || !out) return std::unexpected{"null pointer"};
    std::ranges::for_each(
        std::views::iota(std::size_t{0}, rows),
        [&](std::size_t r) {
            auto row_in  = std::span(in  + r * cols, cols);
            auto row_out = std::span(out + r * cols, cols);

            float mx = *std::ranges::max_element(row_in);

            std::ranges::transform(row_in, row_out.begin(),
                [mx](float x) { return std::exp(x - mx); });

            float sum = 0.0f;
            std::ranges::for_each(row_out, [&sum](float x) { sum += x; });

            std::ranges::transform(row_out, row_out.begin(),
                [sum](float x) { return x / sum; });
        });
    return {};
}

// Approximate GELU with tanh approximation
std::expected<void, std::string> kernel_gelu(
    const float* in, float* out, std::size_t elems) {
    if (!in || !out) return std::unexpected{"null pointer"};
    constexpr float sqrt_2_over_pi = 0.7978845608f;
    constexpr float coef = 0.044715f;
    std::ranges::transform(
        std::span(in, elems), out,
        [sqrt_2_over_pi, coef](float x) {
            float x3 = x * x * x;
            float t = sqrt_2_over_pi * (x + coef * x3);
            return 0.5f * x * (1.0f + std::tanh(t));
        });
    return {};
}

std::expected<void, std::string> kernel_layernorm(
    const float* in, float* out,
    std::size_t rows, std::size_t cols, float eps) {
    if (!in || !out) return std::unexpected{"null pointer"};
    std::ranges::for_each(
        std::views::iota(std::size_t{0}, rows),
        [&](std::size_t r) {
            auto row_in  = std::span(in  + r * cols, cols);
            auto row_out = std::span(out + r * cols, cols);

            float mean = 0.0f;
            std::ranges::for_each(row_in, [&mean](float x) { mean += x; });
            mean /= static_cast<float>(cols);

            float var = 0.0f;
            std::ranges::for_each(row_in, [&var, mean](float x) {
                float d = x - mean;
                var += d * d;
            });
            var /= static_cast<float>(cols);

            float inv_std = 1.0f / std::sqrt(var + eps);
            std::ranges::transform(row_in, row_out.begin(),
                [mean, inv_std](float x) { return (x - mean) * inv_std; });
        });
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

    std::ranges::fill(std::span(output, OH * OW * OC), 0.0f);

    std::ranges::for_each(
        std::views::cartesian_product(
            std::views::iota(std::size_t{0}, OC),
            std::views::iota(std::size_t{0}, OH),
            std::views::iota(std::size_t{0}, OW),
            std::views::iota(std::size_t{0}, KH),
            std::views::iota(std::size_t{0}, KW),
            std::views::iota(std::size_t{0}, C)),
        [&](auto idx) {
            auto [oc, oh, ow, kh, kw, c] = idx;
            float iv = input[(oh + kh) * W * C + (ow + kw) * C + c];
            float wv = weight[kh * KW * C * OC + kw * C * OC + c * OC + oc];
            output[oh * OW * OC + ow * OC + oc] += iv * wv;
        });

    if (bias) {
        std::ranges::for_each(
            std::views::cartesian_product(
                std::views::iota(std::size_t{0}, OH),
                std::views::iota(std::size_t{0}, OW)),
            [&](auto hwoff) {
                auto [oh, ow] = hwoff;
                for (std::size_t oc = 0; oc < OC; ++oc) {
                    output[oh * OW * OC + ow * OC + oc] += bias[oc];
                }
            });
    }
    return {};
}

} // namespace hq::cerberus::native
