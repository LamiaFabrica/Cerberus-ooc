/// @file cerberus_fused_kernels.cpp
/// @copyright Copyright (c) 2026 D Hargreaves. All rights reserved.
///
/// Fused native kernel implementations.
///
/// @version 1.0.0

#include "hq/cerberus_fused_kernels.hpp"
#include <cstring>
#include <cmath>
#include <vector>

namespace hq::cerberus::native {

// ===========================================================================
// Fused multiply-add: out = a * b + c
// ===========================================================================

std::expected<void, std::string> kernel_fma(
    const float* a, const float* b, const float* c, float* out,
    std::size_t elems) {
    if (!a || !b || !c || !out)
        return std::unexpected{"null pointer"};
    if (elems == 0)
        return std::unexpected{"zero size"};

    for (std::size_t i = 0; i < elems; ++i) {
        out[i] = a[i] * b[i] + c[i];
    }
    return {};
}

// ===========================================================================
// Cache-blocked MatMul with micro-kernel tiling
//
// Tiling strategy:
//   - Block C into MC x NC tiles
//   - Block A into MC x KC panels
//   - Block B into KC x NC panels
//   - Micro-kernel: 4x4 register block
//
// This is the classic GotoBLAS / OpenBLAS approach scaled down for
// embedded / NPU host processors.  The blocks are small enough to fit
// in L1 but large enough to amortise loop overhead.
// ===========================================================================

constexpr std::size_t MC = 64;   // rows of C per block
constexpr std::size_t NC = 64;   // cols of C per block
constexpr std::size_t KC = 64;   // panel depth
constexpr std::size_t MR = 4;    // register block rows
constexpr std::size_t NR = 4;    // register block cols

static void micro_kernel_4x4(
    const float* A_panel, const float* B_panel,
    float* C_block, std::size_t ldc,
    std::size_t kc, std::size_t mr, std::size_t nr, std::size_t nc) {
    float c_reg[MR * NR] = {};

    for (std::size_t k = 0; k < kc; ++k) {
        for (std::size_t i = 0; i < mr; ++i) {
            for (std::size_t j = 0; j < nr; ++j) {
                c_reg[i * NR + j] += A_panel[i * kc + k] * B_panel[k * nc + j];
            }
        }
    }

    for (std::size_t i = 0; i < mr; ++i) {
        for (std::size_t j = 0; j < nr; ++j) {
            C_block[i * ldc + j] += c_reg[i * NR + j];
        }
    }
}

std::expected<void, std::string> kernel_matmul_blocked(
    const float* A, const float* B, float* C,
    std::size_t M, std::size_t N, std::size_t K) {
    if (!A || !B || !C)
        return std::unexpected{"null pointer"};
    if (M == 0 || N == 0 || K == 0)
        return std::unexpected{"empty dimension"};

    // Zero C
    std::memset(C, 0, M * N * sizeof(float));

    // Blocked iteration
    for (std::size_t j = 0; j < N; j += NC) {
        std::size_t nc = std::min(NC, N - j);
        for (std::size_t k = 0; k < K; k += KC) {
            std::size_t kc = std::min(KC, K - k);
            for (std::size_t i = 0; i < M; i += MC) {
                std::size_t mc = std::min(MC, M - i);

                // Pack B panel: [kc x nc] row-major
                std::vector<float> B_panel(kc * nc);
                for (std::size_t kk = 0; kk < kc; ++kk) {
                    for (std::size_t jj = 0; jj < nc; ++jj) {
                        B_panel[kk * nc + jj] = B[(k + kk) * N + (j + jj)];
                    }
                }

                // Pack A panel: [mc x kc] row-major
                std::vector<float> A_panel(mc * kc);
                for (std::size_t ii = 0; ii < mc; ++ii) {
                    for (std::size_t kk = 0; kk < kc; ++kk) {
                        A_panel[ii * kc + kk] = A[(i + ii) * K + (k + kk)];
                    }
                }

                // Micro-kernel loop
                for (std::size_t ii = 0; ii < mc; ii += MR) {
                    std::size_t mr = std::min(MR, mc - ii);
                    for (std::size_t jj = 0; jj < nc; jj += NR) {
                        std::size_t nr = std::min(NR, nc - jj);
                        micro_kernel_4x4(
                            A_panel.data() + ii * kc,
                            B_panel.data() + 0,
                            C + (i + ii) * N + (j + jj),
                            N,
                            kc, mr, nr, nc);
                    }
                }
            }
        }
    }
    return {};
}

// ===========================================================================
// Conv+ReLU stub
// ===========================================================================

std::expected<void, std::string> kernel_conv_relu_stub(
    const float*, const float*, const float*, float*,
    std::size_t, std::size_t, std::size_t,
    std::size_t, std::size_t, std::size_t, std::size_t) {
    return std::unexpected{"kernel_conv_relu_stub not yet implemented"};
}

} // namespace hq::cerberus::native
