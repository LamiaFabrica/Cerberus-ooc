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

} // namespace hq::cerberus::native

// ===========================================================================
// AVX2 micro-kernel dispatch (requires __AVX2__)
// ===========================================================================

#if defined(__x86_64__) || defined(_M_X64) || defined(__amd64__)
#if defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>

namespace hq::cerberus::native {

static void avx2_micro_kernel_8x8(
    const float* A_tile, const float* B_tile, float* C_tile,
    std::size_t ldc, std::size_t kc) {
    // A_tile: [kc × 8] row-major  (stride 8)
    // B_tile: [kc × 8] row-major  (stride 8)
    __m256 c0 = _mm256_setzero_ps();
    __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps();
    __m256 c4 = _mm256_setzero_ps();
    __m256 c5 = _mm256_setzero_ps();
    __m256 c6 = _mm256_setzero_ps();
    __m256 c7 = _mm256_setzero_ps();

    for (std::size_t kk = 0; kk < kc; ++kk) {
        __m256 a = _mm256_loadu_ps(A_tile + kk * 8);
        __m256 b0 = _mm256_broadcast_ss(B_tile + kk * 8 + 0);
        __m256 b1 = _mm256_broadcast_ss(B_tile + kk * 8 + 1);
        __m256 b2 = _mm256_broadcast_ss(B_tile + kk * 8 + 2);
        __m256 b3 = _mm256_broadcast_ss(B_tile + kk * 8 + 3);
        __m256 b4 = _mm256_broadcast_ss(B_tile + kk * 8 + 4);
        __m256 b5 = _mm256_broadcast_ss(B_tile + kk * 8 + 5);
        __m256 b6 = _mm256_broadcast_ss(B_tile + kk * 8 + 6);
        __m256 b7 = _mm256_broadcast_ss(B_tile + kk * 8 + 7);
        c0 = _mm256_fmadd_ps(a, b0, c0);
        c1 = _mm256_fmadd_ps(a, b1, c1);
        c2 = _mm256_fmadd_ps(a, b2, c2);
        c3 = _mm256_fmadd_ps(a, b3, c3);
        c4 = _mm256_fmadd_ps(a, b4, c4);
        c5 = _mm256_fmadd_ps(a, b5, c5);
        c6 = _mm256_fmadd_ps(a, b6, c6);
        c7 = _mm256_fmadd_ps(a, b7, c7);
    }

    alignas(32) float tmp[8][8];
    _mm256_storeu_ps(tmp[0], c0); _mm256_storeu_ps(tmp[1], c1);
    _mm256_storeu_ps(tmp[2], c2); _mm256_storeu_ps(tmp[3], c3);
    _mm256_storeu_ps(tmp[4], c4); _mm256_storeu_ps(tmp[5], c5);
    _mm256_storeu_ps(tmp[6], c6); _mm256_storeu_ps(tmp[7], c7);
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            C_tile[i * ldc + j] += tmp[j][i];
}

std::expected<void, std::string> kernel_matmul_blocked_avx2(
    const float* A, const float* B, float* C,
    std::size_t M, std::size_t N, std::size_t K) {
    if (!A || !B || !C)
        return std::unexpected{"null pointer"};
    if (M == 0 || N == 0 || K == 0)
        return std::unexpected{"empty dimension"};
    if (M % 8 != 0 || N % 8 != 0) {
        return kernel_matmul_blocked(A, B, C, M, N, K);
    }

    std::memset(C, 0, M * N * sizeof(float));
    constexpr std::size_t KC = 128;
    constexpr std::size_t NC = 256;
    constexpr std::size_t MC = 128;

    for (std::size_t j = 0; j < N; j += NC) {
        std::size_t nc = (std::min(NC, N - j) / 8) * 8;
        for (std::size_t k = 0; k < K; k += KC) {
            std::size_t kc = std::min(KC, K - k);
            for (std::size_t i = 0; i < M; i += MC) {
                std::size_t mc = (std::min(MC, M - i) / 8) * 8;
                for (std::size_t ii = 0; ii < mc; ii += 8) {
                    for (std::size_t jj = 0; jj < nc; jj += 8) {
                        // On-the-fly pack 8×kc A tile (column-major → row in micro)
                        std::vector<float, std::allocator<float>> A_tile(kc * 8);
                        for (std::size_t kk = 0; kk < kc; ++kk) {
                            for (int r = 0; r < 8; ++r) {
                                A_tile[kk * 8 + r] = A[(i + ii + r) * K + (k + kk)];
                            }
                        }
                        // On-the-fly pack kc×8 B tile
                        std::vector<float> B_tile(kc * 8);
                        for (std::size_t kk = 0; kk < kc; ++kk) {
                            for (int c = 0; c < 8; ++c) {
                                B_tile[kk * 8 + c] = B[(k + kk) * N + (j + jj + c)];
                            }
                        }
                        avx2_micro_kernel_8x8(
                            A_tile.data(), B_tile.data(),
                            C + (i + ii) * N + (j + jj),
                            N, kc);
                    }
                }
            }
        }
    }
    return {};
}

} // namespace hq::cerberus::native
#endif // __AVX2__ || __AVX__
#endif // x86_64
