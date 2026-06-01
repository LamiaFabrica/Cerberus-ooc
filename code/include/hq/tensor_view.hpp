#pragma once
/// @file tensor_view.hpp
/// @copyright Copyright (c) 2026 D Hargreaves (AKA Roylepython). LamiaFabrica. All rights reserved.
/// Self-contained rank-polymorphic tensor view for C++26 inference pipelines.
/// No std::mdspan dependency — fully available on any C++26 sysroot.
///
/// Provides:
///   - TensorView<T, Rank>: non-owning view over contiguous or strided memory
///   - C row-major layout by default; strided views for CHW ↔ HWC transposition
///   - ONNX Runtime zero-copy interop (when <onnxruntime_cxx_api.h> is available)
///   - C++26 concepts: TensorLike, MutableFloatTensor
///   - Gaussian noise fill, element-wise fmadd, make_tensor helpers
///
/// @author LamiaFabrica Team
/// @version 2.0.0

#include "hq/cxx26_features.hpp"
#include "hq/concepts.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#if UM790_HAS_STD_EXPECTED
#  include <expected>
#else
#  error "tensor_view.hpp requires std::expected — GCC >= 14 or Clang >= 18 with C++26"
#endif
#include <functional>
#include <random>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace hq::tensor {

// ===========================================================================
// TensorView<T, Rank>
// ===========================================================================

/// @brief Zero-overhead rank-polymorphic tensor view.
///
/// Stores a non-owning pointer, a shape array, and C row-major stride array.
/// Construction is O(Rank). No heap allocation.
///
/// @tparam T     Element type (may be cv-qualified, e.g. const float).
/// @tparam Rank  Number of dimensions (1–6).
template<hq::HqScalar T, std::size_t Rank>
class TensorView {
public:
    static_assert(Rank >= 1 && Rank <= 6, "TensorView supports ranks 1–6");

    using element_type = T;
    static constexpr std::size_t rank = Rank;

    // ---- Construction -------------------------------------------------------

    TensorView() = default;

    /// Construct from raw pointer + individual dimension sizes.
    template<typename... Sizes>
        requires (sizeof...(Sizes) == Rank) &&
                 (std::convertible_to<Sizes, std::size_t> && ...)
    explicit TensorView(T* data, Sizes... dims) noexcept
        : ptr_{data}, shape_{static_cast<std::size_t>(dims)...}
    {
        compute_strides_();
    }

    /// Construct from raw pointer + shape array (C row-major strides computed).
    explicit TensorView(T* data,
                        const std::array<std::size_t, Rank>& shape) noexcept
        : ptr_{data}, shape_{shape}
    {
        compute_strides_();
    }

    /// Static factory: wrap std::vector storage (view does NOT own the data).
    static TensorView from_storage(
        std::vector<std::remove_const_t<T>>& storage,
        const std::array<std::size_t, Rank>& shape)
    {
        const std::size_t need = flat_size_(shape);
        if (storage.size() < need) storage.resize(need);
        return TensorView{storage.data(), shape};
    }

    // ---- Pointer access -----------------------------------------------------

    [[nodiscard]] constexpr T* data() noexcept { return ptr_; }
    [[nodiscard]] constexpr const T* data() const noexcept { return ptr_; }

    // ---- Shape queries ------------------------------------------------------

    [[nodiscard]] constexpr std::size_t extent(std::size_t dim) const noexcept {
        return shape_[dim];
    }

    [[nodiscard]] constexpr std::array<std::size_t, Rank> shape() const noexcept {
        return shape_;
    }

    [[nodiscard]] std::size_t num_elements() const noexcept {
        return flat_size_(shape_);
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return num_elements();
    }

    [[nodiscard]] std::size_t num_bytes() const noexcept {
        return num_elements() * sizeof(T);
    }

    [[nodiscard]] constexpr bool empty() const noexcept {
        return ptr_ == nullptr || num_elements() == 0;
    }

    /// True when strides match C row-major packed layout.
    [[nodiscard]] bool is_contiguous() const noexcept {
        if constexpr (Rank == 1) return strides_[0] == 1;
        if (strides_[Rank - 1] != 1) return false;
        for (std::size_t d = Rank - 1; d > 0; --d) {
            if (strides_[d - 1] != strides_[d] * shape_[d]) return false;
        }
        return true;
    }

    // ---- Stride access ------------------------------------------------------

    [[nodiscard]] constexpr std::size_t stride(std::size_t dim) const noexcept {
        return strides_[dim];
    }

    [[nodiscard]] constexpr const std::array<std::size_t, Rank>& strides() const noexcept {
        return strides_;
    }

    // ---- Span access --------------------------------------------------------

    [[nodiscard]] std::span<T> flat_span() noexcept {
        return {ptr_, num_elements()};
    }

    [[nodiscard]] std::span<const std::remove_const_t<T>> flat_span() const noexcept {
        return {ptr_, num_elements()};
    }

    // ---- Element access operators -------------------------------------------

    /// Linear index into flat row-major storage.
    [[nodiscard]] constexpr T& operator[](std::size_t linear) const noexcept {
        return ptr_[linear];
    }

    /// Multi-index access: t(i0, i1, ..., iN-1).
    template<typename... Indices>
        requires (sizeof...(Indices) == Rank) &&
                 (std::integral<std::remove_cvref_t<Indices>> && ...)
    [[nodiscard]] constexpr T& operator()(Indices... idx) const noexcept {
        const std::size_t is[]{static_cast<std::size_t>(idx)...};
        std::size_t offset = 0;
        for (std::size_t d = 0; d < Rank; ++d) offset += is[d] * strides_[d];
        return ptr_[offset];
    }

    // ---- Bulk operations (non-const T only) ---------------------------------

    void fill(T value) noexcept
        requires (!std::is_const_v<T>)
    {
        std::ranges::fill(flat_span(), value);
    }

    template<std::invocable<T> Fn>
    void apply(Fn&& fn) noexcept
        requires (!std::is_const_v<T>)
    {
        for (T& v : flat_span()) v = std::invoke(std::forward<Fn>(fn), v);
    }

    // ---- ONNX Runtime interop (available when ORT headers are present) ------
#ifdef ONNXRUNTIME_CXX_API_H
    [[nodiscard]] Ort::Value to_ort_tensor(
        const Ort::MemoryInfo& mem, bool copy = false) const
    {
        using Bare = std::remove_const_t<T>;
        if (copy) {
            std::vector<Bare> buf(data(), data() + size());
            return make_ort_value_(mem, buf.data());
        }
        return make_ort_value_(mem, const_cast<Bare*>(data()));
    }
#endif

    // ---- Static strided-view factory (used by to_hwc_view, to_chw_view) ----
    static TensorView make_strided(
        T* ptr,
        const std::array<std::size_t, Rank>& shape,
        const std::array<std::size_t, Rank>& strides) noexcept
    {
        return TensorView{ptr, shape, strides};
    }

private:
    T*                            ptr_{nullptr};
    std::array<std::size_t, Rank> shape_{};
    std::array<std::size_t, Rank> strides_{};

    /// Private: construct with explicit strides (non-contiguous views).
    TensorView(T* ptr,
               const std::array<std::size_t, Rank>& shape,
               const std::array<std::size_t, Rank>& strides) noexcept
        : ptr_{ptr}, shape_{shape}, strides_{strides} {}

    void compute_strides_() noexcept {
        strides_[Rank - 1] = 1;
        if constexpr (Rank > 1) {
            for (std::size_t d = Rank - 1; d > 0; --d)
                strides_[d - 1] = strides_[d] * shape_[d];
        }
    }

    static constexpr std::size_t flat_size_(
        const std::array<std::size_t, Rank>& s) noexcept
    {
        std::size_t n = 1;
        for (const auto d : s) n *= d;
        return n;
    }

#ifdef ONNXRUNTIME_CXX_API_H
    Ort::Value make_ort_value_(const Ort::MemoryInfo& mem,
                                std::remove_const_t<T>* ptr) const
    {
        std::vector<std::int64_t> ort_shape(Rank);
        for (std::size_t d = 0; d < Rank; ++d)
            ort_shape[d] = static_cast<std::int64_t>(extent(d));
        return Ort::Value::CreateTensor<std::remove_const_t<T>>(
            const_cast<Ort::MemoryInfo&>(mem),
            ptr, size(), ort_shape.data(), ort_shape.size());
    }
#endif
};

// ===========================================================================
// Convenience aliases
// ===========================================================================

template<hq::HqScalar T> using Tensor1D = TensorView<T, 1>;
template<hq::HqScalar T> using Tensor2D = TensorView<T, 2>;
template<hq::HqScalar T> using Tensor3D = TensorView<T, 3>;
template<hq::HqScalar T> using Tensor4D = TensorView<T, 4>;

using FloatTensor1D = Tensor1D<float>;
using FloatTensor2D = Tensor2D<float>;
using FloatTensor3D = Tensor3D<float>;
using FloatTensor4D = Tensor4D<float>;

using Int64Tensor1D = Tensor1D<std::int64_t>;
using ByteTensor1D  = Tensor1D<std::byte>;

/// NHWC/NCHW image tensor: [batch, channels/height, height/width, width/channels]
template<hq::HqScalar T = float>
using ImageTensor = TensorView<T, 4>;

/// Latent tensor for VAE/UNet: [batch=1, channels=4, h/8, w/8]
template<hq::HqScalar T = float>
using LatentTensor = TensorView<T, 4>;

/// Embedding tensor: [batch=1, seq_len=77, hidden_dim]
template<hq::HqScalar T = float>
using EmbeddingTensor = TensorView<T, 3>;

// ===========================================================================
// CHW ↔ HWC strided view helpers (zero-copy layout reinterpretation)
// ===========================================================================

/// Given a NCHW tensor, return a stride-mapped view with NHWC indexing.
template<hq::HqScalar T>
TensorView<T, 4> to_hwc_view(TensorView<T, 4>& chw) noexcept {
    const auto sh = chw.shape();
    const std::size_t N = sh[0], C = sh[1], H = sh[2], W = sh[3];
    return TensorView<T, 4>::make_strided(
        chw.data(),
        std::array<std::size_t, 4>{N, H, W, C},
        std::array<std::size_t, 4>{C * H * W, C * W, C, 1});
}

/// Given a NHWC tensor, return a stride-mapped view with NCHW indexing.
template<hq::HqScalar T>
TensorView<T, 4> to_chw_view(TensorView<T, 4>& hwc) noexcept {
    const auto sh = hwc.shape();
    const std::size_t N = sh[0], H = sh[1], W = sh[2], C = sh[3];
    return TensorView<T, 4>::make_strided(
        hwc.data(),
        std::array<std::size_t, 4>{N, C, H, W},
        std::array<std::size_t, 4>{C * H * W, H * W, W, 1});
}

// ===========================================================================
// Utility free functions
// ===========================================================================

/// Wrap a std::vector in a TensorView (vector must outlive the view).
template<hq::HqScalar T, std::size_t Rank>
auto make_tensor(std::vector<T>& storage,
                 const std::array<std::size_t, Rank>& shape)
    -> TensorView<T, Rank>
{
    return TensorView<T, Rank>::from_storage(storage, shape);
}

/// Fill tensor with Gaussian noise (mean 0, stddev 1 by default).
template<hq::HqScalar T, std::size_t Rank>
void fill_gaussian(TensorView<T, Rank>& tv, std::mt19937& rng,
                   T mean = T{0}, T stddev = T{1})
    requires std::is_floating_point_v<T>
{
    std::normal_distribution<T> dist{mean, stddev};
    for (auto& v : tv.flat_span()) v = dist(rng);
}

/// Element-wise multiply-add: dst[i] = a * src0[i] + b * src1[i].
/// All three tensors must have identical element counts.
template<hq::HqScalar T, std::size_t Rank>
void fmadd(TensorView<T, Rank>& dst,
           T a, const TensorView<const T, Rank>& src0,
           T b, const TensorView<const T, Rank>& src1)
{
    auto ds = dst.flat_span();
    auto s0 = src0.flat_span();
    auto s1 = src1.flat_span();
    for (std::size_t i = 0; i < ds.size(); ++i)
        ds[i] = a * s0[i] + b * s1[i];
}

// ===========================================================================
// ONNX Runtime extraction helpers (require ORT headers)
// ===========================================================================
#ifdef ONNXRUNTIME_CXX_API_H

/// Extract a typed TensorView from an Ort::Value output (zero-copy).
/// The Ort::Value must outlive the returned view.
template<std::size_t Rank>
TensorView<float, Rank> extract_float_tensor(Ort::Value& out) {
    const auto shape_info = out.GetTensorTypeAndShapeInfo();
    const auto ort_shape  = shape_info.GetShape();
    std::array<std::size_t, Rank> shape{};
    for (std::size_t d = 0; d < Rank; ++d) {
        shape[d] = (d < ort_shape.size() && ort_shape[d] > 0)
                       ? static_cast<std::size_t>(ort_shape[d]) : 1;
    }
    float* ptr = const_cast<float*>(out.GetTensorData<float>());
    return TensorView<float, Rank>{ptr, shape};
}

/// Convenience: extract a 4-D latent tensor [1, 4, H, W].
inline FloatTensor4D extract_latent(Ort::Value& v) {
    return extract_float_tensor<4>(v);
}

/// Convenience: extract a 3-D embedding [1, 77, hidden].
inline FloatTensor3D extract_embedding(Ort::Value& v) {
    return extract_float_tensor<3>(v);
}

#endif // ONNXRUNTIME_CXX_API_H

// ===========================================================================
// C++26 concepts for tensor-view constraints
// ===========================================================================

/// Satisfied by any read-only tensor view type.
template<typename T>
concept TensorLike =
    std::is_arithmetic_v<typename T::element_type> &&
    requires(const T& t, std::size_t i) {
        { t.data() }         -> std::convertible_to<const typename T::element_type*>;
        { t.num_elements() } -> std::convertible_to<std::size_t>;
        { t.empty() }        -> std::same_as<bool>;
        { t.size() }         -> std::convertible_to<std::size_t>;
        { t.extent(i) }      -> std::convertible_to<std::size_t>;
    };

/// Satisfied by a mutable float tensor view (pipeline hot path).
template<typename T>
concept MutableFloatTensor =
    TensorLike<T> &&
    std::same_as<typename T::element_type, float> &&
    requires(T& t) {
        { t.data() } -> std::convertible_to<float*>;
    };

static_assert(TensorLike<TensorView<float, 1>>,
              "TensorView<float,1> must satisfy TensorLike");
static_assert(TensorLike<TensorView<float, 4>>,
              "TensorView<float,4> must satisfy TensorLike");
static_assert(MutableFloatTensor<TensorView<float, 4>>,
              "TensorView<float,4> must satisfy MutableFloatTensor");

} // namespace hq::tensor
