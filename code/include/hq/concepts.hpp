#pragma once
/// @file concepts.hpp
/// @copyright Copyright (c) 2026 D Hargreaves (AKA Roylepython). LamiaFabrica. All rights reserved.
///
/// Cerberus C++26 core domain concept library.
///
/// Every template parameter in the Cerberus heterogeneous quantized runtime gets
/// a concept.  No unconstrained templates in this header — any helper must be
/// constrained or non-template.
///
/// Six canonical concepts:
///   - HqScalar     — arithmetic types suitable for tensor elements (integral or floating-point, not bool)
///   - HqBuffer     — contiguous storage with .data() -> T*, .size() -> size_t
///   - HqExecutable — callable pipeline stage with .invoke(), .input_shapes(), .output_shapes()
///   - HqTelemetry  — telemetry surface with .record(), .flush(), timestamping
///   - HqDevice     — device backend with .submit(), .synchronize(), device identification
///   - HqQuantized  — quantized type with .scale(), .zero_point(), dequantize/quantize
///
/// @version 2.0.0

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

namespace hq {

// ===========================================================================
// HqScalar — arithmetic types suitable for tensor elements (integral or
// floating-point, not bool).  std::byte is also accepted for raw weight
// buffers.
//
// Example models: float, int32_t, int8_t, uint8_t, std::byte
// ===========================================================================

template<typename T>
concept HqScalar = (std::integral<std::remove_cv_t<T>> && !std::same_as<std::remove_cv_t<T>, bool>) ||
                   std::floating_point<std::remove_cv_t<T>> ||
                   std::same_as<std::remove_cv_t<T>, std::byte>;

static_assert(HqScalar<float>);
static_assert(HqScalar<std::int32_t>);
static_assert(HqScalar<std::int8_t>);
static_assert(HqScalar<std::byte>);
static_assert(!HqScalar<bool>);
static_assert(!HqScalar<std::vector<float>>);

// ===========================================================================
// HqBuffer — contiguous memory container with .data() -> T*, .size() -> size_t
//
// Example models: std::vector<T>, std::span<T>, std::array<T, N>
// ===========================================================================

template<typename B>
concept HqBuffer = requires(const B& b) {
    { b.data() } -> std::contiguous_iterator;
    { b.size() } -> std::convertible_to<std::size_t>;
} || requires(const B& b) {
    { b.data() } -> std::same_as<const std::remove_pointer_t<decltype(b.data())>* const>;
    { b.size() } -> std::convertible_to<std::size_t>;
} || requires(const B& b) {
    { b.data() } -> std::same_as<const std::remove_pointer_t<decltype(b.data())>* const>;
    { b.size() } -> std::same_as<std::size_t>;
};

// ===========================================================================
// HqExecutable — callable pipeline stage with shape inspection
//
// Example models: a compiled ONNX node, a fused kernel handle, a lambda
// wrapped in an executor
// ===========================================================================

template<typename T>
concept HqExecutable = requires(T& t) {
    { t.invoke() } -> std::same_as<void>;
    { t.input_shapes() } -> std::same_as<std::vector<std::vector<std::size_t>>>;
    { t.output_shapes() } -> std::same_as<std::vector<std::vector<std::size_t>>>;
};

// ===========================================================================
// HqTelemetry — telemetry surface with record/flush and timestamping
//
// Example models: NPU utilization tracker, GPU perf counter aggregator,
// a ring-buffer of (timestamp, metric) pairs
// ===========================================================================

template<typename T>
concept HqTelemetry = requires(T& t) {
    { t.record(std::string{}, double{}) } -> std::same_as<void>;
    { t.flush() } -> std::same_as<void>;
} && requires(const T& t) {
    { t.has_timestamping() } -> std::same_as<bool>;
};

// ===========================================================================
// HqDevice — device backend with submit/synchronize and identification
//
// Example models: NPU backend, GPU HIP stream wrapper, CPU thread-pool
// scheduler with device affinity
// ===========================================================================

template<typename T>
concept HqDevice = requires(T& t, const void* cmd, std::size_t count) {
    { t.submit(cmd, count) } -> std::same_as<bool>;
    { t.synchronize() } -> std::same_as<void>;
} && requires(const T& t) {
    { t.device_id() } -> std::convertible_to<std::string>;
    { t.device_type() } -> std::convertible_to<std::string>;
};

// ===========================================================================
// HqQuantized — quantized type with scale, zero_point, dequantize/quantize
//
// Example models: int8_t with fp32 scale, uint8_t with asymmetric zero-point,
// a Q4_0 block with per-block scale
// ===========================================================================

template<typename T>
concept HqQuantized = requires(const T& t) {
    { t.scale() } -> std::floating_point;
    { t.zero_point() } -> std::integral;
} && requires(T& t, float f) {
    { t.dequantize() } -> std::floating_point;
    { t.quantize(f) } -> std::same_as<void>;
};

// ===========================================================================
// HqFormattableArgs — argument pack where every type is formattable by
// std::format.  Used to constrain HQ_LOG_* macro expansions so that
// ill-typed format strings fail at compile time, not at runtime inside
// std::vformat.
// ===========================================================================

template<typename... Args>
concept HqFormattableArgs = (std::formattable<std::remove_cvref_t<Args>, char> && ...);

// ===========================================================================
// HqChronoRep — valid representation type for std::chrono::duration
// ===========================================================================

template<typename Rep>
concept HqChronoRep = std::integral<Rep> || std::floating_point<Rep>;

// ===========================================================================
// HqChronoPeriod — valid period type for std::chrono::duration
// ===========================================================================

template<typename Period>
concept HqChronoPeriod = requires {
    typename Period::type;
    requires std::same_as<typename Period::type, Period>;
};

// ===========================================================================
// HqGeneratorValue — type that can be yielded from a coroutine generator
// (copyable or movable, not void, not reference)
// ===========================================================================

template<typename T>
concept HqGeneratorValue = std::copyable<T> || std::movable<T>;

// ===========================================================================
// HqCoroValue — type that can be used as a coroutine task return value.
// void is explicitly allowed (task<void> is the primary use-case).
// ===========================================================================

template<typename T>
concept HqCoroValue = std::same_as<T, void> ||
                      ((std::copyable<T> || std::movable<T>) && !std::is_reference_v<T>);

// ===========================================================================
// HqExpectedValue — type that can be carried inside std::expected<T,E>.
// Any non-reference, non-void type, plus void explicitly allowed.
// ===========================================================================

template<typename T>
concept HqExpectedValue = std::same_as<T, void> ||
                          (!std::is_reference_v<T> && !std::is_void_v<T> &&
                           (std::copyable<T> || std::movable<T>));

} // namespace hq
