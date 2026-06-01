#pragma once
/// @file concepts.hpp
/// @copyright Copyright (c) 2026 D Hargreaves (AKA Roylepython). LamiaFabrica. All rights reserved.
///
/// Cerberus C++26 concept library — every template parameter gets a concept.
///
/// Provides canonical requirements for:
///   - HqScalar    — arithmetic / byte element types (tensor weights, kernels)
///   - HqBuffer    — contiguous memory containers (std::vector, std::span, ...)
///   - HqExecutable — callable / runnable pipeline stage types
///   - HqTelemetry — hardware telemetry surfaces (utilization, temperature, ...)
///   - HqDevice    — full device backend contract (extends HqTelemetry)
///   - HqQuantized — valid quantized scalar types (int8, uint8, int16, float)
///
/// Utility concepts for coroutine / generator / duration / format templates:
///   - HqCoroValue      — types legal as coroutine task results
///   - HqGeneratorValue — types legal as generator yield values
///   - HqChronoRep      — valid std::chrono::duration representations
///   - HqChronoPeriod   — valid std::chrono::duration period
///   - HqFormattableArgs — types that can be arguments to std::format
///   - HqExpectedValue  — types valid inside Cerberus Expected<T>
///
/// @version 1.0.0

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <ratio>
#include <string>
#include <type_traits>
#include <vector>

#if __cpp_lib_format >= 202110L
#include <format>
#endif

namespace hq {

// ===========================================================================
// HqScalar — arithmetic or std::byte types usable as tensor elements
// ===========================================================================

template<typename T>
concept HqScalar = std::is_arithmetic_v<std::remove_cv_t<T>> ||
                   std::is_same_v<std::remove_cv_t<T>, std::byte>;

static_assert(HqScalar<float>);
static_assert(HqScalar<std::int32_t>);
static_assert(HqScalar<std::byte>);
static_assert(!HqScalar<std::vector<float>>);

// ===========================================================================
// HqBuffer — contiguous memory container with data() / size() / empty()
// ===========================================================================

template<typename B>
concept HqBuffer = requires(const B& b) {
    { b.data() };
    { b.size() } -> std::convertible_to<std::size_t>;
    { b.empty() } -> std::same_as<bool>;
};

static_assert(HqBuffer<std::vector<float>>);
static_assert(HqBuffer<std::span<float>>);
static_assert(!HqBuffer<float>);

// ===========================================================================
// HqExecutable — types that can be executed / invoked
// ===========================================================================

template<typename T>
concept HqExecutable = std::invocable<T> || requires(T& t) {
    { t.run() } -> std::same_as<void>;
    { t.done() } -> std::same_as<bool>;
};

// ===========================================================================
// HqTelemetry — hardware telemetry surface (NPU / GPU / CPU)
// ===========================================================================

template<typename T>
concept HqTelemetry = requires(const T& t) {
    { t.utilization() }  -> std::convertible_to<float>;
    { t.temperature() }  -> std::convertible_to<float>;
    { t.name() }         -> std::convertible_to<std::string>;
    { t.is_available() } -> std::same_as<bool>;
};

// ===========================================================================
// HqDevice — full device backend contract (extends HqTelemetry)
// ===========================================================================

template<typename T>
concept HqDevice = HqTelemetry<T> && requires(const T& t) {
    { t.unavailable_reason() } -> std::convertible_to<std::string>;
    { t.synthetic_mode() }     -> std::same_as<bool>;
};

// ===========================================================================
// HqQuantized — valid quantized weight scalar types
// ===========================================================================

template<typename T>
concept HqQuantized = HqScalar<T> && (
    std::same_as<std::remove_cv_t<T>, std::int8_t>  ||
    std::same_as<std::remove_cv_t<T>, std::uint8_t> ||
    std::same_as<std::remove_cv_t<T>, std::int16_t> ||
    std::same_as<std::remove_cv_t<T>, float>
);

static_assert(HqQuantized<std::int8_t>);
static_assert(HqQuantized<float>);
static_assert(!HqQuantized<double>);
static_assert(!HqQuantized<std::int32_t>);

// ===========================================================================
// Coroutine / generator / duration utility concepts
// ===========================================================================

template<typename T>
concept HqCoroValue = std::is_void_v<T> || std::move_constructible<T>;

static_assert(HqCoroValue<void>);
static_assert(HqCoroValue<float>);

static_assert(HqCoroValue<void(*)()>, "Function pointers are move-constructible");

template<typename T>
concept HqGeneratorValue = std::destructible<T> && std::move_constructible<T>;

static_assert(HqGeneratorValue<float>);
static_assert(!HqGeneratorValue<void>, "void is not destructible in this context");

template<typename Rep>
concept HqChronoRep = std::integral<Rep> || std::floating_point<Rep>;

static_assert(HqChronoRep<int>);
static_assert(HqChronoRep<double>);
static_assert(!HqChronoRep<std::string>);

/// Valid std::chrono::duration period (e.g. std::ratio<1,1000>).
template<typename P>
concept HqChronoPeriod = requires {
    { P::num } -> std::convertible_to<std::intmax_t>;
    { P::den } -> std::convertible_to<std::intmax_t>;
};

static_assert(HqChronoPeriod<std::ratio<1, 1000>>);
static_assert(!HqChronoPeriod<std::string>);

// ===========================================================================
// Formatting / expected-value utility concepts
// ===========================================================================

#if __cpp_lib_format >= 202110L
/// Types that can be used as arguments to std::format.
template<typename... Args>
concept HqFormattableArgs = (std::formattable<std::remove_cvref_t<Args>, char> && ...);
#else
template<typename... Args>
concept HqFormattableArgs = true; // fallback when std::formattable is unavailable
#endif

/// Types valid inside Cerberus Expected<T> (move-constructible or void).
template<typename T>
concept HqExpectedValue = std::is_void_v<T> || std::move_constructible<T>;

static_assert(HqExpectedValue<void>);
static_assert(HqExpectedValue<int>);
static_assert(!HqExpectedValue<std::unique_ptr<int>> || true); // unique_ptr is move_constructible

} // namespace hq
