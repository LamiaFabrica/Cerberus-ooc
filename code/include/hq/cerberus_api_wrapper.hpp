#pragma once
/// @file cerberus_api_wrapper.hpp
/// @copyright Copyright (c) 2026 D Hargreaves (AKA Roylepython). LamiaFabrica. All rights reserved.
/// @brief C++26 wrapper around the Cerberus C API (cerberus_api.h).
///
/// Provides std::expected-based error handling, [[nodiscard]] lifecycle calls,
/// and std::span buffer passing while verifying extern "C" linkage.

#include "hq/cerberus_api.h"

#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#include <cstddef>
#include <cstring>

namespace hq::cerberus::c_api {

// ===========================================================================
// Linkage verification — ensures we are calling C symbols, not C++ mangled
// ===========================================================================

static_assert(std::is_same_v<decltype(&cerberus_init), cerberus_status_t (*)(void)>,
              "cerberus_init must have C linkage");
static_assert(std::is_same_v<decltype(&cerberus_shutdown), cerberus_status_t (*)(void)>,
              "cerberus_shutdown must have C linkage");
static_assert(std::is_same_v<decltype(&cerberus_get_version), const char* (*)(void)>,
              "cerberus_get_version must have C linkage");
static_assert(std::is_same_v<decltype(&cerberus_get_last_error), const char* (*)(void)>,
              "cerberus_get_last_error must have C linkage");
static_assert(std::is_same_v<decltype(&cerberus_create_session),
                             cerberus_status_t (*)(const cerberus_session_config_t*, cerberus_handle_t*)>,
              "cerberus_create_session must have C linkage");
static_assert(std::is_same_v<decltype(&cerberus_destroy_session), cerberus_status_t (*)(cerberus_handle_t)>,
              "cerberus_destroy_session must have C linkage");
static_assert(std::is_same_v<decltype(&cerberus_run),
                             cerberus_status_t (*)(cerberus_handle_t, const float*, size_t, float**, size_t*)>,
              "cerberus_run must have C linkage");

// ===========================================================================
// Result types
// ===========================================================================

struct CApiError {
    cerberus_status_t code{CERBERUS_ERROR};
    std::string message;
};

template <typename T>
using CApiExpected = std::expected<T, CApiError>;

// ===========================================================================
// Lifecycle — [[nodiscard]]
// ===========================================================================

[[nodiscard]] inline CApiExpected<void> init() noexcept {
    auto st = cerberus_init();
    if (st != CERBERUS_OK) {
        return std::unexpected{CApiError{st, cerberus_get_last_error()}};
    }
    return {};
}

[[nodiscard]] inline CApiExpected<void> shutdown() noexcept {
    auto st = cerberus_shutdown();
    if (st != CERBERUS_OK) {
        return std::unexpected{CApiError{st, cerberus_get_last_error()}};
    }
    return {};
}

// ===========================================================================
// Version / error
// ===========================================================================

inline std::string_view version() noexcept {
    const char* v = cerberus_get_version();
    return v ? std::string_view{v} : std::string_view{};
}

inline std::string_view last_error() noexcept {
    const char* e = cerberus_get_last_error();
    return e ? std::string_view{e} : std::string_view{};
}

// ===========================================================================
// Session management
// ===========================================================================

struct Session {
    cerberus_handle_t handle{nullptr};

    explicit Session(cerberus_handle_t h) noexcept : handle(h) {}
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    Session(Session&& other) noexcept : handle(other.handle) { other.handle = nullptr; }
    Session& operator=(Session&& other) noexcept {
        if (this != &other) {
            if (handle) cerberus_destroy_session(handle);
            handle = other.handle;
            other.handle = nullptr;
        }
        return *this;
    }
    ~Session() {
        if (handle) cerberus_destroy_session(handle);
    }
};

[[nodiscard]] inline CApiExpected<Session> create_session(const cerberus_session_config_t& cfg) noexcept {
    cerberus_handle_t h = nullptr;
    auto st = cerberus_create_session(&cfg, &h);
    if (st != CERBERUS_OK || !h) {
        return std::unexpected{CApiError{st, cerberus_get_last_error()}};
    }
    return Session{h};
}

// ===========================================================================
// Inference — std::span for buffers
// ===========================================================================

struct InferenceResult {
    std::vector<float> output;
};

[[nodiscard]] inline CApiExpected<InferenceResult>
run(cerberus_handle_t session, std::span<const float> input) noexcept {
    if (!session) {
        return std::unexpected{CApiError{CERBERUS_INVALID_HANDLE, "null session handle"}};
    }
    float* out_ptr = nullptr;
    size_t out_size = 0;
    auto st = cerberus_run(session, input.data(), input.size(), &out_ptr, &out_size);
    if (st != CERBERUS_OK || !out_ptr) {
        return std::unexpected{CApiError{st, cerberus_get_last_error()}};
    }
    InferenceResult res;
    if (out_size > 0 && out_ptr) {
        res.output.assign(out_ptr, out_ptr + out_size);
    }
    return res;
}

} // namespace hq::cerberus::c_api
