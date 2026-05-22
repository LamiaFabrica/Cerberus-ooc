#pragma once
/// @file logger.hpp
/// @copyright Copyright (c) 2026 D Hargreaves (AKA Roylepython). LamiaFabrica. All rights reserved.
/// @author LamiaFabrica Team
/// @brief Structured, source-location-aware logger for Cerberus.
///
/// Usage:
///   HQ_LOG_DEBUG("allocating {} bytes in tier {}", n, hq::to_string(tier));
///   HQ_LOG_ERROR("OOM: {} bytes requested", sz);
///   HQ_LOG_FATAL("unrecoverable init failure: {}", msg);
///
/// Every log line carries exact file, line number, and function name via
/// std::source_location — no guessing where an error came from.
///
/// Log levels (ascending severity):
///   TRACE  — per-step hot-path detail (disabled in Release unless HQ_TRACE=1)
///   DEBUG  — subsystem lifecycle and state changes
///   INFO   — normal operational events (startup, shutdown, config)
///   WARN   — recoverable anomalies (fallback paths taken, retries)
///   ERROR  — operation failed, error returned to caller
///   FATAL  — unrecoverable — logs then calls std::terminate()
///
/// Runtime control:
///   Logger::set_level(LogLevel::Debug);   // programmatic
///   HQ_LOG_SET_ENV("HQ_LOG_LEVEL");       // reads env var at startup
///
/// Compile-time silence (zero overhead):
///   -DHQ_LOG_MIN_LEVEL=3  // only WARN and above compiled in

#include "hq/cxx26_features.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <source_location>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>

#if UM790_HAS_STD_FORMAT
#  include <format>
#endif

namespace hq {

// ---------------------------------------------------------------------------
// Log level enum
// ---------------------------------------------------------------------------

enum class LogLevel : int {
    Trace = 0,
    Debug = 1,
    Info  = 2,
    Warn  = 3,
    Error = 4,
    Fatal = 5,
    Off   = 6,  ///< Silence all output
};

[[nodiscard]] inline std::string_view to_string(LogLevel l) noexcept {
    switch (l) {
        case LogLevel::Trace: return "TRACE";
        case LogLevel::Debug: return "DEBUG";
        case LogLevel::Info:  return "INFO ";
        case LogLevel::Warn:  return "WARN ";
        case LogLevel::Error: return "ERROR";
        case LogLevel::Fatal: return "FATAL";
        default:              return "OFF  ";
    }
}

// ---------------------------------------------------------------------------
// ANSI colour helpers (disabled automatically when not a TTY)
// ---------------------------------------------------------------------------

namespace detail {
inline const char* level_colour(LogLevel l) noexcept {
    switch (l) {
        case LogLevel::Trace: return "\033[90m";      // dark grey
        case LogLevel::Debug: return "\033[36m";      // cyan
        case LogLevel::Info:  return "\033[32m";      // green
        case LogLevel::Warn:  return "\033[33m";      // yellow
        case LogLevel::Error: return "\033[31m";      // red
        case LogLevel::Fatal: return "\033[1;31m";    // bold red
        default:              return "";
    }
}
inline const char* reset_colour() noexcept { return "\033[0m"; }

// Returns just the filename portion of a full path (skip directories).
inline const char* basename(const char* path) noexcept {
    const char* last = path;
    for (const char* p = path; *p; ++p)
        if (*p == '/' || *p == '\\') last = p + 1;
    return last;
}
} // namespace detail

// ---------------------------------------------------------------------------
// Logger — singleton, thread-safe
// ---------------------------------------------------------------------------

class Logger {
public:
    // ------------------------------------------------------------------
    // Configuration
    // ------------------------------------------------------------------

    static void set_level(LogLevel l) noexcept {
        s_level_.store(static_cast<int>(l), std::memory_order_relaxed);
    }

    [[nodiscard]] static LogLevel level() noexcept {
        return static_cast<LogLevel>(
            s_level_.load(std::memory_order_relaxed));
    }

    /// Read log level from environment variable (e.g. HQ_LOG_LEVEL=DEBUG).
    static void init_from_env(const char* env_var = "HQ_LOG_LEVEL") noexcept;

    /// Optionally tee output to a file in addition to stderr.
    static void set_log_file(const char* path) noexcept;

    /// Disable ANSI colours (default: auto-detect TTY).
    static void set_colour(bool enabled) noexcept {
        s_colour_.store(enabled, std::memory_order_relaxed);
    }

    // ------------------------------------------------------------------
    // Core log function — called by all HQ_LOG_* macros
    // ------------------------------------------------------------------

    static void write(LogLevel           level,
                      std::source_location loc,
                      std::string_view   message) noexcept;

    // ------------------------------------------------------------------
    // Formatted overload — avoids format overhead when level is filtered
    // ------------------------------------------------------------------

    template<typename... Args>
    static void writef(LogLevel              level,
                       std::source_location  loc,
                       std::string_view      fmt_str,
                       Args&&...             args) noexcept {
        if (static_cast<int>(level) <
            s_level_.load(std::memory_order_relaxed)) return;
#if UM790_HAS_STD_FORMAT
        try {
            // GCC 14 make_format_args takes _Args&... (non-const lvalue refs).
            // std::decay_t (not remove_cvref_t) so that char[N] array args
            // decay to const char* — arrays are not tuple-constructible.
            std::tuple<std::decay_t<Args>...> argv{
                std::forward<Args>(args)...};
            auto msg = std::apply([&fmt_str](auto&... a) {
                return std::vformat(fmt_str, std::make_format_args(a...));
            }, argv);
            write(level, loc, msg);
        } catch (...) {
            write(level, loc, fmt_str);
        }
#else
        write(level, loc, fmt_str);
#endif
    }

private:
    static std::atomic<int>  s_level_;
    static std::atomic<bool> s_colour_;
    static std::mutex        s_mutex_;
    static FILE*             s_file_;

    static void write_line_(LogLevel level,
                             const std::source_location& loc,
                             std::string_view msg) noexcept;
};

} // namespace hq

// ---------------------------------------------------------------------------
// Compile-time minimum level gate
// ---------------------------------------------------------------------------

#ifndef HQ_LOG_MIN_LEVEL
#  ifdef NDEBUG
#    define HQ_LOG_MIN_LEVEL 2   // INFO and above in Release
#  else
#    define HQ_LOG_MIN_LEVEL 0   // All levels in Debug
#  endif
#endif

// ---------------------------------------------------------------------------
// Public macros — each captures exact call-site location automatically
// ---------------------------------------------------------------------------

#define HQ_LOG_(level_, ...)                                            \
    do {                                                                \
        if constexpr (static_cast<int>(hq::LogLevel::level_) >=        \
                      HQ_LOG_MIN_LEVEL) {                               \
            hq::Logger::writef(hq::LogLevel::level_,                   \
                               std::source_location::current(),         \
                               __VA_ARGS__);                            \
        }                                                               \
    } while (false)

/// Per-step hot-path detail — compiled out in Release unless HQ_LOG_MIN_LEVEL=0
#define HQ_LOG_TRACE(...) HQ_LOG_(Trace, __VA_ARGS__)

/// Subsystem lifecycle, state changes
#define HQ_LOG_DEBUG(...) HQ_LOG_(Debug, __VA_ARGS__)

/// Normal operational events
#define HQ_LOG_INFO(...)  HQ_LOG_(Info,  __VA_ARGS__)

/// Recoverable anomaly
#define HQ_LOG_WARN(...)  HQ_LOG_(Warn,  __VA_ARGS__)

/// Operation failed — result returned to caller
#define HQ_LOG_ERROR(...) HQ_LOG_(Error, __VA_ARGS__)

/// Unrecoverable — logs then calls std::terminate()
#define HQ_LOG_FATAL(...) \
    do { HQ_LOG_(Fatal, __VA_ARGS__); std::terminate(); } while (false)

// ---------------------------------------------------------------------------
// Helper: log-and-return a std::unexpected with source context
//
// Usage:
//   return HQ_UNEXPECTED(TierError::OutOfMemory,
//                        "requested {} bytes, only {} available", need, avail);
// ---------------------------------------------------------------------------

#define HQ_UNEXPECTED(err_val_, ...)                                    \
    [&]() {                                                             \
        HQ_LOG_ERROR(__VA_ARGS__);                                      \
        return std::unexpected{err_val_};                               \
    }()

// ---------------------------------------------------------------------------
// Read log level from environment at startup
// ---------------------------------------------------------------------------

#define HQ_LOG_INIT() hq::Logger::init_from_env("HQ_LOG_LEVEL")
