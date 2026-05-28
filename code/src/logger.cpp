/// @file logger.cpp
/// @copyright Copyright © 2026, all rights reserved | David Hargreaves aka Roylepython trading as The Medusa Initiative & Lamia Fabrica
/// Support: https://www.patreon.com/TheMedusaInitiative — £25/month removes ads from all software at 200 subscribers
/// @author LamiaFabrica Team
/// @brief Logger implementation.

#include "hq/logger.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <source_location>
#include <string>

#ifdef _WIN32
#  include <io.h>       // _isatty, _fileno
#  define ISATTY(f) _isatty(_fileno(f))
#else
#  include <unistd.h>   // isatty, fileno
#  define ISATTY(f) isatty(fileno(f))
#endif

namespace hq {

// ---------------------------------------------------------------------------
// Static member definitions
// ---------------------------------------------------------------------------

// Default: INFO in Release, DEBUG in Debug
#ifdef NDEBUG
std::atomic<int>  Logger::s_level_{static_cast<int>(LogLevel::Info)};
#else
std::atomic<int>  Logger::s_level_{static_cast<int>(LogLevel::Debug)};
#endif

std::atomic<bool> Logger::s_colour_{true};   // resolved to TTY-check at first write
std::mutex        Logger::s_mutex_;
FILE*             Logger::s_file_{nullptr};

// ---------------------------------------------------------------------------
// init_from_env
// ---------------------------------------------------------------------------

void Logger::init_from_env(const char* env_var) noexcept {
    const char* val = std::getenv(env_var);
    if (!val) return;

    // Case-insensitive compare helper
    auto ci_eq = [](const char* a, const char* b) {
#ifdef _WIN32
        return _stricmp(a, b) == 0;
#else
        return strcasecmp(a, b) == 0;
#endif
    };

    if      (ci_eq(val, "trace")) set_level(LogLevel::Trace);
    else if (ci_eq(val, "debug")) set_level(LogLevel::Debug);
    else if (ci_eq(val, "info"))  set_level(LogLevel::Info);
    else if (ci_eq(val, "warn") ||
             ci_eq(val, "warning")) set_level(LogLevel::Warn);
    else if (ci_eq(val, "error")) set_level(LogLevel::Error);
    else if (ci_eq(val, "fatal")) set_level(LogLevel::Fatal);
    else if (ci_eq(val, "off"))   set_level(LogLevel::Off);
    // Unknown value: leave default
}

// ---------------------------------------------------------------------------
// set_log_file
// ---------------------------------------------------------------------------

void Logger::set_log_file(const char* path) noexcept {
    std::lock_guard lock{s_mutex_};
    if (s_file_ && s_file_ != stderr) {
        std::fclose(s_file_);
        s_file_ = nullptr;
    }
    if (path && *path) {
        s_file_ = std::fopen(path, "a");
    }
}

// ---------------------------------------------------------------------------
// write (string_view overload — final sink)
// ---------------------------------------------------------------------------

void Logger::write(LogLevel             level,
                   std::source_location loc,
                   std::string_view     message) noexcept {
    if (static_cast<int>(level) < s_level_.load(std::memory_order_relaxed))
        return;

    write_line_(level, loc, message);

    if (level == LogLevel::Fatal) {
        std::fflush(stderr);
        if (s_file_) std::fflush(s_file_);
        std::terminate();
    }
}

// ---------------------------------------------------------------------------
// write_line_ — formats and emits a single log record
//
// Format:
//   [HH:MM:SS.mmm] [LEVEL] basename.cpp:LINE func() — message
// ---------------------------------------------------------------------------

void Logger::write_line_(LogLevel                    level,
                          const std::source_location& loc,
                          std::string_view            msg) noexcept {
    // --- Timestamp ---
    using clock = std::chrono::system_clock;
    const auto now     = clock::now();
    const auto now_t   = clock::to_time_t(now);
    const auto ms      = std::chrono::duration_cast<std::chrono::milliseconds>(
                             now.time_since_epoch()) % 1000;

    std::tm tm_buf{};
#ifdef _WIN32
    localtime_s(&tm_buf, &now_t);
#else
    localtime_r(&now_t, &tm_buf);
#endif

    // --- Source location ---
    const char* file = detail::basename(loc.file_name());
    const int   line = static_cast<int>(loc.line());
    const char* func = loc.function_name();

    // --- Colour ---
    const bool use_colour =
        s_colour_.load(std::memory_order_relaxed) && ISATTY(stderr);

    const char* col   = use_colour ? detail::level_colour(level) : "";
    const char* reset = use_colour ? detail::reset_colour()      : "";

    std::lock_guard lock{s_mutex_};

    // --- Write to stderr ---
    std::fprintf(stderr,
        "%s[%02d:%02d:%02d.%03d] [%s] %s:%d %s() — %.*s%s\n",
        col,
        tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec,
        static_cast<int>(ms.count()),
        to_string(level).data(),
        file, line, func,
        static_cast<int>(msg.size()), msg.data(),
        reset);

    // --- Tee to file (no colour) ---
    if (s_file_) {
        std::fprintf(s_file_,
            "[%02d:%02d:%02d.%03d] [%s] %s:%d %s() — %.*s\n",
            tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec,
            static_cast<int>(ms.count()),
            to_string(level).data(),
            file, line, func,
            static_cast<int>(msg.size()), msg.data());
    }
}

} // namespace hq
