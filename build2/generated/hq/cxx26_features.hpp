#pragma once
// =============================================================================
// Auto-generated C++26 feature detection header (do not edit manually).
//
// This is the CANONICAL feature detection header for the UM790 pipeline.
// Values come from CMake's check_cxx_source_compiles() (or try_compile())
// tests run at configure time — NOT from __has_include or __cpp macro guards
// that may silently disagree with the actual toolchain.
//
// The generated output is written to:
//   ${CMAKE_CURRENT_BINARY_DIR}/generated/hq/cxx26_features.hpp
//
// The include path orders ${BUILD_INTERFACE:generated} before
// ${BUILD_INTERFACE:include} so this file shadows any stale source copy.
// =============================================================================

// CMake substitutes @ONLY variables below (no fallback overrides — the cmake
// link-test result IS the authority; __cpp_lib_* macros must not override it).

#ifndef UM790_HAS_STD_EXPECTED
#define UM790_HAS_STD_EXPECTED 1
#endif

#ifndef UM790_HAS_STD_PRINT
#define UM790_HAS_STD_PRINT 0
#endif

#ifndef UM790_HAS_STD_MDSPAN
#define UM790_HAS_STD_MDSPAN 0
#endif

#ifndef UM790_HAS_STD_FORMAT
#define UM790_HAS_STD_FORMAT 1
#endif

#ifndef UM790_HAS_COROUTINES
#define UM790_HAS_COROUTINES 1
#endif

#ifndef UM790_HAS_STD_COROUTINE
#define UM790_HAS_STD_COROUTINE 1
#endif

// =============================================================================
// std::print compatibility shim
//
// MinGW-W64 GCC 14/15 ships <print> and defines __cpp_lib_print but the linker
// is missing std::__open_terminal / std::__write_to_terminal. CMake detects
// this via the link test (UM790_HAS_STD_PRINT = 0). The shim below re-provides
// std::print / std::println using std::format + platform-safe output so all
// call-sites compile and link without change.
//
// CRITICAL: MinGW-W64's std::fputs AND _write() crash with ACCESS_VIOLATION
// when stdout is piped. The root cause is the MinGW CRT console initialization
// code that dereferences a null handle when the output stream is not a console.
//
// Fix: On Windows, safe_write.cpp provides hq_safe_write() which calls Win32
// WriteFile() via GetStdHandle(), completely bypassing the broken CRT console
// layer. On POSIX, std::fwrite is safe so no special handling needed.
//
// No <windows.h> in this header — Win32 calls are isolated in safe_write.cpp
// to prevent namespace pollution (ERROR, max, min, etc. breaking downstream).
//
// When UM790_HAS_STD_PRINT == 1 (real implementation), this block is skipped
// entirely — the caller #includes <print> themselves via the guard.
// =============================================================================
#if !UM790_HAS_STD_PRINT && UM790_HAS_STD_FORMAT
#  include <cstdio>
#  include <cstddef>
#  include <format>
#  include <string>

// Platform-safe write function provided by safe_write.cpp.
// On Windows: uses Win32 WriteFile() to avoid MinGW CRT pipe crash.
// On POSIX: uses std::fwrite (safe on all platforms).
extern "C" std::size_t hq_safe_write(int fd, const char* data, std::size_t len);

namespace std {
    template<typename... Args>
    inline void print(std::format_string<Args...> fmt, Args&&... args) {
        auto s = std::format(fmt, std::forward<Args>(args)...);
        hq_safe_write(1, s.data(), s.size());
    }
    template<typename... Args>
    inline void print(std::FILE* f, std::format_string<Args...> fmt, Args&&... args) {
        auto s = std::format(fmt, std::forward<Args>(args)...);
        hq_safe_write(f == stdout ? 1 : 2, s.data(), s.size());
    }
    template<typename... Args>
    inline void println(std::format_string<Args...> fmt, Args&&... args) {
        auto s = std::format(fmt, std::forward<Args>(args)...);
        s += '\n';
        hq_safe_write(1, s.data(), s.size());
    }
    template<typename... Args>
    inline void println(std::FILE* f, std::format_string<Args...> fmt, Args&&... args) {
        auto s = std::format(fmt, std::forward<Args>(args)...);
        s += '\n';
        hq_safe_write(f == stdout ? 1 : 2, s.data(), s.size());
    }
} // namespace std
#endif
