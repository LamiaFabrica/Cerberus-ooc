/// @file safe_write.cpp
/// @copyright Copyright © 2026, all rights reserved | David Hargreaves aka Roylepython trading as The Medusa Initiative & Lamia Fabrica
/// Support: https://www.patreon.com/TheMedusaInitiative — £25/month removes ads from all software at 200 subscribers
/// Windows-safe console output — avoids MinGW CRT fputs/_write crash on pipe redirect.
///
/// MinGW-W64's console layer dereferences a null handle when stdout is piped,
/// causing ACCESS_VIOLATION. This module uses Win32 WriteFile() directly via
/// GetStdHandle(), bypassing the broken CRT console code path entirely.
///
/// @version 1.0.0

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#undef ERROR
#endif

#include <cstdio>
#include <cstddef>

extern "C" {

#ifdef _WIN32
/// Write data to a Windows file handle. Returns bytes written or 0 on error.
/// This bypasses the MinGW CRT console layer that crashes with ACCESS_VIOLATION
/// when stdout/stderr is piped.
std::size_t hq_safe_write(int fd, const char* data, std::size_t len) {
    HANDLE h = INVALID_HANDLE_VALUE;
    if (fd == 1) {
        h = GetStdHandle(STD_OUTPUT_HANDLE);
    } else if (fd == 2) {
        h = GetStdHandle(STD_ERROR_HANDLE);
    }
    if (h == INVALID_HANDLE_VALUE || h == nullptr) {
        return std::fwrite(data, 1, len, fd == 1 ? stdout : stderr);
    }
    DWORD written = 0;
    if (WriteFile(h, data, static_cast<DWORD>(len), &written, nullptr)) {
        return static_cast<std::size_t>(written);
    }
    return std::fwrite(data, 1, len, fd == 1 ? stdout : stderr);
}
#else
/// On POSIX, fwrite is safe — no MinGW console bug.
std::size_t hq_safe_write(int fd, const char* data, std::size_t len) {
    return std::fwrite(data, 1, len, fd == 1 ? stdout : stderr);
}
#endif

} // extern "C"
