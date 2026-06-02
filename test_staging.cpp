#include "hq/staging_manager.hpp"
#include <cstddef>

extern "C" std::size_t hq_safe_write(int fd, const char* data, std::size_t len);

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#undef ERROR
#endif
#include <cstdio>

extern "C" std::size_t hq_safe_write(int fd, const char* data, std::size_t len) {
#ifdef _WIN32
    HANDLE h = INVALID_HANDLE_VALUE;
    if (fd == 1) h = GetStdHandle(STD_OUTPUT_HANDLE);
    else if (fd == 2) h = GetStdHandle(STD_ERROR_HANDLE);
    if (h == INVALID_HANDLE_VALUE || h == nullptr) {
        return std::fwrite(data, 1, len, fd == 1 ? stdout : stderr);
    }
    DWORD written = 0;
    if (WriteFile(h, data, static_cast<DWORD>(len), &written, nullptr)) {
        return static_cast<std::size_t>(written);
    }
    DWORD err = GetLastError();
    if (err == ERROR_BROKEN_PIPE || err == ERROR_NO_DATA || err == ERROR_INVALID_HANDLE) {
        return 0;
    }
    return std::fwrite(data, 1, len, fd == 1 ? stdout : stderr);
#else
    return std::fwrite(data, 1, len, fd == 1 ? stdout : stderr);
#endif
}

int main() {
    hq_safe_write(1, "MAIN_START\n", 11);
    hq::StagingConfig cfg;
    cfg.buffer_count = 4;
    cfg.buffer_size_bytes = 1ULL * 1024 * 1024;
    cfg.pinned = false;
    hq_safe_write(1, "Before ctor\n", 12);
    hq::EmbeddingStagingManager mgr(cfg);
    hq_safe_write(1, "After ctor\n", 11);
    return 0;
}
