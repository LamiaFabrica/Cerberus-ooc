#include "hq/staging_manager.hpp"
#include <cstdio>
#include <cstdlib>

extern "C" std::size_t hq_safe_write(int fd, const char* data, std::size_t len) {
    return std::fwrite(data, 1, len, fd == 1 ? stdout : stderr);
}

int main() {
    std::printf("Starting ctor segfault test\n");
    fflush(stdout);
    
    hq::StagingConfig cfg;
    cfg.buffer_count = 4;
    cfg.buffer_size_bytes = 1ULL * 1024 * 1024;
    cfg.pinned = false;
    
    std::printf("Before ctor\n");
    fflush(stdout);
    hq::EmbeddingStagingManager mgr(cfg);
    std::printf("After ctor, total=%zu avail=%zu\n", mgr.total_capacity(), mgr.available_count());
    fflush(stdout);
    
    std::printf("TEST PASSED\n");
    return 0;
}
