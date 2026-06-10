#include "hq/staging_manager.hpp"
#include <cstdio>

extern "C" std::size_t hq_safe_write(int fd, const char* data, std::size_t len);

int main() {
    std::printf("Starting staging test in propup env\n");
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
