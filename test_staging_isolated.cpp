#include <hq/staging_manager.hpp>
#include <cstdio>
#include <cstdlib>

int main() {
    std::printf("Starting isolated staging manager test\n");
    fflush(stdout);
    
    hq::StagingConfig cfg;
    cfg.buffer_count = 4;
    cfg.buffer_size_bytes = 1ULL * 1024 * 1024;
    cfg.pinned = false;
    
    std::printf("Config: count=%zu size=%zu pinned=%d\n", cfg.buffer_count, cfg.buffer_size_bytes, cfg.pinned);
    fflush(stdout);
    
    try {
        std::printf("About to construct EmbeddingStagingManager...\n");
        fflush(stdout);
        hq::EmbeddingStagingManager mgr(cfg);
        std::printf("Constructed successfully!\n");
        fflush(stdout);
        
        std::printf("total_capacity=%zu available=%zu\n", mgr.total_capacity(), mgr.available_count());
        fflush(stdout);
        
        auto buf = mgr.acquire();
        if (buf) {
            std::printf("Acquired buffer: capacity=%zu\n", buf->capacity);
            mgr.release(*buf);
            std::printf("Released buffer\n");
        } else {
            std::printf("Acquire failed\n");
        }
        
        std::printf("TEST PASSED\n");
        return 0;
    } catch (const std::exception& e) {
        std::printf("EXCEPTION: %s\n", e.what());
        return 1;
    }
}
