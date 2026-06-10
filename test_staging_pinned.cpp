#include <hq/staging_manager.hpp>
#include <cstdio>
#include <cstdlib>

extern "C" std::size_t hq_safe_write(int fd, const char* data, std::size_t len) {
    return std::fwrite(data, 1, len, fd == 1 ? stdout : stderr);
}

int main() {
    std::printf("Starting PINNED staging manager test\n");
    fflush(stdout);
    
    hq::StagingConfig cfg;
    cfg.buffer_count = 4;
    cfg.buffer_size_bytes = 1ULL * 1024 * 1024;
    cfg.pinned = true;  // PINNED!
    
    std::printf("Config: count=%zu size=%zu pinned=%d\n", cfg.buffer_count, cfg.buffer_size_bytes, cfg.pinned);
    fflush(stdout);
    
    try {
        std::printf("About to construct EmbeddingStagingManager with pinned=true...\n");
        fflush(stdout);
        hq::EmbeddingStagingManager mgr(cfg);
        std::printf("Constructed successfully!\n");
        fflush(stdout);
        
        std::printf("total_capacity=%zu available=%zu\n", mgr.total_capacity(), mgr.available_count());
        fflush(stdout);
        
        std::printf("TEST PASSED\n");
        return 0;
    } catch (const std::exception& e) {
        std::printf("EXCEPTION: %s\n", e.what());
        return 1;
    }
}
