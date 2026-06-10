#include <hq/staging_manager.hpp>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" std::size_t hq_safe_write(int fd, const char* data, std::size_t len) {
    return std::fwrite(data, 1, len, fd == 1 ? stdout : stderr);
}

int main() {
    std::printf("Starting propup staging test\n");
    fflush(stdout);
    
    hq::StagingConfig cfg;
    cfg.buffer_count = 4;
    cfg.buffer_size_bytes = 1ULL * 1024 * 1024;
    cfg.pinned = false;
    
    std::printf("About to construct EmbeddingStagingManager...\n");
    fflush(stdout);
    
    hq::EmbeddingStagingManager mgr(cfg);
    
    std::printf("Constructed. total_capacity=%zu available=%zu\n", mgr.total_capacity(), mgr.available_count());
    fflush(stdout);
    
    // Acquire all 4
    std::vector<hq::StagingBuffer> acquired;
    for (int i = 0; i < 4; ++i) {
        auto result = mgr.acquire();
        if (!result.has_value()) {
            std::printf("FAILED to acquire buffer %d\n", i);
            return 1;
        }
        acquired.push_back(result.value());
        std::printf("Acquired buffer %d: capacity=%zu\n", i, acquired.back().capacity);
    }
    
    // 5th should fail
    auto result5 = mgr.acquire();
    if (result5.has_value()) {
        std::printf("FAILED: 5th acquire should have failed\n");
        return 1;
    }
    std::printf("5th acquire correctly failed\n");
    
    // Release and reacquire
    mgr.release(acquired[0]);
    auto result_after = mgr.acquire();
    if (!result_after.has_value()) {
        std::printf("FAILED: reacquire after release should succeed\n");
        return 1;
    }
    std::printf("Reacquired after release\n");
    
    // Release another for copy-in test
    mgr.release(acquired[1]);
    
    // Copy-in test
    auto buf_result = mgr.acquire();
    if (!buf_result.has_value()) {
        std::printf("FAILED: acquire for copy-in\n");
        return 1;
    }
    auto buf = buf_result.value();
    
    std::vector<std::byte> test_data(512);
    for (std::size_t i = 0; i < 512; ++i) {
        test_data[i] = static_cast<std::byte>(i % 256);
    }
    
    auto copy_result = mgr.copy_in(buf, test_data);
    if (!copy_result.has_value()) {
        std::printf("FAILED: copy_in\n");
        return 1;
    }
    if (copy_result.value() != 512) {
        std::printf("FAILED: copy_in returned %zu expected 512\n", copy_result.value());
        return 1;
    }
    if (buf.used != 512) {
        std::printf("FAILED: buf.used=%zu expected 512\n", buf.used);
        return 1;
    }
    
    for (std::size_t i = 0; i < 512; ++i) {
        if (buf.data[i] != static_cast<std::byte>(i % 256)) {
            std::printf("FAILED: data mismatch at %zu\n", i);
            return 1;
        }
    }
    
    mgr.release(buf);
    
    std::printf("TEST PASSED\n");
    return 0;
}
