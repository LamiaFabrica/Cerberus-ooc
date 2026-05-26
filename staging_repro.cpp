#include "hq/staging_manager.hpp"
#include <cstdio>
#include <cstdlib>

int main() {
    std::puts("=== Staging reproducer ===");
    {
        std::puts("Test 1: acquire/release");
        hq::StagingConfig cfg;
        cfg.buffer_count = 2;
        cfg.buffer_size_bytes = 4096;
        cfg.pinned = false;
        hq::EmbeddingStagingManager mgr(cfg);
        auto buf = mgr.acquire();
        if (!buf) { std::puts("FAIL: acquire"); return 1; }
        if (buf->capacity != 4096) { std::puts("FAIL: capacity"); return 1; }
        mgr.release(*buf);
        if (mgr.available_count() != 2) { std::puts("FAIL: available_count"); return 1; }
        std::puts("PASS: acquire/release");
    }
    {
        std::puts("Test 2: copy_in");
        hq::StagingConfig cfg; cfg.buffer_count = 1; cfg.buffer_size_bytes = 1024; cfg.pinned = false;
        hq::EmbeddingStagingManager mgr(cfg);
        auto buf = mgr.acquire();
        if (!buf) { std::puts("FAIL: acquire"); return 1; }
        std::vector<std::byte> src(512, static_cast<std::byte>(0xAB));
        auto copied = mgr.copy_in(*buf, src);
        if (!copied || *copied != 512) { std::puts("FAIL: copy_in"); return 1; }
        mgr.release(*buf);
        std::puts("PASS: copy_in");
    }
    {
        std::puts("Test 3: pool_exhausted");
        hq::StagingConfig cfg; cfg.buffer_count = 1; cfg.buffer_size_bytes = 256; cfg.pinned = false;
        hq::EmbeddingStagingManager mgr(cfg);
        auto b1 = mgr.acquire();
        if (!b1) { std::puts("FAIL: first acquire"); return 1; }
        auto b2 = mgr.acquire();
        if (b2) { std::puts("FAIL: second acquire should fail"); return 1; }
        if (b2.error() != hq::HostStagingError::PoolExhausted) { std::puts("FAIL: wrong error"); return 1; }
        mgr.release(*b1);
        std::puts("PASS: pool_exhausted");
    }
    std::puts("=== ALL PASS ===");
    return 0;
}
