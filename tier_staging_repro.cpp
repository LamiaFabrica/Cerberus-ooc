#include "hq/tiered_memory_manager.hpp"
#include "hq/staging_manager.hpp"
#include <cstdio>
#include <vector>

int main() {
    std::puts("=== Tier + Staging reproducer ===");

    // E2E tier cold spill
    {
        std::puts("Test: e2e_tier_cold_spill");
        hq::TieredMemoryConfig cfg; cfg.warm_capacity_bytes = 0; cfg.cool_capacity_bytes = 0;
        hq::TieredMemoryManager tmm(cfg);
        auto r = tmm.allocate(64, hq::MemoryTier::Cold);
        if (!r) { std::puts("FAIL: cold allocate"); return 1; }
        if (r->tier != hq::MemoryTier::Cold) { std::puts("FAIL: expected cold"); return 1; }
        (void)tmm.free(r->handle);
        std::puts("PASS: e2e_tier_cold_spill");
    }
    // E2E tier promote warm
    {
        std::puts("Test: e2e_tier_promote_warm");
        hq::TieredMemoryManager tmm;
        auto r = tmm.allocate(64, hq::MemoryTier::Warm);
        if (!r) { std::puts("FAIL: warm allocate"); return 1; }
        std::uint8_t* p = static_cast<std::uint8_t*>(r->ptr);
        for (int i=0;i<64;++i) p[i]=0xAB;
        bool ok=true;
        for (int i=0;i<64;++i) if (p[i]!=0xAB) {ok=false;break;}
        (void)tmm.free(r->handle);
        if (!ok) { std::puts("FAIL: warm readback"); return 1; }
        std::puts("PASS: e2e_tier_promote_warm");
    }
    // E2E tier demote cool
    {
        std::puts("Test: e2e_tier_demote_cool");
        hq::TieredMemoryConfig cfg; cfg.warm_capacity_bytes = 0;
        hq::TieredMemoryManager tmm(cfg);
        auto r = tmm.allocate(64, hq::MemoryTier::Cool);
        if (!r) { std::puts("FAIL: cool allocate"); return 1; }
        std::uint8_t* p = static_cast<std::uint8_t*>(r->ptr);
        for (int i=0;i<64;++i) p[i]=0xCD;
        bool ok=true;
        for (int i=0;i<64;++i) if (p[i]!=0xCD) {ok=false;break;}
        (void)tmm.free(r->handle);
        if (!ok) { std::puts("FAIL: cool readback"); return 1; }
        std::puts("PASS: e2e_tier_demote_cool");
    }
    // robust tier thrashing
    {
        std::puts("Test: robust_tier_thrashing");
        hq::TieredMemoryConfig cfg;
        cfg.warm_capacity_bytes = 512;
        cfg.warm_watermark_pct = 0.90f;
        hq::TieredMemoryManager tmm(cfg);
        std::vector<hq::TierHandle> handles;
        for (int round = 0; round < 16; ++round) {
            for (int i = 0; i < 8; ++i) {
                auto r = tmm.allocate(64, hq::MemoryTier::Cool);
                if (r) handles.push_back(r->handle);
            }
            for (auto h : handles) { (void)tmm.promote(h); }
            for (auto h : handles) { (void)tmm.demote(h); }
            for (std::size_t i = 0; i < handles.size(); i += 2) {
                (void)tmm.free(handles[i]);
            }
            handles.clear();
        }
        std::puts("PASS: robust_tier_thrashing");
    }
    // Now staging
    {
        std::puts("Test: staging_acquire_release");
        hq::StagingConfig cfg;
        cfg.buffer_count = 2;
        cfg.buffer_size_bytes = 4096;
        cfg.pinned = false;
        hq::EmbeddingStagingManager mgr(cfg);
        auto buf = mgr.acquire();
        if (!buf) { std::puts("FAIL: acquire"); return 1; }
        if (buf->capacity != 4096) { std::puts("FAIL: capacity"); return 1; }
        mgr.release(*buf);
        if (mgr.available_count() != 2) { std::puts("FAIL: release"); return 1; }
        std::puts("PASS: staging_acquire_release");
    }
    std::puts("=== ALL PASS ===");
    return 0;
}
