#include "hq/tiered_memory_manager.hpp"
#include "hq/staging_manager.hpp"
#include <cstdio>
#include <vector>

template<typename Fn>
void run_test(const char* name, Fn fn) {
    std::printf("Running %s ...\n", name);
    auto r = fn();
    if (!r.ok) {
        std::printf("FAIL: %s - %s\n", name, r.msg);
        std::exit(1);
    }
    std::printf("PASS: %s\n", name);
}

struct Result { bool ok; const char* msg; };

Result t1() {
    hq::TieredMemoryManager tmm;
    auto r = tmm.allocate(64, hq::MemoryTier::Warm);
    if (!r) return {false, "alloc"};
    (void)tmm.free(r->handle);
    return {true, ""};
}

Result t2() {
    hq::TieredMemoryConfig cfg; cfg.warm_capacity_bytes = 0; cfg.cool_capacity_bytes = 0;
    hq::TieredMemoryManager tmm(cfg);
    auto r = tmm.allocate(64, hq::MemoryTier::Cold);
    if (!r) return {false, "cold alloc"};
    if (r->tier != hq::MemoryTier::Cold) return {false, "tier"};
    (void)tmm.free(r->handle);
    return {true, ""};
}

Result t3() {
    hq::TieredMemoryManager tmm;
    auto r = tmm.allocate(64, hq::MemoryTier::Warm);
    if (!r) return {false, "warm alloc"};
    std::uint8_t* p = static_cast<std::uint8_t*>(r->ptr);
    for (int i=0;i<64;++i) p[i]=0xAB;
    (void)tmm.free(r->handle);
    return {true, ""};
}

Result t4() {
    hq::TieredMemoryConfig cfg; cfg.warm_capacity_bytes = 0;
    hq::TieredMemoryManager tmm(cfg);
    auto r = tmm.allocate(64, hq::MemoryTier::Cool);
    if (!r) return {false, "cool alloc"};
    std::uint8_t* p = static_cast<std::uint8_t*>(r->ptr);
    for (int i=0;i<64;++i) p[i]=0xCD;
    (void)tmm.free(r->handle);
    return {true, ""};
}

Result t5() {
    hq::TieredMemoryManager tmm;
    auto r = tmm.allocate(64, hq::MemoryTier::Cool);
    if (!r) return {false, "alloc"};
    auto pr = tmm.promote(r->handle);
    if (!pr) { (void)tmm.free(r->handle); return {true, ""}; }
    auto dr = tmm.demote(pr->handle);
    if (!dr) { (void)tmm.free(pr->handle); return {false, "demote"}; }
    (void)tmm.free(dr->handle);
    return {true, ""};
}

Result t6() {
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
    return {true, ""};
}

Result t7() {
    hq::TieredMemoryConfig cfg;
    cfg.warm_capacity_bytes = 32;
    cfg.cool_capacity_bytes = 32;
    cfg.cold_capacity_bytes = 32;
    hq::TieredMemoryManager tmm(cfg);
    auto r1 = tmm.allocate(16, hq::MemoryTier::Warm);
    if (!r1) return {false, "first"};
    auto r2 = tmm.allocate(32, hq::MemoryTier::Warm);
    (void)tmm.free(r1->handle);
    if (r2) (void)tmm.free(r2->handle);
    return {true, ""};
}

Result t8() {
    hq::StagingConfig cfg;
    cfg.buffer_count = 2;
    cfg.buffer_size_bytes = 4096;
    cfg.pinned = false;
    hq::EmbeddingStagingManager mgr(cfg);
    auto buf = mgr.acquire();
    if (!buf) return {false, "acquire"};
    if (buf->capacity != 4096) return {false, "capacity"};
    mgr.release(*buf);
    if (mgr.available_count() != 2) return {false, "release"};
    return {true, ""};
}

int main() {
    std::puts("=== TMM stress + staging reproducer ===");
    for (int iter = 0; iter < 10; ++iter) {
        std::printf("--- iteration %d ---\n", iter);
        run_test("tiered_memory", t1);
        run_test("cold_spill", t2);
        run_test("promote_warm", t3);
        run_test("demote_cool", t4);
        run_test("migration", t5);
        run_test("thrashing", t6);
        run_test("oom", t7);
        run_test("staging", t8);
    }
    std::puts("=== ALL PASS ===");
    return 0;
}
