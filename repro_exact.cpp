/// Reproducer for staging crash after tier/health tests
#include "hq/tiered_memory_manager.hpp"
#include "hq/health_score.hpp"
#include "hq/staging_manager.hpp"
#include <cstdio>
#include <vector>

int main() {
    std::puts("=== Reproducer: tier + health + staging ===");

    // Simulate propup_tiered_memory
    {
        hq::TieredMemoryManager tmm;
        auto r = tmm.allocate(64, hq::MemoryTier::Warm);
        if (r) (void)tmm.free(r->handle);
        std::puts("PASS: tiered_memory");
    }
    // Simulate propup_tier_pressure_demotion
    {
        hq::TieredMemoryConfig cfg;
        cfg.warm_capacity_bytes = 256;
        cfg.cool_capacity_bytes = 256;
        hq::TieredMemoryManager tmm(cfg);
        std::vector<hq::TierAllocation> allocs;
        for (int i = 0; i < 8; ++i) {
            auto r = tmm.allocate(64, hq::MemoryTier::Warm);
            if (r) allocs.push_back(*r);
        }
        for (auto& a : allocs) (void)tmm.free(a.handle);
        std::puts("PASS: tier_pressure_demotion");
    }
    // Simulate propup_e2e_tier_cold_spill
    {
        hq::TieredMemoryConfig cfg; cfg.warm_capacity_bytes = 0; cfg.cool_capacity_bytes = 0;
        hq::TieredMemoryManager tmm(cfg);
        auto r = tmm.allocate(64, hq::MemoryTier::Cold);
        if (r) (void)tmm.free(r->handle);
        std::puts("PASS: e2e_tier_cold_spill");
    }
    // Simulate propup_e2e_tier_promote_warm
    {
        hq::TieredMemoryManager tmm;
        auto r = tmm.allocate(64, hq::MemoryTier::Warm);
        if (r) (void)tmm.free(r->handle);
        std::puts("PASS: e2e_tier_promote_warm");
    }
    // Simulate propup_e2e_tier_demote_cool
    {
        hq::TieredMemoryConfig cfg; cfg.warm_capacity_bytes = 0;
        hq::TieredMemoryManager tmm(cfg);
        auto r = tmm.allocate(64, hq::MemoryTier::Cool);
        if (r) (void)tmm.free(r->handle);
        std::puts("PASS: e2e_tier_demote_cool");
    }
    // Simulate propup_robust_tier_thrashing
    {
        hq::TieredMemoryConfig cfg; cfg.warm_capacity_bytes = 512; cfg.warm_watermark_pct = 0.90f;
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
    // Simulate propup_health_score_grade_a
    {
        hq::PipelineHealthScore scorer;
        scorer.update_gpu(100.0f, 0.0f); scorer.update_hailo(100.0f, 0.0f);
        scorer.update_latency(0.0f); scorer.update_memory(0.0f);
        scorer.update_recovery(true); scorer.update_stability(0.0f);
        scorer.update_npu_utilization(100.0f);
        auto report = scorer.compute();
        if (report.grade != hq::HealthGrade::A) { std::puts("FAIL: grade_a"); return 1; }
        std::puts("PASS: grade_a");
    }
    // Simulate propup_health_score_grade_f
    {
        hq::PipelineHealthScore scorer;
        scorer.update_gpu(5.0f, 95.0f); scorer.update_hailo(0.0f, 95.0f);
        scorer.update_latency(200.0f); scorer.update_memory(5.0f);
        scorer.update_recovery(false); scorer.update_stability(25.0f);
        scorer.update_npu_utilization(0.0f);
        auto report = scorer.compute();
        if (report.grade != hq::HealthGrade::F) { std::puts("FAIL: grade_f"); return 1; }
        std::puts("PASS: grade_f");
    }
    // Simulate propup_health_score_weights_normalize
    {
        hq::HealthWeights w;
        w.gpu_utilization = 0.5f; w.hailo_utilization = 0.5f; w.npu_utilization = 0.5f;
        w.latency = 0.5f; w.memory = 0.5f; w.recovery = 0.5f; w.thermal = 0.5f; w.stability = 0.5f;
        hq::PipelineHealthScore scorer(w);
        scorer.update_gpu(100.0f, 50.0f);
        auto report = scorer.compute();
        if (report.overall_score < 0.0f || report.overall_score > 100.0f) { std::puts("FAIL: weights"); return 1; }
        std::puts("PASS: weights_normalize");
    }
    // Simulate propup_health_score_recovery_rate
    {
        hq::PipelineHealthScore scorer;
        for (int i = 0; i < 10; ++i) scorer.update_recovery(i < 9);
        auto report = scorer.compute();
        if (report.raw_metrics.recovery_success_rate_percent < 80.0f) { std::puts("FAIL: recovery"); return 1; }
        std::puts("PASS: recovery_rate");
    }
    // NOW staging tests
    {
        hq::StagingConfig cfg; cfg.buffer_count = 2; cfg.buffer_size_bytes = 4096; cfg.pinned = false;
        hq::EmbeddingStagingManager mgr(cfg);
        auto buf = mgr.acquire();
        if (!buf) { std::puts("FAIL: staging acquire"); return 1; }
        if (buf->capacity != 4096) { std::puts("FAIL: capacity"); return 1; }
        mgr.release(*buf);
        if (mgr.available_count() != 2) { std::puts("FAIL: release"); return 1; }
        std::puts("PASS: staging_acquire_release");
    }
    {
        hq::StagingConfig cfg; cfg.buffer_count = 1; cfg.buffer_size_bytes = 1024; cfg.pinned = false;
        hq::EmbeddingStagingManager mgr(cfg);
        auto buf = mgr.acquire();
        if (!buf) { std::puts("FAIL: acquire"); return 1; }
        std::vector<std::byte> src(512, static_cast<std::byte>(0xAB));
        auto copied = mgr.copy_in(*buf, src);
        if (!copied || *copied != 512) { std::puts("FAIL: copy_in"); return 1; }
        mgr.release(*buf);
        std::puts("PASS: staging_copy_in");
    }
    {
        hq::StagingConfig cfg; cfg.buffer_count = 1; cfg.buffer_size_bytes = 256; cfg.pinned = false;
        hq::EmbeddingStagingManager mgr(cfg);
        auto b1 = mgr.acquire();
        if (!b1) { std::puts("FAIL: first"); return 1; }
        auto b2 = mgr.acquire();
        if (b2) { std::puts("FAIL: second should fail"); return 1; }
        if (b2.error() != hq::HostStagingError::PoolExhausted) { std::puts("FAIL: wrong error"); return 1; }
        mgr.release(*b1);
        std::puts("PASS: staging_pool_exhausted");
    }
    std::puts("=== ALL PASS ===");
    return 0;
}
