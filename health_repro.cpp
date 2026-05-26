#include "hq/health_score.hpp"
#include <cstdio>

int main() {
    std::puts("=== HealthScore reproducer ===");
    {
        std::puts("Test: grade_a");
        hq::PipelineHealthScore scorer;
        scorer.update_gpu(95.0f, 60.0f);
        scorer.update_hailo(90.0f, 55.0f);
        scorer.update_npu_utilization(88.0f);
        scorer.update_stability(2.0f);
        scorer.update_memory_bw(20.0f);
        scorer.update_tbt(10.0f);
        for (int i = 0; i < 10; ++i) scorer.update_recovery(true);
        auto report = scorer.compute();
        if (report.grade != hq::HealthGrade::A) { std::puts("FAIL: not A"); return 1; }
        std::puts("PASS: grade_a");
    }
    {
        std::puts("Test: grade_f");
        hq::PipelineHealthScore scorer;
        scorer.update_gpu(0.0f, 95.0f);
        scorer.update_hailo(0.0f, 95.0f);
        auto report = scorer.compute();
        if (report.grade != hq::HealthGrade::F) { std::puts("FAIL: not F"); return 1; }
        std::puts("PASS: grade_f");
    }
    {
        std::puts("Test: weights_normalize");
        hq::HealthWeights w{1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f};
        hq::PipelineHealthScore scorer(w);
        auto report = scorer.compute();
        if (report.overall_score < 0.0f || report.overall_score > 100.0f) { std::puts("FAIL: bounds"); return 1; }
        std::puts("PASS: weights_normalize");
    }
    {
        std::puts("Test: recovery_rate");
        hq::PipelineHealthScore scorer;
        for (int i = 0; i < 10; ++i) scorer.update_recovery(i < 9);
        auto report = scorer.compute();
        if (report.raw_metrics.recovery_success_rate_percent < 80.0f) { std::puts("FAIL: rate"); return 1; }
        std::puts("PASS: recovery_rate");
    }
    std::puts("=== ALL PASS ===");
    return 0;
}
