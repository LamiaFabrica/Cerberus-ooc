#include "hq/david_propup_engine.hpp"
#include "hq/safe_write.hpp"
#include <format>

int main() {
    auto r = hq::propup::propup_kernel_fma(nullptr);
    auto s = std::format("{}: {} ms\n", r.name, r.elapsed_ms);
    hq::hq_safe_write(1, s.data(), s.size());
    return 0;
}
