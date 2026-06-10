#include "hq/david_propup_engine.hpp"
#include <cstdio>

extern "C" std::size_t hq_safe_write(int fd, const char* data, std::size_t len) {
    return std::fwrite(data, 1, len, fd == 1 ? stdout : stderr);
}

int main() {
    std::printf("Running staging manager lifecycle test only...\n");
    auto r = hq::propup::propup_staging_manager_lifecycle(nullptr);
    std::printf("Result: passed=%d skipped=%d failed=%d msg=%s\n", r.passed, r.skipped, r.failed, r.message.c_str());
    return r.failed ? 1 : 0;
}
