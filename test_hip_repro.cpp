#include "hq/hip_graph_denoiser.hpp"
#include <iostream>

extern "C" std::size_t hq_safe_write(int fd, const char* data, std::size_t len) {
    (void)fd;
    return std::fwrite(data, 1, len, stdout);
}

int main() {
    std::cout << "Before construction\n";
    {
        hq::GraphConfig cfg;
        cfg.num_steps = 4;
        cfg.latent_count = 1 * 4 * 64 * 64;
        cfg.enable_capture = false;
        hq::HIPGraphDenoiser denoiser(cfg);
        std::cout << "Constructed, available=" << denoiser.is_available() << "\n";
    }
    std::cout << "After destruction\n";
    return 0;
}
