#include <cstdio>
#include <cstdlib>
#include <vector>
#include <queue>

struct StagingConfig {
    std::size_t buffer_count = 8;
    std::size_t buffer_size_bytes = 64ULL * 1024 * 1024;
    bool pinned = true;
};

struct HostBuffer {
    std::vector<unsigned char> vec;
    void* pinned_ptr = nullptr;
    std::size_t buf_size = 0;
    explicit HostBuffer(std::size_t size) : vec(size), buf_size(size) {}
    unsigned char* data() { return pinned_ptr ? static_cast<unsigned char*>(pinned_ptr) : vec.data(); }
    std::size_t size() const { return buf_size; }
    bool empty() const { return pinned_ptr == nullptr && vec.empty(); }
};

class Impl {
public:
    StagingConfig cfg_;
    std::vector<HostBuffer> buffers_;
    std::queue<std::size_t> free_indices_;
    explicit Impl(StagingConfig cfg) : cfg_(std::move(cfg)) {
        buffers_.reserve(cfg_.buffer_count);
        for (std::size_t i = 0; i < cfg_.buffer_count; ++i) {
            buffers_.emplace_back(cfg_.buffer_size_bytes);
            free_indices_.push(i);
        }
    }
};

int main() {
    std::printf("Starting minimal test\n");
    StagingConfig cfg;
    cfg.buffer_count = 4;
    cfg.buffer_size_bytes = 1ULL * 1024 * 1024;
    cfg.pinned = false;
    std::printf("Before Impl ctor\n");
    Impl impl(cfg);
    std::printf("After Impl ctor, buffers=%zu free=%zu\n", impl.buffers_.size(), impl.free_indices_.size());
    std::printf("TEST PASSED\n");
    return 0;
}
