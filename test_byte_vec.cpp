#include <cstdio>
#include <vector>
#include <cstddef>

int main() {
    std::printf("Testing std::vector<std::byte> allocation...\n");
    fflush(stdout);
    
    try {
        std::vector<std::byte> vec(1024 * 1024);
        std::printf("Allocated %zu bytes successfully\n", vec.size());
        
        std::vector<std::byte> vec2;
        vec2.reserve(4);
        for (int i = 0; i < 4; ++i) {
            vec2.emplace_back(static_cast<std::byte>(i));
        }
        std::printf("Emplaced %zu elements\n", vec2.size());
        
        std::printf("TEST PASSED\n");
        return 0;
    } catch (...) {
        std::printf("EXCEPTION\n");
        return 1;
    }
}
