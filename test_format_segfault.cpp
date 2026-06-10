#include <format>
#include <cstdio>
#include <string>

extern "C" std::size_t hq_safe_write(int fd, const char* data, std::size_t len) {
    return std::fwrite(data, 1, len, fd == 1 ? stdout : stderr);
}

int main() {
    std::printf("Test 1: std::format with string only\n");
    std::string msg1 = std::format("[staging] WARNING: pinned=true but HIP not available at compile time, using regular memory\n");
    hq_safe_write(1, msg1.data(), msg1.size());
    
    std::printf("Test 2: std::format with int arg\n");
    std::string msg2 = std::format("Buffer count: {}\n", 4);
    hq_safe_write(1, msg2.data(), msg2.size());
    
    std::printf("Test 3: std::format with size_t arg\n");
    std::size_t sz = 1024;
    std::string msg3 = std::format("Size: {}\n", sz);
    hq_safe_write(1, msg3.data(), msg3.size());
    
    std::printf("Test 4: std::format with const char* arg\n");
    const char* str = "hello";
    std::string msg4 = std::format("String: {}\n", str);
    hq_safe_write(1, msg4.data(), msg4.size());
    
    std::printf("Test 5: std::format with double arg (KNOWN BUG)\n");
    double d = 3.14;
    std::string msg5 = std::format("Double: {}\n", d);
    hq_safe_write(1, msg5.data(), msg5.size());
    
    std::printf("ALL PASSED\n");
    return 0;
}
