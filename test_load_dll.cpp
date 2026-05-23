#include <cstdio>
#include <windows.h>

int main() {
    std::fprintf(stderr, "STEP 1: Start\n"); std::fflush(stderr);

    HMODULE h = LoadLibraryW(L"openvino.dll");
    if (!h) {
        std::fprintf(stderr, "STEP 2: load FAILED\n"); std::fflush(stderr);
        return 1;
    }
    std::fprintf(stderr, "STEP 2: load OK\n"); std::fflush(stderr);

    auto* pfn = GetProcAddress(h, "ov_core_create");
    if (!pfn) {
        std::fprintf(stderr, "STEP 3: get ov_core_create FAILED\n"); std::fflush(stderr);
        return 1;
    }
    std::fprintf(stderr, "STEP 3: get ov_core_create OK\n"); std::fflush(stderr);

    std::fprintf(stderr, "STEP 4: calling ov_core_create...\n"); std::fflush(stderr);
    using fn = int(void**);
    int st = reinterpret_cast<fn*>(pfn)(&h); // dummy, but calls it
    std::fprintf(stderr, "STEP 4: call returned %d\n", st); std::fflush(stderr);

    std::fprintf(stderr, "PASS\n"); std::fflush(stderr);
    return 0;
}
