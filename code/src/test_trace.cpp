#include <windows.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "hq/npu_backend_unified.hpp"

extern "C" void __cyg_profile_func_enter(void* fn, void* call_site);
extern "C" void __cyg_profile_func_exit(void* fn, void* call_site);

static FILE* g_trace = nullptr;

extern "C" void __cyg_profile_func_enter(void* fn, void* call_site) {
    (void)call_site;
    if (!g_trace) g_trace = std::fopen("trace.txt", "w");
    if (g_trace) std::fprintf(g_trace, "ENTER %p\n", fn);
}
extern "C" void __cyg_profile_func_exit(void* fn, void* call_site) {
    (void)call_site;
    if (g_trace) std::fprintf(g_trace, "EXIT %p\n", fn);
}

int main() {
    if (!g_trace) g_trace = std::fopen("trace.txt", "w");
    if (g_trace) std::fprintf(g_trace, "MAIN_START\n");
    std::fflush(g_trace);
    hq::npu::IntelOpenVinoBackend intel;
    if (g_trace) std::fprintf(g_trace, "MAIN_AFTER_CTOR\n");
    std::fflush(g_trace);
    if (!intel.is_available()) {
        if (g_trace) std::fprintf(g_trace, "NOT_AVAIL %s\n", intel.unavailable_reason().c_str());
        std::fflush(g_trace);
        return 1;
    }
    if (g_trace) std::fprintf(g_trace, "AVAIL_YES\n");
    std::fflush(g_trace);
    return 0;
}
