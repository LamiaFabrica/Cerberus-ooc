#include <stdio>
#include <windows.h>
#include <limits.h>

int main() {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    SetUnhandledExceptionFilter([](PEXCEPTION_POINTERS) -> LONG { ExitProcess(42); return EXCEPTION_EXECUTE_HANDLER; });

    std::fprintf(stderr, "LOAD_DLL\n"); std::fflush(stderr);
    HMODULE h = LoadLibraryW(L"openvino.dll");
    if (!h) { std::fprintf(stderr, "LOAD_FAIL\n"); return 1; }
    std::fprintf(stderr, "LOAD_OK\n"); std::fflush(stderr);

    auto create = reinterpret_cast<int (*)(void**)>(GetProcAddress(h, "ov_core_create"));
    if (!create) { std::fprintf(stderr, "SYM_FAIL\n"); return 1; }
    std::fprintf(stderr, "SYM_OK\n"); std::fflush(stderr);

    void* core = nullptr;
    std::fprintf(stderr, "CORE_CREATE\n"); std::fflush(stderr);
    int st = create(&core);
    std::fprintf(stderr, "CORE_DONE st=%d core=%p\n", st, (void*)core); std::fflush(stderr);
    return 0;
}
