#include <cstdio>
#include <windows.h>

int main() {
    const char* msg = "Hello from test_redirect\n";
    DWORD written;
    WriteFile(GetStdHandle(STD_OUTPUT_HANDLE), msg, 25, &written, nullptr);
    return 0;
}
