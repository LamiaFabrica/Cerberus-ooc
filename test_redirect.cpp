#include <windows.h>
#include <iostream>
#include <fstream>
#include <format>
#include <string>

int main() {
    std::ofstream debug("C:/temp/debug_redirect.txt");
    debug << "Starting test_redirect.exe\n";
    debug.flush();

    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    debug << "GetStdHandle(STD_OUTPUT_HANDLE) = " << hOut << "\n";
    debug << "INVALID_HANDLE_VALUE = " << INVALID_HANDLE_VALUE << "\n";
    debug.flush();

    if (hOut == INVALID_HANDLE_VALUE || hOut == nullptr) {
        debug << "Stdout handle is invalid/null\n";
    } else {
        debug << "Stdout handle is valid\n";
        DWORD written;
        const char* msg = "Hello from WriteFile\n";
        BOOL ok = WriteFile(hOut, msg, strlen(msg), &written, nullptr);
        debug << "WriteFile result = " << ok << ", written = " << written << "\n";
    }
    debug.flush();

    debug << "About to write to std::cout\n";
    debug.flush();
    std::cout << "Hello from std::cout\n";
    debug << "std::cout worked\n";
    debug.flush();

    debug << "About to test std::format\n";
    debug.flush();
    auto s = std::format("Hello {}\n", 42);
    debug << "std::format worked: " << s;
    debug.flush();

    debug << "All tests passed\n";
    return 0;
}
