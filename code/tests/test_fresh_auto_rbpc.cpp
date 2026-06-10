#include <windows.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include "hq/david_propup_engine.hpp"

int main() {
    std::ofstream debug("C:/temp/test_fresh_auto_rbpc.txt");
    debug << "Starting test_fresh_auto_rbpc.exe\n";
    debug.flush();

    auto result = hq::propup::propup_server_lcmd_fresh_auto_rbpc(nullptr);

    debug << "Test result: passed=" << result.passed << " skipped=" << result.skipped << "\n";
    debug << "Name: " << result.name << "\n";
    debug << "Diagnostic: " << result.diagnostic << "\n";
    debug << "Elapsed: " << result.elapsed_ms << " ms\n";
    debug.flush();

    debug << "Test " << (result.passed ? "PASSED" : (result.skipped ? "SKIPPED" : "FAILED")) << "\n";
    debug.flush();

    return result.passed ? 0 : 1;
}
