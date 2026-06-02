#include <windows.h>
#include <iostream>
#include <fstream>
#include <format>
#include <cstring>

// Include a Cerberus header to trigger any static initialization
#include "hq/david_propup_engine.hpp"

int main() {
    std::ofstream debug("C:/temp/debug_redirect2.txt");
    debug << "Starting test_redirect2.exe\n";
    debug.flush();

    debug << "About to call run_all_propups\n";
    debug.flush();

    auto report = hq::propup::run_all_propups();

    debug << "run_all_propups returned\n";
    debug.flush();

    report.print();

    debug << "report.print() returned\n";
    debug.flush();

    return report.all_passed() ? 0 : 1;
}
