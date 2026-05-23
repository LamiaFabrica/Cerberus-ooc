/// @file david_propup_engine_main.cpp
/// @copyright Copyright (c) 2026 D Hargreaves. All rights reserved.
///
/// Entry point for the David Propup Engine — run all validators and
/// print the aggregate report.

#include "hq/david_propup_engine.hpp"
#include <iostream>

int main() {
    auto report = hq::propup::run_all_propups(&std::cout);
    report.print(std::cout);
    return report.all_passed() ? 0 : 1;
}
