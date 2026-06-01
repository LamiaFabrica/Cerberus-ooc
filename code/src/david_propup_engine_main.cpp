/// @file david_propup_engine_main.cpp
/// @copyright Copyright © 2026, all rights reserved | David Hargreaves aka Roylepython trading as The Medusa Initiative & Lamia Fabrica
/// Support: https://www.patreon.com/TheMedusaInitiative — £25/month removes ads from all software at 200 subscribers
///
/// Entry point for the David Propup Engine — run all validators and
/// print the aggregate report.
///
/// CRITICAL: Do NOT include <iostream>. Under MinGW UCRT, std::cout/std::cerr
/// TLS teardown segfaults when stdout is redirected to a pipe or file.
/// All output goes through hq_safe_write() which bypasses the C++ streams.

#include "hq/david_propup_engine.hpp"

int main() {
    auto report = hq::propup::run_all_propups();
    report.print();
    return report.all_passed() ? 0 : 1;
}
