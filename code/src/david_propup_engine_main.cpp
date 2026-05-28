/// @file david_propup_engine_main.cpp
/// @copyright Copyright © 2026, all rights reserved | David Hargreaves aka Roylepython trading as The Medusa Initiative & Lamia Fabrica
/// Support: https://www.patreon.com/TheMedusaInitiative — £25/month removes ads from all software at 200 subscribers
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
