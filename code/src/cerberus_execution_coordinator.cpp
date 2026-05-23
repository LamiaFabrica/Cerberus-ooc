/// @file cerberus_execution_coordinator.cpp
/// @copyright Copyright (c) 2026 D Hargreaves (AKA Roylepython). LamiaFabrica. All rights reserved.
/// Cerberus Execution Coordinator — ties tiered memory management to NPU backend dispatch.
///
/// @version 2.0.0 — inputs and outputs now staged through TieredMemoryManager.

#include "hq/cerberus_execution_coordinator.hpp"

#include <algorithm>
#include <cstring>
#include <expected>
#include <span>
#include <string>
#include <vector>

namespace hq {

CerberusExecutionCoordinator::CerberusExecutionCoordinator(TieredMemoryManager& manager)
    : manager_{manager} {}

MemoryTier CerberusExecutionCoordinator::pick_input_tier(
    const npu::CompiledKernel& kernel, std::size_t input_idx) const noexcept {
    MemoryTier preferred = MemoryTier::Cool;
    if (kernel.input_names.size() > input_idx) {
        const auto& name = kernel.input_names[input_idx];
        if (std::find(kernel.high_reuse_tensors.begin(),
                      kernel.high_reuse_tensors.end(),
                      name) != kernel.high_reuse_tensors.end()) {
            preferred = MemoryTier::Warm;
        }
    }
    return preferred;
}

MemoryTier CerberusExecutionCoordinator::pick_output_tier(
    const npu::CompiledKernel& kernel, std::size_t output_idx) const noexcept {
    MemoryTier preferred = MemoryTier::Cool;
    if (kernel.output_names.size() > output_idx) {
        const auto& name = kernel.output_names[output_idx];
        if (std::find(kernel.high_reuse_tensors.begin(),
                      kernel.high_reuse_tensors.end(),
                      name) != kernel.high_reuse_tensors.end()) {
            preferred = MemoryTier::Warm;
        }
    }
    return preferred;
}

std::expected<void, std::string>
CerberusExecutionCoordinator::run(npu::INpuBackend& backend,
                                  const npu::CompiledKernel& kernel,
                                  std::span<const std::byte*> user_inputs,
                                  std::span<std::byte*> user_outputs,
                                  std::ofstream* debug_log) {
    // --- validate counts ------------------------------------------------------
    if (user_inputs.size() != kernel.inputs.size()) {
        return std::unexpected{
            "input count mismatch: got " + std::to_string(user_inputs.size()) +
            ", expected " + std::to_string(kernel.inputs.size())};
    }
    if (user_outputs.size() != kernel.outputs.size()) {
        return std::unexpected{
            "output count mismatch: got " + std::to_string(user_outputs.size()) +
            ", expected " + std::to_string(kernel.outputs.size())};
    }

    auto log = [&](const std::string& msg) {
        if (debug_log && *debug_log) {
            *debug_log << msg << "\n";
            debug_log->flush();
        }
    };

    // --- allocate tiered INPUT buffers and copy from user ---------------------
    std::vector<ScopedTierAlloc> input_allocs;
    input_allocs.reserve(kernel.inputs.size());
    std::vector<const std::byte*> tiered_input_ptrs;
    tiered_input_ptrs.reserve(kernel.inputs.size());

    for (std::size_t i = 0; i < kernel.inputs.size(); ++i) {
        MemoryTier tier = pick_input_tier(kernel, i);
        std::size_t sz = kernel.inputs[i].size_bytes();

        auto alloc_result = manager_.allocate(sz, tier);
        if (!alloc_result) {
            return std::unexpected{
                "tiered allocation failed for input " + std::to_string(i) +
                ": " + to_string(alloc_result.error())};
        }

        input_allocs.emplace_back(manager_, *alloc_result);
        if (!input_allocs.back().valid()) {
            return std::unexpected{
                "tiered allocation returned invalid handle for input " +
                std::to_string(i)};
        }

        std::byte* ptr = static_cast<std::byte*>(input_allocs.back().ptr());
        std::memcpy(ptr, user_inputs[i], sz);
        tiered_input_ptrs.push_back(ptr);

        log("[COORD] input " + std::to_string(i) +
            " staged -> " + to_string(input_allocs.back().tier()) +
            " tier (" + std::to_string(sz) + " bytes)");
    }

    // --- allocate tiered OUTPUT buffers -------------------------------------
    std::vector<ScopedTierAlloc> output_allocs;
    output_allocs.reserve(kernel.outputs.size());
    std::vector<std::byte*> tiered_output_ptrs;
    tiered_output_ptrs.reserve(kernel.outputs.size());

    for (std::size_t i = 0; i < kernel.outputs.size(); ++i) {
        MemoryTier tier = pick_output_tier(kernel, i);
        std::size_t sz = kernel.outputs[i].size_bytes();

        auto alloc_result = manager_.allocate(sz, tier);
        if (!alloc_result) {
            return std::unexpected{
                "tiered allocation failed for output " + std::to_string(i) +
                ": " + to_string(alloc_result.error())};
        }

        output_allocs.emplace_back(manager_, *alloc_result);
        if (!output_allocs.back().valid()) {
            return std::unexpected{
                "tiered allocation returned invalid handle for output " +
                std::to_string(i)};
        }

        std::byte* ptr = static_cast<std::byte*>(output_allocs.back().ptr());
        tiered_output_ptrs.push_back(ptr);

        log("[COORD] output " + std::to_string(i) +
            " allocated -> " + to_string(output_allocs.back().tier()) +
            " tier (" + std::to_string(sz) + " bytes)");
    }

    // --- dispatch to vendor backend with TIERED buffers, not raw user ptrs ---
    std::span<std::byte*> tiered_out_span{tiered_output_ptrs.data(), tiered_output_ptrs.size()};
    auto exec_result = backend.execute(kernel,
                                       std::span<const std::byte*>{tiered_input_ptrs.data(),
                                                                   tiered_input_ptrs.size()},
                                       tiered_out_span);
    if (!exec_result) {
        return std::unexpected{std::move(exec_result.error())};
    }

    // --- copy tiered outputs back to user-provided host buffers ---------------
    for (std::size_t i = 0; i < kernel.outputs.size(); ++i) {
        std::memcpy(user_outputs[i], tiered_output_ptrs[i], kernel.outputs[i].size_bytes());
    }

    // --- log working set summary ----------------------------------------------
    if (kernel.estimated_working_set_bytes > 0) {
        log("[COORD] working_set=" + std::to_string(kernel.estimated_working_set_bytes) +
            " bytes  high_reuse_tensors=" + std::to_string(kernel.high_reuse_tensors.size()));
    }

    return {};
}

} // namespace hq
