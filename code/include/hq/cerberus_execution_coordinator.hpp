#pragma once
/// @file cerberus_execution_coordinator.hpp
/// @copyright Copyright (c) 2026 D Hargreaves (AKA Roylepython). LamiaFabrica. All rights reserved.
///
/// Raw pointer pass-through is incompatible with the Cerberus vision because it
/// treats memory as an undifferentiated heap. Cerberus must own the memory
/// loop: decide tier placement, track residency, and promote/demote based on
/// reuse pressure. This coordinator is the first bespoke execution layer that
/// makes memory decisions before calling the vendor backend.
///
/// ===========================================================================
/// Dependencies
/// ===========================================================================

#include "hq/npu_backend_unified.hpp"
#include "hq/tiered_memory_manager.hpp"

#include <cstring>
#include <expected>
#include <span>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>

namespace hq {

// ===========================================================================
// CerberusExecutionCoordinator — memory-owning execution shim
//
// The coordinator is the MANDATORY path between the pipeline and any backend.
// No caller may pass raw pointers directly to backend->execute().
//
// Memory flow:
//   1. User provides input/output host buffers.
//   2. Coordinator allocates tiered buffers for inputs (copies from host).
//   3. Coordinator allocates tiered buffers for outputs (tiered per reuse analysis).
//   4. Coordinator calls backend->execute() with tiered buffers.
//   5. Coordinator copies tiered outputs back to user host buffers.
//
// This makes Cerberus the owner of all intermediate memory.
// ===========================================================================

class CerberusExecutionCoordinator {
public:
    explicit CerberusExecutionCoordinator(TieredMemoryManager& manager);

    /// @brief Run a compiled kernel through the given backend.
    ///
    /// The coordinator assumes OWNERSHIP of the execution memory loop:
    ///   - Inputs are COPIED into tiered allocations (default: Cool, promoted
    ///     to Warm when the backend is in synthetic_mode() and the input is
    ///     listed in high_reuse_tensors).
    ///   - Outputs are allocated in tiered memory (Warm for high_reuse_tensors,
    ///     Cool otherwise).
    ///   - Results are copied back to user_outputs before return.
    ///
    /// @param backend      Backend to execute on (must outlive this call).
    /// @param kernel       Compiled kernel metadata (including high_reuse_tensors).
    /// @param user_inputs  Pointers to caller-owned input host buffers.
    /// @param user_outputs Pointers to caller-owned output host buffers.
    /// @param debug_log    Optional file stream for visible tier decisions.
    [[nodiscard]] std::expected<void, std::string>
    run(npu::INpuBackend& backend,
        const npu::CompiledKernel& kernel,
        std::span<const std::byte*> user_inputs,
        std::span<std::byte*> user_outputs,
        std::ofstream* debug_log = nullptr);

private:
    TieredMemoryManager& manager_;

    [[nodiscard]] MemoryTier pick_input_tier(const npu::CompiledKernel& kernel,
                                             std::size_t input_idx) const noexcept;

    [[nodiscard]] MemoryTier pick_output_tier(const npu::CompiledKernel& kernel,
                                              std::size_t output_idx) const noexcept;
};

} // namespace hq
