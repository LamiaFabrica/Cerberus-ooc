#pragma once
/// @file cerberus_native_backend.hpp
/// @copyright Copyright (c) 2026 D Hargreaves. All rights reserved.
///
/// CerberusNativeBackend — a real INpuBackend that lowers Cerberus-owned
/// graph nodes to native CPU kernels. This is the honest CPU fallback AND
/// the correctness baseline for numerical validation.
///
/// @version 1.0.0

#include "hq/npu_backend_unified.hpp"
#include "hq/cerberus_native_kernels.hpp"

#include <vector>
#include <string>

namespace hq::cerberus {

// ===========================================================================
// CerberusNativeBackend — pure C++ execution, no vendor DLLs
// ===========================================================================

class CerberusNativeBackend final : public npu::INpuBackend {
public:
    CerberusNativeBackend();
    ~CerberusNativeBackend() override;

    [[nodiscard]] std::expected<npu::CompiledKernel, std::string>
    compile(const npu::KernelGraph& graph, const npu::TargetConfig& cfg) override;

    [[nodiscard]] std::expected<void, std::string>
    execute(const npu::CompiledKernel& kernel,
            std::span<const std::byte*> inputs,
            std::span<std::byte*> outputs) override;

    [[nodiscard]] bool can_compile_for(std::string_view target_name) const override;
    [[nodiscard]] bool is_available() const override { return true; }
    [[nodiscard]] std::string name() const override { return "Cerberus-Native"; }
    [[nodiscard]] bool synthetic_mode() const noexcept override { return false; }
    [[nodiscard]] std::string unavailable_reason() const override { return {}; }
    [[nodiscard]] float utilization() const override { return -1.0f; }
    [[nodiscard]] float temperature() const override { return -1.0f; }

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hq::cerberus
