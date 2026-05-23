#pragma once
/// @file npu_backend_unified.hpp
/// @copyright Copyright (c) 2026 D Hargreaves (AKA Roylepython). LamiaFabrica. All rights reserved.
/// Cerberus Compiler Runtime — unified NPU backend interface.
///
/// Cerberus IS the compiler and runtime, not a wrapper around vendor SDKs.
/// Each backend compiles a computational graph to a target-specific binary,
/// then executes it. The pipeline calls compile() at init, execute() at runtime.
///
/// @version 3.1.0
/// @author LamiaFabrica Team

#include "hq/cxx26_features.hpp"
#include "hq/tensor_view.hpp"

#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace hq::npu {

// ===========================================================================
// Tensor descriptor — minimal, no heavy abstractions
// ===========================================================================

struct TensorDesc {
    std::vector<std::int64_t> shape;
    enum class DataType : std::uint8_t { F32 = 0, F16, I64, I32, I8, U8 };
    DataType dtype{DataType::F32};
    std::size_t size_bytes() const noexcept;
};

// ===========================================================================
// KernelGraph — abstract computational graph (placeholder for MLIR/TVM IR)
// ===========================================================================

struct KernelGraph {
    enum class SourceFormat : std::uint8_t { ONNX = 0, TFLite, Relay, MLIR };
    SourceFormat format{SourceFormat::ONNX};
    std::filesystem::path source_path;   ///< Path to .onnx, .tflite, .mlir, etc.
    std::string entry_point;              ///< Function name to compile (e.g., "text_encoder")
};

// ===========================================================================
// TargetConfig — what the compiled kernel needs to run
// ===========================================================================

struct TargetConfig {
    std::string target_name;       ///< "intel_npu", "amd_xdna", "hailo_8l", "coral", "cuda", "cpu"
    std::filesystem::path output_dir{"./compiled_kernels"};
    bool optimize_for_latency{true};  ///< true= latency, false= throughput
    std::size_t max_memory_bytes{0}; ///< 0= no limit
};

// ===========================================================================
// CompiledKernel — opaque handle to a compiled binary blob
// ===========================================================================

struct CompiledKernel {
    std::string target_name;
    std::filesystem::path binary_path;  ///< Path to .bin, .xclbin, .hef, .ov_ir, etc.
    std::vector<TensorDesc> inputs;
    std::vector<TensorDesc> outputs;
    bool compiled{false};

    /// Opaque handle to the backend-specific compiled object (e.g. ov_compiled_model_t*).
    mutable void* native_handle{nullptr};

    /// Optional cleanup function called when the CompiledKernel is destroyed.
    std::function<void(void*)> cleanup;
};

// ===========================================================================
// INpuBackend — compiler + runtime interface for ALL NPU vendors
// ===========================================================================

class INpuBackend {
public:
    virtual ~INpuBackend() = default;

    /// @brief Compile a computational graph to a target-specific binary.
    [[nodiscard]] virtual std::expected<CompiledKernel, std::string>
    compile(const KernelGraph& graph, const TargetConfig& cfg) = 0;

    /// @brief Execute a compiled kernel on real hardware.
    /// @param inputs  Array of pointers to input buffers (size = kernel.inputs.size()).
    /// @param outputs Array of pointers to output buffers (size = kernel.outputs.size()).
    ///                Each buffer must be sized per the corresponding TensorDesc.
    [[nodiscard]] virtual std::expected<void, std::string>
    execute(const CompiledKernel& kernel,
            std::span<const std::byte*> inputs,
            std::span<std::byte*> outputs) = 0;

    /// @brief True if this backend can compile for the given target name.
    [[nodiscard]] virtual bool can_compile_for(std::string_view target_name) const = 0;

    /// @brief True if the backend is initialized and ready.
    [[nodiscard]] virtual bool is_available() const = 0;

    /// @brief Human-readable name: "Intel-OpenVINO-NPU", "NVIDIA-CUDA", etc.
    [[nodiscard]] virtual std::string name() const = 0;

    /// @brief True when this backend operates without real silicon (CPU fallback).
    [[nodiscard]] virtual bool synthetic_mode() const noexcept = 0;

    /// @brief Diagnostic: why is_available() returned false.
    [[nodiscard]] virtual std::string unavailable_reason() const = 0;

    /// @brief NPU utilization % (-1.0f when not available).
    [[nodiscard]] virtual float utilization() const = 0;

    /// @brief NPU temperature °C (-1.0f when not available).
    [[nodiscard]] virtual float temperature() const = 0;

    // --- Legacy convenience wrappers (deprecated, will be removed) ---
    // These are kept only for compat during the migration. New code should
    // use compile() + execute() directly.

    [[nodiscard]] virtual std::expected<std::vector<float>, std::string>
    text_encode(const std::string& prompt, std::uint32_t max_seq_len = 77) {
        (void)prompt; (void)max_seq_len;
        return std::unexpected{std::string{"text_encode() not migrated to compile+execute yet"}};
    }

    [[nodiscard]] virtual std::expected<void, std::string>
    blend_noise_cfg(std::span<float> noise_out,
                    std::span<const float> noise_uncond,
                    float guidance_scale) noexcept {
        (void)noise_out; (void)noise_uncond; (void)guidance_scale;
        return std::unexpected{std::string{"blend_noise_cfg() not migrated to compile+execute yet"}};
    }
};

// ===========================================================================
// IntelOpenVinoBackend — Intel AI Boost NPU via OpenVINO C API
// ===========================================================================

class IntelOpenVinoBackend final : public INpuBackend {
public:
    IntelOpenVinoBackend();
    ~IntelOpenVinoBackend() override;

    // --- Compiler + Runtime (new contract) ---
    [[nodiscard]] std::expected<CompiledKernel, std::string>
    compile(const KernelGraph& graph, const TargetConfig& cfg) override;

    [[nodiscard]] std::expected<void, std::string>
    execute(const CompiledKernel& kernel,
            std::span<const std::byte*> inputs,
            std::span<std::byte*> outputs) override;

    [[nodiscard]] bool can_compile_for(std::string_view target_name) const override;

    // --- Introspection (unchanged) ---
    [[nodiscard]] bool is_available() const override;
    [[nodiscard]] std::string name() const override;
    [[nodiscard]] bool synthetic_mode() const noexcept override;
    [[nodiscard]] std::string unavailable_reason() const override;
    [[nodiscard]] float utilization() const override;
    [[nodiscard]] float temperature() const override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// ===========================================================================
// CpuFallbackBackend — honest CPU fallback, works on any platform
// ===========================================================================

class CpuFallbackBackend final : public INpuBackend {
public:
    CpuFallbackBackend();

    [[nodiscard]] std::expected<CompiledKernel, std::string>
    compile(const KernelGraph& graph, const TargetConfig& cfg) override;

    [[nodiscard]] std::expected<void, std::string>
    execute(const CompiledKernel& kernel,
            std::span<const std::byte*> inputs,
            std::span<std::byte*> outputs) override;

    [[nodiscard]] bool can_compile_for(std::string_view target_name) const override;
    [[nodiscard]] bool is_available() const override { return true; }
    [[nodiscard]] std::string name() const override { return "ONNX-CPU-Fallback"; }
    [[nodiscard]] bool synthetic_mode() const noexcept override { return true; }
    [[nodiscard]] std::string unavailable_reason() const override { return {}; }
    [[nodiscard]] float utilization() const override { return -1.0f; }
    [[nodiscard]] float temperature() const override { return -1.0f; }
};

// ===========================================================================
// CudaBackend — NVIDIA GPU via CUDA (compile to PTX, execute via cuLaunch)
// ===========================================================================

class CudaBackend final : public INpuBackend {
public:
    CudaBackend();
    ~CudaBackend() override;

    [[nodiscard]] std::expected<CompiledKernel, std::string>
    compile(const KernelGraph& graph, const TargetConfig& cfg) override;

    [[nodiscard]] std::expected<void, std::string>
    execute(const CompiledKernel& kernel,
            std::span<const std::byte*> inputs,
            std::span<std::byte*> outputs) override;

    [[nodiscard]] bool can_compile_for(std::string_view target_name) const override;
    [[nodiscard]] bool is_available() const override;
    [[nodiscard]] std::string name() const override;
    [[nodiscard]] bool synthetic_mode() const noexcept override;
    [[nodiscard]] std::string unavailable_reason() const override;
    [[nodiscard]] float utilization() const override;
    [[nodiscard]] float temperature() const override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// ===========================================================================
// NpuBackendFactory — probes all vendors, returns best per-target
// ===========================================================================

class NpuBackendFactory {
public:
    /// @brief Create all backends and probe hardware. Call once at startup.
    static void initialize();

    /// @brief Get the best backend that can compile for the given target name.
    [[nodiscard]] static INpuBackend* best_for(std::string_view target_name);

    /// @brief Get a specific backend by name (for diagnostics).
    [[nodiscard]] static INpuBackend* by_name(std::string_view name);

    /// @brief List all backends with their availability status.
    static void print_status();
};

} // namespace hq::npu