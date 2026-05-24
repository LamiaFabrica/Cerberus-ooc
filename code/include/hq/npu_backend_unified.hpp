#pragma once
/// @file npu_backend_unified.hpp
/// @copyright Copyright (c) 2026 D Hargreaves (AKA Roylepython). LamiaFabrica. All rights reserved.
/// Cerberus Compiler Runtime — unified NPU backend interface.
///
/// Cerberus IS the compiler and runtime, not a wrapper around vendor SDKs.
/// Each backend compiles a computational graph to a target-specific binary,
/// then executes it. The pipeline calls compile() at init, execute() at runtime.
///
/// @version 3.2.0
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
// Quantization metadata
// ===========================================================================

enum class QuantMethod : std::uint8_t {
    None,   ///< No quantization (FP32)
    PTQ,    ///< Post-training quantization
    QAT,    ///< Quantization-aware training
    LSQ,    ///< Learned Step Size
};

enum class QuantGranularity : std::uint8_t {
    PerTensor,   ///< Single scale for entire tensor
    PerChannel,  ///< Per-output-channel (weights)
    PerToken,    ///< Per-row (activations in LLMs)
};

struct QuantProfile {
    QuantMethod      method{QuantMethod::None};
    std::uint8_t     activation_bits{8};    ///< e.g. 8 for INT8, 4 for INT4
    std::uint8_t     weight_bits{8};
    QuantGranularity act_granularity{QuantGranularity::PerTensor};
    QuantGranularity weight_granularity{QuantGranularity::PerChannel};
};

// ===========================================================================
// KernelNode — a single operation in the Cerberus-owned graph
//
// This is the first step toward Cerberus owning the IR.  Each node is a
// lightweight description of an operator and its connections.  In the future
// this will become the basis for our own lowering, but today it exists so
// that `compile()` can inspect and optionally transform the graph before
// invoking any vendor compiler.
// ===========================================================================

struct KernelNode {
    enum class Op : std::uint8_t {
        Unknown = 0,
        MatMul,
        Add,
        Mul,
        Conv,
        Relu,
        Sigmoid,
        Reshape,
        Transpose,
        Softmax,
        LayerNorm,
        Gelu,
        Constant,
        FusedMatMulBiasRelu,
        // --- extensible ---
    };
    Op op{Op::Unknown};
    std::string name;                        ///< node name (from ONNX or generated)
    std::vector<std::string> inputs;         ///< names of input tensor(s)
    std::vector<std::string> outputs;        ///< names of output tensor(s)
    // --- attributes ---
    std::vector<float>             float_attrs;
    std::vector<std::int64_t>      int_attrs;
    std::vector<std::vector<std::int64_t>> shape_attrs;

    // --- per-node quantization profile (drives decision engine routing) ---
    QuantProfile quant_profile{};
};

// ===========================================================================
// KernelGraph — Cerberus-owned computational graph
//
// The graph can be loaded from an external file (ONNX, TFLite, etc.) but
// Cerberus owns the in-memory representation.  The backend's `compile()`
// receives this structure and decides how to lower it.  Today we still
// delegate to vendor compilers for binary generation, but the graph itself
// is no longer just a file path passed blindly to a library.
// ===========================================================================

struct KernelGraph {
    enum class SourceFormat : std::uint8_t { ONNX = 0, TFLite, Relay, MLIR };
    SourceFormat format{SourceFormat::ONNX};
    std::filesystem::path source_path;   ///< Path to .onnx, .tflite, .mlir, etc.
    std::string entry_point;              ///< Function / model name to compile

    /// --- Cerberus-owned graph data (initially populated by frontend loader) ---
    std::vector<KernelNode> nodes;        ///< topological list of ops
    std::vector<TensorDesc>  graph_inputs; ///< inputs required by the entry point
    std::vector<TensorDesc>  graph_outputs;///< outputs produced by the entry point

    /// Opaque handle to the *frontend* loaded model (e.g. ov_model_t*).
    /// This is the Cerberus-owned intermediate representation before lowering.
    /// Backends may use it to drive their own compilation or to hand off to a
    /// vendor compiler.
    void* frontend_handle{nullptr};
    std::function<void(void*)> frontend_cleanup;

    KernelGraph() = default;
    ~KernelGraph() {
        if (frontend_handle && frontend_cleanup) {
            frontend_cleanup(frontend_handle);
            frontend_handle = nullptr;
        }
    }
    // move-only
    KernelGraph(const KernelGraph&) = delete;
    KernelGraph& operator=(const KernelGraph&) = delete;
    KernelGraph(KernelGraph&& o) noexcept
        : format(o.format)
        , source_path(std::move(o.source_path))
        , entry_point(std::move(o.entry_point))
        , nodes(std::move(o.nodes))
        , graph_inputs(std::move(o.graph_inputs))
        , graph_outputs(std::move(o.graph_outputs))
        , frontend_handle(o.frontend_handle)
        , frontend_cleanup(std::move(o.frontend_cleanup))
    {
        o.frontend_handle = nullptr;
    }
    KernelGraph& operator=(KernelGraph&& o) noexcept {
        if (this != &o) {
            if (frontend_handle && frontend_cleanup) frontend_cleanup(frontend_handle);
            format        = o.format;
            source_path   = std::move(o.source_path);
            entry_point   = std::move(o.entry_point);
            nodes         = std::move(o.nodes);
            graph_inputs  = std::move(o.graph_inputs);
            graph_outputs = std::move(o.graph_outputs);
            frontend_handle = o.frontend_handle;
            frontend_cleanup = std::move(o.frontend_cleanup);
            o.frontend_handle = nullptr;
        }
        return *this;
    }
};

// ===========================================================================
// TargetConfig — what the compiled kernel needs to run
// ===========================================================================

struct TargetConfig {
    std::string target_name;       ///< "intel_npu", "amd_xdna", "hailo_8l", "coral", "cuda", "cpu"
    std::filesystem::path output_dir{"./compiled_kernels"};
    bool optimize_for_latency{true};  ///< true= latency, false= throughput
    std::size_t max_memory_bytes{0}; ///< 0= no limit
    bool keep_intermediates{false};  ///< if true, Cerberus keeps frontend IR after compile()
};

// ===========================================================================
// CompiledKernel — opaque handle to a compiled binary blob
// ===========================================================================

struct CompiledKernel {
    std::string target_name;
    std::filesystem::path binary_path;  ///< Path to .bin, .xclbin, .hef, .ov_ir, etc.
    std::vector<TensorDesc> inputs;
    std::vector<TensorDesc> outputs;

    std::vector<std::string> input_names;
    std::vector<std::string> output_names;

    /// Names of tensors likely reused across steps (e.g. activations passed between layers).
    std::vector<std::string> high_reuse_tensors;
    /// Sum of all input+output sizes in bytes.
    std::size_t estimated_working_set_bytes{0};

    /// Lightweight snapshot of the Cerberus-owned graph nodes used for lowering.
    std::vector<KernelNode> graph_nodes;

    bool compiled{false};

    /// Opaque handle to the backend-specific compiled object (e.g. ov_compiled_model_t*).
    void* native_handle{nullptr};

    /// Optional cleanup function called when the CompiledKernel is destroyed.
    std::function<void(void*)> cleanup;

    CompiledKernel() = default;
    ~CompiledKernel() { release(); }

    CompiledKernel(const CompiledKernel&) = delete;
    CompiledKernel& operator=(const CompiledKernel&) = delete;

    CompiledKernel(CompiledKernel&& o) noexcept
        : target_name(std::move(o.target_name))
        , binary_path(std::move(o.binary_path))
        , inputs(std::move(o.inputs))
        , outputs(std::move(o.outputs))
        , input_names(std::move(o.input_names))
        , output_names(std::move(o.output_names))
        , high_reuse_tensors(std::move(o.high_reuse_tensors))
        , estimated_working_set_bytes(o.estimated_working_set_bytes)
        , graph_nodes(std::move(o.graph_nodes))
        , compiled(o.compiled)
        , native_handle(o.native_handle)
        , cleanup(std::move(o.cleanup))
    {
        o.compiled = false;
        o.native_handle = nullptr;
        o.estimated_working_set_bytes = 0;
    }
    CompiledKernel& operator=(CompiledKernel&& o) noexcept {
        if (this != &o) {
            release();
            target_name = std::move(o.target_name);
            binary_path = std::move(o.binary_path);
            inputs = std::move(o.inputs);
            outputs = std::move(o.outputs);
            input_names = std::move(o.input_names);
            output_names = std::move(o.output_names);
            high_reuse_tensors = std::move(o.high_reuse_tensors);
            estimated_working_set_bytes = o.estimated_working_set_bytes;
            graph_nodes = std::move(o.graph_nodes);
            compiled = o.compiled;
            native_handle = o.native_handle;
            cleanup = std::move(o.cleanup);
            o.compiled = false;
            o.native_handle = nullptr;
            o.estimated_working_set_bytes = 0;
        }
        return *this;
    }

    void release() noexcept {
        if (native_handle && cleanup) {
            cleanup(native_handle);
            native_handle = nullptr;
        }
        compiled = false;
    }
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
    ///
    /// This method is the ONLY valid way to run a backend. The caller is
    /// always CerberusExecutionCoordinator, which owns the memory loop.
    /// No other code may call execute() directly.
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
};

// ===========================================================================
// IntelOpenVinoBackend — Intel AI Boost NPU via OpenVINO C API
// ===========================================================================

class CerberusExecutionCoordinator;  // forward

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
