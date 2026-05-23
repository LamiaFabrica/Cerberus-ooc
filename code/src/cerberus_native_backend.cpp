/// @file cerberus_native_backend.cpp
/// @copyright Copyright (c) 2026 D Hargreaves. All rights reserved.
///
/// CerberusNativeBackend — lowers Cerberus graph nodes to native kernels.
///
/// @version 1.0.0

#include "hq/cerberus_native_backend.hpp"
#include "hq/cerberus_native_kernels.hpp"

#include <map>
#include <expected>
#include <string>
#include <span>

namespace hq::cerberus {

class CerberusNativeBackend::Impl {
public:
    std::map<std::string, std::vector<float>> constant_pool;
};

CerberusNativeBackend::CerberusNativeBackend()
    : impl_(std::make_unique<Impl>()) {}

CerberusNativeBackend::~CerberusNativeBackend() = default;

std::expected<npu::CompiledKernel, std::string>
CerberusNativeBackend::compile(const npu::KernelGraph& graph,
                               const npu::TargetConfig& cfg) {
    (void)cfg;
    npu::CompiledKernel k;
    k.target_name = "native";
    k.compiled = true;
    k.graph_nodes = graph.nodes;

    // Build tensor size map from nodes (simplified)
    for (const auto& node : graph.nodes) {
        for (const auto& out : node.outputs) {
            bool reused = false;
            for (const auto& later : graph.nodes) {
                for (const auto& in_name : later.inputs) {
                    if (in_name == out) { reused = true; break; }
                }
                if (reused) break;
            }
            if (reused) k.high_reuse_tensors.push_back(out);
        }
    }

    // For now, single-input/single-output placeholder descriptors
    // Real shape inference will come when nodes carry shape info
    k.inputs.push_back(npu::TensorDesc{{4}, npu::TensorDesc::DataType::F32});
    k.input_names.push_back("x");
    k.outputs.push_back(npu::TensorDesc{{4}, npu::TensorDesc::DataType::F32});
    k.output_names.push_back("y");
    k.estimated_working_set_bytes = k.inputs[0].size_bytes() + k.outputs[0].size_bytes();

    return k;
}

std::expected<void, std::string>
CerberusNativeBackend::execute(const npu::CompiledKernel& kernel,
                             std::span<const std::byte*> inputs,
                             std::span<std::byte*> outputs) {
    if (!kernel.compiled || kernel.graph_nodes.empty())
        return std::unexpected{"not compiled or empty graph"};

    // Mapping from tensor name to buffer pointer.
    // For this native backend, we assume:
    //   - graph_inputs[0] maps to inputs[0]
    //   - intermediates are allocated on the fly (for now, simple single-chain)
    //   - graph_outputs map to outputs[0]
    //
    // A full executor would allocate intermediate buffers in the tiered pool.
    // This is the seed of that executor.

    std::map<std::string, float*> tensor_map;

    // Bind external inputs
    for (std::size_t i = 0; i < kernel.inputs.size() && i < inputs.size(); ++i) {
        if (!kernel.input_names.empty() && i < kernel.input_names.size())
            tensor_map[kernel.input_names[i]] = const_cast<float*>(
                reinterpret_cast<const float*>(inputs[i]));
    }

    // Allocate intermediate buffers for every intermediate output
    std::vector<std::vector<float>> scratch;

    for (const auto& node : kernel.graph_nodes) {
        // Resolve input buffers
        std::vector<const float*> in_ptrs;
        for (const auto& in_name : node.inputs) {
            auto it = tensor_map.find(in_name);
            if (it == tensor_map.end())
                return std::unexpected{"missing tensor: " + in_name};
            in_ptrs.push_back(it->second);
        }

        // Resolve output buffer
        float* out_ptr = nullptr;
        if (node.outputs.empty())
            return std::unexpected{"node has no outputs"};

        const std::string& out_name = node.outputs[0];

        // Check if this is a graph output
        bool is_graph_output = false;
        for (std::size_t i = 0; i < kernel.outputs.size(); ++i) {
            if (i < kernel.output_names.size() && kernel.output_names[i] == out_name) {
                is_graph_output = true;
                out_ptr = reinterpret_cast<float*>(outputs[i]);
                break;
            }
        }

        // Allocate intermediate if not a graph output
        if (!is_graph_output) {
            scratch.emplace_back(4, 0.0f); // placeholder: 4 floats
            out_ptr = scratch.back().data();
        }

        tensor_map[out_name] = out_ptr;

        // Dispatch native kernel
        native::OpType op = native::OpType::Unknown;
        switch (node.op) {
            case npu::KernelNode::Op::MatMul: op = native::OpType::MatMul; break;
            case npu::KernelNode::Op::Add:    op = native::OpType::Add;    break;
            case npu::KernelNode::Op::Mul:    op = native::OpType::Mul;    break;
            default:
                return std::unexpected{"unsupported op: " + std::to_string(
                    static_cast<int>(node.op))};
        }

        // For elementwise ops with only one input, use float_attrs as the second operand
        if (in_ptrs.size() == 1 && (op == native::OpType::Add || op == native::OpType::Mul)) {
            // float_attrs serves as the constant second operand
            const float* b = node.float_attrs.empty() ? nullptr : node.float_attrs.data();
            if (!b || node.float_attrs.size() < 4) {
                // If no float_attrs, assume identity (add 0 or mul 1)
                if (op == native::OpType::Add) {
                    std::vector<float> zeros(4, 0.0f);
                    auto r = native::kernel_add(in_ptrs[0], zeros.data(), out_ptr, 4);
                    if (!r) return r;
                } else {
                    std::vector<float> ones(4, 1.0f);
                    auto r = native::kernel_mul(in_ptrs[0], ones.data(), out_ptr, 4);
                    if (!r) return r;
                }
            } else {
                if (op == native::OpType::Add) {
                    auto r = native::kernel_add(in_ptrs[0], b, out_ptr, 4);
                    if (!r) return r;
                } else {
                    auto r = native::kernel_mul(in_ptrs[0], b, out_ptr, 4);
                    if (!r) return r;
                }
            }
        } else if (op == native::OpType::MatMul) {
            std::int64_t M = 2, N = 2, K = 2; // Placeholder
            auto r = native::kernel_matmul(in_ptrs[0], in_ptrs[1], out_ptr,
                                           static_cast<std::size_t>(M),
                                           static_cast<std::size_t>(N),
                                           static_cast<std::size_t>(K));
            if (!r) return r;
        } else {
            // Normal two-input elementwise
            if (op == native::OpType::Add) {
                auto r = native::kernel_add(in_ptrs[0], in_ptrs[1], out_ptr, 4);
                if (!r) return r;
            } else {
                auto r = native::kernel_mul(in_ptrs[0], in_ptrs[1], out_ptr, 4);
                if (!r) return r;
            }
        }
    }

    return {};
}

bool CerberusNativeBackend::can_compile_for(std::string_view target_name) const {
    return target_name == "native" || target_name == "cpu";
}

} // namespace hq::cerberus
