/// @file cerberus_graph_engine.cpp
/// @copyright Copyright © 2026, all rights reserved | David Hargreaves aka Roylepython trading as The Medusa Initiative & Lamia Fabrica
/// Support: https://www.patreon.com/TheMedusaInitiative — £25/month removes ads from all software at 200 subscribers
///
/// CerberusGraph engine — graph queries, topological sort, live-memory analysis.
///
/// @version 1.0.0

#include "hq/cerberus_graph_engine.hpp"

#include <algorithm>
#include <queue>
#include <unordered_map>

namespace hq::cerberus {

std::size_t GraphTensor::size_bytes() const noexcept {
    std::size_t elems = 1;
    for (auto d : shape) elems *= static_cast<std::size_t>(d);
    std::size_t dtype_sz = 4; // F32 default
    switch (dtype) {
        case npu::TensorDesc::DataType::F32: dtype_sz = 4; break;
        case npu::TensorDesc::DataType::F16: dtype_sz = 2; break;
        case npu::TensorDesc::DataType::I64: dtype_sz = 8; break;
        case npu::TensorDesc::DataType::I32: dtype_sz = 4; break;
        case npu::TensorDesc::DataType::I8:
        case npu::TensorDesc::DataType::U8:  dtype_sz = 1; break;
        // Ground-up: honest packed sizes for real GGUF block quants (matches TensorDesc + probe driver logic).
        // 4-bit per weight → ~elems/2 bytes (conservative; actual GGUF IQ4_NL is 16 bytes per 32 weights + scales).
        case npu::TensorDesc::DataType::IQ4_NL_Block:
        case npu::TensorDesc::DataType::Q4_K_Block:
            dtype_sz = 1; // treat as packed u8 stream; caller (driver / future lowering) uses exact compressed bytes
            return (elems + 1) / 2; // packed 4-bit approximation (prevents F32 over-allocation in TMM)
    }
    return elems * dtype_sz;
}

// ===========================================================================
// CerberusGraph implementation
// ===========================================================================

std::optional<std::size_t> CerberusGraph::tensor_index(const std::string& name) const noexcept {
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        if (tensors[i].name == name) return i;
    }
    return std::nullopt;
}

std::optional<std::size_t> CerberusGraph::node_index(std::int32_t id) const noexcept {
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        if (nodes[i].id == id) return i;
    }
    return std::nullopt;
}

std::vector<std::int32_t> CerberusGraph::consumers(std::int32_t node_id) const {
    std::vector<std::int32_t> result;
    auto self_opt = node_index(node_id);
    if (!self_opt) return result;
    const auto& self = nodes[*self_opt];
    if (self.outputs.empty()) return result;
    const std::string& out_name = self.outputs[0];

    for (const auto& node : nodes) {
        if (node.id == node_id) continue;
        for (const auto& in : node.inputs) {
            if (in == out_name) {
                result.push_back(node.id);
                break;
            }
        }
    }
    return result;
}

bool CerberusGraph::topo_sort() {
    std::unordered_map<std::int32_t, std::size_t> in_degree;
    for (const auto& n : nodes) in_degree[n.id] = 0;
    for (const auto& n : nodes) {
        if (n.outputs.empty()) continue;
        const std::string& out = n.outputs[0];
        for (const auto& later : nodes) {
            if (later.id == n.id) {
                // Self-loop check: node consumes its own output
                for (const auto& in : later.inputs) {
                    if (in == out) { in_degree[later.id]++; break; }
                }
                continue;
            }
            for (const auto& in : later.inputs) {
                if (in == out) { in_degree[later.id]++; break; }
            }
        }
    }

    std::queue<std::int32_t> q;
    for (const auto& n : nodes) {
        if (in_degree[n.id] == 0) q.push(n.id);
    }

    std::vector<GraphNode> sorted;
    sorted.reserve(nodes.size());
    while (!q.empty()) {
        std::int32_t cur_id = q.front(); q.pop();
        auto idx_opt = node_index(cur_id);
        if (!idx_opt) return false;
        sorted.push_back(nodes[*idx_opt]);

        for (std::int32_t cid : consumers(cur_id)) {
            if (--in_degree[cid] == 0) q.push(cid);
        }
    }

    if (sorted.size() != nodes.size()) return false; // cycle detected
    nodes = std::move(sorted);
    return true;
}

std::size_t CerberusGraph::total_live_bytes() const noexcept {
    std::size_t total = 0;
    for (const auto& t : tensors) total += t.size_bytes();
    return total;
}

// ===========================================================================
// Conversion from KernelGraph
// ===========================================================================

CerberusGraph CerberusGraph::from_kernel_graph(const npu::KernelGraph& kg) {
    CerberusGraph g;
    g.nodes.reserve(kg.nodes.size());

    int32_t next_id = 0;
    for (const auto& kn : kg.nodes) {
        GraphNode gn;
        gn.id   = next_id++;
        gn.name = kn.name;
        gn.op   = kn.op;
        gn.inputs  = kn.inputs;
        gn.outputs = kn.outputs;
        gn.constant_data = kn.float_attrs;
        gn.quant_profile = kn.quant_profile;   // Ground-up fix: propagate real quant metadata (IQ4_NL_Block PerBlock etc.) from KernelGraph
        g.nodes.push_back(std::move(gn));
    }

    // Build tensor list from node I/O (deduplicate by name)
    std::unordered_map<std::string, GraphTensor> tensor_map;
    for (const auto& kn : kg.nodes) {
        for (const auto& in : kn.inputs) {
            if (tensor_map.find(in) == tensor_map.end()) {
                GraphTensor gt;
                gt.name = in;
                tensor_map[in] = std::move(gt);
            }
        }
        for (const auto& out : kn.outputs) {
            if (tensor_map.find(out) == tensor_map.end()) {
                GraphTensor gt;
                gt.name = out;
                tensor_map[out] = std::move(gt);
            }
        }
    }
    for (auto& [name, gt] : tensor_map) {
        (void)name;
        g.tensors.push_back(std::move(gt));
    }

    // Propagate dtype/shape from graph_inputs to tensors with matching names.
    // If a graph_input has an empty name, it will not match any tensor — this is
    // intentional; callers must provide meaningful names in graph_inputs for
    // propagation to occur.  (See propup_graph_engine_dtype_mismatch for the
    // test that guards this behaviour.)
    for (std::size_t i = 0; i < kg.graph_inputs.size(); ++i) {
        for (auto& t : g.tensors) {
            if (t.name == kg.graph_inputs[i].name) {
                if (!kg.graph_inputs[i].shape.empty())
                    t.shape = kg.graph_inputs[i].shape;
                t.dtype = kg.graph_inputs[i].dtype;
                break;
            }
        }
    }

    // Propagate dtype/shape from graph_outputs to tensors with matching names.
    // This ensures output tensors that may not appear as node inputs (e.g. final
    // graph outputs only produced by a node) carry the correct dtype metadata.
    for (std::size_t i = 0; i < kg.graph_outputs.size(); ++i) {
        for (auto& t : g.tensors) {
            if (t.name == kg.graph_outputs[i].name) {
                if (!kg.graph_outputs[i].shape.empty())
                    t.shape = kg.graph_outputs[i].shape;
                t.dtype = kg.graph_outputs[i].dtype;
                break;
            }
        }
    }

    (void)g.topo_sort();
    return g;
}

std::size_t apply_rewrites(CerberusGraph& g, std::span<const GraphRewriteRule> rules) {
    std::size_t rewrites = 0;
    for (auto& rule : rules) {
        for (std::size_t i = 0; i < g.nodes.size(); ++i) {
            if (rule.match(g, i)) {
                rule.replace(g, i);
                ++rewrites;
            }
        }
    }
    return rewrites;
}

GraphRewriteRule make_fuse_matmul_bias_relu_rule() {
    GraphRewriteRule rule;
    rule.name = "fuse_matmul_bias_relu";
    rule.match = [](const CerberusGraph& g, std::size_t idx) -> bool {
        if (idx + 2 >= g.nodes.size()) return false;
        const auto& n0 = g.nodes[idx];
        const auto& n1 = g.nodes[idx + 1];
        const auto& n2 = g.nodes[idx + 2];
        // Pattern: MatMul -> Add -> ReLU on the same single-value chain
        if (n0.op != npu::KernelNode::Op::MatMul) return false;
        if (n1.op != npu::KernelNode::Op::Add) return false;
        if (n2.op != npu::KernelNode::Op::Relu) return false;
        // n0 single output must be n1's single input
        if (n0.outputs.empty() || n1.inputs.empty()) return false;
        if (n0.outputs[0] != n1.inputs[0]) return false;
        // n1 single output must be n2's single input
        if (n1.outputs.empty() || n2.inputs.empty()) return false;
        if (n1.outputs[0] != n2.inputs[0]) return false;
        return true;
    };
    rule.replace = [](CerberusGraph& g, std::size_t idx) {
        auto& n0 = g.nodes[idx];
        auto& n1 = g.nodes[idx + 1];
        auto& n2 = g.nodes[idx + 2];
        // Replace n0 with fused node, absorb bias from n1, absorb ReLU from n2
        n0.name = n0.name + "_fused_bias_relu";
        n0.op = npu::KernelNode::Op::FusedMatMulBiasRelu;
        n0.outputs = n2.outputs; // final fused output
        // Absorb bias tensor from n1 (second input of Add node)
        if (n1.inputs.size() > 1) {
            n0.inputs.push_back(n1.inputs[1]);
        }
        // Remove n1 and n2
        g.nodes.erase(g.nodes.begin() + idx + 1, g.nodes.begin() + idx + 3);
    };
    return rule;
}

} // namespace hq::cerberus
