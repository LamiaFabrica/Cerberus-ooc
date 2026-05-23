/// @file cerberus_graph_engine.cpp
/// @copyright Copyright (c) 2026 D Hargreaves. All rights reserved.
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
            if (later.id == n.id) continue;
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

    // Simplified shape population: map by position in graph_inputs / graph_outputs
    // (TensorDesc has no name field, so we map by tensor index)
    for (std::size_t i = 0; i < kg.graph_inputs.size() && i < g.tensors.size(); ++i) {
        if (!kg.graph_inputs[i].shape.empty())
            g.tensors[i].shape = kg.graph_inputs[i].shape;
    }

    (void)g.topo_sort();
    return g;
}

} // namespace hq::cerberus
