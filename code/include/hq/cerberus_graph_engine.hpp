#pragma once
/// @file cerberus_graph_engine.hpp
/// @copyright Copyright (c) 2026 D Hargreaves (AKA Roylepython). LamiaFabrica. All rights reserved.
///
/// Cerberus Custom Graph Engine — proprietary intermediate representation.
///
/// This is the foundation of compiler ownership.  The graph is a lightweight
/// DAG owned entirely by Cerberus.  Nodes carry enough metadata for the
/// decision engine to route subsets to native kernels, vendor compilers, or
/// fused subgraphs.  No external framework owns this structure.
///
/// @version 1.0.0

#include "hq/npu_backend_unified.hpp"

#include <cstddef>
#include <string>
#include <vector>
#include <optional>
#include <span>

namespace hq::cerberus {

// ===========================================================================
// Tensor metadata carried inside the custom graph (shape + element type)
// ===========================================================================

struct GraphTensor {
    std::string              name;
    std::vector<std::int64_t> shape;
    npu::TensorDesc::DataType dtype{npu::TensorDesc::DataType::F32};
    std::size_t              size_bytes() const noexcept;
};

// ===========================================================================
// GraphNode — a single operator inside the Cerberus DAG
//
// Each node knows its op, its input/output tensor names, and carries a
// routing decision populated by the DecisionEngine ("native", "openvino",
// "cuda", "fused", etc.).
// ===========================================================================

struct GraphNode {
    std::int32_t             id{-1};               ///< unique within graph
    std::string              name;
    npu::KernelNode::Op      op{npu::KernelNode::Op::Unknown};
    std::vector<std::string> inputs;              ///< tensor names consumed
    std::vector<std::string> outputs;             ///< tensor names produced

    // --- routing state (filled by decision engine) ---
    std::string              execution_backend{"auto"};  ///< "native", "openvino", "cuda", "fused"
    bool                     fused{false};              ///< part of a fused subgraph?
    std::int32_t             fused_group_id{-1};       ///< group id if fused

    // --- optional inline tensor data for constants ---
    std::vector<float>       constant_data;
};

// ===========================================================================
// CerberusGraph — the custom DAG
//
// Owned 100% by Cerberus.  The frontend populates it from ONNX (or later
// from our own textual IR).  The compile path builds it, the decision
// engine rewrites routing fields, and the coordinator walks it.
// ===========================================================================

class CerberusGraph {
public:
    std::vector<GraphNode>    nodes;
    std::vector<GraphTensor> tensors;   ///< all live tensors in topological order

    // --- graph queries (used by decision engine) ---
    [[nodiscard]] std::optional<std::size_t> tensor_index(const std::string& name) const noexcept;
    [[nodiscard]] std::optional<std::size_t> node_index(std::int32_t id) const noexcept;

    /// Build a node → consumer edge map.  Returns vector of consumer node ids.
    [[nodiscard]] std::vector<std::int32_t> consumers(std::int32_t node_id) const;

    /// Topological sort (Kahn).  Reorders `nodes` in-place.
    [[nodiscard]] bool topo_sort();

    /// Compute memory footprint of all live tensors.
    [[nodiscard]] std::size_t total_live_bytes() const noexcept;

    /// Convert a KernelGraph into a CerberusGraph (today: minimal, from nodes).
    [[nodiscard]] static CerberusGraph from_kernel_graph(const npu::KernelGraph& kg);
};

} // namespace hq::cerberus
