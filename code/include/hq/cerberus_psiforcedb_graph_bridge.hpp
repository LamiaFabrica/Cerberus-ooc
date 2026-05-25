#pragma once
/// @file cerberus_psiforcedb_graph_bridge.hpp
/// @copyright Copyright (c) 2026 D Hargreaves. All rights reserved.
///
/// Bridge between Cerberus inference model topology and PsiForceDB graph store.
///
/// Cerberus does not own a graph database — PsiForceDB does.  This bridge maps
/// Cerberus KernelGraph nodes/edges to PsiForceDB NodeId/EdgeId so that model
/// topology can be stored, queried, and visualised through PsiForceDB's
/// GraphModelExtension.  Property values are strings for portability; the real
/// PsiForceDB PropertyValue variant will be used when compiled against actual
/// PsiForceDB headers.
///
/// @version 1.0.0

#include <cstdint>
#include <chrono>
#include <map>
#include <vector>
#include <string>
#include <variant>

#include <variant>

namespace hq::cerberus::psiforcedb {

// ==========================================================================
// PsiForceDB graph type aliases (layout-compatible)
// ==========================================================================

using NodeId      = std::uint64_t;
using EdgeId      = std::uint64_t;
using LabelId     = std::uint32_t;
using PropertyKeyId = std::uint32_t;
using VersionId   = std::uint64_t;
using TransactionId = std::uint64_t;

constexpr NodeId INVALID_NODE_ID = 0;
constexpr EdgeId INVALID_EDGE_ID = 0;
constexpr LabelId INVALID_LABEL_ID = 0;
constexpr std::size_t MAX_PATH_LENGTH = 10000;
constexpr std::size_t DEFAULT_BATCH_SIZE = 1000;

// ==========================================================================
// Property system (simplified variant — full PsiForceDB PropertyValue is
// delegated until LFSSL compiles on Windows)
// ==========================================================================

enum class PropertyType : std::uint8_t {
    NULL_TYPE = 0,
    BOOLEAN,
    INTEGER,
    FLOAT,
    STRING,
    BINARY,
    DATETIME,
    ARRAY
};

enum class ElementState : std::uint8_t {
    ACTIVE = 0,
    DELETED,
    MODIFIED
};

enum class Direction : std::uint8_t {
    OUTGOING = 0,
    INCOMING,
    BOTH
};

using PropertyValue = std::variant<
    std::monostate,
    bool,
    std::int64_t,
    double,
    std::string,
    std::vector<std::uint8_t>,
    std::chrono::system_clock::time_point
>;

// ==========================================================================
// GraphNode — bridge representation of a model node
// ==========================================================================

struct GraphNode {
    NodeId id{INVALID_NODE_ID};
    std::string label;                         // legacy single-label (kept for compatibility)
    std::vector<std::string> labels;           // full PsiForceDB label vector
    std::map<std::string, std::string> properties; // flat-string properties (sentinel)
    std::vector<EdgeId> outgoing_edges;
    std::vector<EdgeId> incoming_edges;

    // Fields matching PsiForceDB::Graph::Node (sentinel defaults)
    ElementState state{ElementState::ACTIVE};
    VersionId version{0};
    TransactionId created_tx{0};
    TransactionId deleted_tx{0};
};

// ==========================================================================
// GraphEdge — bridge representation of a model edge
// ==========================================================================

struct GraphEdge {
    EdgeId id{INVALID_EDGE_ID};
    NodeId source_node{INVALID_NODE_ID};
    NodeId target_node{INVALID_NODE_ID};
    std::string relationship_type;               // e.g. "data_flow", "control_flow"
    std::map<std::string, std::string> properties;

    // Fields matching PsiForceDB::Graph::Edge (sentinel defaults)
    ElementState state{ElementState::ACTIVE};
    VersionId version{0};
    TransactionId created_tx{0};
    TransactionId deleted_tx{0};
};

// ==========================================================================
// Path
// ==========================================================================

struct Path {
    std::vector<GraphNode> nodes;
    std::vector<GraphEdge> edges;

    [[nodiscard]] std::size_t length() const noexcept { return edges.size(); }
    [[nodiscard]] std::size_t node_count() const noexcept { return nodes.size(); }
    [[nodiscard]] bool is_empty() const noexcept { return nodes.empty(); }
};

// ==========================================================================
// GraphTopology — in-memory model topology for PsiForceDB ingestion
// ==========================================================================

struct GraphTopology {
    std::map<NodeId, GraphNode> nodes;
    std::map<EdgeId, GraphEdge> edges;
    NodeId next_node_id{1};
    EdgeId next_edge_id{1};

    NodeId add_node(const std::string& label);
    EdgeId add_edge(NodeId src, NodeId dst, const std::string& rel_type);
    bool has_node(NodeId id) const;
    bool has_edge(EdgeId id) const;

    [[nodiscard]] std::size_t node_count() const noexcept { return nodes.size(); }
    [[nodiscard]] std::size_t edge_count() const noexcept { return edges.size(); }

    [[nodiscard]] std::vector<EdgeId> get_outgoing_edges(NodeId node_id) const;
    [[nodiscard]] std::vector<EdgeId> get_incoming_edges(NodeId node_id) const;
    [[nodiscard]] std::vector<NodeId> get_neighbors(NodeId node_id, Direction dir = Direction::BOTH) const;

    /// Mandatory sentinel — unavailable_reason() for features not yet bridged
    [[nodiscard]] static std::string unavailable_reason() noexcept;
};

// ==========================================================================
// ModelTopologyMapper — converts Cerberus KernelGraph → PsiForceDB topology
// ==========================================================================

class ModelTopologyMapper {
public:
    /// Map a synthetic Cerberus model to PsiForceDB-compatible graph topology.
    /// No real file parsing — purely in-memory bridge validation.
    GraphTopology map_synthetic_model(const std::string& model_name);

    /// Export topology as PFQL-compatible row map for graph insertion.
    std::map<std::string, std::string> to_pfql_row(const GraphNode& node) const;
    std::map<std::string, std::string> to_pfql_row(const GraphEdge& edge) const;
};

} // namespace hq::cerberus::psiforcedb
