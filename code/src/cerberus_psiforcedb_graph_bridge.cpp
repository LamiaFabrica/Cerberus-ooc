/// @file cerberus_psiforcedb_graph_bridge.cpp
/// @copyright Copyright © 2026, all rights reserved | David Hargreaves aka Roylepython trading as The Medusa Initiative & Lamia Fabrica
///
/// ModelTopologyMapper implementation — synthetic bridge between Cerberus
/// and PsiForceDB graph store.  No real file dependencies.
///
/// @version 1.0.0

#include "hq/cerberus_psiforcedb_graph_bridge.hpp"

namespace hq::cerberus::psiforcedb {

// ==========================================================================
// GraphTopology
// ==========================================================================

NodeId GraphTopology::add_node(const std::string& label) {
    NodeId id = next_node_id++;
    GraphNode node;
    node.id = id;
    node.label = label;
    node.properties["label"] = label;
    nodes[id] = std::move(node);
    return id;
}

EdgeId GraphTopology::add_edge(NodeId src, NodeId dst, const std::string& rel_type) {
    EdgeId id = next_edge_id++;
    GraphEdge edge;
    edge.id = id;
    edge.source_node = src;
    edge.target_node = dst;
    edge.relationship_type = rel_type;
    edge.properties["type"] = rel_type;
    edges[id] = std::move(edge);

    if (nodes.find(src) != nodes.end()) nodes[src].outgoing_edges.push_back(id);
    if (nodes.find(dst) != nodes.end()) nodes[dst].incoming_edges.push_back(id);
    return id;
}

bool GraphTopology::has_node(NodeId id) const {
    return nodes.find(id) != nodes.end();
}

bool GraphTopology::has_edge(EdgeId id) const {
    return edges.find(id) != edges.end();
}

std::vector<EdgeId> GraphTopology::get_outgoing_edges(NodeId node_id) const {
    auto it = nodes.find(node_id);
    return (it != nodes.end()) ? it->second.outgoing_edges : std::vector<EdgeId>{};
}

std::vector<EdgeId> GraphTopology::get_incoming_edges(NodeId node_id) const {
    auto it = nodes.find(node_id);
    return (it != nodes.end()) ? it->second.incoming_edges : std::vector<EdgeId>{};
}

std::vector<NodeId> GraphTopology::get_neighbors(NodeId node_id, Direction dir) const {
    std::vector<NodeId> out;
    auto it = nodes.find(node_id);
    if (it == nodes.end()) return out;
    if (dir == Direction::OUTGOING || dir == Direction::BOTH) {
        for (EdgeId eid : it->second.outgoing_edges) {
            auto eit = edges.find(eid);
            if (eit != edges.end()) out.push_back(eit->second.target_node);
        }
    }
    if (dir == Direction::INCOMING || dir == Direction::BOTH) {
        for (EdgeId eid : it->second.incoming_edges) {
            auto eit = edges.find(eid);
            if (eit != edges.end()) out.push_back(eit->second.source_node);
        }
    }
    return out;
}

std::string GraphTopology::unavailable_reason() noexcept {
    return "GraphTopology: Cerberus bridge to PsiForceDB graph store. "
           "Full AdjacencyList, AdjacencyMatrix, CSRFormat, LabelIndex, PropertyIndex "
           "and GraphPersistence require PsiForceDB GraphStore runtime linkage. "
           "This Cerberus bridge provides in-memory topology + PFQL row export only. "
           "Persistent graph storage, versioning, and constraint validation are "
           "delegated to PsiForceDB at runtime. (sentinel mode)";
}

// ==========================================================================
// ModelTopologyMapper
// ==========================================================================

GraphTopology ModelTopologyMapper::map_synthetic_model(const std::string& model_name) {
    GraphTopology topo;

    // Build a synthetic Athenea model topology
    auto input   = topo.add_node("Input");
    auto emb     = topo.add_node("TokenEmbedding");
    auto attn0   = topo.add_node("Attention_0");
    auto ffn0    = topo.add_node("FFN_0");
    auto attn1   = topo.add_node("Attention_1");
    auto ffn1    = topo.add_node("FFN_1");
    auto norm    = topo.add_node("LayerNorm");
    auto output  = topo.add_node("Output");

    topo.nodes[input].properties["model"]   = model_name;
    topo.nodes[output].properties["model"]  = model_name;
    topo.nodes[emb].properties["vocab_size"] = std::to_string(151936);
    topo.nodes[emb].properties["hidden_dim"] = std::to_string(4096);

    topo.add_edge(input,  emb,    "data_flow");
    topo.add_edge(emb,    attn0,  "data_flow");
    topo.add_edge(attn0,  ffn0,   "data_flow");
    topo.add_edge(ffn0,   attn1,  "data_flow");
    topo.add_edge(attn1,   ffn1,   "data_flow");
    topo.add_edge(ffn1,   norm,   "data_flow");
    topo.add_edge(norm,   output, "data_flow");

    return topo;
}

std::map<std::string, std::string> ModelTopologyMapper::to_pfql_row(const GraphNode& node) const {
    std::map<std::string, std::string> row;
    row["node_id"] = std::to_string(node.id);
    row["label"] = node.label;
    row["outgoing_edges"] = std::to_string(node.outgoing_edges.size());
    row["incoming_edges"] = std::to_string(node.incoming_edges.size());
    return row;
}

std::map<std::string, std::string> ModelTopologyMapper::to_pfql_row(const GraphEdge& edge) const {
    std::map<std::string, std::string> row;
    row["edge_id"] = std::to_string(edge.id);
    row["source"] = std::to_string(edge.source_node);
    row["target"] = std::to_string(edge.target_node);
    row["type"] = edge.relationship_type;
    return row;
}

} // namespace hq::cerberus::psiforcedb
