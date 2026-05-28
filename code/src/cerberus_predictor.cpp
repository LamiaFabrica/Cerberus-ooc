/// @file cerberus_predictor.cpp
/// @copyright Copyright © 2026, all rights reserved | David Hargreaves aka Roylepython trading as The Medusa Initiative & Lamia Fabrica
///
/// ExecutionPredictor implementation — bounded LRU cache.
///
/// @version 1.0.0

#include "hq/cerberus_predictor.hpp"

namespace hq::cerberus {

ExecutionPredictor::ExecutionPredictor(std::size_t capacity)
    : capacity_{capacity} {
    table_.reserve(capacity);
}

GraphSignature ExecutionPredictor::signature(const CerberusGraph& g) const noexcept {
    GraphSignature s;
    s.num_nodes = g.nodes.size();
    s.total_live_bytes = g.total_live_bytes();
    for (const auto& n : g.nodes) {
        if (n.op == npu::KernelNode::Op::MatMul) ++s.num_matmuls;
    }
    return s;
}

const ExecutionSnapshot* ExecutionPredictor::lookup(const GraphSignature& sig) const noexcept {
    for (const auto& e : table_) {
        if (e.sig == sig) {
            ++hits_;
            return &e.snap;
        }
    }
    ++misses_;
    return nullptr;
}

void ExecutionPredictor::update(const GraphSignature& sig, const ExecutionSnapshot& snap) {
    // Update existing entry
    for (auto& e : table_) {
        if (e.sig == sig) {
            e.snap = snap;
            e.tick = ++clock_;
            return;
        }
    }
    // Insert new entry (evict LRU if at capacity)
    if (table_.size() >= capacity_) {
        // Evict entry with smallest tick
        std::size_t lru_idx = 0;
        for (std::size_t i = 1; i < table_.size(); ++i) {
            if (table_[i].tick < table_[lru_idx].tick)
                lru_idx = i;
        }
        table_[lru_idx] = {sig, snap, ++clock_};
    } else {
        table_.push_back({sig, snap, ++clock_});
    }
}

double ExecutionPredictor::hit_rate() const noexcept {
    std::size_t total = hits_ + misses_;
    if (total == 0) return 0.0;
    return static_cast<double>(hits_) / static_cast<double>(total);
}

} // namespace hq::cerberus
