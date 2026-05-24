#pragma once
/// @file cerberus_predictor.hpp
/// @copyright Copyright (c) 2026 D Hargreaves. All rights reserved.
///
/// ExecutionPredictor — lightweight pattern matcher between DecisionEngine
/// and ExecutionCoordinator. Caches (signature → plan) mappings in a bounded
/// LRU table to avoid recomputing full graph analysis for recurring patterns.
///
/// @version 1.0.0

#include "hq/cerberus_graph_engine.hpp"
#include "hq/tiered_memory_manager.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <span>
#include <vector>
#include <functional>

namespace hq::cerberus {

// ===========================================================================
// GraphSignature — lightweight fingerprint for plan caching
// ===========================================================================

struct GraphSignature {
    std::size_t num_nodes{0};
    std::size_t num_matmuls{0};
    std::size_t total_live_bytes{0};

    [[nodiscard]] bool operator==(const GraphSignature& o) const noexcept {
        return num_nodes == o.num_nodes &&
               num_matmuls == o.num_matmuls &&
               total_live_bytes == o.total_live_bytes;
    }
};

// ===========================================================================
// ExecutionSnapshot — the cached result of DecisionEngine::analyse()
// ===========================================================================

struct ExecutionSnapshot {
    bool fused{false};
    bool quantized{false};
    std::string backend_hint{"native"};
    MemoryTier preferred_tier{MemoryTier::Warm};
    float predicted_ms{0.0f};
};

// ===========================================================================
// ExecutionPredictor — bounded LRU cache
// ===========================================================================

class ExecutionPredictor {
public:
    explicit ExecutionPredictor(std::size_t capacity = 64);

    /// Compute a lightweight signature from the graph.
    [[nodiscard]] GraphSignature signature(const CerberusGraph& g) const noexcept;

    /// Lookup cached snapshot.  Returns nullptr on miss.
    [[nodiscard]] const ExecutionSnapshot* lookup(const GraphSignature& sig) const noexcept;

    /// Insert or update snapshot for signature.
    void update(const GraphSignature& sig, const ExecutionSnapshot& snap);

    /// Hit-rate telemetry.
    [[nodiscard]] double hit_rate() const noexcept;
    [[nodiscard]] std::size_t hits() const noexcept { return hits_; }
    [[nodiscard]] std::size_t misses() const noexcept { return misses_; }

private:
    struct Entry {
        GraphSignature sig;
        ExecutionSnapshot snap;
        std::uint64_t tick{0};
    };
    std::vector<Entry> table_;
    std::size_t capacity_{64};
    mutable std::size_t hits_{0};
    mutable std::size_t misses_{0};
    std::uint64_t clock_{0};
};

} // namespace hq::cerberus

// ===========================================================================
// std::hash specialization for GraphSignature
// ===========================================================================

template<>
struct std::hash<hq::cerberus::GraphSignature> {
    [[nodiscard]] std::size_t operator()(const hq::cerberus::GraphSignature& s) const noexcept {
        std::size_t h = s.num_nodes;
        h ^= s.num_matmuls + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= s.total_live_bytes + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};
