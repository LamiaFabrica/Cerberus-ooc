#pragma once
/// @file cerberus_glow_engine.hpp
/// @copyright Copyright (c) 2026 D Hargreaves. All rights reserved.
///
/// Cerberus GlowEngine — ported from PsiForceDB Nemadic v3 Glowing Strings.
///
/// Learns hot execution paths through a KernelGraph/CerberusGraph by tracking
/// edge usage as "bonds" with usage-based strengths.  Supports:
///   - reinforcement on successful traversal
///   - periodic decay of unused bonds
///   - catchphrase resolution (human name -> node id)
///   - hot-path query for predictor caching / tier promotion
///
/// Every field is consumed — no empty feature-incomplete variables.
///
/// @version 1.0.0

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <chrono>
#include <optional>
#include <algorithm>

namespace hq::cerberus {

// ============================================================================
// GRAPH EDGE BOND
// ============================================================================

struct GraphEdgeBond {
    std::int32_t from_node{-1};
    std::int32_t to_node{-1};
    float base_strength{0.5f};      // static topology weight
    float learned_weight{0.0f};     // usage-based learned weight
    std::uint32_t traversal_count{0};
    std::chrono::steady_clock::time_point last_access;

    /// Combined strength = base + learned (capped at 1.0)
    [[nodiscard]] float combined_strength() const noexcept {
        float v = base_strength + learned_weight * 0.1f;
        return v < 1.0f ? v : 1.0f;
    }

    /// Attenuate amplitude across hops (inverse-linear fade).
    [[nodiscard]] float attenuate(float amplitude, std::uint32_t hops) const noexcept {
        float s = combined_strength();
        return amplitude * s / static_cast<float>(hops + 1);
    }
};

// ============================================================================
// GLOW PATH — a recorded hot execution trace
// ============================================================================

struct GlowPath {
    std::vector<std::int32_t> nodes;      // ordered node ids
    float total_amplitude{0.0f};           // cumulative strength of the path
    std::chrono::steady_clock::time_point recorded_at;
    std::uint32_t hop_count{0};

    [[nodiscard]] bool is_hot(float threshold = 0.5f) const noexcept {
        return total_amplitude >= threshold;
    }
};

// ============================================================================
// CATCHPHRASE REGISTRY — human name -> node id resolution
// ============================================================================

struct GlowCatchphraseResult {
    bool found{false};
    std::int32_t node_id{-1};
    std::string matched_catchphrase;
    float confidence{0.0f};
};

class GlowCatchphraseRegistry {
public:
    static GlowCatchphraseRegistry& instance() {
        static GlowCatchphraseRegistry reg;
        return reg;
    }

    /// Register a literal catchphrase bound to a node id.
    void register_phrase(std::string_view phrase, std::int32_t node_id);

    /// Resolve a natural language query to a node id.
    [[nodiscard]] GlowCatchphraseResult resolve(std::string_view query) const;

    /// Tokenize for diagnostics.
    [[nodiscard]] std::vector<std::string> tokenize(std::string_view query) const;

    /// List all registered catchphrases.
    [[nodiscard]] std::vector<std::string> list_phrases() const;

    /// Reset all registrations.
    void clear();

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::int32_t> phrases_;

    [[nodiscard]] static std::string normalize(std::string_view s);
    [[nodiscard]] static float calculate_similarity(const std::string& a, const std::string& b);
};

// ============================================================================
// GLOW ENGINE — learns and queries hot paths through graphs
// ============================================================================

struct GlowStats {
    std::uint64_t reinforcements_applied{0};
    std::uint64_t decay_cycles_completed{0};
    std::uint64_t paths_learned{0};
    float average_learned_weight{0.0f};
    std::size_t active_bond_count{0};
};

class GlowEngine {
public:
    explicit GlowEngine(float weight_cap = 2.0f);
    ~GlowEngine() = default;

    // Non-copyable
    GlowEngine(const GlowEngine&) = delete;
    GlowEngine& operator=(const GlowEngine&) = delete;

    /// Record that a graph execution traversed these nodes in order.
    void record_execution(const std::vector<std::int32_t>& node_path);

    /// Reinforce a specific path with a reward (strengthens every edge).
    void reinforce_path(const std::vector<std::int32_t>& path, float reward = 0.3f);

    /// Decay all learned weights by a fixed rate; prune zeroed bonds.
    void decay_all(float decay_rate = 0.01f);

    /// Query hot paths starting from a given node, sorted by amplitude descending.
    [[nodiscard]] std::vector<GlowPath> query_hot_paths(
        std::int32_t start_node,
        float min_strength = 0.5f,
        std::uint32_t max_hops = 12,
        std::size_t max_results = 10) const;

    /// Find next best hop from a node (for greedy hot-path walking).
    [[nodiscard]] std::optional<std::int32_t> best_next_hop(
        std::int32_t from_node,
        float min_strength = 0.1f) const;

    /// Get the bond for a specific edge (if it exists).
    [[nodiscard]] std::optional<GraphEdgeBond> get_bond(std::int32_t from, std::int32_t to) const;

    /// Reset all learned state.
    void reset();

    /// Snapshot statistics.
    [[nodiscard]] GlowStats stats() const;

    /// Set the maximum learned weight per bond (prevents runaway).
    void set_weight_cap(float cap) { weight_cap_ = cap; }

private:
    mutable std::mutex mutex_;
    mutable std::mutex stats_mutex_;

    // Bond storage: outer key = from_node, inner key = to_node -> bond
    std::unordered_map<std::int32_t,
        std::unordered_map<std::int32_t, GraphEdgeBond>> bonds_;

    // Recently recorded paths (capped at 10k)
    std::vector<GlowPath> recent_paths_;

    GlowStats stats_;
    float weight_cap_{2.0f};

    void ensure_bond_(std::int32_t from, std::int32_t to);
    void traverse_hot_(std::int32_t current, float amplitude,
                       std::uint32_t hops, std::uint32_t max_hops,
                       std::vector<GlowPath>& results,
                       std::vector<std::int32_t>& path_so_far,
                       float floor) const;
};

} // namespace hq::cerberus
