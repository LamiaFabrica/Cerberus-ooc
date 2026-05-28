/// @file cerberus_glow_engine.cpp
/// @copyright Copyright © 2026, all rights reserved | David Hargreaves aka Roylepython trading as The Medusa Initiative & Lamia Fabrica
/// Support: https://www.patreon.com/TheMedusaInitiative — £25/month removes ads from all software at 200 subscribers
///
/// Cerberus GlowEngine implementation.
///
/// @version 1.0.0

#include "hq/cerberus_glow_engine.hpp"
#include <sstream>
#include <numeric>

namespace hq::cerberus {

// ============================================================================
// GlowCatchphraseRegistry
// ============================================================================

void GlowCatchphraseRegistry::register_phrase(std::string_view phrase, std::int32_t node_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    phrases_[normalize(phrase)] = node_id;
}

GlowCatchphraseResult GlowCatchphraseRegistry::resolve(std::string_view query) const {
    std::lock_guard<std::mutex> lock(mutex_);
    GlowCatchphraseResult result;
    std::string normalized_query = normalize(query);
    if (normalized_query.empty()) return result;

    // Exact match
    auto it = phrases_.find(normalized_query);
    if (it != phrases_.end()) {
        result.found = true;
        result.node_id = it->second;
        result.matched_catchphrase = normalized_query;
        result.confidence = 1.0f;
        return result;
    }

    // Fuzzy match
    float best_confidence = 0.0f;
    std::string best_match;
    std::int32_t best_id = -1;
    for (const auto& [phrase, node_id] : phrases_) {
        float conf = calculate_similarity(normalized_query, phrase);
        if (conf > best_confidence && conf >= 0.5f) {
            best_confidence = conf;
            best_match = phrase;
            best_id = node_id;
        }
    }

    if (!best_match.empty()) {
        result.found = true;
        result.node_id = best_id;
        result.matched_catchphrase = best_match;
        result.confidence = best_confidence;
    }
    return result;
}

std::vector<std::string> GlowCatchphraseRegistry::tokenize(std::string_view query) const {
    std::string s = normalize(query);
    std::vector<std::string> tokens;
    std::istringstream iss(s);
    std::string tok;
    while (iss >> tok) {
        tokens.push_back(tok);
    }
    return tokens;
}

std::vector<std::string> GlowCatchphraseRegistry::list_phrases() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> out;
    out.reserve(phrases_.size());
    for (const auto& kv : phrases_) out.push_back(kv.first);
    return out;
}

void GlowCatchphraseRegistry::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    phrases_.clear();
}

std::string GlowCatchphraseRegistry::normalize(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c >= 'A' && c <= 'Z') out.push_back(c + 32);
        else if (c == '\'' || c == '"' || c == '-' || c == '_') out.push_back(' ');
        else out.push_back(c);
    }
    size_t start = out.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    size_t end = out.find_last_not_of(" \t\n\r");
    return out.substr(start, end - start + 1);
}

float GlowCatchphraseRegistry::calculate_similarity(const std::string& a, const std::string& b) {
    if (a == b) return 1.0f;
    if (a.empty() || b.empty()) return 0.0f;
    if (b.find(a) != std::string::npos || a.find(b) != std::string::npos) {
        float ratio = static_cast<float>(std::min(a.size(), b.size())) /
                      static_cast<float>(std::max(a.size(), b.size()));
        return 0.7f + 0.3f * ratio;
    }
    std::istringstream ssa(a), ssb(b);
    std::vector<std::string> wa, wb;
    std::string w;
    while (ssa >> w) wa.push_back(w);
    while (ssb >> w) wb.push_back(w);
    size_t overlap = 0;
    for (const auto& ca : wa) {
        for (const auto& cb : wb) {
            if (ca == cb) { ++overlap; break; }
        }
    }
    size_t denom = wa.size() + wb.size() - overlap;
    if (denom == 0) return 0.0f;
    float jaccard = static_cast<float>(overlap) / static_cast<float>(denom);
    return jaccard * 0.6f;
}

// ============================================================================
// GlowEngine
// ============================================================================

GlowEngine::GlowEngine(float weight_cap) : weight_cap_(weight_cap) {}

void GlowEngine::ensure_bond_(std::int32_t from, std::int32_t to) {
    auto& inner = bonds_[from];
    if (inner.find(to) == inner.end()) {
        GraphEdgeBond bond;
        bond.from_node = from;
        bond.to_node = to;
        bond.base_strength = 0.5f;
        bond.learned_weight = 0.0f;
        bond.traversal_count = 0;
        bond.last_access = std::chrono::steady_clock::now();
        inner.emplace(to, std::move(bond));
    }
}

void GlowEngine::record_execution(const std::vector<std::int32_t>& node_path) {
    if (node_path.size() < 2) return;
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t i = 0; i + 1 < node_path.size(); ++i) {
        ensure_bond_(node_path[i], node_path[i + 1]);
        auto& bond = bonds_[node_path[i]][node_path[i + 1]];
        ++bond.traversal_count;
        bond.last_access = std::chrono::steady_clock::now();
    }
    // Store recent path
    GlowPath gp;
    gp.nodes = node_path;
    gp.recorded_at = std::chrono::steady_clock::now();
    gp.hop_count = static_cast<std::uint32_t>(node_path.size() - 1);
    // Compute total amplitude as product of combined strengths
    float amp = 1.0f;
    for (size_t i = 0; i + 1 < node_path.size(); ++i) {
        auto it = bonds_.find(node_path[i]);
        if (it != bonds_.end()) {
            auto jt = it->second.find(node_path[i + 1]);
            if (jt != it->second.end()) {
                amp *= jt->second.combined_strength();
            }
        }
    }
    gp.total_amplitude = amp;
    recent_paths_.push_back(std::move(gp));
    if (recent_paths_.size() > 10000) {
        recent_paths_.erase(recent_paths_.begin(), recent_paths_.begin() + 5000);
    }
    {
        std::lock_guard<std::mutex> slock(stats_mutex_);
        ++stats_.paths_learned;
    }
}

void GlowEngine::reinforce_path(const std::vector<std::int32_t>& path, float reward) {
    if (path.size() < 2) return;
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t i = 0; i + 1 < path.size(); ++i) {
        ensure_bond_(path[i], path[i + 1]);
        float& w = bonds_[path[i]][path[i + 1]].learned_weight;
        w = std::min(weight_cap_, w + reward);
    }
    {
        std::lock_guard<std::mutex> slock(stats_mutex_);
        stats_.reinforcements_applied += path.size() - 1;
    }
}

void GlowEngine::decay_all(float decay_rate) {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t decayed_count = 0;
    for (auto from_it = bonds_.begin(); from_it != bonds_.end(); ) {
        auto& targets = from_it->second;
        for (auto it = targets.begin(); it != targets.end(); ) {
            float old = it->second.learned_weight;
            it->second.learned_weight = std::max(0.0f, it->second.learned_weight - decay_rate);
            if (old != it->second.learned_weight) ++decayed_count;
            if (it->second.learned_weight <= 0.0f && it->second.traversal_count == 0) {
                it = targets.erase(it);
            } else {
                ++it;
            }
        }
        if (targets.empty()) {
            from_it = bonds_.erase(from_it);
        } else {
            ++from_it;
        }
    }
    {
        std::lock_guard<std::mutex> slock(stats_mutex_);
        ++stats_.decay_cycles_completed;
    }
}

void GlowEngine::traverse_hot_(
    std::int32_t current, float amplitude,
    std::uint32_t hops, std::uint32_t max_hops,
    std::vector<GlowPath>& results,
    std::vector<std::int32_t>& path_so_far,
    float floor) const {
    if (amplitude < floor || hops > max_hops) return;
    path_so_far.push_back(current);

    // Record if we have traversed at least one edge
    if (path_so_far.size() >= 2) {
        GlowPath gp;
        gp.nodes = path_so_far;
        gp.total_amplitude = amplitude;
        gp.hop_count = hops;
        gp.recorded_at = std::chrono::steady_clock::now();
        results.push_back(std::move(gp));
    }

    auto from_it = bonds_.find(current);
    if (from_it == bonds_.end()) { path_so_far.pop_back(); return; }

    // Collect best outgoing bond per target
    std::vector<GraphEdgeBond> outgoing;
    for (const auto& kv : from_it->second) outgoing.push_back(kv.second);
    std::sort(outgoing.begin(), outgoing.end(),
              [](const GraphEdgeBond& a, const GraphEdgeBond& b) {
                  return a.combined_strength() > b.combined_strength();
              });

    for (const auto& bond : outgoing) {
        float next_amp = bond.attenuate(amplitude, hops);
        traverse_hot_(bond.to_node, next_amp, hops + 1, max_hops,
                      results, path_so_far, floor);
    }
    path_so_far.pop_back();
}

std::vector<GlowPath> GlowEngine::query_hot_paths(
    std::int32_t start_node,
    float min_strength,
    std::uint32_t max_hops,
    std::size_t max_results) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<GlowPath> results;
    std::vector<std::int32_t> path;
    traverse_hot_(start_node, 1.0f, 0, max_hops, results, path, min_strength);
    std::sort(results.begin(), results.end(),
              [](const GlowPath& a, const GlowPath& b) {
                  return a.total_amplitude > b.total_amplitude;
              });
    if (results.size() > max_results) results.resize(max_results);
    return results;
}

std::optional<std::int32_t> GlowEngine::best_next_hop(
    std::int32_t from_node, float min_strength) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto from_it = bonds_.find(from_node);
    if (from_it == bonds_.end()) return std::nullopt;

    std::optional<GraphEdgeBond> best;
    for (const auto& kv : from_it->second) {
        if (kv.second.combined_strength() < min_strength) continue;
        if (!best || kv.second.combined_strength() > best->combined_strength()) {
            best = kv.second;
        }
    }
    if (best) return best->to_node;
    return std::nullopt;
}

std::optional<GraphEdgeBond> GlowEngine::get_bond(std::int32_t from, std::int32_t to) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto from_it = bonds_.find(from);
    if (from_it == bonds_.end()) return std::nullopt;
    auto it = from_it->second.find(to);
    if (it == from_it->second.end()) return std::nullopt;
    return it->second;
}

void GlowEngine::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    bonds_.clear();
    recent_paths_.clear();
    {
        std::lock_guard<std::mutex> slock(stats_mutex_);
        stats_ = GlowStats{};
    }
}

GlowStats GlowEngine::stats() const {
    std::lock_guard<std::mutex> slock(stats_mutex_);
    GlowStats s = stats_;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        s.active_bond_count = 0;
        float total_weight = 0.0f;
        size_t count = 0;
        for (const auto& [from, targets] : bonds_) {
            for (const auto& [to, bond] : targets) {
                ++s.active_bond_count;
                total_weight += bond.learned_weight;
                ++count;
            }
        }
        if (count > 0) s.average_learned_weight = total_weight / static_cast<float>(count);
    }
    return s;
}

} // namespace hq::cerberus
