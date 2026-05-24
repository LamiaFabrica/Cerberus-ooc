#pragma once
/// @file cerberus_command_parser.hpp
/// @copyright Copyright (c) 2026 D Hargreaves. All rights reserved.
///
/// Cerberus Command Parser — PFQL-style grammar native to Cerberus.
///
/// Grammar:
///   cerberus://namespace:command::param1:value1::param2:value2;
///   cbr://namespace:command::param1:value1;
///   ergonomic: cbr:compile model.gguf --backend native --quantize Q4_K_M
///
/// @version 1.0.0

#include <string>
#include <vector>
#include <map>
#include <expected>
#include <cstdint>

namespace hq::cerberus::cli {

// ===========================================================================
// Command Structure
// ===========================================================================

struct CerberusCommand {
    std::string protocol;      ///< "cerberus" or "cbr"
    std::string namespace_;    ///< "graph", "backend", "tier", "model", "system", "quantize"
    std::string command;       ///< "compile", "execute", "set", "get", "list", "promote", "demote"
    std::map<std::string, std::string> params; ///< key-value parameter pairs
    std::vector<std::string> positional;     ///< positional arguments after params

    bool has_param(const std::string& key) const noexcept {
        return params.find(key) != params.end();
    }
    std::string get_param(const std::string& key, const std::string& fallback = {}) const noexcept {
        auto it = params.find(key);
        return (it != params.end()) ? it->second : fallback;
    }
};

// ===========================================================================
// Parser
// ===========================================================================

class CerberusCommandParser {
public:
    /// Parse a raw command string into a structured command.
    /// Supports both protocol URLs and ergonomic short-form.
    [[nodiscard]] static std::expected<CerberusCommand, std::string> parse(const std::string& raw);

    /// Validate that a raw string is a valid Cerberus command.
    [[nodiscard]] static bool is_valid(const std::string& raw) noexcept;

private:
    [[nodiscard]] static std::expected<CerberusCommand, std::string> parse_protocol_(const std::string& raw);
    [[nodiscard]] static std::expected<CerberusCommand, std::string> parse_ergonomic_(const std::string& raw);
};

// ===========================================================================
// Ergonomic Shortcuts — shorthand mappings for common operations
// ===========================================================================

namespace ergonomic {
    /// Expand an ergonomic shortcut into full protocol form.
    /// Returns empty string if no shortcut matches.
    [[nodiscard]] std::string expand(const std::string& shortcut);

    /// Register a custom shortcut.
    void register_shortcut(const std::string& shortcut, std::string expanded);
} // namespace ergonomic

} // namespace hq::cerberus::cli
