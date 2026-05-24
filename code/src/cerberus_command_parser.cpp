/// @file cerberus_command_parser.cpp
/// @copyright Copyright (c) 2026 D Hargreaves. All rights reserved.
///
/// Cerberus Command Parser Implementation — PFQL-style grammar.
///
/// @version 1.0.0

#include "hq/cerberus_command_parser.hpp"

#include <charconv>
#include <sstream>
#include <algorithm>
#include <cctype>

namespace hq::cerberus::cli {

// ===========================================================================
// Ergonomic Shortcuts Registry
// ===========================================================================

namespace {
    struct ShortcutRegistry {
        std::map<std::string, std::string> map{
            {"cbr:compile",          "cerberus://graph:compile::path:model.gguf::backend:native;"},
            {"cbr:run",              "cerberus://graph:execute::backend:native;"},
            {"cbr:status",           "cerberus://system:status;"},
            {"cbr:backends",         "cerberus://system:list-backends;"},
            {"cbr:tier-status",      "cerberus://tier:get-status;"},
            {"cbr:promote",          "cerberus://tier:promote::tensor:output;"},
            {"cbr:demote",           "cerberus://tier:demote::tensor:output;"},
            {"cbr:quantize",         "cerberus://quantize:apply::method:Q4_K_M;"},
            {"cbr:dequantize",       "cerberus://quantize:remove::method:Q4_K_M;"},
            {"cbr:model-load",       "cerberus://model:load::path:model.gguf;"},
            {"cbr:model-info",       "cerberus://model:info;"},
            {"cbr:kernel-list",      "cerberus://system:list-kernels;"},
            {"cbr:benchmark",        "cerberus://system:benchmark::backend:native;"},
            {"cbr:help",             "cerberus://system:help;"},
            {"cbr:version",          "cerberus://system:version;"},
            {"cbr:predictor-stats",  "cerberus://system:predictor-stats;"},
            {"cbr:predictor-clear",  "cerberus://system:predictor-clear;"},
            {"cbr:hash-verify",      "cerberus://system:hash-verify::tensor:input;"},
            {"cbr:shadow-save",      "cerberus://tier:shadow-save::tensor:activations;"},
            {"cbr:shadow-restore",   "cerberus://tier:shadow-restore::tensor:activations;"},
            {"cbr:fusion-enable",    "cerberus://graph:enable-fusion;"},
            {"cbr:fusion-disable",   "cerberus://graph:disable-fusion;"},
        };
    };
    ShortcutRegistry& get_registry() {
        static ShortcutRegistry reg;
        return reg;
    }
} // namespace

namespace ergonomic {
    std::string expand(const std::string& shortcut) {
        auto& reg = get_registry();
        auto it = reg.map.find(shortcut);
        return (it != reg.map.end()) ? it->second : std::string{};
    }
    void register_shortcut(const std::string& shortcut, std::string expanded) {
        get_registry().map[shortcut] = std::move(expanded);
    }
} // namespace ergonomic

// ===========================================================================
// Static Helpers
// ===========================================================================

namespace {
    std::string trim_view(std::string_view sv) {
        std::size_t first = 0;
        while (first < sv.size() && std::isspace(static_cast<unsigned char>(sv[first]))) ++first;
        std::size_t last = sv.size();
        while (last > first && std::isspace(static_cast<unsigned char>(sv[last - 1]))) --last;
        return std::string(sv.substr(first, last - first));
    }

    bool starts_with(std::string_view sv, std::string_view prefix) noexcept {
        return sv.size() >= prefix.size() && sv.substr(0, prefix.size()) == prefix;
    }

    bool is_protocol_prefix(std::string_view sv) noexcept {
        return starts_with(sv, "cerberus://") || starts_with(sv, "cbr://");
    }
} // namespace

// ===========================================================================
// Parser Implementation
// ===========================================================================

std::expected<CerberusCommand, std::string> CerberusCommandParser::parse(const std::string& raw) {
    std::string trimmed = trim_view(raw);
    if (trimmed.empty()) {
        return std::unexpected{"Empty command"};
    }

    // Try ergonomic expansion first
    std::string expanded = ergonomic::expand(trimmed);
    const std::string& target = expanded.empty() ? trimmed : expanded;

    if (is_protocol_prefix(target)) {
        return parse_protocol_(target);
    }

    // Try ergonomic parsing if it wasn't an exact shortcut match
    if (expanded.empty()) {
        auto ergo_result = parse_ergonomic_(trimmed);
        if (ergo_result.has_value()) {
            return ergo_result;
        }
    }

    return std::unexpected{"Unknown command format. Use cerberus:// or cbr:shortcut"};
}

bool CerberusCommandParser::is_valid(const std::string& raw) noexcept {
    auto r = parse(raw);
    return r.has_value();
}

std::expected<CerberusCommand, std::string> CerberusCommandParser::parse_protocol_(const std::string& raw) {
    CerberusCommand cmd;
    std::string_view sv = raw;

    // Extract protocol prefix
    if (starts_with(sv, "cerberus://")) {
        cmd.protocol = "cerberus";
        sv = sv.substr(11);
    } else if (starts_with(sv, "cbr://")) {
        cmd.protocol = "cbr";
        sv = sv.substr(6);
    } else {
        return std::unexpected{"Invalid protocol prefix"};
    }

    // Expect namespace:command
    std::size_t colon_pos = sv.find(':');
    if (colon_pos == std::string_view::npos) {
        return std::unexpected{"Expected namespace:command"};
    }
    cmd.namespace_ = trim_view(sv.substr(0, colon_pos));
    sv = sv.substr(colon_pos + 1);

    // Command may be followed by ::params or ;
    std::size_t sep_pos = sv.find("::");
    std::size_t semi_pos = sv.find(';');

    if (sep_pos == std::string_view::npos && semi_pos == std::string_view::npos) {
        // Just a command with no params
        cmd.command = trim_view(sv);
        return cmd;
    }

    if (semi_pos != std::string_view::npos && (sep_pos == std::string_view::npos || semi_pos < sep_pos)) {
        // command; with no params
        cmd.command = trim_view(sv.substr(0, semi_pos));
        return cmd;
    }

    // Extract command up to first ::
    cmd.command = trim_view(sv.substr(0, sep_pos));
    sv = sv.substr(sep_pos + 2);

    // Parse key:value pairs separated by ::
    while (!sv.empty()) {
        semi_pos = sv.find(';');
        std::size_t next_sep = sv.find("::");

        std::string_view pair;
        if (semi_pos != std::string_view::npos && (next_sep == std::string_view::npos || semi_pos < next_sep)) {
            pair = sv.substr(0, semi_pos);
            sv = sv.substr(semi_pos + 1);
            break; // ; terminates params
        } else if (next_sep != std::string_view::npos) {
            pair = sv.substr(0, next_sep);
            sv = sv.substr(next_sep + 2);
        } else {
            // Look for trailing ;
            if (semi_pos != std::string_view::npos) {
                pair = sv.substr(0, semi_pos);
                sv = sv.substr(semi_pos + 1);
                break;
            }
            pair = sv;
            sv = {};
        }

        std::string pair_str = trim_view(pair);
        if (pair_str.empty()) continue;

        std::size_t eq = pair_str.find(':');
        if (eq == std::string::npos) {
            // Positional param (value without key)
            cmd.positional.push_back(pair_str);
        } else {
            std::string key = trim_view(std::string_view(pair_str.c_str(), eq));
            std::string val = trim_view(std::string_view(pair_str.c_str() + eq + 1, pair_str.size() - eq - 1));
            cmd.params[std::move(key)] = std::move(val);
        }
    }

    // Collect any trailing positional args before ;
    if (!sv.empty()) {
        std::string remaining = trim_view(sv);
        if (!remaining.empty()) {
            // Could be more positional, but typically the ; terminates.
            // For now, just ignore trailing after ;
        }
    }

    if (cmd.namespace_.empty() || cmd.command.empty()) {
        return std::unexpected{"namespace and command are required"};
    }

    return cmd;
}

std::expected<CerberusCommand, std::string> CerberusCommandParser::parse_ergonomic_(const std::string& raw) {
    // Ergonomic form: "compile model.gguf --backend native --quantize Q4_K_M"
    // Also supports: "cbr:compile --path model.gguf --backend native"
    std::istringstream iss(raw);
    std::vector<std::string> tokens;
    std::string tok;
    while (iss >> tok) {
        tokens.push_back(std::move(tok));
    }
    if (tokens.empty()) {
        return std::unexpected{"Empty ergonomic command"};
    }

    std::string first = tokens[0];

    // If the first token starts with "cbr:", expand the base shortcut and apply overrides
    if (first.rfind("cbr:", 0) == 0) {
        std::string base_expanded = ergonomic::expand(first);
        if (!base_expanded.empty()) {
            // Parse the base protocol command
            auto base_cmd = parse_protocol_(base_expanded);
            if (!base_cmd) return base_cmd;
            CerberusCommand cmd = *base_cmd;
            // Override with remaining tokens: --key value
            for (std::size_t i = 1; i < tokens.size(); ++i) {
                if (tokens[i].rfind("--", 0) == 0 && i + 1 < tokens.size()) {
                    std::string key = tokens[i].substr(2);
                    cmd.params[std::move(key)] = tokens[++i];
                } else {
                    cmd.positional.push_back(tokens[i]);
                }
            }
            return cmd;
        }
    }

    // Map first token to namespace:command
    if (first == "compile" || first == "build") {
        CerberusCommand cmd;
        cmd.protocol = "cbr";
        cmd.namespace_ = "graph";
        cmd.command = "compile";
        for (std::size_t i = 1; i < tokens.size(); ++i) {
            if (tokens[i] == "--backend" && i + 1 < tokens.size()) {
                cmd.params["backend"] = tokens[++i];
            } else if (tokens[i] == "--quantize" && i + 1 < tokens.size()) {
                cmd.params["quantize"] = tokens[++i];
            } else if (tokens[i] == "--path" && i + 1 < tokens.size()) {
                cmd.params["path"] = tokens[++i];
            } else {
                cmd.positional.push_back(tokens[i]);
            }
        }
        return cmd;
    }

    if (first == "run" || first == "execute") {
        CerberusCommand cmd;
        cmd.protocol = "cbr";
        cmd.namespace_ = "graph";
        cmd.command = "execute";
        for (std::size_t i = 1; i < tokens.size(); ++i) {
            if (tokens[i] == "--backend" && i + 1 < tokens.size()) {
                cmd.params["backend"] = tokens[++i];
            } else {
                cmd.positional.push_back(tokens[i]);
            }
        }
        return cmd;
    }

    if (first == "status" || first == "info") {
        CerberusCommand cmd;
        cmd.protocol = "cbr";
        cmd.namespace_ = "system";
        cmd.command = (first == "status") ? "status" : "info";
        return cmd;
    }

    if (first == "promote" || first == "demote") {
        CerberusCommand cmd;
        cmd.protocol = "cbr";
        cmd.namespace_ = "tier";
        cmd.command = first;
        for (std::size_t i = 1; i < tokens.size(); ++i) {
            if (tokens[i] == "--tensor" && i + 1 < tokens.size()) {
                cmd.params["tensor"] = tokens[++i];
            } else {
                cmd.positional.push_back(tokens[i]);
            }
        }
        return cmd;
    }

    if (first == "quantize") {
        CerberusCommand cmd;
        cmd.protocol = "cbr";
        cmd.namespace_ = "quantize";
        cmd.command = "apply";
        for (std::size_t i = 1; i < tokens.size(); ++i) {
            if (tokens[i] == "--method" && i + 1 < tokens.size()) {
                cmd.params["method"] = tokens[++i];
            } else {
                cmd.positional.push_back(tokens[i]);
            }
        }
        return cmd;
    }

    if (first == "load" || first == "model-load") {
        CerberusCommand cmd;
        cmd.protocol = "cbr";
        cmd.namespace_ = "model";
        cmd.command = "load";
        for (std::size_t i = 1; i < tokens.size(); ++i) {
            if (tokens[i] == "--path" && i + 1 < tokens.size()) {
                cmd.params["path"] = tokens[++i];
            } else {
                cmd.positional.push_back(tokens[i]);
            }
        }
        return cmd;
    }

    if (first == "benchmark") {
        CerberusCommand cmd;
        cmd.protocol = "cbr";
        cmd.namespace_ = "system";
        cmd.command = "benchmark";
        for (std::size_t i = 1; i < tokens.size(); ++i) {
            if (tokens[i] == "--backend" && i + 1 < tokens.size()) {
                cmd.params["backend"] = tokens[++i];
            } else {
                cmd.positional.push_back(tokens[i]);
            }
        }
        return cmd;
    }

    if (first == "help" || first == "--help" || first == "-h") {
        CerberusCommand cmd;
        cmd.protocol = "cbr";
        cmd.namespace_ = "system";
        cmd.command = "help";
        return cmd;
    }

    if (first == "version" || first == "--version" || first == "-v") {
        CerberusCommand cmd;
        cmd.protocol = "cbr";
        cmd.namespace_ = "system";
        cmd.command = "version";
        return cmd;
    }

    return std::unexpected{"Unknown ergonomic command: " + first};
}

} // namespace hq::cerberus::cli
