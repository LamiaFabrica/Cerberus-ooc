#pragma once
/// @file cerberus_command_executor.hpp
/// @copyright Copyright (c) 2026 D Hargreaves. All rights reserved.
///
/// Cerberus Command Executor — routes parsed commands to Cerberus runtime ops.
///
/// Every command is executed against a live CerberusRuntime instance.
/// Results are returned as structured JSON-like strings for CLI display.
///
/// @version 1.0.0

#include "hq/cerberus_command_parser.hpp"
#include "hq/cerberus_runtime.hpp"

#include <string>
#include <expected>
#include <functional>
#include <map>

namespace hq::cerberus::cli {

// ===========================================================================
// Execution Result
// ===========================================================================

struct CommandResult {
    bool success{false};
    int exit_code{0};
    std::string output;   ///< Human-readable or JSON output
    std::string error;      ///< Error message if !success
    double execution_time_ms{0.0};

    static CommandResult ok(std::string output_text) {
        CommandResult r;
        r.success = true;
        r.output = std::move(output_text);
        return r;
    }
    static CommandResult fail(std::string error_text, int code = 1) {
        CommandResult r;
        r.success = false;
        r.error = std::move(error_text);
        r.exit_code = code;
        return r;
    }
};

// ===========================================================================
// Command Handler Signature
// ===========================================================================

using CommandHandler = std::function<CommandResult(const CerberusCommand&, CerberusRuntime&)>;

// ===========================================================================
// Executor — routes commands to handlers
// ===========================================================================

class CerberusCommandExecutor {
public:
    explicit CerberusCommandExecutor(CerberusRuntime& runtime);
    ~CerberusCommandExecutor() = default;

    // Non-copyable, movable
    CerberusCommandExecutor(const CerberusCommandExecutor&) = delete;
    CerberusCommandExecutor& operator=(const CerberusCommandExecutor&) = delete;
    CerberusCommandExecutor(CerberusCommandExecutor&&) noexcept = default;
    CerberusCommandExecutor& operator=(CerberusCommandExecutor&&) noexcept = default;

    /// Execute a parsed command.
    [[nodiscard]] CommandResult execute(const CerberusCommand& cmd);

    /// Convenience: parse + execute in one call.
    [[nodiscard]] CommandResult execute(const std::string& raw_command);

    /// Register a custom handler for a namespace:command pair.
    void register_handler(const std::string& ns, const std::string& cmd, CommandHandler handler);

    /// Check if a handler exists for a command.
    [[nodiscard]] bool has_handler(const std::string& ns, const std::string& cmd) const noexcept;

    /// Get help text for all registered commands.
    [[nodiscard]] std::string help_text() const;

private:
    CerberusRuntime& runtime_;
    std::map<std::string, CommandHandler> handlers_; // key = "namespace:command"

    void register_default_handlers_();
    static std::string make_key_(const std::string& ns, const std::string& cmd);
};

} // namespace hq::cerberus::cli
