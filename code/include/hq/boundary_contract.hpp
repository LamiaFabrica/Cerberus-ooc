#pragma once
/// @file boundary_contract.hpp
/// @copyright Copyright (c) 2026 D Hargreaves. All rights reserved.
///
/// Boundary Contract — runtime pre/post/invariant checks with ContractViolation.
///
/// C++26 features used:
///   - std::source_location for automatic file/line reporting
///   - Concepts for contract predicate types
///   - [[nodiscard]] on contract check results
///   - std::expected for non-throwing contract results
///
/// © 2026 D Hargreaves | LamiaFabrica Software

#include <concepts>
#include <expected>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace hq::contract {

// ============================================================================
// ContractViolation — thrown when any boundary contract fails.
// ============================================================================
class ContractViolation : public std::runtime_error {
public:
    explicit ContractViolation(const std::string& msg)
        : std::runtime_error(msg) {}

    explicit ContractViolation(std::string_view msg)
        : std::runtime_error(std::string(msg)) {}
};

// ============================================================================
// Concepts
// ============================================================================

template <typename F>
concept ContractPredicate = std::predicate<F>;

// ============================================================================
// Contract check result (non-throwing path)
// ============================================================================

enum class ContractKind { PreCondition, PostCondition, Invariant };

struct [[nodiscard]] ContractResult {
    bool satisfied = true;
    ContractKind kind = ContractKind::PreCondition;
    std::string message;
    std::source_location location = std::source_location::current();

    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return satisfied;
    }

    [[nodiscard]] std::expected<void, ContractViolation> to_expected() const {
        if (satisfied) return {};
        return std::unexpected(ContractViolation(message));
    }
};

// ============================================================================
// Throwing API
// ============================================================================

/// Throws ContractViolation if predicate is false.
inline void pre_condition(
    bool predicate,
    std::string_view msg = "pre-condition violated",
    const std::source_location& loc = std::source_location::current())
{
    if (!predicate) {
        auto full = std::string(msg) + " [" + loc.file_name() + ":"
                    + std::to_string(loc.line()) + "]";
        throw ContractViolation(full);
    }
}

/// Throws ContractViolation if predicate is false.
inline void post_condition(
    bool predicate,
    std::string_view msg = "post-condition violated",
    const std::source_location& loc = std::source_location::current())
{
    if (!predicate) {
        auto full = std::string(msg) + " [" + loc.file_name() + ":"
                    + std::to_string(loc.line()) + "]";
        throw ContractViolation(full);
    }
}

/// Throws ContractViolation if predicate is false.
inline void invariant(
    bool predicate,
    std::string_view msg = "invariant violated",
    const std::source_location& loc = std::source_location::current())
{
    if (!predicate) {
        auto full = std::string(msg) + " [" + loc.file_name() + ":"
                    + std::to_string(loc.line()) + "]";
        throw ContractViolation(full);
    }
}

// ============================================================================
// Predicate overloads (concepts)
// ============================================================================

template <ContractPredicate F>
inline void pre_condition(
    F&& pred,
    std::string_view msg = "pre-condition predicate returned false",
    const std::source_location& loc = std::source_location::current())
{
    pre_condition(std::forward<F>(pred)(), msg, loc);
}

template <ContractPredicate F>
inline void post_condition(
    F&& pred,
    std::string_view msg = "post-condition predicate returned false",
    const std::source_location& loc = std::source_location::current())
{
    post_condition(std::forward<F>(pred)(), msg, loc);
}

template <ContractPredicate F>
inline void invariant(
    F&& pred,
    std::string_view msg = "invariant predicate returned false",
    const std::source_location& loc = std::source_location::current())
{
    invariant(std::forward<F>(pred)(), msg, loc);
}

// ============================================================================
// Non-throwing API (returns std::expected)
// ============================================================================

[[nodiscard]] inline ContractResult check_pre(
    bool predicate,
    std::string_view msg = "pre-condition violated",
    const std::source_location& loc = std::source_location::current())
{
    if (!predicate) {
        auto full = std::string(msg) + " [" + loc.file_name() + ":"
                    + std::to_string(loc.line()) + "]";
        return ContractResult{false, ContractKind::PreCondition, full, loc};
    }
    return ContractResult{true, ContractKind::PreCondition, {}, loc};
}

[[nodiscard]] inline ContractResult check_post(
    bool predicate,
    std::string_view msg = "post-condition violated",
    const std::source_location& loc = std::source_location::current())
{
    if (!predicate) {
        auto full = std::string(msg) + " [" + loc.file_name() + ":"
                    + std::to_string(loc.line()) + "]";
        return ContractResult{false, ContractKind::PostCondition, full, loc};
    }
    return ContractResult{true, ContractKind::PostCondition, {}, loc};
}

[[nodiscard]] inline ContractResult check_invariant(
    bool predicate,
    std::string_view msg = "invariant violated",
    const std::source_location& loc = std::source_location::current())
{
    if (!predicate) {
        auto full = std::string(msg) + " [" + loc.file_name() + ":"
                    + std::to_string(loc.line()) + "]";
        return ContractResult{false, ContractKind::Invariant, full, loc};
    }
    return ContractResult{true, ContractKind::Invariant, {}, loc};
}

} // namespace hq::contract
