#pragma once
/// @file cerberus_error.hpp
/// @copyright Copyright (c) 2026 D Hargreaves. All rights reserved.
///
/// Central error type and expected alias for the Cerberus heterogeneous
/// inference runtime.  All fallible functions return hq::Expected<T>.
///
/// @version 1.0.0

#include "hq/concepts.hpp"
#include <expected>
#include <string>

namespace hq {

/// @brief Unified error enumeration for all Cerberus subsystems.
enum class CerberusError {
    Unknown,
    DeviceNotFound,
    NotImplemented,
    InvalidArgument,
    OutOfMemory,
    RuntimeError,
    CompilationError,
    BackendCompileFailed,
    BackendExecuteFailed,
    EmptyExecutionPlan,
    GraphEmpty,
    ExecutionError,
    UnsupportedOperation,
    ConfigurationError,
    NetworkError,
    FileNotFound,
    PermissionDenied,
    Timeout,
};

/// @brief Convert CerberusError to human-readable string.
inline std::string to_string(CerberusError e) {
    switch (e) {
        case CerberusError::Unknown:               return "Unknown";
        case CerberusError::DeviceNotFound:        return "DeviceNotFound";
        case CerberusError::NotImplemented:        return "NotImplemented";
        case CerberusError::InvalidArgument:       return "InvalidArgument";
        case CerberusError::OutOfMemory:           return "OutOfMemory";
        case CerberusError::RuntimeError:          return "RuntimeError";
        case CerberusError::CompilationError:      return "CompilationError";
        case CerberusError::BackendCompileFailed:  return "BackendCompileFailed";
        case CerberusError::BackendExecuteFailed:  return "BackendExecuteFailed";
        case CerberusError::EmptyExecutionPlan:    return "EmptyExecutionPlan";
        case CerberusError::GraphEmpty:            return "GraphEmpty";
        case CerberusError::ExecutionError:        return "ExecutionError";
        case CerberusError::UnsupportedOperation:  return "UnsupportedOperation";
        case CerberusError::ConfigurationError:    return "ConfigurationError";
        case CerberusError::NetworkError:          return "NetworkError";
        case CerberusError::FileNotFound:          return "FileNotFound";
        case CerberusError::PermissionDenied:      return "PermissionDenied";
        case CerberusError::Timeout:               return "Timeout";
    }
    return "Unknown";
}

/// @brief Convenience alias for std::expected<T, CerberusError>.
template <typename T>
    requires hq::HqExpectedValue<T>
using Expected = std::expected<T, CerberusError>;

/// @brief Convenience alias for std::expected<void, CerberusError>.
using ExpectedVoid = std::expected<void, CerberusError>;

} // namespace hq
