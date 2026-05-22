#pragma once
/// @file inference_server.hpp
/// @copyright Copyright (c) 2026 D Hargreaves (AKA Roylepython). LamiaFabrica. All rights reserved.
/// Cerberus Inference Server — local HTTP server exposing an OpenAI-compatible
/// API for Ollama, LM Studio, and other tools to use the NPU/GPU/CPU cluster.
///
/// Endpoints:
///   POST /v1/chat/completions  — OpenAI-compatible chat completions
///   GET  /v1/models            — list available models
///   GET  /health               — device utilization stats
///
/// Design:
///   - Hand-written HTTP/1.1 parser (no external HTTP library)
///   - Single-threaded event loop with accept/read/write
///   - Minimal JSON via string building or nlohmann/json
///   - Device load balancing across NPU + GPU + CPU
///   - Auto-discovery of NPU devices at startup
///
/// @target MinisForum UM790 Pro (7940HS + Radeon 780M + Hailo-8L)
/// @requires C++26
/// @version 1.0.0

#include "hq/cxx26_features.hpp"
#include "hq/async_pipeline.hpp"
#include "hq/gpu_monitor.hpp"
#include "hq/hailo_monitor.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#if UM790_HAS_STD_EXPECTED
#  include <expected>
#endif
#include <filesystem>
#if UM790_HAS_STD_FORMAT
#  include <format>
#endif
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#if UM790_HAS_STD_PRINT
#  include <print>
#else
#  include <cstdio>
#endif
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

// Forward: ONNX Runtime
namespace Ort { class Env; class Session; class SessionOptions; class MemoryInfo; }

namespace hq {

// ===========================================================================
// DeviceType — which accelerator to target
// ===========================================================================

enum class DeviceType : std::uint8_t {
    CPU     = 0,
    GPU     = 1,
    NPU     = 2,
    Auto    = 0xFF,
};

[[nodiscard]] inline constexpr const char* to_string(DeviceType t) noexcept {
    switch (t) {
        case DeviceType::CPU:  return "cpu";
        case DeviceType::GPU:  return "gpu";
        case DeviceType::NPU:  return "npu";
        case DeviceType::Auto: return "auto";
    }
    return "unknown";
}

// ===========================================================================
// DeviceInfo — static and dynamic info about a compute device
// ===========================================================================

struct DeviceInfo {
    std::string   name;                ///< e.g. "Hailo-8L", "AMD Radeon 780M"
    DeviceType    type{DeviceType::CPU};
    std::uint64_t memory_total_bytes{0};
    std::uint64_t memory_used_bytes{0};
    float         utilization_percent{0.0f};   ///< [0, 100]
    float         temperature_celsius{-1.0f};  ///< -1.0f = unavailable
    float         power_watts{-1.0f};
    bool          thermal_throttling{false};
    bool          available{true};
    std::size_t   active_sessions{0};   ///< how many inference sessions bound
};

// ===========================================================================
// ServerConfig — tunable server parameters
// ===========================================================================

struct ServerConfig {
    std::uint16_t  port{8080};                  ///< HTTP listen port
    std::string    bind_address{"127.0.0.1"};   ///< localhost only by default
    std::filesystem::path model_dir;            ///< directory scanned for models
    std::uint32_t  max_concurrent_requests{4};  ///< inflight-request cap
    std::uint32_t  client_timeout_ms{30'000};   ///< idle client socket timeout
    bool           enable_cors{true};           ///< add CORS headers
    bool           verbose_logging{false};      ///< log every request/response
};

// ===========================================================================
// HttpMethod — minimal HTTP method enum
// ===========================================================================

enum class HttpMethod : std::uint8_t {
    GET  = 0,
    POST = 1,
    OPTIONS = 2,
    UNKNOWN = 0xFF,
};

[[nodiscard]] inline HttpMethod parse_method(std::string_view s) noexcept {
    if (s == "GET")     return HttpMethod::GET;
    if (s == "POST")    return HttpMethod::POST;
    if (s == "OPTIONS") return HttpMethod::OPTIONS;
    return HttpMethod::UNKNOWN;
}

// ===========================================================================
// HttpRequest — parsed incoming request
// ===========================================================================

struct HttpRequest {
    HttpMethod     method{HttpMethod::UNKNOWN};
    std::string    path;               ///< e.g. "/v1/chat/completions"
    std::string    body;               ///< raw body (for POST)
    std::unordered_map<std::string, std::string> headers;
};

// ===========================================================================
// HttpResponse — outgoing response to write back
// ===========================================================================

struct HttpResponse {
    std::uint16_t status_code{200};
    std::string   content_type{"application/json"};
    std::string   body;                ///< response body (JSON string)
    bool          close_connection{true}; ///< HTTP/1.1 Connection: close

    /// Render the full HTTP response line (status line + headers + body).
    [[nodiscard]] std::string render() const;
};

// ===========================================================================
// RouteHandler — typedef for route dispatch callbacks
// ===========================================================================

using RouteHandler = std::function<HttpResponse(const HttpRequest&)>;

// ===========================================================================
// SimpleHttpParser — hand-written HTTP/1.1 parser, reads from an fd
// ===========================================================================

class SimpleHttpParser {
public:
    SimpleHttpParser() = default;

    /// Feed raw bytes received from the socket. Returns std::nullopt while
    /// the request is incomplete; returns a fully-parsed HttpRequest on
    /// completion.  Handles chunked transfer encoding and Content-Length.
    [[nodiscard]] std::optional<HttpRequest> feed(std::span<const std::byte> data);

    /// Reset parser state for a new request on the same connection.
    void reset() noexcept;

    /// True if the parser expects more bytes to complete the current request.
    [[nodiscard]] bool wants_more() const noexcept { return wants_more_; }

private:
    enum class State : std::uint8_t {
        METHOD_LINE, HEADERS, BODY, DONE, ParseError
    };

    State parse_state_{State::METHOD_LINE};
    HttpRequest current_;
    std::string buffer_;
    std::size_t body_expected_{0};
    std::size_t body_received_{0};
    bool wants_more_{true};
    bool chunked_{false};

    void parse_line_(std::string_view line);
    [[nodiscard]] std::string_view get_header_(std::string_view key) const noexcept;
};

// ===========================================================================
// DeviceLoadBalancer — tracks utilization, picks best device for inference
//
// Weights: NPU 2.0x, GPU 1.5x, CPU 1.0x — NPU preferred for inference
// workloads (text embeddings, vision backbone). GPU next. CPU fallback.
// ===========================================================================

class DeviceLoadBalancer {
public:
    struct DeviceChoice {
        std::string device_name;
        float       utilization_percent{0.0f};
        bool        available{false};
    };

    explicit DeviceLoadBalancer(bool verbose = false);
    ~DeviceLoadBalancer() noexcept = default;

    DeviceChoice pick_device();
    bool is_any_available();

    float gpu_utilization()    const noexcept { return last_gpu_util_; }
    float gpu_temperature()    const noexcept { return last_gpu_temp_; }
    float hailo_utilization() const noexcept { return last_hailo_util_; }
    float hailo_temperature() const noexcept { return last_hailo_temp_; }

private:
    bool verbose_{false};
    float last_gpu_util_{0.0f}, last_gpu_temp_{0.0f};
    float last_hailo_util_{0.0f}, last_hailo_temp_{0.0f};
    bool gpu_available_{false}, hailo_available_{false};
    std::unique_ptr<GPUMonitor>   gpu_monitor_;
    std::unique_ptr<HailoMonitor> hailo_monitor_;
};

// ===========================================================================
// OpenAI-compatible message structs for chat completions
// ===========================================================================

struct OpenAIMessage {
    std::string role;     ///< "system", "user", "assistant"
    std::string content;  ///< message body text
};

struct OpenAIChoice {
    std::int32_t  index{0};
    OpenAIMessage message;
    std::string   finish_reason; ///< "stop", "length", "tool_calls"
};

struct OpenAIChatCompletionRequest {
    std::string              model;
    std::vector<OpenAIMessage> messages;
    bool                     stream{false};
    std::uint32_t            max_tokens{2048};
    float                    temperature{1.0f};
    float                    top_p{1.0f};
    std::int64_t             seed{-1};

    /// Parse from raw JSON request body. Returns nullopt on parse failure.
    [[nodiscard]] static std::optional<OpenAIChatCompletionRequest> from_json(
        std::string_view body);
};

struct OpenAIChatCompletionResponse {
    std::string               id;
    std::string               object{"chat.completion"};
    std::int64_t              created{0};
    std::string               model;
    std::vector<OpenAIChoice> choices;

    struct Usage {
        std::uint32_t prompt_tokens{0};
        std::uint32_t completion_tokens{0};
        std::uint32_t total_tokens{0};
    } usage;

    /// Serialise to JSON string for the HTTP response body.
    [[nodiscard]] std::string to_json() const;
};

// ===========================================================================
// ModelEntry + ModelRegistry — scans model_dir, lists models
// ===========================================================================

struct ModelEntry {
    std::string id;          ///< unique model ID (e.g. "mistral-7b-npu")
    std::string object{"model"};
    std::int64_t created{0};
    std::string owned_by{"cerberus"};
    DeviceType preferred_device{DeviceType::Auto};
    std::filesystem::path file_path; ///< full path to .onnx / .hef
};

class ModelRegistry {
public:
    explicit ModelRegistry(std::filesystem::path model_dir);
    ~ModelRegistry() noexcept = default;

    /// Re-scan the model directory for new/removed models.
    void rescan();

    /// List all known models as JSON.
    [[nodiscard]] std::string list_models_json() const;

    /// Look up a model by ID string. Returns nullptr if not found.
    [[nodiscard]] const ModelEntry* find_model(std::string_view id) const noexcept;

    /// Get all model entries (snapshot copy).
    [[nodiscard]] std::vector<ModelEntry> all_models() const noexcept;

    /// Number of known models.
    [[nodiscard]] std::size_t model_count() const noexcept { return models_.size(); }

    /// Set the model directory and rescan.
    void set_model_dir(std::filesystem::path dir);

private:
    std::filesystem::path model_dir_;
    std::vector<ModelEntry> models_;
    mutable std::mutex mutex_;

    void scan_();
    [[nodiscard]] static std::optional<DeviceType> detect_device_from_model_(
        const std::filesystem::path& path) noexcept;
};

// ===========================================================================
// ServerStats — accumulated counters exposed by InferenceServer::get_stats()
// ===========================================================================

struct ServerStats {
    std::atomic<std::uint64_t> requests_served{0};
    std::atomic<std::uint64_t> chat_completions{0};
    std::atomic<std::uint64_t> health_checks{0};
    std::atomic<std::uint64_t> model_lists{0};
    std::atomic<std::uint64_t> errors{0};
    std::chrono::steady_clock::time_point start_time{}; ///< epoch until server starts
};

// ===========================================================================
// InferenceServer — the core HTTP server class
// ===========================================================================

class InferenceServer {
public:
    /// Construct with configuration. Does NOT start listening.
    explicit InferenceServer(ServerConfig cfg);
    ~InferenceServer() noexcept;

    InferenceServer(const InferenceServer&) = delete;
    InferenceServer& operator=(const InferenceServer&) = delete;
    InferenceServer(InferenceServer&&) noexcept;
    InferenceServer& operator=(InferenceServer&&) noexcept;

    // ---- Lifecycle ----

    /// Start listening on the configured port. Does device discovery,
    /// model scanning, then enters the event loop. Blocks until stop().
    /// @return 0 on clean shutdown, non-zero on startup failure.
    [[nodiscard]] int start();

    /// Signal the event loop to stop (non-blocking, thread-safe).
    void stop();

    /// Graceful shutdown: stop accepting, drain in-flight, close socket.
    void shutdown() noexcept;

    /// True if the server is currently listening.
    [[nodiscard]] bool running() const noexcept;

    // ---- Configuration ----

    [[nodiscard]] std::uint16_t port() const noexcept { return cfg_.port; }
    [[nodiscard]] const std::filesystem::path& model_dir() const noexcept {
        return cfg_.model_dir;
    }

    void set_model_dir(std::filesystem::path dir);

    // ---- Health / telemetry ----

    /// Snapshot of current cluster utilisation.
    struct HealthSnapshot {
        float cpu_util_percent{0.0f};
        float gpu_util_percent{-1.0f};
        float npu_util_percent{-1.0f};
        float gpu_temp_c{-1.0f};
        float npu_temp_c{-1.0f};
        std::size_t active_sessions{0};
        std::chrono::steady_clock::time_point timestamp;
    };

    /// Get current health snapshot.
    [[nodiscard]] HealthSnapshot get_health() const noexcept;

    /// Get health as JSON for the /health endpoint.
    [[nodiscard]] std::string get_health_json() const noexcept;

    // ---- Utilisation query ----

    /// Force a utilisation refresh across all monitors.
    void refresh_utilization();

    /// Get all device infos (snapshot copy).
    [[nodiscard]] std::vector<DeviceInfo> get_all_devices() const;

    /// Get the recommended device for the next inference.
    [[nodiscard]] DeviceType get_load_balance_hint() const;

    /// Get a snapshot reference to the server stats (lives in Impl).
    [[nodiscard]] const ServerStats& get_stats() const noexcept;

    // ---- Version ----

    [[nodiscard]] static constexpr const char* version() noexcept {
        return "1.0.0";
    }

    // ---- Error ----

    /// Return the last logged error string (thread-local).
    [[nodiscard]] static const std::string& last_error() noexcept;

private:
    ServerConfig cfg_;
    std::unique_ptr<ModelRegistry> model_registry_;
    std::unique_ptr<DeviceLoadBalancer> load_balancer_;
    std::unique_ptr<hq::async::AsyncPipeline> async_pipeline_;

    // Monitoring
    std::unique_ptr<hq::HailoMonitor>    hailo_monitor_;
    std::unique_ptr<hq::GPUMonitor>      gpu_monitor_;

    struct Impl;
    std::unique_ptr<Impl> impl_;

    // ---- Internal route handlers ----

    [[nodiscard]] HttpResponse handle_chat_completions_(const HttpRequest& req);
    [[nodiscard]] HttpResponse handle_list_models_(const HttpRequest& req);
    [[nodiscard]] HttpResponse handle_health_(const HttpRequest& req);
    [[nodiscard]] HttpResponse handle_options_(const HttpRequest& req);

    /// Route dispatcher: maps path -> handler.
    [[nodiscard]] HttpResponse dispatch_(const HttpRequest& req);

    /// Run inference via the async pipeline and return a response.
    [[nodiscard]] std::expected<std::string, std::string>
        run_inference_(const OpenAIChatCompletionRequest& req);

    /// Discover NPU (Hailo) devices at startup. Populates the load balancer.
    void discover_devices_();

    /// Periodic utilisation refresh for load balancer.
    void refresh_telemetry_loop_(std::stop_token token);

    /// Thread pool for async inference runs.
    struct ThreadPool {
        std::vector<std::jthread> threads;
        std::atomic<std::uint32_t> inflight{0};
        std::atomic<bool> shutdown_flag{false};
    };
    std::unique_ptr<ThreadPool> thread_pool_;

    static thread_local std::string tl_last_error_;
};

// ===========================================================================
// Convenience helpers — JSON building without nlohmann
// ===========================================================================

namespace json {
/// Escape a string for safe JSON embedding (quotes, backslash, control chars).
[[nodiscard]] std::string escape(std::string_view raw);

/// Build a JSON object string from key-value pairs.
/// Each pair is {key, json_value} — value is already a JSON fragment string.
template<std::size_t N>
[[nodiscard]] std::string build_object(
    const std::array<std::pair<std::string_view, std::string_view>, N>& fields)
{
    std::string out{"{"};
    for (std::size_t i = 0; i < N; ++i) {
        if (i > 0) out += ',';
        out += '"';
        out += fields[i].first;
        out += '"';
        out += ':';
        out += fields[i].second;
    }
    out += '}';
    return out;
}

/// Build a JSON array from elements.
template<std::size_t N>
[[nodiscard]] std::string build_array(
    const std::array<std::string_view, N>& elements)
{
    std::string out{"["};
    for (std::size_t i = 0; i < N; ++i) {
        if (i > 0) out += ',';
        out += elements[i];
    }
    out += ']';
    return out;
}

/// Quick JSON string value: "escaped_content"
[[nodiscard]] inline std::string str_val(std::string_view v) {
    return '"' + escape(v) + '"';
}

/// Quick JSON integer value: "123"
[[nodiscard]] inline std::string int_val(std::int64_t v) {
    return std::to_string(v);
}

/// Quick JSON float value: "3.14"
[[nodiscard]] inline std::string float_val(float v) {
    return std::to_string(v);
}

/// Quick JSON bool value: "true" or "false"
[[nodiscard]] inline std::string bool_val(bool v) {
    return v ? "true" : "false";
}
} // namespace json

// ===========================================================================
// HttpResponse::render() — inline implementation
// ===========================================================================

inline std::string HttpResponse::render() const {
    std::string out;
    out.reserve(128 + body.size());

    // Status line
    out += "HTTP/1.1 ";
    out += std::to_string(status_code);
    out += " ";
    switch (status_code) {
        case 200: out += "OK"; break;
        case 400: out += "Bad Request"; break;
        case 404: out += "Not Found"; break;
        case 405: out += "Method Not Allowed"; break;
        case 429: out += "Too Many Requests"; break;
        case 500: out += "Internal Server Error"; break;
        case 503: out += "Service Unavailable"; break;
        default:  out += "Unknown"; break;
    }
    out += "\r\n";

    // Headers
    out += "Content-Type: ";
    out += content_type;
    out += "\r\n";

    out += "Content-Length: ";
    out += std::to_string(body.size());
    out += "\r\n";

    out += "Connection: ";
    out += close_connection ? "close" : "keep-alive";
    out += "\r\n";

    out += "Server: Cerberus/1.0.0\r\n";
    out += "Access-Control-Allow-Origin: *\r\n";
    out += "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n";
    out += "Access-Control-Allow-Headers: Content-Type, Authorization\r\n";

    out += "\r\n";
    out += body;

    return out;
}

} // namespace hq
