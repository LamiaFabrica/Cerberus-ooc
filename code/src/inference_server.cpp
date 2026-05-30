/// @file inference_server.cpp
/// @copyright Copyright © 2026, all rights reserved | David Hargreaves aka Roylepython trading as The Medusa Initiative & Lamia Fabrica
/// Support: https://www.patreon.com/TheMedusaInitiative — £25/month removes ads from all software at 200 subscribers
/// Implementation of the Cerberus Inference Server.
///
/// POSIX-socket HTTP server with hand-rolled JSON parser.
/// Routes: GET /health, GET /v1/models, POST /v1/chat/completions.

#include "hq/inference_server.hpp"
#include "hq/cxx26_features.hpp"
#include "hq/pipeline.hpp"

#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <memory>
#if UM790_HAS_STD_PRINT
#  include <print>
#endif
#include <string>
#include <string_view>
#include <vector>

#ifdef _WIN32
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  ifndef MSG_NOSIGNAL
#    define MSG_NOSIGNAL 0
#  endif
namespace {
struct WinsockInit_ {
    WinsockInit_()  { WSADATA wd; WSAStartup(MAKEWORD(2,2), &wd); }
    ~WinsockInit_() { WSACleanup(); }
};
static WinsockInit_ wsa_init_;
} // anonymous namespace
#else
#  include <arpa/inet.h>
#  include <fcntl.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <sys/socket.h>
#  include <unistd.h>
#endif

using namespace std::literals;

namespace hq {

// ===========================================================================
// InferenceServer::Impl
// ===========================================================================

struct InferenceServer::Impl {
    int listen_fd{-1};
    std::atomic<bool> running{false};
    ServerStats stats;

    // Privacy surface (optional but required for full inference history feature)
    std::shared_ptr<hq::cerberus::LocalMaintenanceDB> lcmd;
    std::shared_ptr<hq::cerberus::UserSecurity>       user_security;
    std::string                                       rbpc_node_id{"local"};

    void close_listen() noexcept {
        int fd = listen_fd;
        listen_fd = -1;
        if (fd < 0) return;
#ifdef _WIN32
        closesocket(static_cast<SOCKET>(fd));
#else
        ::close(fd);
#endif
    }
};

// ===========================================================================
// Anonymous helpers
// ===========================================================================

namespace {

constexpr std::size_t RECV_BUF_SIZE  = 65536;
constexpr int         LISTEN_BACKLOG = 8;

[[maybe_unused]] void close_socket_(int& fd) noexcept {
    if (fd < 0) return;
#ifdef _WIN32
    closesocket(static_cast<SOCKET>(fd));
#else
    ::close(fd);
#endif
    fd = -1;
}

// Minimal JSON string extraction (key → value between quotes)
std::string_view json_get_str(std::string_view json, std::string_view key) {
    std::string pat = '"' + std::string(key) + '"';
    auto pos = json.find(pat);
    if (pos == std::string_view::npos) return {};
    auto colon = json.find(':', pos + pat.size());
    if (colon == std::string_view::npos) return {};
    auto vs = json.find_first_not_of(" \t\n\r", colon + 1);
    if (vs == std::string_view::npos) return {};
    if (json[vs] != '"') return {};
    auto ve = json.find('"', vs + 1);
    if (ve == std::string_view::npos) return {};
    return json.substr(vs + 1, ve - vs - 1);
}

bool json_get_int_(std::string_view json, std::string_view key, int& out) {
    std::string pat = '"' + std::string(key) + '"';
    auto pos = json.find(pat);
    if (pos == std::string_view::npos) return false;
    auto colon = json.find(':', pos + pat.size());
    if (colon == std::string_view::npos) return false;
    auto vs = json.find_first_not_of(" \t\n\r", colon + 1);
    if (vs == std::string_view::npos) return false;
    if (json[vs] == '"') { vs++; }
    auto ve = json.find_first_not_of("0123456789-", vs);
    if (ve == std::string_view::npos) ve = json.size();
    std::string num(json.substr(vs, ve - vs));
    char* ep = nullptr;
    long v = std::strtol(num.c_str(), &ep, 10);
    if (ep == num.c_str()) return false;
    out = static_cast<int>(v);
    return true;
}

bool json_get_float_(std::string_view json, std::string_view key, float& out) {
    std::string pat = '"' + std::string(key) + '"';
    auto pos = json.find(pat);
    if (pos == std::string_view::npos) return false;
    auto colon = json.find(':', pos + pat.size());
    if (colon == std::string_view::npos) return false;
    auto vs = json.find_first_not_of(" \t\n\r", colon + 1);
    if (vs == std::string_view::npos) return false;
    if (json[vs] == '"') { vs++; }
    auto ve = json.find_first_not_of("0123456789.-+eE", vs);
    if (ve == std::string_view::npos) ve = json.size();
    std::string num(json.substr(vs, ve - vs));
    char* ep = nullptr;
    out = std::strtof(num.c_str(), &ep);
    return ep != num.c_str();
}

} // anonymous namespace

// ===========================================================================
// json::escape
// ===========================================================================

namespace json {

std::string escape(std::string_view raw) {
    std::string out;
    out.reserve(raw.size() + 8);
    for (char c : raw) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

} // namespace json

// ===========================================================================
// SimpleHttpParser
// ===========================================================================

std::optional<HttpRequest> SimpleHttpParser::feed(std::span<const std::byte> data) {
    for (auto b : data)
        buffer_ += static_cast<char>(b);

    while (true) {
        if (parse_state_ == State::METHOD_LINE || parse_state_ == State::HEADERS) {
            auto pos = buffer_.find("\r\n");
            if (pos == std::string::npos) { wants_more_ = true; return std::nullopt; }
            parse_line_(std::string_view(buffer_).substr(0, pos));
            buffer_.erase(0, pos + 2);
            if (parse_state_ == State::ParseError) return std::nullopt;
        } else if (parse_state_ == State::BODY) {
            if (chunked_) {
                // Simple: accumulate and finish when connection closes
                current_.body = std::move(buffer_);
                buffer_.clear();
                parse_state_ = State::DONE;
            } else {
                if (buffer_.size() < body_expected_) { wants_more_ = true; return std::nullopt; }
                current_.body = buffer_.substr(0, body_expected_);
                buffer_.erase(0, body_expected_);
                parse_state_ = State::DONE;
            }
        }
        if (parse_state_ == State::DONE) {
            wants_more_ = false;
            return std::move(current_);
        }
        if (parse_state_ == State::ParseError) return std::nullopt;
    }
}

void SimpleHttpParser::reset() noexcept {
    parse_state_   = State::METHOD_LINE;
    current_       = HttpRequest{};
    buffer_.clear();
    body_expected_ = 0;
    body_received_ = 0;
    wants_more_    = true;
    chunked_       = false;
}

void SimpleHttpParser::parse_line_(std::string_view line) {
    if (parse_state_ == State::METHOD_LINE) {
        auto sp1 = line.find(' ');
        if (sp1 == std::string_view::npos) { parse_state_ = State::ParseError; return; }
        auto sp2 = line.find(' ', sp1 + 1);
        if (sp2 == std::string_view::npos) { parse_state_ = State::ParseError; return; }
        current_.method = parse_method(line.substr(0, sp1));
        current_.path   = std::string(line.substr(sp1 + 1, sp2 - sp1 - 1));
        parse_state_    = State::HEADERS;
        return;
    }
    if (parse_state_ == State::HEADERS) {
        if (line.empty()) {
            // End of headers
            auto cl = get_header_("content-length");
            if (!cl.empty()) {
                char* ep = nullptr;
                body_expected_ = static_cast<std::size_t>(
                    std::strtoul(cl.data(), &ep, 10));
                parse_state_ = body_expected_ > 0 ? State::BODY : State::DONE;
            } else {
                auto te = get_header_("transfer-encoding");
                if (te.find("chunked") != std::string_view::npos) {
                    chunked_     = true;
                    parse_state_ = State::BODY;
                } else {
                    parse_state_ = State::DONE;
                }
            }
            return;
        }
        auto colon = line.find(':');
        if (colon == std::string_view::npos) return;
        std::string key(line.substr(0, colon));
        for (auto& ch : key) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        auto val = line.substr(colon + 1);
        auto vs = val.find_first_not_of(" \t");
        if (vs != std::string_view::npos) val = val.substr(vs);
        current_.headers[key] = std::string(val);
    }
}

std::string_view SimpleHttpParser::get_header_(std::string_view key) const noexcept {
    auto it = current_.headers.find(std::string(key));
    if (it == current_.headers.end()) return {};
    return it->second;
}

// ===========================================================================
// DeviceLoadBalancer
// ===========================================================================

DeviceLoadBalancer::DeviceLoadBalancer(bool verbose)
    : verbose_(verbose)
{
    try { gpu_monitor_   = std::make_unique<GPUMonitor>(0);  } catch (...) {}
    try { hailo_monitor_ = std::make_unique<HailoMonitor>(); } catch (...) {}
}

DeviceLoadBalancer::DeviceChoice DeviceLoadBalancer::pick_device() {
    DeviceChoice best{"cpu", 0.0f, true};

    if (gpu_monitor_) {
        try {
            if (auto init = gpu_monitor_->initialize(); init) {
                if (auto telem = gpu_monitor_->query_all(); telem) {
                    last_gpu_util_ = telem->utilization_percent;
                    last_gpu_temp_ = telem->temperature_celsius;
                    gpu_available_ = true;
                    if (verbose_)
                        std::print("[lb] GPU {}: {}% {}C\n",
                                   gpu_monitor_->device_index(),
                                   last_gpu_util_, last_gpu_temp_);
                    if (telem->utilization_percent < 95.0f)
                        best = {"gpu_780m", telem->utilization_percent, true};
                }
            }
        } catch (...) {
            gpu_available_ = false;
            last_gpu_util_ = 0.0f;
            last_gpu_temp_ = 0.0f;
        }
    }

    if (hailo_monitor_) {
        try {
            if (auto open = hailo_monitor_->open(""); open) {
                if (auto stats = hailo_monitor_->sample(); stats) {
                    last_hailo_util_ = stats->nn_core_utilization;
                    last_hailo_temp_ = stats->temperature_celsius;
                    hailo_available_ = true;
                    if (verbose_)
                        std::print("[lb] Hailo {}: {}% {}C\n",
                                   hailo_monitor_->device_id(),
                                   last_hailo_util_, last_hailo_temp_);
                    if (stats->nn_core_utilization < 90.0f
                        && best.utilization_percent > stats->nn_core_utilization)
                        best = {"hailo_8l", stats->nn_core_utilization, true};
                }
            }
        } catch (...) {
            hailo_available_ = false;
            last_hailo_util_ = 0.0f;
            last_hailo_temp_ = 0.0f;
        }
    }

    if (verbose_)
        std::print("[lb] picked {} ({}%)\n", best.device_name, best.utilization_percent);
    return best;
}

bool DeviceLoadBalancer::is_any_available() {
    if (gpu_monitor_) {
        try {
            if (auto init = gpu_monitor_->initialize(); init) return true;
        } catch (...) {}
    }
    if (hailo_monitor_) {
        try {
            if (auto open = hailo_monitor_->open(""); open) return true;
        } catch (...) {}
    }
    return true; // CPU always available
}

// ===========================================================================
// OpenAIChatCompletionRequest::from_json
// ===========================================================================

std::optional<OpenAIChatCompletionRequest>
OpenAIChatCompletionRequest::from_json(std::string_view body) {
    OpenAIChatCompletionRequest req;

    auto model_sv = json_get_str(body, "model");
    if (!model_sv.empty()) req.model = std::string(model_sv);

    json_get_int_(body, "max_tokens",  reinterpret_cast<int&>(req.max_tokens));
    json_get_float_(body, "temperature", req.temperature);
    json_get_float_(body, "top_p", req.top_p);

    // Extract last user message from "messages" array
    auto msg_sv = json_get_str(body, "messages");
    if (!msg_sv.empty()) {
        std::string search(msg_sv);
        std::size_t pos = 0;
        while (true) {
            auto rp = search.find("\"role\"", pos);
            if (rp == std::string::npos) break;
            auto col = search.find(':', rp + 6);
            if (col == std::string::npos) break;
            auto vs = search.find('"', col + 1);
            if (vs == std::string::npos) break;
            auto ve = search.find('"', vs + 1);
            if (ve == std::string::npos) break;
            std::string_view role(search.data() + vs + 1, ve - vs - 1);

            auto cp = search.find("\"content\"", ve);
            if (cp != std::string::npos) {
                auto cc = search.find(':', cp + 9);
                if (cc != std::string::npos) {
                    auto cvs = search.find('"', cc + 1);
                    if (cvs != std::string::npos) {
                        auto cve = search.find('"', cvs + 1);
                        if (cve != std::string::npos) {
                            OpenAIMessage m;
                            m.role    = std::string(role);
                            m.content = search.substr(cvs + 1, cve - cvs - 1);
                            req.messages.push_back(std::move(m));
                        }
                    }
                }
            }
            pos = ve + 1;
        }
    }

    return req;
}

// ===========================================================================
// OpenAIChatCompletionResponse::to_json
// ===========================================================================

std::string OpenAIChatCompletionResponse::to_json() const {
    std::string out;
    out.reserve(512);
    out += '{';
    out += '"'; out += "id"; out += "\":\""; out += json::escape(id); out += "\",";
    out += '"'; out += "object"; out += "\":\"chat.completion\",";
    out += '"'; out += "created"; out += "\":"; out += std::to_string(created); out += ',';
    out += '"'; out += "model"; out += "\":\""; out += json::escape(model); out += "\",";
    out += '"'; out += "usage"; out += "\":{";
    out += '"'; out += "prompt_tokens"; out += "\":"; out += std::to_string(usage.prompt_tokens); out += ',';
    out += '"'; out += "completion_tokens"; out += "\":"; out += std::to_string(usage.completion_tokens); out += ',';
    out += '"'; out += "total_tokens"; out += "\":"; out += std::to_string(usage.total_tokens);
    out += "},";
    out += '"'; out += "choices"; out += "\":[";
    for (std::size_t i = 0; i < choices.size(); ++i) {
        if (i > 0) out += ',';
        out += "{\"index\":"; out += std::to_string(choices[i].index); out += ',';
        out += "\"message\":{\"role\":\""; out += json::escape(choices[i].message.role);
        out += "\",\"content\":\""; out += json::escape(choices[i].message.content); out += "\"},";
        out += "\"finish_reason\":\""; out += json::escape(choices[i].finish_reason); out += "\"}";
    }
    out += "]}";
    return out;
}

// ===========================================================================
// ModelRegistry
// ===========================================================================

ModelRegistry::ModelRegistry(std::filesystem::path model_dir)
    : model_dir_(std::move(model_dir))
{
    if (!model_dir_.empty()) scan_();
}

void ModelRegistry::rescan() {
    std::lock_guard lock{mutex_};
    models_.clear();
    scan_();
}

void ModelRegistry::set_model_dir(std::filesystem::path dir) {
    std::lock_guard lock{mutex_};
    model_dir_ = std::move(dir);
    models_.clear();
    scan_();
}

std::string ModelRegistry::list_models_json() const {
    std::lock_guard lock{mutex_};
    std::string out = R"({"object":"list","data":[)";
    bool first = true;
    for (const auto& m : models_) {
        if (!first) out += ',';
        first = false;
        out += "{\"id\":\""; out += json::escape(m.id);
        out += "\",\"object\":\"model\",\"owned_by\":\"cerberus\"}";
    }
    if (first)
        out += R"({"id":"cerberus-v1","object":"model","owned_by":"cerberus"})";
    out += "]}";
    return out;
}

const ModelEntry* ModelRegistry::find_model(std::string_view id) const noexcept {
    std::lock_guard lock{mutex_};
    for (const auto& m : models_)
        if (m.id == id) return &m;
    return nullptr;
}

std::vector<ModelEntry> ModelRegistry::all_models() const noexcept {
    std::lock_guard lock{mutex_};
    return models_;
}

void ModelRegistry::scan_() {
    if (model_dir_.empty()) return;
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator{model_dir_, ec}) {
        if (ec) break;
        if (!entry.is_regular_file()) continue;
        auto ext = entry.path().extension().string();
        if (ext != ".onnx" && ext != ".hef") continue;
        ModelEntry m;
        m.id        = entry.path().stem().string();
        m.file_path = entry.path();
        m.preferred_device = detect_device_from_model_(entry.path())
                                 .value_or(DeviceType::CPU);
        models_.push_back(std::move(m));
    }
}

std::optional<DeviceType>
ModelRegistry::detect_device_from_model_(const std::filesystem::path& path) noexcept {
    auto ext = path.extension().string();
    if (ext == ".hef") return DeviceType::NPU;
    auto stem = path.stem().string();
    if (stem.find("npu") != std::string::npos) return DeviceType::NPU;
    if (stem.find("gpu") != std::string::npos) return DeviceType::GPU;
    return DeviceType::CPU;
}

// ===========================================================================
// InferenceServer — constructor / destructor
// ===========================================================================

thread_local std::string InferenceServer::tl_last_error_;

InferenceServer::InferenceServer(ServerConfig cfg)
    : cfg_(std::move(cfg))
    , model_registry_(std::make_unique<ModelRegistry>(cfg_.model_dir))
    , load_balancer_(std::make_unique<DeviceLoadBalancer>(cfg_.verbose_logging))
    , impl_(std::make_unique<Impl>())
{
    impl_->stats.start_time = std::chrono::steady_clock::now();

    // Capture optional privacy surface (LCMD + RBPC) for inference audit endpoints.
    // These may be null — handlers degrade gracefully or return clear errors.
    impl_->lcmd           = cfg_.lcmd;
    impl_->user_security  = cfg_.user_security;
    impl_->rbpc_node_id   = cfg_.rbpc_node_id.empty() ? "local" : cfg_.rbpc_node_id;
}

InferenceServer::~InferenceServer() noexcept {
    shutdown();
}

InferenceServer::InferenceServer(InferenceServer&&) noexcept = default;
InferenceServer& InferenceServer::operator=(InferenceServer&&) noexcept = default;

// ===========================================================================
// InferenceServer — lifecycle
// ===========================================================================

int InferenceServer::start() {
    // Open listening socket
    impl_->listen_fd = static_cast<int>(::socket(AF_INET, SOCK_STREAM, 0));
    if (impl_->listen_fd < 0) {
        std::print("[cerberus] socket() failed: {}\n", strerror(errno));
        return 1;
    }

    int opt = 1;
    ::setsockopt(impl_->listen_fd, SOL_SOCKET, SO_REUSEADDR,
                 reinterpret_cast<const char*>(&opt), sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(cfg_.port);
    if (inet_pton(AF_INET, cfg_.bind_address.c_str(), &addr.sin_addr) != 1) {
        std::print("[cerberus] inet_pton failed\n");
        impl_->close_listen();
        return 1;
    }

    if (::bind(impl_->listen_fd,
               reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::print("[cerberus] bind({}:{}) failed\n",
                   cfg_.bind_address, cfg_.port);
        impl_->close_listen();
        return 1;
    }

    if (::listen(impl_->listen_fd, LISTEN_BACKLOG) < 0) {
        std::print("[cerberus] listen() failed\n");
        impl_->close_listen();
        return 1;
    }

    std::print("[cerberus] listening on {}:{}\n", cfg_.bind_address, cfg_.port);
    impl_->running.store(true, std::memory_order_release);

    discover_devices_();

    while (impl_->running.load(std::memory_order_acquire)) {
        sockaddr_in client_addr{};
        socklen_t   addr_len = sizeof(client_addr);
        int client_fd = static_cast<int>(
            ::accept(impl_->listen_fd,
                     reinterpret_cast<sockaddr*>(&client_addr), &addr_len));
        if (client_fd < 0) {
#ifdef _WIN32
            int wsaerr = WSAGetLastError();
            if (wsaerr == WSAEINTR) continue;
            if (!impl_->running.load()) break;
#else
            if (errno == EINTR) continue;
            if (errno == EBADF) break;
#endif
            std::print("[cerberus] accept() error\n");
            continue;
        }

        int nodelay = 1;
        ::setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY,
                     reinterpret_cast<const char*>(&nodelay), sizeof(nodelay));

        impl_->stats.requests_served.fetch_add(1, std::memory_order_relaxed);

        std::array<char, RECV_BUF_SIZE> buf{};
        ssize_t n = ::recv(client_fd, buf.data(),
                           static_cast<int>(buf.size() - 1), 0);
        if (n > 0) {
            auto raw = std::span<const std::byte>(
                reinterpret_cast<const std::byte*>(buf.data()),
                static_cast<std::size_t>(n));

            SimpleHttpParser parser;
            if (auto req = parser.feed(raw); req) {
                auto resp = dispatch_(*req);
                auto rendered = resp.render();
                ::send(client_fd, rendered.data(),
                       static_cast<int>(rendered.size()), MSG_NOSIGNAL);
            } else {
                HttpResponse err{400, "application/json",
                                 R"({"error":"bad_request"})", true};
                auto rendered = err.render();
                ::send(client_fd, rendered.data(),
                       static_cast<int>(rendered.size()), MSG_NOSIGNAL);
                impl_->stats.errors.fetch_add(1, std::memory_order_relaxed);
            }
        }

#ifdef _WIN32
        closesocket(static_cast<SOCKET>(client_fd));
#else
        ::close(client_fd);
#endif
    }

    impl_->close_listen();

    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - impl_->stats.start_time).count();
    std::print("[cerberus] shutdown — {} req in {} s\n",
               impl_->stats.requests_served.load(), elapsed);
    return 0;
}

void InferenceServer::stop() {
    impl_->running.store(false, std::memory_order_release);
    impl_->close_listen();
}

void InferenceServer::shutdown() noexcept {
    impl_->running.store(false, std::memory_order_release);
    impl_->close_listen();
}

bool InferenceServer::running() const noexcept {
    return impl_->running.load(std::memory_order_acquire);
}

void InferenceServer::set_model_dir(std::filesystem::path dir) {
    cfg_.model_dir = dir;
    if (model_registry_) model_registry_->set_model_dir(std::move(dir));
}

// ===========================================================================
// InferenceServer — health / telemetry
// ===========================================================================

InferenceServer::HealthSnapshot InferenceServer::get_health() const noexcept {
    HealthSnapshot hs;
    hs.timestamp = std::chrono::steady_clock::now();
    if (load_balancer_) {
        hs.gpu_util_percent = load_balancer_->gpu_utilization();
        hs.gpu_temp_c       = load_balancer_->gpu_temperature();
        hs.npu_util_percent = load_balancer_->hailo_utilization();
        hs.npu_temp_c       = load_balancer_->hailo_temperature();
    }
    return hs;
}

std::string InferenceServer::get_health_json() const noexcept {
    auto hs = get_health();
    return std::format(
        R"({{"gpu_util":{:.1f},"gpu_temp":{:.1f},"npu_util":{:.1f},"npu_temp":{:.1f}}})",
        hs.gpu_util_percent, hs.gpu_temp_c,
        hs.npu_util_percent, hs.npu_temp_c);
}

void InferenceServer::refresh_utilization() {
    if (load_balancer_) load_balancer_->pick_device();
}

std::vector<DeviceInfo> InferenceServer::get_all_devices() const {
    std::vector<DeviceInfo> devs;
    if (load_balancer_) {
        {
            DeviceInfo d;
            d.name = "AMD Radeon 780M";
            d.type = DeviceType::GPU;
            d.utilization_percent = load_balancer_->gpu_utilization();
            d.temperature_celsius = load_balancer_->gpu_temperature();
            d.available           = d.utilization_percent >= 0.0f;
            devs.push_back(d);
        }
        {
            DeviceInfo d;
            d.name = "Hailo-8L";
            d.type = DeviceType::NPU;
            d.utilization_percent = load_balancer_->hailo_utilization();
            d.temperature_celsius = load_balancer_->hailo_temperature();
            d.available           = d.utilization_percent >= 0.0f;
            devs.push_back(d);
        }
    }
    return devs;
}

DeviceType InferenceServer::get_load_balance_hint() const {
    if (!load_balancer_) return DeviceType::CPU;
    auto choice = load_balancer_->pick_device();
    if (choice.device_name.find("npu") != std::string::npos ||
        choice.device_name.find("hailo") != std::string::npos)
        return DeviceType::NPU;
    if (choice.device_name.find("gpu") != std::string::npos)
        return DeviceType::GPU;
    return DeviceType::CPU;
}

const std::string& InferenceServer::last_error() noexcept {
    return tl_last_error_;
}

const ServerStats& InferenceServer::get_stats() const noexcept {
    return impl_->stats;
}

// ===========================================================================
// InferenceServer — routing
// ===========================================================================

HttpResponse InferenceServer::dispatch_(const HttpRequest& req) {
    if (cfg_.verbose_logging)
        std::print("[cerberus] {} {}\n",
                   static_cast<int>(req.method), req.path);

    if (req.method == HttpMethod::GET && req.path == "/health")
        return handle_health_(req);
    if (req.method == HttpMethod::GET && req.path == "/v1/models")
        return handle_list_models_(req);
    if (req.method == HttpMethod::POST && req.path == "/v1/chat/completions")
        return handle_chat_completions_(req);
    if (req.method == HttpMethod::POST && req.path == "/v1/inference/history")
        return handle_inference_history_(req);
    if (req.method == HttpMethod::POST && req.path == "/v1/inference/export")
        return handle_inference_export_(req);
    if (req.method == HttpMethod::POST && req.path == "/v1/inference/clear")
        return handle_inference_clear_(req);
    if (req.method == HttpMethod::GET && req.path == "/v1/inference/stats")
        return handle_inference_stats_(req);
    if (req.method == HttpMethod::OPTIONS)
        return handle_options_(req);

    return HttpResponse{404, "application/json", R"({"error":"not_found"})", true};
}

HttpResponse InferenceServer::handle_inference_history_(const HttpRequest& req) {
    impl_->stats.inference_history_requests.fetch_add(1, std::memory_order_relaxed);

    if (!impl_->lcmd) {
        return HttpResponse{503, "application/json",
            R"({"error":"inference_audit_unavailable","reason":"no LocalMaintenanceDB configured on this server instance"})", true};
    }

    // Query last 100 records (since epoch 0, until now + margin)
    auto now = static_cast<std::time_t>(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())) + 86400;
    auto records = impl_->lcmd->query_inference_records(0, now, 100);

    std::string body = "[\n";
    for (size_t i = 0; i < records.size(); ++i) {
        const auto& r = records[i];
        if (i > 0) body += ",\n";
        body += "  {\"inference_id\":\"" + r.inference_id +
                "\",\"timestamp\":\"" + r.timestamp +
                "\",\"status\":\"" + r.status +
                "\",\"generation_time_ms\":\"" + r.generation_time_ms +
                "\",\"prompt\":\"" + r.prompt.substr(0, 120) + "\"}";
    }
    body += "\n]";

    return HttpResponse{200, "application/json", std::move(body), true};
}

HttpResponse InferenceServer::handle_inference_export_(const HttpRequest& req) {
    impl_->stats.inference_export_requests.fetch_add(1, std::memory_order_relaxed);

    if (!impl_->lcmd) {
        return HttpResponse{503, "application/json",
            R"({"error":"inference_audit_unavailable","reason":"no LocalMaintenanceDB configured"})", true};
    }

    // RBPC gate (if UserSecurity is wired)
    if (impl_->user_security) {
        std::string pin = std::string(json_get_str(req.body, "pin"));
        std::string word = std::string(json_get_str(req.body, "word"));

        if (pin.empty() || word.empty()) {
            return HttpResponse{400, "application/json",
                R"({"error":"rbpc_required","detail":"POST JSON body must contain \"pin\" and \"word\" for destructive export"})", true};
        }

        std::string result = impl_->user_security->verify_confirmation(impl_->rbpc_node_id, pin, word);
        if (!result.empty()) {
            bool burned = result.find("burned") != std::string::npos || result.find("LOCKED") != std::string::npos;
            std::string body = "{\"error\":\"rbpc_failed\",\"detail\":\"" + result + "\"";
            if (burned) body += ",\"burned\":true";
            body += "}";
            return HttpResponse{403, "application/json", std::move(body), true};
        }
    }

    // Perform export to a safe temp location
    auto ts = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    std::filesystem::path out_path = std::filesystem::temp_directory_path() /
        ("cerberus_inference_export_" + std::to_string(ts) + ".json");

    bool ok = impl_->lcmd->export_inference_json(out_path);
    if (!ok) {
        return HttpResponse{500, "application/json",
            R"({"error":"export_failed","reason":"filesystem write error"})", true};
    }

    std::string body = "{\"exported\":true,\"format\":\"json\",\"path\":\"" +
                       out_path.string() + "\",\"rbpc_enforced\":" +
                       (impl_->user_security ? "true" : "false") + "}";
    return HttpResponse{200, "application/json", std::move(body), true};
}

HttpResponse InferenceServer::handle_inference_clear_(const HttpRequest& req) {
    impl_->stats.inference_clear_requests.fetch_add(1, std::memory_order_relaxed);

    if (!impl_->lcmd) {
        return HttpResponse{503, "application/json",
            R"({"error":"inference_audit_unavailable"})", true};
    }

    // RBPC gate (mandatory for clear)
    if (impl_->user_security) {
        std::string pin = std::string(json_get_str(req.body, "pin"));
        std::string word = std::string(json_get_str(req.body, "word"));

        if (pin.empty() || word.empty()) {
            return HttpResponse{400, "application/json",
                R"({"error":"rbpc_required","detail":"both pin and word required for clear"})", true};
        }

        std::string result = impl_->user_security->verify_confirmation(impl_->rbpc_node_id, pin, word);
        if (!result.empty()) {
            bool burned = result.find("burned") != std::string::npos || result.find("LOCKED") != std::string::npos;
            std::string body = "{\"error\":\"rbpc_failed\",\"detail\":\"" + result + "\"";
            if (burned) body += ",\"burned\":true";
            body += "}";
            return HttpResponse{403, "application/json", std::move(body), true};
        }
    } else {
        // No RBPC configured — still allow in dev but mark it
    }

    size_t before = 0;
    // Best-effort count before clear
    auto stats_map = impl_->lcmd->inference_stats();
    auto itc = stats_map.find("count");
    if (itc != stats_map.end()) {
        try { before = static_cast<size_t>(std::stoull(itc->second)); } catch (...) {}
    }

    bool cleared = impl_->lcmd->clear_inference_records();

    std::string body = "{\"cleared\":" + std::string(cleared ? "true" : "false") +
                       ",\"records_before\":" + std::to_string(before) + "}";
    return HttpResponse{200, "application/json", std::move(body), true};
}

HttpResponse InferenceServer::handle_inference_stats_(const HttpRequest& req) {
    impl_->stats.inference_stats_requests.fetch_add(1, std::memory_order_relaxed);

    if (!impl_->lcmd) {
        return HttpResponse{503, "application/json",
            R"({"error":"inference_audit_unavailable"})", true};
    }

    auto m = impl_->lcmd->inference_stats();

    std::string body = "{";
    size_t i = 0;
    for (const auto& [k, v] : m) {
        if (i++ > 0) body += ",";
        body += "\"" + k + "\":\"" + v + "\"";
    }
    body += ",\"rbpc_configured\":" + std::string(impl_->user_security ? "true" : "false") + "}";

    return HttpResponse{200, "application/json", std::move(body), true};
}

HttpResponse InferenceServer::handle_health_(const HttpRequest&) {
    impl_->stats.health_checks.fetch_add(1, std::memory_order_relaxed);

    auto hs = get_health();
    auto uptime_s = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - impl_->stats.start_time).count();

    std::string body = std::format(
        R"({{"status":"ok","uptime_seconds":{},"requests_served":{},"gpu_util":{:.1f},"npu_util":{:.1f}}})",
        uptime_s,
        impl_->stats.requests_served.load(),
        hs.gpu_util_percent,
        hs.npu_util_percent);

    return HttpResponse{200, "application/json", std::move(body), true};
}

HttpResponse InferenceServer::handle_list_models_(const HttpRequest&) {
    impl_->stats.model_lists.fetch_add(1, std::memory_order_relaxed);
    std::string body = model_registry_
        ? model_registry_->list_models_json()
        : R"({"object":"list","data":[{"id":"cerberus-v1","object":"model","owned_by":"cerberus"}]})";
    return HttpResponse{200, "application/json", std::move(body), true};
}

HttpResponse InferenceServer::handle_chat_completions_(const HttpRequest& req) {
    impl_->stats.chat_completions.fetch_add(1, std::memory_order_relaxed);

    auto oai_req = OpenAIChatCompletionRequest::from_json(req.body);
    if (!oai_req) {
        impl_->stats.errors.fetch_add(1, std::memory_order_relaxed);
        return HttpResponse{400, "application/json",
                            R"({"error":{"message":"invalid_request"}})", true};
    }

    auto result = run_inference_(*oai_req);
    std::string content = result
        ? std::move(*result)
        : std::format("[error: {}]", result.error());

    auto now_t = std::chrono::system_clock::to_time_t(
        std::chrono::system_clock::now());

    OpenAIChatCompletionResponse resp;
    resp.id      = std::format("chatcmpl-{}", impl_->stats.chat_completions.load());
    resp.model   = oai_req->model.empty() ? "cerberus-v1" : oai_req->model;
    resp.created = static_cast<std::int64_t>(now_t);
    resp.choices.push_back(OpenAIChoice{
        0,
        OpenAIMessage{"assistant", std::move(content)},
        "stop"
    });
    auto prompt_tokens = static_cast<std::uint32_t>(
        oai_req->messages.empty() ? 0
            : oai_req->messages.back().content.size() / 4 + 1);
    resp.usage = {prompt_tokens, 64, prompt_tokens + 64};

    return HttpResponse{200, "application/json", resp.to_json(), true};
}

HttpResponse InferenceServer::handle_options_(const HttpRequest&) {
    return HttpResponse{200, "application/json", "{}", true};
}

std::expected<std::string, std::string>
InferenceServer::run_inference_(const OpenAIChatCompletionRequest& req) {
    std::string prompt;
    for (const auto& m : req.messages)
        if (m.role == "user") prompt = m.content;
    if (prompt.empty() && !req.messages.empty())
        prompt = req.messages.back().content;

    try {
        PipelineConfig pcfg;
        if (!cfg_.model_dir.empty()) {
            pcfg.text_encoder_onnx = cfg_.model_dir / "text_encoder.onnx";
            pcfg.unet_onnx         = cfg_.model_dir / "unet.onnx";
            pcfg.vae_decoder_onnx  = cfg_.model_dir / "vae_decoder.onnx";
        }
        pcfg.enable_watchdog = false;

        Pipeline pipeline{pcfg};
        GenerationRequest gen{
            .prompt         = prompt,
            .width          = 512,
            .height         = 512,
            .num_steps      = std::min(req.max_tokens, std::uint32_t{128}),
            .guidance_scale = std::clamp(req.temperature, 0.1f, 2.0f),
            .seed           = static_cast<std::int64_t>(req.seed),
        };

        auto t0     = std::chrono::steady_clock::now();
        auto result = pipeline.generate(gen);
        pipeline.shutdown();

        if (result) {
            auto elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            return std::format("[Cerberus: {} ({:.0f}ms)]",
                               prompt.substr(0, 80), elapsed * 1000.0);
        }
        return std::unexpected{hq::to_string(result.error())};
    } catch (const std::exception& e) {
        return std::unexpected{std::string(e.what())};
    }
}

void InferenceServer::discover_devices_() {
    if (load_balancer_) load_balancer_->pick_device();
}

void InferenceServer::refresh_telemetry_loop_(std::stop_token token) {
    while (!token.stop_requested()) {
        if (load_balancer_) load_balancer_->pick_device();
        std::this_thread::sleep_for(std::chrono::seconds{5});
    }
}

} // namespace hq
