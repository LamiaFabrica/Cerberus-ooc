#pragma once
/// @file cerberus_psiforcedb_extension.hpp
/// @copyright Copyright (c) 2026 D Hargreaves. All rights reserved.
///
/// CerberusExtension — PsiForceDB MultiModelExtension adapter.
///
/// This header is a STANDALONE, layout-compatible replica of PsiForceDB's
/// extension_interface.hpp.  It contains NO external dependencies
/// (<nlohmann/json.hpp>, <lfssl/crypto/random.hpp>, etc.) so it compiles
/// inside the Cerberus tree while remaining binary-compatible with the
/// real PsiForceDB coordinator.
///
/// @version 1.0.0

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <functional>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace hq::cerberus::psiforcedb {

// ==========================================================================
// Data structures — layout-identical to PsiForceDB::MultiModel
// ==========================================================================

struct ExtensionConfig {
    std::string extension_name;
    std::string extension_type;  // "MOD", "EXT", "PLG", "COM"
    std::string extension_path;
    std::map<std::string, std::string> parameters;
    bool enabled = true;
    std::vector<std::string> dependencies;
};

struct ExtensionMetadata {
    std::string name;
    std::string version;
    std::string description;
    std::string author;
    std::string model_type;               // "relational", "document", "graph", "vector", "inference"
    std::vector<std::string> supported_queries;
    std::vector<std::string> supported_tables;
    std::chrono::system_clock::time_point loaded_at;
};

struct UserContext {
    std::string user_id;
    std::string username;
    std::vector<std::string> roles;
    bool is_admin = false;
    bool is_system = false;
};

class TransactionContext {
public:
    std::string transaction_id;
    std::chrono::system_clock::time_point started_at;
    std::vector<std::string> modified_models;
    bool is_committed = false;
    bool is_rolled_back = false;

    TransactionContext() {
        transaction_id = generate_transaction_id();
        started_at = std::chrono::system_clock::now();
    }

private:
    /// Stand-in for PsiForceDB::Security::AuditLoggedSecureRandom —
    /// layout-compatible ID format, entropy source swapped for portability.
    static std::string generate_transaction_id() {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count();
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<std::uint32_t> dist(0, UINT32_MAX);
        uint32_t random_value = dist(gen);
        return "txn_" + std::to_string(time) + "_" + std::to_string(random_value);
    }
};

struct Query {
    std::string query_string;
    std::string query_type;               // "SELECT", "INSERT", "UPDATE", "DELETE", "GRAPH", "VECTOR", "INFERENCE"
    std::map<std::string, std::string> parameters;
    std::vector<std::string> target_models;
    bool in_transaction = false;
    TransactionContext* transaction = nullptr;
    UserContext user_context;
};

struct QueryResult {
    bool success = false;
    std::string error_message;
    std::vector<std::map<std::string, std::string>> rows;
    std::size_t row_count = 0;
    double execution_time_ms = 0.0;
    std::map<std::string, std::string> metadata;
};

// ==========================================================================
// extension_detail — helper utilities (identical semantics to PsiForceDB)
// ==========================================================================

namespace extension_detail {

inline std::string trim(std::string value) {
    auto not_space = [](unsigned char ch) { return std::isspace(ch) == 0; };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

inline std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

inline bool parse_bool(const std::string& value, bool fallback) {
    const std::string normalized = lowercase(trim(value));
    if (normalized == "1" || normalized == "true" || normalized == "yes" ||
        normalized == "on" || normalized == "enabled") {
        return true;
    }
    if (normalized == "0" || normalized == "false" || normalized == "no" ||
        normalized == "off" || normalized == "disabled") {
        return false;
    }
    return fallback;
}

inline std::vector<std::string> split_csv(const std::string& value) {
    std::vector<std::string> tokens;
    std::stringstream stream(value);
    std::string token;
    while (std::getline(stream, token, ',')) {
        token = trim(token);
        if (!token.empty()) tokens.push_back(token);
    }
    return tokens;
}

inline std::string join_csv(const std::vector<std::string>& values) {
    std::ostringstream joined;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) joined << ',';
        joined << values[i];
    }
    return joined.str();
}

inline std::uint64_t stable_fingerprint(const std::string& value) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (unsigned char ch : value) {
        hash ^= static_cast<std::uint64_t>(ch);
        hash *= 1099511628211ULL;
    }
    return hash;
}

inline std::string parameter_or(const std::map<std::string, std::string>& parameters,
                                const std::string& key,
                                const std::string& fallback) {
    const auto found = parameters.find(key);
    if (found != parameters.end()) {
        const std::string candidate = trim(found->second);
        if (!candidate.empty()) return candidate;
    }
    return fallback;
}

inline std::string model_from_type(const ExtensionConfig& config) {
    std::string model = lowercase(parameter_or(config.parameters, "model_type", std::string()));
    if (model.empty()) {
        const std::string type = lowercase(trim(config.extension_type));
        if (type == "mod") model = "native";
        else if (type == "ext") model = "extension";
        else if (type == "plg") model = "plugin";
        else if (type == "com") model = "composite";
        else model = "generic";
    }
    return model;
}

inline bool query_has_balanced_quotes(const std::string& query) {
    bool single = false;
    bool dbl = false;
    bool escaped = false;
    for (char ch : query) {
        if (escaped) { escaped = false; continue; }
        if (ch == '\\') { escaped = true; continue; }
        if (ch == '\'' && !dbl) single = !single;
        else if (ch == '"' && !single) dbl = !dbl;
    }
    return !single && !dbl && !escaped;
}

} // namespace extension_detail

// ==========================================================================
// MultiModelExtension — base class (layout + vtable compatible)
// ==========================================================================

class MultiModelExtension {
public:
    MultiModelExtension();
    virtual ~MultiModelExtension() noexcept;

    virtual bool initialize(const ExtensionConfig& config);
    virtual bool load();
    virtual bool unload();
    virtual ExtensionMetadata getMetadata() const;
    virtual std::string getModelType() const;
    virtual QueryResult executeQuery(const Query& query);
    virtual bool supportsTransaction() const;
    virtual std::unique_ptr<TransactionContext> beginTransaction();
    virtual bool commitTransaction(TransactionContext* transaction);
    virtual bool rollbackTransaction(TransactionContext* transaction);
    virtual bool isHealthy() const;
    virtual std::map<std::string, std::string> getStatistics() const;
    virtual bool validateQuery(const std::string& query_string) const;

protected:
    void setMetadata(ExtensionMetadata metadata);
    void recordExtensionError(std::string error_message);

    mutable std::mutex base_mutex_{};
    ExtensionConfig config_{};
    ExtensionMetadata metadata_{};
    bool initialized_ = false;
    bool loaded_ = false;
    bool transaction_support_ = false;
    std::size_t query_count_ = 0;
    mutable std::size_t validation_failures_ = 0;
    std::size_t error_count_ = 0;
    std::chrono::system_clock::time_point initialized_at_{};
    std::chrono::system_clock::time_point last_activity_{};
    std::string last_error_{};
    std::set<std::string> active_transactions_{};
};

// --------------------------------------------------------------------------
// Inline base implementations (identical to PsiForceDB source of truth)
// --------------------------------------------------------------------------

inline MultiModelExtension::MultiModelExtension()
    : initialized_at_(std::chrono::system_clock::now()),
      last_activity_(initialized_at_)
{
    metadata_.name = "PsiForceDB.GenericExtension";
    metadata_.version = "1.0.0";
    metadata_.description = "PsiForceDB generic multi-model extension contract";
    metadata_.author = "PsiForceDB";
    metadata_.model_type = "generic";
    metadata_.supported_queries = {"PING", "DESCRIBE", "HEALTH", "STATS", "VALIDATE"};
    metadata_.loaded_at = initialized_at_;
}

inline MultiModelExtension::~MultiModelExtension() noexcept {
    std::lock_guard<std::mutex> lock(base_mutex_);
    loaded_ = loaded_ && active_transactions_.empty();
    active_transactions_.clear();
}

inline bool MultiModelExtension::initialize(const ExtensionConfig& config) {
    ExtensionConfig normalized = config;
    normalized.extension_name = extension_detail::trim(normalized.extension_name);
    normalized.extension_type = extension_detail::lowercase(extension_detail::trim(normalized.extension_type));
    normalized.extension_path = extension_detail::trim(normalized.extension_path);

    const bool has_name = !normalized.extension_name.empty();
    const bool has_type = normalized.extension_type == "mod" ||
                          normalized.extension_type == "ext" ||
                          normalized.extension_type == "plg" ||
                          normalized.extension_type == "com";
    const bool path_required = normalized.extension_type == "ext" ||
                                 normalized.extension_type == "plg" ||
                                 normalized.extension_type == "com";
    const bool has_path = !path_required || !normalized.extension_path.empty();
    const bool enabled = normalized.enabled;
    const bool ready = has_name && has_type && has_path && enabled;
    const auto now = std::chrono::system_clock::now();

    ExtensionMetadata metadata;
    metadata.name = has_name ? normalized.extension_name : "PsiForceDB.UnnamedExtension";
    metadata.version = extension_detail::parameter_or(normalized.parameters, "version", "1.0.0");
    metadata.description = extension_detail::parameter_or(normalized.parameters, "description",
                                                            "PsiForceDB multi-model extension");
    metadata.author = extension_detail::parameter_or(normalized.parameters, "author", "PsiForceDB");
    metadata.model_type = extension_detail::model_from_type(normalized);
    metadata.loaded_at = now;
    metadata.supported_queries = extension_detail::split_csv(
        extension_detail::parameter_or(normalized.parameters, "supported_queries",
                                       "PING,DESCRIBE,HEALTH,STATS,VALIDATE"));

    {
        std::lock_guard<std::mutex> lock(base_mutex_);
        config_ = std::move(normalized);
        metadata_ = std::move(metadata);
        initialized_ = ready;
        loaded_ = false;
        transaction_support_ = extension_detail::parse_bool(
            extension_detail::parameter_or(config_.parameters, "transactions",
                                           config_.extension_type == "com" ? "true" : "false"),
            config_.extension_type == "com");
        query_count_ = 0;
        validation_failures_ = 0;
        error_count_ = ready ? error_count_ : error_count_ + 1;
        initialized_at_ = now;
        last_activity_ = now;
        last_error_ = ready ? std::string() : "Extension configuration failed validation";
        active_transactions_.clear();
    }
    return ready;
}

inline bool MultiModelExtension::load() {
    bool loaded = false;
    const auto now = std::chrono::system_clock::now();
    {
        std::lock_guard<std::mutex> lock(base_mutex_);
        const bool dependency_names_valid = std::all_of(
            config_.dependencies.begin(), config_.dependencies.end(),
            [](const std::string& dependency) { return !extension_detail::trim(dependency).empty(); });
        loaded = initialized_ && config_.enabled && dependency_names_valid;
        loaded_ = loaded;
        metadata_.loaded_at = now;
        last_activity_ = now;
        if (!loaded) {
            ++error_count_;
            last_error_ = "Extension load rejected by lifecycle, enabled flag, or dependency names";
        }
    }
    return loaded;
}

inline bool MultiModelExtension::unload() {
    bool unloaded = false;
    {
        std::lock_guard<std::mutex> lock(base_mutex_);
        const bool no_active_transactions = active_transactions_.empty();
        unloaded = initialized_ && no_active_transactions;
        if (unloaded) {
            loaded_ = false;
            last_activity_ = std::chrono::system_clock::now();
        } else {
            ++error_count_;
            last_error_ = "Extension unload rejected while transactions are active or before initialization";
        }
    }
    return unloaded;
}

inline ExtensionMetadata MultiModelExtension::getMetadata() const {
    std::lock_guard<std::mutex> lock(base_mutex_);
    ExtensionMetadata snapshot = metadata_;
    if (snapshot.supported_queries.empty()) {
        snapshot.supported_queries = {"PING", "DESCRIBE", "HEALTH", "STATS", "VALIDATE"};
    }
    return snapshot;
}

inline std::string MultiModelExtension::getModelType() const {
    std::lock_guard<std::mutex> lock(base_mutex_);
    std::string model = extension_detail::trim(metadata_.model_type);
    if (model.empty()) model = extension_detail::model_from_type(config_);
    return model;
}

inline QueryResult MultiModelExtension::executeQuery(const Query& query) {
    const auto started = std::chrono::steady_clock::now();
    QueryResult result;
    result.metadata["extension"] = getMetadata().name;
    result.metadata["model_type"] = getModelType();
    result.metadata["query_type"] = extension_detail::trim(query.query_type);
    result.metadata["fingerprint"] = std::to_string(extension_detail::stable_fingerprint(query.query_string));

    const bool query_valid = validateQuery(query.query_string);
    bool accepted = query_valid;
    {
        std::lock_guard<std::mutex> lock(base_mutex_);
        accepted = accepted && initialized_ && loaded_;
        ++query_count_;
        last_activity_ = std::chrono::system_clock::now();
        if (!accepted) {
            ++error_count_;
            last_error_ = query_valid ? "Extension is not loaded" : "Query validation failed";
        }
    }

    result.success = accepted;
    if (accepted) {
        std::map<std::string, std::string> row;
        row["extension"] = result.metadata["extension"];
        row["model_type"] = result.metadata["model_type"];
        row["query_type"] = result.metadata["query_type"].empty() ? "GENERIC" : result.metadata["query_type"];
        row["query_fingerprint"] = result.metadata["fingerprint"];
        row["target_models"] = extension_detail::join_csv(query.target_models);
        row["user"] = query.user_context.username.empty() ? query.user_context.user_id : query.user_context.username;
        result.rows.push_back(row);
        result.row_count = result.rows.size();
        result.metadata["contract_execution"] = "accepted";
    } else {
        result.error_message = query_valid ? "Multi-model extension is not loaded"
                                           : "Query failed multi-model extension validation";
        result.row_count = result.rows.size();
        result.metadata["contract_execution"] = "rejected";
    }

    const auto finished = std::chrono::steady_clock::now();
    result.execution_time_ms = std::chrono::duration<double, std::milli>(finished - started).count();
    return result;
}

inline bool MultiModelExtension::supportsTransaction() const {
    std::lock_guard<std::mutex> lock(base_mutex_);
    return transaction_support_ && initialized_;
}

inline std::unique_ptr<TransactionContext> MultiModelExtension::beginTransaction() {
    std::unique_ptr<TransactionContext> transaction{};
    const bool supported = supportsTransaction();
    if (supported) {
        auto candidate = std::make_unique<TransactionContext>();
        candidate->modified_models.push_back(getModelType());
        {
            std::lock_guard<std::mutex> lock(base_mutex_);
            active_transactions_.insert(candidate->transaction_id);
            last_activity_ = std::chrono::system_clock::now();
        }
        transaction = std::move(candidate);
    } else {
        recordExtensionError("Transaction begin rejected because transaction boundaries are disabled");
    }
    return transaction;
}

inline bool MultiModelExtension::commitTransaction(TransactionContext* transaction) {
    bool committed = false;
    if (transaction) {
        std::lock_guard<std::mutex> lock(base_mutex_);
        const auto found = active_transactions_.find(transaction->transaction_id);
        const bool known = found != active_transactions_.end();
        const bool open = !transaction->is_committed && !transaction->is_rolled_back;
        committed = transaction_support_ && initialized_ && known && open;
        if (committed) {
            transaction->is_committed = true;
            active_transactions_.erase(found);
            last_activity_ = std::chrono::system_clock::now();
        } else {
            ++error_count_;
            last_error_ = "Transaction commit rejected by state validation";
        }
    } else {
        recordExtensionError("Transaction commit rejected because the context was missing");
    }
    return committed;
}

inline bool MultiModelExtension::rollbackTransaction(TransactionContext* transaction) {
    bool rolled_back = false;
    if (transaction) {
        std::lock_guard<std::mutex> lock(base_mutex_);
        const auto found = active_transactions_.find(transaction->transaction_id);
        const bool known = found != active_transactions_.end();
        const bool open = !transaction->is_committed && !transaction->is_rolled_back;
        rolled_back = transaction_support_ && initialized_ && known && open;
        if (rolled_back) {
            transaction->is_rolled_back = true;
            active_transactions_.erase(found);
            last_activity_ = std::chrono::system_clock::now();
        } else {
            ++error_count_;
            last_error_ = "Transaction rollback rejected by state validation";
        }
    } else {
        recordExtensionError("Transaction rollback rejected because the context was missing");
    }
    return rolled_back;
}

inline bool MultiModelExtension::isHealthy() const {
    std::lock_guard<std::mutex> lock(base_mutex_);
    const bool healthy = initialized_ && (loaded_ || !config_.enabled || !metadata_.name.empty()) &&
                         error_count_ == 0;
    return healthy;
}

inline std::map<std::string, std::string> MultiModelExtension::getStatistics() const {
    std::lock_guard<std::mutex> lock(base_mutex_);
    std::map<std::string, std::string> stats;
    stats["name"] = metadata_.name;
    stats["model_type"] = metadata_.model_type;
    stats["initialized"] = initialized_ ? "true" : "false";
    stats["loaded"] = loaded_ ? "true" : "false";
    stats["transactions_supported"] = transaction_support_ ? "true" : "false";
    stats["active_transactions"] = std::to_string(active_transactions_.size());
    stats["query_count"] = std::to_string(query_count_);
    stats["validation_failures"] = std::to_string(validation_failures_);
    stats["error_count"] = std::to_string(error_count_);
    stats["last_error"] = last_error_;
    stats["supported_queries"] = extension_detail::join_csv(metadata_.supported_queries);
    return stats;
}

inline bool MultiModelExtension::validateQuery(const std::string& query_string) const {
    const std::string query = extension_detail::trim(query_string);
    bool valid = !query.empty();
    valid = valid && query.size() <= 1024U * 1024U;
    valid = valid && extension_detail::query_has_balanced_quotes(query);
    for (unsigned char ch : query) {
        const bool permitted = std::isprint(ch) != 0 || std::isspace(ch) != 0;
        valid = valid && permitted;
    }
    {
        std::lock_guard<std::mutex> lock(base_mutex_);
        if (!valid) ++validation_failures_;
    }
    return valid;
}

inline void MultiModelExtension::setMetadata(ExtensionMetadata metadata) {
    std::lock_guard<std::mutex> lock(base_mutex_);
    if (metadata.name.empty()) metadata.name = metadata_.name;
    if (metadata.version.empty()) metadata.version = metadata_.version;
    if (metadata.model_type.empty()) metadata.model_type = metadata_.model_type;
    if (metadata.supported_queries.empty()) metadata.supported_queries = metadata_.supported_queries;
    metadata_ = std::move(metadata);
    last_activity_ = std::chrono::system_clock::now();
}

inline void MultiModelExtension::recordExtensionError(std::string error_message) {
    std::lock_guard<std::mutex> lock(base_mutex_);
    ++error_count_;
    last_error_ = std::move(error_message);
    last_activity_ = std::chrono::system_clock::now();
}

using ExtensionFactory = std::function<std::unique_ptr<MultiModelExtension>()>;

// ==========================================================================
// CerberusExtension — inference engine plugin
// ==========================================================================

class CerberusExtension : public MultiModelExtension {
public:
    CerberusExtension();
    ~CerberusExtension() override;

    bool initialize(const ExtensionConfig& config) override;
    bool load() override;
    bool unload() override;
    ExtensionMetadata getMetadata() const override;
    std::string getModelType() const override { return "inference"; }
    QueryResult executeQuery(const Query& query) override;
    bool supportsTransaction() const override { return false; }
    bool isHealthy() const override;
    std::map<std::string, std::string> getStatistics() const override;
    bool validateQuery(const std::string& query_string) const override;

private:
    struct RuntimeImpl;
    std::unique_ptr<RuntimeImpl> runtime_;

    QueryResult handle_compile_(const Query& query);
    QueryResult handle_run_(const Query& query);
    QueryResult handle_status_(const Query& query);
    QueryResult handle_gguf_(const Query& query);
    QueryResult handle_telemetry_(const Query& query);
};

} // namespace hq::cerberus::psiforcedb

// ==========================================================================
// C factory for PsiForceDB dynamic loading
// ==========================================================================

extern "C" {
    std::unique_ptr<hq::cerberus::psiforcedb::MultiModelExtension> cerberus_create_extension();
}
