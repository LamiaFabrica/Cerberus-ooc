/// @file cerberus_local_maintenance_db.cpp
/// @copyright Copyright (c) 2026 D Hargreaves. All rights reserved.
///
/// Local Maintenance Database — Carbon Copy of PsiForceDB MaintenanceDatabase
/// =========================================================================
///
/// This file IS the LamiaFabrica MaintenanceDB (carbon copy).
///
/// Architecture (matches PsiForceDB line-for-line where the boundary permits):
///   - All pages encrypted at rest with AES-256-GCM (delegated to LFSSL).
///   - Works in offline and online mode identically.
///   - Sync queue for deferred replay to PsiForceDB.
///   - RBPC state, credentials, audit events all stored encrypted.
///   - Trust policy: server_isolated, no plaintext, no recovery, rebuild only.
///
/// BOUNDARY: We do NOT replicate PsiForceDB proprietary glue, hardware_
/// dependent crypto, or the master SMDU secret store. We replicate the
/// interface, policy enforcement, and encryption-wrapped behavior.
///
/// © 2026 D Hargreaves | LamiaFabrica Software

#include "hq/cerberus_local_maintenance_db.hpp"
#include "hq/cerberus_psiforcedb_security.hpp"

#include <cstring>
#include <cstdint>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <filesystem>

namespace hq::cerberus::privacy {

// ============================================================================
// Sentinel / unavailable_reason
// ============================================================================

std::string LocalMaintenanceDB::unavailable_reason() noexcept {
    return "LocalMaintenanceDB is the carbon-copy replica of PsiForceDB "
           "MaintenanceDatabase. It stores ALL configuration, licenses, RBPC "
           "trust policy, credential commitments, audit events, and user "
           "preferences. All pages use LFSSL AES-256-GCM encryption. Works in "
           "offline and online modes identically. "
           "On this Windows build host, LFSSL.dll is not yet linked; operations "
           "run in sentinel mode with in-memory-only storage. Argon2id, Kyber, "
           "and page encryption are delegated to LFSSL at runtime.";
}

// ============================================================================
// Lifecycle
// ============================================================================

LocalMaintenanceDB::~LocalMaintenanceDB() noexcept {
    shutdown();
}

bool LocalMaintenanceDB::initialize(const std::filesystem::path& db_path,
                                    const std::vector<std::uint8_t>& db_key) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (initialized_) return true;
    if (db_key.size() != 32) return false;

    db_path_ = db_path;
    db_key_ = db_key;

    // Try to load persisted state from disk (encrypted page — LFSSL decrypt)
    // In sentinel mode: start empty but initialized.
    initialized_ = true;
    return true;
}

void LocalMaintenanceDB::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) return;

    // In production: flush encrypted pages to disk via LFSSL.
    // In sentinel: clear everything.

    scrub_(db_key_);
    licenses_.clear();
    credential_records_.clear();
    audit_events_.clear();
    rbpc_state_records_.clear();
    preferences_.clear();
    revoked_hashes_.clear();
    trust_policy_.reset();
    sync_queue_.clear();
    initialized_ = false;
}

std::size_t LocalMaintenanceDB::pending_sync_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return sync_queue_.size();
}

// ============================================================================
// License management
// ============================================================================

bool LocalMaintenanceDB::store_license(const std::string& extension_id,
                                       const std::string& license_key_hash,
                                       const std::string& user_id,
                                       const std::string& license_type,
                                       const std::chrono::system_clock::time_point& expires_at,
                                       const std::map<std::string, std::string>& metadata) {
    if (!initialized_ || extension_id.empty() || license_key_hash.empty()) return false;

    std::lock_guard<std::mutex> lock(mutex_);

    std::map<std::string, std::string> rec;
    rec["extension_id"]     = extension_id;
    rec["license_key_hash"] = license_key_hash;
    rec["user_id"]          = user_id;
    rec["license_type"]     = license_type;

    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(
        expires_at.time_since_epoch()).count();
    rec["expires_at"] = std::to_string(seconds);

    for (const auto& [k, v] : metadata) rec[k] = v;

    const std::string key = extension_id + ":" + user_id;
    licenses_[key] = rec;

    queue_for_sync("licenses", key, rec);
    return true;
}

std::map<std::string, std::string> LocalMaintenanceDB::load_license(
    const std::string& extension_id,
    const std::string& user_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const std::string key = extension_id + ":" + user_id;
    auto it = licenses_.find(key);
    if (it != licenses_.end()) return it->second;
    return {};
}

bool LocalMaintenanceDB::revoke_license(const std::string& license_key_hash, const std::string& reason) {
    if (!initialized_) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    revoked_hashes_[license_key_hash] = reason.empty() ? "revoked" : reason;
    return true;
}

bool LocalMaintenanceDB::is_license_revoked(const std::string& license_key_hash) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return revoked_hashes_.find(license_key_hash) != revoked_hashes_.end();
}

// ============================================================================
// Extension store entries, revenue sharing, reviews, stats, VIP keys
// ============================================================================

bool LocalMaintenanceDB::store_extension_entry(const std::map<std::string, std::string>& entry) {
    if (!initialized_ || entry.empty()) return false;
    auto it = entry.find("extension_id");
    if (it == entry.end() || it->second.empty()) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    extension_entries_[it->second] = entry;
    queue_for_sync("extension_entries", it->second, entry);
    return true;
}

std::map<std::string, std::string> LocalMaintenanceDB::load_extension_entry(
    const std::string& extension_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = extension_entries_.find(extension_id);
    return (it != extension_entries_.end()) ? it->second : std::map<std::string, std::string>{};
}

std::vector<std::map<std::string, std::string>> LocalMaintenanceDB::search_extension_entries(
    const std::string& query,
    const std::map<std::string, std::string>& filters) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::map<std::string, std::string>> out;
    for (const auto& [id, entry] : extension_entries_) {
        bool match = true;
        for (const auto& [fk, fv] : filters) {
            auto fit = entry.find(fk);
            if (fit == entry.end() || fit->second != fv) { match = false; break; }
        }
        if (!match) continue;
        if (!query.empty()) {
            bool found = false;
            for (const auto& [k, v] : entry) {
                if (k.find(query) != std::string::npos || v.find(query) != std::string::npos) {
                    found = true; break;
                }
            }
            if (!found) continue;
        }
        out.push_back(entry);
    }
    return out;
}

bool LocalMaintenanceDB::store_revenue_share_record(const std::map<std::string, std::string>& record) {
    if (!initialized_ || record.empty()) return false;
    auto it = record.find("record_id");
    std::string key = (it != record.end()) ? it->second
                     : ("rev_" + std::to_string(revenue_records_.size()));
    std::lock_guard<std::mutex> lock(mutex_);
    revenue_records_[key] = record;
    queue_for_sync("revenue_records", key, record);
    return true;
}

std::vector<std::map<std::string, std::string>> LocalMaintenanceDB::load_revenue_share_records(
    const std::string& extension_id,
    const std::string& user_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::map<std::string, std::string>> out;
    for (const auto& [key, rec] : revenue_records_) {
        bool match = true;
        if (!extension_id.empty()) {
            auto it = rec.find("extension_id");
            if (it == rec.end() || it->second != extension_id) match = false;
        }
        if (!user_id.empty()) {
            auto it = rec.find("user_id");
            if (it == rec.end() || it->second != user_id) match = false;
        }
        if (match) out.push_back(rec);
    }
    return out;
}

bool LocalMaintenanceDB::store_review(const std::map<std::string, std::string>& review) {
    if (!initialized_ || review.empty()) return false;
    auto it = review.find("review_id");
    std::string key = (it != review.end()) ? it->second
                     : ("review_" + std::to_string(reviews_.size()));
    std::lock_guard<std::mutex> lock(mutex_);
    reviews_[key] = review;
    queue_for_sync("reviews", key, review);
    return true;
}

std::vector<std::map<std::string, std::string>> LocalMaintenanceDB::load_reviews(
    const std::string& extension_id,
    int limit) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::map<std::string, std::string>> out;
    for (const auto& [key, rec] : reviews_) {
        if (!extension_id.empty()) {
            auto it = rec.find("extension_id");
            if (it == rec.end() || it->second != extension_id) continue;
        }
        out.push_back(rec);
        if (limit > 0 && static_cast<int>(out.size()) >= limit) break;
    }
    return out;
}

bool LocalMaintenanceDB::update_extension_stats(const std::string& extension_id,
                                                const std::map<std::string, std::string>& stats) {
    if (!initialized_ || extension_id.empty()) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    extension_stats_[extension_id] = stats;
    queue_for_sync("extension_stats", extension_id, stats);
    return true;
}

std::map<std::string, std::string> LocalMaintenanceDB::get_extension_stats(
    const std::string& extension_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = extension_stats_.find(extension_id);
    return (it != extension_stats_.end()) ? it->second : std::map<std::string, std::string>{};
}

// ============================================================================
// VIP key management
// ============================================================================

bool LocalMaintenanceDB::store_vip_key(const std::string& key_hash,
                                        const std::string& encrypted_metadata,
                                        const std::string& encrypted_key,
                                        std::time_t expiration_timestamp) {
    if (!initialized_ || key_hash.empty()) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    std::map<std::string, std::string> rec;
    rec["key_hash"] = key_hash;
    rec["encrypted_metadata"] = encrypted_metadata;
    rec["encrypted_key"] = encrypted_key;
    rec["expiration_timestamp"] = std::to_string(expiration_timestamp);
    rec["record_class"] = "vip_key";
    rec["plaintext_storage"] = "forbidden";
    vip_keys_[key_hash] = rec;
    return true;
}

std::map<std::string, std::string> LocalMaintenanceDB::load_vip_key(const std::string& key_hash) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = vip_keys_.find(key_hash);
    return (it != vip_keys_.end()) ? it->second : std::map<std::string, std::string>{};
}

bool LocalMaintenanceDB::vip_key_exists(const std::string& key_hash) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return vip_keys_.find(key_hash) != vip_keys_.end();
}

bool LocalMaintenanceDB::update_vip_key_status(const std::string& key_hash,
                                                  const std::map<std::string, std::string>& status_data) {
    if (!initialized_ || key_hash.empty()) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = vip_keys_.find(key_hash);
    if (it == vip_keys_.end()) return false;
    for (const auto& [k, v] : status_data) it->second[k] = v;
    return true;
}

std::vector<std::string> LocalMaintenanceDB::get_all_vip_key_hashes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> out;
    for (const auto& [k, v] : vip_keys_) out.push_back(k);
    return out;
}

// ============================================================================
// Onboarding grants (stubs — full RBPC onboarding grant flow delegated to PsiForceDB)
// ============================================================================

bool LocalMaintenanceDB::store_onboarding_grant(const std::map<std::string, std::string>& grant) {
    if (!initialized_ || grant.empty()) return false;
    auto it = grant.find("grant_id");
    std::string key = (it != grant.end()) ? it->second
                     : ("grant_" + std::to_string(onboarding_grants_.size()));
    std::lock_guard<std::mutex> lock(mutex_);
    std::map<std::string, std::string> safe = grant;
    safe["record_class"] = "rbpc_onboarding_grant";
    safe["plaintext_storage"] = "forbidden";
    if (!safe.count("consumed")) safe["consumed"] = "false";
    onboarding_grants_[key] = safe;
    return true;
}

std::map<std::string, std::string> LocalMaintenanceDB::load_onboarding_grant(const std::string& grant_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = onboarding_grants_.find(grant_id);
    return (it != onboarding_grants_.end()) ? it->second : std::map<std::string, std::string>{};
}

bool LocalMaintenanceDB::consume_onboarding_grant(const std::string& grant_id,
                                                    const std::string& consumed_by,
                                                    const std::string& reason) {
    if (!initialized_ || grant_id.empty()) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = onboarding_grants_.find(grant_id);
    if (it == onboarding_grants_.end()) return false;
    it->second["consumed"] = "true";
    it->second["consumed_by"] = consumed_by;
    it->second["consumed_reason"] = reason.empty() ? "consumed" : reason;
    auto now = std::chrono::system_clock::now();
    it->second["consumed_at"] = std::to_string(std::chrono::duration_cast<std::chrono::seconds>(
        now.time_since_epoch()).count());
    return true;
}

std::vector<std::map<std::string, std::string>> LocalMaintenanceDB::load_onboarding_grants_for_user(
    const std::string& user_id,
    bool include_consumed) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::map<std::string, std::string>> out;
    for (const auto& [id, grant] : onboarding_grants_) {
        auto uit = grant.find("user_id");
        if (uit == grant.end() || uit->second != user_id) continue;
        auto cit = grant.find("consumed");
        bool consumed = (cit != grant.end() && cit->second == "true");
        if (!include_consumed && consumed) continue;
        out.push_back(grant);
    }
    return out;
}

// ============================================================================
// Trust Policy
// ============================================================================

bool LocalMaintenanceDB::store_trust_policy(const TrustPolicy& policy) {
    if (!initialized_) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    trust_policy_ = policy;
    return true;
}

TrustPolicy LocalMaintenanceDB::load_trust_policy() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (trust_policy_.has_value()) return *trust_policy_;

    TrustPolicy tp;
    tp.policy_id = "cerberus_rbpc_trust_default";
    tp.deployment_model = "component_optional";
    tp.credential_authority = "server_isolated";
    tp.central_service_role = "relay_directory_update_only";
    tp.medusaserv_role = "open_protocol_verifier";
    tp.psiforcedb_role = "sealed_psmdb_authority_when_installed";
    tp.psmdb_owner_scope = "d_hargreaves_only";
    tp.proprietary_boundary = "sealed_vault_private";
    tp.plaintext_storage = "forbidden";
    tp.maintenance_encryption_layer = "lamia_fabrica_owned_required";
    tp.psmdb_recovery = "forbidden";
    tp.psmdb_reenrollment_model = "rebuild_not_recover";
    tp.hash_suite = "BLAKE3+SHA256";
    tp.pqc_profile = "hybrid_pqc_required";
    tp.hardware_binding = "required";
    tp.authority_scope = "local_login_pfql_medusaserv_psiforcedb_admin";
    tp.rbpc_pin_source = "system_issued";
    tp.rbpc_word_source = "user_memorized";
    tp.rbpc_confirmation_window_seconds = "30";
    tp.rbpc_failure_burn_threshold = "3";
    tp.temporary_onboarding = "sealed_expiring_consumed_local";
    tp.audit_scope = "local_psmdb";
    tp.step_up_model = "partial_disclosure_rbpc";
    tp.record_class = "rbpc_trust_policy";
    return tp;
}

// ============================================================================
// Credential Records
// ============================================================================

bool LocalMaintenanceDB::store_credential_record(const std::map<std::string, std::string>& record) {
    if (!initialized_ || record.empty()) return false;
    std::lock_guard<std::mutex> lock(mutex_);

    auto user_it = record.find("user_id");
    auto token_it = record.find("token_id");
    if (user_it == record.end() || token_it == record.end()) return false;

    const std::string key = user_it->second + ":" + token_it->second;
    std::map<std::string, std::string> safe = record;
    safe["plaintext_storage"] = "forbidden";
    safe["credential_authority"] = "server_isolated";
    safe["proprietary_boundary"] = "sealed_vault_private";
    safe["record_class"] = "rbpc_credential_commitments";
    safe["hash_suite"] = safe.count("hash_suite") ? safe["hash_suite"] : "BLAKE3+SHA256";
    safe["pqc_profile"] = safe.count("pqc_profile") ? safe["pqc_profile"] : "hybrid_pqc_required";
    safe["hardware_binding"] = "required";
    safe["step_up_model"] = "partial_disclosure_rbpc";

    auto now = std::chrono::system_clock::now();
    auto sec = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
    if (!safe.count("created_at")) safe["created_at"] = std::to_string(sec);
    safe["updated_at"] = std::to_string(sec);

    credential_records_[key] = safe;
    queue_for_sync("credential_records", key, safe);
    return true;
}

std::map<std::string, std::string> LocalMaintenanceDB::load_credential_record(
    const std::string& user_id,
    const std::string& token_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const std::string key = user_id + ":" + token_id;
    auto it = credential_records_.find(key);
    if (it != credential_records_.end()) return it->second;
    return {};
}

// ============================================================================
// RBPC State
// ============================================================================

bool LocalMaintenanceDB::save_rbpc_state(const RBPCState& state) {
    if (!initialized_ || state.node_id.empty() || state.pin_hash.empty()) return false;
    std::lock_guard<std::mutex> lock(mutex_);

    std::map<std::string, std::string> rec;
    rec["node_id"]    = state.node_id;
    rec["pin_hash"]   = state.pin_hash;
    rec["salt"]       = state.salt;
    rec["failed_attempts"] = std::to_string(state.failed_attempts);
    rec["burned"]     = state.burned ? "true" : "false";
    rec["last_auth_timestamp"] = std::to_string(state.last_auth_timestamp);
    rec["created_at"] = std::to_string(state.created_at);
    rec["plaintext_storage"] = "forbidden";
    rec["record_class"] = "rbpc_state";

    rbpc_state_records_[state.node_id] = rec;
    return true;
}

std::optional<RBPCState> LocalMaintenanceDB::load_rbpc_state(const std::string& node_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = rbpc_state_records_.find(node_id);
    if (it == rbpc_state_records_.end()) return std::nullopt;

    const auto& rec = it->second;
    RBPCState st;
    st.node_id = node_id;

    auto get = [&rec](const std::string& k) {
        auto i = rec.find(k);
        return i != rec.end() ? i->second : std::string{};
    };

    st.pin_hash = get("pin_hash");
    st.salt = get("salt");
    try { st.failed_attempts = std::stoi(get("failed_attempts")); } catch (...) { st.failed_attempts = 0; }
    st.burned = get("burned") == "true";
    try { st.last_auth_timestamp = static_cast<std::time_t>(std::stoll(get("last_auth_timestamp"))); } catch (...) { st.last_auth_timestamp = 0; }
    try { st.created_at = static_cast<std::time_t>(std::stoll(get("created_at"))); } catch (...) { st.created_at = 0; }
    return st;
}

bool LocalMaintenanceDB::increment_rbpc_failed_attempts(const std::string& node_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& rec = rbpc_state_records_[node_id];
    if (rec.empty()) {
        rec["node_id"] = node_id;
        rec["failed_attempts"] = "1";
        rec["burned"] = "false";
        rec["plaintext_storage"] = "forbidden";
        rec["record_class"] = "rbpc_state";
    } else {
        int attempts = 0;
        try { attempts = std::stoi(rec["failed_attempts"]); } catch (...) {}
        rec["failed_attempts"] = std::to_string(attempts + 1);

        int threshold = 3;
        try { threshold = std::stoi(rec["burn_threshold"]); } catch (...) {}
        if (attempts + 1 >= threshold) {
            rec["burned"] = "true";
        }
    }
    return true;
}

bool LocalMaintenanceDB::set_rbpc_burned(const std::string& node_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& rec = rbpc_state_records_[node_id];
    if (rec.empty()) {
        rec["node_id"] = node_id;
        rec["failed_attempts"] = "3";
    }
    rec["burned"] = "true";
    rec["plaintext_storage"] = "forbidden";
    rec["record_class"] = "rbpc_state";
    return true;
}

// ============================================================================
// Audit Events
// ============================================================================

bool LocalMaintenanceDB::store_audit_event(const std::map<std::string, std::string>& event) {
    if (!initialized_) return false;
    std::lock_guard<std::mutex> lock(mutex_);

    if (!event.count("user_id") || !event.count("event_type")) return false;

    std::string event_id;
    auto it = event.find("event_id");
    if (it == event.end() || it->second.empty()) {
        event_id = "audit_" + std::to_string(audit_events_.size());
    } else {
        event_id = it->second;
    }

    std::map<std::string, std::string> rec = event;
    rec["event_id"] = event_id;
    rec["record_class"] = "rbpc_audit_event";
    rec["audit_scope"] = "local_psmdb";
    rec["plaintext_storage"] = "forbidden";
    rec["credential_material_present"] = "false";

    auto now = std::chrono::system_clock::now();
    auto sec = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
    if (!rec.count("timestamp")) rec["timestamp"] = std::to_string(sec);

    audit_events_[event_id] = rec;
    queue_for_sync("audit_events", event_id, rec);
    return true;
}

std::vector<std::map<std::string, std::string>> LocalMaintenanceDB::load_audit_events(
    const std::string& user_id,
    const std::string& token_id,
    int limit) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::map<std::string, std::string>> out;
    for (const auto& [id, ev] : audit_events_) {
        bool match = true;
        if (!user_id.empty()) {
            auto it = ev.find("user_id");
            match = match && (it != ev.end() && it->second == user_id);
        }
        if (!token_id.empty()) {
            auto it = ev.find("token_id");
            match = match && (it != ev.end() && it->second == token_id);
        }
        if (match) {
            out.push_back(ev);
            if (limit > 0 && static_cast<int>(out.size()) >= limit) break;
        }
    }
    return out;
}

// ============================================================================
// Preferences
// ============================================================================

bool LocalMaintenanceDB::store_preference(const std::string& key, const std::string& value) {
    if (!initialized_ || key.empty()) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    preferences_[key] = value;
    return true;
}

std::string LocalMaintenanceDB::load_preference(const std::string& key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = preferences_.find(key);
    return it != preferences_.end() ? it->second : std::string{};
}

// ============================================================================
// Sync queue
// ============================================================================

void LocalMaintenanceDB::queue_for_sync(const std::string& table,
                                          const std::string& key,
                                          const std::map<std::string, std::string>& record) {
    SyncRecord sr;
    sr.table = table;
    sr.key = key;
    sr.record = record;
    sr.queued_at = std::chrono::system_clock::now();
    sync_queue_.push_back(std::move(sr));
}

void LocalMaintenanceDB::scrub_(std::vector<std::uint8_t>& buf) const noexcept {
    if (buf.empty()) return;
    volatile std::uint8_t* p = buf.data();
    for (std::size_t i = 0; i < buf.size(); ++i) p[i] = 0;
    buf.clear();
}

} // namespace hq::cerberus::privacy
