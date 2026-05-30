#pragma once
/// @file cerberus_local_maintenance_db.hpp
/// @copyright Copyright (c) 2026 D Hargreaves. All rights reserved.
///
/// Cerberus Local Maintenance Database (Carbon Copy of PsiForceDB MaintenanceDB)
/// ==============================================================================
///
/// A local offline-capable replica of PsiForceDB's MaintenanceDatabase.
///
/// Purpose:
///   - Stores Cerberus configuration, licenses, and secret values locally.
///   - Operates in "local mode" when PsiForceDB is unreachable (offline/downtime).
///   - Syncs to PsiForceDB when connectivity is restored.
///   - ALL pages are encrypted at rest via AES-256-GCM (delegated to LFSSL).
///   - Hardware-bound to this device only.
///
/// Data stored:
///   - Extension licenses and revocation status
///   - Extension store entries
///   - Revenue sharing records
///   - Reviews and ratings
///   - VIP key hashes (NOT plaintext keys!)
///   - RBPC trust policy (replica of PsiForceDB authority)
///   - RBPC credential records (PIN commitments, word commitments, hardware anchors)
///   - RBPC onboarding grants (temporary enrollment codes)
///   - RBPC audit events (local login attempts, changes)
///   - RBPC state records (per-node PIN state with burn policy)
///   - User preferences and sensitive config (encrypted, redacted in exports)
///
/// BOUNDARY:
///   - This is a local REPLICA. The canonical source-of-truth lives in PsiForceDB.
///   - All encryption (AES-256-GCM, Argon2id, Kyber) is delegated to LFSSL.
///   - Redaction of sensitive fields uses PsiForceDB's SAR/FOI redaction engine
///     imported at compile time from LFSSL. No proprietary logic replicated here.
///   - The user PIN and memorable word are NEVER stored in this database.
///     Only their Argon2id-hash commitments and salts are stored.
///
/// Offline mode:
///   - When PsiForceDB is unreachable, this DB operates standalone.
///   - Changes are queued for sync (sync queue stored in its own encrypted table).
///   - On reconnect, the sync queue is replayed via the CerberusExtension
///     query interface to PsiForceDB.
///
/// © 2026 D Hargreaves | LamiaFabrica Software

#include <string>
#include <vector>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <chrono>
#include <filesystem>
#include <ctime>
#include <functional>
#include <atomic>

namespace hq::cerberus::privacy {

// ============================================================================
// RBPC State — per-node PIN security with burn policy
// ============================================================================
struct RBPCState {
    std::string node_id;
    std::string pin_hash;           // Argon2id commitment hash (NOT plaintext!)
    std::string salt;               // per-node unique salt
    int  failed_attempts{0};
    bool burned{false};             // true after 3 failed attempts
    std::time_t last_auth_timestamp{0};
    std::time_t created_at{0};

    [[nodiscard]] bool is_active() const noexcept {
        return !burned && failed_attempts < 3;
    }
};

// ============================================================================
// RBPC Trust Policy (replica — encrypted at rest, validated against PsiForceDB)
// ============================================================================
struct TrustPolicy {
    // These correspond exactly to PsiForceDB::MultiModel::default_rbpc_trust_policy()
    std::string policy_id;
    std::string deployment_model;                 // "component_optional"
    std::string credential_authority;             // "server_isolated"
    std::string central_service_role;             // "relay_directory_update_only"
    std::string medusaserv_role;                  // "open_protocol_verifier"
    std::string psiforcedb_role;                  // "sealed_psmdb_authority_when_installed"
    std::string psmdb_owner_scope;                // "d_hargreaves_only"
    std::string proprietary_boundary;              // "sealed_vault_private"
    std::string plaintext_storage;                 // "forbidden" (MANDATORY)
    std::string maintenance_encryption_layer;      // "lamia_fabrica_owned_required"
    std::string psmdb_recovery;                   // "forbidden"
    std::string psmdb_reenrollment_model;         // "rebuild_not_recover"
    std::string hash_suite;                       // "BLAKE3+SHA256"
    std::string pqc_profile;                      // "hybrid_pqc_required"
    std::string hardware_binding;                 // "required"
    std::string authority_scope;                  // "local_login_pfql_medusaserv_psiforcedb_admin"
    std::string rbpc_pin_source;                  // "system_issued"
    std::string rbpc_word_source;                 // "user_memorized"
    std::string rbpc_confirmation_window_seconds; // "30"
    std::string rbpc_failure_burn_threshold;      // "3"
    std::string temporary_onboarding;             // "sealed_expiring_consumed_local"
    std::string audit_scope;                      // "local_psmdb"
    std::string step_up_model;                    // "partial_disclosure_rbpc"
    std::string record_class;                     // auto-sealed on write
    std::string created_at;
    std::string updated_at;

    /// Returns a default trust policy that matches PsiForceDB canonical defaults.
    static TrustPolicy default_policy() noexcept;

    /// Validates that this policy keeps local authority (no central credential storage, no recovery)
    [[nodiscard]] bool keeps_local_authority() const noexcept;

    /// Serialize to flat map (for storage/encryption)
    [[nodiscard]] std::map<std::string, std::string> to_map() const noexcept;

    /// Deserialize from flat map (after decryption)
    static TrustPolicy from_map(const std::map<std::string, std::string>& m);
};

// ============================================================================
// InferenceRecord — per-generation audit trail (bootup core for Pipeline)
// ============================================================================
struct InferenceRecord {
    std::string inference_id;           // uint64_t as string
    std::string session_id;           // MetroSession token or "local"
    std::string prompt;
    std::string result_summary;       // truncated result or "ok"/"fail"
    std::string status;               // "success", "failed", "cluster_dispatched"
    std::string timestamp;            // epoch seconds
    std::string generation_time_ms;
    std::string width;
    std::string height;
    std::string num_steps;
    std::string guidance_scale;
    std::string encoder_name;
    std::string post_processor_name;
    std::string gpu_backend_name;
    std::string text_encode_used_npu;     // "true"/"false"
    std::string denoise_used_gpu;         // "true"/"false"
    std::string vae_decode_used_gpu;      // "true"/"false"
    std::string post_process_used_npu;    // "true"/"false"
    std::string unet_denoise_used_npu;    // "true"/"false"
    std::string npu_cheap_ops_percent;
    std::string recovery_attempts;
    std::string node_id;              // cluster node id or "local"

    /// Serialize to flat map (for storage/encryption)
    [[nodiscard]] std::map<std::string, std::string> to_map() const noexcept;

    /// Deserialize from flat map (after decryption)
    static InferenceRecord from_map(const std::map<std::string, std::string>& m);
};

// ============================================================================
// FileVaultRecord — encrypted file vault metadata (project/folder indexing)
// ============================================================================
struct FileVaultRecord {
    std::string file_id;            // UUID
    std::string project_name;
    std::string folder_path;        // project-relative folder
    std::string original_name;
    std::string encrypted_disk_path; // absolute path to .enc blob
    std::string size_bytes;
    std::string mime_type;
    std::string thumbnail_base64;   // inline preview for Web UI
    std::string created_at;           // epoch seconds
    std::string last_accessed;
    std::string jwt_audience;         // "cerberus-file-vault"
    std::string permission_level;     // "read", "write", "delete", "admin"
    std::string encryption_iv;        // hex-encoded 12-byte nonce
    std::string encryption_tag;       // hex-encoded 16-byte GCM tag
    std::string status;               // "active", "quarantined", "deleted"
    std::string node_id;              // "local" or cluster node

    /// Serialize to flat map (for storage/encryption)
    [[nodiscard]] std::map<std::string, std::string> to_map() const noexcept;

    /// Deserialize from flat map (after decryption)
    static FileVaultRecord from_map(const std::map<std::string, std::string>& m);
};

// ============================================================================
// Local Maintenance Database
// ============================================================================
class LocalMaintenanceDB {
public:
    LocalMaintenanceDB() = default;
    ~LocalMaintenanceDB() noexcept;

    // Move-only
    LocalMaintenanceDB(LocalMaintenanceDB&&) = default;
    LocalMaintenanceDB& operator=(LocalMaintenanceDB&&) = default;
    LocalMaintenanceDB(const LocalMaintenanceDB&) = delete;
    LocalMaintenanceDB& operator=(const LocalMaintenanceDB&) = delete;

    // ------------------------------------------------------------------------
    // Lifecycle
    // ------------------------------------------------------------------------
    /// Initialize the local maintenance database.
    /// @param db_path  Filesystem path for the local DB file.
    /// @param db_key   AES-256-GCM key (32 bytes) derived from SMDU.
    /// @return true if initialization succeeded.
    bool initialize(const std::filesystem::path& db_path,
                    const std::vector<std::uint8_t>& db_key);

    /// Close the database and clear sensitive memory.
    void shutdown();

    /// Check if the database is initialized and active.
    [[nodiscard]] bool is_initialized() const noexcept { return initialized_; }

    /// Sync queue status: how many pending records need pushing to PsiForceDB.
    [[nodiscard]] std::size_t pending_sync_count() const;

    // ------------------------------------------------------------------------
    // License management (synced to/from PsiForceDB)
    // ------------------------------------------------------------------------
    bool store_license(const std::string& extension_id,
                       const std::string& license_key_hash,
                       const std::string& user_id,
                       const std::string& license_type,
                       const std::chrono::system_clock::time_point& expires_at,
                       const std::map<std::string, std::string>& metadata = {});

    [[nodiscard]] std::map<std::string, std::string> load_license(
        const std::string& extension_id,
        const std::string& user_id) const;

    bool revoke_license(const std::string& license_key_hash, const std::string& reason);
    [[nodiscard]] bool is_license_revoked(const std::string& license_key_hash) const;

    // ------------------------------------------------------------------------
    // Extension store entries (P0 gap: missing from original Carbon Copy)
    // ------------------------------------------------------------------------
    bool store_extension_entry(const std::map<std::string, std::string>& entry);
    [[nodiscard]] std::map<std::string, std::string> load_extension_entry(
        const std::string& extension_id) const;
    [[nodiscard]] std::vector<std::map<std::string, std::string>> search_extension_entries(
        const std::string& query,
        const std::map<std::string, std::string>& filters = {}) const;

    // ------------------------------------------------------------------------
    // Revenue sharing records
    // ------------------------------------------------------------------------
    bool store_revenue_share_record(const std::map<std::string, std::string>& record);
    [[nodiscard]] std::vector<std::map<std::string, std::string>> load_revenue_share_records(
        const std::string& extension_id = "",
        const std::string& user_id = "") const;

    // ------------------------------------------------------------------------
    // Reviews
    // ------------------------------------------------------------------------
    bool store_review(const std::map<std::string, std::string>& review);
    [[nodiscard]] std::vector<std::map<std::string, std::string>> load_reviews(
        const std::string& extension_id,
        int limit = 20) const;

    // ------------------------------------------------------------------------
    // Extension stats
    // ------------------------------------------------------------------------
    bool update_extension_stats(const std::string& extension_id,
                                const std::map<std::string, std::string>& stats);
    [[nodiscard]] std::map<std::string, std::string> get_extension_stats(
        const std::string& extension_id) const;

    // ------------------------------------------------------------------------
    // VIP key management
    // ------------------------------------------------------------------------
    bool store_vip_key(const std::string& key_hash,
                        const std::string& encrypted_metadata,
                        const std::string& encrypted_key,
                        std::time_t expiration_timestamp);
    [[nodiscard]] std::map<std::string, std::string> load_vip_key(const std::string& key_hash) const;
    [[nodiscard]] bool vip_key_exists(const std::string& key_hash) const;
    bool update_vip_key_status(const std::string& key_hash,
                                const std::map<std::string, std::string>& status_data);
    [[nodiscard]] std::vector<std::string> get_all_vip_key_hashes() const;

    // ------------------------------------------------------------------------
    // RBPC Trust Policy
    // ------------------------------------------------------------------------
    bool store_trust_policy(const TrustPolicy& policy);
    [[nodiscard]] TrustPolicy load_trust_policy() const;

    // ------------------------------------------------------------------------
    // RBPC Onboarding Grants
    // ------------------------------------------------------------------------
    bool store_onboarding_grant(const std::map<std::string, std::string>& grant);
    [[nodiscard]] std::map<std::string, std::string> load_onboarding_grant(const std::string& grant_id) const;
    bool consume_onboarding_grant(const std::string& grant_id,
                                    const std::string& consumed_by,
                                    const std::string& reason);
    [[nodiscard]] std::vector<std::map<std::string, std::string>> load_onboarding_grants_for_user(
        const std::string& user_id,
        bool include_consumed = false) const;

    // ------------------------------------------------------------------------
    // RBPC Credential Records (PIN commitments, word commitments)
    // ------------------------------------------------------------------------
    bool store_credential_record(const std::map<std::string, std::string>& record);
    [[nodiscard]] std::map<std::string, std::string> load_credential_record(
        const std::string& user_id,
        const std::string& token_id) const;

    // ------------------------------------------------------------------------
    // RBPC State (per-node PIN with burn policy)
    // ------------------------------------------------------------------------
    bool save_rbpc_state(const RBPCState& state);
    [[nodiscard]] std::optional<RBPCState> load_rbpc_state(const std::string& node_id) const;
    bool increment_rbpc_failed_attempts(const std::string& node_id);
    bool set_rbpc_burned(const std::string& node_id);

    // ------------------------------------------------------------------------
    // RBPC Audit Events
    // ------------------------------------------------------------------------
    bool store_audit_event(const std::map<std::string, std::string>& event);
    [[nodiscard]] std::vector<std::map<std::string, std::string>> load_audit_events(
        const std::string& user_id = "",
        const std::string& token_id = "",
        int limit = 50) const;

    // ------------------------------------------------------------------------
    // User preferences (encrypted, redacted on export)
    // ------------------------------------------------------------------------
    bool store_preference(const std::string& key, const std::string& value);
    [[nodiscard]] std::string load_preference(const std::string& key) const;

    // ------------------------------------------------------------------------
    // Offline mode
    // ------------------------------------------------------------------------
    void set_offline_mode(bool offline) noexcept;
    [[nodiscard]] bool is_offline_mode() const noexcept;

    // ------------------------------------------------------------------------
    // Sync queue (records to replay to PsiForceDB on reconnect)
    // ------------------------------------------------------------------------
    void queue_for_sync(const std::string& table, const std::string& key,
                        const std::map<std::string, std::string>& record);

    /// Replay the sync queue until drained or max_records. Returns number
    /// of records successfully replayed.
    using SyncReplayCallback = std::function<bool(const std::string& table,
                                                   const std::string& key,
                                                   const std::map<std::string,std::string>& record)>;
    std::size_t replay_sync_queue(SyncReplayCallback callback,
                                   std::size_t max_records = 0);

    // ------------------------------------------------------------------------
    // Inference history (bootup core — query, export, clear, stats)
    // ------------------------------------------------------------------------
    bool store_inference_record(const InferenceRecord& record);
    [[nodiscard]] std::optional<InferenceRecord> load_inference_record(
        const std::string& inference_id) const;
    [[nodiscard]] std::vector<InferenceRecord> query_inference_records(
        std::time_t since, std::time_t until, std::size_t limit = 100) const;
    [[nodiscard]] bool export_inference_json(const std::filesystem::path& path) const;
    [[nodiscard]] bool export_inference_csv(const std::filesystem::path& path) const;
    [[nodiscard]] bool export_inference_markdown(const std::filesystem::path& path) const;
    bool clear_inference_records();
    [[nodiscard]] std::map<std::string, std::string> inference_stats() const;

    // MANDATORY unavailable_reason per AGENTS.md
    [[nodiscard]] static std::string unavailable_reason() noexcept;

    // ------------------------------------------------------------------------
    // Encrypted file vault (project/folder indexing + metadata)
    // ------------------------------------------------------------------------
    bool store_file_record(const FileVaultRecord& record);
    [[nodiscard]] std::optional<FileVaultRecord> load_file_record(
        const std::string& file_id) const;
    bool delete_file_record(const std::string& file_id);  // marks "deleted", not erased
    bool update_file_accessed(const std::string& file_id);
    bool update_file_status(const std::string& file_id, const std::string& status);
    [[nodiscard]] std::vector<FileVaultRecord> query_files_by_project(
        const std::string& project_name,
        const std::string& folder_path = "",
        const std::string& status_filter = "active") const;
    [[nodiscard]] std::vector<std::string> list_projects() const;
    [[nodiscard]] std::map<std::string, std::string> file_vault_stats() const;

private:
    mutable std::mutex mutex_;
    bool initialized_{false};
    std::filesystem::path db_path_;
    std::vector<std::uint8_t> db_key_;
    bool offline_mode_{false};
    mutable std::atomic<bool> dirty_{false};   // set on any write, cleared on flush

    // In-memory storage (encrypted pages would be managed by LFSSL in production)
    std::map<std::string, std::map<std::string, std::string>> licenses_;
    std::map<std::string, std::map<std::string, std::string>> extension_entries_;
    std::map<std::string, std::map<std::string, std::string>> revenue_records_;
    std::map<std::string, std::map<std::string, std::string>> reviews_;
    std::map<std::string, std::map<std::string, std::string>> extension_stats_;
    std::map<std::string, std::map<std::string, std::string>> credential_records_;
    std::map<std::string, std::map<std::string, std::string>> audit_events_;
    std::map<std::string, std::map<std::string, std::string>> rbpc_state_records_;
    std::map<std::string, std::map<std::string, std::string>> vip_keys_;
    std::map<std::string, std::map<std::string, std::string>> onboarding_grants_;
    std::map<std::string, std::string> preferences_;
    std::map<std::string, std::string> revoked_hashes_;
    std::optional<TrustPolicy> trust_policy_;
    std::map<std::string, std::map<std::string, std::string>> inference_records_;
    std::map<std::string, std::map<std::string, std::string>> file_vault_records_;

    // Sync queue
    struct SyncRecord {
        std::string table;
        std::string key;
        std::map<std::string, std::string> record;
        std::chrono::system_clock::time_point queued_at;
    };
    std::vector<SyncRecord> sync_queue_;

    void scrub_(std::vector<std::uint8_t>& buf) const noexcept;
    void mark_dirty_() const noexcept { dirty_.store(true, std::memory_order_relaxed); }

    // ------------------------------------------------------------------------
    // Disk persistence (LFSSL AES-256-GCM encrypted at rest)
    // ------------------------------------------------------------------------
    [[nodiscard]] bool flush_to_disk_() const;
    [[nodiscard]] bool load_from_disk_();

    // Serialization helpers
    void serialize_table_(const std::map<std::string, std::string>&,
                          std::vector<std::uint8_t>& out) const;
    void serialize_table_(const std::map<std::string, std::map<std::string,std::string>>&, std::vector<std::uint8_t>& out) const;
    [[nodiscard]] bool deserialize_all_(const std::vector<std::uint8_t>& in);

    // LFSSL AES-256-GCM helpers (forward-declared Windows API in .cpp)
    struct LfsslAesGcm {
        void*   lib{nullptr};
        int (*encrypt)(const std::uint8_t*, const std::uint8_t*, size_t, const std::uint8_t*, size_t, const std::uint8_t*, size_t, std::uint8_t*, size_t, size_t*){nullptr};
        int (*decrypt)(const std::uint8_t*, const std::uint8_t*, size_t, const std::uint8_t*, size_t, const std::uint8_t*, size_t, std::uint8_t*, size_t, size_t*){nullptr};
        int (*random_bytes)(std::uint8_t*, size_t){nullptr}; // LFSSL CSPRNG (GetRandom/BCryptGenRandom)
        bool init();
        bool available() const noexcept { return lib && encrypt && decrypt && random_bytes; }
    };
    static LfsslAesGcm& lfssl_();
};

} // namespace hq::cerberus::privacy
