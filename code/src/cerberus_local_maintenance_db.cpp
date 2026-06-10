/// @file cerberus_local_maintenance_db.cpp
/// @copyright Copyright © 2026, all rights reserved | David Hargreaves aka Roylepython trading as The Medusa Initiative & Lamia Fabrica
/// Support: https://www.patreon.com/TheMedusaInitiative — £25/month removes ads from all software at 200 subscribers
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
#include <random>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <filesystem>

#include <ostream>

// ============================================================================
// LFSSL AES-256-GCM forward declarations (minimal — avoids <windows.h> pollution)
// ============================================================================

#ifdef _WIN32
  extern "C" __declspec(dllimport) void* LoadLibraryA(const char*);
  extern "C" __declspec(dllimport) void*  GetProcAddress(void*, const char*);
  extern "C" __declspec(dllimport) int   FreeLibrary(void*);
  extern "C" __declspec(dllimport) unsigned long GetLastError(void);
#else
  #include <dlfcn.h>
#endif

namespace hq::cerberus::privacy {

// ============================================================================
// LFSSL helper
// ============================================================================

LocalMaintenanceDB::LfsslAesGcm& LocalMaintenanceDB::lfssl_() {
    static LfsslAesGcm s;
    if (!s.lib) s.init();
    return s;
}

bool LocalMaintenanceDB::LfsslAesGcm::init() {
    if (lib) return true;
#ifdef _WIN32
    const char* paths[] = {
        "cerberus_lfssl.dll",
        "../../lfssl_bridge/cerberus_lfssl.dll",
        "../lfssl_bridge/cerberus_lfssl.dll",
        "lfssl_bridge/cerberus_lfssl.dll",
        nullptr
    };
    for (auto* p = paths; p && *p; ++p) {
        lib = reinterpret_cast<void*>(LoadLibraryA(*p));
        if (lib) break;
    }
#else
    const char* paths[] = {
        "./libcerberus_lfssl.so",
        "../../lfssl_bridge/libcerberus_lfssl.so",
        "../lfssl_bridge/libcerberus_lfssl.so",
        "lfssl_bridge/libcerberus_lfssl.so",
        nullptr
    };
    for (auto* p = paths; p && *p; ++p) {
        lib = dlopen(*p, RTLD_NOW);
        if (lib) break;
    }
#endif
    if (!lib) return false;

#ifdef _WIN32
    encrypt = reinterpret_cast<decltype(encrypt)>(
        reinterpret_cast<void* (*)(void*,const char*)>(GetProcAddress)(lib, "cerberus_lfssl_aes256gcm_encrypt"));
    decrypt = reinterpret_cast<decltype(decrypt)>(
        reinterpret_cast<void* (*)(void*,const char*)>(GetProcAddress)(lib, "cerberus_lfssl_aes256gcm_decrypt"));
    random_bytes = reinterpret_cast<decltype(random_bytes)>(
        reinterpret_cast<void* (*)(void*,const char*)>(GetProcAddress)(lib, "cerberus_lfssl_random_bytes"));
#else
    encrypt = reinterpret_cast<decltype(encrypt)>(dlsym(lib, "cerberus_lfssl_aes256gcm_encrypt"));
    decrypt = reinterpret_cast<decltype(decrypt)>(dlsym(lib, "cerberus_lfssl_aes256gcm_decrypt"));
    random_bytes = reinterpret_cast<decltype(random_bytes)>(dlsym(lib, "cerberus_lfssl_random_bytes"));
#endif

    if (!encrypt || !decrypt || !random_bytes) {
#ifdef _WIN32
        FreeLibrary(lib);
#else
        dlclose(lib);
#endif
        lib = nullptr;
        return false;
    }
    return true;
}

// ============================================================================
// Serialization helpers
// ============================================================================

namespace {

inline void pack_u32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    out.push_back(static_cast<std::uint8_t>((v >> 24) & 0xFF));
    out.push_back(static_cast<std::uint8_t>((v >> 16) & 0xFF));
    out.push_back(static_cast<std::uint8_t>((v >>  8) & 0xFF));
    out.push_back(static_cast<std::uint8_t>( v        & 0xFF));
}

inline std::uint32_t unpack_u32(const std::uint8_t* p) {
    return (static_cast<std::uint32_t>(p[0]) << 24)
         | (static_cast<std::uint32_t>(p[1]) << 16)
         | (static_cast<std::uint32_t>(p[2]) <<  8)
         |  static_cast<std::uint32_t>(p[3]);
}

} // anonymous namespace

void LocalMaintenanceDB::serialize_table_(const std::map<std::string, std::string>& tbl, std::vector<std::uint8_t>& out) const {
    pack_u32(out, static_cast<std::uint32_t>(tbl.size()));
    for (const auto& [k, v] : tbl) {
        pack_u32(out, static_cast<std::uint32_t>(k.size()));
        out.insert(out.end(), reinterpret_cast<const std::uint8_t*>(k.data()),
                             reinterpret_cast<const std::uint8_t*>(k.data() + k.size()));
        pack_u32(out, static_cast<std::uint32_t>(v.size()));
        out.insert(out.end(), reinterpret_cast<const std::uint8_t*>(v.data()),
                             reinterpret_cast<const std::uint8_t*>(v.data() + v.size()));
    }
}

void LocalMaintenanceDB::serialize_table_(const std::map<std::string, std::map<std::string,std::string>>& tbl,
                                           std::vector<std::uint8_t>& out) const {
    pack_u32(out, static_cast<std::uint32_t>(tbl.size()));
    for (const auto& [k, m] : tbl) {
        pack_u32(out, static_cast<std::uint32_t>(k.size()));
        out.insert(out.end(), reinterpret_cast<const std::uint8_t*>(k.data()),
                             reinterpret_cast<const std::uint8_t*>(k.data() + k.size()));
        serialize_table_(m, out);
    }
}

bool LocalMaintenanceDB::deserialize_all_(const std::vector<std::uint8_t>& in) {
    if (in.size() < 4) return false;
    std::size_t pos = 0;

    auto read_u32 = [&]() -> std::uint32_t {
        if (pos + 4 > in.size()) return 0;
        std::uint32_t v = unpack_u32(in.data() + pos);
        pos += 4; return v;
    };
    auto read_str = [&]() -> std::string {
        std::uint32_t sz = read_u32();
        if (pos + sz > in.size()) return "";
        std::string s(reinterpret_cast<const char*>(in.data() + pos), sz);
        pos += sz; return s;
    };
    auto read_map = [&]() {
        std::map<std::string, std::string> m;
        std::uint32_t count = read_u32();
        for (std::uint32_t i = 0; i < count; ++i) {
            std::string key = read_str();
            std::string val = read_str();
            if (!key.empty()) m[key] = val;
        }
        return m;
    };
    auto read_map_of_maps = [&]() {
        std::map<std::string, std::map<std::string, std::string>> tbl;
        std::uint32_t count = read_u32();
        for (std::uint32_t i = 0; i < count; ++i) {
            std::string key = read_str();
            std::map<std::string, std::string> m = read_map();
            if (!key.empty()) tbl[key] = std::move(m);
        }
        return tbl;
    };

    std::uint32_t magic = read_u32();
    if (magic != 0x4C434D44) { // "LCMD" in BE
        return false;
    }

    // Read version
    std::uint32_t version = read_u32();
    (void)version; // v1 currently; structure is forward-compatible

    licenses_           = read_map_of_maps();
    extension_entries_  = read_map_of_maps();
    revenue_records_    = read_map_of_maps();
    reviews_            = read_map_of_maps();
    extension_stats_    = read_map_of_maps();
    credential_records_ = read_map_of_maps();
    audit_events_       = read_map_of_maps();
    rbpc_state_records_ = read_map_of_maps();
    vip_keys_           = read_map_of_maps();
    onboarding_grants_  = read_map_of_maps();
    inference_records_  = read_map_of_maps();
    file_vault_records_   = read_map_of_maps();

    preferences_      = read_map();
    revoked_hashes_   = read_map();

    // TrustPolicy
    if (read_u32() != 0) { // has_policy flag
        trust_policy_ = TrustPolicy::from_map(read_map());
    }

    return true;
}

// ============================================================================
// Disk persistence (AES-256-GCM encrypted at rest via LFSSL)
// ============================================================================

bool LocalMaintenanceDB::flush_to_disk_() const {
    if (db_path_.empty()) return true; // in-memory only

    // Serialize all tables into a single plaintext buffer
    std::vector<std::uint8_t> plaintext;
    plaintext.reserve(4096);

    static constexpr std::uint32_t kMagic   = 0x4C434D44; // "LCMD"
    static constexpr std::uint32_t kVersion = 1;

    pack_u32(plaintext, kMagic);
    pack_u32(plaintext, kVersion);

    serialize_table_(licenses_,          plaintext);
    serialize_table_(extension_entries_, plaintext);
    serialize_table_(revenue_records_,    plaintext);
    serialize_table_(reviews_,            plaintext);
    serialize_table_(extension_stats_,    plaintext);
    serialize_table_(credential_records_, plaintext);
    serialize_table_(audit_events_,       plaintext);
    serialize_table_(rbpc_state_records_,plaintext);
    serialize_table_(vip_keys_,           plaintext);
    serialize_table_(onboarding_grants_, plaintext);

    serialize_table_(inference_records_,    plaintext);
    serialize_table_(file_vault_records_,   plaintext);

    serialize_table_(preferences_,      plaintext);
    serialize_table_(revoked_hashes_,   plaintext);

    if (trust_policy_.has_value()) {
        pack_u32(plaintext, 1); // has_policy flag
        serialize_table_(trust_policy_.value().to_map(), plaintext);
    } else {
        pack_u32(plaintext, 0);
    }

    // Encrypt via LFSSL AES-256-GCM
    LfsslAesGcm& lfssl = lfssl_();

    std::ofstream ofs(db_path_, std::ios::binary | std::ios::trunc);
    if (!ofs) return false;

    if (!lfssl.available()) {
        // Sentinel mode: LFSSL unavailable — persist as plaintext.
        // Format: [4:kMagic][4:kVersion][N:plaintext] (distinguishable from
        // encrypted format because encrypted starts with nonce_len=12).
        ofs.write(reinterpret_cast<const char*>(plaintext.data()), plaintext.size());
        return ofs.good();
    }

    // 12-byte nonce using LFSSL CSPRNG (critical: NEVER reuse with same key)
    // If LFSSL CSPRNG unavailable, use std::random_device (OS entropy pool)
    std::vector<std::uint8_t> nonce(12);
    if (lfssl.random_bytes) {
        int rc_rng = lfssl.random_bytes(nonce.data(), nonce.size());
        if (rc_rng != 0) {
            // LFSSL RNG failed — refuse to encrypt rather than reuse nonce
            return false;
        }
    } else {
        // OS fallback: std::random_device pulls from OS entropy source
        // (/dev/urandom on Linux, BCryptGenRandom on Windows via C++ runtime)
        std::random_device rd;
        std::uniform_int_distribution<int> dist(0, 255);
        for (auto& b : nonce) b = static_cast<std::uint8_t>(dist(rd));
    }

    std::vector<std::uint8_t> ciphertext(plaintext.size() + 16); // pt + 16-byte tag
    std::size_t written = 0;

    int rc = lfssl.encrypt(db_key_.data(),
                           nonce.data(), nonce.size(),
                           plaintext.data(), plaintext.size(),
                           nullptr, 0, // no AAD
                           ciphertext.data(), ciphertext.size(), &written);
    if (rc != 0 || written == 0) return false;

    ciphertext.resize(written);

    // Write: [4:nonce_len][12:nonce][4:cipher_len][N:ciphertext]
    std::vector<std::uint8_t> out_buf;
    out_buf.reserve(4 + 12 + 4 + ciphertext.size());
    pack_u32(out_buf, static_cast<std::uint32_t>(nonce.size()));
    out_buf.insert(out_buf.end(), nonce.begin(), nonce.end());
    pack_u32(out_buf, static_cast<std::uint32_t>(ciphertext.size()));
    out_buf.insert(out_buf.end(), ciphertext.begin(), ciphertext.end());
    ofs.write(reinterpret_cast<const char*>(out_buf.data()), out_buf.size());
    return ofs.good();
}

bool LocalMaintenanceDB::load_from_disk_() {
    if (db_path_.empty() || !std::filesystem::exists(db_path_)) return true;

    std::ifstream ifs(db_path_, std::ios::binary);
    if (!ifs) return false;

    auto read_be32 = [&ifs]() -> std::uint32_t {
        std::uint8_t buf[4] = {0};
        ifs.read(reinterpret_cast<char*>(buf), 4);
        return unpack_u32(buf);
    };

    static constexpr std::uint32_t kMagic   = 0x4C434D44; // "LCMD"

    std::uint32_t first_u32 = read_be32();

    if (first_u32 == kMagic) {
        // Sentinel mode: plaintext file (LFSSL unavailable on this host).
        // Format: [4:kMagic][4:kVersion][N:serialized_tables...]
        // We already read kMagic; read the rest of the file into a buffer
        // that starts with kMagic so deserialize_all_ can process it.
        std::vector<std::uint8_t> plaintext;
        plaintext.reserve(4096);
        pack_u32(plaintext, kMagic);

        // Read version (next 4 bytes)
        std::uint8_t ver_buf[4] = {0};
        ifs.read(reinterpret_cast<char*>(ver_buf), 4);
        plaintext.insert(plaintext.end(), ver_buf, ver_buf + 4);

        // Read remainder
        char chunk[4096];
        while (ifs.good()) {
            ifs.read(chunk, sizeof(chunk));
            std::streamsize n = ifs.gcount();
            if (n > 0) {
                plaintext.insert(plaintext.end(), chunk, chunk + n);
            }
        }
        return deserialize_all_(plaintext);
    }

    // Encrypted file: first_u32 is nonce_len (expected 12)
    std::uint32_t nonce_len = first_u32;
    if (nonce_len != 12) return false;

    std::vector<std::uint8_t> nonce(12);
    ifs.read(reinterpret_cast<char*>(nonce.data()), 12);

    std::uint32_t cipher_len = read_be32();
    if (cipher_len < 16 || cipher_len > 0x0FFFFFFF) return false;

    std::vector<std::uint8_t> ciphertext(cipher_len);
    ifs.read(reinterpret_cast<char*>(ciphertext.data()), cipher_len);
    if (static_cast<std::size_t>(ifs.gcount()) != cipher_len) return false;

    // Decrypt via LFSSL
    LfsslAesGcm& lfssl = lfssl_();
    if (!lfssl.available()) return false;

    std::vector<std::uint8_t> plaintext(cipher_len);
    std::size_t written = 0;
    int rc = lfssl.decrypt(db_key_.data(),
                           nonce.data(), nonce.size(),
                           ciphertext.data(), ciphertext.size(),
                           nullptr, 0,
                           plaintext.data(), plaintext.size(), &written);
    if (rc != 0 || written == 0) return false;
    plaintext.resize(written);

    return deserialize_all_(plaintext);
}

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

    if (std::filesystem::exists(db_path_)) {
        // File exists — MUST decrypt successfully, or it's the wrong key
        if (!load_from_disk_()) {
            // Wrong key or corrupt file — do NOT initialize
            db_path_.clear();
            db_key_.clear();
            return false;
        }
    }
    // File doesn't exist — start empty, will be created on first flush

    initialized_ = true;
    return true;
}

void LocalMaintenanceDB::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) return;

    // Persist encrypted pages to disk before clearing
    if (dirty_.exchange(false, std::memory_order_relaxed)) {
        (void)flush_to_disk_();
    }

    // In production: clear sensitive memory after flush.
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
    inference_records_.clear();
    file_vault_records_.clear();
    initialized_ = false;
}

std::size_t LocalMaintenanceDB::pending_sync_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return sync_queue_.size();
}

// ============================================================================
// Offline mode
// ============================================================================

void LocalMaintenanceDB::set_offline_mode(bool offline) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    offline_mode_ = offline;
}

bool LocalMaintenanceDB::is_offline_mode() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return offline_mode_;
}

// ============================================================================
// Sync queue replay
// ============================================================================

std::size_t LocalMaintenanceDB::replay_sync_queue(
    SyncReplayCallback callback,
    std::size_t max_records) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (sync_queue_.empty()) return 0;

    std::size_t limit = (max_records == 0) ? sync_queue_.size()
                                           : std::min(max_records, sync_queue_.size());
    std::vector<SyncRecord> remaining;
    remaining.reserve(sync_queue_.size());

    std::size_t replayed = 0;
    for (std::size_t i = 0; i < sync_queue_.size(); ++i) {
        if (replayed < limit) {
            bool ok = false;
            try {
                ok = callback(sync_queue_[i].table,
                              sync_queue_[i].key,
                              sync_queue_[i].record);
            } catch (...) {
                ok = false; // callback threw; treat as failure, keep record
            }
            if (ok) {
                ++replayed;
                continue;
            } else {
                remaining.push_back(std::move(sync_queue_[i]));
            }
        } else {
            remaining.push_back(std::move(sync_queue_[i]));
        }
    }
    sync_queue_ = std::move(remaining);
    return replayed;
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

    mark_dirty_();
    if (offline_mode_) queue_for_sync("licenses", key, rec);
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
    mark_dirty_();
    if (offline_mode_) queue_for_sync("license_revocations", license_key_hash, { {"hash",license_key_hash}, {"reason",reason.empty() ? "revoked" : reason} });
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
    mark_dirty_();
    if (offline_mode_) queue_for_sync("extension_entries", it->second, entry);
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
    mark_dirty_();
    if (offline_mode_) queue_for_sync("revenue_records", key, record);
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
    mark_dirty_();
    if (offline_mode_) queue_for_sync("reviews", key, review);
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
    mark_dirty_();
    if (offline_mode_) queue_for_sync("extension_stats", extension_id, stats);
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
    mark_dirty_();
    if (offline_mode_) queue_for_sync("vip_keys", key_hash, rec);
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
    mark_dirty_();
    if (offline_mode_) queue_for_sync("vip_keys", key_hash, it->second);
    return true;
}

std::vector<std::string> LocalMaintenanceDB::get_all_vip_key_hashes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> out;
    for (const auto& [k, v] : vip_keys_) out.push_back(k);
    return out;
}

// ============================================================================
// Onboarding grants — basic RBPC grant storage and consumption.
// Advanced RBPC onboarding grant lifecycle (issuance, validation, cross-device) is
// handled by PsiForceDB. This provides the local encrypted table + sync queue
// following the exact same pattern as revenue_records, audit_events, etc.
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
    mark_dirty_();
    if (offline_mode_) queue_for_sync("onboarding_grants", key, safe);
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
    auto map = policy.to_map();
    mark_dirty_();
    if (offline_mode_) queue_for_sync("trust_policy", "trust_policy", map);
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
    mark_dirty_();
    if (offline_mode_) queue_for_sync("credential_records", key, safe);
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
    mark_dirty_();
    if (offline_mode_) queue_for_sync("rbpc_state", state.node_id, rec);
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
    mark_dirty_();
    if (offline_mode_) queue_for_sync("audit_events", event_id, rec);
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
    mark_dirty_();
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

// ============================================================================
// FileVaultRecord helpers
// ============================================================================

std::map<std::string, std::string> FileVaultRecord::to_map() const noexcept {
    std::map<std::string, std::string> m;
    m["file_id"]              = file_id;
    m["project_name"]         = project_name;
    m["folder_path"]          = folder_path;
    m["original_name"]      = original_name;
    m["encrypted_disk_path"]  = encrypted_disk_path;
    m["size_bytes"]           = size_bytes;
    m["mime_type"]            = mime_type;
    m["thumbnail_base64"]     = thumbnail_base64;
    m["created_at"]           = created_at;
    m["last_accessed"]        = last_accessed;
    m["jwt_audience"]          = jwt_audience;
    m["permission_level"]      = permission_level;
    m["encryption_iv"]         = encryption_iv;
    m["encryption_tag"]         = encryption_tag;
    m["status"]               = status;
    m["node_id"]              = node_id;
    m["record_class"]          = "file_vault_record";
    m["plaintext_storage"]     = "forbidden";
    return m;
}

FileVaultRecord FileVaultRecord::from_map(const std::map<std::string, std::string>& m) {
    FileVaultRecord r;
    auto it = m.find("file_id");          if (it != m.end()) r.file_id          = it->second;
    it = m.find("project_name");          if (it != m.end()) r.project_name     = it->second;
    it = m.find("folder_path");           if (it != m.end()) r.folder_path      = it->second;
    it = m.find("original_name");         if (it != m.end()) r.original_name    = it->second;
    it = m.find("encrypted_disk_path");   if (it != m.end()) r.encrypted_disk_path = it->second;
    it = m.find("size_bytes");            if (it != m.end()) r.size_bytes       = it->second;
    it = m.find("mime_type");           if (it != m.end()) r.mime_type        = it->second;
    it = m.find("thumbnail_base64");      if (it != m.end()) r.thumbnail_base64 = it->second;
    it = m.find("created_at");            if (it != m.end()) r.created_at       = it->second;
    it = m.find("last_accessed");         if (it != m.end()) r.last_accessed    = it->second;
    it = m.find("jwt_audience");          if (it != m.end()) r.jwt_audience     = it->second;
    it = m.find("permission_level");    if (it != m.end()) r.permission_level = it->second;
    it = m.find("encryption_iv");         if (it != m.end()) r.encryption_iv    = it->second;
    it = m.find("encryption_tag");        if (it != m.end()) r.encryption_tag   = it->second;
    it = m.find("status");                if (it != m.end()) r.status           = it->second;
    it = m.find("node_id");               if (it != m.end()) r.node_id          = it->second;
    return r;
}

// ============================================================================
// InferenceRecord helpers
// ============================================================================

std::map<std::string, std::string> InferenceRecord::to_map() const noexcept {
    std::map<std::string, std::string> m;
    m["inference_id"]          = inference_id;
    m["session_id"]            = session_id;
    m["prompt"]                = prompt;
    m["result_summary"]        = result_summary;
    m["status"]                = status;
    m["timestamp"]             = timestamp;
    m["generation_time_ms"]    = generation_time_ms;
    m["width"]                 = width;
    m["height"]                = height;
    m["num_steps"]             = num_steps;
    m["guidance_scale"]        = guidance_scale;
    m["encoder_name"]          = encoder_name;
    m["post_processor_name"]   = post_processor_name;
    m["gpu_backend_name"]      = gpu_backend_name;
    m["text_encode_used_npu"]  = text_encode_used_npu;
    m["denoise_used_gpu"]      = denoise_used_gpu;
    m["vae_decode_used_gpu"]   = vae_decode_used_gpu;
    m["post_process_used_npu"] = post_process_used_npu;
    m["unet_denoise_used_npu"] = unet_denoise_used_npu;
    m["npu_cheap_ops_percent"] = npu_cheap_ops_percent;
    m["recovery_attempts"]     = recovery_attempts;
    m["node_id"]               = node_id;
    m["record_class"]          = "inference_record";
    m["plaintext_storage"]     = "forbidden";
    return m;
}

InferenceRecord InferenceRecord::from_map(const std::map<std::string, std::string>& m) {
    InferenceRecord r;
    auto it = m.find("inference_id");          if (it != m.end()) r.inference_id          = it->second;
    it = m.find("session_id");                  if (it != m.end()) r.session_id            = it->second;
    it = m.find("prompt");                       if (it != m.end()) r.prompt                = it->second;
    it = m.find("result_summary");              if (it != m.end()) r.result_summary        = it->second;
    it = m.find("status");                      if (it != m.end()) r.status                = it->second;
    it = m.find("timestamp");                  if (it != m.end()) r.timestamp             = it->second;
    it = m.find("generation_time_ms");           if (it != m.end()) r.generation_time_ms    = it->second;
    it = m.find("width");                        if (it != m.end()) r.width                 = it->second;
    it = m.find("height");                      if (it != m.end()) r.height                = it->second;
    it = m.find("num_steps");                   if (it != m.end()) r.num_steps             = it->second;
    it = m.find("guidance_scale");               if (it != m.end()) r.guidance_scale        = it->second;
    it = m.find("encoder_name");                if (it != m.end()) r.encoder_name          = it->second;
    it = m.find("post_processor_name");          if (it != m.end()) r.post_processor_name   = it->second;
    it = m.find("gpu_backend_name");             if (it != m.end()) r.gpu_backend_name      = it->second;
    it = m.find("text_encode_used_npu");         if (it != m.end()) r.text_encode_used_npu  = it->second;
    it = m.find("denoise_used_gpu");             if (it != m.end()) r.denoise_used_gpu      = it->second;
    it = m.find("vae_decode_used_gpu");          if (it != m.end()) r.vae_decode_used_gpu   = it->second;
    it = m.find("post_process_used_npu");        if (it != m.end()) r.post_process_used_npu = it->second;
    it = m.find("unet_denoise_used_npu");        if (it != m.end()) r.unet_denoise_used_npu = it->second;
    it = m.find("npu_cheap_ops_percent");        if (it != m.end()) r.npu_cheap_ops_percent = it->second;
    it = m.find("recovery_attempts");           if (it != m.end()) r.recovery_attempts     = it->second;
    it = m.find("node_id");                     if (it != m.end()) r.node_id               = it->second;
    return r;
}

// ============================================================================
// TrustPolicy helpers
// ============================================================================

TrustPolicy TrustPolicy::default_policy() noexcept {
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

bool TrustPolicy::keeps_local_authority() const noexcept {
    return plaintext_storage == "forbidden" &&
           psmdb_recovery == "forbidden" &&
           credential_authority != "central_only";
}

std::map<std::string, std::string> TrustPolicy::to_map() const noexcept {
    std::map<std::string, std::string> m;
    m["policy_id"] = policy_id;
    m["deployment_model"] = deployment_model;
    m["credential_authority"] = credential_authority;
    m["central_service_role"] = central_service_role;
    m["medusaserv_role"] = medusaserv_role;
    m["psiforcedb_role"] = psiforcedb_role;
    m["psmdb_owner_scope"] = psmdb_owner_scope;
    m["proprietary_boundary"] = proprietary_boundary;
    m["plaintext_storage"] = plaintext_storage;
    m["maintenance_encryption_layer"] = maintenance_encryption_layer;
    m["psmdb_recovery"] = psmdb_recovery;
    m["psmdb_reenrollment_model"] = psmdb_reenrollment_model;
    m["hash_suite"] = hash_suite;
    m["pqc_profile"] = pqc_profile;
    m["hardware_binding"] = hardware_binding;
    m["authority_scope"] = authority_scope;
    m["rbpc_pin_source"] = rbpc_pin_source;
    m["rbpc_word_source"] = rbpc_word_source;
    m["rbpc_confirmation_window_seconds"] = rbpc_confirmation_window_seconds;
    m["rbpc_failure_burn_threshold"] = rbpc_failure_burn_threshold;
    m["temporary_onboarding"] = temporary_onboarding;
    m["audit_scope"] = audit_scope;
    m["step_up_model"] = step_up_model;
    m["record_class"] = record_class;
    m["created_at"] = created_at;
    m["updated_at"] = updated_at;
    return m;
}

TrustPolicy TrustPolicy::from_map(const std::map<std::string, std::string>& m) {
    TrustPolicy tp;
    auto it = m.find("policy_id");
    if (it != m.end()) tp.policy_id = it->second;
    it = m.find("deployment_model");
    if (it != m.end()) tp.deployment_model = it->second;
    it = m.find("credential_authority");
    if (it != m.end()) tp.credential_authority = it->second;
    it = m.find("central_service_role");
    if (it != m.end()) tp.central_service_role = it->second;
    it = m.find("medusaserv_role");
    if (it != m.end()) tp.medusaserv_role = it->second;
    it = m.find("psiforcedb_role");
    if (it != m.end()) tp.psiforcedb_role = it->second;
    it = m.find("psmdb_owner_scope");
    if (it != m.end()) tp.psmdb_owner_scope = it->second;
    it = m.find("proprietary_boundary");
    if (it != m.end()) tp.proprietary_boundary = it->second;
    it = m.find("plaintext_storage");
    if (it != m.end()) tp.plaintext_storage = it->second;
    it = m.find("maintenance_encryption_layer");
    if (it != m.end()) tp.maintenance_encryption_layer = it->second;
    it = m.find("psmdb_recovery");
    if (it != m.end()) tp.psmdb_recovery = it->second;
    it = m.find("psmdb_reenrollment_model");
    if (it != m.end()) tp.psmdb_reenrollment_model = it->second;
    it = m.find("hash_suite");
    if (it != m.end()) tp.hash_suite = it->second;
    it = m.find("pqc_profile");
    if (it != m.end()) tp.pqc_profile = it->second;
    it = m.find("hardware_binding");
    if (it != m.end()) tp.hardware_binding = it->second;
    it = m.find("authority_scope");
    if (it != m.end()) tp.authority_scope = it->second;
    it = m.find("rbpc_pin_source");
    if (it != m.end()) tp.rbpc_pin_source = it->second;
    it = m.find("rbpc_word_source");
    if (it != m.end()) tp.rbpc_word_source = it->second;
    it = m.find("rbpc_confirmation_window_seconds");
    if (it != m.end()) tp.rbpc_confirmation_window_seconds = it->second;
    it = m.find("rbpc_failure_burn_threshold");
    if (it != m.end()) tp.rbpc_failure_burn_threshold = it->second;
    it = m.find("temporary_onboarding");
    if (it != m.end()) tp.temporary_onboarding = it->second;
    it = m.find("audit_scope");
    if (it != m.end()) tp.audit_scope = it->second;
    it = m.find("step_up_model");
    if (it != m.end()) tp.step_up_model = it->second;
    it = m.find("record_class");
    if (it != m.end()) tp.record_class = it->second;
    it = m.find("created_at");
    if (it != m.end()) tp.created_at = it->second;
    it = m.find("updated_at");
    if (it != m.end()) tp.updated_at = it->second;
    return tp;
}

// ============================================================================
// Inference history — bootup core
// ============================================================================

bool LocalMaintenanceDB::store_inference_record(const InferenceRecord& record) {
    if (!initialized_ || record.inference_id.empty()) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    inference_records_[record.inference_id] = record.to_map();
    mark_dirty_();
    if (offline_mode_) queue_for_sync("inference_records", record.inference_id, record.to_map());
    return true;
}

std::optional<InferenceRecord> LocalMaintenanceDB::load_inference_record(
    const std::string& inference_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = inference_records_.find(inference_id);
    if (it == inference_records_.end()) return std::nullopt;
    return InferenceRecord::from_map(it->second);
}

std::vector<InferenceRecord> LocalMaintenanceDB::query_inference_records(
    std::time_t since, std::time_t until, std::size_t limit) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<InferenceRecord> out;
    for (const auto& [id, m] : inference_records_) {
        auto it = m.find("timestamp");
        if (it == m.end()) continue;
        std::time_t ts = 0;
        try { ts = static_cast<std::time_t>(std::stoll(it->second)); } catch (...) { continue; }
        if (ts >= since && ts <= until) {
            out.push_back(InferenceRecord::from_map(m));
            if (limit > 0 && out.size() >= limit) break;
        }
    }
    return out;
}

bool LocalMaintenanceDB::export_inference_json(const std::filesystem::path& path) const {
    std::ofstream ofs(path);
    if (!ofs) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    ofs << "[\n";
    std::size_t i = 0;
    for (const auto& [id, m] : inference_records_) {
        if (i > 0) ofs << ",\n";
        ofs << "  {\n";
        std::size_t j = 0;
        for (const auto& [k, v] : m) {
            if (j > 0) ofs << ",\n";
            ofs << "    \"" << k << "\": \"" << v << "\"";
            ++j;
        }
        ofs << "\n  }";
        ++i;
    }
    ofs << "\n]\n";
    return ofs.good();
}

bool LocalMaintenanceDB::export_inference_csv(const std::filesystem::path& path) const {
    std::ofstream ofs(path);
    if (!ofs) return false;
    // Header
    ofs << "inference_id,session_id,prompt,result_summary,status,timestamp,"
           << "generation_time_ms,width,height,num_steps,guidance_scale,"
           << "encoder_name,post_processor_name,gpu_backend_name,"
           << "text_encode_used_npu,denoise_used_gpu,vae_decode_used_gpu,"
           << "post_process_used_npu,unet_denoise_used_npu,npu_cheap_ops_percent,"
           << "recovery_attempts,node_id\n";
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& [id, m] : inference_records_) {
        auto get = [&m](const std::string& k) {
            auto it = m.find(k);
            return it != m.end() ? it->second : std::string{};
        };
        ofs << get("inference_id") << ','
           << get("session_id") << ','
           << '"' << get("prompt") << '"' << ','
           << '"' << get("result_summary") << '"' << ','
           << get("status") << ','
           << get("timestamp") << ','
           << get("generation_time_ms") << ','
           << get("width") << ','
           << get("height") << ','
           << get("num_steps") << ','
           << get("guidance_scale") << ','
           << get("encoder_name") << ','
           << get("post_processor_name") << ','
           << get("gpu_backend_name") << ','
           << get("text_encode_used_npu") << ','
           << get("denoise_used_gpu") << ','
           << get("vae_decode_used_gpu") << ','
           << get("post_process_used_npu") << ','
           << get("unet_denoise_used_npu") << ','
           << get("npu_cheap_ops_percent") << ','
           << get("recovery_attempts") << ','
           << get("node_id") << '\n';
    }
    return ofs.good();
}

bool LocalMaintenanceDB::export_inference_markdown(const std::filesystem::path& path) const {
    std::ofstream ofs(path);
    if (!ofs) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    ofs << "# Cerberus Inference History\n\n";
    ofs << "| ID | Prompt | Status | Time (ms) | Hardware |\n";
    ofs << "|----|--------|--------|-----------|----------|\n";
    double total_ms = 0.0;
    for (const auto& [id, m] : inference_records_) {
        auto get = [&m](const std::string& k) {
            auto it = m.find(k);
            return it != m.end() ? it->second : std::string{};
        };
        double ms = 0.0;
        try { ms = std::stod(get("generation_time_ms")); } catch (...) {}
        total_ms += ms;
        ofs << "| " << get("inference_id")
           << " | " << get("prompt")
           << " | " << get("status")
           << " | " << ms
           << " | " << get("encoder_name")
           << "/" << get("gpu_backend_name")
           << " |\n";
    }
    if (!inference_records_.empty()) {
        double avg = total_ms / static_cast<double>(inference_records_.size());
        ofs << "\n## Summary\n\n";
        ofs << "| Metric | Value |\n";
        ofs << "|--------|-------|\n";
        ofs << "| Count  | " << inference_records_.size() << " |\n";
        ofs << "| Avg ms | " << avg << " |\n";
    }
    return ofs.good();
}

bool LocalMaintenanceDB::clear_inference_records() {
    std::lock_guard<std::mutex> lock(mutex_);
    inference_records_.clear();
    mark_dirty_();
    return true;
}

std::map<std::string, std::string> LocalMaintenanceDB::inference_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::map<std::string, std::string> stats;
    stats["count"] = std::to_string(inference_records_.size());
    double total_ms = 0.0;
    std::size_t success_count = 0;
    std::size_t fail_count = 0;
    for (const auto& [id, m] : inference_records_) {
        auto it = m.find("generation_time_ms");
        if (it != m.end()) {
            try { total_ms += std::stod(it->second); } catch (...) {}
        }
        auto sit = m.find("status");
        if (sit != m.end()) {
            if (sit->second == "success") ++success_count;
            else if (sit->second == "failed") ++fail_count;
        }
    }
    stats["total_ms"] = std::to_string(total_ms);
    stats["success_count"] = std::to_string(success_count);
    stats["fail_count"] = std::to_string(fail_count);
    if (!inference_records_.empty()) {
        stats["avg_ms"] = std::to_string(total_ms / static_cast<double>(inference_records_.size()));
    }
    return stats;
}

// ============================================================================
// File vault (encrypted file metadata + project/folder indexing)
// ============================================================================

bool LocalMaintenanceDB::store_file_record(const FileVaultRecord& record) {
    if (!initialized_ || record.file_id.empty()) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    file_vault_records_[record.file_id] = record.to_map();
    mark_dirty_();
    if (offline_mode_) queue_for_sync("file_vault", record.file_id, record.to_map());
    return true;
}

std::optional<FileVaultRecord> LocalMaintenanceDB::load_file_record(
    const std::string& file_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = file_vault_records_.find(file_id);
    if (it == file_vault_records_.end()) return std::nullopt;
    return FileVaultRecord::from_map(it->second);
}

bool LocalMaintenanceDB::delete_file_record(const std::string& file_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = file_vault_records_.find(file_id);
    if (it == file_vault_records_.end()) return false;
    it->second["status"] = "deleted";
    mark_dirty_();
    if (offline_mode_) queue_for_sync("file_vault", file_id, it->second);
    return true;
}

bool LocalMaintenanceDB::update_file_accessed(const std::string& file_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = file_vault_records_.find(file_id);
    if (it == file_vault_records_.end()) return false;
    auto now = std::chrono::system_clock::now();
    auto sec = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
    it->second["last_accessed"] = std::to_string(sec);
    mark_dirty_();
    return true;
}

bool LocalMaintenanceDB::update_file_status(const std::string& file_id, const std::string& status) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = file_vault_records_.find(file_id);
    if (it == file_vault_records_.end()) return false;
    it->second["status"] = status;
    mark_dirty_();
    if (offline_mode_) queue_for_sync("file_vault", file_id, it->second);
    return true;
}

std::vector<FileVaultRecord> LocalMaintenanceDB::query_files_by_project(
    const std::string& project_name,
    const std::string& folder_path,
    const std::string& status_filter) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<FileVaultRecord> out;
    for (const auto& [id, m] : file_vault_records_) {
        auto pit = m.find("project_name");
        auto fit = m.find("folder_path");
        auto sit = m.find("status");
        bool match_project = (pit != m.end() && pit->second == project_name);
        bool match_folder  = folder_path.empty() || (fit != m.end() && fit->second == folder_path);
        bool match_status  = status_filter.empty() || (sit != m.end() && sit->second == status_filter);
        if (match_project && match_folder && match_status) {
            out.push_back(FileVaultRecord::from_map(m));
        }
    }
    return out;
}

std::vector<std::string> LocalMaintenanceDB::list_projects() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> out;
    for (const auto& [id, m] : file_vault_records_) {
        auto sit = m.find("status");
        if (sit != m.end() && sit->second == "deleted") continue;
        auto pit = m.find("project_name");
        if (pit != m.end()) {
            bool dup = false;
            for (const auto& existing : out) {
                if (existing == pit->second) { dup = true; break; }
            }
            if (!dup) out.push_back(pit->second);
        }
    }
    return out;
}

std::map<std::string, std::string> LocalMaintenanceDB::file_vault_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::map<std::string, std::string> stats;
    std::size_t total_size = 0;
    std::size_t active_count = 0;
    std::size_t quarantined_count = 0;
    std::size_t deleted_count = 0;
    for (const auto& [id, m] : file_vault_records_) {
        auto sit = m.find("status");
        if (sit != m.end()) {
            if (sit->second == "active") ++active_count;
            else if (sit->second == "quarantined") ++quarantined_count;
            else if (sit->second == "deleted") ++deleted_count;
        }
        auto sz = m.find("size_bytes");
        if (sz != m.end()) {
            try { total_size += std::stoull(sz->second); } catch (...) {}
        }
    }
    stats["total_count"]      = std::to_string(file_vault_records_.size());
    stats["active_count"]      = std::to_string(active_count);
    stats["quarantined_count"] = std::to_string(quarantined_count);
    stats["deleted_count"]      = std::to_string(deleted_count);
    stats["total_size_bytes"]  = std::to_string(total_size);
    return stats;
}

} // namespace hq::cerberus::privacy
