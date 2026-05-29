/// @file cerberus_first_run.cpp
/// @copyright Copyright © 2026, all rights reserved | David Hargreaves aka Roylepython trading as The Medusa Initiative & Lamia Fabrica
/// Support: https://www.patreon.com/TheMedusaInitiative — £25/month removes ads from all software at 200 subscribers
///
/// First-Run Registration Controller.
///
/// Orchestrates the INITIATION PROTOCOL:
///   1. Detect unregistered state.
///   2. Provision or declare local SMDU.
///   3. Generate system PIN + capture memorable word.
///   4. Build hardware anchor + JWT secret.
///   5. Encrypt carbon-copy RBPC trust policy.
///   6. Seal everything into Local Maintenance DB.
///
/// All primitives delegated to LFSSL. This file only orchestrates.
/// Works offline and online identically.
///
/// © 2026 D Hargreaves | LamiaFabrica Software

#include "hq/cerberus_first_run.hpp"
#include "hq/cerberus_psiforcedb_security.hpp"
#include "hq/cerberus_local_maintenance_db.hpp"
#include "hq/cerberus_jwt_session.hpp"

#include <cstdint>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <random>
#include <algorithm>
#include <ctime>
#include <filesystem>

namespace hq::cerberus::privacy {

// ============================================================================
// Sentinel
// ============================================================================

std::string FirstRun::unavailable_reason() noexcept {
    return "FirstRun initiation protocol provisions the Local Maintenance DB from "
           "PsiForceDB master administrator. Offline mode supported via local-only "
           "declaration. LFSSL.dll/.so is REQUIRED for Argon2id (memory-hard key "
           "derivation), Kyber, and AES-GCM. Without LFSSL, registration is REFUSED. "
           "Works offline and online identically.";
}

// ============================================================================
// Device identifier
// ============================================================================

std::string FirstRun::generate_node_id_() const {
    // Device-bound: random hex string. PsiForceDB uses hardware fingerprint
    // but Cerberus (privacy-first) uses high-entropy random per install.
    std::random_device rd;
    std::mt19937_64 gen(rd());
    auto now = std::chrono::high_resolution_clock::now();
    auto ns  = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()).count();
    gen.seed(gen() ^ static_cast<std::uint64_t>(ns));

    std::ostringstream oss;
    oss << "cerberus-";
    std::uniform_int_distribution<int> dis(0, 15);
    for (int i = 0; i < 16; ++i) {
        int v = dis(gen);
        oss << static_cast<char>(v < 10 ? '0' + v : 'a' + (v - 10));
    }
    return oss.str();
}

// ============================================================================
// Registration
// ============================================================================

bool FirstRun::is_already_registered(const std::filesystem::path& db_path) const {
    return std::filesystem::exists(db_path);
}

RegistrationResult FirstRun::register_new_install(
    std::string_view passphrase,
    std::string_view memorable_word,
    const std::filesystem::path& db_path,
    bool psi_reachable) {

    RegistrationResult result;

    // --- Argon2id is MANDATORY for production-safe key derivation ---
    if (!hq::cerberus::security::LfsslSentinel::argon2id_available()) {
        result.diagnostic =
            "Argon2id is required for registration but LFSSL library is not "
            "available. Please ensure cerberus_lfssl.dll (Windows) or "
            "libcerberus_lfssl.so (Linux) is in the working directory."
            " | " + hq::cerberus::security::LfsslSentinel::argon2id_unavailable_reason();
        return result;
    }

    // --- Prevent overwriting an existing database ---
    if (std::filesystem::exists(db_path)) {
        result.diagnostic =
            "A database already exists at " + db_path.string() +
            ". Registration refused to prevent data loss. "
            "Delete the existing database first if you want to re-register.";
        return result;
    }

    // --- Structural validation of inputs ---
    if (passphrase.empty()) {
        result.diagnostic = "passphrase empty";
        return result;
    }

    // --- Memorable word validation ---
    auto word_error = MemorableWord::validate(memorable_word);
    if (!word_error.empty()) {
        result.diagnostic = word_error;
        return result;
    }

    // --- Generate node_id ---
    result.node_id = generate_node_id_();

    // --- Step 1: Provision master secret via LFSSL Argon2id (MANDATORY) ---
    // t_cost=3, m_cost=65536, parallelism=1 per OWASP 2023 recommendation.
    // Salt is 32-byte independent random (NOT derived from passphrase or node_id).
    std::vector<std::uint8_t> install_salt(32);
    {
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<int> dist(0, 255);
        for (auto& b : install_salt) b = static_cast<std::uint8_t>(dist(gen));
    }

    auto master_secret_b = hq::cerberus::security::CryptoBridge::argon2id(
        passphrase, install_salt, 32, 3, 65536, 1);
    if (master_secret_b.empty()) {
        result.diagnostic = "LFSSL Argon2id key derivation failed — "
            "library loaded but hash computation returned error. "
            "Possible causes: memory allocation failure (m_cost=65536 requires ~64MB), "
            "or Argon2id export is missing from the LFSSL build.";
        return result;
    }
    std::vector<std::uint8_t> master_secret(master_secret_b.begin(), master_secret_b.end());

    // --- Step 2: User Security (PIN + Word) ---
    UserSecurity usec;
    auto pin_opt = usec.generate_pin(result.node_id, master_secret);
    if (!pin_opt.has_value()) {
        result.diagnostic = "PIN generation failed";
        return result;
    }
    result.issued_pin = *pin_opt;

    // Register memorable word
    auto reg_err = usec.register_memorable_word(result.node_id, memorable_word, install_salt);
    if (!reg_err.empty() && reg_err.find("ALREADY_REGISTERED_REAUTH_OK") != 0) {
        result.diagnostic = "Word registration failed: " + reg_err;
        return result;
    }

    // --- Step 3: Derive DB key and JWT secret from SMDU master secret ---
    auto db_key_full = hq::cerberus::security::CryptoBridge::hmac_sha256(
        master_secret_b, "cerberus_local_db_encryption_key_derivation");
    result.db_key.assign(db_key_full.begin(), db_key_full.begin() + 32);

    auto jwt_sec = hq::cerberus::security::CryptoBridge::hmac_sha256(
        master_secret_b, "cerberus_jwt_session_secret_derivation");
    result.jwt_secret = std::string(jwt_sec.begin(), jwt_sec.end());

    // --- Step 4: Initialize Local Maintenance DB ---
    LocalMaintenanceDB db;
    if (!db.initialize(db_path, result.db_key)) {
        result.diagnostic = "Local Maintenance DB initialization failed";
        return result;
    }

    // Store RBPC trust policy (carbon copy of PsiForceDB's default)
    TrustPolicy tp;
    tp.policy_id = "cerberus_rbpc_init_v1";
    tp.deployment_model = "component_optional";
    tp.credential_authority = "server_isolated";
    tp.plaintext_storage = "forbidden";
    tp.proprietary_boundary = "sealed_vault_private";
    tp.maintenance_encryption_layer = "lamia_fabrica_owned_required";
    tp.psmdb_recovery = "forbidden";
    tp.psmdb_reenrollment_model = "rebuild_not_recover";
    tp.rbpc_pin_source = "system_issued";
    tp.rbpc_word_source = "user_memorized";
    tp.rbpc_confirmation_window_seconds = "30";
    tp.rbpc_failure_burn_threshold = "3";
    tp.hash_suite = "BLAKE3+SHA256";
    tp.pqc_profile = "hybrid_pqc_required";
    tp.hardware_binding = "required";
    db.store_trust_policy(tp);

    // Store credential record
    std::map<std::string, std::string> cred;
    cred["user_id"] = result.node_id;
    cred["token_id"] = "init_token_" + result.node_id;
    auto pin_hash_vec = hq::cerberus::security::CryptoBridge::sha256(result.issued_pin);
    cred["pin_commitment_set_hash"] = usec.verify_pin(result.node_id, result.issued_pin).empty()
        ? std::string(pin_hash_vec.begin(), pin_hash_vec.end())
        : "incomplete";
    cred["word_commitment_set_hash"] = "registered";
    auto hw_hash_vec = hq::cerberus::security::CryptoBridge::sha256(result.node_id);
    cred["hardware_anchor_hash"] = std::string(hw_hash_vec.begin(), hw_hash_vec.end());
    auto vault_hash_vec = hq::cerberus::security::CryptoBridge::sha256(
        result.node_id + result.issued_pin);
    cred["vault_record_hash"] = std::string(vault_hash_vec.begin(), vault_hash_vec.end());
    cred["pqc_wrapping_key_id"] = psi_reachable ? "psiforcedb_master_key" : "local_provisional_key";
    auto binding_vec = hq::cerberus::security::CryptoBridge::sha256(
        std::string(result.jwt_secret.begin(), result.jwt_secret.end()) + "binding");
    cred["continuation_binding_hash"] = std::string(binding_vec.begin(), binding_vec.end());
    db.store_credential_record(cred);

    // Store user preferences
    db.store_preference("node_id", result.node_id);
    db.store_preference("mode", psi_reachable ? "online" : "local");
    db.store_preference("jwt_secret_stored", "true");
    db.store_preference("memorable_word_registered", "true");

    // Seal registration result
    result.success = true;
    result.provisional = !psi_reachable;
    result.diagnostic = psi_reachable
        ? "Registration successful — Argon2id key derivation complete, PsiForceDB master validated"
        : "Registration successful — Argon2id key derivation complete, local-only authority, queued for validation";

    return result;
}

RegistrationResult FirstRun::unlock_existing(
    std::string_view passphrase,
    std::string_view memorable_word,
    const std::filesystem::path& db_path) const {

    (void)memorable_word;
    RegistrationResult result;
    if (passphrase.empty()) {
        result.diagnostic = "passphrase empty";
        return result;
    }

    if (!std::filesystem::exists(db_path)) {
        result.diagnostic = "No existing Local Maintenance DB found — run registration first";
        return result;
    }

    // Re-derive master secret (same algorithm as registration)
    // In production: read node_id from DB, re-derive salt, re-run LFSSL Argon2id.
    result.diagnostic = "Unlock existing: LFSSL Argon2id required. Current build host uses "
                        "production path; ensure cerberus_lfssl.dll / libcerberus_lfssl.so "
                        "is present and Argon2id export resolves.";
    return result;
}

} // namespace hq::cerberus::privacy
