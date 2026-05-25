#pragma once
/// @file cerberus_first_run.hpp
/// @copyright Copyright (c) 2026 D Hargreaves. All rights reserved.
///
/// Cerberus First-Run Registration — One-Time PsiForceDB Handshake
/// ==================================================================
///
/// This is the INITIATION PROTOCOL. It happens ONCE per device install.
/// After this, Cerberus operates entirely in offline/local mode, using
/// the Local Maintenance DB as its carbon-copy authority.
///
/// Flow: (matches PsiForceDB install_smdu_root.cpp / SMDUDatabase / RBPC epoch)
///   1. User runs Cerberus for the first time.
///   2. Cerberus detects no Local Maintenance DB on disk.
///   3. Cerberus prompts for the PsiForceDB Master Administrator passphrase.
///   4. If PsiForceDB is reachable: perform live handshake.
///      a. Send anonymous device fingerprint (hash only, no PII).
///      b. PsiForceDB validates master passphrase via SMDU gate.
///      c. PsiForceDB returns: provisioned Kyber public key, per-device salt.
///      d. PsiForceDB returns: RBPC trust policy (carbon-copy sealed).
///      e. PsiForceDB signs a one-time onboarding grant with expiry.
///      f. Onboarding grant is sealed_expiring_consumed_local type.
///   5. If PsiForceDB is NOT reachable: enter LOCAL-ONLY DECLARE mode.
///      a. User declares they are the device owner.
///      b. A local-only SMDU is generated from passphrase + hardware entropy.
///      c. This is marked "local_authority" and queud for later validation.
///      d. The device is fully functional offline but flagged for review.
///   6. Regardless of path, generate:
///      a. SMDU identity (passphrase + salt → Argon2id → Kyber keypair).
///      b. System-issued 6-digit PIN.
///      c. Prompt user for memorable word (8–40 chars).
///      d. Hardware anchor (device-bound, no migration).
///      e. JWT session secret (HMAC-SHA256 key for persistent auth).
///   7. Write ALL of the above into the Local Maintenance DB encrypted.
///   8. Create the carbon-copy RBPC trust policy.
///   9. Cerberus now operates. Every boot: unlock SMDU → validate JWT →
///      optionally verify PIN+Word confirmation window.
///
/// BOUNDARY: All PQC (Kyber, Dilithium), Argon2id, AES-256-GCM are delegated
/// to LFSSL. This orchestrator never implements a primitive.
///
/// Works offline and online identically.
///
/// © 2026 D Hargreaves | LamiaFabrica Software

#include <string>
#include <vector>
#include <cstdint>
#include <optional>
#include <memory>

#include "hq/cerberus_local_maintenance_db.hpp"
#include "hq/cerberus_user_security.hpp"
#include "hq/cerberus_jwt_session.hpp"
#include "hq/cerberus_smdi.hpp"

namespace hq::cerberus::privacy {

// ============================================================================
// Registration Result
// ============================================================================
struct RegistrationResult {
    bool success{false};
    std::string diagnostic;          // human-readable error if failed
    bool provisional{false};           // true if local-only declaration (not yet validated by PsiForceDB)
    std::string node_id;             // device-bound identifier
    std::string issued_pin;          // display-once to user
    std::string jwt_secret;          // session signing secret
    std::vector<std::uint8_t> db_key;   // AES-256-GCM key for Local DB
    std::chrono::system_clock::time_point expires_at;
};

// ============================================================================
// FirstRun — Initiatory Registration Controller
// ============================================================================
class FirstRun {
public:
    FirstRun() = default;

    /// Check whether a local maintenance database already exists on disk.
    /// @param db_path  Candidate path for the local DB file.
    /// @return true if the database file exists and is readable.
    [[nodiscard]] bool is_already_registered(const std::filesystem::path& db_path) const;

    /// Perform first-run registration.
    /// @param passphrase       The PsiForceDB master passphrase (user input).
    /// @param memorable_word   The user-chosen memorable word.
    /// @param db_path          Where to create the Local Maintenance DB.
    /// @param psi_reachable    Whether PsiForceDB is currently online.
    /// @return RegistrationResult with all generated secrets (display-once).
    [[nodiscard]] RegistrationResult register_new_install(
        std::string_view passphrase,
        std::string_view memorable_word,
        const std::filesystem::path& db_path,
        bool psi_reachable);

    /// Re-derive an existing installation (after user provides passphrase + word).
    [[nodiscard]] RegistrationResult unlock_existing(
        std::string_view passphrase,
        std::string_view memorable_word,
        const std::filesystem::path& db_path) const;

    // MANDATORY unavailable_reason per AGENTS.md
    [[nodiscard]] static std::string unavailable_reason() noexcept;

private:
    std::string generate_node_id_() const;
    std::vector<std::uint8_t> derive_db_key_from_smdi_(const SMDU& smdu) const;
    std::string derive_jwt_secret_from_smdi_(const SMDU& smdu) const;
    void setup_local_db_(LocalMaintenanceDB& db, const RegistrationResult& reg, const SMDU& smdu,
                         const std::string& pin, std::string_view memorable_word) const;
};

} // namespace hq::cerberus::privacy
