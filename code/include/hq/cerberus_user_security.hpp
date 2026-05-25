#pragma once
/// @file cerberus_user_security.hpp
/// @copyright Copyright (c) 2026 D Hargreaves. All rights reserved.
///
/// Cerberus User Security — PIN + Memorable Word (RBPC Model)
/// ============================================================
///
/// Implements the PsiForceDB RBPC (Reinforced Biometric/PIN Confirmation) model:
///
///   - PIN: system-issued 6-digit code. NOT stored in plaintext.
///     Only an Argon2id commitment hash (with per-user salt) is stored.
///   - Memorable Word: user-memorized word (8–40 chars). NOT stored in plaintext.
///     Only a commitment hash is stored.
///   - Confirmation: both PIN and word are hashed, combined, and verified.
///     ANY failure increments the burn counter; 3 burns lock the node forever.
///   - Burn policy: 3 failed attempts → permanent lockout (SYSTEM LOCKED status).
///   - Hardware anchor: device-bound, no migration possible. Period.
///   - Works offline and online identically.
///
/// Generation (copied from PsiForceDB SMDU installer / psmdb_manager.cpp):
///   1. LFSSL Argon2id(passphrase + master_salt) → master secret.
///   2. Master secret wraps Kyber-1024 keypair via LFSSL.
///   3. PIN is derived from BLAKE3(master_secret || "rbpc_pin") mod 10 per digit.
///   4. Memorable word commitment = LFSSL Argon2id(word, salt, t=3, m=64K, p=1).
///
/// BOUNDARY: Argon2id, Kyber, BLAKE3, AES-256-GCM all delegated to LFSSL.
/// Cerberus only stores commitments and orchestrates verification.
///
/// © 2026 D Hargreaves | LamiaFabrica Software

#include <string>
#include <vector>
#include <cstdint>
#include <optional>
#include <map>
#include <mutex>

#include "hq/cerberus_local_maintenance_db.hpp"

namespace hq::cerberus::privacy {

// ============================================================================
// PIN Generator — matches PsiForceDB RBPC "system_issued" model
// ============================================================================
struct PINGenerator {
    /// Generate a deterministic 6-digit PIN from a master identity hash.
    /// @param master_hash  High-entropy hash (e.g. from Argon2id or Kyber pub key).
    /// @return 6-digit PIN string (e.g. "739284").
    [[nodiscard]] static std::string generate(const std::vector<std::uint8_t>& master_hash);

    /// Generate a random-looking PIN using std::random_device (fallback).
    /// Documented as a temporary replacement until LFSSL Argon2id linked.
    [[nodiscard]] static std::string generate_fallback();
};

// ============================================================================
// Memorable Word Validator
// ============================================================================
struct MemorableWord {
    /// Minimum and maximum length enforced.
    static constexpr std::size_t MIN_LENGTH = 8;
    static constexpr std::size_t MAX_LENGTH = 40;

    /// Validate a user's memorable word.
    /// @return validation error string, or empty string if valid.
    [[nodiscard]] static std::string validate(std::string_view word);

    /// Derive a commitment hash for storage (does NOT store plaintext).
    /// Uses Argon2id via LFSSL; fallback to HMAC-SHA256 if LFSSL unavailable.
    [[nodiscard]] static std::vector<std::uint8_t> derive_commitment(
        std::string_view word,
        const std::vector<std::uint8_t>& salt);
};

// ============================================================================
// RBPC Confirmation Set — matches PsiForceDB credential record schema
// ============================================================================
struct RBPCConfirmationSet {
    std::string node_id;                    // device-bound node identifier
    std::string pin_commitment_hash;        // hash of PIN commitment
    std::string word_commitment_hash;       // hash of word commitment
    std::string word_salt;                  // salt used for word commitment (must match on verify)
    std::string hardware_anchor_hash;       // device fingerprint hash
    std::string vault_record_hash;          // combined vault hash
    std::string pqc_wrapping_key_id;        // wrapping key reference (PsiForceDB)
    std::string continuation_binding_hash;  // continuation token hash
    std::time_t created_at{0};

    [[nodiscard]] bool is_complete() const noexcept {
        return !node_id.empty() &&
               !pin_commitment_hash.empty() &&
               !word_commitment_hash.empty() &&
               !word_salt.empty() &&
               !hardware_anchor_hash.empty();
    }
};

// ============================================================================
// UserSecurity — combined PIN + Word + Hardware Anchor
// ============================================================================
class UserSecurity {
public:
    /// Generate a new system-issued PIN and store its commitment.
    /// @param node_id          Unique node identifier for this device.
    /// @param master_secret    Derived from SMDU (LFSSL Argon2id).
    /// @return The plaintext PIN (display-once to user), or nullopt.
    [[nodiscard]] std::optional<std::string> generate_pin(
        const std::string& node_id,
        const std::vector<std::uint8_t>& master_secret);

    /// Verify a PIN attempt. Returns failure reason if invalid.
    /// Increments failed_attempts; burns on 3 failures.
    [[nodiscard]] std::string verify_pin(const std::string& node_id,
                                         const std::string& pin_attempt);

    /// Register a user-memorable word. Returns error if invalid.
    [[nodiscard]] std::string register_memorable_word(
        const std::string& node_id,
        std::string_view word,
        const std::vector<std::uint8_t>& salt);

    /// Verify both PIN and word in the confirmation window.
    /// @param pin_attempt  The PIN the user entered.
    /// @param word_attempt The word the user entered.
    /// @return Empty string on success, error message on failure.
    [[nodiscard]] std::string verify_confirmation(
        const std::string& node_id,
        const std::string& pin_attempt,
        std::string_view word_attempt);

    /// Check if a node is burned (permanent lockout after 3 failures).
    [[nodiscard]] bool is_burned(const std::string& node_id) const;

    /// Zero all sensitive memory.
    void scrub_all();

    // MANDATORY unavailable_reason per AGENTS.md
    [[nodiscard]] static std::string unavailable_reason() noexcept;

private:
    mutable std::mutex mutex_;
    std::map<std::string, RBPCState> states_;  // node_id → state
    std::map<std::string, RBPCConfirmationSet> confirmation_sets_;
};

} // namespace hq::cerberus::privacy
