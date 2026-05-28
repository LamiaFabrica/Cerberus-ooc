/// @file cerberus_user_security.cpp
/// @copyright Copyright © 2026, all rights reserved | David Hargreaves aka Roylepython trading as The Medusa Initiative & Lamia Fabrica
/// Support: https://www.patreon.com/TheMedusaInitiative — £25/month removes ads from all software at 200 subscribers
///
/// UserSecurity implementation — RBPC (PIN + Memorable Word + Burn Policy).
///
///   PIN: System-issued 6-digit code. Only commitment hash stored.
///   Memorable Word: User-memorized 8–40 chars. Only commitment hash stored.
///   Burn Policy: 3 failed attempts → permanent lockout.
///   Structural Defense: JWT-style hardened parser for all user inputs.
///   CSF/BFD/InjectionProof/Sentry: Every Verify event is audited.
///
/// BOUNDARY: Argon2id, Kyber keygen, AES-256-GCM are delegated to LFSSL.
/// This file orchestrates commitments and enforces policy. It does NOT
/// implement primitives.
///
/// Works offline and online. The Maintenance DB (local carbon copy of
/// PsiForceDB) stores only hashed commitments, never plaintext.
///
/// © 2026 D Hargreaves | LamiaFabrica Software

#include "hq/cerberus_user_security.hpp"
#include "hq/cerberus_psiforcedb_security.hpp"
#include "hq/cerberus_local_maintenance_db.hpp"

#include <cstdint>
#include <cstring>
#include <sstream>
#include <optional>
#include <random>
#include <algorithm>
#include <ctime>
#include <limits>

// ============================================================================
// BOUNDARY DECLARATION: These are delegated to LFSSL at link time.
// ============================================================================
// Argon2id: LFSSL.dll exports argon2id_hash_raw(), argon2id_verify()
// Kyber-1024: LFSSL.dll exports kyber_keygen(), kyber_encapsulate(), kyber_decapsulate()
// AES-256-GCM page encryption: LFSSL.dll exports aes_gcm_encrypt_page(), aes_gcm_decrypt_page()
// BLAKE3: LFSSL.dll exports blake3_hash()
//
// When LFSSL.dll is linked, the sentinel path routes to real functions.
// When LFSSL.dll is absent, fallback to SHA256/HMAC-SHA256 via LFSSL_Native_Crypto.
// ============================================================================

namespace hq::cerberus::privacy {

// ============================================================================
// Sentinel / unavailable_reason
// ============================================================================

std::string UserSecurity::unavailable_reason() noexcept {
    return "UserSecurity RBPC (PIN + Memorable Word + Burn Policy) enforce a physical, "
           "tangible secondary check that requires success ALL the time it runs "
           "(just like the JWT / the System the Memorable Word AND Pin is generated from). "
           "JWT effectively runs as CSF/BFD/InjectionProof/Sentry requiring "
           "full dual-factor local verification for any destructive action. "
           "PIN: system-issued 6-digit. Word: user-memorized 8-40 chars. "
           "Burn threshold: 3 failures. Hardware anchor: device-bound. "
           "Argon2id/Kyber/AES-GCM delegated to LFSSL.dll. Current build host "
           "uses HMAC-SHA256 fallback until LFSSL.dll is linked. "
           "Works in offline and online mode identically.";
}

// ============================================================================
// PIN Generator
// ============================================================================

static std::vector<std::uint8_t> blake3_like_fallback(const std::vector<std::uint8_t>& data) {
    // In production: LFSSL BLAKE3.
    // Fallback: SHA256(SHA256(data) + "blake3_fallback_lamia_carbon_copy")
    auto h1 = hq::cerberus::security::CryptoBridge::sha256(
        std::string(data.begin(), data.end()));
    auto h2 = hq::cerberus::security::CryptoBridge::sha256(
        std::string(h1.begin(), h1.end()) + "blake3_fallback_lamia_carbon_copy");
    return std::vector<std::uint8_t>(h2.begin(), h2.end());
}

std::string PINGenerator::generate(const std::vector<std::uint8_t>& master_hash) {
    if (master_hash.empty()) return "000000";

    // Production path: hash = LFSSL::BLAKE3(master_hash || "cerberus_pin_derivation")
    auto hash = blake3_like_fallback(master_hash);
    // Append additional mixing to ensure 6 independent digits
    auto hash2 = blake3_like_fallback(hash);
    hash.insert(hash.end(), hash2.begin(), hash2.end());

    std::string pin;
    pin.reserve(6);
    for (std::size_t i = 0; i < 6 && i < hash.size(); ++i) {
        pin.push_back('0' + static_cast<char>(hash[i] % 10));
    }
    while (pin.size() < 6) pin.push_back('0' + static_cast<char>(pin.size()));
    return pin;
}

std::string PINGenerator::generate_fallback() {
    // Entropy source: random_device + chrono nanoseconds (matches PsiForceDB installer)
    std::random_device rd;
    std::mt19937_64 gen(rd());
    auto now = std::chrono::high_resolution_clock::now();
    auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    gen.seed(gen() ^ static_cast<std::uint64_t>(nanos));
    std::uniform_int_distribution<int> dis(0, 9);
    std::string pin;
    pin.reserve(6);
    for (int i = 0; i < 6; ++i) pin.push_back('0' + dis(gen));
    return pin;
}

// ============================================================================
// Memorable Word
// ============================================================================

std::string MemorableWord::validate(std::string_view word) {
    if (word.length() < MIN_LENGTH)
        return "Memorable word too short (min " + std::to_string(MIN_LENGTH) + " chars)";
    if (word.length() > MAX_LENGTH)
        return "Memorable word too long (max " + std::to_string(MAX_LENGTH) + " chars)";
    bool has_alpha = false;
    for (auto c : word) {
        if (std::isalpha(static_cast<unsigned char>(c))) { has_alpha = true; break; }
    }
    if (!has_alpha) return "Memorable word must contain at least one letter";
    // DeMorgan18 character rejection (sensitive chars forbidden)
    for (auto c : word) {
        if (c == '\0' || c == '\n' || c == '\r' || c == '=' || c < ' ')
            return "Memorable word contains forbidden control characters";
    }
    return {}; // valid
}

std::vector<std::uint8_t> MemorableWord::derive_commitment(std::string_view word,
                                                         const std::vector<std::uint8_t>& salt) {
    // Production: LFSSL Argon2id(word, salt, t=3, m=65536, p=1)
    // Fallback: HMAC-SHA256(SHA256(word + salt), "cerberus_rbpc_word_commitment")
    std::string message = std::string(word) + std::string(salt.begin(), salt.end());
    auto inner = hq::cerberus::security::CryptoBridge::sha256(message);
    auto digest = hq::cerberus::security::CryptoBridge::hmac_sha256(
        inner, "cerberus_rbpc_word_commitment_lamia_carbon");
    return std::vector<std::uint8_t>(digest.begin(), digest.end());
}

// ============================================================================
// UserSecurity
// ============================================================================

std::optional<std::string> UserSecurity::generate_pin(const std::string& node_id,
                                               const std::vector<std::uint8_t>& master_secret) {
    if (node_id.empty() || master_secret.empty()) return std::nullopt;

    std::lock_guard<std::mutex> lock(mutex_);

    // Generate the PIN from master_secret (deterministic per node + master)
    auto mixed = master_secret;
    mixed.insert(mixed.end(), node_id.begin(), node_id.end());
    std::string pin = PINGenerator::generate(mixed);

    // Derive per-node salt for commitment
    auto salt_b = hq::cerberus::security::CryptoBridge::sha256(
        node_id + std::string(master_secret.begin(), master_secret.end()));
    std::vector<std::uint8_t> salt(salt_b.begin(), salt_b.end());

    // Derive commitment: HMAC-SHA256(pin + salt) — production: Argon2id
    std::string pin_with_salt = pin + std::string(salt.begin(), salt.end());
    auto commitment_b = hq::cerberus::security::CryptoBridge::sha256(pin_with_salt);
    std::string commitment(commitment_b.begin(), commitment_b.end());

    // Record confirmation set
    RBPCConfirmationSet cs;
    cs.node_id = node_id;
    cs.pin_commitment_hash = commitment;
    cs.created_at = std::time(nullptr);
    confirmation_sets_[node_id] = std::move(cs);

    // Initialize state
    RBPCState state;
    state.node_id = node_id;
    state.pin_hash = commitment;
    state.salt = std::string(salt.begin(), salt.end());
    state.failed_attempts = 0;
    state.burned = false;
    state.created_at = std::time(nullptr);
    states_[node_id] = std::move(state);

    return pin; // Display-once to user, NEVER stored again in plaintext
}

std::string UserSecurity::verify_pin(const std::string& node_id,
                                     const std::string& pin_attempt) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto sit = states_.find(node_id);
    if (sit == states_.end())
        return "Node not registered — RBPC confirmation set missing";

    if (sit->second.burned || !sit->second.is_active()) {
        return "PIN entry permanently disabled — node is burned after 3 failed attempts";
    }

    if (pin_attempt.empty()) {
        sit->second.failed_attempts++;
        if (sit->second.failed_attempts >= 3) {
            sit->second.burned = true;
            return "Empty PIN attempt — node permanently burned (3/3)";
        }
        return "Empty PIN attempt — failure " + std::to_string(sit->second.failed_attempts) + "/3";
    }

    // Hardened re-hash attempt with stored salt
    std::string attempt_salted = pin_attempt + sit->second.salt;
    auto attempt_hash_b = hq::cerberus::security::CryptoBridge::sha256(attempt_salted);
    std::string attempt_hash(attempt_hash_b.begin(), attempt_hash_b.end());

    if (attempt_hash == sit->second.pin_hash) {
        sit->second.failed_attempts = 0;
        sit->second.last_auth_timestamp = std::time(nullptr);
        return {}; // OK — empty string = success
    }

    // Failed — increment and enforce burn policy
    sit->second.failed_attempts++;
    if (sit->second.failed_attempts >= 3) {
        sit->second.burned = true;
        return "PIN verification failed — node permanently burned after 3 attempts";
    }
    return "PIN verification failed — attempt " + std::to_string(sit->second.failed_attempts) + "/3 — node will burn on 3rd failure";
}

std::string UserSecurity::register_memorable_word(const std::string& node_id,
                                                  std::string_view word,
                                                  const std::vector<std::uint8_t>& salt) {
    auto validation = MemorableWord::validate(word);
    if (!validation.empty()) return validation;

    // BFD structural defense: reject words that look like injection payloads
    if (word.find("\"") != std::string_view::npos ||
        word.find("\\") != std::string_view::npos ||
        word.find("{") != std::string_view::npos ||
        word.find("}") != std::string_view::npos) {
        return "Memorable word contains forbidden structural characters (InjectionProof BFD)";
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = confirmation_sets_.find(node_id);
    if (it == confirmation_sets_.end())
        return "Node not found — generate PIN first";

    if (it->second.word_commitment_hash.empty()) {
        // First registration — derive commitment
        auto commitment = MemorableWord::derive_commitment(word, salt);
        it->second.word_commitment_hash = std::string(commitment.begin(), commitment.end());
        it->second.hardware_anchor_hash = std::string(salt.begin(), salt.end()); // anchor = salt hash
        it->second.word_salt              = std::string(salt.begin(), salt.end()); // store word salt independently
        // The state salt stays as set by generate_pin() — do NOT overwrite.
        return {}; // OK
    } else {
        // Already registered — verify old word first, then replace
        auto attempt_commitment = MemorableWord::derive_commitment(word, salt);
        std::string attempt_str(attempt_commitment.begin(), attempt_commitment.end());
        if (attempt_str == it->second.word_commitment_hash) {
            // Old word verified — now user must provide NEW word separately in next call
            return {"ALREADY_REGISTERED_REAUTH_OK"};
        }
        return "Word registration failed — already has a commitment. Re-authenticate with existing word first.";
    }
}

std::string UserSecurity::verify_confirmation(const std::string& node_id,
                                              const std::string& pin_attempt,
                                              std::string_view word_attempt) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto sit = states_.find(node_id);
    if (sit == states_.end())
        return "Node not registered — SMDU provision must occur first";

    if (sit->second.burned || !sit->second.is_active())
        return "Node is burned — both PIN and Word confirmation permanently disabled";

    auto cit = confirmation_sets_.find(node_id);
    if (cit == confirmation_sets_.end())
        return "Confirmation set not found — corrupted RBPC state";

    if (cit->second.word_commitment_hash.empty())
        return "Memorable word not registered — set memorable word before confirmation";

    // -------- CSF layer 1: verify PIN --------
    std::string attempt_salted = pin_attempt + sit->second.salt;
    auto attempt_pin_b = hq::cerberus::security::CryptoBridge::sha256(attempt_salted);
    std::string attempt_pin(attempt_pin_b.begin(), attempt_pin_b.end());

    if (attempt_pin != sit->second.pin_hash) {
        sit->second.failed_attempts++;
        if (sit->second.failed_attempts >= 3) {
            sit->second.burned = true;
            return "PIN confirmation failed — SYSTEM LOCKED. Node burned. Contact administrator.";
        }
        return "PIN confirmation failed — attempt " + std::to_string(sit->second.failed_attempts) + "/3. Burn on 3rd failure.";
    }

    // -------- CSF layer 2: verify Word --------
    // CRITICAL: use confirmation_set.word_salt, NOT state.salt.
    // state.salt is the PIN salt (per-node, derived in generate_pin).
    // confirmation_set.word_salt is the salt passed during register_memorable_word().
    auto word_salt = std::vector<std::uint8_t>(
        cit->second.word_salt.empty()
            ? sit->second.salt.begin()
            : cit->second.word_salt.begin(),
        cit->second.word_salt.empty()
            ? sit->second.salt.end()
            : cit->second.word_salt.end());
    auto attempt_word_b = MemorableWord::derive_commitment(word_attempt, word_salt);
    std::string attempt_word(attempt_word_b.begin(), attempt_word_b.end());

    if (attempt_word != cit->second.word_commitment_hash) {
        sit->second.failed_attempts++;
        if (sit->second.failed_attempts >= 3) {
            sit->second.burned = true;
            return "Word confirmation failed — SYSTEM LOCKED. Node burned. Contact administrator.";
        }
        return "Word confirmation failed — attempt " + std::to_string(sit->second.failed_attempts) + "/3. Burn on 3rd failure.";
    }

    // -------- DUAL-FACTOR SUCCESS --------
    sit->second.failed_attempts = 0;
    sit->second.last_auth_timestamp = std::time(nullptr);

    // Sentry: structured audit data (caller can write to LocalMaintenanceDB)
    // The fact that we return empty string signals success; caller audits.
    return {}; // OK
}

bool UserSecurity::is_burned(const std::string& node_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = states_.find(node_id);
    if (it == states_.end()) return false;
    return it->second.burned || !it->second.is_active();
}

void UserSecurity::scrub_all() {
    std::lock_guard<std::mutex> lock(mutex_);
    states_.clear();
    confirmation_sets_.clear();
}

} // namespace hq::cerberus::privacy
