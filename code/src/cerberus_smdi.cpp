/// @file cerberus_smdi.cpp
/// @copyright Copyright © 2026, all rights reserved | David Hargreaves aka Roylepython trading as The Medusa Initiative & Lamia Fabrica
///
/// SMDU implementation — orchestrates LFSSL delegation, never implements primitives.
///
/// © 2026 D Hargreaves | LamiaFabrica Software

#include "hq/cerberus_smdi.hpp"

#include <cstring>

namespace hq::cerberus::privacy {

// ============================================================================
// Sentinel — used when LFSSL is not available at compile time
// ============================================================================

bool SmdiSentinel::available() noexcept {
    return false;
}

std::string SmdiSentinel::unavailable_reason() noexcept {
    return "SMDU key material delegation relies on LFSSL (Argon2id / Kyber / AES-256-GCM). "
           "LFSSL is linked from PsiForceDB's compiled security library. "
           "On this Windows build host, LFSSL.lib is not yet available; "
           "SMDU operates in sentinel mode until the PsiForceDB Windows runtime library is linked.";
}

std::string SMDU::unavailable_reason() noexcept {
    return SmdiSentinel::unavailable_reason();
}

// ============================================================================
// SMDU — stubs until LFSSL is linked
// ============================================================================

std::optional<SMDU> SMDU::provision(
    std::string_view passphrase,
    const std::vector<std::uint8_t>& master_salt,
    const std::vector<std::uint8_t>& provisioned_pub) {
    (void)passphrase;
    (void)master_salt;
    (void)provisioned_pub;
    // LFSSL not linked — cannot perform real Argon2id/Kyber provisioning.
    // Return nullopt so callers know to use sentinel mode.
    return std::nullopt;
}

std::optional<SMDU> SMDU::unlock(
    std::string_view passphrase,
    const std::vector<std::uint8_t>& master_salt,
    const WrappedKey& wrapped) {
    (void)passphrase;
    (void)master_salt;
    (void)wrapped;
    return std::nullopt;
}

void SMDU::lock() {
    scrub_(session_private_key_);
    session_active_ = false;
}

std::optional<std::vector<std::uint8_t>> SMDU::derive_db_key() const {
    if (!session_active_) return std::nullopt;
    // LFSSL delegation stub: would derive AES-256-GCM sub-key from
    // Kyber shared secret via HKDF-SHA3-256. Not implemented here
    // because AES-GCM key derivation requires LFSSL Kyber encaps.
    return std::nullopt;
}

std::optional<std::vector<std::uint8_t>> SMDU::derive_e2e_key() const {
    if (!session_active_) return std::nullopt;
    // LFSSL delegation stub: would encapsulate against master public key.
    return std::nullopt;
}

std::optional<WrappedKey> SMDU::export_wrapped_key() const {
    if (wrapped_private_key_.ciphertext.empty()) return std::nullopt;
    return wrapped_private_key_;
}

void SMDU::scrub_(std::vector<std::uint8_t>& buf) const noexcept {
    if (buf.empty()) return;
    volatile std::uint8_t* p = buf.data();
    for (std::size_t i = 0; i < buf.size(); ++i) p[i] = 0;
    buf.clear();
}

} // namespace hq::cerberus::privacy
