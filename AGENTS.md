# Cerberus — AGENTS.md

## Goal
Cerberus registers as a PsiForceDB MultiModelExtension (type "EXT", model "inference") and reuses PsiForceDB's graph engine, security (JWT/Kyber/AES/PQC), datatypes, and web tier (MedusaServ/BertieBot) instead of building redundant standalone layers. Expand David Propup Engine to 99+ tests.
- **Current: 160/160 passed, ~1354 ms, zero blockers.**

## Constraints & Preferences
- **No standalone database in Cerberus, but Local Cerberus Maintenance Database (LCMD) for offline operation.** LCMD is a carbon copy of PsiForceDB's MaintenanceDatabase — encrypted at rest with AES-256-GCM (delegated to LFSSL), stores RBPC state, licenses, audit events, preferences. Works offline and online identically, syncs to PsiForceDB on reconnect.
- **Privacy is the #1 concern.** End-to-end encryption down to intranet and device level. LFSSL (Kyber + Argon2id + AES-256-GCM) works online AND offline. No plaintext secrets stored anywhere.
- **RBPC model enforced for ALL destructive actions.** System-issued 6-digit PIN + user-memorized word. Minimum 1 character from EACH required for destructive operations. 3 failed attempts = permanent burn (SYSTEM LOCKED). 30-second confirmation window.
- **JWT runs as CSF/BFD/InjectionProof/Sentry.** Not convenience — security perimeter. Every validation event audited.
- **One ecosystem.** All projects under `C:\McMaker Projects\Projects\` are part of the David Propup ecosystem; Cerberus consumes PsiForceDB, PFQL, MedusaServ, and BertieBot APIs.
- **Cerberus is a small system within a BIG ecosystem.** It does NOT govern FortressAuth, AuditLog, JWT, or session management. It consumes them from PsiForceDB.
- **Carbon copy PsiForceDB patterns.** Copy from PsiForceDB `.cpp` implementations as source of truth. No reinvention. Boundary contract (`cerberus_boundary_contract.hpp`) prevents accidental secret replication.
- **No empty none-feature-complete variables.** Every field consumed. `-1.0f` sentinels and `unavailable_reason()` mandatory.
- **No external file dependencies in propup tests.** Synthetic tests only; no parsing real GGUF files in the test suite.
- **Windows/MinGW build host** (G18 Strix, GCC 15.1.0 from WinLibs MCF, GCC 15.2.0 hardcoded in CMakeLists.txt), MinGW-w64, AVX2-capable.
- **David Propup Engine is the KPI.** 160/160 passing. Zero tolerance for regressions.
- **`<windows.h>` is BANNED from propup compilation units.** Any Windows API needed for runtime DLL loading must use **minimal forward declarations** (e.g., `extern "C" __declspec(dllimport) HMODULE LoadLibraryA(const char*);`) or be isolated to a separate `.cpp` translation unit. This prevents `winerror.h` macro pollution (e.g., `ERROR_NOT_FOUND`) from corrupting Cerberus enums (`CerberusOpcode`, `SlipstreamMessageType`) on MinGW GCC 15.x.

## Progress
### Done
- **150/150 David Propup Engine tests passing** (~1354 ms):
  - **Core:** tiered memory, AVX2 matmul, quantized kernels, graph engine, command layer, ANBP gateway, slipstream, metro, glow engine.
  - **Glow Engine (10):** `propup_glow_bond_creation`, `reinforcement`, `decay`, `hot_path_query`, `best_next_hop`, `catchphrase_exact`, `catchphrase_fuzzy`, `stats_integrity`, `reset`, `attenuation`.
  - **Adversarial robustness (5):** `propup_adversarial_malformed_anbp`, `invalid_token`, `permission_escalation`, `slipstream_overflow`, `metro_empty_payload`.
  - **GGUF parser synthetic (2):** `propup_gguf_synthetic_header`, `propup_gguf_synthetic_tensor_info`.
  - **PsiForceDB Extension Integration (15):** `propup_psiforcedb_extension_init`, `load_unload`, `inference_query`, `status_query`, `stats`, `validate`, `pfql_routing`, `gguf_loader`, `telemetry`, `health_check`, `transaction_reject`, `coordinator_routing`, `factory`, `dependencies`, `metadata`.
  - **Extension Edge-Cases (4):** `propup_psiforcedb_extension_validate_edge_cases` (oversized, unbalanced quotes, empty), `error_counting`, `detail_helpers`, `glow_integration`.
  - **PsiForceDB Graph Bridge (2):** `propup_psiforcedb_graph_bridge_topology`, `propup_psiforcedb_graph_bridge_pfql_rows`.
  - **AVX-512 Dispatch (2):** `propup_kernel_avx512_detect`, `propup_kernel_matmul_avx512_dispatch`.
  - **Additional Native Kernels (2):** `propup_kernel_relu`, `propup_kernel_sigmoid`.
  - **Security Bridge (5):** `propup_security_sha256`, `propup_security_hmac_sha256`, `propup_security_pbkdf2_sha256`, `propup_security_aes256_gcm_sentinel`, `propup_security_pqc_sentinel`.
  - **Real PsiForceDB Header Compile Proof (1):** `propup_psiforcedb_extension_real_header_compile`.
  - **Privacy/RBPC/JWT/LCMD (12):** `propup_privacy_local_maintenance_db`, `propup_privacy_pin_generation`, `propup_privacy_pin_burn_policy`, `propup_privacy_word_commitment`, `propup_privacy_dual_factor_confirmation`, `propup_privacy_jwt_session`, `propup_privacy_jwt_concurrent`, `propup_lcmd_extension_entry`, `propup_lcmd_revenue_share`, `propup_lcmd_vip_keys`, `propup_lcmd_onboarding_grant`, `propup_lcmd_offline_sync_ready`.
  - **LFSSL DLL Crypto (4):** `propup_lfssl_dll_smoke`, `propup_lfssl_dll_sha256`, `propup_lfssl_dll_hmac`, `propup_lfssl_dll_aes256gcm`.
  - **LFSSL DLL PQC (2):** `propup_lfssl_dll_kyber`, `propup_lfssl_dll_dilithium`.
  - **LFSSL DLL Memory-Hard (2):** `propup_lfssl_dll_argon2id`, `propup_lfssl_dll_argon2id_verify`.
- **Real `cerberus_lfssl.dll` built and verified** — valid PE (now 600+ KB with PQC + Argon2), exports 12 C-linkage symbols including:
  - SHA-256, HMAC-SHA256, PBKDF2-SHA256, AES-256 block, AES-256-GCM encrypt/decrypt, random bytes.
  - **Kyber KEM** (`cerberus_lfssl_kyber_keypair`, `_encapsulate`, `_decapsulate` — all three levels 512/768/1024).
  - **Dilithium signatures** (`cerberus_lfssl_dilithium_keypair`, `_sign`, `_verify` — all three levels 2/3/5).
  - **Argon2id memory-hard password hashing** (`cerberus_lfssl_argon2id`, `_argon2id_verify`).
- **AES-256-GCM decrypt authenticity fixed in LFSSL** (`aes256_gcm_hardware.cpp`):
  - `encrypt_aesni` tag mask was using the wrong counter (post-increment instead of `J0`).
  - `decrypt_aesni` had no GCM tag verification at all — effectively CTR mode with zero auth.
  - Fixed by extracting shared `gcm_ghash()` and `gcm_encrypt_ctr()` helpers, using `J0` for tag mask on both sides, and performing volatile constant-time tag comparison in decrypt before returning plaintext.
  - Propup tamper check now **rejects** flipped-bit ciphertext (returns `-5` auth failure).
- **Dilithium secret key size bug fixed:** `dilithium.hpp` listed 2528 bytes for DILITHIUM2, but the reference implementation requires **2560**. The wrapper now uses reference sizes, matching the actual compiled objects.
- **Cerberus dynamic loader wired:** `LfsslSentinel::aes256_gcm_available()`, `kyber_available()`, `dilithium_available()` now probe `cerberus_lfssl.dll` at runtime. Sentinels return `true` when DLL present, with updated `unavailable_reason()` reflecting linkage state.
- **Windows header macro pollution fixed:** `winerror.h` defines `ERROR_NOT_FOUND` as `1168`, which broke `enum class CerberusOpcode`. Fixed by replacing `#include <windows.h>` with minimal forward declarations in `cerberus_psiforcedb_security.cpp` and `david_propup_engine.cpp`.
- **AGENTS.md updated** with current KPI, build artifacts, and blockers.

### In Progress
- Nothing.

### Blocked
- **Hot tier** still stub (no HIP/CUDA runtime linked).
- **OpenVINO NPU** unavailable on build host.
- **CUDA backend** unavailable on build host.
- **AVX-512 execution** — dispatch implemented, host lacks AVX-512F hardware.
- **CerberusExtension wired into actual PsiForceDB runtime** — compiles against real headers (`RealCerberusExtension`), but linking into `MultiModelCoordinator::loadExtension()` requires PsiForceDB Windows `.lib`/`.dll` export. PsiForceDB has **never been built on this Windows host** (no build artifacts exist). Blocked until PsiForceDB produces a compiled Windows library exporting `MultiModelExtension` symbols.

## Key Decisions
- **PsiForceDB owns the heart; Cerberus is the integrator.** Proprietary logic stays in PsiForceDB/LamiaFabrica. Cerberus only sees the integration surface.
- **`<windows.h>` is banned from propup compilation.** Forward declarations or isolated translation units only. Prevents `winerror.h` macro pollution on MinGW GCC 15.x.
- **DLL is per-process loaded by propup, not linked at compile time.** No build changes required to add crypto features — just drop a new DLL and the next test run picks it up. Propup tests are runtime-probed.
- **PQC key sizes come from reference `api.h`/`params.h`, not C++ wrapper headers.** `dilithium.hpp` was wrong (2528 vs 2560 for mode 2). Reference sizes are the source of truth.

## Next Steps
1. **Build PsiForceDB Windows `.lib`/`dll`** so Cerberus can link `RealCerberusExtension` into `MultiModelCoordinator::loadExtension()`.
2. **Hot tier / CUDA / AVX-512** — link when hardware/drivers available.
3. **Add more propup tests** to reach 200+ (kpi was 99+, now 150/150).
4. **Resolve GCC 15.1.0 vs 15.2.0 CMake mismatch** — CMakeLists.txt hardcodes a path that doesn't exist.

## Critical Context
- **150/150 passed, ~1354 ms.** KPI is strict.
- **`cerberus_lfssl.dll` is real** at `lfssl_bridge/cerberus_lfssl.dll` (600+ KB PE, MZ header `4D 5A`). 
- **AES-256-GCM now has full auth.** Tag verification with constant-time compare.
- **Dilithium2 verify round-trip confirmed.** Signature fixed by using correct SK size.
- **PsiForceDB has never been built on this host.** No `.lib`/`.dll`/`.a` artifacts. Root CMakeLists.txt requires building into `Binaries/Windows` subdir.
- **Build host:** G18 Strix, Win11, MinGW GCC 15.1.0 (WinLibs MCF), AVX2-capable, AVX-512F absent.

## Relevant Files
- `code/include/hq/cerberus_psiforcedb_security.hpp/.cpp` — CryptoBridge + LfsslSentinel (runtime DLL probing).
- `code/include/hq/cerberus_local_maintenance_db.hpp/.cpp` — LCMD: full MaintenanceDatabase carbon copy.
- `code/include/hq/cerberus_user_security.hpp/.cpp` — UserSecurity: RBPC PIN + memorable word + burn policy.
- `code/include/hq/cerberus_jwt_session.hpp/.cpp` — JWTSession: expanded SessionConfig, HMAC-SHA256.
- `code/include/hq/cerberus_psiforcedb_extension.hpp` — Standalone extension interface + `ExtensionFactory` typedef.
- `code/src/cerberus_psiforcedb_real_integration.cpp` — Real header compile proof (`RealCerberusExtension`).
- `code/src/david_propup_engine.cpp` — 148-test suite.
- `code/CMakeLists.txt` — Auto-detects PsiForceDB headers, conditional real-header compilation.
- `lfssl_bridge/cerberus_lfssl.dll` — Real LFSSL PE DLL with 12 C exports (SHA-256, HMAC, PBKDF2, AES-256, AES-256-GCM, random, Kyber, Dilithium, Argon2id).
- `lfssl_bridge/cerberus_lfssl_wrapper.cpp` — C-export wrapper.
- `lfssl_bridge/build_dll.bat` — Batch build script for full DLL with PQC + Argon2id.
- `../PsiForceDB/Src/Database/Intake/include/multimodel/extension_interface.hpp` — PsiForceDB real extension base class.
- `../LFSSL - Lamia Fabrica SSL/src/crypto/aes256_gcm_hardware.cpp` — Fixed GCM decrypt auth.
- `../LFSSL - Lamia Fabrica SSL/phc-argon2/` — Argon2 reference implementation (public domain).
