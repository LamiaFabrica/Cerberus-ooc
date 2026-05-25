# Cerberus — AGENTS.md

## Goal
Cerberus registers as a PsiForceDB MultiModelExtension (type "EXT", model "inference") and reuses PsiForceDB's graph engine, security (JWT/Kyber/AES/PQC), datatypes, and web tier (MedusaServ/BertieBot) instead of building redundant standalone layers. Expand David Propup Engine to 99+ tests.
- **Current: 144/144 passed, ~337 ms, zero blockers.**

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
- **Windows/MinGW build host** (G18 Strix, GCC 15.2.0).
- **David Propup Engine is the KPI.** 141/141 passing. Zero tolerance for regressions.

## Progress
### Done
- **141/141 David Propup Engine tests passing** (~328 ms):
  - **Original 82**: tiered memory, AVX2 matmul, quantized kernels, graph engine, command layer, ANBP gateway, slipstream, metro, glow engine.
  - **Glow Engine (10):** `propup_glow_bond_creation`, `reinforcement`, `decay`, `hot_path_query`, `best_next_hop`, `catchphrase_exact`, `catchphrase_fuzzy`, `stats_integrity`, `reset`, `attenuation`.
  - **Adversarial robustness (5):** `propup_adversarial_malformed_anbp`, `invalid_token`, `permission_escalation`, `slipstream_overflow`, `metro_empty_payload`.
  - **GGUF parser synthetic (2):** `propup_gguf_synthetic_header`, `propup_gguf_synthetic_tensor_info`.
  - **PsiForceDB Extension Integration (15):** `propup_psiforcedb_extension_init`, `load_unload`, `inference_query`, `status_query`, `stats`, `validate`, `pfql_routing`, `gguf_loader`, `telemetry`, `health_check`, `transaction_reject`, `coordinator_routing`, `factory`, `dependencies`, `metadata`.
  - **Extension Edge-Cases (4):** `propup_psiforcedb_extension_validate_edge_cases` (oversized, unbalanced quotes, empty), `error_counting`, `detail_helpers` (trim, lowercase, parse_bool, stable_fingerprint, model_from_type), `glow_integration` (GLOW query after INFERENCE triggers GlowEngine recording).
  - **PsiForceDB Graph Bridge (2):** `propup_psiforcedb_graph_bridge_topology`, `propup_psiforcedb_graph_bridge_pfql_rows`.
  - **AVX-512 Dispatch (2):** `propup_kernel_avx512_detect` (runtime cpuid, reports avx2=yes avx512f=no on this host), `propup_kernel_matmul_avx512_dispatch` (64x64 synthetic dispatch with blocked fallback on non-AVX512 builds).
  - **Additional Native Kernels (2):** `propup_kernel_relu`, `propup_kernel_sigmoid`.
  - **Security Bridge (5):** `propup_security_sha256`, `propup_security_hmac_sha256`, `propup_security_pbkdf2_sha256`, `propup_security_aes256_gcm_sentinel`, `propup_security_pqc_sentinel`.
  - **Real PsiForceDB Header Compile Proof (1):** `propup_psiforcedb_extension_real_header_compile` — Cerberus now compiles and links against the **real** PsiForceDB `extension_interface.hpp`.
  - **Privacy/RBPC/JWT/LCMD (11):** `propup_privacy_local_maintenance_db`, `propup_privacy_pin_generation`, `propup_privacy_pin_burn_policy`, `propup_privacy_word_commitment`, `propup_privacy_dual_factor_confirmation`, `propup_privacy_jwt_session`, `propup_privacy_jwt_concurrent`, `propup_lcmd_extension_entry`, `propup_lcmd_revenue_share`, `propup_lcmd_vip_keys`, `propup_lcmd_onboarding_grant`.
- **LFSSL shim headers created** for Windows compatibility:
  - `code/include/lfssl/crypto/random.hpp` — `SecureRandom` + `secure_random_bytes()` shim.
  - `code/include/lfssl/crypto/aes256_gcm.hpp` — `AES256GCM` class declaration.
  - `code/include/lfssl/crypto/SecureRandom.hpp` — compatibility forwarder.
  - `code/include/psiforcedb/security/crypto_audit_log.hpp` — `AuditLoggedSecureRandom` + `CryptoAuditLog` stubs.
- **Thin LFSSL_Native_Crypto bridge** (`cerberus_psiforcedb_security.hpp/.cpp`): wraps ONLY the inline-safe subset (SHA256, HMAC-SHA256, PBKDF2-SHA256). `LfsslSentinel` documents that AES-256-GCM, Kyber, Dilithium are delegated to PsiForceDB's LFSSL library at runtime.
- **LocalMaintenanceDB (LCMD) fully expanded** to match PsiForceDB `MaintenanceDatabase` interface:
  - **TrustPolicy** expanded from 14 → 24 fields with `default_policy()`, `keeps_local_authority()`, `to_map()`, `from_map()`.
  - **All 11 missing methods implemented**: extension entries, revenue share, reviews, extension stats, VIP keys, onboarding grants.
  - **RBPC credential record sealing**: auto-injects `record_class`, `hash_suite`, `pqc_profile`, `hardware_binding`, `step_up_model`, `plaintext_storage=forbidden`.
  - **Sync queue** for deferred replay to PsiForceDB.
- **Graph bridge types expanded** to PsiForceDB canonical:
  - Added `PropertyType`, `ElementState`, `Direction`, `PropertyValue` variant, `Path`, and graph constants (`INVALID_NODE_ID`, `MAX_PATH_LENGTH`).
  - `GraphNode` and `GraphEdge` now include `state`, `version`, `created_tx`, `deleted_tx`.
  - `GraphTopology` gained `get_outgoing_edges`, `get_incoming_edges`, `get_neighbors`, `unavailable_reason()`.
- **ExtensionFactory typedef** added to `cerberus_psiforcedb_extension.hpp`.
- **SessionConfig expanded** with all PsiForceDB canonical fields (`jwt_secret_key`, `issuer`, `access_token_ttl`, `refresh_token_ttl`, `enable_refresh_tokens`…`revocation_endpoints`). `unavailable_reason()` added and inlined.
- **JWT base64url_decode bug fixed**: lookup table was corrupted (shifted elements), which silently corrupted decoded payloads. Fixed by rewriting `tbl[256]` with 256 explicit entries matching RFC 4648.
- **Word salt isolation fixed**: `verify_confirmation()` now reads `word_salt` from `RBPCConfirmationSet` instead of the PIN `state.salt`, preventing cross-factor salt contamination.
- **CMakeLists.txt updated**:
  - Auto-detects real `extension_interface.hpp`; sets `PSIFORCEDB_HEADERS_FOUND` and `CERBERUS_USE_REAL_PSIFORCEDB_HEADERS`.
  - PsiForceDB include dirs wired into `um790_includes`.
  - GCC 15.2.0 hardcoded.
  - `cerberus_psiforcedb_real_integration.cpp` added conditionally.
- **Blocked matmul bugfix**: `kernel_matmul_blocked` was passing `B_panel.data() + 0` regardless of `jj`; fixed to `B_panel.data() + jj`.
- **Build passes all targets**: `david_propup_engine.exe`, `libum790_pipeline.a`, `cerberus_npu.dll`, `cerberus_server.exe`.

### In Progress
- Nothing — all P0 audit gaps closed.

### Blocked
- **Hot tier** still stub (no HIP/CUDA runtime linked).
- **OpenVINO NPU** unavailable on build host.
- **CUDA backend** unavailable on build host.
- **AVX-512 execution** — dispatch implemented, host lacks AVX-512F hardware.
- **PQC (Kyber/Dilithium)** — requires LFSSL compiled library. Linux `.so` available; Windows DLL pending. Cerberus delegates to PsiForceDB.
- **AES-256-GCM page encryption** — implementation is in LFSSL library, not inline. Cerberus operates in sentinel mode with HMAC-SHA256 fallback.
- **LFSSL DLL is a corrupted/placeholder** (`lfssl.dll` from `MedusaMail_1.0.0\lib\` is not a real PE). Quarantined to `lfssl.dll.CORRUPTED_PLACEHOLDER`. A real Windows LFSSL build is required for runtime linking.
- **CerberusExtension wired into actual PsiForceDB runtime** — compiles against real headers, but linking into `MultiModelCoordinator::loadExtension()` requires PsiForceDB Windows `.lib`/`.dll` export. Blocked until PsiForceDB produces linked LFSSL runtime.

## Key Decisions
- **PsiForceDB is the database, graph engine, security layer, and web tier.** Cerberus does not bring its own.
- **Cerberus does NOT implement FortressAuth, AuditLog, or SessionToken.** These are PsiForceDB domains.
- **CerberusExtension type = "EXT", model_type = "inference".** Query routing: `INFERENCE`/`COMPILE` → `CerberusRuntime::execute_command()`, `STATUS` → telemetry, `GLOW` → `GlowEngine::stats()`, `GGUF` → synthetic tensor metadata, `TELEMETRY` → MedusaServ/BertieBot payload.
- **GGUF parser is runtime infrastructure, not test infrastructure.** Propup tests are synthetic (in-memory struct validation). Real GGUF file parsing is for production load paths.
- **No middleware.** Cerberus hooks into PsiForceDB's existing security (JWT, Kyber, AES, PQC) without building parallel auth.
- **Conditional real-header compilation** (`CERBERUS_USE_REAL_PSIFORCEDB_HEADERS`). When real PsiForceDB headers present: `RealCerberusExtension` compiled against actual `MultiModelExtension`. When absent: standalone replica fallback.

## Next Steps
1. **Build real LFSSL Windows DLL** — produce a PE `.dll` with exported AES-GCM, Argon2id, Kyber, Dilithium symbols. Replace quarantined placeholder.
2. **Link CerberusExtension into PsiForceDB runtime** — once PsiForceDB Windows build produces `.dll`/`.lib` exporting `MultiModelExtension` symbols, link Cerberus and register via `MultiModelCoordinator::loadExtension()`.
3. **Wire into PsiForceDB MultiModelCoordinator** — register `RealCerberusExtension` and confirm `INFERENCE` queries route correctly.
4. **Hot tier / CUDA / AVX-512** — link runtime backends when build host has the hardware/drivers.

## Critical Context
- **141/141 passed, ~328 ms.** KPI is strict.
- **PsiForceDB MultiModel architecture:** `ExtensionInterface` base class, `MultiModelCoordinator` routes queries, `GraphModelExtension` handles property graphs. CerberusExtension follows this pattern.
- **LFSSL_Native_Crypto.hpp:** header-only for SHA256/HMAC/PBKDF2. AES256GCM, Kyber, Dilithium declared but not defined inline.
- **Athenea GGUF files:** `lamia-fabrica-athenea-Q4_K_M.gguf` (2.50 GB), `Q5_K_M.gguf` (2.89 GB), `IQ4_NL.gguf` (2.39 GB). Rebranded Qwen3 4B Coding.
- **Build host:** G18 Strix, Win11, MinGW GCC 15.2.0, AVX2-capable, AVX-512F absent.

## Relevant Files
- `code/include/hq/cerberus_local_maintenance_db.hpp/.cpp` — LCMD: full MaintenanceDatabase carbon copy.
- `code/include/hq/cerberus_user_security.hpp/.cpp` — UserSecurity: RBPC PIN + memorable word + burn policy.
- `code/include/hq/cerberus_jwt_session.hpp/.cpp` — JWTSession: expanded SessionConfig, HMAC-SHA256, sentinels for PQC/hardware/fingerprinting.
- `code/include/hq/cerberus_first_run.hpp/.cpp` — FirstRun: SMDU provision, PIN+word capture, hardware anchor, JWT secret.
- `code/include/hq/cerberus_smdi.hpp/.cpp` — SMDU identity: Kyber keypair sentinel stubs.
- `code/include/hq/cerberus_boundary_contract.hpp` — Boundary contract.
- `code/include/hq/cerberus_psiforcedb_extension.hpp` — Standalone extension interface + `ExtensionFactory` typedef.
- `code/src/cerberus_psiforcedb_extension.cpp` — CerberusExtension implementation.
- `code/include/hq/cerberus_psiforcedb_graph_bridge.hpp/.cpp` — GraphTopology, GraphNode, GraphEdge with PsiForceDB canonical fields + `unavailable_reason()`.
- `code/include/hq/cerberus_psiforcedb_security.hpp/.cpp` — CryptoBridge (SHA256/HMAC/PBKDF2) + LfsslSentinel.
- `code/src/cerberus_psiforcedb_real_integration.cpp` — Real header compile proof.
- `code/include/hq/cerberus_glow_engine.hpp/.cpp` — GlowEngine.
- `code/include/hq/cerberus_gguf_parser.hpp/.cpp` — GGUF v3 parser.
- `code/src/david_propup_engine.cpp` — 141-test suite.
- `code/CMakeLists.txt` — Auto-detects PsiForceDB headers, GCC 15.2.0, conditional real-header compilation.
- `../PsiForceDB/Src/Database/Intake/include/multimodel/extension_interface.hpp` — PsiForceDB real extension base class.
- `../PsiForceDB/Src/Database/Intake/include/multimodel/maintenance_database_interface.hpp` — PsiForceDB MaintenanceDatabase interface (source of truth for LCMD).
- `../PsiForceDB/Src/Database/Intake/include/psiforcedb/graph/graph_types.hpp` — PsiForceDB canonical graph types (carbon-copied into bridge).
