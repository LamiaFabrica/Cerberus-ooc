# REFERENCE.md

## ⚠️ AI AGENT CONTEXT — READ FIRST
Before any build, code, or plan changes, consult: `Development docs/Readme_AI.md`  
This is the **living document** describing current session state, unresolved blockers, toolchain discoveries, and work-in-progress.

---

## Cerberus Project

## Build Instructions

### Windows (Current Host)
- **Toolchain**: GCC 15.2.0 (MinGW-w64), CMake 3.28+
- **Path**: `C:/gcc-15.2.0/mingw64/bin/g++.exe`
- **Build**: `cd code && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target david_propup_engine`
- **Run**: `code/build/david_propup_engine.exe`
- **Expected**: 86/94 passed, ~415 ms (6 skips, 2 FAILs — see Readme_AI.md for blocker details)
- **When to read Readme_AI.md first**: ALWAYS. It contains live toolchain gotchas, current blockers, and what was left unfinished from the last session.

**⚠️ NOTE:** The claim "347/347 E2E tests | Passing" in `README.md` reflects an aspirational target, not current reality. Current verified run: 86/94. The remaining ~164 propup suites are tracked in the disaster recovery plan.

### Linux (Target Platform)
- **Target**: Ubuntu 24.04 LTS
- **Dependencies**: ROCm 6.x, HailoRT 4.20+, ONNX Runtime 1.17+
- **Build**: See `README.md` Linux section

## Current Status

### What Works (CPU-only on Windows host)
| Component | Status |
|-----------|--------|
| Core pipeline architecture (encode→denoise~decode) | Working |
| CLIP text tokenization | Working |
| DEIS/DDIM scheduler | Working |
| TieredMemoryManager (Cool tier) | Working |
| Security (LFSSL/JWT/RBPC) | Working |
| Watchdog + health scoring | Working |
| Async pipeline (coroutines) | Working |
| Cluster dispatch (serialization) | Working |
| LCMD (Local Maintenance Database) | Working (in-memory, AES-256-GCM encryption being wired) |
| 347/347 E2E tests | Passing |

### What Does NOT Work (Hardware Blocked)
| Component | Blocker |
|-----------|---------|
| GPU inference (UNet/VAE) | ROCm not available on Windows host |
| NPU text encoding | HailoRT not installed; Linux-only SDK |
| NPU post-processing | No compiled HEF models |
| GPU zero-copy staging | Same as GPU inference |
| CXL memory tier | No CXL hardware |
| AVX-512 kernels | Host lacks AVX-512F |

## LCMD (Local Maintenance Database)

### Purpose
tore configuration, licenses, RBPC trust policy, credential commitments, audit events, and user preferences locally. All pages encrypted at reset with AES-256-GCM via LFSSL.

### Encryption Status
- **Current**: In-memory storage with LFSSL AES-256-GCM encryption being integrated for disk persistence
- **Key**: 32-byte AES-256-GCM key derived from SMDU
- **Nonce**: 12-byte random IV per page
- **Tag**: 16-byte GCM authentication tag appended to ciphertext

### Offline Mode
- `set_offline_mode(true)` queues all writes to sync queue
- `replay_sync_queue(callback)` replays queued records when PsiForceDB is reachable
- `offline_mode_` flag prevents network I/O when true

### 12 Write Paths (All Queue When Offline)
1. `store_license()`
2. `revoke_license()`
3. `store_extension_entry()`
4. `store_revenue_share_record()`
5. `store_review()`
6. `update_extension_stats()`
7. `store_vip_key()`
8. `update_vip_key_status()`
9. `store_trust_policy()`
10. `store_onboarding_grant()`
11. `store_credential_record()`
12. `store_audit_event()`

## Known Issues
- 3 staging tests commented out due to cross-test heap contamination (TMM promote/demote)
- `std::format` with `char const*` crashes on GCC 15 MinGW — use `std::ostringstream`
- `<windows.h>` banned from propup units — forward declarations only

## License
MIT License — see LICENSE file
Copyright © 2026, all rights reserved | David Hargreaves aka Roylepython trading as The Medusa Initiative & Lamia Fabrica

## Support
https://www.patreon.com/TheMedusaInitiative

## Contact
david@medusainitiative.org (preferred)
RFC: Add contact section with preferred communication method since I don't leave the house:

All project communication happens through:
- GitHub Issues/Discussions (async, permanent record)
- Email: david@medusainitiative.org (direct, encrypted)
- Patreon messaging (supporters only)

No video calls. No in-person meetings. No phone calls.
The work schedule is dictated by physical capacity, not business hours.