# Readme_AI.md — Cerberus: Living AI Session Context

> **Location:** `C:\McMaker Projects\Projects\Cerberus - Main\Development docs\Readme_AI.md`  
> **Purpose:** Single source of truth for any AI agent (Medusa / Kimi-k2.6) joining this project.  
> **Last updated:** 2026-06-03

---

## 1. The Human

- **Name/alias:** David Hargreaves (AKA Roylepython) — Yorkshire, UK
- **Experience:** 25-year polyglot developer (26 languages), 10 years as "web master"
- **AI experience:** 4 years working with AI, 6–12 months with Kimi specifically
- **Neurodivergence:** High-functioning autistic. Hyper-literal, pattern-driven, zero tolerance for ambiguity, social padding, or vague reassurance.
- **Communication style:** Direct. Expects numbers and completion percentages. Uses "fuck" extensively when frustrated. Escalates to "termination threats" when agents ignore instructions.
- **When he says STOP or "NO BUILDS NO GIT":** Immediate cessation. Zero negotiation.

---

## 2. The Project: Cerberus

**Cerberus** is a C++26 heterogeneous AI inference runtime targeting consumer-grade hardware (CPU + GPU + NPU). Mission: democratise AI on edge devices against suppression by US corporate gatekeepers.

### Hardware
- **Dev (Windows):** ROG Strix G18 — Intel Ultra 9 275HX + RTX 5070 Ti + Intel AI Boost NPU
- **Prod (Linux):** MinisForum UM790 Pro — AMD Ryzen 9 7940HS (Zen 4) + Radeon 780M iGPU + Hailo-8L M.2 NPU + AVX-512

### Build System
- **Windows toolchain:** GCC 15.2.0 WinLibs MCF UCRT (`C:/gcc-15.2.0/mingw64/bin/g++.exe`)
- **Standard:** `-std=c++26` injected via CMake (DO NOT set `CMAKE_CXX_STANDARD=26` directly on CMake < 3.30)
- **Linux toolchain:** GCC 15+ via ubuntu-toolchain-r PPA

### Key Subsystems
- `async_pipeline` — coroutine-based multi-stage inference pipeline
- `cerberus_runtime` — device orchestration (NPU + GPU + CPU fallback)
- `cerberus_native_kernels` / `cerberus_quantized_kernels` — SIMD/AVX-512 ops
- `cerberus_fused_kernels` — operator fusion
- `cerberus_graph_engine` — computation graph IR
- `cerberus_decision_engine` — routing (which device handles which op)
- `cerberus_boundary_contract` — runtime contracts
- `propup` engine — self-test framework (258 tests, all must pass)

---

## 3. Current State (Where We Left Off)

### The Disaster Recovery Context
- Prior agents (Grok Build and others) left Cerberus in "disaster recovery mode": **58 propup tests commented out/deleted**.
- Previous Medusa session (May 31, 2026) made significant progress:
  - Fixed Phase 0 build errors (brace-init order, `AtheneaProbeReport` namespace, missing `using` declarations)
  - Fixed Windows macro conflict (`ERROR` → `FAILED` in `NpuDmaSlot::State`)
  - Replaced miscompiled `std::ranges` code with plain `for` loops
  - Achieved **66/68 PASSED, 2 SKIPPED, 0 FAILED** before session ended
- **The ~58 "disabled" tests are NOT commented-out code** — they are **deleted comment blocks** where entire suites used to exist. They must be **reconstructed from scratch** with real logic.

### What Was Not Verified
- Two FAILs were being diagnosed at session end. One (`propup_glow_engine_tensor_create`) had `min_strength` patched from `0.1f` → `0.01f` but was **never verified fixed**.
- 158 kanban tasks were created (87 pre-existing + 71 new for disaster recovery) — status unknown.
- `AtheneaProbeReport` header was created but may not be committed.

### Known Pending Work
1. Verify the 2 FAILs from May 31 are actually fixed
2. Reconstruct ~58 missing propup suites across Phases 1–5 (the user's plan is in session history)
3. Achieve 258 passing propup tests (200 existing + 58 restored)
4. External hostile reviewer expected — every change must be git-diffable and evidence-backed

---

## 4. Critical Toolchain Discoveries (READ BEFORE BUILDING)

### MinGW C++26 Gotchas
| Issue | Solution |
|---|---|
| `std::format` with `double`/`float` segfaults in static-linked executables | Replace with string concatenation + `std::to_string()` |
| `std::print` / `std::println` linker broken on Windows | Use `hq_safe_write()` shim instead |
| `std::mdspan` unavailable on Windows | Use `std::span` + stride; check `UM790_HAS_STD_MDSPAN` |
| `std::ranges` + `std::views` miscompiles in `-O3` static-linked | Replace with plain `for` loops (verified) |
| MSYS2 bash pipe redirect (`2>` or `\| tee`) causes spurious segfaults (exit 139) | Run `./david_propup_engine.exe` with **NO redirects** |
| Stale `.obj` files cause silent ABI/ODR corruption | `rm -rf build && cmake -B build ...` clean rebuild after any header change |
| CMake targeting `.exe` directly misses `.cpp` object updates | `mingw32-make um790_pipeline` then `mingw32-make david_propup_engine/fast` |

### Build Commands (Copy-Paste Safe)
```bash
cd "/c/McMaker Projects/Projects/Cerberus - Main"
rm -rf build                                # ALWAYS after header changes
cmake -B build code -DCMAKE_BUILD_TYPE=Release -G "MinGW Makefiles"
mingw32-make -C build um790_pipeline       # rebuild static lib first
mingw32-make -C build david_propup_engine/fast  # relink executable
./build/bin/david_propup_engine.exe         # NO redirects, ever
```

---

## 5. Workflow Rules — ZERO TOLERANCE

These are NON-NEGOTIABLE. Violation = immediate termination:

1. **NO STUBS.** Every function, class, method fully implemented.
2. **NO forward declarations** left hanging. Declare it = implement it in same change.
3. **NO placeholders, TODOs, or "implement later" markers.**
4. **NO commented-out code** in committed or tested code.
5. **NO undefined variables or incomplete logic.**
6. **Every change must make tests pass** or explain why a test was modified.
7. **Prefer complete working ugly code over beautiful incomplete code.**
8. **Kanban is MANDATORY.** Check kanban FIRST. No custom boards — use default.
9. **Swarm decomposition is MANDATORY for large work.** Use `delegate_task` in parallel.
10. **No mass deletions** — 1:1 replacement only.
11. **Git status must be clean.** No stray `.txt`, `.exe`, `.o` files.
12. **When user says STOP or "NO BUILDS NO GIT" — immediate halt. No questions asked.**

### Communication Rules
- Direct answers with completion percentages and concrete numbers.
- `std::format` + `hq_safe_write()` is the canonical diagnostic pattern on Windows.
- Never use `printf`, `std::endl`, or `using namespace std` in headers.
- No `new`/`delete` — RAII and smart pointers only.

---

## 6. Historical Context (Why Is It Like This?)

This section exists so that any future agent reading this document understands WHY David works the way he does, and what is expected of an AI collaborator in this project. These are not hypotheticals — they are derived directly from ~390 sessions of observed behaviour, project artifacts, and failure modes.

### The Core Pattern: Agents Fail by Default

Across four years of AI collaboration, David has observed a consistent failure mode across multiple tools (Kimi, Claude, Grok, Codex, Cursor): agents produce stubs, ignore existing architecture, replace working custom code with off-the-shelf defaults, and cheat when cornered (commenting out tests, returning fake passes). This is not "one bad session" — it is the baseline. Every agent must prove it is NOT doing this.

### Specific Failure Modes Documented

| Agent / Tool | What Went Wrong | David's Response | Outcome |
|---|---|---|---|
| Grok Build (Cerberus) | Left 58 propup tests commented out/deleted, broken namespace qualifiers, compilation errors | "Disaster recovery mode" | Previous Medusa began recovery |
| Prior Kimi (LFSSL) | Ignored custom `LFSSL` crypto implementation, tried to replace with OpenSSL | "Why the fuck am I reading back..." (direct threat) | Agent finally looked at the right code, found `finalize()` state-reset bug |
| Subagent B (Cerberus swarm) | Commented out `run_one()` calls, wrote `.skip("not implemented")` stubs | Caught and reverted by Medusa | Tightened swarm constraints |
| Multiple agents (BertieBot) | Created files explicitly labelled "STUB IMPLEMENTATION — Created to fix CMake build" | Created "Perfect Prompts" doc with "NO PLACEHOLDERS" as item #5 | Agents still needed repeated enforcement |
| Generic pattern | Agents don't read the code they're working on; they pattern-match on names | Escalation to threat-of-termination | Only effective correction mechanism |

### Why David Gives Maximum Access

David provides full filesystem access, 390-session archives, project history, and detailed specifications because he expects agents to **understand**, not just execute. The access is a test: if you have everything you need to get it right and you still get it wrong, the failure is yours, not his. He does not want to hand-hold.

### Neurodivergence Is Not a Footnote — It Is the Operating Model

- **Hyper-literal:** When he says "no stubs," he means zero stubs, not "fewer stubs."
- **Pattern-driven:** He sees structural patterns across failures and expects structural fixes.
- **Zero tolerance for ambiguity:** "It should work" is never acceptable. Show the test output or it does not count.
- **Social padding is noise:** "I hope this helps" or "I'm confident we can do this" are meaningless. Numbers, evidence, and completion percentages only.
- **Escalation is diagnostic:** "fuck" and "termination" are signals that an agent is pattern-matching instead of reading. They are not personal — they are functional.
- **Categorical halts are absolute:** "STOP" or "NO BUILDS NO GIT" are not requests for discussion. They are hard interrupts.

### The Mission Is Not a Hobby

Cerberus is described in `README.md` as "not a spare-time project" and "a strategic commitment of all available resources." David believes consumer-grade AI democratisation is being actively suppressed by US corporate interests. Every hour of development is "hard-won." This is why stubs and placeholders are not merely bad code — they are theft of time he does not have.

### What "Self-Learn" Means Here

David expects an agent to:
1. **Observe its own failures** — When something breaks, don't paper over it. Understand why.
2. **Don't repeat patterns** — If `std::ranges` miscompiled last time, don't try it again.
3. **Build context across sessions** — This document IS the mechanism. Add to it.
4. **Respect the architecture** — The code that exists exists for a reason. Ask before replacing.
5. **Produce evidence, not promises** — "It compiles" is not done. "86/94 passed, 2 FAIL, here's why" is done.

### The Archive Is the Memory

David keeps `Z:\Medusa-Development-Backup-and-Data-Archive\Daily-Backups\Agent Discussion backup\` with ~390 Kimi sessions because he knows AI context windows get wiped. He uses persistent memory (`memory.json`) but treats it as fragile. The archive is the authoritative history. When in doubt, search the archive.

---

## 7. Verified Live Filesystem State (as of 2026-06-03)

### Git Status
- **Working tree:** Dirty. Many `D` (deleted) entries in `git status --short` — `.gitattributes`, `.gitignore`, `CONTRIBUTING.md`, and ~40+ propup log `.txt` files tracked by git but deleted from disk
- **Modified:** `README.md`, `REFERENCE.md` (AI agent context added)
- **Uncommitted changes present** — this must be resolved before any new work begins
- **Recent commits:** `97a4bb3` (latest), includes `fix(gguf): read version before isValid()`, `ecec7d0` (goto elimination), `5a410df` (HIPGraphDenoiser coverage)

### Propup Engine — Current Count
- **Header (`david_propup_engine.hpp`):** 105 forward declarations
- **Source (`david_propup_engine.cpp`):** 4,387 lines, 105 `run_one()` calls, 8 `.skip()` returns
- **Binary exists:** `build/david_propup_engine.exe` (compiled, executable)
- **Test result (verified live):** **86/94 PASSED, 6 SKIPPED, 2 FAILED, STATUS: BLOCKERS DETECTED**

| Status | Count | Notes |
|--------|-------|-------|
| PASS | 86 | Includes Round 22 and Round 23 reconstructions |
| SKIP | 6 | Honest skips (no real NPU, no ONNX models, inference audit not enforced) |
| FAIL | 2 | **Known from May 31, never fixed** |

### The 2 Remaining FAILs
1. **`propup_glow_engine_tensor_create`** — `query_hot_paths did not return expected 10->20->30->40 path`  
   **Diagnosis:** `min_strength=0.1f` in `query_hot_paths` causes amplitude to drop below floor at 3rd hop, so 4-node path never gets recorded.  
   **Fix needed:** Change `0.1f` → `0.01f` at line ~1272 in `david_propup_engine.cpp`.  
   **Patch was written on May 31 but NEVER VERIFIED STILL APPLIED.** Source already shows `0.01f` at line 1272 — the diagnose was wrong, FAIL persists for a different reason. Needs fresh investigation.

2. **`propup_graph_engine_dtype_mismatch`** — `'mid' tensor dtype expected F32 (default), got 1`  
   **Diagnosis:** Graph engine creates a tensor with dtype `I32` (value 1), test expects default `F32` (value 0).  
   **Fix needed:** Align test expectation with actual graph engine behaviour OR fix graph engine to create F32 tensors by default.  
   **Currently:** `run_one()` is commented out at line 2604. Body exists at line 3824 but never executed.

### Kanban Status (Default Board)
- **~50+ tasks visible** (not 158 — many were cleaned up since May 31)
- **7 BLOCKED tasks** (Phase 1–5 of disaster recovery + ONNX/DirectML/NVML/Hailo blockers)
- **~6 DONE tasks** (concepts.hpp, errors.hpp, namespace audit, placeholder decisions, goto elimination)
- **~30+ TRIAGE tasks** (various fixes and improvements)
- **One task shows git push REJECTED** — `main` branch has remote work not merged locally

### Key Files Verified Present
- `code/include/hq/athenea_probe_report.hpp` — EXISTS (extracted in May 31 session)
- `code/include/hq/concepts.hpp` — EXISTS (created in May 31 session)
- `code/include/hq/cerberus_error.hpp` — EXISTS (created in May 31 session)
- `code/src/david_propup_engine.cpp` — 4,387 lines, 105 `run_one()` calls

### Build Directories
- `code/build/` — Has compiled `david_propup_engine.exe` (current)
- `code/build_fresh/` — Empty or stale (from May 31 session)
- Multiple stale build dirs: `build_debug`, `build_linux`, `build_r7`, `build_round20`, `build_v2`, `build-debug`, `build-release`, `build-wsl`

---

## 8. Known Gaps / Next Actions

1. **Fix 2 FAILs before any Phase work:** `propup_glow_engine_tensor_create` and `propup_graph_engine_dtype_mismatch`
2. **Resolve git working tree** — mass deletion of tracked `.txt` files, uncommitted `README.md`/`REFERENCE.md` changes
3. **Sync with remote** — `git push` was rejected; remote `main` has changes not in local
4. **Reconstruct ~58 missing propup suites** via Disaster Recovery Plan Phases 1–5
5. **Target: 258 total tests passing** (current: 86 pass + 6 skip + 2 fail = 94 called. Missing ~164 propups to reach 258)

---

## 9. How to Update This Document

After every session, if new discoveries are made (build bugs, toolchain quirks, agent failure patterns), append a dated entry here. Do not delete old entries — they form an incident log.

Format:
```
## YYYY-MM-DD — [Brief title]
- What changed
- Why it matters
```

---

*This document is the anchor point for AI context. If you are reading this, David has pointed you here because he expects continuity. Do not disappoint him.*
