# Round 21: Repository Cleanup Report

## Stage 1: Locate and Inventory

### Directory investigated

**`Kimi_Agent_UM790 Pro C++26 Battle (5)/`**

Path: `C:\McMaker Projects\Projects\Cerberus - Copy\Kimi_Agent_UM790 Pro C++26 Battle (5)\`

Contents (recursively):

| Path | Size | Type |
|---|---|---|
| `code/` | — | Empty directory |
| `code/build/` | — | Empty directory |
| `code/src/` | — | Empty directory |
| `code/src/staging_manager.cpp` | 169 lines | Superseded C++ source |
| `code/src/utilization_watchdog.cpp` | 506 lines | Superseded C++ source |

### Additional items found

| Path | Type | Status |
|---|---|---|
| `doc/C01_fix_strategy.md` | Design rationale document | Valuable context |
| `include/hq/health_score.hpp` | Stale header copy | Superseded |
| `src/health_score.cpp` | Stale source copy | Superseded |
| `New folder/` | Empty directory | Empty |
| `UM790_Battle_Test_Report.docx` | Third-party test report | Superseded |
| `UM790_Battle_Test_Report.docx:Zone.Identifier` | Windows metadata | Junk |

## Stage 2: Evaluation

### `code/src/staging_manager.cpp` (Kimi version, 169 lines)
- **Still referenced?** No. CMakeLists.txt uses `code/src/staging_manager.cpp`
- **Unique content?** No. Production version (167 lines) is a superset. Kimi version uses an outdated `StagingBuffer` struct with a `.cdata` field that was removed from the production header. The Kimi version also uses `#include <print>` (requiring native `std::print`) while the production version uses the `UM790_HAS_STD_PRINT` shim.
- **Deletion risk:** None — production version is strictly superior.

### `code/src/utilization_watchdog.cpp` (Kimi version, 506 lines)
- **Still referenced?** No. CMakeLists.txt uses `code/src/utilization_watchdog.cpp`
- **Unique content?** No. While the Kimi version is longer (506 vs 367 lines), it is an older revision that uses `gmtime_r` (POSIX-only) without a Windows fallback, while the production version at line 328-330 uses both `gmtime_s` (Windows) and `gmtime_r` (POSIX) with proper `#ifdef` guards. The Kimi version also lacks atomic operations on `stats_.gpu_state` and `stats_.hailo_state` that were added to fix the C01 data race.
- **Deletion risk:** None — production version has bug fixes not present in Kimi version.

### `doc/C01_fix_strategy.md` (554 lines)
- **Still referenced?** No — build doesn't reference it.
- **Unique content?** **Yes.** This document contains the detailed audit and fix strategy for the UtilizationWatchdog data race. It documents the atomic migration strategy for `stats_.gpu_state`, `stats_.hailo_state`, `stats_.gpu_recovery_count`, and `stats_.hailo_recovery_count`. While the fix is already implemented in production code, the rationale documentation has historical value.
- **Action:** Moved to `research/C01_fix_strategy.md` alongside the other Round reports.

### `include/hq/health_score.hpp` (stale, different hash from production)
- **Still referenced?** No — build uses `${CMAKE_CURRENT_SOURCE_DIR}/include` which resolves to `code/include/`.
- **Unique content?** No — older revision with different hash. Production version under `code/include/hq/` is newer.
- **Deletion risk:** None.

### `src/health_score.cpp` (stale, 290 lines vs production 375 lines)
- **Still referenced?** No — build uses `code/src/`.
- **Unique content?** No — older, smaller version.
- **Deletion risk:** None.

### `New folder/`
- Empty directory. Zero contents.
- **Deletion risk:** None.

### `UM790_Battle_Test_Report.docx`
- Third-party agent test report. No code references.
- **Deletion risk:** None — historical only.

## Stage 3: Decision & Action

| Item | Decision | Rationale |
|---|---|---|
| `Kimi_Agent_UM790 Pro C++26 Battle (5)/` | **Delete** | Production code in `code/` is a strict superset. Kimi versions are older, lack Windows compatibility, and lack bug fixes. |
| `doc/C01_fix_strategy.md` | **Move to `research/`** | Contains valuable design rationale for the watchdog data-race fix. Preserved alongside other research documents. |
| `include/hq/health_score.hpp` | **Delete** | Stale copy. Production version lives in `code/include/hq/`. |
| `src/health_score.cpp` | **Delete** | Stale copy (290 vs 375 lines). Production version lives in `code/src/`. |
| `New folder/` | **Delete** | Empty directory. |
| `UM790_Battle_Test_Report.docx` | **Delete** | Third-party output, no code value. |
| `UM790_Battle_Test_Report.docx:Zone.Identifier` | **Delete** | Windows metadata junk. |

## Stage 4: Verification

- **Build:** `py build.py --clean` → 0 errors, 0 warnings (GCC 15.2.0)
- **Tests:** Core test suite passes (CLIPTokenizer, PinnedStaging, HealthScore, TensorView, DEISScheduler, BenchmarkLogger, TieredMemory, Coroutines including Task_ReturnValue, Task_MoveSemantics, Generator, AsyncPipeline)
- **git status:** All deletions and the rename are clean. No orphan references in CMakeLists.txt.
- **Deleted directories verified absent:** `Kimi_Agent_...`, `New folder/`, `doc/`, `include/`, `src/`
- **C01_fix_strategy.md verified present** at `research/C01_fix_strategy.md`
- **No build references** to any deleted paths (CMakeLists references `code/src/` and `code/include/`, not top-level)

## Stage 5: Recommendations for Preventing Residue

1. **Add `.gitignore` entries for agent workspaces:**
   ```
   *Agent*
   *Battle*
   *Zone.Identifier
   New folder/
   ```
   These patterns prevent third-party agent session directories from being accidentally committed.

2. **Enforce single source tree:** The `code/` directory should be the only location for source code. Any top-level `src/`, `include/`, or `doc/` directories are stale and should be treated as suspicious.

3. **Code review gate:** Before merging any PR, verify that new files are placed under `code/` (or `research/` for documentation), not at the repository root.

4. **Pre-commit hook suggestion:** A pre-commit hook that rejects files matching `*Agent*Battle*` or `*Zone.Identifier` patterns would prevent future residue.