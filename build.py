#!/usr/bin/env python3
"""
build.py — Cerberus verbose build driver
Copyright (c) 2026 D Hargreaves (AKA Roylepython). LamiaFabrica.

Runs cmake + mingw32-make, captures all compiler output, then splits into:
  build/cerberus_errors.log   — every error with full caret context
  build/cerberus_warnings.log — every warning with full caret context
  build/cerberus_build.log    — complete raw output

Each block in the split files is self-contained: the source line, the caret
line (^~~~), and the note lines are kept together so each error is fully
readable in isolation.

Usage:
  python build.py             # full rebuild
  python build.py --clean     # wipe build dir first
  python build.py --jobs 1    # single-threaded (easier to read sequential output)
  python build.py --config Debug
"""

import subprocess
import sys
import os
import re
import shutil
import stat
import argparse
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).parent.resolve()
CODE_DIR    = SCRIPT_DIR / "code"
BUILD_DIR   = CODE_DIR   / "build"
LOG_DIR     = BUILD_DIR

GXX   = r"C:\gcc-15.2.0\mingw64\bin\g++.exe"
GCC   = r"C:\gcc-15.2.0\mingw64\bin\gcc.exe"
MAKE  = r"C:\gcc-15.2.0\mingw64\bin\mingw32-make.exe"

# ---------------------------------------------------------------------------
# ANSI colour codes for terminal output
# ---------------------------------------------------------------------------
RED    = "\033[31m"
YELLOW = "\033[33m"
GREEN  = "\033[32m"
CYAN   = "\033[36m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def colour(text, code):
    return f"{code}{text}{RESET}" if sys.stdout.isatty() else text

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Cerberus build driver")
    p.add_argument("--clean",   action="store_true", help="Delete build dir first")
    p.add_argument("--jobs",    type=int, default=4,  help="Parallel jobs (default 4)")
    p.add_argument("--config",  default="Release",    help="Build type (default Release)")
    p.add_argument("--no-cmake",action="store_true",  help="Skip cmake configure step")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Run cmake configure
# ---------------------------------------------------------------------------
def run_cmake(config: str) -> tuple[int, str]:
    cmd = [
        "cmake",
        "-B", str(BUILD_DIR),
        "-G", "MinGW Makefiles",
        f"-DCMAKE_BUILD_TYPE={config}",
        f"-DCMAKE_CXX_COMPILER={GXX}",
        f"-DCMAKE_C_COMPILER={GCC}",
        f"-DCMAKE_MAKE_PROGRAM={MAKE}",
        "-Wno-dev",
        str(CODE_DIR),
    ]
    print(colour("==> cmake configure", CYAN))
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(CODE_DIR))
    output = result.stdout + result.stderr
    return result.returncode, output

# ---------------------------------------------------------------------------
# Run make
# ---------------------------------------------------------------------------
def run_make(jobs: int) -> tuple[int, str]:
    cmd = [MAKE, "-C", str(BUILD_DIR), f"-j{jobs}"]
    print(colour(f"==> make -j{jobs}", CYAN))
    result = subprocess.run(cmd, capture_output=True, text=True)
    # make mixes stdout/stderr — capture both
    output = result.stdout + result.stderr
    return result.returncode, output

# ---------------------------------------------------------------------------
# Diagnostic block splitter
#
# GCC emits diagnostics as multi-line blocks:
#
#   src/foo.cpp:42:17: error: 'bar' was not declared [-Werror=undeclared]
#      42 |     auto x = bar();
#         |              ^~~
#   src/foo.cpp:40:5: note: declared here
#      40 |     int bar_v = 0;
#         |     ^~~
#
# We want to capture each block (primary line + its source + caret + notes)
# as a single self-contained unit.
# ---------------------------------------------------------------------------

# Regex: primary diagnostic line
DIAG_RE = re.compile(
    r'^(?P<file>[^:]+):(?P<line>\d+):(?P<col>\d+):\s+'
    r'(?P<kind>error|warning|note|fatal error):\s+'
    r'(?P<msg>.+)$'
)

def split_diagnostics(raw: str) -> tuple[list[str], list[str]]:
    """
    Returns (error_blocks, warning_blocks).
    Each block is a list of lines forming one complete diagnostic.
    """
    lines = raw.splitlines(keepends=True)
    errors   = []
    warnings = []

    i = 0
    while i < len(lines):
        m = DIAG_RE.match(lines[i].rstrip('\n'))
        if not m:
            i += 1
            continue

        kind = m.group('kind')
        if kind not in ('error', 'warning', 'fatal error'):
            i += 1
            continue

        # Collect the block: primary line + source listing + caret + notes
        block_lines = [lines[i]]
        i += 1
        while i < len(lines):
            l = lines[i]
            # A new primary error/warning line ends the block
            m2 = DIAG_RE.match(l.rstrip('\n'))
            if m2 and m2.group('kind') in ('error', 'warning', 'fatal error'):
                break
            # note lines, source lines (starting with digits+|), caret lines (^) belong to block
            block_lines.append(l)
            i += 1

        block_text = "".join(block_lines)
        if kind in ('error', 'fatal error'):
            errors.append(block_text)
        else:
            warnings.append(block_text)

    return errors, warnings

# ---------------------------------------------------------------------------
# Write log files
# ---------------------------------------------------------------------------
def write_log(path: Path, header: str, blocks: list[str], total: int):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*78}\n")
        f.write(f"{header}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total: {total}\n")
        f.write(f"{'='*78}\n\n")
        for n, block in enumerate(blocks, 1):
            f.write(f"{'─'*78}\n")
            f.write(f"[{n}/{total}]\n")
            f.write(block)
            if not block.endswith('\n'):
                f.write('\n')
        if not blocks:
            f.write("(none)\n")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    if args.clean and BUILD_DIR.exists():
        print(colour(f"Removing {BUILD_DIR}", YELLOW))

        def _remove_readonly(func, path, exc):
            os.chmod(path, stat.S_IWRITE)
            func(path)

        shutil.rmtree(BUILD_DIR, onerror=_remove_readonly)

    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    # --- cmake ---
    if not args.no_cmake:
        rc, cmake_out = run_cmake(args.config)
        (BUILD_DIR / "cerberus_cmake.log").write_text(cmake_out, encoding='utf-8')
        if rc != 0:
            print(colour("cmake FAILED", RED))
            print(cmake_out[-2000:])
            sys.exit(rc)
        print(colour("cmake OK", GREEN))

    # --- make ---
    rc, make_out = run_make(args.jobs)

    # Save raw log
    raw_log = BUILD_DIR / "cerberus_build.log"
    raw_log.write_text(make_out, encoding='utf-8')

    # Split into errors and warnings
    errors, warnings = split_diagnostics(make_out)

    err_log  = BUILD_DIR / "cerberus_errors.log"
    warn_log = BUILD_DIR / "cerberus_warnings.log"

    write_log(err_log,  "CERBERUS BUILD — ERRORS",   errors,   len(errors))
    write_log(warn_log, "CERBERUS BUILD — WARNINGS",  warnings, len(warnings))

    # --- Summary ---
    print()
    if rc == 0:
        print(colour("BUILD SUCCEEDED", GREEN + BOLD))
    else:
        print(colour("BUILD FAILED", RED + BOLD))

    print(f"  Errors:   {colour(str(len(errors)),   RED    if errors   else GREEN)} "
          f"-> {err_log.name}")
    print(f"  Warnings: {colour(str(len(warnings)), YELLOW if warnings else GREEN)} "
          f"-> {warn_log.name}")
    print(f"  Raw log:  {raw_log.name}")
    print(f"  All logs: {BUILD_DIR}")
    print()

    # Print the first error in full so it's visible immediately in the terminal
    if errors:
        print(colour("── First error ──────────────────────────────────────────", RED))
        print(errors[0])

    sys.exit(rc)


if __name__ == "__main__":
    main()
