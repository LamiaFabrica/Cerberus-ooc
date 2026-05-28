# Contributing to Cerberus

Thank you for your interest! Cerberus is a **work-in-progress passion project**
and I welcome contributions of all kinds — code, documentation, design feedback,
and especially **hardware expertise** for the GPU/NPU paths that are currently stubbed.

## Quick Start for Contributors

1. **Fork the repository**
2. **Build the project** (see [README.md](README.md#build))
3. **Run the KPI test suite**: `code/build/david_propup_engine.exe` — must show **347/347 passed**
4. **Make your changes**
5. **Run the suite again** — zero regressions allowed
6. **Open a Pull Request** with a clear description of what changed and why

## What I'm Looking For

### High Priority

- **Ubuntu + ROCm build verification** — I can't test this on Windows
- **HIP zero-copy staging** — Replace the `false` gate in `hip_staging_` with real `hipMalloc`/`hipMemcpy`
- **HailoRT integration** — Linux-only; I have no HailoRT on this host
- **AVX-512 kernel correctness** — Dispatch exists but hardware is unavailable
- **End-to-end inference with real ONNX models** — Text encoder / UNet / VAE on real weights

### Medium Priority

- **Code review of safety-critical paths** — `denoise_step_()`, `Scheduler::step()`, `TMM::promote()`
- **AddressSanitizer / Valgrind runs** — There is a cross-test heap corruption I'm chasing
- **Better error messages** — `unavailable_reason()` strings should be actionable
- **Documentation** — The C++26 concepts (`NpuBackend<T>`, `NpuAccelerator<T>`) need examples

### Low Priority / Nice-to-Have

- **README translations** — I only write English
- **Build scripts for other platforms** — macOS, WSL2
- **Pretty printing / TUI improvements** — The `cerberus monitor` dashboard is functional but ugly

## Code Conventions

- **C++26 minimum** — `std::expected`, `std::format`, designated initializers are required
- **Zero tolerance for commented-out code** — If it's dead, delete it. Exception: 3 staging tests with known cross-test heap contamination
- **Honest unavailability** — Every hardware-gated component must implement `unavailable_reason()` returning a descriptive string (never empty, never `false` without explanation)
- **No `std::format` with `char const*` args** — GCC 15 MinGW segfaults. Use `std::ostringstream` instead
- **`<windows.h>` banned from propup units** — Forward declarations in headers; isolated `.cpp` files only
- **No empty feature stubs** — `-1.0f` sentinels and `unavailable_reason()` mandatory

## Test Philosophy

- **The KPI is sacred.** `david_propup_engine` must always show **347/347 passed**
- **Synthetic tests only** — No external file dependencies in the test suite
- **E2E detectable** — Every test must be callable via `run_one<>` in `david_propup_engine.cpp`

## Security

Do **not** open a public issue for security vulnerabilities. Email me directly at the address
in my GitHub profile with "Cerberus Security" in the subject line.

## License

By contributing, you agree that your contributions will be licensed under the
**MIT License** (see [LICENSE](LICENSE)).

---

## Code of Conduct

Be respectful. Be honest. Be patient. I am a solo developer doing this in my spare time.
I will review PRs as quickly as I can, but sometimes real life gets in the way.

If you are frustrated that something is "only 65% done" — that's because it genuinely is only
65% done, not because I'm lazy. Help me get to 100%.
