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

## About the Author

I am a **systems analyst, network engineer, programmer, and web developer** who moved to passion projects after leaving paid corporate employment. After years working in systems architecture, infrastructure, and full-stack development, I am now building these projects independently while navigating physical disability and limited capacity for traditional full-time work.

Cerberus is built by one person in their spare time, heavily constrained by health. Every hour of development is hard-won. If you find this project valuable and want to see it reach production quality — real CXL memory tiering, proper Thunderbolt 5 clustering, broader hardware support, and post-quantum security — support is welcome.

**Patreon**: [https://www.patreon.com/TheMedusaInitiative](https://www.patreon.com/TheMedusaInitiative) — £25/month removes ads from all software at 200 subscribers

Every contribution, code or financial, directly funds continued development and keeps the hardware running.

— David Hargreaves (Roylepython), May 2026
