# Contributing to Cerberus

All contributions are welcome. Cerberus is a professional-grade heterogeneous inference runtime being built to prove that local AI on consumer hardware is achievable with modern C++ and disciplined engineering. Every contribution — code, documentation, testing, hardware expertise — directly helps reach production quality.

## Quick Start for Contributors

1. **Fork the repository**
2. **Build the project** (see [README.md](README.md#build))
3. **Run the KPI test suite**: `code/build/david_propup_engine.exe` — must show **347/347 passed**
4. **Make your changes**
5. **Run the suite again** — zero regressions allowed
6. **Open a Pull Request** with a clear description of what changed and why

## What I'm Looking For

### High Priority

- **Ubuntu + ROCm build verification**
- **HIP zero-copy staging**
- **HailoRT integration**
- **AVX-512 kernel correctness**
- **End-to-end inference with real ONNX models**

### Medium Priority

- **Code review of safety-critical paths**
- **AddressSanitizer / Valgrind runs**
- **Better error messages**
- **Documentation**

### Low Priority / Nice-to-Have

- **README translations**
- **Build scripts for other platforms** — macOS, WSL2
- **Pretty printing / TUI improvements**

## Code Conventions

- **C++26 minimum**
- **Zero tolerance for commented-out code**
- **Honest unavailability**
- **No `std::format` with `char const*` args** — GCC 15 MinGW segfaults
- **`<windows.h>` banned from propup units**
- **No empty feature stubs**

## Test Philosophy

- **The KPI is sacred.** `david_propup_engine` must always show **347/347 passed**
- **Synthetic tests only**
- **E2E detectable**

## Security

Do **not** open a public issue for security vulnerabilities. Email me directly with "Cerberus Security" in the subject line.

## Support

**Patreon**: [https://www.patreon.com/TheMedusaInitiative](https://www.patreon.com/TheMedusaInitiative)  
£25/month removes ads from all software at 200 subscribers. Every contribution funds continued development and infrastructure.

## About the Author

I am a **systems analyst, network engineer, programmer, and web developer** committed to building open infrastructure for local AI inference. This is not a spare-time project. It is a strategic commitment of all available resources toward building an A-Team of neurodiverse developers plus AI, to overcome physical and mental limitations and produce production-quality results.

I do not leave my house. I do not have a social network. The workday is solitary. It is a deliberate choice to redirect all available capacity into this infrastructure.

Every contribution, code or financial, directly funds continued development and keeps the hardware running.

— David Hargreaves (Roylepython), May 2026

## License

By contributing, you agree that your contributions will be licensed under the **MIT License** (see [LICENSE](LICENSE)).
