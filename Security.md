### 2. SECURITY.md

```markdown
# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.7.x   | ✅ Active development |
| < 0.7   | ❌ Not supported     |

## Reporting a Vulnerability

If you discover a security vulnerability in Cerberus, please report it responsibly.

**Do not** open a public GitHub issue.

Instead, email: **hi@poweredbymedusa.com** (or open a private security advisory on GitHub if the repository is public).

We aim to respond within 48 hours and will work with you to resolve the issue before any public disclosure.

## Current Security Posture (May 2026)

- All code is open source under MIT license.
- No user data, API keys, or sensitive credentials are stored or transmitted by default.
- The project uses modern C++26 safety features (`std::expected`, RAII, no raw pointers in hot paths where possible).
- Dependencies (ROCm, HailoRT, ONNX Runtime) are used via official APIs.

## Planned Future Security Features

The following advanced security features are **planned for future releases** once the core inference engine is stable:

- **Post-quantum cryptography** (Kyber / Dilithium for key exchange and signatures)
- **Argon2id** for any password or key derivation
- **libfssl** (or equivalent) for hardened TLS / secure transport
- **JWT-based authentication** with short-lived tokens for the inference server and clustering protocol

These features will be implemented only after the project reaches a stable, production-grade state and proper security auditing has been completed.

## Responsible Disclosure

We appreciate responsible disclosure and will publicly credit researchers who help improve the security of Cerberus (unless they prefer to remain anonymous).

Thank you for helping keep Cerberus secure.