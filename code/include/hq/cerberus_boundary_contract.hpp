#pragma once
/// @file cerberus_boundary_contract.hpp
/// @copyright Copyright (c) 2026 D Hargreaves. All rights reserved.
///
/// Cerberus / PsiForceDB Boundary Contract
/// ========================================
///
/// This file is the single source of truth for what Cerberus MAY and MAY NOT
/// contain. It is not compiled; it is read by agents and reviewers.
///
/// Any code, comment, or header that contradicts this contract is a bug.
/// Any code that silently replicates PsiForceDB / LamiaFabrica logic is a
/// critical violation.
///
/// © 2026 D Hargreaves | LamiaFabrica Software
///
/// ---------------------------------------------------------------------------
/// RULE 1: PsiForceDB Owns the Heart
/// ---------------------------------------------------------------------------
/// The following systems are PROPRIETARY to PsiForceDB / LamiaFabrica.
/// Cerberus MUST delegate to them; it MUST NOT replicate or approximate them:
///
///   - Authentication, JWT, session management, FortressAuth
///   - The real LFSSL library (Kyber, Dilithium, AES-256-GCM production paths)
///   - PQC-secured model splitting / cross-device federation
///   - Deep graph rewriting and advanced quantization planning
///   - Persistent knowledge graph (Glow state across sessions)
///   - Queryable encryption, tamper-proof audit logging
///   - The Athenea weight files and their proprietary GGUF packing schemas
///
/// Cerberus may contain:
///   - Thin wrappers that CALL into PsiForceDB (e.g., LfsslSentinel)
///   - Header shims so PsiForceDB headers compile (documented placeholders)
///   - Extension classes that inherit from PsiForceDB MultiModelExtension
///   - Local telemetry that is REPORTED to PsiForceDB, not learned from it
///
/// ---------------------------------------------------------------------------
/// RULE 2: Cerberus Owns the Integrator Layer
/// ---------------------------------------------------------------------------
/// The following are legitimate Cerberus responsibilities:
///
///   - Local native kernel dispatch (CPU AVX2 / blocked matmul as fallback)
///   - Lightweight local graph representation (nodes, edges, tensors)
///   - Thin routing decisions: which LOCAL backend stub to use
///   - Memory tier placement for LOCAL buffers (Hot/Warm/Cool/Cold)
///   - Recording local execution traces (Glow bonds that are REPORTED to PsiForceDB)
///   - Parsing GGUF metadata headers (NOT the proprietary packed weights)
///   - Command parsing and ergonomic error messages for end users
///
/// ---------------------------------------------------------------------------
/// RULE 3: Comments Must Not Claim Provenance from PsiForceDB
/// ---------------------------------------------------------------------------
/// No Cerberus header may contain text like:
///   - "ported from PsiForceDB ..."
///   - "this implements the proprietary ..."
///   - "the secret sauce ..."
///
/// Cerberus headers should describe what THIS layer does, not where PsiForceDB's
/// equivalent lives.
///
/// ---------------------------------------------------------------------------
/// RULE 4: No Silent Reinvention
/// ---------------------------------------------------------------------------
/// If a Cerberus module starts to look like it needs:
///   - A real vector-native graph database → STOP. That is PsiForceDB.
///   - Production-grade PQC key generation → STOP. That is LFSSL / PsiForceDB.
///   - Cross-session model learning → STOP. That is PsiForceDB.
///   - Tamper-proof audit log chains → STOP. That is PsiForceDB.
///
/// If in doubt, the answer is "delegate to PsiForceDB".
///
/// ---------------------------------------------------------------------------
/// RULE 5: Test Purity
/// ---------------------------------------------------------------------------
/// Propup tests MUST NOT parse real GGUF files, load real model weights, or
/// exercise PsiForceDB proprietary paths. Synthetic in-memory tests only.
/// Real file paths are for runtime production code, not the test suite.
///
/// ---------------------------------------------------------------------------

namespace hq::cerberus::boundary {

/// A dummy marker type that agents can grep for when auditing boundary hygiene.
struct BoundaryContractMarker {};

} // namespace hq::cerberus::boundary
