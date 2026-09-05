# Cerberus — Hardware Platform

This document describes the physical hardware used to develop and run Cerberus, the reasoning
behind each choice, and how the physical build came together.

---

## Primary Development Platform: MinisForum UM790 Pro

**Form factor:** Mini PC (approximately 200 × 200 × 55 mm)

| Component | Detail |
|-----------|--------|
| CPU | AMD Ryzen 9 7940HS — Zen 4, 8-core/16-thread, up to 5.2 GHz |
| iGPU | AMD Radeon 780M — RDNA 3, 12 CUs, HIP/ROCm compute |
| RAM | 64 GB DDR5-5600 (dual-channel) |
| Storage | 2 × M.2 NVMe slots (one used for primary SSD) |
| Connectivity | Thunderbolt 4, USB4, 2.5 GbE |
| OS | Ubuntu 24.04 LTS (primary) / Windows 11 (development build) |

### Why the UM790 Pro?

The 7940HS has an RDNA 3 iGPU that is fully supported by AMD ROCm via the HIP runtime.
This means a £600 mini PC can run real GPGPU workloads — denoising UNet inference,
VAE decoding — that would otherwise require a discrete GPU costing several times more.

The 64 GB DDR5 shared-memory architecture also means the iGPU has access to a large
address space without a separate VRAM budget. This is a practical advantage for running
multi-GB models on consumer hardware.

---

## NPU: Hailo-8L M.2

**Hailo-8L** is a neural processing unit from Hailo AI, available in M.2 form factor
(2242/2280). It delivers up to 13 TOPS at very low power (< 2W typical), making it ideal
for offloading text encoding (CLIP/T5) from the CPU/GPU.

| Spec | Value |
|------|-------|
| Peak TOPS | 13 |
| Typical power | < 2 W |
| Interface | PCIe Gen 3 x1 (M.2 M-key) |
| SDK | HailoRT >= 4.20 |

The UM790 Pro has one of its M.2 slots occupied by the primary NVMe SSD. Adding the Hailo-8L
requires a **PCIe riser solution** — which leads to the 3D-printing work described below.

---

## 3D-Printed Riser Brackets (Bambu Lab)

The Hailo-8L M.2 card cannot be mounted directly inside the UM790 Pro's existing M.2 slot
without displacing the primary SSD. The solution is an external PCIe M.2 riser:

1. A **M.2 to PCIe x1 adapter card** routes the Hailo-8L outside the chassis.
2. A custom **3D-printed bracket** mounts the adapter to the side of the UM790 Pro case,
   maintaining proper clearance and cable management.

The brackets are designed and printed on a **Bambu Lab A1 Mini** using PETG filament
(heat-resistant, low warp). The design accounts for:

- PCB standoff heights
- Ribbon cable bend radius for the PCIe riser
- Ventilation clearance around the Hailo-8L
- Aesthetic alignment with the UM790 Pro's case lines

CAD files and print profiles will be published in a future release under a permissive licence
so other makers can replicate the build.

**Note:** Bambu Lab hardware is used purely as a fabrication tool. No Bambu Lab software,
firmware, or intellectual property is part of the Cerberus project.

---

## Constraint-Driven Design Philosophy

Every major architectural decision in Cerberus traces back to a specific hardware constraint
on this platform:

| Constraint | Architectural Response |
|------------|----------------------|
| Shared CPU/GPU memory (no discrete VRAM) | `TieredMemoryManager` manages Hot/Warm/Cool/Cold tiers across DDR5 address space |
| iGPU H2D copies waste bandwidth | `PinnedStagingPool` + documented `BUG B3` zero-copy path for future fix |
| Hailo-8L SDK absent in dev environment | `INpuEncoder` abstraction with `SyntheticNpuEncoder` fallback for CI/testing |
| Single-node during development | `ClusterTransport` designed but marked `@experimental` until multi-node hardware is ready |
| Power envelope (mini PC, no active cooling for NPU) | `PipelineHealthScore` thermal sub-score + `UtilizationWatchdog` thermal threshold |

The goal has always been: write real production-quality code against the actual hardware
constraints, not against idealised server hardware. Cerberus is honest about what works today
and what is aspirational.

---

## Future Hardware Targets

| Hardware | Purpose | Status |
|----------|---------|--------|
| Second UM790 Pro | Multi-node ClusterTransport testing | Planned |
| CXL memory expansion card | Hot-tier memory beyond DDR5 | Researching |
| Hailo-8 (full, 26 TOPS) | Higher-throughput text encoding | Target for Hailo path |
| 10 GbE switch | Inter-node Ethernet clustering | Planned with second node |

---

Copyright (c) 2026 D Hargreaves (AKA Roylepython). LamiaFabrica. All rights reserved.
