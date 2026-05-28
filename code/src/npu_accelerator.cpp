/// @file npu_accelerator.cpp
/// @copyright Copyright © 2026, all rights reserved | David Hargreaves aka Roylepython trading as The Medusa Initiative & Lamia Fabrica
/// Support: https://www.patreon.com/TheMedusaInitiative — £25/month removes ads from all software at 200 subscribers
/// NPU post-processing accelerator — implementation.
///
/// Provides concrete post-processor implementations and the factory that
/// probes hardware to select the best available backend.
///
/// @version 2.0.0 — synthetic data eradicated. No fabricated telemetry.
/// @author LamiaFabrica Team

#include "hq/npu_accelerator.hpp"
#include "hq/logger.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <type_traits>

// ===========================================================================
// HailoRT three-tier detection
// ===========================================================================
#ifdef HAILO_BUILD
#  define HAILO_ACCEL_HAS_HAILORT 1
#  include <hailort/hailort.h>
#else
#  if __has_include(<hailort/hailort.h>)
#    define HAILO_ACCEL_HAS_HAILORT 1
#    include <hailort/hailort.h>
#  elif __has_include(<hailort/hailort.hpp>)
#    define HAILO_ACCEL_HAS_HAILORT 1
#    include <hailort/hailort.hpp>
#  else
#    define HAILO_ACCEL_HAS_HAILORT 0
#    pragma message("HailoRT headers not found — HailoNpuPostProcessor compiled without HailoRT support")
#  endif
#endif

namespace hq::npu {

// ===========================================================================
// CpuPostProcessor — honest CPU pass-through (no NPU acceleration)
// ===========================================================================

std::expected<NpuPostProcessResult, std::string>
CpuPostProcessor::post_process(const NpuPostProcessRequest& req) {
    if (req.pixels.empty()) {
        return std::unexpected{"CpuPostProcessor: empty pixel input"};
    }
    if (req.width == 0 || req.height == 0) {
        return std::unexpected{"CpuPostProcessor: zero dimensions"};
    }

    const auto t0 = std::chrono::high_resolution_clock::now();

    NpuPostProcessResult result;
    result.width  = req.width;
    result.height = req.height;

    // CPU pass-through: copy pixels unchanged.
    // When a real NPU post-processing network (e.g. ESRGAN on Hailo-8L) is
    // available, HailoNpuPostProcessor will perform the work. Until then,
    // the pipeline uses this honest CPU fallback.
    result.pixels.resize(req.pixels.size());
    std::memcpy(result.pixels.data(), req.pixels.data(), req.pixels.size());

    const auto t1 = std::chrono::high_resolution_clock::now();
    result.processing_time_us = static_cast<float>(
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    result.npu_utilization    = -1.0f;  // sentinel: no NPU hardware in CPU fallback
    result.was_npu_accelerated = false;

    return result;
}

std::expected<void, std::string>
CpuPostProcessor::blend_noise_cfg(
    std::span<float>       noise_out,
    std::span<const float> noise_uncond,
    float                  guidance_scale) noexcept {
    if (noise_out.empty()) {
        return std::unexpected{std::string{"blend_noise_cfg: empty noise_out"}};
    }
    if (noise_out.size() != noise_uncond.size()) {
        return std::unexpected{std::string{"blend_noise_cfg: size mismatch cond="}
            + std::to_string(noise_out.size())
            + " uncond=" + std::to_string(noise_uncond.size())};
    }

    // CFG blend: noise_out[i] = noise_uncond[i] + scale*(noise_out[i] - noise_uncond[i])
    //
    // This is pure CPU scalar math. A real Hailo-8L implementation would
    // submit this as a SAXPY kernel on the NN core, but that requires:
    //   1. HailoRT SDK installed (Linux only)
    //   2. A compiled SAXPY HEF for Hailo-8L
    //   3. PCIe DMA path for input/output tensors
    //
    // Until those conditions are met, this honest CPU fallback performs
    // the blend without claiming NPU acceleration.
    for (std::size_t i = 0; i < noise_out.size(); ++i) {
        noise_out[i] = noise_uncond[i] + guidance_scale * (noise_out[i] - noise_uncond[i]);
    }
    return {};
}

bool CpuPostProcessor::can_handle(NpuTaskType task) const {
    // CPU pass-through does NOT claim to accelerate any NPU task.
    // It exists as a safe fallback when no NPU backend is available.
    // The factory selects it explicitly, not because it "can handle" NPU tasks.
    (void)task;
    return false;
}

std::string CpuPostProcessor::unavailable_reason() const {
    return "CPU pass-through performs no NPU acceleration";
}

bool CpuPostProcessor::synthetic_mode() const noexcept {
    // CPU pass-through is always a non-hardware fallback path
    return true;
}

// ===========================================================================
// HailoNpuPostProcessor — production Hailo-8L post-processing
// ===========================================================================

struct HailoNpuPostProcessor::Impl {
    bool hailo_available{false};
    bool post_hef_loaded{false};
    std::string unavailable_reason{"Not initialized"};
    CpuPostProcessor cpu_fallback{};  // Honest CPU delegate when HEF missing

#if HAILO_ACCEL_HAS_HAILORT
    std::unique_ptr<hailort::Device> device;
    std::unique_ptr<hailort::VDevice> vdevice;
#endif
};

HailoNpuPostProcessor::HailoNpuPostProcessor()
    : impl_{std::make_unique<Impl>()}
{
#if HAILO_ACCEL_HAS_HAILORT
    auto& m = *impl_;
    auto scan_result = hailort::Device::scan_pcie();
    if (scan_result && !scan_result.value().empty()) {
        m.hailo_available = true;
        const auto& dev = scan_result.value()[0];
        auto create_result = hailort::Device::create_pcie(dev);
        if (create_result) {
            m.device.reset(create_result.release());
        }
        HQ_LOG_INFO("[HailoNpuPostProcessor] Hailo-8L detected at {} — "
                    "post-processing available when post-HEF loaded",
                    dev.dev_id);
    } else {
        m.hailo_available = false;
        m.unavailable_reason = "No Hailo-8L device detected on PCIe bus";
        HQ_LOG_INFO("[HailoNpuPostProcessor] {}", m.unavailable_reason);
    }
#else
    impl_->hailo_available = false;
    impl_->unavailable_reason = "HailoRT SDK not compiled — rebuild with -DHAILO_BUILD or install HailoRT";
    HQ_LOG_INFO("[HailoNpuPostProcessor] {}", impl_->unavailable_reason);
#endif
}

HailoNpuPostProcessor::~HailoNpuPostProcessor() = default;

std::expected<NpuPostProcessResult, std::string>
HailoNpuPostProcessor::post_process(const NpuPostProcessRequest& req) {
    auto& m = *impl_;
    // No post-processing HEF loaded → delegate to honest CPU fallback
    if (!m.post_hef_loaded) {
        HQ_LOG_DEBUG("[HailoNpuPostProcessor] No post-HEF loaded — delegating to CPU fallback");
        return m.cpu_fallback.post_process(req);
    }
    return std::unexpected{"Hailo-8L post-HEF inference unavailable: HEF loaded but execution path not compiled"};
}

std::expected<void, std::string>
HailoNpuPostProcessor::blend_noise_cfg(
    std::span<float>       noise_out,
    std::span<const float> noise_uncond,
    float                  guidance_scale) noexcept {
    auto& m = *impl_;
    // No SAXPY HEF loaded → delegate to honest CPU fallback
    if (!m.post_hef_loaded) {
        return m.cpu_fallback.blend_noise_cfg(noise_out, noise_uncond, guidance_scale);
    }
    return std::unexpected{"Hailo-8L SAXPY fusion unavailable: SAXPY HEF not loaded"};
}

bool HailoNpuPostProcessor::can_handle(NpuTaskType task) const {
    // Only claim capability for tasks we can actually accelerate on Hailo hardware.
    // When no post-HEF is loaded, we delegate to CPU — do not claim NPU capability.
    if (!impl_->post_hef_loaded) return false;
    (void)task;
    return true;  // HEF loaded → real Hailo acceleration available
}

bool HailoNpuPostProcessor::is_available() const {
    return impl_->hailo_available && impl_->post_hef_loaded;
}

std::string HailoNpuPostProcessor::name() const {
    if (!impl_->hailo_available) return "Hailo-8L PostProcessor (no device)";
    if (!impl_->post_hef_loaded) return "Hailo-8L PostProcessor (no post-HEF)";
    return "Hailo-8L PostProcessor";
}

float HailoNpuPostProcessor::utilization() const {
#if HAILO_ACCEL_HAS_HAILORT
    if (impl_->device) {
        auto power_result = impl_->device->get_power_measurement(
            HAILO_POWER_MEASUREMENT_TYPES__TOTAL_POWER);
        if (power_result) {
            return std::clamp(power_result.value() / 1000.0f / HAILO8L_ACTIVE_POWER_W * 100.0f, 0.0f, 100.0f);
        }
    }
#endif
    return -1.0f;  // No device or power read failed
}

bool HailoNpuPostProcessor::device_present() const noexcept {
    return impl_->hailo_available;
}

bool HailoNpuPostProcessor::load_post_hef(const std::filesystem::path& hef_path) {
#if HAILO_ACCEL_HAS_HAILORT
    auto& m = *impl_;
    if (!m.hailo_available || !m.device) {
        m.unavailable_reason = "No Hailo device available to load post-HEF";
        return false;
    }
    if (hef_path.empty() || !std::filesystem::exists(hef_path)) {
        m.unavailable_reason = std::format(
            "Post-HEF not found: {}", hef_path.string());
        return false;
    }
    m.unavailable_reason = "Post-HEF loading not yet implemented (v2.2)";
    return false;
#else
    (void)hef_path;
    impl_->unavailable_reason = "HailoRT SDK not compiled";
    return false;
#endif
}

std::string HailoNpuPostProcessor::unavailable_reason() const {
    return impl_->unavailable_reason;
}

bool HailoNpuPostProcessor::synthetic_mode() const noexcept {
    // When no post-HEF loaded, all operations delegate to CPU fallback.
    // From the caller's perspective, this is a synthetic/fallback path.
    return !impl_->post_hef_loaded;
}

// ===========================================================================
// NpuPostProcessorFactory
// ===========================================================================

std::unique_ptr<INpuPostProcessor>
NpuPostProcessorFactory::create_best_available() {
    // ---- Priority 1: Hailo-8L ----
    {
        auto hailo = std::make_unique<HailoNpuPostProcessor>();
        // Hailo device present but no post-HEF? Use it for delegation, but
        // is_available() stays false until post-HEF is loaded.
        if (hailo->device_present()) {
            HQ_LOG_INFO("[NpuPostProcessorFactory] → SELECTED: Hailo-8L (device_present=true, "
                        "available={}, synthetic={}, reason='{}') — "
                        "delegating to CPU until post-HEF loaded",
                        hailo->is_available(), hailo->synthetic_mode(),
                        hailo->unavailable_reason());
            return hailo;  // Returns HailoNpuPostProcessor that delegates to CPU
        }
    }

    // ---- Priority 2: CPU pass-through (no NPU acceleration) ----
    {
        auto cpu = std::make_unique<CpuPostProcessor>();
        HQ_LOG_INFO("[NpuPostProcessorFactory] → SELECTED: CPU pass-through "
                    "(no acceleration, synthetic={})", cpu->synthetic_mode());
        return cpu;
    }
}

static_assert(!std::is_abstract_v<hq::npu::CpuPostProcessor>, "CpuPostProcessor must override all pure virtuals (post_process, blend_noise_cfg, can_handle, is_available, name, utilization, synthetic_mode, unavailable_reason)");
static_assert(!std::is_abstract_v<hq::npu::HailoNpuPostProcessor>, "HailoNpuPostProcessor must override all pure virtuals");

} // namespace hq::npu
