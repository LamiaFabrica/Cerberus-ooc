/// @file npu_encoder.cpp
/// @copyright Copyright (c) 2026 D Hargreaves (AKA Roylepython). LamiaFabrica. All rights reserved.
/// NPU Encoder abstraction layer — implementation.
///
/// Houses the concrete encoder implementations and the factory that
/// probes system hardware to select the best available backend.
///
/// @version 2.1.0 — production HailoRT path with HEF loading.
///   - Hailo8lEncoder: real async inference when HailoRT + HEF present.
///   - Graceful CPU fallback when Hailo device present but HEF missing.
///   - Zero fast-fail when silicon exists; factory selects CpuFallbackEncoder.

#include "hq/npu_encoder.hpp"
#include "hq/cxx26_features.hpp"
#include "hq/logger.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

#include <onnxruntime_cxx_api.h>

// ===========================================================================
// HailoRT includes (only when SDK is present)
// ===========================================================================
#ifdef HAILO_BUILD
#  define HAILO_ENCODER_HAS_HAILORT 1
#  include <hailort/hailort.h>
#  include <fstream>
#else
#  if __has_include(<hailort/hailort.h>)
#    define HAILO_ENCODER_HAS_HAILORT 1
#    include <hailort/hailort.h>
#    include <fstream>
#  elif __has_include(<hailort/hailort.hpp>)
#    define HAILO_ENCODER_HAS_HAILORT 1
#    include <hailort/hailort.hpp>
#    include <fstream>
#  else
#    define HAILO_ENCODER_HAS_HAILORT 0
#    pragma message("HailoRT headers not found — Hailo8lEncoder compiled without HailoRT support")
#  endif
#endif

namespace hq::npu {

// ===========================================================================
// Hailo8lEncoder — production Hailo-8L inference
// ===========================================================================

struct Hailo8lEncoder::Impl {
    std::string       pcie_address;
    std::filesystem::path hef_path;
    bool              hailo_available{false};
    bool              hef_loaded{false};
    float             last_utilization{0.0f};
    float             last_temperature{0.0f};
    std::string       unavailable_reason{"Not initialized"};

#if HAILO_ENCODER_HAS_HAILORT
    // HailoRT production handles
    std::unique_ptr<hailort::Device>        device;
    std::unique_ptr<hailort::VDevice>       vdevice;
    std::unique_ptr<hailort::InferModel>    infer_model;
    std::unique_ptr<hailort::AsyncInferRunner> runner;
    std::vector<std::uint8_t>                hef_buffer;
#endif

    explicit Impl(std::string addr, std::filesystem::path hef)
        : pcie_address{std::move(addr)}
        , hef_path{std::move(hef)}
    {}
};

Hailo8lEncoder::Hailo8lEncoder(const std::string& pcie_address,
                                const std::filesystem::path& hef_path)
    : impl_{std::make_unique<Impl>(pcie_address, hef_path)}
{
#if HAILO_ENCODER_HAS_HAILORT
    auto& m = *impl_;

    // --- Step 1: Scan PCIe for Hailo devices ---
    auto scan_result = hailort::Device::scan_pcie();
    if (!scan_result || scan_result.value().empty()) {
        m.hailo_available = false;
        m.unavailable_reason = "No Hailo-8L devices detected on PCIe bus";
        HQ_LOG_WARN("[Hailo8lEncoder] {}", m.unavailable_reason);
        return;
    }

    const auto& devices = scan_result.value();
    std::size_t selected_index = 0;
    if (!m.pcie_address.empty()) {
        bool found = false;
        for (std::size_t i = 0; i < devices.size(); ++i) {
            if (devices[i].dev_id == m.pcie_address ||
                devices[i].func_id == m.pcie_address) {
                selected_index = i;
                found = true;
                break;
            }
        }
        if (!found) {
            m.hailo_available = false;
            m.unavailable_reason = std::format(
                "Requested device '{}' not found among {} scanned device(s)",
                m.pcie_address, devices.size());
            HQ_LOG_WARN("[Hailo8lEncoder] {}", m.unavailable_reason);
            return;
        }
    }

    // --- Step 2: Open the device ---
    const auto& selected = devices[selected_index];
    auto create_result = hailort::Device::create_pcie(selected);
    if (!create_result) {
        m.hailo_available = false;
        m.unavailable_reason = std::format(
            "Failed to open Hailo device at {}: status={}",
            selected.dev_id, static_cast<int>(create_result.status()));
        HQ_LOG_WARN("[Hailo8lEncoder] {}", m.unavailable_reason);
        return;
    }

    m.device.reset(create_result.release());
    m.hailo_available = true;
    HQ_LOG_INFO("[Hailo8lEncoder] Opened Hailo device {} (PCIe {})",
                selected.dev_id, selected.bus_rev);

    // --- Step 3: Create VDevice (virtual device for inference) ---
    {
        hailo::VDeviceParams vdev_params{};
        auto vdev_result = hailort::VDevice::create(vdev_params);
        if (!vdev_result) {
            m.hailo_available = false;
            m.unavailable_reason = std::format(
                "VDevice creation failed: status={}",
                static_cast<int>(vdev_result.status()));
            HQ_LOG_WARN("[Hailo8lEncoder] {}", m.unavailable_reason);
            return;
        }
        m.vdevice.reset(vdev_result.release());
    }

    // --- Step 4: Load HEF if path provided ---
    if (m.hef_path.empty()) {
        m.hef_loaded = false;
        m.unavailable_reason = "HEF path not provided — Hailo device present but no compiled model";
        HQ_LOG_INFO("[Hailo8lEncoder] {}. CPU fallback will be used.", m.unavailable_reason);
        // Device IS present — telemetry can still work. Inference unavailable until HEF provided.
        return;
    }

    if (!std::filesystem::exists(m.hef_path)) {
        m.hef_loaded = false;
        m.unavailable_reason = std::format(
            "HEF file not found: {} — compile CLIP text encoder to Hailo Executable Format",
            m.hef_path.string());
        HQ_LOG_WARN("[Hailo8lEncoder] {}", m.unavailable_reason);
        return;
    }

    // Read HEF into memory
    {
        std::ifstream f{m.hef_path, std::ios::binary | std::ios::ate};
        if (!f) {
            m.hef_loaded = false;
            m.unavailable_reason = std::format("Cannot open HEF file: {}", m.hef_path.string());
            HQ_LOG_WARN("[Hailo8lEncoder] {}", m.unavailable_reason);
            return;
        }
        auto sz = f.tellg();
        f.seekg(0, std::ios::beg);
        m.hef_buffer.resize(static_cast<std::size_t>(sz));
        f.read(reinterpret_cast<char*>(m.hef_buffer.data()), sz);
    }

    // Configure VDevice with HEF
    {
        auto config_result = m.vdevice->configure(
            hailort::BufferView{m.hef_buffer.data(), m.hef_buffer.size()});
        if (!config_result) {
            m.hef_loaded = false;
            m.unavailable_reason = std::format(
                "HEF configuration failed: status={}. HEF may be incompatible with this device.",
                static_cast<int>(config_result.status()));
            HQ_LOG_WARN("[Hailo8lEncoder] {}", m.unavailable_reason);
            return;
        }
    }

    // Get infer model
    {
        auto model_result = m.vdevice->get_infer_model();
        if (!model_result) {
            m.hef_loaded = false;
            m.unavailable_reason = std::format(
                "get_infer_model failed: status={}",
                static_cast<int>(model_result.status()));
            HQ_LOG_WARN("[Hailo8lEncoder] {}", m.unavailable_reason);
            return;
        }
        m.infer_model = std::move(model_result.value());
    }

    // Create async infer runner
    {
        auto runner_result = m.infer_model->create_infer_runner();
        if (!runner_result) {
            m.hef_loaded = false;
            m.unavailable_reason = std::format(
                "create_infer_runner failed: status={}",
                static_cast<int>(runner_result.status()));
            HQ_LOG_WARN("[Hailo8lEncoder] {}", m.unavailable_reason);
            return;
        }
        m.runner = std::move(runner_result.value());
    }

    m.hef_loaded = true;
    m.unavailable_reason.clear();
    HQ_LOG_INFO("[Hailo8lEncoder] HEF loaded successfully: {}. Async inference ready.",
                m.hef_path.string());

#else
    (void)pcie_address;
    (void)hef_path;
    impl_->hailo_available = false;
    impl_->unavailable_reason = "HailoRT SDK not compiled — rebuild with -DHAILO_BUILD or install HailoRT";
    HQ_LOG_WARN("[Hailo8lEncoder] {}", impl_->unavailable_reason);
#endif
}

Hailo8lEncoder::~Hailo8lEncoder() = default;

std::expected<NpuEncodeResult, std::string>
Hailo8lEncoder::encode(const NpuEncodeRequest& req) {
    auto& m = *impl_;

    if (!m.hailo_available) {
        return std::unexpected{
            std::format("Hailo-8L not available: {}", m.unavailable_reason)};
    }

    if (!m.hef_loaded) {
        return std::unexpected{
            std::format("Hailo-8L device present but HEF not loaded: {}", m.unavailable_reason)};
    }

#if HAILO_ENCODER_HAS_HAILORT
    if (!m.runner) {
        return std::unexpected{"Hailo async infer runner not initialized"};
    }

    // --- Tokenize prompt with CLIPTokenizer ---
    const std::size_t seq_len = req.max_seq_len > 0 ? req.max_seq_len : 77;
    CLIPTokenizer tokenizer;
    std::vector<std::int64_t> token_ids = tokenizer.encode(req.prompt, seq_len);

    // NOTE: Hailo HEF expects specific input tensor shape.
    // The HEF compiled for CLIP text encoder should accept int64[1,77].
    // If the HEF uses float input, conversion happens here.

    // --- Build Hailo input tensor ---
    // HailoRT async runner accepts buffer views.
    // Input: token_ids as int64_t array.
    constexpr std::size_t input_bytes = 77 * sizeof(std::int64_t);
    std::array<std::uint8_t, input_bytes> input_buffer{};
    std::memcpy(input_buffer.data(), token_ids.data(), input_bytes);

    hailort::BufferView input_view{input_buffer.data(), input_buffer.size()};

    // --- Allocate output buffer ---
    // CLIP output: float[1, 77, hidden_dim] where hidden_dim is 768 (SD1.5) or 2048 (SDXL)
    const std::size_t hidden_dim = req.max_seq_len == 77 ? 768UL : 2048UL;
    const std::size_t output_elems = seq_len * hidden_dim;
    const std::size_t output_bytes = output_elems * sizeof(float);
    std::vector<std::uint8_t> output_buffer(output_bytes);
    hailort::BufferView output_view{output_buffer.data(), output_buffer.size()};

    // --- Submit async inference ---
    auto submit_result = m.runner->run_async(input_view, output_view);
    if (!submit_result) {
        return std::unexpected{std::format(
            "Hailo async inference submission failed: status={}",
            static_cast<int>(submit_result.status()))};
    }

    // --- Wait for completion ---
    // HailoRT async runner has a wait() method or callback-based completion.
    // For simplicity in v2.1, we block until completion. Future: callback-based
    // overlap with GPU denoising (the real Cerberus power move).
    auto wait_result = m.runner->wait();
    if (!wait_result) {
        return std::unexpected{std::format(
            "Hailo async inference wait failed: status={}",
            static_cast<int>(wait_result.status()))};
    }

    // --- Copy output to PinnedTensor ---
    NpuEncodeResult result;
    result.embeddings = PinnedTensor<float>{output_elems};
    result.embedding_count = output_elems;
    result.hidden_dim = hidden_dim;
    result.cfg_enabled = req.guidance_scale > 1.0f;

    const float* src = reinterpret_cast<const float*>(output_buffer.data());
    auto s = result.embeddings.span();
    std::copy_n(src, std::min(s.size(), output_elems), s.begin());

    // --- Unconditional embeddings (empty prompt for CFG) ---
    result.uncond_embeddings = PinnedTensor<float>{output_elems};
    if (req.guidance_scale > 1.0f) {
        std::vector<std::int64_t> empty_ids(seq_len, 0);
        std::array<std::uint8_t, input_bytes> uncond_input_buffer{};
        std::memcpy(uncond_input_buffer.data(), empty_ids.data(), input_bytes);
        hailort::BufferView uncond_input_view{uncond_input_buffer.data(), uncond_input_buffer.size()};
        std::vector<std::uint8_t> uncond_output_buffer(output_bytes);
        hailort::BufferView uncond_output_view{uncond_output_buffer.data(), uncond_output_buffer.size()};

        auto u_submit = m.runner->run_async(uncond_input_view, uncond_output_view);
        if (u_submit) {
            auto u_wait = m.runner->wait();
            if (u_wait) {
                const float* usrc = reinterpret_cast<const float*>(uncond_output_buffer.data());
                auto us = result.uncond_embeddings.span();
                std::copy_n(usrc, std::min(us.size(), output_elems), us.begin());
            }
        }
    }

    // --- Real telemetry from Hailo device ---
    if (m.device) {
        auto power_result = m.device->get_power_measurement(
            HAILO_POWER_MEASUREMENT_TYPES__TOTAL_POWER);
        if (power_result) {
            m.last_utilization = std::clamp(power_result.value() / 1000.0f / HAILO8L_ACTIVE_POWER_W * 100.0f, 0.0f, 100.0f);
        }
        auto temp_result = m.device->get_chip_temperature();
        if (temp_result) {
            m.last_temperature = static_cast<float>(temp_result.value().ts0_temperature);
        }
    }

    result.npu_utilization = m.last_utilization;
    result.npu_temperature = m.last_temperature;
    result.encode_time_us = 0.0f;  // TODO: measure with high-res clock
    result.slot_index = 0;

    HQ_LOG_INFO("[Hailo8lEncoder] Hailo async inference complete: {}, seq={}, hidden={}, elems={}",
                req.prompt, seq_len, hidden_dim, output_elems);
    return result;

#else
    (void)req;
    return std::unexpected{"HailoRT SDK not compiled — Hailo inference unavailable"};
#endif
}

float Hailo8lEncoder::utilization() const {
    if (impl_->hailo_available)
        return impl_->last_utilization;
    return 0.0f;
}

float Hailo8lEncoder::temperature() const {
    if (impl_->hailo_available)
        return impl_->last_temperature;
    return 0.0f;
}

std::string Hailo8lEncoder::name() const {
    if (!impl_->hailo_available) return "Hailo-8L (unavailable)";
    if (!impl_->hef_loaded)       return "Hailo-8L (no HEF)";
    return "Hailo-8L";
}

bool Hailo8lEncoder::is_available() const {
    return impl_->hailo_available && impl_->hef_loaded;
}

std::string Hailo8lEncoder::unavailable_reason() const {
    return impl_->unavailable_reason;
}

// ===========================================================================
// CpuFallbackEncoder — ONNX Runtime CPU EP
// ===========================================================================

CpuFallbackEncoder::CpuFallbackEncoder(Ort::Session* session,
                                       Ort::MemoryInfo* memory_info)
    : session_{session}, memory_info_{memory_info}
{}

std::expected<NpuEncodeResult, std::string>
CpuFallbackEncoder::encode(const NpuEncodeRequest& req) {
    if (!session_)    return std::unexpected{"ORT session is null"};
    if (!memory_info_) return std::unexpected{"ORT memory info is null"};

    const std::size_t seq_len = req.max_seq_len > 0 ? req.max_seq_len : 77;
    std::vector<std::int64_t> token_ids = tokenizer_.encode(req.prompt, seq_len);

    std::vector<std::int64_t> input_shape = {
        1, static_cast<std::int64_t>(seq_len)};

    Ort::Value input_tensor = Ort::Value::CreateTensor<std::int64_t>(
        *memory_info_,
        token_ids.data(),
        token_ids.size(),
        input_shape.data(),
        input_shape.size());

    const char*  input_names[]  = {"input_ids"};
    const char*  output_names[] = {"last_hidden_state"};
    Ort::Value   output_tensor{};

    try {
        auto outputs = session_->Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor, 1,
            output_names, 1);
        if (outputs.empty()) return std::unexpected{"ORT Run returned no outputs"};
        output_tensor = std::move(outputs[0]);
    } catch (const Ort::Exception& e) {
        return std::unexpected{std::string{"ORT inference failed: "} + e.what()};
    }

    auto info = output_tensor.GetTensorTypeAndShapeInfo();
    auto out_shape = info.GetShape();
    std::size_t elem_count = info.GetElementCount();

    if (out_shape.size() < 3 || out_shape[1] < 0) {
        return std::unexpected{"Unexpected ORT output shape"};
    }

    const std::size_t out_seq_len = static_cast<std::size_t>(out_shape[1]);
    const std::size_t hidden_dim  =
        out_shape.size() >= 3 ? static_cast<std::size_t>(out_shape[2]) : 768UL;
    const std::size_t actual_count = elem_count > 0 ? elem_count : out_seq_len * hidden_dim;

    NpuEncodeResult result;
    result.embeddings    = PinnedTensor<float>{actual_count};
    result.embedding_count = actual_count;
    result.hidden_dim    = hidden_dim;
    result.cfg_enabled   = req.guidance_scale > 1.0f;

    const float* src = output_tensor.GetTensorData<float>();
    auto s = result.embeddings.span();
    std::copy_n(src, std::min(s.size(), actual_count), s.begin());

    result.uncond_embeddings = PinnedTensor<float>{actual_count};
    if (req.guidance_scale > 1.0f) {
        std::vector<std::int64_t> empty_ids(seq_len, 0);
        Ort::Value empty_tensor = Ort::Value::CreateTensor<std::int64_t>(
            *memory_info_,
            empty_ids.data(), empty_ids.size(),
            input_shape.data(), input_shape.size());
        try {
            auto uncond_outs = session_->Run(
                Ort::RunOptions{nullptr},
                input_names, &empty_tensor, 1,
                output_names, 1);
            if (!uncond_outs.empty() && uncond_outs[0].IsTensor()) {
                const float* usrc = uncond_outs[0].GetTensorData<float>();
                const std::size_t ucnt = uncond_outs[0]
                    .GetTensorTypeAndShapeInfo().GetElementCount();
                auto us = result.uncond_embeddings.span();
                std::copy_n(usrc, std::min(us.size(), ucnt), us.begin());
            }
        } catch (const Ort::Exception&) {
            // Zero-fill is safe fallback
        }
    }

    result.npu_utilization  = 0.0f;
    result.npu_temperature  = 0.0f;
    result.encode_time_us   = 0.0f;
    result.slot_index       = 0;

    HQ_LOG_INFO("[CpuFallbackEncoder] ONNX CPU inference complete: {}, seq={}, hidden={}, elems={}",
                req.prompt, out_seq_len, hidden_dim, actual_count);
    return result;
}

float CpuFallbackEncoder::utilization() const {
    return 0.0f;
}

float CpuFallbackEncoder::temperature() const {
    return 0.0f;
}

std::string CpuFallbackEncoder::name() const {
    return session_ ? "ONNX-CPU" : "ONNX-CPU (null session)";
}

bool CpuFallbackEncoder::is_available() const {
    return session_ != nullptr;
}

// ===========================================================================
// NpuEncoderFactory — probe hardware -> best encoder
// ===========================================================================

std::unique_ptr<INpuEncoder>
NpuEncoderFactory::create_best_available(
    Ort::Session*    ort_session,
    Ort::MemoryInfo* ort_memory_info,
    const std::filesystem::path& hef_path)
{
    // ---- Priority 1: Hailo-8L ----
    {
        auto hailo = std::make_unique<Hailo8lEncoder>("", hef_path);
        if (hailo->is_available()) {
            HQ_LOG_INFO("[NpuEncoderFactory] Using Hailo-8L encoder with HEF: {}",
                        hef_path.string());
            return hailo;
        }
        // Device present but no HEF? Log diagnostic, then try CPU fallback.
        if (hailo->unavailable_reason().find("HEF") != std::string::npos ||
            hailo->unavailable_reason().find("No Hailo") == std::string::npos) {
            HQ_LOG_INFO("[NpuEncoderFactory] Hailo-8L detected but unavailable: {} — "
                        "falling back to CPU. Provide HEF to enable NPU acceleration.",
                        hailo->unavailable_reason());
        }
    }

    // ---- Priority 2: ONNX Runtime CPU EP ----
    if (ort_session != nullptr) {
        HQ_LOG_INFO("[NpuEncoderFactory] Using ONNX Runtime CPU encoder");
        return std::make_unique<CpuFallbackEncoder>(
            ort_session, ort_memory_info);
    }

    // ---- No encoder available ----
    HQ_LOG_WARN("[NpuEncoderFactory] No NPU or ORT encoder available — returning nullptr");
    return nullptr;
}

// ===========================================================================
// NpuEncoderFactory::enumerate_devices — multi-device enumeration for clustering
// ===========================================================================

std::vector<NpuEncoderFactory::DeviceInfo>
NpuEncoderFactory::enumerate_devices(const std::filesystem::path& hef_path) {
    std::vector<DeviceInfo> devices;

#if HAILO_ENCODER_HAS_HAILORT
    auto scan_result = hailort::Device::scan_pcie();
    if (!scan_result || scan_result.value().empty()) {
        HQ_LOG_DEBUG("[NpuEncoderFactory] enumerate_devices: no Hailo devices found");
        return devices;
    }

    for (const auto& dev : scan_result.value()) {
        DeviceInfo info;
        info.pcie_address = dev.dev_id;

        // Try to open the device to check if HEF loads
        auto probe = std::make_unique<Hailo8lEncoder>(dev.dev_id, hef_path);
        if (probe->is_available()) {
            info.hef_loaded = true;
            info.status = "ready";
        } else {
            info.hef_loaded = false;
            std::string reason = probe->unavailable_reason();
            if (reason.find("HEF") != std::string::npos ||
                reason.find("hef") != std::string::npos) {
                info.status = "no HEF";
            } else if (reason.find("open") != std::string::npos ||
                       reason.find("Failed") != std::string::npos) {
                info.status = "busy";
            } else {
                info.status = "unavailable";
            }
        }
        devices.push_back(std::move(info));
    }

    HQ_LOG_INFO("[NpuEncoderFactory] enumerate_devices: {} Hailo device(s) found", devices.size());
#else
    (void)hef_path;
    HQ_LOG_DEBUG("[NpuEncoderFactory] enumerate_devices: HailoRT SDK not compiled");
#endif

    return devices;
}

} // namespace hq::npu
