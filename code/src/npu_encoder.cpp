/// @file npu_encoder.cpp
/// @copyright Copyright (c) 2026 D Hargreaves (AKA Roylepython). LamiaFabrica. All rights reserved.
/// NPU Encoder abstraction layer — implementation.
///
/// Houses the concrete encoder implementations and the factory that
/// probes system hardware to select the best available backend.
///
/// @version 1.0.0

#include "hq/npu_encoder.hpp"
#include "hq/cxx26_features.hpp"
#include "hq/logger.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace hq::npu {

// ===========================================================================
// Hailo8lEncoder — skeleton implementation
// ===========================================================================

struct Hailo8lEncoder::Impl {
    std::string pcie_address;
    bool        hailo_available{false};
    float       last_utilization{0.0f};
    float       last_temperature{0.0f};

    std::unique_ptr<SyntheticNpuEncoder> synthetic_fallback;

    explicit Impl(std::string addr)
        : pcie_address{std::move(addr)}
        , synthetic_fallback{std::make_unique<SyntheticNpuEncoder>()}
    {}
};

Hailo8lEncoder::Hailo8lEncoder(const std::string& pcie_address)
    : impl_{std::make_unique<Impl>(pcie_address)}
{
    // Production builds would call HailoRT here:
    //   auto devices = hailort::Device::scan_pcie();
    //   auto dev = hailort::Device::create_pcie(impl_->pcie_address);
    //   impl_->hailo_available = true;
    //
    // Skeleton: always unavailable until the HailoRT SDK + HEF are
    // compiled on the actual UM790 Pro hardware.
    impl_->hailo_available = false;
}

Hailo8lEncoder::~Hailo8lEncoder() = default;

std::expected<NpuEncodeResult, std::string>
Hailo8lEncoder::encode(const NpuEncodeRequest& req) {
    if (impl_->hailo_available) {
        // ---- Production path: HailoRT async inference ----
        //
        // In production this would:
        //   1. Tokenize req.prompt → int64_t[77] token ids
        //   2. Create hailo_async_infer_runner from compiled HEF
        //   3. Submit token ids as input tensor
        //   4. Wait for async callback with output embeddings
        //   5. Copy float* embeddings to NpuEncodeResult::embeddings
        //
        // Skeleton: not yet wired — fall through to synthetic.
        HQ_LOG_INFO("[Hailo8lEncoder] Production path — HailoRT async infer for prompt: {}", req.prompt);
    }

    HQ_LOG_INFO("[Hailo8lEncoder] Skeleton mode — delegating to synthetic encoder for: {}", req.prompt);

    return impl_->synthetic_fallback->encode(req);
}

float Hailo8lEncoder::utilization() const {
    if (impl_->hailo_available)
        return impl_->last_utilization;
    return impl_->synthetic_fallback->utilization();
}

float Hailo8lEncoder::temperature() const {
    if (impl_->hailo_available)
        return impl_->last_temperature;
    return impl_->synthetic_fallback->temperature();
}

std::string Hailo8lEncoder::name() const {
    return "Hailo-8L (skeleton)";
}

bool Hailo8lEncoder::is_available() const {
    return impl_->hailo_available;
}

// ===========================================================================
// SyntheticNpuEncoder — deterministic XOR embeddings for CI/testing
// ===========================================================================

SyntheticNpuEncoder::SyntheticNpuEncoder() = default;

std::expected<NpuEncodeResult, std::string>
SyntheticNpuEncoder::encode(const NpuEncodeRequest& req) {
    // ---- Simulated NPU text encoding ----
    // Mimics Hailo-8L behaviour: burst encode → idle → burst.
    // This is the logic formerly in NpuDmaPipeline::Impl::simulate_encode().

    std::this_thread::sleep_for(std::chrono::microseconds{
        2000 + static_cast<long long>(encode_count_ % 5) * 1200});

    NpuEncodeResult result;
    constexpr std::size_t embedding_bytes = MAX_EMBEDDING_BYTES;
    const std::size_t count = embedding_bytes / sizeof(float);

    // Allocate both conditional and unconditional embeddings.
    result.embeddings         = PinnedTensor<float>{count};
    result.uncond_embeddings  = PinnedTensor<float>{count};
    result.embedding_count    = count;
    result.hidden_dim         = req.max_seq_len == 77 ? 768UL : 2048UL;
    result.cfg_enabled        = req.guidance_scale > 1.0f;

    // Fill conditional buffer with synthetic embeddings.
    {
        auto s = result.embeddings.span();
        for (std::size_t i = 0; i < std::min(s.size(), count); ++i) {
            s[i] = 0.1f * static_cast<float>(
                (req.prompt[i % req.prompt.size()] ^
                 (i * 7 + encode_count_)) & 0xFF) / 25.6f;
        }
    }

    // Fill unconditional buffer (empty-prompt embeddings for CFG).
    {
        auto s = result.uncond_embeddings.span();
        for (std::size_t i = 0; i < std::min(s.size(), count); ++i) {
            s[i] = 0.1f * static_cast<float>(
                ((' ' ^ (i * 7 + encode_count_)) & 0xFF)) / 25.6f;
        }
    }

    // ---- Time-varying NPU pseudo-telemetry ----
    const auto now  = std::chrono::steady_clock::now().time_since_epoch();
    const auto ms   = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
    last_utilization_  = 65.0f + 25.0f * std::sin(static_cast<float>(ms) * 0.003f);
    last_temperature_  = 48.0f + 7.0f * std::sin(static_cast<float>(ms) * 0.0015f);

    result.npu_utilization  = last_utilization_;
    result.npu_temperature  = last_temperature_;
    result.encode_time_us   = static_cast<float>(2000 + (encode_count_ % 5) * 1200);
    result.slot_index       = 0;  // caller will fill if needed

    ++encode_count_;
    return result;
}

float SyntheticNpuEncoder::utilization() const {
    return last_utilization_;
}

float SyntheticNpuEncoder::temperature() const {
    return last_temperature_;
}

std::string SyntheticNpuEncoder::name() const {
    return "Synthetic-XOR";
}

bool SyntheticNpuEncoder::is_available() const {
    return true;  // always available
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

    // ---- Tokenize prompt with CLIPTokenizer ----
    const std::size_t seq_len = req.max_seq_len > 0 ? req.max_seq_len : 77;
    std::vector<std::int64_t> token_ids = tokenizer_.encode(req.prompt, seq_len);

    // ---- Build ONNX input tensor ----
    std::vector<std::int64_t> input_shape = {
        1, static_cast<std::int64_t>(seq_len)};

    Ort::Value input_tensor = Ort::Value::CreateTensor<std::int64_t>(
        *memory_info_,
        token_ids.data(),
        token_ids.size(),
        input_shape.data(),
        input_shape.size());

    // ---- Run inference ----
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

    // ---- Copy output embeddings to result ----
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

    // Unconditional pass: run CLIP with an empty (zero-padded) prompt for CFG.
    // When CFG is disabled (guidance_scale <= 1.0), zero-fill is sufficient.
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
            // Zero-fill is safe fallback: unconditional branch stays at zeros
        }
    }

    result.npu_utilization  = 0.0f;     // CPU EP — no NPU
    result.npu_temperature  = 0.0f;
    result.encode_time_us   = 0.0f;     // caller can measure
    result.slot_index       = 0;

    HQ_LOG_INFO("[CpuFallbackEncoder] ONNX CPU inference complete: {}, seq={}, hidden={}, elems={}",
                req.prompt, out_seq_len, hidden_dim, actual_count);

    return result;
}

float CpuFallbackEncoder::utilization() const {
    return 0.0f;  // CPU encoding doesn't use the NPU
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
// NpuEncoderFactory — probe hardware → best encoder
// ===========================================================================

std::unique_ptr<INpuEncoder>
NpuEncoderFactory::create_best_available(
    Ort::Session*    ort_session,
    Ort::MemoryInfo* ort_memory_info)
{
    // ---- Priority 1: Hailo-8L ----
    {
        auto hailo = std::make_unique<Hailo8lEncoder>("");
        if (hailo->is_available()) {
            HQ_LOG_INFO("[NpuEncoderFactory] Using Hailo-8L encoder");
            return hailo;
        }
    }

    // ---- Priority 2: ONNX Runtime CPU EP ----
    if (ort_session != nullptr) {
        HQ_LOG_INFO("[NpuEncoderFactory] Hailo-8L unavailable, using ONNX Runtime CPU encoder");
        return std::make_unique<CpuFallbackEncoder>(
            ort_session, ort_memory_info);
    }

    // ---- Priority 3: Synthetic (always available) ----
    HQ_LOG_INFO("[NpuEncoderFactory] Falling back to Synthetic encoder (CI/demo mode)");
    return std::make_unique<SyntheticNpuEncoder>();
}

} // namespace hq::npu
