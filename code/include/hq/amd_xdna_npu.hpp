#pragma once
/// @file amd_xdna_npu.hpp
/// @copyright Copyright (c) 2026 D Hargreaves (AKA Roylepython). LamiaFabrica. All rights reserved.
/// AMD XDNA NPU backend for Ryzen AI (7940HS, UM790 Pro).
///
/// Wraps AMD XRT (Xilinx Runtime) to submit inference jobs to the XDNA NPU.
/// The amdxdna.ko driver is open-source and upstreamed in Linux 6.14+.
///
/// Architecture:
///   - Windows/MinGW host: compiles as stub, returns honest errors.
///   - Linux host with XRT installed: real XRT device/kernel execution.
///
/// Requires:
///   - Linux kernel >= 6.10 (6.14 has upstream amdxdna driver)
///   - XRT base package + XDNA plugin (from amd/xdna-driver repo)
///   - Valid XCLBIN or XSA file compiled for XDNA target
///
/// @version 1.0.0
/// @author LamiaFabrica Team

#include "hq/cxx26_features.hpp"
#include "hq/npu_encoder.hpp"
#include "hq/npu_accelerator.hpp"

#include <filesystem>
#include <memory>
#include <span>
#include <string>

namespace hq::npu {

// ===========================================================================
// AmdXdnaNpu — low-level XRT wrapper for AMD XDNA NPU
// ===========================================================================

class AmdXdnaNpu {
public:
    AmdXdnaNpu();
    ~AmdXdnaNpu() noexcept;

    // Non-copyable (unique XRT device handle)
    AmdXdnaNpu(const AmdXdnaNpu&) = delete;
    AmdXdnaNpu& operator=(const AmdXdnaNpu&) = delete;

    // Movable
    AmdXdnaNpu(AmdXdnaNpu&&) noexcept;
    AmdXdnaNpu& operator=(AmdXdnaNpu&&) noexcept;

    /// @brief Open the XDNA NPU device and load the XCLBIN.
    /// @param xclbin_path Path to compiled XCLBIN/XSA for XDNA target.
    /// @return std::expected<void, std::string>
    [[nodiscard]] std::expected<void, std::string> open(
        const std::filesystem::path& xclbin_path);

    /// @brief Close device and release XRT resources.
    void close() noexcept;

    /// @brief True when device is open and XCLBIN loaded.
    [[nodiscard]] bool is_open() const noexcept;

    /// @brief Run a SAXPY kernel: y[i] = a * x[i] + y[i]
    /// @param a Scalar multiplier
    /// @param x Input vector (device-readable)
    /// @param y In/out vector (device-read/write)
    /// @return std::expected<void, std::string>
    [[nodiscard]] std::expected<void, std::string> run_saxpy(
        float a,
        std::span<const float> x,
        std::span<float> y);

    /// @brief Run a small 1D convolution kernel.
    /// @param input Input tensor
    /// @param kernel Convolution weights
    /// @param output Output tensor
    /// @return std::expected<void, std::string>
    [[nodiscard]] std::expected<void, std::string> run_conv1d(
        std::span<const float> input,
        std::span<const float> kernel,
        std::span<float> output);

    /// @brief NPU utilization % (from XRT telemetry).
    /// @return -1.0f when not available.
    [[nodiscard]] float utilization() const noexcept;

    /// @brief NPU temperature °C (from XRT telemetry).
    /// @return -1.0f when not available.
    [[nodiscard]] float temperature() const noexcept;

    /// @brief Human-readable backend name.
    [[nodiscard]] std::string name() const noexcept;

    /// @brief Why open() failed.
    [[nodiscard]] std::string unavailable_reason() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// ===========================================================================
// AmdXdnaEncoder — INpuEncoder implementation using XDNA
// ===========================================================================

class AmdXdnaEncoder final : public INpuEncoder {
public:
    explicit AmdXdnaEncoder(const std::filesystem::path& xclbin_path);
    ~AmdXdnaEncoder() override = default;

    [[nodiscard]] std::expected<NpuEncodeResult, std::string>
    encode(const NpuEncodeRequest& req) override;

    [[nodiscard]] float utilization() const override;
    [[nodiscard]] float temperature() const override;
    [[nodiscard]] std::string name() const override;
    [[nodiscard]] bool is_available() const override;
    [[nodiscard]] bool synthetic_mode() const noexcept override;
    [[nodiscard]] std::string unavailable_reason() const override;

private:
    AmdXdnaNpu npu_;
    std::string unavailable_reason_;
};

// ===========================================================================
// AmdXdnaPostProcessor — INpuPostProcessor implementation using XDNA
// ===========================================================================

class AmdXdnaPostProcessor final : public INpuPostProcessor {
public:
    explicit AmdXdnaPostProcessor(const std::filesystem::path& xclbin_path);
    ~AmdXdnaPostProcessor() override = default;

    [[nodiscard]] std::expected<NpuPostProcessResult, std::string>
    post_process(const NpuPostProcessRequest& req) override;

    [[nodiscard]] std::expected<void, std::string>
    blend_noise_cfg(std::span<float> noise_out,
                    std::span<const float> noise_uncond,
                    float guidance_scale) noexcept override;

    [[nodiscard]] bool can_handle(NpuTaskType task) const override;
    [[nodiscard]] bool is_available() const override;
    [[nodiscard]] std::string name() const override;
    [[nodiscard]] float utilization() const override;
    [[nodiscard]] bool synthetic_mode() const noexcept override;
    [[nodiscard]] std::string unavailable_reason() const override;

private:
    AmdXdnaNpu npu_;
    std::string unavailable_reason_;
};

} // namespace hq::npu