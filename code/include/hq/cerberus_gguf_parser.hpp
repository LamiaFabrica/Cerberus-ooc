#pragma once
/// @file cerberus_gguf_parser.hpp
/// @copyright Copyright (c) 2026 D Hargreaves. All rights reserved.
///
/// Lightweight GGUF metadata parser for Cerberus.
/// Reads headers and tensor metadata without loading multi-gigabyte weights.
///
/// @version 1.0.0

#include <cstdint>
#include <string>
#include <vector>
#include <optional>
#include <string_view>
#include <fstream>
#include <memory>
#include <variant>

namespace hq::cerberus {

// ==========================================================================
// GGUF Format Constants
// ==========================================================================

inline constexpr uint32_t GGUF_MAGIC_BE = 0x47475546; ///< "GGUF" big-endian
inline constexpr uint32_t GGUF_MAGIC_LE = 0x46554747; ///< "GGUF" little-endian
inline constexpr uint32_t GGUF_VERSION_V3 = 3;

// ==========================================================================
// GGML Quantization Type (subset relevant to Cerberus)
// ==========================================================================

enum class GgmlType : uint32_t {
    F32     = 0,
    F16     = 1,
    Q4_0    = 2,
    Q5_0    = 6,
    Q8_0    = 7,
    Q4_K    = 12, ///< Q4_K_M uses this
    Q5_K    = 13, ///< Q5_K_M uses this
    Q6_K    = 14,
    IQ4_NL  = 23,
};

[[nodiscard]] const char* ggml_type_name(GgmlType t) noexcept;
[[nodiscard]] bool ggml_type_is_quantized(GgmlType t) noexcept;

// ==========================================================================
// Tensor Info
// ==========================================================================

struct GgufTensorInfo {
    std::string name;
    std::vector<uint64_t> shape;
    GgmlType dtype{GgmlType::F32};
    uint64_t offset_in_file{0};
    uint64_t size_bytes{0};

    [[nodiscard]] uint64_t num_elements() const noexcept;
    [[nodiscard]] bool is_quantized() const noexcept;
};

// ==========================================================================
// Metadata Value (simplified KV store)
// ==========================================================================

enum class GgufMetadataType : uint32_t {
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12,
};

struct GgufMetadataValue {
    std::string key;
    GgufMetadataType type{GgufMetadataType::STRING};
    std::variant<
        uint64_t,
        int64_t,
        double,
        bool,
        std::string,
        std::vector<GgufMetadataValue>
    > value;
};

// ==========================================================================
// Parsed Header
// ==========================================================================

struct GgufHeader {
    uint32_t magic{0};
    uint32_t version{0};
    uint64_t tensor_count{0};
    uint64_t metadata_kv_count{0};
    [[nodiscard]] bool isValid() const noexcept;
    [[nodiscard]] bool isLittleEndian() const noexcept;
};

// ==========================================================================
// GGUF Parser
// ==========================================================================

class GgufParser {
public:
    GgufParser() = default;
    ~GgufParser() = default;

    /// Open and parse the header of a GGUF file (does not load weights).
    [[nodiscard]] bool parse_header(const std::string& filepath);

    [[nodiscard]] const GgufHeader& header() const noexcept { return header_; }
    [[nodiscard]] const std::vector<GgufTensorInfo>& tensors() const noexcept { return tensors_; }
    [[nodiscard]] std::vector<GgufTensorInfo> tensors_with_type(GgmlType t) const;
    [[nodiscard]] std::optional<std::string> get_metadata_string(std::string_view key) const;
    [[nodiscard]] std::optional<uint64_t> get_metadata_uint64(std::string_view key) const;
    [[nodiscard]] std::optional<int64_t> get_metadata_int64(std::string_view key) const;

    /// Detect quantization family from metadata / tensor types.
    [[nodiscard]] std::optional<std::string> detect_quantization_family() const;

    /// Reset all parsed state.
    void reset();

private:
    GgufHeader header_;
    std::vector<GgufTensorInfo> tensors_;
    std::vector<GgufMetadataValue> metadata_;

    [[nodiscard]] uint32_t read_u32_le(std::istream& in) const;
    [[nodiscard]] uint64_t read_u64_le(std::istream& in) const;
    [[nodiscard]] int64_t read_i64_le(std::istream& in) const;
    [[nodiscard]] double read_f64_le(std::istream& in) const;
    [[nodiscard]] std::string read_string(std::istream& in) const;
    bool parse_metadata_kv(std::istream& in);
    bool parse_tensor_info(std::istream& in);
};

} // namespace hq::cerberus
