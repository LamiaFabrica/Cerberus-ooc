/// @file cerberus_gguf_parser.cpp
/// @copyright Copyright © 2026, all rights reserved | David Hargreaves aka Roylepython trading as The Medusa Initiative & Lamia Fabrica
///
/// Lightweight GGUF metadata parser.
///
/// @version 1.0.0

#include "hq/cerberus_gguf_parser.hpp"
#include <cstring>
#include <array>

namespace hq::cerberus {

// ==========================================================================
// Helpers
// ==========================================================================

const char* ggml_type_name(GgmlType t) noexcept {
    switch (t) {
        case GgmlType::F32:    return "F32";
        case GgmlType::F16:    return "F16";
        case GgmlType::Q4_0:   return "Q4_0";
        case GgmlType::Q5_0:   return "Q5_0";
        case GgmlType::Q8_0:   return "Q8_0";
        case GgmlType::Q4_K:   return "Q4_K";
        case GgmlType::Q5_K:   return "Q5_K";
        case GgmlType::Q6_K:   return "Q6_K";
        case GgmlType::IQ4_NL: return "IQ4_NL";
        default: return "UNKNOWN";
    }
}

bool ggml_type_is_quantized(GgmlType t) noexcept {
    switch (t) {
        case GgmlType::Q4_0:
        case GgmlType::Q5_0:
        case GgmlType::Q8_0:
        case GgmlType::Q4_K:
        case GgmlType::Q5_K:
        case GgmlType::Q6_K:
        case GgmlType::IQ4_NL:
            return true;
        default:
            return false;
    }
}

uint64_t GgufTensorInfo::num_elements() const noexcept {
    uint64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}

bool GgufTensorInfo::is_quantized() const noexcept {
    return ggml_type_is_quantized(dtype);
}

bool GgufHeader::isValid() const noexcept {
    return (magic == GGUF_MAGIC_LE || magic == GGUF_MAGIC_BE) && version == GGUF_VERSION_V3;
}

bool GgufHeader::isLittleEndian() const noexcept {
    return magic == GGUF_MAGIC_LE;
}

// ==========================================================================
// Reader helpers (little-endian GGUF files)
// ==========================================================================

uint32_t GgufParser::read_u32_le(std::istream& in) const {
    uint32_t v = 0;
    in.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}

uint64_t GgufParser::read_u64_le(std::istream& in) const {
    uint64_t v = 0;
    in.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}

int64_t GgufParser::read_i64_le(std::istream& in) const {
    int64_t v = 0;
    in.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}

double GgufParser::read_f64_le(std::istream& in) const {
    double v = 0;
    in.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}

std::string GgufParser::read_string(std::istream& in) const {
    uint64_t len = read_u64_le(in);
    std::string s;
    s.resize(len);
    in.read(s.data(), static_cast<std::streamsize>(len));
    return s;
}

// ==========================================================================
// Metadata KV parsing (simplified — reads and stores strings/ints)
// ==========================================================================

static bool skip_metadata_value(std::istream& in, GgufMetadataType type) {
    switch (type) {
        case GgufMetadataType::UINT8: {
            uint8_t _; in.read(reinterpret_cast<char*>(&_), sizeof(_));
            break;
        }
        case GgufMetadataType::INT8: {
            int8_t _; in.read(reinterpret_cast<char*>(&_), sizeof(_));
            break;
        }
        case GgufMetadataType::UINT16: {
            uint16_t _; in.read(reinterpret_cast<char*>(&_), sizeof(_));
            break;
        }
        case GgufMetadataType::INT16: {
            int16_t _; in.read(reinterpret_cast<char*>(&_), sizeof(_));
            break;
        }
        case GgufMetadataType::UINT32: {
            uint32_t _; in.read(reinterpret_cast<char*>(&_), sizeof(_));
            break;
        }
        case GgufMetadataType::INT32: {
            int32_t _; in.read(reinterpret_cast<char*>(&_), sizeof(_));
            break;
        }
        case GgufMetadataType::FLOAT32: {
            float _; in.read(reinterpret_cast<char*>(&_), sizeof(_));
            break;
        }
        case GgufMetadataType::BOOL: {
            uint8_t _; in.read(reinterpret_cast<char*>(&_), sizeof(_));
            break;
        }
        case GgufMetadataType::STRING: {
            uint64_t len = 0;
            in.read(reinterpret_cast<char*>(&len), sizeof(len));
            if (len > 0) in.seekg(static_cast<std::streamoff>(len), std::ios::cur);
            break;
        }
        case GgufMetadataType::UINT64: {
            uint64_t _; in.read(reinterpret_cast<char*>(&_), sizeof(_));
            break;
        }
        case GgufMetadataType::INT64: {
            int64_t _; in.read(reinterpret_cast<char*>(&_), sizeof(_));
            break;
        }
        case GgufMetadataType::FLOAT64: {
            double _; in.read(reinterpret_cast<char*>(&_), sizeof(_));
            break;
        }
        case GgufMetadataType::ARRAY: {
            uint32_t arr_type_raw = 0;
            in.read(reinterpret_cast<char*>(&arr_type_raw), sizeof(arr_type_raw));
            auto arr_type = static_cast<GgufMetadataType>(arr_type_raw);
            uint64_t arr_len = 0;
            in.read(reinterpret_cast<char*>(&arr_len), sizeof(arr_len));
            for (uint64_t i = 0; i < arr_len; ++i) {
                if (!skip_metadata_value(in, arr_type)) return false;
            }
            break;
        }
        // No default — every GGUF type 0..12 is explicitly handled.
    }
    return in.good();
}

bool GgufParser::parse_metadata_kv(std::istream& in) {
    // Read key
    std::string key = read_string(in);
    // Read value type
    uint32_t type_raw = read_u32_le(in);
    auto type = static_cast<GgufMetadataType>(type_raw);

    GgufMetadataValue mv;
    mv.key = std::move(key);
    mv.type = type;

    // Handle all GGUF metadata types (0-12)
    switch (type) {
        case GgufMetadataType::UINT8: {
            uint8_t v = 0; in.read(reinterpret_cast<char*>(&v), sizeof(v));
            mv.value = static_cast<uint64_t>(v);
            break;
        }
        case GgufMetadataType::INT8: {
            int8_t v = 0; in.read(reinterpret_cast<char*>(&v), sizeof(v));
            mv.value = static_cast<int64_t>(v);
            break;
        }
        case GgufMetadataType::UINT16: {
            uint16_t v = 0; in.read(reinterpret_cast<char*>(&v), sizeof(v));
            mv.value = static_cast<uint64_t>(v);
            break;
        }
        case GgufMetadataType::INT16: {
            int16_t v = 0; in.read(reinterpret_cast<char*>(&v), sizeof(v));
            mv.value = static_cast<int64_t>(v);
            break;
        }
        case GgufMetadataType::UINT32: {
            uint32_t v = 0; in.read(reinterpret_cast<char*>(&v), sizeof(v));
            mv.value = static_cast<uint64_t>(v);
            break;
        }
        case GgufMetadataType::INT32: {
            int32_t v = 0; in.read(reinterpret_cast<char*>(&v), sizeof(v));
            mv.value = static_cast<int64_t>(v);
            break;
        }
        case GgufMetadataType::FLOAT32: {
            float v = 0.0f; in.read(reinterpret_cast<char*>(&v), sizeof(v));
            mv.value = static_cast<double>(v);
            break;
        }
        case GgufMetadataType::BOOL: {
            uint8_t v = 0; in.read(reinterpret_cast<char*>(&v), sizeof(v));
            mv.value = (v != 0);
            break;
        }
        case GgufMetadataType::STRING: {
            mv.value = read_string(in);
            break;
        }
        case GgufMetadataType::UINT64: {
            uint64_t v = 0; in.read(reinterpret_cast<char*>(&v), sizeof(v));
            mv.value = v;
            break;
        }
        case GgufMetadataType::INT64: {
            int64_t v = 0; in.read(reinterpret_cast<char*>(&v), sizeof(v));
            mv.value = v;
            break;
        }
        case GgufMetadataType::FLOAT64: {
            double v = 0.0; in.read(reinterpret_cast<char*>(&v), sizeof(v));
            mv.value = v;
            break;
        }
        case GgufMetadataType::ARRAY: {
            // Array parsing: read type + length, then seek over values
            uint32_t arr_type_raw = read_u32_le(in);
            auto arr_type = static_cast<GgufMetadataType>(arr_type_raw);
            uint64_t arr_len = read_u64_le(in);
            // For metadata KV we store the array as a single string placeholder
            for (uint64_t i = 0; i < arr_len; ++i) {
                skip_metadata_value(in, arr_type);
            }
            mv.value = std::string("[array]");
            break;
        }
    }

    if (!in.good()) return false;
    metadata_.push_back(std::move(mv));
    return true;
}

bool GgufParser::parse_tensor_info(std::istream& in) {
    GgufTensorInfo info;
    info.name = read_string(in);
    uint32_t ndim = read_u32_le(in);
    for (uint32_t i = 0; i < ndim; ++i) {
        uint64_t dim = read_u64_le(in);
        info.shape.push_back(dim);
    }
    uint32_t dtype_raw = read_u32_le(in);
    info.dtype = static_cast<GgmlType>(dtype_raw);
    info.offset_in_file = read_u64_le(in);

    // Estimate size: for quantized types we do a rough placeholder
    info.size_bytes = ggml_type_is_quantized(info.dtype)
        ? (info.num_elements() / 2 + 32)
        : (info.num_elements() * 4);

    tensors_.push_back(std::move(info));
    return in.good();
}

// ==========================================================================
// Main parse
// ==========================================================================

bool GgufParser::parse_header(const std::string& filepath) {
    reset();
    std::ifstream in(filepath, std::ios::binary);
    if (!in) return false;

    // Magic
    header_.magic = read_u32_le(in);
    if (!header_.isValid()) return false;

    // Version
    header_.version = read_u32_le(in);
    if (header_.version != GGUF_VERSION_V3) return false;

    // Tensor count
    header_.tensor_count = read_u64_le(in);

    // Metadata KV count
    header_.metadata_kv_count = read_u64_le(in);

    // Parse metadata
    for (uint64_t i = 0; i < header_.metadata_kv_count; ++i) {
        if (!parse_metadata_kv(in)) return false;
    }

    // Parse tensor info
    for (uint64_t i = 0; i < header_.tensor_count; ++i) {
        if (!parse_tensor_info(in)) return false;
    }

    return true;
}

std::vector<GgufTensorInfo> GgufParser::tensors_with_type(GgmlType t) const {
    std::vector<GgufTensorInfo> out;
    for (const auto& ti : tensors_) {
        if (ti.dtype == t) out.push_back(ti);
    }
    return out;
}

std::optional<std::string> GgufParser::get_metadata_string(std::string_view key) const {
    for (const auto& kv : metadata_) {
        if (kv.key == key) {
            if (std::holds_alternative<std::string>(kv.value)) {
                return std::get<std::string>(kv.value);
            }
        }
    }
    return std::nullopt;
}

std::optional<uint64_t> GgufParser::get_metadata_uint64(std::string_view key) const {
    for (const auto& kv : metadata_) {
        if (kv.key == key) {
            if (std::holds_alternative<uint64_t>(kv.value)) {
                return std::get<uint64_t>(kv.value);
            }
        }
    }
    return std::nullopt;
}

std::optional<int64_t> GgufParser::get_metadata_int64(std::string_view key) const {
    for (const auto& kv : metadata_) {
        if (kv.key == key) {
            if (std::holds_alternative<int64_t>(kv.value)) {
                return std::get<int64_t>(kv.value);
            }
        }
    }
    return std::nullopt;
}

std::optional<std::string> GgufParser::detect_quantization_family() const {
    // Count tensors by family
    size_t q4_k = tensors_with_type(GgmlType::Q4_K).size();
    size_t q5_k = tensors_with_type(GgmlType::Q5_K).size();
    size_t iq4_nl = tensors_with_type(GgmlType::IQ4_NL).size();
    if (q4_k > 0 && q5_k == 0 && iq4_nl == 0) return std::string("Q4_K_M");
    if (q5_k > 0 && q4_k == 0 && iq4_nl == 0) return std::string("Q5_K_M");
    if (iq4_nl > 0 && q4_k == 0 && q5_k == 0) return std::string("IQ4_NL");
    return std::nullopt;
}

void GgufParser::reset() {
    header_ = GgufHeader{};
    tensors_.clear();
    metadata_.clear();
}

} // namespace hq::cerberus
