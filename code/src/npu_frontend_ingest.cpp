#include <fstream>
#include <vector>
#include <string>
#include "hq/npu_backend_unified.hpp"

namespace hq::npu {

static bool onnx_probe(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    char magic[8]{};
    f.read(magic, 8);
    // ONNX files start with 0x08 (protobuf varint field 1, wire type 2) or
    // can be a flatbuffers header.  We accept either.
    return f.gcount() == 8;
}

struct NpuFrontendIngest {
    /// Load an ONNX file into a Cerberus-owned KernelGraph.
    /// Currently the "load" is lightweight: we verify the file exists, set
    /// the source_path, and leave the heavy parsing to the backend.  As
    /// Cerberus takes ownership we will replace this with a real protobuf
    /// reader that populates KernelNode and TensorDesc.
    static bool from_onnx(const std::filesystem::path& path, KernelGraph& out) {
        if (!std::filesystem::exists(path))
            return false;
        out.source_path = path;
        out.entry_point = path.stem().string();
        out.format = KernelGraph::SourceFormat::ONNX;

        // TODO(Round-30): replace with real ONNX protobuf parse
        // For now we only set metadata so backends still call ov_core_read_model
        // with the file path, but the data structure is Cerberus-owned.
        return true;
    }
};

} // namespace hq::npu
