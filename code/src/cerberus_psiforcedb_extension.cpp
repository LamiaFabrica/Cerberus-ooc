/// @file cerberus_psiforcedb_extension.cpp
/// @copyright Copyright © 2026, all rights reserved | David Hargreaves aka Roylepython trading as The Medusa Initiative & Lamia Fabrica
/// Support: https://www.patreon.com/TheMedusaInitiative — £25/month removes ads from all software at 200 subscribers
///
/// CerberusExtension implementation — runtime wiring only.
///
/// The MultiModelExtension base is inline in the header (copied from
/// PsiForceDB extension_interface.hpp).  This file contains ONLY
/// CerberusExtension-specific behaviour: RuntimeImpl, query routing,
/// and the C factory.
///
/// @version 1.0.0

#include "hq/cerberus_psiforcedb_extension.hpp"
#include "hq/npu_backend_unified.hpp"
#include "hq/cerberus_runtime.hpp"
#include "hq/cerberus_glow_engine.hpp"
#include "hq/cerberus_gguf_parser.hpp"
#include "hq/cerberus_api_gateway.hpp"
#include "hq/cerberus_slipstream.hpp"
#include "hq/cerberus_metro.hpp"

#include <sstream>
#include <chrono>

namespace hq::cerberus::psiforcedb {

// ==========================================================================
// CerberusExtension — RuntimeImpl
// ==========================================================================

struct CerberusExtension::RuntimeImpl {
    std::unique_ptr<hq::cerberus::CerberusRuntime> runtime;
    std::unique_ptr<hq::cerberus::gateway::CerberusApiGateway> gateway;
    std::unique_ptr<hq::cerberus::slipstream::CerberusSlipstreamEngine> slipstream;

    RuntimeImpl() {
        runtime = std::make_unique<hq::cerberus::CerberusRuntime>();
        gateway = std::make_unique<hq::cerberus::gateway::CerberusApiGateway>();
        gateway->initialize();
        slipstream = std::make_unique<hq::cerberus::slipstream::CerberusSlipstreamEngine>();
        slipstream->initialize(2);
        slipstream->start();
    }
    ~RuntimeImpl() {
        if (slipstream) slipstream->stop();
        if (gateway) gateway->shutdown();
    }
};

// ==========================================================================
// CerberusExtension
// ==========================================================================

CerberusExtension::CerberusExtension()
    : MultiModelExtension(),
      runtime_(std::make_unique<RuntimeImpl>())
{
    metadata_.name = "Cerberus.InferenceEngine";
    metadata_.version = "1.0.0";
    metadata_.description = "Cerberus AI-native inference extension for PsiForceDB";
    metadata_.author = "LamiaFabrica";
    metadata_.model_type = "inference";
    metadata_.supported_queries = {"INFERENCE", "COMPILE", "STATUS", "GLOW", "GGUF", "TELEMETRY"};
}

CerberusExtension::~CerberusExtension() = default;

bool CerberusExtension::initialize(const ExtensionConfig& config) {
    if (!MultiModelExtension::initialize(config)) return false;
    if (!runtime_->gateway->isInitialized()) {
        recordExtensionError("Gateway initialization failed");
        return false;
    }
    return true;
}

bool CerberusExtension::load() {
    if (!MultiModelExtension::load()) return false;
    return runtime_ != nullptr;
}

bool CerberusExtension::unload() {
    runtime_.reset();
    return MultiModelExtension::unload();
}

ExtensionMetadata CerberusExtension::getMetadata() const {
    auto m = MultiModelExtension::getMetadata();
    m.model_type = "inference";
    m.supported_queries = {"INFERENCE", "COMPILE", "STATUS", "GLOW", "GGUF", "TELEMETRY"};
    return m;
}

bool CerberusExtension::isHealthy() const {
    return initialized_ && runtime_ != nullptr &&
           runtime_->gateway->isInitialized();
}

std::map<std::string, std::string> CerberusExtension::getStatistics() const {
    auto stats = MultiModelExtension::getStatistics();
    stats["backend"] = runtime_ && runtime_->runtime ? "native_ready" : "not_ready";
    stats["gateway_sessions"] = runtime_ ? std::to_string(runtime_->gateway->sessionCount()) : "0";
    return stats;
}

bool CerberusExtension::validateQuery(const std::string& query_string) const {
    if (!MultiModelExtension::validateQuery(query_string)) return false;
    // Accept protocol-style commands, JSON payloads, plain keywords, PFQL, or GGUF
    return query_string.find("cerberus://") == 0 ||
           query_string.find("cbr:") == 0 ||
           query_string.find("{") == 0 ||
           query_string == "status" ||
           query_string == "compile" ||
           query_string == "glow" ||
           query_string == "telemetry" ||
           query_string.find("pf://") == 0 ||
           query_string.find("gguf://") == 0;
}

QueryResult CerberusExtension::executeQuery(const Query& query) {
    auto t0 = std::chrono::steady_clock::now();
    QueryResult result = MultiModelExtension::executeQuery(query);
    if (!result.success) return result;

    if (query.query_type == "INFERENCE") {
        result = handle_run_(query);
    } else if (query.query_type == "COMPILE") {
        result = handle_compile_(query);
    } else if (query.query_type == "STATUS") {
        result = handle_status_(query);
    } else if (query.query_type == "GLOW") {
        if (runtime_ && runtime_->runtime) {
            auto* glow = runtime_->runtime->glow_engine();
            if (glow) {
                auto stats = glow->stats();
                std::map<std::string, std::string> row;
                row["active_bonds"] = std::to_string(stats.active_bond_count);
                row["reinforcements"] = std::to_string(stats.reinforcements_applied);
                row["paths_learned"] = std::to_string(stats.paths_learned);
                result.rows.push_back(row);
                result.row_count = 1;
            }
        }
    } else if (query.query_type == "GGUF") {
        result = handle_gguf_(query);
    } else if (query.query_type == "TELEMETRY") {
        result = handle_telemetry_(query);
    } else {
        result.success = false;
        result.error_message = "Unknown query type: " + query.query_type;
    }

    auto t1 = std::chrono::steady_clock::now();
    result.execution_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.metadata["extension"] = "Cerberus.InferenceEngine";
    return result;
}

// ==========================================================================
// Query handlers
// ==========================================================================

QueryResult CerberusExtension::handle_compile_(const Query& query) {
    QueryResult result;
    result.success = true;
    if (!runtime_ || !runtime_->runtime) {
        result.success = false;
        result.error_message = "CerberusRuntime not initialized";
        return result;
    }

    auto cmd_it = query.parameters.find("command");
    std::string cmd = (cmd_it != query.parameters.end()) ? cmd_it->second : query.query_string;
    auto output = runtime_->runtime->execute_command(cmd);

    hq::npu::KernelGraph graph;
    hq::npu::KernelNode node;
    node.op = hq::npu::KernelNode::Op::MatMul;
    node.name = "matmul_0";
    node.inputs = {"input_a", "input_b"};
    node.outputs = {"output_c"};
    graph.nodes.push_back(std::move(node));

    std::map<std::string, std::string> row;
    row["command"] = cmd;
    row["result"] = output;
    row["graph_nodes"] = std::to_string(graph.nodes.size());
    result.rows.push_back(row);
    result.row_count = 1;
    return result;
}

QueryResult CerberusExtension::handle_run_(const Query& query) {
    QueryResult result;
    result.success = true;
    if (!runtime_ || !runtime_->runtime) {
        result.success = false;
        result.error_message = "CerberusRuntime not initialized";
        return result;
    }

    auto cmd_it = query.parameters.find("command");
    std::string cmd = (cmd_it != query.parameters.end()) ? cmd_it->second : query.query_string;

    auto output = runtime_->runtime->execute_command(cmd);
    std::map<std::string, std::string> row;
    row["command"] = cmd;
    row["result"] = output;
    result.rows.push_back(row);
    result.row_count = 1;
    return result;
}

QueryResult CerberusExtension::handle_status_(const Query& /*query*/) {
    QueryResult result;
    result.success = true;
    std::map<std::string, std::string> row;
    row["name"] = "Cerberus.InferenceEngine";
    row["version"] = "1.0.0";
    row["healthy"] = isHealthy() ? "true" : "false";
    if (runtime_ && runtime_->gateway) {
        row["gateway_sessions"] = std::to_string(runtime_->gateway->sessionCount());
    }
    result.rows.push_back(row);
    result.row_count = 1;
    return result;
}

QueryResult CerberusExtension::handle_gguf_(const Query& query) {
    QueryResult result;
    result.success = true;

    // Synthetic GGUF header roundtrip — validates parser structures without touching files
    using namespace hq::cerberus;
    GgufHeader hdr;
    hdr.magic = GGUF_MAGIC_LE;
    hdr.version = 3;
    hdr.tensor_count = 24;
    hdr.metadata_kv_count = 12;

    if (!hdr.isValid()) {
        result.success = false;
        result.error_message = "Synthetic GGUF header invalid";
        return result;
    }

    // Build a synthetic tensor info for Athenea model topology
    GgufTensorInfo info;
    info.name = "token_embd.weight";
    info.shape = {151936, 4096};
    info.dtype = GgmlType::Q4_K;
    info.offset_in_file = 128;
    info.size_bytes = 151936ULL * 4096ULL / 2; // Q4_K ~ half of raw

    std::map<std::string, std::string> row;
    row["model_name"] = query.parameters.count("model") ? query.parameters.at("model") : "athenea";
    row["gguf_valid"] = "true";
    row["gguf_version"] = std::to_string(hdr.version);
    row["tensor_count"] = std::to_string(hdr.tensor_count);
    row["metadata_kv_count"] = std::to_string(hdr.metadata_kv_count);
    row["sample_tensor"] = info.name;
    row["sample_shape"] = std::to_string(info.shape[0]) + "x" + std::to_string(info.shape[1]);
    row["sample_dtype"] = ggml_type_name(info.dtype);
    row["sample_quantized"] = info.is_quantized() ? "true" : "false";
    result.rows.push_back(row);
    result.row_count = 1;
    return result;
}

QueryResult CerberusExtension::handle_telemetry_(const Query& /*query*/) {
    QueryResult result;
    result.success = true;

    // Package STATUS + GLOW rows into a telemetry payload for MedusaServ/BertieBot
    std::map<std::string, std::string> row;
    row["extension"] = "Cerberus.InferenceEngine";
    row["version"] = "1.0.0";
    row["healthy"] = isHealthy() ? "true" : "false";
    row["model_type"] = "inference";

    if (runtime_ && runtime_->gateway) {
        row["gateway_sessions"] = std::to_string(runtime_->gateway->sessionCount());
    }
    if (runtime_ && runtime_->runtime) {
        auto* glow = runtime_->runtime->glow_engine();
        if (glow) {
            auto stats = glow->stats();
            row["glow_bonds"] = std::to_string(stats.active_bond_count);
            row["glow_paths"] = std::to_string(stats.paths_learned);
            row["glow_reinforcements"] = std::to_string(stats.reinforcements_applied);
        }
    }

    result.rows.push_back(row);
    result.row_count = 1;
    return result;
}

} // namespace hq::cerberus::psiforcedb

// ==========================================================================
// C factory — returns std::unique_ptr for modern C++ consumers.
// PsiForceDB's dlopen/GetProcAddress path will use a thin wrapper when linked.
// ==========================================================================

extern "C" std::unique_ptr<hq::cerberus::psiforcedb::MultiModelExtension> cerberus_create_extension() {
    return std::make_unique<hq::cerberus::psiforcedb::CerberusExtension>();
}
