// Minimal stub for ONNX Runtime C++ API — used when ONNX Runtime is not installed.
// This allows compilation to proceed for CI/build verification.
// DO NOT use in production; link against real ONNX Runtime instead.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>

// C API constants (global namespace, as in real ONNX Runtime)
enum OrtLoggingLevel {
    ORT_LOGGING_LEVEL_VERBOSE = 0,
    ORT_LOGGING_LEVEL_INFO = 1,
    ORT_LOGGING_LEVEL_WARNING = 2,
    ORT_LOGGING_LEVEL_ERROR = 3,
    ORT_LOGGING_LEVEL_FATAL = 4
};

enum OrtMemType {
    OrtMemTypeDefault = 0
};

enum OrtAllocatorType {
    OrtArenaAllocator = 0
};

enum GraphOptimizationLevel {
    ORT_DISABLE_ALL = 0,
    ORT_ENABLE_BASIC = 1,
    ORT_ENABLE_EXTENDED = 2,
    ORT_ENABLE_ALL = 99
};

using OrtEnv = void*;
using OrtSessionOptions = void*;
using OrtSession = void*;
using OrtStatus = void*;
using OrtMemoryInfo = void*;
using OrtValue = void*;
using OrtAllocator = void*;

namespace Ort {

struct Exception : std::exception {
    explicit Exception(const std::string& msg) : msg_(msg) {}
    const char* what() const noexcept override { return msg_.c_str(); }
private:
    std::string msg_;
};

inline void ThrowOnError(int) {}

struct Env {
    Env() = default;
    explicit Env(OrtLoggingLevel, const char*) {}
};

struct MemoryInfo {
    static MemoryInfo CreateCpu(OrtAllocatorType, OrtMemType) { return {}; }
};

struct TensorTypeAndShapeInfo {
    std::vector<int64_t> GetShape() const { return {}; }
    std::size_t GetElementCount() const { return 0; }
};

struct Value {
    template<typename T>
    static Value CreateTensor(const MemoryInfo&, T*, size_t, const int64_t*, size_t) { return {}; }

    template<typename T>
    T* GetTensorMutableData() { return nullptr; }

    template<typename T>
    const T* GetTensorData() const { return nullptr; }

    TensorTypeAndShapeInfo GetTensorTypeAndShapeInfo() const { return {}; }
    bool IsTensor() const { return true; }
};

struct RunOptions {
    RunOptions() = default;
    explicit RunOptions(void*) {}
};

struct SessionOptions {
    void SetGraphOptimizationLevel(GraphOptimizationLevel) {}
    void SetIntraOpNumThreads(int) {}
    operator OrtSessionOptions*() { return nullptr; }
};

struct Session {
    Session(const Env&, const char*, const SessionOptions&) {}
    Session(const Env&, const std::vector<uint8_t>&, const SessionOptions&) {}
    std::vector<Value> Run(const RunOptions&, const char* const*, const Value*, size_t,
                           const char* const*, size_t) { return {}; }
};

struct AllocatorWithDefaultOptions {
    const char* GetInputNameAllocated(size_t, void*) const { return nullptr; }
    const char* GetOutputNameAllocated(size_t, void*) const { return nullptr; }
};

struct TypeInfo {};

struct Allocator {};

} // namespace Ort

inline int OrtSessionOptionsAppendExecutionProvider_ROCM(OrtSessionOptions*, int) { return 0; }
inline int OrtSessionOptionsAppendExecutionProvider(OrtSessionOptions*, const char*, const void*, size_t) { return 0; }
