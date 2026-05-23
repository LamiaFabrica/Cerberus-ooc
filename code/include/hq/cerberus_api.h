#pragma once
/// @file cerberus_api.h
// Copyright (c) 2026 D Hargreaves (AKA Roylepython). LamiaFabrica. All rights reserved.
/// @brief C API for the Cerberus NPU shared library (libcerberus_npu.so).
///
/// Exposes NPU tensor staging, multi-device telemetry, ONNX Runtime inference,
/// and load-balanced device selection as a pure C interface.

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ===========================================================================
// Status codes
// ===========================================================================

typedef enum {
    CERBERUS_OK = 0,
    CERBERUS_ERROR = -1,
    CERBERUS_NOT_INITIALIZED = -2,
    CERBERUS_INVALID_HANDLE = -3,
    CERBERUS_INVALID_ARGUMENT = -4,
    CERBERUS_DEVICE_NOT_FOUND = -5,
    CERBERUS_OUT_OF_MEMORY = -6,
    CERBERUS_SESSION_LIMIT_REACHED = -7,
    CERBERUS_THREAD_ERROR = -8,
    CERBERUS_TIMEOUT = -9,
    CERBERUS_ALREADY_SHUTDOWN = -10,
    CERBERUS_INFERENCE_FAILED = -11,
} cerberus_status_t;

// ===========================================================================
// Device types
// ===========================================================================

typedef enum {
    CERBERUS_DEVICE_CPU = 0,
    CERBERUS_DEVICE_GPU = 1,
    CERBERUS_DEVICE_NPU = 2,
    CERBERUS_DEVICE_ANY = 3,
} cerberus_device_type_t;

// ===========================================================================
// Device info
// ===========================================================================

#define CERBERUS_MAX_DEVICE_NAME 64

typedef struct {
    int    index;               ///< Device index
    int    device_type;          ///< CERBERUS_DEVICE_GPU or CERBERUS_DEVICE_NPU
    char   name[CERBERUS_MAX_DEVICE_NAME]; ///< Device name string
    int    available;            ///< 1 if device is usable, 0 otherwise
    float  utilization_percent;  ///< [0, 100] GPU/NPU utilization
    float  temperature_celsius;  ///< Die temperature
} cerberus_device_info_t;
/// Callers MUST zero-initialize: cerberus_device_info_t info = {0};

// ===========================================================================
// Session configuration
// ===========================================================================

typedef struct {
    const char* model_path;      ///< Path to .onnx model file
    int32_t     width;           ///< Output image width
    int32_t     height;          ///< Output image height
    int32_t     num_steps;       ///< Number of denoising steps
    float       guidance_scale;  ///< CFG guidance scale
    int32_t     preferred_device;///< CERBERUS_DEVICE_GPU or CERBERUS_DEVICE_NPU
    int32_t     num_threads;     ///< ORT session thread count
    int32_t     preferred_backend; /**< 0=auto, 1=native(Cerberus), 2=openvino,
                                        3=cuda, 4=onnx-cpu (legacy pipeline) */
} cerberus_session_config_t;
/// Callers MUST zero-initialize: cerberus_session_config_t cfg = {0};
/// then set required fields.

// ===========================================================================
// Opaque handle
// ===========================================================================

typedef void* cerberus_handle_t;

// ===========================================================================
// Callback type for async operations
// ===========================================================================

typedef void (*cerberus_callback_t)(
    cerberus_status_t status,
    const float* output,
    size_t output_size,
    void* user_data);

// ===========================================================================
// Lifecycle
// ===========================================================================

cerberus_status_t cerberus_init(void);

cerberus_status_t cerberus_shutdown(void);

// ===========================================================================
// Version / error
// ===========================================================================

const char* cerberus_get_version(void);

const char* cerberus_get_last_error(void);

// ===========================================================================
// Device discovery
// ===========================================================================

cerberus_status_t cerberus_get_device_count(int* count);

cerberus_status_t cerberus_get_device_info(int index,
                                           cerberus_device_info_t* info);

cerberus_status_t cerberus_get_utilization(
    cerberus_device_type_t device,
    float* utilization_percent);

cerberus_status_t cerberus_get_load_balance_hint(
    cerberus_device_type_t* recommended_device);

// ===========================================================================
// Session management
// ===========================================================================

cerberus_status_t cerberus_create_session(
    const cerberus_session_config_t* config,
    cerberus_handle_t* session);

cerberus_status_t cerberus_destroy_session(cerberus_handle_t session);

// ===========================================================================
// Inference
// ===========================================================================

cerberus_status_t cerberus_run(cerberus_handle_t session,
                               const float* input,
                               size_t input_size,
                               float** output,
                               size_t* output_size);

cerberus_status_t cerberus_run_async(cerberus_handle_t session,
                                     const float* input,
                                     size_t input_size,
                                     cerberus_callback_t callback,
                                     void* user_data);

// ===========================================================================
// Cerberus-native graph execution (engine loop-in)
// ===========================================================================

/// @brief Opaque handle to a Cerberus-native computation graph.
typedef void* cerberus_graph_handle_t;

/// @brief Create an empty Cerberus graph.
/// @param graph  Output handle (must be destroyed with cerberus_graph_destroy).
cerberus_status_t cerberus_graph_create(cerberus_graph_handle_t* graph);

/// @brief Destroy a Cerberus graph and free all associated resources.
cerberus_status_t cerberus_graph_destroy(cerberus_graph_handle_t graph);

/// @brief Add a MatMul node to the graph.
/// @param graph   Graph handle.
/// @param a_idx   Input tensor index A (or -1 for external input).
/// @param b_idx   Input tensor index B (or -1 for external input).
/// @param out_idx Output tensor index (must be unique).
/// @param m,n,k   GEMM dimensions (C = A[m×k] × B[k×n]).
cerberus_status_t cerberus_graph_add_matmul(cerberus_graph_handle_t graph,
                                              int a_idx, int b_idx, int out_idx,
                                              int m, int n, int k);

/// @brief Add an element-wise Add node to the graph.
cerberus_status_t cerberus_graph_add_add(cerberus_graph_handle_t graph,
                                         int a_idx, int b_idx, int out_idx,
                                         int count);

/// @brief Add an element-wise Mul node to the graph.
cerberus_status_t cerberus_graph_add_mul(cerberus_graph_handle_t graph,
                                         int a_idx, int b_idx, int out_idx,
                                         int count);

/// @brief Declare an external input to the graph.
cerberus_status_t cerberus_graph_add_input(cerberus_graph_handle_t graph,
                                           int tensor_idx,
                                           int count);

/// @brief Declare an external output of the graph.
cerberus_status_t cerberus_graph_add_output(cerberus_graph_handle_t graph,
                                            int tensor_idx,
                                            int count);

/// @brief Compile and execute the Cerberus graph natively.
///
/// This is the production loop-in for the Cerberus compiler+runtime.
/// It goes through: CerberusGraph → DecisionEngine → ExecutionCoordinator → Backend.
///
/// @param graph       Compiled graph handle.
/// @param inputs      Array of input buffers (length = number of graph inputs).
/// @param input_sizes Array of buffer sizes in floats.
/// @param outputs     Array of output buffers (caller-allocated, length = number of graph outputs).
/// @param output_sizes Array of buffer sizes in floats.
cerberus_status_t cerberus_run_graph(cerberus_graph_handle_t graph,
                                     const float** inputs, const size_t* input_sizes,
                                     float** outputs, size_t* output_sizes);

// ===========================================================================
// Pinned memory allocation
// ===========================================================================

cerberus_status_t cerberus_alloc_pinned(size_t bytes, void** ptr);

cerberus_status_t cerberus_free_pinned(void* ptr);

#ifdef __cplusplus
}
#endif
