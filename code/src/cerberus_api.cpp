/// @file cerberus_api.cpp
/// @copyright Copyright © 2026, all rights reserved | David Hargreaves aka Roylepython trading as The Medusa Initiative & Lamia Fabrica
/// Support: https://www.patreon.com/TheMedusaInitiative — £25/month removes ads from all software at 200 subscribers
/// @brief C API implementation for libcerberus_npu.so shared library.
///
/// Wraps C++ components (NpuDmaPipeline, HailoMonitor, GPUMonitor, Pipeline)
/// behind a pure C ABI. Provides device discovery, load-balanced session
/// creation, synchronous/async inference, and pinned memory allocation.
///
/// @version 1.0.0

#include "hq/cerberus_api.h"

#include "hq/npu_pipeline.hpp"
#include "hq/hailo_monitor.hpp"
#include "hq/gpu_monitor.hpp"
#include "hq/pipeline.hpp"
#include "hq/cerberus_runtime.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdlib>
#include <cstring>
#ifdef _WIN32
#  include <malloc.h>
#endif
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#ifdef UM790_HAS_HIP
#include <hip/hip_runtime_api.h>
#include <format>
#endif

extern "C" std::size_t hq_safe_write(int fd, const char* data, std::size_t len);

namespace {
    inline void hq_print(std::string msg) {
        hq_safe_write(1, msg.data(), msg.size());
    }
    inline void hq_println(std::string msg) {
        hq_safe_write(1, msg.data(), msg.size());
        hq_safe_write(1, "\n", 1);
    }
}

// =============================================================================
// Internal state
// =============================================================================

namespace {

struct CerberusHandle {
    uint64_t id;
    explicit CerberusHandle(uint64_t i) : id{i} {}
};

struct SessionState {
    cerberus_handle_t    handle{nullptr};
    cerberus_session_config_t config{};
    std::unique_ptr<hq::Pipeline> pipeline;
    std::unique_ptr<hq::npu::NpuDmaPipeline> npu_pipeline;
    std::unique_ptr<hq::cerberus::CerberusRuntime> cerberus_rt;
    std::string model_path_owned;
    std::vector<float> output_buffer;
    bool active{false};
};

struct ThreadPool {
    struct Task {
        std::function<void()> work;
    };

    std::vector<std::thread> workers;
    std::queue<Task>          tasks;
    std::mutex                queue_mutex;
    std::condition_variable   cv;
    std::atomic<bool>         stop{false};

    explicit ThreadPool(std::size_t num_threads) {
        num_threads = std::min(num_threads, std::size_t{4});
        for (std::size_t i = 0; i < num_threads; ++i) {
            workers.emplace_back([this]() {
                while (true) {
                    Task task;
                    {
                        std::unique_lock lock{queue_mutex};
                        cv.wait(lock, [this]() {
                            return stop.load(std::memory_order_acquire)
                                   || !tasks.empty();
                        });
                        if (stop.load(std::memory_order_acquire) && tasks.empty())
                            return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task.work();
                }
            });
        }
    }

    ~ThreadPool() {
        stop.store(true, std::memory_order_release);
        cv.notify_all();
        for (auto& w : workers) {
            if (w.joinable()) w.join();
        }
    }

    void enqueue(std::function<void()> fn) {
        {
            std::lock_guard lock{queue_mutex};
            tasks.push({std::move(fn)});
        }
        cv.notify_one();
    }
};

struct GlobalState {
    bool initialized{false};
    std::mutex mutex;
    std::string last_error;
    std::vector<SessionState> sessions;
    std::unique_ptr<ThreadPool> thread_pool;
    std::unique_ptr<hq::HailoMonitor> hailo_monitor;
    std::unique_ptr<hq::GPUMonitor> gpu_monitor;
    std::unique_ptr<hq::npu::NpuDmaPipeline> shared_npu;
    bool hailo_available{false};
    bool gpu_available{false};
    int  max_sessions{16};
};

GlobalState& g_state() {
    static GlobalState s;
    return s;
}

void set_error(const char* msg) {
    auto& gs = g_state();
    std::lock_guard lock{gs.mutex};
    gs.last_error = msg ? msg : "";
}

void set_error_fmt(const std::string& msg) {
    auto& gs = g_state();
    std::lock_guard lock{gs.mutex};
    gs.last_error = msg;
}

SessionState* find_session(cerberus_handle_t h) {
    if (!h) return nullptr;
    auto& gs = g_state();
    for (auto& s : gs.sessions) {
        if (s.active && s.handle == h) return &s;
    }
    return nullptr;
}

bool probe_hailo() {
    try {
        hq::HailoMonitor mon;
        auto open_result = mon.open("");
        return open_result.has_value();
    } catch (...) {
        return false;
    }
}

bool probe_gpu() {
    try {
        hq::GPUMonitor mon{0};
        auto init = mon.initialize();
        return init.has_value();
    } catch (...) {
        return false;
    }
}

constexpr const char* kVersion = "1.0.0";
constexpr int kMaxSessions = 16;
constexpr int kMaxThreads = 4;

float clamp_util(float v) {
    return std::clamp(v, 0.0f, 100.0f);
}

cerberus_status_t to_status(hq::PipelineError e) {
    switch (e) {
        case hq::PipelineError::Ok:                    return CERBERUS_OK;
        case hq::PipelineError::InvalidRequest:        return CERBERUS_INVALID_ARGUMENT;
        case hq::PipelineError::HailoNotAvailable:     return CERBERUS_DEVICE_NOT_FOUND;
        case hq::PipelineError::HailoTimeout:          return CERBERUS_TIMEOUT;
        case hq::PipelineError::GPUOutOfMemory:        return CERBERUS_OUT_OF_MEMORY;
        case hq::PipelineError::ONNXSessionLoadFailed: return CERBERUS_INFERENCE_FAILED;
        case hq::PipelineError::ONNXRunFailed:         return CERBERUS_INFERENCE_FAILED;
        case hq::PipelineError::ShutdownInProgress:    return CERBERUS_ALREADY_SHUTDOWN;
        default: return CERBERUS_ERROR;
    }
}

hq::PipelineConfig build_pipeline_config(const cerberus_session_config_t& cfg,
                                         const std::string& model_path) {
    hq::PipelineConfig pcfg;
    pcfg.enable_watchdog = false;

    if (!model_path.empty()) {
        pcfg.text_encoder_onnx = model_path + "/text_encoder.onnx";
        pcfg.unet_onnx         = model_path + "/unet.onnx";
        pcfg.vae_decoder_onnx  = model_path + "/vae_decoder.onnx";
    }

    if (cfg.width > 0)  { /* dimensions pass through request */ }
    if (cfg.height > 0) { /* dimensions pass through request */ }

    return pcfg;
}

} // anonymous namespace

// =============================================================================
// Exported API — extern "C"
// =============================================================================

extern "C" {

cerberus_status_t cerberus_init(void) {
    auto& gs = g_state();
    std::lock_guard lock{gs.mutex};

    if (gs.initialized) {
        set_error("cerberus_init: already initialized");
        return CERBERUS_OK;
    }

    try {
        gs.thread_pool = std::make_unique<ThreadPool>(kMaxThreads);
    } catch (const std::exception& e) {
        set_error_fmt(std::string("cerberus_init: thread pool creation failed: ") + e.what());
        return CERBERUS_THREAD_ERROR;
    }

    gs.hailo_available = probe_hailo();
    gs.gpu_available   = probe_gpu();

    if (gs.hailo_available) {
        try {
            gs.hailo_monitor = std::make_unique<hq::HailoMonitor>();
            auto open_res = gs.hailo_monitor->open("");
            if (!open_res.has_value()) {
                gs.hailo_available = false;
                gs.hailo_monitor.reset();
            }
        } catch (...) {
            gs.hailo_available = false;
            gs.hailo_monitor.reset();
        }
    }

    if (gs.gpu_available) {
        try {
            gs.gpu_monitor = std::make_unique<hq::GPUMonitor>(0);
            auto init_res = gs.gpu_monitor->initialize();
            if (!init_res.has_value()) {
                gs.gpu_available = false;
                gs.gpu_monitor.reset();
            }
        } catch (...) {
            gs.gpu_available = false;
            gs.gpu_monitor.reset();
        }
    }

    try {
        hq::npu::NpuDmaPipeline::Config ncfg{
            .num_slots       = 3,
            .embedding_bytes = hq::npu::MAX_EMBEDDING_BYTES,
            .enable_gpu_staging = gs.gpu_available,
            .enable_npu_util_tracking = true,
        };
        gs.shared_npu = std::make_unique<hq::npu::NpuDmaPipeline>(ncfg);
    } catch (const std::exception& e) {
        set_error_fmt(std::string("cerberus_init: NPU pipeline creation failed: ") + e.what());
    }

    gs.sessions.clear();
    gs.sessions.reserve(kMaxSessions);
    gs.initialized = true;

    printf("[cerberus] initialized: hailo=%d gpu=%d cpu=1\n",
           gs.hailo_available ? 1 : 0,
           gs.gpu_available ? 1 : 0);

    return CERBERUS_OK;
}

cerberus_status_t cerberus_shutdown(void) {
    auto& gs = g_state();
    std::lock_guard lock{gs.mutex};

    if (!gs.initialized) {
        set_error("cerberus_shutdown: not initialized");
        return CERBERUS_ALREADY_SHUTDOWN;
    }

    for (auto& s : gs.sessions) {
        if (s.active && s.pipeline) {
            s.pipeline->shutdown();
            s.active = false;
        }
    }
    gs.sessions.clear();

    gs.thread_pool.reset();
    gs.hailo_monitor.reset();
    gs.gpu_monitor.reset();
    gs.shared_npu.reset();
    gs.initialized = false;
    gs.hailo_available = false;
    gs.gpu_available = false;

    printf("[cerberus] shutdown complete\n");
    return CERBERUS_OK;
}

const char* cerberus_get_version(void) {
    return kVersion;
}

const char* cerberus_get_last_error(void) {
    return g_state().last_error.c_str();
}

cerberus_status_t cerberus_get_device_count(int* count) {
    if (!count) {
        set_error("cerberus_get_device_count: null pointer");
        return CERBERUS_INVALID_ARGUMENT;
    }

    auto& gs = g_state();
    std::lock_guard lock{gs.mutex};

    if (!gs.initialized) {
        set_error("cerberus_get_device_count: not initialized");
        return CERBERUS_NOT_INITIALIZED;
    }

    int n = 1; // CPU always available
    if (gs.gpu_available) ++n;
    if (gs.hailo_available) ++n;

    *count = n;
    return CERBERUS_OK;
}

cerberus_status_t cerberus_get_device_info(int index,
                                           cerberus_device_info_t* info) {
    if (!info) {
        set_error("cerberus_get_device_info: null pointer");
        return CERBERUS_INVALID_ARGUMENT;
    }

    auto& gs = g_state();
    std::lock_guard lock{gs.mutex};

    if (!gs.initialized) {
        set_error("cerberus_get_device_info: not initialized");
        return CERBERUS_NOT_INITIALIZED;
    }

    memset(info, 0, sizeof(*info));

    int dev_idx = 0;
    bool found = false;

    if (index == dev_idx++) {
        info->index      = 0;
        info->device_type = CERBERUS_DEVICE_CPU;
        snprintf(info->name, CERBERUS_MAX_DEVICE_NAME, "CPU (always available)");
        info->available   = 1;
        info->utilization_percent = 0.0f;
        info->temperature_celsius = 0.0f;
        found = true;
    }

    if (!found && gs.gpu_available && index == dev_idx++) {
        info->index       = 1;
        info->device_type = CERBERUS_DEVICE_GPU;
        snprintf(info->name, CERBERUS_MAX_DEVICE_NAME, "AMD GPU (ROCm)");
        info->available   = 1;
        info->utilization_percent = 0.0f;
        info->temperature_celsius = 0.0f;

        if (gs.gpu_monitor) {
            try {
                auto telem = gs.gpu_monitor->query_all();
                if (telem) {
                    info->utilization_percent = telem->utilization_percent;
                    info->temperature_celsius = telem->temperature_celsius;
                }
            } catch (...) {}
        }
        found = true;
    }

    if (!found && gs.hailo_available && index == dev_idx++) {
        info->index       = 2;
        info->device_type = CERBERUS_DEVICE_NPU;
        snprintf(info->name, CERBERUS_MAX_DEVICE_NAME, "Hailo-8L NPU");
        info->available   = 1;
        info->utilization_percent = 0.0f;
        info->temperature_celsius = 0.0f;

        if (gs.hailo_monitor && gs.hailo_monitor->is_open()) {
            try {
                auto stats = gs.hailo_monitor->sample();
                if (stats) {
                    info->utilization_percent = stats->nn_core_utilization;
                    info->temperature_celsius = stats->temperature_celsius;
                }
            } catch (...) {}
        }
        found = true;
    }

    if (!found) {
        set_error("cerberus_get_device_info: index out of range");
        return CERBERUS_INVALID_ARGUMENT;
    }

    return CERBERUS_OK;
}

cerberus_status_t cerberus_get_utilization(
    cerberus_device_type_t device,
    float* utilization_percent) {
    if (!utilization_percent) {
        set_error("cerberus_get_utilization: null pointer");
        return CERBERUS_INVALID_ARGUMENT;
    }

    auto& gs = g_state();
    std::lock_guard lock{gs.mutex};

    if (!gs.initialized) {
        set_error("cerberus_get_utilization: not initialized");
        *utilization_percent = 0.0f;
        return CERBERUS_NOT_INITIALIZED;
    }

    *utilization_percent = -1.0f;  // Default sentinel: unknown

    switch (device) {
        case CERBERUS_DEVICE_CPU: {
#if defined(__linux__)
            double loadavg[1];
            if (getloadavg(loadavg, 1) == 1) {
                float load_pct = static_cast<float>(loadavg[0]) * 10.0f;
                *utilization_percent = clamp_util(load_pct);
                return CERBERUS_OK;
            } else {
                hq_println(std::format("[cerberus] WARNING CPU utilization unavailable on this platform."));
                return CERBERUS_NOT_INITIALIZED;
            }
#else
            hq_println(std::format("[cerberus] WARNING CPU utilization unavailable on non-Linux platforms."));
            return CERBERUS_NOT_INITIALIZED;
#endif
        }

        case CERBERUS_DEVICE_GPU:
            if (gs.gpu_monitor) {
                try {
                    auto util = gs.gpu_monitor->query_utilization();
                    if (util) {
                        *utilization_percent = clamp_util(*util);
                        return CERBERUS_OK;
                    } else {
                        hq_println(std::format("[cerberus] WARNING GPU utilization query failed: {}",
                                   util.error().message));
                        return CERBERUS_DEVICE_NOT_FOUND;
                    }
                } catch (...) {
                    return CERBERUS_DEVICE_NOT_FOUND;
                }
            }
            return CERBERUS_NOT_INITIALIZED;

        case CERBERUS_DEVICE_NPU:
            if (gs.shared_npu) {
                float npu_util = gs.shared_npu->last_npu_utilization();
                if (npu_util >= 0.0f) {
                    *utilization_percent = clamp_util(npu_util);
                    return CERBERUS_OK;
                }
            }
            return CERBERUS_NOT_INITIALIZED;

        case CERBERUS_DEVICE_ANY:
        default: {
            float npu_util = -1.0f;
            float gpu_util = -1.0f;
            if (gs.shared_npu) {
                npu_util = gs.shared_npu->last_npu_utilization();
            }
            if (gs.gpu_monitor) {
                try {
                    auto util = gs.gpu_monitor->query_utilization();
                    if (util) gpu_util = clamp_util(*util);
                } catch (...) {}
            }
            // Return the best available, or error if none
            if (gpu_util >= 0.0f) {
                *utilization_percent = gpu_util;
                return CERBERUS_OK;
            }
            if (npu_util >= 0.0f) {
                *utilization_percent = npu_util;
                return CERBERUS_OK;
            }
            return CERBERUS_NOT_INITIALIZED;
        }
    }
}

cerberus_status_t cerberus_get_load_balance_hint(
    cerberus_device_type_t* recommended_device) {
    if (!recommended_device) {
        set_error("cerberus_get_load_balance_hint: null pointer");
        return CERBERUS_INVALID_ARGUMENT;
    }

    auto& gs = g_state();

    if (!gs.initialized) {
        set_error("cerberus_get_load_balance_hint: not initialized");
        *recommended_device = CERBERUS_DEVICE_CPU;
        return CERBERUS_NOT_INITIALIZED;
    }

    float npu_util = 0.0f;
    float gpu_util = 0.0f;
    float cpu_util = 0.0f;
#if defined(__linux__)
    {
        double loadavg[1];
        if (getloadavg(loadavg, 1) == 1) {
            float load_pct = static_cast<float>(loadavg[0]) * 10.0f;
            cpu_util = clamp_util(load_pct);
        }
    }
#else
    cpu_util = 0.0f;
    hq_println(std::format("[cerberus] WARNING CPU utilization unavailable on this platform — load balance hint may be inaccurate."));
#endif

    if (gs.shared_npu)
        npu_util = clamp_util(gs.shared_npu->last_npu_utilization());

    {
        std::lock_guard lock{gs.mutex};
        if (gs.gpu_monitor) {
            try {
                auto util = gs.gpu_monitor->query_utilization();
                if (util) gpu_util = clamp_util(*util);
            } catch (...) {}
        }
    }

    // Weighted effective load (lower is better)
    // Weight NPU 2x, GPU 1.5x, CPU 1x for inference workloads
    const float npu_weight = 2.0f;
    const float gpu_weight = 1.5f;
    const float cpu_weight = 1.0f;

    struct Candidate {
        cerberus_device_type_t type;
        float weighted_load;
        bool available;
    };

    Candidate candidates[3] = {
        { CERBERUS_DEVICE_NPU, npu_util * npu_weight, gs.hailo_available },
        { CERBERUS_DEVICE_GPU, gpu_util * gpu_weight, gs.gpu_available },
        { CERBERUS_DEVICE_CPU, cpu_util * cpu_weight, true },
    };

    Candidate* best = nullptr;
    for (auto& c : candidates) {
        if (!c.available) continue;
        if (!best || c.weighted_load < best->weighted_load) {
            best = &c;
        }
    }

    *recommended_device = best ? best->type : CERBERUS_DEVICE_CPU;
    return CERBERUS_OK;
}

cerberus_status_t cerberus_create_session(
    const cerberus_session_config_t* config,
    cerberus_handle_t* session) {
    if (!config || !session) {
        set_error("cerberus_create_session: null pointer");
        return CERBERUS_INVALID_ARGUMENT;
    }

    auto& gs = g_state();
    std::lock_guard lock{gs.mutex};

    if (!gs.initialized) {
        set_error("cerberus_create_session: not initialized");
        return CERBERUS_NOT_INITIALIZED;
    }

    int active_count = 0;
    for (auto& s : gs.sessions) {
        if (s.active) ++active_count;
    }
    if (active_count >= gs.max_sessions) {
        set_error("cerberus_create_session: session limit reached");
        return CERBERUS_SESSION_LIMIT_REACHED;
    }

    try {
        SessionState ss;
        ss.config = *config;
        if (config->model_path) {
            ss.model_path_owned = config->model_path;
        }

        auto pcfg = build_pipeline_config(ss.config, ss.model_path_owned);
        ss.pipeline = std::make_unique<hq::Pipeline>(pcfg);

        hq::npu::NpuDmaPipeline::Config ncfg{
            .num_slots       = 3,
            .embedding_bytes = hq::npu::MAX_EMBEDDING_BYTES,
            .enable_gpu_staging = gs.gpu_available,
            .enable_npu_util_tracking = true,
        };
        ss.npu_pipeline = std::make_unique<hq::npu::NpuDmaPipeline>(ncfg);

        static std::atomic<uint64_t> handle_counter{1};
        ss.handle = reinterpret_cast<cerberus_handle_t>(
            new CerberusHandle{handle_counter.fetch_add(1, std::memory_order_relaxed)});
        ss.active = true;

        gs.sessions.push_back(std::move(ss));
        *session = gs.sessions.back().handle;

        printf("[cerberus] session created: handle=%p\n", *session);
        return CERBERUS_OK;

    } catch (const std::exception& e) {
        set_error_fmt(std::string("cerberus_create_session: ") + e.what());
        return CERBERUS_ERROR;
    }
}

cerberus_status_t cerberus_destroy_session(cerberus_handle_t session) {
    if (!session) {
        set_error("cerberus_destroy_session: invalid handle");
        return CERBERUS_INVALID_HANDLE;
    }

    auto& gs = g_state();
    std::lock_guard lock{gs.mutex};

    for (auto it = gs.sessions.begin(); it != gs.sessions.end(); ++it) {
        if (it->active && it->handle == session) {
            if (it->pipeline) {
                it->pipeline->shutdown();
            }
            it->npu_pipeline.reset();
            it->active = false;
            gs.sessions.erase(it);
            printf("[cerberus] session destroyed: handle=%p\n", session);
            return CERBERUS_OK;
        }
    }

    set_error("cerberus_destroy_session: session not found");
    return CERBERUS_INVALID_HANDLE;
}

cerberus_status_t cerberus_run(cerberus_handle_t session,
                                const float* input,
                                size_t input_size,
                                float** output,
                                size_t* output_size) {
    if (!session || !output || !output_size) {
        set_error("cerberus_run: invalid arguments");
        return CERBERUS_INVALID_ARGUMENT;
    }

    auto& gs = g_state();
    std::lock_guard lock{gs.mutex};

    auto* ss = find_session(session);
    if (!ss) {
        set_error("cerberus_run: session not found");
        return CERBERUS_INVALID_HANDLE;
    }

    try {
        std::string prompt;
        if (input != nullptr && input_size > 0) {
            const char* prompt_cstr = reinterpret_cast<const char*>(input);
            prompt = std::string(prompt_cstr, strnlen(prompt_cstr,
                std::min(input_size, size_t{1024})));
        }
        if (prompt.empty()) {
            prompt = "inference from C API";
        }

        hq::GenerationRequest req{
            .prompt         = prompt,
            .width          = static_cast<uint32_t>(
                ss->config.width > 0 ? ss->config.width : 512),
            .height         = static_cast<uint32_t>(
                ss->config.height > 0 ? ss->config.height : 512),
            .num_steps      = static_cast<uint32_t>(
                ss->config.num_steps > 0 ? ss->config.num_steps : 20),
            .guidance_scale = ss->config.guidance_scale > 0.0f
                                  ? ss->config.guidance_scale : 7.5f,
            .seed           = -1,
        };

        auto result = ss->pipeline->generate(req);
        if (!result) {
            set_error_fmt(std::string("cerberus_run: inference failed: ")
                          + hq::to_string(result.error()));
            return to_status(result.error());
        }

        size_t pixel_count = result->pixels.size();
        size_t rgb_count = pixel_count / 4 * 3;
        ss->output_buffer.resize(rgb_count);
        for (size_t i = 0, j = 0; i < pixel_count; i += 4, j += 3) {
            ss->output_buffer[j]   = static_cast<float>(result->pixels[i])   / 255.0f;
            ss->output_buffer[j+1] = static_cast<float>(result->pixels[i+1]) / 255.0f;
            ss->output_buffer[j+2] = static_cast<float>(result->pixels[i+2]) / 255.0f;
        }
        *output = ss->output_buffer.data();
        *output_size = ss->output_buffer.size();

        return CERBERUS_OK;

    } catch (const std::exception& e) {
        set_error_fmt(std::string("cerberus_run: exception: ") + e.what());
        return CERBERUS_ERROR;
    }
}

cerberus_status_t cerberus_run_async(cerberus_handle_t session,
                                     const float* input,
                                     size_t input_size,
                                     cerberus_callback_t callback,
                                     void* user_data) {
    if (!session || !callback) {
        set_error("cerberus_run_async: invalid arguments");
        return CERBERUS_INVALID_ARGUMENT;
    }

    auto& gs = g_state();

    {
        std::lock_guard lock{gs.mutex};
        auto* ss = find_session(session);
        if (!ss) {
            set_error("cerberus_run_async: session not found");
            return CERBERUS_INVALID_HANDLE;
        }
    }

    if (!gs.thread_pool) {
        set_error("cerberus_run_async: thread pool not available");
        return CERBERUS_THREAD_ERROR;
    }

    gs.thread_pool->enqueue([session, input, input_size, callback, user_data]() {
        float* output = nullptr;
        size_t output_sz = 0;
        auto status = cerberus_run(session, input, input_size, &output, &output_sz);
        callback(status, output, output_sz, user_data);
    });

    return CERBERUS_OK;
}

cerberus_status_t cerberus_alloc_pinned(size_t bytes, void** ptr) {
    if (!ptr || bytes == 0) {
        set_error("cerberus_alloc_pinned: invalid arguments");
        return CERBERUS_INVALID_ARGUMENT;
    }

#if defined(UM790_HAS_HIP)
    hipError_t err = hipHostMalloc(ptr, bytes, hipHostMallocPortable);
    if (err != hipSuccess) {
        printf("[cerberus] hipHostMalloc failed (code=%d), falling back to "
               "aligned_alloc\n", static_cast<int>(err));
#ifdef _WIN32
        *ptr = _aligned_malloc(((bytes + 255) / 256) * 256, 256);
#else
        *ptr = ::aligned_alloc(256, ((bytes + 255) / 256) * 256);
#endif
        if (!*ptr) {
            set_error("cerberus_alloc_pinned: allocation failed");
            return CERBERUS_OUT_OF_MEMORY;
        }
        return CERBERUS_OK;
    }
#else
#  ifdef _WIN32
    *ptr = _aligned_malloc(((bytes + 255) / 256) * 256, 256);
#  else
    *ptr = ::aligned_alloc(256, ((bytes + 255) / 256) * 256);
#  endif
    if (!*ptr) {
        set_error("cerberus_alloc_pinned: allocation failed");
        return CERBERUS_OUT_OF_MEMORY;
    }
#endif

    return CERBERUS_OK;
}

cerberus_status_t cerberus_free_pinned(void* ptr) {
    if (!ptr) {
        set_error("cerberus_free_pinned: null pointer");
        return CERBERUS_INVALID_ARGUMENT;
    }

#if defined(UM790_HAS_HIP)
    // Try HIP-free first; if not HIP memory, fall-back to aligned free.
    hipError_t err = hipHostFree(ptr);
    if (err != hipSuccess) {
#  ifdef _WIN32
        _aligned_free(ptr);
#  else
        ::free(ptr);
#  endif
    }
#else
#  ifdef _WIN32
    _aligned_free(ptr);
#  else
    ::free(ptr);
#  endif
#endif

    return CERBERUS_OK;
}

// ===========================================================================
// Cerberus-native graph API (production engine loop-in)
// ===========================================================================

namespace {

struct CerberusGraphHolder {
    hq::npu::KernelGraph graph;
    std::vector<std::vector<float>> constant_buffers; ///< ownership for float_attrs
};

} // anonymous namespace

cerberus_status_t cerberus_graph_create(cerberus_graph_handle_t* graph) {
    if (!graph) {
        set_error("cerberus_graph_create: null pointer");
        return CERBERUS_INVALID_ARGUMENT;
    }
    try {
        auto* gh = new CerberusGraphHolder{};
        *graph = reinterpret_cast<cerberus_graph_handle_t>(gh);
        return CERBERUS_OK;
    } catch (const std::exception& e) {
        set_error_fmt(std::string("cerberus_graph_create: ") + e.what());
        return CERBERUS_ERROR;
    }
}

cerberus_status_t cerberus_graph_destroy(cerberus_graph_handle_t graph) {
    if (!graph) {
        set_error("cerberus_graph_destroy: null pointer");
        return CERBERUS_INVALID_ARGUMENT;
    }
    delete reinterpret_cast<CerberusGraphHolder*>(graph);
    return CERBERUS_OK;
}

cerberus_status_t cerberus_graph_add_matmul(cerberus_graph_handle_t graph,
                                              int a_idx, int b_idx, int out_idx,
                                              int m, int n, int k) {
    if (!graph) return CERBERUS_INVALID_ARGUMENT;
    auto* gh = reinterpret_cast<CerberusGraphHolder*>(graph);
    hq::npu::KernelNode node;
    node.op = hq::npu::KernelNode::Op::MatMul;
    node.inputs.push_back("tensor_" + std::to_string(a_idx));
    node.inputs.push_back("tensor_" + std::to_string(b_idx));
    node.outputs.push_back("tensor_" + std::to_string(out_idx));
    node.int_attrs = {m, n, k};
    gh->graph.nodes.push_back(std::move(node));
    return CERBERUS_OK;
}

cerberus_status_t cerberus_graph_add_add(cerberus_graph_handle_t graph,
                                           int a_idx, int b_idx, int out_idx,
                                           int count) {
    if (!graph) return CERBERUS_INVALID_ARGUMENT;
    auto* gh = reinterpret_cast<CerberusGraphHolder*>(graph);
    hq::npu::KernelNode node;
    node.op = hq::npu::KernelNode::Op::Add;
    node.inputs.push_back("tensor_" + std::to_string(a_idx));
    node.inputs.push_back("tensor_" + std::to_string(b_idx));
    node.outputs.push_back("tensor_" + std::to_string(out_idx));
    node.int_attrs = {count};
    gh->graph.nodes.push_back(std::move(node));
    return CERBERUS_OK;
}

cerberus_status_t cerberus_graph_add_mul(cerberus_graph_handle_t graph,
                                          int a_idx, int b_idx, int out_idx,
                                          int count) {
    if (!graph) return CERBERUS_INVALID_ARGUMENT;
    auto* gh = reinterpret_cast<CerberusGraphHolder*>(graph);
    hq::npu::KernelNode node;
    node.op = hq::npu::KernelNode::Op::Mul;
    node.inputs.push_back("tensor_" + std::to_string(a_idx));
    node.inputs.push_back("tensor_" + std::to_string(b_idx));
    node.outputs.push_back("tensor_" + std::to_string(out_idx));
    node.int_attrs = {count};
    gh->graph.nodes.push_back(std::move(node));
    return CERBERUS_OK;
}

cerberus_status_t cerberus_graph_add_input(cerberus_graph_handle_t graph,
                                           int tensor_idx,
                                           int count) {
    if (!graph) return CERBERUS_INVALID_ARGUMENT;
    (void)tensor_idx;
    auto* gh = reinterpret_cast<CerberusGraphHolder*>(graph);
    hq::npu::TensorDesc desc;
    desc.shape = {count};
    desc.dtype = hq::npu::TensorDesc::DataType::F32;
    gh->graph.graph_inputs.push_back(std::move(desc));
    return CERBERUS_OK;
}

cerberus_status_t cerberus_graph_add_output(cerberus_graph_handle_t graph,
                                            int tensor_idx,
                                            int count) {
    if (!graph) return CERBERUS_INVALID_ARGUMENT;
    (void)tensor_idx;
    auto* gh = reinterpret_cast<CerberusGraphHolder*>(graph);
    hq::npu::TensorDesc desc;
    desc.shape = {count};
    desc.dtype = hq::npu::TensorDesc::DataType::F32;
    gh->graph.graph_outputs.push_back(std::move(desc));
    return CERBERUS_OK;
}

cerberus_status_t cerberus_run_graph(cerberus_graph_handle_t graph,
                                     const float** inputs, const size_t* input_sizes,
                                     float** outputs, size_t* output_sizes) {
    (void)input_sizes; (void)output_sizes;
    if (!graph) {
        set_error("cerberus_run_graph: null graph");
        return CERBERUS_INVALID_ARGUMENT;
    }

    auto* gh = reinterpret_cast<CerberusGraphHolder*>(graph);
    auto& kg = gh->graph;

    if (kg.nodes.empty()) {
        set_error("cerberus_run_graph: graph has no nodes");
        return CERBERUS_INVALID_ARGUMENT;
    }

    try {
        hq::cerberus::CerberusRuntime rt;

        std::vector<const std::byte*> in_ptrs;
        std::vector<std::byte*> out_ptrs;

        size_t num_inputs  = kg.graph_inputs.size();
        size_t num_outputs = kg.graph_outputs.size();

        if (inputs) {
            for (size_t i = 0; i < num_inputs; ++i)
                in_ptrs.push_back(reinterpret_cast<const std::byte*>(inputs[i]));
        }
        if (outputs) {
            for (size_t i = 0; i < num_outputs; ++i)
                out_ptrs.push_back(reinterpret_cast<std::byte*>(outputs[i]));
        }

        auto result = rt.run_graph(kg, out_ptrs, in_ptrs);
        if (!result) {
            set_error_fmt(std::string("cerberus_run_graph: ") + hq::to_string(result.error()));
            return CERBERUS_INFERENCE_FAILED;
        }

        return CERBERUS_OK;

    } catch (const std::exception& e) {
        set_error_fmt(std::string("cerberus_run_graph: exception: ") + e.what());
        return CERBERUS_ERROR;
    }
}

} // extern "C"
