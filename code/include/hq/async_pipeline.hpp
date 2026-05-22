#pragma once
/// @file async_pipeline.hpp
/// @copyright Copyright (c) 2026 D Hargreaves (AKA Roylepython). LamiaFabrica. All rights reserved.
///
/// TRADE SECRET — Proprietary and Confidential.
/// This file contains trade secrets of LamiaFabrica / D Hargreaves.
/// Unauthorized use, reproduction, or disclosure is strictly prohibited.
/// See LICENSE for terms.
///
/// C++26 coroutine-based async generation pipeline for UM790 Pro.
///
/// Provides:
///   1. task<T> — minimal coroutine task type (co_await / co_return)
///   2. Generator<T> — minimal generator for streaming step results
///   3. GPUEventAwaiter — co_await hipEventSynchronize
///   4. NpuReadyAwaiter — co_await NpuDmaPipeline::wait_gpu_ready
///   5. SleepAwaiter — co_await timeout durations
///   6. AsyncPipeline — coroutine entry points wrapping hq::Pipeline
///
/// @target MinisForum UM790 Pro (7940HS + Radeon 780M + Hailo-8L)
/// @requires C++26 coroutines
/// @version 1.0.0

#include "hq/cxx26_features.hpp"
#include "hq/pipeline.hpp"
#include "hq/npu_pipeline.hpp"

#include <atomic>
#include <chrono>
#include <concepts>
#if UM790_HAS_COROUTINES
#  include <coroutine>
#else
#  error "async_pipeline.hpp requires C++26 coroutines (<coroutine>) — GCC >= 14 or Clang >= 18 with C++26 enabled"
#endif
#include <cstddef>
#include <cstdint>
#if UM790_HAS_STD_EXPECTED
#  include <expected>
#else
#  error "async_pipeline.hpp requires std::expected (<expected>) — GCC >= 14 or Clang >= 18 with C++26 enabled"
#endif
#if UM790_HAS_STD_FORMAT
#  include <format>
#endif
#include <memory>
#include <mutex>
#include <optional>
#if UM790_HAS_STD_PRINT
#  include <print>
#endif
#include <stop_token>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef UM790_HAS_HIP
#include <hip/hip_runtime_api.h>
#endif

namespace hq::async {

// ===========================================================================
// StepResult: streaming progress from each denoising step
// ===========================================================================

struct StepResult {
    std::uint32_t step_index{0};
    float         gpu_util{0.0f};
    float         hailo_util{0.0f};
    float         temperature{0.0f};
    double        latency_ms{0.0};
    const float*  latents_ptr{nullptr};
};

// ===========================================================================
// Awaiter concept
// ===========================================================================

template<typename T>
concept Awaiter = requires(T a, std::coroutine_handle<> h) {
    { a.await_ready() } -> std::same_as<bool>;
    a.await_suspend(h);
    a.await_resume();
};

// ===========================================================================
// task<T>: minimal C++26 coroutine task type
// ===========================================================================

template<typename T>
class task;

namespace detail {

struct task_promise_base {
    std::coroutine_handle<> continuation_{nullptr};
    std::exception_ptr     exception_;

    std::suspend_never initial_suspend() noexcept { return {}; }
    void unhandled_exception() noexcept { exception_ = std::current_exception(); }
};

template<typename T>
struct task_promise final : task_promise_base {
    task<T> get_return_object();

    auto final_suspend() noexcept {
        struct final_awaiter {
            std::coroutine_handle<> continuation_{nullptr};
            bool await_ready() noexcept { return false; }
            std::coroutine_handle<> await_suspend(
                std::coroutine_handle<>) noexcept {
                return continuation_;
            }
            void await_resume() noexcept {}
        };
        return final_awaiter{continuation_};
    }

    template<typename U>
        requires std::convertible_to<U, T>
    void return_value(U&& value) {
        result_ = std::forward<U>(value);
    }

    std::optional<T> result_;
};

template<>
struct task_promise<void> : task_promise_base {
    task<void> get_return_object();

    auto final_suspend() noexcept {
        struct final_awaiter {
            std::coroutine_handle<> continuation_{nullptr};
            bool await_ready() noexcept { return false; }
            std::coroutine_handle<> await_suspend(
                std::coroutine_handle<>) noexcept {
                return continuation_;
            }
            void await_resume() noexcept {}
        };
        return final_awaiter{continuation_};
    }

    void return_void() noexcept {}
};

} // namespace detail

template<typename T>
class task {
public:
    using promise_type = detail::task_promise<T>;

    explicit task(std::coroutine_handle<promise_type> h) noexcept
        : handle_{h} {}

    task(const task&) = delete;
    task& operator=(const task&) = delete;

    task(task&& other) noexcept
        : handle_{std::exchange(other.handle_, nullptr)} {}

    task& operator=(task&& other) noexcept {
        if (this != &other) {
            if (handle_) handle_.destroy();
            handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
    }

    ~task() noexcept {
        if (handle_) handle_.destroy();
    }

    [[nodiscard]] bool done() const noexcept {
        return !handle_ || handle_.done();
    }

    [[nodiscard]] bool has_exception() const noexcept {
        return handle_ && static_cast<bool>(handle_.promise().exception_);
    }

    void resume() {
        if (handle_ && !handle_.done()) handle_.resume();
    }

    [[nodiscard]] T& result() & {
        if (handle_ && !handle_.done()) handle_.resume();
        if (auto& ex = handle_.promise().exception_; ex)
            std::rethrow_exception(ex);
        return *handle_.promise().result_;
    }

    [[nodiscard]] const T& result() const& {
        if (auto& ex = handle_.promise().exception_; ex)
            std::rethrow_exception(ex);
        return *handle_.promise().result_;
    }

    [[nodiscard]] T&& result() && {
        if (handle_ && !handle_.done()) handle_.resume();
        if (auto& ex = handle_.promise().exception_; ex)
            std::rethrow_exception(ex);
        return std::move(*handle_.promise().result_);
    }

    struct awaiter {
        std::coroutine_handle<promise_type> handle_;
        bool await_ready() noexcept {
            return !handle_ || handle_.done();
        }
        void await_suspend(std::coroutine_handle<> caller) noexcept {
            handle_.promise().continuation_ = caller;
            handle_.resume();
        }
        decltype(auto) await_resume() {
            if (auto& ex = handle_.promise().exception_; ex)
                std::rethrow_exception(ex);
            if constexpr (!std::is_void_v<T>) {
                return std::move(*handle_.promise().result_);
            }
        }
    };

    awaiter operator co_await() && noexcept { return awaiter{handle_}; }
    awaiter operator co_await() & noexcept { return awaiter{handle_}; }

private:
    std::coroutine_handle<promise_type> handle_;
};

// void specialization
template<>
class task<void> {
public:
    using promise_type = detail::task_promise<void>;

    explicit task(std::coroutine_handle<promise_type> h) noexcept
        : handle_{h} {}

    task(const task&) = delete;
    task& operator=(const task&) = delete;

    task(task&& other) noexcept
        : handle_{std::exchange(other.handle_, nullptr)} {}

    task& operator=(task&& other) noexcept {
        if (this != &other) {
            if (handle_) handle_.destroy();
            handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
    }

    ~task() noexcept {
        if (handle_) handle_.destroy();
    }

    [[nodiscard]] bool done() const noexcept {
        return !handle_ || handle_.done();
    }

    [[nodiscard]] bool has_exception() const noexcept {
        return handle_ && static_cast<bool>(handle_.promise().exception_);
    }

    void resume() {
        if (handle_ && !handle_.done()) handle_.resume();
    }

    void result() const {
        if (handle_ && handle_.promise().exception_)
            std::rethrow_exception(handle_.promise().exception_);
    }

    struct awaiter {
        std::coroutine_handle<promise_type> handle_;
        bool await_ready() noexcept {
            return !handle_ || handle_.done();
        }
        void await_suspend(std::coroutine_handle<> caller) noexcept {
            handle_.promise().continuation_ = caller;
            handle_.resume();
        }
        void await_resume() {
            if (auto& ex = handle_.promise().exception_; ex)
                std::rethrow_exception(ex);
        }
    };

    awaiter operator co_await() && noexcept { return awaiter{handle_}; }
    awaiter operator co_await() & noexcept { return awaiter{handle_}; }

private:
    std::coroutine_handle<promise_type> handle_;
};

namespace detail {
template<typename T>
task<T> task_promise<T>::get_return_object() {
    return task<T>{
        std::coroutine_handle<task_promise<T>>::from_promise(*this)};
}
inline task<void> task_promise<void>::get_return_object() {
    return task<void>{
        std::coroutine_handle<task_promise<void>>::from_promise(*this)};
}
} // namespace detail

// ===========================================================================
// Generator<T>: minimal lazy generator
// ===========================================================================

template<typename T>
class Generator;

namespace detail {

template<typename T>
struct generator_promise {
    std::optional<T>   current_value_;
    std::exception_ptr exception_;

    Generator<T> get_return_object();

    std::suspend_always initial_suspend() noexcept { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }

    std::suspend_always yield_value(T value) noexcept {
        current_value_ = std::move(value);
        return {};
    }

    void return_void() noexcept {}
    void unhandled_exception() noexcept { exception_ = std::current_exception(); }
};

} // namespace detail

template<typename T>
class Generator {
public:
    using promise_type = detail::generator_promise<T>;

    explicit Generator(std::coroutine_handle<promise_type> h) noexcept
        : handle_{h} {}

    Generator(const Generator&) = delete;
    Generator& operator=(const Generator&) = delete;

    Generator(Generator&& other) noexcept
        : handle_{std::exchange(other.handle_, nullptr)} {}

    Generator& operator=(Generator&& other) noexcept {
        if (this != &other) {
            if (handle_) handle_.destroy();
            handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
    }

    ~Generator() noexcept {
        if (handle_) handle_.destroy();
    }

    struct iterator {
        using iterator_category = std::input_iterator_tag;
        using value_type        = T;
        using difference_type   = std::ptrdiff_t;
        using pointer           = const T*;
        using reference         = const T&;

        std::coroutine_handle<promise_type> handle_{nullptr};

        iterator() = default;
        explicit iterator(std::coroutine_handle<promise_type> h) noexcept
            : handle_{h} {}

        reference operator*() const noexcept {
            return *handle_.promise().current_value_;
        }

        pointer operator->() const noexcept {
            return &*handle_.promise().current_value_;
        }

        iterator& operator++() {
            handle_.resume();
            if (handle_.done()) {
                if (auto& ex = handle_.promise().exception_; ex)
                    std::rethrow_exception(ex);
                handle_ = nullptr;
            }
            return *this;
        }

        void operator++(int) { ++*this; }

        bool operator==(const iterator& other) const noexcept {
            return handle_ == other.handle_;
        }
    };

    [[nodiscard]] iterator begin() {
        if (handle_ && !handle_.done()) {
            handle_.resume();
            if (handle_.done()) {
                if (auto& ex = handle_.promise().exception_; ex)
                    std::rethrow_exception(ex);
                return iterator{nullptr};
            }
            return iterator{handle_};
        }
        return iterator{nullptr};
    }

    [[nodiscard]] iterator end() noexcept { return iterator{nullptr}; }

private:
    std::coroutine_handle<promise_type> handle_;
};

namespace detail {
template<typename T>
Generator<T> generator_promise<T>::get_return_object() {
    return Generator<T>{
        std::coroutine_handle<generator_promise<T>>::from_promise(*this)};
}
} // namespace detail

// ===========================================================================
// GPUEventAwaiter: co_await hipEventSynchronize
// ===========================================================================

class GPUEventAwaiter {
public:
#ifdef UM790_HAS_HIP
    explicit GPUEventAwaiter(hipEvent_t event) noexcept : event_{event} {}

    ~GPUEventAwaiter() {
        stop_source_.request_stop();
    }

    GPUEventAwaiter(const GPUEventAwaiter&) = delete;
    GPUEventAwaiter& operator=(const GPUEventAwaiter&) = delete;
    GPUEventAwaiter(GPUEventAwaiter&&) = delete;
    GPUEventAwaiter& operator=(GPUEventAwaiter&&) = delete;

    [[nodiscard]] bool await_ready() noexcept {
        if (!event_) return true;
        return hipEventQuery(event_) == hipSuccess;
    }

    void await_suspend(std::coroutine_handle<> caller) {
        auto tok = stop_source_.get_token();
        thread_ = std::jthread{[ev = event_, c = caller, tok] {
            if (ev) hipEventSynchronize(ev);
            if (!tok.stop_requested()) c.resume();
        }};
    }

    void await_resume() noexcept {}

private:
    hipEvent_t        event_{nullptr};
    std::stop_source  stop_source_;
    std::jthread      thread_;
#else
    explicit GPUEventAwaiter(void* = nullptr) noexcept {}

    [[nodiscard]] bool await_ready() noexcept { return true; }
    void await_suspend(std::coroutine_handle<>) noexcept {}
    void await_resume() noexcept {}
#endif
};

// ===========================================================================
// NpuReadyAwaiter: co_await NpuDmaPipeline::wait_gpu_ready
// ===========================================================================

class NpuReadyAwaiter {
public:
    NpuReadyAwaiter(hq::npu::NpuDmaPipeline& pipeline, std::size_t slot,
                    std::chrono::milliseconds timeout)
        : pipeline_{&pipeline}, slot_{slot}, timeout_{timeout} {}

    ~NpuReadyAwaiter() {
        stop_source_.request_stop();
    }

    NpuReadyAwaiter(const NpuReadyAwaiter&) = delete;
    NpuReadyAwaiter& operator=(const NpuReadyAwaiter&) = delete;
    NpuReadyAwaiter(NpuReadyAwaiter&&) = delete;
    NpuReadyAwaiter& operator=(NpuReadyAwaiter&&) = delete;

    [[nodiscard]] bool await_ready() {
        auto ready = pipeline_->try_get_gpu_handle(slot_);
        if (ready.has_value()) {
            result_.emplace(std::move(*ready));
            return true;
        }
        return false;
    }

    void await_suspend(std::coroutine_handle<> caller) {
        auto tok = stop_source_.get_token();
        thread_ = std::jthread{[this, c = caller, tok] {
            auto h = pipeline_->wait_gpu_ready(slot_, timeout_);
            if (!tok.stop_requested()) {
                if (h.has_value()) result_.emplace(std::move(*h));
                c.resume();
            }
        }};
    }

    [[nodiscard]] hq::npu::NpuTensorHandle await_resume() {
        if (result_.has_value()) return std::move(*result_);
        return hq::npu::NpuTensorHandle{};
    }

private:
    hq::npu::NpuDmaPipeline*            pipeline_{nullptr};
    std::size_t                         slot_{0};
    std::chrono::milliseconds           timeout_{5000};
    std::optional<hq::npu::NpuTensorHandle> result_;
    std::stop_source                    stop_source_;
    std::jthread                        thread_;
};

// ===========================================================================
// SleepAwaiter: co_await a duration
// ===========================================================================

template<typename Rep, typename Period>
class SleepAwaiter {
public:
    explicit SleepAwaiter(std::chrono::duration<Rep, Period> d) noexcept
        : duration_{std::chrono::duration_cast<std::chrono::milliseconds>(d)}
    {}

    ~SleepAwaiter() {
        stop_source_.request_stop();
    }

    SleepAwaiter(const SleepAwaiter&) = delete;
    SleepAwaiter& operator=(const SleepAwaiter&) = delete;
    SleepAwaiter(SleepAwaiter&&) = delete;
    SleepAwaiter& operator=(SleepAwaiter&&) = delete;

    [[nodiscard]] bool await_ready() const noexcept {
        return duration_.count() <= 0;
    }

    void await_suspend(std::coroutine_handle<> caller) {
        auto d = duration_;
        auto tok = stop_source_.get_token();
        thread_ = std::jthread{[d, c = caller, tok] {
            std::this_thread::sleep_for(d);
            if (!tok.stop_requested()) c.resume();
        }};
    }

    void await_resume() noexcept {}

private:
    std::chrono::milliseconds duration_;
    std::stop_source          stop_source_;
    std::jthread              thread_;
};

// ===========================================================================
// AsyncPipeline: coroutine entry points wrapping hq::Pipeline
// ===========================================================================

class AsyncPipeline {
public:
    explicit AsyncPipeline(const PipelineConfig& cfg);
    ~AsyncPipeline() noexcept;

    AsyncPipeline(const AsyncPipeline&) = delete;
    AsyncPipeline& operator=(const AsyncPipeline&) = delete;
    AsyncPipeline(AsyncPipeline&&) noexcept;
    AsyncPipeline& operator=(AsyncPipeline&&) noexcept;

    [[nodiscard]] task<GeneratedImage> generate_async(GenerationRequest req);

    [[nodiscard]] Generator<StepResult> generate_streaming(
        GenerationRequest req);

    [[nodiscard]] PipelineStats get_stats() const;
    void shutdown() noexcept;

    [[nodiscard]] hq::Pipeline& inner() noexcept { return *pipeline_; }
    [[nodiscard]] hq::npu::NpuDmaPipeline& npu() noexcept {
        return *npu_pipeline_;
    }

private:
    std::unique_ptr<hq::Pipeline> pipeline_;
    std::unique_ptr<hq::npu::NpuDmaPipeline> npu_pipeline_;
};

} // namespace hq::async
