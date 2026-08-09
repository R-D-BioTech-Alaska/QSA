#include "qubit/qnumeric.hpp"
#include "qnumeric_backend.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

#if defined(QSTATE_NUMERIC_HAS_AVX2) && defined(_MSC_VER)
#include <intrin.h>
#endif

namespace qubit {
namespace {

constexpr std::size_t kMaxNumericWorkers = 128U;

[[nodiscard]] std::size_t resolve_workers(std::size_t requested) {
    if (requested > kMaxNumericWorkers) {
        throw std::invalid_argument("NumericExecutor worker_count exceeds the supported limit");
    }
    if (requested != 0U) {
        return requested;
    }
    const unsigned int hardware = std::thread::hardware_concurrency();
    return std::max<std::size_t>(
        1U, std::min<std::size_t>(hardware == 0U ? 1U : hardware, kMaxNumericWorkers));
}

[[nodiscard]] std::size_t chunks(std::size_t size, std::size_t grain) noexcept {
    return size / grain + static_cast<std::size_t>(size % grain != 0U);
}

template <typename Value>
[[nodiscard]] bool overlaps(std::span<const Value> input, std::span<Value> output) noexcept {
    if (input.empty() || output.empty()) {
        return false;
    }
    const auto a = reinterpret_cast<std::uintptr_t>(input.data());
    const auto b = reinterpret_cast<std::uintptr_t>(output.data());
    return a < b + output.size_bytes() && b < a + input.size_bytes();
}

template <typename Value>
void validate_output(
    std::span<const Value> first,
    std::span<const Value> second,
    std::span<Value> output) {
    if (first.size() != second.size() || first.size() != output.size()) {
        throw std::invalid_argument("NumericExecutor input and output sizes must match");
    }
    if ((overlaps(first, output) && first.data() != output.data()) ||
        (overlaps(second, output) && second.data() != output.data())) {
        throw std::invalid_argument("NumericExecutor does not allow partial output overlap");
    }
}

[[nodiscard]] QComplex multiply(QComplex first, QComplex second) noexcept {
    return {
        std::fma(first.re, second.re, -first.im * second.im),
        std::fma(first.re, second.im, first.im * second.re),
    };
}

[[nodiscard]] QComplex affine(
    QComplex first,
    QComplex second,
    QComplex first_scale,
    QComplex second_scale,
    QComplex bias) noexcept {
    const QComplex left = multiply(first, first_scale);
    const QComplex right = multiply(second, second_scale);
    return {left.re + right.re + bias.re, left.im + right.im + bias.im};
}

[[nodiscard]] double norm2(QComplex value) noexcept {
    return std::fma(value.re, value.re, value.im * value.im);
}

[[nodiscard]] QComplex inner_term(QComplex first, QComplex second) noexcept {
    return {
        std::fma(first.re, second.re, first.im * second.im),
        std::fma(first.re, second.im, -first.im * second.re),
    };
}

}  // namespace

#if defined(QSTATE_NUMERIC_HAS_AVX2)
namespace numeric_detail {

bool avx2_fma_available() noexcept {
#if defined(_MSC_VER)
    int registers[4]{};
    __cpuid(registers, 1);
    const bool fma = (registers[2] & (1 << 12)) != 0;
    const bool osxsave = (registers[2] & (1 << 27)) != 0;
    const bool avx = (registers[2] & (1 << 28)) != 0;
    if (!fma || !osxsave || !avx || (_xgetbv(0) & 0x6) != 0x6) {
        return false;
    }
    __cpuidex(registers, 7, 0);
    return (registers[1] & (1 << 5)) != 0;
#elif defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

}  // namespace numeric_detail
#endif

class NumericExecutor::Impl {
public:
    explicit Impl(NumericConfig config)
        : worker_count_(resolve_workers(config.worker_count)), grain_size_(config.grain_size) {
        if (grain_size_ == 0U) {
            throw std::invalid_argument("NumericExecutor grain_size must be positive");
        }
#if defined(QSTATE_NUMERIC_HAS_AVX2)
        if (config.enable_simd && numeric_detail::avx2_fma_available()) {
            backend_ = NumericBackend::Avx2Fma;
        }
#endif
        helpers_.reserve(worker_count_ - 1U);
        try {
            for (std::size_t index = 0; index + 1U < worker_count_; ++index) {
                helpers_.emplace_back([this, index] { worker_loop(index); });
            }
        } catch (...) {
            stop_workers();
            throw;
        }
    }

    ~Impl() {
        std::unique_lock<std::mutex> lock(run_mutex_);
        stop_workers();
    }

    [[nodiscard]] std::size_t worker_count() const noexcept { return worker_count_; }
    [[nodiscard]] std::size_t grain_size() const noexcept { return grain_size_; }
    [[nodiscard]] NumericBackend backend() const noexcept { return backend_; }
    [[nodiscard]] const char* backend_name() const noexcept {
        return backend_ == NumericBackend::Avx2Fma ? "avx2_fma" : "scalar";
    }

    void fused_affine(
        std::span<const double> first,
        std::span<const double> second,
        double first_scale,
        double second_scale,
        double bias,
        std::span<double> output) {
        validate_output(first, second, output);
        if (output.empty()) {
            return;
        }
        std::unique_lock<std::mutex> lock(run_mutex_);
        const std::size_t task_count = chunks(output.size(), grain_size_);
        auto task = [&](std::size_t task_index) noexcept {
            const std::size_t begin = task_index * grain_size_;
            const std::size_t end = std::min(output.size(), begin + grain_size_);
#if defined(QSTATE_NUMERIC_HAS_AVX2)
            if (backend_ == NumericBackend::Avx2Fma) {
                numeric_detail::fused_affine_avx2(
                    first.data() + begin, second.data() + begin,
                    first_scale, second_scale, bias, output.data() + begin, end - begin);
                return;
            }
#endif
            for (std::size_t index = begin; index < end; ++index) {
                output[index] = std::fma(
                    first[index], first_scale,
                    std::fma(second[index], second_scale, bias));
            }
        };
        run_tasks_locked(task_count, task);
    }

    [[nodiscard]] double fused_affine_norm2(
        std::span<const double> first,
        std::span<const double> second,
        double first_scale,
        double second_scale,
        double bias,
        std::span<double> output) {
        validate_output(first, second, output);
        if (output.empty()) {
            return 0.0;
        }
        std::unique_lock<std::mutex> lock(run_mutex_);
        const std::size_t task_count = chunks(output.size(), grain_size_);
        real_partials_.resize(task_count);
        auto task = [&](std::size_t task_index) noexcept {
            const std::size_t begin = task_index * grain_size_;
            const std::size_t end = std::min(output.size(), begin + grain_size_);
#if defined(QSTATE_NUMERIC_HAS_AVX2)
            if (backend_ == NumericBackend::Avx2Fma) {
                real_partials_[task_index] = numeric_detail::fused_affine_norm2_avx2(
                    first.data() + begin, second.data() + begin,
                    first_scale, second_scale, bias, output.data() + begin, end - begin);
                return;
            }
#endif
            double sum = 0.0;
            for (std::size_t index = begin; index < end; ++index) {
                const double value = std::fma(
                    first[index], first_scale,
                    std::fma(second[index], second_scale, bias));
                output[index] = value;
                sum = std::fma(value, value, sum);
            }
            real_partials_[task_index] = sum;
        };
        run_tasks_locked(task_count, task);
        return reduce_real(task_count);
    }

    void fused_complex_affine(
        std::span<const QComplex> first,
        std::span<const QComplex> second,
        QComplex first_scale,
        QComplex second_scale,
        QComplex bias,
        std::span<QComplex> output) {
        validate_output(first, second, output);
        if (output.empty()) {
            return;
        }
        std::unique_lock<std::mutex> lock(run_mutex_);
        const std::size_t task_count = chunks(output.size(), grain_size_);
        auto task = [&](std::size_t task_index) noexcept {
            const std::size_t begin = task_index * grain_size_;
            const std::size_t end = std::min(output.size(), begin + grain_size_);
#if defined(QSTATE_NUMERIC_HAS_AVX2)
            if (backend_ == NumericBackend::Avx2Fma) {
                numeric_detail::fused_complex_affine_avx2(
                    first.data() + begin, second.data() + begin,
                    first_scale, second_scale, bias, output.data() + begin, end - begin);
                return;
            }
#endif
            for (std::size_t index = begin; index < end; ++index) {
                output[index] = affine(
                    first[index], second[index], first_scale, second_scale, bias);
            }
        };
        run_tasks_locked(task_count, task);
    }

    [[nodiscard]] double fused_complex_affine_norm2(
        std::span<const QComplex> first,
        std::span<const QComplex> second,
        QComplex first_scale,
        QComplex second_scale,
        QComplex bias,
        std::span<QComplex> output) {
        validate_output(first, second, output);
        if (output.empty()) {
            return 0.0;
        }
        std::unique_lock<std::mutex> lock(run_mutex_);
        const std::size_t task_count = chunks(output.size(), grain_size_);
        real_partials_.resize(task_count);
        auto task = [&](std::size_t task_index) noexcept {
            const std::size_t begin = task_index * grain_size_;
            const std::size_t end = std::min(output.size(), begin + grain_size_);
#if defined(QSTATE_NUMERIC_HAS_AVX2)
            if (backend_ == NumericBackend::Avx2Fma) {
                real_partials_[task_index] = numeric_detail::fused_complex_affine_norm2_avx2(
                    first.data() + begin, second.data() + begin,
                    first_scale, second_scale, bias, output.data() + begin, end - begin);
                return;
            }
#endif
            double sum = 0.0;
            for (std::size_t index = begin; index < end; ++index) {
                const QComplex value = affine(
                    first[index], second[index], first_scale, second_scale, bias);
                output[index] = value;
                sum += norm2(value);
            }
            real_partials_[task_index] = sum;
        };
        run_tasks_locked(task_count, task);
        return reduce_real(task_count);
    }

    [[nodiscard]] double dot(
        std::span<const double> first,
        std::span<const double> second) {
        if (first.size() != second.size()) {
            throw std::invalid_argument("NumericExecutor dot input sizes must match");
        }
        if (first.empty()) {
            return 0.0;
        }
        std::unique_lock<std::mutex> lock(run_mutex_);
        const std::size_t task_count = chunks(first.size(), grain_size_);
        real_partials_.resize(task_count);
        auto task = [&](std::size_t task_index) noexcept {
            const std::size_t begin = task_index * grain_size_;
            const std::size_t end = std::min(first.size(), begin + grain_size_);
#if defined(QSTATE_NUMERIC_HAS_AVX2)
            if (backend_ == NumericBackend::Avx2Fma) {
                real_partials_[task_index] = numeric_detail::dot_avx2(
                    first.data() + begin, second.data() + begin, end - begin);
                return;
            }
#endif
            double sum = 0.0;
            for (std::size_t index = begin; index < end; ++index) {
                sum = std::fma(first[index], second[index], sum);
            }
            real_partials_[task_index] = sum;
        };
        run_tasks_locked(task_count, task);
        return reduce_real(task_count);
    }

    [[nodiscard]] QComplex inner_product(
        std::span<const QComplex> first,
        std::span<const QComplex> second) {
        if (first.size() != second.size()) {
            throw std::invalid_argument("NumericExecutor inner-product sizes must match");
        }
        if (first.empty()) {
            return {};
        }
        std::unique_lock<std::mutex> lock(run_mutex_);
        const std::size_t task_count = chunks(first.size(), grain_size_);
        complex_partials_.resize(task_count);
        auto task = [&](std::size_t task_index) noexcept {
            const std::size_t begin = task_index * grain_size_;
            const std::size_t end = std::min(first.size(), begin + grain_size_);
            QComplex sum{};
            for (std::size_t index = begin; index < end; ++index) {
                const QComplex term = inner_term(first[index], second[index]);
                sum.re += term.re;
                sum.im += term.im;
            }
            complex_partials_[task_index] = sum;
        };
        run_tasks_locked(task_count, task);
        QComplex total{};
        for (std::size_t index = 0; index < task_count; ++index) {
            total += complex_partials_[index];
        }
        return total;
    }

    void matrix2_batch(
        std::span<const QComplex, 4> matrix,
        std::span<const QComplex> input,
        std::span<QComplex> output) {
        matrix_io(input, output, 2U);
        const std::size_t vector_count = input.size() / 2U;
        if (vector_count == 0U) {
            return;
        }
        const std::size_t grain = std::max<std::size_t>(1U, grain_size_ / 2U);
        std::unique_lock<std::mutex> lock(run_mutex_);
        const std::size_t task_count = chunks(vector_count, grain);
        auto task = [&](std::size_t task_index) noexcept {
            const std::size_t begin = task_index * grain;
            const std::size_t end = std::min(vector_count, begin + grain);
#if defined(QSTATE_NUMERIC_HAS_AVX2)
            if (backend_ == NumericBackend::Avx2Fma) {
                numeric_detail::matrix2_batch_avx2(
                    matrix.data(), input.data() + begin * 2U,
                    output.data() + begin * 2U, end - begin);
                return;
            }
#endif
            for (std::size_t vector = begin; vector < end; ++vector) {
                const std::size_t offset = vector * 2U;
                const QComplex first = input[offset];
                const QComplex second = input[offset + 1U];
                output[offset] = multiply(matrix[0], first) + multiply(matrix[1], second);
                output[offset + 1U] = multiply(matrix[2], first) + multiply(matrix[3], second);
            }
        };
        run_tasks_locked(task_count, task);
    }

    void matrix4_batch(
        std::span<const QComplex, 16> matrix,
        std::span<const QComplex> input,
        std::span<QComplex> output) {
        matrix_io(input, output, 4U);
        const std::size_t vector_count = input.size() / 4U;
        if (vector_count == 0U) {
            return;
        }
        const std::size_t grain = std::max<std::size_t>(1U, grain_size_ / 4U);
        std::unique_lock<std::mutex> lock(run_mutex_);
        const std::size_t task_count = chunks(vector_count, grain);
        auto task = [&](std::size_t task_index) noexcept {
            const std::size_t begin = task_index * grain;
            const std::size_t end = std::min(vector_count, begin + grain);
#if defined(QSTATE_NUMERIC_HAS_AVX2)
            if (backend_ == NumericBackend::Avx2Fma) {
                numeric_detail::matrix4_batch_avx2(
                    matrix.data(), input.data() + begin * 4U,
                    output.data() + begin * 4U, end - begin);
                return;
            }
#endif
            for (std::size_t vector = begin; vector < end; ++vector) {
                const std::size_t offset = vector * 4U;
                const std::array<QComplex, 4> values{
                    input[offset], input[offset + 1U], input[offset + 2U], input[offset + 3U]};
                for (std::size_t row = 0; row < 4U; ++row) {
                    QComplex sum{};
                    for (std::size_t column = 0; column < 4U; ++column) {
                        sum += multiply(matrix[row * 4U + column], values[column]);
                    }
                    output[offset + row] = sum;
                }
            }
        };
        run_tasks_locked(task_count, task);
    }

private:
    using Invoke = void (*)(void*, std::size_t) noexcept;

    std::size_t worker_count_{1U};
    std::size_t grain_size_{1U};
    NumericBackend backend_{NumericBackend::Scalar};
    std::vector<std::thread> helpers_{};
    std::mutex run_mutex_{};
    std::mutex state_mutex_{};
    std::condition_variable start_cv_{};
    std::condition_variable done_cv_{};
    bool stopping_{false};
    std::size_t generation_{0U};
    std::size_t active_helpers_{0U};
    std::size_t pending_helpers_{0U};
    std::size_t task_count_{0U};
    std::atomic<std::size_t> next_task_{0U};
    void* task_context_{nullptr};
    Invoke invoke_{nullptr};
    std::vector<double> real_partials_{};
    std::vector<QComplex> complex_partials_{};

    void stop_workers() noexcept {
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            stopping_ = true;
            ++generation_;
        }
        start_cv_.notify_all();
        for (std::thread& helper : helpers_) {
            if (helper.joinable()) {
                helper.join();
            }
        }
    }

    void worker_loop(std::size_t worker_index) noexcept {
        std::size_t seen = 0U;
        for (;;) {
            void* context = nullptr;
            Invoke invoke = nullptr;
            std::size_t count = 0U;
            bool active = false;
            {
                std::unique_lock<std::mutex> lock(state_mutex_);
                start_cv_.wait(lock, [&] { return stopping_ || generation_ != seen; });
                if (stopping_) {
                    return;
                }
                seen = generation_;
                active = worker_index < active_helpers_;
                if (active) {
                    context = task_context_;
                    invoke = invoke_;
                    count = task_count_;
                }
            }
            if (!active) {
                continue;
            }
            while (true) {
                const std::size_t task = next_task_.fetch_add(1U, std::memory_order_relaxed);
                if (task >= count) {
                    break;
                }
                invoke(context, task);
            }
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (--pending_helpers_ == 0U) {
                done_cv_.notify_one();
            }
        }
    }

    template <typename Function>
    void run_tasks_locked(std::size_t count, Function& function) {
        if (count == 0U) {
            return;
        }
        if (count == 1U || helpers_.empty()) {
            for (std::size_t task = 0; task < count; ++task) {
                function(task);
            }
            return;
        }
        using FunctionType = std::remove_reference_t<Function>;
        const std::size_t active = std::min(helpers_.size(), count - 1U);
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            active_helpers_ = active;
            pending_helpers_ = active;
            task_count_ = count;
            next_task_.store(0U, std::memory_order_relaxed);
            task_context_ = std::addressof(function);
            invoke_ = [](void* context, std::size_t task) noexcept {
                (*static_cast<FunctionType*>(context))(task);
            };
            ++generation_;
        }
        start_cv_.notify_all();
        while (true) {
            const std::size_t task = next_task_.fetch_add(1U, std::memory_order_relaxed);
            if (task >= count) {
                break;
            }
            function(task);
        }
        std::unique_lock<std::mutex> lock(state_mutex_);
        done_cv_.wait(lock, [&] { return pending_helpers_ == 0U; });
        task_context_ = nullptr;
        invoke_ = nullptr;
        task_count_ = 0U;
        active_helpers_ = 0U;
    }

    [[nodiscard]] double reduce_real(std::size_t count) const noexcept {
        double total = 0.0;
        for (std::size_t index = 0; index < count; ++index) {
            total += real_partials_[index];
        }
        return total;
    }

    static void matrix_io(
        std::span<const QComplex> input,
        std::span<QComplex> output,
        std::size_t width) {
        if (input.size() != output.size()) {
            throw std::invalid_argument("NumericExecutor matrix input and output sizes must match");
        }
        if (input.size() % width != 0U) {
            throw std::invalid_argument("NumericExecutor matrix batch has an incomplete vector");
        }
        if (overlaps(input, output) && input.data() != output.data()) {
            throw std::invalid_argument("NumericExecutor does not allow partial matrix output overlap");
        }
    }
};

NumericExecutor::NumericExecutor(NumericConfig config)
    : impl_(std::make_unique<Impl>(config)) {}
NumericExecutor::~NumericExecutor() = default;

std::size_t NumericExecutor::worker_count() const noexcept { return impl_->worker_count(); }
std::size_t NumericExecutor::grain_size() const noexcept { return impl_->grain_size(); }
NumericBackend NumericExecutor::backend() const noexcept { return impl_->backend(); }
const char* NumericExecutor::backend_name() const noexcept { return impl_->backend_name(); }

void NumericExecutor::fused_affine(
    std::span<const double> first,
    std::span<const double> second,
    double first_scale,
    double second_scale,
    double bias,
    std::span<double> output) const {
    impl_->fused_affine(first, second, first_scale, second_scale, bias, output);
}

double NumericExecutor::fused_affine_norm2(
    std::span<const double> first,
    std::span<const double> second,
    double first_scale,
    double second_scale,
    double bias,
    std::span<double> output) const {
    return impl_->fused_affine_norm2(first, second, first_scale, second_scale, bias, output);
}

void NumericExecutor::fused_complex_affine(
    std::span<const QComplex> first,
    std::span<const QComplex> second,
    QComplex first_scale,
    QComplex second_scale,
    QComplex bias,
    std::span<QComplex> output) const {
    impl_->fused_complex_affine(first, second, first_scale, second_scale, bias, output);
}

double NumericExecutor::fused_complex_affine_norm2(
    std::span<const QComplex> first,
    std::span<const QComplex> second,
    QComplex first_scale,
    QComplex second_scale,
    QComplex bias,
    std::span<QComplex> output) const {
    return impl_->fused_complex_affine_norm2(
        first, second, first_scale, second_scale, bias, output);
}

double NumericExecutor::dot(
    std::span<const double> first,
    std::span<const double> second) const {
    return impl_->dot(first, second);
}

QComplex NumericExecutor::inner_product(
    std::span<const QComplex> first,
    std::span<const QComplex> second) const {
    return impl_->inner_product(first, second);
}

void NumericExecutor::matrix2_batch(
    std::span<const QComplex, 4> matrix,
    std::span<const QComplex> input,
    std::span<QComplex> output) const {
    impl_->matrix2_batch(matrix, input, output);
}

void NumericExecutor::matrix4_batch(
    std::span<const QComplex, 16> matrix,
    std::span<const QComplex> input,
    std::span<QComplex> output) const {
    impl_->matrix4_batch(matrix, input, output);
}

}  // namespace qubit
