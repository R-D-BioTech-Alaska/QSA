#include "qubit/qkron.hpp"
#include "qubit/qnumeric.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using qubit::KronVector;
using qubit::NumericConfig;
using qubit::NumericExecutor;
using qubit::QComplex;

[[nodiscard]] QComplex multiply(QComplex first, QComplex second) noexcept {
    return {
        std::fma(first.re, second.re, -first.im * second.im),
        std::fma(first.re, second.im, first.im * second.re),
    };
}

[[nodiscard]] double norm2(QComplex value) noexcept {
    return std::fma(value.re, value.re, value.im * value.im);
}

template <typename Function>
[[nodiscard]] double best_milliseconds(Function&& function, std::size_t repetitions = 5U) {
    double best = std::numeric_limits<double>::infinity();
    for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
        const auto start = Clock::now();
        function();
        const auto finish = Clock::now();
        best = std::min(
            best,
            std::chrono::duration<double, std::milli>(finish - start).count());
    }
    return best;
}

[[nodiscard]] std::vector<double> real_data(std::size_t size, double shift) {
    std::vector<double> values(size);
    for (std::size_t index = 0; index < size; ++index) {
        const double x = static_cast<double>(index + 1U);
        values[index] = std::sin(0.00017 * x + shift) +
                        0.35 * std::cos(0.00011 * x - shift);
    }
    return values;
}

[[nodiscard]] std::vector<QComplex> complex_data(std::size_t size, double shift) {
    std::vector<QComplex> values(size);
    for (std::size_t index = 0; index < size; ++index) {
        const double x = static_cast<double>(index + 1U);
        values[index] = {
            std::sin(0.00013 * x + shift),
            0.8 * std::cos(0.00009 * x - shift),
        };
    }
    return values;
}

[[nodiscard]] KronVector structured_vector(std::size_t sites) {
    KronVector result(std::vector<std::size_t>(sites, 2U), 4U);
    std::vector<std::vector<QComplex>> first;
    std::vector<std::vector<QComplex>> second;
    first.reserve(sites);
    second.reserve(sites);
    for (std::size_t site = 0; site < sites; ++site) {
        const double angle = 0.01 * static_cast<double>(site + 1U);
        const double phase = 0.013 * static_cast<double>(site + 1U);
        first.push_back({{std::cos(angle), 0.0}, {std::sin(angle), 0.0}});
        second.push_back({{std::cos(phase), 0.0}, {0.0, std::sin(phase)}});
    }
    result.add_term({0.75, 0.125}, first);
    result.add_term({-0.2, 0.05}, second);
    return result;
}

}  // namespace

int main() {
    std::cout << std::setprecision(12);

    NumericExecutor serial(NumericConfig{1U, 16'384U});
    NumericExecutor parallel(NumericConfig{0U, 16'384U});
    std::cout << "numeric_worker_count=" << parallel.worker_count() << '\n';
    std::cout << "numeric_grain_size=" << parallel.grain_size() << '\n';

    {
        constexpr std::size_t size = 2'000'000U;
        constexpr double first_scale = 1.25;
        constexpr double second_scale = -0.375;
        constexpr double bias = 0.0625;
        const std::vector<double> first = real_data(size, 0.17);
        const std::vector<double> second = real_data(size, -0.29);
        std::vector<double> temporary(size);
        std::vector<double> split_output(size);
        std::vector<double> serial_output(size);
        std::vector<double> parallel_output(size);
        double split_norm = 0.0;
        double serial_norm = 0.0;
        double parallel_norm = 0.0;

        static_cast<void>(parallel.fused_affine_norm2(
            first, second, first_scale, second_scale, bias, parallel_output));

        const double split_ms = best_milliseconds([&] {
            for (std::size_t index = 0; index < size; ++index) {
                temporary[index] = std::fma(second[index], second_scale, bias);
            }
            for (std::size_t index = 0; index < size; ++index) {
                split_output[index] = std::fma(first[index], first_scale, temporary[index]);
            }
            double sum = 0.0;
            for (const double value : split_output) {
                sum = std::fma(value, value, sum);
            }
            split_norm = sum;
        });
        const double serial_ms = best_milliseconds([&] {
            serial_norm = serial.fused_affine_norm2(
                first, second, first_scale, second_scale, bias, serial_output);
        });
        const double parallel_ms = best_milliseconds([&] {
            parallel_norm = parallel.fused_affine_norm2(
                first, second, first_scale, second_scale, bias, parallel_output);
        });

        double max_error = 0.0;
        for (std::size_t index = 0; index < size; ++index) {
            max_error = std::max(max_error, std::abs(split_output[index] - parallel_output[index]));
        }
        std::cout << "real_elements=" << size << '\n';
        std::cout << "real_split_scalar_ms=" << split_ms << '\n';
        std::cout << "real_fused_serial_ms=" << serial_ms << '\n';
        std::cout << "real_fused_parallel_ms=" << parallel_ms << '\n';
        std::cout << "real_fusion_ratio=" << split_ms / serial_ms << '\n';
        std::cout << "real_parallel_ratio=" << serial_ms / parallel_ms << '\n';
        std::cout << "real_max_error=" << max_error << '\n';
        std::cout << "real_split_norm2=" << split_norm << '\n';
        std::cout << "real_serial_norm2=" << serial_norm << '\n';
        std::cout << "real_parallel_norm2=" << parallel_norm << '\n';
    }

    {
        constexpr std::size_t size = 1'000'000U;
        const QComplex first_scale{0.75, -0.25};
        const QComplex second_scale{-0.125, 0.5};
        const QComplex bias{0.03125, -0.0625};
        const std::vector<QComplex> first = complex_data(size, 0.23);
        const std::vector<QComplex> second = complex_data(size, -0.31);
        std::vector<QComplex> temporary(size);
        std::vector<QComplex> split_output(size);
        std::vector<QComplex> serial_output(size);
        std::vector<QComplex> parallel_output(size);
        double split_norm = 0.0;
        double serial_norm = 0.0;
        double parallel_norm = 0.0;

        static_cast<void>(parallel.fused_complex_affine_norm2(
            first, second, first_scale, second_scale, bias, parallel_output));

        const double split_ms = best_milliseconds([&] {
            for (std::size_t index = 0; index < size; ++index) {
                const QComplex right = multiply(second[index], second_scale);
                temporary[index] = right + bias;
            }
            for (std::size_t index = 0; index < size; ++index) {
                split_output[index] = multiply(first[index], first_scale) + temporary[index];
            }
            double sum = 0.0;
            for (const QComplex value : split_output) {
                sum += norm2(value);
            }
            split_norm = sum;
        });
        const double serial_ms = best_milliseconds([&] {
            serial_norm = serial.fused_complex_affine_norm2(
                first, second, first_scale, second_scale, bias, serial_output);
        });
        const double parallel_ms = best_milliseconds([&] {
            parallel_norm = parallel.fused_complex_affine_norm2(
                first, second, first_scale, second_scale, bias, parallel_output);
        });

        double max_error = 0.0;
        for (std::size_t index = 0; index < size; ++index) {
            max_error = std::max(
                max_error,
                (split_output[index] - parallel_output[index]).magnitude());
        }
        std::cout << "complex_elements=" << size << '\n';
        std::cout << "complex_split_scalar_ms=" << split_ms << '\n';
        std::cout << "complex_fused_serial_ms=" << serial_ms << '\n';
        std::cout << "complex_fused_parallel_ms=" << parallel_ms << '\n';
        std::cout << "complex_fusion_ratio=" << split_ms / serial_ms << '\n';
        std::cout << "complex_parallel_ratio=" << serial_ms / parallel_ms << '\n';
        std::cout << "complex_max_error=" << max_error << '\n';
        std::cout << "complex_split_norm2=" << split_norm << '\n';
        std::cout << "complex_serial_norm2=" << serial_norm << '\n';
        std::cout << "complex_parallel_norm2=" << parallel_norm << '\n';
    }

    {
        constexpr std::size_t vector_count = 500'000U;
        const std::array<QComplex, 4> matrix{{
            {0.5, 0.25}, {-0.125, 0.75},
            {0.625, -0.375}, {0.25, 0.125},
        }};
        const std::vector<QComplex> input = complex_data(vector_count * 2U, 0.41);
        std::vector<QComplex> scalar_output(input.size());
        std::vector<QComplex> serial_output(input.size());
        std::vector<QComplex> parallel_output(input.size());

        parallel.matrix2_batch(
            std::span<const QComplex, 4>(matrix), input, parallel_output);

        const double scalar_ms = best_milliseconds([&] {
            for (std::size_t vector = 0; vector < vector_count; ++vector) {
                const std::size_t offset = vector * 2U;
                const QComplex first = input[offset];
                const QComplex second = input[offset + 1U];
                scalar_output[offset] = multiply(matrix[0], first) + multiply(matrix[1], second);
                scalar_output[offset + 1U] = multiply(matrix[2], first) + multiply(matrix[3], second);
            }
        });
        const double serial_ms = best_milliseconds([&] {
            serial.matrix2_batch(
                std::span<const QComplex, 4>(matrix), input, serial_output);
        });
        const double parallel_ms = best_milliseconds([&] {
            parallel.matrix2_batch(
                std::span<const QComplex, 4>(matrix), input, parallel_output);
        });

        double max_error = 0.0;
        double checksum = 0.0;
        for (std::size_t index = 0; index < scalar_output.size(); ++index) {
            max_error = std::max(
                max_error,
                (scalar_output[index] - parallel_output[index]).magnitude());
            checksum += parallel_output[index].norm2();
        }
        std::cout << "matrix2_vectors=" << vector_count << '\n';
        std::cout << "matrix2_scalar_ms=" << scalar_ms << '\n';
        std::cout << "matrix2_serial_ms=" << serial_ms << '\n';
        std::cout << "matrix2_parallel_ms=" << parallel_ms << '\n';
        std::cout << "matrix2_parallel_ratio=" << serial_ms / parallel_ms << '\n';
        std::cout << "matrix2_max_error=" << max_error << '\n';
        std::cout << "matrix2_checksum=" << checksum << '\n';
    }

    {
        constexpr std::size_t sites = 20U;
        KronVector structured = structured_vector(sites);
        double structured_norm = 0.0;
        const double structured_ms = best_milliseconds([&] {
            structured_norm = structured.norm2();
        }, 25U);
        std::cout << "kron20_sites=" << sites << '\n';
        std::cout << "kron20_logical_elements=" << structured.logical_size() << '\n';
        std::cout << "kron20_terms=" << structured.term_count() << '\n';
        std::cout << "kron20_norm_ms=" << structured_ms << '\n';
        std::cout << "kron20_norm2=" << structured_norm << '\n';
        std::cout << "kron20_bytes=" << structured.estimated_bytes() << '\n';
    }

    {
        constexpr std::size_t sites = 100U;
        KronVector structured = structured_vector(sites);
        double structured_norm = 0.0;
        const double structured_ms = best_milliseconds([&] {
            structured_norm = structured.norm2();
        }, 25U);
        std::cout << "kron100_sites=" << sites << '\n';
        std::cout << "kron100_log2_elements="
                  << static_cast<double>(structured.log2_logical_size()) << '\n';
        std::cout << "kron100_dense_size_fits="
                  << static_cast<int>(structured.logical_size_fits()) << '\n';
        std::cout << "kron100_terms=" << structured.term_count() << '\n';
        std::cout << "kron100_norm_ms=" << structured_ms << '\n';
        std::cout << "kron100_norm2=" << structured_norm << '\n';
        std::cout << "kron100_bytes=" << structured.estimated_bytes() << '\n';
    }

    return 0;
}
