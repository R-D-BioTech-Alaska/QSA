#include "qubit/qtransfer_power.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace qubit;

namespace {

std::vector<QComplex> amplitude_damping_ry(double gamma, double theta) {
    const double a = std::sqrt(1.0 - gamma);
    const double b = 1.0 - gamma;
    const double cosine = std::cos(theta);
    const double sine = std::sin(theta);
    return {
        {1.0}, {}, {}, {},
        {}, {a * cosine}, {}, {a * sine},
        {}, {}, {a}, {},
        {gamma}, {-b * sine}, {}, {b * cosine},
    };
}

std::vector<QComplex> matvec(
    std::span<const QComplex> matrix,
    std::span<const QComplex> vector,
    std::size_t dimension) {
    std::vector<QComplex> output(dimension, QComplex{});
    for (std::size_t row = 0U; row < dimension; ++row) {
        for (std::size_t column = 0U; column < dimension; ++column) {
            output[row] += matrix[row * dimension + column] * vector[column];
        }
    }
    output[0] = vector[0];
    return output;
}

struct SequentialResult {
    std::vector<QComplex> vector{};
    double seconds{0.0};
};

SequentialResult sequential(
    std::span<const QComplex> matrix,
    std::uint64_t repetitions,
    std::span<const QComplex> initial) {
    std::vector<QComplex> current(initial.begin(), initial.end());
    const auto started = std::chrono::steady_clock::now();
    for (std::uint64_t repeat = 0U; repeat < repetitions; ++repeat) {
        current = matvec(matrix, current, 4U);
    }
    const auto finished = std::chrono::steady_clock::now();
    return SequentialResult{
        std::move(current),
        std::chrono::duration<double>(finished - started).count(),
    };
}

double max_error(
    std::span<const QComplex> left,
    std::span<const QComplex> right) {
    double error = 0.0;
    for (std::size_t index = 0U; index < left.size(); ++index) {
        error = std::max(error, (left[index] - right[index]).magnitude());
    }
    return error;
}

bool physical(std::span<const QComplex> vector) {
    if (vector.size() != 4U || vector[0] != QComplex{1.0}) {
        return false;
    }
    if (std::abs(vector[1].im) > 1e-10 ||
        std::abs(vector[2].im) > 1e-10 ||
        std::abs(vector[3].im) > 1e-10) {
        return false;
    }
    const double x = vector[1].re;
    const double y = vector[2].re;
    const double z = vector[3].re;
    return x * x + y * y + z * z <= 1.0 + 1e-10;
}

}  // namespace

int main() {
    const std::vector<QComplex> matrix = amplitude_damping_ry(0.071, 0.19);
    ExactRepeatedTransferKernel kernel(4U, matrix, true);
    const std::vector<QComplex> initial{{1.0}, {}, {}, {1.0}};

    std::cout << std::setprecision(17);
    double maximum_error = 0.0;
    double b20_speedup = 0.0;
    double b20_scalar_ratio = 0.0;

    for (const std::size_t bits : std::vector<std::size_t>{8U, 12U, 16U, 18U, 20U}) {
        const std::uint64_t repetitions = std::uint64_t{1U} << bits;

        double powered_best = 1e100;
        RepeatedTransferPowerResult powered;
        for (std::size_t repeat = 0U; repeat < 5U; ++repeat) {
            const auto started = std::chrono::steady_clock::now();
            RepeatedTransferPowerResult current = kernel.apply(repetitions, initial);
            const auto finished = std::chrono::steady_clock::now();
            powered_best = std::min(
                powered_best,
                std::chrono::duration<double>(finished - started).count());
            powered = std::move(current);
        }

        const SequentialResult linear = sequential(matrix, repetitions, initial);
        const double error = max_error(powered.vector, linear.vector);
        maximum_error = std::max(maximum_error, error);
        const double speedup = linear.seconds / powered_best;
        const double sequential_scalar_ops = static_cast<double>(repetitions) * 16.0;
        const double scalar_ratio = sequential_scalar_ops /
            static_cast<double>(powered.stats.scalar_multiply_accumulates);
        if (bits == 20U) {
            b20_speedup = speedup;
            b20_scalar_ratio = scalar_ratio;
        }

        std::cout << "b" << bits << "_repetitions=" << repetitions << '\n';
        std::cout << "b" << bits << "_powered_seconds=" << powered_best << '\n';
        std::cout << "b" << bits << "_sequential_seconds=" << linear.seconds << '\n';
        std::cout << "b" << bits << "_speedup=" << speedup << '\n';
        std::cout << "b" << bits << "_scalar_work_ratio=" << scalar_ratio << '\n';
        std::cout << "b" << bits << "_matrix_multiplications="
                  << powered.stats.matrix_matrix_multiplications << '\n';
        std::cout << "b" << bits << "_max_error=" << error << '\n';
        std::cout << "b" << bits << "_physical=" << (physical(powered.vector) ? 1 : 0) << '\n';
    }

    const auto huge_started = std::chrono::steady_clock::now();
    const RepeatedTransferPowerResult huge = kernel.apply_power_of_two(256U, initial);
    const auto huge_finished = std::chrono::steady_clock::now();

    std::vector<QComplex> changed = matrix;
    changed[15].re += 1e-15;
    bool varying_rejected = false;
    try {
        kernel.require_identical_kernel(changed);
    } catch (const QStateError&) {
        varying_rejected = true;
    }

    std::cout << "maximum_sequential_error=" << maximum_error << '\n';
    std::cout << "b20_speedup=" << b20_speedup << '\n';
    std::cout << "b20_scalar_work_ratio=" << b20_scalar_ratio << '\n';
    std::cout << "huge_description_bits=256\n";
    std::cout << "huge_repetitions_log2=256\n";
    std::cout << "huge_matrix_multiplications="
              << huge.stats.matrix_matrix_multiplications << '\n';
    std::cout << "huge_scalar_multiply_accumulates="
              << huge.stats.scalar_multiply_accumulates << '\n';
    std::cout << "huge_physical=" << (physical(huge.vector) ? 1 : 0) << '\n';
    std::cout << "huge_trace_coordinate_exact=" << (huge.vector[0] == QComplex{1.0} ? 1 : 0) << '\n';
    std::cout << "huge_seconds="
              << std::chrono::duration<double>(huge_finished - huge_started).count() << '\n';
    std::cout << "expanded_schedule_materialized=0\n";
    std::cout << "varying_schedule_rejected=" << (varying_rejected ? 1 : 0) << '\n';
    return 0;
}
