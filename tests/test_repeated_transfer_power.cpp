#include "qubit/qtransfer_power.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

using namespace qubit;

namespace {

bool close(double left, double right, double tolerance = 2e-10) {
    return std::abs(left - right) <= tolerance;
}

bool close(QComplex left, QComplex right, double tolerance = 2e-10) {
    return almost_equal(left, right, tolerance);
}

void require(bool condition, const char* message) {
    if (!condition) {
        throw QStateError(message);
    }
}

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
    return output;
}

std::vector<QComplex> sequential(
    std::span<const QComplex> matrix,
    std::uint64_t repetitions,
    std::span<const QComplex> input,
    std::size_t dimension) {
    std::vector<QComplex> current(input.begin(), input.end());
    for (std::uint64_t repeat = 0U; repeat < repetitions; ++repeat) {
        current = matvec(matrix, current, dimension);
    }
    return current;
}

void powered_matches_sequential() {
    const std::vector<QComplex> matrix = amplitude_damping_ry(0.071, 0.19);
    ExactRepeatedTransferKernel kernel(4U, matrix, true);
    const std::vector<QComplex> initial{{1.0}, {}, {}, {1.0}};

    for (const std::uint64_t repetitions :
         std::vector<std::uint64_t>{0U, 1U, 2U, 3U, 7U, 31U, 256U, 4096U}) {
        const RepeatedTransferPowerResult powered = kernel.apply(repetitions, initial);
        const std::vector<QComplex> expected = sequential(matrix, repetitions, initial, 4U);
        require(powered.vector.size() == expected.size(),
                "repeated transfer output dimension mismatch");
        for (std::size_t index = 0U; index < expected.size(); ++index) {
            require(close(powered.vector[index], expected[index]),
                    "repeated transfer powered/sequential mismatch");
        }
        if (repetitions == 0U) {
            require(powered.stats.matrix_matrix_multiplications == 0U,
                    "zero repetition unexpectedly multiplied matrices");
            require(powered.stats.matrix_vector_applications == 0U,
                    "zero repetition unexpectedly applied a matrix");
        } else {
            require(powered.stats.matrix_vector_applications == 1U,
                    "powered transfer did not finish with one matrix/vector apply");
        }
        require(powered.vector[0] == QComplex{1.0},
                "homogeneous trace coordinate drifted");
    }
}

void huge_power_of_two_is_structural() {
    const std::vector<QComplex> matrix = amplitude_damping_ry(0.071, 0.19);
    ExactRepeatedTransferKernel kernel(4U, matrix, true);
    const std::vector<QComplex> initial{{1.0}, {}, {}, {1.0}};
    const RepeatedTransferPowerResult result = kernel.apply_power_of_two(256U, initial);

    require(result.stats.description_bits == 256U,
            "power-of-two transfer description bit count mismatch");
    require(result.stats.matrix_matrix_multiplications == 257U,
            "power-of-two transfer multiplication count mismatch");
    require(result.stats.matrix_vector_applications == 1U,
            "power-of-two transfer matrix/vector count mismatch");
    require(result.stats.scalar_multiply_accumulates == 16464U,
            "power-of-two transfer scalar work mismatch");
    require(result.vector[0] == QComplex{1.0},
            "huge power-of-two homogeneous coordinate drifted");

    const double x = result.vector[1].re;
    const double y = result.vector[2].re;
    const double z = result.vector[3].re;
    require(std::abs(result.vector[1].im) < 1e-12 &&
                std::abs(result.vector[2].im) < 1e-12 &&
                std::abs(result.vector[3].im) < 1e-12,
            "huge powered Bloch coordinates became complex");
    require(x * x + y * y + z * z <= 1.0 + 1e-10,
            "huge powered Bloch vector became unphysical");
}

void schedule_identity_is_explicit() {
    const std::vector<QComplex> matrix = amplitude_damping_ry(0.071, 0.19);
    ExactRepeatedTransferKernel kernel(4U, matrix, true);
    require(kernel.same_kernel(matrix), "repeated transfer rejected its own kernel");

    std::vector<QComplex> changed = matrix;
    changed[15].re += 1e-15;
    require(!kernel.same_kernel(changed),
            "repeated transfer treated a changed kernel as identical");

    bool rejected = false;
    try {
        kernel.require_identical_kernel(changed);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "repeated transfer did not reject a varying schedule kernel");
}

void rejection_cases() {
    bool rejected = false;
    try {
        std::vector<QComplex> bad = amplitude_damping_ry(0.071, 0.19);
        bad[0] = QComplex{0.999};
        ExactRepeatedTransferKernel invalid(4U, bad, true);
        (void)invalid;
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "repeated transfer accepted an invalid homogeneous certificate");

    rejected = false;
    try {
        std::vector<QComplex> bad = amplitude_damping_ry(0.071, 0.19);
        bad[5].re = std::numeric_limits<double>::infinity();
        ExactRepeatedTransferKernel invalid(4U, bad, false);
        (void)invalid;
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "repeated transfer accepted a non-finite kernel");

    rejected = false;
    try {
        const std::vector<QComplex> matrix = amplitude_damping_ry(0.071, 0.19);
        ExactRepeatedTransferKernel kernel(4U, matrix, true);
        const std::vector<QComplex> wrong{{1.0}, {}, {1.0}};
        (void)kernel.apply(8U, wrong);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "repeated transfer accepted a wrong-size vector");

    rejected = false;
    try {
        const std::vector<QComplex> matrix = amplitude_damping_ry(0.071, 0.19);
        RepeatedTransferConfig config;
        config.max_description_bits = 8U;
        ExactRepeatedTransferKernel kernel(4U, matrix, true, config);
        const std::vector<QComplex> initial{{1.0}, {}, {}, {1.0}};
        (void)kernel.apply_power_of_two(9U, initial);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "repeated transfer ignored its description-bit cap");

    rejected = false;
    try {
        const std::vector<QComplex> matrix = amplitude_damping_ry(0.071, 0.19);
        RepeatedTransferConfig config;
        config.max_matrix_multiplications = 4U;
        ExactRepeatedTransferKernel kernel(4U, matrix, true, config);
        const std::vector<QComplex> initial{{1.0}, {}, {}, {1.0}};
        (void)kernel.apply_power_of_two(8U, initial);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "repeated transfer ignored its multiplication cap");
}

}  // namespace

int main() {
    powered_matches_sequential();
    huge_power_of_two_is_structural();
    schedule_identity_is_explicit();
    rejection_cases();
    std::cout << "repeated transfer power tests passed\n";
    return 0;
}
