#pragma once

#include "qubit/qcomplex.hpp"
#include "qubit/qstate.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace qubit {

struct RepeatedTransferConfig {
    std::size_t max_dimension{256U};
    std::size_t max_matrix_entries{1U << 20U};
    std::size_t max_description_bits{4096U};
    std::size_t max_matrix_multiplications{1U << 20U};
    std::size_t max_scalar_multiply_accumulates{
        std::numeric_limits<std::size_t>::max()};
};

struct RepeatedTransferPowerStats {
    std::size_t dimension{0U};
    std::size_t description_bits{0U};
    std::size_t matrix_matrix_multiplications{0U};
    std::size_t matrix_vector_applications{0U};
    std::size_t scalar_multiply_accumulates{0U};
};

struct RepeatedTransferPowerResult {
    std::vector<QComplex> vector{};
    RepeatedTransferPowerStats stats{};
};

class ExactRepeatedTransferKernel {
public:
    ExactRepeatedTransferKernel(
        std::size_t dimension,
        std::span<const QComplex> matrix,
        bool fixed_homogeneous_coordinate = false,
        RepeatedTransferConfig config = {})
        : dimension_(dimension),
          fixed_homogeneous_coordinate_(fixed_homogeneous_coordinate),
          config_(config),
          matrix_(matrix.begin(), matrix.end()) {
        validate();
    }

    [[nodiscard]] std::size_t dimension() const noexcept { return dimension_; }
    [[nodiscard]] bool fixed_homogeneous_coordinate() const noexcept {
        return fixed_homogeneous_coordinate_;
    }
    [[nodiscard]] const std::vector<QComplex>& matrix() const noexcept { return matrix_; }
    [[nodiscard]] const RepeatedTransferConfig& config() const noexcept { return config_; }

    [[nodiscard]] bool same_kernel(std::span<const QComplex> candidate) const noexcept {
        if (candidate.size() != matrix_.size()) {
            return false;
        }
        for (std::size_t index = 0U; index < matrix_.size(); ++index) {
            if (!(matrix_[index] == candidate[index])) {
                return false;
            }
        }
        return true;
    }

    void require_identical_kernel(std::span<const QComplex> candidate) const {
        if (!same_kernel(candidate)) {
            throw QStateError(
                "Repeated transfer schedule is not one exactly identical kernel");
        }
    }

    [[nodiscard]] RepeatedTransferPowerResult apply(
        std::uint64_t repetitions,
        std::span<const QComplex> input) const {
        validate_vector(input);
        if (repetitions == 0U) {
            return RepeatedTransferPowerResult{
                std::vector<QComplex>(input.begin(), input.end()),
                RepeatedTransferPowerStats{dimension_, 0U, 0U, 0U, 0U},
            };
        }

        const std::size_t description_bits = bit_length(repetitions);
        std::vector<QComplex> result = identity();
        std::vector<QComplex> base = matrix_;
        std::uint64_t power = repetitions;
        std::size_t multiplications = 0U;

        while (power != 0U) {
            if ((power & 1U) != 0U) {
                result = multiply(result, base);
                ++multiplications;
                preflight(multiplications);
            }
            power >>= 1U;
            if (power != 0U) {
                base = multiply(base, base);
                ++multiplications;
                preflight(multiplications);
            }
        }

        return finish(std::move(result), input, description_bits, multiplications);
    }

    [[nodiscard]] RepeatedTransferPowerResult apply_power_of_two(
        std::size_t description_bits,
        std::span<const QComplex> input) const {
        validate_vector(input);
        if (description_bits > config_.max_description_bits) {
            throw QStateError("Repeated transfer description bits exceed configured cap");
        }
        const std::size_t multiplications = checked_sum(
            description_bits, 1U, "Repeated transfer multiplication count overflowed");
        preflight(multiplications);

        std::vector<QComplex> base = matrix_;
        for (std::size_t bit = 0U; bit < description_bits; ++bit) {
            base = multiply(base, base);
        }
        std::vector<QComplex> result = multiply(identity(), base);
        return finish(std::move(result), input, description_bits, multiplications);
    }

private:
    std::size_t dimension_{0U};
    bool fixed_homogeneous_coordinate_{false};
    RepeatedTransferConfig config_{};
    std::vector<QComplex> matrix_{};

    void validate() const {
        if (dimension_ == 0U || dimension_ > config_.max_dimension ||
            config_.max_matrix_entries == 0U || config_.max_matrix_multiplications == 0U ||
            config_.max_scalar_multiply_accumulates == 0U) {
            throw QStateError("Repeated transfer dimensions or configuration are invalid");
        }
        const std::size_t entries = checked_product(
            dimension_, dimension_, "Repeated transfer matrix dimension overflowed");
        if (entries > config_.max_matrix_entries || matrix_.size() != entries) {
            throw QStateError("Repeated transfer matrix shape exceeds configured contract");
        }
        for (const QComplex value : matrix_) {
            if (!finite(value)) {
                throw QStateError("Repeated transfer matrix contains non-finite values");
            }
        }
        if (fixed_homogeneous_coordinate_) {
            for (std::size_t column = 0U; column < dimension_; ++column) {
                const QComplex expected = column == 0U ? QComplex{1.0} : QComplex{};
                if (!(matrix_[column] == expected)) {
                    throw QStateError(
                        "Repeated transfer homogeneous certificate requires exact first row [1,0,...]");
                }
            }
        }
    }

    void validate_vector(std::span<const QComplex> input) const {
        if (input.size() != dimension_) {
            throw QStateError("Repeated transfer vector dimension mismatch");
        }
        for (const QComplex value : input) {
            if (!finite(value)) {
                throw QStateError("Repeated transfer vector contains non-finite values");
            }
        }
    }

    [[nodiscard]] RepeatedTransferPowerResult finish(
        std::vector<QComplex> powered,
        std::span<const QComplex> input,
        std::size_t description_bits,
        std::size_t multiplications) const {
        preflight(multiplications);
        std::vector<QComplex> output = multiply_vector(powered, input);
        const std::size_t scalar_ops = scalar_work(multiplications);
        return RepeatedTransferPowerResult{
            std::move(output),
            RepeatedTransferPowerStats{
                dimension_, description_bits, multiplications, 1U, scalar_ops,
            },
        };
    }

    [[nodiscard]] std::vector<QComplex> identity() const {
        std::vector<QComplex> result(matrix_.size(), QComplex{});
        for (std::size_t index = 0U; index < dimension_; ++index) {
            result[index * dimension_ + index] = QComplex{1.0};
        }
        return result;
    }

    [[nodiscard]] std::vector<QComplex> multiply(
        std::span<const QComplex> left,
        std::span<const QComplex> right) const {
        if (left.size() != matrix_.size() || right.size() != matrix_.size()) {
            throw QStateError("Repeated transfer matrix multiplication shape mismatch");
        }
        std::vector<QComplex> output(matrix_.size(), QComplex{});
        for (std::size_t row = 0U; row < dimension_; ++row) {
            for (std::size_t column = 0U; column < dimension_; ++column) {
                QComplex total{};
                for (std::size_t inner = 0U; inner < dimension_; ++inner) {
                    total += left[row * dimension_ + inner] *
                             right[inner * dimension_ + column];
                }
                output[row * dimension_ + column] = total;
            }
        }
        if (fixed_homogeneous_coordinate_) {
            for (std::size_t column = 0U; column < dimension_; ++column) {
                output[column] = column == 0U ? QComplex{1.0} : QComplex{};
            }
        }
        return output;
    }

    [[nodiscard]] std::vector<QComplex> multiply_vector(
        std::span<const QComplex> matrix,
        std::span<const QComplex> input) const {
        if (matrix.size() != matrix_.size() || input.size() != dimension_) {
            throw QStateError("Repeated transfer matrix/vector multiplication shape mismatch");
        }
        std::vector<QComplex> output(dimension_, QComplex{});
        for (std::size_t row = 0U; row < dimension_; ++row) {
            QComplex total{};
            for (std::size_t column = 0U; column < dimension_; ++column) {
                total += matrix[row * dimension_ + column] * input[column];
            }
            output[row] = total;
        }
        if (fixed_homogeneous_coordinate_) {
            output[0] = input[0];
        }
        return output;
    }

    void preflight(std::size_t multiplications) const {
        if (multiplications > config_.max_matrix_multiplications) {
            throw QStateError("Repeated transfer matrix multiplication count exceeds configured cap");
        }
        if (scalar_work(multiplications) > config_.max_scalar_multiply_accumulates) {
            throw QStateError("Repeated transfer scalar work exceeds configured cap");
        }
    }

    [[nodiscard]] std::size_t scalar_work(std::size_t multiplications) const {
        const std::size_t square = checked_product(
            dimension_, dimension_, "Repeated transfer scalar work overflowed");
        const std::size_t cube = checked_product(
            square, dimension_, "Repeated transfer scalar work overflowed");
        const std::size_t matrix_work = checked_product(
            multiplications, cube, "Repeated transfer scalar work overflowed");
        return checked_sum(
            matrix_work, square, "Repeated transfer scalar work overflowed");
    }

    [[nodiscard]] static std::size_t bit_length(std::uint64_t value) noexcept {
        std::size_t bits = 0U;
        while (value != 0U) {
            ++bits;
            value >>= 1U;
        }
        return bits;
    }

    [[nodiscard]] static bool finite(QComplex value) noexcept {
        return std::isfinite(value.re) && std::isfinite(value.im);
    }

    [[nodiscard]] static std::size_t checked_product(
        std::size_t left, std::size_t right, const char* message) {
        if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) {
            throw QStateError(message);
        }
        return left * right;
    }

    [[nodiscard]] static std::size_t checked_sum(
        std::size_t left, std::size_t right, const char* message) {
        if (right > std::numeric_limits<std::size_t>::max() - left) {
            throw QStateError(message);
        }
        return left + right;
    }
};

}  // namespace qubit
