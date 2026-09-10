#pragma once

#include "qubit/qstate.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace qubit {

struct FermionicGaussianConfig {
    std::size_t max_modes{4096U};
    std::size_t max_covariance_scalars{1U << 24U};
};

enum class FermionicGaussianEligibilityCode {
    eligible,
    zero_modes,
    non_gaussian_input,
    non_quadratic_hamiltonian,
    mode_cap_exceeded,
    size_overflow,
    scalar_cap_exceeded,
    memory_budget_exceeded,
};

struct FermionicGaussianEligibility {
    FermionicGaussianEligibilityCode code{FermionicGaussianEligibilityCode::zero_modes};
    std::size_t required_bytes{0U};

    [[nodiscard]] bool accepted() const noexcept {
        return code == FermionicGaussianEligibilityCode::eligible;
    }
};

[[nodiscard]] inline FermionicGaussianEligibility fermionic_gaussian_eligibility(
    std::size_t modes,
    bool gaussian_input,
    std::size_t maximum_majorana_degree,
    std::size_t memory_budget_bytes,
    FermionicGaussianConfig config = {}) noexcept {
    if (modes == 0U) {
        return {FermionicGaussianEligibilityCode::zero_modes, 0U};
    }
    if (!gaussian_input) {
        return {FermionicGaussianEligibilityCode::non_gaussian_input, 0U};
    }
    if (maximum_majorana_degree != 2U) {
        return {FermionicGaussianEligibilityCode::non_quadratic_hamiltonian, 0U};
    }
    if (config.max_modes == 0U || config.max_covariance_scalars == 0U ||
        modes > config.max_modes) {
        return {FermionicGaussianEligibilityCode::mode_cap_exceeded, 0U};
    }
    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    if (modes > maximum / 2U) {
        return {FermionicGaussianEligibilityCode::size_overflow, maximum};
    }
    const std::size_t dimension = 2U * modes;
    if (dimension > maximum / dimension) {
        return {FermionicGaussianEligibilityCode::size_overflow, maximum};
    }
    const std::size_t scalars = dimension * dimension;
    if (scalars > config.max_covariance_scalars) {
        return {FermionicGaussianEligibilityCode::scalar_cap_exceeded, 0U};
    }
    if (scalars > maximum / sizeof(double)) {
        return {FermionicGaussianEligibilityCode::size_overflow, maximum};
    }
    const std::size_t required = scalars * sizeof(double);
    if (required > memory_budget_bytes) {
        return {FermionicGaussianEligibilityCode::memory_budget_exceeded, required};
    }
    return {FermionicGaussianEligibilityCode::eligible, required};
}

class FermionicGaussianState {
public:
    explicit FermionicGaussianState(
        std::size_t modes,
        FermionicGaussianConfig config = {})
        : modes_(modes), config_(config) {
        const auto eligibility = fermionic_gaussian_eligibility(
            modes_, true, 2U, std::numeric_limits<std::size_t>::max(), config_);
        if (!eligibility.accepted()) {
            throw QStateError("Fermionic Gaussian dimensions exceed configured resource limits");
        }
        const std::size_t dimension = 2U * modes_;
        covariance_.assign(dimension * dimension, 0.0);
        for (std::size_t mode = 0U; mode < modes_; ++mode) {
            const std::size_t first = 2U * mode;
            const std::size_t second = first + 1U;
            covariance_[first * dimension + second] = -1.0;
            covariance_[second * dimension + first] = 1.0;
        }
    }

    [[nodiscard]] std::size_t mode_count() const noexcept { return modes_; }
    [[nodiscard]] std::size_t majorana_count() const noexcept { return 2U * modes_; }
    [[nodiscard]] const FermionicGaussianConfig& config() const noexcept { return config_; }
    [[nodiscard]] const std::vector<double>& covariance() const noexcept { return covariance_; }
    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        return sizeof(*this) + covariance_.capacity() * sizeof(double);
    }

    void rotate_majoranas(std::size_t first, std::size_t second, double angle) {
        validate_majorana(first);
        validate_majorana(second);
        if (first == second || !std::isfinite(angle)) {
            throw QStateError(
                "Fermionic Majorana rotation requires distinct indices and a finite angle");
        }

        const std::size_t dimension = majorana_count();
        const double c = std::cos(angle);
        const double s = std::sin(angle);
        if (!std::isfinite(c) || !std::isfinite(s)) {
            throw QStateError("Fermionic Majorana rotation produced a non-finite transform");
        }

        std::vector<double> first_row(dimension);
        std::vector<double> second_row(dimension);
        std::vector<double> first_column(dimension);
        std::vector<double> second_column(dimension);
        for (std::size_t index = 0U; index < dimension; ++index) {
            const double left_row = covariance_[first * dimension + index];
            const double right_row = covariance_[second * dimension + index];
            const double left_column = covariance_[index * dimension + first];
            const double right_column = covariance_[index * dimension + second];
            first_row[index] = c * left_row - s * right_row;
            second_row[index] = s * left_row + c * right_row;
            first_column[index] = c * left_column - s * right_column;
            second_column[index] = s * left_column + c * right_column;
            if (!std::isfinite(first_row[index]) || !std::isfinite(second_row[index]) ||
                !std::isfinite(first_column[index]) || !std::isfinite(second_column[index])) {
                throw QStateError("Fermionic Majorana rotation became non-finite");
            }
        }

        const double ff = c * first_row[first] - s * first_row[second];
        const double fs = s * first_row[first] + c * first_row[second];
        const double sf = c * second_row[first] - s * second_row[second];
        const double ss = s * second_row[first] + c * second_row[second];
        if (!std::isfinite(ff) || !std::isfinite(fs) ||
            !std::isfinite(sf) || !std::isfinite(ss)) {
            throw QStateError("Fermionic Majorana rotation became non-finite");
        }

        for (std::size_t index = 0U; index < dimension; ++index) {
            if (index != first && index != second) {
                covariance_[first * dimension + index] = first_row[index];
                covariance_[second * dimension + index] = second_row[index];
                covariance_[index * dimension + first] = first_column[index];
                covariance_[index * dimension + second] = second_column[index];
            }
        }
        covariance_[first * dimension + first] = ff;
        covariance_[first * dimension + second] = fs;
        covariance_[second * dimension + first] = sf;
        covariance_[second * dimension + second] = ss;
    }

    [[nodiscard]] double occupation(std::size_t mode) const {
        validate_mode(mode);
        const std::size_t dimension = majorana_count();
        const std::size_t first = 2U * mode;
        const double value = 0.5 * (1.0 + covariance_[first * dimension + first + 1U]);
        if (!std::isfinite(value)) {
            throw QStateError("Fermionic Gaussian occupation became non-finite");
        }
        if (value < 0.0 && value >= -1e-12) {
            return 0.0;
        }
        if (value > 1.0 && value <= 1.0 + 1e-12) {
            return 1.0;
        }
        return value;
    }

    [[nodiscard]] double total_occupation() const {
        double total = 0.0;
        for (std::size_t mode = 0U; mode < modes_; ++mode) {
            const double next = total + occupation(mode);
            if (!std::isfinite(next)) {
                throw QStateError("Fermionic Gaussian total occupation became non-finite");
            }
            total = next;
        }
        return total;
    }

    [[nodiscard]] bool validate(double tolerance = 1e-10) const noexcept {
        if (!valid_tolerance(tolerance)) {
            return false;
        }
        const std::size_t dimension = majorana_count();
        if (modes_ == 0U || covariance_.size() != dimension * dimension) {
            return false;
        }
        for (std::size_t row = 0U; row < dimension; ++row) {
            for (std::size_t column = row; column < dimension; ++column) {
                const double left = covariance_[row * dimension + column];
                const double right = covariance_[column * dimension + row];
                if (!std::isfinite(left) || !std::isfinite(right) ||
                    std::abs(left + right) > tolerance) {
                    return false;
                }
            }
        }
        return true;
    }

    [[nodiscard]] bool validate_pure(double tolerance = 1e-10) const noexcept {
        if (!validate(tolerance)) {
            return false;
        }
        const std::size_t dimension = majorana_count();
        for (std::size_t row = 0U; row < dimension; ++row) {
            for (std::size_t column = 0U; column < dimension; ++column) {
                double value = row == column ? 1.0 : 0.0;
                for (std::size_t inner = 0U; inner < dimension; ++inner) {
                    value += covariance_[row * dimension + inner] *
                             covariance_[inner * dimension + column];
                }
                if (!std::isfinite(value) || std::abs(value) > tolerance) {
                    return false;
                }
            }
        }
        return true;
    }

private:
    std::size_t modes_{0U};
    FermionicGaussianConfig config_{};
    std::vector<double> covariance_{};

    void validate_mode(std::size_t mode) const {
        if (mode >= modes_) {
            throw QStateError("Fermionic mode is outside logical shape");
        }
    }

    void validate_majorana(std::size_t index) const {
        if (index >= majorana_count()) {
            throw QStateError("Fermionic Majorana index is outside logical shape");
        }
    }

    [[nodiscard]] static bool valid_tolerance(double tolerance) noexcept {
        return tolerance >= 0.0 && std::isfinite(tolerance);
    }
};

}  // namespace qubit
