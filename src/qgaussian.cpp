#include "qubit/qgaussian.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace qubit {

GaussianState::GaussianState(std::size_t modes)
    : modes_(modes), mean_(2U * modes, 0.0), covariance_(4U * modes * modes, 0.0) {
    if (modes_ == 0U) {
        throw QStateError("Gaussian state requires at least one mode");
    }
    for (std::size_t index = 0U; index < 2U * modes_; ++index) {
        covariance_[index * 2U * modes_ + index] = 0.5;
    }
}

std::size_t GaussianState::estimated_bytes() const noexcept {
    return sizeof(*this) + mean_.capacity() * sizeof(double) +
           covariance_.capacity() * sizeof(double);
}

void GaussianState::validate_mode(std::size_t mode) const {
    if (mode >= modes_) {
        throw QStateError("Gaussian mode is out of range");
    }
}

void GaussianState::apply_local(
    std::span<const std::size_t> indices,
    std::span<const double> transform) {
    const std::size_t width = indices.size();
    if (width == 0U || transform.size() != width * width) {
        throw QStateError("Gaussian local transform shape mismatch");
    }
    for (const std::size_t index : indices) {
        if (index >= 2U * modes_) {
            throw QStateError("Gaussian quadrature index is out of range");
        }
    }

    std::vector<double> next_mean(width, 0.0);
    for (std::size_t row = 0U; row < width; ++row) {
        for (std::size_t column = 0U; column < width; ++column) {
            next_mean[row] += transform[row * width + column] * mean_[indices[column]];
        }
    }
    for (std::size_t row = 0U; row < width; ++row) {
        mean_[indices[row]] = next_mean[row];
    }

    const std::size_t dimension = 2U * modes_;
    std::vector<double> rows(width * dimension, 0.0);
    for (std::size_t row = 0U; row < width; ++row) {
        for (std::size_t source = 0U; source < width; ++source) {
            const double weight = transform[row * width + source];
            for (std::size_t column = 0U; column < dimension; ++column) {
                rows[row * dimension + column] +=
                    weight * covariance_[indices[source] * dimension + column];
            }
        }
    }
    for (std::size_t row = 0U; row < width; ++row) {
        for (std::size_t column = 0U; column < dimension; ++column) {
            covariance_[indices[row] * dimension + column] = rows[row * dimension + column];
        }
    }

    std::vector<double> columns(dimension * width, 0.0);
    for (std::size_t row = 0U; row < dimension; ++row) {
        for (std::size_t column = 0U; column < width; ++column) {
            for (std::size_t source = 0U; source < width; ++source) {
                columns[row * width + column] +=
                    covariance_[row * dimension + indices[source]] *
                    transform[column * width + source];
            }
        }
    }
    for (std::size_t row = 0U; row < dimension; ++row) {
        for (std::size_t column = 0U; column < width; ++column) {
            covariance_[row * dimension + indices[column]] = columns[row * width + column];
        }
    }
}

void GaussianState::displace(std::size_t mode, double q, double p) {
    validate_mode(mode);
    if (!std::isfinite(q) || !std::isfinite(p)) {
        throw QStateError("Gaussian displacement must be finite");
    }
    mean_[2U * mode] += q;
    mean_[2U * mode + 1U] += p;
}

void GaussianState::phase_shift(std::size_t mode, double theta) {
    validate_mode(mode);
    if (!std::isfinite(theta)) {
        throw QStateError("Gaussian phase shift must be finite");
    }
    const double c = std::cos(theta);
    const double s = std::sin(theta);
    const std::array<std::size_t, 2> indices{2U * mode, 2U * mode + 1U};
    const std::array<double, 4> transform{c, -s, s, c};
    apply_local(indices, transform);
}

void GaussianState::squeeze(std::size_t mode, double r) {
    validate_mode(mode);
    if (!std::isfinite(r)) {
        throw QStateError("Gaussian squeeze must be finite");
    }
    const std::array<std::size_t, 2> indices{2U * mode, 2U * mode + 1U};
    const std::array<double, 4> transform{
        std::exp(-r), 0.0,
        0.0, std::exp(r),
    };
    apply_local(indices, transform);
}

void GaussianState::beam_splitter(
    std::size_t first,
    std::size_t second,
    double theta) {
    validate_mode(first);
    validate_mode(second);
    if (first == second || !std::isfinite(theta)) {
        throw QStateError("Gaussian beam splitter requires distinct modes and finite angle");
    }
    const double c = std::cos(theta);
    const double s = std::sin(theta);
    const std::array<std::size_t, 4> indices{
        2U * first, 2U * first + 1U, 2U * second, 2U * second + 1U,
    };
    const std::array<double, 16> transform{
        c, 0.0, s, 0.0,
        0.0, c, 0.0, s,
        -s, 0.0, c, 0.0,
        0.0, -s, 0.0, c,
    };
    apply_local(indices, transform);
}

void GaussianState::two_mode_squeeze(
    std::size_t first,
    std::size_t second,
    double r) {
    validate_mode(first);
    validate_mode(second);
    if (first == second || !std::isfinite(r)) {
        throw QStateError("Gaussian two-mode squeeze requires distinct modes and finite strength");
    }
    const double c = std::cosh(r);
    const double s = std::sinh(r);
    const std::array<std::size_t, 4> indices{
        2U * first, 2U * first + 1U, 2U * second, 2U * second + 1U,
    };
    const std::array<double, 16> transform{
        c, 0.0, s, 0.0,
        0.0, c, 0.0, -s,
        s, 0.0, c, 0.0,
        0.0, -s, 0.0, c,
    };
    apply_local(indices, transform);
}

double GaussianState::mean_photon_number(std::size_t mode) const {
    validate_mode(mode);
    const std::size_t dimension = 2U * modes_;
    const std::size_t q = 2U * mode;
    const std::size_t p = q + 1U;
    return 0.5 * (
        covariance_[q * dimension + q] + covariance_[p * dimension + p] +
        mean_[q] * mean_[q] + mean_[p] * mean_[p] - 1.0);
}

double GaussianState::total_mean_photon_number() const {
    double total = 0.0;
    for (std::size_t mode = 0U; mode < modes_; ++mode) {
        total += mean_photon_number(mode);
    }
    return total;
}

bool GaussianState::validate(double tolerance) const noexcept {
    if (!(tolerance >= 0.0) || !std::isfinite(tolerance)) {
        return false;
    }
    const std::size_t dimension = 2U * modes_;
    if (mean_.size() != dimension || covariance_.size() != dimension * dimension) {
        return false;
    }
    for (const double value : mean_) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    for (std::size_t row = 0U; row < dimension; ++row) {
        for (std::size_t column = 0U; column < dimension; ++column) {
            const double value = covariance_[row * dimension + column];
            if (!std::isfinite(value) ||
                std::abs(value - covariance_[column * dimension + row]) > tolerance) {
                return false;
            }
        }
    }
    for (std::size_t mode = 0U; mode < modes_; ++mode) {
        const std::size_t q = 2U * mode;
        const std::size_t p = q + 1U;
        const double qq = covariance_[q * dimension + q];
        const double pp = covariance_[p * dimension + p];
        const double qp = covariance_[q * dimension + p];
        if (qq * pp - qp * qp < 0.25 - tolerance) {
            return false;
        }
    }
    return true;
}

std::size_t fermionic_gaussian_required_bytes(std::size_t modes) noexcept {
    if (modes == 0U) {
        return 0U;
    }
    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    if (modes > maximum / 2U) {
        return maximum;
    }
    const std::size_t dimension = 2U * modes;
    if (dimension > maximum / dimension) {
        return maximum;
    }
    const std::size_t entries = dimension * dimension;
    if (entries > maximum / sizeof(double)) {
        return maximum;
    }
    return entries * sizeof(double);
}

FermionicGaussianEligibility fermionic_gaussian_eligibility(
    std::size_t modes,
    bool gaussian_input,
    std::size_t maximum_majorana_degree,
    std::size_t memory_budget_bytes) noexcept {
    if (modes == 0U) {
        return {FermionicGaussianEligibilityCode::zero_modes, 0U};
    }
    if (!gaussian_input) {
        return {FermionicGaussianEligibilityCode::non_gaussian_input, 0U};
    }
    if (maximum_majorana_degree != 2U) {
        return {FermionicGaussianEligibilityCode::non_quadratic_hamiltonian, 0U};
    }
    const std::size_t required = fermionic_gaussian_required_bytes(modes);
    if (required == std::numeric_limits<std::size_t>::max()) {
        return {FermionicGaussianEligibilityCode::size_overflow, required};
    }
    if (required > memory_budget_bytes) {
        return {FermionicGaussianEligibilityCode::memory_budget_exceeded, required};
    }
    return {FermionicGaussianEligibilityCode::eligible, required};
}

FermionicGaussianState::FermionicGaussianState(std::size_t modes) : modes_(modes) {
    if (modes_ == 0U) {
        throw QStateError("Fermionic Gaussian state requires at least one mode");
    }
    const std::size_t required = fermionic_gaussian_required_bytes(modes_);
    if (required == std::numeric_limits<std::size_t>::max()) {
        throw QStateError("Fermionic Gaussian covariance size overflow");
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

std::size_t FermionicGaussianState::estimated_bytes() const noexcept {
    return sizeof(*this) + covariance_.capacity() * sizeof(double);
}

void FermionicGaussianState::validate_majorana(std::size_t index) const {
    if (index >= 2U * modes_) {
        throw QStateError("Fermionic Majorana index is out of range");
    }
}

void FermionicGaussianState::validate_mode(std::size_t mode) const {
    if (mode >= modes_) {
        throw QStateError("Fermionic mode is out of range");
    }
}

void FermionicGaussianState::rotate_majoranas(
    std::size_t first,
    std::size_t second,
    double theta) {
    validate_majorana(first);
    validate_majorana(second);
    if (first == second || !std::isfinite(theta)) {
        throw QStateError("Fermionic Majorana rotation requires distinct indices and finite angle");
    }
    const std::size_t dimension = 2U * modes_;
    const double c = std::cos(theta);
    const double s = std::sin(theta);
    std::vector<double> first_row(dimension, 0.0);
    std::vector<double> second_row(dimension, 0.0);
    for (std::size_t column = 0U; column < dimension; ++column) {
        const double left = covariance_[first * dimension + column];
        const double right = covariance_[second * dimension + column];
        first_row[column] = c * left - s * right;
        second_row[column] = s * left + c * right;
    }
    for (std::size_t column = 0U; column < dimension; ++column) {
        covariance_[first * dimension + column] = first_row[column];
        covariance_[second * dimension + column] = second_row[column];
    }

    std::vector<double> first_column(dimension, 0.0);
    std::vector<double> second_column(dimension, 0.0);
    for (std::size_t row = 0U; row < dimension; ++row) {
        const double left = covariance_[row * dimension + first];
        const double right = covariance_[row * dimension + second];
        first_column[row] = c * left - s * right;
        second_column[row] = s * left + c * right;
    }
    for (std::size_t row = 0U; row < dimension; ++row) {
        covariance_[row * dimension + first] = first_column[row];
        covariance_[row * dimension + second] = second_column[row];
    }
}

double FermionicGaussianState::occupation(std::size_t mode) const {
    validate_mode(mode);
    const std::size_t dimension = 2U * modes_;
    const std::size_t first = 2U * mode;
    return 0.5 * (1.0 + covariance_[first * dimension + first + 1U]);
}

double FermionicGaussianState::total_occupation() const {
    double total = 0.0;
    for (std::size_t mode = 0U; mode < modes_; ++mode) {
        total += occupation(mode);
    }
    return total;
}

bool FermionicGaussianState::validate(double tolerance) const noexcept {
    if (!(tolerance >= 0.0) || !std::isfinite(tolerance)) {
        return false;
    }
    const std::size_t dimension = 2U * modes_;
    if (covariance_.size() != dimension * dimension) {
        return false;
    }
    for (std::size_t row = 0U; row < dimension; ++row) {
        for (std::size_t column = 0U; column < dimension; ++column) {
            const double value = covariance_[row * dimension + column];
            if (!std::isfinite(value) ||
                std::abs(value + covariance_[column * dimension + row]) > tolerance) {
                return false;
            }
        }
    }
    for (std::size_t row = 0U; row < dimension; ++row) {
        for (std::size_t column = 0U; column < dimension; ++column) {
            double value = row == column ? 1.0 : 0.0;
            for (std::size_t inner = 0U; inner < dimension; ++inner) {
                value += covariance_[row * dimension + inner] *
                         covariance_[inner * dimension + column];
            }
            if (std::abs(value) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace qubit
