#include "qubit/qgaussian.hpp"

#include <algorithm>
#include <array>
#include <cmath>

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

}  // namespace qubit
