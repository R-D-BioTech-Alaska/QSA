#pragma once

#include "qubit/qstate.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace qubit {

struct GaussianConfig {
    std::size_t max_modes{1U << 20U};
    std::size_t max_component_modes{64U};
    std::size_t max_component_scalars{1U << 20U};
    std::size_t max_total_scalars{1U << 24U};
    double max_abs_squeeze{20.0};
};

struct GaussianStats {
    std::size_t modes{0U};
    std::size_t components{0U};
    std::size_t largest_component_modes{0U};
    std::size_t descriptor_scalars{0U};
};

struct GaussianComponent {
    std::vector<std::size_t> modes{};
    std::vector<double> mean{};
    std::vector<double> covariance{};
};

class StructuredGaussianState {
public:
    [[nodiscard]] static StructuredGaussianState vacuum(
        std::size_t mode_count, GaussianConfig config = {}) {
        if (mode_count == 0U || mode_count > config.max_modes) {
            throw QStateError("Gaussian mode count is zero or exceeds configured cap");
        }
        std::vector<double> occupations(mode_count, 0.0);
        return thermal(occupations, config);
    }

    [[nodiscard]] static StructuredGaussianState thermal(
        std::span<const double> mean_occupations, GaussianConfig config = {}) {
        if (mean_occupations.empty() || mean_occupations.size() > config.max_modes) {
            throw QStateError("Gaussian thermal state mode count is zero or exceeds configured cap");
        }
        StructuredGaussianState state(mean_occupations.size(), config);
        state.components_.reserve(mean_occupations.size());
        for (std::size_t mode = 0U; mode < mean_occupations.size(); ++mode) {
            const double occupation = mean_occupations[mode];
            if (!std::isfinite(occupation) || occupation < 0.0) {
                throw QStateError("Gaussian thermal occupation must be finite and nonnegative");
            }
            const double variance = occupation + 0.5;
            state.components_.push_back(GaussianComponent{
                {mode},
                {0.0, 0.0},
                {variance, 0.0, 0.0, variance},
            });
        }
        state.refresh_stats();
        return state;
    }

    [[nodiscard]] std::size_t mode_count() const noexcept { return mode_count_; }
    [[nodiscard]] const GaussianConfig& config() const noexcept { return config_; }
    [[nodiscard]] const GaussianStats& stats() const noexcept { return stats_; }
    [[nodiscard]] const std::vector<GaussianComponent>& components() const noexcept {
        return components_;
    }

    void displace(std::size_t mode, double dq, double dp) {
        if (!std::isfinite(dq) || !std::isfinite(dp)) {
            throw QStateError("Gaussian displacement must be finite");
        }
        auto [component, local] = locate(mode);
        component->mean[2U * local] += dq;
        component->mean[2U * local + 1U] += dp;
        require_finite(*component);
    }

    void rotate(std::size_t mode, double angle) {
        if (!std::isfinite(angle)) {
            throw QStateError("Gaussian phase rotation must be finite");
        }
        const double c = std::cos(angle);
        const double s = std::sin(angle);
        apply_single(mode, {c, -s, s, c});
    }

    void squeeze(std::size_t mode, double parameter) {
        if (!std::isfinite(parameter) || std::abs(parameter) > config_.max_abs_squeeze) {
            throw QStateError("Gaussian squeezing is non-finite or exceeds configured magnitude cap");
        }
        const double q = std::exp(-parameter);
        const double p = std::exp(parameter);
        if (!std::isfinite(q) || !std::isfinite(p)) {
            throw QStateError("Gaussian squeezing produced a non-finite symplectic factor");
        }
        apply_single(mode, {q, 0.0, 0.0, p});
    }

    void two_mode_squeeze(std::size_t first, std::size_t second, double parameter) {
        validate_mode(first);
        validate_mode(second);
        if (first == second) {
            throw QStateError("Gaussian two-mode squeezing requires distinct modes");
        }
        if (!std::isfinite(parameter) || std::abs(parameter) > config_.max_abs_squeeze) {
            throw QStateError(
                "Gaussian two-mode squeezing is non-finite or exceeds configured magnitude cap");
        }
        const double c = std::cosh(parameter);
        const double s = std::sinh(parameter);
        if (!std::isfinite(c) || !std::isfinite(s)) {
            throw QStateError("Gaussian two-mode squeezing produced a non-finite symplectic factor");
        }
        merge_components(first, second);
        auto [component, first_local] = locate(first);
        const std::size_t second_local = local_index(*component, second);
        const std::array<std::size_t, 4> selected{
            2U * first_local, 2U * first_local + 1U,
            2U * second_local, 2U * second_local + 1U,
        };
        const std::array<double, 16> transform{
            c, 0.0, s, 0.0,
            0.0, c, 0.0, -s,
            s, 0.0, c, 0.0,
            0.0, -s, 0.0, c,
        };
        apply_selected(*component, selected, transform);
        require_finite(*component);
        refresh_stats();
    }

    void beam_splitter(std::size_t first, std::size_t second, double transmissivity) {
        validate_mode(first);
        validate_mode(second);
        if (first == second) {
            throw QStateError("Gaussian beam splitter requires distinct modes");
        }
        if (!std::isfinite(transmissivity) || transmissivity < 0.0 || transmissivity > 1.0) {
            throw QStateError("Gaussian beam-splitter transmissivity must lie in [0,1]");
        }
        merge_components(first, second);
        auto [component, first_local] = locate(first);
        const std::size_t second_local = local_index(*component, second);
        const double c = std::sqrt(transmissivity);
        const double s = std::sqrt(1.0 - transmissivity);
        const std::array<std::size_t, 4> selected{
            2U * first_local, 2U * first_local + 1U,
            2U * second_local, 2U * second_local + 1U,
        };
        const std::array<double, 16> transform{
            c, 0.0, s, 0.0,
            0.0, c, 0.0, s,
            -s, 0.0, c, 0.0,
            0.0, -s, 0.0, c,
        };
        apply_selected(*component, selected, transform);
        require_finite(*component);
        refresh_stats();
    }

    void loss(std::size_t mode, double transmissivity, double environment_occupation = 0.0) {
        if (!std::isfinite(transmissivity) || transmissivity < 0.0 || transmissivity > 1.0 ||
            !std::isfinite(environment_occupation) || environment_occupation < 0.0) {
            throw QStateError("Gaussian loss parameters are outside the physical range");
        }
        auto [component, local] = locate(mode);
        const std::size_t dimension = component->mean.size();
        const std::size_t q = 2U * local;
        const std::size_t p = q + 1U;
        const double scale = std::sqrt(transmissivity);
        component->mean[q] *= scale;
        component->mean[p] *= scale;
        for (std::size_t column = 0U; column < dimension; ++column) {
            component->covariance[q * dimension + column] *= scale;
            component->covariance[p * dimension + column] *= scale;
        }
        for (std::size_t row = 0U; row < dimension; ++row) {
            component->covariance[row * dimension + q] *= scale;
            component->covariance[row * dimension + p] *= scale;
        }
        const double noise = (1.0 - transmissivity) * (environment_occupation + 0.5);
        component->covariance[q * dimension + q] += noise;
        component->covariance[p * dimension + p] += noise;
        require_finite(*component);
    }

    [[nodiscard]] std::array<double, 2> mean(std::size_t mode) const {
        const auto [component, local] = locate_const(mode);
        return {component->mean[2U * local], component->mean[2U * local + 1U]};
    }

    [[nodiscard]] std::array<double, 4> covariance_block(
        std::size_t first, std::size_t second) const {
        validate_mode(first);
        validate_mode(second);
        const auto [left_component, left_local] = locate_const(first);
        const auto [right_component, right_local] = locate_const(second);
        if (left_component != right_component) {
            return {0.0, 0.0, 0.0, 0.0};
        }
        const std::size_t dimension = left_component->mean.size();
        const std::size_t a = 2U * left_local;
        const std::size_t b = 2U * right_local;
        return {
            left_component->covariance[a * dimension + b],
            left_component->covariance[a * dimension + b + 1U],
            left_component->covariance[(a + 1U) * dimension + b],
            left_component->covariance[(a + 1U) * dimension + b + 1U],
        };
    }

    [[nodiscard]] double mean_occupation(std::size_t mode) const {
        const auto d = mean(mode);
        const auto v = covariance_block(mode, mode);
        const double result = 0.5 * (v[0] + v[3] + d[0] * d[0] + d[1] * d[1] - 1.0);
        if (!std::isfinite(result)) {
            throw QStateError("Gaussian mean occupation became non-finite");
        }
        return result < 0.0 && std::abs(result) <= 1e-12 ? 0.0 : result;
    }

    [[nodiscard]] double total_mean_occupation() const {
        double result = 0.0;
        for (std::size_t mode = 0U; mode < mode_count_; ++mode) {
            result += mean_occupation(mode);
        }
        return result;
    }

private:
    std::size_t mode_count_{0U};
    GaussianConfig config_{};
    std::vector<GaussianComponent> components_{};
    GaussianStats stats_{};

    StructuredGaussianState(std::size_t mode_count, GaussianConfig config)
        : mode_count_(mode_count), config_(config) {
        if (config_.max_modes == 0U || config_.max_component_modes == 0U ||
            config_.max_component_scalars == 0U || config_.max_total_scalars == 0U ||
            !std::isfinite(config_.max_abs_squeeze) || config_.max_abs_squeeze < 0.0) {
            throw QStateError("Gaussian configuration contains an invalid resource cap");
        }
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

    void validate_mode(std::size_t mode) const {
        if (mode >= mode_count_) {
            throw QStateError("Gaussian mode lies outside logical shape");
        }
    }

    [[nodiscard]] static std::size_t local_index(
        const GaussianComponent& component, std::size_t mode) {
        const auto it = std::find(component.modes.begin(), component.modes.end(), mode);
        if (it == component.modes.end()) {
            throw QStateError("Gaussian component does not contain requested mode");
        }
        return static_cast<std::size_t>(std::distance(component.modes.begin(), it));
    }

    [[nodiscard]] std::pair<GaussianComponent*, std::size_t> locate(std::size_t mode) {
        validate_mode(mode);
        for (GaussianComponent& component : components_) {
            const auto it = std::find(component.modes.begin(), component.modes.end(), mode);
            if (it != component.modes.end()) {
                return {&component,
                        static_cast<std::size_t>(std::distance(component.modes.begin(), it))};
            }
        }
        throw QStateError("Gaussian mode is missing from component partition");
    }

    [[nodiscard]] std::pair<const GaussianComponent*, std::size_t> locate_const(
        std::size_t mode) const {
        validate_mode(mode);
        for (const GaussianComponent& component : components_) {
            const auto it = std::find(component.modes.begin(), component.modes.end(), mode);
            if (it != component.modes.end()) {
                return {&component,
                        static_cast<std::size_t>(std::distance(component.modes.begin(), it))};
            }
        }
        throw QStateError("Gaussian mode is missing from component partition");
    }

    void apply_single(std::size_t mode, const std::array<double, 4>& matrix) {
        auto [component, local] = locate(mode);
        const std::array<std::size_t, 2> selected{2U * local, 2U * local + 1U};
        apply_selected(*component, selected, matrix);
        require_finite(*component);
    }

    static void apply_selected(
        GaussianComponent& component,
        std::span<const std::size_t> selected,
        std::span<const double> matrix) {
        const std::size_t count = selected.size();
        if (count == 0U || matrix.size() != count * count) {
            throw QStateError("Gaussian local transform shape is invalid");
        }
        const std::size_t dimension = component.mean.size();
        const std::vector<double> old_mean = component.mean;
        const std::vector<double> old_covariance = component.covariance;
        for (std::size_t row = 0U; row < count; ++row) {
            double value = 0.0;
            for (std::size_t column = 0U; column < count; ++column) {
                value += matrix[row * count + column] * old_mean[selected[column]];
            }
            component.mean[selected[row]] = value;
        }
        std::vector<double> left = old_covariance;
        for (std::size_t row = 0U; row < count; ++row) {
            for (std::size_t column = 0U; column < dimension; ++column) {
                double value = 0.0;
                for (std::size_t source = 0U; source < count; ++source) {
                    value += matrix[row * count + source] *
                             old_covariance[selected[source] * dimension + column];
                }
                left[selected[row] * dimension + column] = value;
            }
        }
        component.covariance = left;
        for (std::size_t row = 0U; row < dimension; ++row) {
            for (std::size_t column = 0U; column < count; ++column) {
                double value = 0.0;
                for (std::size_t source = 0U; source < count; ++source) {
                    value += left[row * dimension + selected[source]] *
                             matrix[column * count + source];
                }
                component.covariance[row * dimension + selected[column]] = value;
            }
        }
    }

    void merge_components(std::size_t first, std::size_t second) {
        std::size_t a = components_.size();
        std::size_t b = components_.size();
        for (std::size_t index = 0U; index < components_.size(); ++index) {
            if (std::find(components_[index].modes.begin(), components_[index].modes.end(), first) !=
                components_[index].modes.end()) {
                a = index;
            }
            if (std::find(components_[index].modes.begin(), components_[index].modes.end(), second) !=
                components_[index].modes.end()) {
                b = index;
            }
        }
        if (a == components_.size() || b == components_.size()) {
            throw QStateError("Gaussian component partition is incomplete");
        }
        if (a == b) {
            return;
        }
        if (a > b) {
            std::swap(a, b);
        }
        const GaussianComponent& left = components_[a];
        const GaussianComponent& right = components_[b];
        const std::size_t merged_modes = checked_sum(
            left.modes.size(), right.modes.size(), "Gaussian component mode count overflowed");
        if (merged_modes > config_.max_component_modes) {
            throw QStateError("Gaussian mode coupling exceeds configured component-mode cap");
        }
        const std::size_t dimension =
            checked_product(merged_modes, 2U, "Gaussian component dimension overflowed");
        const std::size_t covariance_scalars = checked_product(
            dimension, dimension, "Gaussian component covariance size overflowed");
        const std::size_t component_scalars = checked_sum(
            dimension, covariance_scalars, "Gaussian component scalar count overflowed");
        if (component_scalars > config_.max_component_scalars) {
            throw QStateError("Gaussian mode coupling exceeds configured component scalar cap");
        }

        GaussianComponent merged;
        merged.modes = left.modes;
        merged.modes.insert(merged.modes.end(), right.modes.begin(), right.modes.end());
        merged.mean = left.mean;
        merged.mean.insert(merged.mean.end(), right.mean.begin(), right.mean.end());
        merged.covariance.assign(covariance_scalars, 0.0);
        const std::size_t left_dimension = left.mean.size();
        const std::size_t right_dimension = right.mean.size();
        for (std::size_t row = 0U; row < left_dimension; ++row) {
            for (std::size_t column = 0U; column < left_dimension; ++column) {
                merged.covariance[row * dimension + column] =
                    left.covariance[row * left_dimension + column];
            }
        }
        for (std::size_t row = 0U; row < right_dimension; ++row) {
            for (std::size_t column = 0U; column < right_dimension; ++column) {
                merged.covariance[(left_dimension + row) * dimension + left_dimension + column] =
                    right.covariance[row * right_dimension + column];
            }
        }
        components_[a] = std::move(merged);
        components_.erase(components_.begin() + static_cast<std::ptrdiff_t>(b));
        refresh_stats();
    }

    static void require_finite(const GaussianComponent& component) {
        for (double value : component.mean) {
            if (!std::isfinite(value)) {
                throw QStateError("Gaussian first moment became non-finite");
            }
        }
        for (double value : component.covariance) {
            if (!std::isfinite(value)) {
                throw QStateError("Gaussian covariance became non-finite");
            }
        }
    }

    void refresh_stats() {
        std::size_t largest = 0U;
        std::size_t total = 0U;
        for (const GaussianComponent& component : components_) {
            require_finite(component);
            largest = std::max(largest, component.modes.size());
            const std::size_t component_scalars = checked_sum(
                component.mean.size(), component.covariance.size(),
                "Gaussian component scalar count overflowed");
            total = checked_sum(total, component_scalars, "Gaussian total scalar count overflowed");
            if (component.modes.size() > config_.max_component_modes ||
                component_scalars > config_.max_component_scalars) {
                throw QStateError("Gaussian component exceeds configured resource cap");
            }
        }
        if (total > config_.max_total_scalars) {
            throw QStateError("Gaussian state exceeds configured total scalar cap");
        }
        stats_ = GaussianStats{mode_count_, components_.size(), largest, total};
    }
};

}  // namespace qubit
