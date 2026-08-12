#pragma once

#include "qubit/qqtt_sp_invariants.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <map>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace qubit {

namespace qtt_split_detail {

[[nodiscard]] inline std::size_t checked_product(
    std::size_t left, std::size_t right, const char* message) {
    if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) {
        throw QStateError(message);
    }
    return left * right;
}

[[nodiscard]] inline std::size_t checked_sum(
    std::size_t left, std::size_t right, const char* message) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        throw QStateError(message);
    }
    return left + right;
}

[[nodiscard]] inline std::vector<std::size_t> xor_support(
    std::span<const std::size_t> left,
    std::span<const std::size_t> right) {
    std::vector<std::size_t> result;
    result.reserve(left.size() + right.size());
    std::set_symmetric_difference(
        left.begin(), left.end(), right.begin(), right.end(), std::back_inserter(result));
    return result;
}

[[nodiscard]] inline ComplexWalshConfig combined_config(
    const ExactComplexWalshField& left,
    const ExactComplexWalshField& right) noexcept {
    return ComplexWalshConfig{
        std::min(left.config().max_terms, right.config().max_terms),
        std::min(left.config().max_support_entries, right.config().max_support_entries),
        std::min(left.config().max_products, right.config().max_products),
    };
}

}  // namespace qtt_split_detail

[[nodiscard]] inline ExactComplexWalshField multiply_complex_walsh(
    const ExactComplexWalshField& left,
    const ExactComplexWalshField& right) {
    if (left.logical_bits() != right.logical_bits()) {
        throw QStateError("complex Walsh product requires equal logical shapes");
    }
    const ComplexWalshConfig config = qtt_split_detail::combined_config(left, right);
    const std::size_t products = qtt_split_detail::checked_product(
        left.term_count(), right.term_count(), "complex Walsh product count overflowed");
    if (products > config.max_products) {
        throw QStateError("complex Walsh product exceeds configured product cap");
    }

    std::map<std::vector<std::size_t>, QComplex> merged;
    for (const ComplexWalshTerm& a : left.terms()) {
        for (const ComplexWalshTerm& b : right.terms()) {
            merged[qtt_split_detail::xor_support(a.active_bits, b.active_bits)] +=
                a.coefficient * b.coefficient;
        }
    }
    if (merged.size() > config.max_terms) {
        throw QStateError("complex Walsh product exceeds configured canonical term cap");
    }
    std::vector<ComplexWalshTerm> terms;
    terms.reserve(merged.size());
    for (auto& [support, coefficient] : merged) {
        if (!std::isfinite(coefficient.re) || !std::isfinite(coefficient.im)) {
            throw QStateError("complex Walsh product produced a non-finite coefficient");
        }
        if (coefficient != QComplex{}) {
            terms.push_back(ComplexWalshTerm{coefficient, std::move(support)});
        }
    }
    return ExactComplexWalshField::from_terms(left.logical_bits(), std::move(terms), config);
}

[[nodiscard]] inline ExactComplexWalshField real_walsh_phase(
    const ExactWalshField& field,
    double phase_scale,
    ComplexWalshConfig config = {}) {
    if (!std::isfinite(phase_scale)) {
        throw QStateError("Walsh phase scale must be finite");
    }
    std::map<std::vector<std::size_t>, QComplex> current;
    current[{}] = QComplex{1.0};
    std::size_t operations = 0U;
    for (const WalshFieldTerm& term : field.terms()) {
        const std::size_t stage = qtt_split_detail::checked_product(
            current.size(), 2U, "Walsh phase expansion count overflowed");
        operations = qtt_split_detail::checked_sum(
            operations, stage, "Walsh phase expansion work overflowed");
        if (operations > config.max_products) {
            throw QStateError("Walsh phase expansion exceeds configured product cap");
        }
        const double angle = phase_scale * term.coefficient;
        if (!std::isfinite(angle)) {
            throw QStateError("Walsh phase angle became non-finite");
        }
        const QComplex diagonal{std::cos(angle)};
        const QComplex parity{0.0, std::sin(angle)};
        std::map<std::vector<std::size_t>, QComplex> next;
        for (const auto& [support, coefficient] : current) {
            next[support] += coefficient * diagonal;
            next[qtt_split_detail::xor_support(support, term.active_bits)] +=
                coefficient * parity;
        }
        if (next.size() > config.max_terms) {
            throw QStateError("Walsh phase expansion exceeds configured term cap");
        }
        current = std::move(next);
    }

    std::vector<ComplexWalshTerm> terms;
    terms.reserve(current.size());
    for (auto& [support, coefficient] : current) {
        if (!std::isfinite(coefficient.re) || !std::isfinite(coefficient.im)) {
            throw QStateError("Walsh phase expansion produced a non-finite coefficient");
        }
        if (coefficient != QComplex{}) {
            terms.push_back(ComplexWalshTerm{coefficient, std::move(support)});
        }
    }
    return ExactComplexWalshField::from_terms(field.logical_bits(), std::move(terms), config);
}

[[nodiscard]] inline ExactComplexWalshField kinetic_walsh_phase(
    const ExactComplexWalshField& wave,
    std::span<const double> weights,
    double phase_scale) {
    if (weights.size() != wave.logical_bits() || !std::isfinite(phase_scale)) {
        throw QStateError("kinetic Walsh phase requires matching shape and finite scale");
    }
    for (double weight : weights) {
        if (!std::isfinite(weight) || weight < 0.0) {
            throw QStateError("kinetic Walsh weights must be finite and nonnegative");
        }
    }
    std::vector<ComplexWalshTerm> terms = wave.terms();
    for (ComplexWalshTerm& term : terms) {
        double eigenvalue = 0.0;
        for (std::size_t bit : term.active_bits) {
            eigenvalue += 2.0 * weights[bit];
        }
        const double angle = phase_scale * eigenvalue;
        if (!std::isfinite(angle)) {
            throw QStateError("kinetic Walsh phase angle became non-finite");
        }
        term.coefficient *= QComplex::from_polar(1.0, angle);
    }
    return ExactComplexWalshField::from_terms(
        wave.logical_bits(), std::move(terms), wave.config());
}

struct QTTSplitStepStats {
    std::size_t logical_bits{0U};
    std::size_t input_terms{0U};
    std::size_t density_terms{0U};
    std::size_t potential_terms{0U};
    std::size_t phase_terms{0U};
    std::size_t output_terms{0U};
    double input_mean_norm_squared{0.0};
    double output_mean_norm_squared{0.0};
    double norm_error{0.0};
};

struct QTTSplitStepResult {
    ExactComplexWalshField wave;
    ExactWalshField density;
    ExactWalshField potential;
    ExactComplexWalshField local_phase;
    QTTSplitStepStats stats;
};

class ExactQTTSchrodingerPoissonSplitStep {
public:
    ExactQTTSchrodingerPoissonSplitStep(
        std::vector<double> weights,
        double regularizer,
        double field_scale,
        double kinetic_coupling,
        double nonlinear_coupling,
        ComplexWalshConfig phase_config = {})
        : weights_(std::move(weights)),
          poisson_(weights_, regularizer, field_scale),
          kinetic_coupling_(kinetic_coupling),
          nonlinear_coupling_(nonlinear_coupling),
          phase_config_(phase_config) {
        if (!std::isfinite(kinetic_coupling_) || !std::isfinite(nonlinear_coupling_)) {
            throw QStateError("split-step couplings must be finite");
        }
        if (phase_config_.max_terms == 0U || phase_config_.max_support_entries == 0U ||
            phase_config_.max_products == 0U) {
            throw QStateError("split-step phase configuration contains a zero resource cap");
        }
    }

    [[nodiscard]] QTTSplitStepResult step(
        const ExactComplexWalshField& wave,
        double dt,
        double background_density = 0.0,
        WalshFieldConfig real_config = {}) const {
        if (wave.logical_bits() != weights_.size()) {
            throw QStateError("split-step wave does not match solver shape");
        }
        if (!std::isfinite(dt) || !std::isfinite(background_density)) {
            throw QStateError("split-step dt and background density must be finite");
        }

        const double input_norm = wave.mean_inner_product(wave).re;
        const ExactComplexWalshField half_kinetic = kinetic_walsh_phase(
            wave, weights_, -0.5 * dt * kinetic_coupling_);
        const ExactWalshField density = half_kinetic.density(real_config);
        ExactWalshField source = density;
        if (background_density != 0.0) {
            source = source.add(ExactWalshField::from_terms(
                weights_.size(), {WalshFieldTerm{-background_density, {}}}, real_config));
        }
        const ExactWalshField potential = poisson_.solve(source);
        const ExactWalshField local_field =
            potential.add(density.scaled(nonlinear_coupling_));
        const ExactComplexWalshField local_phase =
            real_walsh_phase(local_field, -dt, phase_config_);
        const ExactComplexWalshField after_local =
            multiply_complex_walsh(half_kinetic, local_phase);
        const ExactComplexWalshField output = kinetic_walsh_phase(
            after_local, weights_, -0.5 * dt * kinetic_coupling_);
        const double output_norm = output.mean_inner_product(output).re;
        if (!std::isfinite(input_norm) || !std::isfinite(output_norm)) {
            throw QStateError("split-step norm diagnostic became non-finite");
        }

        return QTTSplitStepResult{
            output,
            density,
            potential,
            local_phase,
            QTTSplitStepStats{
                weights_.size(),
                wave.term_count(),
                density.term_count(),
                potential.term_count(),
                local_phase.term_count(),
                output.term_count(),
                input_norm,
                output_norm,
                std::abs(output_norm - input_norm),
            },
        };
    }

    [[nodiscard]] const ExactQTTRegularizedPoisson& poisson() const noexcept { return poisson_; }
    [[nodiscard]] double kinetic_coupling() const noexcept { return kinetic_coupling_; }
    [[nodiscard]] double nonlinear_coupling() const noexcept { return nonlinear_coupling_; }
    [[nodiscard]] const ComplexWalshConfig& phase_config() const noexcept { return phase_config_; }

private:
    std::vector<double> weights_{};
    ExactQTTRegularizedPoisson poisson_;
    double kinetic_coupling_{0.0};
    double nonlinear_coupling_{0.0};
    ComplexWalshConfig phase_config_{};
};

}  // namespace qubit
