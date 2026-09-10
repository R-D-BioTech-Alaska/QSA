#pragma once

#include "qubit/qqtt_poisson.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <iterator>
#include <map>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace qubit {

struct ComplexWalshTerm {
    QComplex coefficient{};
    std::vector<std::size_t> active_bits{};
};

struct ComplexWalshConfig {
    std::size_t max_terms{1024U};
    std::size_t max_support_entries{1U << 20U};
    std::size_t max_products{1U << 20U};
};

class ExactComplexWalshField {
public:
    [[nodiscard]] static ExactComplexWalshField from_terms(
        std::size_t logical_bits,
        std::vector<ComplexWalshTerm> terms,
        ComplexWalshConfig config = {}) {
        return ExactComplexWalshField(logical_bits, std::move(terms), config);
    }

    [[nodiscard]] std::size_t logical_bits() const noexcept { return logical_bits_; }
    [[nodiscard]] std::size_t term_count() const noexcept { return terms_.size(); }
    [[nodiscard]] std::size_t support_entries() const noexcept { return support_entries_; }
    [[nodiscard]] const std::vector<ComplexWalshTerm>& terms() const noexcept { return terms_; }
    [[nodiscard]] const ComplexWalshConfig& config() const noexcept { return config_; }

    [[nodiscard]] QComplex value_bits(std::span<const std::uint8_t> bits) const {
        validate_bits(bits);
        QComplex value{};
        for (const ComplexWalshTerm& term : terms_) {
            std::uint8_t parity = 0U;
            for (std::size_t bit : term.active_bits) {
                parity ^= bits[bit];
            }
            value += term.coefficient * (parity == 0U ? 1.0 : -1.0);
        }
        return value;
    }

    [[nodiscard]] ExactComplexWalshField scaled(QComplex scalar) const {
        if (!finite(scalar)) {
            throw QStateError("complex Walsh field scale must be finite");
        }
        std::vector<ComplexWalshTerm> next = terms_;
        for (ComplexWalshTerm& term : next) {
            term.coefficient *= scalar;
        }
        return ExactComplexWalshField(logical_bits_, std::move(next), config_);
    }

    [[nodiscard]] ExactComplexWalshField add(const ExactComplexWalshField& other) const {
        require_shape(other, "complex Walsh addition");
        const ComplexWalshConfig output = combined_config(other);
        std::map<std::vector<std::size_t>, QComplex> merged;
        accumulate(merged, terms_);
        accumulate(merged, other.terms_);
        return ExactComplexWalshField(logical_bits_, map_terms(std::move(merged)), output);
    }

    [[nodiscard]] ExactComplexWalshField positive_hypercube_laplacian(
        std::span<const double> weights) const {
        validate_weights(weights);
        std::vector<ComplexWalshTerm> next = terms_;
        for (ComplexWalshTerm& term : next) {
            double eigenvalue = 0.0;
            for (std::size_t bit : term.active_bits) {
                eigenvalue += 2.0 * weights[bit];
            }
            term.coefficient *= eigenvalue;
        }
        return ExactComplexWalshField(logical_bits_, std::move(next), config_);
    }

    [[nodiscard]] ExactComplexWalshField multiply_real(const ExactWalshField& other) const {
        if (logical_bits_ != other.logical_bits()) {
            throw QStateError("complex/real Walsh product requires equal logical shapes");
        }
        const std::size_t products = checked_product(
            terms_.size(), other.terms().size(), "complex/real Walsh product count overflowed");
        if (products > config_.max_products) {
            throw QStateError("complex/real Walsh product exceeds configured product cap");
        }
        std::map<std::vector<std::size_t>, QComplex> merged;
        for (const ComplexWalshTerm& left : terms_) {
            for (const WalshFieldTerm& right : other.terms()) {
                merged[xor_support(left.active_bits, right.active_bits)] +=
                    left.coefficient * right.coefficient;
            }
        }
        return ExactComplexWalshField(logical_bits_, map_terms(std::move(merged)), config_);
    }

    [[nodiscard]] ExactWalshField density(WalshFieldConfig output_config = {}) const {
        const std::size_t products = checked_product(
            terms_.size(), terms_.size(), "complex Walsh density product count overflowed");
        if (products > config_.max_products) {
            throw QStateError("complex Walsh density exceeds configured product cap");
        }
        std::map<std::vector<std::size_t>, double> merged;
        for (const ComplexWalshTerm& left : terms_) {
            for (const ComplexWalshTerm& right : terms_) {
                const QComplex coefficient = left.coefficient.conjugate() * right.coefficient;
                merged[xor_support(left.active_bits, right.active_bits)] += coefficient.re;
            }
        }
        std::vector<WalshFieldTerm> terms;
        terms.reserve(merged.size());
        for (auto& [support, coefficient] : merged) {
            if (!std::isfinite(coefficient)) {
                throw QStateError("complex Walsh density produced a non-finite coefficient");
            }
            if (coefficient != 0.0) {
                terms.push_back(WalshFieldTerm{coefficient, std::move(support)});
            }
        }
        return ExactWalshField::from_terms(logical_bits_, std::move(terms), output_config);
    }

    [[nodiscard]] QComplex mean_inner_product(const ExactComplexWalshField& other) const {
        require_shape(other, "complex Walsh inner product");
        QComplex result{};
        std::size_t left = 0U;
        std::size_t right = 0U;
        while (left < terms_.size() && right < other.terms_.size()) {
            if (terms_[left].active_bits < other.terms_[right].active_bits) {
                ++left;
            } else if (other.terms_[right].active_bits < terms_[left].active_bits) {
                ++right;
            } else {
                result += terms_[left].coefficient.conjugate() * other.terms_[right].coefficient;
                ++left;
                ++right;
            }
        }
        return result;
    }

    [[nodiscard]] ExactQTTFunction to_qtt(QTTConfig qtt_config = {}) const {
        preflight_qtt(qtt_config);
        std::vector<std::array<QComplex, 2>> base(
            logical_bits_, std::array<QComplex, 2>{QComplex{1.0}, QComplex{1.0}});
        if (terms_.empty()) {
            return ExactQTTFunction::product(base, qtt_config).scaled(QComplex{});
        }
        ExactQTTFunction result = term_qtt(terms_.front(), base, qtt_config);
        for (std::size_t i = 1U; i < terms_.size(); ++i) {
            result = result.add(term_qtt(terms_[i], base, qtt_config));
        }
        return result;
    }

private:
    std::size_t logical_bits_{0U};
    std::vector<ComplexWalshTerm> terms_{};
    ComplexWalshConfig config_{};
    std::size_t support_entries_{0U};

    ExactComplexWalshField(
        std::size_t logical_bits,
        std::vector<ComplexWalshTerm> terms,
        ComplexWalshConfig config)
        : logical_bits_(logical_bits), config_(config) {
        if (logical_bits_ == 0U) {
            throw QStateError("complex Walsh field requires at least one binary mode");
        }
        if (config_.max_terms == 0U || config_.max_support_entries == 0U ||
            config_.max_products == 0U) {
            throw QStateError("complex Walsh field configuration contains a zero resource cap");
        }
        if (terms.size() > config_.max_terms) {
            throw QStateError("complex Walsh field exceeds configured input term cap");
        }
        std::map<std::vector<std::size_t>, QComplex> merged;
        std::size_t input_support = 0U;
        for (ComplexWalshTerm& term : terms) {
            if (!finite(term.coefficient)) {
                throw QStateError("complex Walsh field coefficient must be finite");
            }
            std::sort(term.active_bits.begin(), term.active_bits.end());
            if (std::adjacent_find(term.active_bits.begin(), term.active_bits.end()) !=
                term.active_bits.end()) {
                throw QStateError("complex Walsh term contains a duplicate active bit");
            }
            for (std::size_t bit : term.active_bits) {
                if (bit >= logical_bits_) {
                    throw QStateError("complex Walsh active bit lies outside logical shape");
                }
            }
            input_support = checked_sum(
                input_support, term.active_bits.size(), "complex Walsh support count overflowed");
            if (input_support > config_.max_support_entries) {
                throw QStateError("complex Walsh field exceeds configured support-entry cap");
            }
            merged[term.active_bits] += term.coefficient;
        }
        terms_ = map_terms(std::move(merged));
        for (const ComplexWalshTerm& term : terms_) {
            support_entries_ = checked_sum(
                support_entries_, term.active_bits.size(), "complex Walsh support count overflowed");
        }
        if (terms_.size() > config_.max_terms || support_entries_ > config_.max_support_entries) {
            throw QStateError("complex Walsh canonical form exceeds configured resource caps");
        }
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

    [[nodiscard]] static std::vector<std::size_t> xor_support(
        std::span<const std::size_t> left,
        std::span<const std::size_t> right) {
        std::vector<std::size_t> result;
        result.reserve(left.size() + right.size());
        std::set_symmetric_difference(
            left.begin(), left.end(), right.begin(), right.end(), std::back_inserter(result));
        return result;
    }

    static void accumulate(
        std::map<std::vector<std::size_t>, QComplex>& target,
        std::span<const ComplexWalshTerm> terms) {
        for (const ComplexWalshTerm& term : terms) {
            target[term.active_bits] += term.coefficient;
        }
    }

    [[nodiscard]] static std::vector<ComplexWalshTerm> map_terms(
        std::map<std::vector<std::size_t>, QComplex> merged) {
        std::vector<ComplexWalshTerm> result;
        result.reserve(merged.size());
        for (auto& [support, coefficient] : merged) {
            if (!finite(coefficient)) {
                throw QStateError("complex Walsh merged coefficient became non-finite");
            }
            if (coefficient != QComplex{}) {
                result.push_back(ComplexWalshTerm{coefficient, std::move(support)});
            }
        }
        return result;
    }

    [[nodiscard]] ComplexWalshConfig combined_config(
        const ExactComplexWalshField& other) const noexcept {
        return ComplexWalshConfig{
            std::min(config_.max_terms, other.config_.max_terms),
            std::min(config_.max_support_entries, other.config_.max_support_entries),
            std::min(config_.max_products, other.config_.max_products),
        };
    }

    void require_shape(const ExactComplexWalshField& other, const char* label) const {
        if (logical_bits_ != other.logical_bits_) {
            throw QStateError(std::string(label) + " requires equal logical shapes");
        }
    }

    void validate_bits(std::span<const std::uint8_t> bits) const {
        if (bits.size() != logical_bits_) {
            throw QStateError("complex Walsh bit string does not match logical shape");
        }
        for (std::uint8_t bit : bits) {
            if (bit > 1U) {
                throw QStateError("complex Walsh physical bits must be 0 or 1");
            }
        }
    }

    void validate_weights(std::span<const double> weights) const {
        if (weights.size() != logical_bits_) {
            throw QStateError("complex Walsh Laplacian weights do not match logical shape");
        }
        for (double weight : weights) {
            if (!std::isfinite(weight) || weight < 0.0) {
                throw QStateError("complex Walsh Laplacian weights must be finite and nonnegative");
            }
        }
    }

    void preflight_qtt(const QTTConfig& config) const {
        const std::size_t rank = std::max<std::size_t>(1U, terms_.size());
        if (rank > config.max_rank) {
            throw QStateError("complex Walsh-to-QTT conversion exceeds configured rank cap");
        }
        std::size_t total = 0U;
        for (std::size_t i = 0U; i < logical_bits_; ++i) {
            const std::size_t left = i == 0U ? 1U : rank;
            const std::size_t right = i + 1U == logical_bits_ ? 1U : rank;
            const std::size_t matrix = checked_product(
                left, right, "complex Walsh-to-QTT core shape overflowed");
            const std::size_t scalars = checked_product(
                matrix, 2U, "complex Walsh-to-QTT scalar count overflowed");
            if (scalars > config.max_core_scalars) {
                throw QStateError("complex Walsh-to-QTT conversion exceeds configured core scalar cap");
            }
            total = checked_sum(total, scalars, "complex Walsh-to-QTT total scalar count overflowed");
            if (total > config.max_total_scalars) {
                throw QStateError("complex Walsh-to-QTT conversion exceeds configured total scalar cap");
            }
        }
    }

    [[nodiscard]] static ExactQTTFunction term_qtt(
        const ComplexWalshTerm& term,
        const std::vector<std::array<QComplex, 2>>& base,
        const QTTConfig& config) {
        std::vector<std::array<QComplex, 2>> factors = base;
        for (std::size_t bit : term.active_bits) {
            factors[bit][1] = QComplex{-1.0};
        }
        return ExactQTTFunction::product(factors, config).scaled(term.coefficient);
    }
};

struct QTTSchrodingerPoissonStats {
    std::size_t logical_bits{0U};
    std::size_t wave_terms{0U};
    std::size_t density_terms{0U};
    std::size_t potential_terms{0U};
    std::size_t hamiltonian_terms{0U};
    double inverse_poisson_norm_bound{0.0};
    double poisson_condition_bound{0.0};
};

struct QTTSchrodingerPoissonResult {
    ExactWalshField density;
    ExactWalshField source;
    ExactWalshField potential;
    ExactComplexWalshField hamiltonian_wave;
    ExactComplexWalshField rhs;
    QTTSchrodingerPoissonStats stats;
};

class ExactQTTSchrodingerPoisson {
public:
    ExactQTTSchrodingerPoisson(
        std::vector<double> weights,
        double regularizer,
        double field_scale,
        double kinetic_coupling,
        double nonlinear_coupling)
        : weights_(std::move(weights)),
          poisson_(weights_, regularizer, field_scale),
          kinetic_coupling_(kinetic_coupling),
          nonlinear_coupling_(nonlinear_coupling) {
        if (!std::isfinite(kinetic_coupling_) || !std::isfinite(nonlinear_coupling_)) {
            throw QStateError("Schrodinger-Poisson couplings must be finite");
        }
    }

    [[nodiscard]] QTTSchrodingerPoissonResult evaluate(
        const ExactComplexWalshField& wave,
        double background_density = 0.0,
        WalshFieldConfig real_config = {}) const {
        if (wave.logical_bits() != weights_.size()) {
            throw QStateError("Schrodinger-Poisson wave does not match solver shape");
        }
        if (!std::isfinite(background_density)) {
            throw QStateError("Schrodinger-Poisson background density must be finite");
        }
        const ExactWalshField density = wave.density(real_config);
        ExactWalshField source = density;
        if (background_density != 0.0) {
            source = source.add(ExactWalshField::from_terms(
                weights_.size(), {WalshFieldTerm{-background_density, {}}}, real_config));
        }
        const ExactWalshField potential = poisson_.solve(source);
        const ExactComplexWalshField kinetic =
            wave.positive_hypercube_laplacian(weights_).scaled(kinetic_coupling_);
        const ExactComplexWalshField potential_wave = wave.multiply_real(potential);
        const ExactComplexWalshField nonlinear_wave =
            wave.multiply_real(density).scaled(nonlinear_coupling_);
        const ExactComplexWalshField hamiltonian =
            kinetic.add(potential_wave).add(nonlinear_wave);
        const ExactComplexWalshField rhs = hamiltonian.scaled(QComplex{0.0, -1.0});
        const auto poisson_stats = poisson_.stats(source);
        return QTTSchrodingerPoissonResult{
            density,
            source,
            potential,
            hamiltonian,
            rhs,
            QTTSchrodingerPoissonStats{
                weights_.size(),
                wave.term_count(),
                density.term_count(),
                potential.term_count(),
                hamiltonian.term_count(),
                poisson_stats.inverse_norm_bound,
                poisson_stats.condition_bound,
            },
        };
    }

    [[nodiscard]] const ExactQTTRegularizedPoisson& poisson() const noexcept { return poisson_; }
    [[nodiscard]] double kinetic_coupling() const noexcept { return kinetic_coupling_; }
    [[nodiscard]] double nonlinear_coupling() const noexcept { return nonlinear_coupling_; }

private:
    std::vector<double> weights_{};
    ExactQTTRegularizedPoisson poisson_;
    double kinetic_coupling_{0.0};
    double nonlinear_coupling_{0.0};
};

}  // namespace qubit
