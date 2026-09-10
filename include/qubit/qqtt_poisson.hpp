#pragma once

#include "qubit/qqtt.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <span>
#include <utility>
#include <vector>

namespace qubit {

struct WalshFieldTerm {
    double coefficient{0.0};
    std::vector<std::size_t> active_bits{};
};

struct WalshFieldConfig {
    std::size_t max_terms{1024U};
    std::size_t max_support_entries{1U << 20U};
};

class ExactWalshField {
public:
    [[nodiscard]] static ExactWalshField from_terms(
        std::size_t logical_bits,
        std::vector<WalshFieldTerm> terms,
        WalshFieldConfig config = {}) {
        return ExactWalshField(logical_bits, std::move(terms), config);
    }

    [[nodiscard]] std::size_t logical_bits() const noexcept { return logical_bits_; }
    [[nodiscard]] std::size_t term_count() const noexcept { return terms_.size(); }
    [[nodiscard]] std::size_t support_entries() const noexcept { return support_entries_; }
    [[nodiscard]] const std::vector<WalshFieldTerm>& terms() const noexcept { return terms_; }
    [[nodiscard]] const WalshFieldConfig& config() const noexcept { return config_; }

    [[nodiscard]] double value_bits(std::span<const std::uint8_t> bits) const {
        if (bits.size() != logical_bits_) {
            throw QStateError("Walsh field bit string does not match logical shape");
        }
        for (std::uint8_t bit : bits) {
            if (bit > 1U) {
                throw QStateError("Walsh field physical bits must be 0 or 1");
            }
        }
        double value = 0.0;
        for (const WalshFieldTerm& term : terms_) {
            std::uint8_t parity = 0U;
            for (std::size_t bit : term.active_bits) {
                parity ^= bits[bit];
            }
            value += parity == 0U ? term.coefficient : -term.coefficient;
        }
        return value;
    }

    [[nodiscard]] ExactWalshField scaled(double scalar) const {
        if (!std::isfinite(scalar)) {
            throw QStateError("Walsh field scale must be finite");
        }
        std::vector<WalshFieldTerm> next = terms_;
        for (WalshFieldTerm& term : next) {
            term.coefficient *= scalar;
        }
        return ExactWalshField(logical_bits_, std::move(next), config_);
    }

    [[nodiscard]] ExactWalshField add(const ExactWalshField& other) const {
        if (logical_bits_ != other.logical_bits_) {
            throw QStateError("Walsh field addition requires equal logical shapes");
        }
        WalshFieldConfig output{
            std::min(config_.max_terms, other.config_.max_terms),
            std::min(config_.max_support_entries, other.config_.max_support_entries),
        };
        std::map<std::vector<std::size_t>, double> merged;
        for (const WalshFieldTerm& term : terms_) {
            merged[term.active_bits] += term.coefficient;
        }
        for (const WalshFieldTerm& term : other.terms_) {
            merged[term.active_bits] += term.coefficient;
        }
        std::vector<WalshFieldTerm> next;
        next.reserve(merged.size());
        for (auto& [support, coefficient] : merged) {
            if (!std::isfinite(coefficient)) {
                throw QStateError("Walsh field addition produced a non-finite coefficient");
            }
            if (coefficient != 0.0) {
                next.push_back(WalshFieldTerm{coefficient, std::move(support)});
            }
        }
        return ExactWalshField(logical_bits_, std::move(next), output);
    }

    [[nodiscard]] ExactWalshField positive_hypercube_laplacian(
        std::span<const double> weights) const {
        validate_weights(weights);
        std::vector<WalshFieldTerm> next = terms_;
        for (WalshFieldTerm& term : next) {
            double eigenvalue = 0.0;
            for (std::size_t bit : term.active_bits) {
                eigenvalue += 2.0 * weights[bit];
            }
            term.coefficient *= eigenvalue;
        }
        return ExactWalshField(logical_bits_, std::move(next), config_);
    }

    [[nodiscard]] double maximum_absolute_coefficient() const noexcept {
        double maximum = 0.0;
        for (const WalshFieldTerm& term : terms_) {
            maximum = std::max(maximum, std::abs(term.coefficient));
        }
        return maximum;
    }

    [[nodiscard]] ExactQTTFunction to_qtt(QTTConfig qtt_config = {}) const {
        preflight_qtt(qtt_config);
        std::vector<std::array<QComplex, 2>> factors(
            logical_bits_, std::array<QComplex, 2>{QComplex{1.0}, QComplex{1.0}});
        if (terms_.empty()) {
            return ExactQTTFunction::product(factors, qtt_config).scaled(QComplex{});
        }

        ExactQTTFunction result = term_qtt(terms_.front(), factors, qtt_config);
        for (std::size_t i = 1U; i < terms_.size(); ++i) {
            result = result.add(term_qtt(terms_[i], factors, qtt_config));
        }
        return result;
    }

private:
    std::size_t logical_bits_{0U};
    std::vector<WalshFieldTerm> terms_{};
    WalshFieldConfig config_{};
    std::size_t support_entries_{0U};

    ExactWalshField(
        std::size_t logical_bits,
        std::vector<WalshFieldTerm> terms,
        WalshFieldConfig config)
        : logical_bits_(logical_bits), config_(config) {
        if (logical_bits_ == 0U) {
            throw QStateError("Walsh field requires at least one binary mode");
        }
        if (config_.max_terms == 0U || config_.max_support_entries == 0U) {
            throw QStateError("Walsh field configuration contains a zero resource cap");
        }
        if (terms.size() > config_.max_terms) {
            throw QStateError("Walsh field exceeds configured input term cap");
        }

        std::map<std::vector<std::size_t>, double> merged;
        std::size_t input_support = 0U;
        for (WalshFieldTerm& term : terms) {
            if (!std::isfinite(term.coefficient)) {
                throw QStateError("Walsh field coefficient must be finite");
            }
            std::sort(term.active_bits.begin(), term.active_bits.end());
            if (std::adjacent_find(term.active_bits.begin(), term.active_bits.end()) !=
                term.active_bits.end()) {
                throw QStateError("Walsh field term contains a duplicate active bit");
            }
            for (std::size_t bit : term.active_bits) {
                if (bit >= logical_bits_) {
                    throw QStateError("Walsh field active bit lies outside logical shape");
                }
            }
            input_support = checked_sum(
                input_support, term.active_bits.size(), "Walsh field support count overflowed");
            if (input_support > config_.max_support_entries) {
                throw QStateError("Walsh field exceeds configured support-entry cap");
            }
            merged[term.active_bits] += term.coefficient;
        }

        terms_.reserve(merged.size());
        for (auto& [support, coefficient] : merged) {
            if (!std::isfinite(coefficient)) {
                throw QStateError("Walsh field merged coefficient became non-finite");
            }
            if (coefficient != 0.0) {
                support_entries_ = checked_sum(
                    support_entries_, support.size(), "Walsh field support count overflowed");
                terms_.push_back(WalshFieldTerm{coefficient, std::move(support)});
            }
        }
        if (terms_.size() > config_.max_terms || support_entries_ > config_.max_support_entries) {
            throw QStateError("Walsh field canonical form exceeds configured resource caps");
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

    void validate_weights(std::span<const double> weights) const {
        if (weights.size() != logical_bits_) {
            throw QStateError("Walsh field Laplacian weights do not match logical shape");
        }
        for (double weight : weights) {
            if (!std::isfinite(weight) || weight < 0.0) {
                throw QStateError("Walsh field Laplacian weights must be finite and nonnegative");
            }
        }
    }

    void preflight_qtt(const QTTConfig& config) const {
        const std::size_t rank = std::max<std::size_t>(1U, terms_.size());
        if (rank > config.max_rank) {
            throw QStateError("Walsh-to-QTT conversion exceeds configured rank cap");
        }
        std::size_t total = 0U;
        for (std::size_t i = 0U; i < logical_bits_; ++i) {
            const std::size_t left = i == 0U ? 1U : rank;
            const std::size_t right = i + 1U == logical_bits_ ? 1U : rank;
            const std::size_t matrix = checked_product(
                left, right, "Walsh-to-QTT core shape overflowed");
            const std::size_t scalars = checked_product(
                matrix, 2U, "Walsh-to-QTT scalar count overflowed");
            if (scalars > config.max_core_scalars) {
                throw QStateError("Walsh-to-QTT conversion exceeds configured core scalar cap");
            }
            total = checked_sum(total, scalars, "Walsh-to-QTT total scalar count overflowed");
            if (total > config.max_total_scalars) {
                throw QStateError("Walsh-to-QTT conversion exceeds configured total scalar cap");
            }
        }
    }

    [[nodiscard]] static ExactQTTFunction term_qtt(
        const WalshFieldTerm& term,
        const std::vector<std::array<QComplex, 2>>& base,
        const QTTConfig& config) {
        std::vector<std::array<QComplex, 2>> factors = base;
        for (std::size_t bit : term.active_bits) {
            factors[bit][1] = QComplex{-1.0};
        }
        return ExactQTTFunction::product(factors, config).scaled(QComplex{term.coefficient});
    }
};

struct QTTRegularizedPoissonStats {
    std::size_t logical_bits{0U};
    std::size_t source_terms{0U};
    std::size_t source_support_entries{0U};
    double minimum_eigenvalue{0.0};
    double maximum_eigenvalue{0.0};
    double inverse_norm_bound{0.0};
    double condition_bound{0.0};
};

class ExactQTTRegularizedPoisson {
public:
    ExactQTTRegularizedPoisson(
        std::vector<double> weights,
        double regularizer,
        double source_scale = 1.0)
        : weights_(std::move(weights)), regularizer_(regularizer), source_scale_(source_scale) {
        if (weights_.empty()) {
            throw QStateError("QTT regularized Poisson solver requires at least one binary mode");
        }
        if (!std::isfinite(regularizer_) || regularizer_ <= 0.0 ||
            !std::isfinite(source_scale_)) {
            throw QStateError("QTT regularized Poisson solver requires finite mu > 0 and finite source scale");
        }
        double sum = 0.0;
        for (double weight : weights_) {
            if (!std::isfinite(weight) || weight < 0.0) {
                throw QStateError("QTT regularized Poisson weights must be finite and nonnegative");
            }
            sum += weight;
            if (!std::isfinite(sum)) {
                throw QStateError("QTT regularized Poisson weight sum became non-finite");
            }
        }
        maximum_eigenvalue_ = regularizer_ + 2.0 * sum;
    }

    [[nodiscard]] ExactWalshField solve(const ExactWalshField& source) const {
        require_shape(source);
        std::vector<WalshFieldTerm> potential = source.terms();
        for (WalshFieldTerm& term : potential) {
            double eigenvalue = regularizer_;
            for (std::size_t bit : term.active_bits) {
                eigenvalue += 2.0 * weights_[bit];
            }
            term.coefficient *= source_scale_ / eigenvalue;
        }
        return ExactWalshField::from_terms(
            source.logical_bits(), std::move(potential), source.config());
    }

    [[nodiscard]] ExactWalshField residual(
        const ExactWalshField& source,
        const ExactWalshField& potential) const {
        require_shape(source);
        require_shape(potential);
        return potential.scaled(regularizer_)
            .add(potential.positive_hypercube_laplacian(weights_))
            .add(source.scaled(-source_scale_));
    }

    [[nodiscard]] QTTRegularizedPoissonStats stats(const ExactWalshField& source) const {
        require_shape(source);
        return QTTRegularizedPoissonStats{
            weights_.size(),
            source.term_count(),
            source.support_entries(),
            regularizer_,
            maximum_eigenvalue_,
            1.0 / regularizer_,
            maximum_eigenvalue_ / regularizer_,
        };
    }

    [[nodiscard]] std::size_t logical_bits() const noexcept { return weights_.size(); }
    [[nodiscard]] double regularizer() const noexcept { return regularizer_; }
    [[nodiscard]] double source_scale() const noexcept { return source_scale_; }
    [[nodiscard]] const std::vector<double>& weights() const noexcept { return weights_; }

private:
    std::vector<double> weights_{};
    double regularizer_{0.0};
    double source_scale_{1.0};
    double maximum_eigenvalue_{0.0};

    void require_shape(const ExactWalshField& field) const {
        if (field.logical_bits() != weights_.size()) {
            throw QStateError("QTT regularized Poisson field does not match solver shape");
        }
    }
};

}  // namespace qubit
