#pragma once

#include "qubit/qcomplex.hpp"
#include "qubit/qstate.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qubit {

struct KronTerm {
    QComplex coefficient{1.0, 0.0};
    std::vector<std::vector<QComplex>> factors{};
};

class KronVector {
public:
    explicit KronVector(
        std::vector<std::size_t> dimensions,
        std::size_t max_terms = 4096U)
        : dimensions_(std::move(dimensions)), max_terms_(max_terms) {
        if (dimensions_.empty()) {
            throw QStateError("KronVector requires at least one factor dimension");
        }
        if (max_terms_ == 0U) {
            throw QStateError("KronVector max_terms must be positive");
        }
        for (const std::size_t dimension : dimensions_) {
            if (dimension == 0U) {
                throw QStateError("KronVector factor dimensions must be positive");
            }
        }
    }

    void add_term(
        QComplex coefficient,
        std::span<const std::vector<QComplex>> factors) {
        if (!finite(coefficient)) {
            throw QStateError("KronVector coefficient must be finite");
        }
        if (coefficient.norm2() == 0.0) {
            return;
        }
        validate_factors(factors);
        if (terms_.size() >= max_terms_) {
            throw QStateError("KronVector term count exceeds max_terms");
        }
        KronTerm term;
        term.coefficient = coefficient;
        term.factors.assign(factors.begin(), factors.end());
        terms_.push_back(std::move(term));
    }

    [[nodiscard]] std::size_t factor_count() const noexcept {
        return dimensions_.size();
    }

    [[nodiscard]] std::size_t term_count() const noexcept {
        return terms_.size();
    }

    [[nodiscard]] std::size_t max_terms() const noexcept {
        return max_terms_;
    }

    [[nodiscard]] const std::vector<std::size_t>& dimensions() const noexcept {
        return dimensions_;
    }

    [[nodiscard]] bool logical_size_fits() const noexcept {
        std::size_t size = 1U;
        for (const std::size_t dimension : dimensions_) {
            if (size > std::numeric_limits<std::size_t>::max() / dimension) {
                return false;
            }
            size *= dimension;
        }
        return true;
    }

    [[nodiscard]] std::size_t logical_size() const {
        std::size_t size = 1U;
        for (const std::size_t dimension : dimensions_) {
            if (size > std::numeric_limits<std::size_t>::max() / dimension) {
                throw QStateError("KronVector logical size exceeds size_t");
            }
            size *= dimension;
        }
        return size;
    }

    [[nodiscard]] long double log2_logical_size() const noexcept {
        long double bits = 0.0L;
        for (const std::size_t dimension : dimensions_) {
            bits += std::log2(static_cast<long double>(dimension));
        }
        return bits;
    }

    [[nodiscard]] QComplex element(std::span<const std::size_t> index) const {
        if (index.size() != dimensions_.size()) {
            throw QStateError("KronVector element index rank does not match factor count");
        }
        for (std::size_t site = 0; site < index.size(); ++site) {
            if (index[site] >= dimensions_[site]) {
                throw QStateError("KronVector element index is out of range");
            }
        }
        QComplex result{};
        for (const KronTerm& term : terms_) {
            QComplex value = term.coefficient;
            for (std::size_t site = 0; site < index.size(); ++site) {
                value *= term.factors[site][index[site]];
            }
            result += value;
        }
        return result;
    }

    [[nodiscard]] QComplex inner_product(const KronVector& other) const {
        validate_compatible(other);
        QComplex total{};
        for (const KronTerm& left : terms_) {
            for (const KronTerm& right : other.terms_) {
                QComplex value = left.coefficient.conjugate() * right.coefficient;
                for (std::size_t site = 0; site < dimensions_.size(); ++site) {
                    QComplex local{};
                    for (std::size_t index = 0; index < dimensions_[site]; ++index) {
                        local += left.factors[site][index].conjugate() *
                                 right.factors[site][index];
                    }
                    value *= local;
                    if (value.norm2() == 0.0) {
                        break;
                    }
                }
                total += value;
            }
        }
        return total;
    }

    [[nodiscard]] double norm2() const {
        const QComplex value = inner_product(*this);
        const double scale = std::max(1.0, std::abs(value.re));
        if (std::abs(value.im) > 64.0 * std::numeric_limits<double>::epsilon() * scale) {
            throw QStateError("KronVector norm accumulated a non-real value");
        }
        if (value.re < 0.0 &&
            std::abs(value.re) > 64.0 * std::numeric_limits<double>::epsilon() * scale) {
            throw QStateError("KronVector norm accumulated a negative value");
        }
        return std::max(0.0, value.re);
    }

    void scale(QComplex coefficient) {
        if (!finite(coefficient)) {
            throw QStateError("KronVector scale must be finite");
        }
        if (coefficient.norm2() == 0.0) {
            terms_.clear();
            return;
        }
        for (KronTerm& term : terms_) {
            term.coefficient *= coefficient;
        }
    }

    void add(const KronVector& other) {
        validate_compatible(other);
        if (other.terms_.size() > max_terms_ - std::min(max_terms_, terms_.size())) {
            throw QStateError("KronVector addition exceeds max_terms");
        }
        terms_.insert(terms_.end(), other.terms_.begin(), other.terms_.end());
    }

    void apply_local_operator(
        std::size_t site,
        std::span<const QComplex> matrix) {
        if (site >= dimensions_.size()) {
            throw QStateError("KronVector local operator site is out of range");
        }
        const std::size_t dimension = dimensions_[site];
        if (dimension > std::numeric_limits<std::size_t>::max() / dimension ||
            matrix.size() != dimension * dimension) {
            throw QStateError("KronVector local operator shape is invalid");
        }
        for (const QComplex value : matrix) {
            if (!finite(value)) {
                throw QStateError("KronVector local operator must be finite");
            }
        }

        std::vector<QComplex> transformed(dimension);
        for (KronTerm& term : terms_) {
            const std::vector<QComplex>& input = term.factors[site];
            for (std::size_t row = 0; row < dimension; ++row) {
                QComplex sum{};
                for (std::size_t column = 0; column < dimension; ++column) {
                    sum += matrix[row * dimension + column] * input[column];
                }
                transformed[row] = sum;
            }
            term.factors[site] = transformed;
        }
    }

    [[nodiscard]] KronVector tensor_product(const KronVector& other) const {
        if (terms_.empty() || other.terms_.empty()) {
            std::vector<std::size_t> dimensions = dimensions_;
            dimensions.insert(dimensions.end(), other.dimensions_.begin(), other.dimensions_.end());
            return KronVector(std::move(dimensions), std::min(max_terms_, other.max_terms_));
        }
        if (terms_.size() > std::numeric_limits<std::size_t>::max() / other.terms_.size()) {
            throw QStateError("KronVector tensor-product rank overflowed");
        }
        const std::size_t result_terms = terms_.size() * other.terms_.size();
        const std::size_t result_limit = std::min(max_terms_, other.max_terms_);
        if (result_terms > result_limit) {
            throw QStateError("KronVector tensor product exceeds max_terms");
        }

        std::vector<std::size_t> dimensions = dimensions_;
        dimensions.insert(dimensions.end(), other.dimensions_.begin(), other.dimensions_.end());
        KronVector result(std::move(dimensions), result_limit);
        result.terms_.reserve(result_terms);
        for (const KronTerm& left : terms_) {
            for (const KronTerm& right : other.terms_) {
                KronTerm term;
                term.coefficient = left.coefficient * right.coefficient;
                term.factors = left.factors;
                term.factors.insert(
                    term.factors.end(), right.factors.begin(), right.factors.end());
                result.terms_.push_back(std::move(term));
            }
        }
        return result;
    }

    [[nodiscard]] std::vector<QComplex> materialize(
        std::size_t max_entries = 1U << 20U) const {
        const std::size_t size = logical_size();
        if (size > max_entries) {
            throw QStateError("KronVector materialization exceeds max_entries");
        }
        std::vector<QComplex> output(size);
        std::vector<std::size_t> index(dimensions_.size(), 0U);
        for (std::size_t flat = 0; flat < size; ++flat) {
            std::size_t remainder = flat;
            for (std::size_t site = dimensions_.size(); site-- > 0U;) {
                index[site] = remainder % dimensions_[site];
                remainder /= dimensions_[site];
            }
            output[flat] = element(index);
        }
        return output;
    }

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        std::size_t bytes = sizeof(*this) +
                            dimensions_.capacity() * sizeof(std::size_t) +
                            terms_.capacity() * sizeof(KronTerm);
        for (const KronTerm& term : terms_) {
            bytes += term.factors.capacity() * sizeof(std::vector<QComplex>);
            for (const auto& factor : term.factors) {
                bytes += factor.capacity() * sizeof(QComplex);
            }
        }
        return bytes;
    }

private:
    std::vector<std::size_t> dimensions_{};
    std::size_t max_terms_{4096U};
    std::vector<KronTerm> terms_{};

    [[nodiscard]] static bool finite(QComplex value) noexcept {
        return std::isfinite(value.re) && std::isfinite(value.im);
    }

    void validate_factors(std::span<const std::vector<QComplex>> factors) const {
        if (factors.size() != dimensions_.size()) {
            throw QStateError("KronVector term factor count does not match dimensions");
        }
        for (std::size_t site = 0; site < factors.size(); ++site) {
            if (factors[site].size() != dimensions_[site]) {
                throw QStateError("KronVector term factor dimension does not match");
            }
            for (const QComplex value : factors[site]) {
                if (!finite(value)) {
                    throw QStateError("KronVector term factor must be finite");
                }
            }
        }
    }

    void validate_compatible(const KronVector& other) const {
        if (dimensions_ != other.dimensions_) {
            throw QStateError("KronVector dimensions do not match");
        }
    }
};

}  // namespace qubit