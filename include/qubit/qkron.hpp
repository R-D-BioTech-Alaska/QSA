#pragma once

#include "qubit/qcomplex.hpp"
#include "qubit/qstate.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace qubit {

class KronOperator;

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
        validate_dimensions();
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
        validate_index(index);
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
        const double scale_value = std::max(1.0, std::abs(value.re));
        if (std::abs(value.im) >
            64.0 * std::numeric_limits<double>::epsilon() * scale_value) {
            throw QStateError("KronVector norm accumulated a non-real value");
        }
        if (value.re < 0.0 &&
            std::abs(value.re) >
                64.0 * std::numeric_limits<double>::epsilon() * scale_value) {
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
        validate_square_matrix(matrix, dimension, "KronVector local operator");

        std::vector<QComplex> transformed(dimension);
        for (KronTerm& term : terms_) {
            transform_factor(matrix, term.factors[site], transformed);
            term.factors[site] = transformed;
        }
    }

    [[nodiscard]] KronVector tensor_product(const KronVector& other) const {
        std::vector<std::size_t> result_dimensions = dimensions_;
        result_dimensions.insert(
            result_dimensions.end(), other.dimensions_.begin(), other.dimensions_.end());
        const std::size_t result_limit = std::min(max_terms_, other.max_terms_);
        KronVector result(std::move(result_dimensions), result_limit);
        if (terms_.empty() || other.terms_.empty()) {
            return result;
        }
        if (terms_.size() > std::numeric_limits<std::size_t>::max() / other.terms_.size()) {
            throw QStateError("KronVector tensor-product rank overflowed");
        }
        const std::size_t result_terms = terms_.size() * other.terms_.size();
        if (result_terms > result_limit) {
            throw QStateError("KronVector tensor product exceeds max_terms");
        }

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
            decode_index(flat, index);
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

    static void validate_square_matrix(
        std::span<const QComplex> matrix,
        std::size_t dimension,
        const char* context) {
        if (dimension > std::numeric_limits<std::size_t>::max() / dimension ||
            matrix.size() != dimension * dimension) {
            throw QStateError(std::string(context) + " shape is invalid");
        }
        for (const QComplex value : matrix) {
            if (!finite(value)) {
                throw QStateError(std::string(context) + " must be finite");
            }
        }
    }

    static void transform_factor(
        std::span<const QComplex> matrix,
        std::span<const QComplex> input,
        std::span<QComplex> output) {
        const std::size_t dimension = input.size();
        for (std::size_t row = 0; row < dimension; ++row) {
            QComplex sum{};
            for (std::size_t column = 0; column < dimension; ++column) {
                sum += matrix[row * dimension + column] * input[column];
            }
            output[row] = sum;
        }
    }

    void validate_dimensions() const {
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

    void validate_index(std::span<const std::size_t> index) const {
        if (index.size() != dimensions_.size()) {
            throw QStateError("KronVector element index rank does not match factor count");
        }
        for (std::size_t site = 0; site < index.size(); ++site) {
            if (index[site] >= dimensions_[site]) {
                throw QStateError("KronVector element index is out of range");
            }
        }
    }

    void decode_index(std::size_t flat, std::span<std::size_t> index) const noexcept {
        for (std::size_t site = dimensions_.size(); site-- > 0U;) {
            index[site] = flat % dimensions_[site];
            flat /= dimensions_[site];
        }
    }

    friend class KronOperator;
};

struct KronOperatorTerm {
    QComplex coefficient{1.0, 0.0};
    std::vector<std::vector<QComplex>> factors{};
};

class KronOperator {
public:
    explicit KronOperator(
        std::vector<std::size_t> dimensions,
        std::size_t max_terms = 4096U)
        : dimensions_(std::move(dimensions)), max_terms_(max_terms) {
        if (dimensions_.empty()) {
            throw QStateError("KronOperator requires at least one factor dimension");
        }
        if (max_terms_ == 0U) {
            throw QStateError("KronOperator max_terms must be positive");
        }
        for (const std::size_t dimension : dimensions_) {
            if (dimension == 0U) {
                throw QStateError("KronOperator factor dimensions must be positive");
            }
        }
    }

    void add_term(
        QComplex coefficient,
        std::span<const std::vector<QComplex>> factors) {
        if (!finite(coefficient)) {
            throw QStateError("KronOperator coefficient must be finite");
        }
        if (coefficient.norm2() == 0.0) {
            return;
        }
        validate_factors(factors);
        if (terms_.size() >= max_terms_) {
            throw QStateError("KronOperator term count exceeds max_terms");
        }
        KronOperatorTerm term;
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

    [[nodiscard]] KronVector apply(const KronVector& input) const {
        validate_vector(input);
        const std::size_t output_limit = std::min(max_terms_, input.max_terms_);
        KronVector output(dimensions_, output_limit);
        if (terms_.empty() || input.terms_.empty()) {
            return output;
        }
        if (terms_.size() > std::numeric_limits<std::size_t>::max() / input.terms_.size()) {
            throw QStateError("KronOperator application rank overflowed");
        }
        const std::size_t output_terms = terms_.size() * input.terms_.size();
        if (output_terms > output_limit) {
            throw QStateError("KronOperator application exceeds output max_terms");
        }

        output.terms_.reserve(output_terms);
        for (const KronOperatorTerm& operation : terms_) {
            for (const KronTerm& source : input.terms_) {
                KronTerm result;
                result.coefficient = operation.coefficient * source.coefficient;
                result.factors.resize(dimensions_.size());
                for (std::size_t site = 0; site < dimensions_.size(); ++site) {
                    result.factors[site].resize(dimensions_[site]);
                    transform_factor(
                        operation.factors[site],
                        source.factors[site],
                        result.factors[site]);
                }
                output.terms_.push_back(std::move(result));
            }
        }
        return output;
    }

    [[nodiscard]] QComplex matrix_element(
        const KronVector& bra,
        const KronVector& ket) const {
        validate_vector(bra);
        validate_vector(ket);
        QComplex total{};
        for (const KronTerm& left : bra.terms_) {
            for (const KronOperatorTerm& operation : terms_) {
                for (const KronTerm& right : ket.terms_) {
                    QComplex value = left.coefficient.conjugate() *
                                     operation.coefficient * right.coefficient;
                    for (std::size_t site = 0; site < dimensions_.size(); ++site) {
                        const std::size_t dimension = dimensions_[site];
                        QComplex local{};
                        for (std::size_t row = 0; row < dimension; ++row) {
                            const QComplex bra_value = left.factors[site][row].conjugate();
                            for (std::size_t column = 0; column < dimension; ++column) {
                                local += bra_value *
                                         operation.factors[site][row * dimension + column] *
                                         right.factors[site][column];
                            }
                        }
                        value *= local;
                        if (value.norm2() == 0.0) {
                            break;
                        }
                    }
                    total += value;
                }
            }
        }
        return total;
    }

    [[nodiscard]] QComplex expectation(const KronVector& state) const {
        return matrix_element(state, state);
    }

    void scale(QComplex coefficient) {
        if (!finite(coefficient)) {
            throw QStateError("KronOperator scale must be finite");
        }
        if (coefficient.norm2() == 0.0) {
            terms_.clear();
            return;
        }
        for (KronOperatorTerm& term : terms_) {
            term.coefficient *= coefficient;
        }
    }

    void add(const KronOperator& other) {
        validate_compatible(other);
        if (other.terms_.size() > max_terms_ - std::min(max_terms_, terms_.size())) {
            throw QStateError("KronOperator addition exceeds max_terms");
        }
        terms_.insert(terms_.end(), other.terms_.begin(), other.terms_.end());
    }

    [[nodiscard]] KronOperator tensor_product(const KronOperator& other) const {
        std::vector<std::size_t> result_dimensions = dimensions_;
        result_dimensions.insert(
            result_dimensions.end(), other.dimensions_.begin(), other.dimensions_.end());
        const std::size_t result_limit = std::min(max_terms_, other.max_terms_);
        KronOperator result(std::move(result_dimensions), result_limit);
        if (terms_.empty() || other.terms_.empty()) {
            return result;
        }
        if (terms_.size() > std::numeric_limits<std::size_t>::max() / other.terms_.size()) {
            throw QStateError("KronOperator tensor-product rank overflowed");
        }
        const std::size_t result_terms = terms_.size() * other.terms_.size();
        if (result_terms > result_limit) {
            throw QStateError("KronOperator tensor product exceeds max_terms");
        }
        result.terms_.reserve(result_terms);
        for (const KronOperatorTerm& left : terms_) {
            for (const KronOperatorTerm& right : other.terms_) {
                KronOperatorTerm term;
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
        if (size > 0U && size > std::numeric_limits<std::size_t>::max() / size) {
            throw QStateError("KronOperator dense matrix size exceeds size_t");
        }
        const std::size_t entries = size * size;
        if (entries > max_entries) {
            throw QStateError("KronOperator materialization exceeds max_entries");
        }
        std::vector<QComplex> output(entries);
        std::vector<std::size_t> row_index(dimensions_.size(), 0U);
        std::vector<std::size_t> column_index(dimensions_.size(), 0U);
        for (std::size_t row = 0; row < size; ++row) {
            decode_index(row, row_index);
            for (std::size_t column = 0; column < size; ++column) {
                decode_index(column, column_index);
                QComplex value{};
                for (const KronOperatorTerm& term : terms_) {
                    QComplex product = term.coefficient;
                    for (std::size_t site = 0; site < dimensions_.size(); ++site) {
                        product *= term.factors[site][
                            row_index[site] * dimensions_[site] + column_index[site]];
                    }
                    value += product;
                }
                output[row * size + column] = value;
            }
        }
        return output;
    }

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        std::size_t bytes = sizeof(*this) +
                            dimensions_.capacity() * sizeof(std::size_t) +
                            terms_.capacity() * sizeof(KronOperatorTerm);
        for (const KronOperatorTerm& term : terms_) {
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
    std::vector<KronOperatorTerm> terms_{};

    [[nodiscard]] static bool finite(QComplex value) noexcept {
        return std::isfinite(value.re) && std::isfinite(value.im);
    }

    [[nodiscard]] std::size_t logical_size() const {
        std::size_t size = 1U;
        for (const std::size_t dimension : dimensions_) {
            if (size > std::numeric_limits<std::size_t>::max() / dimension) {
                throw QStateError("KronOperator logical size exceeds size_t");
            }
            size *= dimension;
        }
        return size;
    }

    static void transform_factor(
        std::span<const QComplex> matrix,
        std::span<const QComplex> input,
        std::span<QComplex> output) {
        const std::size_t dimension = input.size();
        for (std::size_t row = 0; row < dimension; ++row) {
            QComplex sum{};
            for (std::size_t column = 0; column < dimension; ++column) {
                sum += matrix[row * dimension + column] * input[column];
            }
            output[row] = sum;
        }
    }

    void validate_factors(std::span<const std::vector<QComplex>> factors) const {
        if (factors.size() != dimensions_.size()) {
            throw QStateError("KronOperator term factor count does not match dimensions");
        }
        for (std::size_t site = 0; site < factors.size(); ++site) {
            const std::size_t dimension = dimensions_[site];
            if (dimension > std::numeric_limits<std::size_t>::max() / dimension ||
                factors[site].size() != dimension * dimension) {
                throw QStateError("KronOperator term factor shape does not match dimensions");
            }
            for (const QComplex value : factors[site]) {
                if (!finite(value)) {
                    throw QStateError("KronOperator term factor must be finite");
                }
            }
        }
    }

    void validate_vector(const KronVector& vector) const {
        if (dimensions_ != vector.dimensions_) {
            throw QStateError("KronOperator and KronVector dimensions do not match");
        }
    }

    void validate_compatible(const KronOperator& other) const {
        if (dimensions_ != other.dimensions_) {
            throw QStateError("KronOperator dimensions do not match");
        }
    }

    void decode_index(std::size_t flat, std::span<std::size_t> index) const noexcept {
        for (std::size_t site = dimensions_.size(); site-- > 0U;) {
            index[site] = flat % dimensions_[site];
            flat /= dimensions_[site];
        }
    }
};

}  // namespace qubit