#pragma once

#include "qubit/qfactor.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace qubit {

struct ExactFactorAffineConfig {
    std::size_t max_variables{1'000'000U};
    std::size_t max_equations{1'000'000U};
    std::size_t max_row_terms{1'000'000U};
    std::size_t max_basis_terms{16'000'000U};
};

struct ExactFactorAffineStats {
    std::size_t variable_count{0U};
    std::size_t source_factors{0U};
    std::size_t dense_factors{0U};
    std::size_t sparse_factors{0U};
    std::size_t scalar_factors{0U};
    std::size_t parity_factors{0U};
    std::size_t zero_factors{0U};
    std::size_t equation_count{0U};
    std::size_t rank{0U};
    std::size_t free_variables{0U};
    std::size_t basis_terms{0U};
    std::size_t peak_row_terms{0U};
    std::size_t row_reductions{0U};
    std::size_t retained_variables{0U};
    std::size_t output_entries{0U};
    bool inconsistent{false};
};

class ExactFactorAffinePlan {
public:
    ExactFactorAffinePlan(
        const ExactFactorGraph& graph,
        std::span<const FactorVariableId> retained_variables = {},
        ExactFactorAffineConfig config = {})
        : config_(config),
          variable_count_(graph.dimensions_.size()),
          retained_variables_(retained_variables.begin(), retained_variables.end()) {
        std::string reason;
        if (!graph.validate(&reason)) {
            throw QStateError("Cannot compile invalid exact affine factor graph: " + reason);
        }
        if (config_.max_variables == 0U || config_.max_equations == 0U ||
            config_.max_row_terms == 0U || config_.max_basis_terms == 0U) {
            throw QStateError("Exact affine factor configuration contains a zero resource cap");
        }
        if (variable_count_ == 0U) {
            throw QStateError("Exact affine factor plan requires at least one variable");
        }
        if (variable_count_ > config_.max_variables) {
            throw QStateError("Exact affine factor graph exceeds configured variable cap");
        }
        if (retained_variables_.size() > 1U) {
            throw QStateError("Exact affine factor plan supports at most one retained variable");
        }
        if (!retained_variables_.empty() &&
            static_cast<std::size_t>(retained_variables_.front()) >= variable_count_) {
            throw QStateError("Exact affine retained variable is out of range");
        }
        for (const std::size_t dimension : graph.dimensions_) {
            if (dimension != 2U) {
                throw QStateError("Exact affine factor plan requires binary variables");
            }
        }

        const std::size_t missing = std::numeric_limits<std::size_t>::max();
        pivot_to_basis_.assign(variable_count_, missing);
        basis_.reserve(std::min(graph.factors_.size(), variable_count_));
        coefficient_ = {1.0, 0.0};
        stats_.variable_count = variable_count_;
        stats_.source_factors = graph.factors_.size();
        stats_.retained_variables = retained_variables_.size();
        stats_.output_entries = retained_variables_.empty() ? 1U : 2U;

        for (const ExactFactorGraph::Factor& factor : graph.factors_) {
            if (factor.storage == FactorStorageMode::Dense) {
                ++stats_.dense_factors;
            } else {
                ++stats_.sparse_factors;
            }
            if (factor.variables.empty()) {
                ++stats_.scalar_factors;
                const QComplex scalar = factor_value(factor, 0U);
                coefficient_ *= scalar;
                if (zero(scalar)) {
                    zero_result_ = true;
                    ++stats_.zero_factors;
                }
                continue;
            }

            bool parity_rhs = false;
            QComplex factor_coefficient{};
            std::size_t support = 0U;
            if (factor.storage == FactorStorageMode::Dense) {
                for (std::size_t index = 0U; index < factor.logical_entries; ++index) {
                    const QComplex value = factor.dense[index];
                    if (zero(value)) {
                        continue;
                    }
                    const bool parity = (std::popcount(index) & 1U) != 0U;
                    if (support == 0U) {
                        parity_rhs = parity;
                        factor_coefficient = value;
                    } else if (parity != parity_rhs || value != factor_coefficient) {
                        throw QStateError(
                            "Exact affine factor support is not one uniform XOR parity class");
                    }
                    ++support;
                }
            } else {
                for (const FactorSparseEntry& entry : factor.sparse) {
                    if (zero(entry.value)) {
                        continue;
                    }
                    const bool parity = (std::popcount(entry.index) & 1U) != 0U;
                    if (support == 0U) {
                        parity_rhs = parity;
                        factor_coefficient = entry.value;
                    } else if (parity != parity_rhs || entry.value != factor_coefficient) {
                        throw QStateError(
                            "Exact affine factor support is not one uniform XOR parity class");
                    }
                    ++support;
                }
            }

            if (support == 0U) {
                zero_result_ = true;
                ++stats_.zero_factors;
                continue;
            }
            if (support != factor.logical_entries / 2U) {
                throw QStateError(
                    "Exact affine factor support does not cover a complete XOR parity class");
            }
            if (stats_.equation_count >= config_.max_equations) {
                throw QStateError("Exact affine factor plan exceeds configured equation cap");
            }

            coefficient_ *= factor_coefficient;
            ++stats_.parity_factors;
            ++stats_.equation_count;
            Row row;
            row.variables = factor.variables;
            std::sort(row.variables.begin(), row.variables.end());
            row.rhs = parity_rhs;
            reduce_and_insert(std::move(row));
        }

        stats_.rank = basis_.size();
        stats_.free_variables = inconsistent_ ? 0U : variable_count_ - stats_.rank;
        stats_.inconsistent = inconsistent_;

        if (!retained_variables_.empty() && !inconsistent_ && !zero_result_) {
            Row probe;
            probe.variables.push_back(retained_variables_.front());
            reduce(probe);
            retained_fixed_ = probe.variables.empty();
            retained_value_ = probe.rhs;
        }
    }

    [[nodiscard]] std::vector<QComplex> evaluate() const {
        std::vector<QComplex> output(stats_.output_entries);
        evaluate(output);
        return output;
    }

    void evaluate(std::span<QComplex> output) const {
        if (output.size() != stats_.output_entries) {
            throw QStateError("Exact affine factor output size does not match its plan");
        }
        if (inconsistent_ || zero_result_) {
            std::fill(output.begin(), output.end(), QComplex{});
            return;
        }
        if (retained_variables_.empty()) {
            output[0] = scaled_coefficient(stats_.free_variables);
            return;
        }
        if (retained_fixed_) {
            output[0] = {};
            output[1] = {};
            output[retained_value_ ? 1U : 0U] = scaled_coefficient(stats_.free_variables);
            return;
        }
        if (stats_.free_variables == 0U) {
            throw QStateError("Exact affine retained-variable balance invariant failed");
        }
        const QComplex half = scaled_coefficient(stats_.free_variables - 1U);
        output[0] = half;
        output[1] = half;
    }

    [[nodiscard]] QComplex partition() const {
        if (inconsistent_ || zero_result_) {
            return {};
        }
        return scaled_coefficient(stats_.free_variables);
    }

    [[nodiscard]] std::vector<QComplex> normalized_marginal() const {
        std::vector<QComplex> values = evaluate();
        QComplex normalization{};
        for (const QComplex& value : values) {
            normalization += value;
        }
        if (normalization.norm2() <= std::numeric_limits<double>::min()) {
            throw QStateError("Cannot normalize an exact affine factor plan with zero partition");
        }
        for (QComplex& value : values) {
            value /= normalization;
        }
        return values;
    }

    [[nodiscard]] const char* route_name() const noexcept {
        return "ExactFactorAffineXOR";
    }

    [[nodiscard]] const ExactFactorAffineStats& stats() const noexcept {
        return stats_;
    }

    [[nodiscard]] std::span<const FactorVariableId> retained_variables() const noexcept {
        return retained_variables_;
    }

    [[nodiscard]] std::size_t output_entries() const noexcept {
        return stats_.output_entries;
    }

    [[nodiscard]] bool consistent() const noexcept {
        return !inconsistent_;
    }

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        std::size_t bytes = sizeof(*this) +
                            retained_variables_.capacity() * sizeof(FactorVariableId) +
                            pivot_to_basis_.capacity() * sizeof(std::size_t) +
                            basis_.capacity() * sizeof(Row);
        for (const Row& row : basis_) {
            bytes += row.variables.capacity() * sizeof(FactorVariableId);
        }
        return bytes;
    }

private:
    struct Row {
        std::vector<FactorVariableId> variables{};
        bool rhs{false};
    };

    ExactFactorAffineConfig config_{};
    std::size_t variable_count_{0U};
    std::vector<FactorVariableId> retained_variables_{};
    std::vector<std::size_t> pivot_to_basis_{};
    std::vector<Row> basis_{};
    QComplex coefficient_{1.0, 0.0};
    ExactFactorAffineStats stats_{};
    bool inconsistent_{false};
    bool zero_result_{false};
    bool retained_fixed_{false};
    bool retained_value_{false};

    [[nodiscard]] static bool zero(const QComplex& value) noexcept {
        return value.re == 0.0 && value.im == 0.0;
    }

    [[nodiscard]] static QComplex factor_value(
        const ExactFactorGraph::Factor& factor,
        std::size_t index) {
        if (factor.storage == FactorStorageMode::Dense) {
            return factor.dense[index];
        }
        const auto found = std::lower_bound(
            factor.sparse.begin(), factor.sparse.end(), index,
            [](const FactorSparseEntry& entry, std::size_t target) {
                return entry.index < target;
            });
        if (found == factor.sparse.end() || found->index != index) {
            return {};
        }
        return found->value;
    }

    [[nodiscard]] QComplex scaled_coefficient(std::size_t exponent) const noexcept {
        QComplex value = coefficient_;
        if (exponent == 0U || zero(value)) {
            return value;
        }
        if (exponent > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            value.re = value.re == 0.0
                ? 0.0
                : std::copysign(std::numeric_limits<double>::infinity(), value.re);
            value.im = value.im == 0.0
                ? 0.0
                : std::copysign(std::numeric_limits<double>::infinity(), value.im);
            return value;
        }
        value.re = std::ldexp(value.re, static_cast<int>(exponent));
        value.im = std::ldexp(value.im, static_cast<int>(exponent));
        return value;
    }

    [[nodiscard]] std::vector<FactorVariableId> xor_variables(
        std::span<const FactorVariableId> first,
        std::span<const FactorVariableId> second) const {
        std::vector<FactorVariableId> result;
        if (first.size() > config_.max_row_terms || second.size() > config_.max_row_terms) {
            throw QStateError("Exact affine row exceeds configured term cap");
        }
        const std::size_t reserve_terms =
            first.size() > std::numeric_limits<std::size_t>::max() - second.size()
            ? config_.max_row_terms
            : std::min(config_.max_row_terms, first.size() + second.size());
        result.reserve(reserve_terms);
        std::size_t left = 0U;
        std::size_t right = 0U;
        while (left < first.size() || right < second.size()) {
            if (right == second.size() ||
                (left < first.size() && first[left] < second[right])) {
                result.push_back(first[left++]);
            } else if (left == first.size() || second[right] < first[left]) {
                result.push_back(second[right++]);
            } else {
                ++left;
                ++right;
            }
            if (result.size() > config_.max_row_terms) {
                throw QStateError("Exact affine elimination exceeds configured row fill cap");
            }
        }
        return result;
    }

    void reduce(Row& row) const {
        const std::size_t missing = std::numeric_limits<std::size_t>::max();
        while (!row.variables.empty()) {
            const FactorVariableId pivot = row.variables.front();
            const std::size_t basis_index = pivot_to_basis_[pivot];
            if (basis_index == missing) {
                return;
            }
            const Row& basis_row = basis_[basis_index];
            row.variables = xor_variables(row.variables, basis_row.variables);
            row.rhs = row.rhs != basis_row.rhs;
        }
    }

    void reduce_and_insert(Row row) {
        const std::size_t missing = std::numeric_limits<std::size_t>::max();
        while (!row.variables.empty()) {
            stats_.peak_row_terms = std::max(stats_.peak_row_terms, row.variables.size());
            if (row.variables.size() > config_.max_row_terms) {
                throw QStateError("Exact affine equation exceeds configured row term cap");
            }
            const FactorVariableId pivot = row.variables.front();
            const std::size_t basis_index = pivot_to_basis_[pivot];
            if (basis_index == missing) {
                if (row.variables.size() > config_.max_basis_terms ||
                    stats_.basis_terms > config_.max_basis_terms - row.variables.size()) {
                    throw QStateError("Exact affine basis exceeds configured term cap");
                }
                const std::size_t index = basis_.size();
                stats_.basis_terms += row.variables.size();
                basis_.push_back(std::move(row));
                pivot_to_basis_[pivot] = index;
                return;
            }
            const Row& basis_row = basis_[basis_index];
            row.variables = xor_variables(row.variables, basis_row.variables);
            row.rhs = row.rhs != basis_row.rhs;
            ++stats_.row_reductions;
        }
        if (row.rhs) {
            inconsistent_ = true;
        }
    }
};

}  // namespace qubit
