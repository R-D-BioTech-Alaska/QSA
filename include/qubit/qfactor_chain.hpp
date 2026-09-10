#pragma once

#include "qubit/qfactor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace qubit {

struct ExactFactorChainStats {
    std::size_t variable_count{0U};
    std::size_t factor_count{0U};
    std::size_t factor_width{0U};
    std::size_t retained_variables{0U};
    std::size_t source_dense_factors{0U};
    std::size_t source_sparse_factors{0U};
    std::size_t transfer_steps{0U};
    std::size_t max_separator_entries{0U};
    std::size_t output_entries{0U};
};

class ExactFactorChainWorkspace {
public:
    ExactFactorChainWorkspace() = default;

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        return sizeof(*this) +
               first_.capacity() * sizeof(QComplex) +
               second_.capacity() * sizeof(QComplex);
    }

private:
    std::vector<QComplex> first_{};
    std::vector<QComplex> second_{};

    friend class ExactFactorChainPlan;
};

class ExactFactorChainPlan {
public:
    ExactFactorChainPlan(
        const ExactFactorGraph& graph,
        std::span<const FactorVariableId> retained_variables)
        : config_(graph.config_),
          dimensions_(graph.dimensions_),
          retained_variables_(retained_variables.begin(), retained_variables.end()) {
        std::string reason;
        if (!graph.validate(&reason)) {
            throw QStateError("Cannot compile invalid exact factor chain: " + reason);
        }
        if (graph.factors_.empty()) {
            throw QStateError("Exact factor chain requires at least one factor");
        }

        factor_width_ = graph.factors_.front().variables.size();
        if (factor_width_ < 2U) {
            throw QStateError("Exact factor chain requires factor width of at least two");
        }
        if (graph.factors_.size() >
            std::numeric_limits<std::size_t>::max() - (factor_width_ - 1U)) {
            throw QStateError("Exact factor chain topology size overflowed");
        }
        if (graph.factors_.size() + factor_width_ - 1U != dimensions_.size()) {
            throw QStateError("Exact factor chain factor count does not span every variable");
        }
        if (retained_variables_.size() != factor_width_ - 1U) {
            throw QStateError("Exact factor chain must retain the final separator");
        }

        const std::size_t factor_count = graph.factors_.size();
        for (std::size_t position = 0U; position < retained_variables_.size(); ++position) {
            const std::size_t expected = factor_count + position;
            if (expected > std::numeric_limits<FactorVariableId>::max() ||
                retained_variables_[position] != static_cast<FactorVariableId>(expected)) {
                throw QStateError("Exact factor chain retained variables are not the final separator");
            }
        }

        separator_entries_.resize(factor_count + 1U);
        for (std::size_t step = 0U; step <= factor_count; ++step) {
            separator_entries_[step] = separator_entries(step);
            stats_.max_separator_entries =
                std::max(stats_.max_separator_entries, separator_entries_[step]);
        }

        factors_.reserve(factor_count);
        for (std::size_t index = 0U; index < factor_count; ++index) {
            const ExactFactorGraph::Factor& source = graph.factors_[index];
            if (source.variables.size() != factor_width_) {
                throw QStateError("Exact factor chain factor width changed");
            }
            for (std::size_t position = 0U; position < factor_width_; ++position) {
                const std::size_t expected = index + position;
                if (expected > std::numeric_limits<FactorVariableId>::max() ||
                    source.variables[position] != static_cast<FactorVariableId>(expected)) {
                    throw QStateError("Exact factor chain contains a noncontiguous or permuted scope");
                }
            }
            const std::size_t new_dimension = dimensions_[index + factor_width_ - 1U];
            if (separator_entries_[index] >
                std::numeric_limits<std::size_t>::max() / new_dimension ||
                separator_entries_[index] * new_dimension != source.logical_entries) {
                throw QStateError("Exact factor chain table shape is inconsistent");
            }

            SourceFactor factor;
            factor.logical_entries = source.logical_entries;
            factor.storage = source.storage;
            factor.dense = source.dense;
            factor.sparse = source.sparse;
            factors_.push_back(std::move(factor));
            if (source.storage == FactorStorageMode::Dense) {
                ++stats_.source_dense_factors;
            } else {
                ++stats_.source_sparse_factors;
            }
        }

        stats_.variable_count = dimensions_.size();
        stats_.factor_count = factors_.size();
        stats_.factor_width = factor_width_;
        stats_.retained_variables = retained_variables_.size();
        stats_.transfer_steps = factors_.size();
        stats_.output_entries = separator_entries_.back();
    }

    [[nodiscard]] ExactFactorChainWorkspace workspace() const {
        ExactFactorChainWorkspace result;
        result.first_.resize(stats_.max_separator_entries);
        result.second_.resize(stats_.max_separator_entries);
        return result;
    }

    [[nodiscard]] std::vector<QComplex> evaluate() const {
        ExactFactorChainWorkspace local = workspace();
        return evaluate(local);
    }

    [[nodiscard]] std::vector<QComplex> evaluate(
        ExactFactorChainWorkspace& workspace_value) const {
        std::vector<QComplex> result(stats_.output_entries);
        evaluate(result, workspace_value);
        return result;
    }

    void evaluate(
        std::span<QComplex> output,
        ExactFactorChainWorkspace& workspace_value) const {
        if (output.size() != stats_.output_entries) {
            throw QStateError("Exact factor chain output size does not match its plan");
        }
        validate_workspace(workspace_value);

        std::vector<QComplex>* current = &workspace_value.first_;
        std::vector<QComplex>* next = &workspace_value.second_;
        const std::size_t initial_entries = separator_entries_.front();
        std::fill_n(current->begin(), initial_entries, QComplex{1.0, 0.0});

        for (std::size_t step = 0U; step < factors_.size(); ++step) {
            const SourceFactor& factor = factors_[step];
            const std::size_t old_entries = separator_entries_[step];
            const std::size_t next_entries = separator_entries_[step + 1U];
            const std::size_t first_dimension = dimensions_[step];
            const std::size_t tail_entries = old_entries / first_dimension;
            const std::size_t new_dimension = dimensions_[step + factor_width_ - 1U];
            if (tail_entries > std::numeric_limits<std::size_t>::max() / new_dimension ||
                tail_entries * new_dimension != next_entries) {
                throw QStateError("Exact factor chain separator shape changed");
            }
            std::fill_n(next->begin(), next_entries, QComplex{});

            if (factor.storage == FactorStorageMode::Dense) {
                for (std::size_t old_index = 0U; old_index < old_entries; ++old_index) {
                    const QComplex prefix = (*current)[old_index];
                    if (prefix.re == 0.0 && prefix.im == 0.0) {
                        continue;
                    }
                    const std::size_t tail_index = old_index / first_dimension;
                    for (std::size_t new_value = 0U;
                         new_value < new_dimension;
                         ++new_value) {
                        const QComplex weight =
                            factor.dense[old_index + new_value * old_entries];
                        if (weight.re == 0.0 && weight.im == 0.0) {
                            continue;
                        }
                        (*next)[tail_index + new_value * tail_entries] += prefix * weight;
                    }
                }
            } else {
                for (const FactorSparseEntry& entry : factor.sparse) {
                    const std::size_t old_index = entry.index % old_entries;
                    const std::size_t new_value = entry.index / old_entries;
                    if (new_value >= new_dimension) {
                        throw QStateError("Exact factor chain sparse index is inconsistent");
                    }
                    const QComplex prefix = (*current)[old_index];
                    if (prefix.re == 0.0 && prefix.im == 0.0) {
                        continue;
                    }
                    const std::size_t tail_index = old_index / first_dimension;
                    (*next)[tail_index + new_value * tail_entries] += prefix * entry.value;
                }
            }
            std::swap(current, next);
        }

        std::copy_n(current->begin(), stats_.output_entries, output.begin());
    }

    [[nodiscard]] QComplex partition() const {
        ExactFactorChainWorkspace local = workspace();
        return partition(local);
    }

    [[nodiscard]] QComplex partition(ExactFactorChainWorkspace& workspace_value) const {
        const std::vector<QComplex> values = evaluate(workspace_value);
        QComplex result{};
        for (const QComplex& value : values) {
            result += value;
        }
        return result;
    }

    [[nodiscard]] std::vector<QComplex> normalized_marginal() const {
        ExactFactorChainWorkspace local = workspace();
        return normalized_marginal(local);
    }

    [[nodiscard]] std::vector<QComplex> normalized_marginal(
        ExactFactorChainWorkspace& workspace_value) const {
        std::vector<QComplex> values = evaluate(workspace_value);
        QComplex normalization{};
        for (const QComplex& value : values) {
            normalization += value;
        }
        if (normalization.norm2() <= std::numeric_limits<double>::min()) {
            throw QStateError("Cannot normalize an exact factor chain with zero partition");
        }
        for (QComplex& value : values) {
            value /= normalization;
        }
        return values;
    }

    void rebind_dense_factor(FactorId factor_id, std::span<const QComplex> values) {
        SourceFactor& factor = mutable_factor(factor_id);
        if (values.size() != factor.logical_entries) {
            throw QStateError("Exact factor chain dense rebind size changed");
        }
        std::vector<QComplex> replacement(values.begin(), values.end());
        for (const QComplex& value : replacement) {
            if (!finite(value)) {
                throw QStateError("Exact factor chain dense rebind contains non-finite values");
            }
        }
        const FactorStorageMode previous = factor.storage;
        factor.storage = FactorStorageMode::Dense;
        factor.dense = std::move(replacement);
        factor.sparse.clear();
        if (previous == FactorStorageMode::Sparse) {
            --stats_.source_sparse_factors;
            ++stats_.source_dense_factors;
        }
        ++rebind_count_;
    }

    void rebind_sparse_factor(
        FactorId factor_id,
        std::span<const FactorSparseEntry> entries) {
        SourceFactor& factor = mutable_factor(factor_id);
        std::vector<FactorSparseEntry> replacement(entries.begin(), entries.end());
        for (const FactorSparseEntry& entry : replacement) {
            if (entry.index >= factor.logical_entries) {
                throw QStateError("Exact factor chain sparse rebind index is out of range");
            }
            if (!finite(entry.value)) {
                throw QStateError("Exact factor chain sparse rebind contains non-finite values");
            }
        }
        std::sort(
            replacement.begin(), replacement.end(),
            [](const FactorSparseEntry& first, const FactorSparseEntry& second) {
                return first.index < second.index;
            });
        for (std::size_t index = 1U; index < replacement.size(); ++index) {
            if (replacement[index - 1U].index == replacement[index].index) {
                throw QStateError("Exact factor chain sparse rebind repeats an index");
            }
        }
        const FactorStorageMode previous = factor.storage;
        factor.storage = FactorStorageMode::Sparse;
        factor.sparse = std::move(replacement);
        factor.dense.clear();
        if (previous == FactorStorageMode::Dense) {
            --stats_.source_dense_factors;
            ++stats_.source_sparse_factors;
        }
        ++rebind_count_;
    }

    [[nodiscard]] const char* route_name() const noexcept {
        return "ExactFactorChainTransfer";
    }

    [[nodiscard]] std::span<const FactorVariableId> retained_variables() const noexcept {
        return retained_variables_;
    }

    [[nodiscard]] const ExactFactorChainStats& stats() const noexcept {
        return stats_;
    }

    [[nodiscard]] std::size_t rebind_count() const noexcept {
        return rebind_count_;
    }

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        std::size_t bytes = sizeof(*this) +
                            dimensions_.capacity() * sizeof(std::size_t) +
                            retained_variables_.capacity() * sizeof(FactorVariableId) +
                            separator_entries_.capacity() * sizeof(std::size_t) +
                            factors_.capacity() * sizeof(SourceFactor);
        for (const SourceFactor& factor : factors_) {
            bytes += factor.dense.capacity() * sizeof(QComplex) +
                     factor.sparse.capacity() * sizeof(FactorSparseEntry);
        }
        return bytes;
    }

private:
    struct SourceFactor {
        std::size_t logical_entries{0U};
        FactorStorageMode storage{FactorStorageMode::Dense};
        std::vector<QComplex> dense{};
        std::vector<FactorSparseEntry> sparse{};
    };

    ExactFactorConfig config_{};
    std::vector<std::size_t> dimensions_{};
    std::vector<FactorVariableId> retained_variables_{};
    std::vector<std::size_t> separator_entries_{};
    std::vector<SourceFactor> factors_{};
    ExactFactorChainStats stats_{};
    std::size_t factor_width_{0U};
    std::size_t rebind_count_{0U};

    [[nodiscard]] static bool finite(const QComplex& value) noexcept {
        return std::isfinite(value.re) && std::isfinite(value.im);
    }

    [[nodiscard]] std::size_t separator_entries(std::size_t start) const {
        std::size_t entries = 1U;
        for (std::size_t offset = 0U; offset + 1U < factor_width_; ++offset) {
            const std::size_t dimension = dimensions_[start + offset];
            if (entries > std::numeric_limits<std::size_t>::max() / dimension) {
                throw QStateError("Exact factor chain separator size overflowed");
            }
            entries *= dimension;
        }
        if (entries > config_.max_factor_entries) {
            throw QStateError("Exact factor chain separator exceeds max_factor_entries");
        }
        return entries;
    }

    [[nodiscard]] SourceFactor& mutable_factor(FactorId factor_id) {
        const std::size_t index = static_cast<std::size_t>(factor_id);
        if (index >= factors_.size()) {
            throw QStateError("Exact factor chain factor id is out of range");
        }
        return factors_[index];
    }

    void validate_workspace(const ExactFactorChainWorkspace& workspace_value) const {
        if (workspace_value.first_.size() != stats_.max_separator_entries ||
            workspace_value.second_.size() != stats_.max_separator_entries) {
            throw QStateError("Exact factor chain workspace does not match its plan");
        }
    }
};

}  // namespace qubit
