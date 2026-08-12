#pragma once

#include "qubit/qstate.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace qubit {

using FactorVariableId = std::uint32_t;
using FactorId = std::uint32_t;

enum class FactorStorageMode : std::uint8_t {
    Dense = 0,
    Sparse = 1,
};

enum class ExactFactorRoute : std::uint8_t {
    VariableElimination = 0,
};

struct FactorSparseEntry {
    std::size_t index{0U};
    QComplex value{};
};

struct ExactFactorConfig {
    std::size_t max_factor_entries{1U << 20U};
    std::size_t max_factors{1'000'000U};
    std::size_t max_variables{1'000'000U};
    std::size_t max_compiled_index_entries{1U << 20U};
    bool reuse_workspace_slots{true};
};

struct ExactFactorStats {
    std::size_t variable_count{0U};
    std::size_t source_factors{0U};
    std::size_t source_dense_factors{0U};
    std::size_t source_sparse_factors{0U};
    std::size_t eliminated_variables{0U};
    std::size_t retained_variables{0U};
    std::size_t peak_union_variables{0U};
    std::size_t peak_factor_entries{0U};
    std::size_t output_entries{0U};
    std::size_t compiled_index_entries{0U};
    std::size_t workspace_slots{0U};
};

class ExactFactorPlan;
class ExactFactorWorkspace;

class ExactFactorGraph {
public:
    explicit ExactFactorGraph(ExactFactorConfig config = {});

    [[nodiscard]] FactorVariableId add_variable(std::size_t dimension);
    [[nodiscard]] FactorId add_dense_factor(
        std::span<const FactorVariableId> variables,
        std::span<const QComplex> values);
    [[nodiscard]] FactorId add_sparse_factor(
        std::span<const FactorVariableId> variables,
        std::span<const FactorSparseEntry> entries);

    void set_dense_factor(FactorId factor, std::span<const QComplex> values);
    void set_sparse_factor(FactorId factor, std::span<const FactorSparseEntry> entries);

    [[nodiscard]] ExactFactorPlan compile(
        std::span<const FactorVariableId> retained_variables = {}) const;
    [[nodiscard]] QComplex partition() const;
    [[nodiscard]] std::vector<QComplex> marginal(
        std::span<const FactorVariableId> retained_variables) const;
    [[nodiscard]] std::vector<QComplex> normalized_marginal(
        std::span<const FactorVariableId> retained_variables) const;

    [[nodiscard]] std::size_t variable_count() const noexcept { return dimensions_.size(); }
    [[nodiscard]] std::size_t factor_count() const noexcept { return factors_.size(); }
    [[nodiscard]] std::size_t dimension(FactorVariableId variable) const;
    [[nodiscard]] FactorStorageMode factor_storage(FactorId factor) const;
    [[nodiscard]] const ExactFactorConfig& config() const noexcept { return config_; }
    [[nodiscard]] std::size_t estimated_bytes() const noexcept;
    [[nodiscard]] bool validate(std::string* reason = nullptr) const noexcept;

private:
    struct Factor {
        std::vector<FactorVariableId> variables{};
        std::size_t logical_entries{1U};
        FactorStorageMode storage{FactorStorageMode::Dense};
        std::vector<QComplex> dense{};
        std::vector<FactorSparseEntry> sparse{};
    };

    ExactFactorConfig config_{};
    std::vector<std::size_t> dimensions_{};
    std::vector<Factor> factors_{};

    [[nodiscard]] std::size_t factor_entries(
        std::span<const FactorVariableId> variables) const;
    void validate_factor_scope(std::span<const FactorVariableId> variables) const;
    [[nodiscard]] Factor& mutable_factor(FactorId factor);
    [[nodiscard]] const Factor& factor(FactorId factor) const;

    friend class ExactFactorPlan;
};

class ExactFactorWorkspace {
public:
    ExactFactorWorkspace() = default;

    [[nodiscard]] std::size_t estimated_bytes() const noexcept;

private:
    std::vector<std::vector<QComplex>> outputs_{};
    std::vector<std::size_t> coordinates_{};
    std::vector<std::size_t> retained_coordinates_{};

    friend class ExactFactorPlan;
};

class ExactFactorPlan {
public:
    ExactFactorPlan(
        const ExactFactorGraph& graph,
        std::span<const FactorVariableId> retained_variables = {});

    [[nodiscard]] ExactFactorWorkspace workspace() const;
    [[nodiscard]] std::vector<QComplex> evaluate() const;
    [[nodiscard]] std::vector<QComplex> evaluate(ExactFactorWorkspace& workspace) const;
    void evaluate(std::span<QComplex> output, ExactFactorWorkspace& workspace) const;
    [[nodiscard]] QComplex partition() const;
    [[nodiscard]] QComplex partition(ExactFactorWorkspace& workspace) const;
    [[nodiscard]] std::vector<QComplex> normalized_marginal() const;
    [[nodiscard]] std::vector<QComplex> normalized_marginal(
        ExactFactorWorkspace& workspace) const;

    void rebind(const ExactFactorGraph& graph);

    void rebind_dense_factor(FactorId factor, std::span<const QComplex> values) {
        const std::size_t index = static_cast<std::size_t>(factor);
        if (index >= graph_factor_count_) {
            throw QStateError("Exact factor targeted rebind factor is out of range");
        }
        SourceFactor& source = sources_[index];
        if (values.size() != source.logical_entries) {
            throw QStateError("Exact factor targeted dense rebind size changed");
        }
        std::vector<QComplex> replacement(values.begin(), values.end());
        if (!std::all_of(
                replacement.begin(), replacement.end(),
                [](const QComplex& value) {
                    return std::isfinite(value.re) && std::isfinite(value.im);
                })) {
            throw QStateError("Exact factor targeted dense rebind contains non-finite values");
        }

        const FactorStorageMode previous = source.storage;
        source.storage = FactorStorageMode::Dense;
        source.dense = std::move(replacement);
        source.sparse = {};
        if (previous == FactorStorageMode::Sparse) {
            --stats_.source_sparse_factors;
            ++stats_.source_dense_factors;
        }
        ++rebind_count_;
    }

    void rebind_sparse_factor(
        FactorId factor,
        std::span<const FactorSparseEntry> entries) {
        const std::size_t index = static_cast<std::size_t>(factor);
        if (index >= graph_factor_count_) {
            throw QStateError("Exact factor targeted rebind factor is out of range");
        }
        SourceFactor& source = sources_[index];
        std::vector<FactorSparseEntry> replacement(entries.begin(), entries.end());
        for (const FactorSparseEntry& entry : replacement) {
            if (entry.index >= source.logical_entries) {
                throw QStateError("Exact factor targeted sparse rebind index is out of range");
            }
            if (!std::isfinite(entry.value.re) || !std::isfinite(entry.value.im)) {
                throw QStateError("Exact factor targeted sparse rebind contains non-finite values");
            }
        }
        std::sort(
            replacement.begin(), replacement.end(),
            [](const FactorSparseEntry& lhs, const FactorSparseEntry& rhs) {
                return lhs.index < rhs.index;
            });
        for (std::size_t entry = 1U; entry < replacement.size(); ++entry) {
            if (replacement[entry - 1U].index == replacement[entry].index) {
                throw QStateError("Exact factor targeted sparse rebind repeats an index");
            }
        }

        const FactorStorageMode previous = source.storage;
        source.storage = FactorStorageMode::Sparse;
        source.sparse = std::move(replacement);
        source.dense = {};
        if (previous == FactorStorageMode::Dense) {
            --stats_.source_dense_factors;
            ++stats_.source_sparse_factors;
        }
        ++rebind_count_;
    }

    [[nodiscard]] ExactFactorRoute route() const noexcept {
        return ExactFactorRoute::VariableElimination;
    }
    [[nodiscard]] const ExactFactorStats& stats() const noexcept { return stats_; }
    [[nodiscard]] std::span<const FactorVariableId> retained_variables() const noexcept {
        return retained_variables_;
    }
    [[nodiscard]] std::size_t output_entries() const noexcept { return stats_.output_entries; }
    [[nodiscard]] std::size_t step_count() const noexcept { return steps_.size(); }
    [[nodiscard]] std::size_t rebind_count() const noexcept { return rebind_count_; }
    [[nodiscard]] std::size_t estimated_bytes() const noexcept;

private:
    struct SourceFactor {
        std::vector<FactorVariableId> variables{};
        std::vector<std::size_t> strides{};
        std::size_t logical_entries{1U};
        FactorStorageMode storage{FactorStorageMode::Dense};
        std::vector<QComplex> dense{};
        std::vector<FactorSparseEntry> sparse{};
        bool synthetic{false};
    };

    struct InputMap {
        std::size_t node{0U};
        std::vector<std::size_t> positions{};
        std::vector<std::size_t> strides{};
    };

    struct Step {
        FactorVariableId eliminated{0U};
        std::vector<FactorVariableId> union_variables{};
        std::vector<FactorVariableId> output_variables{};
        std::vector<InputMap> inputs{};
        std::vector<std::size_t> compiled_input_indices{};
        std::size_t selected_position{0U};
        std::size_t selected_dimension{0U};
        std::size_t output_entries{0U};
        std::size_t workspace_slot{0U};
    };

    struct TerminalMap {
        std::size_t node{0U};
        std::vector<std::size_t> retained_positions{};
        std::vector<std::size_t> strides{};
    };

    ExactFactorConfig config_{};
    std::vector<std::size_t> dimensions_{};
    std::vector<FactorVariableId> retained_variables_{};
    std::vector<std::vector<FactorVariableId>> factor_topology_{};
    std::vector<std::size_t> factor_logical_entries_{};
    std::vector<SourceFactor> sources_{};
    std::vector<Step> steps_{};
    std::vector<TerminalMap> terminals_{};
    std::vector<std::size_t> workspace_slot_sizes_{};
    ExactFactorStats stats_{};
    std::size_t graph_factor_count_{0U};
    std::size_t rebind_count_{0U};

    [[nodiscard]] QComplex source_value(const SourceFactor& source, std::size_t index) const;
    [[nodiscard]] QComplex node_value(
        std::size_t node,
        std::size_t index,
        const ExactFactorWorkspace& workspace) const;
    void validate_workspace(const ExactFactorWorkspace& workspace) const;
    void validate_topology(const ExactFactorGraph& graph) const;
};

[[nodiscard]] const char* exact_factor_route_name(ExactFactorRoute route) noexcept;

}  // namespace qubit
