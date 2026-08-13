#pragma once

#include "qubit/qstate.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
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
    bool reuse_workspace_slots{false};
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
class ExactFactorChainPlan;
class ExactFactorDecisionPlan;
class ExactFactorAffinePlan;

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
    friend class ExactFactorChainPlan;
    friend class ExactFactorDecisionPlan;
    friend class ExactFactorAffinePlan;
};

class ExactFactorWorkspace {
public:
    ExactFactorWorkspace() = default;

    [[nodiscard]] std::size_t estimated_bytes() const noexcept;

    [[nodiscard]] std::size_t binding_estimated_bytes() const noexcept {
        std::size_t bytes =
            bound_source_slots_.capacity() * sizeof(FactorId) +
            bound_dense_values_.capacity() * sizeof(std::vector<QComplex>);
        for (const auto& values : bound_dense_values_) {
            const std::size_t value_bytes = values.capacity() * sizeof(QComplex);
            if (value_bytes > std::numeric_limits<std::size_t>::max() - bytes) {
                return std::numeric_limits<std::size_t>::max();
            }
            bytes += value_bytes;
        }
        return bytes;
    }

    [[nodiscard]] std::size_t bound_rebind_count() const noexcept {
        return bound_rebind_count_;
    }

private:
    std::vector<std::vector<QComplex>> outputs_{};
    std::vector<std::size_t> coordinates_{};
    std::vector<std::size_t> retained_coordinates_{};
    std::vector<FactorId> bound_source_slots_{};
    std::vector<std::vector<QComplex>> bound_dense_values_{};
    std::size_t bound_rebind_count_{0U};

    friend class ExactFactorPlan;
};

class ExactFactorPlan {
public:
    ExactFactorPlan(
        const ExactFactorGraph& graph,
        std::span<const FactorVariableId> retained_variables = {});

    [[nodiscard]] ExactFactorWorkspace workspace() const;

    [[nodiscard]] ExactFactorWorkspace workspace(
        std::span<const FactorId> bound_dense_factors) const {
        ExactFactorWorkspace result = workspace();
        if (bound_dense_factors.empty()) {
            return result;
        }
        if (graph_factor_count_ >
            static_cast<std::size_t>(std::numeric_limits<FactorId>::max())) {
            throw QStateError("Exact factor bound workspace source count exceeds FactorId range");
        }
        constexpr FactorId unbound = std::numeric_limits<FactorId>::max();
        result.bound_source_slots_.assign(graph_factor_count_, unbound);
        result.bound_dense_values_.reserve(bound_dense_factors.size());
        for (const FactorId factor : bound_dense_factors) {
            const std::size_t source_index = static_cast<std::size_t>(factor);
            if (source_index >= graph_factor_count_) {
                throw QStateError("Exact factor bound workspace factor is out of range");
            }
            if (result.bound_source_slots_[source_index] != unbound) {
                throw QStateError("Exact factor bound workspace repeats a factor");
            }
            const SourceFactor& source = sources_[source_index];
            if (source.synthetic || source.storage != FactorStorageMode::Dense) {
                throw QStateError("Exact factor workspace binding requires a dense source factor");
            }
            if (result.bound_dense_values_.size() >=
                static_cast<std::size_t>(std::numeric_limits<FactorId>::max())) {
                throw QStateError("Exact factor bound workspace slot count exceeds FactorId range");
            }
            const FactorId slot =
                static_cast<FactorId>(result.bound_dense_values_.size());
            result.bound_source_slots_[source_index] = slot;
            result.bound_dense_values_.push_back(source.dense);
        }
        return result;
    }

    void bind_dense_factor(
        ExactFactorWorkspace& workspace_value,
        FactorId factor,
        std::span<const QComplex> values) const {
        validate_workspace(workspace_value);
        const std::size_t source_index = static_cast<std::size_t>(factor);
        if (source_index >= graph_factor_count_ ||
            workspace_value.bound_source_slots_.size() != graph_factor_count_) {
            throw QStateError("Exact factor workspace binding is not declared for this plan");
        }
        constexpr FactorId unbound = std::numeric_limits<FactorId>::max();
        const FactorId slot_id = workspace_value.bound_source_slots_[source_index];
        if (slot_id == unbound ||
            static_cast<std::size_t>(slot_id) >= workspace_value.bound_dense_values_.size()) {
            throw QStateError("Exact factor workspace binding factor was not declared");
        }
        const SourceFactor& source = sources_[source_index];
        if (source.storage != FactorStorageMode::Dense ||
            values.size() != source.logical_entries) {
            throw QStateError("Exact factor workspace dense binding size or storage changed");
        }
        if (!std::all_of(
                values.begin(), values.end(),
                [](const QComplex& value) {
                    return std::isfinite(value.re) && std::isfinite(value.im);
                })) {
            throw QStateError("Exact factor workspace dense binding contains non-finite values");
        }
        std::vector<QComplex>& target =
            workspace_value.bound_dense_values_[static_cast<std::size_t>(slot_id)];
        if (target.size() != values.size()) {
            throw QStateError("Exact factor workspace dense binding shape changed");
        }
        std::copy(values.begin(), values.end(), target.begin());
        ++workspace_value.bound_rebind_count_;
    }

    [[nodiscard]] std::vector<QComplex> bound_evaluate(
        ExactFactorWorkspace& workspace_value) const {
        std::vector<QComplex> result(stats_.output_entries);
        bound_evaluate(result, workspace_value);
        return result;
    }

    void bound_evaluate(
        std::span<QComplex> output,
        ExactFactorWorkspace& workspace_value) const {
        if (output.size() != stats_.output_entries) {
            throw QStateError("Exact factor bound output size does not match its plan");
        }
        validate_bound_workspace(workspace_value);

        for (std::size_t step_index = 0U; step_index < steps_.size(); ++step_index) {
            const Step& step = steps_[step_index];
            std::vector<QComplex>& step_output =
                workspace_value.outputs_[step.workspace_slot];
            if (!step.compiled_input_indices.empty()) {
                const std::size_t input_count = step.inputs.size();
                for (std::size_t output_index = 0U;
                     output_index < step.output_entries;
                     ++output_index) {
                    QComplex sum{};
                    const std::size_t output_base =
                        output_index * step.selected_dimension * input_count;
                    for (std::size_t selected_value = 0U;
                         selected_value < step.selected_dimension;
                         ++selected_value) {
                        QComplex product{1.0, 0.0};
                        const std::size_t input_base =
                            output_base + selected_value * input_count;
                        for (std::size_t input_index = 0U;
                             input_index < input_count;
                             ++input_index) {
                            product *= bound_node_value(
                                step.inputs[input_index].node,
                                step.compiled_input_indices[input_base + input_index],
                                workspace_value);
                        }
                        sum += product;
                    }
                    step_output[output_index] = sum;
                }
                continue;
            }

            for (std::size_t output_index = 0U;
                 output_index < step.output_entries;
                 ++output_index) {
                std::size_t remaining = output_index;
                for (std::size_t position = 0U;
                     position < step.union_variables.size();
                     ++position) {
                    if (position == step.selected_position) {
                        continue;
                    }
                    const std::size_t dimension_value =
                        dimensions_[step.union_variables[position]];
                    workspace_value.coordinates_[position] = remaining % dimension_value;
                    remaining /= dimension_value;
                }

                QComplex sum{};
                for (std::size_t selected_value = 0U;
                     selected_value < step.selected_dimension;
                     ++selected_value) {
                    workspace_value.coordinates_[step.selected_position] = selected_value;
                    QComplex product{1.0, 0.0};
                    for (const InputMap& input : step.inputs) {
                        std::size_t local_index = 0U;
                        for (std::size_t position = 0U;
                             position < input.positions.size();
                             ++position) {
                            local_index +=
                                workspace_value.coordinates_[input.positions[position]] *
                                input.strides[position];
                        }
                        product *= bound_node_value(
                            input.node, local_index, workspace_value);
                    }
                    sum += product;
                }
                step_output[output_index] = sum;
            }
        }

        for (std::size_t output_index = 0U; output_index < output.size(); ++output_index) {
            std::size_t remaining = output_index;
            for (std::size_t position = 0U;
                 position < retained_variables_.size();
                 ++position) {
                const std::size_t dimension_value =
                    dimensions_[retained_variables_[position]];
                workspace_value.retained_coordinates_[position] =
                    remaining % dimension_value;
                remaining /= dimension_value;
            }

            QComplex product{1.0, 0.0};
            for (const TerminalMap& terminal : terminals_) {
                std::size_t local_index = 0U;
                for (std::size_t position = 0U;
                     position < terminal.retained_positions.size();
                     ++position) {
                    local_index +=
                        workspace_value.retained_coordinates_[
                            terminal.retained_positions[position]] *
                        terminal.strides[position];
                }
                product *= bound_node_value(
                    terminal.node, local_index, workspace_value);
            }
            output[output_index] = product;
        }
    }

    [[nodiscard]] QComplex bound_partition(
        ExactFactorWorkspace& workspace_value) const {
        const std::vector<QComplex> values = bound_evaluate(workspace_value);
        QComplex result{};
        for (const QComplex value : values) {
            result += value;
        }
        return result;
    }

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

    [[nodiscard]] QComplex bound_source_value(
        std::size_t source_index,
        std::size_t index,
        const ExactFactorWorkspace& workspace_value) const {
        if (source_index >= sources_.size()) {
            throw QStateError("Exact factor bound source index is out of range");
        }
        if (!workspace_value.bound_source_slots_.empty() &&
            source_index < graph_factor_count_) {
            if (workspace_value.bound_source_slots_.size() != graph_factor_count_) {
                throw QStateError("Exact factor bound source map has the wrong shape");
            }
            constexpr FactorId unbound = std::numeric_limits<FactorId>::max();
            const FactorId slot_id = workspace_value.bound_source_slots_[source_index];
            if (slot_id != unbound) {
                const std::size_t slot = static_cast<std::size_t>(slot_id);
                if (slot >= workspace_value.bound_dense_values_.size() ||
                    index >= workspace_value.bound_dense_values_[slot].size()) {
                    throw QStateError("Exact factor bound source value is out of range");
                }
                return workspace_value.bound_dense_values_[slot][index];
            }
        }
        return source_value(sources_[source_index], index);
    }

    [[nodiscard]] QComplex bound_node_value(
        std::size_t node,
        std::size_t index,
        const ExactFactorWorkspace& workspace_value) const {
        if (node < sources_.size()) {
            return bound_source_value(node, index, workspace_value);
        }
        const std::size_t producer = node - sources_.size();
        if (producer >= steps_.size()) {
            throw QStateError("Exact factor bound plan references an invalid dynamic node");
        }
        const Step& step = steps_[producer];
        if (step.workspace_slot >= workspace_value.outputs_.size() ||
            index >= step.output_entries ||
            index >= workspace_value.outputs_[step.workspace_slot].size()) {
            throw QStateError("Exact factor bound plan references an invalid workspace node");
        }
        return workspace_value.outputs_[step.workspace_slot][index];
    }

    void validate_bound_workspace(const ExactFactorWorkspace& workspace_value) const {
        validate_workspace(workspace_value);
        if (workspace_value.bound_source_slots_.size() != graph_factor_count_) {
            throw QStateError("Exact factor bound workspace has no declared source-binding map");
        }
        constexpr FactorId unbound = std::numeric_limits<FactorId>::max();
        std::size_t bound_count = 0U;
        for (const FactorId slot_id : workspace_value.bound_source_slots_) {
            if (slot_id == unbound) {
                continue;
            }
            if (static_cast<std::size_t>(slot_id) >=
                workspace_value.bound_dense_values_.size()) {
                throw QStateError("Exact factor bound workspace contains an invalid source slot");
            }
            ++bound_count;
        }
        if (bound_count != workspace_value.bound_dense_values_.size()) {
            throw QStateError("Exact factor bound workspace source-slot accounting is inconsistent");
        }
    }

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
