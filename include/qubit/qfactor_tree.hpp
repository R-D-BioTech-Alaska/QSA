#pragma once

#include "qubit/qfactor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace qubit {

struct ExactFactorTreeStats {
    std::size_t variable_count{0U};
    std::size_t factor_count{0U};
    std::size_t retained_variables{0U};
    std::size_t source_dense_factors{0U};
    std::size_t source_sparse_factors{0U};
    std::size_t incidence_edges{0U};
    std::size_t message_count{0U};
    std::size_t message_entries{0U};
    std::size_t max_message_entries{0U};
    std::size_t output_entries{0U};
};

class ExactFactorTreeWorkspace {
public:
    ExactFactorTreeWorkspace() = default;

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        std::size_t bytes = sizeof(*this) +
                            messages_.capacity() * sizeof(std::vector<QComplex>) +
                            root_.capacity() * sizeof(QComplex);
        for (const auto& message : messages_) {
            bytes += message.capacity() * sizeof(QComplex);
        }
        return bytes;
    }

private:
    std::vector<std::vector<QComplex>> messages_{};
    std::vector<QComplex> root_{};

    friend class ExactFactorTreePlan;
};

class ExactFactorTreePlan {
public:
    ExactFactorTreePlan(
        const ExactFactorGraph& graph,
        std::span<const FactorVariableId> retained_variables = {})
        : config_(graph.config_),
          dimensions_(graph.dimensions_),
          retained_variables_(retained_variables.begin(), retained_variables.end()) {
        std::string reason;
        if (!graph.validate(&reason)) {
            throw QStateError("Cannot compile invalid exact factor tree: " + reason);
        }
        if (dimensions_.empty()) {
            throw QStateError("Exact factor tree requires at least one variable");
        }
        if (graph.factors_.empty()) {
            throw QStateError("Exact factor tree requires at least one factor");
        }
        if (retained_variables_.size() > 1U) {
            throw QStateError("Exact factor tree supports at most one retained variable");
        }
        if (!retained_variables_.empty() &&
            static_cast<std::size_t>(retained_variables_.front()) >= dimensions_.size()) {
            throw QStateError("Exact factor tree retained variable is out of range");
        }

        root_variable_ = retained_variables_.empty() ? 0U : retained_variables_.front();
        const std::size_t variable_count = dimensions_.size();
        if (graph.factors_.size() >
            std::numeric_limits<std::size_t>::max() - variable_count) {
            throw QStateError("Exact factor tree topology size overflowed");
        }
        const std::size_t node_count = variable_count + graph.factors_.size();
        adjacency_.resize(node_count);
        sources_.reserve(graph.factors_.size());

        std::size_t edge_count = 0U;
        for (std::size_t factor_index = 0U;
             factor_index < graph.factors_.size();
             ++factor_index) {
            const ExactFactorGraph::Factor& source = graph.factors_[factor_index];
            if (source.variables.empty()) {
                throw QStateError("Exact factor tree does not support scalar factors");
            }
            if (edge_count > std::numeric_limits<std::size_t>::max() - source.variables.size()) {
                throw QStateError("Exact factor tree incidence edge count overflowed");
            }
            edge_count += source.variables.size();

            SourceFactor factor;
            factor.variables = source.variables;
            factor.logical_entries = source.logical_entries;
            factor.storage = source.storage;
            factor.dense = source.dense;
            factor.sparse = source.sparse;
            factor.strides.resize(source.variables.size(), 1U);
            std::size_t stride = 1U;
            for (std::size_t position = 0U; position < source.variables.size(); ++position) {
                factor.strides[position] = stride;
                const std::size_t dimension = dimensions_[source.variables[position]];
                if (stride > std::numeric_limits<std::size_t>::max() / dimension) {
                    throw QStateError("Exact factor tree stride overflowed");
                }
                stride *= dimension;
            }
            if (stride != source.logical_entries) {
                throw QStateError("Exact factor tree factor shape is inconsistent");
            }
            sources_.push_back(std::move(factor));
            if (source.storage == FactorStorageMode::Dense) {
                ++stats_.source_dense_factors;
            } else {
                ++stats_.source_sparse_factors;
            }

            const std::size_t factor_node = variable_count + factor_index;
            for (const FactorVariableId variable : source.variables) {
                adjacency_[variable].push_back(factor_node);
                adjacency_[factor_node].push_back(variable);
            }
        }

        if (node_count == 0U || edge_count != node_count - 1U) {
            throw QStateError("Exact factor tree incidence graph is not acyclic and connected");
        }

        const std::size_t no_parent = std::numeric_limits<std::size_t>::max();
        parent_.assign(node_count, no_parent);
        parent_[root_variable_] = root_variable_;
        std::vector<std::size_t> order;
        order.reserve(node_count);
        order.push_back(root_variable_);
        for (std::size_t cursor = 0U; cursor < order.size(); ++cursor) {
            const std::size_t node = order[cursor];
            for (const std::size_t neighbor : adjacency_[node]) {
                if (parent_[neighbor] != no_parent) {
                    continue;
                }
                parent_[neighbor] = node;
                order.push_back(neighbor);
            }
        }
        if (order.size() != node_count) {
            throw QStateError("Exact factor tree incidence graph is disconnected");
        }

        postorder_.reserve(node_count - 1U);
        for (auto iterator = order.rbegin(); iterator != order.rend(); ++iterator) {
            if (*iterator != root_variable_) {
                postorder_.push_back(*iterator);
            }
        }

        message_slot_.assign(node_count, no_parent);
        message_entries_.reserve(postorder_.size());
        for (std::size_t slot = 0U; slot < postorder_.size(); ++slot) {
            const std::size_t node = postorder_[slot];
            const std::size_t entries = node < variable_count
                ? dimensions_[node]
                : dimensions_[parent_[node]];
            message_slot_[node] = slot;
            message_entries_.push_back(entries);
            stats_.max_message_entries = std::max(stats_.max_message_entries, entries);
            if (stats_.message_entries >
                std::numeric_limits<std::size_t>::max() - entries) {
                throw QStateError("Exact factor tree message storage size overflowed");
            }
            stats_.message_entries += entries;
        }

        stats_.variable_count = variable_count;
        stats_.factor_count = sources_.size();
        stats_.retained_variables = retained_variables_.size();
        stats_.incidence_edges = edge_count;
        stats_.message_count = postorder_.size();
        stats_.output_entries = retained_variables_.empty()
            ? 1U
            : dimensions_[root_variable_];
    }

    [[nodiscard]] ExactFactorTreeWorkspace workspace() const {
        ExactFactorTreeWorkspace result;
        result.messages_.resize(message_entries_.size());
        for (std::size_t slot = 0U; slot < message_entries_.size(); ++slot) {
            result.messages_[slot].resize(message_entries_[slot]);
        }
        result.root_.resize(dimensions_[root_variable_]);
        return result;
    }

    [[nodiscard]] std::vector<QComplex> evaluate() const {
        ExactFactorTreeWorkspace local = workspace();
        return evaluate(local);
    }

    [[nodiscard]] std::vector<QComplex> evaluate(
        ExactFactorTreeWorkspace& workspace_value) const {
        std::vector<QComplex> result(stats_.output_entries);
        evaluate(result, workspace_value);
        return result;
    }

    void evaluate(
        std::span<QComplex> output,
        ExactFactorTreeWorkspace& workspace_value) const {
        if (output.size() != stats_.output_entries) {
            throw QStateError("Exact factor tree output size does not match its plan");
        }
        validate_workspace(workspace_value);
        const std::size_t variable_count = dimensions_.size();

        for (const std::size_t node : postorder_) {
            const std::size_t slot = message_slot_[node];
            std::vector<QComplex>& target = workspace_value.messages_[slot];
            if (node < variable_count) {
                std::fill(target.begin(), target.end(), QComplex{1.0, 0.0});
                for (const std::size_t neighbor : adjacency_[node]) {
                    if (neighbor == parent_[node]) {
                        continue;
                    }
                    const std::size_t child_slot = message_slot_[neighbor];
                    const std::vector<QComplex>& child = workspace_value.messages_[child_slot];
                    for (std::size_t value = 0U; value < target.size(); ++value) {
                        target[value] *= child[value];
                    }
                }
                continue;
            }

            const SourceFactor& factor = sources_[node - variable_count];
            const FactorVariableId parent_variable =
                static_cast<FactorVariableId>(parent_[node]);
            std::fill(target.begin(), target.end(), QComplex{});
            const auto accumulate = [&](std::size_t index, QComplex value) {
                if (value.re == 0.0 && value.im == 0.0) {
                    return;
                }
                std::size_t parent_value = 0U;
                for (std::size_t position = 0U;
                     position < factor.variables.size();
                     ++position) {
                    const FactorVariableId variable = factor.variables[position];
                    const std::size_t coordinate =
                        (index / factor.strides[position]) % dimensions_[variable];
                    if (variable == parent_variable) {
                        parent_value = coordinate;
                        continue;
                    }
                    const std::size_t child_slot = message_slot_[variable];
                    value *= workspace_value.messages_[child_slot][coordinate];
                    if (value.re == 0.0 && value.im == 0.0) {
                        return;
                    }
                }
                target[parent_value] += value;
            };

            if (factor.storage == FactorStorageMode::Dense) {
                for (std::size_t index = 0U; index < factor.logical_entries; ++index) {
                    accumulate(index, factor.dense[index]);
                }
            } else {
                for (const FactorSparseEntry& entry : factor.sparse) {
                    accumulate(entry.index, entry.value);
                }
            }
        }

        std::fill(
            workspace_value.root_.begin(),
            workspace_value.root_.end(),
            QComplex{1.0, 0.0});
        for (const std::size_t neighbor : adjacency_[root_variable_]) {
            const std::size_t child_slot = message_slot_[neighbor];
            const std::vector<QComplex>& child = workspace_value.messages_[child_slot];
            for (std::size_t value = 0U; value < workspace_value.root_.size(); ++value) {
                workspace_value.root_[value] *= child[value];
            }
        }

        if (retained_variables_.empty()) {
            QComplex partition_value{};
            for (const QComplex& value : workspace_value.root_) {
                partition_value += value;
            }
            output[0] = partition_value;
        } else {
            std::copy(workspace_value.root_.begin(), workspace_value.root_.end(), output.begin());
        }
    }

    [[nodiscard]] QComplex partition() const {
        ExactFactorTreeWorkspace local = workspace();
        return partition(local);
    }

    [[nodiscard]] QComplex partition(ExactFactorTreeWorkspace& workspace_value) const {
        const std::vector<QComplex> values = evaluate(workspace_value);
        QComplex result{};
        for (const QComplex& value : values) {
            result += value;
        }
        return result;
    }

    [[nodiscard]] std::vector<QComplex> normalized_marginal() const {
        ExactFactorTreeWorkspace local = workspace();
        return normalized_marginal(local);
    }

    [[nodiscard]] std::vector<QComplex> normalized_marginal(
        ExactFactorTreeWorkspace& workspace_value) const {
        std::vector<QComplex> values = evaluate(workspace_value);
        QComplex normalization{};
        for (const QComplex& value : values) {
            normalization += value;
        }
        if (normalization.norm2() <= std::numeric_limits<double>::min()) {
            throw QStateError("Cannot normalize an exact factor tree with zero partition");
        }
        for (QComplex& value : values) {
            value /= normalization;
        }
        return values;
    }

    void rebind_dense_factor(FactorId factor_id, std::span<const QComplex> values) {
        SourceFactor& factor = mutable_factor(factor_id);
        if (values.size() != factor.logical_entries) {
            throw QStateError("Exact factor tree dense rebind size changed");
        }
        std::vector<QComplex> replacement(values.begin(), values.end());
        for (const QComplex& value : replacement) {
            if (!finite(value)) {
                throw QStateError("Exact factor tree dense rebind contains non-finite values");
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
                throw QStateError("Exact factor tree sparse rebind index is out of range");
            }
            if (!finite(entry.value)) {
                throw QStateError("Exact factor tree sparse rebind contains non-finite values");
            }
        }
        std::sort(
            replacement.begin(), replacement.end(),
            [](const FactorSparseEntry& first, const FactorSparseEntry& second) {
                return first.index < second.index;
            });
        for (std::size_t index = 1U; index < replacement.size(); ++index) {
            if (replacement[index - 1U].index == replacement[index].index) {
                throw QStateError("Exact factor tree sparse rebind repeats an index");
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
        return "ExactFactorTreeTransfer";
    }

    [[nodiscard]] std::span<const FactorVariableId> retained_variables() const noexcept {
        return retained_variables_;
    }

    [[nodiscard]] const ExactFactorTreeStats& stats() const noexcept {
        return stats_;
    }

    [[nodiscard]] std::size_t output_entries() const noexcept {
        return stats_.output_entries;
    }

    [[nodiscard]] std::size_t rebind_count() const noexcept {
        return rebind_count_;
    }

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        std::size_t bytes = sizeof(*this) +
                            dimensions_.capacity() * sizeof(std::size_t) +
                            retained_variables_.capacity() * sizeof(FactorVariableId) +
                            sources_.capacity() * sizeof(SourceFactor) +
                            adjacency_.capacity() * sizeof(std::vector<std::size_t>) +
                            parent_.capacity() * sizeof(std::size_t) +
                            postorder_.capacity() * sizeof(std::size_t) +
                            message_slot_.capacity() * sizeof(std::size_t) +
                            message_entries_.capacity() * sizeof(std::size_t);
        for (const SourceFactor& factor : sources_) {
            bytes += factor.variables.capacity() * sizeof(FactorVariableId) +
                     factor.strides.capacity() * sizeof(std::size_t) +
                     factor.dense.capacity() * sizeof(QComplex) +
                     factor.sparse.capacity() * sizeof(FactorSparseEntry);
        }
        for (const auto& neighbors : adjacency_) {
            bytes += neighbors.capacity() * sizeof(std::size_t);
        }
        return bytes;
    }

private:
    struct SourceFactor {
        std::vector<FactorVariableId> variables{};
        std::vector<std::size_t> strides{};
        std::size_t logical_entries{0U};
        FactorStorageMode storage{FactorStorageMode::Dense};
        std::vector<QComplex> dense{};
        std::vector<FactorSparseEntry> sparse{};
    };

    ExactFactorConfig config_{};
    std::vector<std::size_t> dimensions_{};
    std::vector<FactorVariableId> retained_variables_{};
    std::vector<SourceFactor> sources_{};
    std::vector<std::vector<std::size_t>> adjacency_{};
    std::vector<std::size_t> parent_{};
    std::vector<std::size_t> postorder_{};
    std::vector<std::size_t> message_slot_{};
    std::vector<std::size_t> message_entries_{};
    ExactFactorTreeStats stats_{};
    FactorVariableId root_variable_{0U};
    std::size_t rebind_count_{0U};

    [[nodiscard]] static bool finite(const QComplex& value) noexcept {
        return std::isfinite(value.re) && std::isfinite(value.im);
    }

    [[nodiscard]] SourceFactor& mutable_factor(FactorId factor_id) {
        const std::size_t index = static_cast<std::size_t>(factor_id);
        if (index >= sources_.size()) {
            throw QStateError("Exact factor tree factor is out of range");
        }
        return sources_[index];
    }

    void validate_workspace(const ExactFactorTreeWorkspace& workspace_value) const {
        if (workspace_value.messages_.size() != message_entries_.size() ||
            workspace_value.root_.size() != dimensions_[root_variable_]) {
            throw QStateError("Exact factor tree workspace does not match its plan");
        }
        for (std::size_t slot = 0U; slot < message_entries_.size(); ++slot) {
            if (workspace_value.messages_[slot].size() != message_entries_[slot]) {
                throw QStateError("Exact factor tree workspace message shape does not match its plan");
            }
        }
    }
};

}  // namespace qubit
