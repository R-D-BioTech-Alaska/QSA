#pragma once

#include "qubit/qfactor.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace qubit {

struct ExactFactorDecisionConfig {
    std::size_t max_nodes{1U << 20U};
    std::size_t max_apply_pairs{1U << 22U};
    std::size_t max_variables{2'048U};
};

struct ExactFactorDecisionStats {
    std::size_t variable_count{0U};
    std::size_t source_factors{0U};
    std::size_t source_dense_factors{0U};
    std::size_t source_sparse_factors{0U};
    std::size_t retained_variables{0U};
    std::size_t peak_compile_nodes{0U};
    std::size_t apply_pairs_peak{0U};
    std::size_t compiled_nodes{0U};
    std::size_t terminal_nodes{0U};
    std::size_t internal_nodes{0U};
    std::size_t output_entries{0U};
};

class ExactFactorDecisionWorkspace {
public:
    ExactFactorDecisionWorkspace() = default;

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        return sizeof(*this) + sums_.capacity() * sizeof(std::array<QComplex, 2>);
    }

private:
    std::vector<std::array<QComplex, 2>> sums_{};

    friend class ExactFactorDecisionPlan;
};

class ExactFactorDecisionPlan {
public:
    ExactFactorDecisionPlan(
        const ExactFactorGraph& graph,
        std::span<const FactorVariableId> retained_variables = {},
        ExactFactorDecisionConfig config = {})
        : config_(config),
          variable_count_(graph.dimensions_.size()),
          retained_variables_(retained_variables.begin(), retained_variables.end()) {
        std::string reason;
        if (!graph.validate(&reason)) {
            throw QStateError("Cannot compile invalid exact factor decision graph: " + reason);
        }
        if (variable_count_ == 0U) {
            throw QStateError("Exact factor decision graph requires at least one variable");
        }
        if (config_.max_nodes == 0U || config_.max_apply_pairs == 0U ||
            config_.max_variables == 0U) {
            throw QStateError("Exact factor decision configuration contains a zero resource cap");
        }
        if (variable_count_ > config_.max_variables) {
            throw QStateError("Exact factor decision graph exceeds configured variable recursion cap");
        }
        if (retained_variables_.size() > 1U) {
            throw QStateError("Exact factor decision graph supports at most one retained variable");
        }
        if (!retained_variables_.empty() &&
            static_cast<std::size_t>(retained_variables_.front()) >= variable_count_) {
            throw QStateError("Exact factor decision retained variable is out of range");
        }
        for (const std::size_t dimension : graph.dimensions_) {
            if (dimension != 2U) {
                throw QStateError("Exact factor decision graph currently requires binary variables");
            }
        }

        stats_.variable_count = variable_count_;
        stats_.source_factors = graph.factors_.size();
        stats_.retained_variables = retained_variables_.size();
        for (const ExactFactorGraph::Factor& factor : graph.factors_) {
            if (factor.storage == FactorStorageMode::Dense) {
                ++stats_.source_dense_factors;
            } else {
                ++stats_.source_sparse_factors;
            }
        }

        root_ = intern_terminal({1.0, 0.0});
        for (const ExactFactorGraph::Factor& factor : graph.factors_) {
            const NodeId factor_root = build_factor(factor);
            ApplyMemo memo;
            memo.reserve(std::min(config_.max_apply_pairs, std::size_t{4096U}));
            root_ = multiply(root_, factor_root, memo, 0U);
            stats_.apply_pairs_peak = std::max(stats_.apply_pairs_peak, memo.size());
        }
        compact();
        stats_.compiled_nodes = nodes_.size();
        for (const Node& node : nodes_) {
            if (node.terminal) {
                ++stats_.terminal_nodes;
            } else {
                ++stats_.internal_nodes;
            }
        }
        stats_.output_entries = retained_variables_.empty() ? 1U : 2U;

        terminal_unique_.clear();
        terminal_unique_.rehash(0U);
        internal_unique_.clear();
        internal_unique_.rehash(0U);
    }

    [[nodiscard]] ExactFactorDecisionWorkspace workspace() const {
        ExactFactorDecisionWorkspace result;
        result.sums_.resize(nodes_.size());
        return result;
    }

    [[nodiscard]] std::vector<QComplex> evaluate() const {
        ExactFactorDecisionWorkspace local = workspace();
        return evaluate(local);
    }

    [[nodiscard]] std::vector<QComplex> evaluate(
        ExactFactorDecisionWorkspace& workspace_value) const {
        std::vector<QComplex> result(stats_.output_entries);
        evaluate(result, workspace_value);
        return result;
    }

    void evaluate(
        std::span<QComplex> output,
        ExactFactorDecisionWorkspace& workspace_value) const {
        if (output.size() != stats_.output_entries) {
            throw QStateError("Exact factor decision output size does not match its plan");
        }
        if (workspace_value.sums_.size() != nodes_.size()) {
            throw QStateError("Exact factor decision workspace does not match its plan");
        }

        const bool retained = !retained_variables_.empty();
        const FactorVariableId retained_variable = retained ? retained_variables_.front() : 0U;
        for (NodeId id = 0U; id < nodes_.size(); ++id) {
            const Node& node = nodes_[id];
            auto& sums = workspace_value.sums_[id];
            if (node.terminal) {
                sums[0] = node.value;
                sums[1] = node.value;
                continue;
            }

            if (retained && node.variable == retained_variable) {
                sums[0] = edge_value(
                    node.variable, node.low, 0U, retained_variable, workspace_value);
                sums[1] = edge_value(
                    node.variable, node.high, 1U, retained_variable, workspace_value);
                continue;
            }

            for (std::size_t bit = 0U; bit < 2U; ++bit) {
                sums[bit] = edge_value(
                    node.variable, node.low, bit, retained_variable, workspace_value) +
                    edge_value(
                        node.variable, node.high, bit, retained_variable, workspace_value);
            }
        }

        const Node& root_node = nodes_[root_];
        const std::size_t root_variable = root_node.terminal
            ? variable_count_
            : static_cast<std::size_t>(root_node.variable);
        if (!retained) {
            output[0] = scale_pow2(workspace_value.sums_[root_][0], root_variable);
            return;
        }

        std::size_t leading_free = root_variable;
        if (static_cast<std::size_t>(retained_variable) < root_variable) {
            --leading_free;
        }
        output[0] = scale_pow2(workspace_value.sums_[root_][0], leading_free);
        output[1] = scale_pow2(workspace_value.sums_[root_][1], leading_free);
    }

    [[nodiscard]] QComplex partition() const {
        ExactFactorDecisionWorkspace local = workspace();
        return partition(local);
    }

    [[nodiscard]] QComplex partition(ExactFactorDecisionWorkspace& workspace_value) const {
        const std::vector<QComplex> values = evaluate(workspace_value);
        if (retained_variables_.empty()) {
            return values.front();
        }
        return values[0] + values[1];
    }

    [[nodiscard]] std::vector<QComplex> normalized_marginal() const {
        ExactFactorDecisionWorkspace local = workspace();
        return normalized_marginal(local);
    }

    [[nodiscard]] std::vector<QComplex> normalized_marginal(
        ExactFactorDecisionWorkspace& workspace_value) const {
        std::vector<QComplex> values = evaluate(workspace_value);
        QComplex normalization{};
        for (const QComplex& value : values) {
            normalization += value;
        }
        if (normalization.norm2() <= std::numeric_limits<double>::min()) {
            throw QStateError("Cannot normalize an exact factor decision graph with zero partition");
        }
        for (QComplex& value : values) {
            value /= normalization;
        }
        return values;
    }

    [[nodiscard]] const char* route_name() const noexcept {
        return "ExactFactorDecisionDiagram";
    }

    [[nodiscard]] const ExactFactorDecisionStats& stats() const noexcept {
        return stats_;
    }

    [[nodiscard]] std::span<const FactorVariableId> retained_variables() const noexcept {
        return retained_variables_;
    }

    [[nodiscard]] std::size_t output_entries() const noexcept {
        return stats_.output_entries;
    }

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        return sizeof(*this) +
               retained_variables_.capacity() * sizeof(FactorVariableId) +
               nodes_.capacity() * sizeof(Node);
    }

private:
    using NodeId = std::size_t;

    struct Node {
        FactorVariableId variable{0U};
        NodeId low{0U};
        NodeId high{0U};
        QComplex value{};
        bool terminal{false};
    };

    struct Axis {
        FactorVariableId variable{0U};
        std::size_t stride{0U};
    };

    struct TerminalKey {
        std::uint64_t re{0U};
        std::uint64_t im{0U};

        [[nodiscard]] bool operator==(const TerminalKey&) const noexcept = default;
    };

    struct InternalKey {
        FactorVariableId variable{0U};
        NodeId low{0U};
        NodeId high{0U};

        [[nodiscard]] bool operator==(const InternalKey&) const noexcept = default;
    };

    struct ApplyKey {
        NodeId first{0U};
        NodeId second{0U};

        [[nodiscard]] bool operator==(const ApplyKey&) const noexcept = default;
    };

    struct TerminalHash {
        [[nodiscard]] std::size_t operator()(const TerminalKey& key) const noexcept {
            return mix(static_cast<std::size_t>(key.re), static_cast<std::size_t>(key.im));
        }
    };

    struct InternalHash {
        [[nodiscard]] std::size_t operator()(const InternalKey& key) const noexcept {
            return mix(
                mix(static_cast<std::size_t>(key.variable), key.low),
                key.high);
        }
    };

    struct ApplyHash {
        [[nodiscard]] std::size_t operator()(const ApplyKey& key) const noexcept {
            return mix(key.first, key.second);
        }
    };

    using ApplyMemo = std::unordered_map<ApplyKey, NodeId, ApplyHash>;

    ExactFactorDecisionConfig config_{};
    std::size_t variable_count_{0U};
    std::vector<FactorVariableId> retained_variables_{};
    std::vector<Node> nodes_{};
    NodeId root_{0U};
    ExactFactorDecisionStats stats_{};
    std::unordered_map<TerminalKey, NodeId, TerminalHash> terminal_unique_{};
    std::unordered_map<InternalKey, NodeId, InternalHash> internal_unique_{};

    [[nodiscard]] static std::size_t mix(std::size_t first, std::size_t second) noexcept {
        return first ^ (second + 0x9e3779b97f4a7c15ULL + (first << 6U) + (first >> 2U));
    }

    [[nodiscard]] static QComplex canonical(QComplex value) noexcept {
        if (value.re == 0.0) {
            value.re = 0.0;
        }
        if (value.im == 0.0) {
            value.im = 0.0;
        }
        return value;
    }

    [[nodiscard]] static TerminalKey terminal_key(QComplex value) noexcept {
        value = canonical(value);
        return {
            std::bit_cast<std::uint64_t>(value.re),
            std::bit_cast<std::uint64_t>(value.im),
        };
    }

    [[nodiscard]] static bool zero(const QComplex& value) noexcept {
        return value.re == 0.0 && value.im == 0.0;
    }

    [[nodiscard]] static bool one(const QComplex& value) noexcept {
        return value.re == 1.0 && value.im == 0.0;
    }

    void require_node_capacity() {
        if (nodes_.size() >= config_.max_nodes) {
            throw QStateError("Exact factor decision graph exceeded configured node cap");
        }
    }

    [[nodiscard]] NodeId intern_terminal(QComplex value) {
        value = canonical(value);
        if (!std::isfinite(value.re) || !std::isfinite(value.im)) {
            throw QStateError("Exact factor decision graph produced a non-finite terminal");
        }
        const TerminalKey key = terminal_key(value);
        const auto found = terminal_unique_.find(key);
        if (found != terminal_unique_.end()) {
            return found->second;
        }
        require_node_capacity();
        const NodeId id = nodes_.size();
        nodes_.push_back({0U, 0U, 0U, value, true});
        terminal_unique_.emplace(key, id);
        stats_.peak_compile_nodes = std::max(stats_.peak_compile_nodes, nodes_.size());
        return id;
    }

    [[nodiscard]] NodeId intern_internal(
        FactorVariableId variable,
        NodeId low,
        NodeId high) {
        if (low == high) {
            return low;
        }
        const InternalKey key{variable, low, high};
        const auto found = internal_unique_.find(key);
        if (found != internal_unique_.end()) {
            return found->second;
        }
        require_node_capacity();
        const NodeId id = nodes_.size();
        nodes_.push_back({variable, low, high, {}, false});
        internal_unique_.emplace(key, id);
        stats_.peak_compile_nodes = std::max(stats_.peak_compile_nodes, nodes_.size());
        return id;
    }

    [[nodiscard]] QComplex factor_value(
        const ExactFactorGraph::Factor& factor,
        std::size_t index) const {
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

    [[nodiscard]] NodeId build_factor(const ExactFactorGraph::Factor& factor) {
        std::vector<Axis> axes;
        axes.reserve(factor.variables.size());
        std::size_t stride = 1U;
        for (const FactorVariableId variable : factor.variables) {
            axes.push_back({variable, stride});
            if (stride > std::numeric_limits<std::size_t>::max() / 2U) {
                throw QStateError("Exact factor decision factor stride overflowed");
            }
            stride *= 2U;
        }
        if (stride != factor.logical_entries) {
            throw QStateError("Exact factor decision factor shape changed");
        }
        std::sort(
            axes.begin(), axes.end(),
            [](const Axis& first, const Axis& second) {
                return first.variable < second.variable;
            });
        return build_factor_recursive(factor, axes, 0U, 0U);
    }

    [[nodiscard]] NodeId build_factor_recursive(
        const ExactFactorGraph::Factor& factor,
        std::span<const Axis> axes,
        std::size_t position,
        std::size_t table_index) {
        if (position == axes.size()) {
            return intern_terminal(factor_value(factor, table_index));
        }
        const Axis& axis = axes[position];
        const NodeId low = build_factor_recursive(
            factor, axes, position + 1U, table_index);
        const NodeId high = build_factor_recursive(
            factor, axes, position + 1U, table_index + axis.stride);
        return intern_internal(axis.variable, low, high);
    }

    [[nodiscard]] NodeId multiply(
        NodeId first,
        NodeId second,
        ApplyMemo& memo,
        std::size_t depth) {
        if (depth > variable_count_) {
            throw QStateError("Exact factor decision apply exceeded variable recursion bound");
        }
        const Node& first_node = nodes_[first];
        const Node& second_node = nodes_[second];
        if (first_node.terminal) {
            if (zero(first_node.value)) {
                return first;
            }
            if (one(first_node.value)) {
                return second;
            }
        }
        if (second_node.terminal) {
            if (zero(second_node.value)) {
                return second;
            }
            if (one(second_node.value)) {
                return first;
            }
        }
        if (first_node.terminal && second_node.terminal) {
            return intern_terminal(first_node.value * second_node.value);
        }

        ApplyKey key{std::min(first, second), std::max(first, second)};
        const auto found = memo.find(key);
        if (found != memo.end()) {
            return found->second;
        }
        if (memo.size() >= config_.max_apply_pairs) {
            throw QStateError("Exact factor decision apply exceeded configured pair cap");
        }

        const FactorVariableId first_variable = first_node.terminal
            ? std::numeric_limits<FactorVariableId>::max()
            : first_node.variable;
        const FactorVariableId second_variable = second_node.terminal
            ? std::numeric_limits<FactorVariableId>::max()
            : second_node.variable;
        const FactorVariableId variable = std::min(first_variable, second_variable);
        const NodeId first_low = first_variable == variable ? first_node.low : first;
        const NodeId first_high = first_variable == variable ? first_node.high : first;
        const NodeId second_low = second_variable == variable ? second_node.low : second;
        const NodeId second_high = second_variable == variable ? second_node.high : second;
        const NodeId low = multiply(first_low, second_low, memo, depth + 1U);
        const NodeId high = multiply(first_high, second_high, memo, depth + 1U);
        const NodeId result = intern_internal(variable, low, high);
        memo.emplace(key, result);
        return result;
    }

    void compact() {
        const NodeId missing = std::numeric_limits<NodeId>::max();
        std::vector<NodeId> remap(nodes_.size(), missing);
        std::vector<std::uint8_t> state(nodes_.size(), 0U);
        std::vector<std::pair<NodeId, bool>> stack;
        stack.reserve(std::min(nodes_.size() * 2U, config_.max_nodes));
        stack.push_back({root_, false});
        std::vector<Node> compact_nodes;
        compact_nodes.reserve(nodes_.size());

        while (!stack.empty()) {
            const auto [id, expanded] = stack.back();
            stack.pop_back();
            if (state[id] == 2U) {
                continue;
            }
            const Node& node = nodes_[id];
            if (!expanded && !node.terminal) {
                if (state[id] == 1U) {
                    throw QStateError("Exact factor decision graph contains an internal cycle");
                }
                state[id] = 1U;
                stack.push_back({id, true});
                if (state[node.high] == 0U) {
                    stack.push_back({node.high, false});
                }
                if (state[node.low] == 0U) {
                    stack.push_back({node.low, false});
                }
                continue;
            }

            Node replacement = node;
            if (!node.terminal) {
                if (remap[node.low] == missing || remap[node.high] == missing) {
                    throw QStateError("Exact factor decision compaction order is invalid");
                }
                replacement.low = remap[node.low];
                replacement.high = remap[node.high];
            }
            remap[id] = compact_nodes.size();
            compact_nodes.push_back(replacement);
            state[id] = 2U;
        }

        if (remap[root_] == missing) {
            throw QStateError("Exact factor decision compaction lost the root");
        }
        root_ = remap[root_];
        nodes_ = std::move(compact_nodes);
    }

    [[nodiscard]] std::size_t node_variable(NodeId id) const noexcept {
        return nodes_[id].terminal
            ? variable_count_
            : static_cast<std::size_t>(nodes_[id].variable);
    }

    [[nodiscard]] static QComplex scale_pow2(QComplex value, std::size_t exponent) noexcept {
        if (exponent == 0U) {
            return value;
        }
        const int power = exponent > static_cast<std::size_t>(std::numeric_limits<int>::max())
            ? std::numeric_limits<int>::max()
            : static_cast<int>(exponent);
        value.re = std::ldexp(value.re, power);
        value.im = std::ldexp(value.im, power);
        return value;
    }

    [[nodiscard]] QComplex edge_value(
        FactorVariableId parent_variable,
        NodeId child,
        std::size_t bit,
        FactorVariableId retained_variable,
        const ExactFactorDecisionWorkspace& workspace_value) const noexcept {
        const std::size_t child_variable = node_variable(child);
        std::size_t free_variables =
            child_variable - static_cast<std::size_t>(parent_variable) - 1U;
        if (!retained_variables_.empty() &&
            static_cast<std::size_t>(retained_variable) > static_cast<std::size_t>(parent_variable) &&
            static_cast<std::size_t>(retained_variable) < child_variable) {
            --free_variables;
        }
        return scale_pow2(workspace_value.sums_[child][bit], free_variables);
    }
};

}  // namespace qubit
