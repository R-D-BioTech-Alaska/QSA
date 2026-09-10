#pragma once

#include "qubit/qfactor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <span>
#include <utility>
#include <vector>

namespace qubit {

struct ExactFactorMaxSumStats {
    std::size_t variable_count{0U};
    std::size_t source_factors{0U};
    std::size_t source_dense_factors{0U};
    std::size_t source_sparse_factors{0U};
    std::size_t eliminated_variables{0U};
    std::size_t peak_union_variables{0U};
    std::size_t peak_factor_entries{0U};
    std::size_t backpointer_entries{0U};
};

struct ExactFactorMaxSumResult {
    bool feasible{false};
    double score{0.0};
    std::vector<std::size_t> assignment{};
};

class ExactFactorMaxSumPlan {
public:
    explicit ExactFactorMaxSumPlan(const ExactFactorGraph& graph)
        : config_(graph.config_), dimensions_(graph.dimensions_) {
        compile(graph);
    }

    [[nodiscard]] const ExactFactorMaxSumResult& result() const noexcept {
        return result_;
    }

    [[nodiscard]] const ExactFactorMaxSumStats& stats() const noexcept {
        return stats_;
    }

    [[nodiscard]] const char* route_name() const noexcept {
        return "ExactFactorMaxSumVariableElimination";
    }

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        return sizeof(*this) + result_.assignment.capacity() * sizeof(std::size_t) +
               dimensions_.capacity() * sizeof(std::size_t);
    }

private:
    struct Node {
        const ExactFactorGraph::Factor* source{nullptr};
        std::vector<FactorVariableId> variables{};
        std::vector<std::size_t> strides{};
        std::vector<double> scores{};
        std::vector<std::uint8_t> feasible{};
    };

    struct InputMap {
        std::size_t node{0U};
        std::vector<std::size_t> positions{};
    };

    struct Trace {
        FactorVariableId eliminated{0U};
        std::vector<FactorVariableId> output_variables{};
        std::vector<std::size_t> choices{};
    };

    struct Candidate {
        std::size_t entries{0U};
        FactorVariableId variable{0U};
        std::uint64_t revision{0U};
    };

    struct CandidateGreater {
        [[nodiscard]] bool operator()(
            const Candidate& first,
            const Candidate& second) const noexcept {
            if (first.entries != second.entries) {
                return first.entries > second.entries;
            }
            return first.variable > second.variable;
        }
    };

    ExactFactorConfig config_{};
    std::vector<std::size_t> dimensions_{};
    ExactFactorMaxSumResult result_{};
    ExactFactorMaxSumStats stats_{};

    [[nodiscard]] std::size_t entries(
        std::span<const FactorVariableId> variables) const noexcept {
        std::size_t count = 1U;
        for (const FactorVariableId variable : variables) {
            if (static_cast<std::size_t>(variable) >= dimensions_.size()) {
                return std::numeric_limits<std::size_t>::max();
            }
            const std::size_t dimension = dimensions_[variable];
            if (dimension == 0U ||
                count > std::numeric_limits<std::size_t>::max() / dimension) {
                return std::numeric_limits<std::size_t>::max();
            }
            count *= dimension;
        }
        return count;
    }

    [[nodiscard]] std::vector<std::size_t> strides(
        std::span<const FactorVariableId> variables) const {
        std::vector<std::size_t> result(variables.size(), 1U);
        std::size_t stride = 1U;
        for (std::size_t position = 0U; position < variables.size(); ++position) {
            result[position] = stride;
            const std::size_t dimension = dimensions_[variables[position]];
            if (stride > std::numeric_limits<std::size_t>::max() / dimension) {
                throw QStateError("Exact factor max-sum stride overflowed");
            }
            stride *= dimension;
        }
        return result;
    }

    [[nodiscard]] static std::vector<FactorVariableId> bucket_union(
        std::span<const std::size_t> bucket,
        const std::vector<Node>& nodes) {
        std::vector<FactorVariableId> variables;
        for (const std::size_t node : bucket) {
            variables.insert(
                variables.end(), nodes[node].variables.begin(), nodes[node].variables.end());
        }
        std::sort(variables.begin(), variables.end());
        variables.erase(std::unique(variables.begin(), variables.end()), variables.end());
        return variables;
    }

    [[nodiscard]] static bool node_score(
        const Node& node,
        std::size_t index,
        double* score) {
        if (node.source == nullptr) {
            if (index >= node.scores.size() || index >= node.feasible.size()) {
                throw QStateError("Exact factor max-sum node index is out of range");
            }
            if (node.feasible[index] == 0U) {
                return false;
            }
            *score = node.scores[index];
            return true;
        }

        if (index >= node.source->logical_entries) {
            throw QStateError("Exact factor max-sum source index is out of range");
        }
        if (node.source->storage == FactorStorageMode::Dense) {
            *score = node.source->dense[index].re;
            return true;
        }
        const auto found = std::lower_bound(
            node.source->sparse.begin(), node.source->sparse.end(), index,
            [](const FactorSparseEntry& entry, std::size_t target) {
                return entry.index < target;
            });
        if (found == node.source->sparse.end() || found->index != index) {
            return false;
        }
        *score = found->value.re;
        return true;
    }

    void compile(const ExactFactorGraph& graph) {
        std::string reason;
        if (!graph.validate(&reason)) {
            throw QStateError("Cannot compile invalid exact factor max-sum graph: " + reason);
        }

        stats_.variable_count = dimensions_.size();
        stats_.source_factors = graph.factors_.size();

        std::vector<Node> nodes;
        nodes.reserve(graph.factors_.size() + dimensions_.size());
        std::vector<std::uint8_t> variable_seen(dimensions_.size(), 0U);
        for (const ExactFactorGraph::Factor& factor : graph.factors_) {
            const auto require_real = [](const QComplex& value) {
                if (value.im != 0.0) {
                    throw QStateError("Exact factor max-sum requires real additive scores");
                }
            };
            if (factor.storage == FactorStorageMode::Dense) {
                for (const QComplex& value : factor.dense) {
                    require_real(value);
                }
                ++stats_.source_dense_factors;
            } else {
                for (const FactorSparseEntry& entry : factor.sparse) {
                    require_real(entry.value);
                }
                ++stats_.source_sparse_factors;
            }

            Node node;
            node.source = &factor;
            node.variables = factor.variables;
            node.strides = strides(node.variables);
            nodes.push_back(std::move(node));
            for (const FactorVariableId variable : factor.variables) {
                variable_seen[variable] = 1U;
            }
            stats_.peak_union_variables =
                std::max(stats_.peak_union_variables, factor.variables.size());
            stats_.peak_factor_entries =
                std::max(stats_.peak_factor_entries, factor.logical_entries);
        }

        for (std::size_t variable = 0U; variable < dimensions_.size(); ++variable) {
            if (variable_seen[variable] != 0U) {
                continue;
            }
            Node unit;
            unit.variables.push_back(static_cast<FactorVariableId>(variable));
            unit.strides.push_back(1U);
            unit.scores.assign(dimensions_[variable], 0.0);
            unit.feasible.assign(dimensions_[variable], 1U);
            nodes.push_back(std::move(unit));
            stats_.peak_union_variables = std::max<std::size_t>(stats_.peak_union_variables, 1U);
            stats_.peak_factor_entries =
                std::max(stats_.peak_factor_entries, dimensions_[variable]);
        }
        if (nodes.size() > config_.max_factors) {
            throw QStateError("Exact factor max-sum exceeds max_factors after isolated-variable units");
        }

        std::vector<bool> active(nodes.size(), true);
        std::vector<std::vector<std::size_t>> incidence(dimensions_.size());
        for (std::size_t node = 0U; node < nodes.size(); ++node) {
            for (const FactorVariableId variable : nodes[node].variables) {
                incidence[variable].push_back(node);
            }
        }

        std::vector<std::uint64_t> revisions(dimensions_.size(), 0U);
        std::priority_queue<Candidate, std::vector<Candidate>, CandidateGreater> candidates;
        const auto enqueue = [&](FactorVariableId variable) {
            if (incidence[variable].empty()) {
                return;
            }
            const std::vector<FactorVariableId> variables =
                bucket_union(incidence[variable], nodes);
            candidates.push({entries(variables), variable, revisions[variable]});
        };
        for (std::size_t variable = 0U; variable < dimensions_.size(); ++variable) {
            enqueue(static_cast<FactorVariableId>(variable));
        }

        std::vector<Trace> traces;
        traces.reserve(dimensions_.size());
        constexpr std::size_t no_choice = std::numeric_limits<std::size_t>::max();

        while (true) {
            while (!candidates.empty()) {
                const Candidate& candidate = candidates.top();
                if (candidate.revision == revisions[candidate.variable] &&
                    !incidence[candidate.variable].empty()) {
                    break;
                }
                candidates.pop();
            }
            if (candidates.empty()) {
                break;
            }

            const Candidate candidate = candidates.top();
            candidates.pop();
            const FactorVariableId selected = candidate.variable;
            const std::vector<std::size_t> selected_bucket = incidence[selected];
            const std::vector<FactorVariableId> selected_union =
                bucket_union(selected_bucket, nodes);
            const std::size_t selected_entries = entries(selected_union);
            if (selected_entries == std::numeric_limits<std::size_t>::max() ||
                selected_entries > config_.max_factor_entries) {
                throw QStateError("Exact factor max-sum elimination exceeded max_factor_entries");
            }

            const auto selected_it =
                std::lower_bound(selected_union.begin(), selected_union.end(), selected);
            if (selected_it == selected_union.end() || *selected_it != selected) {
                throw QStateError("Exact factor max-sum planner lost its selected variable");
            }
            const std::size_t selected_position =
                static_cast<std::size_t>(selected_it - selected_union.begin());
            const std::size_t selected_dimension = dimensions_[selected];

            std::vector<FactorVariableId> output_variables = selected_union;
            output_variables.erase(
                output_variables.begin() + static_cast<std::ptrdiff_t>(selected_position));
            const std::size_t output_entries = entries(output_variables);
            if (output_entries == std::numeric_limits<std::size_t>::max() ||
                output_entries > config_.max_factor_entries) {
                throw QStateError("Exact factor max-sum reduced table exceeded max_factor_entries");
            }
            if (output_entries >
                std::numeric_limits<std::size_t>::max() / selected_dimension ||
                output_entries * selected_dimension != selected_entries) {
                throw QStateError("Exact factor max-sum assignment count is inconsistent");
            }
            if (stats_.backpointer_entries > config_.max_compiled_index_entries ||
                output_entries > config_.max_compiled_index_entries - stats_.backpointer_entries) {
                throw QStateError("Exact factor max-sum backpointers exceed max_compiled_index_entries");
            }

            std::vector<InputMap> inputs;
            inputs.reserve(selected_bucket.size());
            for (const std::size_t node : selected_bucket) {
                InputMap input;
                input.node = node;
                input.positions.reserve(nodes[node].variables.size());
                for (const FactorVariableId variable : nodes[node].variables) {
                    const auto position =
                        std::lower_bound(selected_union.begin(), selected_union.end(), variable);
                    if (position == selected_union.end() || *position != variable) {
                        throw QStateError("Exact factor max-sum input variable is missing from its union");
                    }
                    input.positions.push_back(
                        static_cast<std::size_t>(position - selected_union.begin()));
                }
                inputs.push_back(std::move(input));
            }

            Node output;
            output.variables = output_variables;
            output.strides = strides(output.variables);
            output.scores.resize(output_entries);
            output.feasible.assign(output_entries, 0U);

            Trace trace;
            trace.eliminated = selected;
            trace.output_variables = output_variables;
            trace.choices.assign(output_entries, no_choice);

            std::vector<std::size_t> coordinates(selected_union.size(), 0U);
            for (std::size_t output_index = 0U; output_index < output_entries; ++output_index) {
                std::size_t remaining = output_index;
                for (std::size_t position = 0U; position < selected_union.size(); ++position) {
                    if (position == selected_position) {
                        continue;
                    }
                    const std::size_t dimension = dimensions_[selected_union[position]];
                    coordinates[position] = remaining % dimension;
                    remaining /= dimension;
                }

                bool found = false;
                double best = 0.0;
                std::size_t best_value = 0U;
                for (std::size_t selected_value = 0U;
                     selected_value < selected_dimension;
                     ++selected_value) {
                    coordinates[selected_position] = selected_value;
                    bool feasible = true;
                    double total = 0.0;
                    for (const InputMap& input : inputs) {
                        const Node& node = nodes[input.node];
                        std::size_t local_index = 0U;
                        for (std::size_t position = 0U;
                             position < input.positions.size();
                             ++position) {
                            local_index +=
                                coordinates[input.positions[position]] * node.strides[position];
                        }
                        double value = 0.0;
                        if (!node_score(node, local_index, &value)) {
                            feasible = false;
                            break;
                        }
                        total += value;
                        if (!std::isfinite(total)) {
                            throw QStateError("Exact factor max-sum score overflowed");
                        }
                    }
                    if (feasible && (!found || total > best)) {
                        found = true;
                        best = total;
                        best_value = selected_value;
                    }
                }
                if (found) {
                    output.scores[output_index] = best;
                    output.feasible[output_index] = 1U;
                    trace.choices[output_index] = best_value;
                }
            }

            stats_.peak_union_variables =
                std::max(stats_.peak_union_variables, selected_union.size());
            stats_.peak_factor_entries =
                std::max(stats_.peak_factor_entries, selected_entries);
            stats_.backpointer_entries += output_entries;
            ++stats_.eliminated_variables;

            const std::size_t output_node = nodes.size();
            for (const FactorVariableId variable : selected_union) {
                std::vector<std::size_t>& incident = incidence[variable];
                std::erase_if(incident, [&](std::size_t node) {
                    return std::binary_search(
                        selected_bucket.begin(), selected_bucket.end(), node);
                });
                ++revisions[variable];
            }
            for (const std::size_t node : selected_bucket) {
                active[node] = false;
                if (nodes[node].source == nullptr) {
                    nodes[node].scores = {};
                    nodes[node].feasible = {};
                }
            }
            nodes.push_back(std::move(output));
            active.push_back(true);
            for (const FactorVariableId variable : output_variables) {
                incidence[variable].push_back(output_node);
            }
            for (const FactorVariableId variable : selected_union) {
                enqueue(variable);
            }
            traces.push_back(std::move(trace));
        }

        if (stats_.eliminated_variables != dimensions_.size()) {
            throw QStateError("Exact factor max-sum planner did not eliminate every variable");
        }

        double score = 0.0;
        for (std::size_t node = 0U; node < nodes.size(); ++node) {
            if (!active[node]) {
                continue;
            }
            if (!nodes[node].variables.empty()) {
                throw QStateError("Exact factor max-sum left an active variable");
            }
            double value = 0.0;
            if (!node_score(nodes[node], 0U, &value)) {
                return;
            }
            score += value;
            if (!std::isfinite(score)) {
                throw QStateError("Exact factor max-sum terminal score overflowed");
            }
        }

        result_.feasible = true;
        result_.score = score;
        result_.assignment.assign(dimensions_.size(), 0U);
        for (auto trace = traces.rbegin(); trace != traces.rend(); ++trace) {
            std::size_t output_index = 0U;
            std::size_t stride = 1U;
            for (const FactorVariableId variable : trace->output_variables) {
                output_index += result_.assignment[variable] * stride;
                if (stride > std::numeric_limits<std::size_t>::max() / dimensions_[variable]) {
                    throw QStateError("Exact factor max-sum backtrace stride overflowed");
                }
                stride *= dimensions_[variable];
            }
            if (output_index >= trace->choices.size() ||
                trace->choices[output_index] == no_choice) {
                throw QStateError("Exact factor max-sum backtrace reached an infeasible choice");
            }
            result_.assignment[trace->eliminated] = trace->choices[output_index];
        }
    }
};

}  // namespace qubit
