#pragma once

#include "qubit/qfactor.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace qubit {

struct ExactFactorMessageCacheStats {
    std::size_t step_count{0U};
    std::size_t bound_factors{0U};
    std::size_t cached_message_entries{0U};
    std::size_t dependency_edges{0U};
    std::size_t evaluations{0U};
    std::size_t rebind_count{0U};
    std::size_t last_recomputed_steps{0U};
    std::size_t last_reused_steps{0U};
    std::size_t total_recomputed_steps{0U};
    std::size_t total_reused_steps{0U};
};

class ExactFactorMessageCache {
public:
    ExactFactorMessageCache(
        const ExactFactorPlan& plan,
        std::span<const FactorId> bound_dense_factors)
        : plan_(&plan),
          workspace_(plan.workspace(bound_dense_factors)),
          plan_rebind_count_(plan.rebind_count()) {
        if (bound_dense_factors.empty()) {
            throw QStateError("Exact factor message cache requires a bound dense factor");
        }
        plan_->validate_bound_workspace(workspace_);

        stats_.step_count = plan_->steps_.size();
        stats_.bound_factors = bound_dense_factors.size();
        std::size_t cache_entries = 0U;
        for (const ExactFactorPlan::Step& step : plan_->steps_) {
            if (step.output_entries >
                std::numeric_limits<std::size_t>::max() - stats_.cached_message_entries) {
                throw QStateError("Exact factor message cache entry count overflowed");
            }
            stats_.cached_message_entries += step.output_entries;
            if (step.inputs.size() >
                std::numeric_limits<std::size_t>::max() - stats_.dependency_edges) {
                throw QStateError("Exact factor message cache dependency count overflowed");
            }
            stats_.dependency_edges += step.inputs.size();
        }
        if (stats_.cached_message_entries >
                std::numeric_limits<std::size_t>::max() - stats_.dependency_edges) {
            throw QStateError("Exact factor message cache resource count overflowed");
        }
        cache_entries = stats_.cached_message_entries + stats_.dependency_edges;
        if (cache_entries > plan_->config_.max_compiled_index_entries) {
            throw QStateError("Exact factor message cache exceeds max_compiled_index_entries");
        }

        if (plan_->steps_.size() >
            std::numeric_limits<std::size_t>::max() - plan_->sources_.size()) {
            throw QStateError("Exact factor message cache node count overflowed");
        }
        consumers_.resize(plan_->sources_.size() + plan_->steps_.size());
        for (std::size_t step_index = 0U; step_index < plan_->steps_.size(); ++step_index) {
            for (const ExactFactorPlan::InputMap& input : plan_->steps_[step_index].inputs) {
                if (input.node >= consumers_.size()) {
                    throw QStateError("Exact factor message cache dependency is out of range");
                }
                consumers_[input.node].push_back(step_index);
            }
        }

        messages_.resize(plan_->steps_.size());
        for (std::size_t step_index = 0U; step_index < plan_->steps_.size(); ++step_index) {
            messages_[step_index].resize(plan_->steps_[step_index].output_entries);
        }
        dirty_.assign(plan_->steps_.size(), 1U);
        coordinates_.resize(plan_->stats_.peak_union_variables);
        retained_coordinates_.resize(plan_->retained_variables_.size());
    }

    void bind_dense_factor(
        FactorId factor,
        std::span<const QComplex> values) {
        validate_plan_state();
        plan_->bind_dense_factor(workspace_, factor, values);
        mark_dirty(static_cast<std::size_t>(factor));
        ++stats_.rebind_count;
    }

    [[nodiscard]] std::vector<QComplex> evaluate() {
        std::vector<QComplex> result(plan_->stats_.output_entries);
        evaluate(result);
        return result;
    }

    void evaluate(std::span<QComplex> output) {
        validate_plan_state();
        if (output.size() != plan_->stats_.output_entries) {
            throw QStateError("Exact factor message cache output size does not match its plan");
        }
        plan_->validate_bound_workspace(workspace_);

        stats_.last_recomputed_steps = 0U;
        stats_.last_reused_steps = 0U;
        for (std::size_t step_index = 0U; step_index < plan_->steps_.size(); ++step_index) {
            if (dirty_[step_index] == 0U) {
                ++stats_.last_reused_steps;
                continue;
            }
            recompute(step_index);
            dirty_[step_index] = 0U;
            ++stats_.last_recomputed_steps;
        }
        stats_.total_recomputed_steps += stats_.last_recomputed_steps;
        stats_.total_reused_steps += stats_.last_reused_steps;
        ++stats_.evaluations;

        for (std::size_t output_index = 0U; output_index < output.size(); ++output_index) {
            std::size_t remaining = output_index;
            for (std::size_t position = 0U;
                 position < plan_->retained_variables_.size();
                 ++position) {
                const std::size_t dimension =
                    plan_->dimensions_[plan_->retained_variables_[position]];
                retained_coordinates_[position] = remaining % dimension;
                remaining /= dimension;
            }

            QComplex product{1.0, 0.0};
            for (const ExactFactorPlan::TerminalMap& terminal : plan_->terminals_) {
                std::size_t local_index = 0U;
                for (std::size_t position = 0U;
                     position < terminal.retained_positions.size();
                     ++position) {
                    local_index +=
                        retained_coordinates_[terminal.retained_positions[position]] *
                        terminal.strides[position];
                }
                product *= node_value(terminal.node, local_index);
            }
            output[output_index] = product;
        }
    }

    [[nodiscard]] QComplex partition() {
        const std::vector<QComplex> values = evaluate();
        QComplex result{};
        for (const QComplex& value : values) {
            result += value;
        }
        return result;
    }

    [[nodiscard]] std::vector<QComplex> normalized_marginal() {
        std::vector<QComplex> values = evaluate();
        QComplex normalization{};
        for (const QComplex& value : values) {
            normalization += value;
        }
        if (normalization.norm2() <= std::numeric_limits<double>::min()) {
            throw QStateError("Cannot normalize an exact factor message cache with zero partition");
        }
        for (QComplex& value : values) {
            value /= normalization;
        }
        return values;
    }

    [[nodiscard]] const ExactFactorMessageCacheStats& stats() const noexcept {
        return stats_;
    }

    [[nodiscard]] const char* route_name() const noexcept {
        return "ExactFactorPersistentMessages";
    }

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        std::size_t bytes = sizeof(*this);
        if (workspace_.estimated_bytes() >= sizeof(ExactFactorWorkspace)) {
            bytes += workspace_.estimated_bytes() - sizeof(ExactFactorWorkspace);
        }
        bytes += workspace_.binding_estimated_bytes();
        bytes += messages_.capacity() * sizeof(std::vector<QComplex>);
        for (const auto& message : messages_) {
            bytes += message.capacity() * sizeof(QComplex);
        }
        bytes += consumers_.capacity() * sizeof(std::vector<std::size_t>);
        for (const auto& consumer : consumers_) {
            bytes += consumer.capacity() * sizeof(std::size_t);
        }
        bytes += dirty_.capacity() * sizeof(std::uint8_t);
        bytes += coordinates_.capacity() * sizeof(std::size_t);
        bytes += retained_coordinates_.capacity() * sizeof(std::size_t);
        return bytes;
    }

private:
    const ExactFactorPlan* plan_{nullptr};
    ExactFactorWorkspace workspace_{};
    std::vector<std::vector<QComplex>> messages_{};
    std::vector<std::vector<std::size_t>> consumers_{};
    std::vector<std::uint8_t> dirty_{};
    std::vector<std::size_t> coordinates_{};
    std::vector<std::size_t> retained_coordinates_{};
    std::size_t plan_rebind_count_{0U};
    ExactFactorMessageCacheStats stats_{};

    void validate_plan_state() const {
        if (plan_ == nullptr || plan_->rebind_count() != plan_rebind_count_) {
            throw QStateError("Exact factor message cache plan source state changed");
        }
    }

    void mark_dirty(std::size_t source) {
        if (source >= plan_->sources_.size()) {
            throw QStateError("Exact factor message cache source is out of range");
        }
        std::vector<std::size_t> pending(
            consumers_[source].begin(), consumers_[source].end());
        while (!pending.empty()) {
            const std::size_t step = pending.back();
            pending.pop_back();
            if (step >= dirty_.size()) {
                throw QStateError("Exact factor message cache dirty step is out of range");
            }
            if (dirty_[step] != 0U) {
                continue;
            }
            dirty_[step] = 1U;
            const std::size_t node = plan_->sources_.size() + step;
            pending.insert(
                pending.end(), consumers_[node].begin(), consumers_[node].end());
        }
    }

    [[nodiscard]] QComplex node_value(std::size_t node, std::size_t index) const {
        if (node < plan_->sources_.size()) {
            return plan_->bound_source_value(node, index, workspace_);
        }
        const std::size_t producer = node - plan_->sources_.size();
        if (producer >= messages_.size() || index >= messages_[producer].size()) {
            throw QStateError("Exact factor message cache node is out of range");
        }
        if (dirty_[producer] != 0U) {
            throw QStateError("Exact factor message cache reached a dirty predecessor");
        }
        return messages_[producer][index];
    }

    void recompute(std::size_t step_index) {
        const ExactFactorPlan::Step& step = plan_->steps_[step_index];
        std::vector<QComplex>& output = messages_[step_index];
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
                        product *= node_value(
                            step.inputs[input_index].node,
                            step.compiled_input_indices[input_base + input_index]);
                    }
                    sum += product;
                }
                output[output_index] = sum;
            }
            return;
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
                const std::size_t dimension =
                    plan_->dimensions_[step.union_variables[position]];
                coordinates_[position] = remaining % dimension;
                remaining /= dimension;
            }

            QComplex sum{};
            for (std::size_t selected_value = 0U;
                 selected_value < step.selected_dimension;
                 ++selected_value) {
                coordinates_[step.selected_position] = selected_value;
                QComplex product{1.0, 0.0};
                for (const ExactFactorPlan::InputMap& input : step.inputs) {
                    std::size_t local_index = 0U;
                    for (std::size_t position = 0U;
                         position < input.positions.size();
                         ++position) {
                        local_index +=
                            coordinates_[input.positions[position]] * input.strides[position];
                    }
                    product *= node_value(input.node, local_index);
                }
                sum += product;
            }
            output[output_index] = sum;
        }
    }
};

}  // namespace qubit
