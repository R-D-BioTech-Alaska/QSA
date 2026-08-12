#pragma once

#include "qubit/qfactor_chain.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace qubit {

enum class ExactFactorBrokerRoute : std::uint8_t {
    ChainTransfer = 0,
    VariableElimination = 1,
};

[[nodiscard]] inline const char* exact_factor_broker_route_name(
    ExactFactorBrokerRoute route) noexcept {
    switch (route) {
        case ExactFactorBrokerRoute::ChainTransfer:
            return "ExactFactorChainTransfer";
        case ExactFactorBrokerRoute::VariableElimination:
            return "ExactFactorVariableElimination";
    }
    return "Unknown";
}

class ExactFactorBrokerPlan;

class ExactFactorBrokerWorkspace {
public:
    ExactFactorBrokerWorkspace() = default;

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        std::size_t bytes = sizeof(*this);
        if (const auto* chain = std::get_if<ExactFactorChainWorkspace>(&workspace_)) {
            bytes += chain->estimated_bytes();
        } else if (const auto* generic = std::get_if<ExactFactorWorkspace>(&workspace_)) {
            bytes += generic->estimated_bytes();
        }
        return bytes;
    }

private:
    std::variant<std::monostate, ExactFactorChainWorkspace, ExactFactorWorkspace> workspace_{};

    friend class ExactFactorBrokerPlan;
};

class ExactFactorBrokerPlan {
public:
    ExactFactorBrokerPlan(
        const ExactFactorGraph& graph,
        std::span<const FactorVariableId> retained_variables = {})
        : retained_variables_(retained_variables.begin(), retained_variables.end()) {
        std::vector<FactorVariableId> chain_retained = retained_variables_;
        if (chain_retained.empty()) {
            chain_retained = partition_chain_separator(graph);
        }

        try {
            plan_.template emplace<1>(graph, chain_retained);
            route_ = ExactFactorBrokerRoute::ChainTransfer;
        } catch (const QStateError& error) {
            chain_rejection_ = error.what();
            plan_.template emplace<2>(graph, retained_variables_);
            route_ = ExactFactorBrokerRoute::VariableElimination;
        }
    }

    [[nodiscard]] ExactFactorBrokerWorkspace workspace() const {
        ExactFactorBrokerWorkspace result;
        if (route_ == ExactFactorBrokerRoute::ChainTransfer) {
            result.workspace_.template emplace<1>(chain_plan().workspace());
        } else {
            result.workspace_.template emplace<2>(generic_plan().workspace());
        }
        return result;
    }

    [[nodiscard]] std::vector<QComplex> evaluate() const {
        ExactFactorBrokerWorkspace local = workspace();
        return evaluate(local);
    }

    [[nodiscard]] std::vector<QComplex> evaluate(
        ExactFactorBrokerWorkspace& workspace_value) const {
        std::vector<QComplex> result(output_entries());
        evaluate(result, workspace_value);
        return result;
    }

    void evaluate(
        std::span<QComplex> output,
        ExactFactorBrokerWorkspace& workspace_value) const {
        if (output.size() != output_entries()) {
            throw QStateError("Exact factor broker output size does not match its plan");
        }
        if (route_ == ExactFactorBrokerRoute::ChainTransfer) {
            ExactFactorChainWorkspace& chain_workspace = chain_workspace_of(workspace_value);
            if (retained_variables_.empty()) {
                output[0] = chain_plan().partition(chain_workspace);
            } else {
                chain_plan().evaluate(output, chain_workspace);
            }
            return;
        }
        generic_plan().evaluate(output, generic_workspace_of(workspace_value));
    }

    [[nodiscard]] QComplex partition() const {
        ExactFactorBrokerWorkspace local = workspace();
        return partition(local);
    }

    [[nodiscard]] QComplex partition(ExactFactorBrokerWorkspace& workspace_value) const {
        if (route_ == ExactFactorBrokerRoute::ChainTransfer) {
            return chain_plan().partition(chain_workspace_of(workspace_value));
        }
        return generic_plan().partition(generic_workspace_of(workspace_value));
    }

    [[nodiscard]] std::vector<QComplex> normalized_marginal() const {
        ExactFactorBrokerWorkspace local = workspace();
        return normalized_marginal(local);
    }

    [[nodiscard]] std::vector<QComplex> normalized_marginal(
        ExactFactorBrokerWorkspace& workspace_value) const {
        if (route_ == ExactFactorBrokerRoute::ChainTransfer) {
            ExactFactorChainWorkspace& chain_workspace = chain_workspace_of(workspace_value);
            if (retained_variables_.empty()) {
                const QComplex normalization = chain_plan().partition(chain_workspace);
                if (normalization.norm2() <= std::numeric_limits<double>::min()) {
                    throw QStateError("Cannot normalize an exact factor broker with zero partition");
                }
                return {QComplex{1.0, 0.0}};
            }
            return chain_plan().normalized_marginal(chain_workspace);
        }
        return generic_plan().normalized_marginal(generic_workspace_of(workspace_value));
    }

    void rebind_dense_factor(FactorId factor, std::span<const QComplex> values) {
        if (route_ == ExactFactorBrokerRoute::ChainTransfer) {
            chain_plan().rebind_dense_factor(factor, values);
        } else {
            generic_plan().rebind_dense_factor(factor, values);
        }
    }

    void rebind_sparse_factor(
        FactorId factor,
        std::span<const FactorSparseEntry> entries) {
        if (route_ == ExactFactorBrokerRoute::ChainTransfer) {
            chain_plan().rebind_sparse_factor(factor, entries);
        } else {
            generic_plan().rebind_sparse_factor(factor, entries);
        }
    }

    [[nodiscard]] ExactFactorBrokerRoute route() const noexcept {
        return route_;
    }

    [[nodiscard]] const char* route_name() const noexcept {
        return exact_factor_broker_route_name(route_);
    }

    [[nodiscard]] const std::string& chain_rejection() const noexcept {
        return chain_rejection_;
    }

    [[nodiscard]] std::span<const FactorVariableId> retained_variables() const noexcept {
        return retained_variables_;
    }

    [[nodiscard]] std::size_t output_entries() const noexcept {
        if (retained_variables_.empty()) {
            return 1U;
        }
        if (route_ == ExactFactorBrokerRoute::ChainTransfer) {
            return chain_plan().stats().output_entries;
        }
        return generic_plan().output_entries();
    }

    [[nodiscard]] std::size_t rebind_count() const noexcept {
        if (route_ == ExactFactorBrokerRoute::ChainTransfer) {
            return chain_plan().rebind_count();
        }
        return generic_plan().rebind_count();
    }

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        std::size_t bytes = sizeof(*this) +
                            retained_variables_.capacity() * sizeof(FactorVariableId) +
                            chain_rejection_.capacity();
        if (route_ == ExactFactorBrokerRoute::ChainTransfer) {
            bytes += chain_plan().estimated_bytes();
        } else {
            bytes += generic_plan().estimated_bytes();
        }
        return bytes;
    }

private:
    std::vector<FactorVariableId> retained_variables_{};
    std::string chain_rejection_{};
    ExactFactorBrokerRoute route_{ExactFactorBrokerRoute::VariableElimination};
    std::variant<std::monostate, ExactFactorChainPlan, ExactFactorPlan> plan_{};

    [[nodiscard]] static std::vector<FactorVariableId> partition_chain_separator(
        const ExactFactorGraph& graph) {
        if (graph.factor_count() == 0U || graph.variable_count() <= graph.factor_count()) {
            return {};
        }
        const std::size_t width = graph.variable_count() - graph.factor_count() + 1U;
        if (width < 2U) {
            return {};
        }
        std::vector<FactorVariableId> retained(width - 1U);
        for (std::size_t position = 0U; position < retained.size(); ++position) {
            const std::size_t variable = graph.factor_count() + position;
            if (variable > std::numeric_limits<FactorVariableId>::max()) {
                return {};
            }
            retained[position] = static_cast<FactorVariableId>(variable);
        }
        return retained;
    }

    [[nodiscard]] ExactFactorChainPlan& chain_plan() {
        return std::get<ExactFactorChainPlan>(plan_);
    }

    [[nodiscard]] const ExactFactorChainPlan& chain_plan() const {
        return std::get<ExactFactorChainPlan>(plan_);
    }

    [[nodiscard]] ExactFactorPlan& generic_plan() {
        return std::get<ExactFactorPlan>(plan_);
    }

    [[nodiscard]] const ExactFactorPlan& generic_plan() const {
        return std::get<ExactFactorPlan>(plan_);
    }

    [[nodiscard]] static ExactFactorChainWorkspace& chain_workspace_of(
        ExactFactorBrokerWorkspace& workspace_value) {
        auto* workspace = std::get_if<ExactFactorChainWorkspace>(&workspace_value.workspace_);
        if (workspace == nullptr) {
            throw QStateError("Exact factor broker workspace route does not match ChainTransfer");
        }
        return *workspace;
    }

    [[nodiscard]] static ExactFactorWorkspace& generic_workspace_of(
        ExactFactorBrokerWorkspace& workspace_value) {
        auto* workspace = std::get_if<ExactFactorWorkspace>(&workspace_value.workspace_);
        if (workspace == nullptr) {
            throw QStateError("Exact factor broker workspace route does not match variable elimination");
        }
        return *workspace;
    }
};

}  // namespace qubit
