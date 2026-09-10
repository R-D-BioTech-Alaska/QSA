#pragma once

#include "qubit/qfactor_affine.hpp"
#include "qubit/qfactor_chain.hpp"
#include "qubit/qfactor_decision.hpp"

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
    DecisionDiagram = 1,
    VariableElimination = 2,
    AffineXOR = 3,
};

struct ExactFactorBrokerConfig {
    ExactFactorAffineConfig affine{};
    ExactFactorDecisionConfig decision{};
};

[[nodiscard]] inline const char* exact_factor_broker_route_name(
    ExactFactorBrokerRoute route) noexcept {
    switch (route) {
        case ExactFactorBrokerRoute::ChainTransfer:
            return "ExactFactorChainTransfer";
        case ExactFactorBrokerRoute::DecisionDiagram:
            return "ExactFactorDecisionDiagram";
        case ExactFactorBrokerRoute::VariableElimination:
            return "ExactFactorVariableElimination";
        case ExactFactorBrokerRoute::AffineXOR:
            return "ExactFactorAffineXOR";
    }
    return "Unknown";
}

namespace detail {

struct ExactFactorBrokerAffineWorkspace {
    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        return sizeof(*this);
    }
};

class ExactFactorBrokerAffineBinding {
public:
    ExactFactorBrokerAffineBinding(
        const ExactFactorGraph& graph,
        std::span<const FactorVariableId> retained_variables,
        ExactFactorAffineConfig config)
        : plan_(graph, retained_variables, config),
          graph_(graph),
          retained_variables_(retained_variables.begin(), retained_variables.end()),
          config_(config) {}

    [[nodiscard]] ExactFactorBrokerAffineWorkspace workspace() const noexcept {
        return {};
    }

    void evaluate(
        std::span<QComplex> output,
        ExactFactorBrokerAffineWorkspace&) const {
        plan_.evaluate(output);
    }

    [[nodiscard]] QComplex partition(
        ExactFactorBrokerAffineWorkspace&) const {
        return plan_.partition();
    }

    [[nodiscard]] std::vector<QComplex> normalized_marginal(
        ExactFactorBrokerAffineWorkspace&) const {
        return plan_.normalized_marginal();
    }

    void rebind_dense_factor(FactorId factor, std::span<const QComplex> values) {
        ExactFactorGraph next_graph = graph_;
        next_graph.set_dense_factor(factor, values);
        ExactFactorAffinePlan next_plan(
            next_graph, retained_variables_, config_);
        graph_ = std::move(next_graph);
        plan_ = std::move(next_plan);
        ++rebind_count_;
    }

    void rebind_sparse_factor(
        FactorId factor,
        std::span<const FactorSparseEntry> entries) {
        ExactFactorGraph next_graph = graph_;
        next_graph.set_sparse_factor(factor, entries);
        ExactFactorAffinePlan next_plan(
            next_graph, retained_variables_, config_);
        graph_ = std::move(next_graph);
        plan_ = std::move(next_plan);
        ++rebind_count_;
    }

    [[nodiscard]] std::size_t output_entries() const noexcept {
        return plan_.output_entries();
    }

    [[nodiscard]] std::size_t rebind_count() const noexcept {
        return rebind_count_;
    }

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        return sizeof(*this) +
               plan_.estimated_bytes() +
               graph_.estimated_bytes() +
               retained_variables_.capacity() * sizeof(FactorVariableId);
    }

private:
    ExactFactorAffinePlan plan_;
    ExactFactorGraph graph_;
    std::vector<FactorVariableId> retained_variables_{};
    ExactFactorAffineConfig config_{};
    std::size_t rebind_count_{0U};
};

struct ExactFactorBrokerDecisionWorkspace {
    ExactFactorDecisionWorkspace workspace{};
    std::size_t generation{0U};

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        return sizeof(*this) + workspace.estimated_bytes();
    }
};

class ExactFactorBrokerDecisionBinding {
public:
    ExactFactorBrokerDecisionBinding(
        const ExactFactorGraph& graph,
        std::span<const FactorVariableId> retained_variables,
        ExactFactorDecisionConfig config)
        : plan_(graph, retained_variables, config),
          graph_(graph),
          retained_variables_(retained_variables.begin(), retained_variables.end()),
          config_(config) {}

    [[nodiscard]] ExactFactorBrokerDecisionWorkspace workspace() const {
        ExactFactorBrokerDecisionWorkspace result;
        result.workspace = plan_.workspace();
        result.generation = generation_;
        return result;
    }

    void evaluate(
        std::span<QComplex> output,
        ExactFactorBrokerDecisionWorkspace& workspace_value) const {
        refresh(workspace_value);
        plan_.evaluate(output, workspace_value.workspace);
    }

    [[nodiscard]] QComplex partition(
        ExactFactorBrokerDecisionWorkspace& workspace_value) const {
        refresh(workspace_value);
        return plan_.partition(workspace_value.workspace);
    }

    [[nodiscard]] std::vector<QComplex> normalized_marginal(
        ExactFactorBrokerDecisionWorkspace& workspace_value) const {
        refresh(workspace_value);
        return plan_.normalized_marginal(workspace_value.workspace);
    }

    void rebind_dense_factor(FactorId factor, std::span<const QComplex> values) {
        ExactFactorGraph next_graph = graph_;
        next_graph.set_dense_factor(factor, values);
        ExactFactorDecisionPlan next_plan(
            next_graph, retained_variables_, config_);
        graph_ = std::move(next_graph);
        plan_ = std::move(next_plan);
        ++generation_;
        ++rebind_count_;
    }

    void rebind_sparse_factor(
        FactorId factor,
        std::span<const FactorSparseEntry> entries) {
        ExactFactorGraph next_graph = graph_;
        next_graph.set_sparse_factor(factor, entries);
        ExactFactorDecisionPlan next_plan(
            next_graph, retained_variables_, config_);
        graph_ = std::move(next_graph);
        plan_ = std::move(next_plan);
        ++generation_;
        ++rebind_count_;
    }

    [[nodiscard]] std::size_t output_entries() const noexcept {
        return plan_.output_entries();
    }

    [[nodiscard]] std::size_t rebind_count() const noexcept {
        return rebind_count_;
    }

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        return sizeof(*this) +
               plan_.estimated_bytes() +
               graph_.estimated_bytes() +
               retained_variables_.capacity() * sizeof(FactorVariableId);
    }

private:
    ExactFactorDecisionPlan plan_;
    ExactFactorGraph graph_;
    std::vector<FactorVariableId> retained_variables_{};
    ExactFactorDecisionConfig config_{};
    std::size_t generation_{0U};
    std::size_t rebind_count_{0U};

    void refresh(ExactFactorBrokerDecisionWorkspace& workspace_value) const {
        if (workspace_value.generation == generation_) {
            return;
        }
        workspace_value.workspace = plan_.workspace();
        workspace_value.generation = generation_;
    }
};

}  // namespace detail

class ExactFactorBrokerPlan;

class ExactFactorBrokerWorkspace {
public:
    ExactFactorBrokerWorkspace() = default;

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        std::size_t bytes = sizeof(*this);
        if (const auto* chain = std::get_if<ExactFactorChainWorkspace>(&workspace_)) {
            bytes += chain->estimated_bytes();
        } else if (const auto* affine =
                       std::get_if<detail::ExactFactorBrokerAffineWorkspace>(&workspace_)) {
            bytes += affine->estimated_bytes();
        } else if (const auto* decision =
                       std::get_if<detail::ExactFactorBrokerDecisionWorkspace>(&workspace_)) {
            bytes += decision->estimated_bytes();
        } else if (const auto* generic = std::get_if<ExactFactorWorkspace>(&workspace_)) {
            bytes += generic->estimated_bytes();
        }
        return bytes;
    }

private:
    std::variant<
        std::monostate,
        ExactFactorChainWorkspace,
        detail::ExactFactorBrokerAffineWorkspace,
        detail::ExactFactorBrokerDecisionWorkspace,
        ExactFactorWorkspace> workspace_{};

    friend class ExactFactorBrokerPlan;
};

class ExactFactorBrokerPlan {
public:
    ExactFactorBrokerPlan(
        const ExactFactorGraph& graph,
        std::span<const FactorVariableId> retained_variables = {},
        ExactFactorBrokerConfig config = {})
        : config_(config),
          retained_variables_(retained_variables.begin(), retained_variables.end()) {
        std::vector<FactorVariableId> chain_retained = retained_variables_;
        if (chain_retained.empty()) {
            chain_retained = partition_chain_separator(graph);
        }

        try {
            plan_.template emplace<1>(graph, chain_retained);
            route_ = ExactFactorBrokerRoute::ChainTransfer;
            return;
        } catch (const QStateError& error) {
            chain_rejection_ = error.what();
        }

        try {
            plan_.template emplace<2>(
                graph, retained_variables_, config_.affine);
            route_ = ExactFactorBrokerRoute::AffineXOR;
            return;
        } catch (const QStateError& error) {
            affine_rejection_ = error.what();
        }

        try {
            plan_.template emplace<3>(
                graph, retained_variables_, config_.decision);
            route_ = ExactFactorBrokerRoute::DecisionDiagram;
            return;
        } catch (const QStateError& error) {
            decision_rejection_ = error.what();
        }

        plan_.template emplace<4>(graph, retained_variables_);
        route_ = ExactFactorBrokerRoute::VariableElimination;
    }

    [[nodiscard]] ExactFactorBrokerWorkspace workspace() const {
        ExactFactorBrokerWorkspace result;
        switch (route_) {
            case ExactFactorBrokerRoute::ChainTransfer:
                result.workspace_.template emplace<1>(chain_plan().workspace());
                break;
            case ExactFactorBrokerRoute::AffineXOR:
                result.workspace_.template emplace<2>(affine_plan().workspace());
                break;
            case ExactFactorBrokerRoute::DecisionDiagram:
                result.workspace_.template emplace<3>(decision_plan().workspace());
                break;
            case ExactFactorBrokerRoute::VariableElimination:
                result.workspace_.template emplace<4>(generic_plan().workspace());
                break;
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
        switch (route_) {
            case ExactFactorBrokerRoute::ChainTransfer: {
                ExactFactorChainWorkspace& chain_workspace =
                    chain_workspace_of(workspace_value);
                if (retained_variables_.empty()) {
                    output[0] = chain_plan().partition(chain_workspace);
                } else {
                    chain_plan().evaluate(output, chain_workspace);
                }
                return;
            }
            case ExactFactorBrokerRoute::AffineXOR:
                affine_plan().evaluate(
                    output, affine_workspace_of(workspace_value));
                return;
            case ExactFactorBrokerRoute::DecisionDiagram:
                decision_plan().evaluate(
                    output, decision_workspace_of(workspace_value));
                return;
            case ExactFactorBrokerRoute::VariableElimination:
                generic_plan().evaluate(
                    output, generic_workspace_of(workspace_value));
                return;
        }
    }

    [[nodiscard]] QComplex partition() const {
        ExactFactorBrokerWorkspace local = workspace();
        return partition(local);
    }

    [[nodiscard]] QComplex partition(
        ExactFactorBrokerWorkspace& workspace_value) const {
        switch (route_) {
            case ExactFactorBrokerRoute::ChainTransfer:
                return chain_plan().partition(chain_workspace_of(workspace_value));
            case ExactFactorBrokerRoute::AffineXOR:
                return affine_plan().partition(affine_workspace_of(workspace_value));
            case ExactFactorBrokerRoute::DecisionDiagram:
                return decision_plan().partition(decision_workspace_of(workspace_value));
            case ExactFactorBrokerRoute::VariableElimination:
                return generic_plan().partition(generic_workspace_of(workspace_value));
        }
        throw QStateError("Exact factor broker route is invalid");
    }

    [[nodiscard]] std::vector<QComplex> normalized_marginal() const {
        ExactFactorBrokerWorkspace local = workspace();
        return normalized_marginal(local);
    }

    [[nodiscard]] std::vector<QComplex> normalized_marginal(
        ExactFactorBrokerWorkspace& workspace_value) const {
        switch (route_) {
            case ExactFactorBrokerRoute::ChainTransfer: {
                ExactFactorChainWorkspace& chain_workspace =
                    chain_workspace_of(workspace_value);
                if (retained_variables_.empty()) {
                    const QComplex normalization =
                        chain_plan().partition(chain_workspace);
                    if (normalization.norm2() <=
                        std::numeric_limits<double>::min()) {
                        throw QStateError(
                            "Cannot normalize an exact factor broker with zero partition");
                    }
                    return {QComplex{1.0, 0.0}};
                }
                return chain_plan().normalized_marginal(chain_workspace);
            }
            case ExactFactorBrokerRoute::AffineXOR:
                return affine_plan().normalized_marginal(
                    affine_workspace_of(workspace_value));
            case ExactFactorBrokerRoute::DecisionDiagram:
                return decision_plan().normalized_marginal(
                    decision_workspace_of(workspace_value));
            case ExactFactorBrokerRoute::VariableElimination:
                return generic_plan().normalized_marginal(
                    generic_workspace_of(workspace_value));
        }
        throw QStateError("Exact factor broker route is invalid");
    }

    void rebind_dense_factor(FactorId factor, std::span<const QComplex> values) {
        switch (route_) {
            case ExactFactorBrokerRoute::ChainTransfer:
                chain_plan().rebind_dense_factor(factor, values);
                break;
            case ExactFactorBrokerRoute::AffineXOR:
                affine_plan().rebind_dense_factor(factor, values);
                break;
            case ExactFactorBrokerRoute::DecisionDiagram:
                decision_plan().rebind_dense_factor(factor, values);
                break;
            case ExactFactorBrokerRoute::VariableElimination:
                generic_plan().rebind_dense_factor(factor, values);
                break;
        }
    }

    void rebind_sparse_factor(
        FactorId factor,
        std::span<const FactorSparseEntry> entries) {
        switch (route_) {
            case ExactFactorBrokerRoute::ChainTransfer:
                chain_plan().rebind_sparse_factor(factor, entries);
                break;
            case ExactFactorBrokerRoute::AffineXOR:
                affine_plan().rebind_sparse_factor(factor, entries);
                break;
            case ExactFactorBrokerRoute::DecisionDiagram:
                decision_plan().rebind_sparse_factor(factor, entries);
                break;
            case ExactFactorBrokerRoute::VariableElimination:
                generic_plan().rebind_sparse_factor(factor, entries);
                break;
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

    [[nodiscard]] const std::string& affine_rejection() const noexcept {
        return affine_rejection_;
    }

    [[nodiscard]] const std::string& decision_rejection() const noexcept {
        return decision_rejection_;
    }

    [[nodiscard]] std::span<const FactorVariableId> retained_variables() const noexcept {
        return retained_variables_;
    }

    [[nodiscard]] std::size_t output_entries() const noexcept {
        if (retained_variables_.empty()) {
            return 1U;
        }
        switch (route_) {
            case ExactFactorBrokerRoute::ChainTransfer:
                return chain_plan().stats().output_entries;
            case ExactFactorBrokerRoute::AffineXOR:
                return affine_plan().output_entries();
            case ExactFactorBrokerRoute::DecisionDiagram:
                return decision_plan().output_entries();
            case ExactFactorBrokerRoute::VariableElimination:
                return generic_plan().output_entries();
        }
        return 0U;
    }

    [[nodiscard]] std::size_t rebind_count() const noexcept {
        switch (route_) {
            case ExactFactorBrokerRoute::ChainTransfer:
                return chain_plan().rebind_count();
            case ExactFactorBrokerRoute::AffineXOR:
                return affine_plan().rebind_count();
            case ExactFactorBrokerRoute::DecisionDiagram:
                return decision_plan().rebind_count();
            case ExactFactorBrokerRoute::VariableElimination:
                return generic_plan().rebind_count();
        }
        return 0U;
    }

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        std::size_t bytes = sizeof(*this) +
                            retained_variables_.capacity() * sizeof(FactorVariableId) +
                            chain_rejection_.capacity() +
                            affine_rejection_.capacity() +
                            decision_rejection_.capacity();
        switch (route_) {
            case ExactFactorBrokerRoute::ChainTransfer:
                bytes += chain_plan().estimated_bytes();
                break;
            case ExactFactorBrokerRoute::AffineXOR:
                bytes += affine_plan().estimated_bytes();
                break;
            case ExactFactorBrokerRoute::DecisionDiagram:
                bytes += decision_plan().estimated_bytes();
                break;
            case ExactFactorBrokerRoute::VariableElimination:
                bytes += generic_plan().estimated_bytes();
                break;
        }
        return bytes;
    }

private:
    ExactFactorBrokerConfig config_{};
    std::vector<FactorVariableId> retained_variables_{};
    std::string chain_rejection_{};
    std::string affine_rejection_{};
    std::string decision_rejection_{};
    ExactFactorBrokerRoute route_{ExactFactorBrokerRoute::VariableElimination};
    std::variant<
        std::monostate,
        ExactFactorChainPlan,
        detail::ExactFactorBrokerAffineBinding,
        detail::ExactFactorBrokerDecisionBinding,
        ExactFactorPlan> plan_{};

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

    [[nodiscard]] detail::ExactFactorBrokerAffineBinding& affine_plan() {
        return std::get<detail::ExactFactorBrokerAffineBinding>(plan_);
    }

    [[nodiscard]] const detail::ExactFactorBrokerAffineBinding& affine_plan() const {
        return std::get<detail::ExactFactorBrokerAffineBinding>(plan_);
    }

    [[nodiscard]] detail::ExactFactorBrokerDecisionBinding& decision_plan() {
        return std::get<detail::ExactFactorBrokerDecisionBinding>(plan_);
    }

    [[nodiscard]] const detail::ExactFactorBrokerDecisionBinding& decision_plan() const {
        return std::get<detail::ExactFactorBrokerDecisionBinding>(plan_);
    }

    [[nodiscard]] ExactFactorPlan& generic_plan() {
        return std::get<ExactFactorPlan>(plan_);
    }

    [[nodiscard]] const ExactFactorPlan& generic_plan() const {
        return std::get<ExactFactorPlan>(plan_);
    }

    [[nodiscard]] static ExactFactorChainWorkspace& chain_workspace_of(
        ExactFactorBrokerWorkspace& workspace_value) {
        auto* workspace =
            std::get_if<ExactFactorChainWorkspace>(&workspace_value.workspace_);
        if (workspace == nullptr) {
            throw QStateError(
                "Exact factor broker workspace route does not match ChainTransfer");
        }
        return *workspace;
    }

    [[nodiscard]] static detail::ExactFactorBrokerAffineWorkspace& affine_workspace_of(
        ExactFactorBrokerWorkspace& workspace_value) {
        auto* workspace = std::get_if<detail::ExactFactorBrokerAffineWorkspace>(
            &workspace_value.workspace_);
        if (workspace == nullptr) {
            throw QStateError(
                "Exact factor broker workspace route does not match AffineXOR");
        }
        return *workspace;
    }

    [[nodiscard]] static detail::ExactFactorBrokerDecisionWorkspace& decision_workspace_of(
        ExactFactorBrokerWorkspace& workspace_value) {
        auto* workspace = std::get_if<detail::ExactFactorBrokerDecisionWorkspace>(
            &workspace_value.workspace_);
        if (workspace == nullptr) {
            throw QStateError(
                "Exact factor broker workspace route does not match DecisionDiagram");
        }
        return *workspace;
    }

    [[nodiscard]] static ExactFactorWorkspace& generic_workspace_of(
        ExactFactorBrokerWorkspace& workspace_value) {
        auto* workspace = std::get_if<ExactFactorWorkspace>(&workspace_value.workspace_);
        if (workspace == nullptr) {
            throw QStateError(
                "Exact factor broker workspace route does not match variable elimination");
        }
        return *workspace;
    }
};

}  // namespace qubit
