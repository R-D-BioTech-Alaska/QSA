#pragma once

#include "qubit/qtensor_adjoint.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace qubit {

struct ExactCausalAdjointGradientConfig {
    ExactAdjointGradientConfig adjoint{};
};

struct ExactCausalAdjointGradientStats {
    std::size_t parameter_count{0U};
    std::size_t parameterized_operation_count{0U};
    std::size_t observable_count{0U};
    std::size_t dynamic_term_count{0U};
    std::size_t static_term_count{0U};
    std::size_t dynamic_observable_count{0U};
    std::size_t static_contribution_observable_count{0U};
    ExactAdjointGradientStats dynamic{};
};

class ExactCausalAdjointGradientPlan;

class ExactCausalAdjointGradientWorkspace {
public:
    ExactCausalAdjointGradientWorkspace() = default;

    [[nodiscard]] std::size_t rebind_count() const noexcept {
        return dynamic_workspace_.has_value()
            ? dynamic_workspace_->rebind_count()
            : 0U;
    }

    [[nodiscard]] std::size_t execution_worker_count() const noexcept {
        return dynamic_workspace_.has_value()
            ? dynamic_workspace_->execution_worker_count()
            : 1U;
    }

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        std::size_t bytes = sizeof(*this) +
                            dynamic_values_.capacity() * sizeof(QComplex) +
                            dynamic_gradients_.capacity() * sizeof(QComplex) +
                            values_.capacity() * sizeof(QComplex) +
                            gradients_.capacity() * sizeof(QComplex);
        if (dynamic_workspace_.has_value()) {
            bytes += dynamic_workspace_->estimated_bytes();
        }
        return bytes;
    }

private:
    std::uint64_t token_{0U};
    std::optional<ExactAdjointGradientWorkspace> dynamic_workspace_{};
    std::vector<QComplex> dynamic_values_{};
    std::vector<QComplex> dynamic_gradients_{};
    std::vector<QComplex> values_{};
    std::vector<QComplex> gradients_{};

    friend class ExactCausalAdjointGradientPlan;
};

class ExactCausalAdjointGradientPlan {
public:
    ExactCausalAdjointGradientPlan(
        std::size_t qubit_count,
        std::span<const ParameterizedOperation> operations,
        std::span<const PauliObservable> observables,
        ExactCausalAdjointGradientConfig config = {})
        : qubit_count_(qubit_count),
          operations_(operations.begin(), operations.end()),
          observable_count_(observables.size()),
          config_(config),
          workspace_token_(next_workspace_token()) {
        if (qubit_count_ == 0U) {
            throw QStateError("Exact causal adjoint gradient requires at least one qubit");
        }
        if (observables.empty()) {
            throw QStateError("Exact causal adjoint gradient requires at least one observable");
        }
        if (config_.adjoint.worker_count > kAdjointWorkerLimit) {
            throw QStateError(
                "Exact causal adjoint gradient worker_count exceeds the supported limit");
        }

        for (const PauliObservable& observable : observables) {
            if (observable.qubit_count() != qubit_count_) {
                throw QStateError(
                    "Exact causal adjoint gradient observable width does not match circuit");
            }
            std::string reason;
            if (!observable.validate(&reason)) {
                throw QStateError(
                    "Exact causal adjoint gradient received an invalid observable: " + reason);
            }
        }

        validate_parameterized_operations();
        const std::vector<double> zeros(parameter_count_, 0.0);
        const std::vector<Operation> initial_operations = bind_operations(zeros);
        const TensorNetworkCircuit initial(
            qubit_count_, initial_operations, config_.adjoint.tensor);

        std::vector<PauliObservable> dynamic_observables;
        std::vector<PauliObservable> static_observables;
        dynamic_observables.reserve(observables.size());
        static_observables.reserve(observables.size());

        for (const PauliObservable& observable : observables) {
            PauliObservable dynamic(qubit_count_, observable.config());
            PauliObservable cached(qubit_count_, observable.config());
            bool observable_dynamic = false;
            bool observable_static = false;
            for (const PauliTerm& term : observable.terms()) {
                if (term.coefficient.norm2() == 0.0) {
                    continue;
                }
                if (term_depends_on_parameters(term)) {
                    dynamic.add_term(term.coefficient, term.factors);
                    ++dynamic_term_count_;
                    observable_dynamic = true;
                } else {
                    cached.add_term(term.coefficient, term.factors);
                    ++static_term_count_;
                    observable_static = true;
                }
            }
            if (observable_dynamic) {
                ++dynamic_observable_count_;
            }
            if (observable_static) {
                ++static_contribution_observable_count_;
            }
            dynamic_observables.push_back(std::move(dynamic));
            static_observables.push_back(std::move(cached));
        }

        static_values_.assign(observable_count_, QComplex{});
        if (static_term_count_ != 0U) {
            TensorExpectationPlan static_plan(initial, static_observables);
            TensorExpectationWorkspace static_workspace = static_plan.workspace();
            static_plan.expectations(static_values_, static_workspace);
        }

        if (dynamic_term_count_ != 0U) {
            dynamic_plan_ = std::make_unique<ExactAdjointGradientPlan>(
                qubit_count_, operations_, dynamic_observables, config_.adjoint);
        }
    }

    [[nodiscard]] ExactCausalAdjointGradientWorkspace workspace() const {
        return workspace(worker_count());
    }

    [[nodiscard]] ExactCausalAdjointGradientWorkspace workspace(
        std::size_t execution_worker_count) const {
        ExactCausalAdjointGradientWorkspace result;
        result.token_ = workspace_token_;
        result.dynamic_values_.resize(observable_count_);
        result.dynamic_gradients_.resize(observable_count_ * parameter_count_);
        result.values_.resize(observable_count_);
        result.gradients_.resize(observable_count_ * parameter_count_);

        if (dynamic_plan_) {
            if (execution_worker_count == 0U ||
                execution_worker_count > dynamic_plan_->worker_count()) {
                throw QStateError(
                    "Exact causal adjoint gradient workspace worker count is invalid");
            }
            result.dynamic_workspace_.emplace(
                dynamic_plan_->workspace(execution_worker_count));
        } else if (execution_worker_count != 1U) {
            throw QStateError(
                "Exact causal adjoint gradient static-only workspace uses one worker");
        }
        return result;
    }

    void value_and_gradient(
        std::span<const double> parameters,
        std::span<QComplex> values_result,
        std::span<QComplex> gradients) const {
        ExactCausalAdjointGradientWorkspace local = workspace();
        value_and_gradient(parameters, values_result, gradients, local);
    }

    void value_and_gradient(
        std::span<const double> parameters,
        std::span<QComplex> values_result,
        std::span<QComplex> gradients,
        ExactCausalAdjointGradientWorkspace& workspace_value) const {
        validate_parameters(parameters);
        validate_workspace(workspace_value);
        if (values_result.size() != observable_count_) {
            throw QStateError(
                "Exact causal adjoint gradient value result count is invalid");
        }
        if (gradients.size() != observable_count_ * parameter_count_) {
            throw QStateError(
                "Exact causal adjoint gradient result count is invalid");
        }

        if (dynamic_plan_) {
            dynamic_plan_->value_and_gradient(
                parameters,
                workspace_value.dynamic_values_,
                workspace_value.dynamic_gradients_,
                *workspace_value.dynamic_workspace_);
        } else {
            std::fill(
                workspace_value.dynamic_values_.begin(),
                workspace_value.dynamic_values_.end(),
                QComplex{});
            std::fill(
                workspace_value.dynamic_gradients_.begin(),
                workspace_value.dynamic_gradients_.end(),
                QComplex{});
        }

        for (std::size_t observable = 0U;
             observable < observable_count_;
             ++observable) {
            workspace_value.values_[observable] =
                static_values_[observable] + workspace_value.dynamic_values_[observable];
        }
        workspace_value.gradients_ = workspace_value.dynamic_gradients_;

        std::copy(
            workspace_value.values_.begin(), workspace_value.values_.end(),
            values_result.begin());
        std::copy(
            workspace_value.gradients_.begin(), workspace_value.gradients_.end(),
            gradients.begin());
    }

    [[nodiscard]] std::size_t gradient_index(
        std::size_t observable,
        std::size_t parameter) const {
        if (observable >= observable_count_ || parameter >= parameter_count_) {
            throw QStateError("Exact causal adjoint gradient index is out of range");
        }
        return observable * parameter_count_ + parameter;
    }

    [[nodiscard]] std::size_t qubit_count() const noexcept { return qubit_count_; }
    [[nodiscard]] std::size_t operation_count() const noexcept { return operations_.size(); }
    [[nodiscard]] std::size_t parameter_count() const noexcept { return parameter_count_; }
    [[nodiscard]] std::size_t parameterized_operation_count() const noexcept {
        return parameterized_operation_count_;
    }
    [[nodiscard]] std::size_t observable_count() const noexcept { return observable_count_; }
    [[nodiscard]] std::size_t worker_count() const noexcept {
        return dynamic_plan_ ? dynamic_plan_->worker_count() : 1U;
    }

    [[nodiscard]] ExactCausalAdjointGradientStats stats() const noexcept {
        ExactCausalAdjointGradientStats result;
        result.parameter_count = parameter_count_;
        result.parameterized_operation_count = parameterized_operation_count_;
        result.observable_count = observable_count_;
        result.dynamic_term_count = dynamic_term_count_;
        result.static_term_count = static_term_count_;
        result.dynamic_observable_count = dynamic_observable_count_;
        result.static_contribution_observable_count =
            static_contribution_observable_count_;
        if (dynamic_plan_) {
            result.dynamic = dynamic_plan_->stats();
        } else {
            result.dynamic.parameter_count = parameter_count_;
            result.dynamic.parameterized_operation_count =
                parameterized_operation_count_;
            result.dynamic.observable_count = observable_count_;
            result.dynamic.parameter_shift_equivalent_evaluations =
                1U + 2U * parameterized_operation_count_;
            result.dynamic.worker_count = 1U;
        }
        return result;
    }

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        std::size_t bytes = sizeof(*this) +
                            operations_.capacity() * sizeof(ParameterizedOperation) +
                            static_values_.capacity() * sizeof(QComplex);
        if (dynamic_plan_) {
            bytes += dynamic_plan_->estimated_bytes();
        }
        return bytes;
    }

private:
    static constexpr std::size_t kAdjointWorkerLimit = 32U;

    std::size_t qubit_count_{0U};
    std::vector<ParameterizedOperation> operations_{};
    std::size_t observable_count_{0U};
    ExactCausalAdjointGradientConfig config_{};
    std::size_t parameter_count_{0U};
    std::size_t parameterized_operation_count_{0U};
    std::vector<QComplex> static_values_{};
    std::unique_ptr<ExactAdjointGradientPlan> dynamic_plan_{};
    std::size_t dynamic_term_count_{0U};
    std::size_t static_term_count_{0U};
    std::size_t dynamic_observable_count_{0U};
    std::size_t static_contribution_observable_count_{0U};
    std::uint64_t workspace_token_{0U};

    [[nodiscard]] static std::uint64_t next_workspace_token() noexcept {
        static std::atomic<std::uint64_t> next{1U};
        return next.fetch_add(1U, std::memory_order_relaxed);
    }

    void validate_parameterized_operations() {
        for (const ParameterizedOperation& templated : operations_) {
            if (templated.parameter_slot < -1 || templated.sample_slot < -1) {
                throw QStateError(
                    "Exact causal adjoint gradient slot indices must be -1 or nonnegative");
            }
            if (templated.sample_slot >= 0) {
                throw QStateError(
                    "Exact causal adjoint gradient does not accept stochastic sample slots");
            }
            switch (templated.operation.code) {
                case OperationCode::BitFlipTrajectory:
                case OperationCode::PhaseFlipTrajectory:
                case OperationCode::DepolarizingTrajectory:
                case OperationCode::AmplitudeDampingTrajectory:
                    throw QStateError(
                        "Exact causal adjoint gradient does not accept trajectory noise");
                default:
                    break;
            }
            if (templated.parameter_slot < 0) {
                continue;
            }
            switch (templated.operation.code) {
                case OperationCode::Rx:
                case OperationCode::Ry:
                case OperationCode::Rz:
                    break;
                default:
                    throw QStateError(
                        "Exact causal adjoint gradient only parameterizes Rx, Ry, and Rz operations");
            }
            parameter_count_ = std::max(
                parameter_count_,
                static_cast<std::size_t>(templated.parameter_slot) + 1U);
            ++parameterized_operation_count_;
        }
    }

    [[nodiscard]] std::vector<Operation> bind_operations(
        std::span<const double> parameters) const {
        if (parameters.size() != parameter_count_) {
            throw QStateError(
                "Exact causal adjoint gradient parameter count is invalid");
        }
        std::vector<Operation> bound;
        bound.reserve(operations_.size());
        for (const ParameterizedOperation& templated : operations_) {
            Operation operation = templated.operation;
            if (templated.parameter_slot >= 0) {
                operation.parameter =
                    parameters[static_cast<std::size_t>(templated.parameter_slot)];
            }
            bound.push_back(operation);
        }
        return bound;
    }

    [[nodiscard]] bool term_depends_on_parameters(const PauliTerm& term) const {
        std::vector<std::uint8_t> support(qubit_count_, 0U);
        for (const PauliFactor& factor : term.factors) {
            if (factor.qubit >= qubit_count_) {
                throw QStateError(
                    "Exact causal adjoint gradient Pauli factor is out of range");
            }
            support[factor.qubit] = 1U;
        }

        for (auto position = operations_.rbegin();
             position != operations_.rend();
             ++position) {
            const ParameterizedOperation& templated = *position;
            const Operation& operation = templated.operation;

            if (templated.parameter_slot >= 0 && support[operation.first] != 0U) {
                return true;
            }

            switch (operation.code) {
                case OperationCode::Cnot:
                case OperationCode::Cz:
                    if (support[operation.first] != 0U ||
                        support[operation.second] != 0U) {
                        support[operation.first] = 1U;
                        support[operation.second] = 1U;
                    }
                    break;
                case OperationCode::Swap:
                    std::swap(support[operation.first], support[operation.second]);
                    break;
                default:
                    break;
            }
        }
        return false;
    }

    void validate_parameters(std::span<const double> parameters) const {
        if (parameters.size() != parameter_count_) {
            throw QStateError(
                "Exact causal adjoint gradient parameter count is invalid");
        }
        for (const double parameter : parameters) {
            if (!std::isfinite(parameter)) {
                throw QStateError(
                    "Exact causal adjoint gradient parameters must be finite");
            }
        }
    }

    void validate_workspace(
        const ExactCausalAdjointGradientWorkspace& workspace_value) const {
        if (workspace_value.token_ != workspace_token_ ||
            workspace_value.dynamic_values_.size() != observable_count_ ||
            workspace_value.dynamic_gradients_.size() !=
                observable_count_ * parameter_count_ ||
            workspace_value.values_.size() != observable_count_ ||
            workspace_value.gradients_.size() !=
                observable_count_ * parameter_count_) {
            throw QStateError(
                "Exact causal adjoint gradient workspace does not match its plan");
        }
        if (dynamic_plan_) {
            if (!workspace_value.dynamic_workspace_.has_value()) {
                throw QStateError(
                    "Exact causal adjoint gradient workspace is missing dynamic state");
            }
        } else if (workspace_value.dynamic_workspace_.has_value()) {
            throw QStateError(
                "Exact causal adjoint gradient static workspace contains dynamic state");
        }
    }
};

}  // namespace qubit
