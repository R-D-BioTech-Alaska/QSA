#pragma once

#include "qubit/qbroker.hpp"

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace qubit {

struct ExactEstimatorConfig {
    ExactExecutionBrokerConfig execution{};
    std::size_t max_pauli_terms{512U};
    std::size_t tensor_worker_count{0U};
};

struct ExactEstimatorResult {
    QComplex value{};
    ExactExecutionRoute route{ExactExecutionRoute::Register};
    PauliPropagationStats pauli_stats{};
    TensorContractionStats tensor_stats{};
    std::string fallback_reason{};
};

class ExactEstimatorBatchPlan;

class ExactEstimatorBatchWorkspace {
public:
    ExactEstimatorBatchWorkspace() = default;

    [[nodiscard]] std::size_t estimated_bytes() const noexcept;

private:
    std::vector<TensorExpectationWorkspace> tensor_{};
    std::vector<std::vector<QComplex>> tensor_values_{};

    friend class ExactEstimatorBatchPlan;
};

class ExactEstimatorPlan {
public:
    ExactEstimatorPlan(
        std::size_t qubit_count,
        std::span<const Operation> operations,
        ExactEstimatorConfig config = {});

    [[nodiscard]] ExactEstimatorBatchPlan compile(
        std::span<const PauliObservable> observables) const;
    [[nodiscard]] ExactEstimatorResult estimate(
        const PauliObservable& observable) const;
    void estimate(
        std::span<const PauliObservable> observables,
        std::span<ExactEstimatorResult> results) const;

    [[nodiscard]] std::size_t qubit_count() const noexcept { return qubit_count_; }
    [[nodiscard]] std::size_t operation_count() const noexcept { return operations_.size(); }
    [[nodiscard]] std::size_t estimated_bytes() const noexcept;
    [[nodiscard]] bool tensor_available() const noexcept { return tensor_.has_value(); }
    [[nodiscard]] const ExactEstimatorConfig& config() const noexcept { return config_; }

private:
    std::size_t qubit_count_{0};
    std::vector<Operation> operations_{};
    ExactEstimatorConfig config_{};
    std::optional<PauliPropagationPlan> pauli_plan_{};
    std::optional<TensorNetworkCircuit> tensor_{};
    std::string pauli_unavailable_reason_{};
    std::string tensor_unavailable_reason_{};

    [[nodiscard]] static QComplex zero_state_expectation(
        const PauliObservable& observable) noexcept;
    [[nodiscard]] bool pauli_within_budget(
        const PauliObservable& observable) const noexcept;
    void validate_observable(const PauliObservable& observable) const;

    friend class ExactEstimatorBatchPlan;
};

class ExactEstimatorBatchPlan {
public:
    ExactEstimatorBatchPlan(
        const ExactEstimatorPlan& circuit,
        std::span<const PauliObservable> observables);

    [[nodiscard]] ExactEstimatorBatchWorkspace workspace() const;
    void estimate(std::span<ExactEstimatorResult> results) const;
    void estimate(
        std::span<ExactEstimatorResult> results,
        ExactEstimatorBatchWorkspace& workspace) const;

    [[nodiscard]] std::size_t observable_count() const noexcept {
        return results_.size();
    }
    [[nodiscard]] std::size_t tensor_observable_count() const noexcept;
    [[nodiscard]] std::size_t tensor_worker_count() const noexcept {
        return tensor_plans_.size();
    }
    [[nodiscard]] std::size_t register_observable_count() const noexcept {
        return register_indices_.size();
    }
    [[nodiscard]] std::size_t estimated_bytes() const noexcept;

private:
    std::size_t qubit_count_{0};
    QStateConfig register_state_{};
    std::vector<Operation> operations_{};
    std::vector<ExactEstimatorResult> results_{};
    std::vector<std::vector<std::size_t>> tensor_indices_{};
    std::vector<TensorExpectationPlan> tensor_plans_{};
    std::vector<std::size_t> register_indices_{};
    std::vector<PauliObservable> register_observables_{};
    std::optional<OperationPlan> register_plan_{};
};

}  // namespace qubit