#include "qubit/qestimator.hpp"

#include <algorithm>
#include <atomic>
#include <exception>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

namespace qubit {
namespace {

constexpr std::size_t kMaxEstimatorTensorWorkers = 128U;

void append_failure(std::string& destination, const char* route, const std::string& message) {
    if (!destination.empty()) {
        destination += "; ";
    }
    destination += route;
    destination += ": ";
    destination += message;
}

[[nodiscard]] std::size_t resolved_tensor_workers(
    std::size_t requested,
    std::size_t jobs) {
    if (requested > kMaxEstimatorTensorWorkers) {
        throw QStateError("Exact estimator tensor_worker_count exceeds the supported limit");
    }
    if (jobs == 0U) {
        return 0U;
    }
    if (requested == 0U) {
        const unsigned int hardware = std::thread::hardware_concurrency();
        requested = hardware == 0U ? 1U : static_cast<std::size_t>(hardware);
    }
    return std::max<std::size_t>(
        1U, std::min({requested, jobs, kMaxEstimatorTensorWorkers}));
}

}  // namespace

ExactEstimatorPlan::ExactEstimatorPlan(
    std::size_t qubit_count,
    std::span<const Operation> operations,
    ExactEstimatorConfig config)
    : qubit_count_(qubit_count),
      operations_(operations.begin(), operations.end()),
      config_(config) {
    if (qubit_count_ == 0U) {
        throw QStateError("Exact estimator requires at least one qubit");
    }
    if (config_.execution.tensor.max_contraction_entries < 2U ||
        config_.execution.tensor.max_factors == 0U) {
        throw QStateError("Exact estimator tensor limits are invalid");
    }
    if (config_.max_pauli_terms == 0U) {
        throw QStateError("Exact estimator Pauli term budget must be positive");
    }
    if (config_.tensor_worker_count > kMaxEstimatorTensorWorkers) {
        throw QStateError("Exact estimator tensor_worker_count exceeds the supported limit");
    }
    try {
        pauli_plan_.emplace(qubit_count_, operations_);
    } catch (const QStateError& error) {
        pauli_unavailable_reason_ = error.what();
    }
    try {
        tensor_.emplace(qubit_count_, operations_, config_.execution.tensor);
    } catch (const QStateError& error) {
        tensor_unavailable_reason_ = error.what();
    }
}

QComplex ExactEstimatorPlan::zero_state_expectation(
    const PauliObservable& observable) noexcept {
    QComplex result{};
    for (const PauliTerm& term : observable.terms()) {
        bool survives = true;
        for (const PauliFactor& factor : term.factors) {
            if (factor.axis != PauliAxis::Z) {
                survives = false;
                break;
            }
        }
        if (survives) {
            result += term.coefficient;
        }
    }
    return result;
}

bool ExactEstimatorPlan::pauli_within_budget(
    const PauliObservable& observable) const noexcept {
    std::size_t total_bound = 0U;
    for (const PauliTerm& term : observable.terms()) {
        if (term.coefficient.norm2() == 0.0) {
            continue;
        }
        std::vector<std::uint8_t> support(qubit_count_, 0U);
        for (const PauliFactor& factor : term.factors) {
            if (factor.axis != PauliAxis::I) {
                support[factor.qubit] = 1U;
            }
        }

        std::size_t term_bound = 1U;
        for (auto operation = operations_.rbegin(); operation != operations_.rend(); ++operation) {
            switch (operation->code) {
                case OperationCode::BitFlipTrajectory:
                case OperationCode::PhaseFlipTrajectory:
                case OperationCode::DepolarizingTrajectory:
                case OperationCode::AmplitudeDampingTrajectory:
                    return false;
                case OperationCode::Swap:
                    std::swap(support[operation->first], support[operation->second]);
                    break;
                case OperationCode::Cnot:
                case OperationCode::Cz:
                    if (support[operation->first] != 0U ||
                        support[operation->second] != 0U) {
                        support[operation->first] = 1U;
                        support[operation->second] = 1U;
                    }
                    break;
                case OperationCode::T:
                case OperationCode::Tdg:
                case OperationCode::Rx:
                case OperationCode::Ry:
                case OperationCode::Rz:
                    if (support[operation->first] != 0U) {
                        if (term_bound > config_.max_pauli_terms / 2U) {
                            return false;
                        }
                        term_bound *= 2U;
                    }
                    break;
                case OperationCode::X:
                case OperationCode::Y:
                case OperationCode::Z:
                case OperationCode::H:
                case OperationCode::S:
                case OperationCode::Sdg:
                    break;
            }
        }
        if (total_bound > config_.max_pauli_terms - term_bound) {
            return false;
        }
        total_bound += term_bound;
    }
    return total_bound <= config_.max_pauli_terms;
}

void ExactEstimatorPlan::validate_observable(const PauliObservable& observable) const {
    if (observable.qubit_count() != qubit_count_) {
        throw QStateError("Exact estimator observable width does not match circuit");
    }
    std::string reason;
    if (!observable.validate(&reason)) {
        throw QStateError("Exact estimator received an invalid observable: " + reason);
    }
}

ExactEstimatorBatchPlan ExactEstimatorPlan::compile(
    std::span<const PauliObservable> observables) const {
    return ExactEstimatorBatchPlan(*this, observables);
}

ExactEstimatorResult ExactEstimatorPlan::estimate(
    const PauliObservable& observable) const {
    ExactEstimatorResult result;
    estimate(
        std::span<const PauliObservable>(&observable, 1U),
        std::span<ExactEstimatorResult>(&result, 1U));
    return result;
}

void ExactEstimatorPlan::estimate(
    std::span<const PauliObservable> observables,
    std::span<ExactEstimatorResult> results) const {
    ExactEstimatorBatchPlan batch(*this, observables);
    ExactEstimatorBatchWorkspace workspace = batch.workspace();
    batch.estimate(results, workspace);
}

std::size_t ExactEstimatorPlan::estimated_bytes() const noexcept {
    std::size_t bytes = sizeof(*this) +
                        operations_.capacity() * sizeof(Operation) +
                        pauli_unavailable_reason_.capacity() +
                        tensor_unavailable_reason_.capacity();
    if (pauli_plan_.has_value()) {
        bytes += pauli_plan_->estimated_bytes();
    }
    if (tensor_.has_value()) {
        bytes += tensor_->estimated_bytes();
    }
    return bytes;
}

ExactEstimatorBatchPlan::ExactEstimatorBatchPlan(
    const ExactEstimatorPlan& circuit,
    std::span<const PauliObservable> observables)
    : qubit_count_(circuit.qubit_count_),
      register_state_(circuit.config_.execution.register_state),
      operations_(circuit.operations_),
      results_(observables.size()) {
    for (const PauliObservable& observable : observables) {
        circuit.validate_observable(observable);
    }

    std::vector<std::size_t> tensor_candidates;
    tensor_candidates.reserve(observables.size());
    for (std::size_t index = 0; index < observables.size(); ++index) {
        ExactEstimatorResult& result = results_[index];
        if (!circuit.pauli_plan_.has_value()) {
            append_failure(
                result.fallback_reason,
                "CausalPauli",
                circuit.pauli_unavailable_reason_.empty()
                    ? std::string("unavailable")
                    : circuit.pauli_unavailable_reason_);
            tensor_candidates.push_back(index);
            continue;
        }
        try {
            if (!circuit.pauli_within_budget(observables[index])) {
                throw QStateError("causal Pauli branch bound exceeds the estimator budget");
            }
            const std::size_t term_limit = std::min(
                observables[index].config().max_terms,
                circuit.config_.max_pauli_terms);
            PauliObservable bounded(qubit_count_, PauliPropagationConfig{term_limit});
            for (const PauliTerm& term : observables[index].terms()) {
                bounded.add_term(term.coefficient, term.factors);
            }
            const PauliObservable propagated = circuit.pauli_plan_->propagate_backward(
                bounded, &result.pauli_stats);
            result.value = ExactEstimatorPlan::zero_state_expectation(propagated);
            result.route = ExactExecutionRoute::CausalPauli;
        } catch (const QStateError& error) {
            append_failure(result.fallback_reason, "CausalPauli", error.what());
            tensor_candidates.push_back(index);
        }
    }

    if (!tensor_candidates.empty() && circuit.tensor_.has_value()) {
        const std::size_t workers = resolved_tensor_workers(
            circuit.config_.tensor_worker_count, tensor_candidates.size());
        std::vector<std::vector<std::size_t>> lane_indices(workers);
        std::vector<std::vector<PauliObservable>> lane_observables(workers);
        for (std::size_t local = 0; local < tensor_candidates.size(); ++local) {
            const std::size_t lane = local % workers;
            const std::size_t index = tensor_candidates[local];
            lane_indices[lane].push_back(index);
            lane_observables[lane].push_back(observables[index]);
        }

        std::vector<std::optional<TensorExpectationPlan>> compiled(workers);
        std::vector<std::exception_ptr> compile_errors(workers);
        const auto compile_lane = [&](std::size_t lane) {
            try {
                compiled[lane].emplace(
                    *circuit.tensor_,
                    std::span<const PauliObservable>(
                        lane_observables[lane].data(), lane_observables[lane].size()));
            } catch (...) {
                compile_errors[lane] = std::current_exception();
            }
        };

        std::vector<std::jthread> threads;
        threads.reserve(workers > 0U ? workers - 1U : 0U);
        for (std::size_t lane = 1U; lane < workers; ++lane) {
            threads.emplace_back(compile_lane, lane);
        }
        compile_lane(0U);
        for (std::jthread& thread : threads) {
            thread.join();
        }

        std::vector<std::size_t> remaining;
        tensor_plans_.reserve(workers);
        tensor_indices_.reserve(workers);
        for (std::size_t lane = 0; lane < workers; ++lane) {
            if (compile_errors[lane] != nullptr) {
                try {
                    std::rethrow_exception(compile_errors[lane]);
                } catch (const QStateError& error) {
                    for (const std::size_t index : lane_indices[lane]) {
                        append_failure(
                            results_[index].fallback_reason,
                            "TensorNetwork",
                            error.what());
                        remaining.push_back(index);
                    }
                    continue;
                }
            }
            if (!compiled[lane].has_value()) {
                throw QStateError("Exact estimator tensor lane did not produce a plan");
            }
            TensorExpectationPlan plan = std::move(*compiled[lane]);
            for (const std::size_t index : lane_indices[lane]) {
                results_[index].route = ExactExecutionRoute::TensorNetwork;
                results_[index].tensor_stats = plan.stats();
            }
            tensor_plans_.push_back(std::move(plan));
            tensor_indices_.push_back(std::move(lane_indices[lane]));
        }
        tensor_candidates = std::move(remaining);
    } else if (!tensor_candidates.empty()) {
        for (const std::size_t index : tensor_candidates) {
            append_failure(
                results_[index].fallback_reason,
                "TensorNetwork",
                circuit.tensor_unavailable_reason_.empty()
                    ? std::string("unavailable")
                    : circuit.tensor_unavailable_reason_);
        }
    }

    register_indices_ = tensor_candidates;
    if (!register_indices_.empty()) {
        register_observables_.reserve(register_indices_.size());
        for (const std::size_t index : register_indices_) {
            register_observables_.push_back(observables[index]);
            results_[index].route = ExactExecutionRoute::Register;
        }
        register_plan_.emplace(operations_);
    }
}

std::size_t ExactEstimatorBatchPlan::tensor_observable_count() const noexcept {
    std::size_t count = 0U;
    for (const auto& lane : tensor_indices_) {
        count += lane.size();
    }
    return count;
}

ExactEstimatorBatchWorkspace ExactEstimatorBatchPlan::workspace() const {
    ExactEstimatorBatchWorkspace result;
    result.tensor_.reserve(tensor_plans_.size());
    result.tensor_values_.reserve(tensor_plans_.size());
    for (const TensorExpectationPlan& plan : tensor_plans_) {
        result.tensor_.push_back(plan.workspace());
        result.tensor_values_.emplace_back(plan.observable_count());
    }
    return result;
}

void ExactEstimatorBatchPlan::estimate(
    std::span<ExactEstimatorResult> results) const {
    ExactEstimatorBatchWorkspace local = workspace();
    estimate(results, local);
}

void ExactEstimatorBatchPlan::estimate(
    std::span<ExactEstimatorResult> results,
    ExactEstimatorBatchWorkspace& workspace_value) const {
    if (results.size() != results_.size()) {
        throw QStateError("Compiled estimator result count does not match observable count");
    }
    if (workspace_value.tensor_.size() != tensor_plans_.size() ||
        workspace_value.tensor_values_.size() != tensor_plans_.size()) {
        throw QStateError("Compiled estimator workspace does not match its tensor lanes");
    }
    std::copy(results_.begin(), results_.end(), results.begin());

    if (!tensor_plans_.empty()) {
        std::atomic<bool> stop{false};
        std::mutex error_mutex;
        std::exception_ptr first_error;

        const auto execute_lane = [&](std::size_t lane) {
            if (stop.load(std::memory_order_relaxed)) {
                return;
            }
            try {
                if (workspace_value.tensor_values_[lane].size() !=
                    tensor_indices_[lane].size()) {
                    throw QStateError("Compiled estimator tensor value lane has the wrong size");
                }
                tensor_plans_[lane].expectations(
                    workspace_value.tensor_values_[lane],
                    workspace_value.tensor_[lane]);
                for (std::size_t local = 0; local < tensor_indices_[lane].size(); ++local) {
                    results[tensor_indices_[lane][local]].value =
                        workspace_value.tensor_values_[lane][local];
                }
            } catch (...) {
                {
                    std::lock_guard<std::mutex> lock(error_mutex);
                    if (first_error == nullptr) {
                        first_error = std::current_exception();
                    }
                }
                stop.store(true, std::memory_order_relaxed);
            }
        };

        std::vector<std::thread> threads;
        threads.reserve(tensor_plans_.size() - 1U);
        for (std::size_t lane = 1U; lane < tensor_plans_.size(); ++lane) {
            threads.emplace_back(execute_lane, lane);
        }
        execute_lane(0U);
        for (std::thread& thread : threads) {
            thread.join();
        }
        if (first_error != nullptr) {
            std::rethrow_exception(first_error);
        }
    }

    if (!register_indices_.empty()) {
        QRegister state(qubit_count_, register_state_);
        register_plan_->execute(state);
        for (std::size_t local = 0; local < register_indices_.size(); ++local) {
            results[register_indices_[local]].value =
                register_observables_[local].expectation(state);
        }
    }
}

std::size_t ExactEstimatorBatchWorkspace::estimated_bytes() const noexcept {
    std::size_t bytes = sizeof(*this) +
                        tensor_.capacity() * sizeof(TensorExpectationWorkspace) +
                        tensor_values_.capacity() * sizeof(std::vector<QComplex>);
    for (const TensorExpectationWorkspace& workspace : tensor_) {
        bytes += workspace.estimated_bytes();
    }
    for (const auto& values : tensor_values_) {
        bytes += values.capacity() * sizeof(QComplex);
    }
    return bytes;
}

std::size_t ExactEstimatorBatchPlan::estimated_bytes() const noexcept {
    std::size_t bytes = sizeof(*this) +
                        operations_.capacity() * sizeof(Operation) +
                        results_.capacity() * sizeof(ExactEstimatorResult) +
                        tensor_indices_.capacity() * sizeof(std::vector<std::size_t>) +
                        tensor_plans_.capacity() * sizeof(TensorExpectationPlan) +
                        register_indices_.capacity() * sizeof(std::size_t) +
                        register_observables_.capacity() * sizeof(PauliObservable);
    for (const ExactEstimatorResult& result : results_) {
        bytes += result.fallback_reason.capacity();
    }
    for (const auto& lane : tensor_indices_) {
        bytes += lane.capacity() * sizeof(std::size_t);
    }
    for (const TensorExpectationPlan& plan : tensor_plans_) {
        bytes += plan.estimated_bytes();
    }
    for (const PauliObservable& observable : register_observables_) {
        bytes += observable.terms().capacity() * sizeof(PauliTerm);
        for (const PauliTerm& term : observable.terms()) {
            bytes += term.factors.capacity() * sizeof(PauliFactor);
        }
    }
    return bytes;
}

}  // namespace qubit