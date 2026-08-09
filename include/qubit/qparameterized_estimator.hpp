#pragma once

#include "qubit/qestimator.hpp"
#include "qubit/qtensor_rebind.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace qubit {

struct ExactParameterizedEstimatorConfig {
    TensorNetworkConfig tensor{};
    QStateConfig register_state{};
    std::size_t point_worker_count{0U};
};

struct ExactParameterizedEstimatorStats {
    std::size_t parameter_count{0U};
    std::size_t parameterized_operation_count{0U};
    std::size_t operation_count{0U};
    std::size_t observable_count{0U};
    std::size_t point_worker_limit{0U};
    ExactExecutionRoute route{ExactExecutionRoute::Register};
    TensorContractionStats tensor{};
};

class ExactParameterizedEstimatorPlan;

class ExactParameterizedEstimatorWorkspace {
public:
    ExactParameterizedEstimatorWorkspace() = default;

    [[nodiscard]] std::size_t point_count() const noexcept { return point_count_; }
    [[nodiscard]] std::size_t worker_count() const noexcept { return worker_count_; }

    [[nodiscard]] std::size_t rebind_count() const noexcept {
        std::size_t count = 0U;
        for (const TensorExpectationRebindPlan& lane : tensor_lanes_) {
            count += lane.rebind_count();
        }
        return count;
    }

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        std::size_t bytes = sizeof(*this) +
                            tensor_lanes_.capacity() * sizeof(TensorExpectationRebindPlan) +
                            tensor_workspaces_.capacity() * sizeof(TensorExpectationWorkspace) +
                            staging_.capacity() * sizeof(QComplex);
        for (const TensorExpectationRebindPlan& lane : tensor_lanes_) {
            bytes += lane.estimated_bytes();
        }
        for (const TensorExpectationWorkspace& workspace_value : tensor_workspaces_) {
            bytes += workspace_value.estimated_bytes();
        }
        return bytes;
    }

private:
    std::uint64_t token_{0U};
    std::size_t point_count_{0U};
    std::size_t worker_count_{0U};
    std::vector<TensorExpectationRebindPlan> tensor_lanes_{};
    std::vector<TensorExpectationWorkspace> tensor_workspaces_{};
    std::vector<QComplex> staging_{};

    friend class ExactParameterizedEstimatorPlan;
};

class ExactParameterizedEstimatorPlan {
public:
    ExactParameterizedEstimatorPlan(
        std::size_t qubit_count,
        std::span<const ParameterizedOperation> operations,
        std::span<const PauliObservable> observables,
        ExactParameterizedEstimatorConfig config = {})
        : qubit_count_(qubit_count),
          operations_(operations.begin(), operations.end()),
          observables_(observables.begin(), observables.end()),
          config_(config),
          register_plan_(operations),
          workspace_token_(next_workspace_token()) {
        if (qubit_count_ == 0U) {
            throw QStateError("Exact parameterized estimator requires at least one qubit");
        }
        if (observables_.empty()) {
            throw QStateError("Exact parameterized estimator requires at least one observable");
        }
        if (config_.tensor.max_contraction_entries < 2U ||
            config_.tensor.max_factors == 0U) {
            throw QStateError("Exact parameterized estimator tensor limits are invalid");
        }
        if (config_.point_worker_count > kMaxWorkers) {
            throw QStateError(
                "Exact parameterized estimator point_worker_count exceeds the supported limit");
        }

        for (const PauliObservable& observable : observables_) {
            if (observable.qubit_count() != qubit_count_) {
                throw QStateError(
                    "Exact parameterized estimator observable width does not match circuit");
            }
            std::string reason;
            if (!observable.validate(&reason)) {
                throw QStateError(
                    "Exact parameterized estimator received an invalid observable: " + reason);
            }
        }

        for (const ParameterizedOperation& templated : operations_) {
            if (templated.parameter_slot < -1 || templated.sample_slot < -1) {
                throw QStateError(
                    "Exact parameterized estimator slot indices must be -1 or nonnegative");
            }
            if (templated.sample_slot >= 0) {
                throw QStateError(
                    "Exact parameterized estimator does not accept stochastic sample slots");
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
                        "Exact parameterized estimator only parameterizes Rx, Ry, and Rz operations");
            }
            const std::size_t slot =
                static_cast<std::size_t>(templated.parameter_slot);
            parameter_count_ = std::max(parameter_count_, slot + 1U);
            ++parameterized_operation_count_;
        }

        worker_limit_ = config_.point_worker_count;
        if (worker_limit_ == 0U) {
            const unsigned int hardware = std::thread::hardware_concurrency();
            worker_limit_ = hardware == 0U ? 1U : static_cast<std::size_t>(hardware);
            worker_limit_ = std::min(worker_limit_, kMaxWorkers);
        }
        worker_limit_ = std::max<std::size_t>(1U, worker_limit_);

        const std::vector<double> zeros(parameter_count_, 0.0);
        const std::vector<Operation> initial = bind_operations(zeros);
        try {
            const TensorNetworkCircuit tensor(
                qubit_count_, initial, config_.tensor);
            tensor_prototype_.emplace(tensor, observables_);
            route_ = ExactExecutionRoute::TensorNetwork;
        } catch (const QStateError& error) {
            tensor_unavailable_reason_ = error.what();
            route_ = ExactExecutionRoute::Register;
        }
    }

    [[nodiscard]] ExactParameterizedEstimatorWorkspace workspace(
        std::size_t point_count) const {
        ExactParameterizedEstimatorWorkspace result;
        result.token_ = workspace_token_;
        result.point_count_ = point_count;
        result.worker_count_ = resolved_workers(point_count);
        result.staging_.resize(
            checked_product(point_count, observable_count(), "result shape"));

        if (route_ == ExactExecutionRoute::TensorNetwork && point_count != 0U) {
            result.tensor_lanes_.reserve(result.worker_count_);
            result.tensor_workspaces_.reserve(result.worker_count_);
            for (std::size_t lane = 0U; lane < result.worker_count_; ++lane) {
                result.tensor_lanes_.push_back(*tensor_prototype_);
                result.tensor_workspaces_.push_back(
                    result.tensor_lanes_.back().workspace());
            }
        }
        return result;
    }

    void estimate(
        std::span<const double> parameters,
        std::span<QComplex> results) const {
        ExactParameterizedEstimatorWorkspace local = workspace(1U);
        estimate(parameters, results, local);
    }

    void estimate(
        std::span<const double> parameters,
        std::span<QComplex> results,
        ExactParameterizedEstimatorWorkspace& workspace_value) const {
        if (workspace_value.point_count_ != 1U) {
            throw QStateError(
                "Exact parameterized estimator single-point workspace has the wrong point count");
        }
        estimate_points(parameters, 1U, results, workspace_value);
    }

    void estimate_points(
        std::span<const double> parameters,
        std::size_t point_count,
        std::span<QComplex> results) const {
        ExactParameterizedEstimatorWorkspace local = workspace(point_count);
        estimate_points(parameters, point_count, results, local);
    }

    void estimate_points(
        std::span<const double> parameters,
        std::size_t point_count,
        std::span<QComplex> results,
        ExactParameterizedEstimatorWorkspace& workspace_value) const {
        const std::size_t parameter_values =
            checked_product(point_count, parameter_count_, "parameter shape");
        const std::size_t result_values =
            checked_product(point_count, observable_count(), "result shape");
        if (parameters.size() != parameter_values) {
            throw QStateError(
                "Exact parameterized estimator flattened parameter shape is invalid");
        }
        if (results.size() != result_values) {
            throw QStateError(
                "Exact parameterized estimator flattened result shape is invalid");
        }
        validate_workspace(workspace_value, point_count);
        for (const double parameter : parameters) {
            if (!std::isfinite(parameter)) {
                throw QStateError(
                    "Exact parameterized estimator parameters must be finite");
            }
        }
        if (point_count == 0U) {
            return;
        }

        std::atomic<std::size_t> next{0U};
        std::atomic<bool> stop{false};
        std::mutex error_mutex;
        std::exception_ptr first_error;

        const auto execute_point = [&](std::size_t lane) {
            while (!stop.load(std::memory_order_relaxed)) {
                const std::size_t point = next.fetch_add(1U, std::memory_order_relaxed);
                if (point >= point_count) {
                    return;
                }
                try {
                    const std::span<const double> point_parameters = parameters.subspan(
                        point * parameter_count_, parameter_count_);
                    std::span<QComplex> point_results(
                        workspace_value.staging_.data() + point * observable_count(),
                        observable_count());

                    if (route_ == ExactExecutionRoute::TensorNetwork) {
                        const std::vector<Operation> bound = bind_operations(point_parameters);
                        const TensorNetworkCircuit tensor(
                            qubit_count_, bound, config_.tensor);
                        workspace_value.tensor_lanes_[lane].rebind(tensor);
                        workspace_value.tensor_lanes_[lane].expectations(
                            point_results, workspace_value.tensor_workspaces_[lane]);
                    } else {
                        OperationPlan bound = register_plan_.bind(point_parameters);
                        QRegister state(qubit_count_, config_.register_state);
                        bound.execute(state);
                        for (std::size_t observable = 0U;
                             observable < observables_.size();
                             ++observable) {
                            point_results[observable] = observables_[observable].expectation(state);
                        }
                    }
                } catch (...) {
                    {
                        std::lock_guard<std::mutex> lock(error_mutex);
                        if (first_error == nullptr) {
                            first_error = std::current_exception();
                        }
                    }
                    stop.store(true, std::memory_order_relaxed);
                    return;
                }
            }
        };

        std::vector<std::thread> threads;
        threads.reserve(workspace_value.worker_count_ > 0U
                            ? workspace_value.worker_count_ - 1U
                            : 0U);
        for (std::size_t lane = 1U; lane < workspace_value.worker_count_; ++lane) {
            threads.emplace_back(execute_point, lane);
        }
        execute_point(0U);
        for (std::thread& thread : threads) {
            thread.join();
        }
        if (first_error != nullptr) {
            std::rethrow_exception(first_error);
        }

        std::copy(
            workspace_value.staging_.begin(),
            workspace_value.staging_.end(),
            results.begin());
    }

    [[nodiscard]] std::size_t qubit_count() const noexcept { return qubit_count_; }
    [[nodiscard]] std::size_t operation_count() const noexcept { return operations_.size(); }
    [[nodiscard]] std::size_t parameter_count() const noexcept { return parameter_count_; }
    [[nodiscard]] std::size_t parameterized_operation_count() const noexcept {
        return parameterized_operation_count_;
    }
    [[nodiscard]] std::size_t observable_count() const noexcept { return observables_.size(); }
    [[nodiscard]] std::size_t point_worker_limit() const noexcept { return worker_limit_; }
    [[nodiscard]] ExactExecutionRoute route() const noexcept { return route_; }
    [[nodiscard]] const std::string& fallback_reason() const noexcept {
        return tensor_unavailable_reason_;
    }

    [[nodiscard]] ExactParameterizedEstimatorStats stats() const noexcept {
        ExactParameterizedEstimatorStats result;
        result.parameter_count = parameter_count_;
        result.parameterized_operation_count = parameterized_operation_count_;
        result.operation_count = operations_.size();
        result.observable_count = observables_.size();
        result.point_worker_limit = worker_limit_;
        result.route = route_;
        if (tensor_prototype_.has_value()) {
            result.tensor = tensor_prototype_->stats();
        }
        return result;
    }

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        std::size_t bytes = sizeof(*this) +
                            operations_.capacity() * sizeof(ParameterizedOperation) +
                            observables_.capacity() * sizeof(PauliObservable) +
                            tensor_unavailable_reason_.capacity();
        for (const PauliObservable& observable : observables_) {
            bytes += observable.terms().capacity() * sizeof(PauliTerm);
            for (const PauliTerm& term : observable.terms()) {
                bytes += term.factors.capacity() * sizeof(PauliFactor);
            }
        }
        if (tensor_prototype_.has_value()) {
            bytes += tensor_prototype_->estimated_bytes();
        }
        return bytes;
    }

private:
    static constexpr std::size_t kMaxWorkers = 128U;

    std::size_t qubit_count_{0U};
    std::vector<ParameterizedOperation> operations_{};
    std::vector<PauliObservable> observables_{};
    ExactParameterizedEstimatorConfig config_{};
    ParameterizedOperationPlan register_plan_;
    std::size_t parameter_count_{0U};
    std::size_t parameterized_operation_count_{0U};
    std::size_t worker_limit_{1U};
    ExactExecutionRoute route_{ExactExecutionRoute::Register};
    std::string tensor_unavailable_reason_{};
    std::optional<TensorExpectationRebindPlan> tensor_prototype_{};
    std::uint64_t workspace_token_{0U};

    [[nodiscard]] static std::uint64_t next_workspace_token() noexcept {
        static std::atomic<std::uint64_t> next{1U};
        return next.fetch_add(1U, std::memory_order_relaxed);
    }

    [[nodiscard]] static std::size_t checked_product(
        std::size_t first,
        std::size_t second,
        const char* label) {
        if (first != 0U && second > std::numeric_limits<std::size_t>::max() / first) {
            throw QStateError(
                std::string("Exact parameterized estimator ") + label + " overflows size_t");
        }
        return first * second;
    }

    [[nodiscard]] std::size_t resolved_workers(std::size_t point_count) const noexcept {
        if (point_count == 0U) {
            return 0U;
        }
        return std::max<std::size_t>(1U, std::min(worker_limit_, point_count));
    }

    [[nodiscard]] std::vector<Operation> bind_operations(
        std::span<const double> parameters) const {
        if (parameters.size() != parameter_count_) {
            throw QStateError(
                "Exact parameterized estimator parameter count is invalid");
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

    void validate_workspace(
        const ExactParameterizedEstimatorWorkspace& workspace_value,
        std::size_t point_count) const {
        const std::size_t worker_count = resolved_workers(point_count);
        const std::size_t staging_count =
            checked_product(point_count, observable_count(), "result shape");
        if (workspace_value.token_ != workspace_token_ ||
            workspace_value.point_count_ != point_count ||
            workspace_value.worker_count_ != worker_count ||
            workspace_value.staging_.size() != staging_count) {
            throw QStateError(
                "Exact parameterized estimator workspace does not match its plan");
        }
        if (route_ == ExactExecutionRoute::TensorNetwork) {
            if (workspace_value.tensor_lanes_.size() != worker_count ||
                workspace_value.tensor_workspaces_.size() != worker_count) {
                throw QStateError(
                    "Exact parameterized estimator tensor workspace shape is invalid");
            }
        } else if (!workspace_value.tensor_lanes_.empty() ||
                   !workspace_value.tensor_workspaces_.empty()) {
            throw QStateError(
                "Exact parameterized estimator register workspace contains tensor lanes");
        }
    }
};

}  // namespace qubit
