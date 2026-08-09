#pragma once

#include "qubit/qestimator.hpp"

#include <algorithm>
#include <array>
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
    [[nodiscard]] std::size_t rebind_count() const noexcept { return rebind_count_; }

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        std::size_t bytes = sizeof(*this) +
                            tensor_lanes_.capacity() * sizeof(TensorExpectationPlan) +
                            tensor_workspaces_.capacity() * sizeof(TensorExpectationWorkspace) +
                            tensor_parameter_values_.capacity() *
                                sizeof(std::vector<std::array<QComplex, 4>>) +
                            staging_.capacity() * sizeof(QComplex);
        for (const TensorExpectationPlan& lane : tensor_lanes_) {
            bytes += lane.estimated_bytes();
        }
        for (const TensorExpectationWorkspace& workspace_value : tensor_workspaces_) {
            bytes += workspace_value.estimated_bytes();
        }
        for (const auto& values : tensor_parameter_values_) {
            bytes += values.capacity() * sizeof(std::array<QComplex, 4>);
        }
        return bytes;
    }

private:
    std::uint64_t token_{0U};
    std::size_t point_count_{0U};
    std::size_t worker_count_{0U};
    std::size_t rebind_count_{0U};
    std::vector<TensorExpectationPlan> tensor_lanes_{};
    std::vector<TensorExpectationWorkspace> tensor_workspaces_{};
    std::vector<std::vector<std::array<QComplex, 4>>> tensor_parameter_values_{};
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

        for (std::size_t index = 0U; index < operations_.size(); ++index) {
            const ParameterizedOperation& templated = operations_[index];
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
            occurrences_.push_back(
                {index, slot, qubit_count_ + index, templated.operation.code});
        }

        worker_limit_ = config_.point_worker_count;
        if (worker_limit_ == 0U) {
            const unsigned int hardware = std::thread::hardware_concurrency();
            worker_limit_ = hardware == 0U ? 1U : static_cast<std::size_t>(hardware);
            worker_limit_ = std::min(worker_limit_, kMaxWorkers);
        }
        worker_limit_ = std::max<std::size_t>(1U, worker_limit_);

        const std::vector<double> zeros(parameter_count_, 0.0);
        const std::vector<Operation> initial_operations = bind_operations(zeros);
        try {
            const TensorNetworkCircuit initial(
                qubit_count_, initial_operations, config_.tensor);
            tensor_prototype_.emplace(initial, observables_);
            build_bindings(initial, observables_);
            certify_parameter_bindings(initial);
            route_ = ExactExecutionRoute::TensorNetwork;
        } catch (const QStateError& error) {
            tensor_unavailable_reason_ = error.what();
            tensor_prototype_.reset();
            bindings_.clear();
            factor_bindings_.clear();
            occurrence_bindings_.clear();
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
            result.tensor_parameter_values_.resize(result.worker_count_);
            for (std::size_t lane = 0U; lane < result.worker_count_; ++lane) {
                result.tensor_lanes_.push_back(*tensor_prototype_);
                result.tensor_workspaces_.push_back(
                    result.tensor_lanes_.back().workspace());
                result.tensor_parameter_values_[lane].resize(occurrences_.size());
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
        std::atomic<std::size_t> completed_rebinds{0U};
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
                        rebind_parameters(
                            workspace_value.tensor_lanes_[lane],
                            point_parameters,
                            workspace_value.tensor_parameter_values_[lane]);
                        workspace_value.tensor_lanes_[lane].expectations(
                            point_results, workspace_value.tensor_workspaces_[lane]);
                        completed_rebinds.fetch_add(1U, std::memory_order_relaxed);
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

        workspace_value.rebind_count_ +=
            completed_rebinds.load(std::memory_order_relaxed);
        std::copy(
            workspace_value.staging_.begin(),
            workspace_value.staging_.end(),
            results.begin());
    }

    [[nodiscard]] std::size_t qubit_count() const noexcept { return qubit_count_; }
    [[nodiscard]] std::size_t operation_count() const noexcept { return operations_.size(); }
    [[nodiscard]] std::size_t parameter_count() const noexcept { return parameter_count_; }
    [[nodiscard]] std::size_t parameterized_operation_count() const noexcept {
        return occurrences_.size();
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
        result.parameterized_operation_count = occurrences_.size();
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
                            occurrences_.capacity() * sizeof(Occurrence) +
                            bindings_.capacity() * sizeof(Binding) +
                            factor_bindings_.capacity() * sizeof(std::vector<std::size_t>) +
                            occurrence_bindings_.capacity() * sizeof(std::vector<std::size_t>) +
                            tensor_unavailable_reason_.capacity();
        for (const PauliObservable& observable : observables_) {
            bytes += observable.terms().capacity() * sizeof(PauliTerm);
            for (const PauliTerm& term : observable.terms()) {
                bytes += term.factors.capacity() * sizeof(PauliFactor);
            }
        }
        for (const auto& list : factor_bindings_) {
            bytes += list.capacity() * sizeof(std::size_t);
        }
        for (const auto& list : occurrence_bindings_) {
            bytes += list.capacity() * sizeof(std::size_t);
        }
        if (tensor_prototype_.has_value()) {
            bytes += tensor_prototype_->estimated_bytes();
        }
        return bytes;
    }

private:
    using VariableId = std::uint32_t;

    struct Binding {
        std::size_t term_index{0U};
        std::size_t source_index{0U};
        std::size_t factor_index{0U};
        std::size_t xor_mask{0U};
        bool bra{false};
    };

    struct Occurrence {
        std::size_t operation_index{0U};
        std::size_t parameter_slot{0U};
        std::size_t factor_index{0U};
        OperationCode code{OperationCode::Rx};
    };

    static constexpr std::size_t kMaxWorkers = 128U;

    std::size_t qubit_count_{0U};
    std::vector<ParameterizedOperation> operations_{};
    std::vector<PauliObservable> observables_{};
    ExactParameterizedEstimatorConfig config_{};
    ParameterizedOperationPlan register_plan_;
    std::size_t parameter_count_{0U};
    std::vector<Occurrence> occurrences_{};
    std::size_t worker_limit_{1U};
    ExactExecutionRoute route_{ExactExecutionRoute::Register};
    std::string tensor_unavailable_reason_{};
    std::optional<TensorExpectationPlan> tensor_prototype_{};
    std::vector<Binding> bindings_{};
    std::vector<std::vector<std::size_t>> factor_bindings_{};
    std::vector<std::vector<std::size_t>> occurrence_bindings_{};
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

    [[nodiscard]] static std::size_t factor_index(
        const TensorNetworkCircuit& circuit,
        std::span<const VariableId> variables) {
        std::size_t match = circuit.factors_.size();
        for (std::size_t index = 0U; index < circuit.factors_.size(); ++index) {
            if (circuit.factors_[index].variables.size() != variables.size()) {
                continue;
            }
            if (!std::equal(
                    variables.begin(), variables.end(),
                    circuit.factors_[index].variables.begin())) {
                continue;
            }
            if (match != circuit.factors_.size()) {
                throw QStateError(
                    "Exact parameterized estimator topology has duplicate factors");
            }
            match = index;
        }
        if (match == circuit.factors_.size()) {
            throw QStateError(
                "Exact parameterized estimator source factor was not found");
        }
        return match;
    }

    void build_bindings(
        const TensorNetworkCircuit& circuit,
        std::span<const PauliObservable> observables) {
        factor_bindings_.assign(circuit.factors_.size(), {});
        std::vector<const PauliTerm*> compiled_terms;
        for (const PauliObservable& observable : observables) {
            for (const PauliTerm& term : observable.terms()) {
                if (term.coefficient.norm2() != 0.0) {
                    compiled_terms.push_back(&term);
                }
            }
        }
        if (!tensor_prototype_.has_value() ||
            compiled_terms.size() != tensor_prototype_->terms_.size()) {
            throw QStateError(
                "Exact parameterized estimator term map does not match the tensor plan");
        }

        for (std::size_t term_index = 0U;
             term_index < tensor_prototype_->terms_.size();
             ++term_index) {
            const PauliTerm& observable_term = *compiled_terms[term_index];
            const auto& term = tensor_prototype_->terms_[term_index];
            if (term.identity) {
                continue;
            }

            std::size_t bra_count = 0U;
            for (const auto& source : term.sources) {
                const bool bra_source = !source.variables.empty() &&
                    std::all_of(
                        source.variables.begin(), source.variables.end(),
                        [&](VariableId variable) {
                            return variable >= circuit.next_variable_;
                        });
                bra_count += static_cast<std::size_t>(bra_source);
            }
            if (bra_count == 0U || term.sources.size() < bra_count * 2U) {
                throw QStateError(
                    "Exact parameterized estimator tensor source layout is invalid");
            }

            std::vector<std::uint8_t> invert(circuit.next_variable_, 0U);
            for (const PauliFactor& factor : observable_term.factors) {
                if (factor.axis != PauliAxis::X && factor.axis != PauliAxis::Y) {
                    continue;
                }
                if (factor.qubit >= circuit.current_wires_.size()) {
                    throw QStateError(
                        "Exact parameterized estimator Pauli factor is out of range");
                }
                invert[circuit.current_wires_[factor.qubit]] = 1U;
            }

            for (std::size_t local = 0U; local < bra_count; ++local) {
                const auto& ket_source = term.sources[local];
                const auto& bra_source = term.sources[bra_count + local];
                if (ket_source.variables.empty() ||
                    std::any_of(
                        ket_source.variables.begin(), ket_source.variables.end(),
                        [&](VariableId variable) {
                            return variable >= circuit.next_variable_;
                        })) {
                    throw QStateError(
                        "Exact parameterized estimator ket source layout is invalid");
                }

                const std::size_t source_factor =
                    factor_index(circuit, ket_source.variables);
                const auto& topology = circuit.factors_[source_factor].variables;
                if (bra_source.variables.size() != topology.size()) {
                    throw QStateError(
                        "Exact parameterized estimator bra source rank changed");
                }

                std::size_t xor_mask = 0U;
                for (std::size_t position = 0U; position < topology.size(); ++position) {
                    const std::uint64_t bra_variable =
                        static_cast<std::uint64_t>(topology[position]) +
                        static_cast<std::uint64_t>(circuit.next_variable_);
                    if (bra_variable > std::numeric_limits<VariableId>::max() ||
                        bra_source.variables[position] !=
                            static_cast<VariableId>(bra_variable)) {
                        throw QStateError(
                            "Exact parameterized estimator bra topology changed");
                    }
                    if (invert[topology[position]] != 0U) {
                        if (position >= std::numeric_limits<std::size_t>::digits) {
                            throw QStateError(
                                "Exact parameterized estimator bra mask overflowed");
                        }
                        xor_mask |= std::size_t{1} << position;
                    }
                }

                bindings_.push_back(
                    {term_index, local, source_factor, 0U, false});
                factor_bindings_[source_factor].push_back(bindings_.size() - 1U);
                bindings_.push_back(
                    {term_index, bra_count + local, source_factor, xor_mask, true});
                factor_bindings_[source_factor].push_back(bindings_.size() - 1U);
            }
        }
    }

    void certify_parameter_bindings(const TensorNetworkCircuit& circuit) {
        if (circuit.factors_.size() != qubit_count_ + operations_.size()) {
            throw QStateError(
                "Exact parameterized estimator circuit factor layout is not one-per-operation");
        }
        occurrence_bindings_.resize(occurrences_.size());
        for (std::size_t occurrence_index = 0U;
             occurrence_index < occurrences_.size();
             ++occurrence_index) {
            const Occurrence& occurrence = occurrences_[occurrence_index];
            if (occurrence.factor_index >= circuit.factors_.size() ||
                occurrence.factor_index >= factor_bindings_.size()) {
                throw QStateError(
                    "Exact parameterized estimator parameter factor is out of range");
            }
            const auto& factor = circuit.factors_[occurrence.factor_index];
            if (factor.variables.size() != 2U || factor.values.size() != 4U) {
                throw QStateError(
                    "Exact parameterized estimator parameter factor is not rank two");
            }
            occurrence_bindings_[occurrence_index] =
                factor_bindings_[occurrence.factor_index];
            for (const std::size_t binding_index :
                 occurrence_bindings_[occurrence_index]) {
                if (binding_index >= bindings_.size()) {
                    throw QStateError(
                        "Exact parameterized estimator parameter binding is invalid");
                }
                const Binding& binding = bindings_[binding_index];
                const auto& source = tensor_prototype_->terms_[binding.term_index]
                                         .sources[binding.source_index];
                if (source.values.size() != 4U || binding.xor_mask >= 4U) {
                    throw QStateError(
                        "Exact parameterized estimator parameter source shape changed");
                }
            }
        }
    }

    [[nodiscard]] static std::array<QComplex, 4> gate_values(
        OperationCode code,
        double parameter) {
        QMatrix2 matrix{};
        switch (code) {
            case OperationCode::Rx:
                matrix = gates::rx(parameter);
                break;
            case OperationCode::Ry:
                matrix = gates::ry(parameter);
                break;
            case OperationCode::Rz:
                matrix = gates::rz(parameter);
                break;
            default:
                throw QStateError(
                    "Exact parameterized estimator direct rebind gate is unsupported");
        }
        return matrix.values;
    }

    void rebind_parameters(
        TensorExpectationPlan& plan,
        std::span<const double> parameters,
        std::vector<std::array<QComplex, 4>>& staged_values) const {
        if (parameters.size() != parameter_count_ ||
            staged_values.size() != occurrences_.size()) {
            throw QStateError(
                "Exact parameterized estimator direct rebind shape is invalid");
        }

        for (std::size_t occurrence_index = 0U;
             occurrence_index < occurrences_.size();
             ++occurrence_index) {
            const Occurrence& occurrence = occurrences_[occurrence_index];
            staged_values[occurrence_index] = gate_values(
                occurrence.code, parameters[occurrence.parameter_slot]);
            for (const std::size_t binding_index :
                 occurrence_bindings_[occurrence_index]) {
                if (binding_index >= bindings_.size()) {
                    throw QStateError(
                        "Exact parameterized estimator direct binding is invalid");
                }
                const Binding& binding = bindings_[binding_index];
                if (binding.term_index >= plan.terms_.size() ||
                    binding.source_index >= plan.terms_[binding.term_index].sources.size() ||
                    plan.terms_[binding.term_index]
                            .sources[binding.source_index]
                            .values.size() != 4U ||
                    binding.xor_mask >= 4U) {
                    throw QStateError(
                        "Exact parameterized estimator direct source shape changed");
                }
            }
        }

        for (std::size_t occurrence_index = 0U;
             occurrence_index < occurrences_.size();
             ++occurrence_index) {
            const auto& values = staged_values[occurrence_index];
            for (const std::size_t binding_index :
                 occurrence_bindings_[occurrence_index]) {
                const Binding& binding = bindings_[binding_index];
                auto& source = plan.terms_[binding.term_index]
                                   .sources[binding.source_index]
                                   .values;
                if (!binding.bra) {
                    std::copy(values.begin(), values.end(), source.begin());
                } else {
                    for (std::size_t index = 0U; index < values.size(); ++index) {
                        source[index ^ binding.xor_mask] = values[index].conjugate();
                    }
                }
            }
        }
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
                workspace_value.tensor_workspaces_.size() != worker_count ||
                workspace_value.tensor_parameter_values_.size() != worker_count) {
                throw QStateError(
                    "Exact parameterized estimator tensor workspace shape is invalid");
            }
            for (std::size_t lane = 0U; lane < worker_count; ++lane) {
                if (workspace_value.tensor_parameter_values_[lane].size() !=
                    occurrences_.size()) {
                    throw QStateError(
                        "Exact parameterized estimator parameter workspace shape is invalid");
                }
            }
        } else if (!workspace_value.tensor_lanes_.empty() ||
                   !workspace_value.tensor_workspaces_.empty() ||
                   !workspace_value.tensor_parameter_values_.empty()) {
            throw QStateError(
                "Exact parameterized estimator register workspace contains tensor lanes");
        }
    }
};

}  // namespace qubit
