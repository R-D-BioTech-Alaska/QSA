#pragma once

#include "qubit/qtensor.hpp"

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
#include <thread>
#include <utility>
#include <vector>

namespace qubit {

class TensorExpectationRebindPlan {
public:
    TensorExpectationRebindPlan(
        const TensorNetworkCircuit& circuit,
        const PauliObservable& observable)
        : TensorExpectationRebindPlan(
              circuit,
              std::span<const PauliObservable>(&observable, 1U)) {}

    TensorExpectationRebindPlan(
        const TensorNetworkCircuit& circuit,
        std::span<const PauliObservable> observables)
        : plan_(circuit, observables),
          qubit_count_(circuit.qubit_count_),
          operation_count_(circuit.operation_count_),
          config_(circuit.config_),
          next_variable_(circuit.next_variable_),
          current_wires_(circuit.current_wires_) {
        factor_variables_.reserve(circuit.factors_.size());
        factor_value_sizes_.reserve(circuit.factors_.size());
        for (const auto& factor : circuit.factors_) {
            factor_variables_.push_back(factor.variables);
            factor_value_sizes_.push_back(factor.values.size());
        }
        build_bindings(circuit, observables);
    }

    void rebind(const TensorNetworkCircuit& circuit) {
        validate_topology(circuit);

        struct Replacement {
            std::size_t term_index{0U};
            std::size_t source_index{0U};
            std::vector<QComplex> values{};
        };

        std::vector<Replacement> replacements;
        replacements.reserve(bindings_.size());
        for (const Binding& binding : bindings_) {
            const auto& factor = circuit.factors_[binding.factor_index];
            const auto& source =
                plan_.terms_[binding.term_index].sources[binding.source_index];
            if (factor.values.size() != source.values.size()) {
                throw QStateError("Tensor expectation rebind source size changed");
            }

            Replacement replacement;
            replacement.term_index = binding.term_index;
            replacement.source_index = binding.source_index;
            replacement.values.resize(factor.values.size());
            if (!binding.bra) {
                replacement.values = factor.values;
            } else {
                for (std::size_t index = 0; index < factor.values.size(); ++index) {
                    const std::size_t target = index ^ binding.xor_mask;
                    if (target >= replacement.values.size()) {
                        throw QStateError("Tensor expectation rebind bra permutation is invalid");
                    }
                    replacement.values[target] = factor.values[index].conjugate();
                }
            }
            replacements.push_back(std::move(replacement));
        }

        for (Replacement& replacement : replacements) {
            plan_.terms_[replacement.term_index]
                .sources[replacement.source_index]
                .values = std::move(replacement.values);
        }
        ++rebind_count_;
    }

    [[nodiscard]] TensorExpectationWorkspace workspace() const {
        return plan_.workspace();
    }

    [[nodiscard]] QComplex expectation(
        TensorContractionStats* stats = nullptr) const {
        return plan_.expectation(stats);
    }

    [[nodiscard]] QComplex expectation(
        TensorExpectationWorkspace& workspace_value,
        TensorContractionStats* stats = nullptr) const {
        return plan_.expectation(workspace_value, stats);
    }

    void expectations(
        std::span<QComplex> results,
        TensorContractionStats* stats = nullptr) const {
        plan_.expectations(results, stats);
    }

    void expectations(
        std::span<QComplex> results,
        TensorExpectationWorkspace& workspace_value,
        TensorContractionStats* stats = nullptr) const {
        plan_.expectations(results, workspace_value, stats);
    }

    [[nodiscard]] std::size_t qubit_count() const noexcept {
        return qubit_count_;
    }

    [[nodiscard]] std::size_t observable_count() const noexcept {
        return plan_.observable_count();
    }

    [[nodiscard]] std::size_t term_count() const noexcept {
        return plan_.term_count();
    }

    [[nodiscard]] std::size_t step_count() const noexcept {
        return plan_.step_count();
    }

    [[nodiscard]] std::size_t rebind_count() const noexcept {
        return rebind_count_;
    }

    [[nodiscard]] const TensorContractionStats& stats() const noexcept {
        return plan_.stats();
    }

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        std::size_t bytes = sizeof(*this) + plan_.estimated_bytes() +
                            current_wires_.capacity() * sizeof(VariableId) +
                            factor_variables_.capacity() * sizeof(std::vector<VariableId>) +
                            factor_value_sizes_.capacity() * sizeof(std::size_t) +
                            bindings_.capacity() * sizeof(Binding);
        for (const auto& variables : factor_variables_) {
            bytes += variables.capacity() * sizeof(VariableId);
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

    TensorExpectationPlan plan_;
    std::size_t qubit_count_{0U};
    std::size_t operation_count_{0U};
    TensorNetworkConfig config_{};
    VariableId next_variable_{0U};
    std::vector<VariableId> current_wires_{};
    std::vector<std::vector<VariableId>> factor_variables_{};
    std::vector<std::size_t> factor_value_sizes_{};
    std::vector<Binding> bindings_{};
    std::size_t rebind_count_{0U};

    [[nodiscard]] std::size_t factor_index(
        const TensorNetworkCircuit& circuit,
        std::span<const VariableId> variables) const {
        std::size_t match = circuit.factors_.size();
        for (std::size_t index = 0; index < circuit.factors_.size(); ++index) {
            if (circuit.factors_[index].variables.size() != variables.size()) {
                continue;
            }
            if (!std::equal(
                    variables.begin(), variables.end(),
                    circuit.factors_[index].variables.begin())) {
                continue;
            }
            if (match != circuit.factors_.size()) {
                throw QStateError("Tensor expectation rebind topology has duplicate factors");
            }
            match = index;
        }
        if (match == circuit.factors_.size()) {
            throw QStateError("Tensor expectation rebind source factor was not found");
        }
        return match;
    }

    void build_bindings(
        const TensorNetworkCircuit& circuit,
        std::span<const PauliObservable> observables) {
        std::vector<const PauliTerm*> compiled_terms;
        for (const PauliObservable& observable : observables) {
            for (const PauliTerm& term : observable.terms()) {
                if (term.coefficient.norm2() != 0.0) {
                    compiled_terms.push_back(&term);
                }
            }
        }
        if (compiled_terms.size() != plan_.terms_.size()) {
            throw QStateError("Tensor expectation rebind term map does not match the plan");
        }

        for (std::size_t term_index = 0;
             term_index < plan_.terms_.size();
             ++term_index) {
            const PauliTerm& observable_term = *compiled_terms[term_index];
            const auto& term = plan_.terms_[term_index];
            if (term.identity) {
                if (!term.sources.empty()) {
                    throw QStateError("Tensor expectation identity term has dynamic sources");
                }
                continue;
            }

            std::size_t bra_count = 0U;
            for (const auto& source : term.sources) {
                const bool bra_source = !source.variables.empty() &&
                    std::all_of(
                        source.variables.begin(), source.variables.end(),
                        [&](VariableId variable) { return variable >= next_variable_; });
                bra_count += static_cast<std::size_t>(bra_source);
            }
            if (bra_count == 0U || term.sources.size() < bra_count * 2U) {
                throw QStateError("Tensor expectation rebind source layout is invalid");
            }

            std::vector<std::uint8_t> invert(next_variable_, 0U);
            for (const PauliFactor& factor : observable_term.factors) {
                if (factor.axis != PauliAxis::X && factor.axis != PauliAxis::Y) {
                    continue;
                }
                if (factor.qubit >= current_wires_.size()) {
                    throw QStateError("Tensor expectation rebind Pauli factor is out of range");
                }
                invert[current_wires_[factor.qubit]] = 1U;
            }

            for (std::size_t local = 0U; local < bra_count; ++local) {
                const auto& ket_source = term.sources[local];
                const auto& bra_source = term.sources[bra_count + local];
                if (ket_source.variables.empty() ||
                    std::any_of(
                        ket_source.variables.begin(), ket_source.variables.end(),
                        [&](VariableId variable) { return variable >= next_variable_; })) {
                    throw QStateError("Tensor expectation rebind ket source layout is invalid");
                }

                const std::size_t source_factor =
                    factor_index(circuit, ket_source.variables);
                const auto& topology = circuit.factors_[source_factor].variables;
                if (bra_source.variables.size() != topology.size()) {
                    throw QStateError("Tensor expectation rebind bra source rank changed");
                }

                std::size_t xor_mask = 0U;
                for (std::size_t position = 0U; position < topology.size(); ++position) {
                    const std::uint64_t bra_variable =
                        static_cast<std::uint64_t>(topology[position]) +
                        static_cast<std::uint64_t>(next_variable_);
                    if (bra_variable > std::numeric_limits<VariableId>::max() ||
                        bra_source.variables[position] !=
                            static_cast<VariableId>(bra_variable)) {
                        throw QStateError("Tensor expectation rebind bra topology changed");
                    }
                    if (invert[topology[position]] != 0U) {
                        if (position >= std::numeric_limits<std::size_t>::digits) {
                            throw QStateError("Tensor expectation rebind bra mask overflowed");
                        }
                        xor_mask |= std::size_t{1} << position;
                    }
                }

                bindings_.push_back(
                    {term_index, local, source_factor, 0U, false});
                bindings_.push_back(
                    {term_index, bra_count + local, source_factor, xor_mask, true});
            }
        }
    }

    void validate_topology(const TensorNetworkCircuit& circuit) const {
        std::string reason;
        if (!circuit.validate(&reason)) {
            throw QStateError(
                "Cannot rebind invalid tensor network: " + reason);
        }
        if (circuit.qubit_count_ != qubit_count_ ||
            circuit.operation_count_ != operation_count_ ||
            circuit.next_variable_ != next_variable_ ||
            circuit.current_wires_ != current_wires_ ||
            circuit.factors_.size() != factor_variables_.size() ||
            circuit.config_.max_contraction_entries != config_.max_contraction_entries ||
            circuit.config_.max_factors != config_.max_factors) {
            throw QStateError("Tensor expectation rebind topology does not match its plan");
        }
        for (std::size_t index = 0; index < circuit.factors_.size(); ++index) {
            if (circuit.factors_[index].variables != factor_variables_[index] ||
                circuit.factors_[index].values.size() != factor_value_sizes_[index]) {
                throw QStateError("Tensor expectation rebind factor topology changed");
            }
        }
    }
};

struct ExactParameterShiftConfig {
    TensorNetworkConfig tensor{};
    std::size_t worker_count{0U};
};

struct ExactParameterShiftStats {
    std::size_t parameter_count{0U};
    std::size_t parameterized_operation_count{0U};
    std::size_t observable_count{0U};
    std::size_t worker_count{0U};
    std::size_t value_and_gradient_evaluations{0U};
    TensorContractionStats tensor{};
};

class ExactParameterShiftPlan;

class ExactParameterShiftWorkspace {
public:
    ExactParameterShiftWorkspace() = default;

    [[nodiscard]] std::size_t rebind_count() const noexcept {
        std::size_t count = 0U;
        for (const TensorExpectationRebindPlan& lane : lanes_) {
            count += lane.rebind_count();
        }
        return count;
    }

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        std::size_t bytes = sizeof(*this) +
                            lanes_.capacity() * sizeof(TensorExpectationRebindPlan) +
                            tensor_.capacity() * sizeof(TensorExpectationWorkspace) +
                            plus_.capacity() * sizeof(std::vector<QComplex>) +
                            minus_.capacity() * sizeof(std::vector<QComplex>) +
                            lane_gradients_.capacity() * sizeof(std::vector<QComplex>) +
                            values_.capacity() * sizeof(QComplex);
        for (const TensorExpectationRebindPlan& lane : lanes_) {
            bytes += lane.estimated_bytes();
        }
        for (const TensorExpectationWorkspace& workspace_value : tensor_) {
            bytes += workspace_value.estimated_bytes();
        }
        for (const auto& values : plus_) {
            bytes += values.capacity() * sizeof(QComplex);
        }
        for (const auto& values : minus_) {
            bytes += values.capacity() * sizeof(QComplex);
        }
        for (const auto& values : lane_gradients_) {
            bytes += values.capacity() * sizeof(QComplex);
        }
        return bytes;
    }

private:
    std::uint64_t token_{0U};
    std::vector<TensorExpectationRebindPlan> lanes_{};
    std::vector<TensorExpectationWorkspace> tensor_{};
    std::vector<std::vector<QComplex>> plus_{};
    std::vector<std::vector<QComplex>> minus_{};
    std::vector<std::vector<QComplex>> lane_gradients_{};
    std::vector<QComplex> values_{};

    friend class ExactParameterShiftPlan;
};

class ExactParameterShiftPlan {
public:
    ExactParameterShiftPlan(
        std::size_t qubit_count,
        std::span<const ParameterizedOperation> operations,
        std::span<const PauliObservable> observables,
        ExactParameterShiftConfig config = {})
        : qubit_count_(qubit_count),
          operations_(operations.begin(), operations.end()),
          config_(config),
          observable_count_(observables.size()),
          workspace_token_(next_workspace_token()) {
        if (qubit_count_ == 0U) {
            throw QStateError("Exact parameter shift requires at least one qubit");
        }
        if (observables.empty()) {
            throw QStateError("Exact parameter shift requires at least one observable");
        }
        if (config_.tensor.max_contraction_entries < 2U ||
            config_.tensor.max_factors == 0U) {
            throw QStateError("Exact parameter shift tensor limits are invalid");
        }
        if (config_.worker_count > kMaxWorkers) {
            throw QStateError("Exact parameter shift worker_count exceeds the supported limit");
        }

        for (std::size_t index = 0U; index < operations_.size(); ++index) {
            const ParameterizedOperation& templated = operations_[index];
            if (templated.parameter_slot < -1 || templated.sample_slot < -1) {
                throw QStateError("Exact parameter shift slot indices must be -1 or nonnegative");
            }
            if (templated.sample_slot >= 0) {
                throw QStateError("Exact parameter shift does not accept stochastic sample slots");
            }
            switch (templated.operation.code) {
                case OperationCode::BitFlipTrajectory:
                case OperationCode::PhaseFlipTrajectory:
                case OperationCode::DepolarizingTrajectory:
                case OperationCode::AmplitudeDampingTrajectory:
                    throw QStateError("Exact parameter shift does not accept trajectory noise");
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
                        "Exact parameter shift only parameterizes Rx, Ry, and Rz operations");
            }
            const std::size_t slot =
                static_cast<std::size_t>(templated.parameter_slot);
            parameter_count_ = std::max(parameter_count_, slot + 1U);
            occurrences_.push_back({index, slot});
        }

        std::vector<double> zeros(parameter_count_, 0.0);
        const std::vector<Operation> initial_operations = bind_operations(zeros);
        const TensorNetworkCircuit initial(
            qubit_count_, initial_operations, config_.tensor);
        prototype_.emplace(initial, observables);

        std::size_t requested = config_.worker_count;
        if (requested == 0U) {
            const unsigned int hardware = std::thread::hardware_concurrency();
            requested = hardware == 0U ? 1U : static_cast<std::size_t>(hardware);
            requested = std::min(requested, kMaxWorkers);
        }
        const std::size_t jobs = std::max<std::size_t>(1U, occurrences_.size());
        worker_count_ = std::max<std::size_t>(1U, std::min(requested, jobs));
    }

    [[nodiscard]] ExactParameterShiftWorkspace workspace() const {
        ExactParameterShiftWorkspace result;
        result.token_ = workspace_token_;
        result.lanes_.reserve(worker_count_);
        result.tensor_.reserve(worker_count_);
        result.plus_.reserve(worker_count_);
        result.minus_.reserve(worker_count_);
        result.lane_gradients_.reserve(worker_count_);
        const std::size_t gradient_size = observable_count_ * parameter_count_;
        for (std::size_t lane = 0U; lane < worker_count_; ++lane) {
            result.lanes_.push_back(*prototype_);
            result.tensor_.push_back(result.lanes_.back().workspace());
            result.plus_.emplace_back(observable_count_);
            result.minus_.emplace_back(observable_count_);
            result.lane_gradients_.emplace_back(gradient_size);
        }
        result.values_.resize(observable_count_);
        return result;
    }

    void values(
        std::span<const double> parameters,
        std::span<QComplex> results) const {
        ExactParameterShiftWorkspace local = workspace();
        values(parameters, results, local);
    }

    void values(
        std::span<const double> parameters,
        std::span<QComplex> results,
        ExactParameterShiftWorkspace& workspace_value) const {
        validate_parameters(parameters);
        validate_workspace(workspace_value);
        if (results.size() != observable_count_) {
            throw QStateError("Exact parameter shift value result count is invalid");
        }

        const std::vector<Operation> bound = bind_operations(parameters);
        TensorNetworkCircuit circuit(qubit_count_, bound, config_.tensor);
        workspace_value.lanes_.front().rebind(circuit);
        workspace_value.lanes_.front().expectations(
            workspace_value.values_, workspace_value.tensor_.front());
        std::copy(workspace_value.values_.begin(), workspace_value.values_.end(), results.begin());
    }

    void value_and_gradient(
        std::span<const double> parameters,
        std::span<QComplex> values_result,
        std::span<QComplex> gradients) const {
        ExactParameterShiftWorkspace local = workspace();
        value_and_gradient(parameters, values_result, gradients, local);
    }

    void value_and_gradient(
        std::span<const double> parameters,
        std::span<QComplex> values_result,
        std::span<QComplex> gradients,
        ExactParameterShiftWorkspace& workspace_value) const {
        validate_parameters(parameters);
        validate_workspace(workspace_value);
        if (values_result.size() != observable_count_) {
            throw QStateError("Exact parameter shift value result count is invalid");
        }
        if (gradients.size() != observable_count_ * parameter_count_) {
            throw QStateError("Exact parameter shift gradient result count is invalid");
        }

        const std::vector<Operation> bound = bind_operations(parameters);
        TensorNetworkCircuit baseline(qubit_count_, bound, config_.tensor);
        workspace_value.lanes_.front().rebind(baseline);
        workspace_value.lanes_.front().expectations(
            workspace_value.values_, workspace_value.tensor_.front());

        for (auto& lane_gradient : workspace_value.lane_gradients_) {
            std::fill(lane_gradient.begin(), lane_gradient.end(), QComplex{});
        }

        if (!occurrences_.empty()) {
            std::atomic<std::size_t> next{0U};
            std::atomic<bool> stop{false};
            std::mutex error_mutex;
            std::exception_ptr first_error;

            const auto worker = [&](std::size_t lane) {
                std::vector<Operation> shifted = bound;
                while (!stop.load(std::memory_order_relaxed)) {
                    const std::size_t occurrence_index =
                        next.fetch_add(1U, std::memory_order_relaxed);
                    if (occurrence_index >= occurrences_.size()) {
                        return;
                    }
                    const Occurrence occurrence = occurrences_[occurrence_index];
                    const double original = shifted[occurrence.operation_index].parameter;
                    try {
                        shifted[occurrence.operation_index].parameter = original + kShift;
                        TensorNetworkCircuit plus_circuit(
                            qubit_count_, shifted, config_.tensor);
                        workspace_value.lanes_[lane].rebind(plus_circuit);
                        workspace_value.lanes_[lane].expectations(
                            workspace_value.plus_[lane], workspace_value.tensor_[lane]);

                        shifted[occurrence.operation_index].parameter = original - kShift;
                        TensorNetworkCircuit minus_circuit(
                            qubit_count_, shifted, config_.tensor);
                        workspace_value.lanes_[lane].rebind(minus_circuit);
                        workspace_value.lanes_[lane].expectations(
                            workspace_value.minus_[lane], workspace_value.tensor_[lane]);
                        shifted[occurrence.operation_index].parameter = original;

                        for (std::size_t observable = 0U;
                             observable < observable_count_;
                             ++observable) {
                            const std::size_t gradient_index =
                                observable * parameter_count_ + occurrence.parameter_slot;
                            workspace_value.lane_gradients_[lane][gradient_index] +=
                                (workspace_value.plus_[lane][observable] -
                                 workspace_value.minus_[lane][observable]) * 0.5;
                        }
                    } catch (...) {
                        shifted[occurrence.operation_index].parameter = original;
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
            threads.reserve(worker_count_ > 0U ? worker_count_ - 1U : 0U);
            for (std::size_t lane = 1U; lane < worker_count_; ++lane) {
                threads.emplace_back(worker, lane);
            }
            worker(0U);
            for (std::thread& thread : threads) {
                thread.join();
            }
            if (first_error != nullptr) {
                std::rethrow_exception(first_error);
            }
        }

        std::vector<QComplex> gradient_result(gradients.size(), QComplex{});
        for (const auto& lane_gradient : workspace_value.lane_gradients_) {
            for (std::size_t index = 0U; index < gradient_result.size(); ++index) {
                gradient_result[index] += lane_gradient[index];
            }
        }

        std::copy(
            workspace_value.values_.begin(), workspace_value.values_.end(),
            values_result.begin());
        std::copy(gradient_result.begin(), gradient_result.end(), gradients.begin());
    }

    [[nodiscard]] std::size_t gradient_index(
        std::size_t observable,
        std::size_t parameter) const {
        if (observable >= observable_count_ || parameter >= parameter_count_) {
            throw QStateError("Exact parameter shift gradient index is out of range");
        }
        return observable * parameter_count_ + parameter;
    }

    [[nodiscard]] std::size_t qubit_count() const noexcept {
        return qubit_count_;
    }

    [[nodiscard]] std::size_t operation_count() const noexcept {
        return operations_.size();
    }

    [[nodiscard]] std::size_t parameter_count() const noexcept {
        return parameter_count_;
    }

    [[nodiscard]] std::size_t parameterized_operation_count() const noexcept {
        return occurrences_.size();
    }

    [[nodiscard]] std::size_t observable_count() const noexcept {
        return observable_count_;
    }

    [[nodiscard]] std::size_t worker_count() const noexcept {
        return worker_count_;
    }

    [[nodiscard]] ExactParameterShiftStats stats() const noexcept {
        ExactParameterShiftStats result;
        result.parameter_count = parameter_count_;
        result.parameterized_operation_count = occurrences_.size();
        result.observable_count = observable_count_;
        result.worker_count = worker_count_;
        result.value_and_gradient_evaluations = 1U + 2U * occurrences_.size();
        result.tensor = prototype_->stats();
        return result;
    }

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        return sizeof(*this) +
               operations_.capacity() * sizeof(ParameterizedOperation) +
               occurrences_.capacity() * sizeof(Occurrence) +
               (prototype_.has_value() ? prototype_->estimated_bytes() : 0U);
    }

private:
    struct Occurrence {
        std::size_t operation_index{0U};
        std::size_t parameter_slot{0U};
    };

    static constexpr std::size_t kMaxWorkers = 128U;
    static constexpr double kShift = 1.57079632679489661923;

    std::size_t qubit_count_{0U};
    std::vector<ParameterizedOperation> operations_{};
    ExactParameterShiftConfig config_{};
    std::vector<Occurrence> occurrences_{};
    std::size_t parameter_count_{0U};
    std::size_t observable_count_{0U};
    std::size_t worker_count_{1U};
    std::uint64_t workspace_token_{0U};
    std::optional<TensorExpectationRebindPlan> prototype_{};

    [[nodiscard]] static std::uint64_t next_workspace_token() noexcept {
        static std::atomic<std::uint64_t> next{1U};
        return next.fetch_add(1U, std::memory_order_relaxed);
    }

    void validate_parameters(std::span<const double> parameters) const {
        if (parameters.size() != parameter_count_) {
            throw QStateError("Exact parameter shift parameter count is invalid");
        }
        for (const double parameter : parameters) {
            if (!std::isfinite(parameter)) {
                throw QStateError("Exact parameter shift parameters must be finite");
            }
        }
    }

    [[nodiscard]] std::vector<Operation> bind_operations(
        std::span<const double> parameters) const {
        if (parameters.size() != parameter_count_) {
            throw QStateError("Exact parameter shift parameter count is invalid");
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
        const ExactParameterShiftWorkspace& workspace_value) const {
        const std::size_t gradient_size = observable_count_ * parameter_count_;
        if (workspace_value.token_ != workspace_token_ ||
            workspace_value.lanes_.size() != worker_count_ ||
            workspace_value.tensor_.size() != worker_count_ ||
            workspace_value.plus_.size() != worker_count_ ||
            workspace_value.minus_.size() != worker_count_ ||
            workspace_value.lane_gradients_.size() != worker_count_ ||
            workspace_value.values_.size() != observable_count_) {
            throw QStateError("Exact parameter shift workspace does not match its plan");
        }
        for (std::size_t lane = 0U; lane < worker_count_; ++lane) {
            if (workspace_value.plus_[lane].size() != observable_count_ ||
                workspace_value.minus_[lane].size() != observable_count_ ||
                workspace_value.lane_gradients_[lane].size() != gradient_size) {
                throw QStateError("Exact parameter shift workspace shape is invalid");
            }
        }
    }
};

}  // namespace qubit
