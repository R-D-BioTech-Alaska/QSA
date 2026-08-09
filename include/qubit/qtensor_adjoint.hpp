#pragma once

#include "qubit/qtensor.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace qubit {

struct ExactAdjointGradientConfig {
    TensorNetworkConfig tensor{};
    std::size_t worker_count{1U};
};

struct ExactAdjointGradientStats {
    std::size_t parameter_count{0U};
    std::size_t parameterized_operation_count{0U};
    std::size_t observable_count{0U};
    std::size_t differentiated_term_count{0U};
    std::size_t source_derivative_bindings{0U};
    std::size_t parameter_shift_equivalent_evaluations{0U};
    std::size_t worker_count{1U};
    std::uint64_t estimated_work{0U};
    std::uint64_t balanced_peak_estimated_work{0U};
    std::uint64_t round_robin_peak_estimated_work{0U};
    TensorContractionStats tensor{};
};

class ExactAdjointGradientPlan;

class ExactAdjointGradientWorkspace {
public:
    ExactAdjointGradientWorkspace() = default;

    [[nodiscard]] std::size_t rebind_count() const noexcept {
        return rebind_count_;
    }

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        std::size_t bytes = sizeof(*this) +
                            values_.capacity() * sizeof(QComplex) +
                            gradients_.capacity() * sizeof(QComplex) +
                            scratch_.capacity() * sizeof(Scratch) +
                            lane_values_.capacity() * sizeof(std::vector<QComplex>) +
                            lane_gradients_.capacity() * sizeof(std::vector<QComplex>);
        if (plan_.has_value()) {
            bytes += plan_->estimated_bytes();
        }
        for (const Scratch& scratch : scratch_) {
            bytes += scratch.forward_steps.capacity() * sizeof(std::vector<QComplex>) +
                     scratch.source_adjoints.capacity() * sizeof(std::vector<QComplex>) +
                     scratch.step_adjoints.capacity() * sizeof(std::vector<QComplex>) +
                     scratch.input_indices.capacity() * sizeof(std::size_t) +
                     scratch.input_values.capacity() * sizeof(QComplex) +
                     scratch.prefix.capacity() * sizeof(QComplex) +
                     scratch.suffix.capacity() * sizeof(QComplex) +
                     scratch.terminal_values.capacity() * sizeof(QComplex);
            for (const auto& values : scratch.forward_steps) {
                bytes += values.capacity() * sizeof(QComplex);
            }
            for (const auto& values : scratch.source_adjoints) {
                bytes += values.capacity() * sizeof(QComplex);
            }
            for (const auto& values : scratch.step_adjoints) {
                bytes += values.capacity() * sizeof(QComplex);
            }
        }
        for (const auto& values : lane_values_) {
            bytes += values.capacity() * sizeof(QComplex);
        }
        for (const auto& values : lane_gradients_) {
            bytes += values.capacity() * sizeof(QComplex);
        }
        return bytes;
    }

private:
    struct Scratch {
        std::vector<std::vector<QComplex>> forward_steps{};
        std::vector<std::vector<QComplex>> source_adjoints{};
        std::vector<std::vector<QComplex>> step_adjoints{};
        std::vector<std::size_t> input_indices{};
        std::vector<QComplex> input_values{};
        std::vector<QComplex> prefix{};
        std::vector<QComplex> suffix{};
        std::vector<QComplex> terminal_values{};
    };

    std::uint64_t token_{0U};
    std::optional<TensorExpectationPlan> plan_{};
    std::vector<QComplex> values_{};
    std::vector<QComplex> gradients_{};
    std::vector<Scratch> scratch_{};
    std::vector<std::vector<QComplex>> lane_values_{};
    std::vector<std::vector<QComplex>> lane_gradients_{};
    std::size_t rebind_count_{0U};

    friend class ExactAdjointGradientPlan;
};

class ExactAdjointGradientPlan {
public:
    ExactAdjointGradientPlan(
        std::size_t qubit_count,
        std::span<const ParameterizedOperation> operations,
        std::span<const PauliObservable> observables,
        ExactAdjointGradientConfig config = {})
        : qubit_count_(qubit_count),
          operations_(operations.begin(), operations.end()),
          config_(config),
          observable_count_(observables.size()),
          workspace_token_(next_workspace_token()) {
        if (qubit_count_ == 0U) {
            throw QStateError("Exact adjoint gradient requires at least one qubit");
        }
        if (observables.empty()) {
            throw QStateError("Exact adjoint gradient requires at least one observable");
        }
        if (config_.tensor.max_contraction_entries < 2U ||
            config_.tensor.max_factors == 0U) {
            throw QStateError("Exact adjoint gradient tensor limits are invalid");
        }
        if (config_.worker_count > kMaxWorkers) {
            throw QStateError("Exact adjoint gradient worker_count exceeds the supported limit");
        }

        for (std::size_t index = 0U; index < operations_.size(); ++index) {
            const ParameterizedOperation& templated = operations_[index];
            if (templated.parameter_slot < -1 || templated.sample_slot < -1) {
                throw QStateError("Exact adjoint gradient slot indices must be -1 or nonnegative");
            }
            if (templated.sample_slot >= 0) {
                throw QStateError("Exact adjoint gradient does not accept stochastic sample slots");
            }
            switch (templated.operation.code) {
                case OperationCode::BitFlipTrajectory:
                case OperationCode::PhaseFlipTrajectory:
                case OperationCode::DepolarizingTrajectory:
                case OperationCode::AmplitudeDampingTrajectory:
                    throw QStateError("Exact adjoint gradient does not accept trajectory noise");
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
                        "Exact adjoint gradient only parameterizes Rx, Ry, and Rz operations");
            }
            const std::size_t slot =
                static_cast<std::size_t>(templated.parameter_slot);
            parameter_count_ = std::max(parameter_count_, slot + 1U);
            occurrences_.push_back({index, slot, qubit_count_ + index});
        }

        std::vector<double> zeros(parameter_count_, 0.0);
        const std::vector<Operation> initial_operations = bind_operations(zeros);
        const TensorNetworkCircuit initial(
            qubit_count_, initial_operations, config_.tensor);
        prototype_.emplace(initial, observables);
        snapshot_topology(initial);
        build_bindings(initial, observables);

        occurrence_bindings_.resize(occurrences_.size());
        for (std::size_t occurrence_index = 0U;
             occurrence_index < occurrences_.size();
             ++occurrence_index) {
            const Occurrence& occurrence = occurrences_[occurrence_index];
            if (occurrence.factor_index >= factor_bindings_.size()) {
                throw QStateError("Exact adjoint gradient parameter factor is out of range");
            }
            const auto& factor = initial.factors_[occurrence.factor_index];
            if (factor.variables.size() != 2U || factor.values.size() != 4U) {
                throw QStateError("Exact adjoint gradient parameter factor is not rank two");
            }
            occurrence_bindings_[occurrence_index] =
                factor_bindings_[occurrence.factor_index];
        }

        term_derivative_bindings_.resize(prototype_->terms_.size());
        for (std::size_t occurrence_index = 0U;
             occurrence_index < occurrence_bindings_.size();
             ++occurrence_index) {
            for (const std::size_t binding_index : occurrence_bindings_[occurrence_index]) {
                const Binding& binding = bindings_[binding_index];
                term_derivative_bindings_[binding.term_index].push_back(
                    {occurrence_index, binding.source_index, binding.xor_mask, binding.bra});
            }
        }

        identity_values_.assign(observable_count_, QComplex{});
        for (std::size_t observable_index = 0U;
             observable_index < prototype_->observables_.size();
             ++observable_index) {
            const auto observable = prototype_->observables_[observable_index];
            for (std::size_t term_index = observable.term_begin;
                 term_index < observable.term_end;
                 ++term_index) {
                const auto& term = prototype_->terms_[term_index];
                if (term.identity) {
                    identity_values_[observable_index] += term.coefficient;
                } else {
                    jobs_.push_back({
                        observable_index,
                        term_index,
                        estimate_term_work(
                            term,
                            term_derivative_bindings_[term_index].size()),
                    });
                }
            }
        }

        std::size_t requested = config_.worker_count;
        if (requested == 0U) {
            const unsigned int hardware = std::thread::hardware_concurrency();
            requested = hardware == 0U ? 1U : static_cast<std::size_t>(hardware);
            requested = std::min(requested, kMaxWorkers);
        }
        const std::size_t job_count = std::max<std::size_t>(1U, jobs_.size());
        worker_count_ = std::max<std::size_t>(1U, std::min(requested, job_count));
        build_lane_schedule();
    }

    [[nodiscard]] ExactAdjointGradientWorkspace workspace() const {
        ExactAdjointGradientWorkspace result;
        result.token_ = workspace_token_;
        result.plan_ = *prototype_;
        result.values_.resize(observable_count_);
        result.gradients_.resize(observable_count_ * parameter_count_);
        result.scratch_.resize(worker_count_);
        result.lane_values_.resize(worker_count_);
        result.lane_gradients_.resize(worker_count_);
        for (std::size_t lane = 0U; lane < worker_count_; ++lane) {
            result.lane_values_[lane].resize(observable_count_);
            result.lane_gradients_[lane].resize(observable_count_ * parameter_count_);
        }
        return result;
    }

    void value_and_gradient(
        std::span<const double> parameters,
        std::span<QComplex> values_result,
        std::span<QComplex> gradients) const {
        ExactAdjointGradientWorkspace local = workspace();
        value_and_gradient(parameters, values_result, gradients, local);
    }

    void value_and_gradient(
        std::span<const double> parameters,
        std::span<QComplex> values_result,
        std::span<QComplex> gradients,
        ExactAdjointGradientWorkspace& workspace_value) const {
        validate_parameters(parameters);
        validate_workspace(workspace_value);
        if (values_result.size() != observable_count_) {
            throw QStateError("Exact adjoint gradient value result count is invalid");
        }
        if (gradients.size() != observable_count_ * parameter_count_) {
            throw QStateError("Exact adjoint gradient result count is invalid");
        }

        const std::vector<Operation> bound = bind_operations(parameters);
        const TensorNetworkCircuit circuit(qubit_count_, bound, config_.tensor);
        rebind(*workspace_value.plan_, circuit);

        for (std::size_t lane = 0U; lane < worker_count_; ++lane) {
            std::fill(
                workspace_value.lane_values_[lane].begin(),
                workspace_value.lane_values_[lane].end(),
                QComplex{});
            std::fill(
                workspace_value.lane_gradients_[lane].begin(),
                workspace_value.lane_gradients_[lane].end(),
                QComplex{});
        }

        const TensorExpectationPlan& plan = *workspace_value.plan_;
        std::vector<std::exception_ptr> errors(worker_count_);
        const auto worker = [&](std::size_t lane) {
            try {
                for (const std::size_t job_index : lane_jobs_[lane]) {
                    const TermJob job = jobs_[job_index];
                    const auto& term = plan.terms_[job.term_index];
                    const QComplex raw_value = forward_and_reverse(
                        term, workspace_value.scratch_[lane]);
                    workspace_value.lane_values_[lane][job.observable_index] +=
                        term.coefficient * raw_value;

                    for (const DerivativeBinding& derivative_binding :
                         term_derivative_bindings_[job.term_index]) {
                        const Occurrence& occurrence =
                            occurrences_[derivative_binding.occurrence_index];
                        const std::array<QComplex, 4> derivative = gate_derivative(
                            bound[occurrence.operation_index]);
                        const auto& source_adjoint =
                            workspace_value.scratch_[lane].source_adjoints[
                                derivative_binding.source_index];
                        if (source_adjoint.size() != derivative.size()) {
                            throw QStateError(
                                "Exact adjoint gradient source derivative rank changed");
                        }

                        QComplex contribution{};
                        if (!derivative_binding.bra) {
                            for (std::size_t index = 0U;
                                 index < derivative.size();
                                 ++index) {
                                contribution += source_adjoint[index] * derivative[index];
                            }
                        } else {
                            for (std::size_t index = 0U;
                                 index < derivative.size();
                                 ++index) {
                                const std::size_t target =
                                    index ^ derivative_binding.xor_mask;
                                if (target >= source_adjoint.size()) {
                                    throw QStateError(
                                        "Exact adjoint gradient bra derivative permutation is invalid");
                                }
                                contribution += source_adjoint[target] *
                                                derivative[index].conjugate();
                            }
                        }
                        const std::size_t gradient_offset =
                            job.observable_index * parameter_count_ +
                            occurrence.parameter_slot;
                        workspace_value.lane_gradients_[lane][gradient_offset] +=
                            contribution;
                    }
                }
            } catch (...) {
                errors[lane] = std::current_exception();
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
        for (const std::exception_ptr& error : errors) {
            if (error != nullptr) {
                std::rethrow_exception(error);
            }
        }

        workspace_value.values_ = identity_values_;
        std::fill(
            workspace_value.gradients_.begin(),
            workspace_value.gradients_.end(),
            QComplex{});
        for (std::size_t lane = 0U; lane < worker_count_; ++lane) {
            for (std::size_t observable = 0U;
                 observable < observable_count_;
                 ++observable) {
                workspace_value.values_[observable] +=
                    workspace_value.lane_values_[lane][observable];
            }
            for (std::size_t index = 0U;
                 index < workspace_value.gradients_.size();
                 ++index) {
                workspace_value.gradients_[index] +=
                    workspace_value.lane_gradients_[lane][index];
            }
        }

        ++workspace_value.rebind_count_;
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
            throw QStateError("Exact adjoint gradient index is out of range");
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

    [[nodiscard]] ExactAdjointGradientStats stats() const noexcept {
        ExactAdjointGradientStats result;
        result.parameter_count = parameter_count_;
        result.parameterized_operation_count = occurrences_.size();
        result.observable_count = observable_count_;
        result.differentiated_term_count = jobs_.size();
        for (const auto& bindings : term_derivative_bindings_) {
            result.source_derivative_bindings += bindings.size();
        }
        result.parameter_shift_equivalent_evaluations =
            1U + 2U * occurrences_.size();
        result.worker_count = worker_count_;
        result.estimated_work = estimated_work_;
        result.balanced_peak_estimated_work = balanced_peak_estimated_work_;
        result.round_robin_peak_estimated_work = round_robin_peak_estimated_work_;
        result.tensor = prototype_->stats();
        return result;
    }

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        std::size_t bytes = sizeof(*this) +
                            operations_.capacity() * sizeof(ParameterizedOperation) +
                            occurrences_.capacity() * sizeof(Occurrence) +
                            current_wires_.capacity() * sizeof(VariableId) +
                            factor_variables_.capacity() * sizeof(std::vector<VariableId>) +
                            factor_value_sizes_.capacity() * sizeof(std::size_t) +
                            bindings_.capacity() * sizeof(Binding) +
                            factor_bindings_.capacity() * sizeof(std::vector<std::size_t>) +
                            occurrence_bindings_.capacity() * sizeof(std::vector<std::size_t>) +
                            term_derivative_bindings_.capacity() *
                                sizeof(std::vector<DerivativeBinding>) +
                            identity_values_.capacity() * sizeof(QComplex) +
                            jobs_.capacity() * sizeof(TermJob) +
                            lane_jobs_.capacity() * sizeof(std::vector<std::size_t>) +
                            lane_estimated_work_.capacity() * sizeof(std::uint64_t);
        if (prototype_.has_value()) {
            bytes += prototype_->estimated_bytes();
        }
        for (const auto& variables : factor_variables_) {
            bytes += variables.capacity() * sizeof(VariableId);
        }
        for (const auto& list : factor_bindings_) {
            bytes += list.capacity() * sizeof(std::size_t);
        }
        for (const auto& list : occurrence_bindings_) {
            bytes += list.capacity() * sizeof(std::size_t);
        }
        for (const auto& list : term_derivative_bindings_) {
            bytes += list.capacity() * sizeof(DerivativeBinding);
        }
        for (const auto& lane : lane_jobs_) {
            bytes += lane.capacity() * sizeof(std::size_t);
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
    };

    struct DerivativeBinding {
        std::size_t occurrence_index{0U};
        std::size_t source_index{0U};
        std::size_t xor_mask{0U};
        bool bra{false};
    };

    struct TermJob {
        std::size_t observable_index{0U};
        std::size_t term_index{0U};
        std::uint64_t estimated_work{1U};
    };

    static constexpr std::size_t kMaxWorkers = 32U;

    std::size_t qubit_count_{0U};
    std::vector<ParameterizedOperation> operations_{};
    ExactAdjointGradientConfig config_{};
    std::size_t parameter_count_{0U};
    std::size_t observable_count_{0U};
    std::vector<Occurrence> occurrences_{};
    std::uint64_t workspace_token_{0U};
    std::optional<TensorExpectationPlan> prototype_{};

    std::size_t operation_count_{0U};
    VariableId next_variable_{0U};
    std::vector<VariableId> current_wires_{};
    std::vector<std::vector<VariableId>> factor_variables_{};
    std::vector<std::size_t> factor_value_sizes_{};
    std::vector<Binding> bindings_{};
    std::vector<std::vector<std::size_t>> factor_bindings_{};
    std::vector<std::vector<std::size_t>> occurrence_bindings_{};
    std::vector<std::vector<DerivativeBinding>> term_derivative_bindings_{};
    std::vector<QComplex> identity_values_{};
    std::vector<TermJob> jobs_{};
    std::size_t worker_count_{1U};
    std::vector<std::vector<std::size_t>> lane_jobs_{};
    std::vector<std::uint64_t> lane_estimated_work_{};
    std::uint64_t estimated_work_{0U};
    std::uint64_t balanced_peak_estimated_work_{0U};
    std::uint64_t round_robin_peak_estimated_work_{0U};

    [[nodiscard]] static std::uint64_t next_workspace_token() noexcept {
        static std::atomic<std::uint64_t> next{1U};
        return next.fetch_add(1U, std::memory_order_relaxed);
    }

    [[nodiscard]] static std::uint64_t saturating_add(
        std::uint64_t first,
        std::uint64_t second) noexcept {
        if (second > std::numeric_limits<std::uint64_t>::max() - first) {
            return std::numeric_limits<std::uint64_t>::max();
        }
        return first + second;
    }

    [[nodiscard]] static std::uint64_t saturating_multiply(
        std::uint64_t first,
        std::uint64_t second) noexcept {
        if (first == 0U || second == 0U) {
            return 0U;
        }
        if (first > std::numeric_limits<std::uint64_t>::max() / second) {
            return std::numeric_limits<std::uint64_t>::max();
        }
        return first * second;
    }

    [[nodiscard]] static std::uint64_t estimate_term_work(
        const TensorExpectationPlan::TermPlan& term,
        std::size_t derivative_bindings) noexcept {
        std::uint64_t work = 0U;
        for (const auto& step : term.steps) {
            const std::uint64_t entries = static_cast<std::uint64_t>(
                std::max<std::size_t>(1U, step.output_entries));
            const std::uint64_t inputs = static_cast<std::uint64_t>(
                std::max<std::size_t>(1U, step.inputs.size()));
            const std::uint64_t per_entry = saturating_add(
                saturating_multiply(6U, inputs), 4U);
            work = saturating_add(
                work,
                saturating_multiply(entries, per_entry));
        }
        work = saturating_add(
            work,
            saturating_multiply(
                static_cast<std::uint64_t>(derivative_bindings), 16U));
        return std::max<std::uint64_t>(1U, work);
    }

    void build_lane_schedule() {
        lane_jobs_.assign(worker_count_, {});
        lane_estimated_work_.assign(worker_count_, 0U);
        estimated_work_ = 0U;
        balanced_peak_estimated_work_ = 0U;
        round_robin_peak_estimated_work_ = 0U;

        std::vector<std::uint64_t> round_robin(worker_count_, 0U);
        for (std::size_t job_index = 0U; job_index < jobs_.size(); ++job_index) {
            estimated_work_ = saturating_add(
                estimated_work_, jobs_[job_index].estimated_work);
            const std::size_t lane = job_index % worker_count_;
            round_robin[lane] = saturating_add(
                round_robin[lane], jobs_[job_index].estimated_work);
        }
        for (const std::uint64_t lane_work : round_robin) {
            round_robin_peak_estimated_work_ =
                std::max(round_robin_peak_estimated_work_, lane_work);
        }

        if (worker_count_ == 1U) {
            lane_jobs_.front().reserve(jobs_.size());
            for (std::size_t job_index = 0U; job_index < jobs_.size(); ++job_index) {
                lane_jobs_.front().push_back(job_index);
            }
            lane_estimated_work_.front() = estimated_work_;
            balanced_peak_estimated_work_ = estimated_work_;
            return;
        }

        std::vector<std::size_t> order(jobs_.size());
        for (std::size_t index = 0U; index < order.size(); ++index) {
            order[index] = index;
        }
        std::stable_sort(
            order.begin(), order.end(),
            [&](std::size_t first, std::size_t second) {
                if (jobs_[first].estimated_work != jobs_[second].estimated_work) {
                    return jobs_[first].estimated_work > jobs_[second].estimated_work;
                }
                if (jobs_[first].observable_index != jobs_[second].observable_index) {
                    return jobs_[first].observable_index < jobs_[second].observable_index;
                }
                return jobs_[first].term_index < jobs_[second].term_index;
            });

        for (const std::size_t job_index : order) {
            std::size_t selected_lane = 0U;
            for (std::size_t lane = 1U; lane < worker_count_; ++lane) {
                if (lane_estimated_work_[lane] <
                    lane_estimated_work_[selected_lane]) {
                    selected_lane = lane;
                }
            }
            lane_jobs_[selected_lane].push_back(job_index);
            lane_estimated_work_[selected_lane] = saturating_add(
                lane_estimated_work_[selected_lane],
                jobs_[job_index].estimated_work);
        }
        for (const std::uint64_t lane_work : lane_estimated_work_) {
            balanced_peak_estimated_work_ =
                std::max(balanced_peak_estimated_work_, lane_work);
        }
        if (balanced_peak_estimated_work_ > round_robin_peak_estimated_work_) {
            throw QStateError("Exact adjoint gradient load balancing regressed estimated peak work");
        }
    }

    void validate_parameters(std::span<const double> parameters) const {
        if (parameters.size() != parameter_count_) {
            throw QStateError("Exact adjoint gradient parameter count is invalid");
        }
        for (const double parameter : parameters) {
            if (!std::isfinite(parameter)) {
                throw QStateError("Exact adjoint gradient parameters must be finite");
            }
        }
    }

    void validate_workspace(
        const ExactAdjointGradientWorkspace& workspace_value) const {
        if (workspace_value.token_ != workspace_token_ ||
            !workspace_value.plan_.has_value() ||
            workspace_value.values_.size() != observable_count_ ||
            workspace_value.gradients_.size() != observable_count_ * parameter_count_ ||
            workspace_value.scratch_.size() != worker_count_ ||
            workspace_value.lane_values_.size() != worker_count_ ||
            workspace_value.lane_gradients_.size() != worker_count_) {
            throw QStateError("Exact adjoint gradient workspace does not match its plan");
        }
        for (std::size_t lane = 0U; lane < worker_count_; ++lane) {
            if (workspace_value.lane_values_[lane].size() != observable_count_ ||
                workspace_value.lane_gradients_[lane].size() !=
                    observable_count_ * parameter_count_) {
                throw QStateError("Exact adjoint gradient workspace lane shape is invalid");
            }
        }
    }

    [[nodiscard]] std::vector<Operation> bind_operations(
        std::span<const double> parameters) const {
        if (parameters.size() != parameter_count_) {
            throw QStateError("Exact adjoint gradient parameter count is invalid");
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

    void snapshot_topology(const TensorNetworkCircuit& circuit) {
        operation_count_ = circuit.operation_count_;
        next_variable_ = circuit.next_variable_;
        current_wires_ = circuit.current_wires_;
        factor_variables_.reserve(circuit.factors_.size());
        factor_value_sizes_.reserve(circuit.factors_.size());
        for (const auto& factor : circuit.factors_) {
            factor_variables_.push_back(factor.variables);
            factor_value_sizes_.push_back(factor.values.size());
        }
        factor_bindings_.resize(circuit.factors_.size());
    }

    [[nodiscard]] std::size_t factor_index(
        const TensorNetworkCircuit& circuit,
        std::span<const VariableId> variables) const {
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
                throw QStateError("Exact adjoint gradient topology has duplicate factors");
            }
            match = index;
        }
        if (match == circuit.factors_.size()) {
            throw QStateError("Exact adjoint gradient source factor was not found");
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
        if (compiled_terms.size() != prototype_->terms_.size()) {
            throw QStateError("Exact adjoint gradient term map does not match the plan");
        }

        for (std::size_t term_index = 0U;
             term_index < prototype_->terms_.size();
             ++term_index) {
            const PauliTerm& observable_term = *compiled_terms[term_index];
            const auto& term = prototype_->terms_[term_index];
            if (term.identity) {
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
                throw QStateError("Exact adjoint gradient source layout is invalid");
            }

            std::vector<std::uint8_t> invert(next_variable_, 0U);
            for (const PauliFactor& factor : observable_term.factors) {
                if (factor.axis != PauliAxis::X && factor.axis != PauliAxis::Y) {
                    continue;
                }
                if (factor.qubit >= current_wires_.size()) {
                    throw QStateError("Exact adjoint gradient Pauli factor is out of range");
                }
                invert[current_wires_[factor.qubit]] = 1U;
            }

            for (std::size_t local = 0U; local < bra_count; ++local) {
                const auto& ket_source = term.sources[local];
                const auto& bra_source = term.sources[bra_count + local];
                const std::size_t source_factor =
                    factor_index(circuit, ket_source.variables);
                const auto& topology = circuit.factors_[source_factor].variables;
                if (bra_source.variables.size() != topology.size()) {
                    throw QStateError("Exact adjoint gradient bra source rank changed");
                }

                std::size_t xor_mask = 0U;
                for (std::size_t position = 0U; position < topology.size(); ++position) {
                    const std::uint64_t bra_variable =
                        static_cast<std::uint64_t>(topology[position]) +
                        static_cast<std::uint64_t>(next_variable_);
                    if (bra_variable > std::numeric_limits<VariableId>::max() ||
                        bra_source.variables[position] !=
                            static_cast<VariableId>(bra_variable)) {
                        throw QStateError("Exact adjoint gradient bra topology changed");
                    }
                    if (invert[topology[position]] != 0U) {
                        if (position >= std::numeric_limits<std::size_t>::digits) {
                            throw QStateError("Exact adjoint gradient bra mask overflowed");
                        }
                        xor_mask |= std::size_t{1} << position;
                    }
                }

                bindings_.push_back({term_index, local, source_factor, 0U, false});
                factor_bindings_[source_factor].push_back(bindings_.size() - 1U);
                bindings_.push_back({
                    term_index, bra_count + local, source_factor, xor_mask, true});
                factor_bindings_[source_factor].push_back(bindings_.size() - 1U);
            }
        }
    }

    void validate_topology(const TensorNetworkCircuit& circuit) const {
        std::string reason;
        if (!circuit.validate(&reason)) {
            throw QStateError("Cannot rebind invalid adjoint tensor network: " + reason);
        }
        if (circuit.qubit_count_ != qubit_count_ ||
            circuit.operation_count_ != operation_count_ ||
            circuit.next_variable_ != next_variable_ ||
            circuit.current_wires_ != current_wires_ ||
            circuit.factors_.size() != factor_variables_.size() ||
            circuit.config_.max_contraction_entries !=
                config_.tensor.max_contraction_entries ||
            circuit.config_.max_factors != config_.tensor.max_factors) {
            throw QStateError("Exact adjoint gradient topology does not match its plan");
        }
        for (std::size_t index = 0U; index < circuit.factors_.size(); ++index) {
            if (circuit.factors_[index].variables != factor_variables_[index] ||
                circuit.factors_[index].values.size() != factor_value_sizes_[index]) {
                throw QStateError("Exact adjoint gradient factor topology changed");
            }
        }
    }

    void rebind(
        TensorExpectationPlan& plan,
        const TensorNetworkCircuit& circuit) const {
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
                plan.terms_[binding.term_index].sources[binding.source_index];
            if (factor.values.size() != source.values.size()) {
                throw QStateError("Exact adjoint gradient source size changed");
            }
            Replacement replacement;
            replacement.term_index = binding.term_index;
            replacement.source_index = binding.source_index;
            replacement.values.resize(factor.values.size());
            if (!binding.bra) {
                replacement.values = factor.values;
            } else {
                for (std::size_t index = 0U; index < factor.values.size(); ++index) {
                    const std::size_t target = index ^ binding.xor_mask;
                    if (target >= replacement.values.size()) {
                        throw QStateError(
                            "Exact adjoint gradient bra permutation is invalid");
                    }
                    replacement.values[target] = factor.values[index].conjugate();
                }
            }
            replacements.push_back(std::move(replacement));
        }
        for (Replacement& replacement : replacements) {
            plan.terms_[replacement.term_index]
                .sources[replacement.source_index]
                .values = std::move(replacement.values);
        }
    }

    [[nodiscard]] static std::array<QComplex, 4> gate_derivative(
        const Operation& operation) {
        const double half = operation.parameter / 2.0;
        const double cosine = std::cos(half);
        const double sine = std::sin(half);
        switch (operation.code) {
            case OperationCode::Rx:
                return {{{-0.5 * sine, 0.0},
                         {0.0, -0.5 * cosine},
                         {0.0, -0.5 * cosine},
                         {-0.5 * sine, 0.0}}};
            case OperationCode::Ry:
                return {{{-0.5 * sine, 0.0},
                         {-0.5 * cosine, 0.0},
                         {0.5 * cosine, 0.0},
                         {-0.5 * sine, 0.0}}};
            case OperationCode::Rz: {
                const QComplex zero = QComplex::from_polar(1.0, -half);
                const QComplex one = QComplex::from_polar(1.0, half);
                return {{{0.5 * zero.im, -0.5 * zero.re},
                         {},
                         {},
                         {-0.5 * one.im, 0.5 * one.re}}};
            }
            default:
                throw QStateError("Exact adjoint gradient derivative gate is unsupported");
        }
    }

    [[nodiscard]] static std::size_t mapped_factor_index(
        std::size_t union_index,
        const std::vector<std::size_t>& positions) noexcept {
        std::size_t result = 0U;
        for (std::size_t local = 0U; local < positions.size(); ++local) {
            result |= ((union_index >> positions[local]) & 1U) << local;
        }
        return result;
    }

    [[nodiscard]] static std::span<const QComplex> node_values(
        const TensorExpectationPlan::TermPlan& term,
        std::size_t node,
        const ExactAdjointGradientWorkspace::Scratch& scratch) {
        if (node < term.sources.size()) {
            return term.sources[node].values;
        }
        if (node < term.step_node_begin) {
            const std::size_t operator_index = node - term.sources.size();
            if (operator_index >= term.operators.size()) {
                throw QStateError("Exact adjoint gradient operator node is invalid");
            }
            return term.operators[operator_index];
        }
        const std::size_t step_index = node - term.step_node_begin;
        if (step_index >= scratch.forward_steps.size()) {
            throw QStateError("Exact adjoint gradient step node is invalid");
        }
        return scratch.forward_steps[step_index];
    }

    static void add_adjoint(
        const TensorExpectationPlan::TermPlan& term,
        std::size_t node,
        std::size_t index,
        QComplex value,
        ExactAdjointGradientWorkspace::Scratch& scratch) {
        if (node < term.sources.size()) {
            if (index >= scratch.source_adjoints[node].size()) {
                throw QStateError("Exact adjoint gradient source adjoint index is invalid");
            }
            scratch.source_adjoints[node][index] += value;
            return;
        }
        if (node < term.step_node_begin) {
            return;
        }
        const std::size_t step_index = node - term.step_node_begin;
        if (step_index >= scratch.step_adjoints.size() ||
            index >= scratch.step_adjoints[step_index].size()) {
            throw QStateError("Exact adjoint gradient step adjoint index is invalid");
        }
        scratch.step_adjoints[step_index][index] += value;
    }

    [[nodiscard]] static QComplex forward_and_reverse(
        const TensorExpectationPlan::TermPlan& term,
        ExactAdjointGradientWorkspace::Scratch& scratch) {
        scratch.forward_steps.resize(term.steps.size());
        scratch.step_adjoints.resize(term.steps.size());
        for (std::size_t step_index = 0U; step_index < term.steps.size(); ++step_index) {
            const auto& step = term.steps[step_index];
            scratch.forward_steps[step_index].assign(step.output_entries, QComplex{});
            scratch.step_adjoints[step_index].assign(step.output_entries, QComplex{});
            auto& output = scratch.forward_steps[step_index];
            const std::size_t lower_mask =
                step.selected_position == 0U
                    ? 0U
                    : (std::size_t{1} << step.selected_position) - 1U;
            for (std::size_t reduced_index = 0U;
                 reduced_index < output.size();
                 ++reduced_index) {
                const std::size_t low = reduced_index & lower_mask;
                const std::size_t high = reduced_index >> step.selected_position;
                const std::size_t base_union_index =
                    low | (high << (step.selected_position + 1U));
                QComplex sum{};
                for (std::size_t selected_bit = 0U; selected_bit < 2U; ++selected_bit) {
                    const std::size_t union_index =
                        base_union_index | (selected_bit << step.selected_position);
                    QComplex product{1.0, 0.0};
                    for (const auto& input : step.inputs) {
                        const std::span<const QComplex> values =
                            node_values(term, input.node, scratch);
                        const std::size_t factor_index =
                            mapped_factor_index(union_index, input.positions);
                        if (factor_index >= values.size()) {
                            throw QStateError(
                                "Exact adjoint gradient forward input index is invalid");
                        }
                        product *= values[factor_index];
                    }
                    sum += product;
                }
                output[reduced_index] = sum;
            }
        }

        scratch.source_adjoints.resize(term.sources.size());
        for (std::size_t source = 0U; source < term.sources.size(); ++source) {
            scratch.source_adjoints[source].assign(
                term.sources[source].values.size(), QComplex{});
        }

        scratch.terminal_values.resize(term.terminal_nodes.size());
        QComplex raw_result{1.0, 0.0};
        for (std::size_t terminal = 0U;
             terminal < term.terminal_nodes.size();
             ++terminal) {
            const std::span<const QComplex> values =
                node_values(term, term.terminal_nodes[terminal], scratch);
            if (values.size() != 1U) {
                throw QStateError("Exact adjoint gradient terminal node is not scalar");
            }
            scratch.terminal_values[terminal] = values.front();
            raw_result *= values.front();
        }

        for (std::size_t terminal = 0U;
             terminal < term.terminal_nodes.size();
             ++terminal) {
            QComplex seed = term.coefficient;
            for (std::size_t other = 0U;
                 other < term.terminal_nodes.size();
                 ++other) {
                if (other != terminal) {
                    seed *= scratch.terminal_values[other];
                }
            }
            add_adjoint(
                term, term.terminal_nodes[terminal], 0U, seed, scratch);
        }

        for (std::size_t reverse_index = term.steps.size();
             reverse_index-- > 0U;) {
            const auto& step = term.steps[reverse_index];
            const auto& output_adjoint = scratch.step_adjoints[reverse_index];
            const std::size_t input_count = step.inputs.size();
            scratch.input_indices.resize(input_count);
            scratch.input_values.resize(input_count);
            scratch.prefix.resize(input_count + 1U);
            scratch.suffix.resize(input_count + 1U);
            const std::size_t lower_mask =
                step.selected_position == 0U
                    ? 0U
                    : (std::size_t{1} << step.selected_position) - 1U;

            for (std::size_t reduced_index = 0U;
                 reduced_index < output_adjoint.size();
                 ++reduced_index) {
                const QComplex output_seed = output_adjoint[reduced_index];
                if (output_seed.re == 0.0 && output_seed.im == 0.0) {
                    continue;
                }
                const std::size_t low = reduced_index & lower_mask;
                const std::size_t high = reduced_index >> step.selected_position;
                const std::size_t base_union_index =
                    low | (high << (step.selected_position + 1U));

                for (std::size_t selected_bit = 0U; selected_bit < 2U; ++selected_bit) {
                    const std::size_t union_index =
                        base_union_index | (selected_bit << step.selected_position);
                    for (std::size_t input_index = 0U;
                         input_index < input_count;
                         ++input_index) {
                        const auto& input = step.inputs[input_index];
                        const std::span<const QComplex> values =
                            node_values(term, input.node, scratch);
                        const std::size_t factor_index =
                            mapped_factor_index(union_index, input.positions);
                        if (factor_index >= values.size()) {
                            throw QStateError(
                                "Exact adjoint gradient reverse input index is invalid");
                        }
                        scratch.input_indices[input_index] = factor_index;
                        scratch.input_values[input_index] = values[factor_index];
                    }

                    scratch.prefix[0] = {1.0, 0.0};
                    for (std::size_t input_index = 0U;
                         input_index < input_count;
                         ++input_index) {
                        scratch.prefix[input_index + 1U] =
                            scratch.prefix[input_index] *
                            scratch.input_values[input_index];
                    }
                    scratch.suffix[input_count] = {1.0, 0.0};
                    for (std::size_t input_index = input_count;
                         input_index-- > 0U;) {
                        scratch.suffix[input_index] =
                            scratch.input_values[input_index] *
                            scratch.suffix[input_index + 1U];
                    }

                    for (std::size_t input_index = 0U;
                         input_index < input_count;
                         ++input_index) {
                        const QComplex contribution =
                            output_seed * scratch.prefix[input_index] *
                            scratch.suffix[input_index + 1U];
                        add_adjoint(
                            term,
                            step.inputs[input_index].node,
                            scratch.input_indices[input_index],
                            contribution,
                            scratch);
                    }
                }
            }
        }
        return raw_result;
    }
};

}  // namespace qubit
