#include "qubit/qplan.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

namespace qubit {
namespace {

[[nodiscard]] bool single_qubit_matrix(const Operation& operation, QMatrix2& matrix) {
    switch (operation.code) {
        case OperationCode::X:
            matrix = gates::x();
            return true;
        case OperationCode::Y:
            matrix = gates::y();
            return true;
        case OperationCode::Z:
            matrix = gates::z();
            return true;
        case OperationCode::H:
            matrix = gates::h();
            return true;
        case OperationCode::S:
            matrix = gates::s();
            return true;
        case OperationCode::Sdg:
            matrix = gates::sdg();
            return true;
        case OperationCode::T:
            matrix = gates::t();
            return true;
        case OperationCode::Tdg:
            matrix = gates::tdg();
            return true;
        case OperationCode::Rx:
            matrix = gates::rx(operation.parameter);
            return true;
        case OperationCode::Ry:
            matrix = gates::ry(operation.parameter);
            return true;
        case OperationCode::Rz:
            matrix = gates::rz(operation.parameter);
            return true;
        default:
            return false;
    }
}

[[nodiscard]] QMatrix2 multiply(const QMatrix2& left, const QMatrix2& right) {
    QMatrix2 result{};
    for (std::size_t row = 0; row < 2; ++row) {
        for (std::size_t column = 0; column < 2; ++column) {
            QComplex value{};
            for (std::size_t inner = 0; inner < 2; ++inner) {
                value += left(row, inner) * right(inner, column);
            }
            result.values[row * 2U + column] = value;
        }
    }
    return result;
}

[[nodiscard]] bool is_identity(const QMatrix2& matrix) {
    constexpr double tolerance = 2e-13;
    return almost_equal(matrix(0, 0), QComplex{1.0, 0.0}, tolerance) &&
           almost_equal(matrix(0, 1), QComplex{}, tolerance) &&
           almost_equal(matrix(1, 0), QComplex{}, tolerance) &&
           almost_equal(matrix(1, 1), QComplex{1.0, 0.0}, tolerance);
}

[[nodiscard]] bool diagonal_phase(const Operation& operation, QDiagonalPhase& phase) {
    phase.qubit = operation.first;
    switch (operation.code) {
        case OperationCode::Z:
            phase.one = {-1.0, 0.0};
            return true;
        case OperationCode::S:
            phase.one = {0.0, 1.0};
            return true;
        case OperationCode::Sdg:
            phase.one = {0.0, -1.0};
            return true;
        case OperationCode::T:
            phase.one = QComplex::from_polar(1.0, 3.14159265358979323846 / 4.0);
            return true;
        case OperationCode::Tdg:
            phase.one = QComplex::from_polar(1.0, -3.14159265358979323846 / 4.0);
            return true;
        case OperationCode::Rz:
            phase.zero = QComplex::from_polar(1.0, -operation.parameter / 2.0);
            phase.one = QComplex::from_polar(1.0, operation.parameter / 2.0);
            return true;
        default:
            return false;
    }
}

[[nodiscard]] bool phase_is_identity(const QDiagonalPhase& phase) {
    constexpr double tolerance = 2e-13;
    return almost_equal(phase.zero, QComplex{1.0, 0.0}, tolerance) &&
           almost_equal(phase.one, QComplex{1.0, 0.0}, tolerance);
}

} 

OperationPlan::OperationPlan(std::span<const Operation> operations, bool optimize)
    : source_operation_count_(operations.size()), optimized_(optimize) {
    steps_.reserve(operations.size());

    if (!optimize) {
        for (std::size_t index = 0; index < operations.size(); ++index) {
            Step step;
            step.kind = StepKind::Direct;
            step.operation = operations[index];
            step.source_begin = index;
            step.source_end = index + 1U;
            steps_.push_back(std::move(step));
        }
        return;
    }

    bool have_fused = false;
    QubitId fused_qubit = 0;
    QMatrix2 fused_matrix{};
    Operation fused_first{};
    std::size_t fused_count = 0;
    std::size_t fused_source_begin = 0;
    std::size_t fused_source_end = 0;

    bool have_diagonal = false;
    std::vector<QDiagonalPhase> diagonal;
    std::unordered_map<QubitId, std::size_t> diagonal_positions;
    Operation diagonal_first{};
    std::size_t diagonal_count = 0;
    std::size_t diagonal_source_begin = 0;
    std::size_t diagonal_source_end = 0;

    auto flush = [&]() {
        if (!have_fused) {
            return;
        }
        if (fused_count == 1U) {
            Step step;
            step.kind = StepKind::Direct;
            step.operation = fused_first;
            step.source_begin = fused_source_begin;
            step.source_end = fused_source_end;
            steps_.push_back(std::move(step));
        } else if (!is_identity(fused_matrix)) {
            Step step;
            step.kind = StepKind::Matrix2;
            step.operation.first = fused_qubit;
            step.matrix = fused_matrix;
            step.source_begin = fused_source_begin;
            step.source_end = fused_source_end;
            steps_.push_back(std::move(step));
        }
        have_fused = false;
        fused_count = 0;
    };

    auto flush_diagonal = [&]() {
        if (!have_diagonal) {
            return;
        }
        if (diagonal_count == 1U) {
            Step step;
            step.kind = StepKind::Direct;
            step.operation = diagonal_first;
            step.source_begin = diagonal_source_begin;
            step.source_end = diagonal_source_end;
            steps_.push_back(std::move(step));
        } else {
            std::erase_if(diagonal, phase_is_identity);
            if (!diagonal.empty()) {
                Step step;
                step.kind = StepKind::Diagonal;
                step.diagonal = diagonal;
                step.source_begin = diagonal_source_begin;
                step.source_end = diagonal_source_end;
                steps_.push_back(std::move(step));
            }
        }
        have_diagonal = false;
        diagonal.clear();
        diagonal_positions.clear();
        diagonal_count = 0;
    };

    for (std::size_t index = 0; index < operations.size(); ++index) {
        const Operation& operation = operations[index];
        QDiagonalPhase phase{};
        if (diagonal_phase(operation, phase)) {
            flush();
            if (!have_diagonal) {
                have_diagonal = true;
                diagonal_first = operation;
                diagonal_source_begin = index;
            }
            const auto [iterator, inserted] = diagonal_positions.emplace(
                phase.qubit, diagonal.size());
            if (inserted) {
                diagonal.push_back(phase);
            } else {
                QDiagonalPhase& existing = diagonal[iterator->second];
                existing.zero = phase.zero * existing.zero;
                existing.one = phase.one * existing.one;
            }
            ++diagonal_count;
            diagonal_source_end = index + 1U;
            continue;
        }

        flush_diagonal();
        QMatrix2 matrix{};
        if (single_qubit_matrix(operation, matrix)) {
            if (have_fused && operation.first == fused_qubit) {
                fused_matrix = multiply(matrix, fused_matrix);
                ++fused_count;
                fused_source_end = index + 1U;
            } else {
                flush();
                have_fused = true;
                fused_qubit = operation.first;
                fused_matrix = matrix;
                fused_first = operation;
                fused_count = 1U;
                fused_source_begin = index;
                fused_source_end = index + 1U;
            }
            continue;
        }

        flush();
        Step step;
        step.kind = StepKind::Direct;
        step.operation = operation;
        step.source_begin = index;
        step.source_end = index + 1U;
        steps_.push_back(std::move(step));
    }
    flush();
    flush_diagonal();
}

void OperationPlan::apply_direct(QRegister& state, const Operation& operation) {
    switch (operation.code) {
        case OperationCode::X:
            state.apply_x(operation.first);
            break;
        case OperationCode::Y:
            state.apply_y(operation.first);
            break;
        case OperationCode::Z:
            state.apply_z(operation.first);
            break;
        case OperationCode::H:
            state.apply_h(operation.first);
            break;
        case OperationCode::S:
            state.apply_s(operation.first);
            break;
        case OperationCode::Sdg:
            state.apply_sdg(operation.first);
            break;
        case OperationCode::T:
            state.apply_t(operation.first);
            break;
        case OperationCode::Tdg:
            state.apply_tdg(operation.first);
            break;
        case OperationCode::Rx:
            state.apply_rx(operation.first, operation.parameter);
            break;
        case OperationCode::Ry:
            state.apply_ry(operation.first, operation.parameter);
            break;
        case OperationCode::Rz:
            state.apply_rz(operation.first, operation.parameter);
            break;
        case OperationCode::Cnot:
            state.apply_cnot(operation.first, operation.second);
            break;
        case OperationCode::Cz:
            state.apply_cz(operation.first, operation.second);
            break;
        case OperationCode::Swap:
            state.apply_swap(operation.first, operation.second);
            break;
        case OperationCode::BitFlipTrajectory:
            state.apply_bit_flip_trajectory(operation.first, operation.parameter, operation.sample);
            break;
        case OperationCode::PhaseFlipTrajectory:
            state.apply_phase_flip_trajectory(operation.first, operation.parameter, operation.sample);
            break;
        case OperationCode::DepolarizingTrajectory:
            state.apply_depolarizing_trajectory(operation.first, operation.parameter, operation.sample);
            break;
        case OperationCode::AmplitudeDampingTrajectory:
            state.apply_amplitude_damping_trajectory(
                operation.first, operation.parameter, operation.sample);
            break;
        default:
            throw QStateError("Operation plan contains an unknown opcode");
    }
}

void OperationPlan::execute(QRegister& state, std::size_t* completed_operations) const {
    if (completed_operations != nullptr) {
        *completed_operations = 0U;
    }
    for (std::size_t index = 0; index < steps_.size(); ++index) {
        const Step& step = steps_[index];
        if (completed_operations != nullptr) {
            *completed_operations = step.source_begin;
        }
        try {
            if (step.kind == StepKind::Matrix2) {
                state.apply_single(step.operation.first, step.matrix);
            } else if (step.kind == StepKind::Diagonal) {
                state.apply_diagonal(step.diagonal);
            } else {
                apply_direct(state, step.operation);
            }
        } catch (const std::exception& error) {
            throw QStateError(
                "Compiled plan step " + std::to_string(index) + " failed: " + error.what());
        }
        if (completed_operations != nullptr) {
            *completed_operations = step.source_end;
        }
    }
    if (completed_operations != nullptr) {
        *completed_operations = source_operation_count_;
    }
}

void OperationPlan::execute_many(
    std::span<QRegister* const> states,
    std::size_t worker_count,
    std::size_t* completed_states) const {
    if (completed_states != nullptr) {
        *completed_states = 0U;
    }
    if (states.empty()) {
        return;
    }
    for (QRegister* state : states) {
        if (state == nullptr) {
            throw QStateError("Operation plan state list contains a null register");
        }
    }

    if (worker_count == 0U) {
        worker_count = static_cast<std::size_t>(std::thread::hardware_concurrency());
        if (worker_count == 0U) {
            worker_count = 1U;
        }
    }
    worker_count = std::min(worker_count, states.size());
    if (worker_count <= 1U || states.size() == 1U) {
        for (std::size_t index = 0; index < states.size(); ++index) {
            execute(*states[index]);
            if (completed_states != nullptr) {
                *completed_states = index + 1U;
            }
        }
        return;
    }

    std::atomic<std::size_t> next{0U};
    std::atomic<std::size_t> completed{0U};
    std::atomic<bool> stop{false};
    std::mutex error_mutex;
    std::exception_ptr first_error;

    const auto worker = [&]() {
        while (!stop.load(std::memory_order_relaxed)) {
            const std::size_t index = next.fetch_add(1U, std::memory_order_relaxed);
            if (index >= states.size()) {
                return;
            }
            try {
                execute(*states[index]);
                completed.fetch_add(1U, std::memory_order_relaxed);
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
    threads.reserve(worker_count);
    for (std::size_t index = 0; index < worker_count; ++index) {
        threads.emplace_back(worker);
    }
    for (std::thread& thread : threads) {
        thread.join();
    }

    if (completed_states != nullptr) {
        *completed_states = completed.load(std::memory_order_relaxed);
    }
    if (first_error != nullptr) {
        std::rethrow_exception(first_error);
    }
}

ParameterizedOperationPlan::ParameterizedOperationPlan(
    std::span<const ParameterizedOperation> operations,
    bool optimize)
    : operations_(operations.begin(), operations.end()), optimize_(optimize) {
    for (const ParameterizedOperation& operation : operations_) {
        if (operation.parameter_slot < -1 || operation.sample_slot < -1) {
            throw QStateError("Parameterized plan slot indices must be -1 or nonnegative");
        }
        if (operation.parameter_slot >= 0) {
            parameter_count_ = std::max(
                parameter_count_, static_cast<std::size_t>(operation.parameter_slot) + 1U);
        }
        if (operation.sample_slot >= 0) {
            parameter_count_ = std::max(
                parameter_count_, static_cast<std::size_t>(operation.sample_slot) + 1U);
        }
    }
}

OperationPlan ParameterizedOperationPlan::bind(std::span<const double> parameters) const {
    if (parameters.size() < parameter_count_) {
        throw QStateError("Parameterized plan received too few parameter values");
    }
    std::vector<Operation> bound;
    bound.reserve(operations_.size());
    for (const ParameterizedOperation& templated : operations_) {
        Operation operation = templated.operation;
        if (templated.parameter_slot >= 0) {
            operation.parameter = parameters[static_cast<std::size_t>(templated.parameter_slot)];
        }
        if (templated.sample_slot >= 0) {
            operation.sample = parameters[static_cast<std::size_t>(templated.sample_slot)];
        }
        bound.push_back(operation);
    }
    return OperationPlan(bound, optimize_);
}

void ParameterizedOperationPlan::execute(
    QRegister& state,
    std::span<const double> parameters,
    std::size_t* completed_operations) const {
    OperationPlan plan = bind(parameters);
    plan.execute(state, completed_operations);
}

void ParameterizedOperationPlan::execute_many(
    std::span<QRegister* const> states,
    std::span<const double> parameters,
    std::size_t worker_count,
    std::size_t* completed_states) const {
    OperationPlan plan = bind(parameters);
    plan.execute_many(states, worker_count, completed_states);
}

} 
