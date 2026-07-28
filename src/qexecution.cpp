#include "qubit/qexecution.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <limits>
#include <mutex>
#include <optional>
#include <thread>

namespace qubit {
namespace {

struct ActiveOperation {
    Operation operation{};
    bool active{true};
};

[[nodiscard]] bool is_rotation(OperationCode code) noexcept {
    return code == OperationCode::Rx || code == OperationCode::Ry ||
           code == OperationCode::Rz;
}

[[nodiscard]] bool is_single_qubit(OperationCode code) {
    switch (code) {
        case OperationCode::X:
        case OperationCode::Y:
        case OperationCode::Z:
        case OperationCode::H:
        case OperationCode::S:
        case OperationCode::Sdg:
        case OperationCode::T:
        case OperationCode::Tdg:
        case OperationCode::Rx:
        case OperationCode::Ry:
        case OperationCode::Rz:
        case OperationCode::BitFlipTrajectory:
        case OperationCode::PhaseFlipTrajectory:
        case OperationCode::DepolarizingTrajectory:
        case OperationCode::AmplitudeDampingTrajectory:
            return true;
        case OperationCode::Cnot:
        case OperationCode::Cz:
        case OperationCode::Swap:
            return false;
    }
    throw QStateError("Execution plan contains an unknown opcode");
}

[[nodiscard]] std::vector<QubitId> touched_qubits(const Operation& operation) {
    if (is_single_qubit(operation.code)) {
        return {operation.first};
    }
    if (operation.first == operation.second) {
        throw QStateError("Execution plan two-qubit operation uses one qubit twice");
    }
    return {operation.first, operation.second};
}

[[nodiscard]] bool same_pair(
    const Operation& first,
    const Operation& second) noexcept {
    if (first.code == OperationCode::Cnot) {
        return first.first == second.first && first.second == second.second;
    }
    return (first.first == second.first && first.second == second.second) ||
           (first.first == second.second && first.second == second.first);
}

[[nodiscard]] bool inverse_pair(
    const Operation& first,
    const Operation& second) noexcept {
    if (first.code == second.code) {
        switch (first.code) {
            case OperationCode::X:
            case OperationCode::Y:
            case OperationCode::Z:
            case OperationCode::H:
                return first.first == second.first;
            case OperationCode::Cnot:
            case OperationCode::Cz:
            case OperationCode::Swap:
                return same_pair(first, second);
            default:
                break;
        }
    }
    if (first.first != second.first) {
        return false;
    }
    return (first.code == OperationCode::S && second.code == OperationCode::Sdg) ||
           (first.code == OperationCode::Sdg && second.code == OperationCode::S) ||
           (first.code == OperationCode::T && second.code == OperationCode::Tdg) ||
           (first.code == OperationCode::Tdg && second.code == OperationCode::T);
}

void apply_single_operation(QRegister& state, const Operation& operation) {
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
        default:
            throw QStateError("Independent component plan contains an unsupported opcode");
    }
}

}  // namespace

DependencyOperationPlan::DependencyOperationPlan(
    std::span<const Operation> operations)
    : source_operation_count_(operations.size()),
      operations_(optimize(operations)),
      compiled_(operations_, true) {}

std::vector<Operation> DependencyOperationPlan::optimize(
    std::span<const Operation> operations) {
    std::size_t qubit_count = 0U;
    for (const Operation& operation : operations) {
        const auto touched = touched_qubits(operation);
        for (QubitId qubit : touched) {
            qubit_count = std::max(
                qubit_count, static_cast<std::size_t>(qubit) + 1U);
        }
        if (is_rotation(operation.code) && !std::isfinite(operation.parameter)) {
            throw QStateError("Execution plan rotation parameter must be finite");
        }
    }

    std::vector<ActiveOperation> records;
    records.reserve(operations.size());
    std::vector<std::vector<std::size_t>> stacks(qubit_count);

    for (const Operation& operation : operations) {
        const auto touched = touched_qubits(operation);
        std::optional<std::size_t> candidate;
        if (touched.size() == 1U) {
            if (!stacks[touched[0]].empty()) {
                candidate = stacks[touched[0]].back();
            }
        } else if (!stacks[touched[0]].empty() && !stacks[touched[1]].empty() &&
                   stacks[touched[0]].back() == stacks[touched[1]].back()) {
            candidate = stacks[touched[0]].back();
        }

        if (candidate.has_value()) {
            ActiveOperation& previous = records[*candidate];
            if (inverse_pair(previous.operation, operation)) {
                previous.active = false;
                for (QubitId qubit : touched) {
                    stacks[qubit].pop_back();
                }
                continue;
            }
            if (is_rotation(previous.operation.code) &&
                previous.operation.code == operation.code &&
                previous.operation.first == operation.first) {
                const double parameter = previous.operation.parameter + operation.parameter;
                if (!std::isfinite(parameter)) {
                    throw QStateError("Execution plan fused rotation overflowed");
                }
                previous.operation.parameter = parameter;
                if (parameter == 0.0) {
                    previous.active = false;
                    stacks[touched[0]].pop_back();
                }
                continue;
            }
        }

        const std::size_t index = records.size();
        records.push_back(ActiveOperation{operation, true});
        for (QubitId qubit : touched) {
            stacks[qubit].push_back(index);
        }
    }

    std::vector<Operation> result;
    result.reserve(records.size());
    for (const ActiveOperation& record : records) {
        if (record.active) {
            result.push_back(record.operation);
        }
    }
    return result;
}

void DependencyOperationPlan::execute(QRegister& state) const {
    compiled_.execute(state);
}

void DependencyOperationPlan::execute_many(
    std::span<QRegister* const> states,
    std::size_t worker_count) const {
    compiled_.execute_many(states, worker_count);
}

IndependentComponentPlan::IndependentComponentPlan(
    std::span<const Operation> operations)
    : operations_(operations.begin(), operations.end()) {
    for (const Operation& operation : operations_) {
        if (!supported(operation.code)) {
            throw QStateError(
                "Independent component plan supports single-qubit unitaries only");
        }
        if (is_rotation(operation.code) && !std::isfinite(operation.parameter)) {
            throw QStateError("Independent component rotation parameter must be finite");
        }
    }
}

bool IndependentComponentPlan::supported(OperationCode code) noexcept {
    switch (code) {
        case OperationCode::X:
        case OperationCode::Y:
        case OperationCode::Z:
        case OperationCode::H:
        case OperationCode::S:
        case OperationCode::Sdg:
        case OperationCode::T:
        case OperationCode::Tdg:
        case OperationCode::Rx:
        case OperationCode::Ry:
        case OperationCode::Rz:
            return true;
        default:
            return false;
    }
}

void IndependentComponentPlan::execute(
    QRegister& state,
    std::size_t worker_count) const {
    if (operations_.empty()) {
        return;
    }

    std::vector<std::vector<Operation>> grouped(state.components_.size());
    for (const Operation& operation : operations_) {
        state.validate_qubit(operation.first);
        grouped[state.component_index(operation.first)].push_back(operation);
    }

    std::vector<std::size_t> active;
    active.reserve(grouped.size());
    for (std::size_t component = 0; component < grouped.size(); ++component) {
        if (!grouped[component].empty()) {
            active.push_back(component);
        }
    }
    if (active.empty()) {
        return;
    }

    if (worker_count == 0U) {
        worker_count = static_cast<std::size_t>(std::thread::hardware_concurrency());
        if (worker_count == 0U) {
            worker_count = 1U;
        }
    }
    worker_count = std::min(worker_count, active.size());
    if (worker_count <= 1U) {
        for (std::size_t component : active) {
            for (const Operation& operation : grouped[component]) {
                apply_single_operation(state, operation);
            }
        }
        return;
    }

    std::atomic<std::size_t> next{0U};
    std::atomic<bool> stop{false};
    std::mutex error_mutex;
    std::exception_ptr first_error;
    const auto worker = [&] {
        while (!stop.load(std::memory_order_relaxed)) {
            const std::size_t position = next.fetch_add(1U, std::memory_order_relaxed);
            if (position >= active.size()) {
                return;
            }
            const std::size_t component = active[position];
            try {
                for (const Operation& operation : grouped[component]) {
                    apply_single_operation(state, operation);
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

    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (std::size_t index = 0; index < worker_count; ++index) {
        workers.emplace_back(worker);
    }
    for (std::thread& thread : workers) {
        thread.join();
    }
    if (first_error != nullptr) {
        std::rethrow_exception(first_error);
    }
}

}  // namespace qubit
