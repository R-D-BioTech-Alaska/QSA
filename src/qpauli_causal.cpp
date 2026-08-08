#include "qubit/qpauli.hpp"

#include <algorithm>
#include <iterator>
#include <limits>

namespace qubit {
namespace {

[[nodiscard]] bool two_qubit_operation(OperationCode code) {
    switch (code) {
        case OperationCode::Cnot:
        case OperationCode::Cz:
        case OperationCode::Swap:
            return true;
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
            return false;
        case OperationCode::BitFlipTrajectory:
        case OperationCode::PhaseFlipTrajectory:
        case OperationCode::DepolarizingTrajectory:
        case OperationCode::AmplitudeDampingTrajectory:
            throw QStateError("Pauli propagation plans do not accept trajectory noise operations");
    }
    throw QStateError("Pauli propagation plan received an unknown opcode");
}

[[nodiscard]] std::vector<QubitId> observable_support(const PauliObservable& observable) {
    std::vector<QubitId> support;
    support.reserve(observable.support_size());
    for (const PauliTerm& term : observable.terms()) {
        for (const PauliFactor& factor : term.factors) {
            support.push_back(factor.qubit);
        }
    }
    std::sort(support.begin(), support.end());
    support.erase(std::unique(support.begin(), support.end()), support.end());
    return support;
}

[[nodiscard]] std::size_t latest_before(
    const std::vector<std::size_t>& indices,
    std::size_t limit) noexcept {
    const auto position = std::lower_bound(indices.begin(), indices.end(), limit);
    if (position == indices.begin()) {
        return std::numeric_limits<std::size_t>::max();
    }
    return *std::prev(position);
}

}  // namespace

PauliPropagationPlan::PauliPropagationPlan(
    std::size_t qubit_count,
    std::span<const Operation> operations)
    : qubit_count_(qubit_count),
      operations_(operations.begin(), operations.end()),
      operation_indices_(qubit_count) {
    if (qubit_count_ == 0U) {
        throw QStateError("Pauli propagation plan requires at least one qubit");
    }

    for (std::size_t index = 0; index < operations_.size(); ++index) {
        const Operation& operation = operations_[index];
        if (operation.first >= qubit_count_) {
            throw QStateError("Pauli propagation plan operation is outside the register");
        }
        const bool two_qubit = two_qubit_operation(operation.code);
        operation_indices_[operation.first].push_back(index);
        ++indexed_reference_count_;
        if (!two_qubit) {
            continue;
        }
        if (operation.second >= qubit_count_ || operation.second == operation.first) {
            throw QStateError("Pauli propagation plan two-qubit operation is invalid");
        }
        operation_indices_[operation.second].push_back(index);
        ++indexed_reference_count_;
    }
}

std::size_t PauliPropagationPlan::estimated_bytes() const noexcept {
    std::size_t total = sizeof(*this);
    total += operations_.capacity() * sizeof(Operation);
    total += operation_indices_.capacity() * sizeof(std::vector<std::size_t>);
    for (const auto& indices : operation_indices_) {
        total += indices.capacity() * sizeof(std::size_t);
    }
    return total;
}

PauliObservable PauliPropagationPlan::propagate_backward(
    const PauliObservable& observable,
    PauliPropagationStats* stats) const {
    if (observable.qubit_count() != qubit_count_) {
        throw QStateError("Pauli observable width does not match propagation plan");
    }
    if (!observable.validate()) {
        throw QStateError("Pauli observable failed exact validation");
    }

    PauliObservable result = observable;
    PauliPropagationStats local_stats;
    local_stats.source_operations = operations_.size();
    local_stats.peak_terms = result.term_count();
    local_stats.peak_support = result.support_size();

    std::vector<QubitId> support = observable_support(result);
    std::size_t limit = operations_.size();
    constexpr std::size_t none = std::numeric_limits<std::size_t>::max();

    while (!support.empty()) {
        std::size_t latest = none;
        for (QubitId qubit : support) {
            const std::size_t candidate = latest_before(operation_indices_[qubit], limit);
            if (candidate != none && (latest == none || candidate > latest)) {
                latest = candidate;
            }
        }
        if (latest == none) {
            break;
        }

        result.apply_backward(operations_[latest]);
        ++local_stats.visited_operations;
        local_stats.peak_terms = std::max(local_stats.peak_terms, result.term_count());
        local_stats.peak_support = std::max(local_stats.peak_support, result.support_size());
        support = observable_support(result);
        limit = latest;
    }

    if (stats != nullptr) {
        *stats = local_stats;
    }
    return result;
}

}  // namespace qubit
