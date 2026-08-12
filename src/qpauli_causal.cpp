#include "qubit/qpauli.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <utility>

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

[[nodiscard]] bool term_expanding_operation(OperationCode code) noexcept {
    switch (code) {
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

[[nodiscard]] bool parameterized_rotation(OperationCode code) noexcept {
    return code == OperationCode::Rx ||
           code == OperationCode::Ry ||
           code == OperationCode::Rz;
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

std::optional<PauliObservable> PauliPropagationPlan::try_propagate_backward(
    const PauliObservable& observable,
    const char** reason,
    PauliPropagationStats* stats) const {
    const auto fail = [reason](const char* message) -> std::optional<PauliObservable> {
        if (reason != nullptr) {
            *reason = message;
        }
        return std::nullopt;
    };

    if (observable.qubit_count() != qubit_count_) {
        return fail("Pauli observable width does not match propagation plan");
    }
    if (!observable.validate()) {
        return fail("Pauli observable failed exact validation");
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

        const Operation& operation = operations_[latest];
        if (parameterized_rotation(operation.code) && !std::isfinite(operation.parameter)) {
            return fail("Pauli rotation angle must be finite");
        }

        const std::size_t term_limit = result.config_.max_terms;
        if (term_expanding_operation(operation.code) &&
            term_limit <= std::numeric_limits<std::size_t>::max() / 2U) {
            result.config_.max_terms = term_limit * 2U;
        }
        result.apply_backward(operation);
        result.config_.max_terms = term_limit;
        if (result.term_count() > term_limit) {
            return fail("Pauli propagation exceeded max_terms");
        }

        ++local_stats.visited_operations;
        local_stats.peak_terms = std::max(local_stats.peak_terms, result.term_count());
        local_stats.peak_support = std::max(local_stats.peak_support, result.support_size());
        support = observable_support(result);
        limit = latest;
    }

    if (stats != nullptr) {
        *stats = local_stats;
    }
    if (reason != nullptr) {
        *reason = nullptr;
    }
    return result;
}

PauliObservable PauliPropagationPlan::propagate_backward(
    const PauliObservable& observable,
    PauliPropagationStats* stats) const {
    const char* reason = nullptr;
    auto result = try_propagate_backward(observable, &reason, stats);
    if (!result.has_value()) {
        throw QStateError(reason != nullptr ? reason : "Pauli propagation failed exact routing");
    }
    return std::move(*result);
}

}  // namespace qubit
