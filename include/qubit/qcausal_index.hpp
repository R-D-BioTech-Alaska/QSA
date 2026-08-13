#pragma once

#include "qubit/qplan.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <span>
#include <unordered_set>
#include <vector>

namespace qubit {

struct ExactCausalIndexConfig {
    std::size_t max_qubits{1U << 24U};
    std::size_t max_operations{1U << 26U};
    std::size_t max_query_qubits{1U << 20U};
};

struct ExactCausalSlice {
    std::vector<QubitId> global_qubits{};
    std::vector<QubitId> local_query_qubits{};
    std::vector<Operation> operations{};
};

struct ExactCausalIndexStats {
    std::size_t qubits{0U};
    std::size_t operations{0U};
    std::size_t dependency_edges{0U};
    std::size_t estimated_bytes{0U};
};

class ExactCausalOperationIndex {
public:
    ExactCausalOperationIndex(
        std::size_t qubit_count,
        std::span<const Operation> operations,
        ExactCausalIndexConfig config = {})
        : qubit_count_(qubit_count), config_(config), operations_(operations.begin(), operations.end()) {
        validate_configuration();
        last_operation_.assign(qubit_count_, npos());
        predecessors_.resize(operations_.size(), {npos(), npos()});

        std::size_t dependency_edges = 0U;
        for (std::size_t index = 0U; index < operations_.size(); ++index) {
            const Operation& operation = operations_[index];
            const bool pair = validate_operation(operation);
            const std::size_t first = static_cast<std::size_t>(operation.first);
            predecessors_[index][0] = last_operation_[first];
            if (predecessors_[index][0] != npos()) {
                ++dependency_edges;
            }
            if (pair) {
                const std::size_t second = static_cast<std::size_t>(operation.second);
                predecessors_[index][1] = last_operation_[second];
                if (predecessors_[index][1] != npos() &&
                    predecessors_[index][1] != predecessors_[index][0]) {
                    ++dependency_edges;
                }
                last_operation_[second] = index;
            }
            last_operation_[first] = index;
        }

        std::size_t estimated = sizeof(*this);
        estimated = checked_sum(
            estimated,
            checked_product(operations_.capacity(), sizeof(Operation),
                "Causal index operation storage overflowed"),
            "Causal index estimated storage overflowed");
        estimated = checked_sum(
            estimated,
            checked_product(predecessors_.capacity(), sizeof(std::array<std::size_t, 2>),
                "Causal index predecessor storage overflowed"),
            "Causal index estimated storage overflowed");
        estimated = checked_sum(
            estimated,
            checked_product(last_operation_.capacity(), sizeof(std::size_t),
                "Causal index terminal storage overflowed"),
            "Causal index estimated storage overflowed");
        stats_ = ExactCausalIndexStats{
            qubit_count_, operations_.size(), dependency_edges, estimated,
        };
    }

    [[nodiscard]] ExactCausalSlice slice(std::span<const QubitId> query_qubits) const {
        validate_query(query_qubits);
        ExactCausalSlice result;
        if (query_qubits.empty()) {
            return result;
        }

        std::unordered_set<std::size_t> visited;
        visited.reserve(query_qubits.size() * 8U + 1U);
        std::vector<std::size_t> stack;
        stack.reserve(query_qubits.size() * 4U + 1U);
        for (const QubitId qubit : query_qubits) {
            const std::size_t terminal = last_operation_[static_cast<std::size_t>(qubit)];
            if (terminal != npos()) {
                stack.push_back(terminal);
            }
        }

        std::vector<std::size_t> selected;
        while (!stack.empty()) {
            const std::size_t index = stack.back();
            stack.pop_back();
            if (!visited.insert(index).second) {
                continue;
            }
            selected.push_back(index);
            const auto& predecessors = predecessors_[index];
            if (predecessors[0] != npos()) {
                stack.push_back(predecessors[0]);
            }
            if (predecessors[1] != npos()) {
                stack.push_back(predecessors[1]);
            }
        }
        std::sort(selected.begin(), selected.end());

        result.global_qubits.reserve(query_qubits.size() + selected.size() * 2U);
        result.global_qubits.insert(
            result.global_qubits.end(), query_qubits.begin(), query_qubits.end());
        for (const std::size_t index : selected) {
            const Operation& operation = operations_[index];
            result.global_qubits.push_back(operation.first);
            if (two_qubit(operation.code)) {
                result.global_qubits.push_back(operation.second);
            }
        }
        std::sort(result.global_qubits.begin(), result.global_qubits.end());
        result.global_qubits.erase(
            std::unique(result.global_qubits.begin(), result.global_qubits.end()),
            result.global_qubits.end());

        if (result.global_qubits.size() >
            static_cast<std::size_t>(std::numeric_limits<QubitId>::max())) {
            throw QStateError("Causal index local qubit numbering overflowed");
        }

        result.local_query_qubits.reserve(query_qubits.size());
        for (const QubitId global : query_qubits) {
            result.local_query_qubits.push_back(local_id(result.global_qubits, global));
        }

        result.operations.reserve(selected.size());
        for (const std::size_t index : selected) {
            Operation local = operations_[index];
            local.first = local_id(result.global_qubits, local.first);
            if (two_qubit(local.code)) {
                local.second = local_id(result.global_qubits, local.second);
            }
            result.operations.push_back(local);
        }
        return result;
    }

    [[nodiscard]] const ExactCausalIndexStats& stats() const noexcept { return stats_; }
    [[nodiscard]] const ExactCausalIndexConfig& config() const noexcept { return config_; }
    [[nodiscard]] std::size_t qubit_count() const noexcept { return qubit_count_; }
    [[nodiscard]] std::size_t operation_count() const noexcept { return operations_.size(); }

private:
    std::size_t qubit_count_{0U};
    ExactCausalIndexConfig config_{};
    std::vector<Operation> operations_{};
    std::vector<std::array<std::size_t, 2>> predecessors_{};
    std::vector<std::size_t> last_operation_{};
    ExactCausalIndexStats stats_{};

    void validate_configuration() const {
        if (qubit_count_ == 0U || qubit_count_ > config_.max_qubits ||
            qubit_count_ > static_cast<std::size_t>(std::numeric_limits<QubitId>::max())) {
            throw QStateError("Causal index qubit count is zero or exceeds configured/index cap");
        }
        if (config_.max_operations == 0U || operations_.size() > config_.max_operations ||
            config_.max_query_qubits == 0U) {
            throw QStateError("Causal index configuration or operation count is invalid");
        }
    }

    [[nodiscard]] bool validate_operation(const Operation& operation) const {
        const std::size_t first = static_cast<std::size_t>(operation.first);
        if (first >= qubit_count_) {
            throw QStateError("Causal index operation target is out of range");
        }
        const bool pair = two_qubit(operation.code);
        if (pair) {
            const std::size_t second = static_cast<std::size_t>(operation.second);
            if (second >= qubit_count_ || first == second) {
                throw QStateError("Causal index two-qubit operation support is invalid");
            }
        }
        switch (operation.code) {
            case OperationCode::Rx:
            case OperationCode::Ry:
            case OperationCode::Rz:
                if (!std::isfinite(operation.parameter)) {
                    throw QStateError("Causal index rotation parameter must be finite");
                }
                break;
            case OperationCode::BitFlipTrajectory:
            case OperationCode::PhaseFlipTrajectory:
            case OperationCode::DepolarizingTrajectory:
            case OperationCode::AmplitudeDampingTrajectory:
                throw QStateError("Causal index supports unitary circuit operations only");
            default:
                break;
        }
        return pair;
    }

    void validate_query(std::span<const QubitId> query_qubits) const {
        if (query_qubits.size() > config_.max_query_qubits) {
            throw QStateError("Causal index query exceeds configured qubit cap");
        }
        std::vector<QubitId> sorted(query_qubits.begin(), query_qubits.end());
        std::sort(sorted.begin(), sorted.end());
        for (std::size_t index = 0U; index < sorted.size(); ++index) {
            if (static_cast<std::size_t>(sorted[index]) >= qubit_count_) {
                throw QStateError("Causal index query qubit is out of range");
            }
            if (index != 0U && sorted[index] == sorted[index - 1U]) {
                throw QStateError("Causal index query contains a duplicate qubit");
            }
        }
    }

    [[nodiscard]] static QubitId local_id(
        const std::vector<QubitId>& global_qubits, QubitId global) {
        const auto found = std::lower_bound(global_qubits.begin(), global_qubits.end(), global);
        if (found == global_qubits.end() || *found != global) {
            throw QStateError("Causal index failed to remap retained qubit");
        }
        return static_cast<QubitId>(std::distance(global_qubits.begin(), found));
    }

    [[nodiscard]] static bool two_qubit(OperationCode code) {
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
            case OperationCode::BitFlipTrajectory:
            case OperationCode::PhaseFlipTrajectory:
            case OperationCode::DepolarizingTrajectory:
            case OperationCode::AmplitudeDampingTrajectory:
                return false;
            default:
                throw QStateError("Causal index received an unknown opcode");
        }
    }

    [[nodiscard]] static constexpr std::size_t npos() noexcept {
        return std::numeric_limits<std::size_t>::max();
    }

    [[nodiscard]] static std::size_t checked_product(
        std::size_t left, std::size_t right, const char* message) {
        if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) {
            throw QStateError(message);
        }
        return left * right;
    }

    [[nodiscard]] static std::size_t checked_sum(
        std::size_t left, std::size_t right, const char* message) {
        if (right > std::numeric_limits<std::size_t>::max() - left) {
            throw QStateError(message);
        }
        return left + right;
    }
};

}  // namespace qubit
