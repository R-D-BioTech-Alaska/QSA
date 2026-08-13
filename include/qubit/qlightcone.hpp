#pragma once

#include "qubit/qplan.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace qubit {

struct ExactCircuitLightConeConfig {
    std::size_t max_qubits{1U << 24U};
    std::size_t max_operations{1U << 26U};
    std::size_t max_active_qubits{1U << 20U};
    std::size_t max_active_operations{1U << 24U};
};

struct ExactCircuitLightConeStats {
    std::size_t full_qubits{0U};
    std::size_t full_operations{0U};
    std::size_t retained_qubits{0U};
    std::size_t active_qubits{0U};
    std::size_t active_operations{0U};
    std::size_t pruned_qubits{0U};
    std::size_t pruned_operations{0U};
    std::size_t estimated_bytes{0U};
};

class ExactCircuitLightCone {
public:
    ExactCircuitLightCone(
        std::size_t qubit_count,
        std::span<const Operation> operations,
        std::span<const std::size_t> retained_qubits,
        ExactCircuitLightConeConfig config = {})
        : qubit_count_(qubit_count),
          retained_qubits_(retained_qubits.begin(), retained_qubits.end()),
          config_(config) {
        validate_configuration(operations.size());
        validate_retained();
        build(operations);
    }

    [[nodiscard]] std::size_t qubit_count() const noexcept { return qubit_count_; }
    [[nodiscard]] std::span<const std::size_t> active_qubits() const noexcept {
        return active_qubits_;
    }
    [[nodiscard]] std::span<const std::size_t> retained_qubits() const noexcept {
        return retained_qubits_;
    }
    [[nodiscard]] std::span<const std::size_t> retained_local_qubits() const noexcept {
        return retained_local_;
    }
    [[nodiscard]] std::span<const std::size_t> selected_operation_indices() const noexcept {
        return selected_operation_indices_;
    }
    [[nodiscard]] std::span<const Operation> operations() const noexcept {
        return operations_;
    }
    [[nodiscard]] const ExactCircuitLightConeStats& stats() const noexcept { return stats_; }
    [[nodiscard]] const ExactCircuitLightConeConfig& config() const noexcept { return config_; }

    [[nodiscard]] std::size_t local_qubit(std::size_t global_qubit) const {
        if (global_qubit >= full_to_local_.size() || full_to_local_[global_qubit] == npos()) {
            throw QStateError("Light-cone qubit is outside the active support");
        }
        return full_to_local_[global_qubit];
    }

private:
    std::size_t qubit_count_{0U};
    std::vector<std::size_t> retained_qubits_{};
    ExactCircuitLightConeConfig config_{};
    std::vector<std::size_t> active_qubits_{};
    std::vector<std::size_t> full_to_local_{};
    std::vector<std::size_t> retained_local_{};
    std::vector<std::size_t> selected_operation_indices_{};
    std::vector<Operation> operations_{};
    ExactCircuitLightConeStats stats_{};

    void build(std::span<const Operation> source) {
        std::vector<std::uint8_t> active(qubit_count_, 0U);
        std::size_t active_count = 0U;
        for (const std::size_t qubit : retained_qubits_) {
            active[qubit] = 1U;
            ++active_count;
        }
        if (active_count > config_.max_active_qubits) {
            throw QStateError("Light-cone retained support exceeds active-qubit cap");
        }

        selected_operation_indices_.reserve(
            std::min(source.size(), config_.max_active_operations));
        for (std::size_t reverse = source.size(); reverse != 0U; --reverse) {
            const std::size_t index = reverse - 1U;
            const Operation& operation = source[index];
            validate_operation(operation);
            const std::size_t first = static_cast<std::size_t>(operation.first);
            const bool pair = two_qubit(operation.code);
            const std::size_t second = pair
                ? static_cast<std::size_t>(operation.second)
                : first;
            if (active[first] == 0U && (!pair || active[second] == 0U)) {
                continue;
            }
            if (selected_operation_indices_.size() >= config_.max_active_operations) {
                throw QStateError("Light-cone active operation count exceeds configured cap");
            }
            selected_operation_indices_.push_back(index);
            if (active[first] == 0U) {
                active[first] = 1U;
                ++active_count;
            }
            if (pair && active[second] == 0U) {
                active[second] = 1U;
                ++active_count;
            }
            if (active_count > config_.max_active_qubits) {
                throw QStateError("Light-cone active support exceeds configured cap");
            }
        }
        std::reverse(selected_operation_indices_.begin(), selected_operation_indices_.end());

        active_qubits_.reserve(active_count);
        full_to_local_.assign(qubit_count_, npos());
        for (std::size_t qubit = 0U; qubit < qubit_count_; ++qubit) {
            if (active[qubit] == 0U) {
                continue;
            }
            full_to_local_[qubit] = active_qubits_.size();
            active_qubits_.push_back(qubit);
        }
        if (active_qubits_.size() != active_count) {
            throw QStateError("Light-cone active support accounting is inconsistent");
        }

        retained_local_.reserve(retained_qubits_.size());
        for (const std::size_t qubit : retained_qubits_) {
            retained_local_.push_back(full_to_local_[qubit]);
        }

        operations_.reserve(selected_operation_indices_.size());
        for (const std::size_t index : selected_operation_indices_) {
            Operation operation = source[index];
            operation.first = checked_qubit_id(full_to_local_[operation.first]);
            if (two_qubit(operation.code)) {
                operation.second = checked_qubit_id(full_to_local_[operation.second]);
            }
            operations_.push_back(operation);
        }

        const std::size_t bytes = estimate_bytes();
        stats_ = ExactCircuitLightConeStats{
            qubit_count_,
            source.size(),
            retained_qubits_.size(),
            active_qubits_.size(),
            operations_.size(),
            qubit_count_ - active_qubits_.size(),
            source.size() - operations_.size(),
            bytes,
        };
    }

    void validate_configuration(std::size_t operation_count) const {
        if (qubit_count_ == 0U || qubit_count_ > config_.max_qubits ||
            qubit_count_ > static_cast<std::size_t>(std::numeric_limits<QubitId>::max()) ||
            operation_count > config_.max_operations ||
            config_.max_active_qubits == 0U || config_.max_active_operations == 0U) {
            throw QStateError("Light-cone dimensions or configuration are invalid");
        }
    }

    void validate_retained() const {
        std::vector<std::uint8_t> seen(qubit_count_, 0U);
        for (const std::size_t qubit : retained_qubits_) {
            if (qubit >= qubit_count_) {
                throw QStateError("Light-cone retained qubit is out of range");
            }
            if (seen[qubit] != 0U) {
                throw QStateError("Light-cone retained qubits contain duplicates");
            }
            seen[qubit] = 1U;
        }
    }

    void validate_operation(const Operation& operation) const {
        const std::size_t first = static_cast<std::size_t>(operation.first);
        if (first >= qubit_count_) {
            throw QStateError("Light-cone operation target is out of range");
        }
        if (two_qubit(operation.code)) {
            const std::size_t second = static_cast<std::size_t>(operation.second);
            if (second >= qubit_count_ || first == second) {
                throw QStateError("Light-cone two-qubit support is invalid");
            }
        }
        if ((operation.code == OperationCode::Rx ||
             operation.code == OperationCode::Ry ||
             operation.code == OperationCode::Rz) &&
            !std::isfinite(operation.parameter)) {
            throw QStateError("Light-cone rotation parameter must be finite");
        }
        switch (operation.code) {
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
            case OperationCode::Cnot:
            case OperationCode::Cz:
            case OperationCode::Swap:
                return;
            default:
                throw QStateError("Light-cone compiler accepts unitary circuit operations only");
        }
    }

    [[nodiscard]] std::size_t estimate_bytes() const {
        std::size_t total = sizeof(*this);
        total = checked_add(total, retained_qubits_.capacity() * sizeof(std::size_t));
        total = checked_add(total, active_qubits_.capacity() * sizeof(std::size_t));
        total = checked_add(total, full_to_local_.capacity() * sizeof(std::size_t));
        total = checked_add(total, retained_local_.capacity() * sizeof(std::size_t));
        total = checked_add(
            total, selected_operation_indices_.capacity() * sizeof(std::size_t));
        total = checked_add(total, operations_.capacity() * sizeof(Operation));
        return total;
    }

    [[nodiscard]] static bool two_qubit(OperationCode code) noexcept {
        return code == OperationCode::Cnot ||
               code == OperationCode::Cz ||
               code == OperationCode::Swap;
    }

    [[nodiscard]] static QubitId checked_qubit_id(std::size_t value) {
        if (value == npos() ||
            value > static_cast<std::size_t>(std::numeric_limits<QubitId>::max())) {
            throw QStateError("Light-cone local qubit id overflowed");
        }
        return static_cast<QubitId>(value);
    }

    [[nodiscard]] static std::size_t checked_add(
        std::size_t left,
        std::size_t right) {
        if (right > std::numeric_limits<std::size_t>::max() - left) {
            throw QStateError("Light-cone metadata size overflowed");
        }
        return left + right;
    }

    [[nodiscard]] static constexpr std::size_t npos() noexcept {
        return std::numeric_limits<std::size_t>::max();
    }
};

}  // namespace qubit
