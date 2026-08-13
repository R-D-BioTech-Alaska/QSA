#pragma once

#include "qubit/qcausal_index.hpp"
#include "qubit/qfabric.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace qubit {

struct ExactMarginalCompilerConfig {
    ExactComponentFabricConfig fabric{};
    std::size_t max_causal_operations{1U << 24U};
};

struct ExactMarginalCompilerStats {
    std::size_t original_qubits{0U};
    std::size_t original_operations{0U};
    std::size_t causal_qubits{0U};
    std::size_t causal_operations{0U};
    std::size_t removed_qubits{0U};
    std::size_t removed_operations{0U};
    std::size_t components{0U};
    std::size_t prepared_components{0U};
    std::size_t estimated_bytes{0U};
};

class ExactMarginalCompilerPlan {
public:
    ExactMarginalCompilerPlan(
        std::size_t qubit_count,
        std::span<const Operation> operations,
        std::span<const QubitId> query_qubits,
        ExactMarginalCompilerConfig config = {})
        : qubit_count_(qubit_count), config_(config) {
        validate_configuration(operations.size());
        validate_query(query_qubits);

        if (query_qubits.empty()) {
            stats_ = ExactMarginalCompilerStats{
                qubit_count_, operations.size(), 0U, 0U,
                qubit_count_, operations.size(), 0U, 0U, 0U,
            };
            return;
        }

        std::vector<std::uint8_t> active(qubit_count_, 0U);
        for (const QubitId qubit : query_qubits) {
            active[static_cast<std::size_t>(qubit)] = 1U;
        }

        std::vector<std::uint8_t> keep(operations.size(), 0U);
        std::size_t retained_operations = 0U;
        for (std::size_t cursor = operations.size(); cursor-- > 0U;) {
            const Operation& operation = operations[cursor];
            validate_support(operation);
            const std::size_t first = static_cast<std::size_t>(operation.first);
            if (two_qubit(operation.code)) {
                const std::size_t second = static_cast<std::size_t>(operation.second);
                if (active[first] != 0U || active[second] != 0U) {
                    keep[cursor] = 1U;
                    active[first] = 1U;
                    active[second] = 1U;
                    ++retained_operations;
                }
            } else if (active[first] != 0U) {
                keep[cursor] = 1U;
                ++retained_operations;
            }
        }
        if (retained_operations > config_.max_causal_operations) {
            throw QStateError("Marginal compiler causal operation count exceeds configured cap");
        }

        global_to_local_.assign(qubit_count_, npos());
        causal_global_qubits_.reserve(qubit_count_);
        for (std::size_t global = 0U; global < qubit_count_; ++global) {
            if (active[global] == 0U) {
                continue;
            }
            if (causal_global_qubits_.size() >=
                static_cast<std::size_t>(std::numeric_limits<QubitId>::max())) {
                throw QStateError("Marginal compiler causal qubit index overflowed");
            }
            global_to_local_[global] = causal_global_qubits_.size();
            causal_global_qubits_.push_back(static_cast<QubitId>(global));
        }

        local_query_qubits_.reserve(query_qubits.size());
        for (const QubitId qubit : query_qubits) {
            const std::size_t local = global_to_local_[static_cast<std::size_t>(qubit)];
            if (local == npos()) {
                throw QStateError("Marginal compiler lost a query qubit from its causal support");
            }
            local_query_qubits_.push_back(static_cast<QubitId>(local));
        }

        causal_operations_.reserve(retained_operations);
        for (std::size_t index = 0U; index < operations.size(); ++index) {
            if (keep[index] == 0U) {
                continue;
            }
            Operation local = operations[index];
            local.first = local_id(operation_global(local.first));
            if (two_qubit(local.code)) {
                local.second = local_id(operation_global(local.second));
            }
            causal_operations_.push_back(local);
        }
        if (causal_operations_.size() != retained_operations) {
            throw QStateError("Marginal compiler causal operation accounting mismatch");
        }

        fabric_.emplace(ExactComponentProbabilityPlan::for_marginals(
            causal_global_qubits_.size(),
            causal_operations_,
            local_query_qubits_,
            config_.fabric));

        const ExactComponentFabricStats& fabric_stats = fabric_->stats();
        std::size_t estimated = fabric_stats.estimated_bytes;
        estimated = checked_sum(
            estimated,
            checked_product(global_to_local_.capacity(), sizeof(std::size_t),
                "Marginal compiler global map storage overflowed"),
            "Marginal compiler estimated storage overflowed");
        estimated = checked_sum(
            estimated,
            checked_product(causal_global_qubits_.capacity(), sizeof(QubitId),
                "Marginal compiler causal qubit storage overflowed"),
            "Marginal compiler estimated storage overflowed");
        estimated = checked_sum(
            estimated,
            checked_product(local_query_qubits_.capacity(), sizeof(QubitId),
                "Marginal compiler query storage overflowed"),
            "Marginal compiler estimated storage overflowed");
        estimated = checked_sum(
            estimated,
            checked_product(causal_operations_.capacity(), sizeof(Operation),
                "Marginal compiler operation storage overflowed"),
            "Marginal compiler estimated storage overflowed");

        stats_ = ExactMarginalCompilerStats{
            qubit_count_,
            operations.size(),
            causal_global_qubits_.size(),
            causal_operations_.size(),
            qubit_count_ - causal_global_qubits_.size(),
            operations.size() - causal_operations_.size(),
            fabric_stats.components,
            fabric_stats.prepared_components,
            estimated,
        };
    }

    [[nodiscard]] double probability(std::span<const std::uint8_t> bits) const {
        if (bits.size() != local_query_qubits_.size()) {
            throw QStateError("Marginal compiler query bit count does not match prepared query");
        }
        if (local_query_qubits_.empty()) {
            return 1.0;
        }
        return fabric_->marginal_probability(local_query_qubits_, bits);
    }

    [[nodiscard]] const ExactMarginalCompilerStats& stats() const noexcept { return stats_; }
    [[nodiscard]] const std::vector<QubitId>& causal_global_qubits() const noexcept {
        return causal_global_qubits_;
    }
    [[nodiscard]] const std::vector<Operation>& causal_operations() const noexcept {
        return causal_operations_;
    }
    [[nodiscard]] const std::vector<ExactComponentReceipt>& component_receipts() const noexcept {
        return fabric_ ? fabric_->receipts() : empty_receipts_;
    }
    [[nodiscard]] const ExactMarginalCompilerConfig& config() const noexcept { return config_; }

private:
    std::size_t qubit_count_{0U};
    ExactMarginalCompilerConfig config_{};
    std::vector<std::size_t> global_to_local_{};
    std::vector<QubitId> causal_global_qubits_{};
    std::vector<QubitId> local_query_qubits_{};
    std::vector<Operation> causal_operations_{};
    std::optional<ExactComponentProbabilityPlan> fabric_{};
    ExactMarginalCompilerStats stats_{};
    std::vector<ExactComponentReceipt> empty_receipts_{};

    void validate_configuration(std::size_t operation_count) const {
        if (qubit_count_ == 0U ||
            qubit_count_ > config_.fabric.max_qubits ||
            qubit_count_ > static_cast<std::size_t>(std::numeric_limits<QubitId>::max())) {
            throw QStateError("Marginal compiler qubit count is zero or exceeds configured/index cap");
        }
        if (config_.max_causal_operations == 0U ||
            operation_count > config_.fabric.max_operations) {
            throw QStateError("Marginal compiler configuration or operation count is invalid");
        }
    }

    void validate_query(std::span<const QubitId> query_qubits) const {
        std::vector<std::uint8_t> seen(qubit_count_, 0U);
        for (const QubitId qubit : query_qubits) {
            const std::size_t global = static_cast<std::size_t>(qubit);
            if (global >= qubit_count_) {
                throw QStateError("Marginal compiler query qubit is out of range");
            }
            if (seen[global] != 0U) {
                throw QStateError("Marginal compiler query contains a duplicate qubit");
            }
            seen[global] = 1U;
        }
    }

    void validate_support(const Operation& operation) const {
        const std::size_t first = static_cast<std::size_t>(operation.first);
        if (first >= qubit_count_) {
            throw QStateError("Marginal compiler operation target is out of range");
        }
        if (two_qubit(operation.code)) {
            const std::size_t second = static_cast<std::size_t>(operation.second);
            if (second >= qubit_count_) {
                throw QStateError("Marginal compiler two-qubit target is out of range");
            }
            if (first == second) {
                throw QStateError("Marginal compiler two-qubit operation requires distinct qubits");
            }
        }
    }

    [[nodiscard]] QubitId local_id(std::size_t global) const {
        const std::size_t local = global_to_local_[global];
        if (local == npos()) {
            throw QStateError("Marginal compiler retained operation references a pruned qubit");
        }
        return static_cast<QubitId>(local);
    }

    [[nodiscard]] static std::size_t operation_global(QubitId qubit) noexcept {
        return static_cast<std::size_t>(qubit);
    }

    [[nodiscard]] static constexpr std::size_t npos() noexcept {
        return std::numeric_limits<std::size_t>::max();
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
                return false;
            case OperationCode::BitFlipTrajectory:
            case OperationCode::PhaseFlipTrajectory:
            case OperationCode::DepolarizingTrajectory:
            case OperationCode::AmplitudeDampingTrajectory:
                throw QStateError(
                    "Marginal compiler causal pruning does not accept trajectory operations");
            default:
                throw QStateError("Marginal compiler received an unknown opcode");
        }
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

class ExactIndexedMarginalCompilerPlan {
public:
    ExactIndexedMarginalCompilerPlan(
        const ExactCausalOperationIndex& index,
        std::span<const QubitId> query_qubits,
        ExactMarginalCompilerConfig config = {})
        : config_(config), slice_(index.slice(query_qubits)) {
        if (config_.max_causal_operations == 0U) {
            throw QStateError("Indexed marginal compiler max_causal_operations must be positive");
        }
        if (slice_.operations.size() > config_.max_causal_operations ||
            slice_.operations.size() > config_.fabric.max_operations) {
            throw QStateError("Indexed marginal compiler causal operation count exceeds configured cap");
        }
        if (slice_.global_qubits.size() > config_.fabric.max_qubits) {
            throw QStateError("Indexed marginal compiler causal qubit count exceeds configured cap");
        }

        if (!slice_.local_query_qubits.empty()) {
            fabric_.emplace(ExactComponentProbabilityPlan::for_marginals(
                slice_.global_qubits.size(),
                slice_.operations,
                slice_.local_query_qubits,
                config_.fabric));
        }

        const std::size_t components = fabric_ ? fabric_->stats().components : 0U;
        const std::size_t prepared_components = fabric_ ? fabric_->stats().prepared_components : 0U;
        std::size_t estimated = fabric_ ? fabric_->stats().estimated_bytes : 0U;
        estimated = checked_sum(
            estimated,
            checked_product(slice_.global_qubits.capacity(), sizeof(QubitId),
                "Indexed marginal compiler causal qubit storage overflowed"),
            "Indexed marginal compiler estimated storage overflowed");
        estimated = checked_sum(
            estimated,
            checked_product(slice_.local_query_qubits.capacity(), sizeof(QubitId),
                "Indexed marginal compiler query storage overflowed"),
            "Indexed marginal compiler estimated storage overflowed");
        estimated = checked_sum(
            estimated,
            checked_product(slice_.operations.capacity(), sizeof(Operation),
                "Indexed marginal compiler operation storage overflowed"),
            "Indexed marginal compiler estimated storage overflowed");

        stats_ = ExactMarginalCompilerStats{
            index.qubit_count(),
            index.operation_count(),
            slice_.global_qubits.size(),
            slice_.operations.size(),
            index.qubit_count() - slice_.global_qubits.size(),
            index.operation_count() - slice_.operations.size(),
            components,
            prepared_components,
            estimated,
        };
    }

    [[nodiscard]] double probability(std::span<const std::uint8_t> bits) const {
        if (bits.size() != slice_.local_query_qubits.size()) {
            throw QStateError("Indexed marginal compiler query bit count does not match prepared query");
        }
        if (slice_.local_query_qubits.empty()) {
            return 1.0;
        }
        return fabric_->marginal_probability(slice_.local_query_qubits, bits);
    }

    [[nodiscard]] const ExactMarginalCompilerStats& stats() const noexcept { return stats_; }
    [[nodiscard]] const std::vector<QubitId>& causal_global_qubits() const noexcept {
        return slice_.global_qubits;
    }
    [[nodiscard]] const std::vector<Operation>& causal_operations() const noexcept {
        return slice_.operations;
    }
    [[nodiscard]] const std::vector<ExactComponentReceipt>& component_receipts() const noexcept {
        return fabric_ ? fabric_->receipts() : empty_receipts_;
    }
    [[nodiscard]] const ExactMarginalCompilerConfig& config() const noexcept { return config_; }

private:
    ExactMarginalCompilerConfig config_{};
    ExactCausalSlice slice_{};
    std::optional<ExactComponentProbabilityPlan> fabric_{};
    ExactMarginalCompilerStats stats_{};
    std::vector<ExactComponentReceipt> empty_receipts_{};

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
