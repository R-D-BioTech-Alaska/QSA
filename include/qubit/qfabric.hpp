#pragma once

#include "qubit/qbroker.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace qubit {

struct ExactComponentFabricConfig {
    ExactExecutionBrokerConfig broker{};
    std::size_t max_qubits{1U << 24U};
    std::size_t max_components{1U << 20U};
    std::size_t max_operations{1U << 24U};
};

struct ExactComponentReceipt {
    std::size_t index{0U};
    std::size_t qubits{0U};
    std::size_t operations{0U};
    ExactExecutionRoute route{ExactExecutionRoute::Register};
    std::size_t estimated_bytes{0U};
};

struct ExactComponentFabricStats {
    std::size_t qubits{0U};
    std::size_t operations{0U};
    std::size_t components{0U};
    std::size_t largest_component_qubits{0U};
    std::size_t prepared_components{0U};
    std::size_t estimated_bytes{0U};
};

class ExactComponentProbabilityPlan {
public:
    ExactComponentProbabilityPlan(
        std::size_t qubit_count,
        std::span<const Operation> operations,
        ExactComponentFabricConfig config = {})
        : qubit_count_(qubit_count), config_(config) {
        if (qubit_count_ == 0U || qubit_count_ > config_.max_qubits ||
            qubit_count_ > static_cast<std::size_t>(std::numeric_limits<QubitId>::max())) {
            throw QStateError("Component fabric qubit count is zero or exceeds configured/index cap");
        }
        if (config_.max_components == 0U || config_.max_operations == 0U ||
            operations.size() > config_.max_operations) {
            throw QStateError("Component fabric configuration or operation count is invalid");
        }

        std::vector<std::size_t> parent(qubit_count_);
        std::vector<std::uint8_t> rank(qubit_count_, 0U);
        for (std::size_t qubit = 0U; qubit < qubit_count_; ++qubit) {
            parent[qubit] = qubit;
        }

        const auto find_root = [&](std::size_t value, auto&& self) -> std::size_t {
            while (parent[value] != value) {
                parent[value] = parent[parent[value]];
                value = parent[value];
            }
            return value;
        };
        const auto unite = [&](std::size_t left, std::size_t right, auto&& find, auto&& self) -> void {
            std::size_t a = find(left, find);
            std::size_t b = find(right, find);
            if (a == b) {
                return;
            }
            if (rank[a] < rank[b]) {
                std::swap(a, b);
            }
            parent[b] = a;
            if (rank[a] == rank[b]) {
                ++rank[a];
            }
        };

        for (const Operation& operation : operations) {
            validate_support(operation);
            if (two_qubit(operation.code)) {
                unite(
                    static_cast<std::size_t>(operation.first),
                    static_cast<std::size_t>(operation.second),
                    find_root,
                    unite);
            }
        }

        global_component_.assign(qubit_count_, 0U);
        global_local_.assign(qubit_count_, 0U);
        std::vector<std::size_t> root_component(qubit_count_, npos());
        for (std::size_t qubit = 0U; qubit < qubit_count_; ++qubit) {
            const std::size_t root = find_root(qubit, find_root);
            if (root_component[root] == npos()) {
                if (components_.size() >= config_.max_components) {
                    throw QStateError("Component fabric exceeds configured component cap");
                }
                root_component[root] = components_.size();
                components_.push_back(Component{});
            }
            const std::size_t component_index = root_component[root];
            global_component_[qubit] = component_index;
            global_local_[qubit] = components_[component_index].global_qubits.size();
            components_[component_index].global_qubits.push_back(static_cast<QubitId>(qubit));
        }

        for (const Operation& operation : operations) {
            const std::size_t component_index = global_component_[operation.first];
            Operation local = operation;
            local.first = static_cast<QubitId>(global_local_[operation.first]);
            if (two_qubit(operation.code)) {
                if (global_component_[operation.second] != component_index) {
                    throw QStateError("Component fabric dependency partition is inconsistent");
                }
                local.second = static_cast<QubitId>(global_local_[operation.second]);
            }
            components_[component_index].operations.push_back(local);
        }

        std::size_t estimated = checked_product(
            qubit_count_, sizeof(std::size_t) * 2U,
            "Component fabric index storage overflowed");
        std::size_t largest = 0U;
        std::size_t prepared = 0U;
        receipts_.reserve(components_.size());
        for (std::size_t index = 0U; index < components_.size(); ++index) {
            Component& component = components_[index];
            largest = std::max(largest, component.global_qubits.size());
            std::size_t component_bytes = checked_sum(
                checked_product(
                    component.global_qubits.size(), sizeof(QubitId),
                    "Component fabric qubit receipt overflowed"),
                checked_product(
                    component.operations.size(), sizeof(Operation),
                    "Component fabric operation storage overflowed"),
                "Component fabric component storage overflowed");
            ExactExecutionRoute route = ExactExecutionRoute::BasisPermutation;
            if (!component.operations.empty()) {
                auto plan = ExactPreparedProbabilityPlan::for_marginals(
                    component.global_qubits.size(), component.operations, config_.broker);
                component.plan = std::make_unique<ExactPreparedProbabilityPlan>(std::move(plan));
                route = component.plan->prepared_route();
                component_bytes = checked_sum(
                    component_bytes, component.plan->estimated_bytes(),
                    "Component fabric prepared storage overflowed");
                ++prepared;
            }
            estimated = checked_sum(
                estimated, component_bytes,
                "Component fabric total estimated storage overflowed");
            receipts_.push_back(ExactComponentReceipt{
                index,
                component.global_qubits.size(),
                component.operations.size(),
                route,
                component_bytes,
            });
        }

        stats_ = ExactComponentFabricStats{
            qubit_count_,
            operations.size(),
            components_.size(),
            largest,
            prepared,
            estimated,
        };
    }

    [[nodiscard]] double basis_probability(std::span<const std::uint8_t> bits) const {
        if (bits.size() != qubit_count_) {
            throw QStateError("Component fabric basis query size does not match qubit count");
        }
        double probability = 1.0;
        std::vector<QubitId> local_qubits;
        std::vector<std::uint8_t> local_bits;
        for (const Component& component : components_) {
            local_qubits.resize(component.global_qubits.size());
            local_bits.resize(component.global_qubits.size());
            for (std::size_t local = 0U; local < component.global_qubits.size(); ++local) {
                const std::uint8_t bit = bits[component.global_qubits[local]];
                validate_bit(bit);
                local_qubits[local] = static_cast<QubitId>(local);
                local_bits[local] = bit;
            }
            if (component.plan == nullptr) {
                if (std::any_of(local_bits.begin(), local_bits.end(), [](std::uint8_t bit) {
                        return bit != 0U;
                    })) {
                    return 0.0;
                }
                continue;
            }
            probability *= component.plan->marginal_probability(local_qubits, local_bits).value;
            if (probability == 0.0) {
                return 0.0;
            }
        }
        return probability;
    }

    [[nodiscard]] double marginal_probability(
        std::span<const QubitId> qubits,
        std::span<const std::uint8_t> bits) const {
        if (qubits.size() != bits.size()) {
            throw QStateError("Component fabric marginal qubit/bit sizes differ");
        }
        std::vector<std::uint8_t> seen(qubit_count_, 0U);
        std::vector<std::vector<QubitId>> local_qubits(components_.size());
        std::vector<std::vector<std::uint8_t>> local_bits(components_.size());
        for (std::size_t index = 0U; index < qubits.size(); ++index) {
            const std::size_t global = static_cast<std::size_t>(qubits[index]);
            if (global >= qubit_count_) {
                throw QStateError("Component fabric marginal qubit is out of range");
            }
            validate_bit(bits[index]);
            if (seen[global] != 0U) {
                throw QStateError("Component fabric marginal contains a duplicate qubit");
            }
            seen[global] = 1U;
            const std::size_t component_index = global_component_[global];
            local_qubits[component_index].push_back(
                static_cast<QubitId>(global_local_[global]));
            local_bits[component_index].push_back(bits[index]);
        }

        double probability = 1.0;
        for (std::size_t index = 0U; index < components_.size(); ++index) {
            if (local_qubits[index].empty()) {
                continue;
            }
            const Component& component = components_[index];
            if (component.plan == nullptr) {
                if (std::any_of(local_bits[index].begin(), local_bits[index].end(), [](std::uint8_t bit) {
                        return bit != 0U;
                    })) {
                    return 0.0;
                }
                continue;
            }
            probability *= component.plan->marginal_probability(
                local_qubits[index], local_bits[index]).value;
            if (probability == 0.0) {
                return 0.0;
            }
        }
        return probability;
    }

    [[nodiscard]] const ExactComponentFabricStats& stats() const noexcept { return stats_; }
    [[nodiscard]] const std::vector<ExactComponentReceipt>& receipts() const noexcept {
        return receipts_;
    }
    [[nodiscard]] const ExactComponentFabricConfig& config() const noexcept { return config_; }

private:
    struct Component {
        std::vector<QubitId> global_qubits{};
        std::vector<Operation> operations{};
        std::unique_ptr<ExactPreparedProbabilityPlan> plan{};
    };

    std::size_t qubit_count_{0U};
    ExactComponentFabricConfig config_{};
    std::vector<Component> components_{};
    std::vector<std::size_t> global_component_{};
    std::vector<std::size_t> global_local_{};
    std::vector<ExactComponentReceipt> receipts_{};
    ExactComponentFabricStats stats_{};

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
            case OperationCode::BitFlipTrajectory:
            case OperationCode::PhaseFlipTrajectory:
            case OperationCode::DepolarizingTrajectory:
            case OperationCode::AmplitudeDampingTrajectory:
                return false;
            default:
                throw QStateError("Component fabric received an unknown opcode");
        }
    }

    void validate_support(const Operation& operation) const {
        if (operation.first >= qubit_count_) {
            throw QStateError("Component fabric operation target is out of range");
        }
        if (two_qubit(operation.code)) {
            if (operation.second >= qubit_count_) {
                throw QStateError("Component fabric two-qubit target is out of range");
            }
            if (operation.first == operation.second) {
                throw QStateError("Component fabric two-qubit operation requires distinct qubits");
            }
        }
    }

    static void validate_bit(std::uint8_t bit) {
        if (bit > 1U) {
            throw QStateError("Component fabric query bit must be zero or one");
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

}  // namespace qubit
