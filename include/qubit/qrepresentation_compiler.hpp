#pragma once

#include "qubit/qcompiler.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace qubit {

enum class ExactRepresentationDisposition : std::uint8_t {
    LocalDenseEligible = 0,
    BrokerRequired = 1,
};

struct ExactRepresentationCompilerConfig {
    ExactMarginalCompilerConfig execution{};
    std::size_t max_query_qubits{1U << 20U};
    std::size_t max_components{1U << 20U};
    std::size_t max_local_dense_qubits{20U};
    std::size_t max_certificate_bytes{std::size_t{1U} << 30U};
};

struct ExactRepresentationComponentCertificate {
    std::vector<QubitId> global_qubits{};
    std::size_t operations{0U};
    std::size_t query_qubits{0U};
    ExactRepresentationDisposition disposition{ExactRepresentationDisposition::BrokerRequired};
    std::size_t local_dense_state_scalars{0U};
};

struct ExactRepresentationCompilerStats {
    std::size_t original_qubits{0U};
    std::size_t original_operations{0U};
    std::size_t causal_qubits{0U};
    std::size_t causal_operations{0U};
    std::size_t components{0U};
    std::size_t largest_component_qubits{0U};
    std::size_t local_dense_eligible_components{0U};
    std::size_t broker_required_components{0U};
    std::size_t local_dense_state_scalar_envelope{0U};
    std::size_t estimated_bytes{0U};
};

class ExactRepresentationCompilerPlan {
public:
    ExactRepresentationCompilerPlan(
        std::size_t qubit_count,
        std::span<const Operation> operations,
        std::span<const QubitId> query_qubits,
        ExactRepresentationCompilerConfig config = {})
        : qubit_count_(qubit_count),
          config_(config),
          source_operations_(operations.begin(), operations.end()),
          query_qubits_(query_qubits.begin(), query_qubits.end()),
          index_(qubit_count_, source_operations_, causal_config(config_)),
          slice_(index_.slice(query_qubits_)) {
        if (config_.execution.max_causal_operations == 0U ||
            config_.execution.fabric.max_components == 0U ||
            config_.max_components == 0U ||
            config_.max_certificate_bytes == 0U ||
            config_.max_local_dense_qubits >= std::numeric_limits<std::size_t>::digits) {
            throw QStateError("Representation compiler configuration is invalid");
        }
        if (slice_.operations.size() > config_.execution.max_causal_operations) {
            throw QStateError("Representation compiler causal operation count exceeds configured cap");
        }
        build_components();
        build_stats();
        if (stats_.estimated_bytes > config_.max_certificate_bytes) {
            throw QStateError("Representation compiler certificate exceeds configured byte cap");
        }
    }

    [[nodiscard]] const ExactRepresentationCompilerStats& stats() const noexcept {
        return stats_;
    }
    [[nodiscard]] const std::vector<ExactRepresentationComponentCertificate>&
    component_certificates() const noexcept {
        return components_;
    }
    [[nodiscard]] const std::vector<QubitId>& causal_global_qubits() const noexcept {
        return slice_.global_qubits;
    }
    [[nodiscard]] const std::vector<Operation>& causal_operations() const noexcept {
        return slice_.operations;
    }

    [[nodiscard]] bool matches_source(
        std::size_t qubit_count,
        std::span<const Operation> operations,
        std::span<const QubitId> query_qubits) const noexcept {
        if (qubit_count != qubit_count_ || operations.size() != source_operations_.size() ||
            query_qubits.size() != query_qubits_.size()) {
            return false;
        }
        for (std::size_t index = 0U; index < query_qubits.size(); ++index) {
            if (query_qubits[index] != query_qubits_[index]) {
                return false;
            }
        }
        for (std::size_t index = 0U; index < operations.size(); ++index) {
            const Operation& left = operations[index];
            const Operation& right = source_operations_[index];
            if (left.code != right.code || left.first != right.first || left.second != right.second ||
                left.parameter != right.parameter || left.sample != right.sample) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] ExactIndexedMarginalCompilerPlan prepare() const {
        ExactIndexedMarginalCompilerPlan prepared(index_, query_qubits_, config_.execution);
        validate_prepared(prepared);
        return prepared;
    }

    [[nodiscard]] ExactIndexedMarginalCompilerPlan prepare_verified(
        std::size_t qubit_count,
        std::span<const Operation> operations,
        std::span<const QubitId> query_qubits) const {
        if (!matches_source(qubit_count, operations, query_qubits)) {
            throw QStateError("Representation compiler source binding does not match certificate");
        }
        return prepare();
    }

private:
    std::size_t qubit_count_{0U};
    ExactRepresentationCompilerConfig config_{};
    std::vector<Operation> source_operations_{};
    std::vector<QubitId> query_qubits_{};
    ExactCausalOperationIndex index_;
    ExactCausalSlice slice_{};
    std::vector<ExactRepresentationComponentCertificate> components_{};
    ExactRepresentationCompilerStats stats_{};

    [[nodiscard]] static ExactCausalIndexConfig causal_config(
        const ExactRepresentationCompilerConfig& config) noexcept {
        ExactCausalIndexConfig result;
        result.max_qubits = config.execution.fabric.max_qubits;
        result.max_operations = config.execution.fabric.max_operations;
        result.max_query_qubits = config.max_query_qubits;
        return result;
    }

    void build_components() {
        const std::size_t qubits = slice_.global_qubits.size();
        std::vector<std::size_t> parent(qubits);
        for (std::size_t index = 0U; index < qubits; ++index) {
            parent[index] = index;
        }
        const auto root = [&](std::size_t value) {
            std::size_t current = value;
            while (parent[current] != current) {
                current = parent[current];
            }
            return current;
        };
        const auto unite = [&](std::size_t left, std::size_t right) {
            const std::size_t first = root(left);
            const std::size_t second = root(right);
            if (first != second) {
                parent[second] = first;
            }
        };

        for (const Operation& operation : slice_.operations) {
            const std::size_t first = static_cast<std::size_t>(operation.first);
            if (first >= qubits) {
                throw QStateError("Representation compiler causal target is out of range");
            }
            if (is_pair(operation.code)) {
                const std::size_t second = static_cast<std::size_t>(operation.second);
                if (second >= qubits || first == second) {
                    throw QStateError("Representation compiler causal dependency is invalid");
                }
                unite(first, second);
            }
        }

        const std::size_t npos = std::numeric_limits<std::size_t>::max();
        const std::size_t cap = std::min(config_.max_components, config_.execution.fabric.max_components);
        std::vector<std::size_t> by_root(qubits, npos);
        std::vector<std::size_t> by_qubit(qubits, npos);
        for (std::size_t local = 0U; local < qubits; ++local) {
            const std::size_t component_root = root(local);
            std::size_t component = by_root[component_root];
            if (component == npos) {
                if (components_.size() >= cap) {
                    throw QStateError("Representation compiler exceeds configured component cap");
                }
                component = components_.size();
                by_root[component_root] = component;
                components_.push_back({});
            }
            by_qubit[local] = component;
            components_[component].global_qubits.push_back(slice_.global_qubits[local]);
        }

        for (const QubitId query : slice_.local_query_qubits) {
            const std::size_t local = static_cast<std::size_t>(query);
            if (local >= qubits || by_qubit[local] == npos) {
                throw QStateError("Representation compiler lost a query qubit");
            }
            ++components_[by_qubit[local]].query_qubits;
        }
        for (const Operation& operation : slice_.operations) {
            const std::size_t first = static_cast<std::size_t>(operation.first);
            const std::size_t component = by_qubit[first];
            if (is_pair(operation.code) &&
                by_qubit[static_cast<std::size_t>(operation.second)] != component) {
                throw QStateError("Representation compiler dependency crosses components");
            }
            ++components_[component].operations;
        }

        for (ExactRepresentationComponentCertificate& component : components_) {
            if (component.query_qubits == 0U) {
                throw QStateError("Representation compiler retained a component without a query root");
            }
            const std::size_t width = component.global_qubits.size();
            if (width <= config_.max_local_dense_qubits) {
                component.disposition = ExactRepresentationDisposition::LocalDenseEligible;
                component.local_dense_state_scalars = std::size_t{1U} << width;
            }
        }
    }

    void build_stats() {
        std::size_t largest = 0U;
        std::size_t local_dense = 0U;
        std::size_t broker = 0U;
        std::size_t dense_scalars = 0U;
        std::size_t bytes = index_.stats().estimated_bytes;
        bytes = add_bytes(bytes, source_operations_.capacity(), sizeof(Operation));
        bytes = add_bytes(bytes, query_qubits_.capacity(), sizeof(QubitId));
        bytes = add_bytes(bytes, slice_.global_qubits.capacity(), sizeof(QubitId));
        bytes = add_bytes(bytes, slice_.operations.capacity(), sizeof(Operation));
        bytes = add_bytes(bytes, components_.capacity(), sizeof(ExactRepresentationComponentCertificate));
        for (const ExactRepresentationComponentCertificate& component : components_) {
            largest = std::max(largest, component.global_qubits.size());
            bytes = add_bytes(bytes, component.global_qubits.capacity(), sizeof(QubitId));
            if (component.disposition == ExactRepresentationDisposition::LocalDenseEligible) {
                ++local_dense;
                dense_scalars = checked_sum(dense_scalars, component.local_dense_state_scalars);
            } else {
                ++broker;
            }
        }
        stats_ = ExactRepresentationCompilerStats{
            qubit_count_, source_operations_.size(), slice_.global_qubits.size(), slice_.operations.size(),
            components_.size(), largest, local_dense, broker, dense_scalars, bytes,
        };
    }

    void validate_prepared(const ExactIndexedMarginalCompilerPlan& prepared) const {
        const ExactMarginalCompilerStats& observed = prepared.stats();
        if (observed.original_qubits != stats_.original_qubits ||
            observed.original_operations != stats_.original_operations ||
            observed.causal_qubits != stats_.causal_qubits ||
            observed.causal_operations != stats_.causal_operations ||
            observed.components != stats_.components ||
            prepared.causal_global_qubits() != slice_.global_qubits) {
            throw QStateError("Representation compiler execution diverged from certificate");
        }
        const std::vector<ExactComponentReceipt>& receipts = prepared.component_receipts();
        if (receipts.size() != components_.size()) {
            throw QStateError("Representation compiler receipt count diverged from certificate");
        }
        for (std::size_t index = 0U; index < receipts.size(); ++index) {
            const ExactComponentReceipt& receipt = receipts[index];
            const ExactRepresentationComponentCertificate& component = components_[index];
            if (receipt.qubits != component.global_qubits.size() ||
                receipt.operations != component.operations ||
                !receipt.active || receipt.prepared != (component.operations != 0U)) {
                throw QStateError("Representation compiler component receipt diverged from certificate");
            }
        }
    }

    [[nodiscard]] static bool is_pair(OperationCode code) {
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
            default:
                throw QStateError("Representation compiler supports unitary circuit operations only");
        }
    }

    [[nodiscard]] static std::size_t add_bytes(
        std::size_t bytes, std::size_t count, std::size_t width) {
        if (count != 0U && width > std::numeric_limits<std::size_t>::max() / count) {
            throw QStateError("Representation compiler storage estimate overflowed");
        }
        const std::size_t added = count * width;
        if (added > std::numeric_limits<std::size_t>::max() - bytes) {
            throw QStateError("Representation compiler storage estimate overflowed");
        }
        return bytes + added;
    }

    [[nodiscard]] static std::size_t checked_sum(std::size_t left, std::size_t right) {
        if (right > std::numeric_limits<std::size_t>::max() - left) {
            throw QStateError("Representation compiler scalar envelope overflowed");
        }
        return left + right;
    }
};

}  // namespace qubit
