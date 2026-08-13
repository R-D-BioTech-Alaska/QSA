#pragma once

#include "qubit/qprogram.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <span>
#include <vector>

namespace qubit {

struct ExactRepresentationFabricConfig {
    ExactCompiledMarginalProgramConfig program{};
    std::size_t max_active_components{1U << 20U};
    std::size_t max_component_qubits{1U << 20U};
    std::size_t max_component_operations{1U << 24U};
    std::size_t max_operations{1U << 26U};
    std::size_t max_dependency_width{1U << 20U};
};

struct ExactRepresentationFabricStats {
    std::size_t qubits{0U};
    std::size_t operations{0U};
    std::size_t declared_dependencies{0U};
    std::size_t active_components{0U};
    std::size_t component_merges{0U};
    std::size_t component_invalidations{0U};
    std::size_t program_builds{0U};
    std::size_t queries{0U};
    std::size_t cache_hits{0U};
    std::size_t cache_misses{0U};
    std::size_t largest_component_qubits{0U};
    std::size_t largest_component_operations{0U};
};

struct ExactRepresentationComponentReceipt {
    std::size_t component{0U};
    std::size_t generation{0U};
    std::size_t qubits{0U};
    std::size_t operations{0U};
    std::size_t declared_dependencies{0U};
    bool program_ready{false};
    std::size_t cached_plans{0U};
};

struct ExactRepresentationIslandExecutionReceipt {
    std::size_t component{0U};
    std::size_t generation{0U};
    std::size_t component_qubits{0U};
    std::size_t component_operations{0U};
    std::size_t query_qubits{0U};
    std::size_t causal_qubits{0U};
    std::size_t causal_operations{0U};
    std::size_t executor_components{0U};
    std::size_t estimated_executor_bytes{0U};
    std::size_t cache_hits{0U};
    std::size_t cache_misses{0U};
    std::array<std::size_t, 9> route_counts{};
};

struct ExactRepresentationQueryReceipt {
    std::size_t global_qubits{0U};
    std::size_t query_qubits{0U};
    std::size_t untouched_query_qubits{0U};
    std::size_t fabric_components{0U};
    std::size_t executor_components{0U};
    std::size_t causal_qubits{0U};
    std::size_t causal_operations{0U};
    std::size_t noncausal_qubits{0U};
    std::size_t estimated_executor_bytes{0U};
    std::size_t cache_hits{0U};
    std::size_t cache_misses{0U};
    std::array<std::size_t, 9> route_counts{};
};

struct ExactRepresentationProbabilityResult {
    double value{0.0};
    ExactRepresentationQueryReceipt receipt{};
    std::vector<ExactRepresentationIslandExecutionReceipt> islands{};
};

class ExactRepresentationFabric {
public:
    ExactRepresentationFabric(
        std::size_t qubit_count,
        ExactRepresentationFabricConfig config = {})
        : qubit_count_(qubit_count), config_(config), component_of_(qubit_count, npos()) {
        if (qubit_count_ == 0U ||
            qubit_count_ > static_cast<std::size_t>(std::numeric_limits<QubitId>::max()) ||
            config_.max_active_components == 0U || config_.max_component_qubits == 0U ||
            config_.max_component_operations == 0U || config_.max_operations == 0U ||
            config_.max_dependency_width < 2U ||
            config_.max_component_qubits > config_.program.index.max_qubits ||
            config_.max_component_qubits > config_.program.marginal.fabric.max_qubits ||
            config_.max_component_operations > config_.program.index.max_operations ||
            config_.max_component_operations > config_.program.marginal.fabric.max_operations) {
            throw QStateError("Representation fabric configuration is invalid");
        }
        stats_.qubits = qubit_count_;
    }

    void append(const Operation& operation) {
        validate_operation(operation);
        if (stats_.operations >= config_.max_operations) {
            throw QStateError("Representation fabric reached its operation cap");
        }
        if (pair(operation.code)) {
            const std::array<QubitId, 2> support{operation.first, operation.second};
            attach(normalize(support, 2U), &operation, false);
        } else {
            const std::array<QubitId, 1> support{operation.first};
            attach(normalize(support, 1U), &operation, false);
        }
        ++stats_.operations;
    }

    void declare_dependency(std::span<const QubitId> support) {
        attach(normalize(support, 2U), nullptr, true);
        ++stats_.declared_dependencies;
    }

    [[nodiscard]] ExactRepresentationProbabilityResult marginal_probability(
        std::span<const QubitId> qubits,
        std::span<const std::uint8_t> bits) {
        validate_query(qubits, bits);
        ExactRepresentationProbabilityResult result;
        result.value = 1.0;
        result.receipt.global_qubits = qubit_count_;
        result.receipt.query_qubits = qubits.size();
        ++stats_.queries;

        struct Term {
            std::size_t component;
            QubitId local;
            std::uint8_t bit;
        };
        std::vector<Term> terms;
        terms.reserve(qubits.size());
        for (std::size_t index = 0U; index < qubits.size(); ++index) {
            const QubitId global = qubits[index];
            const std::size_t component = component_of_[static_cast<std::size_t>(global)];
            if (component == npos()) {
                ++result.receipt.untouched_query_qubits;
                if (bits[index] != 0U) {
                    result.value = 0.0;
                    result.receipt.noncausal_qubits = qubit_count_;
                    return result;
                }
                continue;
            }
            terms.push_back(Term{
                component,
                local_id(components_[component].qubits, global),
                bits[index],
            });
        }
        std::sort(terms.begin(), terms.end(), [](const Term& left, const Term& right) {
            return left.component != right.component
                ? left.component < right.component
                : left.local < right.local;
        });

        std::size_t cursor = 0U;
        while (cursor < terms.size()) {
            const std::size_t component_id = terms[cursor].component;
            std::size_t end = cursor + 1U;
            while (end < terms.size() && terms[end].component == component_id) {
                ++end;
            }
            Component& component = components_[component_id];
            ensure_program(component);
            std::vector<QubitId> local_qubits;
            std::vector<std::uint8_t> local_bits;
            local_qubits.reserve(end - cursor);
            local_bits.reserve(end - cursor);
            for (std::size_t index = cursor; index < end; ++index) {
                local_qubits.push_back(terms[index].local);
                local_bits.push_back(terms[index].bit);
            }

            ExactCompiledMarginalProgram& program = *component.program;
            const ExactCompiledMarginalProgramStats before = program.stats();
            const ExactIndexedMarginalCompilerPlan& plan = program.prepare(local_qubits);
            const ExactCompiledMarginalProgramStats after = program.stats();
            const std::size_t hits = after.cache_hits - before.cache_hits;
            const std::size_t misses = after.cache_misses - before.cache_misses;
            ExactRepresentationIslandExecutionReceipt island;
            island.component = component_id;
            island.generation = component.generation;
            island.component_qubits = component.qubits.size();
            island.component_operations = component.operations.size();
            island.query_qubits = local_qubits.size();
            island.causal_qubits = plan.stats().causal_qubits;
            island.causal_operations = plan.stats().causal_operations;
            island.executor_components = plan.stats().components;
            island.estimated_executor_bytes = plan.stats().estimated_bytes;
            island.cache_hits = hits;
            island.cache_misses = misses;

            result.receipt.cache_hits += hits;
            result.receipt.cache_misses += misses;
            stats_.cache_hits += hits;
            stats_.cache_misses += misses;
            ++result.receipt.fabric_components;
            result.receipt.executor_components += plan.stats().components;
            result.receipt.causal_qubits = checked_sum(
                result.receipt.causal_qubits, plan.stats().causal_qubits);
            result.receipt.causal_operations = checked_sum(
                result.receipt.causal_operations, plan.stats().causal_operations);
            result.receipt.estimated_executor_bytes = checked_sum(
                result.receipt.estimated_executor_bytes, plan.stats().estimated_bytes);
            for (const ExactComponentReceipt& receipt : plan.component_receipts()) {
                if (!receipt.prepared) {
                    continue;
                }
                const std::size_t route = static_cast<std::size_t>(receipt.route);
                if (route >= result.receipt.route_counts.size()) {
                    throw QStateError("Representation fabric received an unknown execution route");
                }
                ++result.receipt.route_counts[route];
                ++island.route_counts[route];
            }
            result.islands.push_back(std::move(island));
            result.value *= plan.probability(local_bits);
            if (result.value == 0.0) {
                break;
            }
            cursor = end;
        }
        if (result.receipt.causal_qubits > qubit_count_) {
            throw QStateError("Representation fabric causal receipt exceeds global qubit count");
        }
        result.receipt.noncausal_qubits = qubit_count_ - result.receipt.causal_qubits;
        return result;
    }

    [[nodiscard]] std::vector<ExactRepresentationComponentReceipt> component_receipts() const {
        std::vector<ExactRepresentationComponentReceipt> result;
        result.reserve(stats_.active_components);
        for (std::size_t index = 0U; index < components_.size(); ++index) {
            const Component& component = components_[index];
            if (!component.active) {
                continue;
            }
            result.push_back(ExactRepresentationComponentReceipt{
                index,
                component.generation,
                component.qubits.size(),
                component.operations.size(),
                component.dependencies,
                component.program != nullptr,
                component.program ? component.program->stats().cached_plans : 0U,
            });
        }
        return result;
    }

    void reset() noexcept {
        components_.clear();
        std::fill(component_of_.begin(), component_of_.end(), npos());
        stats_ = ExactRepresentationFabricStats{};
        stats_.qubits = qubit_count_;
    }

    [[nodiscard]] const ExactRepresentationFabricStats& stats() const noexcept {
        return stats_;
    }

private:
    struct Component {
        bool active{false};
        std::size_t generation{1U};
        std::size_t dependencies{0U};
        std::vector<QubitId> qubits{};
        std::vector<Operation> operations{};
        std::unique_ptr<ExactCompiledMarginalProgram> program{};
    };

    std::size_t qubit_count_{0U};
    ExactRepresentationFabricConfig config_{};
    std::vector<std::size_t> component_of_{};
    std::vector<Component> components_{};
    ExactRepresentationFabricStats stats_{};

    [[nodiscard]] std::vector<QubitId> normalize(
        std::span<const QubitId> support,
        std::size_t minimum) const {
        if (support.size() < minimum || support.size() > config_.max_dependency_width) {
            throw QStateError("Representation fabric dependency width is outside configured bounds");
        }
        std::vector<QubitId> result(support.begin(), support.end());
        std::sort(result.begin(), result.end());
        for (std::size_t index = 0U; index < result.size(); ++index) {
            if (static_cast<std::size_t>(result[index]) >= qubit_count_) {
                throw QStateError("Representation fabric dependency qubit is out of range");
            }
            if (index != 0U && result[index] == result[index - 1U]) {
                throw QStateError("Representation fabric dependency repeats a qubit");
            }
        }
        return result;
    }

    void validate_operation(const Operation& operation) const {
        const bool two = pair(operation.code);
        if (static_cast<std::size_t>(operation.first) >= qubit_count_ ||
            (two && static_cast<std::size_t>(operation.second) >= qubit_count_) ||
            (two && operation.first == operation.second)) {
            throw QStateError("Representation fabric operation support is invalid");
        }
    }

    void validate_query(
        std::span<const QubitId> qubits,
        std::span<const std::uint8_t> bits) const {
        if (qubits.size() != bits.size()) {
            throw QStateError("Representation fabric query qubit/bit sizes differ");
        }
        std::vector<QubitId> ordered(qubits.begin(), qubits.end());
        std::sort(ordered.begin(), ordered.end());
        for (std::size_t index = 0U; index < ordered.size(); ++index) {
            if (static_cast<std::size_t>(ordered[index]) >= qubit_count_ ||
                (index != 0U && ordered[index] == ordered[index - 1U])) {
                throw QStateError("Representation fabric query support is invalid");
            }
        }
        for (const std::uint8_t bit : bits) {
            if (bit > 1U) {
                throw QStateError("Representation fabric query bit must be zero or one");
            }
        }
    }

    void attach(
        const std::vector<QubitId>& support,
        const Operation* operation,
        bool dependency) {
        std::vector<std::size_t> ids;
        for (const QubitId qubit : support) {
            const std::size_t id = component_of_[static_cast<std::size_t>(qubit)];
            if (id != npos()) {
                ids.push_back(id);
            }
        }
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());

        if (ids.empty()) {
            if (stats_.active_components >= config_.max_active_components ||
                support.size() > config_.max_component_qubits) {
                throw QStateError("Representation fabric active-component cap exceeded");
            }
            Component component;
            component.active = true;
            component.qubits = support;
            component.dependencies = dependency ? 1U : 0U;
            if (operation != nullptr) {
                component.operations.push_back(*operation);
            }
            components_.push_back(std::move(component));
            const std::size_t id = components_.size() - 1U;
            for (const QubitId qubit : support) {
                component_of_[static_cast<std::size_t>(qubit)] = id;
            }
            ++stats_.active_components;
            update_maxima(components_[id]);
            return;
        }

        if (ids.size() == 1U) {
            Component& component = components_[ids.front()];
            std::vector<QubitId> additions;
            for (const QubitId qubit : support) {
                if (component_of_[static_cast<std::size_t>(qubit)] == npos()) {
                    additions.push_back(qubit);
                }
            }
            const std::size_t next_qubits = checked_sum(component.qubits.size(), additions.size());
            const std::size_t next_operations = checked_sum(
                component.operations.size(), operation != nullptr ? 1U : 0U);
            if (next_qubits > config_.max_component_qubits ||
                next_operations > config_.max_component_operations) {
                throw QStateError("Representation fabric component growth exceeds configured cap");
            }
            component.qubits.reserve(next_qubits);
            component.operations.reserve(next_operations);
            if (operation != nullptr || !additions.empty()) {
                invalidate(component);
            } else if (dependency) {
                increment_generation(component);
            }
            component.qubits.insert(component.qubits.end(), additions.begin(), additions.end());
            std::sort(component.qubits.begin(), component.qubits.end());
            if (operation != nullptr) {
                component.operations.push_back(*operation);
            }
            if (dependency) {
                ++component.dependencies;
            }
            for (const QubitId qubit : additions) {
                component_of_[static_cast<std::size_t>(qubit)] = ids.front();
            }
            update_maxima(component);
            return;
        }

        std::vector<QubitId> combined = support;
        std::vector<Operation> operations;
        std::size_t operation_count = operation != nullptr ? 1U : 0U;
        std::size_t dependencies = dependency ? 1U : 0U;
        std::size_t generation = 1U;
        for (const std::size_t id : ids) {
            combined.insert(combined.end(), components_[id].qubits.begin(), components_[id].qubits.end());
            operation_count = checked_sum(operation_count, components_[id].operations.size());
            dependencies = checked_sum(dependencies, components_[id].dependencies);
            generation = std::max(generation, components_[id].generation);
        }
        std::sort(combined.begin(), combined.end());
        combined.erase(std::unique(combined.begin(), combined.end()), combined.end());
        if (combined.size() > config_.max_component_qubits ||
            operation_count > config_.max_component_operations) {
            throw QStateError("Representation fabric merge exceeds configured cap");
        }
        operations.reserve(operation_count);
        for (const std::size_t id : ids) {
            operations.insert(
                operations.end(), components_[id].operations.begin(), components_[id].operations.end());
        }
        if (operation != nullptr) {
            operations.push_back(*operation);
        }
        if (generation == std::numeric_limits<std::size_t>::max()) {
            throw QStateError("Representation fabric generation counter overflowed");
        }

        const std::size_t target_id = ids.front();
        for (const std::size_t id : ids) {
            if (components_[id].program) {
                ++stats_.component_invalidations;
            }
        }
        Component& target = components_[target_id];
        target.active = true;
        target.generation = generation + 1U;
        target.dependencies = dependencies;
        target.qubits = std::move(combined);
        target.operations = std::move(operations);
        target.program.reset();
        for (std::size_t index = 1U; index < ids.size(); ++index) {
            components_[ids[index]] = Component{};
        }
        for (const QubitId qubit : target.qubits) {
            component_of_[static_cast<std::size_t>(qubit)] = target_id;
        }
        stats_.component_merges = checked_sum(stats_.component_merges, ids.size() - 1U);
        stats_.active_components -= ids.size() - 1U;
        update_maxima(target);
    }

    void invalidate(Component& component) {
        if (component.program) {
            component.program.reset();
            ++stats_.component_invalidations;
        }
        increment_generation(component);
    }

    static void increment_generation(Component& component) {
        if (component.generation == std::numeric_limits<std::size_t>::max()) {
            throw QStateError("Representation fabric generation counter overflowed");
        }
        ++component.generation;
    }

    void ensure_program(Component& component) {
        if (component.program) {
            return;
        }
        std::vector<Operation> local;
        local.reserve(component.operations.size());
        for (const Operation& source : component.operations) {
            Operation operation = source;
            operation.first = local_id(component.qubits, source.first);
            if (pair(source.code)) {
                operation.second = local_id(component.qubits, source.second);
            }
            local.push_back(operation);
        }
        component.program = std::make_unique<ExactCompiledMarginalProgram>(
            component.qubits.size(), local, config_.program);
        ++stats_.program_builds;
    }

    void update_maxima(const Component& component) noexcept {
        stats_.largest_component_qubits =
            std::max(stats_.largest_component_qubits, component.qubits.size());
        stats_.largest_component_operations =
            std::max(stats_.largest_component_operations, component.operations.size());
    }

    [[nodiscard]] static QubitId local_id(
        const std::vector<QubitId>& qubits,
        QubitId global) {
        const auto found = std::lower_bound(qubits.begin(), qubits.end(), global);
        if (found == qubits.end() || *found != global) {
            throw QStateError("Representation fabric failed to remap a component qubit");
        }
        return static_cast<QubitId>(std::distance(qubits.begin(), found));
    }

    [[nodiscard]] static bool pair(OperationCode code) {
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
                throw QStateError("Representation fabric supports unitary circuit operations only");
        }
    }

    [[nodiscard]] static std::size_t checked_sum(std::size_t left, std::size_t right) {
        if (right > std::numeric_limits<std::size_t>::max() - left) {
            throw QStateError("Representation fabric resource accounting overflowed");
        }
        return left + right;
    }

    [[nodiscard]] static constexpr std::size_t npos() noexcept {
        return std::numeric_limits<std::size_t>::max();
    }
};

}  // namespace qubit
