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
#include <utility>
#include <vector>

namespace qubit {

using ExactRepresentationDependencyType = std::uint64_t;

enum class ExactRepresentationExecutionKind : std::uint8_t {
    LocalDense = 0,
    Broker = 1,
};

enum class ExactRepresentationTransitionKind : std::uint8_t {
    None = 0,
    SameGenerationReplay = 1,
    GenerationReplay = 2,
};

struct ExactRepresentationFabricConfig {
    ExactCompiledMarginalProgramConfig program{};
    std::size_t max_active_components{1U << 20U};
    std::size_t max_component_qubits{1U << 20U};
    std::size_t max_component_operations{1U << 24U};
    std::size_t max_operations{1U << 26U};
    std::size_t max_dependency_width{1U << 20U};
    std::size_t max_declared_dependencies{1U << 20U};
    std::size_t max_dependency_support_entries{1U << 24U};
    std::size_t max_local_dense_qubits{18U};
    std::size_t max_local_dense_state_scalars{1U << 18U};
    std::size_t max_local_dense_scalar_sweep_units{1U << 28U};
    std::size_t max_local_dense_cached_plans{256U};
    std::size_t max_local_dense_cached_scalars{1U << 22U};
};

struct ExactRepresentationFabricStats {
    std::size_t qubits{0U};
    std::size_t operations{0U};
    std::size_t declared_dependencies{0U};
    std::size_t dependency_support_entries{0U};
    std::size_t active_components{0U};
    std::size_t component_merges{0U};
    std::size_t component_invalidations{0U};
    std::size_t program_builds{0U};
    std::size_t queries{0U};
    std::size_t cache_hits{0U};
    std::size_t cache_misses{0U};
    std::size_t local_dense_executions{0U};
    std::size_t broker_executions{0U};
    std::size_t representation_transitions{0U};
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
    std::size_t broker_cached_plans{0U};
    std::size_t local_dense_cached_plans{0U};
    std::size_t local_dense_cached_scalars{0U};
};

struct ExactRepresentationDependencyReceipt {
    std::size_t component{0U};
    std::size_t generation{0U};
    ExactRepresentationDependencyType type{0U};
    std::vector<QubitId> support{};
};

struct ExactRepresentationTransitionReceipt {
    ExactRepresentationTransitionKind kind{ExactRepresentationTransitionKind::None};
    ExactRepresentationExecutionKind from{ExactRepresentationExecutionKind::LocalDense};
    ExactRepresentationExecutionKind to{ExactRepresentationExecutionKind::LocalDense};
    std::size_t from_generation{0U};
    std::size_t to_generation{0U};
    bool exact_replay{false};
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
    bool local_dense_eligible{false};
    std::size_t local_dense_state_scalars{0U};
    std::size_t local_dense_payload_bytes{0U};
    std::size_t local_dense_scalar_sweep_units{0U};
    ExactRepresentationExecutionKind execution{ExactRepresentationExecutionKind::Broker};
    ExactRepresentationTransitionReceipt transition{};
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
    std::size_t local_dense_islands{0U};
    std::size_t broker_islands{0U};
    std::size_t representation_transitions{0U};
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
            config_.max_dependency_width < 2U || config_.max_declared_dependencies == 0U ||
            config_.max_dependency_support_entries < config_.max_dependency_width ||
            config_.max_local_dense_qubits >= std::numeric_limits<std::size_t>::digits ||
            config_.max_local_dense_state_scalars == 0U ||
            config_.max_local_dense_scalar_sweep_units == 0U ||
            config_.max_local_dense_cached_plans == 0U ||
            config_.max_local_dense_cached_scalars == 0U ||
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
            attach(normalize(support, 2U), &operation, false, 0U);
        } else {
            const std::array<QubitId, 1> support{operation.first};
            attach(normalize(support, 1U), &operation, false, 0U);
        }
        ++stats_.operations;
    }

    void declare_dependency(std::span<const QubitId> support) {
        declare_dependency(0U, support);
    }

    void declare_dependency(
        ExactRepresentationDependencyType type,
        std::span<const QubitId> support) {
        const std::vector<QubitId> normalized = normalize(support, 2U);
        if (stats_.declared_dependencies >= config_.max_declared_dependencies ||
            stats_.dependency_support_entries > config_.max_dependency_support_entries ||
            normalized.size() >
                config_.max_dependency_support_entries - stats_.dependency_support_entries) {
            throw QStateError("Representation fabric dependency resource cap exceeded");
        }
        attach(normalized, nullptr, true, type);
        ++stats_.declared_dependencies;
        stats_.dependency_support_entries = checked_sum(
            stats_.dependency_support_entries, normalized.size());
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
            Component& query_component = components_[component];
            ensure_qubit_order(query_component);
            terms.push_back(Term{
                component,
                local_id(query_component.qubits, global),
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
            const ExactCausalSlice slice = program.causal_index().slice(local_qubits);
            DenseCertificate dense = dense_certificate(component, slice, local_qubits);
            DenseCacheEntry* dense_cache = find_dense_cache(component, local_qubits);
            if (dense_cache != nullptr) {
                dense.eligible = true;
                dense.state_scalars = dense_cache->state_scalars;
                dense.payload_bytes = checked_product(
                    dense.state_scalars,
                    sizeof(QComplex),
                    "Representation fabric local dense payload overflowed");
            }

            const ExactRepresentationExecutionKind execution = dense.eligible
                ? ExactRepresentationExecutionKind::LocalDense
                : ExactRepresentationExecutionKind::Broker;

            ExactRepresentationIslandExecutionReceipt island;
            island.component = component_id;
            island.generation = component.generation;
            island.component_qubits = component.qubits.size();
            island.component_operations = component.operations.size();
            island.query_qubits = local_qubits.size();
            island.causal_qubits = slice.global_qubits.size();
            island.causal_operations = slice.operations.size();
            island.local_dense_eligible = dense.eligible;
            island.local_dense_state_scalars = dense.state_scalars;
            island.local_dense_payload_bytes = dense.payload_bytes;
            island.local_dense_scalar_sweep_units = dense.scalar_sweep_units;
            island.execution = execution;
            island.transition = transition(component, execution);
            if (island.transition.kind != ExactRepresentationTransitionKind::None) {
                ++result.receipt.representation_transitions;
                ++stats_.representation_transitions;
            }

            double value = 0.0;
            if (execution == ExactRepresentationExecutionKind::LocalDense) {
                const bool hit = dense_cache != nullptr;
                if (dense_cache == nullptr) {
                    dense_cache = &prepare_dense(component, local_qubits, slice, dense);
                }
                island.cache_hits = hit ? 1U : 0U;
                island.cache_misses = hit ? 0U : 1U;
                island.executor_components = 1U;
                island.estimated_executor_bytes = dense_cache->state->estimated_bytes();
                ++island.route_counts[static_cast<std::size_t>(ExactExecutionRoute::Register)];
                value = dense_cache->state->marginal_probability(
                    dense_cache->local_query_qubits,
                    local_bits);
                ++result.receipt.local_dense_islands;
                ++stats_.local_dense_executions;
            } else {
                const ExactCompiledMarginalProgramStats before = program.stats();
                const ExactIndexedMarginalCompilerPlan& plan = program.prepare(local_qubits);
                const ExactCompiledMarginalProgramStats after = program.stats();
                island.cache_hits = after.cache_hits - before.cache_hits;
                island.cache_misses = after.cache_misses - before.cache_misses;
                island.causal_qubits = plan.stats().causal_qubits;
                island.causal_operations = plan.stats().causal_operations;
                island.executor_components = plan.stats().components;
                island.estimated_executor_bytes = plan.stats().estimated_bytes;
                for (const ExactComponentReceipt& receipt : plan.component_receipts()) {
                    if (!receipt.prepared) {
                        continue;
                    }
                    const std::size_t route = static_cast<std::size_t>(receipt.route);
                    if (route >= island.route_counts.size()) {
                        throw QStateError("Representation fabric received an unknown execution route");
                    }
                    ++island.route_counts[route];
                }
                value = plan.probability(local_bits);
                ++result.receipt.broker_islands;
                ++stats_.broker_executions;
            }

            result.receipt.cache_hits = checked_sum(
                result.receipt.cache_hits, island.cache_hits);
            result.receipt.cache_misses = checked_sum(
                result.receipt.cache_misses, island.cache_misses);
            stats_.cache_hits = checked_sum(stats_.cache_hits, island.cache_hits);
            stats_.cache_misses = checked_sum(stats_.cache_misses, island.cache_misses);
            ++result.receipt.fabric_components;
            result.receipt.executor_components = checked_sum(
                result.receipt.executor_components, island.executor_components);
            result.receipt.causal_qubits = checked_sum(
                result.receipt.causal_qubits, island.causal_qubits);
            result.receipt.causal_operations = checked_sum(
                result.receipt.causal_operations, island.causal_operations);
            result.receipt.estimated_executor_bytes = checked_sum(
                result.receipt.estimated_executor_bytes, island.estimated_executor_bytes);
            for (std::size_t route = 0U; route < island.route_counts.size(); ++route) {
                result.receipt.route_counts[route] = checked_sum(
                    result.receipt.route_counts[route], island.route_counts[route]);
            }
            result.islands.push_back(std::move(island));
            result.value *= value;
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
                component.dense_cache.size(),
                component.dense_cached_scalars,
            });
        }
        return result;
    }

    [[nodiscard]] std::vector<ExactRepresentationDependencyReceipt> dependency_receipts() const {
        std::vector<ExactRepresentationDependencyReceipt> result;
        result.reserve(stats_.declared_dependencies);
        for (std::size_t index = 0U; index < components_.size(); ++index) {
            const Component& component = components_[index];
            if (!component.active) {
                continue;
            }
            for (const DependencyRecord& dependency : component.dependency_records) {
                result.push_back(ExactRepresentationDependencyReceipt{
                    index,
                    component.generation,
                    dependency.type,
                    dependency.support,
                });
            }
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
    struct DenseCertificate {
        bool eligible{false};
        std::size_t state_scalars{0U};
        std::size_t payload_bytes{0U};
        std::size_t scalar_sweep_units{0U};
    };

    struct DenseCacheEntry {
        std::size_t generation{0U};
        std::vector<QubitId> query_qubits{};
        std::vector<QubitId> local_query_qubits{};
        std::size_t state_scalars{0U};
        std::unique_ptr<QRegister> state{};
    };

    struct DependencyRecord {
        ExactRepresentationDependencyType type{0U};
        std::vector<QubitId> support{};
    };

    struct Component {
        bool active{false};
        bool qubits_sorted{true};
        std::size_t generation{1U};
        std::size_t dependencies{0U};
        std::vector<QubitId> qubits{};
        std::vector<Operation> operations{};
        std::vector<DependencyRecord> dependency_records{};
        std::unique_ptr<ExactCompiledMarginalProgram> program{};
        std::vector<DenseCacheEntry> dense_cache{};
        std::size_t dense_cached_scalars{0U};
        bool has_last_execution{false};
        ExactRepresentationExecutionKind last_execution{ExactRepresentationExecutionKind::LocalDense};
        std::size_t last_execution_generation{0U};
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
        bool dependency,
        ExactRepresentationDependencyType dependency_type) {
        std::vector<std::size_t> ids;
        ids.reserve(support.size());
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
            if (dependency) {
                component.dependency_records.push_back(DependencyRecord{dependency_type, support});
            }
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
            additions.reserve(support.size());
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
            if (operation != nullptr || !additions.empty()) {
                invalidate(component);
            } else if (dependency) {
                increment_generation(component);
            }
            if (!additions.empty()) {
                component.qubits.insert(
                    component.qubits.end(), additions.begin(), additions.end());
                component.qubits_sorted = false;
            }
            if (operation != nullptr) {
                component.operations.push_back(*operation);
            }
            if (dependency) {
                component.dependencies = checked_sum(component.dependencies, 1U);
                component.dependency_records.push_back(DependencyRecord{dependency_type, support});
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
        std::vector<DependencyRecord> dependency_records;
        dependency_records.reserve(dependencies);
        for (const std::size_t id : ids) {
            operations.insert(
                operations.end(), components_[id].operations.begin(), components_[id].operations.end());
            dependency_records.insert(
                dependency_records.end(),
                components_[id].dependency_records.begin(),
                components_[id].dependency_records.end());
        }
        if (operation != nullptr) {
            operations.push_back(*operation);
        }
        if (dependency) {
            dependency_records.push_back(DependencyRecord{dependency_type, support});
        }
        if (generation == std::numeric_limits<std::size_t>::max()) {
            throw QStateError("Representation fabric generation counter overflowed");
        }

        const std::size_t target_id = ids.front();
        for (const std::size_t id : ids) {
            if (components_[id].program || !components_[id].dense_cache.empty()) {
                ++stats_.component_invalidations;
            }
        }
        Component& target = components_[target_id];
        target.active = true;
        target.qubits_sorted = true;
        target.generation = generation + 1U;
        target.dependencies = dependencies;
        target.qubits = std::move(combined);
        target.operations = std::move(operations);
        target.dependency_records = std::move(dependency_records);
        target.program.reset();
        target.dense_cache.clear();
        target.dense_cached_scalars = 0U;
        target.has_last_execution = false;
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
        if (component.program || !component.dense_cache.empty()) {
            ++stats_.component_invalidations;
        }
        component.program.reset();
        component.dense_cache.clear();
        component.dense_cached_scalars = 0U;
        increment_generation(component);
    }

    static void increment_generation(Component& component) {
        if (component.generation == std::numeric_limits<std::size_t>::max()) {
            throw QStateError("Representation fabric generation counter overflowed");
        }
        ++component.generation;
    }

    static void ensure_qubit_order(Component& component) {
        if (component.qubits_sorted) {
            return;
        }
        std::sort(component.qubits.begin(), component.qubits.end());
        component.qubits_sorted = true;
    }

    void ensure_program(Component& component) {
        if (component.program) {
            return;
        }
        ensure_qubit_order(component);
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

    [[nodiscard]] DenseCertificate dense_certificate(
        const Component& component,
        const ExactCausalSlice& slice,
        std::span<const QubitId> query_qubits) const {
        DenseCertificate result;
        const std::size_t width = slice.global_qubits.size();
        if (width == 0U || width > config_.max_local_dense_qubits ||
            width >= std::numeric_limits<std::size_t>::digits) {
            return result;
        }
        const std::size_t scalars = std::size_t{1U} << width;
        if (scalars > config_.max_local_dense_state_scalars ||
            component.dense_cache.size() >= config_.max_local_dense_cached_plans ||
            scalars > config_.max_local_dense_cached_scalars -
                std::min(component.dense_cached_scalars, config_.max_local_dense_cached_scalars)) {
            return result;
        }
        const std::size_t sweeps = checked_sum(
            checked_sum(slice.operations.size(), query_qubits.size()), 1U);
        if (scalars > config_.max_local_dense_scalar_sweep_units / sweeps) {
            return result;
        }
        result.eligible = true;
        result.state_scalars = scalars;
        result.payload_bytes = checked_product(
            scalars,
            sizeof(QComplex),
            "Representation fabric local dense payload overflowed");
        result.scalar_sweep_units = scalars * sweeps;
        return result;
    }

    [[nodiscard]] DenseCacheEntry* find_dense_cache(
        Component& component,
        std::span<const QubitId> query_qubits) noexcept {
        for (DenseCacheEntry& entry : component.dense_cache) {
            if (entry.generation != component.generation ||
                entry.query_qubits.size() != query_qubits.size()) {
                continue;
            }
            bool equal = true;
            for (std::size_t index = 0U; index < query_qubits.size(); ++index) {
                if (entry.query_qubits[index] != query_qubits[index]) {
                    equal = false;
                    break;
                }
            }
            if (equal) {
                return &entry;
            }
        }
        return nullptr;
    }

    DenseCacheEntry& prepare_dense(
        Component& component,
        std::span<const QubitId> query_qubits,
        const ExactCausalSlice& slice,
        const DenseCertificate& certificate) {
        if (!certificate.eligible || certificate.state_scalars == 0U) {
            throw QStateError("Representation fabric local dense route lacks a valid certificate");
        }
        auto state = std::make_unique<QRegister>(slice.global_qubits.size());
        OperationPlan plan(slice.operations);
        plan.execute(*state);
        if (component.dense_cache.size() >= config_.max_local_dense_cached_plans ||
            certificate.state_scalars > config_.max_local_dense_cached_scalars -
                component.dense_cached_scalars) {
            throw QStateError("Representation fabric local dense cache changed after certification");
        }
        DenseCacheEntry entry;
        entry.generation = component.generation;
        entry.query_qubits.assign(query_qubits.begin(), query_qubits.end());
        entry.local_query_qubits = slice.local_query_qubits;
        entry.state_scalars = certificate.state_scalars;
        entry.state = std::move(state);
        component.dense_cached_scalars = checked_sum(
            component.dense_cached_scalars, certificate.state_scalars);
        component.dense_cache.push_back(std::move(entry));
        return component.dense_cache.back();
    }

    ExactRepresentationTransitionReceipt transition(
        Component& component,
        ExactRepresentationExecutionKind execution) {
        ExactRepresentationTransitionReceipt result;
        result.to = execution;
        result.to_generation = component.generation;
        if (component.has_last_execution) {
            result.from = component.last_execution;
            result.from_generation = component.last_execution_generation;
            if (component.last_execution_generation != component.generation) {
                result.kind = ExactRepresentationTransitionKind::GenerationReplay;
                result.exact_replay = true;
            } else if (component.last_execution != execution) {
                result.kind = ExactRepresentationTransitionKind::SameGenerationReplay;
                result.exact_replay = true;
            }
        }
        component.has_last_execution = true;
        component.last_execution = execution;
        component.last_execution_generation = component.generation;
        return result;
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

    [[nodiscard]] static std::size_t checked_product(
        std::size_t left,
        std::size_t right,
        const char* message) {
        if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) {
            throw QStateError(message);
        }
        return left * right;
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
