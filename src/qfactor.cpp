#include "qubit/qfactor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <utility>

namespace qubit {
namespace {

[[nodiscard]] bool finite_value(const QComplex& value) noexcept {
    return std::isfinite(value.re) && std::isfinite(value.im);
}

[[nodiscard]] std::size_t saturated_product(
    std::span<const FactorVariableId> variables,
    std::span<const std::size_t> dimensions) noexcept {
    std::size_t entries = 1U;
    for (const FactorVariableId variable : variables) {
        if (static_cast<std::size_t>(variable) >= dimensions.size()) {
            return std::numeric_limits<std::size_t>::max();
        }
        const std::size_t dimension = dimensions[variable];
        if (dimension == 0U || entries > std::numeric_limits<std::size_t>::max() / dimension) {
            return std::numeric_limits<std::size_t>::max();
        }
        entries *= dimension;
    }
    return entries;
}

[[nodiscard]] std::vector<std::size_t> scope_strides(
    std::span<const FactorVariableId> variables,
    std::span<const std::size_t> dimensions) {
    std::vector<std::size_t> strides(variables.size(), 1U);
    std::size_t stride = 1U;
    for (std::size_t position = 0U; position < variables.size(); ++position) {
        strides[position] = stride;
        const std::size_t dimension = dimensions[variables[position]];
        if (stride > std::numeric_limits<std::size_t>::max() / dimension) {
            throw QStateError("Exact factor stride overflowed");
        }
        stride *= dimension;
    }
    return strides;
}

[[nodiscard]] std::vector<FactorVariableId> bucket_union(
    std::span<const std::size_t> bucket,
    const std::vector<std::vector<FactorVariableId>>& scopes) {
    std::vector<FactorVariableId> variables;
    for (const std::size_t node : bucket) {
        variables.insert(variables.end(), scopes[node].begin(), scopes[node].end());
    }
    std::sort(variables.begin(), variables.end());
    variables.erase(std::unique(variables.begin(), variables.end()), variables.end());
    return variables;
}

struct PlannerCandidate {
    std::size_t entries{0U};
    FactorVariableId variable{0U};
    std::uint64_t revision{0U};
};

struct PlannerCandidateGreater {
    [[nodiscard]] bool operator()(
        const PlannerCandidate& first,
        const PlannerCandidate& second) const noexcept {
        if (first.entries != second.entries) {
            return first.entries > second.entries;
        }
        return first.variable > second.variable;
    }
};

[[nodiscard]] bool same_config(
    const ExactFactorConfig& first,
    const ExactFactorConfig& second) noexcept {
    return first.max_factor_entries == second.max_factor_entries &&
           first.max_factors == second.max_factors &&
           first.max_variables == second.max_variables &&
           first.max_compiled_index_entries == second.max_compiled_index_entries &&
           first.reuse_workspace_slots == second.reuse_workspace_slots;
}

}  // namespace

ExactFactorGraph::ExactFactorGraph(ExactFactorConfig config)
    : config_(config) {
    if (config_.max_factor_entries == 0U ||
        config_.max_factors == 0U ||
        config_.max_variables == 0U) {
        throw QStateError("Exact factor graph resource limits must be nonzero");
    }
}

FactorVariableId ExactFactorGraph::add_variable(std::size_t dimension_value) {
    if (dimension_value == 0U) {
        throw QStateError("Exact factor variable dimension must be nonzero");
    }
    if (dimension_value > config_.max_factor_entries) {
        throw QStateError("Exact factor variable dimension exceeds max_factor_entries");
    }
    if (dimensions_.size() >= config_.max_variables ||
        dimensions_.size() >= std::numeric_limits<FactorVariableId>::max()) {
        throw QStateError("Exact factor graph exceeded its variable limit");
    }
    const FactorVariableId id = static_cast<FactorVariableId>(dimensions_.size());
    dimensions_.push_back(dimension_value);
    return id;
}

std::size_t ExactFactorGraph::factor_entries(
    std::span<const FactorVariableId> variables) const {
    const std::size_t entries = saturated_product(variables, dimensions_);
    if (entries == std::numeric_limits<std::size_t>::max()) {
        throw QStateError("Exact factor table size overflowed");
    }
    if (entries > config_.max_factor_entries) {
        throw QStateError("Exact factor table exceeds max_factor_entries");
    }
    return entries;
}

void ExactFactorGraph::validate_factor_scope(
    std::span<const FactorVariableId> variables) const {
    for (std::size_t index = 0U; index < variables.size(); ++index) {
        if (static_cast<std::size_t>(variables[index]) >= dimensions_.size()) {
            throw QStateError("Exact factor scope variable is out of range");
        }
        for (std::size_t previous = 0U; previous < index; ++previous) {
            if (variables[previous] == variables[index]) {
                throw QStateError("Exact factor scope contains duplicate variables");
            }
        }
    }
}

FactorId ExactFactorGraph::add_dense_factor(
    std::span<const FactorVariableId> variables,
    std::span<const QComplex> values) {
    validate_factor_scope(variables);
    if (factors_.size() >= config_.max_factors ||
        factors_.size() >= std::numeric_limits<FactorId>::max()) {
        throw QStateError("Exact factor graph exceeded its factor limit");
    }
    const std::size_t expected = factor_entries(variables);
    if (values.size() != expected) {
        throw QStateError("Exact dense factor table size does not match its scope dimensions");
    }
    for (const QComplex& value : values) {
        if (!finite_value(value)) {
            throw QStateError("Exact factor values must be finite");
        }
    }

    Factor factor_value;
    factor_value.variables.assign(variables.begin(), variables.end());
    factor_value.logical_entries = expected;
    factor_value.storage = FactorStorageMode::Dense;
    factor_value.dense.assign(values.begin(), values.end());
    const FactorId id = static_cast<FactorId>(factors_.size());
    factors_.push_back(std::move(factor_value));
    return id;
}

FactorId ExactFactorGraph::add_sparse_factor(
    std::span<const FactorVariableId> variables,
    std::span<const FactorSparseEntry> entries) {
    validate_factor_scope(variables);
    if (factors_.size() >= config_.max_factors ||
        factors_.size() >= std::numeric_limits<FactorId>::max()) {
        throw QStateError("Exact factor graph exceeded its factor limit");
    }
    const std::size_t logical_entries = factor_entries(variables);
    std::vector<FactorSparseEntry> sorted(entries.begin(), entries.end());
    std::sort(sorted.begin(), sorted.end(), [](const auto& first, const auto& second) {
        return first.index < second.index;
    });
    for (std::size_t index = 0U; index < sorted.size(); ++index) {
        if (sorted[index].index >= logical_entries) {
            throw QStateError("Exact sparse factor index is out of range");
        }
        if (!finite_value(sorted[index].value)) {
            throw QStateError("Exact factor values must be finite");
        }
        if (index != 0U && sorted[index - 1U].index == sorted[index].index) {
            throw QStateError("Exact sparse factor contains duplicate indices");
        }
    }

    Factor factor_value;
    factor_value.variables.assign(variables.begin(), variables.end());
    factor_value.logical_entries = logical_entries;
    factor_value.storage = FactorStorageMode::Sparse;
    factor_value.sparse = std::move(sorted);
    const FactorId id = static_cast<FactorId>(factors_.size());
    factors_.push_back(std::move(factor_value));
    return id;
}

ExactFactorGraph::Factor& ExactFactorGraph::mutable_factor(FactorId id) {
    if (static_cast<std::size_t>(id) >= factors_.size()) {
        throw QStateError("Exact factor id is out of range");
    }
    return factors_[id];
}

const ExactFactorGraph::Factor& ExactFactorGraph::factor(FactorId id) const {
    if (static_cast<std::size_t>(id) >= factors_.size()) {
        throw QStateError("Exact factor id is out of range");
    }
    return factors_[id];
}

void ExactFactorGraph::set_dense_factor(
    FactorId id,
    std::span<const QComplex> values) {
    Factor& target = mutable_factor(id);
    if (values.size() != target.logical_entries) {
        throw QStateError("Exact dense factor rebind size changed");
    }
    std::vector<QComplex> replacement(values.begin(), values.end());
    for (const QComplex& value : replacement) {
        if (!finite_value(value)) {
            throw QStateError("Exact factor values must be finite");
        }
    }
    target.storage = FactorStorageMode::Dense;
    target.dense = std::move(replacement);
    target.sparse.clear();
}

void ExactFactorGraph::set_sparse_factor(
    FactorId id,
    std::span<const FactorSparseEntry> entries) {
    Factor& target = mutable_factor(id);
    std::vector<FactorSparseEntry> replacement(entries.begin(), entries.end());
    std::sort(replacement.begin(), replacement.end(), [](const auto& first, const auto& second) {
        return first.index < second.index;
    });
    for (std::size_t index = 0U; index < replacement.size(); ++index) {
        if (replacement[index].index >= target.logical_entries) {
            throw QStateError("Exact sparse factor rebind index is out of range");
        }
        if (!finite_value(replacement[index].value)) {
            throw QStateError("Exact factor values must be finite");
        }
        if (index != 0U && replacement[index - 1U].index == replacement[index].index) {
            throw QStateError("Exact sparse factor rebind contains duplicate indices");
        }
    }
    target.storage = FactorStorageMode::Sparse;
    target.sparse = std::move(replacement);
    target.dense.clear();
}

std::size_t ExactFactorGraph::dimension(FactorVariableId variable) const {
    if (static_cast<std::size_t>(variable) >= dimensions_.size()) {
        throw QStateError("Exact factor variable id is out of range");
    }
    return dimensions_[variable];
}

FactorStorageMode ExactFactorGraph::factor_storage(FactorId id) const {
    return factor(id).storage;
}

std::size_t ExactFactorGraph::estimated_bytes() const noexcept {
    std::size_t bytes = sizeof(*this) + dimensions_.capacity() * sizeof(std::size_t) +
                        factors_.capacity() * sizeof(Factor);
    for (const Factor& factor_value : factors_) {
        bytes += factor_value.variables.capacity() * sizeof(FactorVariableId) +
                 factor_value.dense.capacity() * sizeof(QComplex) +
                 factor_value.sparse.capacity() * sizeof(FactorSparseEntry);
    }
    return bytes;
}

bool ExactFactorGraph::validate(std::string* reason) const noexcept {
    try {
        if (config_.max_factor_entries == 0U ||
            config_.max_factors == 0U ||
            config_.max_variables == 0U) {
            throw QStateError("Exact factor graph resource limits must be nonzero");
        }
        if (dimensions_.size() > config_.max_variables || factors_.size() > config_.max_factors) {
            throw QStateError("Exact factor graph exceeds configured topology limits");
        }
        for (const std::size_t dimension_value : dimensions_) {
            if (dimension_value == 0U || dimension_value > config_.max_factor_entries) {
                throw QStateError("Exact factor graph contains an invalid variable dimension");
            }
        }
        for (const Factor& factor_value : factors_) {
            validate_factor_scope(factor_value.variables);
            const std::size_t expected = factor_entries(factor_value.variables);
            if (factor_value.logical_entries != expected) {
                throw QStateError("Exact factor logical table size changed");
            }
            if (factor_value.storage == FactorStorageMode::Dense) {
                if (factor_value.dense.size() != expected || !factor_value.sparse.empty()) {
                    throw QStateError("Exact dense factor storage is inconsistent");
                }
                for (const QComplex& value : factor_value.dense) {
                    if (!finite_value(value)) {
                        throw QStateError("Exact factor values must be finite");
                    }
                }
            } else {
                if (!factor_value.dense.empty()) {
                    throw QStateError("Exact sparse factor storage is inconsistent");
                }
                for (std::size_t index = 0U; index < factor_value.sparse.size(); ++index) {
                    const FactorSparseEntry& entry = factor_value.sparse[index];
                    if (entry.index >= expected || !finite_value(entry.value)) {
                        throw QStateError("Exact sparse factor storage is invalid");
                    }
                    if (index != 0U && factor_value.sparse[index - 1U].index >= entry.index) {
                        throw QStateError("Exact sparse factor indices are not unique and sorted");
                    }
                }
            }
        }
        return true;
    } catch (const std::exception& error) {
        if (reason != nullptr) {
            *reason = error.what();
        }
        return false;
    }
}

ExactFactorPlan ExactFactorGraph::compile(
    std::span<const FactorVariableId> retained_variables) const {
    return ExactFactorPlan(*this, retained_variables);
}

QComplex ExactFactorGraph::partition() const {
    return compile().partition();
}

std::vector<QComplex> ExactFactorGraph::marginal(
    std::span<const FactorVariableId> retained_variables) const {
    return compile(retained_variables).evaluate();
}

std::vector<QComplex> ExactFactorGraph::normalized_marginal(
    std::span<const FactorVariableId> retained_variables) const {
    return compile(retained_variables).normalized_marginal();
}

const char* exact_factor_route_name(ExactFactorRoute route) noexcept {
    switch (route) {
        case ExactFactorRoute::VariableElimination:
            return "ExactFactorVariableElimination";
    }
    return "unknown";
}

std::size_t ExactFactorWorkspace::estimated_bytes() const noexcept {
    std::size_t bytes = sizeof(*this) + outputs_.capacity() * sizeof(std::vector<QComplex>) +
                        coordinates_.capacity() * sizeof(std::size_t) +
                        retained_coordinates_.capacity() * sizeof(std::size_t);
    for (const auto& output : outputs_) {
        bytes += output.capacity() * sizeof(QComplex);
    }
    return bytes;
}

ExactFactorPlan::ExactFactorPlan(
    const ExactFactorGraph& graph,
    std::span<const FactorVariableId> retained_variables)
    : config_(graph.config_),
      dimensions_(graph.dimensions_),
      retained_variables_(retained_variables.begin(), retained_variables.end()),
      graph_factor_count_(graph.factors_.size()) {
    std::string reason;
    if (!graph.validate(&reason)) {
        throw QStateError("Cannot compile invalid exact factor graph: " + reason);
    }

    std::vector<std::uint8_t> retained_mask(dimensions_.size(), 0U);
    for (const FactorVariableId variable : retained_variables_) {
        if (static_cast<std::size_t>(variable) >= dimensions_.size()) {
            throw QStateError("Exact factor retained variable is out of range");
        }
        if (retained_mask[variable] != 0U) {
            throw QStateError("Exact factor retained variables contain duplicates");
        }
        retained_mask[variable] = 1U;
    }

    factor_topology_.reserve(graph.factors_.size());
    factor_logical_entries_.reserve(graph.factors_.size());
    sources_.reserve(graph.factors_.size() + dimensions_.size());
    std::vector<std::uint8_t> variable_seen(dimensions_.size(), 0U);
    for (const ExactFactorGraph::Factor& factor_value : graph.factors_) {
        SourceFactor source;
        source.variables = factor_value.variables;
        source.strides = scope_strides(source.variables, dimensions_);
        source.logical_entries = factor_value.logical_entries;
        source.storage = factor_value.storage;
        source.dense = factor_value.dense;
        source.sparse = factor_value.sparse;
        sources_.push_back(std::move(source));
        factor_topology_.push_back(factor_value.variables);
        factor_logical_entries_.push_back(factor_value.logical_entries);
        for (const FactorVariableId variable : factor_value.variables) {
            variable_seen[variable] = 1U;
        }
        if (factor_value.storage == FactorStorageMode::Dense) {
            ++stats_.source_dense_factors;
        } else {
            ++stats_.source_sparse_factors;
        }
        stats_.peak_union_variables =
            std::max(stats_.peak_union_variables, factor_value.variables.size());
        stats_.peak_factor_entries =
            std::max(stats_.peak_factor_entries, factor_value.logical_entries);
    }

    for (std::size_t variable = 0U; variable < dimensions_.size(); ++variable) {
        if (variable_seen[variable] != 0U) {
            continue;
        }
        SourceFactor unit;
        unit.variables.push_back(static_cast<FactorVariableId>(variable));
        unit.strides.push_back(1U);
        unit.logical_entries = dimensions_[variable];
        unit.storage = FactorStorageMode::Dense;
        unit.synthetic = true;
        stats_.peak_union_variables = std::max<std::size_t>(stats_.peak_union_variables, 1U);
        stats_.peak_factor_entries =
            std::max(stats_.peak_factor_entries, unit.logical_entries);
        sources_.push_back(std::move(unit));
    }
    if (sources_.size() > config_.max_factors) {
        throw QStateError("Exact factor plan exceeds max_factors after isolated-variable units");
    }

    std::vector<std::vector<FactorVariableId>> scopes;
    scopes.reserve(sources_.size() + dimensions_.size());
    for (const SourceFactor& source : sources_) {
        scopes.push_back(source.variables);
    }
    std::vector<bool> active(scopes.size(), true);
    std::vector<std::vector<std::size_t>> incidence(dimensions_.size());
    for (std::size_t node = 0U; node < scopes.size(); ++node) {
        for (const FactorVariableId variable : scopes[node]) {
            incidence[variable].push_back(node);
        }
    }

    std::vector<std::uint64_t> revisions(dimensions_.size(), 0U);
    std::priority_queue<
        PlannerCandidate,
        std::vector<PlannerCandidate>,
        PlannerCandidateGreater> candidates;

    const auto enqueue = [&](FactorVariableId variable) {
        if (retained_mask[variable] != 0U || incidence[variable].empty()) {
            return;
        }
        const std::vector<FactorVariableId> variables = bucket_union(incidence[variable], scopes);
        candidates.push({
            saturated_product(variables, dimensions_),
            variable,
            revisions[variable],
        });
    };
    for (std::size_t variable = 0U; variable < dimensions_.size(); ++variable) {
        enqueue(static_cast<FactorVariableId>(variable));
    }

    stats_.variable_count = dimensions_.size();
    stats_.source_factors = graph.factors_.size();
    stats_.retained_variables = retained_variables_.size();
    std::size_t compiled_index_entries = 0U;

    while (true) {
        while (!candidates.empty()) {
            const PlannerCandidate& candidate = candidates.top();
            if (candidate.revision == revisions[candidate.variable] &&
                !incidence[candidate.variable].empty()) {
                break;
            }
            candidates.pop();
        }
        if (candidates.empty()) {
            break;
        }

        const PlannerCandidate candidate = candidates.top();
        candidates.pop();
        const FactorVariableId selected = candidate.variable;
        const std::vector<std::size_t> selected_bucket = incidence[selected];
        const std::vector<FactorVariableId> selected_union = bucket_union(selected_bucket, scopes);
        const std::size_t selected_entries = saturated_product(selected_union, dimensions_);
        if (selected_entries == std::numeric_limits<std::size_t>::max() ||
            selected_entries > config_.max_factor_entries) {
            throw QStateError("Exact factor elimination exceeded max_factor_entries");
        }

        const auto selected_it =
            std::lower_bound(selected_union.begin(), selected_union.end(), selected);
        if (selected_it == selected_union.end() || *selected_it != selected) {
            throw QStateError("Exact factor planner lost its selected variable");
        }

        Step step;
        step.eliminated = selected;
        step.union_variables = selected_union;
        step.selected_position =
            static_cast<std::size_t>(selected_it - selected_union.begin());
        step.selected_dimension = dimensions_[selected];
        step.output_variables = selected_union;
        step.output_variables.erase(
            step.output_variables.begin() + static_cast<std::ptrdiff_t>(step.selected_position));
        step.output_entries = saturated_product(step.output_variables, dimensions_);
        if (step.output_entries == std::numeric_limits<std::size_t>::max() ||
            step.output_entries > config_.max_factor_entries) {
            throw QStateError("Exact factor reduced table exceeded max_factor_entries");
        }
        if (step.output_entries >
            std::numeric_limits<std::size_t>::max() / step.selected_dimension ||
            step.output_entries * step.selected_dimension != selected_entries) {
            throw QStateError("Exact factor elimination assignment count is inconsistent");
        }

        step.inputs.reserve(selected_bucket.size());
        for (const std::size_t node : selected_bucket) {
            InputMap input;
            input.node = node;
            input.positions.reserve(scopes[node].size());
            for (const FactorVariableId variable : scopes[node]) {
                const auto position =
                    std::lower_bound(selected_union.begin(), selected_union.end(), variable);
                if (position == selected_union.end() || *position != variable) {
                    throw QStateError("Exact factor input variable is missing from its union");
                }
                input.positions.push_back(
                    static_cast<std::size_t>(position - selected_union.begin()));
            }
            input.strides = scope_strides(scopes[node], dimensions_);
            step.inputs.push_back(std::move(input));
            active[node] = false;
        }

        const std::size_t assignment_count = step.output_entries * step.selected_dimension;
        std::size_t mapping_entries = 0U;
        if (!step.inputs.empty() &&
            assignment_count <=
                std::numeric_limits<std::size_t>::max() / step.inputs.size()) {
            mapping_entries = assignment_count * step.inputs.size();
        }
        if (mapping_entries != 0U &&
            compiled_index_entries <= config_.max_compiled_index_entries &&
            mapping_entries <= config_.max_compiled_index_entries - compiled_index_entries) {
            step.compiled_input_indices.resize(mapping_entries);
            std::vector<std::size_t> coordinates(step.union_variables.size(), 0U);
            for (std::size_t output_index = 0U;
                 output_index < step.output_entries;
                 ++output_index) {
                std::size_t remaining = output_index;
                for (std::size_t position = 0U;
                     position < step.union_variables.size();
                     ++position) {
                    if (position == step.selected_position) {
                        continue;
                    }
                    const std::size_t dimension_value =
                        dimensions_[step.union_variables[position]];
                    coordinates[position] = remaining % dimension_value;
                    remaining /= dimension_value;
                }
                for (std::size_t selected_value = 0U;
                     selected_value < step.selected_dimension;
                     ++selected_value) {
                    coordinates[step.selected_position] = selected_value;
                    const std::size_t base =
                        (output_index * step.selected_dimension + selected_value) *
                        step.inputs.size();
                    for (std::size_t input_index = 0U;
                         input_index < step.inputs.size();
                         ++input_index) {
                        const InputMap& input = step.inputs[input_index];
                        std::size_t local_index = 0U;
                        for (std::size_t position = 0U;
                             position < input.positions.size();
                             ++position) {
                            local_index += coordinates[input.positions[position]] *
                                           input.strides[position];
                        }
                        step.compiled_input_indices[base + input_index] = local_index;
                    }
                }
            }
            compiled_index_entries += mapping_entries;
        }

        stats_.peak_union_variables =
            std::max(stats_.peak_union_variables, step.union_variables.size());
        stats_.peak_factor_entries =
            std::max(stats_.peak_factor_entries, selected_entries);
        ++stats_.eliminated_variables;

        const std::size_t output_node = scopes.size();
        for (const FactorVariableId variable : selected_union) {
            std::vector<std::size_t>& nodes = incidence[variable];
            std::erase_if(nodes, [&](std::size_t node) {
                return std::binary_search(selected_bucket.begin(), selected_bucket.end(), node);
            });
            ++revisions[variable];
        }

        scopes.push_back(step.output_variables);
        active.push_back(true);
        for (const FactorVariableId variable : step.output_variables) {
            incidence[variable].push_back(output_node);
        }
        for (const FactorVariableId variable : selected_union) {
            enqueue(variable);
        }
        steps_.push_back(std::move(step));
    }
    stats_.compiled_index_entries = compiled_index_entries;

    for (std::size_t node = 0U; node < active.size(); ++node) {
        if (!active[node]) {
            continue;
        }
        for (const FactorVariableId variable : scopes[node]) {
            if (retained_mask[variable] == 0U) {
                throw QStateError("Exact factor planner left an eliminable variable active");
            }
        }

        TerminalMap terminal;
        terminal.node = node;
        terminal.strides = scope_strides(scopes[node], dimensions_);
        terminal.retained_positions.reserve(scopes[node].size());
        for (const FactorVariableId variable : scopes[node]) {
            const auto position = std::find(
                retained_variables_.begin(), retained_variables_.end(), variable);
            if (position == retained_variables_.end()) {
                throw QStateError("Exact factor terminal variable is not retained");
            }
            terminal.retained_positions.push_back(
                static_cast<std::size_t>(position - retained_variables_.begin()));
        }
        terminals_.push_back(std::move(terminal));
    }

    stats_.output_entries = saturated_product(retained_variables_, dimensions_);
    if (stats_.output_entries == std::numeric_limits<std::size_t>::max() ||
        stats_.output_entries > config_.max_factor_entries) {
        throw QStateError("Exact factor marginal output exceeds max_factor_entries");
    }
    stats_.peak_factor_entries =
        std::max(stats_.peak_factor_entries, stats_.output_entries);

    std::vector<std::size_t> last_use(steps_.size());
    for (std::size_t step = 0U; step < steps_.size(); ++step) {
        last_use[step] = step;
    }
    for (std::size_t consumer = 0U; consumer < steps_.size(); ++consumer) {
        for (const InputMap& input : steps_[consumer].inputs) {
            if (input.node < sources_.size()) {
                continue;
            }
            const std::size_t producer = input.node - sources_.size();
            if (producer >= consumer || producer >= steps_.size()) {
                throw QStateError("Exact factor workspace dependency is not topological");
            }
            last_use[producer] = std::max(last_use[producer], consumer);
        }
    }
    for (const TerminalMap& terminal : terminals_) {
        if (terminal.node < sources_.size()) {
            continue;
        }
        const std::size_t producer = terminal.node - sources_.size();
        if (producer >= steps_.size()) {
            throw QStateError("Exact factor terminal references an invalid dynamic node");
        }
        last_use[producer] = steps_.size();
    }

    if (!config_.reuse_workspace_slots) {
        workspace_slot_sizes_.resize(steps_.size());
        for (std::size_t step = 0U; step < steps_.size(); ++step) {
            steps_[step].workspace_slot = step;
            workspace_slot_sizes_[step] = steps_[step].output_entries;
        }
    } else {
        std::vector<std::vector<std::size_t>> release_before(steps_.size() + 1U);
        for (std::size_t producer = 0U; producer < steps_.size(); ++producer) {
            if (last_use[producer] < steps_.size()) {
                release_before[last_use[producer] + 1U].push_back(producer);
            }
        }
        std::vector<std::size_t> free_slots;
        for (std::size_t step = 0U; step < steps_.size(); ++step) {
            for (const std::size_t producer : release_before[step]) {
                free_slots.push_back(steps_[producer].workspace_slot);
            }

            std::size_t slot = 0U;
            if (free_slots.empty()) {
                slot = workspace_slot_sizes_.size();
                workspace_slot_sizes_.push_back(0U);
            } else {
                const auto smallest = std::min_element(free_slots.begin(), free_slots.end());
                slot = *smallest;
                free_slots.erase(smallest);
            }
            steps_[step].workspace_slot = slot;
            workspace_slot_sizes_[slot] =
                std::max(workspace_slot_sizes_[slot], steps_[step].output_entries);
        }
    }
    stats_.workspace_slots = workspace_slot_sizes_.size();
}

QComplex ExactFactorPlan::source_value(
    const SourceFactor& source,
    std::size_t index) const {
    if (index >= source.logical_entries) {
        throw QStateError("Exact factor source index is out of range");
    }
    if (source.synthetic) {
        return {1.0, 0.0};
    }
    if (source.storage == FactorStorageMode::Dense) {
        return source.dense[index];
    }
    const auto position = std::lower_bound(
        source.sparse.begin(), source.sparse.end(), index,
        [](const FactorSparseEntry& entry, std::size_t target) {
            return entry.index < target;
        });
    if (position == source.sparse.end() || position->index != index) {
        return {};
    }
    return position->value;
}

QComplex ExactFactorPlan::node_value(
    std::size_t node,
    std::size_t index,
    const ExactFactorWorkspace& workspace_value) const {
    if (node < sources_.size()) {
        return source_value(sources_[node], index);
    }
    const std::size_t producer = node - sources_.size();
    if (producer >= steps_.size()) {
        throw QStateError("Exact factor plan references an invalid dynamic node");
    }
    const Step& step = steps_[producer];
    if (step.workspace_slot >= workspace_value.outputs_.size() ||
        index >= step.output_entries ||
        index >= workspace_value.outputs_[step.workspace_slot].size()) {
        throw QStateError("Exact factor plan references an invalid workspace node");
    }
    return workspace_value.outputs_[step.workspace_slot][index];
}

ExactFactorWorkspace ExactFactorPlan::workspace() const {
    ExactFactorWorkspace result;
    result.outputs_.resize(workspace_slot_sizes_.size());
    for (std::size_t slot = 0U; slot < workspace_slot_sizes_.size(); ++slot) {
        result.outputs_[slot].resize(workspace_slot_sizes_[slot]);
    }
    result.coordinates_.resize(stats_.peak_union_variables);
    result.retained_coordinates_.resize(retained_variables_.size());
    return result;
}

void ExactFactorPlan::validate_workspace(
    const ExactFactorWorkspace& workspace_value) const {
    if (workspace_value.outputs_.size() != workspace_slot_sizes_.size() ||
        workspace_value.coordinates_.size() != stats_.peak_union_variables ||
        workspace_value.retained_coordinates_.size() != retained_variables_.size()) {
        throw QStateError("Exact factor workspace does not match its plan");
    }
    for (std::size_t slot = 0U; slot < workspace_slot_sizes_.size(); ++slot) {
        if (workspace_value.outputs_[slot].size() != workspace_slot_sizes_[slot]) {
            throw QStateError("Exact factor workspace output shape does not match its plan");
        }
    }
}

void ExactFactorPlan::evaluate(
    std::span<QComplex> output,
    ExactFactorWorkspace& workspace_value) const {
    if (output.size() != stats_.output_entries) {
        throw QStateError("Exact factor output size does not match its plan");
    }
    validate_workspace(workspace_value);

    for (std::size_t step_index = 0U; step_index < steps_.size(); ++step_index) {
        const Step& step = steps_[step_index];
        std::vector<QComplex>& step_output = workspace_value.outputs_[step.workspace_slot];
        if (!step.compiled_input_indices.empty()) {
            const std::size_t input_count = step.inputs.size();
            for (std::size_t output_index = 0U;
                 output_index < step.output_entries;
                 ++output_index) {
                QComplex sum{};
                const std::size_t output_base =
                    output_index * step.selected_dimension * input_count;
                for (std::size_t selected_value = 0U;
                     selected_value < step.selected_dimension;
                     ++selected_value) {
                    QComplex product{1.0, 0.0};
                    const std::size_t input_base =
                        output_base + selected_value * input_count;
                    for (std::size_t input_index = 0U;
                         input_index < input_count;
                         ++input_index) {
                        product *= node_value(
                            step.inputs[input_index].node,
                            step.compiled_input_indices[input_base + input_index],
                            workspace_value);
                    }
                    sum += product;
                }
                step_output[output_index] = sum;
            }
            continue;
        }

        for (std::size_t output_index = 0U; output_index < step.output_entries; ++output_index) {
            std::size_t remaining = output_index;
            for (std::size_t position = 0U; position < step.union_variables.size(); ++position) {
                if (position == step.selected_position) {
                    continue;
                }
                const std::size_t dimension_value = dimensions_[step.union_variables[position]];
                workspace_value.coordinates_[position] = remaining % dimension_value;
                remaining /= dimension_value;
            }

            QComplex sum{};
            for (std::size_t selected_value = 0U;
                 selected_value < step.selected_dimension;
                 ++selected_value) {
                workspace_value.coordinates_[step.selected_position] = selected_value;
                QComplex product{1.0, 0.0};
                for (const InputMap& input : step.inputs) {
                    std::size_t local_index = 0U;
                    for (std::size_t position = 0U; position < input.positions.size(); ++position) {
                        local_index += workspace_value.coordinates_[input.positions[position]] *
                                       input.strides[position];
                    }
                    product *= node_value(input.node, local_index, workspace_value);
                }
                sum += product;
            }
            step_output[output_index] = sum;
        }
    }

    for (std::size_t output_index = 0U; output_index < output.size(); ++output_index) {
        std::size_t remaining = output_index;
        for (std::size_t position = 0U; position < retained_variables_.size(); ++position) {
            const std::size_t dimension_value = dimensions_[retained_variables_[position]];
            workspace_value.retained_coordinates_[position] = remaining % dimension_value;
            remaining /= dimension_value;
        }

        QComplex product{1.0, 0.0};
        for (const TerminalMap& terminal : terminals_) {
            std::size_t local_index = 0U;
            for (std::size_t position = 0U;
                 position < terminal.retained_positions.size();
                 ++position) {
                local_index +=
                    workspace_value.retained_coordinates_[terminal.retained_positions[position]] *
                    terminal.strides[position];
            }
            product *= node_value(terminal.node, local_index, workspace_value);
        }
        output[output_index] = product;
    }
}

std::vector<QComplex> ExactFactorPlan::evaluate() const {
    ExactFactorWorkspace local = workspace();
    return evaluate(local);
}

std::vector<QComplex> ExactFactorPlan::evaluate(
    ExactFactorWorkspace& workspace_value) const {
    std::vector<QComplex> result(stats_.output_entries);
    evaluate(result, workspace_value);
    return result;
}

QComplex ExactFactorPlan::partition() const {
    ExactFactorWorkspace local = workspace();
    return partition(local);
}

QComplex ExactFactorPlan::partition(ExactFactorWorkspace& workspace_value) const {
    const std::vector<QComplex> values = evaluate(workspace_value);
    QComplex result{};
    for (const QComplex& value : values) {
        result += value;
    }
    return result;
}

std::vector<QComplex> ExactFactorPlan::normalized_marginal() const {
    ExactFactorWorkspace local = workspace();
    return normalized_marginal(local);
}

std::vector<QComplex> ExactFactorPlan::normalized_marginal(
    ExactFactorWorkspace& workspace_value) const {
    std::vector<QComplex> values = evaluate(workspace_value);
    QComplex normalization{};
    for (const QComplex& value : values) {
        normalization += value;
    }
    if (normalization.norm2() <= std::numeric_limits<double>::min()) {
        throw QStateError("Cannot normalize an exact factor marginal with zero partition");
    }
    for (QComplex& value : values) {
        value /= normalization;
    }
    return values;
}

void ExactFactorPlan::validate_topology(const ExactFactorGraph& graph) const {
    std::string reason;
    if (!graph.validate(&reason)) {
        throw QStateError("Cannot rebind invalid exact factor graph: " + reason);
    }
    if (!same_config(graph.config_, config_) ||
        graph.dimensions_ != dimensions_ ||
        graph.factors_.size() != graph_factor_count_) {
        throw QStateError("Exact factor rebind topology does not match its plan");
    }
    for (std::size_t index = 0U; index < graph_factor_count_; ++index) {
        if (graph.factors_[index].variables != factor_topology_[index] ||
            graph.factors_[index].logical_entries != factor_logical_entries_[index]) {
            throw QStateError("Exact factor rebind factor topology changed");
        }
    }
}

void ExactFactorPlan::rebind(const ExactFactorGraph& graph) {
    validate_topology(graph);

    struct Replacement {
        FactorStorageMode storage{FactorStorageMode::Dense};
        std::vector<QComplex> dense{};
        std::vector<FactorSparseEntry> sparse{};
    };
    std::vector<Replacement> replacements;
    replacements.reserve(graph_factor_count_);
    std::size_t dense_count = 0U;
    std::size_t sparse_count = 0U;
    for (std::size_t index = 0U; index < graph_factor_count_; ++index) {
        const ExactFactorGraph::Factor& factor_value = graph.factors_[index];
        Replacement replacement;
        replacement.storage = factor_value.storage;
        replacement.dense = factor_value.dense;
        replacement.sparse = factor_value.sparse;
        replacements.push_back(std::move(replacement));
        if (factor_value.storage == FactorStorageMode::Dense) {
            ++dense_count;
        } else {
            ++sparse_count;
        }
    }

    for (std::size_t index = 0U; index < graph_factor_count_; ++index) {
        sources_[index].storage = replacements[index].storage;
        sources_[index].dense = std::move(replacements[index].dense);
        sources_[index].sparse = std::move(replacements[index].sparse);
    }
    stats_.source_dense_factors = dense_count;
    stats_.source_sparse_factors = sparse_count;
    ++rebind_count_;
}

std::size_t ExactFactorPlan::estimated_bytes() const noexcept {
    std::size_t bytes = sizeof(*this) +
                        dimensions_.capacity() * sizeof(std::size_t) +
                        retained_variables_.capacity() * sizeof(FactorVariableId) +
                        factor_topology_.capacity() * sizeof(std::vector<FactorVariableId>) +
                        factor_logical_entries_.capacity() * sizeof(std::size_t) +
                        sources_.capacity() * sizeof(SourceFactor) +
                        steps_.capacity() * sizeof(Step) +
                        terminals_.capacity() * sizeof(TerminalMap) +
                        workspace_slot_sizes_.capacity() * sizeof(std::size_t);
    for (const auto& variables : factor_topology_) {
        bytes += variables.capacity() * sizeof(FactorVariableId);
    }
    for (const SourceFactor& source : sources_) {
        bytes += source.variables.capacity() * sizeof(FactorVariableId) +
                 source.strides.capacity() * sizeof(std::size_t) +
                 source.dense.capacity() * sizeof(QComplex) +
                 source.sparse.capacity() * sizeof(FactorSparseEntry);
    }
    for (const Step& step : steps_) {
        bytes += step.union_variables.capacity() * sizeof(FactorVariableId) +
                 step.output_variables.capacity() * sizeof(FactorVariableId) +
                 step.inputs.capacity() * sizeof(InputMap) +
                 step.compiled_input_indices.capacity() * sizeof(std::size_t);
        for (const InputMap& input : step.inputs) {
            bytes += input.positions.capacity() * sizeof(std::size_t) +
                     input.strides.capacity() * sizeof(std::size_t);
        }
    }
    for (const TerminalMap& terminal : terminals_) {
        bytes += terminal.retained_positions.capacity() * sizeof(std::size_t) +
                 terminal.strides.capacity() * sizeof(std::size_t);
    }
    return bytes;
}

}  // namespace qubit
