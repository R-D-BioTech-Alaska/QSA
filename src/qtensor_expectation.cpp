#include "qubit/qtensor.hpp"

#include <algorithm>
#include <limits>
#include <string>

namespace qubit {
namespace {

[[nodiscard]] std::size_t expectation_entries(std::size_t variables) noexcept {
    if (variables >= std::numeric_limits<std::size_t>::digits) {
        return std::numeric_limits<std::size_t>::max();
    }
    return std::size_t{1} << variables;
}

[[nodiscard]] std::size_t mapped_factor_index(
    std::size_t union_index,
    const std::vector<std::size_t>& positions) noexcept {
    switch (positions.size()) {
        case 0U:
            return 0U;
        case 1U:
            return (union_index >> positions[0]) & 1U;
        case 2U:
            return ((union_index >> positions[0]) & 1U) |
                   (((union_index >> positions[1]) & 1U) << 1U);
        case 3U:
            return ((union_index >> positions[0]) & 1U) |
                   (((union_index >> positions[1]) & 1U) << 1U) |
                   (((union_index >> positions[2]) & 1U) << 2U);
        case 4U:
            return ((union_index >> positions[0]) & 1U) |
                   (((union_index >> positions[1]) & 1U) << 1U) |
                   (((union_index >> positions[2]) & 1U) << 2U) |
                   (((union_index >> positions[3]) & 1U) << 3U);
        default:
            break;
    }
    std::size_t factor_index = 0U;
    for (std::size_t local = 0; local < positions.size(); ++local) {
        factor_index |= ((union_index >> positions[local]) & 1U) << local;
    }
    return factor_index;
}

}  // namespace

TensorExpectationPlan TensorNetworkCircuit::compile_expectation(
    const PauliObservable& observable) const {
    return TensorExpectationPlan(*this, observable);
}

TensorExpectationPlan TensorNetworkCircuit::compile_expectations(
    std::span<const PauliObservable> observables) const {
    return TensorExpectationPlan(*this, observables);
}

std::size_t TensorExpectationWorkspace::estimated_bytes() const noexcept {
    std::size_t bytes = sizeof(*this) +
                        outputs_.capacity() * sizeof(std::vector<QComplex>);
    for (const auto& output : outputs_) {
        bytes += output.capacity() * sizeof(QComplex);
    }
    return bytes;
}

void TensorExpectationPlan::set_operator(
    std::array<QComplex, 4>& values,
    PauliAxis axis) {
    switch (axis) {
        case PauliAxis::I:
            values = {{{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}}};
            return;
        case PauliAxis::X:
            values = {{{0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}}};
            return;
        case PauliAxis::Y:
            values = {{{0.0, 0.0}, {0.0, -1.0}, {0.0, 1.0}, {0.0, 0.0}}};
            return;
        case PauliAxis::Z:
            values = {{{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}}};
            return;
    }
    throw QStateError("Tensor expectation received an invalid Pauli axis");
}

TensorExpectationPlan::TensorExpectationPlan(
    const TensorNetworkCircuit& circuit,
    const PauliObservable& observable)
    : TensorExpectationPlan(
          circuit,
          std::span<const PauliObservable>(&observable, 1U)) {}

TensorExpectationPlan::TensorExpectationPlan(
    const TensorNetworkCircuit& circuit,
    std::span<const PauliObservable> observables)
    : qubit_count_(circuit.qubit_count_) {
    if (observables.empty()) {
        throw QStateError("Tensor expectation requires at least one observable");
    }
    std::string reason;
    if (!circuit.validate(&reason)) {
        throw QStateError("Cannot compile expectation from invalid tensor network: " + reason);
    }

    const std::uint64_t variable_count =
        static_cast<std::uint64_t>(circuit.next_variable_) * 2ULL;
    const std::uint64_t variable_limit =
        static_cast<std::uint64_t>(std::numeric_limits<VariableId>::max()) + 1ULL;
    if (variable_count > variable_limit) {
        throw QStateError("Tensor expectation variable range exhausted");
    }
    const VariableId bra_offset = circuit.next_variable_;
    const std::size_t total_variables = static_cast<std::size_t>(variable_count);

    stats_.source_operations = circuit.operation_count_;
    observables_.reserve(observables.size());

    auto compile_term = [&](const PauliTerm& observable_term) {
        if (observable_term.coefficient.norm2() == 0.0) {
            return;
        }

        TermPlan term;
        term.coefficient = observable_term.coefficient;

        std::vector<std::uint8_t> support(qubit_count_, 0U);
        std::vector<PauliFactor> factors;
        factors.reserve(observable_term.factors.size());
        for (const PauliFactor& factor : observable_term.factors) {
            if (factor.axis == PauliAxis::I) {
                continue;
            }
            const std::size_t qubit = static_cast<std::size_t>(factor.qubit);
            if (qubit >= qubit_count_) {
                throw QStateError("Tensor expectation Pauli factor is out of range");
            }
            if (support[qubit] != 0U) {
                throw QStateError("Tensor expectation Pauli term repeats a qubit");
            }
            support[qubit] = 1U;
            factors.push_back(factor);
        }

        if (factors.empty()) {
            term.identity = true;
            term.stats.source_operations = circuit.operation_count_;
            terms_.push_back(std::move(term));
            return;
        }

        std::vector<std::uint8_t> relevant(circuit.next_variable_, 0U);
        for (const PauliFactor& factor : factors) {
            relevant[circuit.current_wires_[factor.qubit]] = 1U;
        }

        std::vector<std::size_t> ket_factor_indices;
        ket_factor_indices.reserve(circuit.factors_.size());
        std::vector<VariableId> identity_terminals;
        for (std::size_t factor_index = circuit.factors_.size();
             factor_index-- > qubit_count_;) {
            const TensorNetworkCircuit::Factor& factor = circuit.factors_[factor_index];
            if (factor.variables.size() == 2U) {
                if (relevant[factor.variables[1]] == 0U) {
                    continue;
                }
                ket_factor_indices.push_back(factor_index);
                relevant[factor.variables[0]] = 1U;
                continue;
            }
            if (factor.variables.size() != 4U) {
                throw QStateError("Tensor expectation found an invalid circuit factor");
            }
            const bool first_relevant = relevant[factor.variables[2]] != 0U;
            const bool second_relevant = relevant[factor.variables[3]] != 0U;
            if (!first_relevant && !second_relevant) {
                continue;
            }
            ket_factor_indices.push_back(factor_index);
            if (!first_relevant) {
                identity_terminals.push_back(factor.variables[2]);
            }
            if (!second_relevant) {
                identity_terminals.push_back(factor.variables[3]);
            }
            relevant[factor.variables[0]] = 1U;
            relevant[factor.variables[1]] = 1U;
        }
        for (std::size_t qubit = 0; qubit < qubit_count_; ++qubit) {
            const TensorNetworkCircuit::Factor& initial = circuit.factors_[qubit];
            if (relevant[initial.variables.front()] != 0U) {
                ket_factor_indices.push_back(qubit);
            }
        }
        std::sort(ket_factor_indices.begin(), ket_factor_indices.end());

        if (factors.size() > circuit.config_.max_factors ||
            ket_factor_indices.size() >
                (circuit.config_.max_factors - factors.size()) / 2U) {
            throw QStateError("Tensor expectation term exceeded max_factors");
        }

        std::vector<VariableId> variable_map(total_variables);
        std::vector<std::uint8_t> variable_invert(total_variables, 0U);
        for (std::size_t variable = 0; variable < total_variables; ++variable) {
            variable_map[variable] = static_cast<VariableId>(variable);
        }
        for (const VariableId terminal : identity_terminals) {
            variable_map[static_cast<std::size_t>(terminal + bra_offset)] = terminal;
        }
        for (const PauliFactor& factor : factors) {
            const VariableId ket_wire = circuit.current_wires_[factor.qubit];
            const VariableId bra_wire = static_cast<VariableId>(ket_wire + bra_offset);
            variable_map[bra_wire] = ket_wire;
            if (factor.axis == PauliAxis::X || factor.axis == PauliAxis::Y) {
                variable_invert[bra_wire] = 1U;
            }
        }

        term.sources.reserve(ket_factor_indices.size() * 2U + factors.size());
        for (const std::size_t factor_index : ket_factor_indices) {
            const TensorNetworkCircuit::Factor& factor = circuit.factors_[factor_index];
            SourceFactor source;
            source.variables = factor.variables;
            source.values = factor.values;
            term.sources.push_back(std::move(source));
        }
        for (const std::size_t factor_index : ket_factor_indices) {
            const TensorNetworkCircuit::Factor& factor = circuit.factors_[factor_index];
            SourceFactor source;
            source.variables.reserve(factor.variables.size());
            for (const VariableId variable : factor.variables) {
                source.variables.push_back(static_cast<VariableId>(variable + bra_offset));
            }
            source.values.resize(factor.values.size());
            for (std::size_t source_index = 0; source_index < factor.values.size(); ++source_index) {
                std::size_t mapped_index = source_index;
                for (std::size_t local = 0; local < factor.variables.size(); ++local) {
                    const std::size_t bra_variable =
                        static_cast<std::size_t>(factor.variables[local] + bra_offset);
                    if (variable_invert[bra_variable] != 0U) {
                        mapped_index ^= std::size_t{1} << local;
                    }
                }
                source.values[mapped_index] = factor.values[source_index].conjugate();
            }
            term.sources.push_back(std::move(source));
        }
        for (const PauliFactor& factor : factors) {
            if (factor.axis != PauliAxis::Y && factor.axis != PauliAxis::Z) {
                continue;
            }
            SourceFactor phase;
            phase.variables.push_back(circuit.current_wires_[factor.qubit]);
            if (factor.axis == PauliAxis::Y) {
                phase.values = {{0.0, 1.0}, {0.0, -1.0}};
            } else {
                phase.values = {{1.0, 0.0}, {-1.0, 0.0}};
            }
            term.sources.push_back(std::move(phase));
        }

        std::vector<std::vector<VariableId>> scopes;
        scopes.reserve(term.sources.size() + total_variables);
        for (const SourceFactor& source : term.sources) {
            std::vector<VariableId> scope;
            scope.reserve(source.variables.size());
            for (const VariableId variable : source.variables) {
                scope.push_back(variable_map[variable]);
            }
            scopes.push_back(std::move(scope));
        }
        term.step_node_begin = scopes.size();

        std::vector<bool> active(scopes.size(), true);
        term.stats.source_operations = circuit.operation_count_;
        term.stats.source_factors = scopes.size();

        while (true) {
            std::vector<std::vector<std::size_t>> incidence(total_variables);
            bool has_variable = false;
            for (std::size_t node = 0; node < scopes.size(); ++node) {
                if (!active[node]) {
                    continue;
                }
                for (const VariableId variable : scopes[node]) {
                    incidence[variable].push_back(node);
                    has_variable = true;
                }
            }
            if (!has_variable) {
                break;
            }

            VariableId selected = 0U;
            std::vector<std::size_t> selected_bucket;
            std::vector<VariableId> selected_union;
            std::size_t selected_entries = std::numeric_limits<std::size_t>::max();
            bool found = false;

            for (std::size_t candidate = 0; candidate < incidence.size(); ++candidate) {
                if (incidence[candidate].empty()) {
                    continue;
                }
                std::vector<VariableId> variables;
                for (const std::size_t node : incidence[candidate]) {
                    variables.insert(
                        variables.end(), scopes[node].begin(), scopes[node].end());
                }
                std::sort(variables.begin(), variables.end());
                variables.erase(std::unique(variables.begin(), variables.end()), variables.end());
                const std::size_t entries = expectation_entries(variables.size());
                if (!found || entries < selected_entries ||
                    (entries == selected_entries && candidate < selected)) {
                    found = true;
                    selected = static_cast<VariableId>(candidate);
                    selected_bucket = incidence[candidate];
                    selected_union = std::move(variables);
                    selected_entries = entries;
                }
            }
            if (!found) {
                throw QStateError("Tensor expectation plan lost its active variables");
            }
            if (selected_entries > circuit.config_.max_contraction_entries) {
                throw QStateError("Tensor expectation exceeded max_contraction_entries");
            }

            const auto selected_it =
                std::lower_bound(selected_union.begin(), selected_union.end(), selected);
            if (selected_it == selected_union.end() || *selected_it != selected) {
                throw QStateError("Tensor expectation variable is missing from its bucket");
            }

            Step step;
            step.eliminated = selected;
            step.union_variables = selected_union;
            step.selected_position =
                static_cast<std::size_t>(selected_it - selected_union.begin());
            step.output_variables = selected_union;
            step.output_variables.erase(
                step.output_variables.begin() +
                static_cast<std::ptrdiff_t>(step.selected_position));
            step.output_entries = expectation_entries(step.output_variables.size());
            if (step.output_entries > circuit.config_.max_contraction_entries) {
                throw QStateError(
                    "Tensor expectation reduced factor exceeded max_contraction_entries");
            }

            step.inputs.reserve(selected_bucket.size());
            for (const std::size_t node : selected_bucket) {
                InputMap input;
                input.node = node;
                input.positions.reserve(scopes[node].size());
                for (const VariableId variable : scopes[node]) {
                    const auto position =
                        std::lower_bound(selected_union.begin(), selected_union.end(), variable);
                    if (position == selected_union.end() || *position != variable) {
                        throw QStateError(
                            "Tensor expectation input variable is missing from union");
                    }
                    input.positions.push_back(
                        static_cast<std::size_t>(position - selected_union.begin()));
                }
                step.inputs.push_back(std::move(input));
                active[node] = false;
            }

            term.stats.peak_union_variables =
                std::max(term.stats.peak_union_variables, step.union_variables.size());
            term.stats.peak_contraction_entries =
                std::max(term.stats.peak_contraction_entries, selected_entries);
            ++term.stats.eliminated_variables;

            scopes.push_back(step.output_variables);
            active.push_back(true);
            term.steps.push_back(std::move(step));
        }

        for (std::size_t node = 0; node < active.size(); ++node) {
            if (!active[node]) {
                continue;
            }
            if (!scopes[node].empty()) {
                throw QStateError("Tensor expectation did not eliminate every variable");
            }
            term.terminal_nodes.push_back(node);
        }
        if (term.terminal_nodes.empty()) {
            throw QStateError("Tensor expectation produced no terminal scalar");
        }

        std::vector<std::size_t> free_slots;
        std::vector<std::size_t> term_slot_sizes;
        for (std::size_t step_index = 0; step_index < term.steps.size(); ++step_index) {
            Step& step = term.steps[step_index];
            if (free_slots.empty()) {
                step.workspace_slot = term_slot_sizes.size();
                term_slot_sizes.push_back(step.output_entries);
            } else {
                step.workspace_slot = free_slots.back();
                free_slots.pop_back();
                term_slot_sizes[step.workspace_slot] =
                    std::max(term_slot_sizes[step.workspace_slot], step.output_entries);
            }
            for (const InputMap& input : step.inputs) {
                if (input.node < term.step_node_begin) {
                    continue;
                }
                const std::size_t previous = input.node - term.step_node_begin;
                if (previous >= step_index) {
                    throw QStateError("Tensor expectation step dependency is not topological");
                }
                free_slots.push_back(term.steps[previous].workspace_slot);
            }
        }
        if (slot_sizes_.size() < term_slot_sizes.size()) {
            slot_sizes_.resize(term_slot_sizes.size(), 0U);
        }
        for (std::size_t slot = 0; slot < term_slot_sizes.size(); ++slot) {
            slot_sizes_[slot] = std::max(slot_sizes_[slot], term_slot_sizes[slot]);
        }

        stats_.source_factors = std::max(stats_.source_factors, term.stats.source_factors);
        stats_.eliminated_variables =
            std::max(stats_.eliminated_variables, term.stats.eliminated_variables);
        stats_.peak_union_variables =
            std::max(stats_.peak_union_variables, term.stats.peak_union_variables);
        stats_.peak_contraction_entries =
            std::max(stats_.peak_contraction_entries, term.stats.peak_contraction_entries);
        terms_.push_back(std::move(term));
    };

    for (const PauliObservable& observable : observables) {
        if (observable.qubit_count() != qubit_count_) {
            throw QStateError("Tensor expectation observable width does not match circuit");
        }
        if (!observable.validate(&reason)) {
            throw QStateError("Cannot compile invalid tensor observable: " + reason);
        }
        ObservablePlan observable_plan;
        observable_plan.term_begin = terms_.size();
        for (const PauliTerm& term : observable.terms()) {
            compile_term(term);
        }
        observable_plan.term_end = terms_.size();
        observables_.push_back(observable_plan);
    }
}

std::size_t TensorExpectationPlan::step_count() const noexcept {
    std::size_t count = 0U;
    for (const TermPlan& term : terms_) {
        count += term.steps.size();
    }
    return count;
}

TensorExpectationWorkspace TensorExpectationPlan::workspace() const {
    TensorExpectationWorkspace result;
    result.outputs_.resize(slot_sizes_.size());
    for (std::size_t slot = 0; slot < slot_sizes_.size(); ++slot) {
        result.outputs_[slot].resize(slot_sizes_[slot]);
    }
    return result;
}

void TensorExpectationPlan::validate_workspace(
    const TensorExpectationWorkspace& workspace_value) const {
    if (workspace_value.outputs_.size() != slot_sizes_.size()) {
        throw QStateError("Tensor expectation workspace does not match its plan");
    }
    for (std::size_t slot = 0; slot < slot_sizes_.size(); ++slot) {
        if (workspace_value.outputs_[slot].size() < slot_sizes_[slot]) {
            throw QStateError(
                "Tensor expectation workspace output shape does not match its plan");
        }
    }
}

std::span<const QComplex> TensorExpectationPlan::node_values(
    const TermPlan& term,
    std::size_t node,
    const TensorExpectationWorkspace& workspace_value) const {
    if (node < term.sources.size()) {
        return term.sources[node].values;
    }
    if (node < term.step_node_begin) {
        return term.operators[node - term.sources.size()];
    }
    const std::size_t local_output = node - term.step_node_begin;
    if (local_output >= term.steps.size()) {
        throw QStateError("Tensor expectation references an invalid workspace node");
    }
    const Step& step = term.steps[local_output];
    const std::vector<QComplex>& storage = workspace_value.outputs_[step.workspace_slot];
    return std::span<const QComplex>(storage.data(), step.output_entries);
}

QComplex TensorExpectationPlan::contract(
    const TermPlan& term,
    TensorExpectationWorkspace& workspace_value) const {
    for (std::size_t step_index = 0; step_index < term.steps.size(); ++step_index) {
        const Step& step = term.steps[step_index];
        std::vector<QComplex>& output_storage =
            workspace_value.outputs_[step.workspace_slot];
        const std::span<QComplex> output(
            output_storage.data(), step.output_entries);
        const std::size_t lower_mask =
            step.selected_position == 0U
                ? 0U
                : (std::size_t{1} << step.selected_position) - 1U;

        for (std::size_t reduced_index = 0; reduced_index < output.size(); ++reduced_index) {
            const std::size_t low = reduced_index & lower_mask;
            const std::size_t high = reduced_index >> step.selected_position;
            const std::size_t base_union_index =
                low | (high << (step.selected_position + 1U));
            QComplex sum{};
            for (std::size_t selected_bit = 0; selected_bit < 2U; ++selected_bit) {
                const std::size_t union_index =
                    base_union_index | (selected_bit << step.selected_position);
                QComplex product{1.0, 0.0};
                for (const InputMap& input : step.inputs) {
                    const std::span<const QComplex> values =
                        node_values(term, input.node, workspace_value);
                    const QComplex value = values[mapped_factor_index(union_index, input.positions)];
                    if (value.re == 0.0 && value.im == 0.0) {
                        product = {};
                        break;
                    }
                    if (value.re != 1.0 || value.im != 0.0) {
                        product *= value;
                    }
                }
                sum += product;
            }
            output[reduced_index] = sum;
        }
    }

    QComplex result{1.0, 0.0};
    for (const std::size_t node : term.terminal_nodes) {
        const std::span<const QComplex> values =
            node_values(term, node, workspace_value);
        if (values.size() != 1U) {
            throw QStateError("Tensor expectation terminal node is not scalar");
        }
        result *= values.front();
    }
    return result;
}

QComplex TensorExpectationPlan::expectation(
    TensorContractionStats* stats) const {
    TensorExpectationWorkspace local = workspace();
    return expectation(local, stats);
}

QComplex TensorExpectationPlan::expectation(
    TensorExpectationWorkspace& workspace_value,
    TensorContractionStats* stats) const {
    if (observables_.size() != 1U) {
        throw QStateError(
            "Tensor expectation single-result query requires exactly one observable");
    }
    validate_workspace(workspace_value);
    const ObservablePlan observable = observables_.front();
    QComplex result{};
    for (std::size_t term_index = observable.term_begin;
         term_index < observable.term_end;
         ++term_index) {
        const TermPlan& term = terms_[term_index];
        const QComplex value = term.identity
            ? QComplex{1.0, 0.0}
            : contract(term, workspace_value);
        result += term.coefficient * value;
    }
    if (stats != nullptr) {
        *stats = stats_;
    }
    return result;
}

void TensorExpectationPlan::expectations(
    std::span<QComplex> results,
    TensorContractionStats* stats) const {
    TensorExpectationWorkspace local = workspace();
    expectations(results, local, stats);
}

void TensorExpectationPlan::expectations(
    std::span<QComplex> results,
    TensorExpectationWorkspace& workspace_value,
    TensorContractionStats* stats) const {
    if (results.size() != observables_.size()) {
        throw QStateError(
            "Tensor expectation result count does not match observable count");
    }
    validate_workspace(workspace_value);
    for (std::size_t observable_index = 0;
         observable_index < observables_.size();
         ++observable_index) {
        const ObservablePlan observable = observables_[observable_index];
        QComplex result{};
        for (std::size_t term_index = observable.term_begin;
             term_index < observable.term_end;
             ++term_index) {
            const TermPlan& term = terms_[term_index];
            const QComplex value = term.identity
                ? QComplex{1.0, 0.0}
                : contract(term, workspace_value);
            result += term.coefficient * value;
        }
        results[observable_index] = result;
    }
    if (stats != nullptr) {
        *stats = stats_;
    }
}

std::size_t TensorExpectationPlan::estimated_bytes() const noexcept {
    std::size_t bytes = sizeof(*this) +
                        observables_.capacity() * sizeof(ObservablePlan) +
                        terms_.capacity() * sizeof(TermPlan) +
                        slot_sizes_.capacity() * sizeof(std::size_t);
    for (const TermPlan& term : terms_) {
        bytes += term.sources.capacity() * sizeof(SourceFactor);
        for (const SourceFactor& source : term.sources) {
            bytes += source.variables.capacity() * sizeof(VariableId);
            bytes += source.values.capacity() * sizeof(QComplex);
        }
        bytes += term.operators.capacity() * sizeof(std::array<QComplex, 4>);
        bytes += term.steps.capacity() * sizeof(Step);
        bytes += term.terminal_nodes.capacity() * sizeof(std::size_t);
        for (const Step& step : term.steps) {
            bytes += step.union_variables.capacity() * sizeof(VariableId);
            bytes += step.output_variables.capacity() * sizeof(VariableId);
            bytes += step.inputs.capacity() * sizeof(InputMap);
            for (const InputMap& input : step.inputs) {
                bytes += input.positions.capacity() * sizeof(std::size_t);
            }
        }
    }
    return bytes;
}

}  // namespace qubit