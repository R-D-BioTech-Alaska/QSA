#include "qubit/qtensor.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <queue>
#include <string>

namespace qubit {
namespace {

using VariableId = std::uint32_t;

[[nodiscard]] std::size_t compiled_binary_entries(std::size_t variables) noexcept {
    if (variables >= std::numeric_limits<std::size_t>::digits) {
        return std::numeric_limits<std::size_t>::max();
    }
    return std::size_t{1} << variables;
}

[[nodiscard]] std::vector<VariableId> bucket_union(
    std::span<const std::size_t> bucket,
    const std::vector<std::vector<VariableId>>& scopes) {
    std::vector<VariableId> variables;
    for (const std::size_t node : bucket) {
        variables.insert(variables.end(), scopes[node].begin(), scopes[node].end());
    }
    std::sort(variables.begin(), variables.end());
    variables.erase(std::unique(variables.begin(), variables.end()), variables.end());
    return variables;
}

struct PlannerCandidate {
    std::size_t entries{0U};
    VariableId variable{0U};
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

}  // namespace

TensorContractionPlan TensorNetworkCircuit::compile() const {
    return TensorContractionPlan(*this);
}

std::size_t TensorContractionWorkspace::estimated_bytes() const noexcept {
    std::size_t bytes = sizeof(*this) +
                        pins_.capacity() * sizeof(std::array<QComplex, 2>) +
                        outputs_.capacity() * sizeof(std::vector<QComplex>);
    for (const auto& output : outputs_) {
        bytes += output.capacity() * sizeof(QComplex);
    }
    return bytes;
}

TensorContractionPlan::TensorContractionPlan(const TensorNetworkCircuit& circuit)
    : qubit_count_(circuit.qubit_count_),
      source_factor_count_(circuit.factors_.size()),
      pin_node_begin_(source_factor_count_),
      step_node_begin_(source_factor_count_ + qubit_count_) {
    std::string reason;
    if (!circuit.validate(&reason)) {
        throw QStateError("Cannot compile invalid tensor network: " + reason);
    }
    if (source_factor_count_ + qubit_count_ > circuit.config_.max_factors) {
        throw QStateError("Tensor network query exceeded max_factors");
    }

    sources_.reserve(source_factor_count_);
    std::vector<std::vector<VariableId>> scopes;
    scopes.reserve(source_factor_count_ + qubit_count_ + circuit.next_variable_);
    for (const TensorNetworkCircuit::Factor& factor : circuit.factors_) {
        SourceFactor source;
        source.variables = factor.variables;
        source.values = factor.values;
        scopes.push_back(source.variables);
        sources_.push_back(std::move(source));
    }
    for (const VariableId wire : circuit.current_wires_) {
        scopes.push_back({wire});
    }

    std::vector<bool> active(scopes.size(), true);
    std::vector<std::vector<std::size_t>> incidence(circuit.next_variable_);
    for (std::size_t node = 0U; node < scopes.size(); ++node) {
        for (const VariableId variable : scopes[node]) {
            if (variable >= incidence.size()) {
                throw QStateError("Tensor contraction plan scope contains an invalid variable");
            }
            incidence[variable].push_back(node);
        }
    }

    std::vector<std::uint64_t> revisions(circuit.next_variable_, 0U);
    std::priority_queue<
        PlannerCandidate,
        std::vector<PlannerCandidate>,
        PlannerCandidateGreater> candidates;

    const auto enqueue = [&](VariableId variable) {
        if (incidence[variable].empty()) {
            return;
        }
        const std::vector<VariableId> variables = bucket_union(incidence[variable], scopes);
        candidates.push({
            compiled_binary_entries(variables.size()),
            variable,
            revisions[variable],
        });
    };
    for (std::size_t variable = 0U; variable < incidence.size(); ++variable) {
        if (!incidence[variable].empty()) {
            enqueue(static_cast<VariableId>(variable));
        }
    }

    stats_.source_operations = circuit.operation_count_;
    stats_.source_factors = circuit.factors_.size();

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
        const VariableId selected = candidate.variable;
        const std::vector<std::size_t> selected_bucket = incidence[selected];
        const std::vector<VariableId> selected_union = bucket_union(selected_bucket, scopes);
        const std::size_t selected_entries = compiled_binary_entries(selected_union.size());
        if (selected_entries > circuit.config_.max_contraction_entries) {
            throw QStateError("Tensor network contraction exceeded max_contraction_entries");
        }

        const auto selected_it =
            std::lower_bound(selected_union.begin(), selected_union.end(), selected);
        if (selected_it == selected_union.end() || *selected_it != selected) {
            throw QStateError("Tensor contraction plan variable is missing from its bucket");
        }

        Step step;
        step.eliminated = selected;
        step.union_variables = selected_union;
        step.selected_position =
            static_cast<std::size_t>(selected_it - selected_union.begin());
        step.output_variables = selected_union;
        step.output_variables.erase(
            step.output_variables.begin() + static_cast<std::ptrdiff_t>(step.selected_position));
        step.output_entries = compiled_binary_entries(step.output_variables.size());
        if (step.output_entries > circuit.config_.max_contraction_entries) {
            throw QStateError("Tensor network reduced factor exceeded max_contraction_entries");
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
                    throw QStateError("Tensor contraction plan input variable is missing from union");
                }
                input.positions.push_back(
                    static_cast<std::size_t>(position - selected_union.begin()));
            }
            step.inputs.push_back(std::move(input));
            active[node] = false;
        }

        stats_.peak_union_variables =
            std::max(stats_.peak_union_variables, step.union_variables.size());
        stats_.peak_contraction_entries =
            std::max(stats_.peak_contraction_entries, selected_entries);
        ++stats_.eliminated_variables;

        const std::size_t output_node = scopes.size();
        for (const VariableId variable : selected_union) {
            std::vector<std::size_t>& nodes = incidence[variable];
            std::erase_if(nodes, [&](std::size_t node) {
                return std::binary_search(
                    selected_bucket.begin(), selected_bucket.end(), node);
            });
            ++revisions[variable];
        }

        scopes.push_back(step.output_variables);
        active.push_back(true);
        for (const VariableId variable : step.output_variables) {
            incidence[variable].push_back(output_node);
        }
        for (const VariableId variable : selected_union) {
            enqueue(variable);
        }
        steps_.push_back(std::move(step));
    }

    for (std::size_t node = 0; node < active.size(); ++node) {
        if (active[node]) {
            if (!scopes[node].empty()) {
                throw QStateError("Tensor contraction plan did not eliminate every variable");
            }
            terminal_nodes_.push_back(node);
        }
    }
    if (terminal_nodes_.empty()) {
        throw QStateError("Tensor contraction plan produced no terminal scalar");
    }
}

TensorContractionWorkspace TensorContractionPlan::workspace() const {
    TensorContractionWorkspace result;
    result.pins_.resize(qubit_count_);
    result.outputs_.resize(steps_.size());
    for (std::size_t index = 0; index < steps_.size(); ++index) {
        result.outputs_[index].resize(steps_[index].output_entries);
    }
    return result;
}

void TensorContractionPlan::validate_workspace(
    const TensorContractionWorkspace& workspace) const {
    if (workspace.pins_.size() != qubit_count_ ||
        workspace.outputs_.size() != steps_.size()) {
        throw QStateError("Tensor contraction workspace does not match its plan");
    }
    for (std::size_t index = 0; index < steps_.size(); ++index) {
        if (workspace.outputs_[index].size() != steps_[index].output_entries) {
            throw QStateError("Tensor contraction workspace output shape does not match its plan");
        }
    }
}

std::span<const QComplex> TensorContractionPlan::node_values(
    std::size_t node,
    const TensorContractionWorkspace& workspace) const {
    if (node < source_factor_count_) {
        return sources_[node].values;
    }
    if (node < step_node_begin_) {
        const std::size_t pin = node - pin_node_begin_;
        return workspace.pins_[pin];
    }
    const std::size_t output = node - step_node_begin_;
    if (output >= workspace.outputs_.size()) {
        throw QStateError("Tensor contraction plan references an invalid workspace node");
    }
    return workspace.outputs_[output];
}

QComplex TensorContractionPlan::amplitude(
    std::span<const std::uint8_t> basis_bits,
    TensorContractionStats* stats) const {
    TensorContractionWorkspace local = workspace();
    return amplitude(basis_bits, local, stats);
}

QComplex TensorContractionPlan::amplitude(
    std::span<const std::uint8_t> basis_bits,
    TensorContractionWorkspace& workspace_value,
    TensorContractionStats* stats) const {
    if (basis_bits.size() != qubit_count_) {
        throw QStateError("Tensor contraction plan basis width does not match qubit count");
    }
    validate_workspace(workspace_value);

    for (std::size_t qubit = 0; qubit < qubit_count_; ++qubit) {
        if (basis_bits[qubit] > 1U) {
            throw QStateError("Tensor contraction plan basis bits must be zero or one");
        }
        auto& pin = workspace_value.pins_[qubit];
        if (basis_bits[qubit] == 0U) {
            pin[0] = {1.0, 0.0};
            pin[1] = {0.0, 0.0};
        } else {
            pin[0] = {0.0, 0.0};
            pin[1] = {1.0, 0.0};
        }
    }

    for (std::size_t step_index = 0; step_index < steps_.size(); ++step_index) {
        const Step& step = steps_[step_index];
        std::vector<QComplex>& output = workspace_value.outputs_[step_index];
        const std::size_t lower_mask =
            step.selected_position == 0U
                ? 0U
                : (std::size_t{1} << step.selected_position) - 1U;

        for (std::size_t reduced_index = 0; reduced_index < output.size(); ++reduced_index) {
            QComplex sum{};
            for (std::size_t selected_bit = 0; selected_bit < 2U; ++selected_bit) {
                const std::size_t low = reduced_index & lower_mask;
                const std::size_t high = reduced_index >> step.selected_position;
                const std::size_t union_index =
                    low |
                    (selected_bit << step.selected_position) |
                    (high << (step.selected_position + 1U));

                QComplex product{1.0, 0.0};
                for (const InputMap& input : step.inputs) {
                    const std::span<const QComplex> values = node_values(input.node, workspace_value);
                    std::size_t input_index = 0U;
                    for (std::size_t position = 0; position < input.positions.size(); ++position) {
                        input_index |= ((union_index >> input.positions[position]) & 1U) << position;
                    }
                    product *= values[input_index];
                }
                sum += product;
            }
            output[reduced_index] = sum;
        }
    }

    QComplex result{1.0, 0.0};
    for (const std::size_t node : terminal_nodes_) {
        const std::span<const QComplex> values = node_values(node, workspace_value);
        if (values.size() != 1U) {
            throw QStateError("Tensor contraction plan terminal node is not scalar");
        }
        result *= values.front();
    }
    if (stats != nullptr) {
        *stats = stats_;
    }
    return result;
}

QComplex TensorContractionPlan::amplitude(
    BasisIndex basis,
    TensorContractionStats* stats) const {
    TensorContractionWorkspace local = workspace();
    return amplitude(basis, local, stats);
}

QComplex TensorContractionPlan::amplitude(
    BasisIndex basis,
    TensorContractionWorkspace& workspace_value,
    TensorContractionStats* stats) const {
    if (qubit_count_ > std::numeric_limits<BasisIndex>::digits) {
        throw QStateError("Tensor contraction plan BasisIndex query is limited to 64 qubits");
    }
    if (qubit_count_ < std::numeric_limits<BasisIndex>::digits &&
        basis >= (BasisIndex{1} << qubit_count_)) {
        throw QStateError("Tensor contraction plan basis index is out of range");
    }
    std::array<std::uint8_t, std::numeric_limits<BasisIndex>::digits> bits{};
    for (std::size_t qubit = 0; qubit < qubit_count_; ++qubit) {
        bits[qubit] = static_cast<std::uint8_t>((basis >> qubit) & BasisIndex{1});
    }
    return amplitude(
        std::span<const std::uint8_t>(bits.data(), qubit_count_),
        workspace_value,
        stats);
}

std::size_t TensorContractionPlan::estimated_bytes() const noexcept {
    std::size_t bytes = sizeof(*this) +
                        sources_.capacity() * sizeof(SourceFactor) +
                        steps_.capacity() * sizeof(Step) +
                        terminal_nodes_.capacity() * sizeof(std::size_t);
    for (const SourceFactor& source : sources_) {
        bytes += source.variables.capacity() * sizeof(VariableId) +
                 source.values.capacity() * sizeof(QComplex);
    }
    for (const Step& step : steps_) {
        bytes += step.union_variables.capacity() * sizeof(VariableId) +
                 step.output_variables.capacity() * sizeof(VariableId) +
                 step.inputs.capacity() * sizeof(InputMap);
        for (const InputMap& input : step.inputs) {
            bytes += input.positions.capacity() * sizeof(std::size_t);
        }
    }
    return bytes;
}

}  // namespace qubit
