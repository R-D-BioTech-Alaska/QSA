#include "qubit/qtensor.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <string>

namespace qubit {
namespace {

[[nodiscard]] std::size_t compiled_binary_entries(std::size_t variables) noexcept {
    if (variables >= std::numeric_limits<std::size_t>::digits) {
        return std::numeric_limits<std::size_t>::max();
    }
    return std::size_t{1} << variables;
}

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
    stats_.source_operations = circuit.operation_count_;
    stats_.source_factors = circuit.factors_.size();

    while (true) {
        std::vector<std::vector<std::size_t>> incidence(circuit.next_variable_);
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
                variables.insert(variables.end(), scopes[node].begin(), scopes[node].end());
            }
            std::sort(variables.begin(), variables.end());
            variables.erase(std::unique(variables.begin(), variables.end()), variables.end());
            const std::size_t entries = compiled_binary_entries(variables.size());
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
            throw QStateError("Tensor contraction plan lost its active variables");
        }
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

        scopes.push_back(step.output_variables);
        active.push_back(true);
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
                    const std::span<const QComplex> values =
                        node_values(input.node, workspace_value);
                    std::size_t factor_index = 0U;
                    for (std::size_t local = 0; local < input.positions.size(); ++local) {
                        const std::size_t bit =
                            (union_index >> input.positions[local]) & 1U;
                        factor_index |= bit << local;
                    }
                    product *= values[factor_index];
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
        bytes += source.variables.capacity() * sizeof(VariableId);
        bytes += source.values.capacity() * sizeof(QComplex);
    }
    for (const Step& step : steps_) {
        bytes += step.union_variables.capacity() * sizeof(VariableId);
        bytes += step.output_variables.capacity() * sizeof(VariableId);
        bytes += step.inputs.capacity() * sizeof(InputMap);
        for (const InputMap& input : step.inputs) {
            bytes += input.positions.capacity() * sizeof(std::size_t);
        }
    }
    return bytes;
}

}  // namespace qubit
