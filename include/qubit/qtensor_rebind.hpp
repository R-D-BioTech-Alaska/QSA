#pragma once

#include "qubit/qtensor.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace qubit {

class TensorExpectationRebindPlan {
public:
    TensorExpectationRebindPlan(
        const TensorNetworkCircuit& circuit,
        const PauliObservable& observable)
        : TensorExpectationRebindPlan(
              circuit,
              std::span<const PauliObservable>(&observable, 1U)) {}

    TensorExpectationRebindPlan(
        const TensorNetworkCircuit& circuit,
        std::span<const PauliObservable> observables)
        : plan_(circuit, observables),
          qubit_count_(circuit.qubit_count_),
          operation_count_(circuit.operation_count_),
          config_(circuit.config_),
          next_variable_(circuit.next_variable_),
          current_wires_(circuit.current_wires_) {
        factor_variables_.reserve(circuit.factors_.size());
        factor_value_sizes_.reserve(circuit.factors_.size());
        for (const auto& factor : circuit.factors_) {
            factor_variables_.push_back(factor.variables);
            factor_value_sizes_.push_back(factor.values.size());
        }
        build_bindings(circuit, observables);
    }

    void rebind(const TensorNetworkCircuit& circuit) {
        validate_topology(circuit);

        struct Replacement {
            std::size_t term_index{0U};
            std::size_t source_index{0U};
            std::vector<QComplex> values{};
        };

        std::vector<Replacement> replacements;
        replacements.reserve(bindings_.size());
        for (const Binding& binding : bindings_) {
            const auto& factor = circuit.factors_[binding.factor_index];
            const auto& source =
                plan_.terms_[binding.term_index].sources[binding.source_index];
            if (factor.values.size() != source.values.size()) {
                throw QStateError("Tensor expectation rebind source size changed");
            }

            Replacement replacement;
            replacement.term_index = binding.term_index;
            replacement.source_index = binding.source_index;
            replacement.values.resize(factor.values.size());
            if (!binding.bra) {
                replacement.values = factor.values;
            } else {
                for (std::size_t index = 0; index < factor.values.size(); ++index) {
                    const std::size_t target = index ^ binding.xor_mask;
                    if (target >= replacement.values.size()) {
                        throw QStateError("Tensor expectation rebind bra permutation is invalid");
                    }
                    replacement.values[target] = factor.values[index].conjugate();
                }
            }
            replacements.push_back(std::move(replacement));
        }

        for (Replacement& replacement : replacements) {
            plan_.terms_[replacement.term_index]
                .sources[replacement.source_index]
                .values = std::move(replacement.values);
        }
        ++rebind_count_;
    }

    [[nodiscard]] TensorExpectationWorkspace workspace() const {
        return plan_.workspace();
    }

    [[nodiscard]] QComplex expectation(
        TensorContractionStats* stats = nullptr) const {
        return plan_.expectation(stats);
    }

    [[nodiscard]] QComplex expectation(
        TensorExpectationWorkspace& workspace_value,
        TensorContractionStats* stats = nullptr) const {
        return plan_.expectation(workspace_value, stats);
    }

    void expectations(
        std::span<QComplex> results,
        TensorContractionStats* stats = nullptr) const {
        plan_.expectations(results, stats);
    }

    void expectations(
        std::span<QComplex> results,
        TensorExpectationWorkspace& workspace_value,
        TensorContractionStats* stats = nullptr) const {
        plan_.expectations(results, workspace_value, stats);
    }

    [[nodiscard]] std::size_t qubit_count() const noexcept {
        return qubit_count_;
    }

    [[nodiscard]] std::size_t observable_count() const noexcept {
        return plan_.observable_count();
    }

    [[nodiscard]] std::size_t term_count() const noexcept {
        return plan_.term_count();
    }

    [[nodiscard]] std::size_t step_count() const noexcept {
        return plan_.step_count();
    }

    [[nodiscard]] std::size_t rebind_count() const noexcept {
        return rebind_count_;
    }

    [[nodiscard]] const TensorContractionStats& stats() const noexcept {
        return plan_.stats();
    }

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        std::size_t bytes = sizeof(*this) + plan_.estimated_bytes() +
                            current_wires_.capacity() * sizeof(VariableId) +
                            factor_variables_.capacity() * sizeof(std::vector<VariableId>) +
                            factor_value_sizes_.capacity() * sizeof(std::size_t) +
                            bindings_.capacity() * sizeof(Binding);
        for (const auto& variables : factor_variables_) {
            bytes += variables.capacity() * sizeof(VariableId);
        }
        return bytes;
    }

private:
    using VariableId = std::uint32_t;

    struct Binding {
        std::size_t term_index{0U};
        std::size_t source_index{0U};
        std::size_t factor_index{0U};
        std::size_t xor_mask{0U};
        bool bra{false};
    };

    TensorExpectationPlan plan_;
    std::size_t qubit_count_{0U};
    std::size_t operation_count_{0U};
    TensorNetworkConfig config_{};
    VariableId next_variable_{0U};
    std::vector<VariableId> current_wires_{};
    std::vector<std::vector<VariableId>> factor_variables_{};
    std::vector<std::size_t> factor_value_sizes_{};
    std::vector<Binding> bindings_{};
    std::size_t rebind_count_{0U};

    [[nodiscard]] std::size_t factor_index(
        const TensorNetworkCircuit& circuit,
        std::span<const VariableId> variables) const {
        std::size_t match = circuit.factors_.size();
        for (std::size_t index = 0; index < circuit.factors_.size(); ++index) {
            if (circuit.factors_[index].variables.size() != variables.size()) {
                continue;
            }
            if (!std::equal(
                    variables.begin(), variables.end(),
                    circuit.factors_[index].variables.begin())) {
                continue;
            }
            if (match != circuit.factors_.size()) {
                throw QStateError("Tensor expectation rebind topology has duplicate factors");
            }
            match = index;
        }
        if (match == circuit.factors_.size()) {
            throw QStateError("Tensor expectation rebind source factor was not found");
        }
        return match;
    }

    void build_bindings(
        const TensorNetworkCircuit& circuit,
        std::span<const PauliObservable> observables) {
        std::vector<const PauliTerm*> compiled_terms;
        for (const PauliObservable& observable : observables) {
            for (const PauliTerm& term : observable.terms()) {
                if (term.coefficient.norm2() != 0.0) {
                    compiled_terms.push_back(&term);
                }
            }
        }
        if (compiled_terms.size() != plan_.terms_.size()) {
            throw QStateError("Tensor expectation rebind term map does not match the plan");
        }

        for (std::size_t term_index = 0;
             term_index < plan_.terms_.size();
             ++term_index) {
            const PauliTerm& observable_term = *compiled_terms[term_index];
            const auto& term = plan_.terms_[term_index];
            if (term.identity) {
                if (!term.sources.empty()) {
                    throw QStateError("Tensor expectation identity term has dynamic sources");
                }
                continue;
            }

            std::size_t bra_count = 0U;
            for (const auto& source : term.sources) {
                const bool bra_source = !source.variables.empty() &&
                    std::all_of(
                        source.variables.begin(), source.variables.end(),
                        [&](VariableId variable) { return variable >= next_variable_; });
                bra_count += static_cast<std::size_t>(bra_source);
            }
            if (bra_count == 0U || term.sources.size() < bra_count * 2U) {
                throw QStateError("Tensor expectation rebind source layout is invalid");
            }

            std::vector<std::uint8_t> invert(next_variable_, 0U);
            for (const PauliFactor& factor : observable_term.factors) {
                if (factor.axis != PauliAxis::X && factor.axis != PauliAxis::Y) {
                    continue;
                }
                if (factor.qubit >= current_wires_.size()) {
                    throw QStateError("Tensor expectation rebind Pauli factor is out of range");
                }
                invert[current_wires_[factor.qubit]] = 1U;
            }

            for (std::size_t local = 0U; local < bra_count; ++local) {
                const auto& ket_source = term.sources[local];
                const auto& bra_source = term.sources[bra_count + local];
                if (ket_source.variables.empty() ||
                    std::any_of(
                        ket_source.variables.begin(), ket_source.variables.end(),
                        [&](VariableId variable) { return variable >= next_variable_; })) {
                    throw QStateError("Tensor expectation rebind ket source layout is invalid");
                }

                const std::size_t source_factor =
                    factor_index(circuit, ket_source.variables);
                const auto& topology = circuit.factors_[source_factor].variables;
                if (bra_source.variables.size() != topology.size()) {
                    throw QStateError("Tensor expectation rebind bra source rank changed");
                }

                std::size_t xor_mask = 0U;
                for (std::size_t position = 0U; position < topology.size(); ++position) {
                    const std::uint64_t bra_variable =
                        static_cast<std::uint64_t>(topology[position]) +
                        static_cast<std::uint64_t>(next_variable_);
                    if (bra_variable > std::numeric_limits<VariableId>::max() ||
                        bra_source.variables[position] !=
                            static_cast<VariableId>(bra_variable)) {
                        throw QStateError("Tensor expectation rebind bra topology changed");
                    }
                    if (invert[topology[position]] != 0U) {
                        if (position >= std::numeric_limits<std::size_t>::digits) {
                            throw QStateError("Tensor expectation rebind bra mask overflowed");
                        }
                        xor_mask |= std::size_t{1} << position;
                    }
                }

                bindings_.push_back(
                    {term_index, local, source_factor, 0U, false});
                bindings_.push_back(
                    {term_index, bra_count + local, source_factor, xor_mask, true});
            }
        }
    }

    void validate_topology(const TensorNetworkCircuit& circuit) const {
        std::string reason;
        if (!circuit.validate(&reason)) {
            throw QStateError(
                "Cannot rebind invalid tensor network: " + reason);
        }
        if (circuit.qubit_count_ != qubit_count_ ||
            circuit.operation_count_ != operation_count_ ||
            circuit.next_variable_ != next_variable_ ||
            circuit.current_wires_ != current_wires_ ||
            circuit.factors_.size() != factor_variables_.size() ||
            circuit.config_.max_contraction_entries != config_.max_contraction_entries ||
            circuit.config_.max_factors != config_.max_factors) {
            throw QStateError("Tensor expectation rebind topology does not match its plan");
        }
        for (std::size_t index = 0; index < circuit.factors_.size(); ++index) {
            if (circuit.factors_[index].variables != factor_variables_[index] ||
                circuit.factors_[index].values.size() != factor_value_sizes_[index]) {
                throw QStateError("Tensor expectation rebind factor topology changed");
            }
        }
    }
};

}  // namespace qubit