#pragma once

#include "qubit/qstate.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace qubit {

// CausalState is a branchable exact QSA register. Forking shares one immutable
// register in O(1). A branch deep-copies only when it is first mutated, and
// adopting a selected branch transfers ownership in O(1).
class CausalState {
public:
    explicit CausalState(std::size_t qubit_count, QStateConfig config = {})
        : state_(std::make_shared<QRegister>(qubit_count, config)) {}

    explicit CausalState(QRegister state)
        : state_(std::make_shared<QRegister>(std::move(state))) {}

    [[nodiscard]] CausalState fork() const {
        ensure_valid();
        return CausalState(state_);
    }

    [[nodiscard]] std::vector<CausalState> fork_many(std::size_t branch_count) const {
        ensure_valid();
        std::vector<CausalState> branches;
        branches.reserve(branch_count);
        for (std::size_t index = 0; index < branch_count; ++index) {
            branches.push_back(CausalState(state_));
        }
        return branches;
    }

    [[nodiscard]] const QRegister& read() const {
        ensure_valid();
        return *state_;
    }

    [[nodiscard]] QRegister& write() {
        ensure_valid();
        if (state_.use_count() != 1L) {
            state_ = std::make_shared<QRegister>(*state_);
        }
        return *state_;
    }

    template <class Function>
    decltype(auto) mutate(Function&& function) {
        return std::forward<Function>(function)(write());
    }

    void adopt(CausalState&& selected) {
        selected.ensure_valid();
        state_ = std::move(selected.state_);
    }

    void reset(QRegister state) {
        state_ = std::make_shared<QRegister>(std::move(state));
    }

    [[nodiscard]] bool valid() const noexcept { return static_cast<bool>(state_); }
    [[nodiscard]] bool unique() const noexcept {
        return state_ && state_.use_count() == 1L;
    }
    [[nodiscard]] long shared_owner_count() const noexcept {
        return state_ ? state_.use_count() : 0L;
    }
    [[nodiscard]] bool shares_state_with(const CausalState& other) const noexcept {
        return state_ && state_ == other.state_;
    }
    [[nodiscard]] std::size_t estimated_bytes() const {
        return read().estimated_bytes();
    }

private:
    explicit CausalState(std::shared_ptr<QRegister> shared_state)
        : state_(std::move(shared_state)) {}

    void ensure_valid() const {
        if (!state_) {
            throw QStateError("Causal state has no active register");
        }
    }

    std::shared_ptr<QRegister> state_{};
};

struct PauliObservableConfig {
    double imaginary_tolerance{1e-10};
};

// Exact component-wise Pauli expectation plan. Each word uses one character
// per logical qubit: I, X, Y, or Z. Independent QSA components are evaluated
// separately and multiplied, so a global statevector is never required merely
// to read observables.
class PauliObservablePlan {
public:
    PauliObservablePlan(
        std::size_t qubit_count,
        std::span<const std::string> words,
        PauliObservableConfig config = {})
        : qubit_count_(qubit_count), config_(config) {
        if (qubit_count_ == 0U) {
            throw QStateError("Pauli observable qubit count must be positive");
        }
        if (!std::isfinite(config_.imaginary_tolerance) ||
            config_.imaginary_tolerance <= 0.0) {
            throw QStateError("Pauli observable tolerance must be finite and positive");
        }
        words_.reserve(words.size());
        for (const std::string& word : words) {
            words_.push_back(normalize_word(word));
        }
    }

    [[nodiscard]] std::size_t qubit_count() const noexcept { return qubit_count_; }
    [[nodiscard]] std::size_t observable_count() const noexcept { return words_.size(); }
    [[nodiscard]] const std::vector<std::string>& words() const noexcept { return words_; }

    [[nodiscard]] std::vector<double> execute(const QRegister& state) const {
        if (state.qubit_count() != qubit_count_) {
            throw QStateError("Pauli observable plan qubit count does not match register");
        }
        const std::vector<QComponentReadView> views = state.component_read_views();
        std::vector<double> values;
        values.reserve(words_.size());
        for (const std::string& word : words_) {
            QComplex total{1.0, 0.0};
            for (const QComponentReadView& view : views) {
                total *= component_expectation(view, word);
            }
            const double scale = 1.0 + std::abs(total.re);
            if (std::abs(total.im) > config_.imaginary_tolerance * scale) {
                throw QStateError("Hermitian Pauli expectation developed a non-real value");
            }
            values.push_back(total.re);
        }
        return values;
    }

    [[nodiscard]] std::vector<double> execute(const CausalState& state) const {
        return execute(state.read());
    }

    [[nodiscard]] std::vector<std::vector<double>> execute_many(
        std::span<const CausalState> states) const {
        std::vector<std::vector<double>> results;
        results.reserve(states.size());
        for (const CausalState& state : states) {
            results.push_back(execute(state));
        }
        return results;
    }

private:
    [[nodiscard]] std::string normalize_word(std::string_view word) const {
        if (word.size() != qubit_count_) {
            throw QStateError("Pauli word length does not match qubit count");
        }
        std::string normalized;
        normalized.reserve(word.size());
        for (char symbol : word) {
            switch (symbol) {
                case 'I': case 'i': normalized.push_back('I'); break;
                case 'X': case 'x': normalized.push_back('X'); break;
                case 'Y': case 'y': normalized.push_back('Y'); break;
                case 'Z': case 'z': normalized.push_back('Z'); break;
                default:
                    throw QStateError("Pauli words may contain only I, X, Y, and Z");
            }
        }
        return normalized;
    }

    [[nodiscard]] static QComplex source_coefficient(
        BasisIndex source,
        std::span<const QubitId> component_qubits,
        const std::string& word,
        BasisIndex& target) {
        QComplex coefficient{1.0, 0.0};
        target = source;
        for (std::size_t position = 0; position < component_qubits.size(); ++position) {
            const QubitId qubit = component_qubits[position];
            const char symbol = word[static_cast<std::size_t>(qubit)];
            const bool one = ((source >> position) & BasisIndex{1}) != 0U;
            switch (symbol) {
                case 'I':
                    break;
                case 'X':
                    target ^= BasisIndex{1} << position;
                    break;
                case 'Y':
                    target ^= BasisIndex{1} << position;
                    coefficient *= one ? QComplex{0.0, -1.0} : QComplex{0.0, 1.0};
                    break;
                case 'Z':
                    if (one) {
                        coefficient *= -1.0;
                    }
                    break;
                default:
                    throw QStateError("Invalid normalized Pauli operator");
            }
        }
        return coefficient;
    }

    [[nodiscard]] static QComplex sparse_amplitude(
        std::span<const AmplitudeStore::SparseEntry> entries,
        BasisIndex index) {
        const auto iterator = std::lower_bound(
            entries.begin(), entries.end(), index,
            [](const AmplitudeStore::SparseEntry& entry, BasisIndex target) {
                return entry.first < target;
            });
        return iterator != entries.end() && iterator->first == index
            ? iterator->second
            : QComplex{};
    }

    [[nodiscard]] static QComplex component_expectation(
        const QComponentReadView& view,
        const std::string& word) {
        bool identity_only = true;
        for (QubitId qubit : view.qubits) {
            if (word[static_cast<std::size_t>(qubit)] != 'I') {
                identity_only = false;
                break;
            }
        }
        if (identity_only) {
            return {1.0, 0.0};
        }

        if (view.kind == ComponentKind::Cell) {
            if (view.cell == nullptr || view.qubits.size() != 1U) {
                throw QStateError("Invalid Bloch-cell read view");
            }
            const char symbol = word[static_cast<std::size_t>(view.qubits.front())];
            switch (symbol) {
                case 'I': return {1.0, 0.0};
                case 'X': return {view.cell->x, 0.0};
                case 'Y': return {view.cell->y, 0.0};
                case 'Z': return {view.cell->z, 0.0};
                default: throw QStateError("Invalid Pauli operator for Bloch cell");
            }
        }

        QComplex expectation{};
        if (view.kind == ComponentKind::Dense) {
            if (view.dense.size() != static_cast<std::size_t>(view.dimension)) {
                throw QStateError("Invalid dense component read view");
            }
            for (BasisIndex source = 0; source < view.dimension; ++source) {
                BasisIndex target = 0;
                const QComplex coefficient = source_coefficient(
                    source, view.qubits, word, target);
                expectation += view.dense[static_cast<std::size_t>(target)].conjugate()
                    * coefficient
                    * view.dense[static_cast<std::size_t>(source)];
            }
            return expectation;
        }

        if (view.kind == ComponentKind::Sparse) {
            for (const auto& [source, source_amplitude] : view.sparse) {
                BasisIndex target = 0;
                const QComplex coefficient = source_coefficient(
                    source, view.qubits, word, target);
                const QComplex target_amplitude = sparse_amplitude(view.sparse, target);
                if (target_amplitude.norm2() == 0.0) {
                    continue;
                }
                expectation += target_amplitude.conjugate()
                    * coefficient
                    * source_amplitude;
            }
            return expectation;
        }

        throw QStateError("Unknown QSA component kind");
    }

    std::size_t qubit_count_{0};
    PauliObservableConfig config_{};
    std::vector<std::string> words_{};
};

}  // namespace qubit
