#pragma once

#include "qubit/qstate.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace qubit {

struct PauliSupportTerm {
    QubitId qubit{0};
    char axis{'Z'};
};

struct PauliSupportConfig {
    double imaginary_tolerance{1e-10};
};

// Stores only non-identity positions. Execution visits components touched by
// the observable instead of scanning the complete logical register.
class PauliSupportPlan {
public:
    PauliSupportPlan(
        std::size_t qubit_count,
        std::span<const std::vector<PauliSupportTerm>> observables,
        PauliSupportConfig config = {})
        : qubit_count_(qubit_count), config_(config) {
        if (qubit_count_ == 0U) {
            throw QStateError("Pauli support qubit count must be positive");
        }
        if (!std::isfinite(config_.imaginary_tolerance) ||
            config_.imaginary_tolerance <= 0.0) {
            throw QStateError("Pauli support tolerance must be finite and positive");
        }

        observables_.reserve(observables.size());
        for (const auto& observable : observables) {
            observables_.push_back(normalize(observable));
            term_count_ += observables_.back().size();
        }
    }

    [[nodiscard]] std::size_t qubit_count() const noexcept { return qubit_count_; }
    [[nodiscard]] std::size_t observable_count() const noexcept {
        return observables_.size();
    }
    [[nodiscard]] std::size_t term_count() const noexcept { return term_count_; }
    [[nodiscard]] const std::vector<std::vector<PauliSupportTerm>>& observables() const noexcept {
        return observables_;
    }

    [[nodiscard]] std::vector<double> execute(const QRegister& state) const {
        if (state.qubit_count() != qubit_count_) {
            throw QStateError("Pauli support plan qubit count does not match register");
        }

        std::vector<double> values;
        values.reserve(observables_.size());
        for (const auto& observable : observables_) {
            if (observable.empty()) {
                values.push_back(1.0);
                continue;
            }

            std::vector<ComponentGroup> groups;
            groups.reserve(observable.size());
            for (const PauliSupportTerm& term : observable) {
                const QComponentReadView view = state.component_read_view(term.qubit);
                ComponentGroup* group = nullptr;
                for (ComponentGroup& candidate : groups) {
                    if (same_component(candidate.view, view)) {
                        group = &candidate;
                        break;
                    }
                }
                if (group == nullptr) {
                    groups.push_back(ComponentGroup{view, {}});
                    group = &groups.back();
                }

                const auto position = std::find(
                    view.qubits.begin(), view.qubits.end(), term.qubit);
                if (position == view.qubits.end()) {
                    throw QStateError("Pauli support qubit is missing from its component");
                }
                group->operators.push_back(LocalOperator{
                    static_cast<std::size_t>(position - view.qubits.begin()),
                    term.axis,
                });
            }

            QComplex total{1.0, 0.0};
            for (const ComponentGroup& group : groups) {
                total *= component_expectation(group.view, group.operators);
            }
            const double scale = 1.0 + std::abs(total.re);
            if (std::abs(total.im) > config_.imaginary_tolerance * scale) {
                throw QStateError("Hermitian Pauli support developed a non-real value");
            }
            values.push_back(total.re);
        }
        return values;
    }

    [[nodiscard]] std::vector<std::vector<double>> execute_many(
        std::span<const QRegister* const> states) const {
        std::vector<std::vector<double>> values;
        values.reserve(states.size());
        for (const QRegister* state : states) {
            if (state == nullptr) {
                throw QStateError("Pauli support state pointer is null");
            }
            values.push_back(execute(*state));
        }
        return values;
    }

private:
    struct LocalOperator {
        std::size_t position{0};
        char axis{'Z'};
    };

    struct ComponentGroup {
        QComponentReadView view{};
        std::vector<LocalOperator> operators{};
    };

    [[nodiscard]] std::vector<PauliSupportTerm> normalize(
        std::span<const PauliSupportTerm> observable) const {
        std::vector<PauliSupportTerm> normalized;
        normalized.reserve(observable.size());
        for (PauliSupportTerm term : observable) {
            if (static_cast<std::size_t>(term.qubit) >= qubit_count_) {
                throw QStateError("Pauli support qubit is outside the register");
            }
            switch (term.axis) {
                case 'x': term.axis = 'X'; break;
                case 'y': term.axis = 'Y'; break;
                case 'z': term.axis = 'Z'; break;
                case 'X': case 'Y': case 'Z': break;
                case 'I': case 'i':
                    continue;
                default:
                    throw QStateError("Pauli support axes may contain only I, X, Y, and Z");
            }
            normalized.push_back(term);
        }
        std::sort(
            normalized.begin(), normalized.end(),
            [](const PauliSupportTerm& first, const PauliSupportTerm& second) {
                return first.qubit < second.qubit;
            });
        for (std::size_t index = 1; index < normalized.size(); ++index) {
            if (normalized[index - 1U].qubit == normalized[index].qubit) {
                throw QStateError("Pauli support contains a duplicate qubit");
            }
        }
        return normalized;
    }

    [[nodiscard]] static bool same_component(
        const QComponentReadView& first,
        const QComponentReadView& second) noexcept {
        return first.qubits.data() == second.qubits.data() &&
            first.qubits.size() == second.qubits.size();
    }

    [[nodiscard]] static QComplex source_coefficient(
        BasisIndex source,
        std::span<const LocalOperator> operators,
        BasisIndex& target) {
        QComplex coefficient{1.0, 0.0};
        target = source;
        for (const LocalOperator& operation : operators) {
            const bool one =
                ((source >> operation.position) & BasisIndex{1}) != 0U;
            switch (operation.axis) {
                case 'X':
                    target ^= BasisIndex{1} << operation.position;
                    break;
                case 'Y':
                    target ^= BasisIndex{1} << operation.position;
                    coefficient *= one
                        ? QComplex{0.0, -1.0}
                        : QComplex{0.0, 1.0};
                    break;
                case 'Z':
                    if (one) {
                        coefficient *= -1.0;
                    }
                    break;
                default:
                    throw QStateError("Invalid normalized Pauli support axis");
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
        std::span<const LocalOperator> operators) {
        if (view.kind == ComponentKind::Cell) {
            if (view.cell == nullptr || view.qubits.size() != 1U ||
                operators.size() != 1U || operators.front().position != 0U) {
                throw QStateError("Invalid Bloch-cell Pauli support");
            }
            switch (operators.front().axis) {
                case 'X': return {view.cell->x, 0.0};
                case 'Y': return {view.cell->y, 0.0};
                case 'Z': return {view.cell->z, 0.0};
                default: throw QStateError("Invalid Bloch-cell Pauli axis");
            }
        }

        QComplex expectation{};
        if (view.kind == ComponentKind::Dense) {
            if (view.dense.size() != static_cast<std::size_t>(view.dimension)) {
                throw QStateError("Invalid dense Pauli support view");
            }
            for (BasisIndex source = 0; source < view.dimension; ++source) {
                BasisIndex target = 0;
                const QComplex coefficient = source_coefficient(
                    source, operators, target);
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
                    source, operators, target);
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

        throw QStateError("Unknown Pauli support component kind");
    }

    std::size_t qubit_count_{0};
    PauliSupportConfig config_{};
    std::size_t term_count_{0};
    std::vector<std::vector<PauliSupportTerm>> observables_{};
};

}  // namespace qubit
