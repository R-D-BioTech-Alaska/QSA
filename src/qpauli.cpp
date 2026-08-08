#include "qubit/qpauli.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <numbers>
#include <utility>

namespace qubit {
namespace {

struct AxisProduct {
    PauliAxis axis{PauliAxis::I};
    QComplex phase{1.0, 0.0};
};

struct LocalFactor {
    std::size_t position{0};
    PauliAxis axis{PauliAxis::I};
};

struct LocalGroup {
    QComponentReadView view{};
    std::vector<LocalFactor> factors{};
};

[[nodiscard]] bool factor_less(const PauliFactor& left, const PauliFactor& right) noexcept {
    if (left.qubit != right.qubit) {
        return left.qubit < right.qubit;
    }
    return static_cast<std::uint8_t>(left.axis) < static_cast<std::uint8_t>(right.axis);
}

[[nodiscard]] bool same_factors(
    const std::vector<PauliFactor>& left,
    const std::vector<PauliFactor>& right) noexcept {
    if (left.size() != right.size()) {
        return false;
    }
    for (std::size_t index = 0; index < left.size(); ++index) {
        if (left[index].qubit != right[index].qubit || left[index].axis != right[index].axis) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool factors_less(
    const std::vector<PauliFactor>& left,
    const std::vector<PauliFactor>& right) noexcept {
    return std::lexicographical_compare(
        left.begin(), left.end(), right.begin(), right.end(), factor_less);
}

[[nodiscard]] bool zero(const QComplex& value) noexcept {
    return value.re == 0.0 && value.im == 0.0;
}

[[nodiscard]] AxisProduct multiply_axes(PauliAxis left, PauliAxis right) {
    if (left == PauliAxis::I) {
        return {right, {1.0, 0.0}};
    }
    if (right == PauliAxis::I) {
        return {left, {1.0, 0.0}};
    }
    if (left == right) {
        return {PauliAxis::I, {1.0, 0.0}};
    }
    if (left == PauliAxis::X && right == PauliAxis::Y) {
        return {PauliAxis::Z, QI};
    }
    if (left == PauliAxis::Y && right == PauliAxis::Z) {
        return {PauliAxis::X, QI};
    }
    if (left == PauliAxis::Z && right == PauliAxis::X) {
        return {PauliAxis::Y, QI};
    }
    if (left == PauliAxis::Y && right == PauliAxis::X) {
        return {PauliAxis::Z, -QI};
    }
    if (left == PauliAxis::Z && right == PauliAxis::Y) {
        return {PauliAxis::X, -QI};
    }
    if (left == PauliAxis::X && right == PauliAxis::Z) {
        return {PauliAxis::Y, -QI};
    }
    throw QStateError("Pauli axis multiplication received an invalid axis");
}

void multiply_factor(PauliTerm& term, QubitId qubit, PauliAxis axis) {
    if (axis == PauliAxis::I) {
        return;
    }
    const auto position = std::lower_bound(
        term.factors.begin(),
        term.factors.end(),
        PauliFactor{qubit, PauliAxis::I},
        factor_less);
    if (position == term.factors.end() || position->qubit != qubit) {
        term.factors.insert(position, PauliFactor{qubit, axis});
        return;
    }

    const AxisProduct product = multiply_axes(position->axis, axis);
    term.coefficient *= product.phase;
    if (product.axis == PauliAxis::I) {
        term.factors.erase(position);
    } else {
        position->axis = product.axis;
    }
}

void multiply_word(PauliTerm& target, const PauliTerm& source) {
    target.coefficient *= source.coefficient;
    for (const PauliFactor& factor : source.factors) {
        multiply_factor(target, factor.qubit, factor.axis);
    }
}

[[nodiscard]] PauliAxis axis_at(const PauliTerm& term, QubitId qubit) noexcept {
    const auto position = std::lower_bound(
        term.factors.begin(),
        term.factors.end(),
        PauliFactor{qubit, PauliAxis::I},
        factor_less);
    if (position == term.factors.end() || position->qubit != qubit) {
        return PauliAxis::I;
    }
    return position->axis;
}

void set_axis(PauliTerm& term, QubitId qubit, PauliAxis axis) {
    const auto position = std::lower_bound(
        term.factors.begin(),
        term.factors.end(),
        PauliFactor{qubit, PauliAxis::I},
        factor_less);
    if (position == term.factors.end() || position->qubit != qubit) {
        if (axis != PauliAxis::I) {
            term.factors.insert(position, PauliFactor{qubit, axis});
        }
        return;
    }
    if (axis == PauliAxis::I) {
        term.factors.erase(position);
    } else {
        position->axis = axis;
    }
}

void apply_sign(PauliTerm& term, double sign) noexcept {
    term.coefficient *= sign;
}

void map_h(PauliTerm& term, QubitId qubit) {
    switch (axis_at(term, qubit)) {
        case PauliAxis::I:
            break;
        case PauliAxis::X:
            set_axis(term, qubit, PauliAxis::Z);
            break;
        case PauliAxis::Y:
            apply_sign(term, -1.0);
            break;
        case PauliAxis::Z:
            set_axis(term, qubit, PauliAxis::X);
            break;
    }
}

void map_x(PauliTerm& term, QubitId qubit) {
    const PauliAxis axis = axis_at(term, qubit);
    if (axis == PauliAxis::Y || axis == PauliAxis::Z) {
        apply_sign(term, -1.0);
    }
}

void map_y(PauliTerm& term, QubitId qubit) {
    const PauliAxis axis = axis_at(term, qubit);
    if (axis == PauliAxis::X || axis == PauliAxis::Z) {
        apply_sign(term, -1.0);
    }
}

void map_z(PauliTerm& term, QubitId qubit) {
    const PauliAxis axis = axis_at(term, qubit);
    if (axis == PauliAxis::X || axis == PauliAxis::Y) {
        apply_sign(term, -1.0);
    }
}

void map_s(PauliTerm& term, QubitId qubit) {
    switch (axis_at(term, qubit)) {
        case PauliAxis::I:
        case PauliAxis::Z:
            break;
        case PauliAxis::X:
            set_axis(term, qubit, PauliAxis::Y);
            apply_sign(term, -1.0);
            break;
        case PauliAxis::Y:
            set_axis(term, qubit, PauliAxis::X);
            break;
    }
}

void map_sdg(PauliTerm& term, QubitId qubit) {
    switch (axis_at(term, qubit)) {
        case PauliAxis::I:
        case PauliAxis::Z:
            break;
        case PauliAxis::X:
            set_axis(term, qubit, PauliAxis::Y);
            break;
        case PauliAxis::Y:
            set_axis(term, qubit, PauliAxis::X);
            apply_sign(term, -1.0);
            break;
    }
}

void append_generator(
    PauliTerm& term,
    OperationCode code,
    QubitId first,
    QubitId second,
    bool source_first,
    PauliAxis generator) {
    if (generator != PauliAxis::X && generator != PauliAxis::Z) {
        throw QStateError("Pauli Clifford generator must be X or Z");
    }

    switch (code) {
        case OperationCode::Cnot:
            if (source_first && generator == PauliAxis::X) {
                multiply_factor(term, first, PauliAxis::X);
                multiply_factor(term, second, PauliAxis::X);
            } else if (source_first) {
                multiply_factor(term, first, PauliAxis::Z);
            } else if (generator == PauliAxis::X) {
                multiply_factor(term, second, PauliAxis::X);
            } else {
                multiply_factor(term, first, PauliAxis::Z);
                multiply_factor(term, second, PauliAxis::Z);
            }
            return;
        case OperationCode::Cz:
            if (source_first && generator == PauliAxis::X) {
                multiply_factor(term, first, PauliAxis::X);
                multiply_factor(term, second, PauliAxis::Z);
            } else if (source_first) {
                multiply_factor(term, first, PauliAxis::Z);
            } else if (generator == PauliAxis::X) {
                multiply_factor(term, first, PauliAxis::Z);
                multiply_factor(term, second, PauliAxis::X);
            } else {
                multiply_factor(term, second, PauliAxis::Z);
            }
            return;
        case OperationCode::Swap:
            multiply_factor(
                term,
                source_first ? second : first,
                generator);
            return;
        default:
            throw QStateError("Pauli two-qubit mapping received an unsupported gate");
    }
}

void append_mapped_axis(
    PauliTerm& term,
    OperationCode code,
    QubitId first,
    QubitId second,
    bool source_first,
    PauliAxis axis) {
    switch (axis) {
        case PauliAxis::I:
            return;
        case PauliAxis::X:
            append_generator(term, code, first, second, source_first, PauliAxis::X);
            return;
        case PauliAxis::Z:
            append_generator(term, code, first, second, source_first, PauliAxis::Z);
            return;
        case PauliAxis::Y:
            term.coefficient *= QI;
            append_generator(term, code, first, second, source_first, PauliAxis::X);
            append_generator(term, code, first, second, source_first, PauliAxis::Z);
            return;
    }
}

void map_two(PauliTerm& term, OperationCode code, QubitId first, QubitId second) {
    const PauliAxis first_axis = axis_at(term, first);
    const PauliAxis second_axis = axis_at(term, second);
    set_axis(term, first, PauliAxis::I);
    set_axis(term, second, PauliAxis::I);

    PauliTerm mapped;
    append_mapped_axis(mapped, code, first, second, true, first_axis);
    append_mapped_axis(mapped, code, first, second, false, second_axis);
    multiply_word(term, mapped);
}

[[nodiscard]] std::vector<PauliTerm> rotate_term(
    const PauliTerm& term,
    QubitId qubit,
    PauliAxis generator,
    double theta) {
    const PauliAxis current = axis_at(term, qubit);
    if (current == PauliAxis::I || current == generator) {
        return {term};
    }

    PauliTerm first = term;
    first.coefficient *= std::cos(theta);

    PauliTerm second = term;
    const AxisProduct product = multiply_axes(generator, current);
    set_axis(second, qubit, product.axis);
    second.coefficient *= std::sin(theta);
    second.coefficient *= QI;
    second.coefficient *= product.phase;
    return {std::move(first), std::move(second)};
}

[[nodiscard]] QComplex phase_i(std::size_t power) noexcept {
    switch (power & 3U) {
        case 0U:
            return {1.0, 0.0};
        case 1U:
            return QI;
        case 2U:
            return {-1.0, 0.0};
        default:
            return -QI;
    }
}

[[nodiscard]] QComplex component_expectation(
    const QComponentReadView& view,
    std::span<const LocalFactor> factors) {
    if (factors.empty()) {
        return {1.0, 0.0};
    }

    if (view.kind == ComponentKind::Cell) {
        if (view.cell == nullptr || factors.size() != 1U || factors.front().position != 0U) {
            throw QStateError("Pauli cell read view is invalid");
        }
        switch (factors.front().axis) {
            case PauliAxis::I:
                return {1.0, 0.0};
            case PauliAxis::X:
                return {view.cell->x, 0.0};
            case PauliAxis::Y:
                return {view.cell->y, 0.0};
            case PauliAxis::Z:
                return {view.cell->z, 0.0};
        }
    }

    BasisIndex flip_mask = 0U;
    BasisIndex z_mask = 0U;
    std::size_t y_count = 0U;
    for (const LocalFactor& factor : factors) {
        const BasisIndex bit = BasisIndex{1} << factor.position;
        if (factor.axis == PauliAxis::X || factor.axis == PauliAxis::Y) {
            flip_mask |= bit;
        }
        if (factor.axis == PauliAxis::Z || factor.axis == PauliAxis::Y) {
            z_mask |= bit;
        }
        if (factor.axis == PauliAxis::Y) {
            ++y_count;
        }
    }
    const QComplex base_phase = phase_i(y_count);

    QComplex result;
    const auto phase_for = [&](BasisIndex basis) {
        return (std::popcount(basis & z_mask) & 1U) != 0U ? -base_phase : base_phase;
    };

    if (view.kind == ComponentKind::Dense) {
        if (view.dense.size() != static_cast<std::size_t>(view.dimension)) {
            throw QStateError("Pauli dense read view is invalid");
        }
        for (BasisIndex basis = 0U; basis < view.dimension; ++basis) {
            const BasisIndex target = basis ^ flip_mask;
            result += view.dense[static_cast<std::size_t>(target)].conjugate() *
                      phase_for(basis) *
                      view.dense[static_cast<std::size_t>(basis)];
        }
        return result;
    }

    if (view.kind != ComponentKind::Sparse) {
        throw QStateError("Pauli read view has an unknown component kind");
    }
    for (const auto& [basis, source] : view.sparse) {
        const BasisIndex target = basis ^ flip_mask;
        const auto position = std::lower_bound(
            view.sparse.begin(),
            view.sparse.end(),
            target,
            [](const AmplitudeStore::SparseEntry& entry, BasisIndex value) {
                return entry.first < value;
            });
        if (position == view.sparse.end() || position->first != target) {
            continue;
        }
        result += position->second.conjugate() * phase_for(basis) * source;
    }
    return result;
}

}  // namespace

const char* pauli_axis_name(PauliAxis axis) noexcept {
    switch (axis) {
        case PauliAxis::I:
            return "I";
        case PauliAxis::X:
            return "X";
        case PauliAxis::Y:
            return "Y";
        case PauliAxis::Z:
            return "Z";
    }
    return "unknown";
}

PauliObservable::PauliObservable(
    std::size_t qubit_count,
    PauliPropagationConfig config)
    : qubit_count_(qubit_count), config_(config) {
    if (qubit_count_ == 0U) {
        throw QStateError("Pauli observable requires at least one qubit");
    }
    if (config_.max_terms == 0U) {
        throw QStateError("Pauli observable max_terms must be positive");
    }
}

std::size_t PauliObservable::support_size() const {
    std::vector<QubitId> support;
    for (const PauliTerm& term : terms_) {
        for (const PauliFactor& factor : term.factors) {
            support.push_back(factor.qubit);
        }
    }
    std::sort(support.begin(), support.end());
    support.erase(std::unique(support.begin(), support.end()), support.end());
    return support.size();
}

void PauliObservable::add_term(
    QComplex coefficient,
    std::span<const PauliFactor> factors) {
    if (!std::isfinite(coefficient.re) || !std::isfinite(coefficient.im)) {
        throw QStateError("Pauli coefficient must be finite");
    }

    PauliTerm term;
    term.coefficient = coefficient;
    for (const PauliFactor& factor : factors) {
        if (factor.qubit >= qubit_count_) {
            throw QStateError("Pauli factor qubit is outside the observable");
        }
        if (static_cast<std::uint8_t>(factor.axis) >
            static_cast<std::uint8_t>(PauliAxis::Z)) {
            throw QStateError("Pauli factor has an invalid axis");
        }
        multiply_factor(term, factor.qubit, factor.axis);
    }
    if (!zero(term.coefficient)) {
        terms_.push_back(std::move(term));
    }
    normalize_terms();
}

void PauliObservable::normalize_terms() {
    std::stable_sort(terms_.begin(), terms_.end(), [](const PauliTerm& left, const PauliTerm& right) {
        return factors_less(left.factors, right.factors);
    });

    std::vector<PauliTerm> merged;
    merged.reserve(terms_.size());
    for (PauliTerm& term : terms_) {
        if (!merged.empty() && same_factors(merged.back().factors, term.factors)) {
            merged.back().coefficient += term.coefficient;
        } else {
            merged.push_back(std::move(term));
        }
    }
    merged.erase(
        std::remove_if(merged.begin(), merged.end(), [](const PauliTerm& term) {
            return zero(term.coefficient);
        }),
        merged.end());
    if (merged.size() > config_.max_terms) {
        throw QStateError("Pauli propagation exceeded max_terms");
    }
    terms_ = std::move(merged);
}

void PauliObservable::apply_backward(const Operation& operation) {
    if (operation.first >= qubit_count_) {
        throw QStateError("Pauli operation qubit is outside the observable");
    }
    if ((operation.code == OperationCode::Cnot ||
         operation.code == OperationCode::Cz ||
         operation.code == OperationCode::Swap) &&
        (operation.second >= qubit_count_ || operation.first == operation.second)) {
        throw QStateError("Pauli two-qubit operation has invalid qubits");
    }

    switch (operation.code) {
        case OperationCode::X:
            for (PauliTerm& term : terms_) {
                map_x(term, operation.first);
            }
            break;
        case OperationCode::Y:
            for (PauliTerm& term : terms_) {
                map_y(term, operation.first);
            }
            break;
        case OperationCode::Z:
            for (PauliTerm& term : terms_) {
                map_z(term, operation.first);
            }
            break;
        case OperationCode::H:
            for (PauliTerm& term : terms_) {
                map_h(term, operation.first);
            }
            break;
        case OperationCode::S:
            for (PauliTerm& term : terms_) {
                map_s(term, operation.first);
            }
            break;
        case OperationCode::Sdg:
            for (PauliTerm& term : terms_) {
                map_sdg(term, operation.first);
            }
            break;
        case OperationCode::T:
        case OperationCode::Tdg:
        case OperationCode::Rx:
        case OperationCode::Ry:
        case OperationCode::Rz: {
            PauliAxis generator = PauliAxis::Z;
            double theta = operation.parameter;
            if (operation.code == OperationCode::T) {
                theta = std::numbers::pi / 4.0;
            } else if (operation.code == OperationCode::Tdg) {
                theta = -std::numbers::pi / 4.0;
            } else if (operation.code == OperationCode::Rx) {
                generator = PauliAxis::X;
            } else if (operation.code == OperationCode::Ry) {
                generator = PauliAxis::Y;
            }
            if (!std::isfinite(theta)) {
                throw QStateError("Pauli rotation angle must be finite");
            }

            std::vector<PauliTerm> expanded;
            const std::size_t reserve_count = terms_.size() > config_.max_terms
                ? config_.max_terms
                : std::min(config_.max_terms * 2U, terms_.size() * 2U);
            expanded.reserve(reserve_count);
            for (const PauliTerm& term : terms_) {
                auto rotated = rotate_term(term, operation.first, generator, theta);
                expanded.insert(
                    expanded.end(),
                    std::make_move_iterator(rotated.begin()),
                    std::make_move_iterator(rotated.end()));
            }
            terms_ = std::move(expanded);
            break;
        }
        case OperationCode::Cnot:
        case OperationCode::Cz:
        case OperationCode::Swap:
            for (PauliTerm& term : terms_) {
                map_two(term, operation.code, operation.first, operation.second);
            }
            break;
        case OperationCode::BitFlipTrajectory:
        case OperationCode::PhaseFlipTrajectory:
        case OperationCode::DepolarizingTrajectory:
        case OperationCode::AmplitudeDampingTrajectory:
            throw QStateError("Pauli propagation does not accept trajectory noise operations");
        default:
            throw QStateError("Pauli propagation received an unknown opcode");
    }
    normalize_terms();
}

void PauliObservable::propagate_backward(std::span<const Operation> operations) {
    for (auto position = operations.rbegin(); position != operations.rend(); ++position) {
        apply_backward(*position);
    }
}

PauliObservable PauliObservable::propagated_backward(
    std::span<const Operation> operations) const {
    PauliObservable result = *this;
    result.propagate_backward(operations);
    return result;
}

QComplex PauliObservable::expectation(const QRegister& state) const {
    if (state.qubit_count() != qubit_count_) {
        throw QStateError("Pauli observable width does not match QRegister");
    }

    QComplex total;
    for (const PauliTerm& term : terms_) {
        QComplex value{1.0, 0.0};
        std::vector<LocalGroup> groups;
        groups.reserve(term.factors.size());

        for (const PauliFactor& factor : term.factors) {
            const QComponentReadView view = state.component_read_view(factor.qubit);
            const auto local = std::find(view.qubits.begin(), view.qubits.end(), factor.qubit);
            if (local == view.qubits.end()) {
                throw QStateError("Pauli factor was not found in its component");
            }
            const std::size_t position =
                static_cast<std::size_t>(local - view.qubits.begin());

            auto group = std::find_if(groups.begin(), groups.end(), [&](const LocalGroup& candidate) {
                return candidate.view.qubits.data() == view.qubits.data() &&
                       candidate.view.qubits.size() == view.qubits.size();
            });
            if (group == groups.end()) {
                LocalGroup next;
                next.view = view;
                next.factors.push_back(LocalFactor{position, factor.axis});
                groups.push_back(std::move(next));
            } else {
                group->factors.push_back(LocalFactor{position, factor.axis});
            }
        }

        for (const LocalGroup& group : groups) {
            value *= component_expectation(group.view, group.factors);
        }
        total += term.coefficient * value;
    }
    return total;
}

bool PauliObservable::validate(std::string* reason) const {
    const auto fail = [reason](const char* message) {
        if (reason != nullptr) {
            *reason = message;
        }
        return false;
    };

    if (qubit_count_ == 0U) {
        return fail("observable has zero width");
    }
    if (config_.max_terms == 0U || terms_.size() > config_.max_terms) {
        return fail("observable exceeds its term limit");
    }
    for (const PauliTerm& term : terms_) {
        if (!std::isfinite(term.coefficient.re) || !std::isfinite(term.coefficient.im)) {
            return fail("observable contains a non-finite coefficient");
        }
        QubitId previous = 0U;
        bool first = true;
        for (const PauliFactor& factor : term.factors) {
            if (factor.qubit >= qubit_count_) {
                return fail("observable contains an out-of-range qubit");
            }
            if (factor.axis == PauliAxis::I ||
                static_cast<std::uint8_t>(factor.axis) >
                    static_cast<std::uint8_t>(PauliAxis::Z)) {
                return fail("observable contains an invalid factor");
            }
            if (!first && factor.qubit <= previous) {
                return fail("observable factors are not canonical");
            }
            previous = factor.qubit;
            first = false;
        }
    }
    if (reason != nullptr) {
        reason->clear();
    }
    return true;
}

}  // namespace qubit
