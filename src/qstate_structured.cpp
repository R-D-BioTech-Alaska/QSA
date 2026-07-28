#include "qubit/qstate.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>

namespace qubit {
namespace {

enum class SparseXorPhase : std::uint8_t {
    None,
    PauliY,
};

void permute_sparse_xor_bit(
    std::vector<AmplitudeStore::SparseEntry>& entries,
    std::size_t bit_position,
    SparseXorPhase phase_mode) {
    if (entries.empty()) {
        return;
    }
    const BasisIndex mask = BasisIndex{1} << bit_position;
    const std::size_t upper_shift = bit_position + 1U;
    auto block_begin = entries.begin();
    while (block_begin != entries.end()) {
        const BasisIndex block_key = block_begin->first >> upper_shift;
        auto block_end = block_begin;
        while (block_end != entries.end() &&
               (block_end->first >> upper_shift) == block_key) {
            ++block_end;
        }
        const auto split = std::find_if(
            block_begin,
            block_end,
            [mask](const AmplitudeStore::SparseEntry& entry) {
                return (entry.first & mask) != 0U;
            });
        std::rotate(block_begin, split, block_end);
        for (auto iterator = block_begin; iterator != block_end; ++iterator) {
            const bool original_one = (iterator->first & mask) != 0U;
            if (phase_mode == SparseXorPhase::PauliY) {
                iterator->second = original_one
                    ? -QI * iterator->second
                    : QI * iterator->second;
            }
            iterator->first ^= mask;
        }
        block_begin = block_end;
    }
}

}  // namespace

void AmplitudeStore::apply_x_structured(std::size_t bit_position) {
    const std::size_t bits = static_cast<std::size_t>(std::countr_zero(dimension_));
    if (bit_position >= bits) {
        throw QStateError("X gate bit position is out of range");
    }
    if (mode_ == StorageMode::Dense) {
        apply_x(bit_position);
        return;
    }
    permute_sparse_xor_bit(sparse_, bit_position, SparseXorPhase::None);
}

void AmplitudeStore::apply_y_structured(std::size_t bit_position) {
    const std::size_t bits = static_cast<std::size_t>(std::countr_zero(dimension_));
    if (bit_position >= bits) {
        throw QStateError("Y gate bit position is out of range");
    }
    if (mode_ == StorageMode::Dense) {
        apply_y(bit_position);
        return;
    }
    permute_sparse_xor_bit(sparse_, bit_position, SparseXorPhase::PauliY);
}

void QRegister::apply_x_structured(QubitId qubit) {
    const std::size_t index = component_index(qubit);
    if (components_[index].is_cell()) {
        std::get<BlochCell>(components_[index].state).apply_x();
    } else {
        std::get<AmplitudeStore>(components_[index].state)
            .apply_x_structured(local_position(components_[index], qubit));
    }
}

void QRegister::apply_y_structured(QubitId qubit) {
    const std::size_t index = component_index(qubit);
    if (components_[index].is_cell()) {
        std::get<BlochCell>(components_[index].state).apply_y();
    } else {
        std::get<AmplitudeStore>(components_[index].state)
            .apply_y_structured(local_position(components_[index], qubit));
    }
}

void QRegister::apply_diagonal_structured(
    std::span<const QDiagonalPhase> phases) {
    using LocalPhase = std::pair<std::size_t, std::array<QComplex, 2>>;
    const auto validate_phase = [this](const QDiagonalPhase& phase) {
        validate_qubit(phase.qubit);
        const double zero_norm = phase.zero.norm2();
        const double one_norm = phase.one.norm2();
        if (!std::isfinite(zero_norm) || !std::isfinite(one_norm) ||
            std::abs(zero_norm - 1.0) > 1e-10 ||
            std::abs(one_norm - 1.0) > 1e-10) {
            throw QStateError(
                "Diagonal phase entries must be finite unit-magnitude values");
        }
    };
    const auto apply_grouped = [this](
                                   const std::vector<std::vector<LocalPhase>>& grouped) {
        for (std::size_t index = 0; index < grouped.size(); ++index) {
            if (grouped[index].empty()) {
                continue;
            }
            AmplitudeStore& store =
                std::get<AmplitudeStore>(components_[index].state);
            const auto apply_to_value = [&grouped, index](
                                            BasisIndex basis,
                                            QComplex& value) {
                QComplex coefficient{1.0, 0.0};
                for (const auto& [position, values] : grouped[index]) {
                    coefficient *= values[(basis >> position) & 1U];
                }
                value *= coefficient;
            };
            if (store.mode_ == StorageMode::Sparse) {
                for (auto& [basis, value] : store.sparse_) {
                    apply_to_value(basis, value);
                }
            } else {
                for (std::size_t basis = 0; basis < store.dense_.size(); ++basis) {
                    apply_to_value(
                        static_cast<BasisIndex>(basis), store.dense_[basis]);
                }
            }
        }
    };

    if (components_.size() <= 64U) {
        std::vector<std::vector<LocalPhase>> grouped(components_.size());
        for (const QDiagonalPhase& phase : phases) {
            validate_phase(phase);
            const std::size_t index = component_index(phase.qubit);
            StateComponent& component = components_[index];
            if (component.is_cell()) {
                QMatrix2 matrix{};
                matrix.values[0] = phase.zero;
                matrix.values[3] = phase.one;
                apply_cell_matrix(std::get<BlochCell>(component.state), matrix);
            } else {
                grouped[index].push_back(LocalPhase{
                    local_position(component, phase.qubit),
                    std::array<QComplex, 2>{phase.zero, phase.one},
                });
            }
        }
        apply_grouped(grouped);
        return;
    }

    struct PendingPhase {
        std::size_t component_index{0};
        std::size_t local_position{0};
        QComplex zero{1.0, 0.0};
        QComplex one{1.0, 0.0};
    };

    std::vector<PendingPhase> pending;
    bool pending_is_grouped = true;
    for (const QDiagonalPhase& phase : phases) {
        validate_phase(phase);
        const std::size_t index = component_index(phase.qubit);
        StateComponent& component = components_[index];
        if (component.is_cell()) {
            QMatrix2 matrix{};
            matrix.values[0] = phase.zero;
            matrix.values[3] = phase.one;
            apply_cell_matrix(std::get<BlochCell>(component.state), matrix);
            continue;
        }
        if (pending.empty()) {
            pending.reserve(phases.size());
        }
        if (!pending.empty() && index < pending.back().component_index) {
            pending_is_grouped = false;
        }
        pending.push_back(PendingPhase{
            index,
            local_position(component, phase.qubit),
            phase.zero,
            phase.one,
        });
    }

    if (pending.empty()) {
        return;
    }

    const std::size_t half_component_count =
        components_.size() - components_.size() / 2U;
    if (pending.size() >= half_component_count) {
        std::vector<std::vector<LocalPhase>> grouped(components_.size());
        for (const PendingPhase& phase : pending) {
            grouped[phase.component_index].push_back(LocalPhase{
                phase.local_position,
                std::array<QComplex, 2>{phase.zero, phase.one},
            });
        }
        apply_grouped(grouped);
        return;
    }

    if (!pending_is_grouped) {
        std::stable_sort(
            pending.begin(),
            pending.end(),
            [](const PendingPhase& left, const PendingPhase& right) {
                return left.component_index < right.component_index;
            });
    }

    std::size_t first = 0U;
    while (first < pending.size()) {
        const std::size_t component_index_value = pending[first].component_index;
        std::size_t last = first + 1U;
        while (last < pending.size() &&
               pending[last].component_index == component_index_value) {
            ++last;
        }

        AmplitudeStore& store =
            std::get<AmplitudeStore>(components_[component_index_value].state);
        const auto apply_to_value = [&pending, first, last](
                                        BasisIndex basis,
                                        QComplex& value) {
            QComplex coefficient{1.0, 0.0};
            for (std::size_t phase_index = first;
                 phase_index < last;
                 ++phase_index) {
                const PendingPhase& phase = pending[phase_index];
                coefficient *= ((basis >> phase.local_position) & 1U) == 0U
                    ? phase.zero
                    : phase.one;
            }
            value *= coefficient;
        };
        if (store.mode_ == StorageMode::Sparse) {
            for (auto& [basis, value] : store.sparse_) {
                apply_to_value(basis, value);
            }
        } else {
            for (std::size_t basis = 0; basis < store.dense_.size(); ++basis) {
                apply_to_value(
                    static_cast<BasisIndex>(basis), store.dense_[basis]);
            }
        }
        first = last;
    }
}

}  // namespace qubit
