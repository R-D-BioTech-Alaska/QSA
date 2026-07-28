#include "qubit/qstate.hpp"

#include <algorithm>
#include <bit>
#include <limits>
#include <numeric>

namespace qubit {
namespace {

struct StructureSplitMix64 {
    std::uint64_t state;

    [[nodiscard]] std::uint64_t next() noexcept {
        std::uint64_t value = (state += 0x9E3779B97F4A7C15ULL);
        value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
        value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
        return value ^ (value >> 31U);
    }

    [[nodiscard]] double unit() noexcept {
        return static_cast<double>(next() >> 11U) * (1.0 / 9007199254740992.0);
    }
};

[[nodiscard]] BasisIndex remap_structure_index(
    BasisIndex index,
    std::span<const std::size_t> local_to_target) noexcept {
    BasisIndex result = 0U;
    for (std::size_t local = 0; local < local_to_target.size(); ++local) {
        if (((index >> local) & 1U) != 0U) {
            result |= BasisIndex{1} << local_to_target[local];
        }
    }
    return result;
}

}  // namespace

std::optional<int> QRegister::exact_z_basis(const StateComponent& component) {
    if (!component.is_cell()) {
        return std::nullopt;
    }
    const BlochCell& cell = std::get<BlochCell>(component.state);
    if (cell.x != 0.0 || cell.y != 0.0) {
        return std::nullopt;
    }
    if (cell.z == 1.0) {
        return 0;
    }
    if (cell.z == -1.0) {
        return 1;
    }
    return std::nullopt;
}

std::optional<int> QRegister::exact_x_eigenvalue(const StateComponent& component) {
    if (!component.is_cell()) {
        return std::nullopt;
    }
    const BlochCell& cell = std::get<BlochCell>(component.state);
    if (cell.y != 0.0 || cell.z != 0.0) {
        return std::nullopt;
    }
    if (cell.x == 1.0) {
        return 1;
    }
    if (cell.x == -1.0) {
        return -1;
    }
    return std::nullopt;
}

StateComponent QRegister::relabel_component(
    const StateComponent& source,
    QubitId from,
    QubitId to) const {
    StateComponent result = source;
    const std::size_t old_position = local_position(source, from);
    result.qubits[old_position] = to;

    std::vector<std::size_t> order(result.qubits.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(), [&result](std::size_t lhs, std::size_t rhs) {
        return result.qubits[lhs] < result.qubits[rhs];
    });

    std::vector<std::size_t> old_to_new(order.size());
    std::vector<QubitId> sorted_qubits(order.size());
    for (std::size_t next = 0; next < order.size(); ++next) {
        old_to_new[order[next]] = next;
        sorted_qubits[next] = result.qubits[order[next]];
    }
    result.qubits = std::move(sorted_qubits);

    if (result.is_cell() || std::is_sorted(order.begin(), order.end())) {
        return result;
    }

    AmplitudeStore& store = std::get<AmplitudeStore>(result.state);
    if (store.mode_ == StorageMode::Dense) {
        std::vector<QComplex> permuted(store.dense_.size());
        for (BasisIndex index = 0; index < store.dimension_; ++index) {
            permuted[static_cast<std::size_t>(remap_structure_index(index, old_to_new))] =
                store.dense_[static_cast<std::size_t>(index)];
        }
        store.dense_ = std::move(permuted);
    } else {
        for (auto& entry : store.sparse_) {
            entry.first = remap_structure_index(entry.first, old_to_new);
        }
        store.sort_sparse();
    }
    return result;
}

void QRegister::swap_disconnected_qubits(
    std::size_t first_component,
    QubitId first,
    std::size_t second_component,
    QubitId second) {
    if (first_component == second_component || first_component >= components_.size() ||
        second_component >= components_.size()) {
        throw QStateError("Disconnected SWAP received invalid components");
    }

    StateComponent first_next = relabel_component(components_[first_component], first, second);
    StateComponent second_next = relabel_component(components_[second_component], second, first);
    components_[first_component] = std::move(first_next);
    components_[second_component] = std::move(second_next);

    for (QubitId qubit : components_[first_component].qubits) {
        qubit_component_[qubit] = first_component;
    }
    for (QubitId qubit : components_[second_component].qubits) {
        qubit_component_[qubit] = second_component;
    }
}

std::size_t QRegister::merge_components_cached(std::size_t first, std::size_t second) {
    if (first == second) {
        return first;
    }
    if (first >= components_.size() || second >= components_.size()) {
        throw QStateError("Cannot merge an invalid state component");
    }

    const StateComponent& left = components_[first];
    const StateComponent& right = components_[second];
    std::vector<QubitId> merged_qubits = left.qubits;
    merged_qubits.insert(merged_qubits.end(), right.qubits.begin(), right.qubits.end());
    std::sort(merged_qubits.begin(), merged_qubits.end());
    if (merged_qubits.size() > config_.max_component_qubits || merged_qubits.size() >= 63U) {
        throw QStateError("Entangled component exceeds configured qubit limit");
    }

    std::vector<std::size_t> left_positions(left.qubits.size());
    std::vector<std::size_t> right_positions(right.qubits.size());
    std::size_t left_cursor = 0U;
    std::size_t right_cursor = 0U;
    for (std::size_t merged_position = 0; merged_position < merged_qubits.size();
         ++merged_position) {
        const QubitId qubit = merged_qubits[merged_position];
        if (left_cursor < left.qubits.size() && left.qubits[left_cursor] == qubit) {
            left_positions[left_cursor++] = merged_position;
        } else if (right_cursor < right.qubits.size() && right.qubits[right_cursor] == qubit) {
            right_positions[right_cursor++] = merged_position;
        } else {
            throw QStateError("Internal error: component merge mapping is inconsistent");
        }
    }

    const auto component_entries = [this](const StateComponent& component) {
        if (component.is_cell()) {
            const auto amplitudes =
                std::get<BlochCell>(component.state).amplitudes(config_.epsilon);
            std::vector<AmplitudeStore::SparseEntry> result;
            if (amplitudes[0].norm2() > config_.epsilon * config_.epsilon) {
                result.emplace_back(0U, amplitudes[0]);
            }
            if (amplitudes[1].norm2() > config_.epsilon * config_.epsilon) {
                result.emplace_back(1U, amplitudes[1]);
            }
            return result;
        }
        return std::get<AmplitudeStore>(component.state).entries(config_.epsilon);
    };

    const auto left_entries = component_entries(left);
    const auto right_entries = component_entries(right);
    const long double product_size = static_cast<long double>(left_entries.size()) *
                                     static_cast<long double>(right_entries.size());
    const BasisIndex dimension = BasisIndex{1} << merged_qubits.size();
    const long double practical_limit = static_cast<long double>(
        std::max<std::uint64_t>(config_.max_dense_amplitudes, config_.max_sparse_entries));
    if (product_size > practical_limit && dimension > config_.max_dense_amplitudes) {
        throw QStateError("Component merge would exceed configured state capacity");
    }

    const bool cache_right_remap = product_size >= 32.0L;
    std::vector<AmplitudeStore::SparseEntry> remapped_right_entries;
    if (cache_right_remap) {
        remapped_right_entries.reserve(right_entries.size());
        for (const auto& [index, value] : right_entries) {
            remapped_right_entries.emplace_back(
                remap_structure_index(index, right_positions), value);
        }
    }

    std::vector<AmplitudeStore::SparseEntry> merged_entries;
    merged_entries.reserve(static_cast<std::size_t>(product_size));
    for (const auto& [left_index, left_value] : left_entries) {
        const BasisIndex remapped_left = remap_structure_index(left_index, left_positions);
        if (cache_right_remap) {
            for (const auto& [right_index, right_value] : remapped_right_entries) {
                merged_entries.emplace_back(
                    remapped_left | right_index, left_value * right_value);
            }
        } else {
            for (const auto& [right_index, right_value] : right_entries) {
                const BasisIndex remapped_right =
                    remap_structure_index(right_index, right_positions);
                merged_entries.emplace_back(
                    remapped_left | remapped_right, left_value * right_value);
            }
        }
    }

    StateComponent merged{
        std::move(merged_qubits),
        AmplitudeStore::from_entries(dimension, std::move(merged_entries), config_),
    };

    const std::size_t high = std::max(first, second);
    const std::size_t low = std::min(first, second);
    remove_component(high);
    remove_component(low);
    return append_component(std::move(merged));
}

void QRegister::apply_cnot_structured(QubitId control, QubitId target) {
    validate_qubit(control);
    validate_qubit(target);
    if (control == target) {
        throw QStateError("A two-qubit gate requires two distinct qubits");
    }

    std::size_t component = component_index(control);
    const std::size_t other = component_index(target);
    if (component == other) {
        apply_cnot(control, target);
        return;
    }

    if (const auto control_value = exact_z_basis(components_[component]);
        control_value.has_value()) {
        if (*control_value == 1) {
            apply_x(target);
        }
        return;
    }
    if (const auto target_eigenvalue = exact_x_eigenvalue(components_[other]);
        target_eigenvalue.has_value()) {
        if (*target_eigenvalue == -1) {
            apply_z(control);
        }
        return;
    }

    component = merge_components_cached(component, other);
    StateComponent& selected = components_[component];
    const std::size_t control_position = local_position(selected, control);
    const std::size_t target_position = local_position(selected, target);
    std::get<AmplitudeStore>(selected.state).apply_cnot(control_position, target_position);
    const std::array<QubitId, 2> candidates{control, target};
    compact_component_targets(component, candidates);
}

void QRegister::apply_cz_structured(QubitId first, QubitId second) {
    validate_qubit(first);
    validate_qubit(second);
    if (first == second) {
        throw QStateError("A two-qubit gate requires two distinct qubits");
    }

    std::size_t component = component_index(first);
    const std::size_t other = component_index(second);
    if (component == other) {
        apply_cz(first, second);
        return;
    }

    if (const auto first_value = exact_z_basis(components_[component]);
        first_value.has_value()) {
        if (*first_value == 1) {
            apply_z(second);
        }
        return;
    }
    if (const auto second_value = exact_z_basis(components_[other]);
        second_value.has_value()) {
        if (*second_value == 1) {
            apply_z(first);
        }
        return;
    }

    component = merge_components_cached(component, other);
    StateComponent& selected = components_[component];
    const std::size_t first_position = local_position(selected, first);
    const std::size_t second_position = local_position(selected, second);
    std::get<AmplitudeStore>(selected.state).apply_cz(first_position, second_position);
    const std::array<QubitId, 2> candidates{first, second};
    compact_component_targets(component, candidates);
}

void QRegister::apply_swap_structured(QubitId first, QubitId second) {
    validate_qubit(first);
    validate_qubit(second);
    if (first == second) {
        throw QStateError("A two-qubit gate requires two distinct qubits");
    }

    const std::size_t component = component_index(first);
    const std::size_t other = component_index(second);
    if (component == other) {
        apply_swap(first, second);
        return;
    }
    swap_disconnected_qubits(component, first, other, second);
}

std::vector<int> QRegister::measure_all_joint(std::uint64_t seed) {
    StructureSplitMix64 generator{seed};
    std::vector<int> result(qubit_count_, 0);
    const auto ordered = ordered_component_indices();

    for (std::size_t component_index_value : ordered) {
        const StateComponent& component = components_[component_index_value];
        BasisIndex selected_basis = 0U;
        const double sample = generator.unit();
        if (component.is_cell()) {
            selected_basis = sample < std::get<BlochCell>(component.state).probability_one()
                                 ? 1U
                                 : 0U;
        } else {
            const AmplitudeStore& store = std::get<AmplitudeStore>(component.state);
            long double cumulative = 0.0L;
            BasisIndex fallback = 0U;
            bool selected = false;
            if (store.mode_ == StorageMode::Sparse) {
                for (const auto& [basis, amplitude] : store.sparse_) {
                    fallback = basis;
                    cumulative += static_cast<long double>(amplitude.norm2());
                    if (static_cast<long double>(sample) < cumulative) {
                        selected_basis = basis;
                        selected = true;
                        break;
                    }
                }
            } else {
                for (std::size_t basis = 0; basis < store.dense_.size(); ++basis) {
                    const double probability = store.dense_[basis].norm2();
                    if (probability <= 0.0) {
                        continue;
                    }
                    fallback = static_cast<BasisIndex>(basis);
                    cumulative += static_cast<long double>(probability);
                    if (static_cast<long double>(sample) < cumulative) {
                        selected_basis = static_cast<BasisIndex>(basis);
                        selected = true;
                        break;
                    }
                }
            }
            if (!selected) {
                selected_basis = fallback;
            }
        }

        for (std::size_t position = 0; position < component.qubits.size(); ++position) {
            result[component.qubits[position]] =
                static_cast<int>((selected_basis >> position) & 1U);
        }
    }

    components_.clear();
    component_order_.clear();
    components_.reserve(qubit_count_);
    component_order_.reserve(qubit_count_);
    next_component_order_ = 0U;
    qubit_component_.assign(qubit_count_, 0U);
    for (std::size_t qubit = 0; qubit < qubit_count_; ++qubit) {
        const BlochCell cell = result[qubit] == 0
                                   ? BlochCell{0.0, 0.0, 1.0}
                                   : BlochCell{0.0, 0.0, -1.0};
        components_.push_back(StateComponent{{static_cast<QubitId>(qubit)}, cell});
        component_order_.push_back(static_cast<std::uint32_t>(next_component_order_++));
        qubit_component_[qubit] = qubit;
    }
    return result;
}

}  // namespace qubit
