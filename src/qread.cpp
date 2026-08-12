#include "qubit/qstate.hpp"

#include <algorithm>
#include <bit>
#include <limits>

namespace qubit {
namespace {

struct ReadSplitMix64 {
    std::uint64_t state;

    [[nodiscard]] std::uint64_t next() noexcept {
        std::uint64_t value = (state += 0x9E3779B97F4A7C15ULL);
        value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
        value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
        return value ^ (value >> 31U);
    }

    [[nodiscard]] double unit() noexcept {
        return static_cast<double>(next() >> 11U) *
               (1.0 / 9007199254740992.0);
    }
};

}  // namespace

QComponentReadView QRegister::component_read_view(QubitId qubit) const {
    const StateComponent& component = components_[component_index(qubit)];
    QComponentReadView view;
    view.qubits = component.qubits;
    if (component.is_cell()) {
        view.kind = ComponentKind::Cell;
        view.cell = &std::get<BlochCell>(component.state);
        view.dimension = 2U;
        return view;
    }

    const AmplitudeStore& store = std::get<AmplitudeStore>(component.state);
    view.dimension = store.dimension_;
    if (store.mode_ == StorageMode::Dense) {
        view.kind = ComponentKind::Dense;
        view.dense = store.dense_;
    } else {
        view.kind = ComponentKind::Sparse;
        view.sparse = store.sparse_;
    }
    return view;
}

std::vector<QComponentReadView> QRegister::component_read_views() const {
    const auto ordered = ordered_component_indices();
    std::vector<QComponentReadView> result;
    result.reserve(ordered.size());
    for (std::size_t index : ordered) {
        const StateComponent& component = components_[index];
        QComponentReadView view;
        view.qubits = component.qubits;
        if (component.is_cell()) {
            view.kind = ComponentKind::Cell;
            view.cell = &std::get<BlochCell>(component.state);
            view.dimension = 2U;
        } else {
            const AmplitudeStore& store = std::get<AmplitudeStore>(component.state);
            view.dimension = store.dimension_;
            if (store.mode_ == StorageMode::Dense) {
                view.kind = ComponentKind::Dense;
                view.dense = store.dense_;
            } else {
                view.kind = ComponentKind::Sparse;
                view.sparse = store.sparse_;
            }
        }
        result.push_back(view);
    }
    return result;
}

void QRegister::probabilities_one_into(std::span<double> output) const {
    if (output.size() != qubit_count_) {
        throw QStateError(
            "Probability output size does not match QRegister qubit count");
    }
    std::fill(output.begin(), output.end(), 0.0);
    for (const StateComponent& component : components_) {
        if (component.is_cell()) {
            output[component.qubits.front()] =
                std::get<BlochCell>(component.state).probability_one();
            continue;
        }

        const AmplitudeStore& store = std::get<AmplitudeStore>(component.state);
        std::vector<long double> local(component.qubits.size(), 0.0L);
        const auto accumulate = [&local](BasisIndex basis, const QComplex& amplitude) {
            const long double weight =
                static_cast<long double>(amplitude.norm2());
            BasisIndex remaining = basis;
            while (remaining != 0U) {
                const std::size_t position =
                    static_cast<std::size_t>(std::countr_zero(remaining));
                local[position] += weight;
                remaining &= remaining - 1U;
            }
        };

        if (store.mode_ == StorageMode::Sparse) {
            for (const auto& [basis, amplitude] : store.sparse_) {
                accumulate(basis, amplitude);
            }
        } else {
            for (std::size_t position = 0; position < component.qubits.size(); ++position) {
                const std::size_t half_block = std::size_t{1} << position;
                const std::size_t block = half_block << 1U;
                long double probability = 0.0L;
                for (std::size_t base = 0; base < store.dense_.size(); base += block) {
                    const std::size_t end = base + block;
                    for (std::size_t basis = base + half_block; basis < end; ++basis) {
                        probability +=
                            static_cast<long double>(store.dense_[basis].norm2());
                    }
                }
                local[position] = probability;
            }
        }

        for (std::size_t position = 0; position < component.qubits.size(); ++position) {
            output[component.qubits[position]] =
                std::clamp(static_cast<double>(local[position]), 0.0, 1.0);
        }
    }
}

double QRegister::marginal_probability(
    std::span<const QubitId> qubits,
    std::span<const std::uint8_t> bits) const {
    if (qubits.size() != bits.size()) {
        throw QStateError("Marginal query qubit and bit counts do not match");
    }
    if (qubits.empty()) {
        return 1.0;
    }

    std::vector<std::int8_t> requested(qubit_count_, -1);
    for (std::size_t index = 0U; index < qubits.size(); ++index) {
        const QubitId qubit = qubits[index];
        if (static_cast<std::size_t>(qubit) >= qubit_count_) {
            throw QStateError("Marginal query qubit is out of range");
        }
        if (bits[index] > 1U) {
            throw QStateError("Marginal query bits must be zero or one");
        }
        if (requested[qubit] >= 0) {
            throw QStateError("Marginal query contains duplicate qubits");
        }
        requested[qubit] = static_cast<std::int8_t>(bits[index]);
    }

    long double probability = 1.0L;
    for (const StateComponent& component : components_) {
        if (component.is_cell()) {
            const QubitId qubit = component.qubits.front();
            const std::int8_t bit = requested[qubit];
            if (bit < 0) {
                continue;
            }
            const double one = std::get<BlochCell>(component.state).probability_one();
            probability *= static_cast<long double>(bit != 0 ? one : 1.0 - one);
            continue;
        }

        BasisIndex mask = 0U;
        BasisIndex expected = 0U;
        for (std::size_t position = 0U; position < component.qubits.size(); ++position) {
            const std::int8_t bit = requested[component.qubits[position]];
            if (bit < 0) {
                continue;
            }
            const BasisIndex local_bit = BasisIndex{1} << position;
            mask |= local_bit;
            if (bit != 0) {
                expected |= local_bit;
            }
        }
        if (mask == 0U) {
            continue;
        }

        const AmplitudeStore& store = std::get<AmplitudeStore>(component.state);
        long double local_probability = 0.0L;
        if (store.mode_ == StorageMode::Sparse) {
            for (const auto& [basis, amplitude] : store.sparse_) {
                if ((basis & mask) == expected) {
                    local_probability += static_cast<long double>(amplitude.norm2());
                }
            }
        } else {
            for (BasisIndex basis = 0U; basis < store.dimension_; ++basis) {
                if ((basis & mask) == expected) {
                    local_probability += static_cast<long double>(
                        store.dense_[static_cast<std::size_t>(basis)].norm2());
                }
            }
        }
        probability *= std::clamp(local_probability, 0.0L, 1.0L);
    }

    return std::clamp(static_cast<double>(probability), 0.0, 1.0);
}

void QRegister::measure_all_into(
    std::uint64_t seed,
    std::span<int> output) {
    if (output.size() != qubit_count_) {
        throw QStateError(
            "Measurement output size does not match QRegister qubit count");
    }
    ReadSplitMix64 generator{seed};
    for (std::size_t qubit = 0; qubit < qubit_count_; ++qubit) {
        output[qubit] = measure(
            static_cast<QubitId>(qubit), generator.unit());
    }
}

}  // namespace qubit
