#include "qubit/qsymmetry.hpp"

#include <bit>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

namespace qubit {
namespace {

struct ExactAmplitudeKey {
    std::uint64_t real;
    std::uint64_t imag;

    [[nodiscard]] bool operator==(const ExactAmplitudeKey&) const noexcept = default;
};

struct ExactAmplitudeKeyHash {
    [[nodiscard]] std::size_t operator()(const ExactAmplitudeKey& key) const noexcept {
        const std::uint64_t mixed = key.real ^
            (key.imag + 0x9E3779B97F4A7C15ULL + (key.real << 6U) + (key.real >> 2U));
        return static_cast<std::size_t>(mixed ^ (mixed >> 32U));
    }
};

[[nodiscard]] ExactAmplitudeKey amplitude_key(QComplex value) noexcept {
    if (value.re == 0.0) {
        value.re = 0.0;
    }
    if (value.im == 0.0) {
        value.im = 0.0;
    }
    return {
        std::bit_cast<std::uint64_t>(value.re),
        std::bit_cast<std::uint64_t>(value.im),
    };
}

struct AmplitudeClass {
    BasisIndex count{0U};
    QComplex amplitude{};
};

void add_class(
    std::vector<AmplitudeClass>& classes,
    std::unordered_map<ExactAmplitudeKey, std::size_t, ExactAmplitudeKeyHash>& indices,
    BasisIndex count,
    QComplex amplitude,
    std::size_t max_classes) {
    if (count == 0U) {
        return;
    }
    const ExactAmplitudeKey key = amplitude_key(amplitude);
    const auto iterator = indices.find(key);
    if (iterator != indices.end()) {
        AmplitudeClass& existing = classes[iterator->second];
        if (count > std::numeric_limits<BasisIndex>::max() - existing.count) {
            throw QStateError("Component symmetry class count overflow");
        }
        existing.count += count;
        return;
    }
    if (classes.size() >= max_classes) {
        throw QStateError("Component symmetry discovery exceeded the configured class limit");
    }
    indices.emplace(key, classes.size());
    classes.push_back(AmplitudeClass{count, amplitude});
}

[[nodiscard]] std::vector<AmplitudeClass> component_classes(
    const StateComponent& component,
    const QStateConfig& config,
    std::size_t max_classes) {
    std::vector<AmplitudeClass> result;
    std::unordered_map<ExactAmplitudeKey, std::size_t, ExactAmplitudeKeyHash> indices;
    if (component.is_cell()) {
        const auto amplitudes = std::get<BlochCell>(component.state).amplitudes(config.epsilon);
        add_class(result, indices, 1U, amplitudes[0], max_classes);
        add_class(result, indices, 1U, amplitudes[1], max_classes);
        return result;
    }

    const AmplitudeStore& store = std::get<AmplitudeStore>(component.state);
    const auto entries = store.entries(0.0);
    const BasisIndex nonzero = static_cast<BasisIndex>(entries.size());
    if (nonzero > store.dimension()) {
        throw QStateError("Component support exceeds its dimension");
    }
    const BasisIndex zero_count = store.dimension() - nonzero;
    add_class(result, indices, zero_count, QComplex{}, max_classes);
    for (const auto& entry : entries) {
        add_class(result, indices, 1U, entry.second, max_classes);
    }
    return result;
}

}  // namespace

SymmetryState SymmetryState::discover_components(
    const QRegister& state,
    std::size_t max_classes) {
    if (max_classes == 0U) {
        throw QStateError("Component symmetry discovery max_classes must be positive");
    }
    if (state.qubit_count_ == 0U || state.qubit_count_ >= 63U) {
        throw QStateError("Component symmetry discovery requires between 1 and 62 qubits");
    }

    std::vector<AmplitudeClass> combined{{1U, QComplex{1.0, 0.0}}};
    for (const StateComponent& component : state.components_) {
        const auto local = component_classes(component, state.config_, max_classes);
        std::vector<AmplitudeClass> next;
        std::unordered_map<ExactAmplitudeKey, std::size_t, ExactAmplitudeKeyHash> indices;
        const long double candidate_count =
            static_cast<long double>(combined.size()) * static_cast<long double>(local.size());
        next.reserve(static_cast<std::size_t>(
            std::min<long double>(candidate_count, static_cast<long double>(max_classes))));

        for (const AmplitudeClass& left : combined) {
            for (const AmplitudeClass& right : local) {
                if (right.count != 0U &&
                    left.count > std::numeric_limits<BasisIndex>::max() / right.count) {
                    throw QStateError("Component symmetry class product overflow");
                }
                add_class(
                    next,
                    indices,
                    left.count * right.count,
                    left.amplitude * right.amplitude,
                    max_classes);
            }
        }
        combined = std::move(next);
    }

    std::vector<BasisIndex> counts;
    std::vector<QComplex> amplitudes;
    counts.reserve(combined.size());
    amplitudes.reserve(combined.size());
    for (const AmplitudeClass& amplitude_class : combined) {
        counts.push_back(amplitude_class.count);
        amplitudes.push_back(amplitude_class.amplitude);
    }

    SymmetryState result = SymmetryState::from_counts(state.qubit_count_, counts);
    result.set_class_amplitudes(amplitudes, false);
    result.discovery_error_ = 0.0;
    std::string reason;
    if (!result.validate(&reason)) {
        throw QStateError("Component symmetry discovery produced an invalid state: " + reason);
    }
    return result;
}

}  // namespace qubit
