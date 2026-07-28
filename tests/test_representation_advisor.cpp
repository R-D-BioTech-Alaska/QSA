#include "qubit/qrouter.hpp"
#include "qubit/qsymmetry.hpp"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using qubit::BasisIndex;
using qubit::QComplex;
using qubit::QRegister;
using qubit::QStateError;
using qubit::QubitId;
using qubit::RepresentationAdvisor;
using qubit::RepresentationFeatures;
using qubit::RepresentationKind;
using qubit::SymmetryState;

struct ClassValue {
    BasisIndex count;
    QComplex amplitude;
};

[[nodiscard]] std::vector<ClassValue> classes(const SymmetryState& state) {
    std::vector<ClassValue> result;
    result.reserve(state.class_count());
    for (std::size_t index = 0; index < state.class_count(); ++index) {
        result.push_back(ClassValue{state.class_size(index), state.class_amplitude(index)});
    }
    return result;
}

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_same_classes(const SymmetryState& first, const SymmetryState& second) {
    auto left = classes(first);
    auto right = classes(second);
    require(left.size() == right.size(), "symmetry class counts differ");
    std::vector<bool> used(right.size(), false);
    for (const ClassValue& candidate : left) {
        bool matched = false;
        for (std::size_t index = 0; index < right.size(); ++index) {
            if (!used[index] && candidate.count == right[index].count &&
                qubit::almost_equal(candidate.amplitude, right[index].amplitude, 2e-12)) {
                used[index] = true;
                matched = true;
                break;
            }
        }
        require(matched, "component symmetry class did not match dense discovery");
    }
}

}  // namespace

int main() {
    {
        QRegister state(8);
        for (QubitId pair = 0; pair < 4; ++pair) {
            const QubitId first = pair * 2U;
            state.apply_h(first);
            state.apply_cnot(first, first + 1U);
        }
        const SymmetryState component = SymmetryState::discover_components(state);
        const SymmetryState dense = SymmetryState::discover(state, 12, 0.0);
        require(component.membership() == qubit::SymmetryMembership::CountOnly,
                "component discovery must remain count-only");
        require(component.discovery_error() == 0.0,
                "exact component discovery reported an error");
        require(component.validate(), "component symmetry state failed validation");
        require_same_classes(component, dense);
    }

    {
        QRegister state(40);
        for (QubitId qubit = 0; qubit < 40; ++qubit) {
            state.apply_h(qubit);
        }
        const SymmetryState component = SymmetryState::discover_components(state, 64);
        require(component.class_count() <= 41U,
                "uniform 40-qubit register exceeded linear exact class growth");
        BasisIndex total = 0U;
        for (std::size_t index = 0; index < component.class_count(); ++index) {
            total += component.class_size(index);
        }
        require(total == (BasisIndex{1} << 40U),
                "large component discovery class sizes do not cover the state space");
        require(component.validate(), "large component discovery failed validation");
    }

    {
        QRegister state(10);
        for (QubitId qubit = 0; qubit < 10; ++qubit) {
            state.apply_ry(qubit, 0.07 + 0.11 * static_cast<double>(qubit));
            state.apply_rz(qubit, -0.03 + 0.05 * static_cast<double>(qubit));
        }
        bool rejected = false;
        try {
            (void)SymmetryState::discover_components(state, 4);
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "component discovery ignored its class limit");
    }

    RepresentationAdvisor advisor;
    {
        QRegister state(12);
        const auto features = RepresentationAdvisor::inspect(state, 1'000U, 2U, false);
        require(advisor.recommend(features).kind == RepresentationKind::Symmetry,
                "advisor did not select exact symmetry compression");
    }
    {
        QRegister state(12);
        const auto features = RepresentationAdvisor::inspect(state, 1'000U, 0U, true);
        require(advisor.recommend(features).kind == RepresentationKind::QuantumDot,
                "advisor did not select declared quantum-dot structure");
    }
    {
        QRegister state(12);
        const auto features = RepresentationAdvisor::inspect(state, 10U);
        require(advisor.recommend(features).kind == RepresentationKind::Register,
                "advisor did not keep a general workload on QRegister");
    }
    {
        RepresentationFeatures features;
        features.qubit_count = 12U;
        features.component_count = 1U;
        features.largest_component = 12U;
        features.support_entries = 20U;
        features.repeated_steps = 100U;
        features.exact_symmetry_classes = 18U;
        require(advisor.recommend(features).kind == RepresentationKind::Symmetry,
                "close initial recommendation was not symmetry");
        for (int observation = 0; observation < 20; ++observation) {
            advisor.observe(features, RepresentationKind::Symmetry, false);
            advisor.observe(features, RepresentationKind::Register, true);
        }
        require(advisor.recommend(features).kind == RepresentationKind::Register,
                "Bayesian evidence did not change a close recommendation");
        advisor.reset();
        require(advisor.recommend(features).kind == RepresentationKind::Symmetry,
                "advisor reset did not clear learned evidence");
    }

    std::cout << "representation advisor tests passed\n";
    return 0;
}
