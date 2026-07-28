#include "qubit/qdiagonal.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using qubit::CompiledDiagonalConfig;
using qubit::CompiledDiagonalPlan;
using qubit::QComplex;
using qubit::QDiagonalPhase;
using qubit::QRegister;
using qubit::QStateError;
using qubit::QubitId;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

[[nodiscard]] QRegister dense_state(std::size_t qubits) {
    const std::size_t dimension = std::size_t{1} << qubits;
    const double scale = 1.0 / std::sqrt(static_cast<double>(dimension));
    std::vector<QComplex> amplitudes(dimension);
    for (std::size_t basis = 0; basis < dimension; ++basis) {
        amplitudes[basis] = QComplex::from_polar(
            scale,
            0.007 * static_cast<double>((basis * 31U) % 211U));
    }
    return QRegister::from_amplitudes(std::move(amplitudes));
}

[[nodiscard]] double fidelity(
    const std::vector<QComplex>& first,
    const std::vector<QComplex>& second) {
    require(first.size() == second.size(), "state dimensions differ");
    QComplex overlap{};
    for (std::size_t index = 0; index < first.size(); ++index) {
        overlap += second[index].conjugate() * first[index];
    }
    return overlap.norm2();
}

[[nodiscard]] std::vector<QDiagonalPhase> phases(std::size_t qubits) {
    std::vector<QDiagonalPhase> result;
    result.reserve(qubits * 2U);
    for (std::size_t qubit = 0; qubit < qubits; ++qubit) {
        const double first = 0.013 * static_cast<double>(qubit + 1U);
        const double second = -0.009 * static_cast<double>(qubit + 3U);
        result.push_back(QDiagonalPhase{
            static_cast<QubitId>(qubit),
            QComplex::from_polar(1.0, -0.5 * first),
            QComplex::from_polar(1.0, 0.5 * first),
        });
        result.push_back(QDiagonalPhase{
            static_cast<QubitId>(qubit),
            QComplex::from_polar(1.0, -0.5 * second),
            QComplex::from_polar(1.0, 0.5 * second),
        });
    }
    return result;
}

}  // namespace

int main() {
    {
        QRegister prototype = dense_state(12);
        QRegister expected = prototype;
        QRegister actual = prototype;
        const auto diagonal = phases(12);
        CompiledDiagonalPlan plan(prototype, diagonal);
        require(plan.component_count() == 1U, "dense plan component count is wrong");
        require(plan.coefficient_bytes() == (std::size_t{1} << 12U) * sizeof(QComplex),
                "dense coefficient table has the wrong size");
        for (int iteration = 0; iteration < 40; ++iteration) {
            expected.apply_diagonal(diagonal);
            plan.execute(actual);
        }
        require(fidelity(expected.materialize(16), actual.materialize(16)) > 1.0 - 3e-10,
                "compiled dense diagonal plan changed the state");
        require(actual.validate(), "compiled dense diagonal state failed validation");
    }

    {
        QRegister prototype(20);
        prototype.apply_h(0);
        for (QubitId qubit = 1; qubit < 20; ++qubit) {
            prototype.apply_cnot(0, qubit);
        }
        QRegister expected = prototype;
        QRegister actual = prototype;
        const auto diagonal = phases(20);
        CompiledDiagonalPlan plan(prototype, diagonal);
        require(plan.coefficient_bytes() == 0U,
                "sparse plan allocated a dense coefficient table");
        for (int iteration = 0; iteration < 100; ++iteration) {
            expected.apply_diagonal(diagonal);
            plan.execute(actual);
        }
        require(fidelity(expected.materialize(24), actual.materialize(24)) > 1.0 - 3e-10,
                "compiled sparse diagonal plan changed the state");
    }

    {
        QRegister prototype(32);
        QRegister expected = prototype;
        QRegister actual = prototype;
        const auto diagonal = phases(32);
        CompiledDiagonalPlan plan(prototype, diagonal);
        expected.apply_diagonal(diagonal);
        plan.execute(actual);
        require(expected.probabilities_one() == actual.probabilities_one(),
                "compiled cell diagonal plan changed populations");
        require(actual.validate(), "compiled cell diagonal state failed validation");
    }

    {
        QRegister prototype = dense_state(10);
        const auto diagonal = phases(10);
        CompiledDiagonalPlan plan(prototype, diagonal);
        QRegister changed = prototype;
        changed.measure(0, 0.25);
        const auto before = changed.materialize(14);
        bool rejected = false;
        try {
            plan.execute(changed);
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "compiled plan accepted changed component structure");
        require(changed.materialize(14) == before,
                "structure rejection partially mutated the state");
    }

    {
        QRegister prototype = dense_state(8);
        QDiagonalPhase invalid;
        invalid.qubit = 0;
        invalid.zero = {2.0, 0.0};
        bool rejected = false;
        try {
            CompiledDiagonalPlan plan(prototype, std::span<const QDiagonalPhase>(&invalid, 1U));
            (void)plan;
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "compiled plan accepted a nonunitary phase");
    }

    {
        QRegister prototype = dense_state(12);
        const auto diagonal = phases(12);
        CompiledDiagonalConfig config;
        config.max_coefficient_bytes = 1024U;
        bool rejected = false;
        try {
            CompiledDiagonalPlan plan(prototype, diagonal, config);
            (void)plan;
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "compiled plan ignored its coefficient memory limit");
    }

    {
        QRegister prototype = dense_state(10);
        const auto diagonal = phases(10);
        CompiledDiagonalPlan plan(prototype, diagonal);
        std::vector<QRegister> storage(12U, prototype);
        std::vector<QRegister*> pointers;
        pointers.reserve(storage.size());
        for (QRegister& state : storage) {
            pointers.push_back(&state);
        }
        plan.execute_many(pointers, 4U);
        QRegister expected = prototype;
        expected.apply_diagonal(diagonal);
        const auto expected_state = expected.materialize(14);
        for (const QRegister& state : storage) {
            require(fidelity(state.materialize(14), expected_state) > 1.0 - 3e-11,
                    "compiled diagonal ensemble result differs");
        }
    }

    std::cout << "compiled diagonal tests passed\n";
    return 0;
}
