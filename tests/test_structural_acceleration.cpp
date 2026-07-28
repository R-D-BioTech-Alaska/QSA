#include "qubit/qstate.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

using qubit::QComplex;
using qubit::QRegister;
using qubit::QStateConfig;
using qubit::QubitId;

[[nodiscard]] double fidelity(
    const std::vector<QComplex>& first,
    const std::vector<QComplex>& second) {
    if (first.size() != second.size()) {
        return 0.0;
    }
    QComplex overlap{};
    for (std::size_t index = 0; index < first.size(); ++index) {
        overlap += second[index].conjugate() * first[index];
    }
    return overlap.norm2();
}

[[nodiscard]] QRegister make_dense_groups(std::size_t width) {
    QStateConfig config;
    config.max_component_qubits = 60;
    config.max_dense_amplitudes = 1ULL << 26;
    config.max_sparse_entries = 8'000'000;
    QRegister state(width * 2U, config);
    for (std::size_t qubit = 0; qubit < width * 2U; ++qubit) {
        state.apply_ry(
            static_cast<QubitId>(qubit),
            0.27 + 0.013 * static_cast<double>(qubit));
        state.apply_rz(
            static_cast<QubitId>(qubit),
            -0.19 + 0.009 * static_cast<double>(qubit));
    }
    for (std::size_t qubit = 0; qubit + 1U < width; ++qubit) {
        state.apply_cz(
            static_cast<QubitId>(qubit),
            static_cast<QubitId>(qubit + 1U));
        state.apply_cz(
            static_cast<QubitId>(width + qubit),
            static_cast<QubitId>(width + qubit + 1U));
    }
    return state;
}

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main() {
    {
        QRegister state = make_dense_groups(4);
        const auto before = state.materialize(12);
        state.apply_swap_structured(1, 6);

        QRegister reference = QRegister::from_amplitudes(before, state.config());
        reference.apply_swap(1, 6);
        require(
            fidelity(state.materialize(12), reference.materialize(12)) > 1.0 - 1e-12,
            "disconnected structured SWAP changed the state");
        require(state.component_count() == 2U,
                "disconnected structured SWAP merged components");
        require(state.validate(), "disconnected structured SWAP produced an invalid state");
    }

    {
        QRegister state(7);
        for (QubitId qubit = 1; qubit < 7; ++qubit) {
            state.apply_ry(qubit, 0.31 + 0.07 * static_cast<double>(qubit));
        }
        for (QubitId qubit = 1; qubit < 6; ++qubit) {
            state.apply_cz(qubit, qubit + 1U);
        }
        const auto before = state.materialize(12);
        state.apply_cnot_structured(0, 3);
        require(
            fidelity(before, state.materialize(12)) > 1.0 - 1e-12,
            "CNOT with exact |0> control was not an identity");
        require(state.component_count() == 2U,
                "CNOT with exact |0> control merged components");

        state.apply_x(0);
        QRegister reference = QRegister::from_amplitudes(state.materialize(12));
        reference.apply_cnot(0, 3);
        state.apply_cnot_structured(0, 3);
        require(
            fidelity(state.materialize(12), reference.materialize(12)) > 1.0 - 1e-12,
            "CNOT with exact |1> control did not apply X to the target");
        require(state.component_count() == 2U,
                "CNOT with exact |1> control merged components");
    }

    {
        QRegister state(7);
        for (QubitId qubit = 0; qubit < 6; ++qubit) {
            state.apply_ry(qubit, 0.23 + 0.03 * static_cast<double>(qubit));
        }
        for (QubitId qubit = 0; qubit < 5; ++qubit) {
            state.apply_cz(qubit, qubit + 1U);
        }
        state.apply_h(6);
        const auto before = state.materialize(12);
        state.apply_cnot_structured(2, 6);
        require(
            fidelity(before, state.materialize(12)) > 1.0 - 1e-12,
            "CNOT with exact |+> target was not an identity");
        require(state.component_count() == 2U,
                "CNOT with exact |+> target merged components");

        state.apply_z(6);
        QRegister reference = QRegister::from_amplitudes(state.materialize(12));
        reference.apply_cnot(2, 6);
        state.apply_cnot_structured(2, 6);
        require(
            fidelity(state.materialize(12), reference.materialize(12)) > 1.0 - 1e-12,
            "CNOT with exact |-> target did not apply Z to the control");
        require(state.component_count() == 2U,
                "CNOT with exact |-> target merged components");
    }

    {
        QRegister state(6);
        state.apply_x(0);
        state.apply_x(5);
        QRegister reference = state;
        reference.apply_cz(0, 5);
        state.apply_cz_structured(0, 5);
        require(
            fidelity(state.materialize(10), reference.materialize(10)) > 1.0 - 1e-12,
            "structured CZ basis-state path changed the result");
        require(state.component_count() == 6U,
                "structured CZ basis-state path merged components");
    }

    {
        QRegister state = make_dense_groups(16);
        state.apply_swap_structured(0, 16);
        require(state.component_count() == 2U,
                "large structured SWAP merged components");
        require(state.component_size(0) == 16U && state.component_size(16) == 16U,
                "large structured SWAP changed component widths");
        require(state.validate(), "large structured SWAP produced an invalid state");
    }

    {
        std::mt19937_64 generator(0x5354525543545552ULL);
        for (std::size_t test_case = 0; test_case < 400U; ++test_case) {
            const std::size_t width = 2U + test_case % 4U;
            QRegister structured = make_dense_groups(width);
            const auto initial = structured.materialize(14);
            QRegister reference = QRegister::from_amplitudes(initial, structured.config());
            const QubitId first = static_cast<QubitId>(generator() % width);
            const QubitId second = static_cast<QubitId>(width + generator() % width);
            structured.apply_swap_structured(first, second);
            reference.apply_swap(first, second);
            require(
                fidelity(structured.materialize(14), reference.materialize(14)) >
                    1.0 - 2e-12,
                "random structured SWAP diverged from QRegister");
            require(structured.component_count() == 2U,
                    "random structured SWAP merged components");
        }
    }

    std::cout << "structural acceleration tests passed\n";
    return 0;
}
