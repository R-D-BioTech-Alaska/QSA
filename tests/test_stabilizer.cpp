#include "qubit/qstabilizer.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>

namespace {

using qubit::QRegister;
using qubit::QStateError;
using qubit::QubitId;
using qubit::StabilizerConfig;
using qubit::StabilizerState;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void apply_random_gate(
    StabilizerState& stabilizer,
    QRegister& reference,
    std::mt19937_64& generator) {
    const std::size_t qubits = stabilizer.qubit_count();
    const QubitId first = static_cast<QubitId>(generator() % qubits);
    QubitId second = static_cast<QubitId>(generator() % qubits);
    if (second == first) {
        second = static_cast<QubitId>((second + 1U) % qubits);
    }
    switch (generator() % 9U) {
        case 0U:
            stabilizer.apply_h(first);
            reference.apply_h(first);
            break;
        case 1U:
            stabilizer.apply_s(first);
            reference.apply_s(first);
            break;
        case 2U:
            stabilizer.apply_sdg(first);
            reference.apply_sdg(first);
            break;
        case 3U:
            stabilizer.apply_x(first);
            reference.apply_x(first);
            break;
        case 4U:
            stabilizer.apply_y(first);
            reference.apply_y(first);
            break;
        case 5U:
            stabilizer.apply_z(first);
            reference.apply_z(first);
            break;
        case 6U:
            stabilizer.apply_cnot(first, second);
            reference.apply_cnot(first, second);
            break;
        case 7U:
            stabilizer.apply_cz(first, second);
            reference.apply_cz(first, second);
            break;
        default:
            stabilizer.apply_swap(first, second);
            reference.apply_swap(first, second);
            break;
    }
}

void compare_probabilities(
    const StabilizerState& stabilizer,
    const QRegister& reference,
    double tolerance = 2e-10) {
    for (std::size_t qubit = 0; qubit < stabilizer.qubit_count(); ++qubit) {
        const double actual = stabilizer.probability_one(static_cast<QubitId>(qubit));
        const double expected = reference.probability_one(static_cast<QubitId>(qubit));
        require(std::abs(actual - expected) <= tolerance,
                "stabilizer probability differs from QRegister");
    }
}

}  // namespace

int main() {
    {
        StabilizerState state(3);
        require(state.validate_full(), "initial stabilizer tableau failed validation");
        require(state.probability_one(0) == 0.0, "initial stabilizer probability is wrong");
        state.apply_h(0);
        state.apply_cnot(0, 1);
        require(state.probability_one(0) == 0.5, "Bell control probability is wrong");
        require(state.probability_one(1) == 0.5, "Bell target probability is wrong");
        require(state.probability_one(2) == 0.0, "independent probability is wrong");
        require(state.validate_full(), "Bell stabilizer tableau failed validation");
    }

    std::mt19937_64 generator(0x53544142494C495AULL);
    for (std::size_t test_case = 0; test_case < 240U; ++test_case) {
        const std::size_t qubits = 2U + test_case % 9U;
        StabilizerState stabilizer(qubits);
        QRegister reference(qubits);
        const std::size_t gates = 20U + test_case % 90U;
        for (std::size_t gate = 0; gate < gates; ++gate) {
            apply_random_gate(stabilizer, reference, generator);
            if ((gate % 11U) == 0U) {
                compare_probabilities(stabilizer, reference);
                require(stabilizer.validate_full(),
                        "random stabilizer tableau failed full validation");
            }
        }
        compare_probabilities(stabilizer, reference);

        StabilizerState measured = stabilizer;
        QRegister measured_reference = reference;
        for (std::size_t qubit = 0; qubit < qubits; ++qubit) {
            const double sample = static_cast<double>(generator() >> 11U) *
                                  (1.0 / 9007199254740992.0);
            const int actual = measured.measure_z(static_cast<QubitId>(qubit), sample);
            const int expected = measured_reference.measure(static_cast<QubitId>(qubit), sample);
            require(actual == expected, "stabilizer measurement differs from QRegister");
            compare_probabilities(measured, measured_reference);
            require(measured.validate_full(),
                    "measured stabilizer tableau failed full validation");
        }
    }

    {
        StabilizerConfig config;
        config.max_tableau_bytes = 1ULL << 29;
        StabilizerState large(4096, config);
        for (QubitId qubit = 0; qubit < 4096; ++qubit) {
            large.apply_h(qubit);
            large.apply_s(qubit);
        }
        for (QubitId qubit = 1; qubit < 4096; ++qubit) {
            large.apply_cnot(qubit - 1U, qubit);
        }
        require(large.validate(), "large stabilizer tableau failed validation");
        require(large.estimated_bytes() < 140ULL * 1024ULL * 1024ULL,
                "large stabilizer tableau exceeded memory gate");
    }

    {
        bool rejected = false;
        try {
            StabilizerConfig config;
            config.max_tableau_bytes = 1024U;
            StabilizerState state(1024, config);
            (void)state;
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "stabilizer memory limit was not enforced");
    }

    std::cout << "stabilizer tests passed\n";
    return 0;
}
