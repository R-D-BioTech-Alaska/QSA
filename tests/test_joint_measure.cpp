#include "qubit/qstate.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using qubit::QComplex;
using qubit::QRegister;
using qubit::QStateConfig;

[[nodiscard]] QRegister make_dense_state(std::size_t qubits) {
    QStateConfig config;
    config.max_component_qubits = 30;
    config.max_dense_amplitudes = 1ULL << 22;
    const std::size_t dimension = std::size_t{1} << qubits;
    const double scale = 1.0 / std::sqrt(static_cast<double>(dimension));
    std::vector<QComplex> amplitudes(dimension);
    for (std::size_t basis = 0; basis < dimension; ++basis) {
        const double phase =
            0.000013 * static_cast<double>((basis * 2654435761ULL) & 0xFFFFFFU);
        amplitudes[basis] = QComplex::from_polar(scale, phase);
    }
    return QRegister::from_amplitudes(std::move(amplitudes), config);
}

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main() {
    QRegister bell(2);
    bell.apply_h(0);
    bell.apply_cnot(0, 1);
    std::size_t zeros = 0U;
    std::size_t ones = 0U;
    constexpr std::size_t trials = 20'000U;
    for (std::size_t seed = 0; seed < trials; ++seed) {
        QRegister sample = bell;
        const auto bits = sample.measure_all_joint(0xA5A5A5A500000000ULL + seed);
        if (bits[0] == 0 && bits[1] == 0) {
            ++zeros;
        } else if (bits[0] == 1 && bits[1] == 1) {
            ++ones;
        } else {
            throw std::runtime_error("joint Bell measurement produced a forbidden outcome");
        }
        require(sample.component_count() == 2U,
                "joint measurement did not collapse to cells");
        require(sample.validate(), "joint measurement produced an invalid state");
    }
    const double one_fraction = static_cast<double>(ones) / static_cast<double>(trials);
    require(std::abs(one_fraction - 0.5) < 0.02,
            "joint Bell measurement distribution is outside tolerance");
    require(zeros + ones == trials, "joint Bell measurement trial accounting failed");

    for (std::size_t qubits : {12U, 16U, 20U}) {
        QRegister state = make_dense_state(qubits);
        const auto bits = state.measure_all_joint(0x123456789ABCDEF0ULL + qubits);
        require(bits.size() == qubits, "joint measurement returned the wrong bit count");
        require(state.component_count() == qubits,
                "joint measurement did not produce one cell per qubit");
        require(state.validate(), "joint dense measurement produced an invalid state");
        const auto populations = state.probabilities_one();
        for (std::size_t qubit = 0; qubit < qubits; ++qubit) {
            require(populations[qubit] == static_cast<double>(bits[qubit]),
                    "joint measurement collapse disagrees with its returned bit");
        }
    }

    std::cout << "joint measurement tests passed\n";
    return 0;
}
