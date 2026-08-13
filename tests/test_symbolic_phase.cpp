#include "qubit/qsymbolic_phase.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(
    const qubit::QComplex& observed,
    const qubit::QComplex& expected,
    double tolerance,
    const char* message) {
    if (!qubit::almost_equal(observed, expected, tolerance)) {
        throw std::runtime_error(
            std::string(message) + ": observed=(" +
            std::to_string(observed.re) + "," + std::to_string(observed.im) +
            ") expected=(" + std::to_string(expected.re) + "," +
            std::to_string(expected.im) + ")");
    }
}

qubit::QMatrix4 controlled_phase(double angle) {
    qubit::QMatrix4 matrix{};
    matrix.values[0] = {1.0, 0.0};
    matrix.values[5] = {1.0, 0.0};
    matrix.values[10] = {1.0, 0.0};
    matrix.values[15] = qubit::QComplex::from_polar(1.0, angle);
    return matrix;
}

qubit::QRegister plus_state(std::size_t qubits) {
    const std::size_t dimension = std::size_t{1U} << qubits;
    const double amplitude = std::exp2(-0.5 * static_cast<double>(qubits));
    std::vector<qubit::QComplex> values(
        dimension, qubit::QComplex{amplitude, 0.0});
    return qubit::QRegister::from_amplitudes(std::move(values));
}

void dense_equivalence() {
    using qubit::ExactSymbolicPhaseConfig;
    using qubit::ExactSymbolicPhaseGraphSum;
    using qubit::QRegister;

    constexpr std::array<double, 6> angles{
        0.173, -0.411, 0.927, -1.217, 1.733, -2.119,
    };

    for (std::size_t qubits = 2U; qubits <= 6U; ++qubits) {
        ExactSymbolicPhaseConfig config;
        config.max_live_branches = 4096U;
        config.max_intermediate_branches = 8192U;
        config.max_coefficient_terms = 16384U;
        config.max_symbol_terms = 65536U;
        config.max_retained_estimated_bytes = 256U * 1024U * 1024U;
        ExactSymbolicPhaseGraphSum compact(qubits, config);
        for (std::size_t symbol = 0U; symbol < angles.size(); ++symbol) {
            compact.bind_symbol(
                static_cast<qubit::SymbolicPhaseId>(symbol), angles[symbol]);
        }
        QRegister dense = plus_state(qubits);

        for (std::size_t step = 0U; step < 4U; ++step) {
            const auto target = static_cast<qubit::QubitId>(step % qubits);
            const auto symbol = static_cast<qubit::SymbolicPhaseId>(step);
            compact.apply_h(target);
            dense.apply_h(target);

            if ((step & 1U) == 0U) {
                compact.apply_rz_symbol(target, symbol);
                dense.apply_rz(target, angles[step]);
                compact.apply_rz_symbol(target, symbol, -1);
                dense.apply_rz(target, -angles[step]);
            } else {
                const auto other = static_cast<qubit::QubitId>((step + 1U) % qubits);
                if (other != target) {
                    compact.apply_controlled_phase_symbol(target, other, symbol);
                    dense.apply_two(target, other, controlled_phase(angles[step]));
                    compact.apply_controlled_phase_symbol(target, other, symbol, -1);
                    dense.apply_two(target, other, controlled_phase(-angles[step]));
                }
            }

            compact.apply_h(target);
            dense.apply_h(target);
            compact.apply_t(target);
            dense.apply_t(target);
            compact.apply_sdg(target);
            dense.apply_sdg(target);
        }

        compact.apply_h(0U);
        dense.apply_h(0U);
        compact.apply_rz_symbol(0U, 4U);
        dense.apply_rz(0U, angles[4]);
        compact.apply_h(0U);
        dense.apply_h(0U);

        if (qubits > 2U) {
            compact.apply_cz(0U, 2U);
            dense.apply_cz(0U, 2U);
            compact.apply_swap(1U, 2U);
            dense.apply_swap(1U, 2U);
        }

        const auto observed = compact.materialize(6U);
        const auto expected = dense.materialize(6U);
        require(observed.size() == expected.size(), "symbolic dense size mismatch");
        for (std::size_t index = 0U; index < observed.size(); ++index) {
            require_close(
                observed[index], expected[index], 8e-12,
                "symbolic arbitrary-angle amplitude mismatch");
        }
    }
}

void exact_reconvergence() {
    using qubit::ExactSymbolicPhaseGraphSum;

    ExactSymbolicPhaseGraphSum state(4U);
    state.bind_symbol(7U, 0.73123456789);
    state.apply_h(1U);
    state.apply_rz_symbol(1U, 7U);
    state.apply_rz_symbol(1U, 7U, -1);
    state.apply_controlled_phase_symbol(1U, 2U, 7U);
    state.apply_controlled_phase_symbol(1U, 2U, 7U, -1);
    state.apply_h(1U);

    require(state.branch_count() == 1U, "symbolic inverse phase pair did not reconverge");
    require(state.stats().sqrt2_denominator_power == 0, "symbolic denominator did not collapse");
    require(state.stats().coefficient_terms == 1U, "symbolic coefficient did not collapse");
    require(state.stats().graph_merges > 0U, "symbolic graph merge was not observed");
    require(state.stats().exact_cancellations > 0U, "symbolic exact cancellation was not observed");

    const std::array<std::uint8_t, 4> bits{0U, 1U, 0U, 1U};
    const auto exact = state.exact_amplitude_bits(bits);
    require(exact.terms.size() == 1U, "symbolic point amplitude did not remain exact single-term");
    require(exact.sqrt2_denominator_power == 0, "symbolic point denominator mismatch");
}

void connected_branch_growth() {
    using qubit::ExactSymbolicPhaseConfig;
    using qubit::ExactSymbolicPhaseGraphSum;

    ExactSymbolicPhaseConfig config;
    config.max_live_branches = 256U;
    config.max_intermediate_branches = 512U;
    config.max_coefficient_terms = 1024U;
    config.max_symbol_terms = 32768U;
    config.max_retained_estimated_bytes = 128U * 1024U * 1024U;
    ExactSymbolicPhaseGraphSum state(64U, config);

    for (std::uint32_t symbol = 0U; symbol < 16U; ++symbol) {
        state.bind_symbol(symbol, -1.7 + 0.193 * static_cast<double>(symbol + 1U));
    }
    for (std::uint32_t defect = 0U; defect < 8U; ++defect) {
        const auto target = static_cast<qubit::QubitId>(defect + 1U);
        state.apply_controlled_phase_symbol(0U, target, 2U * defect);
        state.apply_h(target);
        state.apply_controlled_phase_symbol(0U, target, 2U * defect + 1U);
    }

    require(state.stats().hadamard_defects == 8U,
        "connected symbolic control Hadamard count mismatch");
    require(state.branch_count() == 256U,
        "connected symbolic control did not retain the full 2^H branch count");
}

void fail_closed() {
    using qubit::ExactSymbolicPhaseConfig;
    using qubit::ExactSymbolicPhaseGraphSum;
    using qubit::OperationCode;
    using qubit::QStateError;

    ExactSymbolicPhaseConfig config;
    config.max_live_branches = 16U;
    config.max_intermediate_branches = 32U;
    config.max_coefficient_terms = 64U;
    config.max_symbol_terms = 256U;
    ExactSymbolicPhaseGraphSum state(5U, config);
    for (std::uint32_t symbol = 0U; symbol < 5U; ++symbol) {
        state.bind_symbol(symbol, 0.21 + 0.17 * static_cast<double>(symbol));
    }

    for (std::uint32_t defect = 0U; defect < 2U; ++defect) {
        const auto qubit = static_cast<qubit::QubitId>(defect);
        state.apply_h(qubit);
        state.apply_rz_symbol(qubit, defect);
        state.apply_h(qubit);
    }
    require(state.branch_count() == 16U, "symbolic branch-cap setup did not reach sixteen branches");
    const auto before = state.stats();
    bool rejected = false;
    try {
        state.apply_h(2U);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "symbolic branch cap did not reject");
    require(state.branch_count() == before.live_branches, "symbolic branch rejection mutated state");
    require(state.stats().hadamard_defects == before.hadamard_defects,
        "symbolic branch rejection mutated counters");

    rejected = false;
    try {
        state.bind_symbol(0U, 9.0);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "symbolic ID rebind mismatch did not reject");

    rejected = false;
    try {
        state.apply_rz_symbol(0U, 99U);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "unbound symbolic Rz did not reject");

    rejected = false;
    try {
        state.apply({OperationCode::Rz, 0U, 0U, 0.3});
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "anonymous floating Rz did not reject from exact symbolic API");

    rejected = false;
    try {
        state.apply({OperationCode::Cnot, 0U, 1U});
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "CNOT did not reject from symbolic PhaseGraph contract");
}

}  // namespace

int main() {
    try {
        dense_equivalence();
        exact_reconvergence();
        connected_branch_growth();
        fail_closed();
        std::cout << "exact symbolic phase graph tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
