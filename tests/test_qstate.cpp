#include "qubit/qstate.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "TEST FAILURE: " << message << '\n';
        std::exit(1);
    }
}

void require_near(double actual, double expected, double tolerance, const std::string& message) {
    if (std::abs(actual - expected) > tolerance) {
        std::cerr << "TEST FAILURE: " << message << " actual=" << actual
                  << " expected=" << expected << '\n';
        std::exit(1);
    }
}

void test_geometric_cells() {
    qubit::QRegister state(128);
    require(state.component_count() == 128, "independent qubits must remain independent cells");
    state.apply_h(17);
    require_near(state.probability_one(17), 0.5, 1e-12, "Hadamard probability");
    require(state.component_count() == 128, "single-qubit gates must not create global state storage");

    state.apply_rz(17, 0.73);
    require_near(state.probability_one(17), 0.5, 1e-12, "Rz must preserve population");
    state.apply_h(17);
    require(state.validate(), "geometric-cell state must validate");
}

void test_bell_state_and_measurement() {
    qubit::QRegister state(2);
    state.apply_h(0);
    state.apply_cnot(0, 1);

    require(state.component_count() == 1, "Bell pair must form one entangled patch");
    require(state.component_size(0) == 2, "Bell component size");
    require_near(state.probability_one(0), 0.5, 1e-12, "Bell q0 probability");
    require_near(state.probability_one(1), 0.5, 1e-12, "Bell q1 probability");

    const auto amplitudes = state.materialize();
    const double inv_sqrt_two = 1.0 / std::sqrt(2.0);
    require_near(amplitudes[0].re, inv_sqrt_two, 1e-12, "Bell |00> amplitude");
    require_near(amplitudes[3].re, inv_sqrt_two, 1e-12, "Bell |11> amplitude");
    require(amplitudes[1].norm2() < 1e-24 && amplitudes[2].norm2() < 1e-24,
            "Bell off-support amplitudes");

    const int outcome = state.measure(0, 0.75);
    require(outcome == 0, "measurement sample should select |0> branch");
    require_near(state.probability_one(1), 0.0, 1e-12, "Bell partner must collapse with q0");
    require(state.component_count() == 2, "measurement must split the collapsed qubit patch");
    require(state.validate(), "collapsed Bell state must validate");
}

void test_non_entangling_gate_compacts() {
    qubit::QRegister state(2);
    state.apply_x(0);
    state.apply_cnot(0, 1);
    require(state.component_count() == 2,
            "CNOT on a computational basis state must compact back to independent cells");
    require_near(state.probability_one(0), 1.0, 1e-12, "control remains one");
    require_near(state.probability_one(1), 1.0, 1e-12, "target becomes one");
    require(state.amplitude(3).norm2() > 1.0 - 1e-12, "state must be |11>");
}

void test_sparse_ghz() {
    constexpr std::size_t qubits = 40;
    qubit::QRegister state(qubits);
    state.apply_h(0);
    for (std::size_t target = 1; target < qubits; ++target) {
        state.apply_cnot(0, static_cast<qubit::QubitId>(target));
    }

    require(state.component_count() == 1, "GHZ qubits must share one component");
    require(state.component_size(0) == qubits, "GHZ component width");
    require(state.component_storage_mode(0) == qubit::StorageMode::Sparse,
            "GHZ must remain sparse");
    require(state.component_nonzero_count(0) == 2, "GHZ support must contain exactly two states");

    std::vector<std::uint8_t> zeros(qubits, 0);
    std::vector<std::uint8_t> ones(qubits, 1);
    const double expected = 1.0 / std::sqrt(2.0);
    require_near(state.amplitude_bits(zeros).re, expected, 1e-12, "GHZ all-zero amplitude");
    require_near(state.amplitude_bits(ones).re, expected, 1e-12, "GHZ all-one amplitude");
    require(state.estimated_bytes() < 100'000,
            "40-qubit GHZ representation should remain far below dense statevector memory");
    require(state.validate(), "sparse GHZ state must validate");
}

void test_qsc_roundtrip() {
    qubit::QRegister original(8);
    original.apply_h(0);
    original.apply_cnot(0, 3);
    original.apply_ry(5, 0.31);
    original.apply_rz(3, -0.82);

    const std::vector<std::uint8_t> encoded = original.encode_qsc();
    qubit::QRegister restored = qubit::QRegister::decode_qsc(encoded);
    require(restored.qubit_count() == original.qubit_count(), "QSC qubit count roundtrip");
    require(restored.component_count() == original.component_count(), "QSC component count roundtrip");
    for (std::uint64_t basis = 0; basis < 256; ++basis) {
        require(qubit::almost_equal(original.amplitude(basis), restored.amplitude(basis), 1e-11),
                "QSC amplitude roundtrip");
    }
    require(restored.validate(), "decoded QSC state must validate");

    std::vector<std::uint8_t> corrupted = encoded;
    corrupted[corrupted.size() / 2] ^= 0x01U;
    bool rejected = false;
    try {
        (void)qubit::QRegister::decode_qsc(corrupted);
    } catch (const qubit::QStateError&) {
        rejected = true;
    }
    require(rejected, "QSC checksum must reject corrupted data");
}

void test_trajectory_noise() {
    qubit::QRegister state(1);
    state.apply_x(0);
    state.apply_amplitude_damping_trajectory(0, 1.0, 0.0);
    require_near(state.probability_one(0), 0.0, 1e-12, "full amplitude damping must decay |1> to |0>");

    state.apply_h(0);
    state.apply_phase_flip_trajectory(0, 1.0, 0.5);
    state.apply_h(0);
    require_near(state.probability_one(0), 1.0, 1e-12, "H-Z-H must act as X");
}

}  // namespace

int main() {
    test_geometric_cells();
    test_bell_state_and_measurement();
    test_non_entangling_gate_compacts();
    test_sparse_ghz();
    test_qsc_roundtrip();
    test_trajectory_noise();
    std::cout << "All Qubit Native State Engine tests passed.\n";
    return 0;
}
