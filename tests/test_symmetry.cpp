#include "qubit/qsymmetry.hpp"

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

void require_complex_near(
    qubit::QComplex actual,
    qubit::QComplex expected,
    double tolerance,
    const std::string& message) {
    if (!qubit::almost_equal(actual, expected, tolerance)) {
        std::cerr << "TEST FAILURE: " << message << " actual=(" << actual.re << ','
                  << actual.im << ") expected=(" << expected.re << ',' << expected.im
                  << ")\n";
        std::exit(1);
    }
}

void test_ordered_classes_and_reflection() {
    const std::vector<qubit::BasisIndex> counts{2, 3, 3};
    qubit::SymmetryState state(3, counts);
    require(state.class_count() == 3, "ordered class count");
    require(state.class_for_basis(0) == 0, "ordered first class");
    require(state.class_for_basis(2) == 1, "ordered second class");
    require(state.class_for_basis(7) == 2, "ordered third class");
    require_near(state.class_probability(0), 0.25, 1e-12, "uniform class probability");

    const double pi = std::acos(-1.0);
    state.apply_class_phase(0, pi);
    const auto before = state.materialize(3);
    state.apply_weighted_reflection();

    qubit::QComplex mean{};
    for (const auto value : before) {
        mean += value / static_cast<double>(before.size());
    }
    for (std::size_t basis = 0; basis < before.size(); ++basis) {
        require_complex_near(
            state.amplitude(basis),
            mean * 2.0 - before[basis],
            1e-12,
            "weighted reflection matches dense reference");
    }
    require(state.validate(), "ordered symmetry state validates");
}

void test_class_unitary_and_fast_forward() {
    const std::vector<qubit::BasisIndex> counts{4, 4};
    qubit::SymmetryState repeated(3, counts);
    qubit::SymmetryState powered(3, counts);
    const double c = std::cos(0.137);
    const double s = std::sin(0.137);
    const std::vector<qubit::QComplex> rotation{
        {c, 0.0}, {-s, 0.0},
        {s, 0.0}, {c, 0.0},
    };
    for (std::size_t index = 0; index < 1000; ++index) {
        repeated.apply_class_unitary(rotation);
    }
    powered.iterate_class_unitary(rotation, 1000);
    require_complex_near(
        repeated.class_amplitude(0), powered.class_amplitude(0), 1e-10,
        "fast-forward class zero");
    require_complex_near(
        repeated.class_amplitude(1), powered.class_amplitude(1), 1e-10,
        "fast-forward class one");
}

void test_explicit_labels_and_fallback() {
    const std::vector<std::uint32_t> labels{0, 1, 1, 0, 2, 2, 2, 2};
    qubit::SymmetryState state = qubit::SymmetryState::from_labels(3, labels);
    const std::vector<qubit::QComplex> amplitudes{
        {0.25, 0.0}, {0.0, 0.25}, {0.25, 0.0},
    };
    state.set_class_amplitudes(amplitudes, true);
    qubit::QRegister materialized = state.to_register(3);
    for (std::size_t basis = 0; basis < labels.size(); ++basis) {
        require_complex_near(
            materialized.amplitude(basis), state.amplitude(basis), 1e-12,
            "QRegister fallback preserves amplitudes");
    }
    require(materialized.validate(), "materialized QRegister validates");
}

void test_class_refinement_and_merge() {
    const std::vector<qubit::BasisIndex> counts{8};
    qubit::SymmetryState state(3, counts);
    const std::size_t second = state.split_class(0, 3);
    require(second == 1, "split returns inserted class index");
    require(state.class_count() == 2, "split increases class count");
    require(state.class_size(0) == 3 && state.class_size(1) == 5,
            "split preserves class sizes");
    require(state.class_for_basis(2) == 0 && state.class_for_basis(3) == 1,
            "ordered split preserves deterministic membership");
    require(state.merge_equivalent() == 1, "equivalent split classes merge");
    require(state.class_count() == 1 && state.class_size(0) == 8,
            "merge restores original class");

    const std::vector<std::uint32_t> labels{0, 0, 1, 1, 1, 1, 0, 0};
    qubit::SymmetryState explicit_state = qubit::SymmetryState::from_labels(3, labels);
    const std::size_t explicit_second = explicit_state.split_class(1, 2);
    require(explicit_second == 2, "explicit split returns appended class index");
    require(explicit_state.class_count() == 3, "explicit split class count");
    require(explicit_state.class_for_basis(2) == 1, "explicit split first membership");
    require(explicit_state.class_for_basis(4) == 2, "explicit split remainder membership");
    require(explicit_state.merge_equivalent() == 2, "explicit equivalent classes merge");
    require(explicit_state.class_count() == 1, "explicit merge compacts all uniform classes");
}

void test_automatic_discovery() {
    qubit::QRegister uniform(10);
    for (qubit::QubitId qubit = 0; qubit < 10; ++qubit) {
        uniform.apply_h(qubit);
    }
    qubit::SymmetryState exact_discovered = qubit::SymmetryState::discover(uniform, 10);
    require(exact_discovered.discovery_error() == 0.0, "bit-exact discovery error");
    qubit::SymmetryState discovered = qubit::SymmetryState::discover(uniform, 10, 1e-12);
    require(discovered.class_count() == 1, "uniform register discovers one tolerant class");
    require(discovered.membership() == qubit::SymmetryMembership::OrderedRanges,
            "uniform discovery selects compact ordered membership");
    require(discovered.discovery_error() < 1e-12, "tolerant uniform discovery error");

    qubit::QRegister bell(3);
    bell.apply_h(0);
    bell.apply_cnot(0, 1);
    qubit::SymmetryState bell_classes = qubit::SymmetryState::discover(bell, 3, 1e-12);
    require(bell_classes.class_count() == 2, "Bell register discovers zero and nonzero classes");
    require(bell_classes.membership() == qubit::SymmetryMembership::ExplicitLabels,
            "noncontiguous Bell classes retain explicit membership");
    qubit::QRegister roundtrip = bell_classes.to_register(3);
    for (qubit::BasisIndex basis = 0; basis < 8; ++basis) {
        require_complex_near(
            roundtrip.amplitude(basis), bell.amplitude(basis), 1e-12,
            "discovered symmetry roundtrip");
    }

    std::vector<qubit::QComplex> near_equal(8, {0.0, 0.0});
    near_equal[0] = {1.0, 0.0};
    near_equal[1] = {1.0 + 1e-13, 0.0};
    qubit::QRegister perturbed = qubit::QRegister::from_amplitudes(std::move(near_equal));
    qubit::SymmetryState tolerant = qubit::SymmetryState::discover(
        perturbed, 3, 1e-12);
    require(tolerant.class_count() == 2, "tolerant discovery groups near-equal nonzero values");
    require(tolerant.discovery_error() <= 2e-12, "tolerant discovery reports bounded error");
}

void test_hamming_weight_symmetry() {
    qubit::SymmetryState state = qubit::SymmetryState::hamming_weight(8);
    require(state.class_count() == 9, "Hamming-weight class count");
    require(state.class_size(0) == 1 && state.class_size(4) == 70 &&
                state.class_size(8) == 1,
            "Hamming-weight binomial class sizes");
    require(state.class_for_basis(0b10110100U) == 4,
            "Hamming-weight basis classification");

    std::vector<qubit::QComplex> amplitudes(9);
    amplitudes[3] = {1.0 / std::sqrt(56.0), 0.0};
    state.set_class_amplitudes(amplitudes, true);
    for (std::uint64_t seed = 0; seed < 100; ++seed) {
        const qubit::BasisIndex basis = state.sample_basis(seed);
        require(std::popcount(basis) == 3, "Hamming-weight sampling preserves weight");
    }
    qubit::QRegister fallback = state.to_register(8);
    for (qubit::BasisIndex basis = 0; basis < 256; ++basis) {
        require_complex_near(
            fallback.amplitude(basis), state.amplitude(basis), 1e-12,
            "Hamming-weight fallback amplitude");
    }

    qubit::SymmetryState huge = qubit::SymmetryState::hamming_weight(60);
    require(huge.class_count() == 61, "60-qubit Hamming-weight class count");
    require(huge.estimated_bytes() < 4096,
            "60-qubit Hamming-weight representation remains compact");
    require(huge.validate(), "60-qubit Hamming-weight state validates");
}

void test_huge_count_only_state() {
    constexpr std::size_t qubits = 60;
    const qubit::BasisIndex space = qubit::BasisIndex{1} << qubits;
    const std::vector<qubit::BasisIndex> counts{1, 7, space - 8};
    qubit::SymmetryState state = qubit::SymmetryState::from_counts(qubits, counts);
    state.apply_class_phase(0, std::acos(-1.0));
    state.apply_weighted_reflection();
    require(state.estimated_bytes() < 1024, "60-qubit symmetry state remains tiny");
    require(state.validate(), "huge count-only symmetry state validates");
}

}  // namespace

int main() {
    test_ordered_classes_and_reflection();
    test_class_unitary_and_fast_forward();
    test_explicit_labels_and_fallback();
    test_class_refinement_and_merge();
    test_automatic_discovery();
    test_hamming_weight_symmetry();
    test_huge_count_only_state();
    std::cout << "All QSA symmetry-algebra tests passed.\n";
    return 0;
}
