#include "qubit/qgrover.hpp"
#include "qubit/qstate.hpp"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <span>
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

void dense_iteration(
    std::vector<qubit::QComplex>& amplitudes,
    std::span<const qubit::BasisIndex> marked) {
    for (qubit::BasisIndex index : marked) {
        amplitudes[static_cast<std::size_t>(index)] =
            -amplitudes[static_cast<std::size_t>(index)];
    }
    long double sum_re = 0.0L;
    long double sum_im = 0.0L;
    for (const auto& value : amplitudes) {
        sum_re += value.re;
        sum_im += value.im;
    }
    const long double inverse = 1.0L / static_cast<long double>(amplitudes.size());
    const qubit::QComplex twice_mean{
        static_cast<double>(2.0L * sum_re * inverse),
        static_cast<double>(2.0L * sum_im * inverse),
    };
    for (auto& value : amplitudes) {
        value = twice_mean - value;
    }
}

void test_single_target_optimum() {
    const qubit::BasisIndex marked = 5;
    qubit::GroverSearch search(3, std::span<const qubit::BasisIndex>(&marked, 1));
    require(search.optimal_iterations() == 2, "8-state single-target optimum must be two iterations");
    require_near(search.success_probability(), 0.125, 1e-14, "initial marked probability");
    search.run_optimal();
    require_near(search.success_probability(), 0.9453125, 1e-12, "optimal Grover probability");
    require(search.amplitude(marked).norm2() > 0.94, "marked amplitude must be amplified");
    require(search.validate(), "single-target compressed Grover state must validate");
}

void test_multiple_targets_match_dense_reference() {
    const std::vector<qubit::BasisIndex> marked{1, 7, 22};
    for (std::uint64_t iterations = 0; iterations <= 8; ++iterations) {
        qubit::GroverSearch search(5, marked);
        search.iterate(iterations);
        const auto compressed = search.materialize();

        const double uniform = 1.0 / std::sqrt(32.0);
        std::vector<qubit::QComplex> dense(32, qubit::QComplex{uniform, 0.0});
        for (std::uint64_t step = 0; step < iterations; ++step) {
            dense_iteration(dense, marked);
        }
        for (std::size_t index = 0; index < dense.size(); ++index) {
            require(
                qubit::almost_equal(compressed[index], dense[index], 2e-12),
                "compressed Grover amplitudes must match dense exact evolution");
        }
    }
}

void test_logarithmic_fast_forward() {
    const qubit::BasisIndex marked = 17;
    qubit::GroverSearch fast(12, std::span<const qubit::BasisIndex>(&marked, 1));
    qubit::GroverSearch repeated(12, std::span<const qubit::BasisIndex>(&marked, 1));
    constexpr std::uint64_t iterations = 10'000;
    fast.iterate(iterations);
    for (std::uint64_t index = 0; index < iterations; ++index) {
        repeated.apply_oracle();
        repeated.apply_diffusion();
    }
    require(
        qubit::almost_equal(fast.marked_amplitude(), repeated.marked_amplitude(), 2e-10),
        "fast-forward marked amplitude must match repeated iterations");
    require(
        qubit::almost_equal(fast.unmarked_amplitude(), repeated.unmarked_amplitude(), 2e-10),
        "fast-forward unmarked amplitude must match repeated iterations");
}

void test_huge_count_only_space() {
    qubit::GroverSearch search = qubit::GroverSearch::from_marked_count(60, 1);
    require(search.space_size() == (qubit::BasisIndex{1} << 60), "60-qubit logical space size");
    require(search.estimated_bytes() < 512, "count-only Grover state must remain constant-size");
    const std::uint64_t optimum = search.optimal_iterations();
    require(optimum > 800'000'000ULL, "60-qubit optimum should exceed 800 million iterations");
    search.run_optimal();
    require(search.success_probability() > 0.999999999, "huge Grover space must reach near-unit success");
    require(search.validate(), "huge count-only Grover state must validate");
}

void test_sampling() {
    const std::vector<qubit::BasisIndex> marked{2, 9};
    qubit::GroverSearch search(4, marked);
    search.run_optimal();
    const qubit::BasisIndex marked_sample = search.sample_basis(0.0, 0.75);
    require(marked_sample == 9, "marked sample rank selection");
    const qubit::BasisIndex unmarked_sample = search.sample_basis(0.999999, 0.0);
    require(unmarked_sample == 0, "unmarked sampling must skip marked states");
}

void test_qregister_exact_grover() {
    const std::vector<qubit::BasisIndex> marked{3, 41};
    qubit::GroverSearch compressed(6, marked);
    compressed.iterate(3);

    qubit::QRegister exact(6);
    for (qubit::QubitId qubit = 0; qubit < 6; ++qubit) {
        exact.apply_h(qubit);
    }
    exact.apply_grover_iterations(marked, 3);
    const auto expected = compressed.materialize();
    const auto actual = exact.materialize();
    for (std::size_t index = 0; index < expected.size(); ++index) {
        require(
            qubit::almost_equal(actual[index], expected[index], 2e-12),
            "QRegister Grover primitives must match compressed exact state");
    }
    require(exact.component_count() == 1, "exact Grover execution must use one global component");
    require(exact.component_storage_mode(0) == qubit::StorageMode::Dense,
            "exact Grover execution must use dense storage");
    require(exact.validate(), "exact Grover register must validate");

    const auto packet = exact.encode_qsc();
    qubit::QRegister restored = qubit::QRegister::decode_qsc(packet);
    for (std::size_t index = 0; index < expected.size(); ++index) {
        require(
            qubit::almost_equal(restored.amplitude(index), expected[index], 2e-12),
            "Grover exact state must survive QSC v1 roundtrip");
    }
}

}

int main() {
    test_single_target_optimum();
    test_multiple_targets_match_dense_reference();
    test_logarithmic_fast_forward();
    test_huge_count_only_space();
    test_sampling();
    test_qregister_exact_grover();
    std::cout << "All QSA Grover tests passed.\n";
    return 0;
}
