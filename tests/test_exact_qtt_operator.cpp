#include "qubit/qqtt_operator.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

using namespace qubit;

namespace {

bool close(QComplex left, QComplex right, double tolerance = 1e-11) {
    return almost_equal(left, right, tolerance);
}

template <class Fn>
bool throws_qstate(Fn&& fn) {
    try {
        fn();
    } catch (const QStateError&) {
        return true;
    }
    return false;
}

std::vector<std::uint8_t> bits_for(std::size_t logical_bits, BasisIndex index) {
    std::vector<std::uint8_t> bits(logical_bits);
    for (std::size_t position = 0U; position < logical_bits; ++position) {
        const std::size_t shift = logical_bits - 1U - position;
        bits[position] = static_cast<std::uint8_t>((index >> shift) & BasisIndex{1});
    }
    return bits;
}

}  // namespace

int main() {
    constexpr std::size_t bits = 5U;
    const std::vector<QComplex> weights{
        QComplex{0.5}, QComplex{-1.25}, QComplex{2.0}, QComplex{0.75}, QComplex{1.5},
    };
    const ExactQTTFunction field = ExactQTTFunction::weighted_bit_sum(weights, QComplex{0.25});

    const ExactQTTOperator identity = ExactQTTOperator::identity(bits);
    const ExactQTTFunction identity_result = identity.apply(field);
    assert(identity.stats().maximum_rank == 1U);
    assert(identity_result.logical_bits() == bits);
    for (BasisIndex index = 0U; index < (BasisIndex{1} << bits); ++index) {
        assert(close(identity_result.value(index), field.value(index)));
    }

    const ExactQTTOperator diagonal = ExactQTTOperator::diagonal(field);
    const ExactQTTFunction diagonal_result = diagonal.apply(field);
    const ExactQTTFunction hadamard_result = field.hadamard(field);
    for (BasisIndex index = 0U; index < (BasisIndex{1} << bits); ++index) {
        assert(close(diagonal_result.value(index), hadamard_result.value(index)));
    }

    const std::vector<double> laplacian_weights{1.0, 2.0, 3.0, 4.0, 5.0};
    const ExactQTTOperator laplacian =
        ExactQTTOperator::weighted_hypercube_laplacian(laplacian_weights);
    const ExactQTTFunction hamming = ExactQTTFunction::hamming_weight(bits);
    const ExactQTTFunction laplacian_result = laplacian.apply(hamming);
    assert(laplacian.stats().maximum_rank == 2U);
    assert(laplacian_result.stats().maximum_rank == 4U);

    for (BasisIndex index = 0U; index < (BasisIndex{1} << bits); ++index) {
        const auto state_bits = bits_for(bits, index);
        double expected = 0.0;
        for (std::size_t position = 0U; position < bits; ++position) {
            expected += laplacian_weights[position] *
                (state_bits[position] == 0U ? 1.0 : -1.0);
        }
        assert(close(laplacian_result.value(index), QComplex{expected}));
    }

    const std::vector<std::uint8_t> zero(bits, 0U);
    assert(close(laplacian.matrix_element_bits(zero, zero), QComplex{-15.0}));
    for (std::size_t position = 0U; position < bits; ++position) {
        std::vector<std::uint8_t> output = zero;
        output[position] = 1U;
        assert(close(
            laplacian.matrix_element_bits(output, zero),
            QComplex{laplacian_weights[position]}));
    }
    std::vector<std::uint8_t> two_flips = zero;
    two_flips[0] = 1U;
    two_flips[3] = 1U;
    assert(close(laplacian.matrix_element_bits(two_flips, zero), QComplex{}));

    const ExactQTTFunction wave = ExactQTTFunction::complex_exponential(bits, 0.173);
    const ExactQTTFunction potential = ExactQTTFunction::weighted_bit_sum(
        std::vector<QComplex>{
            QComplex{0.2}, QComplex{-0.4}, QComplex{0.6}, QComplex{0.1}, QComplex{-0.3},
        },
        QComplex{0.7});
    const ExactQTTOperator hamiltonian =
        laplacian.scaled(QComplex{-0.5}).add(ExactQTTOperator::diagonal(potential));
    const ExactQTTFunction evolved = hamiltonian.apply(wave);
    for (BasisIndex index = 0U; index < (BasisIndex{1} << bits); ++index) {
        QComplex direct_laplacian{};
        for (std::size_t position = 0U; position < bits; ++position) {
            const std::size_t shift = bits - 1U - position;
            const BasisIndex neighbor = index ^ (BasisIndex{1} << shift);
            direct_laplacian +=
                (wave.value(neighbor) - wave.value(index)) * laplacian_weights[position];
        }
        const QComplex expected =
            direct_laplacian * -0.5 + potential.value(index) * wave.value(index);
        assert(close(evolved.value(index), expected, 3e-11));
    }

    QTTOperatorConfig rank_one_only;
    rank_one_only.max_rank = 1U;
    assert(throws_qstate([&] {
        (void)ExactQTTOperator::weighted_hypercube_laplacian(laplacian_weights, rank_one_only);
    }));

    QTTOperatorConfig apply_rank_one;
    apply_rank_one.max_apply_rank = 1U;
    const ExactQTTOperator bounded_laplacian =
        ExactQTTOperator::weighted_hypercube_laplacian(laplacian_weights, apply_rank_one);
    assert(throws_qstate([&] { (void)bounded_laplacian.apply(hamming); }));

    QTTOperatorConfig scalar_bound;
    scalar_bound.max_core_scalars = 4U;
    assert(throws_qstate([&] {
        (void)ExactQTTOperator::weighted_hypercube_laplacian(laplacian_weights, scalar_bound);
    }));

    assert(throws_qstate([&] {
        std::vector<double> invalid = laplacian_weights;
        invalid[2] = std::numeric_limits<double>::infinity();
        (void)ExactQTTOperator::weighted_hypercube_laplacian(invalid);
    }));

    assert(throws_qstate([&] {
        QTTOperatorCore malformed;
        malformed.left_rank = 2U;
        malformed.right_rank = 1U;
        for (auto& block : malformed.blocks) {
            block.assign(2U, QComplex{});
        }
        (void)ExactQTTOperator::from_certified_cores(
            std::vector<QTTOperatorCore>{std::move(malformed)});
    }));

    return 0;
}
