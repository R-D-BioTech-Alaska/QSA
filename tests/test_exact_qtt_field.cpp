#include "qubit/qqtt_field.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

using namespace qubit;

namespace {

bool close(QComplex left, QComplex right, double tolerance = 5e-11) {
    return almost_equal(left, right, tolerance);
}

QComplex direct_hamiltonian(
    const ExactQTTFunction& field,
    const ExactQTTFunction& potential,
    std::span<const double> weights,
    BasisIndex index,
    double kinetic,
    double nonlinear) {
    const std::size_t bits = field.logical_bits();
    const QComplex center = field.value(index);
    QComplex laplacian{};
    for (std::size_t position = 0U; position < bits; ++position) {
        const std::size_t shift = bits - 1U - position;
        const BasisIndex neighbor = index ^ (BasisIndex{1} << shift);
        laplacian += (field.value(neighbor) - center) * weights[position];
    }
    return laplacian * -kinetic + potential.value(index) * center +
           center * (nonlinear * center.norm2());
}

}  // namespace

int main() {
    constexpr std::size_t bits = 4U;
    const std::vector<double> weights{0.5, 1.0, 1.5, 2.0};
    const ExactQTTFunction potential = ExactQTTFunction::weighted_bit_sum(
        std::vector<QComplex>{QComplex{0.25}, QComplex{-0.5}, QComplex{0.75}, QComplex{0.4}},
        QComplex{0.2});
    const ExactQTTFunction field = ExactQTTFunction::complex_exponential(
        bits, 0.071, QComplex{0.8, 0.2});
    constexpr double kinetic = 0.375;
    constexpr double nonlinear = 0.125;

    const ExactQTTFieldHamiltonian hamiltonian = ExactQTTFieldHamiltonian::hypercube(
        weights, potential, kinetic, nonlinear);
    assert(hamiltonian.stats().logical_bits == bits);
    assert(hamiltonian.stats().linear_operator_rank == 4U);

    const ExactQTTFunction rho = ExactQTTFieldHamiltonian::density(field);
    const ExactQTTFunction applied = hamiltonian.apply(field);
    const ExactQTTFunction rhs = hamiltonian.rhs(field);
    QComplex direct_energy{};
    double quartic_sum = 0.0;
    for (BasisIndex index = 0U; index < (BasisIndex{1} << bits); ++index) {
        const QComplex center = field.value(index);
        const QComplex expected = direct_hamiltonian(
            field, potential, weights, index, kinetic, nonlinear);
        assert(close(rho.value(index), QComplex{center.norm2()}));
        assert(close(applied.value(index), expected));
        assert(close(rhs.value(index), expected * QComplex{0.0, -1.0}));

        QComplex linear_expected = direct_hamiltonian(field, potential, weights, index, kinetic, 0.0);
        direct_energy += center.conjugate() * linear_expected;
        quartic_sum += center.norm2() * center.norm2();
    }
    direct_energy += QComplex{0.5 * nonlinear * quartic_sum};
    assert(close(hamiltonian.energy(field), direct_energy, 1e-10));

    bool shape_failed = false;
    try {
        (void)hamiltonian.apply(ExactQTTFunction::hamming_weight(bits + 1U));
    } catch (const QStateError&) {
        shape_failed = true;
    }
    assert(shape_failed);

    QTTConfig tight;
    tight.max_rank = 2U;
    tight.max_core_scalars = 64U;
    tight.max_total_scalars = 1024U;
    const ExactQTTFunction rank_two = ExactQTTFunction::hamming_weight(bits, tight);
    bool nonlinear_rank_failed = false;
    try {
        (void)ExactQTTFieldHamiltonian::density(rank_two).hadamard(rank_two);
    } catch (const QStateError&) {
        nonlinear_rank_failed = true;
    }
    assert(nonlinear_rank_failed);

    return 0;
}
