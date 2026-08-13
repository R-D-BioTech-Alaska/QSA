#include "qubit/qqtt_sp_invariants.hpp"

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

bool close(double left, double right, double tolerance = 5e-11) {
    return std::abs(left - right) <= tolerance * (1.0 + std::max(std::abs(left), std::abs(right)));
}

std::vector<std::uint8_t> bits_for(std::size_t logical_bits, BasisIndex index) {
    std::vector<std::uint8_t> bits(logical_bits);
    for (std::size_t position = 0U; position < logical_bits; ++position) {
        const std::size_t shift = logical_bits - 1U - position;
        bits[position] = static_cast<std::uint8_t>((index >> shift) & BasisIndex{1});
    }
    return bits;
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

}  // namespace

int main() {
    constexpr std::size_t logical_bits = 6U;
    const std::vector<double> weights{0.5, 1.0, 1.5, 0.75, 1.25, 2.0};
    ComplexWalshConfig wave_config;
    wave_config.max_terms = 32U;
    wave_config.max_support_entries = 1024U;
    wave_config.max_products = 4096U;
    const ExactComplexWalshField wave = ExactComplexWalshField::from_terms(
        logical_bits,
        {
            ComplexWalshTerm{QComplex{0.8, 0.1}, {}},
            ComplexWalshTerm{QComplex{0.2, -0.05}, {0U}},
            ComplexWalshTerm{QComplex{-0.1, 0.07}, {1U, 2U}},
        },
        wave_config);

    constexpr double mu = 0.625;
    constexpr double alpha = 1.4;
    constexpr double kinetic = 0.375;
    constexpr double nonlinear = 0.2;
    constexpr double background = 0.3;
    const ExactQTTSchrodingerPoisson solver(weights, mu, alpha, kinetic, nonlinear);
    WalshFieldConfig real_config;
    real_config.max_terms = 32U;
    real_config.max_support_entries = 1024U;
    const QTTSchrodingerPoissonResult result = solver.evaluate(wave, background, real_config);
    const ExactWalshField poisson_residual = solver.poisson().residual(result.source, result.potential);
    const QTTSchrodingerPoissonDiagnostics diagnostics =
        diagnose_schrodinger_poisson(wave, result);
    assert(poisson_residual.maximum_absolute_coefficient() <= 1e-12);

    QTTConfig qtt_config;
    qtt_config.max_rank = 32U;
    qtt_config.max_core_scalars = 4096U;
    qtt_config.max_total_scalars = 1U << 18U;
    const ExactQTTFunction rhs_qtt = result.rhs.to_qtt(qtt_config);

    double direct_mean_norm = 0.0;
    QComplex direct_hamiltonian_expectation{};
    QComplex direct_rhs_overlap{};
    const double point_weight = 1.0 / static_cast<double>(BasisIndex{1} << logical_bits);
    for (BasisIndex index = 0U; index < (BasisIndex{1} << logical_bits); ++index) {
        const std::vector<std::uint8_t> bits = bits_for(logical_bits, index);
        const QComplex center = wave.value_bits(bits);
        const double density = center.norm2();
        assert(close(result.density.value_bits(bits), density));
        assert(close(result.source.value_bits(bits), density - background));

        const double potential = result.potential.value_bits(bits);
        double poisson_laplacian = 0.0;
        QComplex wave_laplacian{};
        for (std::size_t position = 0U; position < logical_bits; ++position) {
            std::vector<std::uint8_t> neighbor = bits;
            neighbor[position] ^= 1U;
            poisson_laplacian += weights[position] *
                (potential - result.potential.value_bits(neighbor));
            wave_laplacian += weights[position] *
                (center - wave.value_bits(neighbor));
        }
        assert(close(mu * potential + poisson_laplacian, alpha * (density - background), 1e-10));

        const QComplex h_expected = wave_laplacian * kinetic + center * potential +
                                    center * (nonlinear * density);
        const QComplex rhs_expected = h_expected * QComplex{0.0, -1.0};
        assert(close(result.hamiltonian_wave.value_bits(bits), h_expected, 1e-10));
        assert(close(result.rhs.value_bits(bits), rhs_expected, 1e-10));
        assert(close(rhs_qtt.value_bits(bits), rhs_expected, 1e-10));
        direct_mean_norm += density * point_weight;
        direct_hamiltonian_expectation += center.conjugate() * h_expected * point_weight;
        direct_rhs_overlap += center.conjugate() * rhs_expected * point_weight;
    }

    assert(result.stats.wave_terms == 3U);
    assert(result.stats.density_terms <= 4U);
    assert(result.stats.potential_terms == result.stats.density_terms);
    assert(result.stats.hamiltonian_terms <= 8U);
    assert(close(result.stats.inverse_poisson_norm_bound, 1.0 / mu));
    assert(close(diagnostics.wave_mean_norm_squared, direct_mean_norm, 1e-10));
    assert(close(diagnostics.hamiltonian_expectation, direct_hamiltonian_expectation, 1e-10));
    assert(close(diagnostics.norm_rate, 2.0 * direct_rhs_overlap.re, 1e-10));
    assert(std::abs(diagnostics.hamiltonian_expectation.im) <= 1e-10);
    assert(std::abs(diagnostics.norm_rate) <= 1e-10);

    ComplexWalshConfig product_cap = wave_config;
    product_cap.max_products = 4U;
    const ExactComplexWalshField capped = ExactComplexWalshField::from_terms(
        logical_bits,
        {
            ComplexWalshTerm{QComplex{1.0}, {}},
            ComplexWalshTerm{QComplex{0.5}, {0U}},
            ComplexWalshTerm{QComplex{0.25}, {1U}},
        },
        product_cap);
    assert(throws_qstate([&] { (void)capped.density(real_config); }));
    assert(throws_qstate([&] { (void)solver.evaluate(capped, 0.0, real_config); }));

    QTTConfig rank_one;
    rank_one.max_rank = 1U;
    rank_one.max_core_scalars = 64U;
    rank_one.max_total_scalars = 4096U;
    assert(throws_qstate([&] { (void)result.rhs.to_qtt(rank_one); }));

    return 0;
}
